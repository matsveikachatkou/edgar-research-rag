"""
answer.py — RAG retrieval and answer generation.

Retrieval pipeline:
    1. Rewrite query (history-aware)
    2. Dual retrieval: original + rewritten query, optionally filtered by ticker and period
    3. Merge and deduplicate chunks
    4. LLM rerank → top FINAL_K
    5. Generate grounded answer with citations
"""

import os
from pathlib import Path
import logging
from datetime import datetime
import re

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from tenacity import retry, wait_exponential

from models.research import RankOrder, Result
from xbrl import get_financial_snapshot, format_snapshot_for_context, FinancialSnapshot

_rewrite_cache: dict[str, str] = {}

load_dotenv(override=True)

# Config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL = "openai/gpt-4.1-mini"
DB_NAME = str(Path(__file__).parent / "edgar_db")
COLLECTION_NAME = "edgar_filings"
EMBEDDING_MODEL = "text-embedding-3-large"
WAIT = wait_exponential(multiplier=1, min=2, max=30)

RETRIEVAL_K = 8        # single-ticker queries
RETRIEVAL_K_BROAD = 15 # unfiltered cross-company queries
FINAL_K = 5
FINAL_K_BROAD = 8

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a financial research assistant that answers questions \
about SEC filings (10-K, 10-Q) and earnings press releases (8-K) for public companies.

Your answers must be:
- Grounded in the provided filing excerpts and structured financials
- Technically precise — use the exact XBRL figures when answering quantitative questions
- Cited by company name and filing type when referencing specific data
- Honest about uncertainty — if the context doesn't contain enough information, say so

Important: When both GAAP and non-GAAP figures are present in the context:
- XBRL structured data contains audited GAAP figures
- 8-K press release excerpts may contain non-GAAP figures
- Always label which standard you are citing

{warning_block}{xbrl_block}Here are relevant excerpts from SEC filings and press releases:

{context}

Answer the user's question based on these sources. \
Always cite the company name and form type when referencing specific findings."""


# Query rewriting


@retry(wait=WAIT)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """
    Rewrite the user's question into a concise retrieval query.
    Takes conversation history into account for follow-up questions.
    Results are cached in-memory to avoid redundant LLM calls.
    """
    history = history or []
    cache_key = question[:100] + str(history[-1:])
    if cache_key in _rewrite_cache:
        return _rewrite_cache[cache_key]

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-4:]
    )
    message = f"""You are helping a financial analyst search through SEC filings.

Conversation history:
{history_text}

Current question:
{question}

Rewrite this into a short, precise search query (5-10 words) optimised to \
retrieve relevant financial filing content. Focus on financial metrics, \
business segments, risk factors, or management commentary as appropriate.

Respond ONLY with the search query — no explanation."""

    response = completion(
        model=MODEL,
        messages=[{"role": "system", "content": message}],
    )
    result = response.choices[0].message.content.strip()
    _rewrite_cache[cache_key] = result
    return result


# Retrieval


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return (
        openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        .data[0]
        .embedding
    )


def fetch_chunks(
    query: str,
    collection,
    ticker: str | None = None,
    period: str | None = None,
    filing_date_lte: str | None = None,
    filing_date_gte: str | None = None,  # added
    k: int = RETRIEVAL_K,
) -> list[Result]:
    query_vec = embed_query(query)

    conditions = []
    if ticker:
        conditions.append({"ticker": {"$eq": ticker.upper()}})
    if period and not filing_date_lte:
        conditions.append({"period_of_report": {"$eq": period}})

    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(k * 2 if filing_date_lte else k, collection.count() or 1),
            where=where,
        )
    except Exception as e:
        log.error(f"Chroma filtering query failed for ticker={ticker} "
                  f"filing_date_lte={filing_date_lte} period={period}: {e}")
        return []

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        if filing_date_lte or filing_date_gte:
            chunk_date = meta.get("filing_date", "")
            if filing_date_lte and chunk_date > filing_date_lte:
                continue
            if filing_date_gte and chunk_date < filing_date_gte:
                continue
        chunks.append(Result(page_content=doc, metadata=meta))
        if len(chunks) >= k:
            break

    return chunks


def merge_chunks(a: list[Result], b: list[Result]) -> list[Result]:
    """Merge two chunk lists, deduplicating by page_content."""
    merged = list(a)
    seen = {c.page_content for c in a}
    for chunk in b:
        if chunk.page_content not in seen:
            merged.append(chunk)
            seen.add(chunk.page_content)
    return merged


def merge_chunks_balanced(
    a: list[Result],
    b: list[Result],
    tickers: list[str],
    k_per_ticker: int = 8,
) -> list[Result]:
    seen = set()
    per_ticker: dict[str, list[Result]] = {t: [] for t in tickers}

    for chunk in list(a) + list(b):
        if chunk.page_content in seen:
            continue
        seen.add(chunk.page_content)
        ticker = chunk.metadata.get("ticker", "")
        if ticker in per_ticker and len(per_ticker[ticker]) < k_per_ticker:
            per_ticker[ticker].append(chunk)
        # Drop overflow entirely — non-target tickers are noise

    merged = []
    for t in tickers:
        merged.extend(per_ticker[t])
    return merged


def detect_tickers_in_query(question: str, collection) -> list[str]:
    """
    Detect which ingested tickers are mentioned in the query.
    Uses regex word boundaries to avoid substring matching traps (e.g., 'A', 'IT').
    Uses targeted metadata-only fetching to prevent OOM memory spikes.
    """
    try:
        # Prevent Memory Bomb: Only fetch metadata, never documents/embeddings here
        results = collection.get(include=["metadatas"])
        available = set(
            m.get("ticker", "") 
            for m in results["metadatas"] 
            if m.get("ticker")
        )
    except Exception as e:
        log.error(f"Failed to fetch available tickers: {e}")
        return []

    question_upper = question.upper()
    matched = []
    
    for t in available:
        # Prevent Substring Trap: Match whole words only
        if re.search(rf'\b{t}\b', question_upper):
            matched.append(t)
            
    return matched


def fetch_chunks_parallel(
    question: str,
    collection,
    tickers: list[str],
    k_per_ticker: int = 8,
    filing_date_lte: str | None = None,
    filing_date_gte: str | None = None,  # added
) -> list[Result]:
    query_vec = embed_query(question)
    all_chunks: list[Result] = []
    seen = set()

    for ticker in tickers:
        where = {"ticker": {"$eq": ticker}}
        try:
            results = collection.query(
                query_embeddings=[query_vec],
                n_results=min(k_per_ticker * 2 if filing_date_lte else k_per_ticker, collection.count() or 1),
                where=where,
            )
            ticker_count = 0
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                if filing_date_lte or filing_date_gte:
                    chunk_date = meta.get("filing_date", "")
                    if filing_date_lte and chunk_date > filing_date_lte:
                        continue
                    if filing_date_gte and chunk_date < filing_date_gte:
                        continue
                if doc not in seen:
                    all_chunks.append(Result(page_content=doc, metadata=meta))
                    seen.add(doc)
                    ticker_count += 1
                    if ticker_count >= k_per_ticker:
                        break
            log.info(f"Parallel fetch: {ticker} → {ticker_count} chunks")
        except Exception as e:
            log.error(f"Parallel fetch failed for {ticker}: {e}")

    return all_chunks


# Reranking


@retry(wait=WAIT)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """
    Re-rank retrieved chunks by relevance to the question using an LLM.
    Returns chunks sorted from most to least relevant.
    """
    if not chunks:
        return []

    system_prompt = (
        "You are a financial document re-ranker. "
        "Given a question and numbered excerpts from SEC filings, "
        "rank all chunks by relevance to the question. "
        "Reply only with the ranked list of chunk ids."
    )
    user_prompt = f"Question:\n{question}\n\nRank all chunks by relevance.\n\n"
    for idx, chunk in enumerate(chunks):
        ticker = chunk.metadata.get("ticker", "")
        form = chunk.metadata.get("form_type", "")
        user_prompt += (
            f"# CHUNK ID: {idx + 1} [{ticker} {form}]:\n\n"
            f"{chunk.page_content}\n\n"
        )
    user_prompt += "Reply only with the list of ranked chunk ids."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(
        model=MODEL, messages=messages, response_format=RankOrder,
        timeout=60,
    )
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order if 1 <= i <= len(chunks)]


# Context building


def build_context(chunks: list[Result]) -> str:
    """Format chunks into a context string for the LLM."""
    parts = []
    for chunk in chunks:
        ticker = chunk.metadata.get("ticker", "Unknown")
        form = chunk.metadata.get("form_type", "")
        period = chunk.metadata.get("period_of_report", "")
        company = chunk.metadata.get("company_name", ticker)
        header = f"[{company} ({ticker}) — {form} {period}]"
        parts.append(f"{header}:\n{chunk.page_content[:2000]}")
    return "\n\n---\n\n".join(parts)


from datetime import datetime

def detect_temporal_mismatch(
    chunks: list[Result],
    snapshot: FinancialSnapshot | None,
) -> str | None:
    """
    Detect if retrieved chunks contain more recent data than the XBRL snapshot.
    Returns a warning string if mismatch detected, None otherwise.
    """
    if not snapshot or not snapshot.period_end:
        return None

    # Find the latest period across all retrieved chunks
    chunk_periods = []
    for chunk in chunks:
        period = chunk.metadata.get("period_of_report", "")
        if period:
            chunk_periods.append(period)

    if not chunk_periods:
        return None

    # Sort chronologically, not lexicographically
    try:
        latest_chunk_period = max(
            chunk_periods, 
            key=lambda d: datetime.strptime(d, "%Y-%m-%d")
        )
        
        chunk_date = datetime.strptime(latest_chunk_period, "%Y-%m-%d")
        snapshot_date = datetime.strptime(snapshot.period_end, "%Y-%m-%d")
        
        if chunk_date > snapshot_date:
            return (
                f"SYSTEM WARNING: The structured financial data (XBRL) is dated "
                f"{snapshot.period_end} ({snapshot.form_type}). "
                f"The retrieved narrative context contains more recent material "
                f"events up to {latest_chunk_period}. "
                f"Do NOT apply the quantitative figures to the more recent narrative "
                f"events. Clearly distinguish between historical GAAP data and "
                f"the more recent narrative disclosures when formulating your answer."
            )
    except ValueError as e:
        # Failsafe: If date parsing fails due to unexpected format, skip the warning 
        # rather than crashing the entire RAG pipeline.
        log.warning(f"Date parsing error in temporal mismatch check: {e}")
        return None

    return None


def make_rag_messages(
    question: str,
    history: list[dict],
    chunks: list[Result],
    xbrl_context: str | None = None,
    temporal_warning: str | None = None,
) -> list[dict]:
    context = build_context(chunks)
    xbrl_block = f"{xbrl_context}\n\n---\n\n" if xbrl_context else ""
    warning_block = f"{temporal_warning}\n\n---\n\n" if temporal_warning else ""
    system = SYSTEM_PROMPT.format(
        context=context,
        xbrl_block=xbrl_block,
        warning_block=warning_block,
    )
    return (
        [{"role": "system", "content": system}]
        + history[-4:]
        + [{"role": "user", "content": question}]
    )


# Main entry point


def fetch_context(
    question: str,
    collection,
    ticker: str | None = None,
    period: str | None = None,
    form_type: str | None = None,
    filing_date_lte: str | None = None,
    final_k: int = FINAL_K,
) -> tuple[list[Result], str | None, str | None]:

    # Compute 90-day PIT window when cutoff is set
    filing_date_gte = None
    if filing_date_lte:
        from datetime import datetime, timedelta
        cutoff = datetime.strptime(filing_date_lte, "%Y-%m-%d")
        filing_date_gte = (cutoff - timedelta(days=90)).strftime("%Y-%m-%d")

    # Detect cross-company query
    if not ticker:
        detected = detect_tickers_in_query(question, collection)
    else:
        detected = []

    if len(detected) >= 2:
        log.info(f"Cross-company query detected: {detected} — using parallel retrieval")
        rewritten = rewrite_query(question)
        chunks_original = fetch_chunks_parallel(
            question, collection, detected,
            filing_date_lte=filing_date_lte,
            filing_date_gte=filing_date_gte,
        )
        chunks_rewritten = fetch_chunks_parallel(
            rewritten, collection, detected,
            filing_date_lte=filing_date_lte,
            filing_date_gte=filing_date_gte,
        )
        merged = merge_chunks_balanced(chunks_original, chunks_rewritten, detected)
        effective_final_k = FINAL_K_BROAD

    elif ticker:
        rewritten = rewrite_query(question)
        chunks_original = fetch_chunks(
            question, collection,
            ticker=ticker, period=period,
            filing_date_lte=filing_date_lte,
            filing_date_gte=filing_date_gte,
            k=RETRIEVAL_K,
        )
        chunks_rewritten = fetch_chunks(
            rewritten, collection,
            ticker=ticker, period=period,
            filing_date_lte=filing_date_lte,
            filing_date_gte=filing_date_gte,
            k=RETRIEVAL_K,
        )
        merged = merge_chunks(chunks_original, chunks_rewritten)
        effective_final_k = final_k

    else:
        rewritten = rewrite_query(question)
        chunks_original = fetch_chunks(
            question, collection,
            filing_date_lte=filing_date_lte,
            filing_date_gte=filing_date_gte,
            k=RETRIEVAL_K_BROAD,
        )
        chunks_rewritten = fetch_chunks(
            rewritten, collection,
            filing_date_lte=filing_date_lte,
            filing_date_gte=filing_date_gte,
            k=RETRIEVAL_K_BROAD,
        )
        merged = merge_chunks(chunks_original, chunks_rewritten)
        effective_final_k = FINAL_K_BROAD

    reranked = rerank(question, merged)
    final_chunks = reranked[:effective_final_k]

    xbrl_context = None
    temporal_warning = None
    if ticker:
        snapshot = get_financial_snapshot(
            ticker=ticker,
            form_type=form_type,
            period_end=period,
            pit_cutoff=filing_date_lte,
        )
        if snapshot:
            xbrl_context = format_snapshot_for_context(snapshot)
            temporal_warning = detect_temporal_mismatch(final_chunks, snapshot)

    return final_chunks, xbrl_context, temporal_warning


@retry(wait=WAIT)
def answer_question(
    question: str,
    history: list[dict] | None = None,
    ticker: str | None = None,
    period: str | None = None,
    form_type: str | None = None,
    filing_date_lte: str | None = None,  
) -> tuple[str, list[Result]]:
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    history = history or []
    chunks, xbrl_context, temporal_warning = fetch_context(
        question, collection,
        ticker=ticker,
        period=period,
        form_type=form_type,
        filing_date_lte=filing_date_lte,
    )

    if not chunks:
        return (
            "No relevant filing data found. "
            "Please run the ingestion pipeline first: "
            "`uv run python ingest.py --tickers AAPL --form-type 10-Q`",
            [],
        )

    messages = make_rag_messages(question, history, chunks, xbrl_context, temporal_warning)
    response = completion(model=MODEL, messages=messages, timeout=60)
    return response.choices[0].message.content, chunks