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

from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from tenacity import retry, wait_exponential

from models.research import RankOrder, Result

load_dotenv(override=True)

# Config

MODEL = "openai/gpt-4.1-mini"
DB_NAME = str(Path(__file__).parent / "edgar_db")
COLLECTION_NAME = "edgar_filings"
EMBEDDING_MODEL = "text-embedding-3-large"
WAIT = wait_exponential(multiplier=1, min=2, max=240)

RETRIEVAL_K = 15
FINAL_K = 8

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma = PersistentClient(path=DB_NAME)
collection = chroma.get_or_create_collection(COLLECTION_NAME)

SYSTEM_PROMPT = """You are a financial research assistant that answers questions \
about SEC filings (10-K, 10-Q) for public companies.

Your answers must be:
- Grounded in the provided filing excerpts
- Technically precise and factual
- Cited by company name and filing type when referencing specific data
- Honest about uncertainty — if the context doesn't contain enough information, say so

Here are relevant excerpts from SEC filings:

{context}

Answer the user's question based on these excerpts. \
Always cite the company name and form type when referencing specific findings."""


# Query rewriting


@retry(wait=WAIT)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """
    Rewrite the user's question into a concise retrieval query.
    Takes conversation history into account for follow-up questions.
    """
    history = history or []
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
    return response.choices[0].message.content.strip()


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
    ticker: str | None = None,
    period: str | None = None,
    k: int = RETRIEVAL_K,
) -> list[Result]:
    """
    Retrieve top-k chunks from ChromaDB.
    Optionally filter by ticker and/or period_of_report.
    """
    query_vec = embed_query(query)

    # Build where filter
    if ticker and period:
        where = {"$and": [
            {"ticker": {"$eq": ticker.upper()}},
            {"period_of_report": {"$eq": period}},
        ]}
    elif ticker:
        where = {"ticker": {"$eq": ticker.upper()}}
    else:
        where = None

    try:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, collection.count() or 1),
            where=where,
        )
    except Exception:
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, collection.count() or 1),
        )

    chunks = []
    for doc, meta in zip(
        results["documents"][0], results["metadatas"][0]
    ):
        chunks.append(Result(page_content=doc, metadata=meta))
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
            f"{chunk.page_content[:300]}\n\n"
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
        parts.append(f"{header}:\n{chunk.page_content}")
    return "\n\n---\n\n".join(parts)


def make_rag_messages(
    question: str,
    history: list[dict],
    chunks: list[Result],
) -> list[dict]:
    """Build the full message list for the final answer LLM call."""
    context = build_context(chunks)
    system = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system}]
        + history
        + [{"role": "user", "content": question}]
    )


# Main entry point


def fetch_context(
    question: str,
    ticker: str | None = None,
    period: str | None = None,
    final_k: int = FINAL_K,
) -> list[Result]:
    rewritten = rewrite_query(question)
    chunks_original = fetch_chunks(question, ticker=ticker, period=period)
    chunks_rewritten = fetch_chunks(rewritten, ticker=ticker, period=period)
    merged = merge_chunks(chunks_original, chunks_rewritten)
    reranked = rerank(question, merged)
    return reranked[:final_k]


@retry(wait=WAIT)
def answer_question(
    question: str,
    history: list[dict] | None = None,
    ticker: str | None = None,
    period: str | None = None,
) -> tuple[str, list[Result]]:
    """
    Answer a question using RAG over ingested SEC filings.

    Args:
        question: The user's question
        history:  Conversation history as list of {role, content} dicts
        ticker:   Optional ticker to restrict retrieval to one company
        period:   Optional period_of_report to restrict to a specific filing

    Returns:
        (answer_text, retrieved_chunks)
    """
    history = history or []
    chunks = fetch_context(question, ticker=ticker, period=period)

    if not chunks:
        return (
            "No relevant filing data found. "
            "Please run the ingestion pipeline first: "
            "`uv run python ingest.py --tickers AAPL --form-type 10-Q`",
            [],
        )

    messages = make_rag_messages(question, history, chunks)
    response = completion(model=MODEL, messages=messages)
    return response.choices[0].message.content, chunks