"""
ingest.py — SEC EDGAR ingestion pipeline.

Pipeline:
    1. Fetch recent filings from SEC EDGAR submissions API
    2. Extract text from HTML filings via BeautifulSoup
    3. Chunk documents semantically via LLM
    4. Embed chunks and store in ChromaDB (incremental — no delete on rerun)

Usage:
    uv run python ingest.py --tickers AAPL MSFT --form-type 10-Q
    uv run python ingest.py --tickers NVDA --form-type 10-K --max-chars 15000
"""

import argparse
import logging
import os
import sys
import re
import warnings
import time
from multiprocessing import Pool
from pathlib import Path

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from openai import OpenAI
from tenacity import retry, wait_exponential

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from models.research import Chunk, Chunks, EdgarFiling, Result

load_dotenv(override=True)


# Config

MODEL = "openai/gpt-4.1-mini"
DB_NAME = str(Path(__file__).parent / "edgar_db")
COLLECTION_NAME = "edgar_filings"
EMBEDDING_MODEL = "text-embedding-3-large"
AVERAGE_CHUNK_SIZE = 500
WORKERS = 3
WAIT = wait_exponential(multiplier=1, min=2, max=240)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ingest] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SEC_HEADERS = {"User-Agent": "edgar-research-rag research@example.com"}


# Step 1 — Fetch filings from SEC EDGAR


def fetch_filings(ticker: str, form_type: str = "10-Q", k: int = 1) -> list[EdgarFiling]:
    """
    Fetch recent filings for a ticker from SEC EDGAR full-text search.
    Returns up to k filings sorted by date descending.
    """
    log.info(f"Fetching {form_type} filings for {ticker}")

    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form_type}"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        log.warning(f"No {form_type} filings found for {ticker}")
        return []

    filings = []
    for hit in hits[:k]:
        src = hit.get("_source", {})
        entity_name = src.get("entity_name", ticker)
        file_date = src.get("file_date", "")
        period = src.get("period_of_report", "")
        accession_raw = hit.get("_id", "")
        accession = accession_raw.replace("-", "")

        # Build the filing index URL
        cik = src.get("_id", accession_raw).split(":")[0] if ":" in accession_raw else ""
        filing_url = src.get("file_date", "")

        # Use the document URL directly from the hit
        doc_url = ""
        for doc in src.get("period_of_report", []) if isinstance(src.get("period_of_report"), list) else []:
            if doc.endswith(".htm") or doc.endswith(".html"):
                doc_url = doc
                break

        # Fallback: construct filing index from accession number
        accession_fmt = accession_raw if accession_raw else ""
        filing_index_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type={form_type}&dateb=&owner=include&count=10"

        from datetime import datetime
        try:
            filed_dt = datetime.strptime(file_date, "%Y-%m-%d") if file_date else datetime.utcnow()
        except ValueError:
            filed_dt = datetime.utcnow()

        filing = EdgarFiling(
            ticker=ticker.upper(),
            company_name=entity_name,
            form_type=form_type,
            filed_at=filed_dt,
            period_of_report=period,
            filing_url=filing_index_url,
            pdf_url=None,
        )
        filings.append(filing)
        log.info(f"Found filing: {entity_name} {form_type} filed {file_date}")

    return filings


def resolve_pdf_url(ticker: str, form_type: str) -> str | None:
    """
    Resolve the most recent PDF/document URL for a ticker+form_type
    via the EDGAR full-text search API.
    """
    log.info(f"Resolving PDF URL for {ticker} {form_type}")
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form_type}"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        hits = resp.json().get("hits", {}).get("hits", [])
        if not hits:
            return None

        hit = hits[0]
        src = hit.get("_source", {})

        # Try to get direct document URLs from the filing
        file_num = src.get("file_num", "")
        accession = hit.get("_id", "")

        # Construct accession-based index URL
        if accession:
            acc_clean = accession.replace("-", "")
            # Extract CIK from entity_id if available
            entity_id = src.get("entity_id", "")
            if entity_id:
                cik_padded = str(entity_id).zfill(10)
                index_url = f"https://www.sec.gov/Archives/edgar/data/{entity_id}/{acc_clean}/{accession}-index.htm"
                log.info(f"Filing index URL: {index_url}")
                return index_url

        return None
    except Exception as e:
        log.warning(f"Could not resolve PDF URL for {ticker}: {e}")
        return None


# CIK resolution


_CIK_CACHE: dict[str, str] = {}


def resolve_cik(ticker: str) -> str | None:
    """
    Resolve a ticker symbol to SEC CIK using the SEC's public mapping file.
    Results are cached in memory for the session.
    """
    ticker = ticker.upper()
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker] = cik
                log.info(f"Resolved {ticker} → CIK {cik} ({entry.get('title', '')})")
                return cik

        log.warning(f"Ticker {ticker} not found in SEC mapping")
        return None

    except Exception as e:
        log.error(f"CIK resolution failed for {ticker}: {e}")
        return None


def fetch_filings_v2(ticker: str, form_type: str = "10-Q", k: int = 1) -> list[EdgarFiling]:
    """
    Fetch recent filings using the EDGAR submissions API.
    Resolves ticker to CIK dynamically via SEC's public mapping.
    """
    from datetime import datetime

    cik = resolve_cik(ticker.upper())
    if not cik:
        log.warning(f"Could not resolve CIK for {ticker} — skipping")
        return []

    log.info(f"Fetching {form_type} for {ticker} via submissions API (CIK: {cik})")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"Submissions API failed for {ticker}: {e}")
        return []

    company_name = data.get("name", ticker)
    filings_data = data.get("filings", {}).get("recent", {})
    forms = filings_data.get("form", [])
    dates = filings_data.get("filingDate", [])
    accessions = filings_data.get("accessionNumber", [])
    primary_docs = filings_data.get("primaryDocument", [])
    periods = filings_data.get("reportDate", [])

    filings = []
    for i, form in enumerate(forms):
        if form != form_type:
            continue
        if len(filings) >= k:
            break

        accession = accessions[i]
        cik_short = cik.lstrip("0")
        acc_clean = accession.replace("-", "")
        primary_doc = primary_docs[i]

        doc_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_short}/{acc_clean}/{primary_doc}"
        )
        filing_url = (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_short}/{acc_clean}/{accession}-index.htm"
        )

        try:
            filed_dt = datetime.strptime(dates[i], "%Y-%m-%d")
        except Exception:
            filed_dt = datetime.utcnow()

        filing = EdgarFiling(
            ticker=ticker.upper(),
            company_name=company_name,
            form_type=form_type,
            filed_at=filed_dt,
            period_of_report=periods[i] if i < len(periods) else "",
            filing_url=filing_url,
            pdf_url=doc_url,
        )
        filings.append(filing)
        log.info(
            f"Resolved: {company_name} | {form_type} | "
            f"filed {dates[i]} | period {periods[i]}"
        )

    return filings


def get_primary_doc_url(filing: EdgarFiling) -> str | None:
    """Return the pre-resolved primary doc URL from the filing."""
    return filing.pdf_url or None


# Step 2 — Extract text from filings


def extract_html(url: str, max_chars: int | None = None) -> str:
    """
    Fetch an SEC HTML filing and extract clean text using BeautifulSoup.
    This is the primary extraction method for modern EDGAR filings.
    """
    log.info(f"Fetching HTML: {url}")
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.content, "lxml")

        # Remove noise: scripts, styles, XBRL metadata
        for tag in soup(["script", "style", "head", "ix:header", "xbrl"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)

        if max_chars:
            text = text[:max_chars]

        log.info(f"Extracted {len(text):,} chars from HTML")
        return text
    except Exception as e:
        log.warning(f"HTML extraction failed for {url}: {e}")
        return ""


def extract_pdf(url: str) -> str:
    """PDF extraction not implemented — add Mistral OCR here if needed."""
    log.warning(f"PDF extraction not supported: {url}")
    return ""


def extract_document(
    url: str,
    max_chars: int | None = None,
) -> str:
    """
    Extract text from a filing document.
    Uses BeautifulSoup for HTML filings.
    PDF support can be added via Mistral OCR if needed.
    """
    if not url:
        return ""
    if url.lower().endswith(".pdf"):
        return extract_pdf(url)
    return extract_html(url, max_chars=max_chars)


def enrich_filings(
    filings: list[EdgarFiling],
    max_chars: int | None = None,
) -> list[EdgarFiling]:
    """Extract text content for each filing."""
    for filing in filings:
        log.info(f"Enriching {filing.ticker} {filing.form_type}")
        if not filing.pdf_url:
            log.warning(f"No document URL for {filing.ticker} — skipping")
            filing.document_markdown = f"No content for {filing.ticker} {filing.form_type}"
            continue

        text = extract_document(
            filing.pdf_url,
            max_chars=max_chars,
        )
        filing.document_markdown = text or f"No content extracted for {filing.ticker}"

    return filings


# Step 3 — LLM chunking


def _make_chunk_prompt(filing: EdgarFiling) -> str:
    how_many = max(5, len(filing.document_markdown) // AVERAGE_CHUNK_SIZE)
    return f"""You are processing a SEC {filing.form_type} filing for a financial knowledge base.

Company: {filing.company_name} ({filing.ticker})
Form type: {filing.form_type}
Period: {filing.period_of_report or "N/A"}
Filed: {filing.filed_at.strftime("%Y-%m-%d")}

A financial research assistant will use these chunks to answer questions about
this company's financial position, risks, opportunities, and outlook.

Divide the document so the entire content is covered — don't leave anything out.
Target at least {how_many} chunks. Use roughly 25% overlap (~50 words) between
adjacent chunks for best retrieval.

Focus especially on:
- Financial highlights (revenue, margins, EPS, guidance)
- Risk factors
- Business segment performance
- Management commentary and outlook
- Balance sheet and cash flow items

For each chunk provide:
- headline: a brief heading capturing the key financial topic
- summary: 2-3 sentences summarising the chunk for retrieval
- original_text: the exact text from the filing

Here is the filing content:

{filing.document_markdown}

Respond with the chunks."""


def _make_messages(filing: EdgarFiling) -> list[dict]:
    return [{"role": "user", "content": _make_chunk_prompt(filing)}]


BATCH_SIZE = 8000  # chars per batch, safe for gpt-4.1-mini context
BATCH_OVERLAP = 500  # overlap between batches to avoid cutting mid-sentence


def split_into_batches(text: str, batch_size: int = BATCH_SIZE, overlap: int = BATCH_OVERLAP) -> list[str]:
    """Split a long document into overlapping batches for LLM chunking."""
    if len(text) <= batch_size:
        return [text]

    batches = []
    start = 0
    while start < len(text):
        end = start + batch_size
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + batch_size // 2:
                end = boundary
        batches.append(text[start:end])
        start = end - overlap

    log.info(f"Split into {len(batches)} batches ({len(text):,} total chars)")
    return batches


@retry(wait=WAIT)
def process_batch(batch_text: str, filing: EdgarFiling, batch_num: int) -> list[Result]:
    """Chunk a single batch of filing text via LLM."""
    batch_filing = filing.model_copy()
    batch_filing.document_markdown = batch_text
    messages = _make_messages(batch_filing)
    response = completion(
        model=MODEL,
        messages=messages,
        response_format=Chunks,
    )
    reply = response.choices[0].message.content
    doc_chunks = Chunks.model_validate_json(reply).chunks
    results = [chunk.as_result(filing) for chunk in doc_chunks]
    log.info(f"Batch {batch_num}: {len(results)} chunks")
    return results


def process_filing(filing: EdgarFiling) -> list[Result]:
    """Chunk a filing via LLM using batched processing."""
    if not filing.document_markdown:
        log.warning(f"No content for {filing.ticker} — skipping")
        return []

    log.info(f"Chunking {filing.ticker} {filing.form_type} ({len(filing.document_markdown):,} chars)")
    batches = split_into_batches(filing.document_markdown)
    all_results: list[Result] = []

    for i, batch in enumerate(batches, 1):
        try:
            results = process_batch(batch, filing, batch_num=i)
            all_results.extend(results)
            time.sleep(3)
        except Exception as e:
            log.error(f"Batch {i} failed for {filing.ticker}: {e}")
            continue

    log.info(f"Total chunks for {filing.ticker}: {len(all_results)}")
    return all_results


def create_chunks(filings: list[EdgarFiling]) -> list[Result]:
    """Chunk all filings using parallel workers."""
    all_chunks: list[Result] = []
    with Pool(processes=WORKERS) as pool:
        for results in pool.imap_unordered(process_filing, filings):
            all_chunks.extend(results)
    return all_chunks


# Step 4 — Embed and store in ChromaDB (incremental)


def _filing_id(filing: EdgarFiling) -> str:
    """Stable dedup key for a filing."""
    return f"{filing.ticker}_{filing.form_type}_{filing.period_of_report or filing.filed_at.strftime('%Y%m%d')}"


def already_ingested(collection, filing: EdgarFiling) -> bool:
    """Check if this filing is already in the vector store."""
    fid = _filing_id(filing)
    results = collection.get(where={"filing_id": fid}, limit=1)
    return len(results["ids"]) > 0


def store_chunks(chunks: list[Result], filing: EdgarFiling) -> None:
    """
    Embed and store chunks for a single filing.
    Uses filing_id metadata for deduplication — safe to rerun.
    """
    if not chunks:
        return

    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    fid = _filing_id(filing)

    # Embed in batches of 100
    texts = [c.page_content for c in chunks]
    all_vectors = []
    for i in range(0, len(texts), 100):
        batch = texts[i: i + 100]
        emb = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=batch
        ).data
        all_vectors.extend([e.embedding for e in emb])

    # Build unique IDs using filing_id + chunk index
    existing_count = collection.count()
    ids = [f"{fid}_{existing_count + i}" for i in range(len(chunks))]
    metas = [{**c.metadata, "filing_id": fid} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=all_vectors,
        documents=texts,
        metadatas=metas,
    )
    log.info(
        f"Stored {len(chunks)} chunks for {filing.ticker} "
        f"(collection total: {collection.count()})"
    )


# Main pipeline


def ingest(
    tickers: list[str],
    form_type: str = "10-Q",
    k: int = 1,
    max_chars: int | None = None,
) -> None:
    """
    Full ingestion pipeline for a list of tickers.

    Args:
        tickers:    List of ticker symbols e.g. ["AAPL", "MSFT"]
        form_type:  SEC form type e.g. "10-K", "10-Q"
        k:          Number of filings per ticker to ingest
        max_chars:  Cap extracted characters per filing (default: None)
    """
    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    for ticker in tickers:
        log.info(f"--- Processing {ticker} ---")
        filings = fetch_filings_v2(ticker, form_type=form_type, k=k)

        if not filings:
            log.warning(f"No filings found for {ticker} — skipping")
            continue

        # Filter already-ingested filings
        new_filings = [f for f in filings if not already_ingested(collection, f)]
        if not new_filings:
            log.info(f"{ticker}: all filings already ingested — skipping")
            continue

        log.info(f"{ticker}: {len(new_filings)} new filing(s) to process")

        # Extract text
        new_filings = enrich_filings(new_filings, max_chars=max_chars)

        # Chunk and store
        for filing in new_filings:
            chunks = process_filing(filing)
            store_chunks(chunks, filing)

    log.info("Ingestion complete")
    final = chroma.get_or_create_collection(COLLECTION_NAME)
    log.info(f"Vector store total: {final.count()} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR filings")
    parser.add_argument(
        "--tickers", nargs="+", required=True,
        help="Ticker symbols e.g. AAPL MSFT NVDA"
    )
    parser.add_argument(
        "--form-type", default="10-Q",
        help="SEC form type: 10-K, 10-Q (default: 10-Q)"
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="Number of filings per ticker (default: 1)"
    )
    parser.add_argument(
        "--max-chars", type=int, default=None,
        help="Cap extracted characters per filing (default: no limit)"
    )
    args = parser.parse_args()
    ingest(
        tickers=args.tickers,
        form_type=args.form_type,
        k=args.k,
        max_chars=args.max_chars,
    )