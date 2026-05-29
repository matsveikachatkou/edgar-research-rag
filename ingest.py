"""
ingest.py — SEC EDGAR ingestion pipeline.

Pipeline:
    1. Fetch recent filings from SEC EDGAR full-text search API
    2. Convert PDFs to markdown via Mistral OCR
    3. Chunk documents semantically via LLM
    4. Embed chunks and store in ChromaDB (incremental — no delete on rerun)

Usage:
    uv run python ingest.py --tickers AAPL MSFT --form-type 10-Q
    uv run python ingest.py --tickers NVDA --form-type 10-K --max-pages 30
"""

import argparse
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import requests
from chromadb import PersistentClient
from dotenv import load_dotenv
from litellm import completion
from mistralai.client import Mistral
from openai import OpenAI
from tenacity import retry, wait_exponential

from models.research import Chunk, Chunks, EdgarFiling, Result

load_dotenv(override=True)


# Config

MODEL = "openai/gpt-4.1-mini"
DB_NAME = str(Path(__file__).parent / "edgar_db")
COLLECTION_NAME = "edgar_filings"
EMBEDDING_MODEL = "text-embedding-3-large"
AVERAGE_CHUNK_SIZE = 500
WORKERS = 3
WAIT = wait_exponential(multiplier=1, min=10, max=240)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [ingest] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))


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


def fetch_filings_v2(ticker: str, form_type: str = "10-Q", k: int = 1) -> list[EdgarFiling]:
    """
    Improved filing fetcher using EDGAR company search API.
    Falls back to a direct URL construction approach.
    """
    from datetime import datetime

    log.info(f"Fetching {form_type} for {ticker} via EDGAR search")

    search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form_type}"
    try:
        resp = requests.get(search_url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error(f"EDGAR search failed for {ticker}: {e}")
        return []

    hits = data.get("hits", {}).get("hits", [])
    if not hits:
        log.warning(f"No hits for {ticker} {form_type}")
        return []

    filings = []
    for hit in hits[:k]:
        src = hit.get("_source", {})
        accession = hit.get("_id", "")
        entity_name = src.get("entity_name", ticker)
        file_date = src.get("file_date", "")
        period = src.get("period_of_report", "")
        entity_id = src.get("entity_id", "")

        try:
            filed_dt = datetime.strptime(file_date, "%Y-%m-%d")
        except Exception:
            filed_dt = datetime.utcnow()

        # Build filing index URL
        if entity_id and accession:
            acc_clean = accession.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{entity_id}/{acc_clean}/{accession}-index.htm"
            pdf_url = filing_url  # will be resolved during OCR step
        else:
            filing_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type={form_type}&dateb=&owner=include&count=5"
            pdf_url = None

        filing = EdgarFiling(
            ticker=ticker.upper(),
            company_name=entity_name,
            form_type=form_type,
            filed_at=filed_dt,
            period_of_report=period,
            filing_url=filing_url,
            pdf_url=pdf_url,
        )
        filings.append(filing)
        log.info(f"Resolved: {entity_name} | {form_type} | filed {file_date} | period {period}")

    return filings


def get_primary_doc_url(filing: EdgarFiling) -> str | None:
    """
    Given a filing index URL, fetch the index page and extract
    the primary document (10-K or 10-Q HTML/PDF) URL.
    """
    try:
        resp = requests.get(filing.filing_url, headers=SEC_HEADERS, timeout=15)
        resp.raise_for_status()
        text = resp.text

        # Look for the primary document link in the index
        import re
        # Match .htm or .pdf links that are not the index itself
        patterns = [
            r'href="(/Archives/edgar/data/[^"]+\.htm)"',
            r'href="(/Archives/edgar/data/[^"]+\.pdf)"',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if "index" not in match.lower():
                    url = f"https://www.sec.gov{match}"
                    log.info(f"Primary doc URL: {url}")
                    return url
    except Exception as e:
        log.warning(f"Could not parse filing index for {filing.ticker}: {e}")
    return None


# Step 2 — OCR via Mistral


def ocr_document(url: str, max_pages: int | None = None) -> str:
    """
    Run Mistral OCR on a document URL (HTML or PDF).
    Returns extracted markdown text.
    """
    log.info(f"Running OCR on: {url}")
    try:
        response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": url},
        )
        pages = [page.markdown for page in response.pages]
        if max_pages:
            pages = pages[:max_pages]
        return "\n\n".join(pages)
    except Exception as e:
        log.warning(f"Mistral OCR failed for {url}: {e}")
        return ""


def enrich_filings(
    filings: list[EdgarFiling], max_pages: int | None = None
) -> list[EdgarFiling]:
    """
    Resolve primary doc URL and run OCR for each filing.
    Falls back to filing summary text if OCR fails.
    """
    for filing in filings:
        log.info(f"Enriching {filing.ticker} {filing.form_type}")
        doc_url = get_primary_doc_url(filing)
        if doc_url:
            filing.pdf_url = doc_url
            markdown = ocr_document(doc_url, max_pages=max_pages)
            if markdown:
                filing.document_markdown = markdown
                log.info(
                    f"OCR complete: {filing.ticker} {filing.form_type} "
                    f"({len(markdown):,} chars)"
                )
                continue

        # Fallback: use filing URL directly
        log.warning(
            f"Falling back to direct OCR for {filing.ticker} — "
            "no primary doc found"
        )
        markdown = ocr_document(filing.filing_url, max_pages=max_pages)
        filing.document_markdown = markdown or f"No content extracted for {filing.ticker} {filing.form_type}"

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


@retry(wait=WAIT)
def process_filing(filing: EdgarFiling) -> list[Result]:
    """Chunk a single filing via LLM and return Results."""
    if not filing.document_markdown:
        log.warning(f"No markdown content for {filing.ticker} — skipping chunking")
        return []

    log.info(f"Chunking {filing.ticker} {filing.form_type}")
    messages = _make_messages(filing)
    response = completion(
        model=MODEL,
        messages=messages,
        response_format=Chunks,
    )
    reply = response.choices[0].message.content
    doc_chunks = Chunks.model_validate_json(reply).chunks
    results = [chunk.as_result(filing) for chunk in doc_chunks]
    log.info(f"Created {len(results)} chunks for {filing.ticker}")
    return results


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
    max_pages: int | None = None,
) -> None:
    """
    Full ingestion pipeline for a list of tickers.

    Args:
        tickers:    List of ticker symbols e.g. ["AAPL", "MSFT"]
        form_type:  SEC form type e.g. "10-K", "10-Q"
        k:          Number of filings per ticker to ingest
        max_pages:  Cap OCR pages per filing (useful during development)
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

        # OCR
        new_filings = enrich_filings(new_filings, max_pages=max_pages)

        # Chunk (parallel)
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
        "--max-pages", type=int, default=None,
        help="Cap OCR pages per filing — useful during development"
    )
    args = parser.parse_args()
    ingest(
        tickers=args.tickers,
        form_type=args.form_type,
        k=args.k,
        max_pages=args.max_pages,
    )