# EDGAR Research RAG

A RAG-powered investment research tool that ingests SEC filings and enables financial Q&A, structured recommendations, and retrieval quality evaluation.

Built as a portfolio project demonstrating production-grade LLM application development across the full stack: data ingestion, retrieval, agent architecture, structured output, and evaluation.

---

## What it does

**Research Chat** — Ask questions about ingested companies in natural language. Retrieval is grounded in actual SEC filings with source citations. Supports cross-company comparison and ticker-filtered deep dives.

**On-demand recommendations** — Select a company and specific filing period to generate a structured buy/hold/sell recommendation with rationale, key risks, and key opportunities — all grounded in filing evidence.

**Data Ingestion** — Ingest any SEC-listed company by ticker. Works for any US-listed company via dynamic CIK resolution. Incremental — safe to rerun without duplicates.

**Eval Dashboard** — Measure retrieval quality across three metrics: context precision, answer faithfulness, and answer relevance. LLM-judged, RAGAS-inspired, no external dependencies.

**Structured + unstructured fusion** — Quantitative questions (revenue, margins, EPS, cash) are answered using exact XBRL figures pulled directly from the SEC EDGAR Frames API, eliminating hallucination risk on financial metrics. Narrative context (management commentary, risk factors, segment performance) comes from the RAG pipeline. Both sources are fused in a single answer with full citations.

**Earnings press release ingestion** — Ingests 8-K earnings press releases  (Exhibit 99.1) alongside 10-Q/10-K filings. Press releases are available ~1 week before the full 10-Q filing, providing earlier access to results. CEO/CFO commentary, non-GAAP metrics, and forward guidance are retrieved alongside audited GAAP figures from the structured XBRL layer.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Ingestion layer                                     │
│  EDGAR submissions API → BeautifulSoup → LLM chunker │
│  10-Q/10-K: full filings                             │
│  8-K: Exhibit 99.1 earnings press releases only      │
│       (item 2.02 filter + exhibit directory scan)    │
│  → ChromaDB (incremental, ticker+period metadata)    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Structured data layer                               │
│  SEC XBRL Frames API → taxonomy fallback resolver    │
│  → FinancialSnapshot (8 metrics) → disk cache        │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Retrieval layer                                     │
│  Query rewrite → dual retrieval → merge → LLM rerank │
│  XBRL context prepended when ticker identified       │
│  Adaptive k: ticker-filtered (8) vs broad (15)       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Agent layer                                         │
│  ScannerAgent → ResearchAgent → RecommendAgent       │
│  Pydantic structured output · JSON memory persistence│
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Eval layer                                          │
│  Context precision · Faithfulness · Relevance        │
│  LLM judge · per-question scores · JSON export       │
└─────────────────────────────────────────────────────┘
```

**Key technical choices:**

- **BeautifulSoup over OCR** — Modern EDGAR filings are HTML, not scanned PDFs. Direct extraction is faster, cheaper, and more reliable than OCR.
- **Async LLM chunking** — `asyncio` with semaphore-controlled concurrency replaces sequential batching. 6-8x faster ingestion for large filings.
- **Dual-query retrieval** — Each question is both used directly and rewritten for retrieval. The two result sets are merged before reranking, improving recall.
- **LLM reranker** — Retrieved chunks are reranked by relevance before being passed to the answer model. Improves precision over pure embedding similarity.
- **Pydantic structured output** — All agent outputs (`Recommendation`, `Chunks`, `RankOrder`) are validated Pydantic models. No string parsing, no pipe-splitting.
- **Dynamic CIK resolution** — Any US-listed ticker resolves to a CIK via SEC's public mapping file. No hardcoded lookup tables.
- **XBRL structured data fusion** — The SEC EDGAR Frames API exposes every reported financial fact as structured XBRL data. Rather than relying on RAG to extract figures from flattened HTML tables (which loses row/column relationships), key metrics (revenue, operating income, net income, EPS, cash, total assets, operating cash flow, gross profit) are fetched directly via a taxonomy fallback resolver that handles tag variations across filers. Snapshots are cached to disk to avoid redundant API calls. Results are prepended to the LLM context when a ticker is identified, giving the model ground-truth numbers alongside filing narrative.
- **Adaptive retrieval** — Retrieval pool size scales with query scope: ticker-filtered queries use k=8 for cost efficiency; unfiltered cross-company queries use k=15 to ensure balanced company representation in results.
- **8-K earnings release extraction** — The scanner filters 8-K filings by SEC item code 2.02 (Results of Operations) to ingest only earnings releases, not unrelated corporate events. Exhibit 99.1 (the press release) is discovered by scanning the filing index directory for `ex99` filename patterns, since the primary document is always just the cover page.
- **GAAP vs non-GAAP source labeling** — The system prompt explicitly instructs the LLM to distinguish between XBRL-sourced GAAP figures and 8-K press release non-GAAP figures. Sources are always cited by form type so the reader knows which standard applies.

---

## Setup

**Prerequisites:** Python 3.11, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/matsveikachatkou/edgar-research-rag.git
cd edgar-research-rag
uv sync
```

> **Intel Mac users:** The project requires Python 3.11 due to `onnxruntime` compatibility.
> Ensure your `pyproject.toml` contains:
> ```toml
> [tool.uv]
> required-environments = [ "sys_platform == 'darwin' and platform_machine == 'x86_64'",]
> ```
> Then run `uv sync` again.

Copy `.env.example` to `.env` and add your OpenAI API key:
```bash
cp .env.example .env
```
Then edit `.env` and fill in your `OPENAI_API_KEY`.

> OpenAI is the only required API. SEC EDGAR is public and requires no authentication.

---

## Quick start

Seed the vector store with three demo companies (AAPL, TSLA, MSFT):

```bash
uv run python scripts/seed.py
```

This takes approximately 15-20 minutes and ingests the latest 10-Q for each company.

Then launch the UI:

```bash
uv run python app.py
```

Opens at `http://localhost:7860`.

---

## Usage

**Ingest any ticker:**
```bash
# Latest 10-Q for Apple and Tesla
uv run python ingest.py --tickers AAPL TSLA --form-type 10-Q --k 1

# Last 4 quarters of 10-K for Microsoft
uv run python ingest.py --tickers MSFT --form-type 10-K --k 4
```

**Run evaluation from CLI:**
```bash
uv run python eval/evaluate.py --ticker AAPL
uv run python eval/evaluate.py --ticker AAPL --output eval_results.json
```

**Run the automated research pipeline from CLI:**
```bash
uv run python research_framework.py --tickers AAPL TSLA --form-type 10-Q
uv run python research_framework.py --summary
```

---

## Project structure

```
edgar-research-rag/
├── ingest.py                 # EDGAR fetch, HTML extraction, async LLM chunking, ChromaDB storage
├── answer.py                 # RAG retrieval: query rewrite, dual fetch, merge, rerank, answer
├── research_framework.py     # CLI orchestrator: scan → ingest → research → recommend
├── app.py                    # Gradio UI: Research Chat, Data Ingestion, Eval Dashboard
├── scripts/
│   └── seed.py               # Ingest demo companies for quick start
├── agents/
│   ├── agent.py              # Base agent class with logging
│   ├── scanner_agent.py      # Scans EDGAR submissions API for new filings
│   ├── research_agent.py     # RAG-powered research summary generation
│   └── recommend_agent.py    # Structured buy/hold/sell recommendation
├── eval/
│   └── evaluate.py           # Context precision, faithfulness, relevance scoring
└── models/
    └── research.py           # Pydantic models: EdgarFiling, Chunk, Result, Recommendation, EvalResult
```

---

## Evaluation results

Evaluated on Apple Inc. (AAPL) 10-Q for Q2 2026, 79 chunks ingested:

| Metric | Score |
|---|---|
| Context precision | 0.53 |
| Answer faithfulness | 0.67 |
| Answer relevance | 1.00 |
| **Overall** | **0.73** |

Context precision improved from 0.10 (11 chunks, 15k chars) to 0.53 (79 chunks, full filing) after switching to async batched ingestion without a character cap. Answer relevance is consistently 1.00 across all question types.

---

## Limitations and next steps

**Current limitations:**
- SEC EDGAR only — covers US-listed companies. European filings (ESMA/ESEF) require a different data source; the architecture is source-agnostic so the scanner and ingestion layer can be swapped.
- LLM-judged evaluation — confidence scores and eval metrics are model-generated, not backtested against ground truth. A golden dataset of question/answer pairs from filings would make evaluation more rigorous.
- Aggregation questions — RAG retrieves by semantic similarity, not by structured data. Questions like "which company has the highest margin?" work by chance, not by design. A structured extraction layer on top would handle this reliably.
- XBRL coverage — the structured data layer covers 8 core metrics via US-GAAP taxonomy. Companies using non-standard tags beyond the fallback list will return partial snapshots; the RAG layer still functions normally as fallback. European filers (IFRS taxonomy) are not covered.
- Cross-company retrieval — unfiltered queries rank by embedding similarity, which can favour one company over another. Per-ticker retrieval with guaranteed minimum chunks per mentioned company is a natural next step.
- 8-K coverage — only earnings press releases (item 2.02) are ingested. Other 8-K events (acquisitions, leadership changes, guidance updates) are filtered out. Non-GAAP figures in press releases are not standardized across companies — the LLM extracts them from text rather than structured data.

**Natural next steps:**
- Scheduled ingestion — GitHub Actions or cron running `ingest.py` weekly to keep the corpus current
- Backtesting — compare recommendations generated from past filings against subsequent price performance using `yfinance`
- European filing support — ESMA ESEF database or a data provider like Refinitiv
- Cross-encoder reranker — replace the LLM reranker with a local `sentence-transformers` cross-encoder for faster, cheaper reranking

---

## Related projects

- [`equity-lens`](https://github.com/matsveikachatkou/equity-lens) — 6-agent CrewAI investment research pipeline