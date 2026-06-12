# EDGAR Research RAG

A RAG-powered investment research tool that ingests SEC filings and enables financial Q&A, structured recommendations, point-in-time backtesting, and retrieval quality evaluation.

Built as a portfolio project demonstrating production-grade LLM application development across the full stack: data ingestion, retrieval, agent architecture, structured output, backtesting, and evaluation.

---

## What it does

**Research Chat** — Ask questions about ingested companies in natural language. Retrieval is grounded in actual SEC filings with source citations. Supports cross-company comparison and ticker-filtered deep dives.

**On-demand recommendations** — Select a company and specific filing period to generate a structured overweight/neutral/underweight recommendation with rationale, key risks, and key opportunities — all grounded in filing evidence.

**Point-in-time backtester** — Walks a company's filing timeline chronologically, generating an investment signal at each filing event using only data available at that moment (no look-ahead bias). Dual-signal architecture: fast signals from 8-K press releases, conviction signals from 10-Q/10-K that cross-reference prior earnings claims against audited figures. Forward returns (30/60/90 day) measured against SPY.

**Data Ingestion** — Ingest any SEC-listed company by ticker. Works for any US-listed company via dynamic CIK resolution. Incremental — safe to rerun without duplicates.

**Eval Dashboard** — Measure retrieval quality across three metrics: context precision, answer faithfulness, and answer relevance. LLM-judged, RAGAS-inspired, no external dependencies.

**Structured + unstructured fusion** — Quantitative questions (revenue, margins, EPS, cash) are answered using exact XBRL figures pulled directly from the SEC EDGAR Frames API, eliminating hallucination risk on financial metrics. Narrative context (management commentary, risk factors, segment performance) comes from the RAG pipeline. Both sources are fused in a single answer with full citations, and the system prompt enforces that audited GAAP figures are always attributed to the 10-Q/10-K, never the 8-K.

**Earnings press release ingestion** — Ingests 8-K earnings press releases (Exhibit 99.1) alongside 10-Q/10-K filings. Press releases are available ~1 day before the full 10-Q filing, providing earlier access to results. CEO/CFO commentary, non-GAAP metrics, and forward guidance are retrieved alongside audited GAAP figures from the structured XBRL layer.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐

│  Ingestion layer                                     │

│  EDGAR submissions API → BeautifulSoup → LLM chunker │

│  10-Q/10-K: full filings                             │

│  8-K: Exhibit 99.1 earnings press releases only      │

│       (item 2.02 filter + exhibit directory scan)    │

│  → ChromaDB (incremental, ticker+filing_date metadata)│

└────────────────────┬────────────────────────────────┘

                     │

┌────────────────────▼────────────────────────────────┐

│  Structured data layer                               │

│  SEC XBRL Frames API → taxonomy fallback resolver    │

│  PIT-aware: fallback filtered by pit_cutoff date     │

│  → FinancialSnapshot (8 metrics) → disk cache        │

└────────────────────┬────────────────────────────────┘

                     │

┌────────────────────▼────────────────────────────────┐

│  Retrieval layer                                     │

│  Query rewrite (cached) → dual retrieval → merge     │

│  → rerank (skipped for small pools)                  │

│  PIT mode: filing_date_lte + 90-day filing_date_gte  │

│  window — current filing + paired prior filing only  │

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

│  Backtest layer                                      │

│  Event-driven chronological walk over filing timeline│

│  Fast signal (8-K) / Conviction signal (10-Q, 10-K)  │

│  Forward returns vs SPY (30/60/90d) · incremental    │

│  save + resume                                       │

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
- **LLM reranker** — Retrieved chunks are reranked by relevance before being passed to the answer model when the candidate pool exceeds the final selection size. Small pools skip reranking entirely to reduce latency.
- **Pydantic structured output** — All agent outputs (`Recommendation`, `Chunks`, `RankOrder`) are validated Pydantic models. No string parsing, no pipe-splitting.
- **Dynamic CIK resolution** — Any US-listed ticker resolves to a CIK via SEC's public mapping file. No hardcoded lookup tables.
- **XBRL structured data fusion** — The SEC EDGAR Frames API exposes every reported financial fact as structured XBRL data. Rather than relying on RAG to extract figures from flattened HTML tables (which loses row/column relationships), key metrics (revenue, operating income, net income, EPS, cash, total assets, operating cash flow, gross profit) are fetched directly via a taxonomy fallback resolver that handles tag variations across filers. Snapshots are cached to disk to avoid redundant API calls. Results are prepended to the LLM context when a ticker is identified, giving the model ground-truth numbers alongside filing narrative.
- **Point-in-time (PIT) retrieval** — Rather than filtering by the period a document covers, retrieval filters by `filing_date` — when the document was actually available to the market. A 90-day lookback window from the cutoff date mirrors how an analyst works: the current filing plus the most recently paired filing, nothing stale. This means a 10-Q conviction signal sees both the audited filing and its paired 8-K press release, enabling cross-referencing of management claims against audited figures. XBRL snapshots use the same `pit_cutoff` to ensure the fallback to a prior period never looks ahead in time.
- **Adaptive retrieval** — Retrieval pool size scales with query scope: ticker-filtered queries use k=8 for cost efficiency; unfiltered cross-company queries use k=15 to ensure balanced company representation in results.
- **8-K earnings release extraction** — The scanner filters 8-K filings by SEC item code 2.02 (Results of Operations) to ingest only earnings releases, not unrelated corporate events. Exhibit 99.1 (the press release) is discovered by parsing the SEC EDGAR filing index table and matching the "EX-99.1" document type, since the primary document is always just the cover page and filename patterns are inconsistent across filers.
- **GAAP vs non-GAAP source labeling** — The system prompt explicitly instructs the LLM to distinguish between XBRL-sourced GAAP figures and 8-K press release non-GAAP figures, and enforces that core GAAP metrics (revenue, EPS, net income, etc.) are always cited as the audited 10-Q/10-K — never the 8-K, even when the 8-K reports the same number.

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

> OpenAI is the only required API. SEC EDGAR and yfinance are public and require no authentication.

---

## Quick start

Seed the vector store with three demo companies (AAPL, TSLA, MSFT):

```bash
uv run python scripts/seed.py
```

This takes approximately 15-20 minutes and ingests the latest 10-Q and 8-K for each company.

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

# Last 4 quarters of 10-Q for Microsoft
uv run python ingest.py --tickers MSFT --form-type 10-Q --k 4

# Last 4 earnings press releases for Apple
uv run python ingest.py --tickers AAPL --form-type 8-K --k 4
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

**Run the point-in-time backtester:**
```bash
uv run python research_backtester.py --tickers AAPL
```

Walks every 10-Q/10-K/8-K filing for AAPL chronologically, generating a signal
at each event using only data available at that point in time, then measures
forward returns vs SPY. Results are saved incrementally to
`backtest_results.json` — safe to interrupt and resume; already-processed
events are skipped on rerun.

---

## Project structure

```
edgar-research-rag/

├── ingest.py                 # EDGAR fetch, HTML extraction, async LLM chunking, ChromaDB storage

├── answer.py                 # RAG retrieval: query rewrite, dual fetch, merge, rerank, answer

├── research_framework.py     # CLI orchestrator: scan → ingest → research → recommend

├── research_backtester.py    # Point-in-time backtester: event-driven signal generation + forward returns

├── xbrl.py                    # SEC XBRL Frames API client, taxonomy fallback resolver, disk cache

├── app.py                    # Gradio UI: Research Chat, Data Ingestion, Eval Dashboard

├── scripts/

│   └── seed.py               # Ingest demo companies for quick start

├── agents/

│   ├── agent.py              # Base agent class with logging

│   ├── scanner_agent.py      # Scans EDGAR submissions API for new filings

│   ├── research_agent.py     # RAG-powered research summary generation

│   └── recommend_agent.py    # Structured overweight/neutral/underweight recommendation

├── eval/

│   └── evaluate.py           # Context precision, faithfulness, relevance scoring

└── models/

└── research.py           # Pydantic models: EdgarFiling, Chunk, Result, Recommendation, FinancialSnapshot, EvalResult
```

---

---

## Point-in-time backtest results

Initial backtest on AAPL, 9 filing events spanning May 2025 – May 2026
(7 events with complete 30-day forward return windows at time of writing):

| Metric | Result |
|---|---|
| Hit rate (30d) | 8/9 = 89% |
| Avg alpha (30d) | +4.8% |
| Fast signal (8-K) avg alpha | +6.4% |
| Conviction signal (10-Q/10-K) avg alpha | +3.5% |

**Methodology:**
- Timeline built from all ingested filings for the ticker, sorted by filing date
- At each event, retrieval and XBRL are restricted to data available as of
  that filing's date (`filing_date_lte`), with a 90-day lookback window
- 8-K events generate a "fast signal" focused on earnings surprise, revenue
  growth, and management tone from the press release
- 10-Q/10-K events generate a "conviction signal" that additionally
  cross-references the paired 8-K's claims against audited figures
- Forward returns measured at 30/60/90 days vs SPY

**Important caveat:** this is a single-ticker backtest over a bull-market
period with every signal landing overweight — it demonstrates that the
infrastructure and PIT discipline work correctly end-to-end, not that the
signal has statistical edge. A meaningful evaluation would require 3-5
tickers across 2+ years with a mix of market regimes, which is primarily
an ingestion-cost question rather than an architectural one.

---

## Evaluation results

Evaluated on Apple Inc. (AAPL), 8 default questions across 10-Q and 8-K filings:

| Metric | v1 | v2 |
|---|---|---|
| Context precision | 0.53 | 0.75 |
| Answer faithfulness | 0.67 | 0.91 |
| Answer relevance | 1.00 | 1.00 |
| **Overall** | **0.73** | **0.89** |

**v1 baseline:** 79 chunks, 10-Q only, eval judge truncated at 400 chars.

**v2 improvements:**
- Truncation removed from eval judge, reranker, and recommendation agent
- 8-K earnings press release chunks added (CEO/CFO commentary, non-GAAP context)
- XBRL structured data block injected for quantitative questions
- Adaptive retrieval (k=8 ticker-filtered, k=15 cross-company)

Notable gaps: context precision is low for forward guidance questions (Apple does not provide formal guidance in filings) and faithfulness is reduced where XBRL cash figures differ from the broader cash + investments figure in filing text.

---

## Limitations and next steps

**Current limitations:**
- SEC EDGAR only — covers US-listed companies. European filings (ESMA/ESEF) require a different data source; the architecture is source-agnostic so the scanner and ingestion layer can be swapped.
- LLM-judged evaluation — confidence scores and eval metrics are model-generated, not backtested against ground truth. A golden dataset of question/answer pairs from filings would make evaluation more rigorous.
- Aggregation questions — RAG retrieves by semantic similarity, not by structured data. Questions like "which company has the highest margin?" work by chance, not by design. A structured extraction layer on top would handle this reliably.
- XBRL coverage — the structured data layer covers 8 core metrics via US-GAAP taxonomy. Companies using non-standard tags beyond the fallback list will return partial snapshots; the RAG layer still functions normally as fallback. European filers (IFRS taxonomy) are not covered.
- Cross-company retrieval — unfiltered queries rank by embedding similarity, which can favour one company over another. Per-ticker retrieval with guaranteed minimum chunks per mentioned company is a natural next step.
- 8-K coverage — only earnings press releases (item 2.02) are ingested. Other 8-K events (acquisitions, leadership changes, guidance updates) are filtered out. Non-GAAP figures in press releases are not standardized across companies — the LLM extracts them from text rather than structured data.
- Backtest scope — currently single-ticker (AAPL) over a single bull-market period. Not statistically significant on its own.

**Natural next steps:**
- Multi-ticker, multi-period backtesting — extend the PIT backtester to MSFT, NVDA, TSLA across 2+ years to test whether signal confidence correlates with forward returns, and whether the model discriminates (currently every signal is overweight)
- Scheduled ingestion — GitHub Actions or cron running `ingest.py` weekly to keep the corpus current
- European filing support — ESMA ESEF database or a data provider like Refinitiv
- Cross-encoder reranker — replace the LLM reranker with a local `sentence-transformers` cross-encoder for faster, cheaper reranking

---

## Related projects

- [`equity-lens`](https://github.com/matsveikachatkou/equity-lens) — 6-agent CrewAI investment research pipeline