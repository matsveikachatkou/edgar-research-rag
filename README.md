# EDGAR Research RAG

A RAG-powered investment research tool that ingests SEC filings and enables financial Q&A, structured recommendations, and retrieval quality evaluation.

Built as a portfolio project demonstrating production-grade LLM application development across the full stack: data ingestion, retrieval, agent architecture, structured output, and evaluation.

---

## What it does

**Research Chat** — Ask questions about ingested companies in natural language. Retrieval is grounded in actual SEC filings with source citations. Supports cross-company comparison and ticker-filtered deep dives.

**On-demand recommendations** — Select a company and specific filing period to generate a structured buy/hold/sell recommendation with rationale, key risks, and key opportunities — all grounded in filing evidence.

**Data Ingestion** — Ingest any SEC-listed company by ticker. Works for any US-listed company via dynamic CIK resolution. Incremental — safe to rerun without duplicates.

**Eval Dashboard** — Measure retrieval quality across three metrics: context precision, answer faithfulness, and answer relevance. LLM-judged, RAGAS-inspired, no external dependencies.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Ingestion layer                                     │
│  EDGAR submissions API → BeautifulSoup → LLM chunker │
│  → ChromaDB (incremental, ticker+period metadata)    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│  Retrieval layer                                     │
│  Query rewrite → dual retrieval → merge → LLM rerank │
│  Optional ticker + period filters                    │
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

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

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

**Natural next steps:**
- Scheduled ingestion — GitHub Actions or cron running `ingest.py` weekly to keep the corpus current
- Backtesting — compare recommendations generated from past filings against subsequent price performance using `yfinance`
- European filing support — ESMA ESEF database or a data provider like Refinitiv
- Cross-encoder reranker — replace the LLM reranker with a local `sentence-transformers` cross-encoder for faster, cheaper reranking

---

## Related projects

- [`equity-lens`](https://github.com/matsveikachatkou/equity-lens) — 6-agent CrewAI investment research pipeline
- [`agentic_ai_course`](https://github.com/matsveikachatkou/agentic_ai_course) — AI agents course covering CrewAI, LangChain, LangGraph, AutoGen, MCP