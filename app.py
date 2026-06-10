"""
app.py — Gradio chat interface for the EDGAR research RAG system.

Three tabs:
    1. Research Chat  — Q&A over ingested SEC filings with source citations
    2. Pipeline       — Run the full scan → ingest → research → recommend pipeline
    3. Eval Dashboard — Run and display RAG evaluation metrics

Usage:
    uv run python app.py
"""

import json
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

from answer import answer_question
from eval.evaluate import DEFAULT_QUESTIONS, print_eval_report, run_evaluation
from models.research import ResearchOpportunity
from research_framework import MEMORY_PATH, ResearchFramework


# Helpers


REC_COLORS = {"overweight": "🟢", "neutral": "🟡", "underweight": "🔴"}


def format_sources(chunks) -> str:
    """Format retrieved chunks as a markdown reference list."""
    if not chunks:
        return "*No sources retrieved.*"

    lines = []
    seen = set()
    for chunk in chunks:
        meta = chunk.metadata
        ticker = meta.get("ticker", "")
        company = meta.get("company_name", ticker)
        form = meta.get("form_type", "")
        period = meta.get("period_of_report", "")
        url = meta.get("filing_url", "")
        snippet = chunk.page_content[:200].replace("\n", " ")

        key = f"{ticker}_{form}_{period}"
        if key not in seen:
            seen.add(key)
            lines.append(
                f"**[{company} ({ticker}) — {form} {period}]({url})**\n\n"
                f"> {snippet}...\n\n---"
            )

    return "\n\n".join(lines) if lines else "*No sources retrieved.*"


def load_memory() -> list[ResearchOpportunity]:
    """Load research memory from disk."""
    if not MEMORY_PATH.exists():
        return []
    try:
        with open(MEMORY_PATH) as f:
            return [ResearchOpportunity(**item) for item in json.load(f)]
    except Exception:
        return []


def format_memory(memory: list[ResearchOpportunity]) -> str:
    """Format research memory as a markdown summary."""
    if not memory:
        return "*No research results yet. Run the pipeline first.*"

    lines = []
    for opp in sorted(memory, key=lambda x: x.researched_at, reverse=True):
        icon = REC_COLORS.get(opp.recommendation, "⚪")
        lines.append(
            f"### {icon} {opp.company_name} ({opp.ticker}) — {opp.form_type}\n\n"
            f"**Period:** {opp.period_of_report or opp.filed_at[:10]}  |  "
            f"**Researched:** {opp.researched_at[:10]}\n\n"
            f"**Recommendation:** `{opp.recommendation.upper()}` "
            f"({opp.confidence:.0%} confidence)\n\n"
            f"**Rationale:** {opp.rationale}\n\n"
            f"**Key risks:** {' · '.join(opp.key_risks) if opp.key_risks else 'N/A'}\n\n"
            f"**Key opportunities:** {' · '.join(opp.key_opportunities) if opp.key_opportunities else 'N/A'}\n\n"
            f"---"
        )
    return "\n\n".join(lines)


def get_coverage_table():
    """Return coverage universe as a dataframe."""
    try:
        from chromadb import PersistentClient
        from pathlib import Path
        import pandas as pd
        db = PersistentClient(path=str(Path("edgar_db")))
        col = db.get_or_create_collection("edgar_filings")
        if col.count() == 0:
            return pd.DataFrame(columns=["Ticker", "Company", "Form", "Period", "Chunks"])
        results = col.get()
        seen = {}
        for meta in results["metadatas"]:
            key = f"{meta.get('ticker')}_{meta.get('form_type')}_{meta.get('period_of_report')}"
            if key not in seen:
                seen[key] = {
                    "Ticker": meta.get("ticker", ""),
                    "Company": meta.get("company_name", ""),
                    "Form": meta.get("form_type", ""),
                    "Period": meta.get("period_of_report", ""),
                    "Chunks": 0,
                }
            seen[key]["Chunks"] += 1
        return pd.DataFrame(seen.values())
    except Exception:
        return pd.DataFrame(columns=["Ticker", "Company", "Form", "Period", "Chunks"])

def get_periods_for_ticker(ticker: str) -> list[str]:
    """Get available filing periods for a ticker from the vector store."""
    try:
        from chromadb import PersistentClient
        from pathlib import Path
        db = PersistentClient(path=str(Path("edgar_db")))
        col = db.get_or_create_collection("edgar_filings")
        if col.count() == 0:
            return []
        results = col.get(where={"ticker": ticker.upper()})
        periods = sorted(set(
            m.get("period_of_report", "")
            for m in results["metadatas"]
            if m.get("period_of_report")
        ), reverse=True)
        return periods
    except Exception:
        return []


def get_available_tickers() -> list[str]:
    """Get list of ingested tickers."""
    try:
        from chromadb import PersistentClient
        from pathlib import Path
        db = PersistentClient(path=str(Path("edgar_db")))
        col = db.get_or_create_collection("edgar_filings")
        if col.count() == 0:
            return []
        results = col.get()
        return sorted(set(m.get("ticker", "") for m in results["metadatas"] if m.get("ticker")))
    except Exception:
        return []


def generate_recommendation(ticker: str, period: str) -> str:
    """Generate a recommendation for a specific ticker and filing period."""
    if not ticker or not period:
        return "Please select a ticker and period."
    try:
        from agents.research_agent import ResearchAgent
        from agents.recommend_agent import RecommendAgent
        researcher = ResearchAgent()
        recommender = RecommendAgent()

        summary, chunks = researcher.research(
            ticker=ticker,
            focus="investment outlook, revenue trends, key risks",
            period=period,
        )
        if not chunks:
            return f"No data found for {ticker} {period}. Please ingest this filing first."

        rec = recommender.recommend(ticker=ticker, summary=summary, chunks=chunks)

        rec_label = {
                  "overweight": '<span style="color: #22c55e; font-weight: bold;">OVERWEIGHT</span>',
                  "neutral": '<span style="color: #eab308; font-weight: bold;">NEUTRAL</span>',
                  "underweight": '<span style="color: #ef4444; font-weight: bold;">UNDERWEIGHT</span>',
              }.get(rec.recommendation, rec.recommendation.upper())
        return f"""## {ticker} — {period}

**Recommendation:** {rec_label} ({rec.confidence:.0%} confidence)

**Rationale:** {rec.rationale}

**Key risks:**
{chr(10).join(f'- {r}' for r in rec.key_risks)}

**Key opportunities:**
{chr(10).join(f'- {o}' for o in rec.key_opportunities)}

---
*Based on {len(chunks)} chunks from {ticker} {period} filing*"""
    except Exception as e:
        return f"Error generating recommendation: {e}"


def send_recommendation_to_chat(rec_text: str, rec_ticker_val: str, history: list):
    if not rec_text or rec_text.startswith("*Select"):
        return history, ""
    history = (history or []) + [
        {"role": "assistant", "content": f"**Recommendation loaded:**\n\n{rec_text}\n\nYou can now ask follow-up questions about this recommendation."}
    ]
    return history, rec_ticker_val


# Tab 1 — Research Chat


def chat(user_message: str, ticker_filter: str, history: list):
    if not user_message.strip():
        return history, "*Ask a question to see sources.*"

    llm_history = []
    for m in (history or []):
        if isinstance(m, dict):
            llm_history.append({"role": m["role"], "content": m["content"]})
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            if m[0]:
                llm_history.append({"role": "user", "content": m[0]})
            if m[1]:
                llm_history.append({"role": "assistant", "content": m[1]})

    ticker = ticker_filter.strip().upper() or None
    answer, chunks = answer_question(user_message, llm_history, ticker=ticker)
    sources_md = format_sources(chunks)

    history = (history or []) + [{"role": "user", "content": user_message}, {"role": "assistant", "content": answer}]
    return history, sources_md


# Tab 2 — Pipeline


def run_ingestion(tickers_input: str, form_type: str, k: int):
    """Ingest filings into the vector store."""
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        yield "Warning: Please enter at least one ticker.", get_coverage_table()
        return

    yield f"Starting ingestion for {', '.join(tickers)}...\n", get_coverage_table()

    try:
        from ingest import ingest
        ingest(
            tickers=tickers,
            form_type=form_type,
            k=int(k) if k else 1,
        )
        yield f"Ingestion complete — {', '.join(tickers)} added to vector store.\n", get_coverage_table()
    except Exception as e:
        yield f"Ingestion error: {e}\n", get_coverage_table()


# Tab 3 — Eval Dashboard


def run_eval(ticker_input: str, custom_questions: str):
    """Run evaluation and return a formatted markdown report."""
    ticker = ticker_input.strip().upper() or None

    if custom_questions.strip():
        questions = [q.strip() for q in custom_questions.strip().split("\n") if q.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    if ticker:
        questions = [q.replace("the company", ticker) for q in questions]

    yield f"Evaluating {len(questions)} question(s)...\n"

    try:
        results = run_evaluation(questions, ticker=ticker)

        if not results:
            yield "No results — make sure filings are ingested first."
            return

        # Build markdown report
        lines = [
            "## RAG Evaluation Report\n",
            f"| Question | Precision | Faithfulness | Relevance | Overall |",
            f"|---|---|---|---|---|",
        ]
        for r in results:
            q = r.question[:50] + "..." if len(r.question) > 50 else r.question
            lines.append(
                f"| {q} | {r.context_precision:.2f} | "
                f"{r.answer_faithfulness:.2f} | "
                f"{r.answer_relevance:.2f} | "
                f"**{r.overall:.2f}** |"
            )

        avg_p = sum(r.context_precision for r in results) / len(results)
        avg_f = sum(r.answer_faithfulness for r in results) / len(results)
        avg_r = sum(r.answer_relevance for r in results) / len(results)
        avg_o = sum(r.overall for r in results) / len(results)

        lines.append(
            f"| **AVERAGE** | **{avg_p:.2f}** | **{avg_f:.2f}** | "
            f"**{avg_r:.2f}** | **{avg_o:.2f}** |"
        )
        lines.append(
            f"\n**Metrics:** Context Precision = are retrieved chunks relevant? · "
            f"Faithfulness = does the answer stay within context? · "
            f"Relevance = does the answer address the question?"
        )

        yield "\n".join(lines)
    except Exception as e:
        yield f"Evaluation error: {e}"


# Build UI


with gr.Blocks(title="EDGAR Research RAG") as app:
    gr.Markdown("#EDGAR Research RAG")
    gr.Markdown(
        "RAG-powered investment research over SEC filings. "
        "Ingest filings, ask questions, generate recommendations, evaluate quality."
    )

    with gr.Tabs():
    
        # Tab 1: Research Chat
        with gr.Tab("Research Chat"):
            with gr.Row():
                coverage_table = gr.Dataframe(
                    value=get_coverage_table(),
                    label="Coverage universe — filings available in the vector store",
                    interactive=False,
                    wrap=True,
                )
                refresh_btn = gr.Button("Refresh", scale=0, min_width=100)
            refresh_btn.click(
                fn=get_coverage_table,
                outputs=[coverage_table],
            )
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=450)
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask about any ingested company...",
                            show_label=False,
                            scale=5,
                        )
                        ticker_filter = gr.Textbox(
                            placeholder="Ticker (clear for cross-company search)",
                            show_label=False,
                            scale=4,
                            min_width=150,
                        )
                    clear = gr.ClearButton([msg, chatbot])
                with gr.Column(scale=2):
                    gr.Markdown("### Retrieved Sources")
                    sources_display = gr.Markdown(
                        "*Sources will appear here after you ask a question.*"
                    )
            def respond(user_message, ticker_filter, history):
                history, sources = chat(user_message, ticker_filter, history or [])
                return "", history, sources
            msg.submit(
                respond,
                [msg, ticker_filter, chatbot],
                [msg, chatbot, sources_display],
            )

            gr.Markdown("---")
            gr.Markdown("### On-demand recommendation")
            gr.Markdown("Generate a structured recommendation from a specific filing already in the vector store.")
            with gr.Row():
                rec_ticker = gr.Dropdown(
                    choices=get_available_tickers(),
                    label="Ticker",
                    scale=1,
                )

                initial_ticker = get_available_tickers()

                rec_period = gr.Dropdown(
                    choices=get_periods_for_ticker(initial_ticker[0]) if initial_ticker else [],
                    label="Filing period",
                    scale=1,
                    value=get_periods_for_ticker(initial_ticker[0])[0] if initial_ticker else None,
                )
                rec_btn = gr.Button(
                    "Generate Recommendation",
                    variant="primary",
                    scale=1,
                )
            rec_status = gr.Markdown("")
            rec_output = gr.Markdown(
                "*Select a ticker and period, then click Generate Recommendation.*"
            )
            send_to_chat_btn = gr.Button("Send to chat", variant="secondary")

            def update_periods(ticker):
              periods = get_periods_for_ticker(ticker)
              return gr.update(choices=periods, value=periods[0] if periods else None)

            rec_ticker.change(
                update_periods,
                inputs=[rec_ticker],
                outputs=[rec_period],
            )
            refresh_btn.click(
                fn=lambda: gr.update(choices=get_available_tickers()),
                outputs=[rec_ticker],
            )
            rec_btn.click(
                fn=lambda: (gr.Button(interactive=False), "Generating recommendation..."),
                outputs=[rec_btn, rec_status],
            ).then(
                generate_recommendation,
                inputs=[rec_ticker, rec_period],
                outputs=[rec_output],
            ).then(
                fn=lambda: (gr.Button(interactive=True), ""),
                outputs=[rec_btn, rec_status],
            )
            send_to_chat_btn.click(
                send_recommendation_to_chat,
                inputs=[rec_output, rec_ticker, chatbot],
                outputs=[chatbot, ticker_filter],
            )

       # Tab 2: Data Ingestion
        with gr.Tab("Data Ingestion"):
            gr.Markdown(
                "Add new SEC filings to the vector store. Once ingested, use Research Chat for Q&A and recommendations."
            )
            with gr.Row():
                tickers_input = gr.Textbox(
                    label="Tickers (comma-separated)",
                    placeholder="AAPL, MSFT, NVDA",
                    scale=3,
                )
                form_type_input = gr.Dropdown(
                    choices=["10-Q", "10-K"],
                    value="10-Q",
                    label="Form type",
                    scale=1,
                )
                k_input = gr.Number(
                    label="Filings per ticker",
                    value=1,
                    scale=1,
                )

            run_btn = gr.Button("Ingest Filings", variant="primary")
            pipeline_status = gr.Textbox(
                label="Status", lines=3, interactive=False
            )
            ingestion_results = gr.Dataframe(
                value=get_coverage_table(),
                label="Coverage universe after ingestion",
                interactive=False,
                wrap=True,
            )

            run_btn.click(
                fn=lambda: gr.Button(interactive=False),
                outputs=[run_btn],
            ).then(
                run_ingestion,
                inputs=[tickers_input, form_type_input, k_input],
                outputs=[pipeline_status, ingestion_results],
            ).then(
                fn=lambda: gr.Button(interactive=True),
                outputs=[run_btn],
            )

        # Tab 3: Eval Dashboard
        with gr.Tab("Eval Dashboard"):
            gr.Markdown(
                "Evaluate RAG quality: context precision, answer faithfulness, answer relevance."
            )
            with gr.Row():
                eval_ticker = gr.Textbox(
                    label="Ticker filter (blank = all ingested)",
                    placeholder="AAPL",
                    scale=1,
                )
                eval_questions = gr.Textbox(
                    label="Custom questions (one per line, blank = defaults)",
                    placeholder="What was revenue last quarter?\nWhat are the main risks?",
                    lines=4,
                    scale=3,
                )

            eval_btn = gr.Button("Run Evaluation", variant="primary")
            eval_display = gr.Markdown("*Evaluation results will appear here.*")

            eval_btn.click(
                fn=lambda: gr.Button(interactive=False),
                outputs=[eval_btn],
            ).then(
                run_eval,
                inputs=[eval_ticker, eval_questions],
                outputs=[eval_display],
            ).then(
                fn=lambda: gr.Button(interactive=True),
                outputs=[eval_btn],
            )


if __name__ == "__main__":
    app.launch(
    inbrowser=True,
    theme=gr.themes.Soft(),
    css="""
        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
        .gradio-container { max-width: 1400px !important; }
    """
)