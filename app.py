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


REC_COLORS = {"buy": "🟢", "hold": "🟡", "sell": "🔴"}


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
            f"**Period:** {opp.filed_at[:10]}  |  "
            f"**Researched:** {opp.researched_at[:10]}\n\n"
            f"**Recommendation:** `{opp.recommendation.upper()}` "
            f"({opp.confidence:.0%} confidence)\n\n"
            f"**Rationale:** {opp.rationale}\n\n"
            f"**Key risks:** {' · '.join(opp.key_risks) if opp.key_risks else 'N/A'}\n\n"
            f"**Key opportunities:** {' · '.join(opp.key_opportunities) if opp.key_opportunities else 'N/A'}\n\n"
            f"---"
        )
    return "\n\n".join(lines)


# Tab 1 — Research Chat


def chat(user_message: str, ticker_filter: str, history: list[dict]):
    """Handle a chat turn and return answer + sources."""
    if not user_message.strip():
        return history, "*Ask a question to see sources.*"

    llm_history = []
    for user_msg, assistant_msg in (history or []):
        if user_msg:
            llm_history.append({"role": "user", "content": user_msg})
        if assistant_msg:
            llm_history.append({"role": "assistant", "content": assistant_msg})

    ticker = ticker_filter.strip().upper() or None
    answer, chunks = answer_question(user_message, llm_history, ticker=ticker)
    sources_md = format_sources(chunks)

    history = (history or []) + [(user_message, answer)]

    return history, sources_md


# Tab 2 — Pipeline


def run_pipeline(tickers_input: str, form_type: str, max_chars_input: int):
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        yield "Warning: Please enter at least one ticker.", ""
        return

    max_chars = int(max_chars_input) if max_chars_input else 15000
    yield f"Starting pipeline for {', '.join(tickers)}...\n", ""

    try:
        framework = ResearchFramework(
            tickers=tickers,
            form_types=[form_type],
            max_chars=max_chars,
        )
        memory = framework.run(max_events=len(tickers))
        results_md = format_memory(memory)
        yield f"Pipeline complete — {len(memory)} total research record(s).\n", results_md
    except Exception as e:
        yield f"Pipeline error: {e}\n", ""


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


with gr.Blocks(
    title="EDGAR Research RAG",
) as app:
    gr.Markdown("#EDGAR Research RAG")
    gr.Markdown(
        "RAG-powered investment research over SEC filings. "
        "Ingest filings, ask questions, generate recommendations, evaluate quality."
    )

    with gr.Tabs():
    
        # Tab 1: Research Chat
        with gr.Tab("Research Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=500)
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Ask about any ingested company...",
                            show_label=False,
                            scale=4,
                        )
                        ticker_filter = gr.Textbox(
                            placeholder="Ticker filter (optional)",
                            show_label=False,
                            scale=1,
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

        # Tab 2: Pipeline
        with gr.Tab("Pipeline"):
            gr.Markdown(
                "Run the full pipeline: scan EDGAR → ingest → research → recommend"
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
                max_chars_input = gr.Number(
                    label="Max chars per filing",
                    value=15000,
                    scale=1,
                )

            run_btn = gr.Button("Run Pipeline", variant="primary")
            pipeline_status = gr.Textbox(
                label="Status", lines=3, interactive=False
            )
            results_display = gr.Markdown("*Results will appear here.*")

            run_btn.click(
                run_pipeline,
                [tickers_input, form_type_input, max_chars_input],
                [pipeline_status, results_display],
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
                run_eval,
                [eval_ticker, eval_questions],
                eval_display,
            )


if __name__ == "__main__":
    app.launch(inbrowser=True, theme=gr.themes.Soft())