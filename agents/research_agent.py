"""
agents/research_agent.py — RAG-powered filing research agent.

Uses the full retrieval pipeline from answer.py to produce
a structured research summary for a given ticker and focus area.
"""

from agents.agent import Agent
from answer import answer_question
from models.research import Result


class ResearchAgent(Agent):
    """
    Produces research summaries for a ticker using RAG over
    ingested SEC filings.
    """

    name = "Research"
    color = Agent.BLUE

    FOCUS_AREAS = [
        "revenue growth and guidance",
        "risk factors and headwinds",
        "balance sheet and cash flow",
        "business segment performance",
        "management commentary and outlook",
    ]

    def __init__(self):
        super().__init__()
        self.log("Research Agent ready")

    def research(
        self,
        ticker: str,
        focus: str = "investment outlook",
        history: list[dict] | None = None,
        period: str | None = None,
        form_type: str | None = None,
        filing_date_lte: str | None = None, 
    ) -> tuple[str, list[Result]]:
        """
        Produce a research summary for a ticker using RAG.

        Args:
            ticker:          Company ticker symbol e.g. "AAPL"
            focus:           Research focus area
            history:         Optional conversation history for follow-ups
            period:          Optional period_of_report to restrict to specific filing
            form_type:       Optional form type for XBRL routing
            filing_date_lte: Optional PIT cutoff — only use data filed on or before
                            this date. When set, supersedes period filter.

        Returns:
            (summary_text, retrieved_chunks)
        """
        pit_note = f" | PIT cutoff: {filing_date_lte}" if filing_date_lte else ""
        self.log(f"Researching {ticker} — focus: {focus}{' | period: ' + period if period else ''}{pit_note}")

        if filing_date_lte:
            period_clause = f"using all filings available as of {filing_date_lte}"
        elif period:
            period_clause = f"for the period ending {period}"
        else:
            period_clause = "from the most recent SEC filing"

        question = (
            f"Based on the SEC filing for {ticker} {period_clause}, "
            f"provide a detailed analysis covering: {focus}. "
            f"Include specific figures, percentages, and direct references "
            f"to management commentary where available."
        )

        summary, chunks = answer_question(
            question=question,
            history=history or [],
            ticker=ticker,
            period=period,
            form_type=form_type,
            filing_date_lte=filing_date_lte,  # added
        )

        if not chunks:
            self.log(f"No filing data found for {ticker} in vector store")
        else:
            self.log(
                f"Research complete for {ticker} "
                f"({len(chunks)} chunks retrieved)"
            )

        return summary, chunks

    def deep_research(
        self, ticker: str
    ) -> tuple[str, list[Result]]:
        """
        Run research across all standard focus areas and combine into
        a comprehensive summary. Not currently wired to the UI —
        available for future deep-dive feature.
        """
        self.log(f"Running deep research for {ticker}")
        all_chunks: list[Result] = []
        sections: list[str] = []

        for focus in self.FOCUS_AREAS:
            summary, chunks = self.research(ticker, focus=focus)
            sections.append(f"### {focus.title()}\n\n{summary}")
            all_chunks.extend(chunks)

        combined = "\n\n".join(sections)
        self.log(f"Deep research complete for {ticker}")
        return combined, all_chunks