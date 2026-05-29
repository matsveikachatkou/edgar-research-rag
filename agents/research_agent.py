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
    ) -> tuple[str, list[Result]]:
        """
        Produce a research summary for a ticker using RAG.

        Args:
            ticker:  Company ticker symbol e.g. "AAPL"
            focus:   Research focus area
            history: Optional conversation history for follow-ups
            period:  Optional period_of_report to restrict to specific filing

        Returns:
            (summary_text, retrieved_chunks)
        """
        self.log(f"Researching {ticker} — focus: {focus}{' | period: ' + period if period else ''}")

        question = (
            f"Based on the most recent SEC filing for {ticker}, "
            f"provide a detailed analysis covering: {focus}. "
            f"Include specific figures, percentages, and direct references "
            f"to management commentary where available."
        )

        summary, chunks = answer_question(
            question=question,
            history=history or [],
            ticker=ticker,
            period=period,
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
        Run research across all standard focus areas and
        combine into a comprehensive summary.
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