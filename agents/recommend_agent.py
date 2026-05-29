"""
agents/recommend_agent.py — Generates structured investment recommendations.

Uses Pydantic structured output (not pipe-splitting) to produce
a validated Recommendation with rationale, risks, and opportunities.
"""

import os

from dotenv import load_dotenv
from litellm import completion
from tenacity import retry, wait_exponential

from agents.agent import Agent
from models.research import Recommendation, Result

load_dotenv(override=True)

MODEL = "openai/gpt-4.1-mini"
WAIT = wait_exponential(multiplier=1, min=10, max=240)


class RecommendAgent(Agent):
    """
    Generates a structured buy/hold/sell recommendation from a
    RAG research summary and retrieved chunks.
    """

    name = "Recommend"
    color = Agent.GREEN

    def __init__(self):
        super().__init__()
        self.log("Recommendation Agent ready")

    @retry(wait=WAIT)
    def recommend(
        self,
        ticker: str,
        summary: str,
        chunks: list[Result] | None = None,
    ) -> Recommendation:
        """
        Generate a structured recommendation for a ticker.

        Args:
            ticker:  Company ticker symbol
            summary: Research summary from ResearchAgent
            chunks:  Retrieved filing chunks for additional context

        Returns:
            Validated Recommendation Pydantic model
        """
        self.log(f"Generating recommendation for {ticker}")

        # Build additional context from chunks if available
        chunk_context = ""
        if chunks:
            snippet_parts = []
            for chunk in chunks[:5]:
                company = chunk.metadata.get("company_name", ticker)
                form = chunk.metadata.get("form_type", "")
                period = chunk.metadata.get("period_of_report", "")
                snippet_parts.append(
                    f"[{company} {form} {period}]: "
                    f"{chunk.page_content[:300]}"
                )
            chunk_context = (
                "\n\nSupporting filing excerpts:\n\n"
                + "\n\n".join(snippet_parts)
            )

        prompt = f"""You are a senior equity research analyst. Based on the following 
research summary and filing excerpts for {ticker}, generate a structured 
investment recommendation.

Research summary:
{summary}
{chunk_context}

Generate a recommendation with:
- recommendation: exactly one of "buy", "hold", or "sell"
- confidence: float between 0 and 1 reflecting certainty
- rationale: 2-3 sentences grounded in specific filing evidence
- key_risks: up to 3 specific risks from the filing (not generic)
- key_opportunities: up to 3 specific opportunities from the filing (not generic)

Base your recommendation strictly on the filing evidence provided. \
Do not introduce external information."""

        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format=Recommendation,
        )

        rec = Recommendation.model_validate_json(
            response.choices[0].message.content
        )

        # Ensure ticker is set correctly
        rec.ticker = ticker.upper()

        self.log(
            f"Recommendation for {ticker}: {rec.recommendation.upper()} "
            f"(confidence: {rec.confidence:.2f})"
        )
        return rec