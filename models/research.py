from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# Ingestion models


class EdgarFiling(BaseModel):
    """A single SEC EDGAR filing fetched from the EDGAR submissions API."""
    ticker: str
    company_name: str
    form_type: str                  # 10-K, 10-Q, 8-K, etc.
    filed_at: datetime
    period_of_report: Optional[str] = None
    filing_url: str                 # index page URL
    pdf_url: Optional[str] = None   # primary document URL (HTML or PDF)
    document_markdown: str = ""     # populated after HTML extraction


class Chunk(BaseModel):
    """A single semantic chunk produced by the LLM chunker."""
    headline: str = Field(
        description="A brief heading capturing the key topic of this chunk"
    )
    summary: str = Field(
        description="A few sentences summarising this chunk for retrieval"
    )
    original_text: str = Field(
        description="The original text from the filing, exactly as extracted"
    )

    def as_result(self, filing: EdgarFiling) -> "Result":
        metadata = {
            "ticker": filing.ticker,
            "company_name": filing.company_name,
            "form_type": filing.form_type,
            "filed_at": filing.filed_at.isoformat(),
            "period_of_report": filing.period_of_report or "",
            "filing_url": filing.filing_url,
        }
        page_content = (
            self.headline + "\n\n" + self.summary + "\n\n" + self.original_text
        )
        return Result(page_content=page_content, metadata=metadata)


class Chunks(BaseModel):
    chunks: list[Chunk]


# Retrieval models


class Result(BaseModel):
    """A retrieved chunk with its metadata."""
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="Chunk ids ordered from most relevant to least relevant"
    )


# Agent models


class SecuritiesEvent(BaseModel):
    """A filing event emitted by the scanner agent."""
    ticker: str
    company_name: str
    form_type: str
    filed_at: datetime
    filing_url: str
    period_of_report: Optional[str] = None


class Recommendation(BaseModel):
    """Structured output from the recommendation agent."""
    ticker: str
    recommendation: str = Field(
        description="One of: buy, hold, sell"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0,
    )
    rationale: str = Field(
        description="2-3 sentence explanation grounded in the filing evidence"
    )
    key_risks: list[str] = Field(
        description="Up to 3 key risks identified from the filing",
        max_length=3,
    )
    key_opportunities: list[str] = Field(
        description="Up to 3 key opportunities identified from the filing",
        max_length=3,
    )


class ResearchOpportunity(BaseModel):
    """Persisted output of a full research cycle."""
    ticker: str
    company_name: str
    form_type: str
    filed_at: str                   # ISO string for JSON serialisation
    period_of_report: str = ""    
    recommendation: str
    confidence: float
    rationale: str
    key_risks: list[str]
    key_opportunities: list[str]
    rag_summary: str                # raw research agent output
    filing_url: str
    researched_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


# Eval models


class EvalSample(BaseModel):
    """A single question/answer/context triple for evaluation."""
    question: str
    answer: str
    chunks: list[Result]
    expected_answer: Optional[str] = None


class EvalResult(BaseModel):
    """Scores for a single eval sample."""
    question: str
    context_precision: float = Field(description="0-1: are retrieved chunks relevant?")
    answer_faithfulness: float = Field(description="0-1: does answer stay within context?")
    answer_relevance: float = Field(description="0-1: does answer address the question?")

    @property
    def overall(self) -> float:
        return round(
            (self.context_precision + self.answer_faithfulness + self.answer_relevance) / 3,
            3,
        )