"""
research_framework.py — CLI orchestrator for the automated research pipeline.

Runs the full scan → ingest → research → recommend pipeline from the command line.
For interactive use, the Gradio UI (app.py) provides the same functionality
with on-demand recommendations per ticker and filing period.

Usage:
    uv run python research_framework.py
    uv run python research_framework.py --tickers AAPL MSFT --form-type 10-Q
    uv run python research_framework.py --summary
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from agents.recommend_agent import RecommendAgent
from agents.research_agent import ResearchAgent
from agents.scanner_agent import ScannerAgent
from ingest import ingest
from models.research import ResearchOpportunity, SecuritiesEvent

MEMORY_PATH = Path(__file__).parent / "research_memory.json"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [Framework] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# Memory persistence


def read_memory() -> list[ResearchOpportunity]:
    """Load persisted research results from disk."""
    if not MEMORY_PATH.exists():
        return []
    try:
        with open(MEMORY_PATH) as f:
            data = json.load(f)
        return [ResearchOpportunity(**item) for item in data]
    except Exception as e:
        log.warning(f"Could not read memory: {e}")
        return []


def write_memory(memory: list[ResearchOpportunity]) -> None:
    """Persist research results to disk."""
    with open(MEMORY_PATH, "w") as f:
        json.dump([m.model_dump() for m in memory], f, indent=2)
    log.info(f"Memory updated — {len(memory)} total research record(s)")


def already_researched(
    memory: list[ResearchOpportunity], event: SecuritiesEvent
) -> bool:
    """Check if a filing has already been researched."""
    for opp in memory:
        if (
            opp.ticker == event.ticker
            and opp.form_type == event.form_type
            and opp.filing_url == event.filing_url
        ):
            return True
    return False


# Core pipeline


class ResearchFramework:
    """
    Orchestrates the full research pipeline:
    scan → ingest → research → recommend → persist
    """

    def __init__(
        self,
        tickers: list[str] | None = None,
        form_types: list[str] | None = None,
        max_chars: int | None = None,
    ):
        self.tickers = tickers
        self.form_types = form_types or ["10-Q"]
        self.max_chars = max_chars
        self.memory = read_memory()

        self.scanner = ScannerAgent(tickers=tickers)
        self.researcher = ResearchAgent()
        self.recommender = RecommendAgent()

        log.info("Research Framework initialised")

    def run_single(self, event: SecuritiesEvent) -> ResearchOpportunity | None:
        """
        Run the full pipeline for a single filing event.
        Returns a ResearchOpportunity or None if pipeline fails.
        """
        log.info(
            f"Processing {event.ticker} {event.form_type} "
            f"(period: {event.period_of_report or 'N/A'})"
        )

        # Step 1 — Ingest filing into ChromaDB
        log.info(f"Ingesting {event.ticker} {event.form_type}")
        try:
            ingest(
                tickers=[event.ticker],
                form_type=event.form_type,
                k=1,
                max_chars=self.max_chars,
            )
        except Exception as e:
            log.error(f"Ingestion failed for {event.ticker}: {e}")
            return None

        # Step 2 — Research via RAG
        log.info(f"Researching {event.ticker}")
        try:
            summary, chunks = self.researcher.research(
                ticker=event.ticker,
                focus="investment outlook, revenue trends, key risks",
            )
        except Exception as e:
            log.error(f"Research failed for {event.ticker}: {e}")
            return None

        if not chunks:
            log.warning(
                f"No chunks retrieved for {event.ticker} — "
                "filing may not have ingested correctly"
            )
            return None

        # Step 3 — Generate recommendation
        log.info(f"Generating recommendation for {event.ticker}")
        try:
            rec = self.recommender.recommend(
                ticker=event.ticker,
                summary=summary,
                chunks=chunks,
            )
        except Exception as e:
            log.error(f"Recommendation failed for {event.ticker}: {e}")
            return None

        # Step 4 — Build and return ResearchOpportunity
        opp = ResearchOpportunity(
            ticker=event.ticker,
            company_name=event.company_name,
            form_type=event.form_type,
            filed_at=event.filed_at.isoformat(),
            period_of_report=event.period_of_report or "",
            recommendation=rec.recommendation,
            confidence=rec.confidence,
            rationale=rec.rationale,
            key_risks=rec.key_risks,
            key_opportunities=rec.key_opportunities,
            rag_summary=summary,
            filing_url=event.filing_url,
        )

        log.info(
            f"Completed {event.ticker}: "
            f"{rec.recommendation.upper()} "
            f"(confidence: {rec.confidence:.2f})"
        )
        return opp

    def run(self, max_events: int = 3) -> list[ResearchOpportunity]:
        """
        Run the full pipeline for all watched tickers.

        Args:
            max_events: Max number of new filings to process per run

        Returns:
            Full memory including new results
        """
        log.info("Starting research pipeline")

        # Scan for new filings
        seen_urls = [opp.filing_url for opp in self.memory]
        events = self.scanner.scan(
            form_types=self.form_types,
            memory=seen_urls,
        )

        if not events:
            log.info("No new filings found — nothing to process")
            return self.memory

        log.info(f"Found {len(events)} new filing(s) to process")
        processed = 0

        for event in events:
            if processed >= max_events:
                log.info(f"Reached max_events limit ({max_events}) — stopping")
                break

            if already_researched(self.memory, event):
                log.info(
                    f"Already researched {event.ticker} "
                    f"{event.form_type} — skipping"
                )
                continue

            opp = self.run_single(event)
            if opp:
                self.memory.append(opp)
                write_memory(self.memory)
                processed += 1

        log.info(
            f"Pipeline complete — "
            f"{processed} new research record(s) added"
        )
        return self.memory

    def print_summary(self) -> None:
        """Print a formatted summary of all research results."""
        if not self.memory:
            print("\nNo research results yet. Run the pipeline first.\n")
            return

        print("\n" + "=" * 60)
        print("RESEARCH SUMMARY")
        print("=" * 60)
        for opp in sorted(self.memory, key=lambda x: x.researched_at, reverse=True):
            rec_color = {
              "overweight": "\033[92m",
              "neutral": "\033[93m",
              "underweight": "\033[91m",
          }.get(opp.recommendation, "")
            reset = "\033[0m"

            print(f"\n{opp.company_name} ({opp.ticker}) — {opp.form_type}")
            print(f"Period: {opp.period_of_report or opp.filed_at[:10]}  |  Researched: {opp.researched_at[:10]}")
            print(
                f"Recommendation: {rec_color}{opp.recommendation.upper()}{reset} "
                f"(confidence: {opp.confidence:.0%})"
            )
            print(f"Rationale: {opp.rationale}")
            if opp.key_risks:
                print(f"Key risks: {' · '.join(opp.key_risks)}")
            if opp.key_opportunities:
                print(f"Key opportunities: {' · '.join(opp.key_opportunities)}")
            print("-" * 60)


# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the EDGAR research pipeline"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Tickers to research (overrides WATCHED_TICKERS env var)"
    )
    parser.add_argument(
        "--form-type", default="10-Q",
        help="SEC form type: 10-K or 10-Q (default: 10-Q)"
    )
    parser.add_argument(
        "--max-chars", type=int, default=None,
        help="Cap extracted characters per filing (default: None)"
    )
    parser.add_argument(
        "--max-events", type=int, default=3,
        help="Max new filings to process per run (default: 3)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary of existing research results and exit"
    )
    args = parser.parse_args()

    framework = ResearchFramework(
        tickers=args.tickers,
        form_types=[args.form_type],
        max_chars=args.max_chars,
    )

    if args.summary:
        framework.print_summary()
    else:
        framework.run(max_events=args.max_events)
        framework.print_summary()