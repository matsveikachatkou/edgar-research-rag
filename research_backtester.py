"""
research_backtester.py — Point-in-time backtester for RAG-generated signals.

Walks chronologically through the filing timeline, generating investment signals
at each event using only information available at that point in time (PIT discipline).
Measures signal quality by comparing recommendations against forward price returns.

Event types:
    8-K  → "Fast Signal"         — tactical reaction to earnings press release
    10-Q → "Conviction Signal"   — audited filing, cross-references prior 8-K claims

Usage:
    uv run python research_backtester.py --tickers AAPL
    uv run python research_backtester.py --tickers AAPL --windows 30 60 90
    uv run python research_backtester.py --tickers AAPL --output backtest_results.json
    uv run python research_backtester.py --summary  # print existing results
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf
from chromadb import PersistentClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(override=True)

sys.path.insert(0, str(Path(__file__).parent))

from agents.recommend_agent import RecommendAgent
from agents.research_agent import ResearchAgent
from models.research import Recommendation

DB_NAME = str(Path(__file__).parent / "edgar_db")
COLLECTION_NAME = "edgar_filings"
BACKTEST_PATH = Path(__file__).parent / "backtest_results.json"

SIGNAL_CORRECT = {
    "overweight": lambda r: r > 0,
    "neutral": lambda r: abs(r) < 0.05,
    "underweight": lambda r: r < 0,
}

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [backtest] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# Data models

@dataclass
class BacktestEvent:
    """A single filing event on the backtest timeline."""
    ticker: str
    company_name: str
    filing_date: str        # YYYY-MM-DD — day market saw the filing
    period_of_report: str   # YYYY-MM-DD — fiscal period end
    form_type: str          # 10-Q, 10-K, 8-K
    filing_url: str


class BacktestResult(BaseModel):
    """Full result for a single backtest event."""
    ticker: str
    company_name: str
    filing_date: str
    period_of_report: str
    form_type: str
    signal_type: str            # "fast" (8-K) or "conviction" (10-Q/10-K)
    recommendation: str         # overweight / neutral / underweight
    confidence: float
    rationale: str
    key_risks: list[str]
    key_opportunities: list[str]

    # Forward returns (None if window not yet closed)
    return_30d: Optional[float] = None
    return_60d: Optional[float] = None
    return_90d: Optional[float] = None
    spy_30d: Optional[float] = None
    spy_60d: Optional[float] = None
    spy_90d: Optional[float] = None
    alpha_30d: Optional[float] = None
    alpha_60d: Optional[float] = None
    alpha_90d: Optional[float] = None

    signal_correct_30d: Optional[bool] = None
    signal_correct_60d: Optional[bool] = None
    signal_correct_90d: Optional[bool] = None

    backtested_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# Timeline builder

def build_timeline(
    tickers: list[str],
    collection,
) -> list[BacktestEvent]:
    """
    Build a chronological timeline of filing events from ChromaDB metadata.
    Deduplicates by ticker + form_type + filing_date.
    Uses filing_date (market availability date) not period_of_report.
    """
    results = collection.get(include=["metadatas"])
    seen = set()
    events = []

    for meta in results["metadatas"]:
        ticker = meta.get("ticker", "")
        if tickers and ticker not in tickers:
            continue

        filing_date = meta.get("filing_date", "")
        form_type = meta.get("form_type", "")
        period = meta.get("period_of_report", "")

        if not filing_date or not form_type:
            continue

        key = f"{ticker}_{form_type}_{filing_date}"
        if key in seen:
            continue
        seen.add(key)

        events.append(BacktestEvent(
            ticker=ticker,
            company_name=meta.get("company_name", ticker),
            filing_date=filing_date,
            period_of_report=period,
            form_type=form_type,
            filing_url=meta.get("filing_url", ""),
        ))

    # Sort chronologically by filing_date
    events.sort(key=lambda e: e.filing_date)
    log.info(f"Timeline: {len(events)} events for {tickers}")
    for e in events:
        log.info(f"  {e.filing_date} | {e.ticker} | {e.form_type} | period {e.period_of_report}")

    return events


# PIT retrieval

def get_pit_context(
    ticker: str,
    event_date: str,
    collection,
) -> list:
    """
    Retrieve chunks available at event_date using strict PIT filtering.
    Only returns chunks where filing_date <= event_date.
    This prevents look-ahead bias — no future filings contaminate the signal.
    """
    try:
        results = collection.get(
            where={"$and": [
                {"ticker": {"$eq": ticker}},
                {"filing_date": {"$lte": event_date}},
            ]},
            include=["documents", "metadatas"],
        )
        log.info(
            f"PIT context for {ticker} at {event_date}: "
            f"{len(results['ids'])} chunks available"
        )
        return results
    except Exception as e:
        log.error(f"PIT retrieval failed for {ticker} at {event_date}: {e}")
        return {"ids": [], "documents": [], "metadatas": []}


# Forward return calculation

def fetch_forward_return(
    ticker: str,
    filing_date: str,
    window_days: int,
) -> Optional[float]:
    """
    Fetch forward price return for a ticker over window_days from filing_date.
    Returns None if window hasn't closed yet or data unavailable.
    """
    try:
        start = datetime.strptime(filing_date, "%Y-%m-%d")
        end = start + timedelta(days=window_days + 5)  # buffer for weekends
        today = datetime.now()

        # Window hasn't closed yet
        if start + timedelta(days=window_days) > today:
            log.info(f"{ticker} {window_days}d window not yet closed — skipping")
            return None

        data = yf.download(
            ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=True,
        )

        if data.empty or len(data) < 2:
            log.warning(f"No price data for {ticker} from {filing_date}")
            return None

        price_start = float(data["Close"].squeeze().iloc[0])

        # Find the closing price closest to window_days out
        target_date = start + timedelta(days=window_days)
        data_in_window = data[data.index <= target_date.strftime("%Y-%m-%d")]
        if data_in_window.empty:
            return None

        price_end = float(data_in_window["Close"].squeeze().iloc[-1])
        return round((price_end - price_start) / price_start, 4)

    except Exception as e:
        log.error(f"Price fetch failed for {ticker}: {e}")
        return None


def fetch_returns(
    ticker: str,
    filing_date: str,
    windows: list[int],
) -> dict[int, Optional[float]]:
    """Fetch forward returns for multiple windows."""
    returns = {}
    for w in windows:
        returns[w] = fetch_forward_return(ticker, filing_date, w)
        spy = fetch_forward_return("SPY", filing_date, w)
        returns[f"spy_{w}"] = spy
        if returns[w] is not None and spy is not None:
            returns[f"alpha_{w}"] = round(returns[w] - spy, 4)
        else:
            returns[f"alpha_{w}"] = None
    return returns


# Signal generation

FAST_SIGNAL_FOCUS = (
    "earnings surprise, revenue growth, non-GAAP metrics, "
    "management tone and forward guidance from the press release"
)

CONVICTION_FOCUS = (
    "audited GAAP financial performance, balance sheet strength, "
    "risk factors, business segment trends, and whether management's "
    "prior earnings release claims are supported by the audited figures"
)


def generate_signal(
    event: BacktestEvent,
    collection,
) -> Optional[Recommendation]:
    """
    Generate an investment signal for a filing event using PIT context.
    Uses dual prompts: fast signal for 8-K, conviction signal for 10-Q/10-K.
    """
    researcher = ResearchAgent()
    recommender = RecommendAgent()

    is_fast_signal = event.form_type == "8-K"
    signal_type = "fast" if is_fast_signal else "conviction"
    focus = FAST_SIGNAL_FOCUS if is_fast_signal else CONVICTION_FOCUS

    log.info(
        f"Generating {signal_type} signal for {event.ticker} "
        f"{event.form_type} {event.filing_date}"
    )

    try:
        # PIT-aware research: only use data available at filing_date
        summary, chunks = researcher.research(
            ticker=event.ticker,
            focus=focus,
            filing_date_lte=event.filing_date,
            form_type=None,
        )

        if not chunks:
            log.warning(f"No chunks for {event.ticker} at {event.filing_date}")
            return None

        rec = recommender.recommend(
            ticker=event.ticker,
            summary=summary,
            chunks=chunks,
        )
        log.info(
            f"Signal: {rec.recommendation.upper()} "
            f"(confidence: {rec.confidence:.0%})"
        )
        return rec

    except Exception as e:
        log.error(f"Signal generation failed for {event.ticker}: {e}")
        return None


# Main backtest engine

class BacktestEngine:

    def __init__(
        self,
        tickers: list[str],
        windows: list[int] | None = None,
    ):
        self.tickers = [t.upper() for t in tickers]
        self.windows = windows or [30, 60, 90]
        self.chroma = PersistentClient(path=DB_NAME)
        self.collection = self.chroma.get_or_create_collection(COLLECTION_NAME)
        self.results: list[BacktestResult] = []

    def run(self) -> list[BacktestResult]:
        """Walk the filing timeline and generate signals with forward returns."""
        timeline = build_timeline(self.tickers, self.collection)

        if not timeline:
            log.warning("No events found — check tickers and ingestion")
            return []

        for event in timeline:
            log.info(
                f"\n{'='*60}\n"
                f"Processing: {event.ticker} {event.form_type} "
                f"filed {event.filing_date}\n"
                f"{'='*60}"
            )

            # Generate signal
            rec = generate_signal(event, self.collection)
            if not rec:
                log.warning(f"Skipping {event.ticker} {event.filing_date} — no signal")
                continue

            # Fetch forward returns
            log.info(f"Fetching forward returns for {event.ticker}...")
            returns = fetch_returns(event.ticker, event.filing_date, self.windows)

            # Evaluate signal correctness
            def correct(signal: str, ret: Optional[float]) -> Optional[bool]:
                if ret is None:
                    return None
                checker = SIGNAL_CORRECT.get(signal)
                return checker(ret) if checker else None

            result = BacktestResult(
                ticker=event.ticker,
                company_name=event.company_name,
                filing_date=event.filing_date,
                period_of_report=event.period_of_report,
                form_type=event.form_type,
                signal_type="fast" if event.form_type == "8-K" else "conviction",
                recommendation=rec.recommendation,
                confidence=rec.confidence,
                rationale=rec.rationale,
                key_risks=rec.key_risks,
                key_opportunities=rec.key_opportunities,
                return_30d=returns.get(30),
                return_60d=returns.get(60),
                return_90d=returns.get(90),
                spy_30d=returns.get("spy_30"),
                spy_60d=returns.get("spy_60"),
                spy_90d=returns.get("spy_90"),
                alpha_30d=returns.get("alpha_30"),
                alpha_60d=returns.get("alpha_60"),
                alpha_90d=returns.get("alpha_90"),
                signal_correct_30d=correct(rec.recommendation, returns.get(30)),
                signal_correct_60d=correct(rec.recommendation, returns.get(60)),
                signal_correct_90d=correct(rec.recommendation, returns.get(90)),
            )

            self.results.append(result)
            self.save()
            log.info(
                f"Result: {rec.recommendation.upper()} | "
                f"30d: {returns.get(30)} | "
                f"alpha_30d: {returns.get('alpha_30')}"
            )

        return self.results

    def save(self, path: Path = BACKTEST_PATH) -> None:
        """Persist results to JSON."""
        with open(path, "w") as f:
            json.dump([r.model_dump() for r in self.results], f, indent=2)
        log.info(f"Results saved to {path}")

    def print_report(self) -> None:
        """Print colored terminal report."""
        if not self.results:
            print("\nNo backtest results.\n")
            return

        # Colors
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RED = "\033[91m"
        RESET = "\033[0m"
        BOLD = "\033[1m"

        rec_colors = {
            "overweight": GREEN,
            "neutral": YELLOW,
            "underweight": RED,
        }

        print(f"\n{BOLD}{'='*100}{RESET}")
        print(f"{BOLD}BACKTEST REPORT{RESET}")
        print(f"{BOLD}{'='*100}{RESET}\n")

        header = (
            f"{'Ticker':<6} {'Date':<12} {'Form':<6} {'Type':<12} "
            f"{'Signal':<12} {'Conf':>5} "
            f"{'30d':>7} {'60d':>7} {'90d':>7} "
            f"{'α30d':>7} {'α60d':>7} {'α90d':>7}"
        )
        print(f"{BOLD}{header}{RESET}")
        print("-" * 100)

        for r in self.results:
            color = rec_colors.get(r.recommendation, "")
            ret_30 = f"{r.return_30d:+.1%}" if r.return_30d is not None else "N/A"
            ret_60 = f"{r.return_60d:+.1%}" if r.return_60d is not None else "N/A"
            ret_90 = f"{r.return_90d:+.1%}" if r.return_90d is not None else "N/A"
            alpha_30 = f"{r.alpha_30d:+.1%}" if r.alpha_30d is not None else "N/A"
            alpha_60 = f"{r.alpha_60d:+.1%}" if r.alpha_60d is not None else "N/A"
            alpha_90 = f"{r.alpha_90d:+.1%}" if r.alpha_90d is not None else "N/A"

            print(
                f"{r.ticker:<6} {r.filing_date:<12} {r.form_type:<6} "
                f"{r.signal_type:<12} "
                f"{color}{r.recommendation:<12}{RESET} "
                f"{r.confidence:>5.0%} "
                f"{ret_30:>7} {ret_60:>7} {ret_90:>7} "
                f"{alpha_30:>7} {alpha_60:>7} {alpha_90:>7}"
            )

        # Summary stats
        completed = [r for r in self.results if r.return_30d is not None]
        if completed:
            print(f"\n{BOLD}Summary ({len(completed)} completed events):{RESET}")

            # Hit rate at 30d
            correct_30 = [r for r in completed if r.signal_correct_30d]
            print(f"  Hit rate (30d):  {len(correct_30)}/{len(completed)} = {len(correct_30)/len(completed):.0%}")

            # Average alpha
            alphas_30 = [r.alpha_30d for r in completed if r.alpha_30d is not None]
            if alphas_30:
                print(f"  Avg alpha (30d): {sum(alphas_30)/len(alphas_30):+.1%}")

            # By signal type
            fast = [r for r in completed if r.signal_type == "fast"]
            conviction = [r for r in completed if r.signal_type == "conviction"]
            if fast:
                fast_alphas = [r.alpha_30d for r in fast if r.alpha_30d is not None]
                if fast_alphas:
                    print(f"  Fast signal avg alpha (30d):       {sum(fast_alphas)/len(fast_alphas):+.1%}")
            if conviction:
                conv_alphas = [r.alpha_30d for r in conviction if r.alpha_30d is not None]
                if conv_alphas:
                    print(f"  Conviction signal avg alpha (30d): {sum(conv_alphas)/len(conv_alphas):+.1%}")

        print(f"\n{BOLD}{'='*100}{RESET}\n")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run point-in-time backtest")
    parser.add_argument(
        "--tickers", nargs="+", required=False, default=None,
        help="Tickers to backtest e.g. AAPL MSFT"
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, default=[30, 60, 90],
        help="Forward return windows in days (default: 30 60 90)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results to JSON file (default: backtest_results.json)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary of existing results and exit"
    )
    args = parser.parse_args()

    if args.summary:
        if not BACKTEST_PATH.exists():
            print("No backtest results found. Run the backtest first.")
            sys.exit(0)
        with open(BACKTEST_PATH) as f:
            data = json.load(f)
        results = [BacktestResult(**r) for r in data]
        engine = BacktestEngine(tickers=[], windows=[30, 60, 90])
        engine.results = results
        engine.print_report()
        sys.exit(0)

    if not args.tickers:
        print("Error: --tickers required unless using --summary")
        sys.exit(1)

    engine = BacktestEngine(
        tickers=args.tickers,
        windows=args.windows,
    )
    engine.run()
    engine.print_report()

    output_path = Path(args.output) if args.output else BACKTEST_PATH
    engine.save(output_path)