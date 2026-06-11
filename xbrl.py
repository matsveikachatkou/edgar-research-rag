"""
xbrl.py — SEC XBRL structured financial data extraction.

Fetches key financial metrics from the SEC EDGAR XBRL Frames API
and fuses them with RAG context for structured + unstructured retrieval.

The companyfacts endpoint returns all reported XBRL facts for a company.
We extract a focused FinancialSnapshot (8 metrics) per ticker/period,
cache to disk, and prepend to LLM context when a ticker is identified.

Usage (standalone):
    uv run python xbrl.py --ticker AAPL
    uv run python xbrl.py --ticker MSFT --period 2025-03-31
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from models.research import FinancialMetric, FinancialSnapshot

load_dotenv(override=True)

log = logging.getLogger(__name__)


# Config


SEC_HEADERS = {"User-Agent": "edgar-research-rag research@example.com"}
XBRL_CACHE_PATH = Path(__file__).parent / "financial_memory.json"

# Fallback tag lists per metric — tried in order until a value is found.
# Covers the most common US-GAAP taxonomy variants across large-cap filers.
METRIC_TAGS: dict[str, list[str]] = {
    "Revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "Gross Profit": [
        "GrossProfit",
    ],
    "Operating Income": [
        "OperatingIncomeLoss",
    ],
    "Net Income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "EPS (Diluted)": [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
    ],
    "Cash & Equivalents": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsPeriodIncreaseDecrease",
    ],
    "Total Assets": [
        "Assets",
    ],
    "Operating Cash Flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
}

# Metrics that are point-in-time (balance sheet) vs flow (income/cash flow)
INSTANT_METRICS = {"Cash & Equivalents", "Total Assets"}


# CIK resolver (mirrors ingest.py — kept local to avoid circular import)

_CIK_CACHE: dict[str, str] = {}


def resolve_cik(ticker: str) -> str | None:
    """Resolve ticker to zero-padded 10-digit CIK via SEC public mapping."""
    ticker = ticker.upper()
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=SEC_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker] = cik
                return cik
        log.warning(f"Ticker {ticker} not found in SEC mapping")
        return None
    except Exception as e:
        log.error(f"CIK resolution failed for {ticker}: {e}")
        return None


# Disk cache helpers

def _cache_key(ticker: str, period_end: str, form_type: str) -> str:
    return f"{ticker.upper()}_{period_end}_{form_type}"


def load_cache() -> dict:
    """Load the financial snapshot cache from disk."""
    if not XBRL_CACHE_PATH.exists():
        return {}
    try:
        with open(XBRL_CACHE_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not read financial cache: {e}")
        return {}


def save_cache(cache: dict) -> None:
    """Persist the financial snapshot cache to disk."""
    try:
        with open(XBRL_CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        log.error(f"Could not write financial cache: {e}")


def get_cached_snapshot(
    ticker: str, period_end: str, form_type: str
) -> FinancialSnapshot | None:
    """Return a cached FinancialSnapshot if it exists."""
    cache = load_cache()
    key = _cache_key(ticker, period_end, form_type)
    if key in cache:
        try:
            return FinancialSnapshot(**cache[key])
        except Exception:
            return None
    return None


def cache_snapshot(snapshot: FinancialSnapshot) -> None:
    """Write a FinancialSnapshot to the disk cache."""
    cache = load_cache()
    key = _cache_key(snapshot.ticker, snapshot.period_end, snapshot.form_type)
    cache[key] = snapshot.model_dump()
    save_cache(cache)


# XBRL fetch and extraction

def fetch_company_facts(cik: str) -> dict | None:
    """
    Fetch the full companyfacts JSON from SEC EDGAR.
    Returns the us-gaap facts dict, or None on failure.

    Note: Response can be several MB — we extract immediately and discard.
    """
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        company_name = data.get("entityName", "")
        facts = data.get("facts", {}).get("us-gaap", {})
        return {"company_name": company_name, "facts": facts}
    except Exception as e:
        log.error(f"XBRL fetch failed for CIK {cik}: {e}")
        return None


def _find_metric_value(
    facts: dict,
    label: str,
    period_end: str,
    form_type: str,
) -> FinancialMetric | None:
    """
    Try each XBRL tag for a metric until a matching period entry is found.

    Handles the SEC taxonomy chaos: different companies use different tags
    for the same concept. We try fallbacks in order.

    Screens for:
    - Correct form_type (10-Q vs 10-K)
    - Correct period_end date
    - period_type: instant (balance sheet) vs duration (income/cash flow)
    - For duration metrics: prefers the shortest period ending on period_end
      to avoid mixing 3-month and 9-month figures.
    """
    is_instant = label in INSTANT_METRICS
    period_type = "instant" if is_instant else "duration"

    for tag in METRIC_TAGS[label]:
        tag_data = facts.get(tag)
        if not tag_data:
            continue

        # Units: usually USD or USD/shares — take the first available unit
        units_dict = tag_data.get("units", {})
        unit_key = next(iter(units_dict), None)
        if not unit_key:
            continue

        entries = units_dict[unit_key]

        if is_instant:
            # Balance sheet: find entries where end == period_end
            candidates = [
                e for e in entries
                if e.get("end") == period_end
                and e.get("form") == form_type
            ]
        else:
            # Flow metric: find entries ending on period_end for correct form
            # For 10-Q: prefer 3-month periods (avoid YTD accumulation)
            # For 10-K: 12-month periods
            candidates = [
                e for e in entries
                if e.get("end") == period_end
                and e.get("form") == form_type
                and e.get("start") is not None
            ]
            if candidates:
                # Sort by period length ascending — shortest = most granular
                candidates.sort(
                    key=lambda e: (
                        datetime.fromisoformat(e["end"])
                        - datetime.fromisoformat(e["start"])
                    ).days
                )

        if not candidates:
            continue

        entry = candidates[0]
        value = entry.get("val")
        if value is None:
            continue

        return FinancialMetric(
            label=label,
            xbrl_tag=tag,
            value=float(value),
            unit=unit_key,
            period_end=period_end,
            form_type=form_type,
            period_type=period_type,
        )

    return None


def _get_available_periods(
    facts: dict, form_type: str
) -> list[str]:
    """
    Scan facts to find all period_end dates available for a given form_type.
    Uses Revenue tags as a proxy — most filers always report revenue.
    Returns sorted descending (most recent first).
    """
    periods = set()
    for tag in METRIC_TAGS["Revenue"]:
        tag_data = facts.get(tag)
        if not tag_data:
            continue
        for unit_entries in tag_data.get("units", {}).values():
            for entry in unit_entries:
                if entry.get("form") == form_type and entry.get("end"):
                    periods.add(entry["end"])
        if periods:
            break
    return sorted(periods, reverse=True)


# Main public interface

def get_financial_snapshot(
    ticker: str,
    form_type: str | None = None,
    period_end: str | None = None,
) -> FinancialSnapshot | None:
    ticker = ticker.upper()
    cik = resolve_cik(ticker)
    if not cik:
        return None

    raw = fetch_company_facts(cik)
    if not raw:
        return None

    company_name = raw["company_name"]
    facts = raw["facts"]

    # Resolve form_type and period_end
    form_types_to_try = [form_type] if form_type else ["10-Q", "10-K"]
    resolved_form_type = None

    for ft in form_types_to_try:
        periods = _get_available_periods(facts, ft)
        if periods:
            resolved_form_type = ft
            if not period_end:
                period_end = periods[0]
                log.info(f"Using most recent period for {ticker}: {period_end}")
            break

    if not resolved_form_type:
        log.warning(f"No periods found for {ticker} in {form_types_to_try}")
        return None

    # Check cache after period resolved
    cached = get_cached_snapshot(ticker, period_end, resolved_form_type)
    if cached:
        log.info(f"Cache hit: {ticker} {resolved_form_type} {period_end}")
        return cached

    log.info(f"Extracting XBRL metrics for {ticker} {resolved_form_type} {period_end}")

    metrics = []
    for label in METRIC_TAGS:
        metric = _find_metric_value(facts, label, period_end, resolved_form_type)
        if metric:
            metrics.append(metric)
            log.info(f"  {label}: {metric.value:,.0f} {metric.unit} [{metric.xbrl_tag}]")
        else:
            log.warning(f"  {label}: not found for {ticker} {period_end}")

    if not metrics:
        log.warning(f"No metrics extracted for {ticker} {period_end}")
        return None

    snapshot = FinancialSnapshot(
        ticker=ticker,
        company_name=company_name,
        cik=cik,
        period_end=period_end,
        form_type=resolved_form_type,
        metrics=metrics,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )

    cache_snapshot(snapshot)
    log.info(f"Snapshot cached: {ticker} {resolved_form_type} {period_end} ({len(metrics)} metrics)")
    return snapshot


# Context formatting — for injection into LLM prompt

def format_snapshot_for_context(snapshot: FinancialSnapshot) -> str:
    """
    Format a FinancialSnapshot as a compact, LLM-readable block.
    Designed to prepend to the RAG context in answer.py.
    Uses human-readable scale (millions/billions) for USD values.
    """
    lines = [
        f"[STRUCTURED FINANCIALS — {snapshot.company_name} ({snapshot.ticker})]",
        f"Source: SEC EDGAR XBRL | Period: {snapshot.period_end} | Form: {snapshot.form_type}",
        "",
    ]

    for metric in snapshot.metrics:
        value = metric.value
        unit = metric.unit

        # Scale USD values for readability
        if unit == "USD":
            if abs(value) >= 1_000_000_000:
                formatted = f"${value / 1_000_000_000:.2f}B"
            elif abs(value) >= 1_000_000:
                formatted = f"${value / 1_000_000:.1f}M"
            else:
                formatted = f"${value:,.0f}"
        elif unit == "USD/shares":
            formatted = f"${value:.4f}"
        else:
            formatted = f"{value:,.2f} {unit}"

        period_note = "(point-in-time)" if metric.period_type == "instant" else "(period)"
        lines.append(f"  {metric.label}: {formatted} {period_note}")

    lines.append("")
    lines.append("Note: These are audited XBRL figures from the SEC filing. Use them for precise quantitative claims.")
    return "\n".join(lines)


# CLI

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [xbrl] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Fetch XBRL financial snapshot")
    parser.add_argument("--ticker", required=True, help="Ticker symbol e.g. AAPL")
    parser.add_argument("--form-type", default="10-Q", help="10-Q or 10-K")
    parser.add_argument("--period", default=None, help="Period end date YYYY-MM-DD")
    args = parser.parse_args()

    snapshot = get_financial_snapshot(
        ticker=args.ticker,
        form_type=args.form_type,
        period_end=args.period,
    )

    if snapshot:
        print("\n" + format_snapshot_for_context(snapshot))
        print(f"\nCached to: {XBRL_CACHE_PATH}")
    else:
        print(f"Could not extract snapshot for {args.ticker}")
        sys.exit(1)