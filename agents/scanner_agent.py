"""
agents/scanner_agent.py — Scans SEC EDGAR RSS for new filings.

Uses the EDGAR full-text search API to find recent filings for
watched tickers. Filters out already-processed filings via memory.

Extend by swapping _fetch_events() with a different data source
(e.g. Refinitiv, Bloomberg, European ESMA filings) while keeping
the SecuritiesEvent output contract unchanged.
"""

import os
from datetime import datetime
from typing import Optional

import requests
from dotenv import load_dotenv

from agents.agent import Agent
from models.research import SecuritiesEvent

load_dotenv(override=True)

SEC_HEADERS = {"User-Agent": "edgar-research-rag research@example.com"}
EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms={form_type}"


class ScannerAgent(Agent):
    """
    Scans SEC EDGAR for recent filings on watched tickers.

    Watched tickers are read from the WATCHED_TICKERS env var
    (comma-separated) or passed directly at construction.
    """

    name = "Scanner"
    color = Agent.CYAN

    def __init__(self, tickers: list[str] | None = None):
        super().__init__()
        if tickers:
            self.watched_tickers = [t.strip().upper() for t in tickers]
        else:
            raw = os.getenv("WATCHED_TICKERS", "AAPL,MSFT,NVDA,GOOGL,AMZN")
            self.watched_tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        self.log(f"Watching tickers: {', '.join(self.watched_tickers)}")

    def _fetch_events(
        self, ticker: str, form_type: str, k: int = 3
    ) -> list[SecuritiesEvent]:
        """
        Fetch recent filings for a single ticker from EDGAR search API.
        Returns up to k SecuritiesEvent objects.
        """
        url = EDGAR_SEARCH.format(ticker=ticker, form_type=form_type)
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
            resp.raise_for_status()
            hits = resp.json().get("hits", {}).get("hits", [])
        except Exception as e:
            self.log(f"EDGAR fetch failed for {ticker}: {e}")
            return []

        events = []
        for hit in hits[:k]:
            src = hit.get("_source", {})
            accession = hit.get("_id", "")
            entity_name = src.get("entity_name", ticker)
            file_date = src.get("file_date", "")
            period = src.get("period_of_report", "")
            entity_id = src.get("entity_id", "")

            try:
                filed_dt = datetime.strptime(file_date, "%Y-%m-%d")
            except Exception:
                filed_dt = datetime.utcnow()

            # Build filing index URL
            if entity_id and accession:
                acc_clean = accession.replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{entity_id}/{acc_clean}/{accession}-index.htm"
                )
            else:
                filing_url = (
                    f"https://www.sec.gov/cgi-bin/browse-edgar?"
                    f"action=getcompany&company={ticker}"
                    f"&type={form_type}&dateb=&owner=include&count=5"
                )

            events.append(
                SecuritiesEvent(
                    ticker=ticker,
                    company_name=entity_name,
                    form_type=form_type,
                    filed_at=filed_dt,
                    filing_url=filing_url,
                    period_of_report=period,
                )
            )
        return events

    def scan(
        self,
        form_types: list[str] | None = None,
        memory: list[str] | None = None,
        k: int = 1,
    ) -> list[SecuritiesEvent]:
        """
        Scan all watched tickers for new filings.

        Args:
            form_types: List of form types to scan e.g. ["10-K", "10-Q"]
            memory:     List of filing_urls already processed (dedup)
            k:          Max filings per ticker per form type

        Returns:
            List of new SecuritiesEvent objects not in memory
        """
        form_types = form_types or ["10-Q", "10-K"]
        memory = memory or []
        seen_urls = set(memory)

        all_events: list[SecuritiesEvent] = []
        for ticker in self.watched_tickers:
            for form_type in form_types:
                self.log(f"Scanning {ticker} {form_type}")
                events = self._fetch_events(ticker, form_type, k=k)
                new = [e for e in events if e.filing_url not in seen_urls]
                if new:
                    self.log(f"Found {len(new)} new {form_type} filing(s) for {ticker}")
                    all_events.extend(new)
                else:
                    self.log(f"No new {form_type} filings for {ticker}")

        self.log(f"Scan complete — {len(all_events)} new filing(s) total")
        return all_events