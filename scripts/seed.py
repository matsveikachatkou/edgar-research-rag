"""
scripts/seed.py — Ingest demo companies for quick start.

Ingests the latest 10-Q and 8-K earnings press release for four demo companies.
Takes approximately 20-30 minutes depending on network and API rate limits.

Usage:
    uv run python scripts/seed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import ingest

DEMO_TICKERS = ["AAPL", "TSLA", "MSFT", "NVDA"]

if __name__ == "__main__":
    print("Seeding vector store with demo companies...")
    print(f"Tickers: {', '.join(DEMO_TICKERS)}")
    print("This will take approximately 20-30 minutes.\n")

    print("Step 1/2: Ingesting 10-Q filings...")
    ingest(tickers=DEMO_TICKERS, form_type="10-Q", k=1)

    print("\nStep 2/2: Ingesting 8-K earnings press releases...")
    ingest(tickers=DEMO_TICKERS, form_type="8-K", k=1)

    print("\nSeeding complete. Launch the app with:")
    print("  uv run python app.py")