"""
scripts/seed.py — Ingest demo companies for quick start.

Ingests the latest 10-Q filing for Apple, Tesla, and Microsoft.
Takes approximately 15-20 minutes depending on network and API rate limits.

Usage:
    uv run python scripts/seed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import ingest

DEMO_TICKERS = ["AAPL", "TSLA", "MSFT"]

if __name__ == "__main__":
    print("Seeding vector store with demo companies...")
    print(f"Tickers: {', '.join(DEMO_TICKERS)}")
    print("This will take approximately 15-20 minutes.\n")

    ingest(tickers=DEMO_TICKERS, form_type="10-Q", k=1)

    print("\nSeeding complete. Launch the app with:")
    print("  uv run python app.py")