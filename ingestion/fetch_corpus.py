from sec_edgar_downloader import Downloader
from pathlib import Path

TICKERS = ["AAPL", "MSI", "GOOGL", "NVDA", "AMZN", "META", "TSLA", "NFLX", "JPM", "JNJ"]
FILING_TYPE = "10-K"
NUM_FILINGS = 2
OUTPUT_DIR = Path("data/raw_filings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_filings():
    
    print(OUTPUT_DIR)

    dl = Downloader(
        company_name="DBCompany", email_address="abc@gmail.com", download_folder=str(OUTPUT_DIR)
    )

    for ticker in TICKERS:
        try:
            dl.get(FILING_TYPE, ticker, limit=NUM_FILINGS)
            print(f"Successfully fetched {FILING_TYPE} for {ticker}")
        except Exception as e:
            print(f"Error fetching {FILING_TYPE} for {ticker}: {e}")


if __name__ == "__main__":
    fetch_filings()