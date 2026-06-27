import sys
from pathlib import Path

import yfinance as yf
import pandas as pd

# Allow imports from src when running as a script
sys.path.append(str(Path(__file__).resolve().parent))

from config import STOCK_PRICE_DIR, PRICE_HISTORY_PERIOD, PRICE_INTERVAL
from ticker_universe import LIQUID_OPTIONS_UNIVERSE


def fetch_one_ticker(ticker: str) -> pd.DataFrame:
    """
    Fetch historical daily stock data for one ticker.
    """
    df = yf.download(
        ticker,
        period=PRICE_HISTORY_PERIOD,
        interval=PRICE_INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten columns if yfinance returns MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()

    # Standardize column names
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"{ticker} missing columns: {missing}")

    df["Ticker"] = ticker

    return df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]


def main():
    STOCK_PRICE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FETCHING STOCK PRICE DATA")
    print("=" * 70)

    success = 0
    failed = []

    for ticker in LIQUID_OPTIONS_UNIVERSE:
        try:
            print(f"Fetching {ticker}...")
            df = fetch_one_ticker(ticker)
            output_path = STOCK_PRICE_DIR / f"{ticker}.csv"
            df.to_csv(output_path, index=False)
            success += 1
        except Exception as e:
            print(f"FAILED: {ticker} -> {e}")
            failed.append((ticker, str(e)))

    print("\n" + "=" * 70)
    print("FETCH COMPLETE")
    print("=" * 70)
    print(f"Successful tickers: {success}")
    print(f"Failed tickers: {len(failed)}")

    if failed:
        print("\nFailures:")
        for ticker, error in failed:
            print(f"{ticker}: {error}")


if __name__ == "__main__":
    main()