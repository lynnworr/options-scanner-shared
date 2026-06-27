import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from config import (
    STOCK_PRICE_DIR,
    PROCESSED_DIR,
    BOARD_DIR,
    MIN_PRICE,
    MAX_PRICE,
    MIN_AVG_DOLLAR_VOLUME,
)


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return true_range.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["RSI14"] = calculate_rsi(df["Close"], 14)
    df["ATR14"] = calculate_atr(df, 14)

    df["High20"] = df["High"].rolling(20).max()
    df["Low20"] = df["Low"].rolling(20).min()

    df["AvgVolume20"] = df["Volume"].rolling(20).mean()
    df["RelVolume"] = df["Volume"] / df["AvgVolume20"]

    df["AvgDollarVolume20"] = df["Close"] * df["AvgVolume20"]

    df["Return5"] = df["Close"].pct_change(5)
    df["Return20"] = df["Close"].pct_change(20)

    df["ATRPercent"] = df["ATR14"] / df["Close"]

    return df


def score_bullish_setup(row: pd.Series) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    close = row["Close"]

    # Trend
    if close > row["SMA50"]:
        score += 15
        reasons.append("Price above 50-day trend")

    if close > row["SMA200"]:
        score += 15
        reasons.append("Price above 200-day trend")

    if row["SMA20"] > row["SMA50"]:
        score += 10
        reasons.append("20-day trend above 50-day trend")

    if row["SMA50"] > row["SMA200"]:
        score += 10
        reasons.append("50-day trend above 200-day trend")

    # Momentum
    if row["Return20"] > 0:
        score += 10
        reasons.append("Positive 20-day momentum")

    if 45 <= row["RSI14"] <= 68:
        score += 10
        reasons.append("RSI bullish but not overextended")

    # Breakout / support logic
    if close >= row["High20"] * 0.98:
        score += 10
        reasons.append("Near 20-day breakout area")

    if abs(close - row["SMA20"]) / close <= 0.03:
        score += 10
        reasons.append("Near rising 20-day moving average")

    # Volume
    if row["RelVolume"] >= 1.10:
        score += 10
        reasons.append("Above-average volume")

    # ATR / movement potential
    if 0.015 <= row["ATRPercent"] <= 0.08:
        score += 10
        reasons.append("ATR range suitable for options movement")

    return min(score, 100), reasons


def score_bearish_setup(row: pd.Series) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    close = row["Close"]

    # Trend weakness
    if close < row["SMA50"]:
        score += 15
        reasons.append("Price below 50-day trend")

    if close < row["SMA200"]:
        score += 15
        reasons.append("Price below 200-day trend")

    if row["SMA20"] < row["SMA50"]:
        score += 10
        reasons.append("20-day trend below 50-day trend")

    if row["SMA50"] < row["SMA200"]:
        score += 10
        reasons.append("50-day trend below 200-day trend")

    # Momentum weakness
    if row["Return20"] < 0:
        score += 10
        reasons.append("Negative 20-day momentum")

    if 32 <= row["RSI14"] <= 55:
        score += 10
        reasons.append("RSI bearish but not extremely oversold")

    # Breakdown / resistance logic
    if close <= row["Low20"] * 1.02:
        score += 10
        reasons.append("Near 20-day breakdown area")

    if abs(close - row["SMA20"]) / close <= 0.03 and close < row["SMA20"]:
        score += 10
        reasons.append("Rejected near 20-day moving average")

    # Volume
    if row["RelVolume"] >= 1.10:
        score += 10
        reasons.append("Above-average volume")

    # ATR / movement potential
    if 0.015 <= row["ATRPercent"] <= 0.08:
        score += 10
        reasons.append("ATR range suitable for options movement")

    return min(score, 100), reasons


def determine_setup_type(row: pd.Series) -> str:
    bullish = row["BullishScore"]
    bearish = row["BearishScore"]

    if bullish >= 80 and bullish > bearish:
        return "Strong Bullish"
    if bullish >= 70 and bullish > bearish:
        return "Bullish"
    if bearish >= 80 and bearish > bullish:
        return "Strong Bearish"
    if bearish >= 70 and bearish > bullish:
        return "Bearish"

    return "No Clear Edge"


def suggested_strategy(row: pd.Series) -> str:
    """
    Temporary strategy suggestion before options-chain data is connected.

    Later this will use:
    - IV rank
    - delta
    - DTE
    - bid/ask spread
    - open interest
    - contract volume
    """
    setup = row["SetupType"]

    if setup in ["Strong Bullish", "Bullish"]:
        if row["RSI14"] < 60:
            return "Check call debit spread or put credit spread"
        return "Check put credit spread; avoid chasing calls"

    if setup in ["Strong Bearish", "Bearish"]:
        if row["RSI14"] > 40:
            return "Check put debit spread or call credit spread"
        return "Check call credit spread; avoid chasing puts"

    return "No trade"


def scan_all_tickers() -> pd.DataFrame:
    rows = []

    csv_files = sorted(STOCK_PRICE_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No stock CSV files found in {STOCK_PRICE_DIR}. "
            "Run fetch_stock_data.py first."
        )

    for file in csv_files:
        ticker = file.stem

        try:
            df = pd.read_csv(file)
            df["Date"] = pd.to_datetime(df["Date"])

            df = add_indicators(df)
            latest = df.dropna().iloc[-1]

            # Basic price and liquidity filters
            if latest["Close"] < MIN_PRICE or latest["Close"] > MAX_PRICE:
                continue

            if latest["AvgDollarVolume20"] < MIN_AVG_DOLLAR_VOLUME:
                continue

            bullish_score, bullish_reasons = score_bullish_setup(latest)
            bearish_score, bearish_reasons = score_bearish_setup(latest)

            row = {
                "Date": latest["Date"].date(),
                "Ticker": ticker,
                "Close": round(float(latest["Close"]), 2),
                "SMA20": round(float(latest["SMA20"]), 2),
                "SMA50": round(float(latest["SMA50"]), 2),
                "SMA200": round(float(latest["SMA200"]), 2),
                "RSI14": round(float(latest["RSI14"]), 2),
                "ATR14": round(float(latest["ATR14"]), 2),
                "ATRPercent": round(float(latest["ATRPercent"] * 100), 2),
                "RelVolume": round(float(latest["RelVolume"]), 2),
                "Return5Pct": round(float(latest["Return5"] * 100), 2),
                "Return20Pct": round(float(latest["Return20"] * 100), 2),
                "AvgDollarVolume20": round(float(latest["AvgDollarVolume20"]), 0),
                "BullishScore": bullish_score,
                "BearishScore": bearish_score,
                "BullishReasons": "; ".join(bullish_reasons),
                "BearishReasons": "; ".join(bearish_reasons),
            }

            temp_row = pd.Series(row)
            row["SetupType"] = determine_setup_type(temp_row)
            row["SuggestedStrategy"] = suggested_strategy(pd.Series(row))

            row["FinalStockSetupScore"] = max(bullish_score, bearish_score)

            rows.append(row)

        except Exception as e:
            print(f"FAILED scanning {ticker}: {e}")

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values(
        by=["FinalStockSetupScore", "RelVolume", "ATRPercent"],
        ascending=[False, False, False],
    )

    return result


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BOARD_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCANNING STOCK SETUPS FOR OPTIONS BOARD")
    print("=" * 70)

    board = scan_all_tickers()

    processed_path = PROCESSED_DIR / "stock_setup_scores.csv"
    board_path = BOARD_DIR / "stock_setup_board.csv"

    board.to_csv(processed_path, index=False)
    board.to_csv(board_path, index=False)

    print("\nTop setups:")
    if board.empty:
        print("No setups found.")
    else:
        display_cols = [
            "Ticker",
            "Close",
            "SetupType",
            "FinalStockSetupScore",
            "BullishScore",
            "BearishScore",
            "RSI14",
            "ATRPercent",
            "RelVolume",
            "SuggestedStrategy",
        ]
        print(board[display_cols].head(15).to_string(index=False))

    print("\nSaved:")
    print(processed_path)
    print(board_path)


if __name__ == "__main__":
    main()