import math
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent))

from config import BOARD_DIR, REPORT_DIR, TARGET_DTE_MIN, TARGET_DTE_MAX
from ticker_universe import LIQUID_OPTIONS_UNIVERSE


# ============================================================
# Options Scanner Settings
# ============================================================

SCAN_TOP_N = 50

RISK_FREE_RATE = 0.045
CONTRACT_MULTIPLIER = 100

MAX_EXPIRATIONS_PER_TICKER = 5
MAX_CANDIDATES_PER_POOL = 25

MAX_TRADES_PER_TICKER_EXPIRATION = 3
MAX_TRADES_PER_TICKER_TOTAL = 3
MAX_TOTAL_REPORT_TRADES = 75

MIN_SPREAD_WIDTH = 0.50
MAX_SPREAD_WIDTH = 50.00

MIN_NET_PRICE = 0.01

MIN_STRIKE_MONEYNESS = 0.70
MAX_STRIKE_MONEYNESS = 1.30

DEFAULT_IV_FOR_DELTA = 0.40

MIN_ROBINHOOD_PRICE_MATCH_PCT = 0.80

# Short-term scanner mode.
# This is now intentionally built for 1-7 DTE trades.
SHORT_TERM_MODE = True


# ============================================================
# Math helpers
# ============================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def safe_float(value, default=np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_strike(value) -> str:
    value = float(value)

    if value.is_integer():
        return str(int(value))

    return str(round(value, 2))


def days_to_expiration(expiration: str) -> int:
    exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
    return (exp_date - date.today()).days


def estimate_delta(
    option_type: str,
    stock_price: float,
    strike: float,
    dte: int,
    implied_volatility: float,
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    if stock_price <= 0 or strike <= 0 or dte <= 0:
        return np.nan

    sigma = safe_float(implied_volatility, DEFAULT_IV_FOR_DELTA)

    if pd.isna(sigma) or sigma <= 0:
        sigma = DEFAULT_IV_FOR_DELTA

    sigma = min(max(float(sigma), 0.05), 3.00)
    t = dte / 365.0

    try:
        d1 = (
            math.log(stock_price / strike)
            + (risk_free_rate + 0.5 * sigma * sigma) * t
        ) / (sigma * math.sqrt(t))

        if option_type == "call":
            return norm_cdf(d1)

        if option_type == "put":
            return norm_cdf(d1) - 1.0

        return np.nan

    except Exception:
        return np.nan


# ============================================================
# Stock setup helpers
# ============================================================

def infer_bias(stock_row: pd.Series) -> str:
    setup_type = str(stock_row.get("SetupType", ""))

    if "Bullish" in setup_type:
        return "Bullish"

    if "Bearish" in setup_type:
        return "Bearish"

    bullish = safe_float(stock_row.get("BullishScore", 0), 0)
    bearish = safe_float(stock_row.get("BearishScore", 0), 0)

    if bullish >= bearish:
        return "Bullish"

    return "Bearish"


def infer_setup_score(stock_row: pd.Series) -> float:
    final_score = safe_float(stock_row.get("FinalStockSetupScore", np.nan), np.nan)

    if not pd.isna(final_score):
        return float(final_score)

    bullish = safe_float(stock_row.get("BullishScore", 0), 0)
    bearish = safe_float(stock_row.get("BearishScore", 0), 0)

    score = max(bullish, bearish)

    if score <= 0:
        return 50.0

    return float(score)


def clean_setup_type(stock_row: pd.Series) -> str:
    setup_type = str(stock_row.get("SetupType", ""))

    if setup_type and setup_type.lower() != "nan":
        return setup_type

    return "Directional Bias"


def get_fallback_stock_row(ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=True)

        if hist.empty:
            return None

        close = float(hist["Close"].iloc[-1])

        if len(hist) >= 6:
            return_5 = (hist["Close"].iloc[-1] / hist["Close"].iloc[-6]) - 1
        else:
            return_5 = 0

        if return_5 >= 0:
            setup_type = "Fallback Bullish Bias"
            bullish_score = 55
            bearish_score = 45
        else:
            setup_type = "Fallback Bearish Bias"
            bullish_score = 45
            bearish_score = 55

        return pd.Series(
            {
                "Ticker": ticker,
                "Close": close,
                "SetupType": setup_type,
                "BullishScore": bullish_score,
                "BearishScore": bearish_score,
                "FinalStockSetupScore": max(bullish_score, bearish_score),
            }
        )

    except Exception:
        return None


def build_scan_rows(stock_board: pd.DataFrame) -> list:
    rows = []

    stock_board = stock_board.copy()
    stock_board["Ticker"] = stock_board["Ticker"].astype(str)

    by_ticker = {
        row["Ticker"]: row
        for _, row in stock_board.iterrows()
    }

    for ticker in LIQUID_OPTIONS_UNIVERSE[:SCAN_TOP_N]:
        if ticker in by_ticker:
            rows.append(by_ticker[ticker])
        else:
            print(f"{ticker}: Missing from stock setup board. Using fallback price/bias.")
            fallback = get_fallback_stock_row(ticker)

            if fallback is not None:
                rows.append(fallback)
            else:
                print(f"{ticker}: Fallback failed. Skipping.")

    return rows


# ============================================================
# Options chain preparation
# ============================================================

def get_valid_expirations(ticker_obj: yf.Ticker) -> list:
    try:
        expirations = list(ticker_obj.options)
    except Exception:
        return []

    if not expirations:
        return []

    target = []

    for exp in expirations:
        try:
            dte = days_to_expiration(exp)

            if TARGET_DTE_MIN <= dte <= TARGET_DTE_MAX:
                target.append(exp)

        except Exception:
            continue

    if target:
        return target[:MAX_EXPIRATIONS_PER_TICKER]

    # Short-term fallback:
    # If nothing exists inside 1-7 DTE, allow up to 14 DTE.
    fallback = []

    for exp in expirations:
        try:
            dte = days_to_expiration(exp)

            if 1 <= dte <= 14:
                fallback.append(exp)

        except Exception:
            continue

    if fallback:
        return fallback[:MAX_EXPIRATIONS_PER_TICKER]

    # Final fallback:
    # Keep this so the scanner does not completely fail if a ticker
    # only has monthly expirations.
    longer_fallback = []

    for exp in expirations:
        try:
            dte = days_to_expiration(exp)

            if 15 <= dte <= 45:
                longer_fallback.append(exp)

        except Exception:
            continue

    return longer_fallback[:MAX_EXPIRATIONS_PER_TICKER]


def prepare_chain_side(
    df: pd.DataFrame,
    option_type: str,
    ticker: str,
    expiration: str,
    stock_price: float,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    dte = days_to_expiration(expiration)

    out = df.copy()

    out["Ticker"] = ticker
    out["OptionType"] = option_type
    out["Expiration"] = expiration
    out["DTE"] = dte

    for col in [
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "volume",
        "openInterest",
        "impliedVolatility",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    out["strike"] = out["strike"].apply(safe_float)
    out["bid"] = out["bid"].apply(safe_float)
    out["ask"] = out["ask"].apply(safe_float)
    out["lastPrice"] = out["lastPrice"].apply(safe_float)
    out["volume"] = out["volume"].fillna(0).apply(safe_float)
    out["openInterest"] = out["openInterest"].fillna(0).apply(safe_float)
    out["impliedVolatility"] = out["impliedVolatility"].apply(safe_float)

    out["HasRealBidAsk"] = (out["bid"] > 0) & (out["ask"] > 0)
    out["Mid"] = np.where(out["HasRealBidAsk"], (out["bid"] + out["ask"]) / 2, np.nan)

    out["CalcPrice"] = out["Mid"]
    out.loc[out["CalcPrice"].isna() | (out["CalcPrice"] <= 0), "CalcPrice"] = out["lastPrice"]

    out["CalcBid"] = out["bid"]
    out.loc[out["CalcBid"].isna() | (out["CalcBid"] <= 0), "CalcBid"] = out["CalcPrice"]

    out["CalcAsk"] = out["ask"]
    out.loc[out["CalcAsk"].isna() | (out["CalcAsk"] <= 0), "CalcAsk"] = out["CalcPrice"]

    out["BidAskSpread"] = out["ask"] - out["bid"]
    out["BidAskSpreadPct"] = np.where(
        out["HasRealBidAsk"] & (out["Mid"] > 0),
        out["BidAskSpread"] / out["Mid"],
        np.nan,
    )

    out["Delta"] = out.apply(
        lambda row: estimate_delta(
            option_type=option_type,
            stock_price=stock_price,
            strike=row["strike"],
            dte=dte,
            implied_volatility=row["impliedVolatility"],
        ),
        axis=1,
    )

    out["Moneyness"] = out["strike"] / stock_price

    out = out[
        [
            "Ticker",
            "OptionType",
            "Expiration",
            "DTE",
            "contractSymbol",
            "strike",
            "bid",
            "ask",
            "lastPrice",
            "Mid",
            "CalcPrice",
            "CalcBid",
            "CalcAsk",
            "HasRealBidAsk",
            "BidAskSpread",
            "BidAskSpreadPct",
            "volume",
            "openInterest",
            "impliedVolatility",
            "Delta",
            "Moneyness",
        ]
    ]

    return out


def filter_usable_contracts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    out = out[
        (out["strike"].notna())
        & (out["CalcPrice"].notna())
        & (out["CalcPrice"] > 0)
        & (out["CalcBid"].notna())
        & (out["CalcAsk"].notna())
        & (out["Delta"].notna())
        & (out["Moneyness"] >= MIN_STRIKE_MONEYNESS)
        & (out["Moneyness"] <= MAX_STRIKE_MONEYNESS)
    ]

    return out


# ============================================================
# Scoring
# ============================================================

def liquidity_score(row: pd.Series) -> float:
    oi = max(float(row["openInterest"]), 0)
    vol = max(float(row["volume"]), 0)

    oi_score = min(25, math.log10(oi + 1) * 8)
    vol_score = min(15, math.log10(vol + 1) * 5)

    if bool(row["HasRealBidAsk"]):
        spread_pct = safe_float(row["BidAskSpreadPct"], 9.99)

        if spread_pct <= 0.08:
            spread_score = 40
        elif spread_pct <= 0.15:
            spread_score = 34
        elif spread_pct <= 0.25:
            spread_score = 26
        elif spread_pct <= 0.40:
            spread_score = 18
        else:
            spread_score = 10
    else:
        spread_score = 6

    return oi_score + vol_score + spread_score


def dte_score(dte: int) -> float:
    # New short-term scoring.
    # Preferred range is 2-7 DTE.
    if 2 <= dte <= 7:
        return 22

    if dte == 1:
        return 16

    if 8 <= dte <= 14:
        return 12

    if 15 <= dte <= 30:
        return 8

    if 31 <= dte <= 60:
        return 5

    return 2


def delta_fit_score(delta: float, target_abs_delta: float) -> float:
    if pd.isna(delta):
        return 0

    d = abs(float(delta))
    distance = abs(d - target_abs_delta)

    if distance <= 0.05:
        return 20
    if distance <= 0.10:
        return 16
    if distance <= 0.15:
        return 12
    if distance <= 0.20:
        return 8

    return 4


def quality_flag(long_leg: pd.Series, short_leg: pd.Series) -> str:
    both_real = bool(long_leg["HasRealBidAsk"]) and bool(short_leg["HasRealBidAsk"])
    min_oi = min(float(long_leg["openInterest"]), float(short_leg["openInterest"]))

    if both_real:
        avg_spread = (
            safe_float(long_leg["BidAskSpreadPct"], 9.99)
            + safe_float(short_leg["BidAskSpreadPct"], 9.99)
        ) / 2

        if avg_spread <= 0.15 and min_oi >= 100:
            return "Good liquidity"

        if avg_spread <= 0.35 and min_oi >= 25:
            return "Tradable check"

        return "Wide spread warning"

    return "Estimated quote — verify"


def action_grade(
    quote_quality: str,
    final_score: float,
    reward_risk: float,
    debit_or_credit: str,
    net_price: float,
) -> str:
    if "Estimated quote" in quote_quality:
        return "WATCH ONLY"

    if "Wide spread" in quote_quality:
        return "VERIFY ONLY"

    if net_price <= 0:
        return "SKIP"

    if reward_risk > 12:
        return "VERIFY ONLY"

    if final_score >= 65 and "Good liquidity" in quote_quality:
        return "ACTIONABLE CHECK"

    if final_score >= 55 and ("Good liquidity" in quote_quality or "Tradable check" in quote_quality):
        return "TRADABLE CHECK"

    return "VERIFY ONLY"


def action_rank(grade: str) -> int:
    if grade == "ACTIONABLE CHECK":
        return 1
    if grade == "TRADABLE CHECK":
        return 2
    if grade == "VERIFY ONLY":
        return 3
    if grade == "WATCH ONLY":
        return 4
    return 5


def candidate_pool(
    df: pd.DataFrame,
    min_abs_delta: float,
    max_abs_delta: float,
    target_abs_delta: float,
    stock_price: float,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    out["AbsDelta"] = out["Delta"].abs()
    out["DeltaDistance"] = (out["AbsDelta"] - target_abs_delta).abs()
    out["LegLiquidityScore"] = out.apply(liquidity_score, axis=1)
    out["DistanceFromStock"] = (out["strike"] - stock_price).abs()

    out = out[
        (out["AbsDelta"] >= min_abs_delta)
        & (out["AbsDelta"] <= max_abs_delta)
    ]

    if out.empty:
        return out

    out = out.sort_values(
        by=[
            "DeltaDistance",
            "DistanceFromStock",
            "LegLiquidityScore",
            "openInterest",
            "volume",
        ],
        ascending=[True, True, False, False, False],
    )

    return out.head(MAX_CANDIDATES_PER_POOL)


def make_trade(
    ticker: str,
    stock_price: float,
    bias: str,
    setup_type: str,
    setup_score: float,
    strategy: str,
    expiration: str,
    long_leg: pd.Series,
    short_leg: pd.Series,
    option_word: str,
    debit_or_credit: str,
    net_price: float,
    width: float,
    breakeven: float,
    reward_risk: float,
    target_delta: float,
    robinhood_action: str,
) -> dict:
    if debit_or_credit == "Debit":
        max_profit = width - net_price
        max_loss = net_price
        min_robinhood_price = net_price * 1.20
        verify_rule = (
            f"For debit trades, Robinhood debit should be no more than "
            f"${min_robinhood_price:.2f}. Lower is better."
        )
    else:
        max_profit = net_price
        max_loss = width - net_price
        min_robinhood_price = net_price * MIN_ROBINHOOD_PRICE_MATCH_PCT
        verify_rule = (
            f"For credit trades, Robinhood credit should be at least "
            f"${min_robinhood_price:.2f}. Higher is better."
        )

    avg_liquidity = (liquidity_score(long_leg) + liquidity_score(short_leg)) / 2
    delta_score = delta_fit_score(short_leg["Delta"], target_delta)
    expiration_score = dte_score(int(long_leg["DTE"]))

    if debit_or_credit == "Debit":
        rr_score = min(20, reward_risk * 8)
    else:
        rr_score = min(20, reward_risk * 55)

    quote_quality = quality_flag(long_leg, short_leg)

    final_score = (
        setup_score * 0.34
        + avg_liquidity * 0.31
        + delta_score * 0.12
        + expiration_score * 0.10
        + rr_score * 0.13
    )

    if "Estimated quote" in quote_quality:
        final_score -= 18

    if "Wide spread" in quote_quality:
        final_score -= 10

    if reward_risk > 8:
        final_score -= 12

    if reward_risk > 20:
        final_score -= 20

    avg_spread_pct = np.nan

    if bool(long_leg["HasRealBidAsk"]) and bool(short_leg["HasRealBidAsk"]):
        avg_spread_pct = (
            safe_float(long_leg["BidAskSpreadPct"], np.nan)
            + safe_float(short_leg["BidAskSpreadPct"], np.nan)
        ) / 2

    grade = action_grade(
        quote_quality=quote_quality,
        final_score=final_score,
        reward_risk=reward_risk,
        debit_or_credit=debit_or_credit,
        net_price=net_price,
    )

    robinhood_chain_url = f"https://robinhood.com/options/chains/{ticker}"

    return {
        "Ticker": ticker,
        "RobinhoodChainUrl": robinhood_chain_url,
        "StockPrice": round(stock_price, 2),
        "Bias": bias,
        "SetupType": setup_type,
        "Strategy": strategy,
        "Expiration": expiration,
        "DTE": int(long_leg["DTE"]),
        "BuyLeg": f"Buy {format_strike(long_leg['strike'])} {option_word}",
        "SellLeg": f"Sell {format_strike(short_leg['strike'])} {option_word}",
        "BuyStrike": float(long_leg["strike"]),
        "SellStrike": float(short_leg["strike"]),
        "NetDebitCredit": round(net_price, 2),
        "DebitOrCredit": debit_or_credit,
        "MinimumRobinhoodPrice": round(min_robinhood_price, 2),
        "VerifyRule": verify_rule,
        "SpreadWidth": round(width, 2),
        "MaxProfit": round(max_profit * CONTRACT_MULTIPLIER, 2),
        "MaxLoss": round(max_loss * CONTRACT_MULTIPLIER, 2),
        "Breakeven": round(breakeven, 2),
        "BuyDelta": round(float(long_leg["Delta"]), 2),
        "SellDelta": round(float(short_leg["Delta"]), 2),
        "BuyIV": round(safe_float(long_leg["impliedVolatility"], 0) * 100, 2),
        "SellIV": round(safe_float(short_leg["impliedVolatility"], 0) * 100, 2),
        "BuyOI": int(safe_float(long_leg["openInterest"], 0)),
        "SellOI": int(safe_float(short_leg["openInterest"], 0)),
        "BuyVolume": int(safe_float(long_leg["volume"], 0)),
        "SellVolume": int(safe_float(short_leg["volume"], 0)),
        "BuyQuoteReal": bool(long_leg["HasRealBidAsk"]),
        "SellQuoteReal": bool(short_leg["HasRealBidAsk"]),
        "AvgBidAskSpreadPct": "" if pd.isna(avg_spread_pct) else round(avg_spread_pct * 100, 2),
        "RewardRisk": round(reward_risk, 2),
        "StockSetupScore": round(setup_score, 2),
        "OptionsLiquidityScore": round(avg_liquidity, 2),
        "QualityFlag": quote_quality,
        "FinalScore": round(final_score, 2),
        "ActionGrade": grade,
        "ActionRank": action_rank(grade),
        "RobinhoodAction": robinhood_action,
    }


# ============================================================
# Spread builders
# ============================================================

def build_call_debit_spreads(
    ticker: str,
    stock_price: float,
    setup_score: float,
    setup_type: str,
    expiration: str,
    calls: pd.DataFrame,
) -> list:
    trades = []

    buy_candidates = candidate_pool(calls, 0.25, 0.80, 0.50, stock_price)
    sell_candidates = candidate_pool(calls, 0.05, 0.60, 0.30, stock_price)

    for _, buy in buy_candidates.iterrows():
        possible_sells = sell_candidates[sell_candidates["strike"] > buy["strike"]]

        for _, sell in possible_sells.iterrows():
            width = float(sell["strike"] - buy["strike"])

            if width < MIN_SPREAD_WIDTH or width > MAX_SPREAD_WIDTH:
                continue

            debit = float(buy["CalcAsk"] - sell["CalcBid"])

            if debit < MIN_NET_PRICE or debit >= width:
                continue

            max_profit = width - debit
            max_loss = debit

            if max_profit <= 0 or max_loss <= 0:
                continue

            reward_risk = max_profit / max_loss
            breakeven = float(buy["strike"] + debit)

            trades.append(
                make_trade(
                    ticker=ticker,
                    stock_price=stock_price,
                    bias="Bullish",
                    setup_type=setup_type,
                    setup_score=setup_score,
                    strategy="Call Debit Spread",
                    expiration=expiration,
                    long_leg=buy,
                    short_leg=sell,
                    option_word="Call",
                    debit_or_credit="Debit",
                    net_price=debit,
                    width=width,
                    breakeven=breakeven,
                    reward_risk=reward_risk,
                    target_delta=0.50,
                    robinhood_action="Open Robinhood → ticker → options → expiration → build call debit spread → verify debit, max loss, and liquidity",
                )
            )

    return sorted(trades, key=lambda x: x["FinalScore"], reverse=True)[:MAX_TRADES_PER_TICKER_EXPIRATION]


def build_put_debit_spreads(
    ticker: str,
    stock_price: float,
    setup_score: float,
    setup_type: str,
    expiration: str,
    puts: pd.DataFrame,
) -> list:
    trades = []

    buy_candidates = candidate_pool(puts, 0.25, 0.80, 0.50, stock_price)
    sell_candidates = candidate_pool(puts, 0.05, 0.60, 0.30, stock_price)

    for _, buy in buy_candidates.iterrows():
        possible_sells = sell_candidates[sell_candidates["strike"] < buy["strike"]]

        for _, sell in possible_sells.iterrows():
            width = float(buy["strike"] - sell["strike"])

            if width < MIN_SPREAD_WIDTH or width > MAX_SPREAD_WIDTH:
                continue

            debit = float(buy["CalcAsk"] - sell["CalcBid"])

            if debit < MIN_NET_PRICE or debit >= width:
                continue

            max_profit = width - debit
            max_loss = debit

            if max_profit <= 0 or max_loss <= 0:
                continue

            reward_risk = max_profit / max_loss
            breakeven = float(buy["strike"] - debit)

            trades.append(
                make_trade(
                    ticker=ticker,
                    stock_price=stock_price,
                    bias="Bearish",
                    setup_type=setup_type,
                    setup_score=setup_score,
                    strategy="Put Debit Spread",
                    expiration=expiration,
                    long_leg=buy,
                    short_leg=sell,
                    option_word="Put",
                    debit_or_credit="Debit",
                    net_price=debit,
                    width=width,
                    breakeven=breakeven,
                    reward_risk=reward_risk,
                    target_delta=0.50,
                    robinhood_action="Open Robinhood → ticker → options → expiration → build put debit spread → verify debit, max loss, and liquidity",
                )
            )

    return sorted(trades, key=lambda x: x["FinalScore"], reverse=True)[:MAX_TRADES_PER_TICKER_EXPIRATION]


def build_put_credit_spreads(
    ticker: str,
    stock_price: float,
    setup_score: float,
    setup_type: str,
    expiration: str,
    puts: pd.DataFrame,
) -> list:
    trades = []

    # Short-term spreads need to stay closer to the money.
    sell_candidates = candidate_pool(puts, 0.12, 0.60, 0.35, stock_price)
    buy_candidates = candidate_pool(puts, 0.02, 0.50, 0.18, stock_price)

    for _, sell in sell_candidates.iterrows():
        possible_buys = buy_candidates[buy_candidates["strike"] < sell["strike"]]

        for _, buy in possible_buys.iterrows():
            width = float(sell["strike"] - buy["strike"])

            if width < MIN_SPREAD_WIDTH or width > MAX_SPREAD_WIDTH:
                continue

            credit = float(sell["CalcBid"] - buy["CalcAsk"])

            if credit < MIN_NET_PRICE or credit >= width:
                continue

            max_profit = credit
            max_loss = width - credit

            if max_profit <= 0 or max_loss <= 0:
                continue

            reward_risk = max_profit / max_loss
            breakeven = float(sell["strike"] - credit)

            trades.append(
                make_trade(
                    ticker=ticker,
                    stock_price=stock_price,
                    bias="Bullish / Neutral-Bullish",
                    setup_type=setup_type,
                    setup_score=setup_score,
                    strategy="Put Credit Spread",
                    expiration=expiration,
                    long_leg=buy,
                    short_leg=sell,
                    option_word="Put",
                    debit_or_credit="Credit",
                    net_price=credit,
                    width=width,
                    breakeven=breakeven,
                    reward_risk=reward_risk,
                    target_delta=0.35,
                    robinhood_action="Open Robinhood → ticker → options → expiration → build put credit spread → verify credit, max loss, and liquidity",
                )
            )

    return sorted(trades, key=lambda x: x["FinalScore"], reverse=True)[:MAX_TRADES_PER_TICKER_EXPIRATION]


def build_call_credit_spreads(
    ticker: str,
    stock_price: float,
    setup_score: float,
    setup_type: str,
    expiration: str,
    calls: pd.DataFrame,
) -> list:
    trades = []

    # Short-term spreads need to stay closer to the money.
    sell_candidates = candidate_pool(calls, 0.12, 0.60, 0.35, stock_price)
    buy_candidates = candidate_pool(calls, 0.02, 0.50, 0.18, stock_price)

    for _, sell in sell_candidates.iterrows():
        possible_buys = buy_candidates[buy_candidates["strike"] > sell["strike"]]

        for _, buy in possible_buys.iterrows():
            width = float(buy["strike"] - sell["strike"])

            if width < MIN_SPREAD_WIDTH or width > MAX_SPREAD_WIDTH:
                continue

            credit = float(sell["CalcBid"] - buy["CalcAsk"])

            if credit < MIN_NET_PRICE or credit >= width:
                continue

            max_profit = credit
            max_loss = width - credit

            if max_profit <= 0 or max_loss <= 0:
                continue

            reward_risk = max_profit / max_loss
            breakeven = float(sell["strike"] + credit)

            trades.append(
                make_trade(
                    ticker=ticker,
                    stock_price=stock_price,
                    bias="Bearish / Neutral-Bearish",
                    setup_type=setup_type,
                    setup_score=setup_score,
                    strategy="Call Credit Spread",
                    expiration=expiration,
                    long_leg=buy,
                    short_leg=sell,
                    option_word="Call",
                    debit_or_credit="Credit",
                    net_price=credit,
                    width=width,
                    breakeven=breakeven,
                    reward_risk=reward_risk,
                    target_delta=0.35,
                    robinhood_action="Open Robinhood → ticker → options → expiration → build call credit spread → verify credit, max loss, and liquidity",
                )
            )

    return sorted(trades, key=lambda x: x["FinalScore"], reverse=True)[:MAX_TRADES_PER_TICKER_EXPIRATION]


# ============================================================
# Main ticker scanner
# ============================================================

def scan_ticker_options(stock_row: pd.Series):
    ticker = str(stock_row["Ticker"])
    stock_price = float(stock_row["Close"])
    setup_type = clean_setup_type(stock_row)
    setup_score = infer_setup_score(stock_row)
    bias = infer_bias(stock_row)

    ticker_obj = yf.Ticker(ticker)

    all_trades = []
    diagnostics = []

    expirations = get_valid_expirations(ticker_obj)

    if not expirations:
        diagnostics.append(
            {
                "Ticker": ticker,
                "Expiration": "None",
                "DTE": None,
                "RawCalls": 0,
                "RawPuts": 0,
                "UsableCalls": 0,
                "UsablePuts": 0,
                "TradesBuilt": 0,
                "Status": "No valid expirations",
            }
        )

        print(f"{ticker}: No valid expirations.")

        return all_trades, diagnostics

    for exp in expirations:
        try:
            dte = days_to_expiration(exp)
            chain = ticker_obj.option_chain(exp)

            calls = prepare_chain_side(
                df=chain.calls,
                option_type="call",
                ticker=ticker,
                expiration=exp,
                stock_price=stock_price,
            )

            puts = prepare_chain_side(
                df=chain.puts,
                option_type="put",
                ticker=ticker,
                expiration=exp,
                stock_price=stock_price,
            )

            raw_calls = len(calls)
            raw_puts = len(puts)

            calls = filter_usable_contracts(calls)
            puts = filter_usable_contracts(puts)

            usable_calls = len(calls)
            usable_puts = len(puts)

            before = len(all_trades)

            if bias == "Bullish":
                all_trades.extend(
                    build_call_debit_spreads(
                        ticker=ticker,
                        stock_price=stock_price,
                        setup_score=setup_score,
                        setup_type=setup_type,
                        expiration=exp,
                        calls=calls,
                    )
                )

                all_trades.extend(
                    build_put_credit_spreads(
                        ticker=ticker,
                        stock_price=stock_price,
                        setup_score=setup_score,
                        setup_type=setup_type,
                        expiration=exp,
                        puts=puts,
                    )
                )

            else:
                all_trades.extend(
                    build_put_debit_spreads(
                        ticker=ticker,
                        stock_price=stock_price,
                        setup_score=setup_score,
                        setup_type=setup_type,
                        expiration=exp,
                        puts=puts,
                    )
                )

                all_trades.extend(
                    build_call_credit_spreads(
                        ticker=ticker,
                        stock_price=stock_price,
                        setup_score=setup_score,
                        setup_type=setup_type,
                        expiration=exp,
                        calls=calls,
                    )
                )

            built = len(all_trades) - before

            diagnostics.append(
                {
                    "Ticker": ticker,
                    "Expiration": exp,
                    "DTE": dte,
                    "RawCalls": raw_calls,
                    "RawPuts": raw_puts,
                    "UsableCalls": usable_calls,
                    "UsablePuts": usable_puts,
                    "TradesBuilt": built,
                    "Status": "OK",
                }
            )

            print(
                f"  {ticker} {exp}: raw calls {raw_calls}, raw puts {raw_puts}, "
                f"usable calls {usable_calls}, usable puts {usable_puts}, built {built}"
            )

        except Exception as e:
            diagnostics.append(
                {
                    "Ticker": ticker,
                    "Expiration": exp,
                    "DTE": None,
                    "RawCalls": 0,
                    "RawPuts": 0,
                    "UsableCalls": 0,
                    "UsablePuts": 0,
                    "TradesBuilt": 0,
                    "Status": f"Error: {e}",
                }
            )

            print(f"  {ticker} {exp}: ERROR -> {e}")

    return all_trades, diagnostics


# ============================================================
# HTML report
# ============================================================

def build_html_report(
    trades: pd.DataFrame,
    diagnostics: pd.DataFrame,
    scanned_count: int,
) -> str:
    if trades.empty:
        rows_html = """
        <tr>
            <td colspan="22">
                No qualifying spread trades found. Check diagnostics to see where candidates are being filtered out.
            </td>
        </tr>
        """
    else:
        rows_html = ""

        top = trades.head(MAX_TOTAL_REPORT_TRADES).copy()

        for _, row in top.iterrows():
            score = float(row["FinalScore"])
            grade = str(row["ActionGrade"])

            if grade == "ACTIONABLE CHECK":
                grade_class = "grade-action"
            elif grade == "TRADABLE CHECK":
                grade_class = "grade-tradable"
            elif grade == "VERIFY ONLY":
                grade_class = "grade-verify"
            else:
                grade_class = "grade-watch"

            if score >= 75:
                score_class = "score-good"
            elif score >= 55:
                score_class = "score-ok"
            else:
                score_class = "score-watch"

            rows_html += f"""
            <tr>
                <td>
                    <strong>{row['Ticker']}</strong><br>
                    <span class="muted">${row['StockPrice']}</span><br>
                    <a href="{row['RobinhoodChainUrl']}" target="_blank">Open Chain</a>
                </td>
                <td><span class="{grade_class}">{row['ActionGrade']}</span></td>
                <td>{row['Bias']}<br><span class="muted">{row['SetupType']}</span></td>
                <td><strong>{row['Strategy']}</strong><br><span class="muted">{row['QualityFlag']}</span></td>
                <td>{row['Expiration']}<br><span class="muted">{row['DTE']} DTE</span></td>
                <td>{row['BuyLeg']}<br>{row['SellLeg']}</td>
                <td>{row['DebitOrCredit']}<br><strong>${row['NetDebitCredit']}</strong></td>
                <td>${row['MinimumRobinhoodPrice']}</td>
                <td>${row['MaxProfit']}</td>
                <td>${row['MaxLoss']}</td>
                <td>${row['Breakeven']}</td>
                <td>{row['BuyDelta']} / {row['SellDelta']}</td>
                <td>{row['BuyIV']}% / {row['SellIV']}%</td>
                <td>{row['BuyOI']} / {row['SellOI']}</td>
                <td>{row['BuyVolume']} / {row['SellVolume']}</td>
                <td>{row['AvgBidAskSpreadPct']}</td>
                <td>{row['RewardRisk']}</td>
                <td>{row['StockSetupScore']}</td>
                <td>{row['OptionsLiquidityScore']}</td>
                <td class="{score_class}">{row['FinalScore']}</td>
                <td>{row['VerifyRule']}</td>
                <td>{row['RobinhoodAction']}</td>
            </tr>
            """

    trades_found = 0 if trades.empty else len(trades)
    top_score = 0 if trades.empty else trades["FinalScore"].max()
    tickers_with_trades = 0 if trades.empty else trades["Ticker"].nunique()

    actionable_count = 0
    tradable_count = 0
    watch_count = 0

    if not trades.empty and "ActionGrade" in trades.columns:
        actionable_count = int((trades["ActionGrade"] == "ACTIONABLE CHECK").sum())
        tradable_count = int((trades["ActionGrade"] == "TRADABLE CHECK").sum())
        watch_count = int((trades["ActionGrade"] == "WATCH ONLY").sum())

    diag_rows = ""

    if not diagnostics.empty:
        diag_top = diagnostics.head(100).copy()

        for _, row in diag_top.iterrows():
            diag_rows += f"""
            <tr>
                <td>{row['Ticker']}</td>
                <td>{row['Expiration']}</td>
                <td>{row['DTE']}</td>
                <td>{row['RawCalls']}</td>
                <td>{row['RawPuts']}</td>
                <td>{row['UsableCalls']}</td>
                <td>{row['UsablePuts']}</td>
                <td>{row['TradesBuilt']}</td>
                <td>{row['Status']}</td>
            </tr>
            """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Robinhood Options Contract Board</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0b0f14;
                color: #e6edf3;
                padding: 24px;
            }}

            h1 {{
                margin-bottom: 6px;
            }}

            h2 {{
                margin-top: 34px;
            }}

            a {{
                color: #7dd3fc;
                text-decoration: none;
                font-size: 11px;
            }}

            a:hover {{
                text-decoration: underline;
            }}

            .subtitle {{
                color: #9ba7b4;
                margin-bottom: 20px;
            }}

            .warning {{
                background: #241a0d;
                border: 1px solid #755118;
                padding: 14px;
                color: #ffd38a;
                margin-bottom: 20px;
                line-height: 1.5;
            }}

            .summary {{
                display: grid;
                grid-template-columns: repeat(6, minmax(130px, 1fr));
                gap: 12px;
                margin-bottom: 20px;
            }}

            .card {{
                background: #111820;
                border: 1px solid #263241;
                padding: 14px;
                border-radius: 10px;
            }}

            .card .label {{
                color: #9ba7b4;
                font-size: 12px;
                margin-bottom: 4px;
            }}

            .card .value {{
                font-size: 22px;
                font-weight: bold;
            }}

            table {{
                width: 100%;
                border-collapse: collapse;
                background: #111820;
                border: 1px solid #263241;
                font-size: 12px;
                margin-bottom: 28px;
            }}

            th {{
                position: sticky;
                top: 0;
                background: #182231;
                color: #ffffff;
                text-align: left;
                padding: 10px;
                z-index: 2;
            }}

            td {{
                border-top: 1px solid #263241;
                padding: 10px;
                vertical-align: top;
            }}

            tr:hover {{
                background: #16202d;
            }}

            .muted {{
                color: #9ba7b4;
                font-size: 11px;
            }}

            .score-good {{
                color: #8ff0a4;
                font-size: 18px;
                font-weight: bold;
            }}

            .score-ok {{
                color: #ffd38a;
                font-size: 18px;
                font-weight: bold;
            }}

            .score-watch {{
                color: #ff9b9b;
                font-size: 18px;
                font-weight: bold;
            }}

            .grade-action {{
                display: inline-block;
                padding: 4px 7px;
                border-radius: 999px;
                background: #10381f;
                color: #8ff0a4;
                font-weight: bold;
                font-size: 11px;
            }}

            .grade-tradable {{
                display: inline-block;
                padding: 4px 7px;
                border-radius: 999px;
                background: #2e2a11;
                color: #ffd38a;
                font-weight: bold;
                font-size: 11px;
            }}

            .grade-verify {{
                display: inline-block;
                padding: 4px 7px;
                border-radius: 999px;
                background: #2d1f12;
                color: #ffbf80;
                font-weight: bold;
                font-size: 11px;
            }}

            .grade-watch {{
                display: inline-block;
                padding: 4px 7px;
                border-radius: 999px;
                background: #331717;
                color: #ff9b9b;
                font-weight: bold;
                font-size: 11px;
            }}
        </style>
    </head>
    <body>
        <h1>Robinhood Options Contract Board</h1>
        <div class="subtitle">
            Short-Term Version — Scanning for 1-7 DTE contracts, with fallback up to 14 DTE if necessary.
        </div>

        <div class="warning">
            This scanner does not place trades. This board is a candidate generator for Robinhood manual verification.
            Short-term options can move quickly. Verify all pricing, max loss, and breakeven inside Robinhood before making any decision.
        </div>

        <div class="summary">
            <div class="card">
                <div class="label">Tickers Scanned</div>
                <div class="value">{scanned_count}</div>
            </div>
            <div class="card">
                <div class="label">Tickers With Trades</div>
                <div class="value">{tickers_with_trades}</div>
            </div>
            <div class="card">
                <div class="label">Trades Shown</div>
                <div class="value">{trades_found}</div>
            </div>
            <div class="card">
                <div class="label">Actionable</div>
                <div class="value">{actionable_count}</div>
            </div>
            <div class="card">
                <div class="label">Tradable Check</div>
                <div class="value">{tradable_count}</div>
            </div>
            <div class="card">
                <div class="label">Watch Only</div>
                <div class="value">{watch_count}</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Action Grade</th>
                    <th>Bias</th>
                    <th>Strategy</th>
                    <th>Expiration</th>
                    <th>Legs</th>
                    <th>Debit/Credit</th>
                    <th>Min RH Price</th>
                    <th>Max Profit</th>
                    <th>Max Loss</th>
                    <th>Breakeven</th>
                    <th>Deltas</th>
                    <th>IVs</th>
                    <th>Open Interest</th>
                    <th>Volume</th>
                    <th>Avg Spread %</th>
                    <th>Reward/Risk</th>
                    <th>Stock Score</th>
                    <th>Liquidity</th>
                    <th>Score</th>
                    <th>Verify Rule</th>
                    <th>Robinhood Action</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <h2>Diagnostics</h2>
        <div class="subtitle">
            This shows whether short-term chains are coming through and how many contracts are usable after filtering.
        </div>

        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Expiration</th>
                    <th>DTE</th>
                    <th>Raw Calls</th>
                    <th>Raw Puts</th>
                    <th>Usable Calls</th>
                    <th>Usable Puts</th>
                    <th>Trades Built</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {diag_rows}
            </tbody>
        </table>
    </body>
    </html>
    """

    return html


# ============================================================
# Main
# ============================================================

def main():
    stock_board_path = BOARD_DIR / "stock_setup_board.csv"

    if not stock_board_path.exists():
        raise FileNotFoundError(
            f"Could not find {stock_board_path}. Run fetch_stock_data.py and stock_setup_scanner.py first."
        )

    stock_board = pd.read_csv(stock_board_path)
    scan_rows = build_scan_rows(stock_board)

    print("=" * 70)
    print("SCANNING OPTIONS CONTRACTS — SHORT-TERM 1-7 DTE MODE")
    print("=" * 70)
    print(f"Tickers requested: {SCAN_TOP_N}")
    print(f"Tickers available to scan: {len(scan_rows)}")
    print(f"Target DTE range: {TARGET_DTE_MIN} to {TARGET_DTE_MAX}")
    print("")

    all_trades = []
    all_diagnostics = []

    for row in scan_rows:
        ticker = str(row["Ticker"])
        bias = infer_bias(row)
        setup_type = clean_setup_type(row)
        score = infer_setup_score(row)

        print(f"Scanning {ticker} | bias {bias} | setup {setup_type} | stock score {score}...")

        try:
            trades, diagnostics = scan_ticker_options(row)

            all_trades.extend(trades)
            all_diagnostics.extend(diagnostics)

            print(f"  Found {len(trades)} qualifying spread candidates for {ticker}.")
            print("")

        except Exception as e:
            print(f"  FAILED {ticker}: {e}")
            print("")

            all_diagnostics.append(
                {
                    "Ticker": ticker,
                    "Expiration": "Error",
                    "DTE": None,
                    "RawCalls": 0,
                    "RawPuts": 0,
                    "UsableCalls": 0,
                    "UsablePuts": 0,
                    "TradesBuilt": 0,
                    "Status": f"Ticker scan failed: {e}",
                }
            )

    trades_df = pd.DataFrame(all_trades)
    diagnostics_df = pd.DataFrame(all_diagnostics)

    if not trades_df.empty:
        trades_df = trades_df.sort_values(
            by=[
                "ActionRank",
                "DTE",
                "FinalScore",
                "StockSetupScore",
                "OptionsLiquidityScore",
                "RewardRisk",
            ],
            ascending=[True, True, False, False, False, False],
        )

        trades_df = (
            trades_df
            .groupby("Ticker", group_keys=False)
            .head(MAX_TRADES_PER_TICKER_TOTAL)
            .reset_index(drop=True)
        )

        trades_df = trades_df.sort_values(
            by=[
                "ActionRank",
                "DTE",
                "FinalScore",
                "StockSetupScore",
                "OptionsLiquidityScore",
                "RewardRisk",
            ],
            ascending=[True, True, False, False, False, False],
        ).head(MAX_TOTAL_REPORT_TRADES)

    BOARD_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    board_output_path = BOARD_DIR / "options_contract_board.csv"
    diagnostics_output_path = BOARD_DIR / "options_scan_diagnostics.csv"
    report_output_path = REPORT_DIR / "options_contract_board.html"

    trades_df.to_csv(board_output_path, index=False)
    diagnostics_df.to_csv(diagnostics_output_path, index=False)

    html = build_html_report(
        trades=trades_df,
        diagnostics=diagnostics_df,
        scanned_count=len(scan_rows),
    )

    with open(report_output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("=" * 70)
    print("OPTIONS CONTRACT SCAN COMPLETE")
    print("=" * 70)

    if trades_df.empty:
        print("No qualifying spread candidates found.")
        print("Open the diagnostics section or CSV to see where candidates are being filtered out.")
    else:
        display_cols = [
            "Ticker",
            "ActionGrade",
            "Bias",
            "Strategy",
            "Expiration",
            "DTE",
            "BuyLeg",
            "SellLeg",
            "DebitOrCredit",
            "NetDebitCredit",
            "MinimumRobinhoodPrice",
            "MaxProfit",
            "MaxLoss",
            "Breakeven",
            "QualityFlag",
            "FinalScore",
        ]

        print(trades_df[display_cols].head(30).to_string(index=False))

    print("")
    print("Saved:")
    print(board_output_path)
    print(diagnostics_output_path)
    print(report_output_path)


if __name__ == "__main__":
    main()