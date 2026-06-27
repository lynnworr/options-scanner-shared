import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

BOARD_DIR = BASE_DIR / "outputs" / "boards"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

OPTIONS_BOARD_PATH = BOARD_DIR / "options_contract_board.csv"
DIAGNOSTICS_PATH = BOARD_DIR / "options_scan_diagnostics.csv"


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Options Scanner Board",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Helpers
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default

        value = str(value).replace("$", "").replace(",", "").strip()

        if value == "":
            return default

        return float(value)

    except Exception:
        return default


def run_script(script_name: str):
    script_path = BASE_DIR / "src" / script_name

    if not script_path.exists():
        return False, f"Could not find {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
    )

    output = ""

    if result.stdout:
        output += result.stdout

    if result.stderr:
        output += "\n\nERRORS:\n" + result.stderr

    return result.returncode == 0, output


@st.cache_data(show_spinner=False)
def load_options_board():
    if not OPTIONS_BOARD_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(OPTIONS_BOARD_PATH)

    if df.empty:
        return df

    numeric_cols = [
        "StockPrice",
        "NetDebitCredit",
        "MinimumRobinhoodPrice",
        "SpreadWidth",
        "MaxProfit",
        "MaxLoss",
        "Breakeven",
        "RewardRisk",
        "StockSetupScore",
        "OptionsLiquidityScore",
        "FinalScore",
        "BuyStrike",
        "SellStrike",
        "DTE",
        "ActionRank",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_diagnostics():
    if not DIAGNOSTICS_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DIAGNOSTICS_PATH)

    return df


def manual_verify_priority(row: pd.Series):
    """
    Ranks which scanner ideas are worth checking manually in Robinhood first.
    This is not a trade signal. It only tells you which ideas deserve attention first.
    """
    final_score = safe_float(row.get("FinalScore", 0), 0)
    stock_score = safe_float(row.get("StockSetupScore", 0), 0)
    reward_risk = safe_float(row.get("RewardRisk", 0), 0)
    dte = safe_float(row.get("DTE", 0), 0)

    quality = str(row.get("QualityFlag", ""))
    strategy = str(row.get("Strategy", ""))

    priority_score = 0
    reasons = []

    if final_score >= 55:
        priority_score += 4
        reasons.append("strong scanner score")
    elif final_score >= 40:
        priority_score += 3
        reasons.append("good scanner score")
    elif final_score >= 30:
        priority_score += 2
        reasons.append("decent scanner score")
    elif final_score >= 25:
        priority_score += 1
        reasons.append("lower but usable scanner score")

    if stock_score >= 90:
        priority_score += 2
        reasons.append("strong stock setup")
    elif stock_score >= 75:
        priority_score += 1
        reasons.append("solid stock setup")

    if 1 <= dte <= 7:
        priority_score += 3
        reasons.append("short-term DTE")
    elif 8 <= dte <= 14:
        priority_score += 1
        reasons.append("near-term fallback DTE")

    if "Credit" in strategy:
        if 0.50 <= reward_risk <= 2.00:
            priority_score += 3
            reasons.append("strong credit spread reward/risk")
        elif 0.30 <= reward_risk < 0.50:
            priority_score += 2
            reasons.append("reasonable credit spread reward/risk")
        elif 0.20 <= reward_risk < 0.30:
            priority_score += 1
            reasons.append("thin but possible credit spread reward/risk")

    elif "Debit" in strategy:
        if reward_risk >= 1.00:
            priority_score += 3
            reasons.append("strong debit spread reward/risk")
        elif reward_risk >= 0.75:
            priority_score += 2
            reasons.append("reasonable debit spread reward/risk")
        elif reward_risk >= 0.50:
            priority_score += 1
            reasons.append("thin but possible debit spread reward/risk")

    if "Good liquidity" in quality:
        priority_score += 2
        reasons.append("good liquidity")
    elif "Tradable check" in quality:
        priority_score += 1
        reasons.append("tradable liquidity check")
    elif "Estimated quote" in quality:
        priority_score -= 1
        reasons.append("estimated quote")
    elif "Wide spread" in quality:
        priority_score -= 2
        reasons.append("wide spread")

    if priority_score >= 8:
        priority = "VERIFY FIRST"
    elif priority_score >= 5:
        priority = "VERIFY"
    elif priority_score >= 3:
        priority = "LOW PRIORITY"
    else:
        priority = "SKIP FIRST PASS"

    reason_text = ", ".join(reasons) if reasons else "No strong reason to prioritize."

    return pd.Series(
        {
            "ManualVerifyPriority": priority,
            "ManualVerifyScore": priority_score,
            "ManualVerifyReason": reason_text,
        }
    )


def add_manual_priority_columns(df: pd.DataFrame):
    if df.empty:
        return df

    priority = df.apply(manual_verify_priority, axis=1)

    base = df.copy()

    for col in ["ManualVerifyPriority", "ManualVerifyScore", "ManualVerifyReason"]:
        if col in base.columns:
            base = base.drop(columns=[col])

    return pd.concat([base.reset_index(drop=True), priority.reset_index(drop=True)], axis=1)


def score_label(score):
    score = safe_float(score, 0)

    if score >= 70:
        return "Strong"
    if score >= 55:
        return "Good"
    if score >= 40:
        return "Usable"
    if score >= 30:
        return "Weak / verify only"

    return "Low quality"


# ============================================================
# Header
# ============================================================

st.title("Options Scanner Board")
st.caption(
    "Shared scanner-only dashboard. The scanner finds ideas; Robinhood is used for manual quote verification."
)

st.warning(
    "This board does not place trades and does not make final trade decisions. "
    "Always verify credit/debit, max profit, max loss, breakeven, liquidity, and risk inside Robinhood."
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Scanner Controls")

if st.sidebar.button("Run Full Scan", use_container_width=True):
    st.cache_data.clear()

    scripts = [
        "fetch_stock_data.py",
        "stock_setup_scanner.py",
        "options_contract_scanner.py",
    ]

    full_log = ""

    with st.spinner("Running full scan..."):
        for script in scripts:
            ok, output = run_script(script)
            full_log += f"\n\n===== {script} =====\n{output}"

            if not ok:
                st.sidebar.error(f"{script} failed.")
                break
        else:
            st.sidebar.success("Full scan complete.")

    st.session_state["scan_log"] = full_log

if st.sidebar.button("Refresh Board", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if "scan_log" in st.session_state:
    with st.sidebar.expander("Scan Log"):
        st.text(st.session_state["scan_log"])


# ============================================================
# Load data
# ============================================================

df = load_options_board()

if df.empty:
    st.info(
        "No options board found yet. Click **Run Full Scan** in the sidebar, "
        "or run `python src\\options_contract_scanner.py` in your terminal."
    )
    st.stop()

df = add_manual_priority_columns(df)


# ============================================================
# Filters
# ============================================================

st.sidebar.header("Filters")

tickers = sorted(df["Ticker"].dropna().unique().tolist()) if "Ticker" in df.columns else []
strategies = sorted(df["Strategy"].dropna().unique().tolist()) if "Strategy" in df.columns else []
biases = sorted(df["Bias"].dropna().unique().tolist()) if "Bias" in df.columns else []
priorities = sorted(df["ManualVerifyPriority"].dropna().unique().tolist()) if "ManualVerifyPriority" in df.columns else []
qualities = sorted(df["QualityFlag"].dropna().unique().tolist()) if "QualityFlag" in df.columns else []
grades = sorted(df["ActionGrade"].dropna().unique().tolist()) if "ActionGrade" in df.columns else []

selected_tickers = st.sidebar.multiselect("Tickers", tickers)
selected_strategies = st.sidebar.multiselect("Strategies", strategies)
selected_biases = st.sidebar.multiselect("Bias", biases)
selected_priorities = st.sidebar.multiselect(
    "Manual Verify Priority",
    priorities,
    default=[p for p in ["VERIFY FIRST", "VERIFY"] if p in priorities],
)
selected_qualities = st.sidebar.multiselect("Quote Quality", qualities)
selected_grades = st.sidebar.multiselect("Action Grade", grades)

min_score = st.sidebar.slider(
    "Minimum Scanner Score",
    min_value=0.0,
    max_value=100.0,
    value=0.0,
    step=1.0,
)

max_dte_default = 7
if "DTE" in df.columns and not df["DTE"].dropna().empty:
    max_dte_default = int(min(max(df["DTE"].dropna()), 14))

max_dte = st.sidebar.slider(
    "Max DTE",
    min_value=1,
    max_value=60,
    value=max_dte_default,
    step=1,
)

min_reward_risk = st.sidebar.slider(
    "Minimum Reward/Risk",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.05,
)

filtered = df.copy()

if selected_tickers:
    filtered = filtered[filtered["Ticker"].isin(selected_tickers)]

if selected_strategies:
    filtered = filtered[filtered["Strategy"].isin(selected_strategies)]

if selected_biases:
    filtered = filtered[filtered["Bias"].isin(selected_biases)]

if selected_priorities:
    filtered = filtered[filtered["ManualVerifyPriority"].isin(selected_priorities)]

if selected_qualities:
    filtered = filtered[filtered["QualityFlag"].isin(selected_qualities)]

if selected_grades:
    filtered = filtered[filtered["ActionGrade"].isin(selected_grades)]

if "FinalScore" in filtered.columns:
    filtered = filtered[filtered["FinalScore"] >= min_score]

if "DTE" in filtered.columns:
    filtered = filtered[filtered["DTE"] <= max_dte]

if "RewardRisk" in filtered.columns:
    filtered = filtered[filtered["RewardRisk"] >= min_reward_risk]

if not filtered.empty:
    filtered = filtered.sort_values(
        by=[
            "ManualVerifyScore",
            "DTE",
            "FinalScore",
            "StockSetupScore",
            "RewardRisk",
        ],
        ascending=[False, True, False, False, False],
    )


# ============================================================
# Metrics
# ============================================================

m1, m2, m3, m4, m5, m6 = st.columns(6)

with m1:
    st.metric("Trades", len(filtered))

with m2:
    st.metric("Tickers", filtered["Ticker"].nunique() if not filtered.empty else 0)

with m3:
    verify_first_count = int((filtered["ManualVerifyPriority"] == "VERIFY FIRST").sum()) if not filtered.empty else 0
    st.metric("Verify First", verify_first_count)

with m4:
    verify_count = int((filtered["ManualVerifyPriority"] == "VERIFY").sum()) if not filtered.empty else 0
    st.metric("Verify", verify_count)

with m5:
    short_dte_count = int((filtered["DTE"] <= 7).sum()) if "DTE" in filtered.columns and not filtered.empty else 0
    st.metric("≤ 7 DTE", short_dte_count)

with m6:
    top_score = filtered["FinalScore"].max() if "FinalScore" in filtered.columns and not filtered.empty else 0
    st.metric("Top Score", f"{top_score:.1f}")


# ============================================================
# Scanner Board
# ============================================================

st.subheader("Scanner Board")

display_cols = [
    "Ticker",
    "StockPrice",
    "ManualVerifyPriority",
    "ManualVerifyScore",
    "ManualVerifyReason",
    "Bias",
    "Strategy",
    "Expiration",
    "DTE",
    "BuyLeg",
    "SellLeg",
    "DebitOrCredit",
    "NetDebitCredit",
    "MinimumRobinhoodPrice",
    "SpreadWidth",
    "MaxProfit",
    "MaxLoss",
    "Breakeven",
    "RewardRisk",
    "QualityFlag",
    "FinalScore",
    "ActionGrade",
    "RobinhoodChainUrl",
]

display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "RobinhoodChainUrl": st.column_config.LinkColumn(
            "Robinhood",
            display_text="Open Chain",
        ),
        "StockPrice": st.column_config.NumberColumn("Stock Price", format="$%.2f"),
        "ManualVerifyScore": st.column_config.NumberColumn("Manual Score", format="%.0f"),
        "NetDebitCredit": st.column_config.NumberColumn("Scanner Credit/Debit", format="$%.2f"),
        "MinimumRobinhoodPrice": st.column_config.NumberColumn("Min RH Price", format="$%.2f"),
        "SpreadWidth": st.column_config.NumberColumn("Width", format="$%.2f"),
        "MaxProfit": st.column_config.NumberColumn("Scanner Max Profit", format="$%.2f"),
        "MaxLoss": st.column_config.NumberColumn("Scanner Max Loss", format="$%.2f"),
        "Breakeven": st.column_config.NumberColumn("Scanner Breakeven", format="$%.2f"),
        "RewardRisk": st.column_config.NumberColumn("Reward/Risk", format="%.2f"),
        "FinalScore": st.column_config.NumberColumn("Score", format="%.2f"),
    },
)


# ============================================================
# Selected Trade Details
# ============================================================

st.subheader("Selected Trade Details")

if filtered.empty:
    st.info("No trades match the current filters.")
else:
    detail_df = filtered.copy()

    detail_df["TradeLabel"] = (
        detail_df["Ticker"].astype(str)
        + " | "
        + detail_df["ManualVerifyPriority"].astype(str)
        + " | "
        + detail_df["Strategy"].astype(str)
        + " | "
        + detail_df["Expiration"].astype(str)
        + " | "
        + detail_df["BuyLeg"].astype(str)
        + " / "
        + detail_df["SellLeg"].astype(str)
    )

    selected_label = st.selectbox(
        "Select a trade to discuss/check in Robinhood",
        detail_df["TradeLabel"].tolist(),
    )

    selected = detail_df[detail_df["TradeLabel"] == selected_label].iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### Setup")
        st.write(f"**Ticker:** {selected.get('Ticker', '')}")
        st.write(f"**Stock Price:** ${safe_float(selected.get('StockPrice', 0), 0):.2f}")
        st.write(f"**Bias:** {selected.get('Bias', '')}")
        st.write(f"**Strategy:** {selected.get('Strategy', '')}")
        st.write(f"**Expiration:** {selected.get('Expiration', '')}")
        st.write(f"**DTE:** {selected.get('DTE', '')}")
        st.write(f"**Buy Leg:** {selected.get('BuyLeg', '')}")
        st.write(f"**Sell Leg:** {selected.get('SellLeg', '')}")

    with c2:
        st.markdown("### Scanner Estimate")
        st.write(f"**Type:** {selected.get('DebitOrCredit', '')}")
        st.write(f"**Scanner Credit/Debit:** ${safe_float(selected.get('NetDebitCredit', 0), 0):.2f}")
        st.write(f"**Minimum Robinhood Price:** ${safe_float(selected.get('MinimumRobinhoodPrice', 0), 0):.2f}")
        st.write(f"**Width:** ${safe_float(selected.get('SpreadWidth', 0), 0):.2f}")
        st.write(f"**Max Profit:** ${safe_float(selected.get('MaxProfit', 0), 0):.2f}")
        st.write(f"**Max Loss:** ${safe_float(selected.get('MaxLoss', 0), 0):.2f}")
        st.write(f"**Breakeven:** ${safe_float(selected.get('Breakeven', 0), 0):.2f}")
        st.write(f"**Reward/Risk:** {safe_float(selected.get('RewardRisk', 0), 0):.2f}")

    with c3:
        st.markdown("### Priority / Score")
        final_score = safe_float(selected.get("FinalScore", 0), 0)
        st.write(f"**Manual Priority:** {selected.get('ManualVerifyPriority', '')}")
        st.write(f"**Manual Score:** {selected.get('ManualVerifyScore', '')}")
        st.write(f"**Scanner Score:** {final_score:.2f} — {score_label(final_score)}")
        st.write(f"**Quote Quality:** {selected.get('QualityFlag', '')}")
        st.write(f"**Action Grade:** {selected.get('ActionGrade', '')}")
        st.write(f"**Reason:** {selected.get('ManualVerifyReason', '')}")

        if "RobinhoodChainUrl" in selected and str(selected["RobinhoodChainUrl"]).startswith("http"):
            st.link_button(
                "Open Robinhood Chain",
                selected["RobinhoodChainUrl"],
                use_container_width=True,
            )

    st.markdown("### How to verify in Robinhood")
    st.write(
        "Open the chain, choose the same expiration, use Builder, add the exact buy and sell legs, "
        "then compare Robinhood's credit/debit, max profit, max loss, and breakeven to the scanner estimate."
    )

    st.markdown("### Score guide")
    st.write(
        "**70+** = strong scanner setup. "
        "**55–70** = good candidate. "
        "**40–55** = usable but verify carefully. "
        "**30–40** = weak / idea only. "
        "**Below 30** = usually skip."
    )


# ============================================================
# Diagnostics
# ============================================================

with st.expander("Diagnostics"):
    diagnostics = load_diagnostics()

    if diagnostics.empty:
        st.info("No diagnostics file found.")
    else:
        st.dataframe(
            diagnostics,
            use_container_width=True,
            hide_index=True,
        )