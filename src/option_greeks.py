import math
from dataclasses import dataclass


# ============================================================
# Black-Scholes Greek Estimator
# ============================================================
# This file estimates option Greeks when the data source does
# not provide complete Greek values.
#
# Inputs:
#   S = stock price
#   K = strike price
#   T = time to expiration in years
#   r = risk-free rate
#   sigma = implied volatility
#   option_type = "call" or "put"
#
# Outputs:
#   delta
#   gamma
#   theta
#   vega
#   rho
#
# Notes:
#   - These are theoretical estimates.
#   - Broker/platform Greeks may differ slightly.
#   - This is good enough for scanner ranking and risk grading.
# ============================================================


@dataclass
class GreekResult:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    theoretical_price: float


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_float(value, default=0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, float) and math.isnan(value):
            return default

        return float(value)

    except Exception:
        return default


def estimate_greeks(
    stock_price,
    strike,
    dte,
    iv,
    option_type,
    risk_free_rate=0.045,
) -> GreekResult:
    """
    Estimate Black-Scholes Greeks for a call or put.

    Parameters
    ----------
    stock_price : float
        Current underlying stock price.

    strike : float
        Option strike price.

    dte : int or float
        Days to expiration.

    iv : float
        Implied volatility as decimal.
        Example: 0.45 means 45%.

    option_type : str
        "call" or "put".

    risk_free_rate : float
        Annualized risk-free rate as decimal.
        Example: 0.045 means 4.5%.

    Returns
    -------
    GreekResult
    """

    S = _safe_float(stock_price, 0.0)
    K = _safe_float(strike, 0.0)
    DTE = _safe_float(dte, 0.0)
    sigma = _safe_float(iv, 0.0)
    r = _safe_float(risk_free_rate, 0.045)

    option_type = str(option_type).lower().strip()

    # Guardrails for bad/missing data.
    if S <= 0 or K <= 0 or DTE <= 0:
        return GreekResult(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            theoretical_price=0.0,
        )

    # If IV is missing or unusable, use a conservative fallback.
    if sigma <= 0 or sigma > 5:
        sigma = 0.40

    # Convert DTE to years.
    T = max(DTE / 365.0, 1.0 / 365.0)

    try:
        d1 = (
            math.log(S / K)
            + (r + 0.5 * sigma * sigma) * T
        ) / (sigma * math.sqrt(T))

        d2 = d1 - sigma * math.sqrt(T)

        pdf_d1 = _norm_pdf(d1)

        if option_type == "call":
            delta = _norm_cdf(d1)

            theoretical_price = (
                S * _norm_cdf(d1)
                - K * math.exp(-r * T) * _norm_cdf(d2)
            )

            theta_annual = (
                -S * pdf_d1 * sigma / (2.0 * math.sqrt(T))
                - r * K * math.exp(-r * T) * _norm_cdf(d2)
            )

            rho = K * T * math.exp(-r * T) * _norm_cdf(d2)

        elif option_type == "put":
            delta = _norm_cdf(d1) - 1.0

            theoretical_price = (
                K * math.exp(-r * T) * _norm_cdf(-d2)
                - S * _norm_cdf(-d1)
            )

            theta_annual = (
                -S * pdf_d1 * sigma / (2.0 * math.sqrt(T))
                + r * K * math.exp(-r * T) * _norm_cdf(-d2)
            )

            rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)

        else:
            return GreekResult(
                delta=0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
                theoretical_price=0.0,
            )

        gamma = pdf_d1 / (S * sigma * math.sqrt(T))

        # Vega is usually interpreted as price change per 1 percentage point move in IV.
        vega = S * pdf_d1 * math.sqrt(T) / 100.0

        # Convert annual theta to daily theta.
        theta = theta_annual / 365.0

        return GreekResult(
            delta=round(delta, 6),
            gamma=round(gamma, 6),
            theta=round(theta, 6),
            vega=round(vega, 6),
            rho=round(rho / 100.0, 6),
            theoretical_price=round(max(theoretical_price, 0.0), 6),
        )

    except Exception:
        return GreekResult(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
            theoretical_price=0.0,
        )


def estimate_leg_greeks(
    stock_price,
    strike,
    dte,
    iv,
    option_type,
    side,
    risk_free_rate=0.045,
):
    """
    Estimate Greeks for a single option leg and adjust sign by side.

    side:
        "buy"  = long option exposure
        "sell" = short option exposure

    Long call:
        positive delta, positive gamma, negative theta, positive vega

    Short call:
        negative delta, negative gamma, positive theta, negative vega

    Long put:
        negative delta, positive gamma, negative theta, positive vega

    Short put:
        positive delta, negative gamma, positive theta, negative vega
    """

    greeks = estimate_greeks(
        stock_price=stock_price,
        strike=strike,
        dte=dte,
        iv=iv,
        option_type=option_type,
        risk_free_rate=risk_free_rate,
    )

    side = str(side).lower().strip()

    multiplier = 1.0

    if side == "sell":
        multiplier = -1.0

    return {
        "Delta": round(greeks.delta * multiplier, 6),
        "Gamma": round(greeks.gamma * multiplier, 6),
        "Theta": round(greeks.theta * multiplier, 6),
        "Vega": round(greeks.vega * multiplier, 6),
        "Rho": round(greeks.rho * multiplier, 6),
        "TheoreticalPrice": greeks.theoretical_price,
    }


def combine_leg_greeks(buy_leg=None, sell_leg=None):
    """
    Combine buy and sell leg Greeks into net strategy Greeks.
    """

    buy_leg = buy_leg or {}
    sell_leg = sell_leg or {}

    net_delta = _safe_float(buy_leg.get("Delta", 0.0)) + _safe_float(sell_leg.get("Delta", 0.0))
    net_gamma = _safe_float(buy_leg.get("Gamma", 0.0)) + _safe_float(sell_leg.get("Gamma", 0.0))
    net_theta = _safe_float(buy_leg.get("Theta", 0.0)) + _safe_float(sell_leg.get("Theta", 0.0))
    net_vega = _safe_float(buy_leg.get("Vega", 0.0)) + _safe_float(sell_leg.get("Vega", 0.0))
    net_rho = _safe_float(buy_leg.get("Rho", 0.0)) + _safe_float(sell_leg.get("Rho", 0.0))

    return {
        "NetDelta": round(net_delta, 6),
        "NetGamma": round(net_gamma, 6),
        "NetTheta": round(net_theta, 6),
        "NetVega": round(net_vega, 6),
        "NetRho": round(net_rho, 6),
    }


def calculate_breakeven_move_pct(stock_price, breakeven):
    """
    How far the stock has to move from current price to reach breakeven.
    """

    S = _safe_float(stock_price, 0.0)
    B = _safe_float(breakeven, 0.0)

    if S <= 0 or B <= 0:
        return 0.0

    return round(abs(B - S) / S * 100.0, 4)


def estimate_expected_move_pct(iv, dte):
    """
    Rough expected move percentage using IV and time.

    Expected move approximation:
        IV * sqrt(DTE / 365)

    Example:
        IV 40%, 7 DTE:
        0.40 * sqrt(7/365) = about 5.54%
    """

    sigma = _safe_float(iv, 0.0)
    DTE = _safe_float(dte, 0.0)

    if sigma <= 0:
        sigma = 0.40

    if DTE <= 0:
        return 0.0

    move = sigma * math.sqrt(DTE / 365.0) * 100.0

    return round(move, 4)


def grade_greek_risk(
    strategy,
    stock_price,
    breakeven,
    net_debit_credit,
    dte,
    buy_delta=0.0,
    sell_delta=0.0,
    net_delta=0.0,
    net_gamma=0.0,
    net_theta=0.0,
    net_vega=0.0,
    avg_iv=0.40,
):
    """
    Convert Greeks into a simple scanner grade.

    Grade:
        A = strong Greek profile
        B = tradable
        C = usable but watch closely
        D = weak
        F = avoid unless there is a very specific reason
    """

    strategy = str(strategy).strip()

    S = _safe_float(stock_price, 0.0)
    B = _safe_float(breakeven, 0.0)
    premium = abs(_safe_float(net_debit_credit, 0.0))
    DTE = _safe_float(dte, 0.0)

    buy_delta_abs = abs(_safe_float(buy_delta, 0.0))
    sell_delta_abs = abs(_safe_float(sell_delta, 0.0))
    net_delta_abs = abs(_safe_float(net_delta, 0.0))
    net_gamma_abs = abs(_safe_float(net_gamma, 0.0))
    net_theta_value = _safe_float(net_theta, 0.0)
    net_vega_abs = abs(_safe_float(net_vega, 0.0))
    avg_iv_value = _safe_float(avg_iv, 0.40)

    breakeven_move_pct = calculate_breakeven_move_pct(S, B)
    expected_move_pct = estimate_expected_move_pct(avg_iv_value, DTE)

    if expected_move_pct > 0:
        breakeven_vs_expected = breakeven_move_pct / expected_move_pct
    else:
        breakeven_vs_expected = 0.0

    # Theta as percent of premium per day.
    if premium > 0:
        theta_pct_of_premium = abs(net_theta_value) / premium * 100.0
    else:
        theta_pct_of_premium = 0.0

    score = 100
    reasons = []

    # --------------------------------------------------------
    # Long Calls / Long Puts
    # --------------------------------------------------------
    if strategy in ["Long Call", "Long Put"]:
        if buy_delta_abs < 0.20:
            score -= 30
            reasons.append("delta too low / far OTM")
        elif buy_delta_abs < 0.30:
            score -= 15
            reasons.append("delta is light")
        elif 0.30 <= buy_delta_abs <= 0.70:
            reasons.append("useful delta exposure")
        elif buy_delta_abs > 0.85:
            score -= 10
            reasons.append("very high delta / expensive contract")

        if theta_pct_of_premium > 20:
            score -= 30
            reasons.append("very high daily theta burn")
        elif theta_pct_of_premium > 12:
            score -= 20
            reasons.append("high daily theta burn")
        elif theta_pct_of_premium > 8:
            score -= 10
            reasons.append("moderate theta burn")
        else:
            reasons.append("manageable theta burn")

        if breakeven_vs_expected > 1.50:
            score -= 30
            reasons.append("breakeven is far beyond expected move")
        elif breakeven_vs_expected > 1.10:
            score -= 15
            reasons.append("breakeven is stretched")
        else:
            reasons.append("breakeven is reasonable vs expected move")

        if avg_iv_value > 1.00:
            score -= 20
            reasons.append("very high IV / expensive premium")
        elif avg_iv_value > 0.70:
            score -= 10
            reasons.append("elevated IV")

        if DTE <= 1:
            score -= 20
            reasons.append("0-1 DTE is very high decay/gamma risk")
        elif DTE <= 3:
            score -= 5
            reasons.append("very short DTE")

        if net_gamma_abs > 0.15:
            score -= 10
            reasons.append("high gamma sensitivity")

    # --------------------------------------------------------
    # Credit Spreads
    # --------------------------------------------------------
    elif "Credit Spread" in strategy:
        if sell_delta_abs > 0.45:
            score -= 30
            reasons.append("short-leg delta is aggressive")
        elif sell_delta_abs > 0.35:
            score -= 15
            reasons.append("short-leg delta is somewhat aggressive")
        elif 0.15 <= sell_delta_abs <= 0.35:
            reasons.append("short-leg delta is reasonable")
        elif sell_delta_abs < 0.10:
            score -= 10
            reasons.append("short-leg delta is low / credit may be thin")

        if breakeven_move_pct < 1.0:
            score -= 20
            reasons.append("breakeven too close to stock price")
        elif breakeven_move_pct < 2.0:
            score -= 10
            reasons.append("breakeven is close")
        else:
            reasons.append("breakeven has some room")

        if DTE <= 1:
            score -= 20
            reasons.append("0-1 DTE has high gamma risk")
        elif DTE <= 3:
            score -= 10
            reasons.append("very short DTE gamma risk")

        if net_theta_value > 0:
            reasons.append("theta works in favor")
        else:
            score -= 10
            reasons.append("theta does not clearly help")

        if net_gamma_abs > 0.10:
            score -= 15
            reasons.append("high net gamma risk")

    # --------------------------------------------------------
    # Debit Spreads
    # --------------------------------------------------------
    elif "Debit Spread" in strategy:
        if net_delta_abs < 0.10:
            score -= 25
            reasons.append("weak net delta exposure")
        elif net_delta_abs < 0.20:
            score -= 10
            reasons.append("light net delta exposure")
        else:
            reasons.append("usable net delta exposure")

        if breakeven_vs_expected > 1.50:
            score -= 25
            reasons.append("breakeven too far vs expected move")
        elif breakeven_vs_expected > 1.10:
            score -= 10
            reasons.append("breakeven somewhat stretched")
        else:
            reasons.append("breakeven is reasonable")

        if DTE <= 1:
            score -= 15
            reasons.append("0-1 DTE high gamma risk")

        if net_gamma_abs > 0.12:
            score -= 10
            reasons.append("high gamma sensitivity")

    # --------------------------------------------------------
    # Unknown Strategy
    # --------------------------------------------------------
    else:
        score -= 20
        reasons.append("unknown strategy Greek rules")

    score = max(0, min(100, score))

    if score >= 85:
        grade = "A"
        risk = "LOW"
    elif score >= 70:
        grade = "B"
        risk = "MODERATE"
    elif score >= 55:
        grade = "C"
        risk = "ELEVATED"
    elif score >= 40:
        grade = "D"
        risk = "HIGH"
    else:
        grade = "F"
        risk = "VERY HIGH"

    return {
        "GreekRiskScore": round(score, 2),
        "GreekRiskGrade": grade,
        "GreekRiskLevel": risk,
        "GreekRiskReason": "; ".join(reasons),
        "ThetaPctOfPremium": round(theta_pct_of_premium, 4),
        "BreakevenMovePct": round(breakeven_move_pct, 4),
        "ExpectedMovePct": round(expected_move_pct, 4),
        "BreakevenVsExpectedMove": round(breakeven_vs_expected, 4),
    }


if __name__ == "__main__":
    # Quick test example.
    result = estimate_greeks(
        stock_price=100,
        strike=102,
        dte=7,
        iv=0.40,
        option_type="call",
        risk_free_rate=0.045,
    )

    print("Example Call Greeks:")
    print(result)

    buy_leg = estimate_leg_greeks(
        stock_price=100,
        strike=102,
        dte=7,
        iv=0.40,
        option_type="call",
        side="buy",
    )

    print("\nExample Buy Leg:")
    print(buy_leg)

    net = combine_leg_greeks(buy_leg=buy_leg)

    print("\nExample Net Greeks:")
    print(net)

    grade = grade_greek_risk(
        strategy="Long Call",
        stock_price=100,
        breakeven=103.25,
        net_debit_credit=1.25,
        dte=7,
        buy_delta=buy_leg["Delta"],
        sell_delta=0.0,
        net_delta=net["NetDelta"],
        net_gamma=net["NetGamma"],
        net_theta=net["NetTheta"],
        net_vega=net["NetVega"],
        avg_iv=0.40,
    )

    print("\nExample Greek Risk Grade:")
    print(grade)