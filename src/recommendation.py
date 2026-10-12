"""
Current BUY / HOLD / SELL recommendation for a stock.

Combines two evidence sources:
  1. Model view: probabilities of SELL / HOLD / BUY from a trained model,
     i.e. how likely the +profit / -stop / neither barrier is hit within the
     time limit (NOT a price forecast).
  2. Technical view: rule-based readings of RSI, MACD, SMA trend, EMA cross.

Combined score = W_MODEL * (P(BUY) - P(SELL)) + W_TECH * technical_score
Score >= +THRESHOLD -> BUY, <= -THRESHOLD -> SELL, otherwise HOLD.

The weights and threshold are transparent heuristics, not fitted values.
Nothing here is financial advice or a guaranteed prediction.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from src.sequence_generation import decode_labels_from_keras  # noqa: F401 (kept for API symmetry)
from src.utils import scale_tabular, scale_sequences

W_MODEL = 0.6
W_TECH = 0.4
THRESHOLD = 0.15
CLASS_ORDER = ["SELL", "HOLD", "BUY"]  # matches encoded classes 0, 1, 2


def technical_view(g: pd.DataFrame) -> Dict[str, object]:
    """
    g: single-ticker feature DataFrame sorted by date (needs >= 2 rows).
    Returns a dict with per-indicator readings, a score in [-1, 1], and a
    volatility regime note.
    """
    g = g.sort_values("date")
    row = g.iloc[-1]
    prev = g.iloc[-2] if len(g) > 1 else row
    price = row["adj_close"] if "adj_close" in g.columns else row["close"]

    readings: List[Dict[str, object]] = []

    # RSI
    rsi = row["rsi_14"]
    if rsi < 30:
        s, txt = 1.0, f"RSI {rsi:.1f} is below 30 (oversold, possible rebound)"
    elif rsi > 70:
        s, txt = -1.0, f"RSI {rsi:.1f} is above 70 (overbought, possible pullback)"
    else:
        s, txt = 0.0, f"RSI {rsi:.1f} is in the neutral 30-70 range"
    readings.append({"indicator": "RSI (14)", "reading": txt, "score": s})

    # MACD histogram + crossover
    hist, prev_hist = row["macd_hist"], prev["macd_hist"]
    if hist > 0 and prev_hist <= 0:
        s, txt = 1.0, "MACD just crossed above its signal line (bullish crossover)"
    elif hist < 0 and prev_hist >= 0:
        s, txt = -1.0, "MACD just crossed below its signal line (bearish crossover)"
    elif hist > 0:
        s, txt = 0.5, "MACD is above its signal line (bullish momentum)"
    else:
        s, txt = -0.5, "MACD is below its signal line (bearish momentum)"
    readings.append({"indicator": "MACD", "reading": txt, "score": s})

    # SMA trend
    sma20, sma50 = row["sma_20"], row["sma_50"]
    if price > sma20 > sma50:
        s, txt = 1.0, "Price is above SMA20 and SMA20 is above SMA50 (uptrend)"
    elif price < sma20 < sma50:
        s, txt = -1.0, "Price is below SMA20 and SMA20 is below SMA50 (downtrend)"
    else:
        s, txt = 0.0, "Price and moving averages are mixed (no clear trend)"
    readings.append({"indicator": "SMA trend", "reading": txt, "score": s})

    # EMA cross
    if row["ema_12"] > row["ema_26"]:
        s, txt = 0.5, "EMA12 is above EMA26 (short-term strength)"
    else:
        s, txt = -0.5, "EMA12 is below EMA26 (short-term weakness)"
    readings.append({"indicator": "EMA 12/26", "reading": txt, "score": s})

    score = float(np.mean([r["score"] for r in readings]))

    # Volatility regime (informational, not directional)
    vol = row["volatility_20"]
    hist_vol = g["volatility_20"].dropna().tail(252)
    if len(hist_vol) >= 20:
        med = hist_vol.median()
        if vol > 1.3 * med:
            regime = f"High volatility ({vol*100:.2f}% daily vs {med*100:.2f}% typical): larger swings, higher risk"
        elif vol < 0.7 * med:
            regime = f"Low volatility ({vol*100:.2f}% daily vs {med*100:.2f}% typical): calmer market"
        else:
            regime = f"Normal volatility ({vol*100:.2f}% daily)"
    else:
        regime = f"Daily volatility {vol*100:.2f}%"

    return {"readings": readings, "score": score, "volatility_note": regime,
            "price": float(price), "date": row["date"], "daily_vol": float(vol)}


def model_probabilities(model_info: dict, g: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Return {'SELL': p, 'HOLD': p, 'BUY': p} for the latest row of g, or None
    if there is not enough history."""
    g = g.sort_values("date")
    cols = config.FEATURE_COLUMNS

    if model_info["type"] == "tabular":
        X = g[cols].values[-1:].astype(float)
        X_s = scale_tabular(X, model_info["scaler"])
        proba = model_info["model"].predict_proba(X_s)[0]
    else:
        if len(g) < config.SEQUENCE_LENGTH:
            return None
        X = g[cols].values[-config.SEQUENCE_LENGTH:].astype(float)[None, ...]
        X_s = scale_sequences(X, model_info["scaler"])
        proba = model_info["model"].predict(X_s, verbose=0)[0]

    return {c: float(p) for c, p in zip(CLASS_ORDER, proba)}


def combine(model_probs: Optional[Dict[str, float]], tech_score: float) -> Dict[str, object]:
    """Blend model and technical evidence into one recommendation."""
    if model_probs is None:
        combined = tech_score
        model_score = None
        basis = "Technical indicators only (no model available)"
    else:
        model_score = model_probs["BUY"] - model_probs["SELL"]
        combined = W_MODEL * model_score + W_TECH * tech_score
        basis = f"{int(W_MODEL*100)}% model + {int(W_TECH*100)}% technical indicators"

    if combined >= THRESHOLD:
        rec = "BUY"
    elif combined <= -THRESHOLD:
        rec = "SELL"
    else:
        rec = "HOLD"

    # Agreement flag: do model and indicators point the same way?
    agreement = None
    if model_score is not None:
        if abs(model_score) < 0.05 or abs(tech_score) < 0.15:
            agreement = "weak / neutral evidence on one side"
        elif np.sign(model_score) == np.sign(tech_score):
            agreement = "model and indicators agree"
        else:
            agreement = "model and indicators DISAGREE (treat with extra caution)"

    return {"recommendation": rec, "combined_score": float(combined),
            "model_score": model_score, "tech_score": float(tech_score),
            "basis": basis, "agreement": agreement}


def volatility_range(price: float, daily_vol: float, horizon_days: int = None, n_sigma: float = 1.0):
    """Statistical price range after `horizon_days` assuming random-walk
    log-returns with the current daily volatility. A range, not a forecast."""
    horizon_days = horizon_days or config.TIME_LIMIT_DAYS
    sigma_h = daily_vol * np.sqrt(horizon_days) * n_sigma
    return float(price * np.exp(-sigma_h)), float(price * np.exp(sigma_h))


def recommend_for_ticker(g: pd.DataFrame, model_info: Optional[dict]) -> Dict[str, object]:
    tech = technical_view(g)
    probs = model_probabilities(model_info, g) if model_info else None
    result = combine(probs, tech["score"])
    low, high = volatility_range(tech["price"], tech["daily_vol"])
    result.update({
        "probs": probs, "technical": tech,
        "range_low": low, "range_high": high,
        "horizon_days": config.TIME_LIMIT_DAYS,
    })
    return result
