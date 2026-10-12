"""
Convert model predictions into a clean BUY / SELL / HOLD signal table.
"""
import numpy as np
import pandas as pd

from config import LABEL_MAP

# Class order after encoding (see sequence_generation.encode_labels_for_keras
# and models/xgboost_model.train_xgboost's label_map): index 0 = SELL,
# index 1 = HOLD, index 2 = BUY. Both XGBoost's predict_proba and the Keras
# models' softmax output follow this same order, so one decoder covers both.
_PROB_COL_TO_LABEL = {0: -1, 1: 0, 2: 1}


def build_signal_table(meta: pd.DataFrame, raw_df: pd.DataFrame, predictions, model_name: str, model_version: str = "v1") -> pd.DataFrame:
    """
    meta: DataFrame with columns [ticker, date] aligned 1:1 with `predictions`
    raw_df: full feature dataframe to pull price/indicator context from
    predictions: array-like of {-1, 0, 1}
    """
    sig = meta.copy()
    sig["prediction"] = predictions
    sig["signal"] = sig["prediction"].map(LABEL_MAP)
    sig["model"] = model_name
    sig["model_version"] = model_version

    lookup_cols = [c for c in [
        "ticker", "date", "close", "adj_close", "rsi_14", "sma_20", "ema_12", "macd", "volatility_20"
    ] if c in raw_df.columns]
    merged = sig.merge(raw_df[lookup_cols], on=["ticker", "date"], how="left")

    price_col = "adj_close" if "adj_close" in merged.columns else "close"
    merged = merged.rename(columns={price_col: "price"})
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def signals_for_ticker(signal_table: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return signal_table[signal_table["ticker"] == ticker].sort_values("date").reset_index(drop=True)


def build_signal_table_with_confidence(
    meta: pd.DataFrame, raw_df: pd.DataFrame, probs: np.ndarray, model_name: str, model_version: str = "v1"
) -> pd.DataFrame:
    """
    Same as build_signal_table, but also carries the model's full probability
    distribution over {SELL, HOLD, BUY} so the recommendation can be shown
    with a confidence level instead of a bare label.

    probs: array of shape (n, 3), columns ordered [SELL_prob, HOLD_prob, BUY_prob]
    """
    predictions = np.argmax(probs, axis=1)
    predictions = np.array([_PROB_COL_TO_LABEL[p] for p in predictions])

    sig = meta.copy()
    sig["prediction"] = predictions
    sig["signal"] = sig["prediction"].map(LABEL_MAP)
    sig["confidence"] = probs.max(axis=1)
    sig["sell_prob"] = probs[:, 0]
    sig["hold_prob"] = probs[:, 1]
    sig["buy_prob"] = probs[:, 2]
    sig["model"] = model_name
    sig["model_version"] = model_version

    lookup_cols = [c for c in [
        "ticker", "date", "close", "adj_close", "rsi_14", "sma_20", "ema_12", "macd", "macd_signal", "volatility_20"
    ] if c in raw_df.columns]
    merged = sig.merge(raw_df[lookup_cols], on=["ticker", "date"], how="left")

    price_col = "adj_close" if "adj_close" in merged.columns else "close"
    merged = merged.rename(columns={price_col: "price"})
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def explain_signal(row: pd.Series) -> list:
    """
    Turn the indicator values behind one row into a short list of plain-
    language observations, so a user can see *why* the model leans the way
    it does before deciding whether to act on it.
    """
    notes = []

    rsi = row.get("rsi_14")
    if rsi is not None and not pd.isna(rsi):
        if rsi >= 70:
            notes.append(f"RSI is {rsi:.1f} — in overbought territory (≥70).")
        elif rsi <= 30:
            notes.append(f"RSI is {rsi:.1f} — in oversold territory (≤30).")
        else:
            notes.append(f"RSI is {rsi:.1f} — neutral range.")

    macd, macd_sig = row.get("macd"), row.get("macd_signal")
    if macd is not None and macd_sig is not None and not pd.isna(macd) and not pd.isna(macd_sig):
        if macd > macd_sig:
            notes.append("MACD is above its signal line — bullish momentum.")
        else:
            notes.append("MACD is below its signal line — bearish momentum.")

    price, sma20 = row.get("price"), row.get("sma_20")
    if price is not None and sma20 is not None and not pd.isna(sma20):
        if price > sma20:
            notes.append(f"Price (${price:.2f}) is above its 20-day average (${sma20:.2f}) — short-term uptrend.")
        else:
            notes.append(f"Price (${price:.2f}) is below its 20-day average (${sma20:.2f}) — short-term downtrend.")

    vol = row.get("volatility_20")
    if vol is not None and not pd.isna(vol):
        notes.append(f"20-day volatility is {vol * 100:.2f}% (daily return std. dev.).")

    return notes
