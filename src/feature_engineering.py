"""
Technical feature engineering.

All indicators are computed per-ticker so no information leaks across stocks,
and only use past/current data at each row (no look-ahead).
"""
import numpy as np
import pandas as pd

from config import (
    SMA_WINDOWS, EMA_WINDOWS, RSI_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL, VOLATILITY_WINDOW,
)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral RSI where undefined


def _compute_for_one_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    price = g["adj_close"] if "adj_close" in g.columns else g["close"]

    g["return_1d"] = price.pct_change()

    for w in SMA_WINDOWS:
        g[f"sma_{w}"] = price.rolling(w, min_periods=w).mean()

    for w in EMA_WINDOWS:
        g[f"ema_{w}"] = price.ewm(span=w, adjust=False).mean()

    g[f"rsi_{RSI_WINDOW}"] = _rsi(price, RSI_WINDOW)

    ema_fast = price.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = price.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    g["macd"] = macd
    g["macd_signal"] = macd_signal
    g["macd_hist"] = macd - macd_signal

    g[f"volatility_{VOLATILITY_WINDOW}"] = g["return_1d"].rolling(
        VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW
    ).std()

    return g


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical features, grouped safely per ticker."""
    pieces = [_compute_for_one_ticker(g) for _, g in df.groupby("ticker", sort=False)]
    out = pd.concat(pieces, ignore_index=True)
    return out


def validate_features(df: pd.DataFrame, feature_cols) -> dict:
    """Sanity-check computed features: missing values, obviously unreasonable values."""
    report = {}
    report["missing_per_feature"] = {c: int(df[c].isna().sum()) for c in feature_cols if c in df.columns}
    rsi_col = f"rsi_{RSI_WINDOW}"
    if rsi_col in df.columns:
        report["rsi_out_of_range"] = int(((df[rsi_col] < 0) | (df[rsi_col] > 100)).sum())
    return report


def drop_warmup_rows(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Drop the initial rows per ticker where rolling windows are still NaN."""
    return df.dropna(subset=[c for c in feature_cols if c in df.columns]).reset_index(drop=True)
