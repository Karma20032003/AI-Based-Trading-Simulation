"""
Data collection from Yahoo Finance.

Requires an internet connection when actually run. Raw data is cached to
data/raw/<TICKER>.csv so repeated runs don't re-download unnecessarily.
"""
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

from config import RAW_DATA_DIR, DEFAULT_START_DATE, DEFAULT_END_DATE


def fetch_stock_data(
    tickers: List[str],
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = DEFAULT_END_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a list of tickers via yfinance.

    Returns a single tidy DataFrame with columns:
    date, ticker, open, high, low, close, adj_close, volume
    """
    import yfinance as yf  # imported lazily so the rest of the package can be
                            # used/tested without yfinance installed

    end = end or datetime.today().strftime("%Y-%m-%d")
    frames = []

    for ticker in tickers:
        cache_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")

        if os.path.exists(cache_path) and not force_refresh:
            df = pd.read_csv(cache_path, parse_dates=["date"])
        else:
            raw = yf.download(
                ticker, start=start, end=end, auto_adjust=False, progress=False
            )
            if raw.empty:
                print(f"[WARN] No data returned for {ticker}, skipping.")
                continue

            # yfinance can return a MultiIndex column frame for a single
            # ticker in some versions -- flatten defensively.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]

            raw = raw.reset_index()
            raw.columns = [str(c).lower().replace(" ", "_") for c in raw.columns]
            raw = raw.rename(columns={"adj_close": "adj_close"})
            raw["ticker"] = ticker
            df = raw[
                ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
            ]
            df.to_csv(cache_path, index=False)

        frames.append(df)

    if not frames:
        raise RuntimeError("No data was collected for any ticker.")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)
    return combined


def load_cached_tickers() -> List[str]:
    """Return tickers currently cached in data/raw/."""
    if not os.path.isdir(RAW_DATA_DIR):
        return []
    return sorted(
        f[:-4] for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")
    )
