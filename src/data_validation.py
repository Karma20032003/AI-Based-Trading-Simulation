"""
Data validation and cleaning.

Checks: missing values, duplicate records, incorrect dates, incorrect values,
data types, chronological order, ticker consistency.
"""
from typing import Dict, List

import numpy as np
import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """Run a battery of checks and return a report dict (does not mutate df)."""
    report: Dict[str, object] = {}

    report["n_rows"] = len(df)
    report["n_tickers"] = df["ticker"].nunique() if "ticker" in df else None
    report["missing_values"] = df.isna().sum().to_dict()
    report["duplicate_rows"] = int(
        df.duplicated(subset=["ticker", "date"]).sum()
        if {"ticker", "date"}.issubset(df.columns)
        else df.duplicated().sum()
    )

    bad_prices = []
    for col in ["open", "high", "low", "close", "adj_close"]:
        if col in df.columns:
            n_bad = int(((df[col] <= 0) | df[col].isna()).sum())
            if n_bad:
                bad_prices.append((col, n_bad))
    report["non_positive_or_missing_prices"] = bad_prices

    if "volume" in df.columns:
        report["negative_volume_rows"] = int((df["volume"] < 0).sum())

    chrono_issues = []
    if {"ticker", "date"}.issubset(df.columns):
        for ticker, g in df.groupby("ticker"):
            dates = g["date"].values
            if not np.all(dates[:-1] <= dates[1:]):
                chrono_issues.append(ticker)
    report["tickers_out_of_chronological_order"] = chrono_issues

    report["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    report["is_clean"] = (
        report["duplicate_rows"] == 0
        and not any(v for v in report["missing_values"].values())
        and not bad_prices
        and not chrono_issues
    )
    return report


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply standard cleaning: drop duplicates, drop rows with missing/invalid
    prices, enforce chronological order per ticker, correct dtypes.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in out.columns]
    out = out.dropna(subset=price_cols + ["date", "ticker"])
    for c in price_cols:
        out = out[out[c] > 0]

    if "volume" in out.columns:
        out = out[out["volume"] >= 0]

    out = out.drop_duplicates(subset=["ticker", "date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def check_ticker_consistency(df: pd.DataFrame, expected_tickers: List[str]) -> Dict[str, object]:
    """Verify every expected ticker is present and has a reasonable row count."""
    present = set(df["ticker"].unique())
    missing = [t for t in expected_tickers if t not in present]
    counts = df.groupby("ticker").size().to_dict()
    return {"missing_tickers": missing, "row_counts": counts}
