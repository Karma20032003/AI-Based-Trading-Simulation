"""
Triple-barrier labeling.

For each row (a potential entry point), look forward up to `time_limit` bars
and check which barrier is hit first:
    - price rises by >= profit_target  -> label 1 (BUY / profitable-long outcome)
    - price falls by <= stop_loss      -> label -1 (SELL / adverse outcome)
    - neither hit within time_limit    -> label 0 (HOLD)

This uses only *future* price data relative to the labeling date, which is
correct -- the label describes what *would have happened*, it is never fed
back in as a feature. The leakage check to watch for is making sure feature
columns at row t never include information from t+1 onward (handled in
feature_engineering.py).
"""
import numpy as np
import pandas as pd

from config import PROFIT_TARGET, STOP_LOSS, TIME_LIMIT_DAYS


def _label_one_ticker(g: pd.DataFrame, profit_target, stop_loss, time_limit) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    price = (g["adj_close"] if "adj_close" in g.columns else g["close"]).values
    n = len(price)
    labels = np.zeros(n, dtype=int)
    hit_day = np.full(n, np.nan)

    for i in range(n):
        entry = price[i]
        end = min(i + time_limit, n - 1)
        if end <= i:
            labels[i] = 0
            continue
        window = price[i + 1: end + 1]
        if len(window) == 0:
            labels[i] = 0
            continue
        returns = (window - entry) / entry

        hit_profit = np.where(returns >= profit_target)[0]
        hit_stop = np.where(returns <= stop_loss)[0]

        first_profit = hit_profit[0] if len(hit_profit) else np.inf
        first_stop = hit_stop[0] if len(hit_stop) else np.inf

        if first_profit == np.inf and first_stop == np.inf:
            labels[i] = 0
            hit_day[i] = time_limit
        elif first_profit <= first_stop:
            labels[i] = 1
            hit_day[i] = first_profit + 1
        else:
            labels[i] = -1
            hit_day[i] = first_stop + 1

    g["label"] = labels
    g["barrier_hit_day"] = hit_day
    return g


def triple_barrier_labels(
    df: pd.DataFrame,
    profit_target: float = PROFIT_TARGET,
    stop_loss: float = STOP_LOSS,
    time_limit: int = TIME_LIMIT_DAYS,
) -> pd.DataFrame:
    """Apply triple-barrier labeling per ticker. Last `time_limit` rows per
    ticker will have a label based on a truncated (shorter) window and should
    typically be excluded from training via `drop_incomplete_tail`."""
    pieces = [
        _label_one_ticker(g, profit_target, stop_loss, time_limit)
        for _, g in df.groupby("ticker", sort=False)
    ]
    return pd.concat(pieces, ignore_index=True)


def drop_incomplete_tail(df: pd.DataFrame, time_limit: int = TIME_LIMIT_DAYS) -> pd.DataFrame:
    """Remove the last `time_limit` rows of each ticker, whose forward window
    was truncated by the end of available data (avoids biased/misleading labels)."""
    def _trim(g):
        return g.sort_values("date").iloc[:-time_limit] if len(g) > time_limit else g.iloc[0:0]

    pieces = [_trim(g) for _, g in df.groupby("ticker", sort=False)]
    return pd.concat(pieces, ignore_index=True)


def label_distribution(df: pd.DataFrame) -> pd.Series:
    return df["label"].value_counts(normalize=True).sort_index()
