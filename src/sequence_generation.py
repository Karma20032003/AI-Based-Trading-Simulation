"""
Sequence generation for temporal models (LSTM / GRU / Transformer-GRU) and
tabular preparation for XGBoost.

Chronological train/val/test splits are done by DATE cutoffs (computed once,
globally) rather than random shuffling, so no future information leaks into
training.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import SEQUENCE_LENGTH, TRAIN_FRAC, VAL_FRAC


def compute_date_splits(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return (train_end_date, val_end_date) chronological cutoffs computed
    over the full universe of dates."""
    dates = np.sort(df["date"].unique())
    n = len(dates)
    train_end = dates[int(n * TRAIN_FRAC) - 1]
    val_end = dates[int(n * (TRAIN_FRAC + VAL_FRAC)) - 1]
    return pd.Timestamp(train_end), pd.Timestamp(val_end)


def split_by_date(df: pd.DataFrame, train_end, val_end):
    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]
    return train, val, test


def make_tabular_dataset(df: pd.DataFrame, feature_cols: List[str], label_col: str = "label"):
    """Row-level tabular dataset for XGBoost: X = features at date t, y = label at t."""
    X = df[feature_cols].values
    y = df[label_col].values
    return X, y


def create_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = SEQUENCE_LENGTH,
    label_col: str = "label",
):
    """
    Build (X, y, meta) sequence arrays PER TICKER so that a sequence never
    mixes rows from two different stocks.

    X shape: (n_sequences, seq_len, n_features)
    y shape: (n_sequences,)
    meta: DataFrame with ticker/date of the label row for each sequence.
    """
    X_list, y_list, meta_rows = [], [], []

    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("date").reset_index(drop=True)
        feats = g[feature_cols].values
        labels = g[label_col].values
        dates = g["date"].values

        if len(g) <= seq_len:
            continue

        for i in range(seq_len, len(g)):
            X_list.append(feats[i - seq_len:i])
            y_list.append(labels[i])
            meta_rows.append({"ticker": ticker, "date": dates[i]})

    if not X_list:
        raise ValueError("No sequences could be generated -- check seq_len vs. data length.")

    X = np.stack(X_list)
    y = np.array(y_list)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def encode_labels_for_keras(y: np.ndarray) -> np.ndarray:
    """Map {-1, 0, 1} -> {0, 1, 2} for categorical training, and return the
    mapping so predictions can be decoded back."""
    mapping = {-1: 0, 0: 1, 1: 2}
    return np.array([mapping[v] for v in y])


def decode_labels_from_keras(y_enc: np.ndarray) -> np.ndarray:
    inverse = {0: -1, 1: 0, 2: 1}
    return np.array([inverse[v] for v in y_enc])
