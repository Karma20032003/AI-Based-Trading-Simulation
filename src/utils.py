"""
Shared utilities: feature scaling and model artifact persistence.
"""
import json
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import MODELS_DIR


def fit_scaler(X_train_2d: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train_2d)
    return scaler


def scale_tabular(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)


def scale_sequences(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """X shape (n, seq_len, n_features) -> scale each feature column."""
    n, seq_len, n_features = X.shape
    flat = X.reshape(-1, n_features)
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(n, seq_len, n_features)


def next_model_version(model_type: str) -> str:
    existing = [
        f for f in os.listdir(MODELS_DIR)
        if f.startswith(f"{model_type}_v") and (f.endswith(".pkl") or f.endswith(".keras"))
    ]
    versions = []
    for f in existing:
        try:
            v = int(f.split("_v")[1].split(".")[0])
            versions.append(v)
        except (IndexError, ValueError):
            continue
    return f"v{(max(versions) + 1) if versions else 1}"


def save_sklearn_model(model, model_type: str, version: str, metrics: dict, extra: dict = None):
    path = os.path.join(MODELS_DIR, f"{model_type}_{version}.pkl")
    joblib.dump(model, path)
    _save_metadata(model_type, version, path, metrics, extra)
    return path


def save_keras_model(model, model_type: str, version: str, metrics: dict, extra: dict = None):
    path = os.path.join(MODELS_DIR, f"{model_type}_{version}.keras")
    model.save(path)
    _save_metadata(model_type, version, path, metrics, extra)
    return path


def _save_metadata(model_type, version, path, metrics, extra):
    meta_path = os.path.join(MODELS_DIR, f"{model_type}_{version}_meta.json")
    meta = {
        "model_type": model_type,
        "version": version,
        "file_path": path,
        "trained_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
    }
    if extra:
        meta.update(extra)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def list_saved_models():
    if not os.path.isdir(MODELS_DIR):
        return []
    metas = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_meta.json"):
            with open(os.path.join(MODELS_DIR, f)) as fh:
                metas.append(json.load(fh))
    return sorted(metas, key=lambda m: m["trained_at"], reverse=True)


def load_model_by_meta(meta: dict):
    path = meta["file_path"]
    if path.endswith(".pkl"):
        return joblib.load(path)
    elif path.endswith(".keras"):
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    raise ValueError(f"Unknown model file type: {path}")
