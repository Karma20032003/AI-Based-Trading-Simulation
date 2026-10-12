"""
XGBoost baseline model (tabular, row-level features -- no sequence).
"""
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support

from config import RANDOM_SEED


def train_xgboost(X_train, y_train, X_val, y_val):
    """y values are expected in {-1, 0, 1}; XGBoost needs {0, 1, 2}."""
    label_map = {-1: 0, 0: 1, 1: 2}
    inv_map = {v: k for k, v in label_map.items()}

    y_train_enc = np.array([label_map[v] for v in y_train])
    y_val_enc = np.array([label_map[v] for v in y_val])

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_SEED,
        eval_metric="mlogloss",
    )
    model.fit(
        X_train, y_train_enc,
        eval_set=[(X_val, y_val_enc)],
        verbose=False,
    )

    val_pred_enc = model.predict(X_val)
    val_pred = np.array([inv_map[v] for v in val_pred_enc])

    acc = accuracy_score(y_val, val_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, val_pred, average="macro", zero_division=0
    )

    # Validation log-loss, the same quantity Keras reports as val_loss, so all
    # four models can be compared on it.
    val_loss = log_loss(y_val_enc, model.predict_proba(X_val), labels=[0, 1, 2])

    metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
               "final_val_loss": float(val_loss)}
    return model, metrics, label_map, inv_map


def predict_xgboost(model, X, inv_map):
    pred_enc = model.predict(X)
    return np.array([inv_map[v] for v in pred_enc])


def predict_proba_xgboost(model, X):
    """Returns array (n, 3) with columns ordered [SELL_prob, HOLD_prob, BUY_prob]."""
    return model.predict_proba(X)
