"""
End-to-end CLI pipeline: collect data -> validate -> clean -> features ->
label -> split -> train all 4 models -> save models + processed data.

Usage:
    python train_pipeline.py --tickers AAPL MSFT GOOGL AMZN --start 2018-01-01
    python train_pipeline.py --level smoke       # 4-stock smoke test
    python train_pipeline.py --level dev         # 10-stock development set
    python train_pipeline.py --epochs 20         # more epochs for real training
"""
import argparse
import json
import sys

import config
from src.data_collection import fetch_stock_data
from src.data_validation import validate_dataframe, clean_data, check_ticker_consistency
from src.feature_engineering import compute_features, drop_warmup_rows, validate_features
from src.labeling import triple_barrier_labels, drop_incomplete_tail, label_distribution
from src.sequence_generation import (
    compute_date_splits, split_by_date, make_tabular_dataset, create_sequences,
    encode_labels_for_keras,
)
from src.utils import fit_scaler, scale_tabular, scale_sequences, save_sklearn_model, save_keras_model, next_model_version
from src.database import init_db, upsert_market_data


def parse_args():
    p = argparse.ArgumentParser(description="Train the multi-stock trading models.")
    p.add_argument("--tickers", nargs="+", default=None, help="Explicit ticker list")
    p.add_argument("--level", choices=["smoke", "dev"], default=None,
                   help="Use a predefined 4-stock (smoke) or 10-stock (dev) universe")
    p.add_argument("--start", default=config.DEFAULT_START_DATE)
    p.add_argument("--end", default=config.DEFAULT_END_DATE)
    p.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    p.add_argument("--skip-db", action="store_true", help="Skip writing to the database")
    p.add_argument("--models", nargs="+", default=["xgboost", "lstm", "gru", "transformer_gru"],
                   choices=["xgboost", "lstm", "gru", "transformer_gru"])
    return p.parse_args()


def resolve_tickers(args):
    if args.tickers:
        return args.tickers
    if args.level == "smoke":
        return config.DEFAULT_TICKERS_SMOKE
    if args.level == "dev":
        return config.DEFAULT_TICKERS_DEV
    return config.DEFAULT_TICKERS_SMOKE


def main():
    args = parse_args()
    tickers = resolve_tickers(args)
    print(f"=== Training pipeline for tickers: {tickers} ===")

    print("\n[1/8] Collecting data from Yahoo Finance...")
    raw = fetch_stock_data(tickers, start=args.start, end=args.end)
    print(f"   Collected {len(raw)} rows across {raw['ticker'].nunique()} tickers.")

    print("\n[2/8] Validating raw data...")
    report = validate_dataframe(raw)
    print(f"   is_clean={report['is_clean']} duplicate_rows={report['duplicate_rows']}")
    consistency = check_ticker_consistency(raw, tickers)
    if consistency["missing_tickers"]:
        print(f"   WARNING: missing tickers: {consistency['missing_tickers']}")

    print("\n[3/8] Cleaning data...")
    clean = clean_data(raw)
    print(f"   {len(clean)} rows after cleaning.")

    if not args.skip_db:
        print("\n   Writing to database...")
        init_db()
        n_new = upsert_market_data(clean)
        print(f"   Inserted {n_new} new rows into the database.")

    clean.to_parquet(f"{config.PARQUET_DIR}/clean_market_data.parquet", index=False)

    print("\n[4/8] Feature engineering...")
    feat = compute_features(clean)
    feat_report = validate_features(feat, config.FEATURE_COLUMNS)
    total_missing = sum(feat_report["missing_per_feature"].values())
    print(f"   Computed features. Total missing (pre-warmup-drop): {total_missing}")
    feat = drop_warmup_rows(feat, config.FEATURE_COLUMNS)
    print(f"   {len(feat)} rows after dropping warmup NaNs.")

    print("\n[5/8] Triple-barrier labeling...")
    labeled = triple_barrier_labels(feat)
    labeled = drop_incomplete_tail(labeled)
    print(f"   Label distribution:\n{label_distribution(labeled)}")
    labeled.to_parquet(f"{config.PARQUET_DIR}/market_features_labeled.parquet", index=False)

    print("\n[6/8] Chronological train/val/test split...")
    train_end, val_end = compute_date_splits(labeled)
    train_df, val_df, test_df = split_by_date(labeled, train_end, val_end)
    print(f"   train={len(train_df)} val={len(val_df)} test={len(test_df)} "
          f"(train_end={train_end.date()}, val_end={val_end.date()})")

    results = {}

    if "xgboost" in args.models:
        print("\n[7/8] Training XGBoost (tabular)...")
        from src.models.xgboost_model import train_xgboost
        X_train, y_train = make_tabular_dataset(train_df, config.FEATURE_COLUMNS)
        X_val, y_val = make_tabular_dataset(val_df, config.FEATURE_COLUMNS)
        scaler = fit_scaler(X_train)
        model, metrics, label_map, inv_map = train_xgboost(
            scale_tabular(X_train, scaler), y_train, scale_tabular(X_val, scaler), y_val
        )
        version = next_model_version("xgboost")
        path = save_sklearn_model(model, "xgboost", version, metrics, extra={"scaler_type": "tabular"})
        import joblib
        joblib.dump(scaler, f"{config.MODELS_DIR}/xgboost_{version}_scaler.pkl")
        print(f"   Saved to {path}. Metrics: {metrics}")
        results["xgboost"] = metrics

    if any(m in args.models for m in ["lstm", "gru", "transformer_gru"]):
        print("\n   Building sequences for temporal models...")
        X_seq, y_seq, meta = create_sequences(labeled, config.FEATURE_COLUMNS, seq_len=config.SEQUENCE_LENGTH)
        train_mask = meta["date"] <= train_end
        val_mask = (meta["date"] > train_end) & (meta["date"] <= val_end)

        X_train_seq, y_train_seq = X_seq[train_mask.values], y_seq[train_mask.values]
        X_val_seq, y_val_seq = X_seq[val_mask.values], y_seq[val_mask.values]

        seq_scaler = fit_scaler(X_train_seq.reshape(-1, X_train_seq.shape[-1]))
        X_train_seq_s = scale_sequences(X_train_seq, seq_scaler)
        X_val_seq_s = scale_sequences(X_val_seq, seq_scaler)
        y_train_enc = encode_labels_for_keras(y_train_seq)
        y_val_enc = encode_labels_for_keras(y_val_seq)

        import joblib

        if "lstm" in args.models:
            print("\n[8/8] Training LSTM...")
            from src.models.lstm_model import train_lstm
            model, metrics, _ = train_lstm(X_train_seq_s, y_train_enc, X_val_seq_s, y_val_enc, epochs=args.epochs)
            version = next_model_version("lstm")
            path = save_keras_model(model, "lstm", version, metrics)
            joblib.dump(seq_scaler, f"{config.MODELS_DIR}/lstm_{version}_scaler.pkl")
            print(f"   Saved to {path}. Metrics: {metrics}")
            results["lstm"] = metrics

        if "gru" in args.models:
            print("\n   Training GRU...")
            from src.models.gru_model import train_gru
            model, metrics, _ = train_gru(X_train_seq_s, y_train_enc, X_val_seq_s, y_val_enc, epochs=args.epochs)
            version = next_model_version("gru")
            path = save_keras_model(model, "gru", version, metrics)
            joblib.dump(seq_scaler, f"{config.MODELS_DIR}/gru_{version}_scaler.pkl")
            print(f"   Saved to {path}. Metrics: {metrics}")
            results["gru"] = metrics

        if "transformer_gru" in args.models:
            print("\n   Training Transformer-GRU...")
            from src.models.transformer_gru_model import train_transformer_gru
            model, metrics, _ = train_transformer_gru(X_train_seq_s, y_train_enc, X_val_seq_s, y_val_enc, epochs=args.epochs)
            version = next_model_version("transformer_gru")
            path = save_keras_model(model, "transformer_gru", version, metrics)
            joblib.dump(seq_scaler, f"{config.MODELS_DIR}/transformer_gru_{version}_scaler.pkl")
            print(f"   Saved to {path}. Metrics: {metrics}")
            results["transformer_gru"] = metrics

    print("\n=== DONE. Model comparison ===")
    print(json.dumps(results, indent=2))
    print(f"\nModels saved under: {config.MODELS_DIR}")
    print("Run the app with:  streamlit run app.py")


if __name__ == "__main__":
    sys.exit(main())
