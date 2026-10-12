"""
AI-Based Multi-Stock Trading Strategy -- Streamlit Application

Run locally:
    streamlit run app.py

Deploy:
    - Streamlit Community Cloud: push this repo to GitHub, point Streamlit
      Cloud at app.py. Requires requirements.txt (already included).
    - Docker: docker build -t trading-app . && docker run -p 8501:8501 trading-app
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src.data_collection import fetch_stock_data, load_cached_tickers
from src.data_validation import validate_dataframe, clean_data
from src.feature_engineering import compute_features, drop_warmup_rows
from src.labeling import triple_barrier_labels, drop_incomplete_tail, label_distribution
from src.sequence_generation import (
    compute_date_splits, split_by_date, make_tabular_dataset, create_sequences,
    encode_labels_for_keras, decode_labels_from_keras,
)
from src.utils import (
    fit_scaler, scale_tabular, scale_sequences, save_sklearn_model, save_keras_model,
    next_model_version, list_saved_models, load_model_by_meta,
)
from src.signals import build_signal_table, signals_for_ticker
from src.simulator import run_simulation
from src.recommendation import recommend_for_ticker, W_MODEL, W_TECH, THRESHOLD
from src.portfolio_simulator import run_portfolio_simulation
from src import manual_trading as mt

st.set_page_config(page_title="AI Multi-Stock Trading", layout="wide")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
for key, default in [
    ("raw_data", None), ("clean_data", None), ("feature_data", None),
    ("labeled_data", None), ("train_val_test", None), ("trained_models", {}),
    ("last_signal_table", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_price_col(df):
    return "adj_close" if "adj_close" in df.columns else "close"


def build_full_signal_table(model_info, data_slice, model_name):
    """Build a signal table across all tickers for the portfolio simulator."""
    if model_info["type"] == "tabular":
        X, _ = make_tabular_dataset(data_slice, config.FEATURE_COLUMNS)
        if len(X) == 0:
            return None
        X_s = scale_tabular(X, model_info["scaler"])
        from src.models.xgboost_model import predict_xgboost
        preds = predict_xgboost(model_info["model"], X_s, model_info["inv_map"])
        meta = data_slice[["ticker", "date"]].reset_index(drop=True)
        return build_signal_table(meta, data_slice, preds, model_name, model_info["version"])
    else:
        X_seq, y_seq, meta = create_sequences(data_slice, config.FEATURE_COLUMNS, seq_len=config.SEQUENCE_LENGTH)
        if len(X_seq) == 0:
            return None
        X_seq_s = scale_sequences(X_seq, model_info["scaler"])
        pred_enc = np.argmax(model_info["model"].predict(X_seq_s, verbose=0), axis=1)
        preds = decode_labels_from_keras(pred_enc)
        return build_signal_table(meta, data_slice, preds, model_name, model_info["version"])


# ---------------------------------------------------------------------------
# Sidebar: global data controls
# ---------------------------------------------------------------------------
st.sidebar.title("Data Controls")

preset = st.sidebar.selectbox(
    "Ticker universe",
    ["Custom", "4-stock smoke test", "10-stock development set"],
)
if preset == "4-stock smoke test":
    default_tickers = ", ".join(config.DEFAULT_TICKERS_SMOKE)
elif preset == "10-stock development set":
    default_tickers = ", ".join(config.DEFAULT_TICKERS_DEV)
else:
    default_tickers = "AAPL, MSFT"

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", value=default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start date", value=pd.Timestamp(config.DEFAULT_START_DATE))
end_date = st.sidebar.date_input("End date", value=pd.Timestamp.today())

force_refresh = st.sidebar.checkbox("Force refresh from Yahoo Finance", value=False)

if st.sidebar.button("Fetch & Process Data", type="primary"):
    with st.spinner("Fetching data from Yahoo Finance..."):
        try:
            raw = fetch_stock_data(
                tickers, start=str(start_date), end=str(end_date), force_refresh=force_refresh
            )
            st.session_state.raw_data = raw
            clean = clean_data(raw)
            st.session_state.clean_data = clean
            feat = compute_features(clean)
            feat = drop_warmup_rows(feat, config.FEATURE_COLUMNS)
            st.session_state.feature_data = feat
            labeled = triple_barrier_labels(feat)
            labeled = drop_incomplete_tail(labeled)
            st.session_state.labeled_data = labeled

            train_end, val_end = compute_date_splits(labeled)
            train_df, val_df, test_df = split_by_date(labeled, train_end, val_end)
            st.session_state.train_val_test = (train_df, val_df, test_df, train_end, val_end)

            st.sidebar.success(f"Loaded {len(clean)} rows for {clean['ticker'].nunique()} tickers.")
        except Exception as e:
            st.sidebar.error(f"Error fetching data: {e}")

cached = load_cached_tickers()
if cached:
    st.sidebar.caption(f"Cached tickers on disk: {', '.join(cached)}")

st.sidebar.divider()
st.sidebar.caption(
    "Educational / research tool for historical analysis and virtual trading. "
    "No live broker execution or real-money trading."
)

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------
st.title("AI-Based Multi-Stock Trading Strategy")
st.caption("LSTM · GRU · Transformer-GRU · XGBoost — with single-asset trading simulation")

home_tab, market_tab, training_tab, rec_tab, manual_tab, simulator_tab, portfolio_tab = st.tabs([
    "Home",
    "Market & Technical Analysis",
    "Model Training & Comparison",
    "Recommendations",
    "Virtual Trading Desk",
    "Trading Simulator",
    "Portfolio Simulator",
])

# ============================== HOME ========================================
with home_tab:
    st.subheader("Welcome")
    st.markdown(
        """
This application implements the full pipeline described in the project spec:

1. **Data collection** from Yahoo Finance, validated and cleaned.
2. **Feature engineering**: returns, SMA/EMA, RSI, MACD, volatility.
3. **Triple-barrier labeling** (profit target / stop loss / time limit) to produce BUY/SELL/HOLD training labels.
4. **Four models** trained and compared: XGBoost, LSTM, GRU, Transformer-GRU.
5. A **single-stock virtual trading simulator** with entry/exit tracking, P&L, and portfolio value.
6. A **multi-stock portfolio simulator** with order and risk controls.
7. **Interactive charts** for price, indicators, signals, and results.

**Workflow:** use the sidebar to fetch & process data first, then go to
*Model Training & Comparison* to train models, then use the trading simulators
to evaluate a single stock or a multi-stock portfolio.
        """
    )
    if st.session_state.clean_data is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Tickers loaded", st.session_state.clean_data["ticker"].nunique())
        c2.metric("Rows", len(st.session_state.clean_data))
        c3.metric("Date range", f"{st.session_state.clean_data['date'].min().date()} → {st.session_state.clean_data['date'].max().date()}")
    else:
        st.info("No data loaded yet. Use the sidebar to fetch data.")

# ===================== MARKET & TECHNICAL ANALYSIS ==========================
with market_tab:
    st.subheader("Market & Technical Analysis")
    if st.session_state.feature_data is None:
        st.info("Fetch data from the sidebar first.")
    else:
        df = st.session_state.feature_data
        ticker_sel = st.selectbox("Select ticker", sorted(df["ticker"].unique()), key="ta_ticker")
        range_sel = st.select_slider(
            "Date range", options=["1M", "3M", "6M", "1Y", "3Y", "5Y", "Full"], value="1Y"
        )
        g = df[df["ticker"] == ticker_sel].sort_values("date")
        if range_sel != "Full":
            days_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "3Y": 756, "5Y": 1260}
            g = g.tail(days_map[range_sel])

        price_col = get_price_col(g)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=g["date"], open=g["open"], high=g["high"], low=g["low"], close=g["close"], name="OHLC"
        ))
        for w in config.SMA_WINDOWS:
            col = f"sma_{w}"
            if col in g.columns:
                fig.add_trace(go.Scatter(x=g["date"], y=g[col], name=f"SMA {w}", line=dict(width=1)))
        fig.update_layout(title=f"{ticker_sel} — Price & Moving Averages", height=450, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            fig_vol = go.Figure(go.Bar(x=g["date"], y=g["volume"], name="Volume"))
            fig_vol.update_layout(title="Volume", height=300)
            st.plotly_chart(fig_vol, width="stretch")

            fig_rsi = go.Figure(go.Scatter(x=g["date"], y=g["rsi_14"], name="RSI(14)"))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="RSI (14)", height=300)
            st.plotly_chart(fig_rsi, width="stretch")

        with c2:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=g["date"], y=g["macd"], name="MACD"))
            fig_macd.add_trace(go.Scatter(x=g["date"], y=g["macd_signal"], name="Signal"))
            fig_macd.add_trace(go.Bar(x=g["date"], y=g["macd_hist"], name="Histogram"))
            fig_macd.update_layout(title="MACD", height=300)
            st.plotly_chart(fig_macd, width="stretch")

            fig_vola = go.Figure(go.Scatter(x=g["date"], y=g["volatility_20"], name="Volatility (20d)"))
            fig_vola.update_layout(title="Rolling Volatility (20d)", height=300)
            st.plotly_chart(fig_vola, width="stretch")

        with st.expander("Data validation report"):
            st.json(validate_dataframe(st.session_state.clean_data))

        with st.expander("Label distribution (triple-barrier)"):
            if st.session_state.labeled_data is not None:
                dist = label_distribution(st.session_state.labeled_data)
                st.bar_chart(dist.rename(index=config.LABEL_MAP))

# ===================== MODEL TRAINING & COMPARISON ===========================
with training_tab:
    st.subheader("Model Training & Comparison")
    if st.session_state.train_val_test is None:
        st.info("Fetch data from the sidebar first.")
    else:
        train_df, val_df, test_df, train_end, val_end = st.session_state.train_val_test
        st.caption(f"Train: {len(train_df)} rows (≤{train_end.date()}) · "
                   f"Val: {len(val_df)} rows (≤{val_end.date()}) · Test: {len(test_df)} rows")

        model_choices = st.multiselect(
            "Models to train", ["XGBoost", "LSTM", "GRU", "Transformer-GRU"],
            default=["XGBoost", "LSTM", "GRU", "Transformer-GRU"],
        )
        epochs = st.slider("Epochs (for LSTM/GRU/Transformer-GRU)", 2, 50, 10)

        if st.button("Train selected models", type="primary"):
            results = {}
            progress = st.progress(0.0, text="Starting...")
            step = 0
            total_steps = len(model_choices)

            if "XGBoost" in model_choices:
                progress.progress(step / total_steps, text="Training XGBoost...")
                from src.models.xgboost_model import train_xgboost
                X_train, y_train = make_tabular_dataset(train_df, config.FEATURE_COLUMNS)
                X_val, y_val = make_tabular_dataset(val_df, config.FEATURE_COLUMNS)
                scaler = fit_scaler(X_train)
                model, metrics, label_map, inv_map = train_xgboost(
                    scale_tabular(X_train, scaler), y_train, scale_tabular(X_val, scaler), y_val
                )
                version = next_model_version("xgboost")
                save_sklearn_model(model, "xgboost", version, metrics)
                st.session_state.trained_models["xgboost"] = {
                    "model": model, "scaler": scaler, "inv_map": inv_map,
                    "metrics": metrics, "version": version, "type": "tabular",
                }
                results["XGBoost"] = metrics
                step += 1

            needs_seq = any(m in model_choices for m in ["LSTM", "GRU", "Transformer-GRU"])
            if needs_seq:
                progress.progress(step / total_steps, text="Building sequences...")
                labeled = st.session_state.labeled_data
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

                if "LSTM" in model_choices:
                    progress.progress(step / total_steps, text="Training LSTM...")
                    from src.models.lstm_model import train_lstm
                    model, metrics, _ = train_lstm(X_train_seq_s, y_train_enc, X_val_seq_s, y_val_enc, epochs=epochs)
                    version = next_model_version("lstm")
                    save_keras_model(model, "lstm", version, metrics)
                    st.session_state.trained_models["lstm"] = {
                        "model": model, "scaler": seq_scaler, "metrics": metrics,
                        "version": version, "type": "sequence",
                    }
                    results["LSTM"] = metrics
                    step += 1

                if "GRU" in model_choices:
                    progress.progress(step / total_steps, text="Training GRU...")
                    from src.models.gru_model import train_gru
                    model, metrics, _ = train_gru(X_train_seq_s, y_train_enc, X_val_seq_s, y_val_enc, epochs=epochs)
                    version = next_model_version("gru")
                    save_keras_model(model, "gru", version, metrics)
                    st.session_state.trained_models["gru"] = {
                        "model": model, "scaler": seq_scaler, "metrics": metrics,
                        "version": version, "type": "sequence",
                    }
                    results["GRU"] = metrics
                    step += 1

                if "Transformer-GRU" in model_choices:
                    progress.progress(step / total_steps, text="Training Transformer-GRU...")
                    from src.models.transformer_gru_model import train_transformer_gru
                    model, metrics, _ = train_transformer_gru(X_train_seq_s, y_train_enc, X_val_seq_s, y_val_enc, epochs=epochs)
                    version = next_model_version("transformer_gru")
                    save_keras_model(model, "transformer_gru", version, metrics)
                    st.session_state.trained_models["transformer_gru"] = {
                        "model": model, "scaler": seq_scaler, "metrics": metrics,
                        "version": version, "type": "sequence",
                    }
                    results["Transformer-GRU"] = metrics
                    step += 1

            progress.progress(1.0, text="Done.")
            st.success("Training complete.")
            st.session_state["last_results"] = results

        if st.session_state.get("last_results"):
            st.markdown("### Comparison")
            comp_df = pd.DataFrame(st.session_state["last_results"]).T
            st.dataframe(comp_df.style.highlight_max(axis=0, color="#264d26"), width="stretch")
            st.bar_chart(comp_df[["accuracy", "f1"]], stack=False)

        with st.expander("Previously saved models (this machine)"):
            saved = list_saved_models()
            if saved:
                st.dataframe(pd.DataFrame(saved)[["model_type", "version", "trained_at"]], width="stretch")
            else:
                st.caption("No saved models yet.")

# ============================ RECOMMENDATIONS ================================
with rec_tab:
    st.subheader("Current BUY / HOLD / SELL Recommendation")
    st.caption(
        f"Blends model probabilities ({int(W_MODEL*100)}%) with technical indicators ({int(W_TECH*100)}%). "
        f"Score ≥ +{THRESHOLD} → BUY, ≤ −{THRESHOLD} → SELL, otherwise HOLD. "
        "The model estimates which barrier (+/- move or neither) is hit within the next "
        f"{config.TIME_LIMIT_DAYS} trading days. It does not predict a price."
    )
    if st.session_state.feature_data is None:
        st.info("Fetch data from the sidebar first.")
    else:
        feat_all = st.session_state.feature_data
        trained = st.session_state.trained_models
        options = ["Technical indicators only"] + list(trained.keys())
        default_idx = 1 if trained else 0
        c1, c2 = st.columns(2)
        rec_ticker = c1.selectbox("Stock", sorted(feat_all["ticker"].unique()), key="rec_ticker")
        rec_model = c2.selectbox("Model", options, index=default_idx, key="rec_model")
        model_info = trained.get(rec_model) if rec_model != "Technical indicators only" else None
        if not trained:
            st.info("No trained model yet, so this uses technical indicators only. "
                    "Train models in the 'Model Training & Comparison' tab to add the model view.")

        g = feat_all[feat_all["ticker"] == rec_ticker].sort_values("date")
        res = recommend_for_ticker(g, model_info)
        tech = res["technical"]

        color = {"BUY": "#2E7D32", "SELL": "#C62828", "HOLD": "#F9A825"}[res["recommendation"]]
        st.markdown(
            f"<div style='padding:18px;border-radius:10px;background:{color}22;border:2px solid {color};'>"
            f"<span style='font-size:14px'>{rec_ticker} as of {tech['date'].date()} at ${tech['price']:,.2f}</span><br>"
            f"<span style='font-size:44px;font-weight:700;color:{color}'>{res['recommendation']}</span>"
            f"<span style='font-size:16px'> &nbsp; combined score {res['combined_score']:+.2f}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Basis: {res['basis']}" + (f" · {res['agreement']}" if res["agreement"] else ""))

        left, right = st.columns(2)
        with left:
            st.markdown("#### Model view")
            if res["probs"]:
                pf = pd.DataFrame({"Outcome within horizon": ["SELL (drop)", "HOLD (no big move)", "BUY (rise)"],
                                   "Probability": [res["probs"]["SELL"], res["probs"]["HOLD"], res["probs"]["BUY"]]})
                st.dataframe(pf.style.format({"Probability": "{:.1%}"}), width="stretch", hide_index=True)
                st.bar_chart(pf.set_index("Outcome within horizon"))
            else:
                st.write("No model probabilities (no model selected, or not enough history).")
        with right:
            st.markdown("#### Technical indicators")
            tdf = pd.DataFrame(tech["readings"])
            tdf["signal"] = tdf["score"].map(lambda v: "Bullish" if v > 0 else ("Bearish" if v < 0 else "Neutral"))
            st.dataframe(tdf[["indicator", "signal", "reading"]], width="stretch", hide_index=True)
            st.caption(f"Indicator score {tech['score']:+.2f} · {tech['volatility_note']}")

        st.markdown("#### Expected price range (statistical, not a forecast)")
        st.write(
            f"Using current volatility, a typical (~68%) range for {rec_ticker} after "
            f"{res['horizon_days']} trading days is **${res['range_low']:,.2f} to ${res['range_high']:,.2f}**. "
            "This shows how far the price could plausibly move in either direction; it does not say which way."
        )
        recent = g.tail(120)
        pcol = get_price_col(g)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["date"], y=recent[pcol], name="Price"))
        future_x = [tech["date"], tech["date"] + pd.tseries.offsets.BDay(res["horizon_days"])]
        fig.add_trace(go.Scatter(x=future_x + future_x[::-1],
                                 y=[tech["price"], res["range_high"], res["range_low"], tech["price"]],
                                 fill="toself", name="1σ range", line=dict(width=0),
                                 fillcolor="rgba(100,150,255,0.25)"))
        fig.update_layout(height=350, title=f"{rec_ticker}: recent price and {res['horizon_days']}-day range")
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### All loaded stocks")
        rows = []
        for t in sorted(feat_all["ticker"].unique()):
            gt = feat_all[feat_all["ticker"] == t].sort_values("date")
            r = recommend_for_ticker(gt, model_info)
            rows.append({"Ticker": t, "Price": r["technical"]["price"], "Recommendation": r["recommendation"],
                         "Score": r["combined_score"],
                         "P(BUY)": r["probs"]["BUY"] if r["probs"] else None,
                         "P(SELL)": r["probs"]["SELL"] if r["probs"] else None,
                         "Indicator score": r["technical"]["score"]})
        st.dataframe(pd.DataFrame(rows).style.format(
            {"Price": "${:,.2f}", "Score": "{:+.2f}", "P(BUY)": "{:.0%}", "P(SELL)": "{:.0%}", "Indicator score": "{:+.2f}"},
            na_rep="-"), width="stretch", hide_index=True)

        st.warning("Educational output only, not financial advice. The score weights and thresholds are simple "
                   "heuristics, and model accuracy on real markets is often modest. Verify against simulation results "
                   "tab before trusting any model.")

# ========================= VIRTUAL TRADING DESK (MANUAL) =====================
with manual_tab:
    st.subheader("Virtual Trading Desk")
    st.caption(
        "The AI only advises here — every Buy, Sell, and Hold is your decision. Nothing executes "
        "automatically. The session starts on the latest available market-data date. Review earlier "
        "dates or move forward to the latest date; at each date you see the AI's recommendation and "
        "indicators, then choose what to do yourself."
    )
    if st.session_state.feature_data is None:
        st.info("Fetch data from the sidebar first.")
    else:
        feat_all = st.session_state.feature_data
        trained = st.session_state.trained_models
        all_tickers = sorted(feat_all["ticker"].unique())

        if st.session_state.get("manual_session") is None:
            st.markdown("#### Set up your session")
            c1, c2 = st.columns(2)
            desk_tickers = c1.multiselect("Stocks to trade", all_tickers, default=all_tickers[:3], key="desk_tickers")
            options = ["Technical indicators only"] + list(trained.keys())
            desk_model = c2.selectbox("Model for recommendations", options,
                                       index=1 if trained else 0, key="desk_model")

            c3, c4 = st.columns(2)
            desk_cash = c3.number_input("Starting cash ($)", min_value=100.0, value=10_000.0, step=500.0, key="desk_cash")
            desk_period = c4.selectbox("Historical date scope", ["Test set only", "Full history"], key="desk_period")

            if not trained:
                st.info("No trained model yet — recommendations will use technical indicators only. "
                        "Train a model in 'Model Training & Comparison' to add the model view.")

            if st.button("Start session", type="primary", disabled=len(desk_tickers) == 0):
                if st.session_state.train_val_test is not None:
                    _, _, _, _, val_end = st.session_state.train_val_test
                    date_source = feat_all[feat_all["date"] > val_end] if desk_period == "Test set only" else feat_all
                else:
                    date_source = feat_all
                date_source = date_source[date_source["ticker"].isin(desk_tickers)]
                session_dates = sorted(date_source["date"].unique())

                if len(session_dates) < 2:
                    st.error("Not enough dates in this range to run a session.")
                else:
                    session = mt.init_session(
                        desk_tickers, list(pd.to_datetime(session_dates)), desk_cash,
                        start_at_latest=True,
                    )
                    st.session_state["manual_session"] = session
                    st.session_state["manual_session_model"] = desk_model
                    st.rerun()

        else:
            session = st.session_state["manual_session"]
            desk_model = st.session_state["manual_session_model"]
            model_info = trained.get(desk_model) if desk_model != "Technical indicators only" else None

            price_series = {t: mt.build_price_series(feat_all, t) for t in session["tickers"]}
            mt.record_equity_point(session, price_series)

            today = mt.current_date(session)
            top_l, top_r = st.columns([3, 1])
            top_l.markdown(f"### {pd.Timestamp(today).strftime('%A, %B %d, %Y')} "
                            f"— day {session['idx'] + 1} of {len(session['dates'])}")
            if top_r.button("Reset session"):
                st.session_state["manual_session"] = None
                st.rerun()

            m1, m2, m3, m4 = st.columns(4)
            cur_value = mt.portfolio_value(session, price_series, today)
            m1.metric("Cash", f"${session['cash']:,.2f}")
            m2.metric("Portfolio value", f"${cur_value:,.2f}",
                      f"{(cur_value / session['starting_cash'] - 1) * 100:+.2f}%")
            m3.metric("Open positions", len(session["positions"]))
            m4.metric("Trades made", len(session["trade_log"]))

            st.markdown("#### Open positions")
            positions_table = mt.open_positions_table(session, feat_all, today)
            if positions_table.empty:
                st.caption("No open positions yet. Buy a stock to start tracking its P/L here.")
            else:
                formatted_positions = positions_table.style.format({
                    "Qty": "{:,.3f}",
                    "Price paid": "${:,.2f}",
                    "Last price": "${:,.2f}",
                    "Day's chg": "${:+,.2f}",
                    "P/L": "${:+,.2f}",
                    "P/L %": "{:+.2f}%",
                    "Market value": "${:,.2f}",
                }, na_rep="—")
                formatted_positions = formatted_positions.map(
                    lambda value: (
                        "color: #2E7D32" if value >= 0 else "color: #C62828"
                    ) if pd.notna(value) else "",
                    subset=["Day's chg", "P/L", "P/L %"],
                )
                st.dataframe(formatted_positions, width="stretch", hide_index=True)
                st.caption("P/L is unrealized for open positions; realized P/L appears in trade history.")

            st.divider()
            st.markdown("#### Recommendations for this date — advice only, you decide")

            for ticker in session["tickers"]:
                g_full = feat_all[feat_all["ticker"] == ticker].sort_values("date")
                g_so_far = g_full[g_full["date"] <= today]  # never let the recommendation see the future
                if len(g_so_far) < 2:
                    continue
                rec = recommend_for_ticker(g_so_far, model_info)
                price = rec["technical"]["price"]
                held = session["positions"].get(ticker, 0.0)

                color = {"BUY": "#2E7D32", "SELL": "#C62828", "HOLD": "#F9A825"}[rec["recommendation"]]
                with st.container(border=True):
                    h1, h2 = st.columns([3, 2])
                    with h1:
                        st.markdown(
                            f"**{ticker}** — ${price:,.2f} &nbsp;·&nbsp; "
                            f"<span style='color:{color};font-weight:700'>{rec['recommendation']}</span> "
                            f"(score {rec['combined_score']:+.2f})",
                            unsafe_allow_html=True,
                        )
                        bullets = "  \n".join(f"- {r['reading']}" for r in rec["technical"]["readings"])
                        st.markdown(bullets)
                        if rec["probs"]:
                            st.caption(
                                f"Model view: P(BUY)={rec['probs']['BUY']:.0%} · "
                                f"P(HOLD)={rec['probs']['HOLD']:.0%} · P(SELL)={rec['probs']['SELL']:.0%}"
                            )
                        if held > 0:
                            st.caption(f"You currently hold {held:.3f} shares.")

                    with h2:
                        amount = st.number_input(
                            "Amount to buy ($)", min_value=0.0, value=min(1000.0, session["cash"]),
                            step=100.0, key=f"buy_amt_{ticker}_{session['idx']}",
                        )
                        bcol, scol = st.columns(2)
                        if bcol.button("Buy", key=f"buy_{ticker}_{session['idx']}", width="stretch"):
                            err = mt.execute_buy(session, ticker, price, amount)
                            if err:
                                st.error(err)
                            else:
                                mt.record_equity_point(session, price_series)
                                st.rerun()
                        if scol.button("Sell all", key=f"sell_{ticker}_{session['idx']}",
                                       width="stretch", disabled=held <= 0):
                            err = mt.execute_sell(session, ticker, price)
                            if err:
                                st.error(err)
                            else:
                                mt.record_equity_point(session, price_series)
                                st.rerun()

            st.divider()
            nav1, nav2, nav3 = st.columns([1, 1, 2])
            if nav1.button("Previous day (Review history)", disabled=session["idx"] == 0):
                mt.rewind(session)
                st.rerun()
            if nav2.button("Next day (Return to latest)", type="primary", disabled=mt.is_last_day(session)):
                mt.advance(session)
                st.rerun()
            skip_n = nav3.number_input("Skip toward latest (days)", min_value=1, value=5, step=1, key="skip_n")
            if nav3.button(f"Advance {int(skip_n)} days", disabled=mt.is_last_day(session)):
                mt.advance(session, int(skip_n))
                st.rerun()

            if mt.is_last_day(session):
                st.success("You're at the latest available date in this session.")

            st.markdown("#### Your portfolio vs. doing nothing (equal-weight Buy & Hold)")
            summary = mt.summary_metrics(session)
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Final value", f"${summary['final_value']:,.2f}", f"{summary['total_return_pct']:+.2f}%")
            s2.metric("Realized P&L", f"${summary['realized_pnl']:,.2f}")
            s3.metric("Sharpe ratio", f"{summary['sharpe_ratio']:.2f}")
            s4.metric("Max drawdown", f"{summary['max_drawdown_pct']:.2f}%")

            your_curve = pd.DataFrame(
                [{"date": d, "portfolio_value": v} for d, v in sorted(session["equity_curve"].items())]
            )
            bh_curve = mt.equal_weight_buy_and_hold_curve(session, price_series)

            fig = go.Figure()
            if len(your_curve):
                fig.add_trace(go.Scatter(x=your_curve["date"], y=your_curve["portfolio_value"], name="Your trades"))
            if len(bh_curve):
                fig.add_trace(go.Scatter(x=bh_curve["date"], y=bh_curve["portfolio_value"],
                                          name="Equal-weight Buy & Hold (reference only)", line=dict(dash="dot")))
            fig.update_layout(title="Portfolio value — your manual trades vs. doing nothing", height=400)
            st.plotly_chart(fig, width="stretch")

            with st.expander(f"Your trade history ({len(session['trade_log'])} trades)"):
                if session["trade_log"]:
                    tj = pd.DataFrame(session["trade_log"])
                    tj["date"] = pd.to_datetime(tj["date"]).dt.date
                    st.dataframe(tj, width="stretch", hide_index=True)
                else:
                    st.caption("No trades yet — use the Buy/Sell buttons above.")

    st.warning(
        "Educational tool. Recommendations are model output plus rule-based technical readings, "
        "not financial advice or a guaranteed prediction of future prices."
    )

# ============================ PORTFOLIO SIMULATOR ============================
with portfolio_tab:
    st.subheader("Multi-Stock Portfolio Simulator")
    st.caption(
        "Trades many stocks at once from one shared cash pool -- with order types, a flat broker "
        "commission, and diversification rules -- the way a StockTrak-style contest account works, "
        "rather than the single-stock simulator's all-in-one-stock approach."
    )
    if not st.session_state.trained_models:
        st.info("Train at least one model in 'Model Training & Comparison' first.")
    else:
        train_df, val_df, test_df, train_end, val_end = st.session_state.train_val_test
        labeled = st.session_state.labeled_data

        c1, c2 = st.columns(2)
        port_model_name = c1.selectbox("Model", list(st.session_state.trained_models.keys()), key="port_model")
        port_period = c2.selectbox("Evaluate on", ["Test set only", "Full history"], key="port_period")

        c3, c4, c5 = st.columns(3)
        port_cash = c3.number_input("Starting cash ($)", min_value=1000.0, value=100_000.0, step=1000.0, key="port_cash")
        port_order_type = c4.selectbox("Order type", config.ORDER_TYPES, key="port_order_type",
                                        help="MARKET fills immediately. LIMIT only buys if the price drops, or "
                                             "only sells if it rises. STOP only buys on a confirmed breakout, or "
                                             "only sells on a confirmed breakdown. Unfilled LIMIT/STOP orders "
                                             f"expire after {config.ORDER_EXPIRY_DAYS} trading days.")
        port_max_positions = c5.slider("Max concurrent positions", 1, 10, config.MAX_OPEN_POSITIONS, key="port_max_pos")

        c6, c7, c8 = st.columns(3)
        port_position_limit = c6.slider("Max % of portfolio per stock", 5, 100, int(config.MAX_POSITION_PCT * 100), key="port_poslimit") / 100
        port_commission = c7.number_input("Commission per trade ($)", min_value=0.0, value=config.COMMISSION_FLAT, step=1.0, key="port_comm")
        port_stop_pct = c8.slider("Protective stop-loss (%, 0 = off)", 0, 30, int(config.PROTECTIVE_STOP_LOSS_PCT * 100), key="port_stop") / 100

        cap_trades = st.checkbox("Cap total trades this period", key="port_cap_trades")
        max_trades_val = None
        if cap_trades:
            max_trades_val = int(st.number_input("Max trades", min_value=1, value=50, step=1, key="port_max_trades_val"))

        if st.button("Run portfolio simulation", type="primary", key="run_portfolio"):
            data_slice = test_df if port_period == "Test set only" else labeled
            model_info = st.session_state.trained_models[port_model_name]
            sig_table = build_full_signal_table(model_info, data_slice, port_model_name)
            if sig_table is None or len(sig_table) == 0:
                st.error("Not enough rows to build signals for this period/model.")
            else:
                res = run_portfolio_simulation(
                    sig_table, starting_cash=port_cash, order_type=port_order_type,
                    commission_flat=port_commission, max_position_pct=port_position_limit,
                    max_open_positions=port_max_positions, max_trades=max_trades_val,
                    protective_stop_loss_pct=port_stop_pct if port_stop_pct > 0 else None,
                )
                st.session_state["last_portfolio_result"] = {"res": res, "model": port_model_name}

        saved = st.session_state.get("last_portfolio_result")
        if saved:
            res = saved["res"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Final value", f"${res.final_value:,.2f}", f"{res.total_return_pct:+.2f}%")
            m2.metric("Sharpe ratio", f"{res.sharpe_ratio:.2f}")
            m3.metric("Max drawdown", f"{res.max_drawdown_pct:.2f}%")
            m4.metric("Trades", res.num_trades)
            m5.metric("Commission paid", f"${res.total_commission_paid:,.2f}")

            fig = go.Figure(go.Scatter(x=res.equity_curve["date"], y=res.equity_curve["portfolio_value"], name="Portfolio value"))
            fig.update_layout(title=f"Portfolio value over time -- {saved['model']}", height=400)
            st.plotly_chart(fig, width="stretch")

            if res.holdings_final:
                st.markdown("#### Open positions at end of period")
                st.dataframe(pd.DataFrame([{"Ticker": t, "Shares": round(s, 3)} for t, s in res.holdings_final.items()]),
                             width="stretch", hide_index=True)
            else:
                st.caption("No open positions at end of period.")

            with st.expander(f"Trade journal ({len(res.trades)} trades)"):
                if res.trades:
                    tj = pd.DataFrame([{
                        "Date": t.date.date(), "Ticker": t.ticker, "Action": t.action, "Order": t.order_type,
                        "Price": round(t.price, 2), "Shares": round(t.shares, 3), "Commission": round(t.commission, 2),
                        "Rationale": t.rationale,
                    } for t in res.trades])
                    st.dataframe(tj, width="stretch", hide_index=True)
                else:
                    st.caption("No trades were made.")

            st.caption(f"Protective stop-loss triggered {res.stopped_out_count} time(s) · "
                       f"{res.expired_orders_count} pending order(s) expired unfilled.")

# ============================ TRADING SIMULATOR ==============================
with simulator_tab:
    st.subheader("Single-Stock Trading Simulator")
    if not st.session_state.trained_models:
        st.info("Train at least one model in the 'Model Training & Comparison' tab first.")
    else:
        available_tickers = sorted(st.session_state.labeled_data["ticker"].unique())
        c1, c2, c3, c4 = st.columns(4)
        sim_ticker = c1.selectbox("Stock", available_tickers)
        sim_model_name = c2.selectbox("Model", list(st.session_state.trained_models.keys()))
        starting_cash = c3.number_input("Starting cash ($)", min_value=100.0, value=10_000.0, step=100.0)
        sim_period = c4.selectbox("Evaluate on", ["Test set only", "Full history"])

        if st.button("Run simulation", type="primary"):
            labeled = st.session_state.labeled_data
            train_df, val_df, test_df, train_end, val_end = st.session_state.train_val_test
            data_slice = test_df if sim_period == "Test set only" else labeled
            data_slice = data_slice[data_slice["ticker"] == sim_ticker].sort_values("date")

            model_info = st.session_state.trained_models[sim_model_name]

            if model_info["type"] == "tabular":
                X, _ = make_tabular_dataset(data_slice, config.FEATURE_COLUMNS)
                X_s = scale_tabular(X, model_info["scaler"])
                from src.models.xgboost_model import predict_xgboost
                preds = predict_xgboost(model_info["model"], X_s, model_info["inv_map"])
                meta = data_slice[["ticker", "date"]].reset_index(drop=True)
                sig_table = build_signal_table(meta, data_slice, preds, sim_model_name, model_info["version"])
            else:
                X_seq, y_seq, meta = create_sequences(data_slice, config.FEATURE_COLUMNS, seq_len=config.SEQUENCE_LENGTH)
                if len(X_seq) == 0:
                    st.error("Not enough rows in this period to build sequences (need > sequence length).")
                    st.stop()
                X_seq_s = scale_sequences(X_seq, model_info["scaler"])
                pred_enc = np.argmax(model_info["model"].predict(X_seq_s, verbose=0), axis=1)
                preds = decode_labels_from_keras(pred_enc)
                sig_table = build_signal_table(meta, data_slice, preds, sim_model_name, model_info["version"])

            one_ticker_sig = signals_for_ticker(sig_table, sim_ticker)
            sim_result = run_simulation(one_ticker_sig, starting_cash=starting_cash)

            st.session_state["last_signal_table"] = {
                "sig": one_ticker_sig, "sim": sim_result,
                "ticker": sim_ticker, "model": sim_model_name,
            }

        if st.session_state.get("last_signal_table"):
            saved = st.session_state["last_signal_table"]
            sig, sim_result = saved["sig"], saved["sim"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Final portfolio value", f"${sim_result.final_value:,.2f}",
                      f"{sim_result.total_return_pct:+.2f}%")
            m2.metric("Sharpe ratio", f"{sim_result.sharpe_ratio:.2f}")
            m3.metric("Max drawdown", f"{sim_result.max_drawdown_pct:.2f}%")
            m4.metric("Trades / Win rate", f"{sim_result.num_trades} / {sim_result.win_rate:.0f}%")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sig["date"], y=sig["price"], name="Price", line=dict(color="gray")))
            buys = sig[sig["signal"] == "BUY"]
            sells = sig[sig["signal"] == "SELL"]
            fig.add_trace(go.Scatter(x=buys["date"], y=buys["price"], mode="markers", name="BUY",
                                       marker=dict(color="green", symbol="triangle-up", size=10)))
            fig.add_trace(go.Scatter(x=sells["date"], y=sells["price"], mode="markers", name="SELL",
                                       marker=dict(color="red", symbol="triangle-down", size=10)))
            fig.update_layout(title=f"{saved['ticker']} — Signals from {saved['model']}", height=400)
            st.plotly_chart(fig, width="stretch")

            with st.expander("Trade history"):
                trades_df = pd.DataFrame([t.__dict__ for t in sim_result.trades])
                st.dataframe(trades_df, width="stretch")
