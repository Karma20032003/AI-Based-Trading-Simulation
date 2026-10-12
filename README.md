# AI-Based Multi-Stock Trading Strategy

LSTM · GRU · Transformer-GRU · XGBoost — with interactive single-asset
trading simulation, model comparison, and periodic retraining.

Research/education tool for historical analysis and **virtual** trading.
It does **not** perform real-money trading or live broker execution.

## What's inside

```
trading_project/
├── app.py                  # Streamlit application (main UI)
├── train_pipeline.py       # CLI: full pipeline, train & save all models
├── config.py                # All tunable settings in one place
├── requirements.txt
├── Dockerfile
├── src/
│   ├── data_collection.py       # Yahoo Finance download + caching
│   ├── data_validation.py       # Data quality checks + cleaning
│   ├── database.py               # SQLite/PostgreSQL persistence (SQLAlchemy)
│   ├── feature_engineering.py   # SMA/EMA/RSI/MACD/volatility
│   ├── labeling.py               # Triple-barrier labeling
│   ├── sequence_generation.py   # Chronological splits + sequence building
│   ├── signals.py                 # BUY/SELL/HOLD signal table construction
│   ├── recommendation.py        # Current BUY/HOLD/SELL: model probabilities + indicators
│   ├── simulator.py               # Single-stock virtual trading simulator
│   ├── benchmark.py               # Buy-and-hold benchmark
│   ├── utils.py                    # Scaling + model persistence/versioning
│   └── models/
│       ├── xgboost_model.py
│       ├── lstm_model.py
│       ├── gru_model.py
│       └── transformer_gru_model.py
├── tests/
│   └── smoke_test.py        # End-to-end test using synthetic data (no internet needed)
├── data/                     # raw/, processed/, parquet/ (created automatically)
└── models/                   # trained model artifacts + metadata (created automatically)
```

## Quick start (local)

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
#    (or, for the exact tested versions: pip install -r requirements-lock.txt)

# 3. (Optional) verify everything works with synthetic data, no internet required
python tests/smoke_test.py

# 4. Run the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

**In the app:** use the sidebar to pick a ticker universe (a 4-stock smoke
test set is a good first run) and click **"Fetch & Process Data"**. Then go
to **Model Training & Comparison** to train models, and **Trading
Simulator** to run a virtual trade and compare it to Buy-and-Hold.

## Current recommendations

The **Recommendations** tab gives a BUY / HOLD / SELL for the latest available
date, per stock, with the reasoning shown:

- **Model view:** probability that the price rises 3%, falls 3%, or does
  neither within the next 10 trading days (the triple-barrier outcome). This is
  not a price forecast.
- **Technical view:** RSI, MACD (including crossovers), SMA trend and EMA
  12/26 readings, plus a volatility note.
- **Combined score:** 60% model + 40% indicators. At or above +0.15 is BUY, at
  or below -0.15 is SELL, otherwise HOLD. These weights are transparent
  heuristics you can change in `src/recommendation.py`.
- **Price range:** a 1-sigma statistical range after 10 days from current
  volatility. It shows how far the price could plausibly move, not which way.

Without a trained model the tab still works using indicators only.

The **Virtual Trading Desk** also uses the latest available market-data date
when a session starts, so its initial recommendations match the latest data
used in the Recommendations tab. Use the day controls to review historical
recommendations or return to the latest date; manual Buy/Sell and portfolio
tracking remain available.

## StockTrak-style portfolio simulation

Looking at how [StockTrak](https://www.stocktrak.com/) actually runs its
contests (a shared multi-asset cash account, order types, flat commissions,
and diversification limits) surfaced things the original single-stock
simulator didn't have. The Portfolio Simulator adds these controls:

### Portfolio Simulator
Trades **many stocks at once from one shared cash pool**, instead of one
stock in isolation:
- **Order types:** MARKET fills immediately; LIMIT only buys if the price
  drops to a target (or only sells if it rises to one); STOP only buys on a
  confirmed breakout (or only sells on a confirmed breakdown). Unfilled
  LIMIT/STOP orders expire after `ORDER_EXPIRY_DAYS` (5 by default).
- **Commission:** a flat $ fee per trade (`COMMISSION_FLAT`), the way
  StockTrak charges a broker commission.
- **Diversification rules:** `MAX_POSITION_PCT` caps any one stock as a % of
  the portfolio (default 25%), and `MAX_OPEN_POSITIONS` caps how many stocks
  can be held at once -- both force spreading risk instead of going all-in
  on one signal.
- **Protective stop-loss:** an account-level safety stop, independent of
  the model's own SELL signal, matching StockTrak's broker-enforced stops.
- **Trade journal:** every fill is logged with a plain-language rationale
  (indicator readings + model signal), similar to StockTrak's trade diary.

The app also includes recommendations, model training and comparison, a
virtual trading desk, and a single-stock trading simulator.

## Command-line pipeline (no UI)

Useful for scripted/scheduled training runs:

```bash
# 4-stock smoke test, quick epochs
python train_pipeline.py --level smoke --epochs 5

# 10-stock development set
python train_pipeline.py --level dev --epochs 20

# Explicit tickers and date range
python train_pipeline.py --tickers AAPL MSFT NVDA --start 2018-01-01 --epochs 25

# Train only specific models
python train_pipeline.py --level smoke --models xgboost lstm
```

Trained models are saved to `models/` with an incrementing version
(`xgboost_v1.pkl`, `lstm_v1.keras`, ...) plus a `_meta.json` file recording
metrics and training time, so retraining never overwrites prior versions.

## Scaling the stock universe

Per the project's staged approach, don't jump straight to the full S&P 500:

```
4 stocks (smoke test) → 10 stocks (dev) → 100 stocks → 250 stocks → selected S&P 500
```

Use `--tickers` with your own list to test at 100/250-stock scale; verify
timing, memory, and database/Parquet performance at each level before
scaling further.

## Database

Defaults to a local SQLite file (`trading.db`, zero configuration). To use
PostgreSQL instead, set an environment variable before running:

```bash
export TRADING_DB_URL="postgresql+psycopg2://user:password@localhost:5432/trading_db"
```

You'll also need `psycopg2-binary` installed (`pip install psycopg2-binary`).

## Deployment

### Option A — Streamlit Community Cloud (easiest)
1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io, connect the repo, set the main file to
   `app.py`.
3. Streamlit Cloud installs `requirements.txt` automatically and deploys.

### Option B — Docker (any host)
```bash
docker build -t ai-trading-app .
docker run -p 8501:8501 ai-trading-app
```
Then open http://localhost:8501.

### Option C — Any VM / server
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Notes & limitations

- **Data collection requires internet access** (Yahoo Finance via `yfinance`).
  The included `tests/smoke_test.py` validates the entire pipeline (features →
  labeling → sequences → all 4 models → signals → simulator → buy-and-hold)
  using synthetic data so you can verify the install without network access.
- Model quality depends heavily on tickers, date range, and epochs chosen —
  the defaults are tuned for a quick, working demo, not for chasing accuracy.
- The trading simulator is a simplified long-only, fully-invested-or-flat
  model with a small flat trading-fee assumption. It is intended for
  research and education, not as a production execution engine.
- No real signal here should be treated as financial advice or a guaranteed
  prediction of future prices.
