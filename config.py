"""
Central configuration for the AI Multi-Stock Trading project.
Edit these values to change project-wide behavior.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, PARQUET_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Database (SQLite by default -- swap for a PostgreSQL URL in production,
# e.g. "postgresql+psycopg2://user:password@localhost:5432/trading_db")
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get(
    "TRADING_DB_URL", f"sqlite:///{os.path.join(BASE_DIR, 'trading.db')}"
)

# ---------------------------------------------------------------------------
# Stock universe (start small -- scale up only after each level is verified)
# ---------------------------------------------------------------------------
DEFAULT_TICKERS_SMOKE = ["AAPL", "MSFT", "GOOGL", "AMZN"]          # 4 stocks
DEFAULT_TICKERS_DEV = DEFAULT_TICKERS_SMOKE + [
    "NVDA", "META", "TSLA", "JPM", "V", "UNH",
]                                                                   # 10 stocks

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = None  # None => today

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
SMA_WINDOWS = [10, 20, 50]
EMA_WINDOWS = [12, 26]
RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLATILITY_WINDOW = 20

FEATURE_COLUMNS = [
    "return_1d",
    "sma_10", "sma_20", "sma_50",
    "ema_12", "ema_26",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "volatility_20",
]

# ---------------------------------------------------------------------------
# Triple-barrier labeling
# ---------------------------------------------------------------------------
PROFIT_TARGET = 0.03   # +3%
STOP_LOSS = -0.03      # -3%
TIME_LIMIT_DAYS = 10   # bars

LABEL_MAP = {-1: "SELL", 0: "HOLD", 1: "BUY"}

# ---------------------------------------------------------------------------
# Sequence generation
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH = 30
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# remainder is test

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Trading simulator (single-stock, original)
# ---------------------------------------------------------------------------
DEFAULT_STARTING_CASH = 10_000.0
TRADE_FEE_PCT = 0.0005  # 5 bps per trade, simple flat cost model

# ---------------------------------------------------------------------------
# Portfolio simulator (StockTrak-style: many positions, order types,
# commissions, diversification rules)
# ---------------------------------------------------------------------------
ORDER_TYPES = ["MARKET", "LIMIT", "STOP"]
DEFAULT_ORDER_TYPE = "MARKET"
LIMIT_OFFSET_PCT = 0.01    # a BUY limit is placed 1% below the signal price ("only buy if it gets cheaper")
STOP_OFFSET_PCT = 0.01     # a BUY stop is placed 1% above the signal price ("only buy on confirmed breakout")
ORDER_EXPIRY_DAYS = 5      # a pending limit/stop order expires (is cancelled) if unfilled after N trading days

COMMISSION_FLAT = 5.00     # $ per trade, StockTrak-style flat broker commission
COMMISSION_PCT = 0.0       # optional additional % of trade value

MAX_POSITION_PCT = 0.25       # diversification rule: no single position > 25% of portfolio value
MAX_OPEN_POSITIONS = 8        # diversification rule: at most N concurrent holdings
MAX_TRADES_PER_PERIOD = None  # optional cap on total trades over the simulation period (None = unlimited)
PROTECTIVE_STOP_LOSS_PCT = 0.08  # broker-enforced safety stop, independent of the model's own SELL signal
