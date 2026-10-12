"""
Structured persistent storage.

Uses SQLAlchemy so the same code works against SQLite (default, zero-config,
good for local/dev/deployment) or PostgreSQL (set TRADING_DB_URL env var to a
postgresql+psycopg2://... URL for production).
"""
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, UniqueConstraint, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker

from config import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Stock(Base):
    __tablename__ = "stocks"
    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)


class MarketData(Base):
    __tablename__ = "market_data"
    id = Column(Integer, primary_key=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(Float)
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True)
    model_type = Column(String, nullable=False)   # xgboost / lstm / gru / transformer_gru
    version = Column(String, nullable=False)       # v1, v2, ...
    trained_at = Column(DateTime, default=datetime.utcnow)
    metrics_json = Column(String)  # serialized JSON metrics
    file_path = Column(String)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(DateTime, index=True, nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    signal = Column(String)   # BUY / SELL / HOLD
    price = Column(Float)


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"))
    date = Column(DateTime)
    action = Column(String)   # BUY / SELL
    price = Column(Float)
    shares = Column(Float)
    cash_after = Column(Float)


class Simulation(Base):
    __tablename__ = "simulations"
    id = Column(Integer, primary_key=True)
    ticker = Column(String)
    model_type = Column(String)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    starting_cash = Column(Float)
    final_value = Column(Float)
    total_return_pct = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown_pct = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()


def upsert_market_data(df):
    """Insert market data rows, skipping ones that already exist (ticker+date)."""
    session = get_session()
    try:
        existing = {
            (r.ticker, r.date) for r in session.query(MarketData.ticker, MarketData.date)
        }
        new_rows = []
        for row in df.itertuples(index=False):
            key = (row.ticker, row.date.to_pydatetime() if hasattr(row.date, "to_pydatetime") else row.date)
            if key in existing:
                continue
            new_rows.append(
                MarketData(
                    ticker=row.ticker,
                    date=row.date,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    adj_close=row.adj_close,
                    volume=row.volume,
                )
            )
        session.bulk_save_objects(new_rows)
        session.commit()
        return len(new_rows)
    finally:
        session.close()
