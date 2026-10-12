"""
Single-stock virtual trading simulator.

Rules (simple, long-only, fully-invested-or-flat, easy to reason about):
- BUY signal and currently flat  -> buy as many shares as cash allows
- SELL signal and currently long -> sell entire position
- HOLD, or a signal that doesn't change state -> do nothing

This is intentionally simple and transparent -- it is a research/education
simulator, not a production execution engine.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from config import DEFAULT_STARTING_CASH, TRADE_FEE_PCT


@dataclass
class Trade:
    date: pd.Timestamp
    action: str
    price: float
    shares: float
    cash_after: float


@dataclass
class SimulationResult:
    equity_curve: pd.DataFrame           # date, portfolio_value
    trades: List[Trade]
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    realized_pnl: float


def _max_drawdown(equity: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min() * 100)


def _sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def run_simulation(
    signal_df: pd.DataFrame,
    starting_cash: float = DEFAULT_STARTING_CASH,
    fee_pct: float = TRADE_FEE_PCT,
) -> SimulationResult:
    """
    signal_df must be sorted by date, for a SINGLE ticker, with columns
    ['date', 'price', 'signal'] where signal in {'BUY','SELL','HOLD'}.
    """
    df = signal_df.sort_values("date").reset_index(drop=True)

    cash = starting_cash
    shares = 0.0
    entry_price = None
    trades: List[Trade] = []
    equity = []
    realized_pnl = 0.0
    wins = 0
    closed_trades = 0

    for _, row in df.iterrows():
        price = row["price"]
        signal = row["signal"]

        if signal == "BUY" and shares == 0 and cash > 0:
            shares_to_buy = (cash * (1 - fee_pct)) / price
            cash -= shares_to_buy * price * (1 + fee_pct)
            shares = shares_to_buy
            entry_price = price
            trades.append(Trade(row["date"], "BUY", price, shares, cash))

        elif signal == "SELL" and shares > 0:
            proceeds = shares * price * (1 - fee_pct)
            cash += proceeds
            pnl = (price - entry_price) * shares
            realized_pnl += pnl
            closed_trades += 1
            if pnl > 0:
                wins += 1
            trades.append(Trade(row["date"], "SELL", price, shares, cash))
            shares = 0.0
            entry_price = None

        portfolio_value = cash + shares * price
        equity.append({"date": row["date"], "portfolio_value": portfolio_value})

    # close any open position at the last available price for reporting
    if shares > 0:
        last_price = df.iloc[-1]["price"]
        pnl = (last_price - entry_price) * shares
        realized_pnl += pnl
        closed_trades += 1
        if pnl > 0:
            wins += 1

    equity_curve = pd.DataFrame(equity)
    final_value = equity_curve["portfolio_value"].iloc[-1] if len(equity_curve) else starting_cash
    total_return_pct = (final_value / starting_cash - 1) * 100

    n_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days if len(df) > 1 else 1
    years = max(n_days / 365.25, 1 / 365.25)
    annualized_return_pct = ((final_value / starting_cash) ** (1 / years) - 1) * 100

    daily_returns = equity_curve["portfolio_value"].pct_change().dropna().values
    sharpe = _sharpe_ratio(daily_returns)
    max_dd = _max_drawdown(equity_curve["portfolio_value"].values) if len(equity_curve) else 0.0
    win_rate = (wins / closed_trades * 100) if closed_trades else 0.0

    return SimulationResult(
        equity_curve=equity_curve,
        trades=trades,
        final_value=float(final_value),
        total_return_pct=float(total_return_pct),
        annualized_return_pct=float(annualized_return_pct),
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=float(win_rate),
        num_trades=len(trades),
        realized_pnl=float(realized_pnl),
    )
