"""
Buy-and-hold benchmark, computed with the same metric definitions as the
AI-strategy simulator so the two are directly comparable.
"""
import numpy as np
import pandas as pd

from config import DEFAULT_STARTING_CASH
from src.simulator import _max_drawdown, _sharpe_ratio, SimulationResult, Trade


def run_buy_and_hold(price_df: pd.DataFrame, starting_cash: float = DEFAULT_STARTING_CASH) -> SimulationResult:
    """
    price_df: single-ticker DataFrame sorted by date with a 'price' column.
    Buys at the first available price and holds until the last.
    """
    df = price_df.sort_values("date").reset_index(drop=True)
    entry_price = df["price"].iloc[0]
    shares = starting_cash / entry_price

    equity_curve = pd.DataFrame({
        "date": df["date"],
        "portfolio_value": shares * df["price"],
    })

    final_value = float(equity_curve["portfolio_value"].iloc[-1])
    total_return_pct = (final_value / starting_cash - 1) * 100

    n_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days if len(df) > 1 else 1
    years = max(n_days / 365.25, 1 / 365.25)
    annualized_return_pct = ((final_value / starting_cash) ** (1 / years) - 1) * 100

    daily_returns = equity_curve["portfolio_value"].pct_change().dropna().values
    sharpe = _sharpe_ratio(daily_returns)
    max_dd = _max_drawdown(equity_curve["portfolio_value"].values)

    trades = [
        Trade(df["date"].iloc[0], "BUY", entry_price, shares, 0.0),
    ]

    return SimulationResult(
        equity_curve=equity_curve,
        trades=trades,
        final_value=final_value,
        total_return_pct=float(total_return_pct),
        annualized_return_pct=float(annualized_return_pct),
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=100.0 if final_value > starting_cash else 0.0,
        num_trades=1,
        realized_pnl=float(final_value - starting_cash),
    )
