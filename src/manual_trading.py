"""
Manual / virtual trading desk.

Unlike src/simulator.py and src/portfolio_simulator.py (which execute every
BUY/SELL signal automatically), NOTHING here executes a trade on its own.
The AI's role is limited to producing a recommendation (via
src/recommendation.py) for the ticker on the currently-viewed date; a human
must explicitly click Buy or Sell for anything to happen to the portfolio.

The session walks forward day by day through the chosen date range so that,
at every point, the recommendation shown only ever uses data up to and
including that day -- never future information.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def build_price_series(feat_all: pd.DataFrame, ticker: str) -> pd.Series:
    """Date-indexed price series for one ticker, for fast 'price as of date' lookups."""
    g = feat_all[feat_all["ticker"] == ticker].sort_values("date")
    price_col = "adj_close" if "adj_close" in g.columns else "close"
    return pd.Series(g[price_col].values, index=pd.DatetimeIndex(g["date"]))


def price_as_of(price_series: pd.Series, date) -> Optional[float]:
    """Last known price on or before `date` (handles any gaps gracefully)."""
    if len(price_series) == 0:
        return None
    try:
        val = price_series.asof(date)
    except KeyError:
        return None
    return None if pd.isna(val) else float(val)


def open_positions_table(
    session: dict, feat_all: pd.DataFrame, date
) -> pd.DataFrame:
    """Build a mark-to-market P&L table for positions at the selected date."""
    columns = [
        "Symbol", "Qty", "Price paid", "Last price", "Day's chg",
        "P/L", "P/L %", "Market value",
    ]
    rows = []

    for ticker, shares in sorted(session["positions"].items()):
        ticker_data = feat_all[
            (feat_all["ticker"] == ticker) & (feat_all["date"] <= date)
        ].sort_values("date")
        price_col = "adj_close" if "adj_close" in ticker_data.columns else "close"
        prices = ticker_data[price_col].dropna()
        cost_basis = session["cost_basis"].get(ticker, 0.0)
        average_cost = cost_basis / shares if shares > 0 else None

        if prices.empty:
            last_price = day_change = unrealized_pnl = pnl_pct = market_value = None
        else:
            last_price = float(prices.iloc[-1])
            day_change = (
                last_price - float(prices.iloc[-2]) if len(prices) > 1 else None
            )
            market_value = shares * last_price
            unrealized_pnl = market_value - cost_basis
            pnl_pct = unrealized_pnl / cost_basis * 100 if cost_basis else None

        rows.append({
            "Symbol": ticker,
            "Qty": shares,
            "Price paid": average_cost,
            "Last price": last_price,
            "Day's chg": day_change,
            "P/L": unrealized_pnl,
            "P/L %": pnl_pct,
            "Market value": market_value,
        })

    return pd.DataFrame(rows, columns=columns)


def init_session(
    tickers: List[str],
    dates: List[pd.Timestamp],
    starting_cash: float,
    start_at_latest: bool = False,
) -> dict:
    sorted_dates = sorted(dates)
    return {
        "tickers": tickers,
        "dates": sorted_dates,
        "idx": len(sorted_dates) - 1 if start_at_latest else 0,
        "cash": float(starting_cash),
        "starting_cash": float(starting_cash),
        "positions": {},     # ticker -> shares held
        "cost_basis": {},    # ticker -> total $ paid for currently-open shares
        "trade_log": [],     # list of dicts, one per USER-executed trade
        "equity_curve": {},  # date -> portfolio value, recorded as the session progresses
    }


def current_date(session: dict) -> pd.Timestamp:
    return session["dates"][session["idx"]]


def is_last_day(session: dict) -> bool:
    return session["idx"] >= len(session["dates"]) - 1


def portfolio_value(session: dict, price_series_by_ticker: Dict[str, pd.Series], date) -> float:
    value = session["cash"]
    for ticker, shares in session["positions"].items():
        price = price_as_of(price_series_by_ticker[ticker], date)
        if price is not None:
            value += shares * price
    return value


def record_equity_point(session: dict, price_series_by_ticker: Dict[str, pd.Series]):
    date = current_date(session)
    session["equity_curve"][date] = portfolio_value(session, price_series_by_ticker, date)


def execute_buy(session: dict, ticker: str, price: float, dollar_amount: float) -> Optional[str]:
    """User-initiated BUY. Returns an error string, or None on success."""
    dollar_amount = min(dollar_amount, session["cash"])
    if dollar_amount <= 0 or price <= 0:
        return "Not enough cash available to buy."
    shares = dollar_amount / price
    session["cash"] -= dollar_amount
    session["positions"][ticker] = session["positions"].get(ticker, 0.0) + shares
    session["cost_basis"][ticker] = session["cost_basis"].get(ticker, 0.0) + dollar_amount
    session["trade_log"].append({
        "date": current_date(session), "ticker": ticker, "action": "BUY",
        "price": price, "shares": shares, "amount": dollar_amount, "cash_after": session["cash"],
    })
    return None


def execute_sell(session: dict, ticker: str, price: float, shares_to_sell: Optional[float] = None) -> Optional[str]:
    """User-initiated SELL. Sells all held shares if shares_to_sell is None. Returns an error string, or None."""
    held = session["positions"].get(ticker, 0.0)
    if held <= 0:
        return f"No open position in {ticker} to sell."
    shares_to_sell = held if shares_to_sell is None else min(shares_to_sell, held)
    if shares_to_sell <= 0:
        return "Nothing to sell."

    avg_cost = session["cost_basis"].get(ticker, 0.0) / held if held > 0 else price
    proceeds = shares_to_sell * price
    realized_pnl = (price - avg_cost) * shares_to_sell

    session["cash"] += proceeds
    session["positions"][ticker] = held - shares_to_sell
    session["cost_basis"][ticker] = session["cost_basis"].get(ticker, 0.0) - avg_cost * shares_to_sell
    if session["positions"][ticker] <= 1e-9:
        del session["positions"][ticker]
        session["cost_basis"].pop(ticker, None)

    session["trade_log"].append({
        "date": current_date(session), "ticker": ticker, "action": "SELL",
        "price": price, "shares": shares_to_sell, "amount": proceeds, "cash_after": session["cash"],
        "realized_pnl": realized_pnl,
    })
    return None


def advance(session: dict, n_days: int = 1):
    session["idx"] = min(session["idx"] + n_days, len(session["dates"]) - 1)


def rewind(session: dict, n_days: int = 1):
    session["idx"] = max(session["idx"] - n_days, 0)


def equal_weight_buy_and_hold_curve(
    session: dict, price_series_by_ticker: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Reference-only benchmark: split starting cash equally across all session
    tickers on day 1 and hold. This is NEVER executed for the user -- it is
    shown purely so they can see how their own manual trades compare to
    doing nothing.
    """
    tickers = session["tickers"]
    if not tickers:
        return pd.DataFrame(columns=["date", "portfolio_value"])

    per_ticker_cash = session["starting_cash"] / len(tickers)
    first_date = session["dates"][0]
    shares = {}
    for t in tickers:
        p0 = price_as_of(price_series_by_ticker[t], first_date)
        shares[t] = (per_ticker_cash / p0) if p0 else 0.0

    rows = []
    for d in session["dates"][: session["idx"] + 1]:
        value = sum(
            shares[t] * (price_as_of(price_series_by_ticker[t], d) or 0.0) for t in tickers
        )
        rows.append({"date": d, "portfolio_value": value})
    return pd.DataFrame(rows)


def summary_metrics(session: dict) -> dict:
    dates = session["dates"][: session["idx"] + 1]
    curve = [session["equity_curve"][d] for d in dates if d in session["equity_curve"]]
    final_value = curve[-1] if curve else session["starting_cash"]
    total_return_pct = (final_value / session["starting_cash"] - 1) * 100

    realized_pnl = sum(t.get("realized_pnl", 0.0) for t in session["trade_log"] if t["action"] == "SELL")
    sells = [t for t in session["trade_log"] if t["action"] == "SELL"]
    wins = sum(1 for t in sells if t.get("realized_pnl", 0.0) > 0)
    win_rate = (wins / len(sells) * 100) if sells else 0.0

    if len(curve) > 1:
        returns = np.diff(curve) / np.array(curve[:-1])
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else 0.0
        running_max = np.maximum.accumulate(curve)
        max_dd = float(((np.array(curve) - running_max) / running_max).min() * 100)
    else:
        sharpe, max_dd = 0.0, 0.0

    return {
        "final_value": final_value, "total_return_pct": total_return_pct,
        "realized_pnl": realized_pnl, "num_trades": len(session["trade_log"]),
        "win_rate": win_rate, "sharpe_ratio": sharpe, "max_drawdown_pct": max_dd,
    }
