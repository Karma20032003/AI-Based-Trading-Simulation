"""
Multi-stock portfolio simulator, blending StockTrak's core contest mechanics
into the project:

  - A single shared cash pool traded across MANY stocks at once (StockTrak's
    "virtual brokerage account"), not the one-stock-at-a-time simulator in
    src/simulator.py (which is kept as-is for quick single-asset checks).
  - Order types: MARKET (fills immediately), LIMIT (only buys if the price
    drops to a target, or only sells if it rises to a target), and STOP
    (only buys on a confirmed breakout above a level, or sells on a confirmed
    breakdown below one). Pending LIMIT/STOP orders expire after
    ORDER_EXPIRY_DAYS if never filled, matching StockTrak's "Day/GTC" orders.
  - A flat broker commission per trade (StockTrak charges a fixed fee).
  - Diversification rules: MAX_POSITION_PCT caps any one stock as a % of the
    portfolio, and MAX_OPEN_POSITIONS caps how many stocks can be held at
    once -- StockTrak enforces the same kind of position limits.
  - A broker-enforced protective stop-loss on every open position, separate
    from whatever the model's own SELL signal says.
  - A trade journal: every fill is stored with a plain-language rationale,
    similar to StockTrak's trade-diary / trade-rationale requirement.

This is still a simplified, educational model -- not a production execution
engine (see the caveats in the README).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from src.simulator import _max_drawdown, _sharpe_ratio


@dataclass
class PendingOrder:
    ticker: str
    side: str          # BUY or SELL
    order_type: str    # LIMIT or STOP (MARKET never becomes pending -- it fills same day)
    trigger_price: float
    placed_date: pd.Timestamp
    expires_date: pd.Timestamp
    shares: Optional[float] = None   # only set for SELL orders (exact shares to sell)


@dataclass
class PortfolioTrade:
    date: pd.Timestamp
    ticker: str
    action: str         # BUY / SELL
    order_type: str
    price: float
    shares: float
    commission: float
    cash_after: float
    rationale: str


@dataclass
class PortfolioResult:
    equity_curve: pd.DataFrame                  # date, portfolio_value
    trades: List[PortfolioTrade]
    holdings_final: Dict[str, float]
    final_value: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    num_trades: int
    total_commission_paid: float
    realized_pnl: float
    stopped_out_count: int
    expired_orders_count: int


def _make_rationale(action: str, ticker: str, order_type: str, row: pd.Series) -> str:
    bits = [f"{action} {ticker} ({order_type})"]
    if "rsi_14" in row and pd.notna(row["rsi_14"]):
        rsi = row["rsi_14"]
        if rsi < 30:
            bits.append(f"RSI {rsi:.0f} oversold")
        elif rsi > 70:
            bits.append(f"RSI {rsi:.0f} overbought")
    if "macd_hist" in row and pd.notna(row["macd_hist"]):
        bits.append("MACD bullish" if row["macd_hist"] > 0 else "MACD bearish")
    if "signal" in row:
        bits.append(f"model signal={row['signal']}")
    return "; ".join(bits)


def _build_price_matrix(signal_df: pd.DataFrame) -> pd.DataFrame:
    """Wide date x ticker price matrix, forward-filled, for daily valuation."""
    wide = signal_df.pivot_table(index="date", columns="ticker", values="price", aggfunc="last")
    wide = wide.sort_index().ffill()
    return wide


def run_portfolio_simulation(
    signal_df: pd.DataFrame,
    starting_cash: float = config.DEFAULT_STARTING_CASH,
    order_type: str = config.DEFAULT_ORDER_TYPE,
    commission_flat: float = config.COMMISSION_FLAT,
    commission_pct: float = config.COMMISSION_PCT,
    max_position_pct: float = config.MAX_POSITION_PCT,
    max_open_positions: int = config.MAX_OPEN_POSITIONS,
    max_trades: Optional[int] = config.MAX_TRADES_PER_PERIOD,
    protective_stop_loss_pct: Optional[float] = config.PROTECTIVE_STOP_LOSS_PCT,
    order_expiry_days: int = config.ORDER_EXPIRY_DAYS,
    limit_offset_pct: float = config.LIMIT_OFFSET_PCT,
    stop_offset_pct: float = config.STOP_OFFSET_PCT,
) -> PortfolioResult:
    """
    signal_df: multi-ticker table with columns
      [ticker, date, price, signal] (+ optional rsi_14/macd_hist for the
      trade journal), one row per ticker per date, signal in
      {'BUY','SELL','HOLD'}.
    """
    df = signal_df.sort_values(["date", "ticker"]).reset_index(drop=True)
    price_matrix = _build_price_matrix(df)
    dates = price_matrix.index

    cash = starting_cash
    holdings: Dict[str, Dict[str, float]] = {}   # ticker -> {shares, entry_price}
    pending: List[PendingOrder] = []
    trades: List[PortfolioTrade] = []
    equity_rows = []
    realized_pnl = 0.0
    total_commission = 0.0
    wins = 0
    closed_trades = 0
    stopped_out_count = 0
    expired_count = 0

    def portfolio_value(today_prices: pd.Series) -> float:
        v = cash
        for t, pos in holdings.items():
            px = today_prices.get(t, pos["entry_price"])
            v += pos["shares"] * px
        return v

    def commission_for(trade_value: float) -> float:
        return commission_flat + commission_pct * trade_value

    def do_buy(ticker, price, date, o_type, row=None):
        nonlocal cash, total_commission
        if ticker in holdings or len(holdings) >= max_open_positions:
            return
        if max_trades is not None and len(trades) >= max_trades:
            return
        pv = portfolio_value(price_matrix.loc[date])
        budget = min(cash, pv * max_position_pct)
        if budget <= commission_flat or budget < 1:
            return
        shares = (budget - commission_flat) / (price * (1 + commission_pct))
        if shares <= 0:
            return
        trade_value = shares * price
        commission = commission_for(trade_value)
        cash -= trade_value + commission
        holdings[ticker] = {"shares": shares, "entry_price": price, "entry_date": date}
        total_commission += commission
        rationale = _make_rationale("BUY", ticker, o_type, row if row is not None else pd.Series({"signal": "BUY"}))
        trades.append(PortfolioTrade(date, ticker, "BUY", o_type, price, shares, commission, cash, rationale))

    def do_sell(ticker, price, date, o_type, forced_reason=None, row=None):
        nonlocal cash, realized_pnl, total_commission, wins, closed_trades
        if ticker not in holdings:
            return
        if max_trades is not None and len(trades) >= max_trades:
            return
        pos = holdings.pop(ticker)
        trade_value = pos["shares"] * price
        commission = commission_for(trade_value)
        cash += trade_value - commission
        total_commission += commission
        pnl = (price - pos["entry_price"]) * pos["shares"] - commission
        realized_pnl += pnl
        closed_trades += 1
        if pnl > 0:
            wins += 1
        rationale = forced_reason or _make_rationale("SELL", ticker, o_type, row if row is not None else pd.Series({"signal": "SELL"}))
        trades.append(PortfolioTrade(date, ticker, "SELL", o_type, price, pos["shares"], commission, cash, rationale))

    by_date = {d: g for d, g in df.groupby("date")}

    for date in dates:
        today_prices = price_matrix.loc[date]

        # 1) Broker-enforced protective stop-loss, checked before anything else
        if protective_stop_loss_pct:
            for t in list(holdings.keys()):
                px = today_prices.get(t)
                pos = holdings[t]
                if pd.notna(px) and px <= pos["entry_price"] * (1 - protective_stop_loss_pct):
                    do_sell(t, px, date, "STOP (protective)", forced_reason=f"Protective stop-loss hit on {t} at {px:.2f}")
                    stopped_out_count += 1

        # 2) Fill or expire pending LIMIT/STOP orders
        still_pending = []
        for order in pending:
            px = today_prices.get(order.ticker)
            if pd.isna(px):
                still_pending.append(order)
                continue
            filled = False
            if order.order_type == "LIMIT":
                if order.side == "BUY" and px <= order.trigger_price:
                    do_buy(order.ticker, px, date, "LIMIT")
                    filled = True
                elif order.side == "SELL" and px >= order.trigger_price:
                    do_sell(order.ticker, px, date, "LIMIT")
                    filled = True
            elif order.order_type == "STOP":
                if order.side == "BUY" and px >= order.trigger_price:
                    do_buy(order.ticker, px, date, "STOP")
                    filled = True
                elif order.side == "SELL" and px <= order.trigger_price:
                    do_sell(order.ticker, px, date, "STOP")
                    filled = True
            if not filled:
                if date >= order.expires_date:
                    expired_count += 1
                else:
                    still_pending.append(order)
        pending = still_pending

        # 3) Process today's new signals
        todays_rows = by_date.get(date)
        if todays_rows is not None:
            for _, row in todays_rows.iterrows():
                ticker, price, sig = row["ticker"], row["price"], row["signal"]
                if pd.isna(price):
                    continue

                if sig == "BUY" and ticker not in holdings:
                    if order_type == "MARKET":
                        do_buy(ticker, price, date, "MARKET", row)
                    elif order_type == "LIMIT":
                        pending.append(PendingOrder(ticker, "BUY", "LIMIT", price * (1 - limit_offset_pct),
                                                     date, date + pd.tseries.offsets.BDay(order_expiry_days)))
                    elif order_type == "STOP":
                        pending.append(PendingOrder(ticker, "BUY", "STOP", price * (1 + stop_offset_pct),
                                                     date, date + pd.tseries.offsets.BDay(order_expiry_days)))

                elif sig == "SELL" and ticker in holdings:
                    if order_type == "MARKET":
                        do_sell(ticker, price, date, "MARKET", row=row)
                    elif order_type == "LIMIT":
                        pending.append(PendingOrder(ticker, "SELL", "LIMIT", price * (1 + limit_offset_pct),
                                                     date, date + pd.tseries.offsets.BDay(order_expiry_days)))
                    elif order_type == "STOP":
                        pending.append(PendingOrder(ticker, "SELL", "STOP", price * (1 - stop_offset_pct),
                                                     date, date + pd.tseries.offsets.BDay(order_expiry_days)))

        equity_rows.append({"date": date, "portfolio_value": portfolio_value(today_prices)})

    # close any remaining open positions at the last available price for reporting
    last_prices = price_matrix.iloc[-1]
    for t in list(holdings.keys()):
        px = last_prices.get(t, holdings[t]["entry_price"])
        pnl = (px - holdings[t]["entry_price"]) * holdings[t]["shares"]
        realized_pnl += pnl
        closed_trades += 1
        if pnl > 0:
            wins += 1

    equity_curve = pd.DataFrame(equity_rows)
    final_value = float(equity_curve["portfolio_value"].iloc[-1]) if len(equity_curve) else starting_cash
    total_return_pct = (final_value / starting_cash - 1) * 100

    n_days = (dates[-1] - dates[0]).days if len(dates) > 1 else 1
    years = max(n_days / 365.25, 1 / 365.25)
    annualized_return_pct = ((final_value / starting_cash) ** (1 / years) - 1) * 100

    daily_returns = equity_curve["portfolio_value"].pct_change().dropna().values
    sharpe = _sharpe_ratio(daily_returns)
    max_dd = _max_drawdown(equity_curve["portfolio_value"].values) if len(equity_curve) else 0.0
    win_rate = (wins / closed_trades * 100) if closed_trades else 0.0

    return PortfolioResult(
        equity_curve=equity_curve,
        trades=trades,
        holdings_final={t: h["shares"] for t, h in holdings.items()},
        final_value=final_value,
        total_return_pct=float(total_return_pct),
        annualized_return_pct=float(annualized_return_pct),
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=float(win_rate),
        num_trades=len(trades),
        total_commission_paid=float(total_commission),
        realized_pnl=float(realized_pnl),
        stopped_out_count=stopped_out_count,
        expired_orders_count=expired_count,
    )


def run_leaderboard(
    signal_tables: Dict[str, pd.DataFrame],
    starting_cash: float = config.DEFAULT_STARTING_CASH,
    **kwargs,
) -> pd.DataFrame:
    """
    StockTrak-style contest leaderboard: run the portfolio simulator once
    per entrant (e.g. one per trained model) on the SAME universe, period and
    starting cash, and rank entrants by return -- mirroring a class/contest
    ranking table.

    signal_tables: {entrant_name: multi-ticker signal_df}
    Returns a ranked DataFrame.
    """
    rows = []
    for name, sig_df in signal_tables.items():
        result = run_portfolio_simulation(sig_df, starting_cash=starting_cash, **kwargs)
        rows.append({
            "Entrant": name,
            "Final Value": result.final_value,
            "Total Return %": result.total_return_pct,
            "Sharpe Ratio": result.sharpe_ratio,
            "Max Drawdown %": result.max_drawdown_pct,
            "Win Rate %": result.win_rate,
            "Trades": result.num_trades,
            "Commission Paid": result.total_commission_paid,
            "Stopped Out": result.stopped_out_count,
        })
    board = pd.DataFrame(rows).sort_values("Total Return %", ascending=False).reset_index(drop=True)
    board.insert(0, "Rank", range(1, len(board) + 1))
    return board
