"""
Core backtesting engine for the ASX contrarian strategy.

Simulation loop:
  For each trading day t:
    1. Compute daily returns for all tradeable stocks.
    2. Within each cap bucket, pick the N worst performers.
    3. Allocate 1/3 of portfolio to each bucket, equally split among N stocks.
    4. "Buy" at the close of day t  (apply slippage + transaction cost).
    5. "Sell" at the *simulated* price 5 min after next open of day t+1.

Because yfinance only provides daily OHLCV, the 5-min-after-open exit is
approximated as the **open price of the next trading day** — this is the
closest available proxy.  A more precise simulation would require intraday
data (e.g., from a paid provider).

The engine records every trade and daily portfolio snapshots so downstream
modules can compute metrics and generate reports.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig, StrategyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Record of a single round-trip trade."""
    date_buy: pd.Timestamp
    date_sell: pd.Timestamp
    ticker: str
    sector: str
    buy_price: float
    sell_price: float
    shares: int
    gross_pnl: float        # before costs
    cost_buy: float
    cost_sell: float
    net_pnl: float           # after costs


@dataclass
class DailySnapshot:
    """Portfolio state at the end of a trading day."""
    date: pd.Timestamp
    portfolio_value: float
    cash: float
    small_cap_value: float
    mid_cap_value: float
    large_cap_value: float


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""
    trades: List[Trade]
    snapshots: List[DailySnapshot]
    strat_cfg: StrategyConfig
    bt_cfg: BacktestConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _worst_performers(
    returns_row: pd.Series,
    tickers: list[str],
    tradeable_row: pd.Series,
    n: int,
) -> list[str]:
    """Return up to *n* tickers with the lowest daily return that are tradeable.

    If fewer than *n* are available, return whatever is available (may be empty).
    """
    # Filter to tickers in this bucket that are tradeable and have a return
    valid = [
        t for t in tickers
        if t in returns_row.index
        and t in tradeable_row.index
        and tradeable_row.get(t, False)
        and pd.notna(returns_row.get(t))
    ]
    if not valid:
        return []
    sub = returns_row[valid].sort_values()
    return list(sub.head(n).index)


def _apply_slippage(price: float, direction: str, slippage_pct: float) -> float:
    """Adjust price for slippage: adverse fill for both buys and sells."""
    if direction == "buy":
        return price * (1 + slippage_pct)   # pay more
    return price * (1 - slippage_pct)        # receive less


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def run_backtest(
    cap_buckets: Dict[str, list[str]],
    closes: pd.DataFrame,
    opens: pd.DataFrame,
    daily_returns: pd.DataFrame,
    tradeable: pd.DataFrame,
    strat_cfg: StrategyConfig,
    bt_cfg: BacktestConfig,
) -> BacktestResult:
    """Execute the contrarian backtest over the supplied data.

    Parameters
    ----------
    cap_buckets : dict mapping sector label → list of tickers
    closes : DataFrame of daily close prices (rows=dates, cols=tickers)
    opens : DataFrame of daily open prices (used for next-day sell)
    daily_returns : DataFrame of daily returns (same shape as closes)
    tradeable : boolean DataFrame (same shape) — True if stock may be traded
    strat_cfg, bt_cfg : configuration objects
    """
    trading_days = closes.index.sort_values()
    cash = bt_cfg.starting_capital
    trades: List[Trade] = []
    snapshots: List[DailySnapshot] = []

    # Track per-sector invested value for snapshot breakdowns.
    # We realise positions daily, so sector value only lives intra-day.
    sector_cum_pnl = {s: 0.0 for s in strat_cfg.sector_labels}

    n = strat_cfg.stocks_per_sector
    total_slots = n * len(strat_cfg.sector_labels)  # e.g., 9

    for i, day in enumerate(trading_days):
        # We need a *next* trading day to sell into.
        if i >= len(trading_days) - 1:
            break

        next_day = trading_days[i + 1]

        # Skip if returns not available for this day
        if day not in daily_returns.index:
            continue
        ret_row = daily_returns.loc[day]
        trd_row = tradeable.loc[day] if day in tradeable.index else pd.Series(dtype=bool)

        # ── 1. Select worst performers per sector ─────────────────────
        selections: Dict[str, list[str]] = {}
        for sector in strat_cfg.sector_labels:
            bucket_tickers = cap_buckets.get(sector, [])
            picks = _worst_performers(ret_row, bucket_tickers, trd_row, n)
            selections[sector] = picks

        # Count how many actual positions we'll open
        actual_positions = sum(len(v) for v in selections.values())
        if actual_positions == 0:
            # Nothing tradeable today — just record snapshot
            snapshots.append(DailySnapshot(
                date=day,
                portfolio_value=cash,
                cash=cash,
                small_cap_value=0.0,
                mid_cap_value=0.0,
                large_cap_value=0.0,
            ))
            continue

        # ── 2. Allocate capital ───────────────────────────────────────
        # Equal weight across all sectors and stocks.
        # Each sector gets 1/3 of portfolio; each stock gets 1/(3*N).
        per_stock_alloc = cash / total_slots

        day_trades: List[Trade] = []

        for sector, picks in selections.items():
            for ticker in picks:
                buy_price_raw = closes.at[day, ticker] if ticker in closes.columns else np.nan
                sell_price_raw = opens.at[next_day, ticker] if ticker in opens.columns else np.nan

                if pd.isna(buy_price_raw) or pd.isna(sell_price_raw):
                    logger.debug("Missing price for %s on %s/%s, skipping", ticker, day, next_day)
                    continue
                if buy_price_raw <= 0:
                    continue

                buy_price = _apply_slippage(buy_price_raw, "buy", bt_cfg.slippage_pct)
                sell_price = _apply_slippage(sell_price_raw, "sell", bt_cfg.slippage_pct)

                shares = int(per_stock_alloc // buy_price)
                if shares <= 0:
                    continue

                cost_buy = shares * buy_price * bt_cfg.transaction_cost_pct
                cost_sell = shares * sell_price * bt_cfg.transaction_cost_pct
                gross_pnl = shares * (sell_price - buy_price)
                net_pnl = gross_pnl - cost_buy - cost_sell

                trade = Trade(
                    date_buy=day,
                    date_sell=next_day,
                    ticker=ticker,
                    sector=sector,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    shares=shares,
                    gross_pnl=gross_pnl,
                    cost_buy=cost_buy,
                    cost_sell=cost_sell,
                    net_pnl=net_pnl,
                )
                day_trades.append(trade)

        # ── 3. Update cash / record snapshot ──────────────────────────
        total_invested = sum(t.shares * t.buy_price + t.cost_buy for t in day_trades)
        total_proceeds = sum(t.shares * t.sell_price - t.cost_sell for t in day_trades)

        # After buying & selling same-day loop (overnight hold), cash changes
        cash = cash - total_invested + total_proceeds

        # Sector PnL for snapshot
        sector_value = {s: 0.0 for s in strat_cfg.sector_labels}
        for t in day_trades:
            sector_value[t.sector] += t.net_pnl
            sector_cum_pnl[t.sector] += t.net_pnl

        trades.extend(day_trades)

        snapshots.append(DailySnapshot(
            date=next_day,     # value is realised on sell day
            portfolio_value=cash,
            cash=cash,
            small_cap_value=sector_value.get("small_cap", 0.0),
            mid_cap_value=sector_value.get("mid_cap", 0.0),
            large_cap_value=sector_value.get("large_cap", 0.0),
        ))

    logger.info("Backtest complete: %d trades over %d days", len(trades), len(snapshots))
    return BacktestResult(
        trades=trades,
        snapshots=snapshots,
        strat_cfg=strat_cfg,
        bt_cfg=bt_cfg,
    )


# ---------------------------------------------------------------------------
# Convenience: extract open prices from multi-index frame
# ---------------------------------------------------------------------------

def get_open_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return DataFrame of open prices: rows=dates, cols=tickers."""
    series = {}
    for t in tickers:
        try:
            if isinstance(prices.columns, pd.MultiIndex):
                s = prices[(t, "Open")].dropna()
            else:
                s = prices["Open"].dropna()
            series[t] = s
        except KeyError:
            series[t] = pd.Series(dtype=float)
    return pd.DataFrame(series)
