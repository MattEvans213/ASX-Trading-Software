"""
Performance metrics for the ASX contrarian backtest.

All calculations assume:
  - 252 trading days per year (ASX standard).
  - Risk-free rate of 4.0 % annualised (adjustable), roughly in line with
    the RBA cash rate during 2020-2024.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .engine import BacktestResult, Trade

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04  # annualised


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    """Holds computed performance metrics."""
    label: str                  # "overall" or a sector name
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    total_pnl: float
    avg_pnl_per_trade: float
    total_costs: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equity_curve(snapshots_df: pd.DataFrame, value_col: str) -> pd.Series:
    """Return cumulative equity series from snapshot dataframe."""
    return snapshots_df.set_index("date")[value_col]


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0–1)."""
    if equity.empty or (equity <= 0).all():
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(abs(dd.min()))


def _sharpe(daily_returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    """Annualised Sharpe ratio from a series of daily returns."""
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free / TRADING_DAYS_PER_YEAR
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_overall_metrics(result: BacktestResult) -> Metrics:
    """Compute performance metrics for the entire portfolio."""
    snap_df = pd.DataFrame([s.__dict__ for s in result.snapshots])
    if snap_df.empty:
        return _empty_metrics("overall")

    equity = snap_df["portfolio_value"]
    start_val = result.bt_cfg.starting_capital
    end_val = equity.iloc[-1]
    total_ret = (end_val - start_val) / start_val

    n_days = len(equity)
    years = n_days / TRADING_DAYS_PER_YEAR if n_days > 0 else 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

    daily_rets = equity.pct_change().dropna()

    trades = result.trades
    wins = [t for t in trades if t.net_pnl > 0]
    win_rate = len(wins) / len(trades) * 100 if trades else 0.0
    total_pnl = sum(t.net_pnl for t in trades)
    total_costs = sum(t.cost_buy + t.cost_sell for t in trades)

    return Metrics(
        label="overall",
        total_return_pct=total_ret * 100,
        annualised_return_pct=ann_ret * 100,
        sharpe_ratio=_sharpe(daily_rets),
        max_drawdown_pct=_max_drawdown(equity) * 100,
        win_rate_pct=win_rate,
        num_trades=len(trades),
        total_pnl=total_pnl,
        avg_pnl_per_trade=total_pnl / len(trades) if trades else 0.0,
        total_costs=total_costs,
    )


def compute_sector_metrics(
    result: BacktestResult,
) -> Dict[str, Metrics]:
    """Compute performance metrics broken down by sector."""
    sector_trades: Dict[str, List[Trade]] = {
        s: [] for s in result.strat_cfg.sector_labels
    }
    for t in result.trades:
        sector_trades[t.sector].append(t)

    out: Dict[str, Metrics] = {}
    for sector, trades in sector_trades.items():
        if not trades:
            out[sector] = _empty_metrics(sector)
            continue

        # Build a pseudo-equity curve from cumulative PnL of this sector.
        # Start with 1/3 of capital allocated to this sector.
        sector_capital = result.bt_cfg.starting_capital / len(result.strat_cfg.sector_labels)
        pnl_by_date: Dict[pd.Timestamp, float] = {}
        for t in trades:
            pnl_by_date[t.date_sell] = pnl_by_date.get(t.date_sell, 0.0) + t.net_pnl

        dates_sorted = sorted(pnl_by_date.keys())
        cum = sector_capital
        eq_values = []
        for d in dates_sorted:
            cum += pnl_by_date[d]
            eq_values.append(cum)
        equity = pd.Series(eq_values, index=dates_sorted)

        total_ret = (equity.iloc[-1] - sector_capital) / sector_capital
        n_days = len(equity)
        years = n_days / TRADING_DAYS_PER_YEAR if n_days > 0 else 1
        ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0.0

        daily_rets = equity.pct_change().dropna()
        wins = [t for t in trades if t.net_pnl > 0]
        total_pnl = sum(t.net_pnl for t in trades)
        total_costs = sum(t.cost_buy + t.cost_sell for t in trades)

        out[sector] = Metrics(
            label=sector,
            total_return_pct=total_ret * 100,
            annualised_return_pct=ann_ret * 100,
            sharpe_ratio=_sharpe(daily_rets),
            max_drawdown_pct=_max_drawdown(equity) * 100,
            win_rate_pct=len(wins) / len(trades) * 100 if trades else 0.0,
            num_trades=len(trades),
            total_pnl=total_pnl,
            avg_pnl_per_trade=total_pnl / len(trades) if trades else 0.0,
            total_costs=total_costs,
        )
    return out


def _empty_metrics(label: str) -> Metrics:
    return Metrics(
        label=label,
        total_return_pct=0.0,
        annualised_return_pct=0.0,
        sharpe_ratio=0.0,
        max_drawdown_pct=0.0,
        win_rate_pct=0.0,
        num_trades=0,
        total_pnl=0.0,
        avg_pnl_per_trade=0.0,
        total_costs=0.0,
    )
