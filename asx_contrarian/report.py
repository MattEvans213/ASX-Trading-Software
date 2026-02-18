"""
Reporting and visualisation for the ASX contrarian backtest.

Outputs:
  1. daily_portfolio.csv  — daily portfolio value with sector breakdown
  2. trades.csv           — every trade record
  3. summary.txt          — human-readable performance report
  4. equity_curve.png     — equity curve chart (overall + per sector)
  5. asx_contrarian_report.xlsx — combined Excel workbook (trades, daily, summary)
"""

import logging
import os
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from .engine import BacktestResult
from .metrics import Metrics, compute_overall_metrics, compute_sector_metrics

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# 1. CSV — daily portfolio value
# ---------------------------------------------------------------------------

def write_daily_csv(result: BacktestResult, output_dir: str) -> str:
    """Write daily portfolio snapshots to CSV. Returns file path."""
    dirp = _ensure_dir(output_dir)
    rows = [s.__dict__ for s in result.snapshots]
    df = pd.DataFrame(rows)
    path = dirp / "daily_portfolio.csv"
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))
    return str(path)


# ---------------------------------------------------------------------------
# 2. CSV — trade log
# ---------------------------------------------------------------------------

def write_trades_csv(result: BacktestResult, output_dir: str) -> str:
    """Write all trades to CSV. Returns file path."""
    dirp = _ensure_dir(output_dir)
    rows = [t.__dict__ for t in result.trades]
    df = pd.DataFrame(rows)
    path = dirp / "trades.csv"
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d trades)", path, len(df))
    return str(path)


# ---------------------------------------------------------------------------
# 3. Text — performance summary
# ---------------------------------------------------------------------------

def _format_metrics(m: Metrics) -> str:
    return (
        f"  Total Return:         {m.total_return_pct:+.2f} %\n"
        f"  Annualised Return:    {m.annualised_return_pct:+.2f} %\n"
        f"  Sharpe Ratio:         {m.sharpe_ratio:.3f}\n"
        f"  Max Drawdown:         {m.max_drawdown_pct:.2f} %\n"
        f"  Win Rate:             {m.win_rate_pct:.1f} %\n"
        f"  Number of Trades:     {m.num_trades}\n"
        f"  Total PnL:            ${m.total_pnl:,.2f}\n"
        f"  Avg PnL / Trade:      ${m.avg_pnl_per_trade:,.2f}\n"
        f"  Total Costs:          ${m.total_costs:,.2f}\n"
    )


def write_summary(result: BacktestResult, output_dir: str) -> str:
    """Write a human-readable performance report. Returns file path."""
    dirp = _ensure_dir(output_dir)
    overall = compute_overall_metrics(result)
    by_sector = compute_sector_metrics(result)

    lines = []
    lines.append("=" * 60)
    lines.append("ASX CONTRARIAN STRATEGY — BACKTEST REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Period:            {result.bt_cfg.start_date} → {result.bt_cfg.end_date}")
    lines.append(f"Starting Capital:  ${result.bt_cfg.starting_capital:,.2f}")
    lines.append(f"Transaction Cost:  {result.bt_cfg.transaction_cost_pct * 100:.2f} % per trade")
    lines.append(f"Slippage:          {result.bt_cfg.slippage_pct * 100:.3f} %")
    lines.append(f"Stocks per day:    {result.strat_cfg.stocks_per_sector}")
    lines.append(f"Universe:          Small cap only (< ${result.strat_cfg.small_cap_upper / 1e9:.1f}B)")
    lines.append(f"Allocation:        {100 / result.strat_cfg.stocks_per_sector:.0f} % per stock")
    lines.append("")
    lines.append("-" * 60)
    lines.append("OVERALL PERFORMANCE")
    lines.append("-" * 60)
    lines.append(_format_metrics(overall))

    for sector, m in by_sector.items():
        lines.append("-" * 60)
        lines.append(f"SECTOR: {sector.upper().replace('_', ' ')}")
        lines.append("-" * 60)
        lines.append(_format_metrics(m))

    lines.append("=" * 60)

    text = "\n".join(lines)
    path = dirp / "summary.txt"
    path.write_text(text)
    logger.info("Wrote %s", path)

    # Also print to stdout
    print(text)

    return str(path)


# ---------------------------------------------------------------------------
# 4. Chart — equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(result: BacktestResult, output_dir: str) -> str:
    """Generate equity curve PNG with overall + sector sub-curves."""
    dirp = _ensure_dir(output_dir)

    snap_df = pd.DataFrame([s.__dict__ for s in result.snapshots])
    if snap_df.empty:
        logger.warning("No snapshots to plot.")
        return ""

    snap_df["date"] = pd.to_datetime(snap_df["date"])
    snap_df.sort_values("date", inplace=True)

    starting = result.bt_cfg.starting_capital

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))

    ax1.plot(snap_df["date"], snap_df["portfolio_value"], linewidth=1.2, color="#1f77b4")
    ax1.axhline(starting, linestyle="--", color="grey", linewidth=0.7, label="Starting Capital")
    ax1.set_title("Small-Cap Contrarian — Portfolio Equity Curve", fontsize=13)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_xlabel("Date")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = dirp / "equity_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path)
    return str(path)


# ---------------------------------------------------------------------------
# 5. Excel — combined multi-sheet workbook
# ---------------------------------------------------------------------------

def write_excel_report(result: BacktestResult, output_dir: str) -> str:
    """Write a multi-sheet Excel workbook with trades, snapshots, and summary."""
    dirp = _ensure_dir(output_dir)
    path = dirp / "asx_contrarian_report.xlsx"

    trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
    daily_df = pd.DataFrame([s.__dict__ for s in result.snapshots])

    overall = compute_overall_metrics(result)
    by_sector = compute_sector_metrics(result)
    all_metrics = [overall] + list(by_sector.values())
    summary_df = pd.DataFrame([m.__dict__ for m in all_metrics])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        trades_df.to_excel(writer, sheet_name="Trades", index=False)
        daily_df.to_excel(writer, sheet_name="Daily Portfolio", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    logger.info("Wrote %s (3 sheets)", path)
    return str(path)


# ---------------------------------------------------------------------------
# All-in-one
# ---------------------------------------------------------------------------

def generate_full_report(result: BacktestResult, output_dir: str = "output") -> None:
    """Generate all outputs: CSVs, summary text, equity chart, and Excel workbook."""
    write_daily_csv(result, output_dir)
    write_trades_csv(result, output_dir)
    write_summary(result, output_dir)
    write_excel_report(result, output_dir)
    plot_equity_curve(result, output_dir)
    logger.info("All reports written to %s/", output_dir)
