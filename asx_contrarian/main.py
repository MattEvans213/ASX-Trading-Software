"""
Entry point for the ASX contrarian backtesting system.

Usage:
    python -m asx_contrarian.main [OPTIONS]

All parameters have sensible defaults (see config.py).  Override via CLI flags.
"""

import argparse
import logging
import sys

from .config import BacktestConfig, DataConfig, StrategyConfig
from .data import load_all, fetch_price_data
from .engine import get_open_prices, run_backtest
from .report import generate_full_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ASX Contrarian Strategy Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Strategy params
    p.add_argument("--small-cap-upper", type=float, default=1e9,
                   help="Upper bound for small cap (AUD)")
    p.add_argument("--mid-cap-upper", type=float, default=7e9,
                   help="Upper bound for mid cap (AUD)")
    p.add_argument("--stocks-per-sector", type=int, default=3,
                   help="Worst-N performers to buy per sector")

    # Backtest params
    p.add_argument("--capital", type=float, default=100_000,
                   help="Starting capital (AUD)")
    p.add_argument("--start-date", type=str, default="2020-01-01")
    p.add_argument("--end-date", type=str, default="2024-12-31")
    p.add_argument("--txn-cost", type=float, default=0.001,
                   help="Transaction cost as decimal (0.001 = 0.1%%)")
    p.add_argument("--slippage", type=float, default=0.0005,
                   help="Slippage as decimal (0.0005 = 0.05%%)")
    p.add_argument("--min-volume", type=int, default=50_000,
                   help="Minimum 20-day avg volume for liquidity filter")
    p.add_argument("--min-price", type=float, default=0.01,
                   help="Minimum stock price")

    # Output
    p.add_argument("--output-dir", type=str, default="output",
                   help="Directory for reports and CSVs")

    # Misc
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable debug logging")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("asx_contrarian")

    # ── Build config objects from CLI args ─────────────────────────────
    strat_cfg = StrategyConfig(
        small_cap_upper=args.small_cap_upper,
        mid_cap_upper=args.mid_cap_upper,
        stocks_per_sector=args.stocks_per_sector,
    )
    bt_cfg = BacktestConfig(
        starting_capital=args.capital,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost_pct=args.txn_cost,
        slippage_pct=args.slippage,
        min_avg_daily_volume=args.min_volume,
        min_price=args.min_price,
    )
    data_cfg = DataConfig()

    logger.info("Starting ASX contrarian backtest …")
    logger.info("  Period: %s → %s", bt_cfg.start_date, bt_cfg.end_date)
    logger.info("  Capital: $%,.0f", bt_cfg.starting_capital)

    # ── 1. Load & validate data ───────────────────────────────────────
    logger.info("Phase 1: Fetching data …")
    cap_buckets, closes, daily_returns, tradeable = load_all(
        data_cfg, bt_cfg, strat_cfg,
    )

    # Also need open prices for next-day sell simulation
    all_tickers = [t for bucket in cap_buckets.values() for t in bucket]
    prices_raw = fetch_price_data(all_tickers, data_cfg, bt_cfg)
    opens = get_open_prices(prices_raw, all_tickers)

    # ── 2. Run backtest ───────────────────────────────────────────────
    logger.info("Phase 2: Running backtest engine …")
    result = run_backtest(
        cap_buckets=cap_buckets,
        closes=closes,
        opens=opens,
        daily_returns=daily_returns,
        tradeable=tradeable,
        strat_cfg=strat_cfg,
        bt_cfg=bt_cfg,
    )

    # ── 3. Generate reports ───────────────────────────────────────────
    logger.info("Phase 3: Generating reports …")
    generate_full_report(result, args.output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
