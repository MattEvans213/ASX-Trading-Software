"""
Configuration for the ASX contrarian backtesting system.

All adjustable parameters are defined here as dataclass fields so they can be
overridden from the CLI or programmatically without touching strategy logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StrategyConfig:
    """Parameters that define the contrarian strategy."""

    # --- Market-cap thresholds (AUD) ---
    small_cap_upper: float = 1_000_000_000       # < $1B
    mid_cap_upper: float = 7_000_000_000          # $1B – $7B
    # Anything above mid_cap_upper is large cap.

    # --- Position sizing ---
    stocks_per_sector: int = 3                    # Worst N performers per cap bucket
    # Portfolio weight is split equally across the 3 sectors, then equally
    # among stocks_per_sector within each sector.

    # --- Holding period ---
    hold_minutes_after_open: int = 5              # Sell 5 min after next open

    # --- Sector labels (do not change order without updating engine logic) ---
    sector_labels: List[str] = field(
        default_factory=lambda: ["small_cap", "mid_cap", "large_cap"]
    )


@dataclass
class BacktestConfig:
    """Parameters that control the backtest execution environment."""

    starting_capital: float = 100_000.0
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"

    # --- Costs ---
    transaction_cost_pct: float = 0.001           # 0.1 % per trade (buy + sell)
    slippage_pct: float = 0.0005                  # 0.05 % adverse fill

    # --- Liquidity filter ---
    min_avg_daily_volume: int = 50_000            # Shares; skip illiquid names
    min_price: float = 0.01                       # Skip sub-penny stocks


@dataclass
class DataConfig:
    """Parameters for data fetching."""

    # Top ASX tickers by market cap. yfinance uses ".AX" suffix for ASX.
    # This is a representative universe — extend as needed.
    tickers: List[str] = field(default_factory=lambda: [
        # Large-cap (typically >$7B)
        "BHP.AX", "CBA.AX", "CSL.AX", "NAB.AX", "WBC.AX",
        "ANZ.AX", "WES.AX", "MQG.AX", "FMG.AX", "WOW.AX",
        "TLS.AX", "RIO.AX", "ALL.AX", "GMG.AX", "TCL.AX",
        "STO.AX", "WDS.AX", "COL.AX", "QBE.AX", "SUN.AX",
        "ORG.AX", "IAG.AX", "REA.AX", "JHX.AX", "AMC.AX",
        "SOL.AX", "NCM.AX", "SHL.AX", "RHC.AX", "MIN.AX",
        # Mid-cap (typically $1B–$7B)
        "OZL.AX", "IGO.AX", "ALD.AX", "EVN.AX", "NST.AX",
        "ILU.AX", "SGP.AX", "GPT.AX", "MGR.AX", "VCX.AX",
        "BEN.AX", "BOQ.AX", "CGF.AX", "PPT.AX", "CPU.AX",
        "NHF.AX", "IPL.AX", "ORI.AX", "BSL.AX", "AWC.AX",
        "DXS.AX", "CHC.AX", "LLC.AX", "CWN.AX", "SGR.AX",
        "TPG.AX", "PMV.AX", "JBH.AX", "SUL.AX", "BXB.AX",
        # Small-cap (typically <$1B)
        "PLS.AX", "LYC.AX", "SYR.AX", "AVZ.AX", "LKE.AX",
        "NVX.AX", "VUL.AX", "DEG.AX", "GOR.AX", "RRL.AX",
        "WAF.AX", "SBM.AX", "RED.AX", "CMM.AX", "PRU.AX",
        "KGN.AX", "APX.AX", "TYR.AX", "BRN.AX", "ZIP.AX",
        "NXS.AX", "PNV.AX", "IMU.AX", "MFG.AX", "PDN.AX",
        "WHC.AX", "NHC.AX", "YAL.AX", "CRN.AX", "STN.AX",
    ])

    # yfinance download settings
    download_threads: int = 4
    chunk_size: int = 20          # Download tickers in chunks to avoid throttle

    # Cache downloaded data to avoid re-fetching
    cache_dir: str = "data_cache"


# ── Convenience: default instances ──────────────────────────────────────────

DEFAULT_STRATEGY = StrategyConfig()
DEFAULT_BACKTEST = BacktestConfig()
DEFAULT_DATA = DataConfig()
