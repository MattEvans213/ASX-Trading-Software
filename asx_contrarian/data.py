"""
Data fetching and validation for ASX stocks.

Uses yfinance as the primary data source.  yfinance pulls from Yahoo Finance,
which provides:
  - Daily OHLCV for ASX tickers (append ".AX")
  - Market-cap via the `info` dict (field "marketCap")

Alternatives considered:
  - Alpha Vantage: limited free tier, no native ASX market-cap
  - ASX API: no official free historical endpoint
  - Quandl/Nasdaq Data Link: ASX coverage is patchy
yfinance is the most practical free option for this use case.

When network access is unavailable (e.g. sandboxed environments), the module
falls back to deterministic synthetic data so the backtest pipeline can still
be exercised end-to-end.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .config import BacktestConfig, DataConfig, StrategyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic / fallback data (used when yfinance is unreachable)
# ---------------------------------------------------------------------------

# Approximate market caps (AUD) based on the intended classification in
# config.py.  These are rough mid-range values for each bucket — only used
# when live data cannot be fetched.
_FALLBACK_MARKET_CAPS: Dict[str, float] = {
    # Large-cap (>$7B)
    "BHP.AX": 200e9, "CBA.AX": 180e9, "CSL.AX": 130e9, "NAB.AX": 95e9,
    "WBC.AX": 80e9,  "ANZ.AX": 78e9,  "WES.AX": 60e9,  "MQG.AX": 75e9,
    "FMG.AX": 55e9,  "WOW.AX": 42e9,  "TLS.AX": 40e9,  "RIO.AX": 38e9,
    "ALL.AX": 20e9,  "GMG.AX": 35e9,  "TCL.AX": 30e9,  "STO.AX": 18e9,
    "WDS.AX": 30e9,  "COL.AX": 17e9,  "QBE.AX": 16e9,  "SUN.AX": 15e9,
    "ORG.AX": 12e9,  "IAG.AX": 11e9,  "REA.AX": 22e9,  "JHX.AX": 14e9,
    "AMC.AX": 13e9,  "SOL.AX": 10e9,  "NCM.AX": 15e9,  "SHL.AX": 12e9,
    "RHC.AX": 10e9,  "MIN.AX": 8e9,
    # Mid-cap ($1B–$7B)
    "OZL.AX": 5e9,   "IGO.AX": 4e9,   "ALD.AX": 3.5e9, "EVN.AX": 5.5e9,
    "NST.AX": 6e9,   "ILU.AX": 4e9,   "SGP.AX": 3e9,   "GPT.AX": 4.5e9,
    "MGR.AX": 3.5e9, "VCX.AX": 2.5e9, "BEN.AX": 5e9,   "BOQ.AX": 3e9,
    "CGF.AX": 2.5e9, "PPT.AX": 2e9,   "CPU.AX": 4e9,   "NHF.AX": 3e9,
    "IPL.AX": 4.5e9, "ORI.AX": 5e9,   "BSL.AX": 3e9,   "AWC.AX": 2.5e9,
    "DXS.AX": 4e9,   "CHC.AX": 3e9,   "LLC.AX": 5e9,   "CWN.AX": 4e9,
    "SGR.AX": 2.5e9, "TPG.AX": 5.5e9, "PMV.AX": 3e9,   "JBH.AX": 5e9,
    "SUL.AX": 2.5e9, "BXB.AX": 6e9,
    # Small-cap (<$1B)
    "PLS.AX": 800e6, "LYC.AX": 700e6, "SYR.AX": 300e6, "AVZ.AX": 200e6,
    "LKE.AX": 250e6, "NVX.AX": 350e6, "VUL.AX": 400e6, "DEG.AX": 500e6,
    "GOR.AX": 600e6, "RRL.AX": 700e6, "WAF.AX": 400e6, "SBM.AX": 350e6,
    "RED.AX": 200e6, "CMM.AX": 500e6, "PRU.AX": 450e6, "KGN.AX": 300e6,
    "APX.AX": 250e6, "TYR.AX": 400e6, "BRN.AX": 350e6, "ZIP.AX": 600e6,
    "NXS.AX": 150e6, "PNV.AX": 300e6, "IMU.AX": 200e6, "MFG.AX": 800e6,
    "PDN.AX": 500e6, "WHC.AX": 700e6, "NHC.AX": 600e6, "YAL.AX": 400e6,
    "CRN.AX": 250e6, "STN.AX": 300e6,
}


def _generate_synthetic_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data for offline testing.

    Uses a seeded random walk per ticker so results are reproducible.
    """
    rng = np.random.RandomState(42)
    dates = pd.bdate_range(start=start_date, end=end_date)

    arrays = {}
    for idx, ticker in enumerate(tickers):
        n = len(dates)
        # Start price between $1 and $50, seeded by ticker index
        start_price = 5.0 + (idx * 7) % 45
        # Daily returns: mean ~0, std ~1.5 %
        daily_ret = rng.normal(0.0002, 0.015, size=n)
        close = start_price * np.cumprod(1 + daily_ret)

        # Derive OHLV from close
        open_prices = close * (1 + rng.normal(0, 0.003, size=n))
        high = np.maximum(close, open_prices) * (1 + rng.uniform(0, 0.01, size=n))
        low = np.minimum(close, open_prices) * (1 - rng.uniform(0, 0.01, size=n))
        volume = rng.randint(50_000, 2_000_000, size=n).astype(float)

        arrays[(ticker, "Open")] = open_prices
        arrays[(ticker, "High")] = high
        arrays[(ticker, "Low")] = low
        arrays[(ticker, "Close")] = close
        arrays[(ticker, "Volume")] = volume

    df = pd.DataFrame(arrays, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ---------------------------------------------------------------------------
# Market-cap snapshot
# ---------------------------------------------------------------------------

def fetch_market_caps(tickers: list[str]) -> Dict[str, float]:
    """Return {ticker: market_cap_aud} using yfinance `info`.

    Tickers that fail or have no market-cap field are silently skipped.
    If *no* tickers succeed (e.g. no network), falls back to built-in
    approximate market caps so the backtest can still run.
    """
    caps: Dict[str, float] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            mc = info.get("marketCap")
            if mc and mc > 0:
                caps[t] = float(mc)
        except Exception as exc:
            logger.warning("Could not fetch market cap for %s: %s", t, exc)

    if caps:
        logger.info("Fetched market caps for %d / %d tickers", len(caps), len(tickers))
        return caps

    # Fallback: use built-in approximate market caps
    logger.warning(
        "Could not fetch any live market caps — falling back to synthetic data"
    )
    fallback = {t: _FALLBACK_MARKET_CAPS[t] for t in tickers if t in _FALLBACK_MARKET_CAPS}
    logger.info("Using fallback market caps for %d / %d tickers", len(fallback), len(tickers))
    return fallback


def classify_by_cap(
    market_caps: Dict[str, float],
    strat: StrategyConfig,
) -> Dict[str, list[str]]:
    """Bucket tickers into small / mid / large cap lists.

    Tickers whose market cap is zero or missing are excluded.
    """
    buckets: Dict[str, list[str]] = {label: [] for label in strat.sector_labels}
    for ticker, cap in market_caps.items():
        if cap < strat.small_cap_upper:
            buckets["small_cap"].append(ticker)
        elif cap < strat.mid_cap_upper:
            buckets["mid_cap"].append(ticker)
        else:
            buckets["large_cap"].append(ticker)
    for label, members in buckets.items():
        logger.info("  %s: %d tickers", label, len(members))
    return buckets


# ---------------------------------------------------------------------------
# Historical price data
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str) -> Path:
    return Path(cache_dir) / "asx_prices.parquet"


def fetch_price_data(
    tickers: list[str],
    data_cfg: DataConfig,
    bt_cfg: BacktestConfig,
) -> pd.DataFrame:
    """Download daily close & volume for all tickers, with disk caching.

    Returns a MultiIndex DataFrame with columns (Close, Volume) × tickers.
    Index is DatetimeIndex of trading days.

    Falls back to synthetic price data when the network is unavailable.
    """
    cache_file = _cache_path(data_cfg.cache_dir)
    if cache_file.exists():
        logger.info("Loading cached price data from %s", cache_file)
        return pd.read_parquet(cache_file)

    logger.info(
        "Downloading price data for %d tickers (%s → %s) …",
        len(tickers), bt_cfg.start_date, bt_cfg.end_date,
    )

    # Download in chunks to reduce throttling risk
    all_frames = []
    download_failed = False
    for i in range(0, len(tickers), data_cfg.chunk_size):
        chunk = tickers[i : i + data_cfg.chunk_size]
        logger.info("  chunk %d–%d …", i, i + len(chunk) - 1)
        try:
            df = yf.download(
                chunk,
                start=bt_cfg.start_date,
                end=bt_cfg.end_date,
                threads=data_cfg.download_threads,
                group_by="ticker",
                auto_adjust=True,
            )
            if df.empty:
                download_failed = True
                break
            all_frames.append(df)
        except Exception as exc:
            logger.warning("Download failed for chunk %d: %s", i, exc)
            download_failed = True
            break
        time.sleep(1)  # polite pause

    if download_failed or not all_frames:
        logger.warning(
            "Could not download price data — falling back to synthetic prices"
        )
        prices = _generate_synthetic_prices(tickers, bt_cfg.start_date, bt_cfg.end_date)
    else:
        prices = pd.concat(all_frames, axis=1)

    # Persist cache
    os.makedirs(data_cfg.cache_dir, exist_ok=True)
    prices.to_parquet(cache_file)
    logger.info("Cached price data to %s", cache_file)

    return prices


# ---------------------------------------------------------------------------
# Extracting close / volume per ticker from the MultiIndex DataFrame
# ---------------------------------------------------------------------------

def _extract_field(prices: pd.DataFrame, ticker: str, field: str) -> pd.Series:
    """Safely pull a single field series for a ticker from the prices frame."""
    try:
        if isinstance(prices.columns, pd.MultiIndex):
            return prices[(ticker, field)].dropna()
        # Single-ticker edge case
        return prices[field].dropna()
    except KeyError:
        return pd.Series(dtype=float)


def get_close_prices(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return DataFrame of close prices: rows=dates, cols=tickers."""
    series = {t: _extract_field(prices, t, "Close") for t in tickers}
    return pd.DataFrame(series)


def get_volumes(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Return DataFrame of daily volumes: rows=dates, cols=tickers."""
    series = {t: _extract_field(prices, t, "Volume") for t in tickers}
    return pd.DataFrame(series)


# ---------------------------------------------------------------------------
# Validation / filtering
# ---------------------------------------------------------------------------

def apply_liquidity_filter(
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    bt_cfg: BacktestConfig,
    window: int = 20,
) -> pd.DataFrame:
    """Return a boolean mask (same shape as closes): True = tradeable.

    A stock is tradeable on day *t* if:
      1. Close price >= min_price
      2. 20-day rolling average volume >= min_avg_daily_volume
      3. Close is not NaN
    """
    avg_vol = volumes.rolling(window, min_periods=5).mean()
    mask = (
        closes.notna()
        & (closes >= bt_cfg.min_price)
        & (avg_vol >= bt_cfg.min_avg_daily_volume)
    )
    n_filtered = (~mask).sum().sum()
    logger.info("Liquidity filter removed %d ticker-day observations", n_filtered)
    return mask


def compute_daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns."""
    return closes.pct_change()


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_all(
    data_cfg: DataConfig,
    bt_cfg: BacktestConfig,
    strat_cfg: StrategyConfig,
) -> Tuple[
    Dict[str, list[str]],   # cap_buckets
    pd.DataFrame,            # closes
    pd.DataFrame,            # daily_returns
    pd.DataFrame,            # tradeable_mask
]:
    """One-call entry point: fetch everything, validate, return clean data."""
    # 1. Market caps & classification
    market_caps = fetch_market_caps(data_cfg.tickers)
    cap_buckets = classify_by_cap(market_caps, strat_cfg)

    # Keep only tickers that got classified
    all_classified = [t for bucket in cap_buckets.values() for t in bucket]
    if not all_classified:
        raise RuntimeError("No tickers could be classified by market cap.")

    # 2. Price data
    prices = fetch_price_data(all_classified, data_cfg, bt_cfg)
    closes = get_close_prices(prices, all_classified)
    volumes = get_volumes(prices, all_classified)

    # 3. Validation
    tradeable = apply_liquidity_filter(closes, volumes, bt_cfg)
    daily_returns = compute_daily_returns(closes)

    return cap_buckets, closes, daily_returns, tradeable
