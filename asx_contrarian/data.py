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
# Market-cap snapshot
# ---------------------------------------------------------------------------

def fetch_market_caps(tickers: list[str]) -> Dict[str, float]:
    """Return {ticker: market_cap_aud} using yfinance `info`.

    Tickers that fail or have no market-cap field are silently skipped.
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
    logger.info("Fetched market caps for %d / %d tickers", len(caps), len(tickers))
    return caps


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
    for i in range(0, len(tickers), data_cfg.chunk_size):
        chunk = tickers[i : i + data_cfg.chunk_size]
        logger.info("  chunk %d–%d …", i, i + len(chunk) - 1)
        df = yf.download(
            chunk,
            start=bt_cfg.start_date,
            end=bt_cfg.end_date,
            threads=data_cfg.download_threads,
            group_by="ticker",
            auto_adjust=True,
        )
        all_frames.append(df)
        time.sleep(1)  # polite pause

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
