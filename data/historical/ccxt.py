from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import ccxt
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

@dataclass
class OHLCVParams:
    """Parameters for fetching OHLCV data.

    Attributes
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USDT').
    start_ms : int | None
        Start time as Unix timestamp in milliseconds, or None for no limit.
    end_ms : int | None
        End time as Unix timestamp in milliseconds, or None for no limit.
    timeframe : str
        Timeframe for candlesticks (e.g., '1d', '1h').
    limit : int
        Maximum number of candles to fetch per request.
    exchanges : Iterable[str] | None
        List of exchange IDs to fetch data from, or None for default set.
    """

    symbol: str
    start_ms: int | None
    end_ms: int | None
    timeframe: str
    limit: int
    exchanges: Iterable[str] | None = None


def fetch_historical_data(
    params: OHLCVParams
) -> pd.DataFrame | None:
    """
    Aggregate OHLCV data for a symbol across multiple exchanges using ccxt.

    Returns a volume-weighted average for OHLC columns and total volume.
    """
    exchange_list = list(params.exchanges or ["binance", "kraken", "coinbase", "bitstamp"])
    exchange_suffix = "-".join(exchange_list)
    cache_dir = Path(__file__).parent.parent / "cached"

    # Create cache key that includes timeframe and time range
    # Using hash to handle time range uniquely while keeping filename manageable
    start_str = (
        params.start_ms if params.start_ms is not None else "none"
    )
    end_str = (
        params.end_ms if params.end_ms is not None else "none"
    )
    time_range_str = f"{start_str}_{end_str}"
    time_range_hash = hashlib.sha256(time_range_str.encode()).hexdigest()[:8]
    cache_filename = f"{params.symbol.replace('/', '')}_{exchange_suffix}_{params.timeframe}_{time_range_hash}_ohlcv.csv"
    cache_path = cache_dir / cache_filename

    if cache_path.exists():
        logger.info("Loading cached data for %s from %s (timeframe=%s, range_hash=%s)",
                    params.symbol, exchange_suffix, params.timeframe, time_range_hash)
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    logger.info("Fetching historical data for %s from exchanges: %s",
                params.symbol,
                exchange_suffix)
    logger.debug("Parameters: timeframe=%s, start=%s, end=%s",
                 params.timeframe,
                 params.start_ms,
                 params.end_ms)

    frames: list[pd.DataFrame] = []
    for exchange_id in exchange_list:
        try:
            logger.info("Fetching data from %s...", exchange_id)
            exchange = _build_exchange(exchange_id)
            ohlcv = _fetch_ohlcv_paginated(exchange, params, params.start_ms, params.end_ms)
        except Exception as exc:
            logger.warning("Skipping %s for %s: %s", exchange_id, params.symbol, exc)
            continue

        if not ohlcv:
            logger.warning("No data returned from %s", exchange_id)
            continue

        logger.info("Retrieved %d candles from %s", len(ohlcv), exchange_id)

        df = pd.DataFrame(ohlcv, columns=["Datetime", "Open", "High", "Low", "Close", "Volume"])
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms", utc=True)
        df = df.set_index("Datetime")
        df.index.name = "Datetime"
        df = df.astype(float)
        frames.append(df)

    if not frames:
        logger.error("No data collected from any exchange for %s", params.symbol)
        return None

    logger.info("Aggregating data from %d exchanges", len(frames))
    combined = pd.concat(frames, axis=0)
    aggregated = _aggregate_ohlcv(combined)

    logger.info("Saving %d aggregated candles to cache: %s", len(aggregated), cache_path.name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(cache_path)
    logger.debug("Cache file saved at: %s", cache_path)
    return aggregated


def _build_exchange(exchange_id: str) -> ccxt.Exchange:
    logger.debug("Building exchange connection: %s", exchange_id)
    # Validate the exchange ID before attempting to instantiate it to avoid AttributeError
    if exchange_id not in getattr(ccxt, "exchanges", []):
        logger.error("Invalid exchange ID provided: %s. Available exchanges: %s", exchange_id, ", ".join(getattr(ccxt, "exchanges", [])))
        raise ValueError(f"Invalid exchange ID: {exchange_id}")
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})
    exchange.load_markets()
    return exchange


def _fetch_ohlcv_paginated(
    exchange: ccxt.Exchange,
    params: OHLCVParams,
    start_ms: int | None,
    end_ms: int | None
) -> list[list[float]]:
    all_ohlcv: list[list[float]] = []
    since = start_ms
    page_count = 0

    while True:
        batch = exchange.fetch_ohlcv(
            params.symbol,
            timeframe=params.timeframe,
            since=since,
            limit=params.limit
        )
        if not batch:
            break

        if end_ms is not None:
            batch = [row for row in batch if row[0] <= end_ms]
            if not batch:
                break

        all_ohlcv.extend(batch)
        page_count += 1
        since = batch[-1][0] + 1

        if page_count % 10 == 0:
            logger.debug("Fetched %d candles so far from %s...", len(all_ohlcv), exchange.id)

        if len(batch) < params.limit:
            break

        time.sleep(exchange.rateLimit / 1000.0)

    logger.debug("Completed pagination for %s: %d total candles", exchange.id, len(all_ohlcv))
    return all_ohlcv


def _aggregate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    logger.debug("Aggregating %d rows into volume-weighted averages", len(df))
    grouped = df.groupby(df.index)
    volume = grouped["Volume"].sum()

    def _weighted_avg(series_name: str) -> pd.Series:
        values = grouped[series_name]
        weighted = (df[series_name] * df["Volume"]).groupby(df.index).sum()
        avg = weighted / volume.replace(0, pd.NA)
        avg = avg.fillna(values.mean())
        return avg

    aggregated = pd.DataFrame(
        {
            "Open": _weighted_avg("Open"),
            "High": _weighted_avg("High"),
            "Low": _weighted_avg("Low"),
            "Close": _weighted_avg("Close"),
            "Volume": volume,
        }
    )

    aggregated = aggregated.sort_index()
    return aggregated
