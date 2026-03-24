from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import ccxt
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

from models.events import StoredCryptoEvent
from utils import ensure_directory, get_workspace_root

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


def fetch_historical_data(params: OHLCVParams) -> pd.DataFrame | None:
    """Fetch and store OHLCV data for a symbol across multiple exchanges.

    Stores individual candles as Parquet file with timestamp-based structure,
    matching the format of streamed crypto data for consistency.

    Parameters
    ----------
    params : OHLCVParams
        Fetching parameters including symbol, timeframe, exchanges.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with stored events, or None if no data collected.
    """
    exchange_list = list(params.exchanges or ["binance", "kraken", "coinbase", "bitstamp"])

    data_root = get_workspace_root() / "data"
    storage_dir = data_root / "historical_ohlcv"
    ensure_directory(storage_dir)

    logger.info("Fetching historical OHLCV data for %s from exchanges: %s",
                params.symbol,
                ", ".join(exchange_list))
    logger.debug("Parameters: timeframe=%s, start=%s, end=%s",
                 params.timeframe,
                 params.start_ms,
                 params.end_ms)

    all_events: list[StoredCryptoEvent] = []
    ts_ingest = datetime.now(UTC)

    for exchange_id in exchange_list:
        try:
            logger.info("Fetching data from %s...", exchange_id)
            exchange = _build_exchange(exchange_id)
            ohlcv = _fetch_ohlcv_paginated(
                exchange, params, params.start_ms, params.end_ms
            )
        except Exception as exc:
            logger.warning("Skipping %s for %s: %s", exchange_id, params.symbol, exc)
            continue

        if not ohlcv:
            logger.warning("No data returned from %s", exchange_id)
            continue

        logger.info("Retrieved %d candles from %s", len(ohlcv), exchange_id)

        # Convert each candle to a StoredCryptoEvent
        for candle in ohlcv:
            timestamp_ms = int(candle[0])
            ts_event = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)

            event = StoredCryptoEvent(
                ts_event=ts_event,
                ts_ingest=ts_ingest,
                source=exchange_id,
                event_type="ohlcv",
                symbol=params.symbol,
                timeframe=params.timeframe,
                data={
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                },
            )
            all_events.append(event)

    if not all_events:
        logger.error("No data collected from any exchange for %s", params.symbol)
        return None

    # Count unique exchanges in the data
    num_exchanges = len({event.source for event in all_events})
    logger.info("Aggregating data from %d exchanges", num_exchanges)

    # Convert events to DataFrame
    events_data = [event.model_dump() for event in all_events]
    df = pd.DataFrame(events_data)

    # Group by date for daily file organization (same as RTDS storage)
    df["ts_date"] = pd.to_datetime(df["ts_event"]).dt.strftime("%Y%m%d")
    grouped = df.groupby("ts_date")

    clean_symbol = params.symbol.replace("/", "_").upper()

    # Save each day to its own file
    for ts_date, day_df in grouped:
        filename = f"{ts_date}_{clean_symbol}_{params.timeframe}.parquet"
        output_path = storage_dir / filename

        # Drop the temporary ts_date column before saving
        day_df_clean = day_df.drop(columns=["ts_date"])
        day_df_clean.to_parquet(output_path, index=False)
        logger.info("Saving %d candles to cache: %s", len(day_df_clean), output_path.name)

    # Return full df without the temporary ts_date column
    df = df.drop(columns=["ts_date"])
    logger.debug("Cache files saved at: %s", storage_dir)
    return df



def _build_exchange(exchange_id: str) -> ccxt.Exchange:
    logger.debug("Building exchange connection: %s", exchange_id)
    # Validate the exchange ID before attempting to instantiate it to avoid AttributeError
    if exchange_id not in getattr(ccxt, "exchanges", []):
        available = ", ".join(getattr(ccxt, "exchanges", []))
        logger.error(
            "Invalid exchange ID provided: %s. Available exchanges: %s",
            exchange_id,
            available,
        )
        msg = f"Invalid exchange ID: {exchange_id}"
        raise ValueError(msg)
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
