"""Data loading and resampling for micro regime classification."""

from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np

from data.historical import fetch_historical_data, OHLCVParams
from regimes.config import MICRO_SYMBOL, MICRO_OHLCV_TIMEFRAME, MICRO_EXCHANGES

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def load_high_frequency_data(
    symbol: str,
    start_date: str | datetime,
    end_date: str | datetime,
    timeframe: str = MICRO_OHLCV_TIMEFRAME,
    exchanges: list[str] | None = MICRO_EXCHANGES,
) -> pd.DataFrame:
    """Load high-frequency OHLCV data from CCXT.

    Parameters
    ----------
    symbol : str
        Trading pair symbol (e.g., 'BTC/USDT').
    start_date : str or datetime
        Start date in format 'YYYY-MM-DD' or datetime object.
    end_date : str or datetime
        End date in format 'YYYY-MM-DD' or datetime object.
    timeframe : str, optional
        CCXT timeframe (default '1m' for 1-minute).
    exchanges : list[str] or None, optional
        List of exchanges to fetch from, or None for defaults.

    Returns
    -------
    pd.DataFrame
        High-frequency OHLCV data with columns:
        ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        Index is datetime.
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    elif start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)

    if isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date).replace(tzinfo=UTC)
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=UTC)

    logger.info(
        f"Loading {timeframe} OHLCV data from {start_date.date()} to {end_date.date()}"
    )

    # Fetch from CCXT
    params = OHLCVParams(
        symbol=symbol,
        start_ms=int(start_date.timestamp() * 1000),
        end_ms=int(end_date.timestamp() * 1000),
        timeframe=timeframe,
        limit=1000,
        exchanges=exchanges,
    )

    df_raw = fetch_historical_data(params)

    if df_raw is None or len(df_raw) == 0:
        raise ValueError(f"No data fetched for {symbol} from {start_date} to {end_date}")

    logger.info(f"Loaded {len(df_raw)} raw records")

    # Flatten multi-exchange data if present
    flat_data = []
    for _, row in df_raw.iterrows():
        flat_row = row.to_dict()
        if isinstance(flat_row.get("data"), dict):
            flat_row.update(flat_row.pop("data"))
        flat_data.append(flat_row)

    df = pd.DataFrame(flat_data)

    # Ensure correct columns and types
    if "ts_event" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ts_event"], utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df.get("timestamp", df.index), utc=True)

    # Extract numeric columns
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select and sort by timestamp
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Cleaned to {len(df)} records with columns {list(df.columns)}"
    )

    return df


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample high-frequency bars to 5-minute intervals.

    Uses OHLC aggregation:
    - open   = first open
    - high   = max(high)
    - low    = min(low)
    - close  = last close
    - volume = sum(volume)

    Parameters
    ----------
    df : pd.DataFrame
        High-frequency data with 'timestamp' and OHLCV columns.

    Returns
    -------
    pd.DataFrame
        5-minute resampled data.
    """
    # Set timestamp as index
    df = df.set_index("timestamp")

    # Resample using OHLC aggregation
    resampled = df["open"].resample("5min").first().to_frame()
    resampled["high"] = df["high"].resample("5min").max()
    resampled["low"] = df["low"].resample("5min").min()
    resampled["close"] = df["close"].resample("5min").last()
    resampled["volume"] = df["volume"].resample("5min").sum()

    # Remove rows with all NaN (gaps in data)
    resampled = resampled.dropna()

    # Reset index to have timestamp as column
    resampled = resampled.reset_index()

    logger.info(
        f"Resampled to {len(resampled)} 5-minute bars "
        f"({resampled['timestamp'].min()} to {resampled['timestamp'].max()})"
    )

    return resampled


def aggregate_multi_exchange_5min(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate 5-minute bars from multiple exchanges using volume-weighting.

    Parameters
    ----------
    df : pd.DataFrame
        5-minute data with 'source' column (exchange name).

    Returns
    -------
    pd.DataFrame
        Aggregated 5-minute data with single row per timestamp.
    """
    if "source" not in df.columns:
        # Single exchange, no aggregation needed
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df.get("timestamp", df.index), utc=True)

    agg_data = []
    for ts, group in df.groupby("timestamp"):
        total_vol = group["volume"].sum()

        # Volume-weighted average for close
        if total_vol > 0:
            weighted_close = (group["close"] * group["volume"]).sum() / total_vol
        else:
            weighted_close = group["close"].mean()

        # Median for OHLH (more robust)
        agg_row = {
            "timestamp": ts,
            "open": group["open"].median(),
            "high": group["high"].median(),
            "low": group["low"].median(),
            "close": weighted_close,
            "volume": total_vol,
            "num_exchanges": group["source"].nunique(),
        }
        agg_data.append(agg_row)

    df_agg = pd.DataFrame(agg_data).sort_values("timestamp").reset_index(drop=True)

    logger.info(f"Aggregated to {len(df_agg)} unique 5-minute bars")

    return df_agg


def load_and_resample_5min(
    symbol: str,
    start_date: str | datetime,
    end_date: str | datetime,
    timeframe: str = MICRO_OHLCV_TIMEFRAME,
    exchanges: list[str] | None = MICRO_EXCHANGES,
) -> pd.DataFrame:
    """End-to-end pipeline: load high-freq data and resample to 5-min bars.

    Parameters
    ----------
    symbol : str
        Trading pair symbol.
    start_date : str or datetime
        Start date.
    end_date : str or datetime
        End date.
    timeframe : str
        CCXT timeframe (default '1m').
    exchanges : list[str] or None
        Exchanges to fetch from.

    Returns
    -------
    pd.DataFrame
        5-minute OHLCV data ready for feature engineering.
    """
    # Load 1-min data
    df = load_high_frequency_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        exchanges=exchanges,
    )

    # Resample to 5-min
    df_5min = resample_to_5min(df)

    # Aggregate multi-exchange if present
    if "source" in df.columns:
        df_5min = aggregate_multi_exchange_5min(df_5min)

    return df_5min
