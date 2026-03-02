"""Storage sink for streaming events with buffering and Parquet output."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from models.events import StoredCryptoEvent, StoredPolymarketEvent
from utils import ensure_directory

logger = logging.getLogger(__name__)


class StreamStorageSink:
    """Manages in-memory buffers for events and writes to Parquet files.

    Parameters
    ----------
    base_path : Path | str
        Root directory for storing event partitions.
    partition_by : str
        Time partition granularity: 'hour', 'day', 'market', or 'none'.
    buffer_size : int
        Number of market events to buffer before writing to disk.
    crypto_buffer_size : int
        Number of crypto events to buffer before writing to disk.
    """

    def __init__(
        self,
        base_path: Path | str = "data/stream_feeds",
        partition_by: str = "hour",
        buffer_size: int = 50,
        crypto_buffer_size: int = 5,
    ) -> None:
        """Initialize storage sink."""
        self.base_path = Path(base_path)
        self.partition_by = partition_by
        self.buffer_size = buffer_size
        self.crypto_buffer_size = crypto_buffer_size

        # Buffers for different event types
        self._market_buffer: dict[str | None, list[dict]] = {}
        self._crypto_buffer: dict[str, list[dict]] = {}

        ensure_directory(self.base_path)
        logger.info(
            "Initialized StreamStorageSink at %s with partition_by=%s, "
            "buffer_size=%d, crypto_buffer_size=%d",
            self.base_path,
            partition_by,
            buffer_size,
            crypto_buffer_size,
        )

    def write_market_event(
        self, data: dict, market_slug: str | None = None
    ) -> StoredPolymarketEvent | None:
        """Write a market event to the buffer.

        Parameters
        ----------
        data : dict
            Market event data.
        market_slug : str | None
            Market slug for organizing files.

        Returns
        -------
        StoredPolymarketEvent | None
            The created event, or None if validation failed.
        """
        try:
            ts_ingest = datetime.now(UTC)
            ts_event_ms = int(data.get("timestamp", ts_ingest.timestamp() * 1000))
            ts_event = datetime.fromtimestamp(ts_event_ms / 1000.0, tz=UTC)

            # Extract identifiers from event data
            market_id = data.get("market", "unknown")
            asset_ids = data.get("asset_ids", [])
            token_id = data.get("asset_id", asset_ids[0] if asset_ids else "unknown")

            # Create event
            event = StoredPolymarketEvent(
                ts_event=ts_event,
                ts_ingest=ts_ingest,
                source="polymarket",
                event_type=data.get("event_type", "unknown"),
                market_id=market_id,
                token_id=token_id,
                market=market_slug,
                data=data,
            )

            # Add to appropriate buffer
            key = market_slug if self.partition_by == "market" else None
            if key not in self._market_buffer:
                self._market_buffer[key] = []
            self._market_buffer[key].append(event.model_dump())

            # Flush if buffer is full
            if len(self._market_buffer[key]) >= self.buffer_size:
                self._flush_market_buffer(key)

            logger.debug(
                "Buffered market event: type=%s, market=%s, buffer_size=%d",
                event.event_type,
                market_id,
                len(self._market_buffer[key]),
            )

        except Exception:
            logger.exception("Failed to write market event")
            return None

        return event

    def write_crypto_price(
        self, data: dict, source: str = "polymarket_rtds"
    ) -> StoredCryptoEvent | None:
        """Write a crypto price event to the buffer.

        Parameters
        ----------
        data : dict
            Crypto price event data.
        source : str
            Event source (e.g., 'polymarket_rtds', 'binance', etc.).

        Returns
        -------
        StoredCryptoEvent | None
            The created event, or None if validation failed.
        """
        try:
            ts_ingest = datetime.now(UTC)
            ts_event_ms = int(data.get("timestamp", ts_ingest.timestamp() * 1000))
            ts_event = datetime.fromtimestamp(ts_event_ms / 1000.0, tz=UTC)

            # Extract symbol
            symbol = data.get("symbol", "unknown")

            # Create event
            event = StoredCryptoEvent(
                ts_event=ts_event,
                ts_ingest=ts_ingest,
                source=source,
                event_type=data.get("event_type", "price"),
                symbol=symbol,
                timeframe=None,
                data=data,
            )

            # Add to buffer
            if source not in self._crypto_buffer:
                self._crypto_buffer[source] = []
            self._crypto_buffer[source].append(event.model_dump())

            # Flush if buffer is full
            if len(self._crypto_buffer[source]) >= self.crypto_buffer_size:
                self._flush_crypto_buffer(source)

            logger.debug(
                "Buffered crypto price: symbol=%s, source=%s, buffer_size=%d",
                symbol,
                source,
                len(self._crypto_buffer[source]),
            )

        except Exception:
            logger.exception("Failed to write crypto price")
            return None

        return event

    def handle_market_resolved(
        self, market_id: str, market_slug: str | None = None
    ) -> None:
        """Flush market buffers when market is resolved.

        Parameters
        ----------
        market_id : str
            Market identifier.
        market_slug : str | None
            Market slug for locating buffer.
        """
        key = market_slug if self.partition_by == "market" else None
        if self._market_buffer.get(key):
            logger.info("Flushing market buffer on resolution: market_id=%s", market_id)
            self._flush_market_buffer(key)

    def _flush_market_buffer(self, key: str | None = None) -> None:
        """Flush market event buffer to Parquet file with appending support."""
        if key not in self._market_buffer or not self._market_buffer[key]:
            return

        try:
            events = self._market_buffer[key]
            df = pd.DataFrame(events)

            # Serialize data field to JSON string to avoid schema mismatches
            df["data"] = df["data"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

            # Create output path
            output_dir = self.base_path / "polymarket_market"
            ensure_directory(output_dir)

            # Group by date and market slug for organized daily storage
            df["ts_date"] = pd.to_datetime(df["ts_event"]).dt.strftime("%Y%m%d")
            grouped = df.groupby(["ts_date", "market"])

            for (ts_date, market_slug), group_df in grouped:
                # Use market slug from the event (default to market_id if slug is None)
                slug = market_slug or group_df["market_id"].iloc[0]

                filename = f"{ts_date}_{slug}.parquet"
                output_path = output_dir / filename

                # Drop ts_date column and prepare data
                write_df = group_df.drop(columns=["ts_date"])

                # Use PyArrow for append support
                table = pa.Table.from_pandas(write_df)

                if output_path.exists():
                    # File exists: read existing, append new, write back
                    existing_table = pq.read_table(output_path)
                    combined_table = pa.concat_tables([existing_table, table])
                    pq.write_table(combined_table, output_path)
                    logger.info(
                        "Appended %d market events to %s (market=%s, total=%d)",
                        len(group_df),
                        filename,
                        slug,
                        len(combined_table),
                    )
                else:
                    # New file: write directly
                    pq.write_table(table, output_path)
                    logger.info(
                        "Created new market file %s with %d events (market=%s, date=%s)",
                        filename,
                        len(group_df),
                        slug,
                        ts_date,
                    )

            # Clear buffer
            self._market_buffer[key] = []

        except Exception:
            logger.exception("Failed to flush market buffer")

    def _flush_crypto_buffer(self, source: str) -> None:
        """Flush crypto price buffer to Parquet file with appending support."""
        if source not in self._crypto_buffer or not self._crypto_buffer[source]:
            return

        try:
            events = self._crypto_buffer[source]
            df = pd.DataFrame(events)

            # Serialize data field to JSON string to avoid schema mismatches
            df["data"] = df["data"].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)

            # Create output path
            output_dir = self.base_path / "polymarket_rtds"
            ensure_directory(output_dir)

            # Group by date, symbol, and timeframe for organized daily storage
            df["ts_date"] = pd.to_datetime(df["ts_event"]).dt.strftime("%Y%m%d")

            # Group by date and symbol (and timeframe if present)
            grouped = df.groupby(["ts_date", "symbol"])

            for (ts_date, symbol), group_df in grouped:
                # Clean symbol for use in filename (e.g., "BTC/USDT" -> "BTC_USDT")
                clean_symbol = str(symbol).replace("/", "_").upper()

                # Get timeframe from first event in group (if available)
                timeframe = group_df["timeframe"].iloc[0]
                if timeframe:
                    # Historical data with timeframe
                    filename = f"{ts_date}_{clean_symbol}_{timeframe!s}.parquet"
                else:
                    # Real-time stream data without timeframe
                    filename = f"{ts_date}_{clean_symbol}_stream.parquet"

                output_path = output_dir / filename

                # Drop ts_date column and prepare data
                write_df = group_df.drop(columns=["ts_date"])

                # Use PyArrow for append support
                table = pa.Table.from_pandas(write_df)

                if output_path.exists():
                    # File exists: read existing, append new, write back
                    existing_table = pq.read_table(output_path)
                    combined_table = pa.concat_tables([existing_table, table])
                    pq.write_table(combined_table, output_path)
                    logger.info(
                        "Appended %d crypto events to %s (symbol=%s, total=%d)",
                        len(group_df),
                        filename,
                        symbol,
                        len(combined_table),
                    )
                else:
                    # New file: write directly
                    pq.write_table(table, output_path)
                    logger.info(
                        "Created new crypto file %s with %d events (symbol=%s, date=%s)",
                        filename,
                        len(group_df),
                        symbol,
                        ts_date,
                    )

            # Clear buffer
            self._crypto_buffer[source] = []

        except Exception:
            logger.exception("Failed to flush crypto buffer")

    def close(self) -> None:
        """Flush all buffers and close the sink."""
        logger.info("Closing StreamStorageSink, flushing all buffers...")

        # Flush market buffers
        for key in list(self._market_buffer.keys()):
            self._flush_market_buffer(key)

        # Flush crypto buffers
        for source in list(self._crypto_buffer.keys()):
            self._flush_crypto_buffer(source)

        logger.info("StreamStorageSink closed")
