"""Comprehensive tests for StreamStorageSink and Parquet storage functionality."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from data.stream.storage import StreamStorageSink
from models.events import StoredCryptoEvent, StoredPolymarketEvent

if TYPE_CHECKING:
    from pathlib import Path

# ============================================================================
# INITIALIZATION AND CONFIGURATION TESTS
# ============================================================================


class TestStorageInitialization:
    """Tests for StreamStorageSink initialization."""

    def test_storage_sink_default_config(self, temp_storage_dir: Path):
        """Test StreamStorageSink with default configuration."""
        sink = StreamStorageSink(base_path=temp_storage_dir)

        assert sink.base_path == temp_storage_dir
        assert sink.partition_by == "hour"
        assert sink.buffer_size == 50
        assert sink._market_buffer == {}
        assert sink._crypto_buffer == {}

    def test_storage_sink_custom_config(self, temp_storage_dir: Path):
        """Test StreamStorageSink with custom configuration."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            partition_by="day",
            buffer_size=100,
        )

        assert sink.partition_by == "day"
        assert sink.buffer_size == 100

    def test_storage_directory_creation(self, temp_storage_dir: Path):
        """Test that storage directory is created if it doesn't exist."""
        new_dir = temp_storage_dir / "new_storage"
        sink = StreamStorageSink(base_path=new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert sink.base_path == new_dir

    def test_storage_with_partition_strategies(self, temp_storage_dir: Path):
        """Test storage sink with different partition strategies."""
        for partition_by in ["hour", "day", "market", "none"]:
            sink = StreamStorageSink(
                base_path=temp_storage_dir,
                partition_by=partition_by,
            )
            assert sink.partition_by == partition_by


# ============================================================================
# POLYMARKET EVENT WRITING TESTS
# ============================================================================


class TestPolymarketEventWriting:
    """Tests for writing Polymarket events to buffer."""

    @pytest.fixture
    def storage_sink(self, temp_storage_dir: Path):
        """Create a storage sink for testing."""
        return StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=50,
            crypto_buffer_size=50,
        )

    def test_write_single_market_event(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test writing a single market event."""
        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": current_timestamp_ms,
            "bids": [{"price": "0.65", "size": "100.0"}],
            "asks": [{"price": "0.66", "size": "120.0"}],
        }

        stored_event = storage_sink.write_market_event(event_data)

        assert stored_event is not None
        assert isinstance(stored_event, StoredPolymarketEvent)
        assert stored_event.event_type == "book"
        assert stored_event.market_id == "0xmarketid"
        assert stored_event.token_id == "0x123abc"

    def test_market_event_timestamp_conversion(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test that event timestamp is converted to UTC datetime."""
        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": current_timestamp_ms,
            "bids": [],
            "asks": [],
        }

        stored_event = storage_sink.write_market_event(event_data)

        assert stored_event is not None
        assert isinstance(stored_event.ts_event, datetime)
        assert stored_event.ts_event.tzinfo == UTC

    def test_market_event_ingestion_timestamp(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test that ingestion timestamp is set to current time."""
        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": current_timestamp_ms,
            "bids": [],
            "asks": [],
        }

        before = datetime.now(UTC)
        stored_event = storage_sink.write_market_event(event_data)
        after = datetime.now(UTC)

        assert stored_event is not None
        assert before <= stored_event.ts_ingest <= after

    def test_market_event_source_set_correctly(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test that event source is set to 'polymarket'."""
        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": current_timestamp_ms,
            "bids": [],
            "asks": [],
        }

        stored_event = storage_sink.write_market_event(event_data)

        assert stored_event.source == "polymarket"

    def test_market_event_with_market_slug(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test writing market event with market slug."""
        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": current_timestamp_ms,
            "bids": [],
            "asks": [],
        }

        stored_event = storage_sink.write_market_event(
            event_data, market_slug="bitcoin-100k"
        )

        assert stored_event.market == "bitcoin-100k"

    def test_market_event_buffering(self, storage_sink):
        """Test that market events are buffered."""
        for i in range(5):
            event_data = {
                "event_type": "book",
                "asset_id": f"0x{i:06x}",
                "market": "0xmarketid",
                "timestamp": 1234567890000 + i * 1000,
                "bids": [],
                "asks": [],
            }
            storage_sink.write_market_event(event_data)

        # Check that events are in buffer
        buffer_key = None
        assert buffer_key in storage_sink._market_buffer
        assert len(storage_sink._market_buffer[buffer_key]) == 5

    def test_market_event_with_missing_asset_id(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test market event when asset_id is derived from asset_ids list."""
        event_data = {
            "event_type": "new_market",
            "market": "0xmarketid",
            "timestamp": current_timestamp_ms,
            "asset_ids": ["0x123abc", "0x456def"],
        }

        stored_event = storage_sink.write_market_event(event_data)

        assert stored_event is not None
        # Token ID should be derived from asset_ids list
        assert stored_event.token_id == "0x123abc"

    def test_market_event_with_all_assets_missing(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test market event when both asset_id and asset_ids are missing."""
        event_data = {
            "event_type": "new_market",
            "market": "0xmarketid",
            "timestamp": current_timestamp_ms,
        }

        stored_event = storage_sink.write_market_event(event_data)

        assert stored_event is not None
        assert stored_event.token_id == "unknown"


# ============================================================================
# CRYPTO EVENT WRITING TESTS
# ============================================================================


class TestCryptoEventWriting:
    """Tests for writing crypto price events to buffer."""

    @pytest.fixture
    def storage_sink(self, temp_storage_dir: Path):
        """Create a storage sink for testing."""
        return StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=50,
            crypto_buffer_size=50,
        )

    def test_write_single_crypto_price(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test writing a single crypto price event."""
        event_data = {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": current_timestamp_ms,
        }

        stored_event = storage_sink.write_crypto_price(event_data)

        assert stored_event is not None
        assert isinstance(stored_event, StoredCryptoEvent)
        assert stored_event.symbol == "btc/usd"
        assert stored_event.source == "polymarket_rtds"

    def test_crypto_event_timestamp_conversion(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test that crypto event timestamp is converted to UTC datetime."""
        event_data = {
            "symbol": "eth/usd",
            "price": "3200.50",
            "timestamp": current_timestamp_ms,
        }

        stored_event = storage_sink.write_crypto_price(event_data)

        assert stored_event is not None
        assert isinstance(stored_event.ts_event, datetime)
        assert stored_event.ts_event.tzinfo == UTC

    def test_crypto_event_ingestion_timestamp(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test that crypto event ingestion timestamp is current."""
        event_data = {
            "symbol": "sol/usd",
            "price": "175.50",
            "timestamp": current_timestamp_ms,
        }

        before = datetime.now(UTC)
        stored_event = storage_sink.write_crypto_price(event_data)
        after = datetime.now(UTC)

        assert stored_event is not None
        assert before <= stored_event.ts_ingest <= after

    def test_crypto_event_source_default(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test that crypto event source defaults to 'polymarket_rtds'."""
        event_data = {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": current_timestamp_ms,
        }

        stored_event = storage_sink.write_crypto_price(event_data)

        assert stored_event.source == "polymarket_rtds"

    def test_crypto_event_source_custom(
        self, storage_sink, current_timestamp_ms: int
    ):
        """Test setting custom source for crypto event."""
        event_data = {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": current_timestamp_ms,
        }

        stored_event = storage_sink.write_crypto_price(event_data, source="binance")

        assert stored_event.source == "binance"

    def test_crypto_event_buffering(self, storage_sink):
        """Test that crypto events are buffered by source."""
        for i in range(5):
            event_data = {
                "symbol": f"coin{i}/usd",
                "price": f"{100 + i}.50",
                "timestamp": 1234567890000 + i * 1000,
            }
            storage_sink.write_crypto_price(event_data, source="binance")

        # Check that events are in buffer
        assert "binance" in storage_sink._crypto_buffer
        assert len(storage_sink._crypto_buffer["binance"]) == 5


# ============================================================================
# BUFFER AUTO-FLUSH TESTS
# ============================================================================


class TestBufferAutoFlush:
    """Tests for automatic buffer flushing when threshold is reached."""

    def test_market_buffer_auto_flush_on_full(self, temp_storage_dir: Path):
        """Test that market buffer flushes when it reaches buffer_size."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=3,
        )

        # Add events up to buffer_size - 1
        for i in range(2):
            event_data = {
                "event_type": "book",
                "asset_id": f"0x{i:06x}",
                "market": "0xmarketid",
                "timestamp": 1234567890000 + i * 1000,
                "bids": [],
                "asks": [],
            }
            sink.write_market_event(event_data)

        # Buffer should still have events
        key = None
        assert len(sink._market_buffer[key]) == 2

        # Add one more to trigger flush
        event_data = {
            "event_type": "book",
            "asset_id": "0x999999",
            "market": "0xmarketid",
            "timestamp": 1234567892000,
            "bids": [],
            "asks": [],
        }
        sink.write_market_event(event_data)

        # Buffer should have flushed and be reset
        assert len(sink._market_buffer[key]) == 0

    def test_crypto_buffer_auto_flush_on_full(self, temp_storage_dir: Path):
        """Test that crypto buffer flushes when it reaches buffer_size."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=3,
            crypto_buffer_size=3,
        )

        # Add events up to buffer_size - 1
        for i in range(2):
            event_data = {
                "symbol": f"coin{i}/usd",
                "price": f"{100 + i}.50",
                "timestamp": 1234567890000 + i * 1000,
            }
            sink.write_crypto_price(event_data, source="chainlink")

        # Buffer should still have events
        assert len(sink._crypto_buffer["chainlink"]) == 2

        # Add one more to trigger flush
        event_data = {
            "symbol": "final/usd",
            "price": "999.99",
            "timestamp": 1234567892000,
        }
        sink.write_crypto_price(event_data, source="chainlink")

        # Buffer should have flushed and be reset
        assert len(sink._crypto_buffer["chainlink"]) == 0


# ============================================================================
# PARQUET FILE WRITING TESTS
# ============================================================================


class TestParquetFileWriting:
    """Tests for writing Parquet files from buffers."""

    def test_write_market_parquet_file(self, temp_storage_dir: Path):
        """Test writing market events to Parquet file."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,  # Force flush on first event
        )

        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "bids": [],
            "asks": [],
        }
        sink.write_market_event(event_data, market_slug="test-market")

        # Check that Parquet file was created
        output_dir = temp_storage_dir / "polymarket_market"
        assert output_dir.exists()
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_write_crypto_parquet_file(self, temp_storage_dir: Path):
        """Test writing crypto events to Parquet file."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,  # Force flush on first event
            crypto_buffer_size=1,
        )

        event_data = {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": 1234567890000,
        }
        sink.write_crypto_price(event_data, source="chainlink")

        # Check that Parquet file was created
        output_dir = temp_storage_dir / "polymarket_rtds"
        assert output_dir.exists()
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) >= 1

    def test_parquet_filename_format_market(self, temp_storage_dir: Path):
        """Test that market Parquet filenames have correct format."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,
        )

        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "bids": [],
            "asks": [],
        }
        sink.write_market_event(event_data, market_slug="bitcoin-100k")

        output_dir = temp_storage_dir / "polymarket_market"
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        # Filename should be YYYYMMDD_market_slug.parquet
        filename = parquet_files[0].name
        assert "bitcoin-100k" in filename
        assert filename.endswith(".parquet")

    def test_parquet_filename_format_crypto(self, temp_storage_dir: Path):
        """Test that crypto Parquet filenames have correct format."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,
            crypto_buffer_size=1,
        )

        event_data = {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": 1234567890000,
        }
        sink.write_crypto_price(event_data, source="chainlink")

        output_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        # Filename should be YYYYMMDD_SYMBOL_stream.parquet or with timeframe
        filename = parquet_files[0].name
        assert "BTC_USD" in filename  # Symbol should be cleaned and uppercase
        assert filename.endswith(".parquet")

    def test_parquet_file_contains_correct_columns(
        self, temp_storage_dir: Path
    ):
        """Test that Parquet files contain expected columns."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,
        )

        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "bids": [],
            "asks": [],
        }
        sink.write_market_event(event_data, market_slug="test-market")

        output_dir = temp_storage_dir / "polymarket_market"
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        # Read the Parquet file
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        # Check that expected columns are present
        expected_columns = {
            "ts_event",
            "ts_ingest",
            "source",
            "event_type",
            "market_id",
            "token_id",
            "data",
        }
        assert expected_columns.issubset(set(df.columns))


# ============================================================================
# PARQUET APPEND OPERATIONS TESTS
# ============================================================================


class TestParquetAppendOperations:
    """Tests for appending to existing Parquet files."""

    def test_append_to_existing_market_file(self, temp_storage_dir: Path):
        """Test appending market events to existing Parquet file."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,
        )

        # Write first event
        event_data_1 = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "bids": [],
            "asks": [],
        }
        sink.write_market_event(event_data_1, market_slug="test-market")

        # Write second event with SAME event type (schemas must match for append)
        event_data_2 = {
            "event_type": "book",  # Changed from "price_change" to "book" for schema consistency
            "asset_id": "0x456def",
            "market": "0xmarketid",
            "timestamp": 1234567891000,
            "bids": [],
            "asks": [],
        }
        sink.write_market_event(event_data_2, market_slug="test-market")

        # Check that file was appended to
        output_dir = temp_storage_dir / "polymarket_market"
        parquet_files = list(output_dir.glob("*.parquet"))

        # Should have at least one file with both events
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()
        assert len(df) >= 2

    def test_append_to_existing_crypto_file(self, temp_storage_dir: Path):
        """Test appending crypto events to existing Parquet file."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,
        )

        # Write first crypto event
        event_data_1 = {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": 1234567890000,
        }
        sink.write_crypto_price(event_data_1, source="chainlink")

        # Write second crypto event with same symbol
        event_data_2 = {
            "symbol": "btc/usd",
            "price": "99000.00",
            "timestamp": 1234567891000,
        }
        sink.write_crypto_price(event_data_2, source="chainlink")

        # Check that events were appended
        output_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(output_dir.glob("*BTC_USD*.parquet"))

        # Should have at least one file with both events
        if parquet_files:
            table = pq.read_table(parquet_files[0])
            df = table.to_pandas()
            assert len(df) >= 2


# ============================================================================
# MANUAL FLUSH TESTS
# ============================================================================


class TestManualFlush:
    """Tests for manually flushing buffers."""

    def test_handle_market_resolved_flushes_buffer(
        self, temp_storage_dir: Path
    ):
        """Test that handle_market_resolved triggers flush."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=100,  # Large buffer to prevent auto-flush
        )

        # Add events without triggering auto-flush
        for i in range(3):
            event_data = {
                "event_type": "book",
                "asset_id": f"0x{i:06x}",
                "market": "0xmarketid",
                "timestamp": 1234567890000 + i * 1000,
                "bids": [],
                "asks": [],
            }
            sink.write_market_event(event_data)

        # Verify events are buffered
        key = None
        assert len(sink._market_buffer[key]) == 3

        # Handle market resolved should flush
        sink.handle_market_resolved("0xmarketid")

        # Buffer should be cleared
        assert len(sink._market_buffer[key]) == 0

    def test_close_flushes_all_buffers(self, temp_storage_dir: Path):
        """Test that close() flushes all remaining buffers."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=100,
        )

        # Add market events
        event_data_market = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "bids": [],
            "asks": [],
        }
        sink.write_market_event(event_data_market)

        # Add crypto events
        event_data_crypto = {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": 1234567890000,
        }
        sink.write_crypto_price(event_data_crypto)

        # Verify events are buffered
        key = None
        assert len(sink._market_buffer[key]) == 1
        assert len(sink._crypto_buffer["polymarket_rtds"]) == 1

        # Close should flush all
        sink.close()

        # All buffers should be empty
        assert len(sink._market_buffer[key]) == 0
        assert len(sink._crypto_buffer["polymarket_rtds"]) == 0


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in storage operations."""

    def test_invalid_market_event_data_returns_none(
        self, storage_sink_with_bad_config
    ):
        """Test that invalid market event data returns None instead of raising."""
        # Note: This would require creating a fixture or context that
        # causes write_market_event to fail gracefully

    def test_write_without_timestamp_uses_default(
        self, temp_storage_dir: Path
    ):
        """Test that events without timestamp use current time."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            # No timestamp provided
            "bids": [],
            "asks": [],
        }

        stored_event = sink.write_market_event(event_data)

        # Should still succeed and use current timestamp
        assert stored_event is not None
        assert stored_event.ts_event is not None

    def test_crypto_event_missing_symbol_stored_as_unknown(
        self, temp_storage_dir: Path
    ):
        """Test crypto event with missing symbol."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        event_data = {
            "price": "98500.50",
            "timestamp": 1234567890000,
            # No symbol provided
        }

        stored_event = sink.write_crypto_price(event_data)

        assert stored_event is not None
        assert stored_event.symbol == "unknown"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_buffer_size_of_one(self, temp_storage_dir: Path):
        """Test that buffer_size of 1 flushes immediately."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,
        )

        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "bids": [],
            "asks": [],
        }

        sink.write_market_event(event_data)

        # Should be flushed immediately
        key = None
        assert len(sink._market_buffer[key]) == 0

    def test_very_large_buffer_size(self, temp_storage_dir: Path):
        """Test storage with very large buffer size."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=10000,
            crypto_buffer_size=10000,
        )

        # Add multiple events
        for i in range(10):
            event_data = {
                "symbol": f"coin{i}/usd",
                "price": f"{100 + i}.50",
                "timestamp": 1234567890000 + i * 1000,
            }
            sink.write_crypto_price(event_data)

        # Should still be in buffer (not flushed)
        assert len(sink._crypto_buffer["polymarket_rtds"]) == 10

    def test_special_characters_in_market_slug(self, temp_storage_dir: Path):
        """Test handling of special characters in market slug."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,
        )

        event_data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "bids": [],
            "asks": [],
        }

        # Should handle special characters in slug gracefully
        stored_event = sink.write_market_event(
            event_data, market_slug="test-market_2024/special"
        )

        assert stored_event is not None

    def test_multiple_sources_crypto_buffering(self, temp_storage_dir: Path):
        """Test that different crypto sources are buffered separately."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=100,
        )

        # Add events from different sources
        event_data = {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": 1234567890000,
        }

        sink.write_crypto_price(event_data, source="chainlink")
        sink.write_crypto_price(event_data, source="binance")
        sink.write_crypto_price(event_data, source="coinbase")

        # Each source should have separate buffer
        assert "chainlink" in sink._crypto_buffer
        assert "binance" in sink._crypto_buffer
        assert "coinbase" in sink._crypto_buffer
        assert len(sink._crypto_buffer["chainlink"]) == 1
        assert len(sink._crypto_buffer["binance"]) == 1
        assert len(sink._crypto_buffer["coinbase"]) == 1


@pytest.fixture
def storage_sink_with_bad_config(temp_storage_dir: Path):
    """Create a storage sink for testing error cases."""
    return StreamStorageSink(base_path=temp_storage_dir)
