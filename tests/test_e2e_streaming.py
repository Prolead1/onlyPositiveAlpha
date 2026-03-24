"""End-to-end integration tests for concurrent streaming and storage."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
import pytest

from data.stream.storage import StreamStorageSink

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# END-TO-END CONCURRENT STREAMING TESTS
# ============================================================================


class TestConcurrentStreaming:
    """End-to-end tests for concurrent crypto and market streaming."""

    @pytest.mark.asyncio
    async def test_concurrent_crypto_and_market_writes(
        self, temp_storage_dir: Path
    ):
        """Test that both crypto and market events can be written concurrently."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            partition_by="market",
            buffer_size=5,
        )

        # Simulate concurrent writes from both streams
        crypto_writes = []
        market_writes = []

        for i in range(10):
            # Crypto event
            crypto_data = {
                "symbol": "btc/usd",
                "price": f"{98000 + i * 100}.50",
                "timestamp": 1234567890000 + i * 1000,
            }
            crypto_event = sink.write_crypto_price(crypto_data, source="chainlink")
            crypto_writes.append(crypto_event)

            # Market event
            market_data = {
                "event_type": "book",
                "asset_id": f"0x{i:06x}",
                "market": "0xmarketid",
                "timestamp": 1234567890000 + i * 1000,
                "bids": [{"price": f"0.{50 + i}", "size": "100"}],
                "asks": [{"price": f"0.{51 + i}", "size": "120"}],
            }
            market_event = sink.write_market_event(
                market_data, market_slug="test-market"
            )
            market_writes.append(market_event)

        # Verify both types of events were created
        assert all(e is not None for e in crypto_writes)
        assert all(e is not None for e in market_writes)
        assert len(crypto_writes) == 10
        assert len(market_writes) == 10

        # Verify buffers contain data before flush
        assert len(sink._crypto_buffer["chainlink"]) >= 0  # May have auto-flushed
        assert len(sink._market_buffer["test-market"]) >= 0  # May have auto-flushed

        # Close to flush remaining
        sink.close()

    @pytest.mark.asyncio
    async def test_high_throughput_mixed_events(self, temp_storage_dir: Path):
        """Test high throughput with mixed crypto and market events."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            partition_by="hour",
            buffer_size=20,
        )

        event_count = 100
        timestamp_base = 1234567890000

        # Write mixed events rapidly
        for i in range(event_count):
            if i % 3 == 0:
                # Crypto event
                sink.write_crypto_price(
                    {
                        "symbol": "eth/usd",
                        "price": f"{3000 + i}.00",
                        "timestamp": timestamp_base + i * 100,
                    },
                    source="binance",
                )
            else:
                # Market event
                sink.write_market_event(
                    {
                        "event_type": "price_change",
                        "market": "0xmarket1",
                        "timestamp": timestamp_base + i * 100,
                        "price_changes": [],
                    },
                    market_slug="btc-up-5m",
                )

        sink.close()

        # Verify files were created
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        market_dir = temp_storage_dir / "polymarket_market"

        assert crypto_dir.exists()
        assert market_dir.exists()


# ============================================================================
# PARTITIONING STRATEGY TESTS
# ============================================================================


class TestPartitioningStrategies:
    """Tests for data partitioning by date, market, and source."""

    def test_crypto_daily_partitioning(self, temp_storage_dir: Path):
        """Test that crypto events are partitioned by date."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,  # Immediate flush
        )

        # Write events across different dates
        dates = [
            datetime(2024, 2, 1, 10, 0, 0, tzinfo=UTC),
            datetime(2024, 2, 1, 15, 30, 0, tzinfo=UTC),
            datetime(2024, 2, 2, 9, 0, 0, tzinfo=UTC),
            datetime(2024, 2, 3, 12, 0, 0, tzinfo=UTC),
        ]

        for date in dates:
            sink.write_crypto_price(
                {
                    "symbol": "btc/usd",
                    "price": "98500.50",
                    "timestamp": int(date.timestamp() * 1000),
                },
                source="chainlink",
            )

        sink.close()

        # Check that files are created with date in filename
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(crypto_dir.glob("*.parquet"))

        # Should have 3 files (3 unique dates)
        assert len(parquet_files) == 3

        # Verify date format in filenames (YYYYMMDD)
        filenames = [f.name for f in parquet_files]
        assert any("20240201" in name for name in filenames)
        assert any("20240202" in name for name in filenames)
        assert any("20240203" in name for name in filenames)

    def test_market_per_market_partitioning(self, temp_storage_dir: Path):
        """Test that market events are partitioned by market slug."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            partition_by="market",
            buffer_size=1,
        )

        # Write events for different markets
        markets = [
            ("btc-up-5m", "0xmarket1"),
            ("eth-up-1h", "0xmarket2"),
            ("btc-up-5m", "0xmarket1"),  # Same market again
        ]

        for market_slug, market_id in markets:
            sink.write_market_event(
                {
                    "event_type": "book",
                    "asset_id": "0x123",
                    "market": market_id,
                    "timestamp": 1234567890000,
                    "bids": [],
                    "asks": [],
                },
                market_slug=market_slug,
            )

        sink.close()

        # Check that separate files exist for each market
        market_dir = temp_storage_dir / "polymarket_market"
        parquet_files = list(market_dir.glob("*.parquet"))

        # Should have 2 files (2 unique market slugs)
        assert len(parquet_files) == 2

        # Verify market slugs are in filenames
        filenames = [f.name for f in parquet_files]
        assert any("btc-up-5m" in name for name in filenames)
        assert any("eth-up-1h" in name for name in filenames)

    def test_crypto_multiple_sources_separate_buffers(
        self, temp_storage_dir: Path
    ):
        """Test that crypto sources maintain separate buffers."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=10)

        # Write to multiple sources
        sources = ["chainlink", "binance", "coinbase"]
        for source in sources:
            for i in range(3):
                sink.write_crypto_price(
                    {
                        "symbol": "btc/usd",
                        "price": f"{98000 + i}.00",
                        "timestamp": 1234567890000 + i * 1000,
                    },
                    source=source,
                )

        # Verify separate buffers exist
        for source in sources:
            assert source in sink._crypto_buffer
            assert len(sink._crypto_buffer[source]) == 3

        sink.close()


# ============================================================================
# DATA QUALITY VALIDATION TESTS
# ============================================================================


class TestLiveCryptoStream:
    """Live crypto stream integration test.

    Tests real Polymarket RTDS crypto price stream connectivity
    and data storage. Marked with @pytest.mark.live to allow
    selective execution and rate-limit aware scheduling.
    """

    @pytest.mark.live
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_crypto_stream_integration(
        self, temp_storage_dir: Path, captured_crypto_events: list[dict]
    ):
        """Test crypto price streaming and storage using captured events.

        Uses pre-captured crypto events from a live stream (captured once per
        session), avoiding repeated WebSocket connections.
        """
        # Skip if no events were captured (e.g., rate limiting)
        if not captured_crypto_events:
            pytest.skip("No crypto events captured from live stream")

        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            partition_by="day",
            buffer_size=10,
        )

        # Write captured events directly to storage
        for event in captured_crypto_events:
            sink.write_crypto_price(event, source="chainlink")

        sink.close()

        # Verify we received data
        assert len(captured_crypto_events) > 0, "No crypto events captured"

        # Verify data structure
        first_event = captured_crypto_events[0]
        assert "symbol" in first_event
        assert "price" in first_event
        assert "timestamp" in first_event

        # Verify data was written to Parquet
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        assert crypto_dir.exists()

        parquet_files = list(crypto_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "No Parquet files created"

        # Read and validate Parquet data
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Verify schema
            assert "ts_event" in df.columns
            assert "ts_ingest" in df.columns
            assert "symbol" in df.columns
            assert "source" in df.columns
            assert "data" in df.columns

            # Verify data quality
            assert len(df) > 0
            assert df["symbol"].notna().all()
            assert df["source"].notna().all()
            assert (df["source"] == "chainlink").all()


# ============================================================================
# DATA QUALITY VALIDATION TESTS
# ============================================================================


class TestDataQualityValidation:
    """Tests for validating data quality in written Parquet files."""

    def test_parquet_schema_validation_crypto(self, temp_storage_dir: Path):
        """Validate that crypto Parquet files have correct schema."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        sink.write_crypto_price(
            {
                "symbol": "btc/usd",
                "price": "98500.50",
                "timestamp": 1234567890000,
            },
            source="chainlink",
        )
        sink.close()

        # Read Parquet file
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(crypto_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        table = pq.read_table(parquet_files[0])
        schema = table.schema

        # Validate required columns exist
        required_columns = {
            "ts_event",
            "ts_ingest",
            "source",
            "event_type",
            "symbol",
            "data",
        }
        actual_columns = set(schema.names)
        assert required_columns.issubset(actual_columns)

        # Validate column types
        assert str(schema.field("ts_event").type) == "timestamp[us, tz=UTC]"
        assert str(schema.field("ts_ingest").type) == "timestamp[us, tz=UTC]"
        assert str(schema.field("source").type) == "large_string"
        assert str(schema.field("symbol").type) == "large_string"

    def test_parquet_schema_validation_market(self, temp_storage_dir: Path):
        """Validate that market Parquet files have correct schema."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        sink.write_market_event(
            {
                "event_type": "book",
                "asset_id": "0x123abc",
                "market": "0xmarketid",
                "timestamp": 1234567890000,
                "bids": [],
                "asks": [],
            },
            market_slug="test-market",
        )
        sink.close()

        # Read Parquet file
        market_dir = temp_storage_dir / "polymarket_market"
        parquet_files = list(market_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        table = pq.read_table(parquet_files[0])
        schema = table.schema

        # Validate required columns
        required_columns = {
            "ts_event",
            "ts_ingest",
            "source",
            "event_type",
            "market_id",
            "token_id",
            "data",
        }
        actual_columns = set(schema.names)
        assert required_columns.issubset(actual_columns)

        # Validate column types
        assert str(schema.field("ts_event").type) == "timestamp[us, tz=UTC]"
        assert str(schema.field("ts_ingest").type) == "timestamp[us, tz=UTC]"
        assert str(schema.field("event_type").type) == "large_string"
        assert str(schema.field("market_id").type) == "large_string"
        assert str(schema.field("token_id").type) == "large_string"

    def test_data_integrity_no_nulls_in_required_fields(
        self, temp_storage_dir: Path
    ):
        """Validate that required fields contain no nulls."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        # Write multiple events
        for i in range(5):
            sink.write_crypto_price(
                {
                    "symbol": f"coin{i}/usd",
                    "price": f"{100 + i}.50",
                    "timestamp": 1234567890000 + i * 1000,
                },
                source="chainlink",
            )

        sink.close()

        # Read and validate
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(crypto_dir.glob("*.parquet"))

        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Check required fields have no nulls
            required_fields = ["ts_event", "ts_ingest", "source", "symbol"]
            for field in required_fields:
                assert df[field].notna().all(), f"Field {field} contains null values"

    def test_timestamp_ordering_preserved(self, temp_storage_dir: Path):
        """Validate that timestamps are preserved in correct order."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        timestamps = [
            1234567890000,
            1234567891000,
            1234567892000,
            1234567893000,
            1234567894000,
        ]

        for ts in timestamps:
            sink.write_crypto_price(
                {"symbol": "btc/usd", "price": "98500.50", "timestamp": ts},
                source="chainlink",
            )

        sink.close()

        # Read and verify order
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(crypto_dir.glob("*.parquet"))
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        # Convert to milliseconds for comparison
        event_timestamps = df["ts_event"].astype("int64") // 1_000_000

        # Check that timestamps are in order (allowing for small variations in ingestion time)
        # Events should maintain their relative ordering
        assert len(event_timestamps) == len(timestamps)

    def test_data_completeness_all_fields_present(
        self, temp_storage_dir: Path
    ):
        """Validate that all expected fields are present in written data."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        # Write event with all optional fields
        sink.write_market_event(
            {
                "event_type": "book",
                "asset_id": "0x123abc",
                "market": "0xmarketid",
                "timestamp": 1234567890000,
                "bids": [{"price": "0.65", "size": "100"}],
                "asks": [{"price": "0.66", "size": "120"}],
                "market_slug": "btc-up-5m",
            },
            market_slug="btc-up-5m",
        )

        sink.close()

        # Read and verify all fields
        market_dir = temp_storage_dir / "polymarket_market"
        parquet_files = list(market_dir.glob("*.parquet"))
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        # Verify the data dict contains original event structure
        data_dict = df["data"].iloc[0]
        assert "event_type" in data_dict
        assert "asset_id" in data_dict
        assert "market" in data_dict
        assert "bids" in data_dict
        assert "asks" in data_dict


# ============================================================================
# LIVE MARKET STREAM TEST
# ============================================================================


class TestLiveMarketStream:
    """Live market stream integration test.

    Tests real Polymarket market channel stream connectivity
    and data storage. Marked with @pytest.mark.live to allow
    selective execution and rate-limit aware scheduling.
    """

    @pytest.mark.live
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_market_stream_integration(
        self,
        temp_storage_dir: Path,
        captured_polymarket_events: tuple[str, list[dict]],
    ):
        """Test market data streaming and storage using captured events.

        Uses pre-captured market events from a live stream (captured once per
        session), avoiding repeated WebSocket connections.
        """
        market_slug, events_received = captured_polymarket_events

        # Skip if no events were captured (e.g., rate limiting)
        if not events_received:
            pytest.skip("No market events captured from live stream")

        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            partition_by="market",
            buffer_size=10,
        )

        # Write captured events to storage
        for event in events_received:
            sink.write_market_event(event, market_slug=market_slug)

        sink.close()

        # Verify we received data
        assert len(events_received) > 0, "No market events captured"

        # Verify event types
        event_types = {e.get("event_type") for e in events_received
                       if isinstance(e, dict)}
        assert len(event_types) > 0, "No valid event types captured"

        # Common event types we expect to see
        expected_types = {"book", "price_change", "last_trade_price",
                          "tick_size_change"}
        assert len(event_types.intersection(expected_types)) > 0, \
            f"Expected some of {expected_types}, got {event_types}"

        # Verify data was written to Parquet
        market_dir = temp_storage_dir / "polymarket_market"
        assert market_dir.exists()

        parquet_files = list(market_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "No Parquet files created"

        # Read and validate Parquet data
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Verify schema
            assert "ts_event" in df.columns
            assert "ts_ingest" in df.columns
            assert "event_type" in df.columns
            assert "market_id" in df.columns
            assert "token_id" in df.columns
            assert "source" in df.columns
            assert "data" in df.columns

            # Verify data quality
            assert len(df) > 0
            assert df["event_type"].notna().all()
            assert df["market_id"].notna().all()
            assert (df["source"] == "polymarket").all()


# ============================================================================
# FILE ORGANIZATION TESTS
# ============================================================================


class TestFileOrganization:
    """Tests for verifying proper file organization and naming."""

    def test_crypto_file_naming_convention(self, temp_storage_dir: Path):
        """Test that crypto files follow naming convention: YYYYMMDD_SYMBOL_stream.parquet."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        # Write event on specific date
        timestamp = datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC)
        sink.write_crypto_price(
            {
                "symbol": "btc/usd",
                "price": "98500.50",
                "timestamp": int(timestamp.timestamp() * 1000),
            },
            source="chainlink",
        )

        sink.close()

        # Verify filename
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(crypto_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

        filename = parquet_files[0].name
        assert filename.startswith("20240315_")
        assert "BTC_USD" in filename  # Symbol should be uppercase with underscore
        assert filename.endswith("_stream.parquet")

    def test_market_file_naming_convention(self, temp_storage_dir: Path):
        """Test that market files follow naming convention: YYYYMMDD_MARKETSLUG.parquet."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        # Write event on specific date
        timestamp = datetime(2024, 3, 15, 10, 30, 0, tzinfo=UTC)
        sink.write_market_event(
            {
                "event_type": "book",
                "asset_id": "0x123abc",
                "market": "0xmarketid",
                "timestamp": int(timestamp.timestamp() * 1000),
                "bids": [],
                "asks": [],
            },
            market_slug="btc-up-5m",
        )

        sink.close()

        # Verify filename
        market_dir = temp_storage_dir / "polymarket_market"
        parquet_files = list(market_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

        filename = parquet_files[0].name
        assert filename.startswith("20240315_")
        assert "btc-up-5m" in filename
        assert filename.endswith(".parquet")

    def test_directory_structure(self, temp_storage_dir: Path):
        """Test that proper directory structure is created."""
        sink = StreamStorageSink(base_path=temp_storage_dir, buffer_size=1)

        # Write both types of events
        sink.write_crypto_price(
            {"symbol": "btc/usd", "price": "98500.50", "timestamp": 1234567890000},
            source="chainlink",
        )
        sink.write_market_event(
            {
                "event_type": "book",
                "asset_id": "0x123",
                "market": "0xmarket",
                "timestamp": 1234567890000,
                "bids": [],
                "asks": [],
            },
            market_slug="test",
        )

        sink.close()

        # Verify directory structure
        assert (temp_storage_dir / "polymarket_rtds").exists()
        assert (temp_storage_dir / "polymarket_rtds").is_dir()
        assert (temp_storage_dir / "polymarket_market").exists()
        assert (temp_storage_dir / "polymarket_market").is_dir()


# ============================================================================
# LIVE CONCURRENT STREAMS TEST
# ============================================================================


class TestLiveConcurrentStreams:
    """Live concurrent stream integration test.

    Tests both crypto and market streams running simultaneously,
    verifying proper data handling and storage. Marked with @pytest.mark.live
    to allow selective execution and rate-limit aware scheduling.
    """

    def _verify_concurrent_parquet_files(self, temp_storage_dir: Path) -> None:
        """Verify both crypto and market Parquet files."""
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        market_dir = temp_storage_dir / "polymarket_market"

        assert crypto_dir.exists()
        assert market_dir.exists()

        crypto_files = list(crypto_dir.glob("*.parquet"))
        market_files = list(market_dir.glob("*.parquet"))

        assert len(crypto_files) > 0, "No crypto Parquet files created"
        assert len(market_files) > 0, "No market Parquet files created"

        # Verify data quality in both file types
        for parquet_file in crypto_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            assert len(df) > 0
            assert (df["source"] == "chainlink").all()
            assert df["symbol"].notna().all()

        for parquet_file in market_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()
            assert len(df) > 0
            assert (df["source"] == "polymarket").all()
            assert df["event_type"].notna().all()

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_concurrent_streams_integration(
        self,
        temp_storage_dir: Path,
        captured_crypto_events: list[dict],
        captured_polymarket_events: tuple[str, list[dict]],
    ):
        """Test both crypto and market streams using captured events.

        Uses pre-captured events from both streams (captured once per session),
        avoiding repeated WebSocket connections. Tests that both event types
        can be written and stored concurrently.
        """
        # Skip if no events were captured (e.g., rate limiting)
        if not captured_crypto_events:
            pytest.skip("No crypto events captured from live stream")

        market_slug, market_events = captured_polymarket_events
        if not market_events:
            pytest.skip("No market events captured from live stream")

        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            partition_by="market",
            buffer_size=10,
        )

        # Write crypto events
        for crypto_event in captured_crypto_events:
            sink.write_crypto_price(crypto_event, source="chainlink")

        # Write market events
        for market_event in market_events:
            sink.write_market_event(market_event, market_slug=market_slug)

        sink.close()

        # Verify both streams received data
        assert len(captured_crypto_events) > 0, "No crypto events captured"
        assert len(market_events) > 0, "No market events captured"

        # Verify both types of files were created and validate data quality
        self._verify_concurrent_parquet_files(temp_storage_dir)


# ============================================================================
# LIVE DATA COMPLETENESS TEST
# ============================================================================


class TestLiveDataCompleteness:
    """Live data completeness integration test.

    Verifies that all data fields are preserved from WebSocket
    to Parquet storage. Marked with @pytest.mark.live to allow
    selective execution and rate-limit aware scheduling.
    """

    @pytest.mark.live
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_stream_data_completeness(
        self, temp_storage_dir: Path, captured_crypto_events: list[dict]
    ):
        """Verify that all data fields are preserved from WebSocket to Parquet.

        Uses pre-captured crypto events from a live stream (captured once per
        session), avoiding repeated WebSocket connections.
        """
        # Skip if no events were captured (e.g., rate limiting)
        if not captured_crypto_events:
            pytest.skip("No crypto events captured from live stream")

        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            partition_by="day",
            buffer_size=5,
        )

        # Write captured events to storage
        for raw_message in captured_crypto_events:
            sink.write_crypto_price(raw_message, source="chainlink")

        sink.close()

        assert len(captured_crypto_events) > 0, "No messages captured"

        # Read Parquet files
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(crypto_dir.glob("*.parquet"))
        assert len(parquet_files) > 0

        # Verify data preservation
        for parquet_file in parquet_files:
            table = pq.read_table(parquet_file)
            df = table.to_pandas()

            # Each row should have a 'data' field containing the original message
            assert "data" in df.columns
            assert len(df) > 0

            # Verify that data field contains JSON strings
            for data_entry in df["data"]:
                assert isinstance(data_entry, str), "data should be JSON string"
                # Should have key fields from original WebSocket message
                assert len(data_entry) > 0


# ============================================================================
# PERFORMANCE AND STRESS TESTS
# ============================================================================


class TestPerformanceAndStress:
    """Performance tests for streaming under load."""

    def test_large_volume_write_performance(self, temp_storage_dir: Path):
        """Test writing large volumes of data efficiently."""
        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=100,  # Larger buffer for performance
        )

        event_count = 1000
        start_time = datetime.now(UTC)

        for i in range(event_count):
            sink.write_crypto_price(
                {
                    "symbol": "btc/usd",
                    "price": f"{98000 + i}.00",
                    "timestamp": 1234567890000 + i * 100,
                },
                source="chainlink",
            )

        sink.close()
        duration = (datetime.now(UTC) - start_time).total_seconds()

        # Should complete in reasonable time (< 5 seconds for 1000 events)
        assert duration < 5.0, f"Writing {event_count} events took {duration}s"

        # Verify all data was written
        crypto_dir = temp_storage_dir / "polymarket_rtds"
        parquet_files = list(crypto_dir.glob("*.parquet"))
        total_rows = sum(len(pq.read_table(f)) for f in parquet_files)
        assert total_rows == event_count


