"""Comprehensive tests for data.historical.ccxt module."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow.parquet as pq
import pytest

try:
    import ccxt  # Optional runtime dependency for integration tests
except Exception:  # pragma: no cover - integration optional
    ccxt = None

from data.historical import OHLCVParams, fetch_historical_data
from data.historical.ccxt import _build_exchange, _fetch_ohlcv_paginated
from models.events import StoredCryptoEvent

if TYPE_CHECKING:
    from pathlib import Path


# ============================================================================
# OHLCV PARAMS TESTS
# ============================================================================


class TestOHLCVParams:
    """Tests for OHLCVParams dataclass."""

    def test_ohlcv_params_creation_minimal(self):
        """Test creating OHLCVParams with minimal required fields."""
        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,  # 2021-01-01 00:00:00 UTC
            end_ms=1609545600000,  # 2021-01-02 00:00:00 UTC
            timeframe="1h",
            limit=1000,
        )

        assert params.symbol == "BTC/USDT"
        assert params.start_ms == 1609459200000
        assert params.end_ms == 1609545600000
        assert params.timeframe == "1h"
        assert params.limit == 1000
        assert params.exchanges is None

    def test_ohlcv_params_creation_with_exchanges(self):
        """Test creating OHLCVParams with custom exchanges."""
        exchanges = ["binance", "coinbase"]
        params = OHLCVParams(
            symbol="ETH/USDT",
            start_ms=None,
            end_ms=None,
            timeframe="1d",
            limit=500,
            exchanges=exchanges,
        )

        assert params.exchanges == exchanges
        assert params.start_ms is None
        assert params.end_ms is None

    def test_ohlcv_params_different_timeframes(self):
        """Test OHLCVParams with different timeframe values."""
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

        for tf in timeframes:
            params = OHLCVParams(
                symbol="BTC/USD",
                start_ms=1609459200000,
                end_ms=1609545600000,
                timeframe=tf,
                limit=1000,
            )
            assert params.timeframe == tf


# ============================================================================
# EXCHANGE BUILD TESTS
# ============================================================================


class TestBuildExchange:
    """Tests for _build_exchange function."""

    @patch("data.historical.ccxt.ccxt")
    def test_build_exchange_success(self, mock_ccxt):
        """Test successful exchange building."""
        mock_exchange_class = Mock()
        mock_exchange_instance = Mock()
        mock_exchange_instance.load_markets = Mock()
        mock_exchange_class.return_value = mock_exchange_instance

        # Setup ccxt module mock
        mock_ccxt.exchanges = ["binance", "coinbase", "kraken"]
        mock_ccxt.binance = mock_exchange_class

        result = _build_exchange("binance")

        assert result == mock_exchange_instance
        mock_exchange_class.assert_called_once_with({"enableRateLimit": True})
        mock_exchange_instance.load_markets.assert_called_once()

    @patch("data.historical.ccxt.ccxt")
    def test_build_exchange_invalid_exchange_id(self, mock_ccxt):
        """Test building exchange with invalid exchange ID."""
        mock_ccxt.exchanges = ["binance", "coinbase", "kraken"]

        with pytest.raises(ValueError, match="Invalid exchange ID"):
            _build_exchange("invalid_exchange")

    @patch("data.historical.ccxt.ccxt")
    def test_build_exchange_validates_before_getattr(self, mock_ccxt):
        """Test that exchange ID is validated before getattr."""
        mock_ccxt.exchanges = ["binance"]

        # Should raise ValueError, not AttributeError
        with pytest.raises(ValueError, match="Invalid exchange ID"):
            _build_exchange("nonexistent")


# ============================================================================
# FETCH OHLCV PAGINATED TESTS
# ============================================================================


class TestFetchOHLCVPaginated:
    """Tests for _fetch_ohlcv_paginated function."""

    def test_fetch_ohlcv_paginated_single_page(self):
        """Test fetching OHLCV data that fits in a single page."""
        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_exchange.rateLimit = 50  # 50ms rate limit

        # Mock data: 5 candles
        mock_data = [
            [1609459200000, 29000, 29500, 28800, 29200, 100],
            [1609462800000, 29200, 29600, 29100, 29400, 110],
            [1609466400000, 29400, 29800, 29300, 29600, 105],
            [1609470000000, 29600, 30000, 29500, 29800, 120],
            [1609473600000, 29800, 30200, 29700, 30000, 115],
        ]
        mock_exchange.fetch_ohlcv.return_value = mock_data

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=None,
            timeframe="1h",
            limit=1000,
        )

        result = _fetch_ohlcv_paginated(mock_exchange, params, params.start_ms, params.end_ms)

        assert len(result) == 5
        assert result == mock_data
        mock_exchange.fetch_ohlcv.assert_called_once()

    def test_fetch_ohlcv_paginated_multiple_pages(self):
        """Test fetching OHLCV data across multiple pages."""
        mock_exchange = Mock()
        mock_exchange.id = "coinbase"
        mock_exchange.rateLimit = 100

        # Mock paginated data
        page1 = [
            [
                1609459200000 + i * 3600000,
                29000 + i * 100,
                29500,
                28800,
                29200,
                100,
            ]
            for i in range(3)
        ]
        page2 = [
            [
                1609470000000 + i * 3600000,
                30000 + i * 100,
                30500,
                29800,
                30200,
                110,
            ]
            for i in range(2)
        ]

        mock_exchange.fetch_ohlcv.side_effect = [page1, page2, []]  # Empty list signals end

        params = OHLCVParams(
            symbol="ETH/USDT",
            start_ms=1609459200000,
            end_ms=None,
            timeframe="1h",
            limit=3,
        )

        with patch("data.historical.ccxt.time.sleep"):  # Skip actual sleep
            result = _fetch_ohlcv_paginated(mock_exchange, params, params.start_ms, params.end_ms)

        assert len(result) == 5
        assert result[:3] == page1
        assert result[3:] == page2

    def test_fetch_ohlcv_paginated_with_end_time_filter(self):
        """Test fetching OHLCV data with end_ms filter."""
        mock_exchange = Mock()
        mock_exchange.id = "kraken"
        mock_exchange.rateLimit = 50

        # Data that goes beyond end_ms
        mock_data = [
            [1609459200000, 29000, 29500, 28800, 29200, 100],  # Within range
            [1609462800000, 29200, 29600, 29100, 29400, 110],  # Within range
            [1609466400000, 29400, 29800, 29300, 29600, 105],  # Within range
            [1609470000000, 29600, 30000, 29500, 29800, 120],  # Beyond end_ms
        ]
        mock_exchange.fetch_ohlcv.return_value = mock_data

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,  # Should include first 3 candles
            timeframe="1h",
            limit=1000,
        )

        result = _fetch_ohlcv_paginated(mock_exchange, params, params.start_ms, params.end_ms)

        # Should stop after filtering out data beyond end_ms
        assert len(result) == 3
        assert result[-1][0] <= params.end_ms

    def test_fetch_ohlcv_paginated_empty_response(self):
        """Test handling empty response from exchange."""
        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_exchange.fetch_ohlcv.return_value = []

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=None,
            timeframe="1h",
            limit=1000,
        )

        result = _fetch_ohlcv_paginated(mock_exchange, params, params.start_ms, params.end_ms)

        assert result == []

    def test_fetch_ohlcv_paginated_respects_rate_limit(self):
        """Test that function respects exchange rate limits during pagination."""
        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_exchange.rateLimit = 200  # 200ms

        page1 = [[1609459200000, 29000, 29500, 28800, 29200, 100]]
        page2 = [[1609462800000, 29200, 29600, 29100, 29400, 110]]

        mock_exchange.fetch_ohlcv.side_effect = [page1, page2, []]

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=None,
            timeframe="1h",
            limit=1,
        )

        with patch("data.historical.ccxt.time.sleep") as mock_sleep:
            _fetch_ohlcv_paginated(mock_exchange, params, params.start_ms, params.end_ms)

        # Sleep should be called between pages
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(0.2)  # rateLimit / 1000


# ============================================================================
# FETCH HISTORICAL DATA INTEGRATION TESTS
# ============================================================================


class TestFetchHistoricalData:
    """Tests for fetch_historical_data function."""

    @pytest.fixture
    def temp_historical_dir(self, tmp_path: Path) -> Path:
        """Create temporary directory for historical data."""
        hist_dir = tmp_path / "data" / "historical_ohlcv"
        hist_dir.mkdir(parents=True)
        return hist_dir

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_fetch_historical_data_success(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test successful historical data fetching."""
        # Setup mocks
        mock_get_root.return_value = tmp_path

        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_build_exchange.return_value = mock_exchange

        # Mock OHLCV data
        mock_ohlcv = [
            [1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0],
            [1609462800000, 29200.0, 29600.0, 29100.0, 29400.0, 110.0],
        ]
        mock_fetch_paginated.return_value = mock_ohlcv

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance"],
        )

        result = fetch_historical_data(params)

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "ts_event" in result.columns
        assert "source" in result.columns
        assert "symbol" in result.columns
        assert result["source"].iloc[0] == "binance"
        assert result["symbol"].iloc[0] == "BTC/USDT"

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_fetch_historical_data_multiple_exchanges(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test fetching data from multiple exchanges."""
        mock_get_root.return_value = tmp_path

        # Mock different exchanges
        def build_exchange_side_effect(exchange_id: str):
            mock = Mock()
            mock.id = exchange_id
            return mock

        mock_build_exchange.side_effect = build_exchange_side_effect

        # Mock OHLCV data for each exchange
        ohlcv_binance = [[1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0]]
        ohlcv_coinbase = [[1609459200000, 29010.0, 29510.0, 28810.0, 29210.0, 105.0]]

        mock_fetch_paginated.side_effect = [ohlcv_binance, ohlcv_coinbase]

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance", "coinbase"],
        )

        result = fetch_historical_data(params)

        assert result is not None
        assert len(result) == 2
        assert set(result["source"].unique()) == {"binance", "coinbase"}

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    def test_fetch_historical_data_exchange_error(
        self, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test handling of exchange connection errors."""
        mock_get_root.return_value = tmp_path
        mock_build_exchange.side_effect = Exception("Connection failed")

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance"],
        )

        result = fetch_historical_data(params)

        # Should return None when no data collected
        assert result is None

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_fetch_historical_data_no_data_returned(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test handling when exchanges return no data."""
        mock_get_root.return_value = tmp_path

        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_build_exchange.return_value = mock_exchange
        mock_fetch_paginated.return_value = []  # No data

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance"],
        )

        result = fetch_historical_data(params)

        assert result is None

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_fetch_historical_data_file_creation(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test that Parquet files are created correctly."""
        mock_get_root.return_value = tmp_path

        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_build_exchange.return_value = mock_exchange

        # Data from a single day
        mock_ohlcv = [
            [1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0],  # 2021-01-01
        ]
        mock_fetch_paginated.return_value = mock_ohlcv

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance"],
        )

        fetch_historical_data(params)

        # Check file was created
        storage_dir = tmp_path / "data" / "historical_ohlcv"
        assert storage_dir.exists()

        # Check for parquet file with expected naming pattern
        parquet_files = list(storage_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

        # File should be named like: 20210101_BTC_USDT_1h.parquet
        assert "20210101" in parquet_files[0].name
        assert "BTC_USDT" in parquet_files[0].name
        assert "1h" in parquet_files[0].name

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_fetch_historical_data_default_exchanges(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test that default exchanges are used when none specified."""
        mock_get_root.return_value = tmp_path

        # Track which exchanges were attempted
        attempted_exchanges = []

        def build_exchange_side_effect(exchange_id: str):
            attempted_exchanges.append(exchange_id)
            mock = Mock()
            mock.id = exchange_id
            return mock

        mock_build_exchange.side_effect = build_exchange_side_effect
        mock_fetch_paginated.return_value = []  # No data

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=None,  # Use defaults
        )

        fetch_historical_data(params)

        # Should attempt default exchanges: binance, kraken, coinbase, bitstamp
        assert "binance" in attempted_exchanges
        assert "kraken" in attempted_exchanges
        assert "coinbase" in attempted_exchanges
        assert "bitstamp" in attempted_exchanges

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_fetch_historical_data_ohlcv_values(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test that OHLCV values are correctly stored in data dict."""
        mock_get_root.return_value = tmp_path

        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_build_exchange.return_value = mock_exchange

        mock_ohlcv = [
            [1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0],
        ]
        mock_fetch_paginated.return_value = mock_ohlcv

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance"],
        )

        result = fetch_historical_data(params)

        assert result is not None
        data_field = result["data"].iloc[0]

        assert data_field["open"] == 29000.0
        assert data_field["high"] == 29500.0
        assert data_field["low"] == 28800.0
        assert data_field["close"] == 29200.0
        assert data_field["volume"] == 100.0

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_fetch_historical_data_partial_exchange_failures(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test that successful exchanges are used even if some fail."""
        mock_get_root.return_value = tmp_path

        # First exchange fails, second succeeds
        def build_exchange_side_effect(exchange_id: str):
            if exchange_id == "binance":
                raise ConnectionError("Connection failed")
            mock = Mock()
            mock.id = exchange_id
            return mock

        mock_build_exchange.side_effect = build_exchange_side_effect

        mock_ohlcv = [[1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0]]
        mock_fetch_paginated.return_value = mock_ohlcv

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance", "coinbase"],
        )

        result = fetch_historical_data(params)

        # Should still get data from coinbase
        assert result is not None
        assert len(result) == 1
        assert result["source"].iloc[0] == "coinbase"


# ============================================================================
# STORED CRYPTO EVENT COMPATIBILITY TESTS
# ============================================================================


class TestStoredCryptoEventStructure:
    """Tests for StoredCryptoEvent structure consistency between RTDS and CCXT."""

    def test_stored_crypto_event_ccxt_structure(self):
        """Test that CCXT creates StoredCryptoEvent with correct structure."""
        ts_event = datetime.now(tz=UTC)
        ts_ingest = datetime.now(tz=UTC)

        event = StoredCryptoEvent(
            ts_event=ts_event,
            ts_ingest=ts_ingest,
            source="binance",
            event_type="ohlcv",
            symbol="BTC/USDT",
            timeframe="1h",
            data={
                "open": 29000.0,
                "high": 29500.0,
                "low": 28800.0,
                "close": 29200.0,
                "volume": 100.0,
            },
        )

        assert event.ts_event == ts_event
        assert event.ts_ingest == ts_ingest
        assert event.source == "binance"
        assert event.event_type == "ohlcv"
        assert event.symbol == "BTC/USDT"
        assert event.timeframe == "1h"
        assert event.data["open"] == 29000.0

    def test_stored_crypto_event_rtds_structure(self):
        """Test that RTDS creates StoredCryptoEvent with correct structure."""
        ts_event = datetime.now(tz=UTC)
        ts_ingest = datetime.now(tz=UTC)

        event = StoredCryptoEvent(
            ts_event=ts_event,
            ts_ingest=ts_ingest,
            source="polymarket_rtds",
            event_type="price",
            symbol="btc/usd",
            timeframe=None,  # RTDS doesn't use timeframe
            data={
                "symbol": "btc/usd",
                "price": "29200.50",
                "timestamp": int(ts_event.timestamp() * 1000),
            },
        )

        assert event.source == "polymarket_rtds"
        assert event.event_type == "price"
        assert event.timeframe is None
        assert event.data["symbol"] == "btc/usd"

    def test_stored_crypto_event_compatible_serialization(self):
        """Test that both CCXT and RTDS events serialize to compatible format."""
        ts = datetime.now(tz=UTC)

        # Create CCXT event
        ccxt_event = StoredCryptoEvent(
            ts_event=ts,
            ts_ingest=ts,
            source="binance",
            event_type="ohlcv",
            symbol="BTC/USDT",
            timeframe="1h",
            data={"open": 29000.0, "close": 29200.0, "volume": 100.0},
        )

        # Create RTDS event
        rtds_event = StoredCryptoEvent(
            ts_event=ts,
            ts_ingest=ts,
            source="polymarket_rtds",
            event_type="price",
            symbol="btc/usd",
            timeframe=None,
            data={"symbol": "btc/usd", "price": "29200.50"},
        )

        # Both should serialize to dict with same keys
        ccxt_dict = ccxt_event.model_dump()
        rtds_dict = rtds_event.model_dump()

        assert set(ccxt_dict.keys()) == set(rtds_dict.keys())
        assert ccxt_dict["ts_event"] == rtds_dict["ts_event"]
        assert ccxt_dict["source"] != rtds_dict["source"]  # Different sources
        assert "data" in ccxt_dict
        assert "data" in rtds_dict

    def test_stored_crypto_event_timeframe_optional(self):
        """Test that timeframe field is optional in StoredCryptoEvent."""
        ts = datetime.now(tz=UTC)

        # Create event without timeframe
        event_no_tf = StoredCryptoEvent(
            ts_event=ts,
            ts_ingest=ts,
            source="rtds",
            event_type="price",
            symbol="btc/usd",
            data={"price": "29200.50"},
        )

        # Create event with timeframe
        event_with_tf = StoredCryptoEvent(
            ts_event=ts,
            ts_ingest=ts,
            source="binance",
            event_type="ohlcv",
            symbol="BTC/USDT",
            timeframe="1h",
            data={"open": 29000.0},
        )

        assert event_no_tf.timeframe is None
        assert event_with_tf.timeframe == "1h"

        # Both should be valid
        assert event_no_tf.model_dump()
        assert event_with_tf.model_dump()


# ============================================================================
# PARQUET STORAGE COMPATIBILITY TESTS
# ============================================================================


class TestCCXTParquetStorage:
    """Tests for Parquet storage and compatibility between CCXT and RTDS."""

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_parquet_file_structure_and_schema(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test that Parquet files have correct structure."""
        mock_get_root.return_value = tmp_path

        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_build_exchange.return_value = mock_exchange

        mock_ohlcv = [
            [1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0],
        ]
        mock_fetch_paginated.return_value = mock_ohlcv

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance"],
        )

        fetch_historical_data(params)

        # Get the file
        storage_dir = tmp_path / "data" / "historical_ohlcv"
        parquet_files = list(storage_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

        # Read parquet file and check schema
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        # Check required columns for StoredCryptoEvent
        required_columns = {
            "ts_event",
            "ts_ingest",
            "source",
            "event_type",
            "symbol",
            "timeframe",
            "data",
        }
        assert required_columns.issubset(set(df.columns))

        # Check data types
        # Timestamps are stored as datetime64[ns] or datetime64[us, UTC]
        assert pd.api.types.is_datetime64_any_dtype(df["ts_event"])
        assert pd.api.types.is_datetime64_any_dtype(df["ts_ingest"])
        # String columns can be object or StringDtype
        assert pd.api.types.is_string_dtype(df["source"]) or df["source"].dtype == "object"
        assert pd.api.types.is_string_dtype(df["event_type"]) or df["event_type"].dtype == "object"
        assert pd.api.types.is_string_dtype(df["symbol"]) or df["symbol"].dtype == "object"
        assert pd.api.types.is_string_dtype(df["timeframe"]) or df["timeframe"].dtype == "object"

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_parquet_data_field_json_format(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test that data field is stored as JSON string in Parquet."""
        mock_get_root.return_value = tmp_path

        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_build_exchange.return_value = mock_exchange

        mock_ohlcv = [
            [1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0],
        ]
        mock_fetch_paginated.return_value = mock_ohlcv

        params = OHLCVParams(
            symbol="ETH/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance"],
        )

        fetch_historical_data(params)

        # Get the file
        storage_dir = tmp_path / "data" / "historical_ohlcv"
        parquet_files = list(storage_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

        # Read parquet file
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        # Data field should be JSON string
        assert len(df) == 1
        data_value = df["data"].iloc[0]

        # If it's a string, it should be valid JSON (stored as JSON)
        # use module-level `json`
        if isinstance(data_value, str):
            parsed = json.loads(data_value)
            assert parsed["open"] == 29000.0
            assert parsed["high"] == 29500.0
            assert parsed["low"] == 28800.0
            assert parsed["close"] == 29200.0
            assert parsed["volume"] == 100.0
        elif isinstance(data_value, dict):
            # Or it could be a dict if Parquet stores it that way
            assert data_value["open"] == 29000.0

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_parquet_timestamps_are_datetime(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test that timestamps are stored as datetime objects."""
        mock_get_root.return_value = tmp_path

        mock_exchange = Mock()
        mock_exchange.id = "binance"
        mock_build_exchange.return_value = mock_exchange

        # Create multiple events at different times
        mock_ohlcv = [
            [1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0],
            [1609462800000, 29200.0, 29600.0, 29100.0, 29400.0, 110.0],
        ]
        mock_fetch_paginated.return_value = mock_ohlcv

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance"],
        )

        fetch_historical_data(params)

        # Get the file
        storage_dir = tmp_path / "data" / "historical_ohlcv"
        parquet_files = list(storage_dir.glob("*.parquet"))
        assert len(parquet_files) == 1

        # Read parquet file
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        # Check that timestamps are different and properly ordered
        assert len(df) == 2
        assert df["ts_event"].iloc[0] < df["ts_event"].iloc[1]

        # Both should be datetime
        assert pd.api.types.is_datetime64_any_dtype(df["ts_event"])
        assert pd.api.types.is_datetime64_any_dtype(df["ts_ingest"])

    @patch("data.historical.ccxt.get_workspace_root")
    @patch("data.historical.ccxt._build_exchange")
    @patch("data.historical.ccxt._fetch_ohlcv_paginated")
    def test_multiple_exchanges_separate_files(
        self, mock_fetch_paginated, mock_build_exchange, mock_get_root, tmp_path: Path
    ):
        """Test that data from different exchanges can be written to separate files."""
        mock_get_root.return_value = tmp_path

        # Mock different exchanges
        def build_exchange_side_effect(exchange_id: str):
            mock = Mock()
            mock.id = exchange_id
            return mock

        mock_build_exchange.side_effect = build_exchange_side_effect

        # Mock different data for each exchange
        ohlcv_binance = [[1609459200000, 29000.0, 29500.0, 28800.0, 29200.0, 100.0]]
        ohlcv_coinbase = [[1609459200000, 29010.0, 29510.0, 28810.0, 29210.0, 105.0]]

        mock_fetch_paginated.side_effect = [ohlcv_binance, ohlcv_coinbase]

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=1609459200000,
            end_ms=1609466400000,
            timeframe="1h",
            limit=1000,
            exchanges=["binance", "coinbase"],
        )

        fetch_historical_data(params)

        # Check files were created
        storage_dir = tmp_path / "data" / "historical_ohlcv"
        parquet_files = list(storage_dir.glob("*.parquet"))
        assert len(parquet_files) == 1  # All data from same date goes to one file

        # Read the combined file
        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        # Should have data from both exchanges
        assert len(df) == 2
        sources = set(df["source"].unique())
        assert "binance" in sources
        assert "coinbase" in sources


# ============================================================================
# INTEGRATION TESTS WITH REAL EXCHANGES
# ============================================================================


class TestCCXTIntegration:
    """Integration tests using real CCXT exchange connections.

    These tests make actual API calls to cryptocurrency exchanges
    and verify the integration works end-to-end. Tests use small
    data ranges to minimize API calls and execution time.
    """

    @pytest.mark.integration
    def test_build_exchange_real_binance(self):
        """Test building a real Binance exchange connection."""
        if ccxt is None:
            pytest.skip("ccxt not installed")

        exchange = _build_exchange("binance")

        assert exchange is not None
        assert isinstance(exchange, ccxt.Exchange)
        assert exchange.id == "binance"
        assert hasattr(exchange, "markets")
        assert len(exchange.markets) > 0

    @pytest.mark.integration
    def test_build_exchange_real_coinbase(self):
        """Test building a real Coinbase exchange connection."""
        if ccxt is None:
            pytest.skip("ccxt not installed")

        exchange = _build_exchange("coinbase")

        assert exchange is not None
        assert isinstance(exchange, ccxt.Exchange)
        assert exchange.id == "coinbase"
        assert hasattr(exchange, "markets")

    @pytest.mark.integration
    def test_fetch_ohlcv_paginated_real_data_single_page(self):
        """Test fetching real OHLCV data that fits in one page."""
        # use module-level `UTC`, `datetime`, `timedelta`

        exchange = _build_exchange("binance")

        # Fetch just 5 hours of data (should fit in one page)
        end_time = datetime.now(tz=UTC)
        start_time = end_time - timedelta(hours=5)

        params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=int(start_time.timestamp() * 1000),
            end_ms=int(end_time.timestamp() * 1000),
            timeframe="1h",
            limit=1000,
        )

        result = _fetch_ohlcv_paginated(exchange, params, params.start_ms, params.end_ms)

        assert isinstance(result, list)
        assert len(result) > 0
        assert len(result) <= 5  # Should be around 5 hours

        # Verify OHLCV structure
        for candle in result:
            assert len(candle) == 6  # timestamp, open, high, low, close, volume
            assert isinstance(candle[0], (int, float))  # timestamp
            assert isinstance(candle[1], (int, float))  # open
            assert candle[2] >= candle[1]  # high >= open (generally true)
            assert candle[5] >= 0  # volume is non-negative

    @pytest.mark.integration
    def test_fetch_historical_data_real_single_exchange(self, tmp_path: Path):
        """Test fetching real historical data from a single exchange."""
        # use module-level `UTC`, `datetime`, `timedelta`

        with patch("data.historical.ccxt.get_workspace_root") as mock_root:
            mock_root.return_value = tmp_path

            # Fetch just 3 hours of recent data
            end_time = datetime.now(tz=UTC)
            start_time = end_time - timedelta(hours=3)

            params = OHLCVParams(
                symbol="BTC/USDT",
                start_ms=int(start_time.timestamp() * 1000),
                end_ms=int(end_time.timestamp() * 1000),
                timeframe="1h",
                limit=1000,
                exchanges=["binance"],  # Use only Binance for speed
            )

            result = fetch_historical_data(params)

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert len(result) <= 3

            # Verify DataFrame structure
            assert "ts_event" in result.columns
            assert "ts_ingest" in result.columns
            assert "source" in result.columns
            assert "event_type" in result.columns
            assert "symbol" in result.columns
            assert "timeframe" in result.columns
            assert "data" in result.columns

            # Verify data values
            assert result["source"].iloc[0] == "binance"
            assert result["event_type"].iloc[0] == "ohlcv"
            assert result["symbol"].iloc[0] == "BTC/USDT"
            assert result["timeframe"].iloc[0] == "1h"

            # Verify data dict contains OHLCV values
            data = result["data"].iloc[0]
            assert "open" in data
            assert "high" in data
            assert "low" in data
            assert "close" in data
            assert "volume" in data
            # Check all OHLCV values are floats
            ohlcv_keys = ["open", "high", "low", "close", "volume"]
            assert all(isinstance(data[k], float) for k in ohlcv_keys)

    @pytest.mark.integration
    def test_fetch_historical_data_real_file_creation(self, tmp_path: Path):
        """Test that real data fetching creates Parquet files correctly."""
        # use module-level `UTC`, `datetime`, `timedelta`

        with patch("data.historical.ccxt.get_workspace_root") as mock_root:
            mock_root.return_value = tmp_path

            # Fetch 2 hours of recent data
            end_time = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
            start_time = end_time - timedelta(hours=2)

            params = OHLCVParams(
                symbol="ETH/USDT",
                start_ms=int(start_time.timestamp() * 1000),
                end_ms=int(end_time.timestamp() * 1000),
                timeframe="1h",
                limit=1000,
                exchanges=["binance"],
            )

            result = fetch_historical_data(params)

            assert result is not None

            # Check Parquet file was created
            storage_dir = tmp_path / "data" / "historical_ohlcv"
            assert storage_dir.exists()

            parquet_files = list(storage_dir.glob("*.parquet"))
            assert len(parquet_files) > 0

            # Verify file naming convention
            for pq_file in parquet_files:
                assert "ETH_USDT" in pq_file.name
                assert "1h" in pq_file.name
                # Should have date in YYYYMMDD format
                # use module-level `re`
                assert re.search(r"\d{8}", pq_file.name)

    @pytest.mark.integration
    def test_fetch_historical_data_real_multiple_exchanges(self, tmp_path: Path):
        """Test fetching real data from multiple exchanges."""
        # use module-level `UTC`, `datetime`, `timedelta`

        with patch("data.historical.ccxt.get_workspace_root") as mock_root:
            mock_root.return_value = tmp_path

            # Fetch just 2 hours to keep test fast
            end_time = datetime.now(tz=UTC)
            start_time = end_time - timedelta(hours=2)

            params = OHLCVParams(
                symbol="BTC/USDT",
                start_ms=int(start_time.timestamp() * 1000),
                end_ms=int(end_time.timestamp() * 1000),
                timeframe="1h",
                limit=1000,
                exchanges=["binance", "coinbase"],
            )

            result = fetch_historical_data(params)

            # Note: Some exchanges may fail or return no data
            # We just verify that the function handles multiple exchanges
            assert isinstance(result, (pd.DataFrame, type(None)))

            if result is not None:
                # Check that we got data from at least one exchange
                sources = result["source"].unique()
                assert len(sources) > 0
                assert all(s in ["binance", "coinbase"] for s in sources)

    @pytest.mark.integration
    def test_fetch_historical_data_real_data_validation(self, tmp_path: Path):
        """Test that real fetched data passes basic validation."""
        # use module-level `UTC`, `datetime`, `timedelta`

        with patch("data.historical.ccxt.get_workspace_root") as mock_root:
            mock_root.return_value = tmp_path

            end_time = datetime.now(tz=UTC)
            start_time = end_time - timedelta(hours=2)

            params = OHLCVParams(
                symbol="BTC/USDT",
                start_ms=int(start_time.timestamp() * 1000),
                end_ms=int(end_time.timestamp() * 1000),
                timeframe="1h",
                limit=1000,
                exchanges=["binance"],
            )

            result = fetch_historical_data(params)

            assert result is not None

            # Validate OHLCV relationships
            for _, row in result.iterrows():
                data = row["data"]

                # High should be >= low
                assert data["high"] >= data["low"], "High should be >= low"

                # High should be >= open and close
                assert data["high"] >= data["open"], "High should be >= open"
                assert data["high"] >= data["close"], "High should be >= close"

                # Low should be <= open and close
                assert data["low"] <= data["open"], "Low should be <= open"
                assert data["low"] <= data["close"], "Low should be <= close"

                # Prices should be positive
                assert data["open"] > 0, "Open should be positive"
                assert data["high"] > 0, "High should be positive"
                assert data["low"] > 0, "Low should be positive"
                assert data["close"] > 0, "Close should be positive"
                assert data["volume"] >= 0, "Volume should be non-negative"

    @pytest.mark.integration
    def test_fetch_historical_data_real_timestamps(self, tmp_path: Path):
        """Test that timestamps in real data are correct and sorted."""
        # use module-level `UTC`, `datetime`, `timedelta`

        with patch("data.historical.ccxt.get_workspace_root") as mock_root:
            mock_root.return_value = tmp_path

            end_time = datetime.now(tz=UTC)
            start_time = end_time - timedelta(hours=3)

            params = OHLCVParams(
                symbol="BTC/USDT",
                start_ms=int(start_time.timestamp() * 1000),
                end_ms=int(end_time.timestamp() * 1000),
                timeframe="1h",
                limit=1000,
                exchanges=["binance"],
            )

            result = fetch_historical_data(params)

            assert result is not None
            assert len(result) > 0

            # Verify timestamps are within requested range
            for _, row in result.iterrows():
                ts = row["ts_event"]
                assert start_time <= ts <= end_time, "Timestamp should be within requested range"

                # ts_ingest should be recent (within last few seconds)
                assert (datetime.now(tz=UTC) - row["ts_ingest"]).total_seconds() < 60

