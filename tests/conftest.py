"""Pytest configuration and shared fixtures for streaming tests."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest

from data.reference import get_updown_asset_ids_with_slug
from data.stream import stream_crypto_prices, stream_polymarket_data
from utils import validate_crypto_price_data

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

# ============================================================================
# WEBSOCKET DATA CAPTURE FIXTURES (Session-scoped)
# ============================================================================
# These fixtures capture live WebSocket data once per test session
# and provide it to multiple tests, avoiding repeated connections

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, Any, Any]:
    """Create and manage event loop for entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def captured_crypto_events(
    event_loop: asyncio.AbstractEventLoop,
) -> list[dict[str, Any]]:
    """Capture crypto events from live stream once per session.

    Stream crypto prices for a short period and capture validated events.
    This data is reused by all crypto-related tests to avoid multiple
    WebSocket connections.
    """
    events: list[dict[str, Any]] = []

    async def capture_crypto_stream() -> None:
        """Stream crypto prices and capture events."""

        def on_crypto_event(data: dict | list) -> None:
            """Capture crypto event data from stream."""
            price_data = validate_crypto_price_data(data)
            if price_data:
                events.append(price_data)

        try:
            stream_task = asyncio.create_task(
                stream_crypto_prices(
                    symbols=["btc/usd", "eth/usd"],
                    source="chainlink",
                    callback=on_crypto_event,
                    max_retries=10,  # More retries for tests
                    retry_delay=5.0,  # Faster initial retry
                )
            )

            # Wait up to 60 seconds, checking periodically for data.
            # This allows time for retries (5s + 10s + 20s + 40s + 60s...
            # = ~135s for 5 retries) but returns early once we have
            # sufficient data.
            max_wait_time = 60
            check_interval = 2
            elapsed = 0
            min_events = 3  # Minimum events before we can return early

            while elapsed < max_wait_time:
                await asyncio.sleep(check_interval)
                elapsed += check_interval

                # Return early if we have enough data
                if len(events) >= min_events:
                    break

            # Cancel the stream
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task
        except Exception as e:
            logger.warning("Crypto stream capture failed: %s", e)
            pytest.skip(f"Crypto stream capture failed: {e}")

    # Run the capture in the event loop
    try:
        event_loop.run_until_complete(capture_crypto_stream())
    except Exception as e:
        pytest.skip(f"Failed to capture crypto events: {e}")

    return events


@pytest.fixture(scope="session")
def captured_polymarket_events(
    event_loop: asyncio.AbstractEventLoop,
) -> tuple[str, list[dict[str, Any]]]:
    """Capture polymarket events from live stream once per session.

    Stream polymarket data for a short period and capture events. Return a
    tuple of (market_slug, events_list) for tests to consume.
    """
    events: list[dict[str, Any]] = []
    market_slug = ""

    async def capture_polymarket_stream() -> None:
        """Stream polymarket data and capture events."""
        nonlocal market_slug

        def on_market_event(data: dict | list) -> None:
            """Capture market events."""
            if isinstance(data, dict):
                events.append(data)

        try:
            # Get current market asset IDs
            utctime = int(datetime.now(tz=UTC).timestamp())
            market_slug, asset_ids = get_updown_asset_ids_with_slug(
                utctime=utctime, resolution="5m"
            )

            if not asset_ids:
                pytest.skip("No asset IDs available for current time")

            stream_task = asyncio.create_task(
                stream_polymarket_data(
                    asset_ids=asset_ids,
                    enable_custom_features=True,
                    callback=on_market_event,
                )
            )

            # Wait up to 60 seconds, checking periodically for data.
            # This allows time for retries (10s + 20s + 40s = 70s
            # for 3 retries) but returns early once we have sufficient
            # data.
            max_wait_time = 60
            check_interval = 2
            elapsed = 0
            min_events = 5  # Minimum events before we can return early

            while elapsed < max_wait_time:
                await asyncio.sleep(check_interval)
                elapsed += check_interval

                # Return early if we have enough data
                if len(events) >= min_events:
                    break

            # Cancel the stream
            stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stream_task
        except Exception as e:
            pytest.skip(f"Polymarket stream capture failed: {e}")

    # Run the capture in the event loop
    try:
        event_loop.run_until_complete(capture_polymarket_stream())
    except Exception as e:
        pytest.skip(f"Failed to capture polymarket events: {e}")

    return market_slug, events

# ============================================================================
# SHARED FIXTURES
# ============================================================================


@pytest.fixture
def temp_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for Parquet storage testing."""
    storage_dir = tmp_path / "stream_feeds"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def current_timestamp_ms() -> int:
    """Return current timestamp in milliseconds."""
    return int(datetime.now(UTC).timestamp() * 1000)


# ============================================================================
# POLYMARKET EVENT MESSAGE FACTORIES
# ============================================================================


@pytest.fixture
def polymarket_orderbook_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Polymarket orderbook (book) event message."""
    return {
        "event_type": "book",
        "asset_id": "0x123abc",
        "market": "0xmarketid",
        "timestamp": current_timestamp_ms,
        "hash": "0xhash123",
        "bids": [
            {"price": "0.65", "size": "100.5"},
            {"price": "0.64", "size": "200.0"},
            {"price": "0.63", "size": "150.25"},
        ],
        "asks": [
            {"price": "0.66", "size": "120.0"},
            {"price": "0.67", "size": "250.5"},
            {"price": "0.68", "size": "300.0"},
        ],
    }


@pytest.fixture
def polymarket_price_change_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Polymarket price_change event message."""
    return {
        "event_type": "price_change",
        "market": "0xmarketid",
        "timestamp": current_timestamp_ms,
        "price_changes": [
            {
                "asset_id": "0x123abc",
                "price": "0.65",
                "size": "100.0",
                "side": "BUY",
                "hash": "0xhash1",
                "best_bid": "0.65",
                "best_ask": "0.66",
            },
            {
                "asset_id": "0x456def",
                "price": "0.35",
                "size": "50.0",
                "side": "SELL",
                "hash": "0xhash2",
                "best_bid": "0.34",
                "best_ask": "0.35",
            },
        ],
    }


@pytest.fixture
def polymarket_trade_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Polymarket last_trade_price event message."""
    return {
        "event_type": "last_trade_price",
        "asset_id": "0x123abc",
        "market": "0xmarketid",
        "timestamp": current_timestamp_ms,
        "price": "0.65",
        "size": "50.0",
        "side": "BUY",
        "fee_rate_bps": "100",
    }


@pytest.fixture
def polymarket_tick_change_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Polymarket tick_size_change event message."""
    return {
        "event_type": "tick_size_change",
        "asset_id": "0x123abc",
        "market": "0xmarketid",
        "timestamp": current_timestamp_ms,
        "old_tick_size": "0.01",
        "new_tick_size": "0.001",
    }


@pytest.fixture
def polymarket_best_bid_ask_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Polymarket best_bid_ask event message."""
    return {
        "event_type": "best_bid_ask",
        "asset_id": "0x123abc",
        "market": "0xmarketid",
        "timestamp": current_timestamp_ms,
        "best_bid": "0.65",
        "best_ask": "0.66",
        "spread": "0.01",
    }


@pytest.fixture
def polymarket_new_market_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Polymarket new_market event message."""
    return {
        "event_type": "new_market",
        "market": "0xmarketid",
        "timestamp": current_timestamp_ms,
        "id": "0xmarketid",
        "question": "Will Bitcoin reach $100,000 by 2025?",
        "slug": "bitcoin-100k-2025",
        "description": "Bitcoin reaches $100,000 USD",
        "assets_ids": ["0x123abc", "0x456def"],
        "outcomes": ["YES", "NO"],
        "event_message": {
            "clob_token_ids": ["0x123abc", "0x456def"],
            "outcome_prices": ["0.65", "0.35"],
        },
    }


@pytest.fixture
def polymarket_market_resolved_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Polymarket market_resolved event message."""
    return {
        "event_type": "market_resolved",
        "market": "0xmarketid",
        "timestamp": current_timestamp_ms,
        "id": "0xmarketid",
        "question": "Will Bitcoin reach $100,000 by 2025?",
        "slug": "bitcoin-100k-2025",
        "description": "Bitcoin reaches $100,000 USD",
        "assets_ids": ["0x123abc", "0x456def"],
        "outcomes": ["YES", "NO"],
        "winning_asset_id": "0x123abc",
        "winning_outcome": "YES",
        "event_message": {"resolution": "YES"},
    }


# ============================================================================
# CRYPTO PRICE EVENT MESSAGE FACTORIES
# ============================================================================


@pytest.fixture
def crypto_binance_single_price_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Binance crypto price message (single value)."""
    return {
        "topic": "crypto_prices",
        "type": "update",
        "timestamp": current_timestamp_ms,
        "payload": {
            "symbol": "btcusdt",
            "value": "98500.50",
            "timestamp": current_timestamp_ms - 100,
        },
    }


@pytest.fixture
def crypto_binance_multiple_prices_message(
    current_timestamp_ms: int,
) -> dict[str, Any]:
    """Generate multiple Binance crypto prices in one message."""
    return {
        "topic": "crypto_prices",
        "type": "update",
        "timestamp": current_timestamp_ms,
        "payload": [
            {
                "symbol": "btcusdt",
                "value": "98500.50",
                "timestamp": current_timestamp_ms - 100,
            },
            {
                "symbol": "ethusdt",
                "value": "3200.75",
                "timestamp": current_timestamp_ms - 100,
            },
        ],
    }


@pytest.fixture
def crypto_chainlink_single_price_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Chainlink crypto price message (from RTDS)."""
    return {
        "topic": "crypto_prices_chainlink",
        "type": "price",
        "timestamp": current_timestamp_ms,
        "payload": {
            "symbol": "btc/usd",
            "value": "98500.50",
            "timestamp": current_timestamp_ms - 50,
        },
    }


@pytest.fixture
def crypto_chainlink_data_series_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a mock Chainlink crypto price with data series (multiple points)."""
    return {
        "topic": "crypto_prices_chainlink",
        "type": "price",
        "timestamp": current_timestamp_ms,
        "payload": {
            "symbol": "eth/usd",
            "value": "3200.75",  # This might be overridden by data series
            "timestamp": current_timestamp_ms - 50,
            "data": [
                {"value": "3198.00", "timestamp": current_timestamp_ms - 3000},
                {"value": "3199.50", "timestamp": current_timestamp_ms - 2000},
                {"value": "3200.75", "timestamp": current_timestamp_ms - 50},
            ],
        },
    }


@pytest.fixture
def crypto_chainlink_with_volume_message(current_timestamp_ms: int) -> dict[str, Any]:
    """Generate a Chainlink crypto price with volume and change metrics."""
    return {
        "topic": "crypto_prices_chainlink",
        "type": "price",
        "timestamp": current_timestamp_ms,
        "payload": {
            "symbol": "sol/usd",
            "value": "175.50",
            "timestamp": current_timestamp_ms - 50,
            "volume_24h": "500000000.50",
            "change_24h": "12.45",
        },
    }


# ============================================================================
# RAW EVENT DATA (for direct model construction)
# ============================================================================


@pytest.fixture
def raw_book_event_data(current_timestamp_ms: int) -> dict[str, Any]:
    """Raw data for BookEvent model validation."""
    return {
        "event_type": "book",
        "asset_id": "0x123abc",
        "market": "0xmarketid",
        "timestamp": current_timestamp_ms,
        "hash": "0xhash123",
        "bids": [["0.65", "100.5"], ["0.64", "200.0"]],  # list of lists format
        "asks": [["0.66", "120.0"], ["0.67", "250.5"]],
    }


@pytest.fixture
def raw_crypto_price_event_data(current_timestamp_ms: int) -> dict[str, Any]:
    """Raw data for CryptoPriceEvent model validation."""
    return {
        "symbol": "btc/usd",
        "price": "98500.50",
        "timestamp": current_timestamp_ms,
        "source": "chainlink",
        "volume_24h": "25000000000.00",
        "change_24h": "5.25",
    }


# ============================================================================
# STORED EVENT FIXTURES (for storage tests)
# ============================================================================


@pytest.fixture
def stored_polymarket_event_dict(current_timestamp_ms: int) -> dict[str, Any]:
    """Create a dictionary representing a StoredPolymarketEvent."""
    ts_event = datetime.fromtimestamp(current_timestamp_ms / 1000.0, tz=UTC)
    ts_ingest = datetime.now(UTC)

    return {
        "ts_event": ts_event,
        "ts_ingest": ts_ingest,
        "source": "polymarket",
        "event_type": "book",
        "market_id": "0xmarketid",
        "token_id": "0x123abc",
        "market": "bitcoin-100k-2025",
        "data": {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": current_timestamp_ms,
            "bids": [{"price": "0.65", "size": "100.0"}],
            "asks": [{"price": "0.66", "size": "120.0"}],
        },
    }


@pytest.fixture
def stored_crypto_event_dict(current_timestamp_ms: int) -> dict[str, Any]:
    """Create a dictionary representing a StoredCryptoEvent."""
    ts_event = datetime.fromtimestamp(current_timestamp_ms / 1000.0, tz=UTC)
    ts_ingest = datetime.now(UTC)

    return {
        "ts_event": ts_event,
        "ts_ingest": ts_ingest,
        "source": "polymarket_rtds",
        "event_type": "price",
        "symbol": "btc/usd",
        "timeframe": None,
        "data": {
            "symbol": "btc/usd",
            "price": "98500.50",
            "timestamp": current_timestamp_ms,
            "source": "chainlink",
        },
    }


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def valid_market_channel_config() -> dict[str, Any]:
    """Return valid MarketChannelConfig parameters."""
    return {
        "asset_ids": ["0x123abc", "0x456def", "0x789ghi"],
        "max_retries": 5,
        "retry_delay": 10.0,
        "custom_feature_enabled": True,
    }


@pytest.fixture
def valid_crypto_price_config_chainlink() -> dict[str, Any]:
    """Return a valid CryptoPriceConfig for the Chainlink source."""
    return {
        "symbols": ["btc/usd", "eth/usd", "sol/usd"],
        "source": "chainlink",
        "max_retries": 5,
        "retry_delay": 10.0,
    }


@pytest.fixture
def valid_crypto_price_config_binance() -> dict[str, Any]:
    """Return a valid CryptoPriceConfig for the Binance source."""
    return {
        "symbols": ["btcusdt", "ethusdt", "solusdt"],
        "source": "binance",
        "max_retries": 5,
        "retry_delay": 10.0,
    }


@pytest.fixture
def valid_crypto_price_config_no_symbols() -> dict[str, Any]:
    """Return a CryptoPriceConfig with no symbol filtering."""
    return {
        "symbols": None,
        "source": "chainlink",
        "max_retries": 3,
        "retry_delay": 5.0,
    }


# ============================================================================
# ASSERTION HELPERS
# ============================================================================


@pytest.fixture
def assert_valid_polymarket_event():
    """Return a factory to validate Polymarket event structure."""

    def _assert(event: dict[str, Any], expected_type: str) -> None:
        assert "event_type" in event
        assert event["event_type"] == expected_type
        assert "market" in event or "asset_id" in event
        assert "timestamp" in event
        assert isinstance(event["timestamp"], int)

    return _assert


@pytest.fixture
def assert_valid_crypto_event():
    """Return a factory to validate Crypto event structure."""

    def _assert(event: dict[str, Any]) -> None:
        assert "symbol" in event
        assert "price" in event or "value" in event
        assert "timestamp" in event
        assert isinstance(event["timestamp"], int)

    return _assert


# ============================================================================
# LIVE TEST RATE LIMITING FIXTURE
# ============================================================================


@pytest.fixture(autouse=True)
def live_test_delay(request: Any) -> Generator[None, Any, Any]:
    """Introduce delay before live tests to avoid rate limiting.

    This fixture automatically adds a 5-second delay before any test
    marked with @pytest.mark.live. This helps prevent rate limiting
    when multiple live WebSocket tests run consecutively.
    """
    # Use module-level `time` import

    # Check if test is marked as live
    if request.node.get_closest_marker("live"):
        # Add delay to avoid rate limiting
        time.sleep(5)

        # For async tests, yield to give async setup time
        yield

        # Add delay after live test as well to ensure rate limit recovery
        time.sleep(2)
    else:
        yield
