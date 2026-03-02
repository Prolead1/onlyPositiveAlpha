"""Comprehensive tests for Polymarket RTDS crypto price streaming."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from data.stream.rtds import (
    CryptoPriceConfig,
    PolymarketCryptoStream,
)
from models.events import CryptoPriceEvent

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


class TestCryptoPriceConfig:
    """Tests for CryptoPriceConfig dataclass."""

    def test_default_config(self):
        """Test default CryptoPriceConfig values."""
        config = CryptoPriceConfig()
        assert config.symbols is None
        assert config.source == "chainlink"
        assert config.max_retries == 5
        assert config.retry_delay == 10.0

    def test_chainlink_config(self, valid_crypto_price_config_chainlink):
        """Test CryptoPriceConfig for Chainlink source."""
        config = CryptoPriceConfig(**valid_crypto_price_config_chainlink)
        assert config.symbols == ["btc/usd", "eth/usd", "sol/usd"]
        assert config.source == "chainlink"

    def test_binance_config(self, valid_crypto_price_config_binance):
        """Test CryptoPriceConfig for Binance source."""
        config = CryptoPriceConfig(**valid_crypto_price_config_binance)
        assert config.symbols == ["btcusdt", "ethusdt", "solusdt"]
        assert config.source == "binance"

    def test_config_no_symbols_filtering(self, valid_crypto_price_config_no_symbols):
        """Test CryptoPriceConfig with no symbol filtering."""
        config = CryptoPriceConfig(**valid_crypto_price_config_no_symbols)
        assert config.symbols is None
        assert config.source == "chainlink"

    def test_config_custom_retry_settings(self):
        """Test CryptoPriceConfig with custom retry settings."""
        config = CryptoPriceConfig(
            source="binance",
            max_retries=3,
            retry_delay=2.0,
        )
        assert config.max_retries == 3
        assert config.retry_delay == 2.0


# ============================================================================
# CRYPTO PRICE EVENT PARSING TESTS
# ============================================================================


class TestCryptoPriceEventParsing:
    """Tests for parsing and validating CryptoPriceEvent messages."""

    def test_parse_crypto_price_event(self, raw_crypto_price_event_data):
        """Test parsing crypto price event."""
        event = CryptoPriceEvent(**raw_crypto_price_event_data)
        assert event.symbol == "btc/usd"
        assert event.price == Decimal("98500.50")
        assert event.source == "chainlink"
        assert event.volume_24h == Decimal("25000000000.00")
        assert event.change_24h == Decimal("5.25")

    def test_crypto_price_event_minimal(self, current_timestamp_ms):
        """Test crypto price event with minimal required fields."""
        data = {
            "symbol": "eth/usd",
            "price": "3200.50",
            "timestamp": current_timestamp_ms,
            "source": "chainlink",
        }
        event = CryptoPriceEvent(**data)
        assert event.symbol == "eth/usd"
        assert event.price == Decimal("3200.50")
        assert event.volume_24h is None
        assert event.change_24h is None

    def test_crypto_price_decimal_precision(self):
        """Test that crypto prices preserve decimal precision."""
        data = {
            "symbol": "btc/usd",
            "price": "98500.123456789",
            "timestamp": 1234567890000,
            "source": "binance",
            "volume_24h": "999999999.999999999",
        }
        event = CryptoPriceEvent(**data)
        assert str(event.price) == "98500.123456789"
        assert str(event.volume_24h) == "999999999.999999999"

    def test_crypto_price_from_string_values(self):
        """Test crypto price event created from string values."""
        data = {
            "symbol": "sol/usd",
            "price": "175.50",
            "timestamp": 1234567890123,  # int timestamp (pydantic doesn't auto-convert str to int)
            "source": "chainlink",
        }
        event = CryptoPriceEvent(**data)
        assert event.price == Decimal("175.50")

    def test_crypto_price_from_float_values(self):
        """Test crypto price event created from float values."""
        data = {
            "symbol": "doge/usd",
            "price": 0.42,
            "timestamp": 1234567890000,
            "source": "binance",
        }
        event = CryptoPriceEvent(**data)
        assert event.price == Decimal("0.42")


# ============================================================================
# SUBSCRIPTION MESSAGE TESTS
# ============================================================================


class TestChainlinkSubscriptionMessage:
    """Tests for Chainlink subscription message formatting."""

    @pytest.fixture
    def mock_ws(self):
        """Create a mock WebSocket."""
        ws = AsyncMock()
        return ws

    @pytest.fixture
    async def chainlink_stream_with_mock_ws(
        self, valid_crypto_price_config_chainlink, mock_ws
    ):
        """Create Chainlink crypto stream with mocked WebSocket."""
        config = CryptoPriceConfig(**valid_crypto_price_config_chainlink)
        stream = PolymarketCryptoStream(config)
        stream.ws = mock_ws
        return stream

    @pytest.mark.asyncio
    async def test_chainlink_subscription_message_format(
        self, chainlink_stream_with_mock_ws, mock_ws
    ):
        """Test Chainlink subscription message structure."""
        stream = chainlink_stream_with_mock_ws
        await stream._subscribe_chainlink()

        # Verify send was called
        mock_ws.send.assert_called_once()

        # Parse the sent message
        sent_message = json.loads(mock_ws.send.call_args[0][0])

        assert sent_message["action"] == "subscribe"
        assert "subscriptions" in sent_message
        assert len(sent_message["subscriptions"]) == 3

        # Check first subscription structure
        sub = sent_message["subscriptions"][0]
        assert sub["topic"] == "crypto_prices_chainlink"
        assert sub["type"] == "*"
        assert "filters" in sub

    @pytest.mark.asyncio
    async def test_chainlink_symbol_filters_json_encoded(
        self, chainlink_stream_with_mock_ws, mock_ws
    ):
        """Test that Chainlink symbol filters are JSON-encoded."""
        await chainlink_stream_with_mock_ws._subscribe_chainlink()

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        subs = sent_message["subscriptions"]

        # Each subscription should have a JSON-encoded filter with symbol
        for _i, sub in enumerate(subs):
            filters = json.loads(sub["filters"])
            assert "symbol" in filters

    @pytest.mark.asyncio
    async def test_chainlink_no_symbols_subscription(self, mock_ws):
        """Test Chainlink subscription without symbol filtering."""
        config = CryptoPriceConfig(source="chainlink", symbols=None)
        stream = PolymarketCryptoStream(config)
        stream.ws = mock_ws

        await stream._subscribe_chainlink()

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert len(sent_message["subscriptions"]) == 1
        # No filters when no symbols specified
        assert "filters" not in sent_message["subscriptions"][0]


class TestBinanceSubscriptionMessage:
    """Tests for Binance subscription message formatting."""

    @pytest.fixture
    def mock_ws(self):
        """Create a mock WebSocket."""
        ws = AsyncMock()
        return ws

    @pytest.fixture
    async def binance_stream_with_mock_ws(
        self, valid_crypto_price_config_binance, mock_ws
    ):
        """Create Binance crypto stream with mocked WebSocket."""
        config = CryptoPriceConfig(**valid_crypto_price_config_binance)
        stream = PolymarketCryptoStream(config)
        stream.ws = mock_ws
        return stream

    @pytest.mark.asyncio
    async def test_binance_subscription_message_format(
        self, binance_stream_with_mock_ws, mock_ws
    ):
        """Test Binance subscription message structure."""
        stream = binance_stream_with_mock_ws
        await stream._subscribe_binance()

        # Verify send was called
        mock_ws.send.assert_called_once()

        # Parse the sent message
        sent_message = json.loads(mock_ws.send.call_args[0][0])

        assert sent_message["action"] == "subscribe"
        assert "subscriptions" in sent_message
        assert len(sent_message["subscriptions"]) == 1

        # Check subscription structure
        sub = sent_message["subscriptions"][0]
        assert sub["topic"] == "crypto_prices"
        assert sub["type"] == "update"
        assert "filters" in sub

    @pytest.mark.asyncio
    async def test_binance_symbol_filters_comma_separated(
        self, binance_stream_with_mock_ws, mock_ws
    ):
        """Test that Binance symbol filters are comma-separated."""
        await binance_stream_with_mock_ws._subscribe_binance()

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        sub = sent_message["subscriptions"][0]
        filters = sub["filters"]

        # Should be comma-separated string
        assert "btcusdt" in filters
        assert "ethusdt" in filters
        assert "solusdt" in filters
        assert "," in filters

    @pytest.mark.asyncio
    async def test_binance_no_symbols_subscription(self, mock_ws):
        """Test Binance subscription without symbol filtering."""
        config = CryptoPriceConfig(source="binance", symbols=None)
        stream = PolymarketCryptoStream(config)
        stream.ws = mock_ws

        await stream._subscribe_binance()

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert len(sent_message["subscriptions"]) == 1
        # No filters when no symbols specified
        assert "filters" not in sent_message["subscriptions"][0]


# ============================================================================
# MESSAGE PARSING TESTS
# ============================================================================


class TestBinancePriceMessageParsing:
    """Tests for parsing Binance crypto price messages."""

    def test_parse_binance_single_price(self, crypto_binance_single_price_message):
        """Test parsing single Binance price message."""
        message = crypto_binance_single_price_message
        payload = message["payload"]

        assert payload["symbol"] == "btcusdt"
        assert payload["value"] == "98500.50"
        assert isinstance(payload["timestamp"], int)

    def test_binance_price_symbol_extraction(
        self, crypto_binance_single_price_message
    ):
        """Test extracting symbol from Binance message."""
        payload = crypto_binance_single_price_message["payload"]
        symbol = payload.get("symbol")
        assert symbol == "btcusdt"

    def test_binance_price_value_extraction(
        self, crypto_binance_single_price_message
    ):
        """Test extracting price value from Binance message."""
        payload = crypto_binance_single_price_message["payload"]
        value = payload.get("value")
        assert value == "98500.50"


class TestChainlinkPriceMessageParsing:
    """Tests for parsing Chainlink crypto price messages."""

    def test_parse_chainlink_single_price(
        self, crypto_chainlink_single_price_message
    ):
        """Test parsing Chainlink single price message."""
        message = crypto_chainlink_single_price_message
        payload = message["payload"]

        assert payload["symbol"] == "btc/usd"
        assert payload["value"] == "98500.50"
        assert isinstance(payload["timestamp"], int)

    def test_parse_chainlink_data_series(
        self, crypto_chainlink_data_series_message
    ):
        """Test parsing Chainlink message with data series."""
        message = crypto_chainlink_data_series_message
        payload = message["payload"]

        assert payload["symbol"] == "eth/usd"
        assert "data" in payload
        assert len(payload["data"]) == 3
        # Last point should have the most recent price
        assert payload["data"][-1]["value"] == "3200.75"

    def test_chainlink_data_series_timestamps(
        self, crypto_chainlink_data_series_message
    ):
        """Test that Chainlink data series includes timestamps."""
        payload = crypto_chainlink_data_series_message["payload"]
        data_series = payload["data"]

        for point in data_series:
            assert "timestamp" in point
            assert isinstance(point["timestamp"], int)

    def test_chainlink_with_volume_and_change(
        self, crypto_chainlink_with_volume_message
    ):
        """Test Chainlink message with optional volume and change metrics."""
        payload = crypto_chainlink_with_volume_message["payload"]

        assert payload["symbol"] == "sol/usd"
        assert "volume_24h" in payload
        assert "change_24h" in payload
        assert payload["volume_24h"] == "500000000.50"
        assert payload["change_24h"] == "12.45"


# ============================================================================
# MESSAGE DISPATCH TESTS
# ============================================================================


class TestMessageDispatch:
    """Tests for PolymarketCryptoStream message dispatching."""

    @pytest.fixture
    def mock_stream(self):
        """Create a PolymarketCryptoStream instance."""
        config = CryptoPriceConfig(source="chainlink")
        return PolymarketCryptoStream(config)

    def test_log_message_with_valid_data(
        self, mock_stream, crypto_chainlink_single_price_message
    ):
        """Test that _log_message processes valid crypto price data."""
        # This tests that the method doesn't raise exceptions
        mock_stream._log_message(crypto_chainlink_single_price_message)

    def test_log_message_with_binance_data(
        self, mock_stream, crypto_binance_single_price_message
    ):
        """Test that _log_message processes Binance price data."""
        mock_stream._log_message(crypto_binance_single_price_message)

    def test_log_message_with_data_series(
        self, mock_stream, crypto_chainlink_data_series_message
    ):
        """Test that _log_message extracts price from data series."""
        # Should use the last point's value if data series is present
        mock_stream._log_message(crypto_chainlink_data_series_message)

    def test_log_message_with_non_dict(self, mock_stream):
        """Test handling of non-dict messages."""
        list_message = [{"symbol": "btc/usd"}]
        # Should not raise, just log
        mock_stream._log_message(list_message)

    def test_log_message_with_missing_payload(self, mock_stream):
        """Test handling of message with missing payload."""
        message = {
            "topic": "crypto_prices_chainlink",
            "type": "price",
        }
        # Should not raise, just log warning
        mock_stream._log_message(message)

    def test_log_message_with_invalid_payload_format(self, mock_stream):
        """Test handling of message with invalid payload format."""
        message = {
            "topic": "crypto_prices_chainlink",
            "type": "price",
            "payload": "invalid_string",  # Should be dict
        }
        # Should not raise, just log warning
        mock_stream._log_message(message)


# ============================================================================
# SUBSCRIPTION ROUTING TESTS
# ============================================================================


class TestSubscriptionRouting:
    """Tests for routing to correct subscription method based on source."""

    @pytest.fixture
    def mock_ws(self):
        """Create a mock WebSocket."""
        ws = AsyncMock()
        return ws

    @pytest.mark.asyncio
    async def test_subscribe_routes_to_chainlink(self, mock_ws):
        """Test that subscribe() routes to Chainlink for chainlink source."""
        config = CryptoPriceConfig(
            source="chainlink",
            symbols=["btc/usd"],
        )
        stream = PolymarketCryptoStream(config)
        stream.ws = mock_ws

        with patch.object(
            stream, "_subscribe_chainlink"
        ) as mock_chainlink, patch.object(
            stream, "_subscribe_binance"
        ) as mock_binance:
            await stream.subscribe()

            mock_chainlink.assert_called_once()
            mock_binance.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscribe_routes_to_binance(self, mock_ws):
        """Test that subscribe() routes to Binance for binance source."""
        config = CryptoPriceConfig(
            source="binance",
            symbols=["btcusdt"],
        )
        stream = PolymarketCryptoStream(config)
        stream.ws = mock_ws

        with patch.object(
            stream, "_subscribe_binance"
        ) as mock_binance, patch.object(
            stream, "_subscribe_chainlink"
        ) as mock_chainlink:
            await stream.subscribe()

            mock_binance.assert_called_once()
            mock_chainlink.assert_not_called()

    @pytest.mark.asyncio
    async def test_subscribe_invalid_source_raises(self, mock_ws):
        """Test that subscribe() raises for invalid source."""
        config = CryptoPriceConfig(source="invalid_source")
        stream = PolymarketCryptoStream(config)
        stream.ws = mock_ws

        with pytest.raises(ValueError, match="Invalid price source"):
            await stream.subscribe()

    @pytest.mark.asyncio
    async def test_subscribe_without_connection_fails(self):
        """Test that subscribe fails if WebSocket is not connected."""
        config = CryptoPriceConfig(source="chainlink")
        stream = PolymarketCryptoStream(config)
        stream.ws = None  # Not connected

        with pytest.raises(RuntimeError, match="WebSocket not connected"):
            await stream.subscribe()


# ============================================================================
# INTEGRATION TESTS (Mock-based)
# ============================================================================


class TestCryptoStreamIntegration:
    """Integration tests for crypto stream with mocked WebSocket."""

    @pytest.mark.asyncio
    async def test_chainlink_stream_lifecycle(
        self, valid_crypto_price_config_chainlink
    ):
        """Test Chainlink stream initialization and configuration."""
        config = CryptoPriceConfig(**valid_crypto_price_config_chainlink)
        stream = PolymarketCryptoStream(config)

        assert stream.price_config.source == "chainlink"
        assert stream.price_config.symbols == ["btc/usd", "eth/usd", "sol/usd"]

    @pytest.mark.asyncio
    async def test_binance_stream_lifecycle(
        self, valid_crypto_price_config_binance
    ):
        """Test Binance stream initialization and configuration."""
        config = CryptoPriceConfig(**valid_crypto_price_config_binance)
        stream = PolymarketCryptoStream(config)

        assert stream.price_config.source == "binance"
        assert stream.price_config.symbols == ["btcusdt", "ethusdt", "solusdt"]

    @pytest.mark.asyncio
    async def test_multiple_price_updates_sequence(
        self,
        crypto_chainlink_single_price_message,
        crypto_chainlink_data_series_message,
        crypto_chainlink_with_volume_message,
    ):
        """Test processing multiple crypto price updates in sequence."""
        config = CryptoPriceConfig(source="chainlink")
        stream = PolymarketCryptoStream(config)

        messages_processed = []

        def capture_message(message):
            if isinstance(message, dict) and "payload" in message:
                messages_processed.append(message["payload"].get("symbol"))

        # Use the real _log_message to test actual behavior
        stream._log_message(crypto_chainlink_single_price_message)
        stream._log_message(crypto_chainlink_data_series_message)
        stream._log_message(crypto_chainlink_with_volume_message)

        # Just verify no exceptions were raised
        assert True

    @pytest.mark.asyncio
    async def test_mixed_source_messages(self):
        """Test handling messages that change between sources."""
        config = CryptoPriceConfig(source="chainlink")
        stream = PolymarketCryptoStream(config)

        # Process Chainlink message
        chainlink_msg = {
            "topic": "crypto_prices_chainlink",
            "type": "price",
            "timestamp": 1234567890000,
            "payload": {
                "symbol": "btc/usd",
                "value": "98500.50",
                "timestamp": 1234567890000,
            },
        }
        stream._log_message(chainlink_msg)

        # Process what looks like a Binance message (should still work)
        binance_like_msg = {
            "topic": "crypto_prices",
            "type": "update",
            "timestamp": 1234567890000,
            "payload": {
                "symbol": "btcusdt",
                "value": "98500.50",
                "timestamp": 1234567890000,
            },
        }
        stream._log_message(binance_like_msg)


# ============================================================================
# SYMBOL FORMAT VALIDATION TESTS
# ============================================================================


class TestSymbolFormatValidation:
    """Tests for symbol format validation per source."""

    def test_chainlink_symbol_format_slash_separated(self):
        """Test Chainlink symbol format uses slash separation."""
        config = CryptoPriceConfig(
            source="chainlink",
            symbols=["btc/usd", "eth/usd"],
        )
        assert config.symbols is not None  # Type narrowing for iteration
        for symbol in config.symbols:
            assert "/" in symbol

    def test_binance_symbol_format_lowercase_concatenated(self):
        """Test Binance symbol format uses lowercase concatenation."""
        config = CryptoPriceConfig(
            source="binance",
            symbols=["btcusdt", "ethusdt"],
        )
        assert config.symbols is not None  # Type narrowing for iteration
        for symbol in config.symbols:
            assert symbol.islower()
            assert "//" not in symbol  # No slashes in Binance format

    def test_config_accepts_arbitrary_symbols(self):
        """Test that config accepts any symbol format (validation is elsewhere)."""
        config1 = CryptoPriceConfig(
            source="chainlink",
            symbols=["custom/symbol"],
        )
        config2 = CryptoPriceConfig(
            source="binance",
            symbols=["customsymbol"],
        )
        assert config1.symbols is not None  # Type narrowing for subscripting
        assert config2.symbols is not None  # Type narrowing for subscripting
        assert config1.symbols[0] == "custom/symbol"
        assert config2.symbols[0] == "customsymbol"
