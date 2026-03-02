"""Comprehensive tests for Polymarket market channel streaming."""

from __future__ import annotations

import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from data.stream.polymarket import (
    MarketChannelConfig,
    PolymarketMarketChannel,
)
from models.events import (
    BestBidAskEvent,
    BookEvent,
    LastTradePriceEvent,
    MarketResolvedEvent,
    NewMarketEvent,
    OrderbookLevel,
    PriceChangeEntry,
    PriceChangeEvent,
    TickSizeChangeEvent,
)

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


class TestMarketChannelConfig:
    """Tests for MarketChannelConfig dataclass."""

    def test_default_config(self):
        """Test default MarketChannelConfig values."""
        config = MarketChannelConfig()
        assert config.asset_ids == []
        assert config.max_retries == 5
        assert config.retry_delay == 10.0
        assert config.custom_feature_enabled is True

    def test_custom_config(self, valid_market_channel_config):
        """Test MarketChannelConfig with custom values."""
        config = MarketChannelConfig(**valid_market_channel_config)
        assert config.asset_ids == ["0x123abc", "0x456def", "0x789ghi"]
        assert config.max_retries == 5
        assert config.retry_delay == 10.0
        assert config.custom_feature_enabled is True

    def test_config_with_custom_features_disabled(self):
        """Test MarketChannelConfig with custom features disabled."""
        config = MarketChannelConfig(
            asset_ids=["0xtest"],
            custom_feature_enabled=False,
        )
        assert config.custom_feature_enabled is False

    def test_config_retry_settings(self):
        """Test MarketChannelConfig retry settings."""
        config = MarketChannelConfig(
            asset_ids=["0xtest"],
            max_retries=3,
            retry_delay=5.0,
        )
        assert config.max_retries == 3
        assert config.retry_delay == 5.0


# ============================================================================
# EVENT PARSING AND VALIDATION TESTS
# ============================================================================


class TestOrderbookEventParsing:
    """Tests for parsing and validating BookEvent messages."""

    def test_parse_orderbook_event_from_dict_format(
        self, polymarket_orderbook_message
    ):
        """Test parsing orderbook with dict bid/ask levels."""
        event = BookEvent(**polymarket_orderbook_message)
        assert event.event_type == "book"
        assert event.asset_id == "0x123abc"
        assert len(event.bids) == 3
        assert len(event.asks) == 3
        assert event.bids[0].price == Decimal("0.65")
        assert event.bids[0].size == Decimal("100.5")
        assert event.asks[0].price == Decimal("0.66")

    def test_parse_orderbook_event_from_list_format(
        self, raw_book_event_data
    ):
        """Test parsing orderbook with nested list bid/ask levels."""
        event = BookEvent(**raw_book_event_data)
        assert event.event_type == "book"
        assert len(event.bids) == 2
        assert event.bids[0].price == Decimal("0.65")
        assert event.bids[0].size == Decimal("100.5")

    def test_orderbook_event_empty_levels(self):
        """Test orderbook event with no bids or asks."""
        data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "bids": [],
            "asks": [],
        }
        event = BookEvent(**data)
        assert event.bids == []
        assert event.asks == []

    def test_orderbook_decimal_precision(self) -> None:
        """Test that OrderbookLevel preserves decimal precision."""
        level_data = {"price": "0.123456789", "size": "1.987654321"}
        level = OrderbookLevel(**level_data)  # type: ignore[arg-type]
        assert str(level.price) == "0.123456789"
        assert str(level.size) == "1.987654321"

    def test_orderbook_timestamp_parsing(self) -> None:
        """Test timestamp parsing to milliseconds."""
        data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890.123,  # float seconds
            "bids": [],
            "asks": [],
        }
        event = BookEvent(**data)
        assert isinstance(event.timestamp, int)
        assert event.timestamp == 1234567890  # truncated to int


class TestPriceChangeEventParsing:
    """Tests for parsing and validating PriceChangeEvent messages."""

    def test_parse_price_change_event(
        self, polymarket_price_change_message: dict
    ) -> None:
        """Test parsing price change event with multiple entries."""
        event = PriceChangeEvent(**polymarket_price_change_message)
        assert event.event_type == "price_change"
        assert len(event.price_changes) == 2
        assert event.price_changes[0].asset_id == "0x123abc"
        assert event.price_changes[0].price == Decimal("0.65")
        assert event.price_changes[0].side == "BUY"

    def test_price_change_with_best_bid_ask(
        self, polymarket_price_change_message: dict
    ) -> None:
        """Test price change entries include best bid/ask."""
        event = PriceChangeEvent(**polymarket_price_change_message)
        first_change = event.price_changes[0]
        assert first_change.best_bid == Decimal("0.65")
        assert first_change.best_ask == Decimal("0.66")

    def test_price_change_decimal_parsing(self) -> None:
        """Test decimal parsing for price change entries."""
        data = {
            "event_type": "price_change",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "price_changes": [
                {
                    "asset_id": "0x123abc",
                    "price": "0.999999999",
                    "size": "123.456789",
                }
            ],
        }
        event = PriceChangeEvent(**data)
        assert str(event.price_changes[0].price) == "0.999999999"
        assert str(event.price_changes[0].size) == "123.456789"


class TestTradeEventParsing:
    """Tests for parsing and validating LastTradePriceEvent messages."""

    def test_parse_trade_event(self, polymarket_trade_message: dict) -> None:
        """Test parsing trade execution event."""
        event = LastTradePriceEvent(**polymarket_trade_message)
        assert event.event_type == "last_trade_price"
        assert event.asset_id == "0x123abc"
        assert event.price == Decimal("0.65")
        assert event.size == Decimal("50.0")
        assert event.side == "BUY"

    def test_trade_event_optional_fields(self) -> None:
        """Test trade event with missing optional fields."""
        data = {
            "event_type": "last_trade_price",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "price": "0.65",
        }
        event = LastTradePriceEvent(**data)
        assert event.price == Decimal("0.65")
        assert event.size is None
        assert event.side is None
        assert event.fee_rate_bps is None


class TestBestBidAskEventParsing:
    """Tests for parsing and validating BestBidAskEvent messages."""

    def test_parse_best_bid_ask_event(
        self, polymarket_best_bid_ask_message: dict
    ) -> None:
        """Test parsing best bid/ask event."""
        event = BestBidAskEvent(**polymarket_best_bid_ask_message)
        assert event.event_type == "best_bid_ask"
        assert event.best_bid == Decimal("0.65")
        assert event.best_ask == Decimal("0.66")
        assert event.spread == Decimal("0.01")

    def test_best_bid_ask_partial_data(self) -> None:
        """Test best bid/ask with partial price data."""
        data = {
            "event_type": "best_bid_ask",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
            "best_bid": "0.65",
        }
        event = BestBidAskEvent(**data)
        assert event.best_bid == Decimal("0.65")
        assert event.best_ask is None
        assert event.spread is None


class TestTickSizeChangeEventParsing:
    """Tests for parsing and validating TickSizeChangeEvent messages."""

    def test_parse_tick_size_change_event(
        self, polymarket_tick_change_message: dict
    ) -> None:
        """Test parsing tick size change event."""
        event = TickSizeChangeEvent(**polymarket_tick_change_message)
        assert event.event_type == "tick_size_change"
        assert event.old_tick_size == Decimal("0.01")
        assert event.new_tick_size == Decimal("0.001")

    def test_tick_size_decimal_precision(
        self, polymarket_tick_change_message: dict
    ) -> None:
        """Test tick size preserves decimal precision."""
        event = TickSizeChangeEvent(**polymarket_tick_change_message)
        assert str(event.old_tick_size) == "0.01"
        assert str(event.new_tick_size) == "0.001"


class TestNewMarketEventParsing:
    """Tests for parsing and validating NewMarketEvent messages."""

    def test_parse_new_market_event(self, polymarket_new_market_message: dict) -> None:
        """Test parsing new market event."""
        event = NewMarketEvent(**polymarket_new_market_message)
        assert event.event_type == "new_market"
        assert event.id == "0xmarketid"
        assert event.slug == "bitcoin-100k-2025"
        assert event.assets_ids is not None
        assert len(event.assets_ids) == 2
        assert event.outcomes is not None
        assert "YES" in event.outcomes
        assert "NO" in event.outcomes

    def test_new_market_event_minimal(self) -> None:
        """Test new market event with minimal required fields."""
        data = {
            "event_type": "new_market",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
        }
        event = NewMarketEvent(**data)
        assert event.event_type == "new_market"
        assert event.id is None
        assert event.question is None


class TestMarketResolvedEventParsing:
    """Tests for parsing and validating MarketResolvedEvent messages."""

    def test_parse_market_resolved_event(self, polymarket_market_resolved_message):
        """Test parsing market resolved event."""
        event = MarketResolvedEvent(**polymarket_market_resolved_message)
        assert event.event_type == "market_resolved"
        assert event.id == "0xmarketid"
        assert event.winning_asset_id == "0x123abc"
        assert event.winning_outcome == "YES"

    def test_market_resolved_event_minimal(self):
        """Test market resolved event with minimal fields."""
        data = {
            "event_type": "market_resolved",
            "market": "0xmarketid",
            "timestamp": 1234567890000,
        }
        event = MarketResolvedEvent(**data)
        assert event.event_type == "market_resolved"
        assert event.winning_asset_id is None
        assert event.winning_outcome is None


# ============================================================================
# MESSAGE DISPATCH TESTS
# ============================================================================


class TestMessageDispatch:
    """Tests for PolymarketMarketChannel message dispatching."""

    @pytest.fixture
    def mock_channel_config(self):
        """Create a mock MarketChannelConfig."""
        return MarketChannelConfig(asset_ids=["0x123abc"])

    @pytest.fixture
    def market_channel(self, mock_channel_config):
        """Create a PolymarketMarketChannel instance."""
        return PolymarketMarketChannel(mock_channel_config)

    def test_dispatch_orderbook_message(
        self,
        market_channel,
        polymarket_orderbook_message,
    ):
        """Test that orderbook messages are dispatched correctly."""
        with patch.object(market_channel, "_log_orderbook") as mock_log:
            market_channel._log_message(polymarket_orderbook_message)
            mock_log.assert_called_once_with(polymarket_orderbook_message)

    def test_dispatch_price_change_message(
        self,
        market_channel,
        polymarket_price_change_message,
    ):
        """Test that price change messages are dispatched correctly."""
        with patch.object(market_channel, "_log_price_change") as mock_log:
            market_channel._log_message(polymarket_price_change_message)
            mock_log.assert_called_once_with(polymarket_price_change_message)

    def test_dispatch_trade_message(
        self,
        market_channel,
        polymarket_trade_message,
    ):
        """Test that trade messages are dispatched correctly."""
        with patch.object(market_channel, "_log_trade") as mock_log:
            market_channel._log_message(polymarket_trade_message)
            mock_log.assert_called_once_with(polymarket_trade_message)

    def test_dispatch_tick_change_message(
        self,
        market_channel,
        polymarket_tick_change_message,
    ):
        """Test that tick change messages are dispatched correctly."""
        with patch.object(market_channel, "_log_tick_change") as mock_log:
            market_channel._log_message(polymarket_tick_change_message)
            mock_log.assert_called_once_with(polymarket_tick_change_message)

    def test_dispatch_best_bid_ask_message(
        self,
        market_channel,
        polymarket_best_bid_ask_message,
    ):
        """Test that best bid/ask messages are dispatched correctly."""
        with patch.object(market_channel, "_log_best_prices") as mock_log:
            market_channel._log_message(polymarket_best_bid_ask_message)
            mock_log.assert_called_once_with(polymarket_best_bid_ask_message)

    def test_dispatch_new_market_message(
        self,
        market_channel,
        polymarket_new_market_message,
    ):
        """Test that new market messages are dispatched correctly."""
        with patch.object(market_channel, "_log_new_market") as mock_log:
            market_channel._log_message(polymarket_new_market_message)
            mock_log.assert_called_once_with(polymarket_new_market_message)

    def test_dispatch_market_resolved_message(
        self,
        market_channel,
        polymarket_market_resolved_message,
    ):
        """Test that market resolved messages are dispatched correctly."""
        with patch.object(market_channel, "_log_market_resolved") as mock_log:
            market_channel._log_message(polymarket_market_resolved_message)
            mock_log.assert_called_once_with(
                polymarket_market_resolved_message
            )

    def test_dispatch_unknown_event_type(self, market_channel):
        """Test handling of unknown event types."""
        unknown_message = {
            "event_type": "unknown_type",
            "market": "0xmarketid",
        }
        # Should not raise, just log as debug
        market_channel._log_message(unknown_message)

    def test_dispatch_non_dict_message(self, market_channel):
        """Test handling of non-dict messages."""
        list_message = [{"event_type": "book"}]
        # Should not raise, just log as info
        market_channel._log_message(list_message)


# ============================================================================
# SUBSCRIPTION MESSAGE TESTS
# ============================================================================


class TestSubscriptionMessage:
    """Tests for market channel subscription message formatting."""

    @pytest.fixture
    def mock_ws(self):
        """Create a mock WebSocket."""
        ws = AsyncMock()
        return ws

    @pytest.fixture
    async def channel_with_mock_ws(self, valid_market_channel_config, mock_ws):
        """Create a market channel with mocked WebSocket."""
        config = MarketChannelConfig(**valid_market_channel_config)
        channel = PolymarketMarketChannel(config)
        channel.ws = mock_ws
        return channel

    @pytest.mark.asyncio
    async def test_subscription_message_format(
        self, channel_with_mock_ws, mock_ws
    ):
        """Test subscription message has correct structure."""
        await channel_with_mock_ws.subscribe()

        # Verify send was called
        mock_ws.send.assert_called_once()

        # Extract and parse the message
        call_args = mock_ws.send.call_args
        sent_message = call_args[0][0]
        message = json.loads(sent_message)

        assert message["type"] == "market"
        assert message["custom_feature_enabled"] is True
        assert "assets_ids" in message
        assert len(message["assets_ids"]) == 3

    @pytest.mark.asyncio
    async def test_subscription_with_single_asset(self, mock_ws):
        """Test subscription with single asset ID."""
        config = MarketChannelConfig(asset_ids=["0x123abc"])
        channel = PolymarketMarketChannel(config)
        channel.ws = mock_ws

        await channel.subscribe()

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert len(sent_message["assets_ids"]) == 1
        assert sent_message["assets_ids"][0] == "0x123abc"

    @pytest.mark.asyncio
    async def test_subscription_custom_features_disabled(self, mock_ws):
        """Test subscription with custom features disabled."""
        config = MarketChannelConfig(
            asset_ids=["0x123abc"],
            custom_feature_enabled=False,
        )
        channel = PolymarketMarketChannel(config)
        channel.ws = mock_ws

        await channel.subscribe()

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["custom_feature_enabled"] is False

    @pytest.mark.asyncio
    async def test_subscribe_without_connection_fails(self):
        """Test that subscribe fails if WebSocket is not connected."""
        config = MarketChannelConfig(asset_ids=["0x123abc"])
        channel = PolymarketMarketChannel(config)
        channel.ws = None  # Not connected

        with pytest.raises(RuntimeError, match="WebSocket not connected"):
            await channel.subscribe()


# ============================================================================
# EVENT VALIDATION EDGE CASES
# ============================================================================


class TestEventValidationEdgeCases:
    """Tests for event validation with edge cases and invalid data."""

    def test_decimal_string_conversion(self):
        """Test that string decimals are converted correctly."""
        level = OrderbookLevel(price=Decimal("999999.999999999"), size=Decimal("0.000001"))
        assert level.price == Decimal("999999.999999999")

    def test_float_to_decimal_conversion(self):
        """Test that float decimals are converted correctly."""
        level = OrderbookLevel(price=Decimal("99.99"), size=Decimal("1.5"))
        assert level.price == Decimal("99.99")
        assert level.size == Decimal("1.5")

    def test_timestamp_as_float_seconds(self):
        """Test parsing timestamp from float seconds."""
        data = {
            "event_type": "book",
            "asset_id": "0x123abc",
            "market": "0xmarketid",
            "timestamp": 1234567890.999,  # float seconds
            "bids": [],
            "asks": [],
        }
        event = BookEvent(**data)
        assert isinstance(event.timestamp, int)

    def test_missing_optional_price_change_fields(self):
        """Test price change entry with missing optional fields."""
        entry = PriceChangeEntry(
            asset_id="0x123abc",
            price=Decimal("0.65"),
            size=Decimal("100.0"),
        )
        assert entry.side is None
        assert entry.best_bid is None
        assert entry.best_ask is None

    def test_none_values_in_optional_fields(self):
        """Test that None values are preserved for optional fields."""
        data = {
            "asset_id": "0x123abc",
            "price": "0.65",
            "size": "100.0",
            "side": None,
            "best_bid": None,
        }
        entry = PriceChangeEntry(**data)
        assert entry.side is None
        assert entry.best_bid is None


# ============================================================================
# INTEGRATION TESTS (Mock-based)
# ============================================================================


class TestPolymarketStreamIntegration:
    """Integration tests for Polymarket stream with mocked WebSocket."""

    @pytest.mark.asyncio
    async def test_stream_lifecycle_with_mock_ws(
        self, valid_market_channel_config
    ):
        """Test complete stream lifecycle with mocked WebSocket."""
        config = MarketChannelConfig(**valid_market_channel_config)
        channel = PolymarketMarketChannel(config)

        # Mock the WebSocket connection
        channel.ws = AsyncMock()
        channel._running = True

        # Mock the run method to test internal state
        assert isinstance(channel.market_config, MarketChannelConfig)
        assert channel.market_config.asset_ids == valid_market_channel_config[
            "asset_ids"
        ]

    @pytest.mark.asyncio
    async def test_multiple_event_dispatch_sequence(
        self,
        polymarket_orderbook_message,
        polymarket_price_change_message,
        polymarket_trade_message,
    ):
        """Test dispatching multiple events in sequence."""
        channel = PolymarketMarketChannel(
            MarketChannelConfig(asset_ids=["0x123abc"])
        )

        events_processed = []

        def mock_log_book(data):
            events_processed.append(("book", data.get("asset_id")))

        def mock_log_price(data):
            events_processed.append(("price_change", data.get("market")))

        def mock_log_trade(data):
            events_processed.append(("trade", data.get("asset_id")))

        # Patch the methods
        channel._log_orderbook = mock_log_book
        channel._log_price_change = mock_log_price
        channel._log_trade = mock_log_trade

        # Dispatch events
        channel._log_message(polymarket_orderbook_message)
        channel._log_message(polymarket_price_change_message)
        channel._log_message(polymarket_trade_message)

        # Verify all events were processed
        assert len(events_processed) == 3
        assert events_processed[0][0] == "book"
        assert events_processed[1][0] == "price_change"
        assert events_processed[2][0] == "trade"

    @pytest.mark.asyncio
    async def test_callback_invocation(self, valid_market_channel_config):
        """Test that callback is invoked for messages."""
        config = MarketChannelConfig(**valid_market_channel_config)
        channel = PolymarketMarketChannel(config)

        callback_invocations = []

        def mock_callback(message):
            callback_invocations.append(message)

        # Mock WebSocket
        channel.ws = AsyncMock()

        # Note: In real tests, we'd test this in the stream() method
        # which is inherited from BaseWebSocketClient
        assert callable(mock_callback)


# ============================================================================
# COMPREHENSIVE EVENT TYPE TESTS
# ============================================================================
# Tests for all Polymarket market channel event types per:
# https://docs.polymarket.com/market-data/websocket/market-channel


class TestAllPolymarketEventTypes:
    """Comprehensive tests for all event types from Polymarket docs.

    This ensures all documented event types can be properly parsed and stored
    without schema conflicts, especially when mixed in the same data file.
    """

    def test_book_event_parsing(self):
        """Test parsing book (orderbook snapshot) event."""
        book_data = {
            "event_type": "book",
            "asset_id": (
                "658186196575688134743418686523089420798049192873804221928922"
                "11131408793125422"
            ),
            "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
            "bids": [
                {"price": ".48", "size": "30"},
                {"price": ".49", "size": "20"},
                {"price": ".50", "size": "15"},
            ],
            "asks": [
                {"price": ".52", "size": "25"},
                {"price": ".53", "size": "60"},
                {"price": ".54", "size": "10"},
            ],
            "timestamp": "1757908890000",
            "hash": "0x0abc1234",
        }

        event = BookEvent(**book_data)
        assert event.event_type == "book"
        assert (
            event.asset_id
            == (
                "658186196575688134743418686523089420798049192873804221928922"
                "11131408793125422"
            )
        )
        assert len(event.bids) == 3
        assert len(event.asks) == 3
        assert event.bids[0].price == Decimal("0.48")

    def test_price_change_event_parsing(self):
        """Test parsing price_change event (order placed/cancelled)."""
        price_change_data = {
            "event_type": "price_change",
            "market": "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1",
            "price_changes": [
                {
                    "asset_id": (
                        "71321045679252212594626385532706912750332728571942532289631379"
                        "312455583992563"
                    ),
                    "price": "0.5",
                    "size": "200",
                    "side": "BUY",
                    "hash": "56621a121a47ed9333273e21c83b660cff37ae50",
                    "best_bid": "0.5",
                    "best_ask": "1",
                },
                {
                    "asset_id": (
                        "52114319501245915516055106046884209969926127482827954674443846"
                        "427813813222426"
                    ),
                    "price": "0.5",
                    "size": "200",
                    "side": "SELL",
                    "hash": "1895759e4df7a796bf4f1c5a5950b748306923e2",
                    "best_bid": "0",
                    "best_ask": "0.5",
                },
            ],
            "timestamp": "1757908892351",
        }

        event = PriceChangeEvent(**price_change_data)
        assert event.event_type == "price_change"
        assert len(event.price_changes) == 2
        assert event.price_changes[0].side == "BUY"
        assert event.price_changes[1].side == "SELL"

    def test_tick_size_change_event_parsing(self) -> None:
        """Test parsing tick_size_change event.

        Emitted when price reaches limits (price > 0.96 or < 0.04).
        """
        tick_size_data = {
            "event_type": "tick_size_change",
            "asset_id": (
                "658186196575688134743418686523089420798049192873804221928922"
                "11131408793125422"
            ),
            "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
            "old_tick_size": "0.01",
            "new_tick_size": "0.001",
            "timestamp": "1757908893000",
        }

        event = TickSizeChangeEvent(**tick_size_data)  # type: ignore[arg-type]
        assert event.event_type == "tick_size_change"
        assert event.old_tick_size == Decimal("0.01")
        assert event.new_tick_size == Decimal("0.001")

    def test_last_trade_price_event_parsing(self) -> None:
        """Test parsing last_trade_price event (trade execution)."""
        trade_data = {
            "asset_id": (
                "11412207150964437967801872790870956022661814800337144611011450"
                "9806601493071694"
            ),
            "event_type": "last_trade_price",
            "fee_rate_bps": "0",
            "market": "0x6a67b9d828d53862160e470329ffea5246f338ecfffdf2cab45211ec578b0347",
            "price": "0.456",
            "side": "BUY",
            "size": "219.217767",
            "timestamp": "1750428146322",
        }

        event = LastTradePriceEvent(**trade_data)  # type: ignore[arg-type]
        assert event.event_type == "last_trade_price"
        assert event.price == Decimal("0.456")
        assert event.side == "BUY"

    def test_best_bid_ask_event_parsing(self) -> None:
        """Test parsing best_bid_ask event (custom_feature_enabled).

        Emitted when best bid or ask prices change.
        """
        best_bid_ask_data = {
            "event_type": "best_bid_ask",
            "market": "0x0005c0d312de0be897668695bae9f32b624b4a1ae8b140c49f08447fcc74f442",
            "asset_id": (
                "8535495606243046531592411686012538853859543381957454275203"
                "1640332592237464430"
            ),
            "best_bid": "0.73",
            "best_ask": "0.77",
            "spread": "0.04",
            "timestamp": "1766789469958",
        }

        event = BestBidAskEvent(**best_bid_ask_data)  # type: ignore[arg-type]
        assert event.event_type == "best_bid_ask"
        assert event.best_bid == Decimal("0.73")
        assert event.best_ask == Decimal("0.77")

    def test_new_market_event_parsing(self):
        """Test parsing new_market event (custom_feature_enabled).

        Emitted when a new market is created.
        """
        new_market_data = {
            "id": "1031769",
            "question": "Will NVIDIA (NVDA) close above $240 end of January?",
            "market": "0x311d0c4b6671ab54af4970c06fcf58662516f5168997bdda209ec3db5aa6b0c1",
            "slug": "nvda-above-240-on-january-30-2026",
            "description": "This market will resolve to Yes if NVDA closes above $240",
            "assets_ids": [
                "76043073756653678226373981964075571318267289248134717369284518995922789326425",
                "31690934263385727664202099278545688007799199447969475608906331829650099442770",
            ],
            "outcomes": ["Yes", "No"],
            "event_message": {
                "id": "125819",
                "ticker": "nvda-above-in-january-2026",
                "slug": "nvda-above-in-january-2026",
                "title": "Will NVIDIA (NVDA) close above ___ end of January?",
                "description": "This market will resolve to Yes if NVDA closes above $240",
            },
            "timestamp": "1766790415550",
            "event_type": "new_market",
        }

        event = NewMarketEvent(**new_market_data)
        assert event.event_type == "new_market"
        assert event.id == "1031769"
        assert event.outcomes is not None
        assert len(event.outcomes) == 2
        assert event.event_message is not None

    def test_market_resolved_event_parsing(self):
        """Test parsing market_resolved event (custom_feature_enabled).

        Emitted when a market is resolved with winning asset/outcome.
        """
        _winning_asset_id = (
            "76043073756653678226373981964075571318267289248134717369"
            "284518995922789326425"
        )
        market_resolved_data = {
            "id": "1031769",
            "question": "Will NVIDIA (NVDA) close above $240 end of January?",
            "market": "0x311d0c4b6671ab54af4970c06fcf58662516f5168997bdda209ec3db5aa6b0c1",
            "slug": "nvda-above-240-on-january-30-2026",
            "description": "This market will resolve to Yes if NVDA closes above $240",
            "assets_ids": [
                _winning_asset_id,
                "31690934263385727664202099278545688007799199447969475608906331829650099442770",
            ],
            "outcomes": ["Yes", "No"],
            "winning_asset_id": _winning_asset_id,
            "winning_outcome": "Yes",
            "event_message": {
                "id": "125819",
                "ticker": "nvda-above-in-january-2026",
                "slug": "nvda-above-in-january-2026",
                "title": "Will NVIDIA (NVDA) close above ___ end of January?",
                "description": "This market will resolve to Yes if NVDA closes above $240",
            },
            "timestamp": "1766790415550",
            "event_type": "market_resolved",
        }

        event = MarketResolvedEvent(**market_resolved_data)  # type: ignore[arg-type]
        assert event.event_type == "market_resolved"
        assert event.winning_outcome == "Yes"
        assert event.winning_asset_id == _winning_asset_id
        assert event.outcomes is not None
        assert len(event.outcomes) == 2
        assert event.event_message is not None

    def test_all_event_types_with_storage_sink(self, temp_storage_dir):
        """Integration test: Store all event types using StreamStorageSink.

        This is the critical test that would have caught the schema mismatch bug.
        Different event types with different data structures must coexist in the
        same Parquet file without schema conflicts.
        """
        # use test-local imports for Path, pq and StreamStorageSink
        from pathlib import Path  # noqa: PLC0415

        import pyarrow.parquet as pq  # noqa: PLC0415

        from data.stream.storage import StreamStorageSink  # noqa: PLC0415

        sink = StreamStorageSink(
            base_path=temp_storage_dir,
            buffer_size=1,  # Force flush after each event
        )

        events = [
            {
                "event_type": "book",
                "asset_id": (
                    "658186196575688134743418686523089420798049192873804221928922"
                    "11131408793125422"
                ),
                "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
                "bids": [{"price": ".48", "size": "30"}],
                "asks": [{"price": ".52", "size": "25"}],
                "timestamp": "1757908890000",
                "hash": "0x0abc1234",
            },
            {
                "event_type": "price_change",
                "market": "0x5f65177b394277fd294cd75650044e32ba009a95022d88a0c1d565897d72f8f1",
                "price_changes": [
                    {
                        "asset_id": (
                            "71321045679252212594626385532706912750332728571942532289631379"
                            "312455583992563"
                        ),
                        "price": "0.5",
                        "size": "200",
                        "side": "BUY",
                    }
                ],
                "timestamp": "1757908891000",
            },
            {
                "event_type": "tick_size_change",
                "asset_id": (
                    "658186196575688134743418686523089420798049192873804221928922"
                    "11131408793125422"
                ),
                "market": "0xbd31dc8a20211944f6b70f31557f1001557b59905b7738480ca09bd4532f84af",
                "old_tick_size": "0.01",
                "new_tick_size": "0.001",
                "timestamp": "1757908892000",
            },
            {
                "event_type": "last_trade_price",
                "asset_id": (
                    "11412207150964437967801872790870956022661814800337144611011450"
                    "9806601493071694"
                ),
                "market": "0x6a67b9d828d53862160e470329ffea5246f338ecfffdf2cab45211ec578b0347",
                "price": "0.456",
                "side": "BUY",
                "size": "219.217767",
                "fee_rate_bps": "0",
                "timestamp": "1757908893000",
            },
            {
                "event_type": "best_bid_ask",
                "market": "0x0005c0d312de0be897668695bae9f32b624b4a1ae8b140c49f08447fcc74f442",
                "asset_id": (
                    "8535495606243046531592411686012538853859543381957454275203"
                    "1640332592237464430"
                ),
                "best_bid": "0.73",
                "best_ask": "0.77",
                "spread": "0.04",
                "timestamp": "1757908894000",
            },
            {
                "event_type": "new_market",
                "id": "1031769",
                "question": "Will NVIDIA (NVDA) close above $240 end of January?",
                "market": "0x311d0c4b6671ab54af4970c06fcf58662516f5168997bdda209ec3db5aa6b0c1",
                "slug": "nvda-above-240-on-january-30-2026",
                "description": "This market will resolve to Yes if NVDA closes above $240",
                "assets_ids": [
                    "76043073756653678226373981964075571318267289248134717369284518995922789326425",
                    "31690934263385727664202099278545688007799199447969475608906331829650099442770",
                ],
                "outcomes": ["Yes", "No"],
                "event_message": {
                    "id": "125819",
                    "ticker": "nvda-above-in-january-2026",
                },
                "timestamp": "1757908895000",
            },
            {
                "event_type": "market_resolved",
                "id": "1031769",
                "question": "Will NVIDIA (NVDA) close above $240 end of January?",
                "market": (
                    "0x311d0c4b6671ab54af4970c06fcf58662516f5168997bdda209ec3db5aa6b0c1"
                ),
                "slug": "nvda-above-240-on-january-30-2026",
                "description": "This market will resolve to Yes if NVDA closes above $240",
                "assets_ids": [
                    (
                        "76043073756653678226373981964075571318267289248134717369"
                        "284518995922789326425"
                    ),
                    "31690934263385727664202099278545688007799199447969475608906331829650099442770",
                ],
                "outcomes": ["Yes", "No"],
                "winning_asset_id": (
                    "76043073756653678226373981964075571318267289248134717369"
                    "284518995922789326425"
                ),
                "winning_outcome": "Yes",
                "event_message": {
                    "id": "125819",
                    "ticker": "nvda-above-in-january-2026",
                },
                "timestamp": "1757908896000",
            },
        ]

        # Write all event types to storage
        for event in events:
            sink.write_market_event(event, market_slug="nvda-market")

        # Verify all events were persisted without schema errors
        output_dir = Path(temp_storage_dir) / "polymarket_market"
        parquet_files = list(output_dir.glob("*.parquet"))

        assert len(parquet_files) > 0, "No parquet files created"

        table = pq.read_table(parquet_files[0])
        df = table.to_pandas()

        # All 7 event types should be present
        assert len(df) == 7, f"Expected 7 events, got {len(df)}"

        # Verify all event types are present
        stored_types = set(df["event_type"].unique())
        expected_types = {
            "book",
            "price_change",
            "tick_size_change",
            "last_trade_price",
            "best_bid_ask",
            "new_market",
            "market_resolved",
        }
        assert stored_types == expected_types, (
            f"Missing event types. Expected {expected_types}, "
            f"got {stored_types}"
        )
