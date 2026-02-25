from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .websocket import BaseWebSocketClient, WebSocketConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Constants
POLYMARKET_MARKET_CHANNEL_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class MarketChannelConfig:
    """Configuration for Polymarket market channel streaming.

    Attributes
    ----------
    asset_ids : list[str]
        List of asset IDs to subscribe to. These are token IDs for specific outcomes
        in the market (e.g., Bitcoin up-down contract outcomes).
    max_retries : int
        Maximum number of connection retry attempts. Default is 5.
    retry_delay : float
        Initial delay in seconds before retrying connection. Default is 10.0.
        Increases exponentially on each retry to avoid rate limiting.
    custom_feature_enabled : bool
        Enable custom features like best_bid_ask, new_market, market_resolved events.
        Default is True.
    """

    asset_ids: list[str] = field(default_factory=list)
    max_retries: int = 5
    retry_delay: float = 10.0
    custom_feature_enabled: bool = True


class PolymarketMarketChannel(BaseWebSocketClient):
    """WebSocket client for Polymarket market channel (L2 orderbook and trade data)."""

    def __init__(self, config: MarketChannelConfig) -> None:
        """Initialize the Polymarket market channel client.

        Parameters
        ----------
        config : MarketChannelConfig
            Configuration for the market channel stream.
        """
        self.market_config = config

        # Create WebSocket config for parent class
        ws_config = WebSocketConfig(
            url=POLYMARKET_MARKET_CHANNEL_URL,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            headers=None,  # No special headers needed for market channel
        )
        super().__init__(ws_config)

    async def subscribe(self) -> None:
        """Send subscription message for market channel data."""
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")

        if not self.market_config.asset_ids:
            logger.warning("No asset IDs configured for subscription")
            return

        message = {
            "assets_ids": self.market_config.asset_ids,
            "type": "market",
            "custom_feature_enabled": self.market_config.custom_feature_enabled,
        }

        logger.info(
            "Subscribing to market channel for %d asset(s): %s",
            len(self.market_config.asset_ids),
            json.dumps(message, indent=2),
        )
        await self.ws.send(json.dumps(message, indent=2))

    def _log_message(self, data: dict[str, Any]) -> None:
        """Log received market data message."""
        if not isinstance(data, dict):
            logger.info("Received non-dict message: %s", json.dumps(data))
        else:
            event_type = data.get("event_type", "unknown")

            if event_type == "book":
                self._log_orderbook(data)
            elif event_type == "price_change":
                self._log_price_change(data)
            elif event_type == "last_trade_price":
                self._log_trade(data)
            elif event_type == "tick_size_change":
                self._log_tick_change(data)
            elif event_type == "best_bid_ask":
                self._log_best_prices(data)
            elif event_type == "new_market":
                self._log_new_market(data)
            elif event_type == "market_resolved":
                self._log_market_resolved(data)
            else:
                logger.debug("Received event type: %s", event_type)

    def _log_orderbook(self, data: dict[str, Any]) -> None:
        """Log L2 orderbook snapshot."""
        asset_id = data.get("asset_id", "unknown")[:16]
        market = data.get("market", "unknown")[:16]
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        timestamp = data.get("timestamp", 0)

        logger.info(
            "Order Book Update - Asset: %s, Market: %s, Bids: %d levels, "
            "Asks: %d levels, Timestamp: %s",
            asset_id,
            market,
            len(bids),
            len(asks),
            timestamp,
        )
        if bids:
            logger.debug("Top Bids: %s", bids[:3])
        if asks:
            logger.debug("Top Asks: %s", asks[:3])

    def _log_price_change(self, data: dict[str, Any]) -> None:
        """Log price changes from new/updated orders."""
        market = data.get("market", "unknown")[:16]
        changes = data.get("price_changes", [])
        timestamp = data.get("timestamp", 0)

        logger.info(
            "Price Change Event - Market: %s, Changes: %d, Timestamp: %s",
            market,
            len(changes),
            timestamp,
        )
        for change in changes[:3]:
            asset = change.get("asset_id", "unknown")[:16]
            price = change.get("price", "N/A")
            size = change.get("size", "N/A")
            side = change.get("side", "unknown")
            logger.debug(
                "  Change - Asset: %s, Price: %s, Size: %s, Side: %s",
                asset,
                price,
                size,
                side,
            )

    def _log_trade(self, data: dict[str, Any]) -> None:
        """Log trade execution."""
        asset_id = data.get("asset_id", "unknown")[:16]
        market = data.get("market", "unknown")[:16]
        price = data.get("price", "N/A")
        size = data.get("size", "N/A")
        side = data.get("side", "unknown")
        timestamp = data.get("timestamp", 0)

        logger.info(
            "Trade Executed - Asset: %s, Market: %s, Price: %s, Size: %s, "
            "Side: %s, Timestamp: %s",
            asset_id,
            market,
            price,
            size,
            side,
            timestamp,
        )

    def _log_tick_change(self, data: dict[str, Any]) -> None:
        """Log tick size change."""
        asset_id = data.get("asset_id", "unknown")[:16]
        market = data.get("market", "unknown")[:16]
        old_tick = data.get("old_tick_size", "N/A")
        new_tick = data.get("new_tick_size", "N/A")
        timestamp = data.get("timestamp", 0)

        logger.info(
            "Tick Size Change - Asset: %s, Market: %s, %s -> %s, Timestamp: %s",
            asset_id,
            market,
            old_tick,
            new_tick,
            timestamp,
        )

    def _log_best_prices(self, data: dict[str, Any]) -> None:
        """Log best bid/ask update."""
        asset_id = data.get("asset_id", "unknown")[:16]
        market = data.get("market", "unknown")[:16]
        best_bid = data.get("best_bid", "N/A")
        best_ask = data.get("best_ask", "N/A")
        spread = data.get("spread", "N/A")
        timestamp = data.get("timestamp", 0)

        logger.info(
            "Best Bid/Ask - Asset: %s, Market: %s, Bid: %s, Ask: %s, "
            "Spread: %s, Timestamp: %s",
            asset_id,
            market,
            best_bid,
            best_ask,
            spread,
            timestamp,
        )

    def _log_new_market(self, data: dict[str, Any]) -> None:
        """Log new market creation."""
        market_id = data.get("id", "unknown")
        question = data.get("question", "unknown")[:50]
        timestamp = data.get("timestamp", 0)

        logger.info(
            "New Market - ID: %s, Question: %s, Timestamp: %s",
            market_id,
            question,
            timestamp,
        )

    def _log_market_resolved(self, data: dict[str, Any]) -> None:
        """Log market resolution."""
        market_id = data.get("id", "unknown")
        winning_outcome = data.get("winning_outcome", "unknown")
        timestamp = data.get("timestamp", 0)

        logger.info(
            "Market Resolved - ID: %s, Winning Outcome: %s, Timestamp: %s",
            market_id,
            winning_outcome,
            timestamp,
        )


async def stream_polymarket_data(
    asset_ids: list[str],
    *,
    enable_custom_features: bool = True,
    callback: Callable[[dict[str, Any] | list[Any]], None] | None = None,
) -> None:
    """Stream market data from Polymarket market channel.

    Parameters
    ----------
    asset_ids : list[str]
        List of asset IDs to subscribe to. These are token IDs for specific
        market outcomes.
    enable_custom_features : bool
        Enable custom features like best_bid_ask, new_market, market_resolved.
        Default is True.
    callback : callable, optional
        Custom callback function for processing messages.
    """
    config = MarketChannelConfig(
        asset_ids=asset_ids,
        custom_feature_enabled=enable_custom_features,
    )
    stream = PolymarketMarketChannel(config)
    await stream.run(callback=callback)
