from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from config import POLYMARKET_RTDS_URL

from .websocket import BaseWebSocketClient, WebSocketConfig

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class CryptoPriceConfig:
    """Configuration for crypto price streaming.

    Attributes
    ----------
    symbols : list[str] | None
        List of symbols to stream. For Binance source use lowercase concatenated format
        (e.g., 'btcusdt', 'ethusdt'). For Chainlink source use slash-separated format
        (e.g., 'btc/usd', 'eth/usd'). If None, subscribes to all symbols.
    source : str
        Price data source: 'binance' or 'chainlink'. Default is 'chainlink'.
    max_retries : int
        Maximum number of connection retry attempts. Default is 5.
    retry_delay : float
        Initial delay in seconds before retrying connection. Default is 10.0.
        Increases exponentially on each retry to avoid rate limiting.
    """

    symbols: list[str] | None = None
    source: str = "chainlink"
    max_retries: int = 5
    retry_delay: float = 10.0


class PolymarketCryptoStream(BaseWebSocketClient):
    """WebSocket client for streaming crypto prices from Polymarket RTDS."""

    def __init__(self, config: CryptoPriceConfig) -> None:
        """Initialize the Polymarket crypto price stream client.

        Parameters
        ----------
        config : CryptoPriceConfig
            Configuration for the stream.
        """
        self.price_config = config

        # Create WebSocket config for parent class
        ws_config = WebSocketConfig(
            url=POLYMARKET_RTDS_URL,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay
        )
        super().__init__(ws_config)


    async def subscribe(self) -> None:
        """Send subscription messages for crypto prices."""
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")
        # Determine topic and filters based on source
        source = self.price_config.source.lower()
        if source == "chainlink":
            await self._subscribe_chainlink()
        elif source == "binance":
            await self._subscribe_binance()
        else:
            msg = (
                f"Invalid price source '{self.price_config.source}'. "
                "Expected 'binance' or 'chainlink'."
            )
            raise ValueError(msg)

    async def _subscribe_binance(self) -> None:
        topic = "crypto_prices"

        subscriptions = []
        if self.price_config.symbols:
            subscriptions.extend([
                {
                    "topic": topic,
                    "type": "update",
                    "filters": ",".join(self.price_config.symbols)
                }
            ])
        else:
            subscriptions.append({
                "topic": topic,
                "type": "update"
            })

        message = {"action": "subscribe", "subscriptions": subscriptions}
        logger.info("Sending subscription message for Binance crypto prices: %s",
                    json.dumps(message))
        await self.ws.send(json.dumps(message))

    async def _subscribe_chainlink(self) -> None:
        topic = "crypto_prices_chainlink"

        if self.price_config.symbols:
            message = {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": topic,
                        "type": "*",
                        "filters": json.dumps({"symbol":symbol}, separators=(",", ":"))
                    } for symbol in self.price_config.symbols
                ]
            }
        else:
            message = {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": topic,
                        "type": "*",
                    }
                ],
            }
        logger.info("Sending subscription message for Chainlink crypto prices: %s",
                    json.dumps(message))
        await self.ws.send(json.dumps(message))

    def _log_message(self, data: dict[str, Any] | list[Any]) -> None:
        """Log received crypto price update."""
        # Override base class method to format price updates specifically
        if not isinstance(data, dict):
            logger.info("Received non-dict message: %s", json.dumps(data))
            return

        topic = data.get("topic", "unknown")
        msg_type = data.get("type", "unknown")
        timestamp = data.get("timestamp", 0)
        payload = data.get("payload", {})

        # Handle case where payload might be None or not a dict
        if not isinstance(payload, dict):
            logger.warning("Unexpected payload format: %r. Full message: %s", payload, data)
            return

        symbol = payload.get("symbol", "unknown")
        value = payload.get("value", "N/A")
        price_timestamp = payload.get("timestamp", 0)

        # Chainlink payloads often include a data series instead of a single value.
        data_series = payload.get("data")
        if isinstance(data_series, list) and data_series:
            last_point = data_series[-1]
            if isinstance(last_point, dict):
                value = last_point.get("value", value)
                price_timestamp = last_point.get("timestamp", price_timestamp)

        # Log full message if critical fields are missing
        if symbol == "unknown" or value == "N/A":
            logger.warning("Incomplete price data. Full message: %s", data)
        else:
            logger.info(
                "Price Update - Symbol: %s, Price: %s, Type: %s, Topic: %s "
                "(msg_ts: %d, price_ts: %d)",
                symbol,
                value,
                msg_type,
                topic,
                timestamp,
                price_timestamp,
            )



async def stream_crypto_prices(
    symbols: list[str] | None = None,
    source: str = "chainlink",
    callback: Callable[[dict[str, Any] | list[Any]], None] | None = None,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> None:
    """Stream cryptocurrency price data from Polymarket RTDS.

    Parameters
    ----------
    symbols : list[str], optional
        List of symbols to stream. Format depends on source:
        - Binance: lowercase concatenated (e.g., ['btcusdt', 'ethusdt', 'solusdt'])
        - Chainlink: slash-separated (e.g., ['btc/usd', 'eth/usd', 'sol/usd'])
        If None, streams all available symbols.
    source : str
        Price data source: 'binance' or 'chainlink'. Default is 'chainlink'.
    callback : callable, optional
        Custom callback function for processing messages.
    max_retries : int, optional
        Maximum number of connection retry attempts. Default is 5.
    retry_delay : float, optional
        Initial delay in seconds before retrying connection. Default is 10.0.
    """
    config = CryptoPriceConfig(
        symbols=symbols,
        source=source,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    stream = PolymarketCryptoStream(config)
    await stream.run(callback=callback)
