"""Base WebSocket client infrastructure for Polymarket data streams."""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
from abc import ABC, abstractmethod
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

import websockets
from websockets.exceptions import ConnectionClosed

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class WebSocketConfig:
    """Base configuration for WebSocket connections.

    Attributes
    ----------
    url : str
        The WebSocket endpoint URL.
    max_retries : int
        Maximum number of connection attempts (including the initial attempt).
        Default is 5. Set to 1 to disable retries and only attempt once.
    retry_delay : float
        Initial delay in seconds before retrying connection. Default is 5.0.
        Increases exponentially on each retry to avoid rate limiting.
    headers : dict[str, str] | None
        Additional HTTP headers for WebSocket connection.
    ssl_context : ssl.SSLContext | None
        Custom SSL context for secure WebSocket connections (wss://).
        If None, a secure default context will be created for wss:// URLs.
    verify_ssl : bool
        Whether to verify SSL certificates. Default is True.
        Set to False only for testing/debugging purposes.
    """

    url: str
    max_retries: int = 5
    retry_delay: float = 5.0
    headers: dict[str, str] | None = None
    ssl_context: ssl.SSLContext | None = None
    verify_ssl: bool = True

    def __post_init__(self) -> None:
        """Validate configuration and set up SSL context for secure connections."""
        if self.url.startswith("wss://") and self.ssl_context is None:
            self.ssl_context = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext:
        if self.verify_ssl:
            # Create context with default secure settings
            ssl_context = ssl.create_default_context()
            # Ensure proper certificate verification
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            # Use TLS 1.2 or higher for security
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            logger.info("Created secure SSL context with certificate verification enabled")
        else:
            # Disable verification (not recommended for production)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.warning(
                "SSL certificate verification disabled - use only for testing/debugging"
            )

        return ssl_context


class BaseWebSocketClient(ABC):
    """Abstract base class for WebSocket clients with reconnection support."""

    def __init__(self, config: WebSocketConfig) -> None:
        """Initialize the WebSocket client.

        Parameters
        ----------
        config : WebSocketConfig
            Configuration for the WebSocket connection.
        """
        self.config = config
        self.ws: Any = None
        self._running = False
        self._rate_limit_count = 0
        self._rate_limit_delay = 1.0  # Initial delay in seconds


    @abstractmethod
    async def subscribe(self) -> None:
        """Send subscription messages. Must be implemented by subclasses."""

    @abstractmethod
    def _log_message(self, data: dict[str, Any] | list[Any]) -> None:
        """Log received messages. Must be implemented by subclasses."""


    async def connect(self) -> None:
        """Establish WebSocket connection with retry logic."""
        logger.info("Connecting to %s", self.config.url)

        retry_count = 0
        delay = self.config.retry_delay

        while retry_count < self.config.max_retries:
            try:
                await self._attempt_connection(retry_count, delay)
            except websockets.exceptions.InvalidStatus as exc:
                delay = self._handle_invalid_status(exc, retry_count)
                retry_count += 1
            except ssl.SSLError:
                logger.exception(
                    "SSL/TLS handshake failed. Check certificate configuration."
                )
                raise
            except Exception:
                logger.exception("Failed to connect to %s", self.config.url)
                raise
            else:
                logger.info("Connected to %s", self.config.url)
                return

        # If we reach this point, all retry attempts have been exhausted
        raise ConnectionError(
            f"Failed to connect to {self.config.url} after {retry_count} attempts"
        )
    async def _attempt_connection(self, retry_count: int, delay: float) -> None:
        # Add initial delay to avoid rate limiting on retry
        if retry_count > 0:
            logger.info(
                "Connection attempt %d/%d after %.1f seconds...",
                retry_count + 1,
                self.config.max_retries,
                delay,
            )
            await asyncio.sleep(delay)

        # Prepare connection kwargs
        connect_kwargs = self._build_connection_kwargs()

        self.ws = await websockets.connect(
            self.config.url,
            **connect_kwargs,
        )

    def _build_connection_kwargs(self) -> dict[str, Any]:
        connect_kwargs: dict[str, Any] = {
            "max_size": None,
            "ping_interval": 20.0,
            "ping_timeout": 20.0,
            "compression": None,
        }

        # Add custom headers if provided
        if self.config.headers:
            connect_kwargs["additional_headers"] = self.config.headers

        # Add SSL context for secure connections (wss://)
        if self.config.ssl_context:
            connect_kwargs["ssl"] = self.config.ssl_context
            logger.debug(
                "Using SSL context with verification: %s",
                self.config.verify_ssl,
            )

        return connect_kwargs

    def _calculate_exponential_backoff(
        self, current_count: int, base_delay: float = 1.0, max_delay: float = 60.0
    ) -> float:
        return min(base_delay * (2 ** (current_count - 1)), max_delay)

    def _handle_invalid_status(
        self, exc: websockets.exceptions.InvalidStatus, retry_count: int
    ) -> float:
        if exc.response.status_code != HTTPStatus.TOO_MANY_REQUESTS:
            logger.exception("HTTP error %d", exc.response.status_code)
            raise exc

        # Rate limit error - use exponential backoff
        if retry_count + 1 >= self.config.max_retries:
            logger.exception(
                "Max attempts (%d) exceeded due to rate limiting (HTTP 429)",
                self.config.max_retries,
            )
            raise exc

        # Exponential backoff with cap at 60 seconds
        new_delay = self._calculate_exponential_backoff(
            retry_count + 1, base_delay=self.config.retry_delay
        )
        logger.warning(
            "Rate limited (HTTP 429). Retrying in %.1f seconds... (attempt %d/%d)",
            new_delay,
            retry_count + 1,
            self.config.max_retries,
        )
        return new_delay

    def _is_empty_message(self, message: str) -> bool:
        return not message or message.isspace()

    async def _handle_rate_limit(self, data: dict[str, Any]) -> bool:
        if data.get("message") == "Too Many Requests":
            self._rate_limit_count += 1
            delay = self._calculate_exponential_backoff(
                self._rate_limit_count, base_delay=self._rate_limit_delay
            )
            logger.warning(
                "Rate limited by server (ConnectionId: %s). "
                "Backing off for %.1f seconds... (rate limit #%d)",
                data.get("connectionId", "unknown"),
                delay,
                self._rate_limit_count,
            )
            await asyncio.sleep(delay)
            return True

        if self._rate_limit_count > 0:
            logger.info(
                "Rate limit cleared after %d warnings. Resuming normal operation.",
                self._rate_limit_count,
            )
            self._rate_limit_count = 0

        return False

    def _handle_error_response(self, data: dict[str, Any]) -> bool:
        if "statusCode" not in data or data.get("statusCode") == HTTPStatus.OK:
            return False

        status_code = data.get("statusCode")
        error_body = data.get("body", {})
        error_msg = error_body.get("message", "Unknown error")
        logger.error(
            "Server error (HTTP %d): %s. Full response: %s",
            status_code,
            error_msg,
            data,
        )
        return True

    def _dispatch_message(
        self,
        data: dict[str, Any] | list[Any],
        callback: Callable[[dict[str, Any] | list[Any]], None] | None,
    ) -> None:
        self._log_message(data)
        if callback:
            callback(data)

    async def _process_message(
        self,
        message: str,
        callback: Callable[[dict[str, Any] | list[Any]], None] | None,
    ) -> None:
        if self._is_empty_message(message):
            logger.debug("Received empty message, skipping")
            return

        try:
            data = json.loads(message)
            logger.debug("Received message: %s", data)

            if isinstance(data, list):
                self._dispatch_message(data, callback)
                return

            if await self._handle_rate_limit(data):
                return

            if self._handle_error_response(data):
                return

            self._dispatch_message(data, callback)

        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to decode message (length: %d): %s. Raw: %r",
                len(message),
                exc,
                message[:200],
            )
        except Exception:
            logger.exception("Error processing message")

    async def stream(
        self, callback: Callable[[dict[str, Any] | list[Any]], None] | None = None
    ) -> None:
        """Start streaming data from WebSocket."""
        if not self.ws:
            raise RuntimeError("WebSocket not connected. Call connect() first.")

        logger.info("Starting to stream from %s...", self.config.url)

        # Reset rate limit counter for fresh connection
        self._rate_limit_count = 0

        try:
            async for message in self.ws:
                if not self._running:
                    break
                await self._process_message(message, callback)
        except (KeyboardInterrupt, asyncio.CancelledError):
            # User interrupted - stop gracefully without traceback
            logger.info("Stream interrupted by user (Ctrl-C)")
            self._running = False
        except ConnectionClosed as exc:
            logger.warning(
                "WebSocket connection closed (code=%s, reason=%s)",
                exc.code,
                exc.reason,
            )
        except Exception:
            logger.exception("Error in stream")

    async def stop(self) -> None:
        """Stop streaming and close the WebSocket connection."""
        logger.info("Stopping stream from %s...", self.config.url)
        self._running = False

        if self.ws:
            await self.ws.close()
            logger.info("WebSocket closed")
            self.ws = None

    async def run(
        self, callback: Callable[[dict[str, Any] | list[Any]], None] | None = None
    ) -> None:
        """Connect, subscribe, and start streaming in one call."""
        self._running = True
        reconnect_delay = self.config.retry_delay

        try:
            while self._running:
                try:
                    await self.connect()
                    await self.subscribe()
                    await self.stream(callback=callback)
                    reconnect_delay = self.config.retry_delay
                except (KeyboardInterrupt, asyncio.CancelledError):
                    logger.info("Interrupted by user (Ctrl-C), shutting down gracefully...")
                    break
                except Exception:
                    logger.exception("Stream error; will attempt to reconnect")

                if not self._running:
                    break

                logger.info("Reconnecting in %.1f seconds...", reconnect_delay)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60.0)
        finally:
            await self.stop()
