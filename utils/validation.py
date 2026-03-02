"""Data validation utilities."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_dict_type(data: Any) -> bool:  # noqa: ANN401
    """Check if data is a dictionary.

    Parameters
    ----------
    data : Any
        Data to validate.

    Returns
    -------
    bool
        True if data is a dict, False otherwise.
    """
    return isinstance(data, dict)


def validate_non_zero_price(price: float | None, symbol: str = "") -> bool:
    """Validate that price is non-zero and valid.

    Parameters
    ----------
    price : float | None
        Price value to validate.
    symbol : str
        Symbol name for logging.

    Returns
    -------
    bool
        True if price is valid, False otherwise.
    """
    if price is None or price == 0:
        if symbol:
            logger.debug("Invalid price for %s: %s", symbol, price)
        return False
    return True


def parse_orderbook_level(level: tuple | dict) -> tuple[float, float]:
    """Parse orderbook level to (price, size) tuple.

    Handles both dict format with 'price' and 'size' keys,
    and tuple/list format with [price, size] values.

    Parameters
    ----------
    level : tuple | dict
        Orderbook level in dict or tuple format.

    Returns
    -------
    tuple[float, float]
        Tuple of (price, size).
    """
    if isinstance(level, dict):
        price = float(level.get("price", 0))
        size = float(level.get("size", 0))
    else:
        price, size = float(level[0]), float(level[1])
    return price, size


def validate_crypto_price_data(data: dict | list) -> dict | None:
    """Validate and extract crypto price data from event payload.

    Handles nested WebSocket message structures and extracts
    symbol, price, and timestamp from various formats.

    Parameters
    ----------
    data : dict | list
        Raw event data from WebSocket.

    Returns
    -------
    dict | None
        Extracted data with keys: 'symbol', 'price', 'timestamp'.
        Returns None if validation fails.
    """
    if not isinstance(data, dict):
        logger.debug("Received non-dict crypto event: %s", type(data))
        return None

    # Extract price data from WebSocket message payload
    payload = data.get("payload")
    if not isinstance(payload, dict):
        logger.warning(
            "Invalid payload format: expected dict, got %s. Full message: %s",
            type(payload),
            data,
        )
        return None

    # Extract symbol and price from payload
    symbol = payload.get("symbol", "")
    price = payload.get("value", payload.get("price", 0))
    timestamp = payload.get("timestamp", data.get("timestamp"))

    # Handle case where actual pricing data is in a series
    data_series = payload.get("data")
    if isinstance(data_series, list) and data_series:
        last_point = data_series[-1]
        if isinstance(last_point, dict):
            price = last_point.get("value", price)
            timestamp = last_point.get("timestamp", timestamp)

    # Validate extracted data (combined validation)
    if not symbol or not price or price == 0:
        logger.warning(
            "Invalid data: symbol='%s', price=%s in event: %s",
            symbol,
            price,
            data,
        )
        return None

    if timestamp is None or timestamp == 0:
        logger.warning(
            "Invalid or missing timestamp for %s. Price: %s. Message: %s",
            symbol,
            price,
            data,
        )
        return None

    # Ensure timestamp is in milliseconds (int)
    try:
        timestamp_ms = int(float(timestamp))
    except (ValueError, TypeError):
        logger.warning(
            "Cannot convert timestamp to int: %s (type: %s). Symbol: %s",
            timestamp,
            type(timestamp),
            symbol,
        )
        return None
    else:
        return {"symbol": symbol, "price": price, "timestamp": timestamp_ms}
