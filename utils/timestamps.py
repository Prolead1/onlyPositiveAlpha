"""Timestamp conversion and handling utilities."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


def parse_timestamp_ms(value: Any, default: int = 0) -> int:  # noqa: ANN401
    """Parse any timestamp value to milliseconds (int).

    Parameters
    ----------
    value : Any
        Timestamp value (int, float, or string representing a number).
    default : int
        Default value to return on parsing failure.

    Returns
    -------
    int
        Timestamp in milliseconds, or default if parsing fails.
    """
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def timestamp_ms_to_datetime(timestamp_ms: int) -> datetime:
    """Convert milliseconds timestamp to UTC datetime.

    Parameters
    ----------
    timestamp_ms : int
        Unix timestamp in milliseconds.

    Returns
    -------
    datetime
        Timezone-aware datetime in UTC.
    """
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)


def datetime_to_timestamp_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds Unix timestamp.

    Parameters
    ----------
    dt : datetime
        Datetime to convert.

    Returns
    -------
    int
        Unix timestamp in milliseconds.
    """
    return int(dt.timestamp() * 1000)


def ts_to_datetime(timestamp: float, unit: str = "ms") -> datetime:
    """Convert timestamp to UTC datetime (legacy compatibility).

    Parameters
    ----------
    timestamp : int | float
        Unix timestamp.
    unit : str
        Unit of timestamp: 'ms' (milliseconds) or 's' (seconds).

    Returns
    -------
    datetime
        Timezone-aware datetime in UTC.
    """
    if unit == "ms":
        return datetime.fromtimestamp(timestamp / 1000.0, tz=UTC)
    if unit == "s":
        return datetime.fromtimestamp(timestamp, tz=UTC)
    msg = f"Invalid unit: {unit}. Use 'ms' or 's'."
    raise ValueError(msg)


def datetime_to_ts(dt: datetime, unit: str = "ms") -> int:
    """Convert datetime to Unix timestamp (legacy compatibility).

    Parameters
    ----------
    dt : datetime
        Datetime to convert.
    unit : str
        Unit of output timestamp: 'ms' (milliseconds) or 's' (seconds).

    Returns
    -------
    int
        Unix timestamp.
    """
    ts = dt.timestamp()
    if unit == "ms":
        return int(ts * 1000)
    if unit == "s":
        return int(ts)
    msg = f"Invalid unit: {unit}. Use 'ms' or 's'."
    raise ValueError(msg)
