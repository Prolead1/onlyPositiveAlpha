"""Logging configuration utilities."""

from __future__ import annotations

import logging

# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_application_logging(
    level: int = logging.INFO,
    format_string: str | None = None,
    date_format: str | None = None,
) -> None:
    """Configure application-wide logging.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).
    format_string : str | None
        Custom format string for log messages. Uses default if None.
    date_format : str | None
        Custom date format. Uses default if None.
    """
    fmt = format_string or DEFAULT_FORMAT
    datefmt = date_format or DEFAULT_DATE_FORMAT

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
    )
