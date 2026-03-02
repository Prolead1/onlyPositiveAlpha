import json
import logging
from http import HTTPStatus
from typing import Any

import requests

logger = logging.getLogger(__name__)

POLYMARKET_GAMMA_API_URL = "https://gamma-api.polymarket.com/markets"

def get_updown_asset_ids(utctime: int, resolution: str) -> list[str]:
    """Get Bitcoin up-down market asset IDs for a given time and resolution.

    Parameters
    ----------
    utctime : int
        Unix timestamp in seconds.
    resolution : str
        Time resolution (e.g., '5m', '1h', '1d').

    Returns
    -------
    list[str]
        List of asset IDs (token IDs) for the market outcomes.
        Returns empty list if no market is found or if the request fails.
    """
    timeslug = _get_btc_slug(utctime, resolution)
    requests_url = (
        f"{POLYMARKET_GAMMA_API_URL}?slug={timeslug}"
    )
    try:
        response = requests.get(requests_url, timeout=10)
    except requests.RequestException as exc:
        logger.warning(
            "Error retrieving asset IDs for UTC time %d and resolution %s: %s",
            utctime,
            resolution,
            exc,
        )
        return []
    if response.status_code == HTTPStatus.OK:
        data = response.json()
        return _extract_asset_ids(data)
    logger.warning(
        "Non-OK response retrieving asset IDs for UTC time %d and resolution %s: "
        "status=%s, body=%r",
        utctime,
        resolution,
        response.status_code,
        response.text,
    )
    return []


def get_updown_asset_ids_with_slug(utctime: int, resolution: str) -> tuple[str, list[str]]:
    """Get Bitcoin up-down market slug and asset IDs for a given time and resolution.

    Parameters
    ----------
    utctime : int
        Unix timestamp in seconds.
    resolution : str
        Time resolution (e.g., '5m', '1h', '1d').

    Returns
    -------
    tuple[str, list[str]]
        Market slug and list of asset IDs (token IDs) for the market outcomes.
        Returns ('', []) if no market is found or if the request fails.
    """
    market_slug = _get_btc_slug(utctime, resolution)
    requests_url = (
        f"{POLYMARKET_GAMMA_API_URL}?slug={market_slug}"
    )
    try:
        response = requests.get(requests_url, timeout=10)
    except requests.RequestException as exc:
        logger.warning(
            "Error retrieving asset IDs for UTC time %d and resolution %s: %s",
            utctime,
            resolution,
            exc,
        )
        return "", []
    if response.status_code == HTTPStatus.OK:
        data = response.json()
        asset_ids = _extract_asset_ids(data)
        return market_slug, asset_ids
    logger.warning(
        "Non-OK response retrieving asset IDs for UTC time %d and resolution %s: "
        "status=%s, body=%r",
        utctime,
        resolution,
        response.status_code,
        response.text,
    )
    return "", []

def _extract_asset_ids(response_data: list[dict[str, Any]]) -> list[str]:
    if not response_data:
        return []
    raw_clob_tokens = response_data[0].get("clobTokenIds", "[]")
    if isinstance(raw_clob_tokens, list):
        clob_tokens = raw_clob_tokens
    elif isinstance(raw_clob_tokens, str):
        clob_tokens = json.loads(raw_clob_tokens)
    else:
        clob_tokens = []
    logger.info("Extracted clobTokens: %s", clob_tokens)
    return clob_tokens

def _get_btc_slug(utctime: int, resolution: str) -> str:
    seconds = _resolution_to_seconds(resolution)
    timeslug = (utctime // seconds) * seconds
    return f"btc-updown-{resolution}-{timeslug}"

def _parse_positive_resolution_value(resolution: str) -> int:
    """Extract and validate the numeric portion of a resolution string.

    The resolution is expected to be of the form "<positive_integer><unit>",
    where <unit> is a single-character suffix such as 'm', 'h', or 'd'.
    """
    min_resolution_length = 2
    if len(resolution) < min_resolution_length:
        msg = f"Invalid resolution format (missing numeric value): {resolution}"
        raise ValueError(msg)
    numeric_part = resolution[:-1]
    if not numeric_part.isdigit():
        msg = f"Invalid resolution format (non-numeric value): {resolution}"
        raise ValueError(msg)
    value = int(numeric_part)
    if value <= 0:
        msg = f"Resolution must be a positive integer: {resolution}"
        raise ValueError(msg)
    return value

def _resolution_to_seconds(resolution: str) -> int:
    if resolution.endswith("m"):
        return _parse_positive_resolution_value(resolution) * 60
    if resolution.endswith("h"):
        return _parse_positive_resolution_value(resolution) * 3600
    if resolution.endswith("d"):
        return _parse_positive_resolution_value(resolution) * 86400
    msg = f"Invalid resolution format: {resolution}"
    raise ValueError(msg)
