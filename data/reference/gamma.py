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
        Returns empty list if no market is found.
    """
    timeslug = _get_btc_slug(utctime, resolution)
    requests_url = (
        f"{POLYMARKET_GAMMA_API_URL}?slug={timeslug}"
    )
    response = requests.get(requests_url, timeout=10)
    if response.status_code == HTTPStatus.OK:
        data = response.json()
        return _extract_asset_ids(data)
    logger.warning(
        "Failed to retrieve asset ID for UTC time %d and resolution %s",
        utctime,
        resolution,
    )
    return []

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

def _resolution_to_seconds(resolution: str) -> int:
    if resolution.endswith("m"):
        return int(resolution[:-1]) * 60
    if resolution.endswith("h"):
        return int(resolution[:-1]) * 3600
    if resolution.endswith("d"):
        return int(resolution[:-1]) * 86400
    msg = f"Invalid resolution format: {resolution}"
    raise ValueError(msg)
