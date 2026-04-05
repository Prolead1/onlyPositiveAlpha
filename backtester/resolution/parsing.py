from __future__ import annotations

import json
from math import isnan

import pandas as pd


def is_missing_resolution_winner(value: object) -> bool:
    """Return True when a resolution winner value is effectively missing."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if value is pd.NA:
        return True
    if isinstance(value, float):
        return isnan(value)
    return False


def parse_clob_token_ids(raw_value: object) -> list[str]:
    """Parse CLOB token ids from either list or JSON-encoded list."""
    if isinstance(raw_value, list):
        return [str(token) for token in raw_value]
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(token) for token in parsed]
    return []


def parse_outcome_prices(raw_value: object) -> list[float]:
    """Parse outcome prices into floats, coercing invalid values to NaN."""
    if not isinstance(raw_value, list):
        return []
    prices: list[float] = []
    for value in raw_value:
        try:
            prices.append(float(value))
        except (TypeError, ValueError):
            prices.append(float("nan"))
    return prices


def select_winning_asset_id(
    token_ids: list[str],
    outcome_prices: list[float],
) -> str | None:
    """Pick the winner token from aligned token ids and outcome prices."""
    if not token_ids or not outcome_prices or len(token_ids) != len(outcome_prices):
        return None

    max_idx: int | None = None
    max_val = float("-inf")
    for idx, value in enumerate(outcome_prices):
        if isnan(value):
            continue
        if max_idx is None or value > max_val:
            max_idx = idx
            max_val = value

    if max_idx is None:
        return None
    return token_ids[max_idx] if 0 <= max_idx < len(token_ids) else None
