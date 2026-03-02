"""JSON and data serialization utilities."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def coerce_to_json_string(value: Any) -> str | None:  # noqa: ANN401
    """Convert Python object to JSON string.

    Parameters
    ----------
    value : Any
        Value to convert to JSON string.

    Returns
    -------
    str | None
        JSON string representation, or None if value is None.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return json.dumps(str(value))


def parse_json_field(value: Any) -> dict | None:  # noqa: ANN401
    """Parse JSON from string or return dict as-is.

    Parameters
    ----------
    value : Any
        Value to parse (string or dict).

    Returns
    -------
    dict | None
        Parsed dictionary, or None on failure.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse JSON: %s", value)
            return None
    return None


def dataframe_json_column_to_dict(
    df: pd.DataFrame,
    *,
    column: str = "data",
    inplace: bool = False,
) -> pd.DataFrame:
    """Convert JSON string column to parsed dict objects.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with JSON string column.
    column : str
        Name of column containing JSON strings.
    inplace : bool
        If True, modify DataFrame in place.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed dict column.
    """
    if column not in df.columns:
        return df

    if not inplace:
        df = df.copy()

    def _safe_parse_json_cell(x: Any) -> Any:
        if not isinstance(x, str):
            return x
        try:
            return json.loads(x)
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse JSON in column '%s': %s", column, x)
            return x

    df[column] = df[column].apply(_safe_parse_json_cell)
    return df


def dataframe_dict_column_to_json(
    df: pd.DataFrame,
    *,
    column: str = "data",
    inplace: bool = False,
) -> pd.DataFrame:
    """Convert dict column to JSON strings for Parquet storage.

    This is commonly used before writing DataFrames to Parquet
    to ensure stable schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with dict column.
    column : str
        Name of column containing dicts.
    inplace : bool
        If True, modify DataFrame in place.

    Returns
    -------
    pd.DataFrame
        DataFrame with JSON string column.
    """
    if column not in df.columns:
        return df

    if not inplace:
        df = df.copy()

    df[column] = df[column].apply(coerce_to_json_string)
    return df
