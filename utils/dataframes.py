"""DataFrame operation utilities."""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


def prepare_timestamp_index(
    df: pd.DataFrame,
    *,
    col: str = "ts_event",
    sort: bool = True,
) -> pd.DataFrame:
    """Convert timestamp column to datetime index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column.
    col : str
        Name of timestamp column to convert.
    sort : bool
        If True, sort by index after conversion.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index.
    """
    if df.empty:
        return df

    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df = df.set_index(col)

    if sort:
        df = df.sort_index()

    return df


def filter_by_time_range(
    df: pd.DataFrame,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Filter DataFrame by time range using index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with datetime index.
    start : datetime | None
        Start time (inclusive). If None, no lower bound.
    end : datetime | None
        End time (inclusive). If None, no upper bound.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if df.empty:
        return df

    result = df
    if start:
        result = result[result.index >= start]
    if end:
        result = result[result.index <= end]

    return result


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Get list of numeric column names from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.

    Returns
    -------
    list[str]
        List of numeric column names.
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()
