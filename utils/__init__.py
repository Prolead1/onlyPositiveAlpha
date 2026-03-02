"""Common utilities for the onlyPositiveAlpha project.

This module provides shared utilities for timestamps, paths, validation,
serialization, logging, and DataFrame operations.
"""

from __future__ import annotations

from .logging_config import setup_application_logging
from .paths import (
    ensure_directory,
    get_storage_path,
    get_workspace_root,
    validate_path_exists,
)

# Core utilities (no heavy dependencies)
from .timestamps import (
    datetime_to_timestamp_ms,
    parse_timestamp_ms,
    timestamp_ms_to_datetime,
)
from .validation import (
    parse_orderbook_level,
    validate_crypto_price_data,
    validate_dict_type,
    validate_non_zero_price,
)


# Lazy imports for modules with heavy dependencies
def _import_serialization() -> dict[str, object]:
    """Lazy import serialization utilities (requires pandas)."""
    from .serialization import (  # noqa: PLC0415 - intentional lazy import
        coerce_to_json_string,
        dataframe_dict_column_to_json,
        dataframe_json_column_to_dict,
        parse_json_field,
    )
    return {
        "coerce_to_json_string": coerce_to_json_string,
        "dataframe_dict_column_to_json": dataframe_dict_column_to_json,
        "dataframe_json_column_to_dict": dataframe_json_column_to_dict,
        "parse_json_field": parse_json_field,
    }


def _import_dataframes() -> dict[str, object]:
    """Lazy import DataFrame utilities (requires pandas/numpy)."""
    from .dataframes import (  # noqa: PLC0415 - intentional lazy import
        filter_by_time_range,
        get_numeric_columns,
        prepare_timestamp_index,
    )
    return {
        "filter_by_time_range": filter_by_time_range,
        "get_numeric_columns": get_numeric_columns,
        "prepare_timestamp_index": prepare_timestamp_index,
    }


# Make heavy modules available with lazy loading
def __getattr__(name: str) -> object:
    """Lazy load heavy dependencies."""
    serialization_funcs = ["coerce_to_json_string", "dataframe_dict_column_to_json",
                           "dataframe_json_column_to_dict", "parse_json_field"]
    dataframe_funcs = ["filter_by_time_range", "get_numeric_columns", "prepare_timestamp_index"]

    if name in serialization_funcs:
        return _import_serialization()[name]
    if name in dataframe_funcs:
        return _import_dataframes()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [  # noqa: RUF022 - intentional grouping order
    # Timestamps
    "datetime_to_timestamp_ms",
    "parse_timestamp_ms",
    "timestamp_ms_to_datetime",
    # Paths
    "ensure_directory",
    "get_storage_path",
    "get_workspace_root",
    "validate_path_exists",
    # Logging
    "setup_application_logging",
    # Validation
    "parse_orderbook_level",
    "validate_crypto_price_data",
    "validate_dict_type",
    "validate_non_zero_price",
    # Serialization (lazy loaded)
    "coerce_to_json_string",
    "dataframe_dict_column_to_json",
    "dataframe_json_column_to_dict",
    "parse_json_field",
    # DataFrames (lazy loaded)
    "filter_by_time_range",
    "get_numeric_columns",
    "prepare_timestamp_index",
]
