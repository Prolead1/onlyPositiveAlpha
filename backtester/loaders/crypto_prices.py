from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from utils.serialization import dataframe_json_column_to_dict

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

logger = logging.getLogger(__name__)


def load_crypto_prices(
    rtds_path: Path,
    *,
    start: datetime | None,
    end: datetime | None,
    prepare_timestamp_index_fn,
    filter_by_time_range_fn,
) -> pd.DataFrame:
    """Load crypto price events from parquet partitions."""
    parquet_files = sorted(rtds_path.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No crypto price files found in %s", rtds_path)
        return pd.DataFrame()

    dfs = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = prepare_timestamp_index_fn(combined, col="ts_event", sort=True)

    if "data" in combined.columns and not combined.empty:
        combined = dataframe_json_column_to_dict(combined, column="data")

    combined = filter_by_time_range_fn(combined, start=start, end=end)

    logger.info("Loaded %d crypto price events", len(combined))
    return combined
