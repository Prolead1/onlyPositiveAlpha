from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pyarrow.parquet as pq

from EDA.core.schema import (
    CORE_SOURCE_COLUMNS,
    MAPPING_SCAN_SCHEMA,
    MAPPING_UNIVERSE_COLUMNS,
    SOURCE_NAMES,
    coin_expr,
    parse_file_metadata,
    resolution_expr,
)
from EDA.core.utils import get_logger

if TYPE_CHECKING:
    from EDA.core.config import RunConfig


@dataclass(frozen=True)
class HourlyBatch:
    """One capture hour plus the parquet files available for that hour."""

    hour_key: str
    mapping_paths: tuple[Path, ...]
    book_snapshot_paths: tuple[Path, ...]
    price_change_paths: tuple[Path, ...]


def build_file_inventory(config: RunConfig) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    for source in SOURCE_NAMES:
        source_dir = config.paths.data_root / source
        for path in sorted(source_dir.glob("*.parquet")):
            metadata = pq.ParquetFile(path).metadata
            parsed = parse_file_metadata(path)
            records.append(
                {
                    "source": source,
                    "hour_key": parsed["hour_key"],
                    "suffix": parsed["suffix"],
                    "chunk_index": parsed["chunk_index"] or 0,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "row_count": metadata.num_rows,
                    "modified_at_utc": path.stat().st_mtime,
                },
            )
    inventory = pl.DataFrame(records).sort(["source", "hour_key", "path"])
    inventory.write_parquet(config.paths.mode_cache_dir(config.mode) / "file_inventory.parquet")
    return inventory


def build_schema_summary(config: RunConfig) -> pl.DataFrame:
    logger = get_logger()
    records: list[dict[str, object]] = []
    for source in SOURCE_NAMES:
        source_dir = config.paths.data_root / source
        variants: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for path in sorted(source_dir.glob("*.parquet")):
            schema = pq.ParquetFile(path).schema_arrow
            variants[tuple(schema.names)].append(str(path))
        for variant_index, (columns, sample_paths) in enumerate(variants.items(), start=1):
            records.append(
                {
                    "source": source,
                    "schema_variant_id": f"{source}_variant_{variant_index}",
                    "column_count": len(columns),
                    "file_count": len(sample_paths),
                    "sample_path": sample_paths[0],
                    "columns": list(columns),
                },
            )
        logger.info("%s schema variants: %s", source, len(variants))
    summary = pl.DataFrame(records).sort(["source", "schema_variant_id"])
    summary.write_parquet(config.paths.mode_cache_dir(config.mode) / "schema_summary.parquet")
    return summary


def discover_batches(inventory: pl.DataFrame) -> list[HourlyBatch]:
    by_hour: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for row in inventory.iter_rows(named=True):
        by_hour[row["hour_key"]][row["source"]].append(Path(row["path"]))
    batches: list[HourlyBatch] = []
    for hour_key in sorted(by_hour):
        source_map = by_hour[hour_key]
        if not all(source in source_map for source in SOURCE_NAMES):
            continue
        batches.append(
            HourlyBatch(
                hour_key=hour_key,
                mapping_paths=tuple(sorted(source_map["mapping"])),
                book_snapshot_paths=tuple(sorted(source_map["book_snapshot"])),
                price_change_paths=tuple(sorted(source_map["price_change"])),
            ),
        )
    return batches


def select_batches(config: RunConfig, batches: list[HourlyBatch]) -> list[HourlyBatch]:
    if config.is_sample:
        logger = get_logger()
        target_universe = set(config.expected_coin_resolution_pairs)
        eligible_batches = [
            batch
            for batch in batches
            if target_universe.issubset(_mapping_coin_resolution_pairs(batch))
        ]
        if eligible_batches:
            selected = eligible_batches[-config.sample_hour_limit :]
            logger.info(
                "Sample mode selected %s latest shared hours with full %s x %s coverage.",
                len(selected),
                len(config.expected_coins),
                len(config.resolution_order),
            )
            return selected
        logger.warning(
            "Sample mode could not find any shared hour that covers the expected "
            "%s x %s universe; falling back to the latest %s shared hours.",
            len(config.expected_coins),
            len(config.resolution_order),
            config.sample_hour_limit,
        )
        return batches[-config.sample_hour_limit :]
    return batches


def _mapping_coin_resolution_pairs(batch: HourlyBatch) -> set[tuple[str, str]]:
    frame = (
        scan_mapping(batch.mapping_paths, columns=MAPPING_UNIVERSE_COLUMNS)
        .with_columns(
            coin_expr().alias("coin"),
            resolution_expr().alias("resolution"),
        )
        .filter(pl.col("coin").is_not_null() & pl.col("resolution").is_not_null())
        .select(["coin", "resolution"])
        .unique()
        .collect()
    )
    return {tuple(row) for row in frame.iter_rows()}


def scan_mapping(
    paths: list[Path] | tuple[Path, ...],
    columns: tuple[str, ...] | None = None,
) -> pl.LazyFrame:
    return pl.scan_parquet(
        [str(path) for path in paths],
        schema=MAPPING_SCAN_SCHEMA,
        extra_columns="ignore",
        missing_columns="insert",
        cast_options=pl.ScanCastOptions(integer_cast="allow-float"),
    ).select(list(columns or CORE_SOURCE_COLUMNS["mapping"]))


def inventory_summary(inventory: pl.DataFrame) -> pl.DataFrame:
    return (
        inventory.group_by("source")
        .agg(
            pl.len().alias("file_count"),
            pl.n_unique("hour_key").alias("distinct_hours"),
            pl.col("row_count").sum().alias("total_rows"),
            (pl.col("size_bytes").sum() / 1_000_000_000).alias("total_size_gb"),
            pl.col("hour_key").min().alias("first_hour"),
            pl.col("hour_key").max().alias("last_hour"),
            (pl.len() - pl.n_unique("hour_key")).alias("extra_files_beyond_hours"),
        )
        .sort("source")
    )


def write_analysis_table(frame: pl.DataFrame, path: Path) -> Path:
    frame.write_parquet(path)
    return path
