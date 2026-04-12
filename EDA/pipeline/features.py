from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from EDA.core.io import HourlyBatch, scan_mapping
from EDA.core.schema import MAPPING_DIMENSION_COLUMNS, coin_expr, resolution_expr
from EDA.core.utils import ensure_directories, get_logger, safe_ratio

if TYPE_CHECKING:
    from pathlib import Path

    from EDA.core.config import RunConfig

BOOK_HOURLY_COLUMNS = (
    "coin",
    "resolution",
    "hour_key",
    "market_id",
    "token_id",
    "side",
    "first_timestamp",
    "snapshot_count",
    "spread_mean",
    "spread_median",
    "relative_spread_mean",
    "relative_spread_median",
    "top_of_book_depth_mean",
    "top_of_book_depth_median",
    "total_visible_depth_mean",
    "total_visible_depth_median",
    "order_book_imbalance_mean",
    "liquidity_concentration_mean",
    "midpoint_change_frequency",
    "stale_quote_frequency",
    "quote_update_intensity_per_min",
    "snapshot_irregularity",
    "crossed_quote_frequency",
    "locked_quote_frequency",
    "missing_depth_side_frequency",
)
PRICE_HOURLY_COLUMNS = (
    "coin",
    "resolution",
    "hour_key",
    "market_id",
    "token_id",
    "side",
    "first_timestamp",
    "last_midpoint",
    "event_count",
    "event_spread_mean",
    "midpoint_change_mean",
    "abs_midpoint_change_mean",
    "abs_midpoint_change_p95",
    "abs_midpoint_change_max",
    "realized_volatility",
    "rms_midpoint_change",
    "buy_event_share",
    "change_side_imbalance",
    "momentum_indicator_mean",
    "mean_reversion_indicator_mean",
    "change_intensity_per_min",
    "jump_rate_small",
    "jump_rate_large",
)


@dataclass(frozen=True)
class FeatureArtifacts:
    """Cached feature outputs produced by the hourly aggregation stage."""

    market_dimension_path: Path
    book_hourly_paths: tuple[Path, ...]
    price_hourly_paths: tuple[Path, ...]
    batch_audit_path: Path


def build_market_dimension(config: RunConfig, _batches: list[HourlyBatch]) -> pl.DataFrame:
    mapping_paths = sorted((config.paths.data_root / "mapping").glob("*.parquet"))
    dimension = (
        scan_mapping(mapping_paths, columns=MAPPING_DIMENSION_COLUMNS)
        .with_columns(
            coin_expr().alias("coin"),
            resolution_expr().alias("resolution"),
        )
        .filter(pl.col("coin").is_not_null() & pl.col("resolution").is_not_null())
        .explode("gamma_market_clob_token_ids")
        .rename(
            {
                "condition_id": "market_id",
                "gamma_market_clob_token_ids": "token_id",
                "gamma_market_start_date": "event_start_time",
                "gamma_market_end_date": "event_end_time",
            },
        )
        .group_by(["market_id", "token_id"])
        .agg(
            pl.col("coin").drop_nulls().first().alias("coin"),
            pl.col("resolution").drop_nulls().first().alias("resolution"),
            pl.col("event_start_time").min().alias("event_start_time"),
            pl.col("event_end_time").max().alias("event_end_time"),
        )
        .sort(["coin", "resolution", "market_id", "token_id"])
        .collect()
    )
    market_dimension_path = config.paths.mode_cache_dir(config.mode) / "market_dimension.parquet"
    dimension.write_parquet(market_dimension_path)
    return dimension


def build_hourly_feature_caches(
    config: RunConfig,
    batches: list[HourlyBatch],
    inventory: pl.DataFrame,
    market_dimension: pl.DataFrame,
) -> FeatureArtifacts:
    logger = get_logger()
    mode_cache_dir = config.paths.mode_cache_dir(config.mode)
    book_cache_dir = mode_cache_dir / "book_hourly"
    price_cache_dir = mode_cache_dir / "price_hourly"
    ensure_directories([book_cache_dir, price_cache_dir])

    existing_artifacts = _existing_feature_artifacts(mode_cache_dir, batches)
    if existing_artifacts is not None:
        logger.info("Reusing existing hourly feature caches under %s", mode_cache_dir)
        return existing_artifacts

    inventory_lookup = {
        (row["source"], row["hour_key"]): int(row["raw_rows"])
        for row in (
            inventory.group_by(["source", "hour_key"])
            .agg(pl.col("row_count").sum().alias("raw_rows"))
            .iter_rows(named=True)
        )
    }

    book_paths: list[Path] = []
    price_paths: list[Path] = []
    audit_rows: list[dict[str, object]] = []

    for batch in batches:
        logger.info("Processing hour %s", batch.hour_key)
        book_summary = summarize_book_hour(batch, market_dimension)
        book_path = book_cache_dir / f"book_{batch.hour_key}.parquet"
        book_summary.write_parquet(book_path)
        book_paths.append(book_path)
        matched_book_rows = int(book_summary["snapshot_count"].sum()) if book_summary.height else 0
        audit_rows.append(
            {
                "source": "book_snapshot",
                "hour_key": batch.hour_key,
                "file_count": len(batch.book_snapshot_paths),
                "raw_rows": inventory_lookup[("book_snapshot", batch.hour_key)],
                "matched_rows": matched_book_rows,
                "output_rows": book_summary.height,
            },
        )

        price_summary = summarize_price_hour(config, batch, market_dimension)
        price_path = price_cache_dir / f"price_{batch.hour_key}.parquet"
        price_summary.write_parquet(price_path)
        price_paths.append(price_path)
        matched_price_rows = int(price_summary["event_count"].sum()) if price_summary.height else 0
        audit_rows.append(
            {
                "source": "price_change",
                "hour_key": batch.hour_key,
                "file_count": len(batch.price_change_paths),
                "raw_rows": inventory_lookup[("price_change", batch.hour_key)],
                "matched_rows": matched_price_rows,
                "output_rows": price_summary.height,
            },
        )

        audit_rows.append(
            {
                "source": "mapping",
                "hour_key": batch.hour_key,
                "file_count": len(batch.mapping_paths),
                "raw_rows": inventory_lookup[("mapping", batch.hour_key)],
                "matched_rows": inventory_lookup[("mapping", batch.hour_key)],
                "output_rows": inventory_lookup[("mapping", batch.hour_key)],
            },
        )

    batch_audit = pl.DataFrame(audit_rows).with_columns(
        (pl.col("raw_rows") - pl.col("matched_rows")).alias("unmatched_rows"),
        safe_ratio(pl.col("matched_rows"), pl.col("raw_rows"), "match_rate"),
    )
    batch_audit_path = mode_cache_dir / "batch_audit.parquet"
    batch_audit.write_parquet(batch_audit_path)

    return FeatureArtifacts(
        market_dimension_path=mode_cache_dir / "market_dimension.parquet",
        book_hourly_paths=tuple(book_paths),
        price_hourly_paths=tuple(price_paths),
        batch_audit_path=batch_audit_path,
    )


def summarize_book_hour(batch: HourlyBatch, market_dimension: pl.DataFrame) -> pl.DataFrame:
    market_keys = _market_keys_lazy(market_dimension)
    market_metadata = _market_metadata_lazy(market_dimension)
    group_cols = ["market_id", "token_id"]
    return (
        pl.scan_parquet([str(path) for path in batch.book_snapshot_paths])
        .select(
            [
                "timestamp_received",
                "market_id",
                "token_id",
                "side",
                "best_bid",
                "best_ask",
                "bids",
                "asks",
            ],
        )
        .join(market_keys, on=["market_id", "token_id"], how="semi")
        .with_columns(
            pl.lit(batch.hour_key).alias("hour_key"),
            pl.col("bids")
            .list.first()
            .struct.field("size")
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("top_bid_size"),
            pl.col("asks")
            .list.first()
            .struct.field("size")
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("top_ask_size"),
            pl.col("bids")
            .list.eval(pl.element().struct.field("size"))
            .list.sum()
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("total_bid_depth"),
            pl.col("asks")
            .list.eval(pl.element().struct.field("size"))
            .list.sum()
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("total_ask_depth"),
        )
        .with_columns(
            (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
            ((pl.col("best_ask") + pl.col("best_bid")) / 2.0).alias("midpoint"),
            (pl.col("top_bid_size") + pl.col("top_ask_size")).alias("top_of_book_depth"),
            (pl.col("total_bid_depth") + pl.col("total_ask_depth")).alias("total_visible_depth"),
        )
        .with_columns(
            safe_ratio(pl.col("spread"), pl.col("midpoint"), "relative_spread"),
            safe_ratio(
                pl.col("total_bid_depth") - pl.col("total_ask_depth"),
                pl.col("total_visible_depth"),
                "order_book_imbalance",
            ),
            safe_ratio(
                pl.col("top_of_book_depth"),
                pl.col("total_visible_depth"),
                "liquidity_concentration",
            ),
        )
        .sort(["market_id", "token_id", "timestamp_received"])
        .with_columns(
            pl.col("midpoint").shift(1).over(group_cols).alias("prev_midpoint"),
            pl.col("best_bid").shift(1).over(group_cols).alias("prev_best_bid"),
            pl.col("best_ask").shift(1).over(group_cols).alias("prev_best_ask"),
            (pl.col("timestamp_received") - pl.col("timestamp_received").shift(1).over(group_cols))
            .dt.total_milliseconds()
            .truediv(1_000.0)
            .alias("snapshot_gap_s"),
        )
        .with_columns(
            pl.when(pl.col("prev_midpoint").is_not_null())
            .then((pl.col("midpoint") != pl.col("prev_midpoint")).cast(pl.Float64))
            .otherwise(None)
            .alias("midpoint_changed"),
            pl.when(
                pl.col("prev_best_bid").is_not_null() & pl.col("prev_best_ask").is_not_null(),
            )
            .then(
                (
                    (pl.col("best_bid") == pl.col("prev_best_bid"))
                    & (pl.col("best_ask") == pl.col("prev_best_ask"))
                ).cast(pl.Float64),
            )
            .otherwise(None)
            .alias("stale_quote"),
            (pl.col("best_ask") < pl.col("best_bid")).cast(pl.Float64).alias("crossed_quote"),
            (pl.col("best_ask") == pl.col("best_bid")).cast(pl.Float64).alias("locked_quote"),
            ((pl.col("total_bid_depth") <= 0.0) | (pl.col("total_ask_depth") <= 0.0))
            .cast(pl.Float64)
            .alias("missing_depth_side"),
        )
        .group_by(["hour_key", "market_id", "token_id", "side"])
        .agg(
            pl.len().alias("snapshot_count"),
            pl.col("timestamp_received").min().alias("first_timestamp"),
            pl.col("timestamp_received").max().alias("last_timestamp"),
            pl.col("spread").mean().alias("spread_mean"),
            pl.col("spread").median().alias("spread_median"),
            pl.col("relative_spread").mean().alias("relative_spread_mean"),
            pl.col("relative_spread").median().alias("relative_spread_median"),
            pl.col("top_of_book_depth").mean().alias("top_of_book_depth_mean"),
            pl.col("top_of_book_depth").median().alias("top_of_book_depth_median"),
            pl.col("total_visible_depth").mean().alias("total_visible_depth_mean"),
            pl.col("total_visible_depth").median().alias("total_visible_depth_median"),
            pl.col("order_book_imbalance").mean().alias("order_book_imbalance_mean"),
            pl.col("liquidity_concentration").mean().alias("liquidity_concentration_mean"),
            pl.col("midpoint_changed").mean().alias("midpoint_change_frequency"),
            pl.col("stale_quote").mean().alias("stale_quote_frequency"),
            pl.col("crossed_quote").mean().alias("crossed_quote_frequency"),
            pl.col("locked_quote").mean().alias("locked_quote_frequency"),
            pl.col("missing_depth_side").mean().alias("missing_depth_side_frequency"),
            pl.col("snapshot_gap_s").mean().alias("snapshot_gap_mean_s"),
            pl.col("snapshot_gap_s").std().alias("snapshot_gap_std_s"),
        )
        .with_columns(
            (pl.col("last_timestamp") - pl.col("first_timestamp"))
            .dt.total_milliseconds()
            .truediv(60_000.0)
            .alias("active_minutes"),
        )
        .with_columns(
            pl.when(pl.col("active_minutes") > 0)
            .then((pl.col("snapshot_count") - 1) / pl.col("active_minutes"))
            .otherwise(None)
            .alias("quote_update_intensity_per_min"),
            safe_ratio(
                pl.col("snapshot_gap_std_s"),
                pl.col("snapshot_gap_mean_s"),
                "snapshot_irregularity",
            ),
        )
        .join(market_metadata, on=["market_id", "token_id"], how="inner")
        .select(list(BOOK_HOURLY_COLUMNS))
        .collect()
    )


def summarize_price_hour(
    config: RunConfig,
    batch: HourlyBatch,
    market_dimension: pl.DataFrame,
) -> pl.DataFrame:
    market_keys = _market_keys_lazy(market_dimension)
    market_metadata = _market_metadata_lazy(market_dimension)
    group_cols = ["market_id", "token_id"]
    return (
        pl.scan_parquet([str(path) for path in batch.price_change_paths])
        .select(
            [
                "timestamp_received",
                "market_id",
                "token_id",
                "side",
                "best_bid",
                "best_ask",
                "change_side",
            ],
        )
        .join(market_keys, on=["market_id", "token_id"], how="semi")
        .with_columns(
            pl.lit(batch.hour_key).alias("hour_key"),
            (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
            ((pl.col("best_ask") + pl.col("best_bid")) / 2.0).alias("midpoint"),
        )
        .sort(["market_id", "token_id", "timestamp_received"])
        .with_columns(
            pl.col("midpoint").shift(1).over(group_cols).alias("prev_midpoint"),
            (pl.col("timestamp_received") - pl.col("timestamp_received").shift(1).over(group_cols))
            .dt.total_milliseconds()
            .truediv(1_000.0)
            .alias("event_gap_s"),
        )
        .with_columns(
            (pl.col("midpoint") - pl.col("prev_midpoint")).alias("midpoint_change"),
        )
        .with_columns(
            pl.col("midpoint_change").abs().alias("abs_midpoint_change"),
            (pl.col("midpoint_change") * pl.col("midpoint_change")).alias("sq_midpoint_change"),
            pl.col("midpoint_change").shift(1).over(group_cols).alias("prev_midpoint_change"),
            (pl.col("change_side") == "BUY").cast(pl.Float64).alias("buy_event"),
            (
                (pl.col("change_side") == "BUY").cast(pl.Float64)
                - (pl.col("change_side") == "SELL").cast(pl.Float64)
            ).alias("change_side_signal"),
        )
        .with_columns(
            pl.when(pl.col("prev_midpoint_change").is_not_null())
            .then(
                (pl.col("midpoint_change") * pl.col("prev_midpoint_change") > 0).cast(
                    pl.Float64,
                ),
            )
            .otherwise(None)
            .alias("momentum_indicator"),
            pl.when(pl.col("prev_midpoint_change").is_not_null())
            .then(
                (pl.col("midpoint_change") * pl.col("prev_midpoint_change") < 0).cast(
                    pl.Float64,
                ),
            )
            .otherwise(None)
            .alias("mean_reversion_indicator"),
        )
        .group_by(["hour_key", "market_id", "token_id", "side"])
        .agg(
            pl.len().alias("event_count"),
            pl.col("timestamp_received").min().alias("first_timestamp"),
            pl.col("timestamp_received").max().alias("last_timestamp"),
            pl.col("midpoint").last().alias("last_midpoint"),
            pl.col("spread").mean().alias("event_spread_mean"),
            pl.col("midpoint_change").mean().alias("midpoint_change_mean"),
            pl.col("abs_midpoint_change").mean().alias("abs_midpoint_change_mean"),
            pl.col("abs_midpoint_change").quantile(0.95).alias("abs_midpoint_change_p95"),
            pl.col("abs_midpoint_change").max().alias("abs_midpoint_change_max"),
            pl.col("sq_midpoint_change").sum().sqrt().alias("realized_volatility"),
            pl.col("sq_midpoint_change").mean().sqrt().alias("rms_midpoint_change"),
            pl.col("buy_event").mean().alias("buy_event_share"),
            pl.col("change_side_signal").mean().alias("change_side_imbalance"),
            pl.col("momentum_indicator").mean().alias("momentum_indicator_mean"),
            pl.col("mean_reversion_indicator").mean().alias("mean_reversion_indicator_mean"),
            pl.col("event_gap_s").mean().alias("event_gap_mean_s"),
            pl.col("event_gap_s").std().alias("event_gap_std_s"),
            (pl.col("abs_midpoint_change") >= config.jump_threshold_small)
            .mean()
            .alias("jump_rate_small"),
            (pl.col("abs_midpoint_change") >= config.jump_threshold_large)
            .mean()
            .alias("jump_rate_large"),
        )
        .with_columns(
            (pl.col("last_timestamp") - pl.col("first_timestamp"))
            .dt.total_milliseconds()
            .truediv(60_000.0)
            .alias("active_minutes"),
        )
        .with_columns(
            pl.when(pl.col("active_minutes") > 0)
            .then((pl.col("event_count") - 1) / pl.col("active_minutes"))
            .otherwise(None)
            .alias("change_intensity_per_min"),
        )
        .join(market_metadata, on=["market_id", "token_id"], how="inner")
        .select(list(PRICE_HOURLY_COLUMNS))
        .collect()
    )


def _market_keys_lazy(market_dimension: pl.DataFrame) -> pl.LazyFrame:
    return market_dimension.lazy().select(["market_id", "token_id"])


def _market_metadata_lazy(market_dimension: pl.DataFrame) -> pl.LazyFrame:
    return market_dimension.lazy().select(["market_id", "token_id", "coin", "resolution"])


def _existing_feature_artifacts(
    mode_cache_dir: Path,
    batches: list[HourlyBatch],
) -> FeatureArtifacts | None:
    book_hourly_paths = tuple(
        mode_cache_dir / "book_hourly" / f"book_{batch.hour_key}.parquet" for batch in batches
    )
    price_hourly_paths = tuple(
        mode_cache_dir / "price_hourly" / f"price_{batch.hour_key}.parquet" for batch in batches
    )
    batch_audit_path = mode_cache_dir / "batch_audit.parquet"
    expected_paths = (*book_hourly_paths, *price_hourly_paths, batch_audit_path)
    if not expected_paths or not all(path.exists() for path in expected_paths):
        return None
    return FeatureArtifacts(
        market_dimension_path=mode_cache_dir / "market_dimension.parquet",
        book_hourly_paths=book_hourly_paths,
        price_hourly_paths=price_hourly_paths,
        batch_audit_path=batch_audit_path,
    )
