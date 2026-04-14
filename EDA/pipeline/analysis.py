from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from EDA.core.io import HourlyBatch, inventory_summary, write_analysis_table
from EDA.core.utils import get_logger, safe_ratio

if TYPE_CHECKING:
    from EDA.core.config import RunConfig
    from EDA.pipeline.features import FeatureArtifacts

SPARSE_MARKET_THRESHOLD = 100


def build_analysis_outputs(
    config: RunConfig,
    inventory: pl.DataFrame,
    schema_summary: pl.DataFrame,
    batches: list[HourlyBatch],
    market_dimension: pl.DataFrame,
    artifacts: FeatureArtifacts,
) -> dict[str, pl.DataFrame]:
    logger = get_logger()
    logger.info("Loading cached hourly summaries")
    book_hourly = pl.scan_parquet([str(path) for path in artifacts.book_hourly_paths]).collect()
    price_hourly = pl.scan_parquet([str(path) for path in artifacts.price_hourly_paths]).collect()
    batch_audit = pl.read_parquet(artifacts.batch_audit_path)

    outputs: dict[str, pl.DataFrame] = {}
    outputs["inventory_summary"] = inventory_summary(inventory)
    outputs["schema_summary"] = schema_summary
    outputs["batch_audit_summary"] = summarize_batch_audit(batch_audit)
    outputs["coverage_table"] = build_coverage_table(market_dimension, book_hourly, price_hourly)
    outputs["sparse_coverage_table"] = (
        outputs["coverage_table"]
        .filter(
            (pl.col("unique_markets") < SPARSE_MARKET_THRESHOLD)
            | (pl.col("snapshot_count") <= 0)
            | (pl.col("event_count") <= 0),
        )
        .sort(["snapshot_count", "event_count", "unique_markets", "coin", "resolution"])
    )

    contract_book = build_contract_book_summary(book_hourly)
    contract_price = build_contract_price_summary(price_hourly)
    daily_stability_panel = build_daily_stability_panel(book_hourly, price_hourly)
    autocorr_summary = build_hourly_autocorr_summary(price_hourly, config)
    outputs["contract_book_summary"] = contract_book
    outputs["contract_price_summary"] = contract_price
    outputs["daily_stability_panel"] = daily_stability_panel
    outputs["autocorr_summary"] = autocorr_summary
    outputs["microstructure_summary"] = build_microstructure_summary(contract_book)
    outputs["return_summary"] = build_return_summary(contract_price, autocorr_summary)
    outputs["autocorr_long"] = build_autocorr_long(autocorr_summary, config)
    outputs["stability_table"] = build_stability_table(daily_stability_panel)
    outputs["ranking_table"] = build_research_rankings(
        outputs["coverage_table"],
        outputs["microstructure_summary"],
        outputs["return_summary"],
        outputs["stability_table"],
    )
    validate_sample_coverage(config, outputs["coverage_table"])
    outputs["coin_summary"] = build_coin_summary(outputs["ranking_table"])
    outputs["resolution_summary"] = build_resolution_summary(outputs["ranking_table"])
    outputs["selected_batches"] = pl.DataFrame(
        [
            {"hour_key": batch.hour_key, "batch_index": idx}
            for idx, batch in enumerate(batches, start=1)
        ],
    )

    tables_dir = config.paths.mode_cache_dir(config.mode) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        write_analysis_table(frame, tables_dir / f"{name}.parquet")
    return outputs


def validate_sample_coverage(config: RunConfig, coverage_table: pl.DataFrame) -> None:
    if not config.is_sample:
        return

    expected_universe = pl.DataFrame(
        [
            {"coin": coin, "resolution": resolution}
            for coin, resolution in config.expected_coin_resolution_pairs
        ],
    )
    coverage_check = (
        expected_universe.join(
            coverage_table.select(["coin", "resolution", "snapshot_count", "event_count"]),
            on=["coin", "resolution"],
            how="left",
        )
        .with_columns(
            pl.col("snapshot_count").fill_null(0),
            pl.col("event_count").fill_null(0),
        )
    )
    missing_pairs = coverage_check.filter(
        (pl.col("snapshot_count") <= 0) | (pl.col("event_count") <= 0),
    ).sort(["coin", "resolution"])
    if missing_pairs.is_empty():
        return

    missing_text = ", ".join(
        f"{row['coin']} {row['resolution']}" for row in missing_pairs.iter_rows(named=True)
    )
    msg = (
        "Sample mode must cover the full expected coin x resolution universe. "
        f"Missing observed activity for: {missing_text}."
    )
    raise RuntimeError(msg)


def summarize_batch_audit(batch_audit: pl.DataFrame) -> pl.DataFrame:
    return (
        batch_audit.group_by("source")
        .agg(
            pl.len().alias("hours_processed"),
            pl.col("file_count").sum().alias("files_processed"),
            pl.col("raw_rows").sum().alias("raw_rows"),
            pl.col("matched_rows").sum().alias("matched_rows"),
            pl.col("output_rows").sum().alias("output_rows"),
            pl.col("unmatched_rows").sum().alias("unmatched_rows"),
            pl.col("match_rate").mean().alias("average_match_rate"),
        )
        .sort("source")
    )


def build_coverage_table(
    market_dimension: pl.DataFrame,
    book_hourly: pl.DataFrame,
    price_hourly: pl.DataFrame,
) -> pl.DataFrame:
    market_coverage = (
        market_dimension.group_by(["coin", "resolution"])
        .agg(
            pl.col("market_id").n_unique().alias("unique_markets"),
            pl.len().alias("unique_contracts"),
            pl.col("event_start_time").min().alias("event_start_time_min"),
            pl.col("event_end_time").max().alias("event_end_time_max"),
        )
        .sort(["coin", "resolution"])
    )
    book_coverage = (
        book_hourly.group_by(["coin", "resolution"])
        .agg(
            pl.len().alias("instrument_hours"),
            pl.col("hour_key").n_unique().alias("active_hours"),
            pl.col("snapshot_count").sum().alias("snapshot_count"),
        )
        .sort(["coin", "resolution"])
    )
    price_coverage = (
        price_hourly.group_by(["coin", "resolution"])
        .agg(
            pl.col("hour_key").n_unique().alias("price_active_hours"),
            pl.col("event_count").sum().alias("event_count"),
        )
        .sort(["coin", "resolution"])
    )
    coverage = (
        market_coverage.join(book_coverage, on=["coin", "resolution"], how="left")
        .join(price_coverage, on=["coin", "resolution"], how="left")
        .with_columns(
            pl.col("snapshot_count").fill_null(0),
            pl.col("event_count").fill_null(0),
            pl.col("instrument_hours").fill_null(0),
            pl.col("active_hours").fill_null(0),
            pl.col("price_active_hours").fill_null(0),
        )
        .sort(["coin", "resolution"])
    )
    return coverage


def build_contract_book_summary(book_hourly: pl.DataFrame) -> pl.DataFrame:
    return book_hourly.group_by(["coin", "resolution", "market_id", "token_id", "side"]).agg(
        pl.len().alias("active_hours"),
        pl.col("snapshot_count").sum().alias("snapshot_count"),
        pl.col("spread_mean").mean().alias("spread_mean"),
        pl.col("spread_median").median().alias("spread_median"),
        pl.col("relative_spread_mean").mean().alias("relative_spread_mean"),
        pl.col("relative_spread_median").median().alias("relative_spread_median"),
        pl.col("top_of_book_depth_mean").mean().alias("top_of_book_depth_mean"),
        pl.col("top_of_book_depth_median").median().alias("top_of_book_depth_median"),
        pl.col("total_visible_depth_mean").mean().alias("total_visible_depth_mean"),
        pl.col("total_visible_depth_median").median().alias("total_visible_depth_median"),
        pl.col("order_book_imbalance_mean").mean().alias("order_book_imbalance_mean"),
        pl.col("liquidity_concentration_mean").mean().alias("liquidity_concentration_mean"),
        pl.col("midpoint_change_frequency").mean().alias("midpoint_change_frequency"),
        pl.col("stale_quote_frequency").mean().alias("stale_quote_frequency"),
        pl.col("quote_update_intensity_per_min").mean().alias("quote_update_intensity_per_min"),
        pl.col("snapshot_irregularity").mean().alias("snapshot_irregularity"),
        pl.col("crossed_quote_frequency").mean().alias("crossed_quote_frequency"),
        pl.col("locked_quote_frequency").mean().alias("locked_quote_frequency"),
        pl.col("missing_depth_side_frequency").mean().alias("missing_depth_side_frequency"),
    )


def build_contract_price_summary(price_hourly: pl.DataFrame) -> pl.DataFrame:
    return price_hourly.group_by(["coin", "resolution", "market_id", "token_id", "side"]).agg(
        pl.len().alias("active_hours"),
        pl.col("event_count").sum().alias("event_count"),
        pl.col("event_spread_mean").mean().alias("event_spread_mean"),
        pl.col("midpoint_change_mean").mean().alias("midpoint_change_mean"),
        pl.col("abs_midpoint_change_mean").mean().alias("abs_midpoint_change_mean"),
        pl.col("abs_midpoint_change_p95").median().alias("abs_midpoint_change_p95"),
        pl.col("abs_midpoint_change_max").max().alias("abs_midpoint_change_max"),
        pl.col("realized_volatility").mean().alias("realized_volatility"),
        pl.col("rms_midpoint_change").mean().alias("rms_midpoint_change"),
        pl.col("buy_event_share").mean().alias("buy_event_share"),
        pl.col("change_side_imbalance").mean().alias("change_side_imbalance"),
        pl.col("momentum_indicator_mean").mean().alias("momentum_indicator_mean"),
        pl.col("mean_reversion_indicator_mean").mean().alias("mean_reversion_indicator_mean"),
        pl.col("change_intensity_per_min").mean().alias("change_intensity_per_min"),
        pl.col("jump_rate_small").mean().alias("jump_rate_small"),
        pl.col("jump_rate_large").mean().alias("jump_rate_large"),
    )


def build_microstructure_summary(contract_book: pl.DataFrame) -> pl.DataFrame:
    return (
        contract_book.group_by(["coin", "resolution"])
        .agg(
            pl.col("market_id").n_unique().alias("contract_markets"),
            pl.len().alias("contract_sides"),
            pl.col("active_hours").median().alias("active_hours_median"),
            pl.col("snapshot_count").median().alias("snapshot_count_median"),
            pl.col("spread_mean").median().alias("spread_median"),
            pl.col("relative_spread_mean").median().alias("relative_spread_median"),
            pl.col("top_of_book_depth_mean").median().alias("top_of_book_depth_median"),
            pl.col("total_visible_depth_mean").median().alias("total_visible_depth_median"),
            pl.col("order_book_imbalance_mean").median().alias("imbalance_median"),
            pl.col("liquidity_concentration_mean")
            .median()
            .alias("liquidity_concentration_median"),
            pl.col("midpoint_change_frequency").median().alias("midpoint_change_frequency_median"),
            pl.col("stale_quote_frequency").median().alias("stale_quote_frequency_median"),
            pl.col("quote_update_intensity_per_min")
            .median()
            .alias("quote_update_intensity_median"),
            pl.col("snapshot_irregularity").median().alias("snapshot_irregularity_median"),
            pl.col("crossed_quote_frequency").median().alias("crossed_quote_frequency_median"),
            pl.col("locked_quote_frequency").median().alias("locked_quote_frequency_median"),
            pl.col("missing_depth_side_frequency")
            .median()
            .alias("missing_depth_side_frequency_median"),
        )
        .sort(["coin", "resolution"])
    )


def build_return_summary(
    contract_price: pl.DataFrame,
    autocorr_summary: pl.DataFrame,
) -> pl.DataFrame:
    summary = (
        contract_price.group_by(["coin", "resolution"])
        .agg(
            pl.col("event_count").median().alias("event_count_median"),
            pl.col("event_spread_mean").median().alias("event_spread_median"),
            pl.col("midpoint_change_mean").median().alias("midpoint_change_median"),
            pl.col("abs_midpoint_change_mean").median().alias("abs_midpoint_change_median"),
            pl.col("abs_midpoint_change_p95").median().alias("abs_midpoint_change_p95"),
            pl.col("abs_midpoint_change_max").max().alias("abs_midpoint_change_max"),
            pl.col("realized_volatility").median().alias("realized_volatility_median"),
            pl.col("rms_midpoint_change").median().alias("rms_midpoint_change_median"),
            pl.col("buy_event_share").median().alias("buy_event_share_median"),
            pl.col("change_side_imbalance").median().alias("change_side_imbalance_median"),
            pl.col("momentum_indicator_mean").median().alias("momentum_indicator_median"),
            pl.col("mean_reversion_indicator_mean")
            .median()
            .alias("mean_reversion_indicator_median"),
            pl.col("change_intensity_per_min").median().alias("change_intensity_median"),
            pl.col("jump_rate_small").median().alias("jump_rate_small_median"),
            pl.col("jump_rate_large").median().alias("jump_rate_large_median"),
        )
        .sort(["coin", "resolution"])
    )
    return summary.join(autocorr_summary, on=["coin", "resolution"], how="left")


def build_hourly_autocorr_summary(
    price_hourly: pl.DataFrame,
    config: RunConfig,
) -> pl.DataFrame:
    panel = (
        price_hourly.sort(["market_id", "token_id", "hour_key"])
        .with_columns(
            pl.col("last_midpoint")
            .shift(1)
            .over(["market_id", "token_id"])
            .alias("prev_hour_close"),
        )
        .with_columns(
            (pl.col("last_midpoint") - pl.col("prev_hour_close")).alias("hourly_return"),
        )
        .with_columns(pl.col("hourly_return").abs().alias("abs_hourly_return"))
        .with_columns(_hourly_lag_columns(config.acf_lags))
    )
    return panel.group_by(["coin", "resolution"]).agg(
        *[
            pl.corr("hourly_return", f"hourly_return_lag_{lag}").alias(
                f"acf_midpoint_lag_{lag}",
            )
            for lag in config.acf_lags
        ],
        *[
            pl.corr("abs_hourly_return", f"abs_hourly_return_lag_{lag}").alias(
                f"acf_abs_midpoint_lag_{lag}",
            )
            for lag in config.acf_lags
        ],
    )


def build_autocorr_long(autocorr_summary: pl.DataFrame, config: RunConfig) -> pl.DataFrame:
    lag_columns = [f"acf_midpoint_lag_{lag}" for lag in config.acf_lags]
    return (
        autocorr_summary.melt(
            id_vars=["coin", "resolution"],
            value_vars=lag_columns,
            variable_name="lag_name",
            value_name="autocorrelation",
        )
        .with_columns(
            pl.col("lag_name").str.extract(r"(\d+)$", 1).cast(pl.Int64).alias("lag"),
        )
        .sort(["resolution", "coin", "lag"])
    )


def build_daily_stability_panel(
    book_hourly: pl.DataFrame,
    price_hourly: pl.DataFrame,
) -> pl.DataFrame:
    book_daily = (
        book_hourly.with_columns(pl.col("first_timestamp").dt.date().alias("observation_day"))
        .group_by(["coin", "resolution", "observation_day"])
        .agg(
            pl.col("spread_mean").mean().alias("spread_mean"),
            pl.col("total_visible_depth_mean").mean().alias("total_visible_depth_mean"),
            pl.col("stale_quote_frequency").mean().alias("stale_quote_frequency"),
        )
    )
    price_daily = (
        price_hourly.with_columns(pl.col("first_timestamp").dt.date().alias("observation_day"))
        .group_by(["coin", "resolution", "observation_day"])
        .agg(
            pl.col("realized_volatility").mean().alias("realized_volatility"),
            pl.col("jump_rate_large").mean().alias("jump_rate_large"),
        )
    )
    return book_daily.join(price_daily, on=["coin", "resolution", "observation_day"], how="inner")


def build_stability_table(daily: pl.DataFrame) -> pl.DataFrame:
    return (
        daily.group_by(["coin", "resolution"])
        .agg(
            pl.len().alias("daily_points"),
            _coefficient_of_variation("spread_mean").alias("spread_cv"),
            _coefficient_of_variation("total_visible_depth_mean").alias("depth_cv"),
            _coefficient_of_variation("realized_volatility").alias("volatility_cv"),
            _coefficient_of_variation("jump_rate_large").alias("tail_risk_cv"),
        )
        .sort(["coin", "resolution"])
    )


def build_research_rankings(
    coverage_table: pl.DataFrame,
    microstructure_summary: pl.DataFrame,
    return_summary: pl.DataFrame,
    stability_table: pl.DataFrame,
) -> pl.DataFrame:
    combined = (
        coverage_table.join(microstructure_summary, on=["coin", "resolution"], how="left")
        .join(return_summary, on=["coin", "resolution"], how="left")
        .join(stability_table, on=["coin", "resolution"], how="left")
        .filter((pl.col("snapshot_count") > 0) & (pl.col("event_count") > 0))
        .with_columns(
            safe_ratio(
                pl.col("relative_spread_median"),
                pl.col("realized_volatility_median"),
                "spread_to_vol_ratio",
            ),
            safe_ratio(
                pl.col("relative_spread_median"),
                pl.col("abs_midpoint_change_p95"),
                "spread_to_tail_move_ratio",
            ),
        )
        .with_columns(
            _zscore_expr("unique_markets").alias("z_coverage"),
            _zscore_expr("total_visible_depth_median").alias("z_depth"),
            _zscore_expr("relative_spread_median").mul(-1).alias("z_spread_efficiency"),
            _zscore_expr("stale_quote_frequency_median").mul(-1).alias("z_freshness"),
            _zscore_expr("realized_volatility_median").mul(-1).alias("z_vol_stability"),
            _zscore_expr("jump_rate_large_median").mul(-1).alias("z_tail_risk"),
            _zscore_expr("spread_cv").mul(-1).alias("z_spread_stability"),
            _zscore_expr("spread_to_vol_ratio").mul(-1).alias("z_friction_to_move"),
        )
        .with_columns(
            (
                pl.col("z_coverage")
                + pl.col("z_depth")
                + pl.col("z_spread_efficiency")
                + pl.col("z_freshness")
                + pl.col("z_vol_stability")
                + pl.col("z_tail_risk")
                + pl.col("z_spread_stability")
                + pl.col("z_friction_to_move")
            ).alias("research_score"),
        )
        .sort("research_score", descending=True)
    )
    return combined


def build_coin_summary(ranking_table: pl.DataFrame) -> pl.DataFrame:
    return (
        ranking_table.group_by("coin")
        .agg(
            pl.col("research_score").mean().alias("average_research_score"),
            pl.col("resolution")
            .sort_by("research_score", descending=True)
            .first()
            .alias(
                "best_resolution",
            ),
            pl.col("relative_spread_median").mean().alias("average_relative_spread"),
            pl.col("realized_volatility_median").mean().alias("average_realized_volatility"),
        )
        .sort("average_research_score", descending=True)
    )


def build_resolution_summary(ranking_table: pl.DataFrame) -> pl.DataFrame:
    return (
        ranking_table.group_by("resolution")
        .agg(
            pl.col("research_score").mean().alias("average_research_score"),
            pl.col("relative_spread_median").mean().alias("average_relative_spread"),
            pl.col("stale_quote_frequency_median").mean().alias("average_stale_quote_frequency"),
            pl.col("realized_volatility_median").mean().alias("average_realized_volatility"),
        )
        .sort("average_research_score", descending=True)
    )


def _coefficient_of_variation(column: str) -> pl.Expr:
    cleaned = pl.col(column).fill_nan(None)
    return (
        pl.when(cleaned.mean().abs() > 0)
        .then(
            cleaned.std() / cleaned.mean().abs(),
        )
        .otherwise(None)
        .fill_nan(None)
    )


def _zscore_expr(column: str) -> pl.Expr:
    cleaned = pl.col(column).fill_nan(None)
    return (
        pl.when(cleaned.std().abs() > 0)
        .then(
            (cleaned - cleaned.mean()) / cleaned.std(),
        )
        .otherwise(0.0)
        .fill_nan(0.0)
        .fill_null(0.0)
    )


def _hourly_lag_columns(lags: tuple[int, ...]) -> list[pl.Expr]:
    expressions: list[pl.Expr] = []
    for lag in lags:
        expressions.append(
            pl.col("hourly_return")
            .shift(lag)
            .over(["market_id", "token_id"])
            .alias(f"hourly_return_lag_{lag}"),
        )
        expressions.append(
            pl.col("abs_hourly_return")
            .shift(lag)
            .over(["market_id", "token_id"])
            .alias(f"abs_hourly_return_lag_{lag}"),
        )
    return expressions
