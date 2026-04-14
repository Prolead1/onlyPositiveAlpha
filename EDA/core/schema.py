from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

SOURCE_NAMES = ("mapping", "book_snapshot", "price_change")
UTC_US_DATETIME = pl.Datetime(time_unit="us", time_zone="UTC")
CORE_SOURCE_COLUMNS: dict[str, tuple[str, ...]] = {
    "mapping": (
        "condition_id",
        "market_question",
        "market_slug",
        "gamma_market_clob_token_ids",
        "gamma_market_outcomes",
        "gamma_market_created_at",
        "gamma_market_start_date",
        "gamma_market_end_date",
        "gamma_market_updated_at",
        "gamma_market_order_price_min_tick_size",
        "gamma_market_order_min_size",
        "gamma_market_best_bid",
        "gamma_market_best_ask",
        "gamma_market_spread",
        "gamma_market_volume",
        "gamma_series_liquidity",
    ),
    "book_snapshot": (
        "timestamp_received",
        "timestamp_created_at",
        "market_id",
        "token_id",
        "side",
        "best_bid",
        "best_ask",
        "payload_timestamp_s",
        "bid_levels",
        "ask_levels",
        "bids",
        "asks",
    ),
    "price_change": (
        "timestamp_received",
        "timestamp_created_at",
        "market_id",
        "token_id",
        "side",
        "best_bid",
        "best_ask",
        "payload_timestamp_s",
        "change_price",
        "change_size",
        "change_side",
    ),
}
MAPPING_DIMENSION_COLUMNS = (
    "condition_id",
    "market_slug",
    "gamma_market_clob_token_ids",
    "gamma_market_start_date",
    "gamma_market_end_date",
)
MAPPING_UNIVERSE_COLUMNS = ("market_slug",)
MAPPING_SCAN_SCHEMA: dict[str, pl.DataType] = {
    "condition_id": pl.String,
    "market_question": pl.String,
    "market_slug": pl.String,
    "gamma_market_clob_token_ids": pl.List(pl.String),
    "gamma_market_outcomes": pl.List(pl.String),
    "gamma_market_created_at": UTC_US_DATETIME,
    "gamma_market_start_date": UTC_US_DATETIME,
    "gamma_market_end_date": UTC_US_DATETIME,
    "gamma_market_updated_at": UTC_US_DATETIME,
    "gamma_market_order_price_min_tick_size": pl.Float64,
    "gamma_market_order_min_size": pl.Float64,
    "gamma_market_best_bid": pl.Float64,
    "gamma_market_best_ask": pl.Float64,
    "gamma_market_spread": pl.Float64,
    "gamma_market_volume": pl.Float64,
    "gamma_series_liquidity": pl.Float64,
}

FILE_NAME_PATTERN = re.compile(
    r"^polymarket_orderbook_(?P<hour>\d{4}-\d{2}-\d{2}T\d{2})_"
    r"(?P<suffix>[a-z_]+?)(?:_chunk_(?P<chunk>\d+))?\.parquet$"
)


def parse_file_metadata(path: Path) -> dict[str, str | int | None]:
    match = FILE_NAME_PATTERN.match(path.name)
    if match is None:
        msg = f"Unexpected parquet filename: {path.name}"
        raise ValueError(msg)
    chunk_raw = match.group("chunk")
    return {
        "hour_key": match.group("hour"),
        "suffix": match.group("suffix"),
        "chunk_index": int(chunk_raw) if chunk_raw is not None else None,
    }


def coin_expr(slug: str = "market_slug") -> pl.Expr:
    return pl.col(slug).str.extract(r"^([a-z0-9]+)-updown-", 1).str.to_uppercase()


def resolution_expr(slug: str = "market_slug") -> pl.Expr:
    return pl.col(slug).str.extract(r"-updown-([0-9]+[mh])-", 1)
