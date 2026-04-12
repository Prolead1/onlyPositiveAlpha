from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
import requests
from tqdm.auto import tqdm

PRICE_CHANGE_SCHEMA = pl.Struct(
    {
        "update_type": pl.String,
        "market_id": pl.String,
        "token_id": pl.String,
        "side": pl.String,
        "best_bid": pl.String,
        "best_ask": pl.String,
        "timestamp": pl.Float64,
        "change_price": pl.String,
        "change_size": pl.String,
        "change_side": pl.String,
    }
)

BOOK_SNAPSHOT_SCHEMA = pl.Struct(
    {
        "update_type": pl.String,
        "market_id": pl.String,
        "token_id": pl.String,
        "side": pl.String,
        "best_bid": pl.String,
        "best_ask": pl.String,
        "timestamp": pl.Float64,
        "bids": pl.List(pl.List(pl.String)),
        "asks": pl.List(pl.List(pl.String)),
    }
)

DEFAULT_RAW_DIRNAME = "raw"
DEFAULT_BOOK_SNAPSHOT_DIRNAME = "book_snapshot"
DEFAULT_PRICE_CHANGE_DIRNAME = "price_change"
DEFAULT_MAPPING_DIRNAME = "mapping"
DEFAULT_INDEX_FILENAME = "split_manifest.json"
MAX_OUTPUT_FILE_SIZE_BYTES = 99 * 1024 * 1024
GAMMA_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
GAMMA_BATCH_SIZE = 20
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5
RAW_FILENAME_PATTERN = re.compile(r"polymarket_orderbook_(?P<capture>.+)\.parquet$")
UPDOWN_RESOLUTION_PATTERN = re.compile(r"-updown-(?P<resolution>[^-]+)-\d+$")
CRYPTO_TAG_SLUGS = frozenset({"crypto", "hype", "btc", "eth", "xrp", "doge", "sol", "bnb"})
MARKET_FIELDS_TO_KEEP = frozenset(
    {
        "id",
        "resolutionSource",
        "startDate",
        "endDate",
        "createdAt",
        "updatedAt",
        "closedTime",
        "outcomes",
        "outcomePrices",
        "volume",
        "active",
        "closed",
        "groupItemThreshold",
        "umaEndDate",
        "enableOrderBook",
        "orderPriceMinTickSize",
        "orderMinSize",
        "umaResolutionStatus",
        "clobTokenIds",
        "acceptingOrders",
        "acceptingOrdersTimestamp",
        "negRisk",
        "spread",
        "lastTradePrice",
        "bestBid",
        "bestAsk",
        "feeType",
        "makerBaseFee",
        "takerBaseFee",
    }
)
EVENT_FIELDS_TO_KEEP = frozenset({"id", "ticker", "startTime", "closedTime", "seriesSlug"})
SERIES_FIELDS_TO_KEEP = frozenset(
    {
        "id",
        "ticker",
        "slug",
        "title",
        "seriesType",
        "recurrence",
        "volume24hr",
        "volume",
        "liquidity",
    }
)
EVENT_METADATA_FIELDS_TO_KEEP = frozenset({"finalPrice", "priceToBeat"})
FEE_SCHEDULE_FIELDS_TO_KEEP = frozenset({"exponent", "rate", "takerOnly", "rebateRate"})
OUTPUT_NAMES = ("price_change", "book_snapshot", "mapping")
GENERAL_INFO_PATH_KEYS = (
    "dataset_root",
    "index_json_path",
    "book_snapshot_dir",
    "price_change_dir",
    "mapping_dir",
)


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """Resolved locations for the batch split workflow."""

    repo_root: Path
    dataset_root: Path
    raw_dir: Path
    book_snapshot_dir: Path
    price_change_dir: Path
    mapping_dir: Path
    index_path: Path


def _common_payload_columns(payload_col: str = "payload") -> list[pl.Expr]:
    return [
        pl.col(payload_col).struct.field("token_id").alias("token_id"),
        pl.col(payload_col).struct.field("side").alias("side"),
        pl.col(payload_col)
        .struct.field("best_bid")
        .cast(pl.Float64, strict=False)
        .alias("best_bid"),
        pl.col(payload_col)
        .struct.field("best_ask")
        .cast(pl.Float64, strict=False)
        .alias("best_ask"),
        pl.col(payload_col)
        .struct.field("timestamp")
        .cast(pl.Float64, strict=False)
        .alias("payload_timestamp_s"),
    ]


def _levels_expr(field_name: str, payload_col: str = "payload") -> pl.Expr:
    return (
        pl.col(payload_col)
        .struct.field(field_name)
        .list.eval(
            pl.struct(
                price=pl.element().list.get(0).cast(pl.Float64, strict=False),
                size=pl.element().list.get(1).cast(pl.Float64, strict=False),
            )
        )
        .alias(field_name)
    )


def _parse_iso_datetime(value: str) -> datetime:
    iso_value = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(iso_value)


def _first_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, dict)), {})
    return {}


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _utc_isoformat(value: object) -> str | None:
    raw = _optional_string(value)
    if raw is None:
        return None

    try:
        parsed = _parse_iso_datetime(raw)
    except ValueError:
        return raw

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _resolution_from_slug(market_slug: str | None) -> str | None:
    if market_slug is None:
        return None

    match = UPDOWN_RESOLUTION_PATTERN.search(market_slug)
    if match is None:
        return None
    return match.group("resolution")


def snake_case(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.replace("-", "_").lower()


def extract_tag_slugs(raw_tags: list[dict[str, Any]] | None) -> list[Any | None]:
    return [
        tag.get("slug")
        for tag in raw_tags or []
        if isinstance(tag, dict) and isinstance(tag.get("slug"), str) and tag.get("slug")
    ]


def parse_temporal_value(column: str, value: str) -> date | datetime | str:
    parsed_value: date | datetime | str = value

    if column.endswith("_date_iso"):
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError:
            parsed_value = value
        return parsed_value

    if not column.endswith(("_date", "_time", "_timestamp", "_at")):
        return parsed_value

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            parsed_value = date.fromisoformat(value)
        except ValueError:
            parsed_value = value
        return parsed_value

    try:
        parsed_value = _parse_iso_datetime(value)
    except ValueError:
        parsed_value = value
    return parsed_value


def should_keep_string(column: str, value: str) -> bool:
    if value.startswith("0x"):
        return True

    if column in {"condition_id", "series_slug", "fee_type"}:
        return True

    return column.endswith(("_ids", "_address"))


def normalize_loaded_json(key: str, value: object) -> object:
    if isinstance(value, list):
        return [normalize_loaded_json(key, item) for item in value]
    if isinstance(value, dict):
        return {
            sub_key: normalize_loaded_json(sub_key, sub_value)
            for sub_key, sub_value in value.items()
        }
    return normalize_value(key, value)


def parse_numeric_value(column: str, value: str) -> int | float | str:
    if should_keep_string(column, value):
        return value

    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)

    if re.fullmatch(
        r"[+-]?(?:\d+\.\d*|\.\d+|\d+(?:[eE][+-]?\d+)|\d+\.\d*(?:[eE][+-]?\d+))",
        value,
    ):
        number = float(value)
        if (
            column
            in {
                "group_item_threshold",
                "order_min_size",
                "maker_base_fee",
                "taker_base_fee",
                "exponent",
            }
            and number.is_integer()
        ):
            return int(number)
        return number

    return value


def normalize_value(key: str, value: object) -> object:
    if value is None:
        return None
    if not isinstance(value, str):
        return value

    stripped_value = value.strip()
    if not stripped_value:
        return None

    column = snake_case(key)

    if stripped_value[0] in "[{":
        try:
            return normalize_loaded_json(key, json.loads(stripped_value))
        except json.JSONDecodeError:
            pass

    temporal_value = parse_temporal_value(column, stripped_value)
    if temporal_value != stripped_value:
        return temporal_value

    return parse_numeric_value(column, stripped_value)


def flatten_selected_value(prefix: str, key: str, value: object) -> dict[str, Any]:
    column = f"{prefix}_{snake_case(key)}"
    normalized = normalize_value(key, value)
    row: dict[str, Any] = {}
    if isinstance(normalized, dict):
        for nested_key, nested_value in normalized.items():
            row.update(flatten_selected_value(column, nested_key, nested_value))
    else:
        row[column] = normalized
    return row


def flatten_selected_fields(
    prefix: str,
    payload: dict[str, Any],
    keep_fields: set[str] | frozenset[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in sorted(keep_fields):
        if key in payload:
            row.update(flatten_selected_value(prefix, key, payload[key]))
    return row


def serialize_manifest_time(value: object) -> str | None:
    serialized: str | None = None

    if value is None:
        return serialized

    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        serialized = normalized.astimezone(UTC).isoformat()
        return serialized

    if isinstance(value, date):
        serialized = value.isoformat()
        return serialized

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return serialized
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
            serialized = stripped
        else:
            serialized = _utc_isoformat(stripped)
        return serialized

    return str(value)


def empty_crypto_markets_mapping_df() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "condition_id": pl.String,
            "market_question": pl.String,
            "market_slug": pl.String,
            "event_title": pl.String,
            "event_slug": pl.String,
            "tags": pl.List(pl.String),
            "crypto_tags": pl.List(pl.String),
        }
    )


def crypto_market_mapping_row_from_payload(
    market: dict[str, Any],
) -> dict[str, Any] | None:
    condition_id = _optional_string(market.get("conditionId"))
    market_slug = _optional_string(market.get("slug"))
    if condition_id is None or market_slug is None or "updown" not in market_slug:
        return None

    tag_records = market.get("tags") if isinstance(market.get("tags"), list) else []
    tags = sorted(set(extract_tag_slugs(tag_records)))
    crypto_tags = sorted(set(tags) & CRYPTO_TAG_SLUGS)
    if not crypto_tags:
        return None

    event = _first_dict(market.get("events"))
    series = _first_dict(event.get("series"))
    fee_schedule = market.get("feeSchedule") if isinstance(market.get("feeSchedule"), dict) else {}
    event_metadata = (
        event.get("eventMetadata") if isinstance(event.get("eventMetadata"), dict) else {}
    )

    row: dict[str, Any] = {
        "condition_id": condition_id,
        "market_question": _optional_string(market.get("question")),
        "market_slug": market_slug,
        "event_title": _optional_string(event.get("title")),
        "event_slug": _optional_string(event.get("slug")),
        "tags": tags,
        "crypto_tags": crypto_tags,
    }
    row.update(flatten_selected_fields("gamma_market", market, MARKET_FIELDS_TO_KEEP))
    row.update(flatten_selected_fields("gamma_event", event, EVENT_FIELDS_TO_KEEP))
    row.update(flatten_selected_fields("gamma_series", series, SERIES_FIELDS_TO_KEEP))
    row.update(
        flatten_selected_fields(
            "gamma_event_metadata",
            event_metadata,
            EVENT_METADATA_FIELDS_TO_KEEP,
        )
    )
    row.update(
        flatten_selected_fields(
            "gamma_fee_schedule",
            fee_schedule,
            FEE_SCHEDULE_FIELDS_TO_KEEP,
        )
    )
    return row


def mapping_row_to_manifest_market(row: dict[str, Any]) -> dict[str, Any] | None:
    condition_id = _optional_string(row.get("condition_id"))
    if condition_id is None:
        return None

    market_slug = _optional_string(row.get("market_slug"))
    resolution = _optional_string(row.get("gamma_series_recurrence")) or _resolution_from_slug(
        market_slug
    )
    event_start_time_utc = serialize_manifest_time(
        row.get("gamma_event_start_time") or row.get("gamma_market_start_date")
    )
    event_end_time_utc = serialize_manifest_time(
        row.get("gamma_market_end_date")
        or row.get("gamma_event_closed_time")
        or row.get("gamma_market_closed_time")
    )

    return {
        "condition_id": condition_id,
        "market_slug": market_slug,
        "market_question": _optional_string(row.get("market_question")),
        "event_start_time_utc": event_start_time_utc,
        "event_end_time_utc": event_end_time_utc,
        "resolution": resolution,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split raw Polymarket orderbook parquet files into book_snapshot and "
            "price_change parquet datasets and maintain an append-only JSON index."
        ),
    )
    parser.add_argument(
        "orderbook_paths",
        nargs="*",
        type=Path,
        help=(
            "Optional raw parquet file paths to process. If omitted, every parquet file "
            "inside raw-dir is processed."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing raw parquet files. Defaults to <script_dir>/raw.",
    )
    parser.add_argument(
        "--book-snapshot-dir",
        type=Path,
        default=None,
        help=(
            "Directory for split book_snapshot parquet files. "
            "Defaults to <script_dir>/book_snapshot."
        ),
    )
    parser.add_argument(
        "--price-change-dir",
        type=Path,
        default=None,
        help=(
            "Directory for split price_change parquet files. "
            "Defaults to <script_dir>/price_change."
        ),
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-source crypto updown mapping parquet files. "
            "Defaults to <script_dir>/mapping."
        ),
    )
    parser.add_argument(
        "--index-json",
        type=Path,
        default=None,
        help="Dataset-level JSON index path. Defaults to <script_dir>/split_manifest.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Reprocess files even if the same source file fingerprint already exists in the index."
        ),
    )
    return parser.parse_args(argv)


def default_dataset_root(script_path: str | Path | None = None) -> Path:
    source = Path(script_path) if script_path is not None else Path(__file__)
    return source.resolve().parent


def find_repo_root(script_path: str | Path | None = None) -> Path:
    source = Path(script_path) if script_path is not None else Path(__file__)
    search_root = source.resolve()
    if search_root.is_file():
        search_root = search_root.parent

    for candidate in (search_root, *search_root.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return search_root


def resolve_runtime_paths(  # noqa: PLR0913
    *,
    script_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
    book_snapshot_dir: str | Path | None = None,
    price_change_dir: str | Path | None = None,
    mapping_dir: str | Path | None = None,
    index_path: str | Path | None = None,
) -> RuntimePaths:
    repo_root = find_repo_root(script_path)
    dataset_root = default_dataset_root(script_path)
    resolved_raw_dir = Path(raw_dir) if raw_dir is not None else dataset_root / DEFAULT_RAW_DIRNAME
    resolved_book_snapshot_dir = (
        Path(book_snapshot_dir)
        if book_snapshot_dir is not None
        else dataset_root / DEFAULT_BOOK_SNAPSHOT_DIRNAME
    )
    resolved_price_change_dir = (
        Path(price_change_dir)
        if price_change_dir is not None
        else dataset_root / DEFAULT_PRICE_CHANGE_DIRNAME
    )
    resolved_mapping_dir = (
        Path(mapping_dir) if mapping_dir is not None else dataset_root / DEFAULT_MAPPING_DIRNAME
    )
    resolved_index_path = (
        Path(index_path) if index_path is not None else dataset_root / DEFAULT_INDEX_FILENAME
    )
    return RuntimePaths(
        repo_root=repo_root,
        dataset_root=dataset_root,
        raw_dir=resolved_raw_dir,
        book_snapshot_dir=resolved_book_snapshot_dir,
        price_change_dir=resolved_price_change_dir,
        mapping_dir=resolved_mapping_dir,
        index_path=resolved_index_path,
    )


def repo_relative_path(path: str | Path, repo_root: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    resolved_repo_root = repo_root.expanduser().resolve()
    try:
        return str(resolved.relative_to(resolved_repo_root))
    except ValueError:
        return str(resolved)


def manifest_path_to_real_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (repo_root.expanduser().resolve() / path).resolve()


def collect_market_ids(source: pl.LazyFrame) -> list[str]:
    market_ids_df = source.select("market_id").drop_nulls().unique().sort("market_id").collect()
    if "market_id" not in market_ids_df.columns:
        return []
    return market_ids_df.get_column("market_id").to_list()


def build_orderbook_views(
    orderbook_path: str | Path,
    *,
    mapping_cache: dict[str, dict[str, Any] | None] | None = None,
    metadata_session: requests.Session | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Build lazy views for a single raw orderbook parquet file.

    Returns:
        tuple:
            - price_change LazyFrame filtered to crypto updown markets
            - book_snapshot LazyFrame filtered to crypto updown markets
            - crypto market mapping LazyFrame
    """
    source = pl.scan_parquet(orderbook_path)
    market_ids = collect_market_ids(source)

    cache = mapping_cache if mapping_cache is not None else {}
    owns_session = metadata_session is None
    session = metadata_session or requests.Session()
    try:
        crypto_markets_mapping = resolve_crypto_markets_mapping(
            market_ids,
            mapping_cache=cache,
            session=session,
        )
    finally:
        if owns_session:
            session.close()

    allowed_market_ids = (
        crypto_markets_mapping["condition_id"].to_list()
        if "condition_id" in crypto_markets_mapping.columns
        else []
    )
    filtered_source = source.filter(pl.col("market_id").is_in(allowed_market_ids))

    price_change = (
        filtered_source.filter(pl.col("update_type") == "price_change")
        .with_columns(payload=pl.col("data").str.json_decode(PRICE_CHANGE_SCHEMA))
        .select(
            "timestamp_received",
            "timestamp_created_at",
            "market_id",
            *_common_payload_columns(),
            pl.col("payload")
            .struct.field("change_price")
            .cast(pl.Float64, strict=False)
            .alias("change_price"),
            pl.col("payload")
            .struct.field("change_size")
            .cast(pl.Float64, strict=False)
            .alias("change_size"),
            pl.col("payload").struct.field("change_side").alias("change_side"),
        )
    )

    book_snapshot = (
        filtered_source.filter(pl.col("update_type") == "book_snapshot")
        .with_columns(payload=pl.col("data").str.json_decode(BOOK_SNAPSHOT_SCHEMA))
        .select(
            "timestamp_received",
            "timestamp_created_at",
            "market_id",
            *_common_payload_columns(),
            pl.col("payload").struct.field("bids").list.len().alias("bid_levels"),
            pl.col("payload").struct.field("asks").list.len().alias("ask_levels"),
            _levels_expr("bids"),
            _levels_expr("asks"),
        )
    )

    return price_change, book_snapshot, crypto_markets_mapping.lazy()


def build_output_paths(
    orderbook_path: str | Path,
    *,
    book_snapshot_dir: str | Path,
    price_change_dir: str | Path,
    mapping_dir: str | Path,
) -> dict[str, Path]:
    input_path = Path(orderbook_path)
    base_name = input_path.stem
    return {
        "price_change": Path(price_change_dir) / f"{base_name}_price_change.parquet",
        "book_snapshot": Path(book_snapshot_dir) / f"{base_name}_book_snapshot.parquet",
        "mapping": Path(mapping_dir) / f"{base_name}_crypto_markets_mapping.parquet",
    }


def chunk_output_path(destination: Path, chunk_index: int) -> Path:
    return destination.with_name(f"{destination.stem}_chunk_{chunk_index:04d}{destination.suffix}")


def cleanup_existing_output_files(destination: Path) -> None:
    destination.unlink(missing_ok=True)
    for chunk_path in destination.parent.glob(f"{destination.stem}_chunk_*{destination.suffix}"):
        chunk_path.unlink(missing_ok=True)


def write_lazyframe_slice_parquet(
    frame: pl.LazyFrame,
    destination: Path,
    *,
    offset: int = 0,
    length: int | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    slice_frame = frame if length is None else frame.slice(offset, length)
    tmp_path.unlink(missing_ok=True)
    try:
        slice_frame.sink_parquet(tmp_path)
        tmp_path.replace(destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def materialize_chunk_temp_files(
    frame: pl.LazyFrame,
    row_count: int,
    *,
    max_file_size_bytes: int,
    tmp_dir: Path,
) -> list[Path]:
    counter = 0

    def split_range(offset: int, length: int) -> list[Path]:
        nonlocal counter

        temp_path = tmp_dir / f"chunk_candidate_{counter:04d}.parquet"
        counter += 1
        write_lazyframe_slice_parquet(frame, temp_path, offset=offset, length=length)
        file_size = temp_path.stat().st_size

        if file_size <= max_file_size_bytes or length <= 1:
            if file_size > max_file_size_bytes and length <= 1:
                msg = f"Unable to split {temp_path.name} below {max_file_size_bytes} bytes."
                raise ValueError(msg)
            return [temp_path]

        temp_path.unlink(missing_ok=True)
        left_length = length // 2
        right_length = length - left_length
        return split_range(offset, left_length) + split_range(offset + left_length, right_length)

    if row_count == 0:
        empty_path = tmp_dir / "chunk_candidate_0000.parquet"
        write_lazyframe_slice_parquet(frame, empty_path, offset=0, length=0)
        return [empty_path]

    return split_range(0, row_count)


def write_lazyframe_parquet(
    frame: pl.LazyFrame,
    destination: Path,
    *,
    max_file_size_bytes: int = MAX_OUTPUT_FILE_SIZE_BYTES,
) -> list[Path]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count_df = frame.select(pl.len().alias("row_count")).collect()
    row_count = int(row_count_df["row_count"][0])

    with TemporaryDirectory(dir=destination.parent) as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        temp_paths = materialize_chunk_temp_files(
            frame,
            row_count,
            max_file_size_bytes=max_file_size_bytes,
            tmp_dir=tmp_dir,
        )
        final_paths = (
            [destination]
            if len(temp_paths) == 1
            else [chunk_output_path(destination, idx) for idx in range(1, len(temp_paths) + 1)]
        )

        cleanup_existing_output_files(destination)
        for temp_path, final_path in zip(temp_paths, final_paths, strict=True):
            temp_path.replace(final_path)

    return final_paths


def write_json_atomic(destination: Path, payload: list[dict[str, Any]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(f"{destination.suffix}.tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(destination)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_index_entries(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.exists():
        return []

    raw_text = index_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    entries = json.loads(raw_text)
    if not isinstance(entries, list):
        msg = f"Expected {index_path} to contain a JSON list."
        raise TypeError(msg)
    if any(not isinstance(entry, dict) for entry in entries):
        msg = f"Expected every entry in {index_path} to be a JSON object."
        raise TypeError(msg)
    return entries


def normalize_output_info(output_info: object, repo_root: Path) -> dict[str, Any] | None:
    if not isinstance(output_info, dict):
        return None

    normalized_files: list[dict[str, Any]] = []
    raw_files = output_info.get("files")
    if isinstance(raw_files, list):
        for file_info in raw_files:
            if not isinstance(file_info, dict):
                continue
            raw_path = file_info.get("path")
            size_bytes = file_info.get("size_bytes")
            if isinstance(raw_path, str) and isinstance(size_bytes, int):
                normalized_files.append(
                    {
                        "path": repo_relative_path(
                            manifest_path_to_real_path(raw_path, repo_root),
                            repo_root,
                        ),
                        "size_bytes": size_bytes,
                    }
                )
    else:
        raw_path = output_info.get("path")
        size_bytes = output_info.get("size_bytes")
        if isinstance(raw_path, str) and isinstance(size_bytes, int):
            normalized_files.append(
                {
                    "path": repo_relative_path(
                        manifest_path_to_real_path(raw_path, repo_root),
                        repo_root,
                    ),
                    "size_bytes": size_bytes,
                }
            )

    if not normalized_files:
        return None

    return {
        "files": normalized_files,
        "chunk_count": len(normalized_files),
        "total_size_bytes": sum(file_info["size_bytes"] for file_info in normalized_files),
    }


def normalize_market_metadata_list(markets: object) -> list[dict[str, Any]] | None:
    if not isinstance(markets, list):
        return None

    normalized_markets: list[dict[str, Any]] = []
    for market in markets:
        if not isinstance(market, dict):
            continue

        condition_id = market.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id:
            continue

        normalized_markets.append(
            {
                "condition_id": condition_id,
                "market_slug": _optional_string(market.get("market_slug")),
                "market_question": _optional_string(market.get("market_question")),
                "event_start_time_utc": _optional_string(market.get("event_start_time_utc")),
                "event_end_time_utc": _optional_string(market.get("event_end_time_utc")),
                "resolution": _optional_string(market.get("resolution")),
            }
        )

    return normalized_markets


def normalize_source_info(source_info: object, repo_root: Path) -> dict[str, Any] | None:
    if not isinstance(source_info, dict):
        return None

    normalized_source = dict(source_info)
    raw_path = source_info.get("path")
    if isinstance(raw_path, str):
        normalized_source["path"] = repo_relative_path(
            manifest_path_to_real_path(raw_path, repo_root),
            repo_root,
        )
    return normalized_source


def normalize_output_files(output_files: object, repo_root: Path) -> dict[str, Any] | None:
    if not isinstance(output_files, dict):
        return None

    normalized_output_files: dict[str, Any] = {}
    for output_name in OUTPUT_NAMES:
        normalized_output = normalize_output_info(output_files.get(output_name), repo_root)
        if normalized_output is not None:
            normalized_output_files[output_name] = normalized_output
    return normalized_output_files


def output_file_names(output_info: object) -> list[str]:
    if not isinstance(output_info, dict):
        return []

    return [
        Path(file_info["path"]).name
        for file_info in output_info.get("files", [])
        if isinstance(file_info, dict) and isinstance(file_info.get("path"), str)
    ]


def normalize_general_info(
    general_info: object,
    repo_root: Path,
    output_files: object,
) -> dict[str, Any] | None:
    if not isinstance(general_info, dict):
        return None

    normalized_general_info = dict(general_info)
    for key in GENERAL_INFO_PATH_KEYS:
        raw_path = general_info.get(key)
        if isinstance(raw_path, str):
            normalized_general_info[key] = repo_relative_path(
                manifest_path_to_real_path(raw_path, repo_root),
                repo_root,
            )

    if not isinstance(output_files, dict):
        return normalized_general_info

    for output_name in OUTPUT_NAMES:
        file_names = output_file_names(output_files.get(output_name))
        normalized_general_info[f"{output_name}_file_names"] = file_names
        singular_key = f"{output_name}_file_name"
        if len(file_names) == 1:
            normalized_general_info[singular_key] = file_names[0]
        else:
            normalized_general_info.pop(singular_key, None)

    return normalized_general_info


def normalize_index_entry(entry: dict[str, Any], runtime_paths: RuntimePaths) -> dict[str, Any]:
    normalized_entry = dict(entry)
    repo_root = runtime_paths.repo_root

    normalized_source = normalize_source_info(entry.get("source_file"), repo_root)
    if normalized_source is not None:
        normalized_entry["source_file"] = normalized_source

    normalized_output_files = normalize_output_files(entry.get("output_files"), repo_root)
    if normalized_output_files is not None:
        normalized_entry["output_files"] = normalized_output_files

    normalized_general_info = normalize_general_info(
        entry.get("general_info"),
        repo_root,
        normalized_entry.get("output_files"),
    )
    if normalized_general_info is not None:
        normalized_entry["general_info"] = normalized_general_info

    normalized_markets = normalize_market_metadata_list(entry.get("markets"))
    if normalized_markets is not None:
        normalized_entry["markets"] = normalized_markets

    return normalized_entry


def normalize_index_entries(
    entries: list[dict[str, Any]],
    runtime_paths: RuntimePaths,
) -> list[dict[str, Any]]:
    return [normalize_index_entry(entry, runtime_paths) for entry in entries]


def source_fingerprint(orderbook_path: Path) -> tuple[str, int, str]:
    stat = orderbook_path.stat()
    modified_at_utc = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return str(orderbook_path.resolve()), stat.st_size, modified_at_utc


def manifest_source_path(entry: dict[str, Any], repo_root: Path) -> str | None:
    source_info = entry.get("source_file")
    if not isinstance(source_info, dict):
        return None

    raw_path = source_info.get("path")
    if not isinstance(raw_path, str):
        return None
    return str(manifest_path_to_real_path(raw_path, repo_root))


def replace_entry_for_source(
    entries: list[dict[str, Any]],
    orderbook_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    resolved_source_path = str(orderbook_path.resolve())
    return [
        entry
        for entry in entries
        if manifest_source_path(entry, repo_root) != resolved_source_path
    ]


def manifest_file_matches(
    file_info: object,
    runtime_paths: RuntimePaths,
    *,
    max_file_size_bytes: int,
) -> bool:
    if not isinstance(file_info, dict):
        return False

    raw_path = file_info.get("path")
    size_bytes = file_info.get("size_bytes")
    if not isinstance(raw_path, str) or not isinstance(size_bytes, int):
        return False
    if size_bytes > max_file_size_bytes:
        return False

    resolved_path = manifest_path_to_real_path(raw_path, runtime_paths.repo_root)
    return resolved_path.is_file() and resolved_path.stat().st_size == size_bytes


def output_file_is_indexed(
    output_info: object,
    runtime_paths: RuntimePaths,
    *,
    max_file_size_bytes: int,
) -> bool:
    normalized_output = normalize_output_info(output_info, runtime_paths.repo_root)
    if normalized_output is None:
        return False

    files = normalized_output.get("files", [])
    if not isinstance(files, list) or not files:
        return False

    return all(
        manifest_file_matches(
            file_info,
            runtime_paths,
            max_file_size_bytes=max_file_size_bytes,
        )
        for file_info in files
    )


def entry_has_market_metadata(entry: dict[str, Any]) -> bool:
    return normalize_market_metadata_list(entry.get("markets")) is not None


def entry_has_complete_outputs(
    entry: dict[str, Any],
    runtime_paths: RuntimePaths,
    *,
    max_file_size_bytes: int,
) -> bool:
    output_files = entry.get("output_files")
    if not isinstance(output_files, dict):
        return False

    if not entry_has_market_metadata(entry):
        return False

    return (
        output_file_is_indexed(
            output_files.get("price_change"),
            runtime_paths,
            max_file_size_bytes=max_file_size_bytes,
        )
        and output_file_is_indexed(
            output_files.get("book_snapshot"),
            runtime_paths,
            max_file_size_bytes=max_file_size_bytes,
        )
        and output_file_is_indexed(
            output_files.get("mapping"),
            runtime_paths,
            max_file_size_bytes=max_file_size_bytes,
        )
    )


def indexed_fingerprints(
    entries: list[dict[str, Any]],
    runtime_paths: RuntimePaths,
    *,
    max_file_size_bytes: int,
) -> set[tuple[str, int, str]]:
    fingerprints: set[tuple[str, int, str]] = set()
    for entry in entries:
        if not entry_has_complete_outputs(
            entry,
            runtime_paths,
            max_file_size_bytes=max_file_size_bytes,
        ):
            continue

        source_info = entry.get("source_file")
        if not isinstance(source_info, dict):
            continue
        raw_path = source_info.get("path")
        size_bytes = source_info.get("size_bytes")
        modified_at_utc = source_info.get("modified_at_utc")
        if (
            isinstance(raw_path, str)
            and isinstance(size_bytes, int)
            and isinstance(modified_at_utc, str)
        ):
            resolved_path = manifest_path_to_real_path(raw_path, runtime_paths.repo_root)
            fingerprints.add((str(resolved_path), size_bytes, modified_at_utc))
    return fingerprints


def extract_capture_time_utc(orderbook_path: Path) -> str | None:
    match = RAW_FILENAME_PATTERN.fullmatch(orderbook_path.name)
    if match is None:
        return None

    capture_label = match.group("capture")
    try:
        capture_time = datetime.fromisoformat(capture_label)
    except ValueError:
        return capture_label

    if capture_time.tzinfo is None:
        capture_time = capture_time.replace(tzinfo=UTC)
    return capture_time.astimezone(UTC).isoformat()


def fetch_markets_batch(
    session: requests.Session,
    condition_ids: list[str],
) -> list[dict[str, Any]]:
    response = session.get(
        GAMMA_MARKETS_URL,
        params={"condition_ids": condition_ids, "include_tag": "true"},
        timeout=30,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError:
        if response.status_code == HTTPStatus.REQUEST_URI_TOO_LONG and len(condition_ids) > 1:
            midpoint = len(condition_ids) // 2
            return fetch_markets_batch(session, condition_ids[:midpoint]) + fetch_markets_batch(
                session,
                condition_ids[midpoint:],
            )
        raise

    payload = response.json()
    return payload if isinstance(payload, list) else []


def fetch_markets_batch_with_retries(
    session: requests.Session,
    condition_ids: list[str],
) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_markets_batch(session, condition_ids)
        except requests.RequestException:
            if attempt == MAX_RETRIES:
                return markets
            time.sleep(RETRY_DELAY_SECONDS)
    return markets


def cache_mapping_rows(
    markets: list[dict[str, Any]],
    mapping_cache: dict[str, dict[str, Any] | None],
) -> set[str]:
    resolved_ids: set[str] = set()
    for market in markets:
        condition_id = _optional_string(market.get("conditionId"))
        if condition_id is None:
            continue
        mapping_cache[condition_id] = crypto_market_mapping_row_from_payload(market)
        resolved_ids.add(condition_id)
    return resolved_ids


def build_mapping_records(
    condition_ids: list[str],
    mapping_cache: dict[str, dict[str, Any] | None],
) -> list[dict[Any, Any]]:
    return [
        dict(mapping_cache[condition_id])
        for condition_id in condition_ids
        if isinstance(mapping_cache.get(condition_id), dict)
    ]


def resolve_crypto_markets_mapping(
    condition_ids: list[str],
    *,
    mapping_cache: dict[str, dict[str, Any] | None],
    session: requests.Session,
) -> pl.DataFrame:
    unique_ids = [
        condition_id
        for condition_id in dict.fromkeys(condition_ids)
        if isinstance(condition_id, str) and condition_id
    ]
    missing_ids = [
        condition_id for condition_id in unique_ids if condition_id not in mapping_cache
    ]

    for start in range(0, len(missing_ids), GAMMA_BATCH_SIZE):
        batch_ids = missing_ids[start : start + GAMMA_BATCH_SIZE]
        markets = fetch_markets_batch_with_retries(session, batch_ids)
        resolved_ids = cache_mapping_rows(markets, mapping_cache)

        for condition_id in batch_ids:
            if condition_id not in resolved_ids:
                mapping_cache.setdefault(condition_id, None)

    records = build_mapping_records(unique_ids, mapping_cache)
    if not records:
        return empty_crypto_markets_mapping_df()

    return (
        pl.DataFrame(records)
        .unique(subset=["condition_id"])
        .sort(["event_title", "market_question"])
    )


def build_output_manifest(paths: list[Path], runtime_paths: RuntimePaths) -> dict[str, Any]:
    files = [
        {
            "path": repo_relative_path(path, runtime_paths.repo_root),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {
        "files": files,
        "chunk_count": len(files),
        "total_size_bytes": sum(file_info["size_bytes"] for file_info in files),
    }


def mapping_frame_to_manifest_markets(
    mapping_frame: pl.DataFrame,
) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    for row in mapping_frame.to_dicts():
        market = mapping_row_to_manifest_market(row)
        if market is not None:
            markets.append(market)
    return markets


def build_index_entry(
    orderbook_path: Path,
    output_paths: dict[str, list[Path]],
    runtime_paths: RuntimePaths,
    markets: list[dict[str, Any]],
) -> dict[str, Any]:
    source_stat = orderbook_path.stat()
    price_change_paths = output_paths["price_change"]
    book_snapshot_paths = output_paths["book_snapshot"]
    mapping_paths = output_paths["mapping"]

    return {
        "indexed_at_utc": datetime.now(tz=UTC).isoformat(),
        "source_file": {
            "name": orderbook_path.name,
            "stem": orderbook_path.stem,
            "path": repo_relative_path(orderbook_path, runtime_paths.repo_root),
            "size_bytes": source_stat.st_size,
            "modified_at_utc": datetime.fromtimestamp(source_stat.st_mtime, tz=UTC).isoformat(),
            "capture_time_utc": extract_capture_time_utc(orderbook_path),
        },
        "markets": markets,
        "output_files": {
            "price_change": build_output_manifest(price_change_paths, runtime_paths),
            "book_snapshot": build_output_manifest(book_snapshot_paths, runtime_paths),
            "mapping": build_output_manifest(mapping_paths, runtime_paths),
        },
        "general_info": {
            "split_types": ["price_change", "book_snapshot", "mapping"],
            "dataset_root": repo_relative_path(
                runtime_paths.dataset_root, runtime_paths.repo_root
            ),
            "index_json_path": repo_relative_path(
                runtime_paths.index_path, runtime_paths.repo_root
            ),
            "book_snapshot_dir": repo_relative_path(
                runtime_paths.book_snapshot_dir,
                runtime_paths.repo_root,
            ),
            "price_change_dir": repo_relative_path(
                runtime_paths.price_change_dir,
                runtime_paths.repo_root,
            ),
            "mapping_dir": repo_relative_path(
                runtime_paths.mapping_dir,
                runtime_paths.repo_root,
            ),
            "book_snapshot_file_names": [path.name for path in book_snapshot_paths],
            "price_change_file_names": [path.name for path in price_change_paths],
            "mapping_file_names": [path.name for path in mapping_paths],
        },
    }


def resolve_orderbook_paths(orderbook_paths: list[Path], raw_dir: Path) -> list[Path]:
    if orderbook_paths:
        candidates = [path.expanduser().resolve() for path in orderbook_paths]
    else:
        candidates = sorted(raw_dir.glob("*.parquet"))

    parquet_files = sorted(
        path for path in candidates if path.is_file() and path.suffix.lower() == ".parquet"
    )
    if not parquet_files:
        msg = "No parquet files found to process."
        raise FileNotFoundError(msg)
    return parquet_files


def split_single_orderbook_file(
    orderbook_path: Path,
    runtime_paths: RuntimePaths,
    *,
    mapping_cache: dict[str, dict[str, Any] | None],
    metadata_session: requests.Session,
    max_file_size_bytes: int,
) -> dict[str, Any]:
    price_change, book_snapshot, crypto_markets_mapping = build_orderbook_views(
        orderbook_path,
        mapping_cache=mapping_cache,
        metadata_session=metadata_session,
    )
    output_paths = build_output_paths(
        orderbook_path,
        book_snapshot_dir=runtime_paths.book_snapshot_dir,
        price_change_dir=runtime_paths.price_change_dir,
        mapping_dir=runtime_paths.mapping_dir,
    )
    price_change_files = write_lazyframe_parquet(
        price_change,
        output_paths["price_change"],
        max_file_size_bytes=max_file_size_bytes,
    )
    book_snapshot_files = write_lazyframe_parquet(
        book_snapshot,
        output_paths["book_snapshot"],
        max_file_size_bytes=max_file_size_bytes,
    )
    mapping_files = write_lazyframe_parquet(
        crypto_markets_mapping,
        output_paths["mapping"],
        max_file_size_bytes=max_file_size_bytes,
    )
    mapping_frame = crypto_markets_mapping.collect()
    markets = mapping_frame_to_manifest_markets(mapping_frame)

    return build_index_entry(
        orderbook_path,
        {
            "price_change": price_change_files,
            "book_snapshot": book_snapshot_files,
            "mapping": mapping_files,
        },
        runtime_paths,
        markets,
    )


def split_orderbook_files(
    orderbook_paths: list[Path] | None = None,
    *,
    runtime_paths: RuntimePaths,
    force: bool = False,
    max_file_size_bytes: int = MAX_OUTPUT_FILE_SIZE_BYTES,
) -> dict[str, Any]:
    source_paths = resolve_orderbook_paths(orderbook_paths or [], runtime_paths.raw_dir)
    raw_entries = load_index_entries(runtime_paths.index_path)
    existing_entries = normalize_index_entries(raw_entries, runtime_paths)
    manifest_needs_write = existing_entries != raw_entries
    seen_fingerprints = indexed_fingerprints(
        existing_entries,
        runtime_paths,
        max_file_size_bytes=max_file_size_bytes,
    )
    mapping_cache: dict[str, dict[str, Any] | None] = {}

    processed_files: list[str] = []
    skipped_files: list[str] = []

    with requests.Session() as metadata_session:
        for orderbook_path in tqdm(
            source_paths,
            desc="Processing orderbook files",
            unit="file",
            dynamic_ncols=True,
        ):
            fingerprint = source_fingerprint(orderbook_path)
            if not force and fingerprint in seen_fingerprints:
                skipped_files.append(str(orderbook_path))
                continue

            entry = split_single_orderbook_file(
                orderbook_path,
                runtime_paths,
                mapping_cache=mapping_cache,
                metadata_session=metadata_session,
                max_file_size_bytes=max_file_size_bytes,
            )
            existing_entries = replace_entry_for_source(
                existing_entries,
                orderbook_path,
                runtime_paths.repo_root,
            )
            existing_entries.append(entry)
            write_json_atomic(runtime_paths.index_path, existing_entries)
            manifest_needs_write = False
            processed_files.append(str(orderbook_path))
            seen_fingerprints.add(fingerprint)

    if manifest_needs_write:
        write_json_atomic(runtime_paths.index_path, existing_entries)

    return {
        "processed_count": len(processed_files),
        "skipped_count": len(skipped_files),
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "index_path": str(runtime_paths.index_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_paths = resolve_runtime_paths(
        script_path=__file__,
        raw_dir=args.raw_dir,
        book_snapshot_dir=args.book_snapshot_dir,
        price_change_dir=args.price_change_dir,
        mapping_dir=args.mapping_dir,
        index_path=args.index_json,
    )
    summary = split_orderbook_files(
        orderbook_paths=args.orderbook_paths,
        runtime_paths=runtime_paths,
        force=args.force,
    )
    print(f"Processed files: {summary['processed_count']}")
    print(f"Skipped files: {summary['skipped_count']}")
    print(f"JSON index: {summary['index_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
