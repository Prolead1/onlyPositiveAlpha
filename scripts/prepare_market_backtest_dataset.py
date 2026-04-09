"""Prepare a market-isolated backtest dataset from PMXT parquet shards.

This script filters a large PMXT source directory down to required markets and writes
partitioned output shards that are easy to scan repeatedly for backtests.
"""

from __future__ import annotations

import argparse
import heapq
import importlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock, current_thread, local
from time import monotonic
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    duckdb = importlib.import_module("duckdb")
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    duckdb = None

from backtester.feature_generator import FeatureGenerator
from backtester.normalize.schema import normalize_market_events_schema
from backtester.resolution.parsing import (
    parse_clob_token_ids,
    parse_outcome_prices,
    select_winning_asset_id,
)

DEFAULT_SOURCE_DIR = Path("data") / "cached" / "pmxt"
DEFAULT_MAPPING_DIR = Path("data") / "cached" / "mapping"
DEFAULT_OUTPUT_DIR = Path("data") / "cached" / "pmxt_backtest"
DEFAULT_FEATURES_SUBDIR = "features"
DEFAULT_RESOLUTION_SUBDIR = "resolution"
DEFAULT_RESOLUTION_FILENAME = "resolution_frame.parquet"
# Conservative auto default to avoid overwhelming local machines with many
# simultaneous parquet scans; users can still increase via --max-workers.
DEFAULT_MAX_WORKERS = min(8, max((os.cpu_count() or 1) // 2, 1))

_TIMESTAMP_CANDIDATES = (
    "ts_event",
    "timestamp_received",
    "timestamp_created_at",
    "timestamp",
)

_SLUG_DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[mhd])$", re.IGNORECASE)
_FILENAME_HOUR_FMT = "%Y-%m-%dT%H"
_SLUG_MIN_PARTS = 2
_FEATURE_CACHE_MIN_PATH_PARTS = 3
_SANITIZE_SEGMENT_RE = re.compile(r"[^a-z0-9]+")
_MANIFEST_SCHEMA_VERSION = 2
_SUPPORTED_SCAN_ENGINES = {"auto", "pyarrow", "duckdb"}
_DUCKDB_AUTO_MARKET_THRESHOLD = 64
_ACTIVE_WORKERS_LOCK = Lock()
_ACTIVE_WORKERS: dict[str, str] = {}
_THREAD_LOCAL_STATE = local()
_STOP_REQUESTED = Event()


def _is_stop_requested() -> bool:
    return _STOP_REQUESTED.is_set()


def _request_stop() -> None:
    _STOP_REQUESTED.set()


def _reset_stop_request() -> None:
    _STOP_REQUESTED.clear()


@dataclass(frozen=True)
class PrepareOptions:
    """Options controlling how prepared output is written."""

    overwrite: bool
    dry_run: bool
    max_workers: int
    compression: str
    compression_level: int
    log_worker_progress: bool = False
    feature_workers: int = 1
    feature_queue_size: int = 0
    scan_engine: str = "duckdb"
    scan_batch_size: int = 250_000


@dataclass(frozen=True)
class FileSummary:
    """Per-source-file processing summary."""

    source_file: str
    source_mtime_ns: int
    source_rows: int
    kept_rows: int
    feature_rows: int
    output_files: list[str]
    feature_output_files: list[str]
    written_files: int
    feature_written_files: int
    skipped_existing: int
    feature_skipped_existing: int
    reused_from_manifest: bool


@dataclass(frozen=True)
class MappingSelection:
    """Deterministic mapping-derived market and hour selection details."""

    selected_market_ids: set[str]
    ordered_selected_market_ids: list[str]
    hour_expectations: dict[datetime, set[str]]
    unresolved_market_ids: set[str]
    normalized_prefixes: list[str]
    market_limit: int | None


def _get_feature_generator() -> FeatureGenerator:
    feature_generator = getattr(_THREAD_LOCAL_STATE, "feature_generator", None)
    if feature_generator is None:
        feature_generator = FeatureGenerator()
        _THREAD_LOCAL_STATE.feature_generator = feature_generator
    return feature_generator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare market-isolated PMXT parquet dataset for scalable backtests.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing PMXT parquet files.",
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=DEFAULT_MAPPING_DIR,
        help="Directory containing daily gamma mapping JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where market-isolated output files will be written.",
    )
    parser.add_argument(
        "--market-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited file of market IDs to include.",
    )
    parser.add_argument(
        "--market-slug-prefix",
        action="append",
        default=[],
        help="Mapping slug prefix to include (can be repeated).",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=0,
        help=(
            "Optional cap for mapping-derived condition IDs. "
            "Uses first N IDs in deterministic order: mapping shard date, slug, condition ID. "
            "Use 0 or a negative value for no cap (default)."
        ),
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD) for source files.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD) for source files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest output path. Defaults to <output-dir>/manifest.json.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help=(
            "Maximum parallel workers for file processing. "
            "Use 0 or a negative value for auto (default)."
        ),
    )
    parser.add_argument(
        "--scan-engine",
        type=str,
        choices=sorted(_SUPPORTED_SCAN_ENGINES),
        default="duckdb",
        help=(
            "Engine for Stage-A parquet filtering. "
            "Choices: auto, pyarrow, duckdb."
        ),
    )
    parser.add_argument(
        "--feature-workers",
        type=int,
        default=0,
        help=(
            "Maximum per-file workers for feature computation across market/date groups. "
            "Use 0 or a negative value for auto (default)."
        ),
    )
    parser.add_argument(
        "--feature-queue-size",
        type=int,
        default=0,
        help=(
            "Maximum in-flight per-file feature tasks buffered between grouping and feature "
            "computation stages. Use 0 or a negative value for auto (default)."
        ),
    )
    parser.add_argument(
        "--scan-batch-size",
        type=int,
        default=0,
        help=(
            "Record batch size used for Stage-A scan streaming. "
            "Use 0 or a negative value for auto (default)."
        ),
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        help="Parquet compression codec for output files.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=6,
        help="Parquet compression level for output files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute manifest and row counts without writing output files.",
    )
    parser.add_argument(
        "--log-worker-progress",
        action="store_true",
        help="Print per-worker file start/finish progress.",
    )
    return parser.parse_args(argv)


def parse_iso_date(raw: str | None) -> datetime | None:
    """Parse YYYY-MM-DD as UTC datetime at midnight."""
    if raw is None:
        return None

    parsed = datetime.strptime(raw, "%Y-%m-%d")  # noqa: DTZ007
    return parsed.replace(tzinfo=UTC)


def extract_hour_from_filename(filename: str) -> datetime | None:
    """Extract UTC hour from polymarket_orderbook_YYYY-MM-DDTHH.parquet."""
    try:
        value = filename.removeprefix("polymarket_orderbook_").removesuffix(".parquet")
        # Accept suffix after <date>T<hour>, e.g., polymarket_orderbook_2026-04-03T12_extra.parquet
        # Only parse the first 13 characters (YYYY-MM-DDTHH)
        hour_str = value[:13]
        return datetime.strptime(hour_str, "%Y-%m-%dT%H").replace(tzinfo=UTC)
    except ValueError:
        return None


def resolve_timestamp_column(columns: pd.Index) -> str | None:
    """Resolve event timestamp column from supported candidates."""
    for name in _TIMESTAMP_CANDIDATES:
        if name in columns:
            return name
    return None


def _to_list_payload(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            converted = to_list()
        except Exception:  # pragma: no cover - defensive conversion
            return []
        if isinstance(converted, list):
            return converted
    return []


def _normalize_chunk_for_features(chunk: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Normalize source chunks from either PMXT event feed or book snapshots."""
    if chunk.empty:
        return chunk

    columns = set(chunk.columns)

    # Standard PMXT event feed path.
    if "event_type" in columns or "update_type" in columns:
        return normalize_market_events_schema(chunk, is_pmxt_mode=True)

    # book_snapshot path: convert columns into canonical event schema.
    if {"market_id", "bids", "asks"}.issubset(columns):
        normalized = chunk.copy()
        ts_col = resolve_timestamp_column(normalized.columns)
        if ts_col is None:
            normalized["ts_event"] = pd.NaT
        elif ts_col != "ts_event":
            normalized = normalized.rename(columns={ts_col: "ts_event"})

        if "token_id" not in normalized.columns:
            normalized["token_id"] = ""

        def _build_snapshot_payload(row: pd.Series) -> dict[str, object]:
            token_id = str(row.get("token_id", "") or "")
            payload: dict[str, object] = {
                "asset_id": token_id,
                "bids": _to_list_payload(row.get("bids")),
                "asks": _to_list_payload(row.get("asks")),
            }
            for key in ("best_bid", "best_ask", "side"):
                if key in row.index:
                    value = row.get(key)
                    if value is not None:
                        payload[key] = value
            return payload

        normalized["event_type"] = "book"
        normalized["data"] = normalized.apply(_build_snapshot_payload, axis=1)

        canonical = normalized[["ts_event", "event_type", "market_id", "token_id", "data"]]
        return normalize_market_events_schema(canonical, is_pmxt_mode=True)

    return normalize_market_events_schema(chunk, is_pmxt_mode=True)


def collect_market_ids_from_file(path: Path | None) -> set[str]:
    """Load explicit market IDs from newline-delimited file."""
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)

    market_ids: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        value = raw_line.strip()
        if value and not value.startswith("#"):
            market_ids.add(value)
    return market_ids


def _iter_mapping_files(mapping_dir: Path) -> list[Path]:
    return sorted(mapping_dir.glob("gamma_updown_markets_*.json"))


def _normalize_slug_prefixes(prefixes: list[str]) -> list[str]:
    return [prefix.strip().lower() for prefix in prefixes if prefix.strip()]


def _sanitize_path_segment(value: str) -> str:
    cleaned = _SANITIZE_SEGMENT_RE.sub("-", value.strip().lower()).strip("-")
    return cleaned or "prefix"


def resolve_run_output_dir(
    *,
    output_dir: Path,
    slug_prefixes: list[str],
) -> tuple[Path, str | None, list[str]]:
    """Resolve run root and validated prefix metadata.

    Prefix-isolated runs use output_dir/runs/<prefix>. Exactly one prefix is
    allowed per run for deterministic, independent feature sets.
    """
    normalized_prefixes = _normalize_slug_prefixes(slug_prefixes)
    if len(normalized_prefixes) > 1:
        raise ValueError(
            "Use one --market-slug-prefix per run to create isolated feature sets."
        )
    if not normalized_prefixes:
        return output_dir, None, []
    normalized_prefix = normalized_prefixes[0]
    return output_dir / "runs" / _sanitize_path_segment(normalized_prefix), normalized_prefix, [
        normalized_prefix,
    ]


def _parse_slug_epoch(slug: str) -> datetime | None:
    # Slugs are expected to end with an epoch-seconds suffix, e.g. ...-1704067200
    raw_tail = slug.rsplit("-", maxsplit=1)[-1].strip()
    if not raw_tail.isdigit():
        return None
    try:
        return datetime.fromtimestamp(int(raw_tail), tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _parse_slug_duration(slug: str) -> timedelta | None:
    # Prefer the closest duration token before the epoch, e.g. 15m or 1h.
    parts = slug.split("-")
    if len(parts) < _SLUG_MIN_PARTS:
        return None
    for part in reversed(parts[:-1]):
        match = _SLUG_DURATION_RE.match(part)
        if match is None:
            continue
        value = int(match.group("value"))
        unit = match.group("unit").lower()
        if value <= 0:
            return None
        if unit == "m":
            return timedelta(minutes=value)
        if unit == "h":
            return timedelta(hours=value)
        if unit == "d":
            return timedelta(days=value)
    return None


def _floor_to_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def _iter_hour_buckets(start: datetime, end: datetime) -> list[datetime]:
    if end < start:
        return []
    buckets: list[datetime] = []
    cursor = _floor_to_hour(start)
    boundary = _floor_to_hour(end)
    while cursor <= boundary:
        buckets.append(cursor)
        cursor = cursor + timedelta(hours=1)
    return buckets


def _load_mapping_payload(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _iter_matching_slug_condition_ids(
    *,
    mapping_dir: Path,
    normalized_prefixes: list[str],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for shard in _iter_mapping_files(mapping_dir):
        payload = _load_mapping_payload(shard)
        if payload is None:
            continue
        for slug, entry in payload.items():
            if not isinstance(slug, str) or not isinstance(entry, dict):
                continue
            slug_lc = slug.lower()
            if not any(slug_lc.startswith(prefix) for prefix in normalized_prefixes):
                continue
            condition_id = entry.get("conditionId")
            if isinstance(condition_id, str):
                pairs.append((slug_lc, condition_id))
    return pairs


def _iter_matching_mapping_records(
    *,
    mapping_dir: Path,
    normalized_prefixes: list[str],
) -> list[tuple[str, str, str]]:
    """Return deterministic mapping records as (shard_name, slug_lc, condition_id)."""
    records: list[tuple[str, str, str]] = []
    for shard in _iter_mapping_files(mapping_dir):
        payload = _load_mapping_payload(shard)
        if payload is None:
            continue

        for slug in sorted(payload):
            entry = payload.get(slug)
            if not isinstance(slug, str) or not isinstance(entry, dict):
                continue
            slug_lc = slug.lower()
            if not any(slug_lc.startswith(prefix) for prefix in normalized_prefixes):
                continue
            condition_id = entry.get("conditionId")
            if isinstance(condition_id, str) and condition_id:
                records.append((shard.name, slug_lc, condition_id))
    return records


def _hour_buckets_for_slug(slug_lc: str) -> list[datetime]:
    anchor_ts = _parse_slug_epoch(slug_lc)
    if anchor_ts is None:
        return []
    duration = _parse_slug_duration(slug_lc) or timedelta(hours=1)
    # Add a one-hour margin on both sides to avoid dropping boundary events.
    window_start = anchor_ts - duration - timedelta(hours=1)
    window_end = anchor_ts + timedelta(hours=1)
    return _iter_hour_buckets(window_start, window_end)


def build_market_hour_expectations(
    mapping_dir: Path,
    prefixes: list[str],
    *,
    target_market_ids: set[str],
) -> tuple[dict[datetime, set[str]], set[str]]:
    """Map UTC hour buckets to market IDs expected from mapping slug timestamps.

    Returns a tuple of:
    - hour_to_market_ids index for slug entries with parseable epochs
    - unresolved market IDs that should still be scanned in all files
    """
    if not prefixes or not target_market_ids:
        return {}, set()
    if not mapping_dir.exists():
        return {}, set(target_market_ids)

    normalized_prefixes = _normalize_slug_prefixes(prefixes)
    if not normalized_prefixes:
        return {}, set(target_market_ids)

    hour_index: dict[datetime, set[str]] = {}
    unresolved_market_ids: set[str] = set()
    scheduled_market_ids: set[str] = set()
    slug_condition_pairs = _iter_matching_slug_condition_ids(
        mapping_dir=mapping_dir,
        normalized_prefixes=normalized_prefixes,
    )
    for slug_lc, condition_id in slug_condition_pairs:
        if condition_id not in target_market_ids:
            continue

        buckets = _hour_buckets_for_slug(slug_lc)
        if not buckets:
            unresolved_market_ids.add(condition_id)
            continue

        for bucket in buckets:
            hour_index.setdefault(bucket, set()).add(condition_id)
            scheduled_market_ids.add(condition_id)

    unresolved_market_ids.update(target_market_ids - scheduled_market_ids)
    return hour_index, unresolved_market_ids


def build_mapping_selection(
    *,
    mapping_dir: Path,
    prefixes: list[str],
    max_markets: int,
) -> MappingSelection:
    """Build deterministic mapping selection with hour expectations.

    Selection order is deterministic by shard filename (date), then slug, then
    condition ID through first-seen unique condition IDs.
    """
    normalized_prefixes = _normalize_slug_prefixes(prefixes)
    if not normalized_prefixes:
        return MappingSelection(
            selected_market_ids=set(),
            ordered_selected_market_ids=[],
            hour_expectations={},
            unresolved_market_ids=set(),
            normalized_prefixes=[],
            market_limit=None,
        )
    if not mapping_dir.exists():
        raise FileNotFoundError(mapping_dir)

    records = _iter_matching_mapping_records(
        mapping_dir=mapping_dir,
        normalized_prefixes=normalized_prefixes,
    )
    ordered_unique_market_ids: list[str] = []
    seen_market_ids: set[str] = set()
    for _, _, condition_id in records:
        if condition_id in seen_market_ids:
            continue
        seen_market_ids.add(condition_id)
        ordered_unique_market_ids.append(condition_id)

    market_limit = max_markets if max_markets > 0 else None
    if market_limit is not None:
        ordered_unique_market_ids = ordered_unique_market_ids[:market_limit]

    selected_market_ids = set(ordered_unique_market_ids)
    hour_index: dict[datetime, set[str]] = {}
    unresolved_market_ids: set[str] = set()
    scheduled_market_ids: set[str] = set()
    for _, slug_lc, condition_id in records:
        if condition_id not in selected_market_ids:
            continue
        buckets = _hour_buckets_for_slug(slug_lc)
        if not buckets:
            unresolved_market_ids.add(condition_id)
            continue
        for bucket in buckets:
            hour_index.setdefault(bucket, set()).add(condition_id)
            scheduled_market_ids.add(condition_id)

    unresolved_market_ids.update(selected_market_ids - scheduled_market_ids)
    return MappingSelection(
        selected_market_ids=selected_market_ids,
        ordered_selected_market_ids=ordered_unique_market_ids,
        hour_expectations=hour_index,
        unresolved_market_ids=unresolved_market_ids,
        normalized_prefixes=normalized_prefixes,
        market_limit=market_limit,
    )


def collect_market_ids_from_mapping(
    mapping_dir: Path,
    prefixes: list[str],
    *,
    max_markets: int = 0,
) -> set[str]:
    """Collect condition IDs from mapping shards that match slug prefixes."""
    selection = build_mapping_selection(
        mapping_dir=mapping_dir,
        prefixes=prefixes,
        max_markets=max_markets,
    )
    return selection.selected_market_ids


def build_per_file_market_targets(
    *,
    source_files: list[Path],
    explicit_ids: set[str],
    mapping_selection: MappingSelection,
) -> tuple[list[Path], dict[str, set[str]], int]:
    """Build scheduled files and per-file target markets using hour indexes."""
    if not source_files:
        return [], {}, 0

    if not mapping_selection.selected_market_ids:
        return (
            list(source_files),
            {
                str(source_file): set(explicit_ids)
                for source_file in source_files
                if explicit_ids
            },
            0,
        )

    hour_to_files: dict[datetime, list[Path]] = {}
    unknown_hour_files: list[Path] = []
    file_hour_cache: dict[str, datetime | None] = {}
    for source_file in source_files:
        hour = extract_hour_from_filename(source_file.name)
        file_hour_cache[str(source_file)] = hour
        if hour is None:
            unknown_hour_files.append(source_file)
            continue
        hour_to_files.setdefault(hour, []).append(source_file)

    scheduled_files: list[Path] = []
    per_file_market_ids: dict[str, set[str]] = {}
    unresolved = set(mapping_selection.unresolved_market_ids)

    for source_file in source_files:
        source_key = str(source_file)
        hour = file_hour_cache.get(source_key)
        target_ids = set(explicit_ids)
        target_ids.update(unresolved)
        if hour is not None:
            target_ids.update(mapping_selection.hour_expectations.get(hour, set()))

        if not target_ids:
            continue
        per_file_market_ids[source_key] = target_ids
        scheduled_files.append(source_file)

    pruned_file_count = len(source_files) - len(scheduled_files)
    return scheduled_files, per_file_market_ids, pruned_file_count


def collect_source_files(
    source_dir: Path,
    *,
    start_date: datetime | None,
    end_date: datetime | None,
) -> list[Path]:
    """Collect source PMXT files constrained by optional date boundaries."""
    all_files = sorted(source_dir.rglob("polymarket_orderbook_*.parquet"))

    if start_date is None and end_date is None:
        return all_files

    selected: list[Path] = []
    end_exclusive = end_date + timedelta(days=1) if end_date is not None else None
    for file in all_files:
        hour = extract_hour_from_filename(file.name)
        if hour is None:
            continue
        if start_date is not None and hour < start_date:
            continue
        if end_exclusive is not None and hour >= end_exclusive:
            continue
        selected.append(file)

    return selected


def build_output_path(
    *,
    output_dir: Path,
    event_date: str,
    market_id: str,
    source_file: Path,
) -> Path:
    """Build deterministic output path for one market/date/source shard."""
    return (
        output_dir
        / event_date
        / market_id
        / f"{source_file.stem}.parquet"
    )


def build_feature_output_path(
    *,
    output_dir: Path,
    event_date: str,
    market_id: str,
    source_file: Path,
) -> Path:
    """Build deterministic output path for one feature shard."""
    return (
        output_dir
        / DEFAULT_FEATURES_SUBDIR
        / event_date
        / market_id
        / f"{source_file.stem}.parquet"
    )


def compute_prepared_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute orderbook features for one normalized market/date frame."""
    if frame.empty:
        return pd.DataFrame()
    if "ts_event" not in frame.columns:
        return pd.DataFrame()

    feature_input = frame.copy()
    feature_input["ts_event"] = pd.to_datetime(
        feature_input["ts_event"],
        utc=True,
        errors="coerce",
    )
    feature_input = feature_input.dropna(subset=["ts_event"]).set_index("ts_event")
    feature_input = feature_input.sort_index()
    if feature_input.empty:
        return pd.DataFrame()

    feature_generator = _get_feature_generator()
    features = feature_generator.generate_orderbook_features(feature_input)
    if features.empty:
        return features

    return features.reset_index()


def build_resolution_output_path(*, output_dir: Path) -> Path:
    return output_dir / DEFAULT_RESOLUTION_SUBDIR / DEFAULT_RESOLUTION_FILENAME


def _build_mapping_condition_index(mapping_dir: Path) -> dict[str, list[dict[str, object]]]:
    entries: dict[str, list[dict[str, object]]] = {}
    for shard in _iter_mapping_files(mapping_dir):
        try:
            payload = json.loads(shard.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            condition_id = entry.get("conditionId")
            if not isinstance(condition_id, str) or not condition_id:
                continue
            entries.setdefault(condition_id, []).append(entry)
    return entries


def collect_date_market_pairs_from_features(features_root: Path) -> set[tuple[str, str]]:
    """Collect (date, market_id) pairs from prepared feature directory layout.
    
    Returns set of (date_str, market_id) tuples representing all feature granularity.
    This ensures resolution has one row per unique (date, market_id) combination.
    Expected layout: features/YYYY-MM-DD/<market_id>/*.parquet
    """
    if not features_root.exists():
        return set()

    pairs: set[tuple[str, str]] = set()
    for day_dir in sorted(features_root.iterdir()):
        if not day_dir.is_dir():
            continue
        date_str = day_dir.name
        for market_dir in sorted(day_dir.iterdir()):
            if market_dir.is_dir():
                market_id = market_dir.name
                pairs.add((date_str, market_id))

    return pairs


def build_resolution_frame_from_mapping(
    *,
    mapping_dir: Path,
    required_market_ids: set[str],
    required_date_market_pairs: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build mapping-based resolution frame for required market IDs.
    
    If required_date_market_pairs is provided, creates one resolution row per
    (date, market_id) pair (accounting for market appearing on different dates).
    If only required_market_ids is provided, creates one row per market.
    """
    columns = [
        "market_id",
        "resolved_at",
        "winning_asset_id",
        "winning_outcome",
        "fees_enabled_market",
        "settlement_source",
        "settlement_confidence",
        "settlement_evidence_ts",
        "feature_date",
    ]

    condition_index = _build_mapping_condition_index(mapping_dir)
    candidate_rows_by_market: dict[str, list[dict[str, object]]] = {}

    for market_id in sorted(required_market_ids):
        market_entries = condition_index.get(market_id, [])
        if not market_entries:
            continue

        for entry in market_entries:
            resolved_raw = entry.get("resolvedAt") or entry.get("endDate")
            resolved_at = pd.to_datetime(str(resolved_raw), utc=True, errors="coerce")
            if pd.isna(resolved_at):
                continue

            winning_asset_id = entry.get("winningAssetId")
            if not isinstance(winning_asset_id, str) or not winning_asset_id.strip():
                token_ids = parse_clob_token_ids(entry.get("clobTokenIds"))
                prices = parse_outcome_prices(entry.get("outcomePrices"))
                winning_asset_id = select_winning_asset_id(token_ids, prices)

            candidate_rows_by_market.setdefault(str(market_id), []).append(
                {
                    "market_id": str(market_id),
                    "resolved_at": resolved_at,
                    "winning_asset_id": (
                        str(winning_asset_id)
                        if winning_asset_id is not None and str(winning_asset_id).strip()
                        else None
                    ),
                    "winning_outcome": entry.get("winningOutcome"),
                    "fees_enabled_market": bool(
                        entry.get("feesEnabledMarket", entry.get("feesEnabled", True))
                    ),
                    "settlement_source": "mapping",
                    "settlement_confidence": 1.0,
                    "settlement_evidence_ts": resolved_at,
                }
            )

    if required_date_market_pairs is not None:
        replicated_rows: list[dict[str, object]] = []
        for date_str, market_id in sorted(required_date_market_pairs):
            candidates = candidate_rows_by_market.get(market_id, [])
            if not candidates:
                continue

            same_day_candidates = [
                candidate
                for candidate in candidates
                if pd.to_datetime(candidate.get("resolved_at"), utc=True, errors="coerce")
                .strftime("%Y-%m-%d")
                == date_str
            ]
            if not same_day_candidates:
                continue

            selected = max(
                same_day_candidates,
                key=lambda candidate: pd.to_datetime(
                    candidate.get("resolved_at"),
                    utc=True,
                    errors="coerce",
                ),
            )
            row_with_date = dict(selected)
            row_with_date["feature_date"] = date_str
            replicated_rows.append(row_with_date)

        return pd.DataFrame(replicated_rows, columns=columns)

    selected_rows: list[dict[str, object]] = []
    for market_id, candidates in candidate_rows_by_market.items():
        if not candidates:
            continue

        selected = max(
            candidates,
            key=lambda candidate: pd.to_datetime(
                candidate.get("resolved_at"),
                utc=True,
                errors="coerce",
            ),
        )
        row_with_date = dict(selected)
        resolved_at = pd.to_datetime(row_with_date.get("resolved_at"), utc=True, errors="coerce")
        row_with_date["feature_date"] = (
            resolved_at.strftime("%Y-%m-%d") if not pd.isna(resolved_at) else None
        )
        row_with_date["market_id"] = market_id
        selected_rows.append(row_with_date)

    return pd.DataFrame(selected_rows, columns=columns)


def _event_date_fallback(source_file: Path) -> str:
    hour = extract_hour_from_filename(source_file.name)
    if hour is not None:
        return hour.strftime("%Y-%m-%d")
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


def _load_manifest_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _resolve_reusable_summaries(  # noqa: C901, PLR0911, PLR0912, PLR0913
    *,
    manifest_path: Path,
    source_dir: Path,
    output_dir: Path,
    required_market_ids: set[str],
    run_slug_prefix: str | None,
    normalized_prefixes: list[str],
    market_limit: int | None,
    overwrite: bool,
) -> dict[str, FileSummary]:
    if overwrite:
        return {}

    payload = _load_manifest_payload(manifest_path)
    if payload is None:
        return {}

    if payload.get("manifest_schema_version") != _MANIFEST_SCHEMA_VERSION:
        return {}

    if payload.get("source_dir") != str(source_dir.resolve()):
        return {}
    if payload.get("output_dir") != str(output_dir.resolve()):
        return {}
    if payload.get("run_slug_prefix") != run_slug_prefix:
        return {}
    if payload.get("market_slug_prefixes") != normalized_prefixes:
        return {}
    if payload.get("selected_market_limit") != market_limit:
        return {}

    raw_required_ids = payload.get("required_market_ids")
    if not isinstance(raw_required_ids, list):
        return {}
    manifest_required_ids = {
        str(item)
        for item in raw_required_ids
        if isinstance(item, str) and item
    }
    if manifest_required_ids != required_market_ids:
        return {}

    raw_entries = payload.get("files")
    if not isinstance(raw_entries, list):
        return {}

    reusable: dict[str, FileSummary] = {}
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            continue

        source_file_raw = raw.get("source_file")
        if not isinstance(source_file_raw, str) or not source_file_raw:
            continue
        source_file = Path(source_file_raw)
        if not source_file.exists():
            continue

        feature_output_files_raw = raw.get("feature_output_files")
        if not isinstance(feature_output_files_raw, list):
            continue

        feature_output_files: list[str] = []
        for item in feature_output_files_raw:
            if not isinstance(item, str) or not item:
                continue
            candidate = Path(item)
            if candidate.exists():
                feature_output_files.append(str(candidate))

        if not feature_output_files:
            continue

        source_mtime_ns = source_file.stat().st_mtime_ns
        manifest_mtime_ns = raw.get("source_mtime_ns")
        if isinstance(manifest_mtime_ns, int) and manifest_mtime_ns != source_mtime_ns:
            continue

        source_rows = int(raw.get("source_rows", 0))
        kept_rows = int(raw.get("kept_rows", 0))
        feature_rows = int(raw.get("feature_rows", 0))
        reusable[str(source_file)] = FileSummary(
            source_file=str(source_file),
            source_mtime_ns=source_mtime_ns,
            source_rows=source_rows,
            kept_rows=kept_rows,
            feature_rows=feature_rows,
            output_files=[],
            feature_output_files=feature_output_files,
            written_files=0,
            feature_written_files=0,
            skipped_existing=0,
            feature_skipped_existing=len(feature_output_files),
            reused_from_manifest=True,
        )

    return reusable


def _build_manifest_payload(  # noqa: PLR0913
    *,
    source_dir: Path,
    mapping_dir: Path,
    output_dir: Path,
    required_market_ids: set[str],
    selected_mapping_market_ids: list[str],
    run_slug_prefix: str | None,
    normalized_prefixes: list[str],
    selected_market_limit: int | None,
    source_files: list[Path],
    source_file_count_total: int,
    source_file_count_pruned: int,
    options: PrepareOptions,
    summaries: list[FileSummary],
    resolution_output_file: Path,
    resolution_rows: int,
    resolution_written: bool,
    resolution_skipped_existing: bool,
    interrupted: bool,
) -> dict[str, Any]:
    total_rows = sum(item.source_rows for item in summaries)
    kept_rows = sum(item.kept_rows for item in summaries)
    feature_rows = sum(item.feature_rows for item in summaries)
    output_files_written = sum(item.written_files for item in summaries)
    feature_output_files_written = sum(item.feature_written_files for item in summaries)
    skipped_existing = sum(item.skipped_existing for item in summaries)
    feature_skipped_existing = sum(item.feature_skipped_existing for item in summaries)

    return {
        "manifest_schema_version": _MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_dir": str(source_dir.resolve()),
        "mapping_dir": str(mapping_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "run_slug_prefix": run_slug_prefix,
        "market_slug_prefixes": normalized_prefixes,
        "selected_market_limit": selected_market_limit,
        "dry_run": options.dry_run,
        "interrupted": interrupted,
        "required_market_count": len(required_market_ids),
        "required_market_ids": sorted(required_market_ids),
        "selected_mapping_market_count": len(selected_mapping_market_ids),
        "selected_mapping_market_ids": selected_mapping_market_ids,
        "source_file_count": len(source_files),
        "source_file_count_total": source_file_count_total,
        "source_file_count_scheduled": len(source_files),
        "source_file_count_pruned": source_file_count_pruned,
        "rows_total": total_rows,
        "rows_kept": kept_rows,
        "feature_rows": feature_rows,
        "resolution_rows": resolution_rows,
        "resolution_output_file": str(resolution_output_file),
        "resolution_written": resolution_written,
        "resolution_skipped_existing": resolution_skipped_existing,
        "output_files_written": output_files_written,
        "feature_output_files_written": feature_output_files_written,
        "output_files_skipped_existing": skipped_existing,
        "feature_output_files_skipped_existing": feature_skipped_existing,
        "files": [
            {
                "source_file": item.source_file,
                "source_mtime_ns": item.source_mtime_ns,
                "source_rows": item.source_rows,
                "kept_rows": item.kept_rows,
                "feature_rows": item.feature_rows,
                "output_files": item.output_files,
                "feature_output_files": item.feature_output_files,
                "written_files": item.written_files,
                "feature_written_files": item.feature_written_files,
                "skipped_existing": item.skipped_existing,
                "feature_skipped_existing": item.feature_skipped_existing,
                "reused_from_manifest": item.reused_from_manifest,
            }
            for item in summaries
        ],
    }


def resolve_worker_count(*, max_workers: int, file_count: int) -> int:
    """Resolve effective worker count from CLI and available files."""
    if file_count <= 0:
        return 1
    if max_workers <= 0:
        return min(file_count, DEFAULT_MAX_WORKERS)
    return min(max(max_workers, 1), file_count)


def resolve_scan_engine(*, scan_engine: str) -> str:
    """Resolve scan engine name to a supported value."""
    normalized = scan_engine.strip().lower()
    if normalized in _SUPPORTED_SCAN_ENGINES:
        return normalized
    return "auto"


def resolve_scan_batch_size(*, scan_batch_size: int) -> int:
    """Resolve streamed scan batch size."""
    if scan_batch_size <= 0:
        return 250_000
    return max(scan_batch_size, 10_000)


def _resolve_scan_projection_columns(*, source_file: Path) -> list[str]:
    """Resolve minimal required scan projection for normalization and features."""
    try:
        source_schema = pq.read_schema(source_file)
    except Exception:  # pragma: no cover - defensive schema read
        return ["market_id"]

    available = set(source_schema.names)
    preferred = [
        "market_id",
        *list(_TIMESTAMP_CANDIDATES),
        "event_type",
        "update_type",
        "token_id",
        "data",
        "side",
        "best_bid",
        "best_ask",
        "bids",
        "asks",
    ]
    selected = [name for name in preferred if name in available]
    if "market_id" not in selected:
        return []
    return selected


def _iter_arrow_batches(
    *,
    result: pa.Table | pa.RecordBatchReader,
    scan_batch_size: int,
) -> Iterator[pa.RecordBatch]:
    """Yield Arrow record batches from either a table or a record batch reader."""
    if isinstance(result, pa.Table):
        yield from result.to_batches(max_chunksize=scan_batch_size)
        return

    if isinstance(result, pa.RecordBatchReader):
        while True:
            try:
                batch = result.read_next_batch()
            except StopIteration:
                break
            if batch.num_rows:
                yield batch
        return


def _project_record_batch(
    *,
    batch: pa.RecordBatch,
    scan_columns: list[str],
) -> pa.RecordBatch:
    existing_columns = [name for name in scan_columns if name in batch.schema.names]
    if not existing_columns:
        return batch
    return batch.select(existing_columns)


def _project_batch_stream(
    *,
    batches: Iterator[pa.RecordBatch],
    scan_columns: list[str],
) -> Iterator[pa.RecordBatch]:
    for batch in batches:
        yield _project_record_batch(batch=batch, scan_columns=scan_columns)

def _scan_filtered_batches_with_pyarrow(
    *,
    source_file: Path,
    target_market_ids: set[str],
    scan_batch_size: int,
) -> Iterator[pa.RecordBatch]:
    scan_columns = _resolve_scan_projection_columns(source_file=source_file)
    if not scan_columns:
        return iter(())

    resolved_batch_size = resolve_scan_batch_size(scan_batch_size=scan_batch_size)
    scanner = ds.dataset(source_file, format="parquet")
    market_filter = ds.field("market_id").isin(sorted(target_market_ids))
    batch_scanner = scanner.scanner(
        filter=market_filter,
        columns=scan_columns,
        batch_size=resolved_batch_size,
        use_threads=True,
    )
    return batch_scanner.to_batches()


def _scan_filtered_batches_with_duckdb(
    *,
    source_file: Path,
    target_market_ids: set[str],
    scan_batch_size: int,
) -> tuple[Iterator[pa.RecordBatch], Callable[[], None]]:
    if duckdb is None:
        msg = (
            "DuckDB scan engine requested but duckdb is not installed. "
            "Install duckdb or use --scan-engine pyarrow."
        )
        raise RuntimeError(msg)

    scan_columns = _resolve_scan_projection_columns(source_file=source_file)
    if not scan_columns:
        return iter(()), lambda: None

    resolved_batch_size = resolve_scan_batch_size(scan_batch_size=scan_batch_size)
    target_ids = sorted(str(market_id) for market_id in target_market_ids)
    query = (
        "SELECT * FROM read_parquet(?) "
        "WHERE CAST(market_id AS VARCHAR) "
        "IN (SELECT UNNEST(?::VARCHAR[]))"
    )
    params: list[object] = [str(source_file), target_ids]

    conn = duckdb.connect(database=":memory:")
    result = conn.execute(query, params).arrow()
    return (
        _project_batch_stream(
            batches=_iter_arrow_batches(
                result=result,
                scan_batch_size=resolved_batch_size,
            ),
            scan_columns=scan_columns,
        ),
        conn.close,
    )


def _scan_filtered_batches(
    *,
    source_file: Path,
    target_market_ids: set[str],
    scan_engine: str,
    scan_batch_size: int,
) -> tuple[Iterator[pa.RecordBatch], str, Callable[[], None]]:
    """Scan one source file for target markets via configured scan engine."""
    engine = resolve_scan_engine(scan_engine=scan_engine)
    if engine == "pyarrow":
        return (
            _scan_filtered_batches_with_pyarrow(

                source_file=source_file,
                target_market_ids=target_market_ids,
                scan_batch_size=scan_batch_size,
            ),
            "pyarrow",
            lambda: None,
        )
    if engine == "duckdb":
        batches, close_conn = _scan_filtered_batches_with_duckdb(
            source_file=source_file,
            target_market_ids=target_market_ids,
            scan_batch_size=scan_batch_size,
        )
        return (batches, "duckdb", close_conn)

    # Auto mode: prefer DuckDB for larger target sets when available.
    prefer_duckdb = (
        duckdb is not None
        and len(target_market_ids) >= _DUCKDB_AUTO_MARKET_THRESHOLD
    )
    if prefer_duckdb:
        try:
            batches, close_conn = _scan_filtered_batches_with_duckdb(
                source_file=source_file,
                target_market_ids=target_market_ids,
                scan_batch_size=scan_batch_size,
            )
        except Exception:
            return (
                _scan_filtered_batches_with_pyarrow(
                    source_file=source_file,
                    target_market_ids=target_market_ids,
                    scan_batch_size=scan_batch_size,
                ),
                "pyarrow",
                lambda: None,
            )
        else:
            return (batches, "duckdb", close_conn)

    return (
        _scan_filtered_batches_with_pyarrow(
            source_file=source_file,
            target_market_ids=target_market_ids,
            scan_batch_size=scan_batch_size,
        ),
        "pyarrow",
        lambda: None,
    )

def resolve_feature_worker_count(*, feature_workers: int, group_count: int) -> int:
    """Resolve per-file feature worker count for grouped market shards."""
    if group_count <= 0:
        return 1
    if feature_workers <= 0:
        # Conservative auto mode: avoid nested threadpool oversubscription when
        # file-level workers are already running in parallel.
        return 1
    return min(max(feature_workers, 1), group_count)


def resolve_feature_queue_size(*, feature_queue_size: int, feature_workers: int) -> int:
    """Resolve bounded per-file queue size for feature stage handoff."""
    if feature_workers <= 1:
        return 1
    if feature_queue_size <= 0:
        return max(feature_workers * 2, feature_workers)
    return max(feature_queue_size, feature_workers)


def _compute_feature_frames(  # noqa: C901, PLR0912
    *,
    grouped_frames: list[tuple[int, str, str, pd.DataFrame]],
    feature_workers: int,
    feature_queue_size: int,
) -> list[tuple[int, str, str, pd.DataFrame]]:
    """Compute feature frames for grouped market/date shards.

    Uses a bounded in-flight queue when running in parallel to keep memory stable
    for large source files while preserving deterministic output ordering.
    """
    if not grouped_frames:
        return []
    if _is_stop_requested():
        return []

    worker_count = resolve_feature_worker_count(
        feature_workers=feature_workers,
        group_count=len(grouped_frames),
    )
    if worker_count <= 1:
        return [
            (idx, event_date, market_id, compute_prepared_features(frame))
            for idx, event_date, market_id, frame in grouped_frames
        ]

    inflight_limit = resolve_feature_queue_size(
        feature_queue_size=feature_queue_size,
        feature_workers=worker_count,
    )
    results: list[tuple[int, str, str, pd.DataFrame]] = []
    ready_results: dict[int, tuple[str, str, pd.DataFrame]] = {}
    next_emit_index = 0
    pending_futures: dict[Future[pd.DataFrame], tuple[int, str, str]] = {}
    grouped_iter = iter(grouped_frames)
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="pmxt-feature",
    )
    try:
        def submit_next_group() -> bool:
            if _is_stop_requested():
                return False
            try:
                idx, event_date, market_id, frame = next(grouped_iter)
            except StopIteration:
                return False
            future = executor.submit(compute_prepared_features, frame)
            pending_futures[future] = (idx, event_date, market_id)
            return True

        while len(pending_futures) < inflight_limit and submit_next_group():
            pass

        while pending_futures:
            if _is_stop_requested():
                break

            done_futures, _ = wait(
                set(pending_futures),
                timeout=0.2,
                return_when=FIRST_COMPLETED,
            )
            if not done_futures:
                continue

            for future in done_futures:
                idx, event_date, market_id = pending_futures.pop(future)
                ready_results[idx] = (event_date, market_id, future.result())

                while next_emit_index in ready_results:
                    ready_event_date, ready_market_id, ready_frame = ready_results.pop(
                        next_emit_index
                    )
                    results.append(
                        (
                            next_emit_index,
                            ready_event_date,
                            ready_market_id,
                            ready_frame,
                        )
                    )
                    next_emit_index += 1

            while len(pending_futures) < inflight_limit and submit_next_group():
                pass
    finally:
        if _is_stop_requested():
            for future in pending_futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True, cancel_futures=False)

    return results


def _estimate_source_processing_cost(*, source_file: Path, target_market_ids: set[str]) -> int:
    """Estimate relative source processing cost for dynamic scheduling."""
    target_count = max(len(target_market_ids), 1)
    try:
        source_bytes = source_file.stat().st_size
    except OSError:
        source_bytes = 0
    return max(source_bytes, 1) * target_count


def _build_pending_source_priority_queue(
    *,
    pending_source_files: list[Path],
    per_file_market_ids: dict[str, set[str]],
) -> list[tuple[int, int, Path]]:
    """Build min-priority queue of source files by estimated processing cost."""
    priority_queue: list[tuple[int, int, Path]] = []
    for index, source_file in enumerate(pending_source_files):
        target_market_ids = per_file_market_ids.get(str(source_file), set())
        estimated_cost = _estimate_source_processing_cost(
            source_file=source_file,
            target_market_ids=target_market_ids,
        )
        # Prefer cheaper files first for faster feedback and steadier progress.
        heapq.heappush(priority_queue, (estimated_cost, index, source_file))
    return priority_queue


def _build_feature_cache_index(*, output_dir: Path) -> dict[str, list[str]]:
    """Index existing feature shards by market ID."""
    features_root = output_dir / DEFAULT_FEATURES_SUBDIR
    if not features_root.exists():
        return {}

    index: dict[str, list[str]] = {}
    for feature_path in sorted(features_root.rglob("*.parquet")):
        parts = feature_path.relative_to(features_root).parts
        if len(parts) < _FEATURE_CACHE_MIN_PATH_PARTS:
            continue

        market_id = str(parts[-2])
        index.setdefault(market_id, []).append(str(feature_path))

    return index


def _resolve_cached_feature_outputs(
    *,
    target_market_ids: set[str],
    feature_cache_index: dict[str, list[str]],
) -> tuple[set[str], list[str]]:
    """Resolve cached feature files for market IDs already on disk."""
    if not target_market_ids:
        return set(), []

    cached_market_ids: set[str] = set()
    cached_feature_files: list[str] = []
    for market_id in sorted(target_market_ids):
        paths = feature_cache_index.get(str(market_id), [])
        if not paths:
            continue
        cached_market_ids.add(str(market_id))
        cached_feature_files.extend(paths)

    return cached_market_ids, sorted(set(cached_feature_files))


def _count_cached_feature_rows(feature_files: list[str]) -> int:
    """Count cached feature rows from parquet metadata."""
    total_rows = 0
    for raw_path in feature_files:
        path = Path(raw_path)
        if not path.exists():
            continue
        metadata = None
        try:
            metadata = pq.ParquetFile(path).metadata
        except Exception:  # pragma: no cover - defensive parquet metadata read
            metadata = None
        if metadata is not None:
            total_rows += int(metadata.num_rows)
    return total_rows


def _to_json_compatible(value: object) -> object:
    """Recursively convert array-like payload values to plain Python containers."""
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_json_compatible(v) for v in value.tolist()]
    to_list = getattr(value, "tolist", None)
    if callable(to_list) and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = to_list()
        except Exception:  # pragma: no cover - defensive fallback
            return value
        return _to_json_compatible(converted)
    return value


def _log_worker_file_start(
    *,
    options: PrepareOptions,
    worker_name: str,
    source_file: Path,
) -> None:
    if not options.log_worker_progress:
        return
    with _ACTIVE_WORKERS_LOCK:
        _ACTIVE_WORKERS[worker_name] = source_file.name
    print(f"[{worker_name}] START {source_file.name}", flush=True)


def _log_worker_file_done(
    *,
    options: PrepareOptions,
    worker_name: str,
    source_file: Path,
    summary: FileSummary,
) -> None:
    if not options.log_worker_progress:
        return
    with _ACTIVE_WORKERS_LOCK:
        _ACTIVE_WORKERS.pop(worker_name, None)
    print(
        (
            f"[{worker_name}] DONE {source_file.name} "
            f"rows={summary.kept_rows}/{summary.source_rows} "
            f"feature_rows={summary.feature_rows} "
            f"written={summary.feature_written_files} "
            f"skipped={summary.feature_skipped_existing}"
        ),
        flush=True,
    )


def _format_active_worker_snapshot() -> str:
    with _ACTIVE_WORKERS_LOCK:
        if not _ACTIVE_WORKERS:
            return "none"
        pairs = sorted(_ACTIVE_WORKERS.items())
    return ", ".join(f"{worker}:{filename}" for worker, filename in pairs)


def process_source_file(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    source_file: Path,
    target_market_ids: set[str],
    output_dir: Path,
    options: PrepareOptions,
    preexisting_feature_output_files: list[str] | None = None,
    preexisting_feature_rows: int = 0,
) -> FileSummary:
    """Filter one source file and write market-isolated output shards."""
    worker_name = current_thread().name
    _log_worker_file_start(
        options=options,
        worker_name=worker_name,
        source_file=source_file,
    )

    source_mtime_ns = source_file.stat().st_mtime_ns
    metadata = pq.ParquetFile(source_file).metadata
    source_rows = int(metadata.num_rows) if metadata is not None else 0
    preexisting_feature_output_files = sorted(set(preexisting_feature_output_files or []))
    preexisting_feature_count = len(preexisting_feature_output_files)
    preexisting_feature_rows = max(int(preexisting_feature_rows), 0)

    if _is_stop_requested():
        summary = FileSummary(
            source_file=str(source_file),
            source_mtime_ns=source_mtime_ns,
            source_rows=source_rows,
            kept_rows=0,
            feature_rows=preexisting_feature_rows,
            output_files=[],
            feature_output_files=preexisting_feature_output_files,
            written_files=0,
            feature_written_files=0,
            skipped_existing=0,
            feature_skipped_existing=preexisting_feature_count,
            reused_from_manifest=False,
        )
        _log_worker_file_done(
            options=options,
            worker_name=worker_name,
            source_file=source_file,
            summary=summary,
        )
        return summary

    if not target_market_ids:
        summary = FileSummary(
            source_file=str(source_file),
            source_mtime_ns=source_mtime_ns,
            source_rows=source_rows,
            kept_rows=0,
            feature_rows=preexisting_feature_rows,
            output_files=[],
            feature_output_files=preexisting_feature_output_files,
            written_files=0,
            feature_written_files=0,
            skipped_existing=0,
            feature_skipped_existing=preexisting_feature_count,
            reused_from_manifest=False,
        )
        _log_worker_file_done(
            options=options,
            worker_name=worker_name,
            source_file=source_file,
            summary=summary,
        )
        return summary

    filtered_batches, _, close_scanner = _scan_filtered_batches(
        source_file=source_file,
        target_market_ids=target_market_ids,
        scan_engine=options.scan_engine,
        scan_batch_size=options.scan_batch_size,
    )
    fallback_date = _event_date_fallback(source_file)
    kept_rows = 0
    grouped_chunks: dict[tuple[str, str], list[pd.DataFrame]] = {}

    try:
        for batch in filtered_batches:
            if _is_stop_requested():
                break
            if batch.num_rows == 0:
                continue

            filtered_chunk = batch.to_pandas()
            normalized_chunk = _normalize_chunk_for_features(filtered_chunk)
            if normalized_chunk.empty:
                continue

            kept_rows += len(normalized_chunk)
            ts_col = resolve_timestamp_column(normalized_chunk.columns)
            if ts_col is None:
                normalized_chunk["_event_date"] = fallback_date
            else:
                normalized_chunk["_event_date"] = (
                    pd.to_datetime(normalized_chunk[ts_col], utc=True, errors="coerce")
                    .dt.strftime("%Y-%m-%d")
                    .fillna(fallback_date)
                )

            normalized_chunk["market_id"] = normalized_chunk["market_id"].astype(str)
            if ts_col is not None:
                normalized_chunk["_event_ts_sort"] = pd.to_datetime(
                    normalized_chunk[ts_col],
                    utc=True,
                    errors="coerce",
                )

            for (event_date, market_id), frame in normalized_chunk.groupby(
                ["_event_date", "market_id"],
                sort=False,
            ):
                chunk_frame = frame.drop(columns=["_event_date"]).copy()
                grouped_chunks.setdefault(
                    (str(event_date), str(market_id)),
                    [],
                ).append(chunk_frame)
    finally:
        close_scanner()

    if kept_rows == 0:
        summary = FileSummary(
            source_file=str(source_file),
            source_mtime_ns=source_mtime_ns,
            source_rows=source_rows,
            kept_rows=0,
            feature_rows=preexisting_feature_rows,
            output_files=[],
            feature_output_files=preexisting_feature_output_files,
            written_files=0,
            feature_written_files=0,
            skipped_existing=0,
            feature_skipped_existing=preexisting_feature_count,
            reused_from_manifest=False,
        )
        _log_worker_file_done(
            options=options,
            worker_name=worker_name,
            source_file=source_file,
            summary=summary,
        )
        return summary

    output_files: list[str] = []
    feature_output_files: list[str] = list(preexisting_feature_output_files)
    feature_output_file_set = set(feature_output_files)
    written_files = 0
    feature_written_files = 0
    skipped_existing = 0
    feature_skipped_existing = preexisting_feature_count
    feature_rows = preexisting_feature_rows

    grouped_frames: list[tuple[int, str, str, pd.DataFrame]] = []
    for index, (group_key, frames) in enumerate(sorted(grouped_chunks.items())):
        event_date, market_id = group_key
        merged = pd.concat(frames, ignore_index=True)
        if "_event_ts_sort" in merged.columns:
            merged = merged.sort_values(
                by=["_event_ts_sort"],
                kind="stable",
                na_position="last",
            ).drop(columns=["_event_ts_sort"])
        grouped_frames.append((index, event_date, market_id, merged))

    feature_results: list[tuple[int, str, str, pd.DataFrame]] = []
    if not _is_stop_requested():
        feature_results = _compute_feature_frames(
            grouped_frames=grouped_frames,
            feature_workers=options.feature_workers,
            feature_queue_size=options.feature_queue_size,
        )

    for _, event_date, market_id, feature_frame in feature_results:
        if _is_stop_requested():
            break
        feature_rows += len(feature_frame)

        if feature_frame.empty:
            continue

        feature_out_path = build_feature_output_path(
            output_dir=output_dir,
            event_date=event_date,
            market_id=market_id,
            source_file=source_file,
        )
        if feature_out_path.exists() and not options.overwrite:
            feature_skipped_existing += 1
            feature_path_str = str(feature_out_path)
            if feature_path_str not in feature_output_file_set:
                feature_output_file_set.add(feature_path_str)
                feature_output_files.append(feature_path_str)
            continue

        if not options.dry_run:
            feature_out_path.parent.mkdir(parents=True, exist_ok=True)
            feature_table = pa.Table.from_pandas(feature_frame, preserve_index=False)
            feature_tmp_path = feature_out_path.with_suffix(f"{feature_out_path.suffix}.tmp")
            pq.write_table(
                feature_table,
                feature_tmp_path,
                compression=options.compression,
                compression_level=options.compression_level,
            )
            feature_tmp_path.replace(feature_out_path)
            feature_written_files += 1

        feature_path_str = str(feature_out_path)
        if feature_path_str not in feature_output_file_set:
            feature_output_file_set.add(feature_path_str)
            feature_output_files.append(feature_path_str)

    summary = FileSummary(
        source_file=str(source_file),
        source_mtime_ns=source_mtime_ns,
        source_rows=source_rows,
        kept_rows=kept_rows,
        feature_rows=feature_rows,
        output_files=output_files,
        feature_output_files=feature_output_files,
        written_files=written_files,
        feature_written_files=feature_written_files,
        skipped_existing=skipped_existing,
        feature_skipped_existing=feature_skipped_existing,
        reused_from_manifest=False,
    )
    _log_worker_file_done(
        options=options,
        worker_name=worker_name,
        source_file=source_file,
        summary=summary,
    )
    return summary


def prepare_market_backtest_dataset(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    source_dir: Path,
    mapping_dir: Path,
    output_dir: Path,
    market_ids_file: Path | None,
    slug_prefixes: list[str],
    max_markets: int,
    start_date: datetime | None,
    end_date: datetime | None,
    options: PrepareOptions,
    manifest_path: Path,
) -> dict[str, Any]:
    """Prepare a market-isolated dataset and write a manifest."""
    _reset_stop_request()
    run_output_dir, run_slug_prefix, normalized_prefixes = resolve_run_output_dir(
        output_dir=output_dir,
        slug_prefixes=slug_prefixes,
    )
    explicit_ids = collect_market_ids_from_file(market_ids_file)
    mapping_selection = build_mapping_selection(
        mapping_dir=mapping_dir,
        prefixes=slug_prefixes,
        max_markets=max_markets,
    )
    mapping_ids = set(mapping_selection.selected_market_ids)
    required_market_ids = explicit_ids | mapping_ids
    if not required_market_ids:
        raise ValueError(
            "No required markets resolved. Provide --market-ids-file and/or --market-slug-prefix.",
        )

    all_source_files = collect_source_files(
        source_dir,
        start_date=start_date,
        end_date=end_date,
    )

    source_files, per_file_market_ids, pruned_file_count = build_per_file_market_targets(
        source_files=all_source_files,
        explicit_ids=explicit_ids,
        mapping_selection=mapping_selection,
    )

    if not source_files:
        raise FileNotFoundError(
            "No source parquet files found for selected date window/slug time buckets."
        )

    reusable_summaries = _resolve_reusable_summaries(
        manifest_path=manifest_path,
        source_dir=source_dir,
        output_dir=run_output_dir,
        required_market_ids=required_market_ids,
        run_slug_prefix=run_slug_prefix,
        normalized_prefixes=normalized_prefixes,
        market_limit=mapping_selection.market_limit,
        overwrite=options.overwrite,
    )

    summaries: list[FileSummary] = []
    pending_source_files: list[Path] = []
    reusable_count = 0
    for source_file in source_files:
        reusable_summary = reusable_summaries.get(str(source_file))
        if reusable_summary is not None:
            summaries.append(reusable_summary)
            reusable_count += 1
        else:
            pending_source_files.append(source_file)

    cache_reused_count = 0
    preexisting_feature_outputs: dict[str, list[str]] = {}
    preexisting_feature_rows_by_source: dict[str, int] = {}
    if pending_source_files and not options.overwrite:
        feature_cache_index = _build_feature_cache_index(output_dir=run_output_dir)
        if feature_cache_index:
            next_pending_source_files: list[Path] = []
            for source_file in pending_source_files:
                source_key = str(source_file)
                target_market_ids = set(per_file_market_ids.get(source_key, set()))
                cached_market_ids, cached_feature_files = _resolve_cached_feature_outputs(
                    target_market_ids=target_market_ids,
                    feature_cache_index=feature_cache_index,
                )
                if not cached_market_ids:
                    next_pending_source_files.append(source_file)
                    continue

                cached_feature_rows = _count_cached_feature_rows(cached_feature_files)

                remaining_market_ids = target_market_ids - cached_market_ids
                if not remaining_market_ids:
                    summaries.append(
                        process_source_file(
                            source_file=source_file,
                            target_market_ids=set(),
                            output_dir=run_output_dir,
                            options=options,
                            preexisting_feature_output_files=cached_feature_files,
                            preexisting_feature_rows=cached_feature_rows,
                        )
                    )
                    cache_reused_count += 1
                    continue

                per_file_market_ids[source_key] = remaining_market_ids
                preexisting_feature_outputs[source_key] = cached_feature_files
                preexisting_feature_rows_by_source[source_key] = cached_feature_rows
                next_pending_source_files.append(source_file)

            pending_source_files = next_pending_source_files

    resolution_frame = build_resolution_frame_from_mapping(
        mapping_dir=mapping_dir,
        required_market_ids=required_market_ids,
    )
    resolution_output_file = build_resolution_output_path(output_dir=run_output_dir)
    resolution_rows = len(resolution_frame)
    resolution_written = False
    resolution_skipped_existing = False

    if resolution_rows > 0:
        if resolution_output_file.exists() and not options.overwrite:
            resolution_skipped_existing = True
        elif not options.dry_run:
            resolution_output_file.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(resolution_frame, preserve_index=False)
            tmp_path = resolution_output_file.with_suffix(f"{resolution_output_file.suffix}.tmp")
            pq.write_table(
                table,
                tmp_path,
                compression=options.compression,
                compression_level=options.compression_level,
            )
            tmp_path.replace(resolution_output_file)
            resolution_written = True

    worker_total = len(pending_source_files)
    worker_completed = 0
    worker_count = resolve_worker_count(
        max_workers=options.max_workers,
        file_count=worker_total,
    )

    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="pmxt-worker",
    )
    interrupted = False
    try:
        priority_queue = _build_pending_source_priority_queue(
            pending_source_files=pending_source_files,
            per_file_market_ids=per_file_market_ids,
        )
        all_futures: set[Future[FileSummary]] = set()
        pending_futures: set[Future[FileSummary]] = set()

        def submit_next() -> Future[FileSummary] | None:
            if _is_stop_requested():
                return None
            if not priority_queue:
                return None
            _, _, source_file = heapq.heappop(priority_queue)
            future = executor.submit(
                process_source_file,
                source_file=source_file,
                target_market_ids=per_file_market_ids.get(str(source_file), set()),
                output_dir=run_output_dir,
                options=options,
                preexisting_feature_output_files=preexisting_feature_outputs.get(
                    str(source_file),
                    [],
                ),
                preexisting_feature_rows=preexisting_feature_rows_by_source.get(
                    str(source_file),
                    0,
                ),
            )
            all_futures.add(future)
            pending_futures.add(future)
            return future

        initial_slots = min(worker_count, len(priority_queue))
        for _ in range(initial_slots):
            submit_next()

        with tqdm(total=len(source_files), desc="Preparing PMXT", unit="file") as pbar:
            total_reused = reusable_count + cache_reused_count
            if total_reused:
                pbar.update(total_reused)

            pbar.set_postfix(
                {
                    "reused": total_reused,
                    "completed": f"{worker_completed}/{worker_total}",
                },
                refresh=True,
            )
            pbar.refresh()

            print(
                (
                    "Preparing files: "
                    f"total={len(source_files)} reused={total_reused} "
                    f"pending={worker_total} max_workers={worker_count} "
                    f"scan_engine={options.scan_engine} "
                    f"scan_batch_size={options.scan_batch_size} "
                    f"feature_workers={options.feature_workers} "
                    f"feature_queue_size={options.feature_queue_size} "
                    "scheduler=cost-priority"
                ),
                flush=True,
            )

            try:
                last_heartbeat = monotonic()
                heartbeat_interval_seconds = 15.0

                while pending_futures:
                    if _is_stop_requested():
                        break
                    done_futures, _ = wait(
                        pending_futures,
                        timeout=1.0,
                        return_when=FIRST_COMPLETED,
                    )

                    for future in done_futures:
                        pending_futures.discard(future)
                        summaries.append(future.result())
                        worker_completed += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "reused": total_reused,
                                "completed": f"{worker_completed}/{worker_total}",
                            },
                            refresh=False,
                        )
                        submit_next()

                    now = monotonic()
                    if now - last_heartbeat >= heartbeat_interval_seconds:
                        active_workers = (
                            _format_active_worker_snapshot()
                            if options.log_worker_progress
                            else "n/a"
                        )
                        print(
                            (
                                "Still processing: "
                                f"completed={worker_completed}/{worker_total} "
                                f"overall={pbar.n}/{pbar.total} "
                                f"active=[{active_workers}]"
                            ),
                            flush=True,
                        )
                        pbar.refresh()
                        last_heartbeat = now
            except KeyboardInterrupt:
                _request_stop()
                interrupted = True
                for future in all_futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

                summaries.sort(key=lambda item: item.source_file)
                partial_manifest = _build_manifest_payload(
                    source_dir=source_dir,
                    mapping_dir=mapping_dir,
                    output_dir=run_output_dir,
                    required_market_ids=required_market_ids,
                    selected_mapping_market_ids=mapping_selection.ordered_selected_market_ids,
                    run_slug_prefix=run_slug_prefix,
                    normalized_prefixes=normalized_prefixes,
                    selected_market_limit=mapping_selection.market_limit,
                    source_files=source_files,
                    source_file_count_total=len(all_source_files),
                    source_file_count_pruned=pruned_file_count,
                    options=options,
                    summaries=summaries,
                    resolution_output_file=resolution_output_file,
                    resolution_rows=resolution_rows,
                    resolution_written=resolution_written,
                    resolution_skipped_existing=resolution_skipped_existing,
                    interrupted=True,
                )
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(
                    json.dumps(partial_manifest, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                raise
    finally:
        if interrupted or _is_stop_requested():
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True, cancel_futures=False)

    summaries.sort(key=lambda item: item.source_file)

    # Rebuild resolution frame with proper (date, market_id) granularity based on actually-prepared features
    features_root = run_output_dir / DEFAULT_FEATURES_SUBDIR
    if features_root.exists():
        feature_date_market_pairs = collect_date_market_pairs_from_features(features_root)
        if feature_date_market_pairs:
            resolution_frame = build_resolution_frame_from_mapping(
                mapping_dir=mapping_dir,
                required_market_ids=required_market_ids,
                required_date_market_pairs=feature_date_market_pairs,
            )
            resolution_rows = len(resolution_frame)

            if resolution_rows > 0 and not options.dry_run:
                resolution_output_file.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pandas(resolution_frame, preserve_index=False)
                tmp_path = resolution_output_file.with_suffix(f"{resolution_output_file.suffix}.tmp")
                pq.write_table(
                    table,
                    tmp_path,
                    compression=options.compression,
                    compression_level=options.compression_level,
                )
                tmp_path.replace(resolution_output_file)
                resolution_written = True

    manifest = _build_manifest_payload(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=run_output_dir,
        required_market_ids=required_market_ids,
        selected_mapping_market_ids=mapping_selection.ordered_selected_market_ids,
        run_slug_prefix=run_slug_prefix,
        normalized_prefixes=normalized_prefixes,
        selected_market_limit=mapping_selection.market_limit,
        source_files=source_files,
        source_file_count_total=len(all_source_files),
        source_file_count_pruned=pruned_file_count,
        options=options,
        summaries=summaries,
        resolution_output_file=resolution_output_file,
        resolution_rows=resolution_rows,
        resolution_written=resolution_written,
        resolution_skipped_existing=resolution_skipped_existing,
        interrupted=False,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)

    source_dir = args.source_dir.resolve()
    mapping_dir = args.mapping_dir.resolve()
    output_dir = args.output_dir.resolve()
    run_output_dir, _, _ = resolve_run_output_dir(
        output_dir=output_dir,
        slug_prefixes=args.market_slug_prefix,
    )
    market_ids_file = args.market_ids_file.resolve() if args.market_ids_file else None
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path
        else (run_output_dir / "manifest.json").resolve()
    )

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)

    options = PrepareOptions(
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        max_workers=int(args.max_workers),
        compression=str(args.compression),
        compression_level=int(args.compression_level),
        log_worker_progress=bool(args.log_worker_progress),
        feature_workers=int(args.feature_workers),
        feature_queue_size=int(args.feature_queue_size),
        scan_engine=str(args.scan_engine),
        scan_batch_size=resolve_scan_batch_size(scan_batch_size=int(args.scan_batch_size)),
    )

    try:
        manifest = prepare_market_backtest_dataset(
            source_dir=source_dir,
            mapping_dir=mapping_dir,
            output_dir=output_dir,
            market_ids_file=market_ids_file,
            slug_prefixes=args.market_slug_prefix,
            max_markets=int(args.max_markets),
            start_date=start_date,
            end_date=end_date,
            options=options,
            manifest_path=manifest_path,
        )
    except KeyboardInterrupt:
        print("Preparation interrupted by user. Partial manifest written if available.")
        return 130
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("Prepared market dataset successfully")
    print(f"Required markets: {manifest['required_market_count']}")
    print(f"Selected mapping markets: {manifest['selected_mapping_market_count']}")
    print(f"Source files scanned: {manifest['source_file_count']}")
    print(f"Source files pruned: {manifest['source_file_count_pruned']}")
    print(f"Rows kept: {manifest['rows_kept']} / {manifest['rows_total']}")
    print(f"Feature rows: {manifest['feature_rows']}")
    print(f"Resolution rows: {manifest['resolution_rows']}")
    print(f"Output files: {manifest['output_files_written']}")
    print(f"Feature output files: {manifest['feature_output_files_written']}")
    print(f"Resolution file: {manifest['resolution_output_file']}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
