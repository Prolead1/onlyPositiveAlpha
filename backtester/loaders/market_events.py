from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

logger = logging.getLogger(__name__)

_TIMESTAMP_CANDIDATES = (
    "ts_event",
    "timestamp_received",
    "timestamp_created_at",
    "timestamp",
)


@dataclass(frozen=True)
class PyArrowModules:
    """Runtime container for optional pyarrow modules."""

    pa: Any | None
    pc: Any | None
    ds: Any | None
    pq: Any | None


@dataclass(frozen=True)
class MarketEventsLoadRequest:
    """Input parameters for loading market events from parquet shards."""

    market_path: Path
    start: datetime | None
    end: datetime | None
    limit_files: int | None
    max_rows_per_file: int | None
    market_slug_prefix: str | None
    is_pmxt_mode: bool
    mapping_path: Path
    manifest_path: Path | None = None
    recursive_scan: bool = True


@dataclass(frozen=True)
class MarketEventsLoadDeps:
    """Dependency callbacks used by the market events loader."""

    load_condition_ids_for_slug_prefix_fn: Callable[[str], set[str]]
    normalize_market_events_schema_fn: Callable[[pd.DataFrame], pd.DataFrame]
    prepare_timestamp_index_fn: Callable[..., pd.DataFrame]
    filter_by_time_range_fn: Callable[..., pd.DataFrame]


@lru_cache(maxsize=64)
def _cached_manifest_files(
    manifest_path_str: str,
    manifest_mtime_ns: int,
) -> tuple[str, ...]:
    if manifest_mtime_ns < 0:
        return ()

    manifest_path = Path(manifest_path_str)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    if not isinstance(payload, dict):
        return ()

    files_payload = payload.get("files")
    if not isinstance(files_payload, list):
        return ()

    resolved: list[str] = []
    for entry in files_payload:
        if not isinstance(entry, dict):
            continue
        output_files = entry.get("output_files")
        if not isinstance(output_files, list):
            continue
        for raw in output_files:
            path = _resolve_manifest_output_file(manifest_path, raw)
            if path is not None:
                resolved.append(path)

    return tuple(dict.fromkeys(sorted(resolved)))


def _resolve_manifest_output_file(
    manifest_path: Path,
    raw: object,
) -> str | None:
    if not isinstance(raw, str) or not raw:
        return None

    path = Path(raw)
    if not path.is_absolute():
        path = manifest_path.parent / path
    if path.suffix.lower() != ".parquet" or not path.exists():
        return None
    return str(path)


@lru_cache(maxsize=64)
def _cached_globbed_files(
    market_path_str: str,
    *,
    recursive: bool,
    market_path_mtime_ns: int,
) -> tuple[str, ...]:
    if market_path_mtime_ns < 0:
        return ()

    market_path = Path(market_path_str)
    pattern = "**/*.parquet" if recursive else "*.parquet"
    files = sorted(market_path.glob(pattern))
    return tuple(str(path) for path in files if path.is_file())


def resolve_market_event_files(request: MarketEventsLoadRequest) -> tuple[list[Path], str]:
    """Resolve candidate parquet files for market event loading."""
    market_path = request.market_path
    if not market_path.exists():
        return [], "missing-path"

    if request.manifest_path is not None:
        manifest_path = request.manifest_path
        manifest_mtime_ns = manifest_path.stat().st_mtime_ns if manifest_path.exists() else -1
        manifest_files = _cached_manifest_files(
            str(manifest_path),
            manifest_mtime_ns,
        )
        if manifest_files:
            return [Path(path) for path in manifest_files], "manifest"

    market_path_mtime_ns = market_path.stat().st_mtime_ns
    flat_files = _cached_globbed_files(
        str(market_path),
        recursive=False,
        market_path_mtime_ns=market_path_mtime_ns,
    )
    if flat_files:
        return [Path(path) for path in flat_files], "flat-glob"

    if request.recursive_scan:
        recursive_files = _cached_globbed_files(
            str(market_path),
            recursive=True,
            market_path_mtime_ns=market_path_mtime_ns,
        )
        if recursive_files:
            return [Path(path) for path in recursive_files], "recursive-glob"

    return [], "no-files"


def market_event_projection_columns(file: Path, *, modules: PyArrowModules) -> list[str]:
    """Return parquet columns needed for market event loading."""
    if modules.pq is None:
        return []

    schema_names = modules.pq.ParquetFile(file).schema_arrow.names
    preferred_columns = [
        "ts_event",
        "timestamp_received",
        "timestamp_created_at",
        "timestamp",
        "market_id",
        "event_type",
        "update_type",
        "token_id",
        "data",
    ]
    return [column for column in preferred_columns if column in schema_names]


def _to_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


def _resolve_source_row_count(file: Path, *, modules: PyArrowModules) -> int | None:
    if modules.pq is None:
        return None
    try:
        metadata = modules.pq.ParquetFile(file).metadata
    except Exception:  # pragma: no cover - defensive fallback
        return None
    if metadata is None:
        return None
    return int(metadata.num_rows)


def _resolve_time_filter_column(file: Path, *, modules: PyArrowModules) -> str | None:
    if modules.pq is None:
        return None
    try:
        schema_names = modules.pq.ParquetFile(file).schema_arrow.names
    except Exception:  # pragma: no cover - defensive fallback
        return None
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in schema_names:
            return candidate
    return None


def _build_arrow_filter_expression(
    file: Path,
    *,
    market_ids_filter: set[str] | None,
    start: datetime | None,
    end: datetime | None,
    modules: PyArrowModules,
) -> tuple[object | None, str | None]:
    filter_expr = None
    if market_ids_filter:
        filter_expr = modules.ds.field("market_id").isin(sorted(market_ids_filter))

    time_column = _resolve_time_filter_column(file, modules=modules)
    start_utc = _to_utc_datetime(start)
    end_utc = _to_utc_datetime(end)
    if time_column is None or (start_utc is None and end_utc is None):
        return filter_expr, time_column

    time_expr = None
    if start_utc is not None:
        time_expr = modules.ds.field(time_column) >= start_utc
    if end_utc is not None:
        end_filter = modules.ds.field(time_column) <= end_utc
        time_expr = end_filter if time_expr is None else (time_expr & end_filter)

    if time_expr is None:
        return filter_expr, time_column
    if filter_expr is None:
        return time_expr, time_column
    return filter_expr & time_expr, time_column


def filter_market_event_rows(
    df: pd.DataFrame,
    *,
    market_ids_filter: set[str] | None,
    market_slug_prefix: str | None,
) -> pd.DataFrame:
    if df.empty or "market_id" not in df.columns:
        return df

    market_ids = df["market_id"].fillna("").astype(str).str.lower()
    if market_ids_filter:
        return df.loc[market_ids.isin(market_ids_filter)].copy()
    if market_slug_prefix:
        return df.loc[market_ids.str.startswith(market_slug_prefix)].copy()
    return df


def truncate_post_resolution_events(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that occur after the first market_resolved timestamp per market."""
    if df.empty:
        return df
    if "event_type" not in df.columns or "market_id" not in df.columns:
        return df

    event_type = df["event_type"].fillna("").astype(str)
    resolution_events = df.loc[event_type == "market_resolved"]
    if resolution_events.empty:
        return df

    resolution_rows = resolution_events.reset_index()
    time_col = resolution_rows.columns[0]
    resolution_rows["market_id"] = resolution_rows["market_id"].astype(str)
    resolution_rows[time_col] = pd.to_datetime(
        resolution_rows[time_col],
        utc=True,
        errors="coerce",
    )
    resolution_rows = resolution_rows.dropna(subset=[time_col])
    if resolution_rows.empty:
        return df

    first_resolution_by_market = resolution_rows.groupby("market_id", sort=False)[time_col].min()

    rows = df.reset_index()
    ts_col = rows.columns[0]
    rows["market_id"] = rows["market_id"].astype(str)
    rows["_ts_event"] = pd.to_datetime(rows[ts_col], utc=True, errors="coerce")
    rows["_resolved_at"] = rows["market_id"].map(first_resolution_by_market)

    keep_mask = rows["_resolved_at"].isna() | (rows["_ts_event"] <= rows["_resolved_at"])
    trimmed_rows = int((~keep_mask).sum())
    if trimmed_rows == 0:
        return df

    trimmed_markets = int(rows.loc[~keep_mask, "market_id"].nunique())
    logger.info(
        "Trimmed %d post-resolution rows across %d markets",
        trimmed_rows,
        trimmed_markets,
    )

    trimmed = rows.loc[keep_mask].drop(columns=["_ts_event", "_resolved_at"])
    trimmed = trimmed.set_index(ts_col)
    return trimmed.sort_index()


def read_market_events_file_arrow(  # noqa: PLR0913
    file: Path,
    *,
    max_rows: int | None,
    market_ids_filter: set[str] | None,
    start: datetime | None = None,
    end: datetime | None = None,
    modules: PyArrowModules,
) -> pd.DataFrame:
    if modules.pa is None or modules.ds is None:
        return pd.DataFrame()

    columns = market_event_projection_columns(file, modules=modules)
    dataset = modules.ds.dataset(file, format="parquet")
    filter_expr, _ = _build_arrow_filter_expression(
        file,
        market_ids_filter=market_ids_filter,
        start=start,
        end=end,
        modules=modules,
    )

    scanner_kwargs: dict[str, object] = {
        "columns": columns or None,
        "filter": filter_expr,
    }
    if max_rows is not None:
        scanner_kwargs["batch_size"] = min(max_rows, 50_000)

    scanner = dataset.scanner(**scanner_kwargs)
    if max_rows is None:
        return scanner.to_table().to_pandas()

    batches = []
    rows_collected = 0
    for batch in scanner.to_batches():
        batches.append(batch)
        rows_collected += batch.num_rows
        if rows_collected >= max_rows:
            break

    if not batches:
        return pd.DataFrame()

    table = modules.pa.Table.from_batches(batches).slice(0, max_rows)
    return table.to_pandas()


def read_parquet_with_row_limit(
    file: Path,
    *,
    max_rows: int | None,
    modules: PyArrowModules,
) -> pd.DataFrame:
    """Read parquet file with optional row limit."""
    if max_rows is None:
        return pd.read_parquet(file)

    if max_rows <= 0:
        return pd.DataFrame()

    if modules.pa is None or modules.pq is None:
        logger.warning("pyarrow unavailable; falling back to full parquet load")
        return pd.read_parquet(file).head(max_rows)

    parquet_file = modules.pq.ParquetFile(file)
    batches = []
    rows_collected = 0

    for batch in parquet_file.iter_batches(batch_size=min(max_rows, 50_000)):
        batches.append(batch)
        rows_collected += batch.num_rows
        if rows_collected >= max_rows:
            break

    if not batches:
        return pd.DataFrame()

    table = modules.pa.Table.from_batches(batches).slice(0, max_rows)
    return table.to_pandas()


def read_market_events_file(  # noqa: PLR0913
    file: Path,
    *,
    max_rows: int | None,
    market_ids_filter: set[str] | None,
    market_slug_prefix: str | None,
    start: datetime | None = None,
    end: datetime | None = None,
    modules: PyArrowModules,
) -> pd.DataFrame:
    """Read one market events parquet file."""
    if modules.pa is None or modules.pc is None or modules.ds is None:
        df = read_parquet_with_row_limit(
            file,
            max_rows=max_rows,
            modules=modules,
        )
    else:
        df = read_market_events_file_arrow(
            file,
            max_rows=max_rows,
            market_ids_filter=market_ids_filter,
            start=start,
            end=end,
            modules=modules,
        )

    return filter_market_event_rows(
        df,
        market_ids_filter=market_ids_filter,
        market_slug_prefix=market_slug_prefix,
    )


def load_market_events(  # noqa: C901, PLR0912, PLR0915
    request: MarketEventsLoadRequest,
    *,
    deps: MarketEventsLoadDeps,
    modules: PyArrowModules,
) -> pd.DataFrame:
    """Load market events from parquet partitions."""
    load_started = perf_counter()
    parquet_files, source = resolve_market_event_files(request)
    discovered_file_count = len(parquet_files)
    if not parquet_files:
        logger.warning("No market event files found in %s", request.market_path)
        return pd.DataFrame()

    logger.info(
        "Market event file discovery source=%s file_count=%d market_path=%s",
        source,
        len(parquet_files),
        request.market_path,
    )

    if request.limit_files is not None:
        parquet_files = parquet_files[: request.limit_files]

    selected_file_count = len(parquet_files)

    normalized_prefix = (
        str(request.market_slug_prefix).strip().lower() if request.market_slug_prefix else ""
    )
    market_ids_filter: set[str] | None = None
    direct_prefix_filter: str | None = None

    if normalized_prefix:
        if request.is_pmxt_mode:
            condition_ids = deps.load_condition_ids_for_slug_prefix_fn(normalized_prefix)
            if condition_ids:
                market_ids_filter = condition_ids
                logger.info(
                    "Using mapping-based slug filter '%s' with %d condition IDs",
                    normalized_prefix,
                    len(condition_ids),
                )
            else:
                direct_prefix_filter = normalized_prefix
                logger.warning(
                    "No mapping entries found for market_slug_prefix='%s' in %s; "
                    "falling back to direct market_id prefix filtering",
                    normalized_prefix,
                    request.mapping_path,
                )
        else:
            direct_prefix_filter = normalized_prefix

    dfs: list[pd.DataFrame] = []
    files_read = 0
    total_source_rows: int | None = 0
    rows_after_pushdown_total = 0
    rows_after_post_filter_total = 0
    pushdown_time_column_hits = 0

    def _read_with_metrics(file: Path) -> tuple[pd.DataFrame, dict[str, object]]:
        source_rows = _resolve_source_row_count(file, modules=modules)
        pushed_rows = source_rows
        time_column = None

        if modules.pa is not None and modules.pc is not None and modules.ds is not None:
            filter_expr, time_column = _build_arrow_filter_expression(
                file,
                market_ids_filter=market_ids_filter,
                start=request.start,
                end=request.end,
                modules=modules,
            )
            if filter_expr is not None:
                dataset = modules.ds.dataset(file, format="parquet")
                try:
                    pushed_rows = int(dataset.count_rows(filter=filter_expr))
                except Exception:  # pragma: no cover - defensive fallback
                    pushed_rows = None

        df = read_market_events_file(
            file,
            max_rows=request.max_rows_per_file,
            market_ids_filter=market_ids_filter,
            market_slug_prefix=direct_prefix_filter,
            start=request.start,
            end=request.end,
            modules=modules,
        )
        return df, {
            "source_rows": source_rows,
            "rows_after_pushdown": pushed_rows,
            "rows_after_post_filter": len(df),
            "time_pushdown_column": time_column,
        }

    if len(parquet_files) == 1:
        df, file_metrics = _read_with_metrics(parquet_files[0])
        files_read += 1
        if file_metrics["source_rows"] is None:
            total_source_rows = None
        elif total_source_rows is not None:
            total_source_rows += int(file_metrics["source_rows"])
        rows_after_pushdown_total += int(file_metrics["rows_after_pushdown"] or 0)
        rows_after_post_filter_total += int(file_metrics["rows_after_post_filter"])
        if file_metrics["time_pushdown_column"] is not None:
            pushdown_time_column_hits += 1

        if not df.empty:
            dfs.append(df)
    else:
        worker_count = min(8, len(parquet_files))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _read_with_metrics,
                    file,
                ): idx
                for idx, file in enumerate(parquet_files)
            }
            indexed_results: list[tuple[int, pd.DataFrame]] = []
            for future in as_completed(futures):
                file_idx = futures[future]
                df, file_metrics = future.result()
                files_read += 1
                if file_metrics["source_rows"] is None:
                    total_source_rows = None
                elif total_source_rows is not None:
                    total_source_rows += int(file_metrics["source_rows"])
                rows_after_pushdown_total += int(file_metrics["rows_after_pushdown"] or 0)
                rows_after_post_filter_total += int(file_metrics["rows_after_post_filter"])
                if file_metrics["time_pushdown_column"] is not None:
                    pushdown_time_column_hits += 1
                if not df.empty:
                    indexed_results.append((file_idx, df))

        indexed_results.sort(key=lambda item: item[0])
        dfs = [df for _, df in indexed_results]

    if not dfs:
        logger.warning("No market events matched requested filters")
        metrics_payload = {
            "source": source,
            "files_discovered": discovered_file_count,
            "files_selected": selected_file_count,
            "files_read": files_read,
            "files_non_empty": 0,
            "source_rows_total": total_source_rows,
            "rows_after_pushdown": rows_after_pushdown_total,
            "rows_after_post_filter": rows_after_post_filter_total,
            "rows_filtered_pre_materialization": (
                (total_source_rows - rows_after_pushdown_total)
                if total_source_rows is not None
                else None
            ),
            "rows_filtered_post_materialization": rows_after_pushdown_total,
            "time_pushdown_files": pushdown_time_column_hits,
            "elapsed_seconds": round(perf_counter() - load_started, 6),
        }
        logger.info(
            "Market events loader metrics: %s",
            json.dumps(metrics_payload, sort_keys=True),
        )
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    rows_pre_normalization = len(combined)
    logger.info(
        "Loaded %d rows from %d parquet files before normalization",
        len(combined),
        len(dfs),
    )
    combined = deps.normalize_market_events_schema_fn(combined)
    rows_post_normalization = len(combined)
    combined = deps.prepare_timestamp_index_fn(combined, col="ts_event", sort=True)
    combined = deps.filter_by_time_range_fn(combined, start=request.start, end=request.end)
    rows_post_time_filter = len(combined)
    combined = truncate_post_resolution_events(combined)
    rows_post_resolution_trim = len(combined)

    metrics_payload = {
        "source": source,
        "files_discovered": discovered_file_count,
        "files_selected": selected_file_count,
        "files_read": files_read,
        "files_non_empty": len(dfs),
        "source_rows_total": total_source_rows,
        "rows_after_pushdown": rows_after_pushdown_total,
        "rows_after_post_filter": rows_after_post_filter_total,
        "rows_pre_normalization": rows_pre_normalization,
        "rows_post_normalization": rows_post_normalization,
        "rows_post_time_filter": rows_post_time_filter,
        "rows_post_resolution_trim": rows_post_resolution_trim,
        "rows_filtered_pre_materialization": (
            (total_source_rows - rows_after_pushdown_total)
            if total_source_rows is not None
            else None
        ),
        "rows_filtered_post_materialization": (
            rows_after_pushdown_total - rows_after_post_filter_total
        ),
        "time_pushdown_files": pushdown_time_column_hits,
        "elapsed_seconds": round(perf_counter() - load_started, 6),
    }
    logger.info("Market events loader metrics: %s", json.dumps(metrics_payload, sort_keys=True))

    logger.info("Loaded %d market events", len(combined))
    return combined
