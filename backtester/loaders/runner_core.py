from __future__ import annotations

import hashlib
import json
import logging
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pa = None
    pc = None
    ds = None
    pq = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from backtester.loaders.crypto_prices import load_crypto_prices as load_crypto_prices_loader
from backtester.loaders.market_events import (
    MarketEventsLoadDeps,
    MarketEventsLoadRequest,
    PyArrowModules,
)
from backtester.loaders.market_events import (
    filter_market_event_rows as filter_market_event_rows_loader,
)
from backtester.loaders.market_events import (
    load_market_events as load_market_events_loader,
)
from backtester.loaders.market_events import (
    market_event_projection_columns as market_event_projection_columns_loader,
)
from backtester.loaders.market_events import (
    read_market_events_file as read_market_events_file_loader,
)
from backtester.loaders.market_events import (
    read_market_events_file_arrow as read_market_events_file_arrow_loader,
)
from backtester.loaders.market_events import (
    read_parquet_with_row_limit as read_parquet_with_row_limit_loader,
)
from backtester.normalize.market_lookup import (
    filter_market_events_by_slug_prefix,
    load_condition_ids_for_slug_prefix,
)
from backtester.normalize.schema import (
    drop_missing_pmxt_tokens,
    extract_token_from_payload,
    normalize_market_event_payload,
    normalize_market_events_schema,
    normalize_price_change_payload,
    resolve_market_event_timestamp_column,
    resolve_market_event_type_column,
)
from utils.dataframes import filter_by_time_range, prepare_timestamp_index

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

logger = logging.getLogger(__name__)


class ProgressBarLike(Protocol):
    """Minimal progress-bar interface used by this runner."""

    def update(self, n: int = 1) -> object:
        """Advance the progress bar by ``n`` steps."""
        ...

    def set_postfix(
        self,
        ordered_dict: Mapping[str, object] | None = None,
        *,
        refresh: bool = True,
        **kwargs: object,
    ) -> object:
        """Set the displayed progress metadata."""
        ...

    def close(self) -> None:
        """Close the progress bar."""
        ...


class BacktestCoreOps:
    """Market/feature loading and lightweight runner utility methods."""

    @staticmethod
    def _to_utc_timestamp(value: object) -> pd.Timestamp | None:
        parsed = pd.to_datetime(str(value), utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed

    @staticmethod
    def _to_float_or_nan(value: object) -> float:
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return float("nan")

    @staticmethod
    def _make_progress_bar(
        *,
        enabled: bool,
        total: int,
        desc: str,
        unit: str,
    ) -> ProgressBarLike | None:
        if not enabled or total <= 0 or tqdm is None:
            return None
        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=False,
            dynamic_ncols=True,
        )

    def _load_condition_ids_for_slug_prefix(self, slug_prefix: str) -> set[str]:
        """Resolve slug prefix to condition IDs using cached mapping shards."""
        return load_condition_ids_for_slug_prefix(
            self.mapping_path,
            slug_prefix,
            cache=self._slug_prefix_condition_ids_cache,
        )

    def _filter_market_events_by_slug_prefix(
        self,
        market_events: pd.DataFrame,
        market_slug_prefix: str | None,
    ) -> pd.DataFrame:
        """Filter events by market slug prefix.

        In PMXT mode, this uses mapping shards (slug -> conditionId) as the default
        filtering method because market_id values are condition IDs rather than slugs.
        """
        return filter_market_events_by_slug_prefix(
            market_events,
            market_slug_prefix,
            is_pmxt_mode=self.is_pmxt_mode,
            mapping_path=self.mapping_path,
            condition_ids_lookup=self._load_condition_ids_for_slug_prefix,
        )

    def _normalize_price_change_payload(self, payload: dict, token_id: str) -> dict:
        """Normalize PMXT-style price_change payload into the canonical schema."""
        return normalize_price_change_payload(payload, token_id)

    @staticmethod
    def _resolve_market_event_timestamp_column(columns: pd.Index) -> str:
        return resolve_market_event_timestamp_column(columns)

    @staticmethod
    def _resolve_market_event_type_column(columns: pd.Index) -> str:
        return resolve_market_event_type_column(columns)

    @staticmethod
    def _extract_token_from_payload(payload: object) -> str:
        return extract_token_from_payload(payload)

    def _normalize_market_event_payload(
        self,
        payload: object,
        token_id: str,
        event_type: str,
    ) -> object:
        return normalize_market_event_payload(payload, token_id, event_type)

    def _drop_missing_pmxt_tokens(self, normalized: pd.DataFrame) -> pd.DataFrame:
        return drop_missing_pmxt_tokens(normalized, is_pmxt_mode=self.is_pmxt_mode)

    def _normalize_market_events_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize market events from supported feeds to a single schema."""
        return normalize_market_events_schema(df, is_pmxt_mode=self.is_pmxt_mode)

    def load_market_events(  # noqa: PLR0913
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit_files: int | None = None,
        max_rows_per_file: int | None = None,
        market_slug_prefix: str | None = None,
        market_events_manifest_path: Path | str | None = None,
        *,
        recursive_scan: bool = True,
    ) -> pd.DataFrame:
        """Load market events from Parquet partitions.

        Parameters
        ----------
        start : datetime | None
            Start of time range (inclusive). If None, loads all.
        end : datetime | None
            End of time range (inclusive). If None, loads all.
        limit_files : int | None
            Limit number of parquet files to load. If None, loads all files.
        max_rows_per_file : int | None
            If provided, read at most this many rows from each parquet file.
            Useful for tests and quick diagnostics on very large partitions.
        market_slug_prefix : str | None
            If provided, keep only rows for this slug prefix. In PMXT mode this
            defaults to mapping-based slug->conditionId filtering.
        market_events_manifest_path : Path | str | None
            Optional path to a manifest produced by the prepare-market-dataset
            script. When available, the loader uses manifest output files directly.
            Defaults to <market_path>/manifest.json when present.
        recursive_scan : bool
            Whether to fall back to recursive parquet discovery when no top-level
            parquet files are present at market_path.

        Returns
        -------
        pd.DataFrame
            Market events dataframe with timestamp index.
        """
        manifest_path: Path | None = None
        if market_events_manifest_path is None:
            auto_manifest = self.market_path / "manifest.json"
            if auto_manifest.exists():
                manifest_path = auto_manifest
        else:
            manifest_path = Path(market_events_manifest_path)

        modules = PyArrowModules(pa=pa, pc=pc, ds=ds, pq=pq)
        request = MarketEventsLoadRequest(
            market_path=self.market_path,
            start=start,
            end=end,
            limit_files=limit_files,
            max_rows_per_file=max_rows_per_file,
            market_slug_prefix=market_slug_prefix,
            is_pmxt_mode=self.is_pmxt_mode,
            mapping_path=self.mapping_path,
            manifest_path=manifest_path,
            recursive_scan=recursive_scan,
        )
        deps = MarketEventsLoadDeps(
            load_condition_ids_for_slug_prefix_fn=self._load_condition_ids_for_slug_prefix,
            normalize_market_events_schema_fn=self._normalize_market_events_schema,
            prepare_timestamp_index_fn=prepare_timestamp_index,
            filter_by_time_range_fn=filter_by_time_range,
        )

        return load_market_events_loader(
            request,
            deps=deps,
            modules=modules,
        )

    @staticmethod
    def _resolve_prepared_feature_files(  # noqa: C901, PLR0912
        *,
        market_path: Path,
        manifest_path: Path | None,
        recursive_scan: bool,
    ) -> list[Path]:
        """Resolve prepared feature file list from manifest or folder scan."""
        if manifest_path is not None and manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None

            if isinstance(payload, dict):
                entries = payload.get("files")
                resolved_files: list[Path] = []
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        feature_files = entry.get("feature_output_files")
                        if not isinstance(feature_files, list):
                            continue
                        for raw in feature_files:
                            if not isinstance(raw, str) or not raw:
                                continue
                            path = Path(raw)
                            if not path.is_absolute():
                                path = manifest_path.parent / path
                            if path.suffix.lower() == ".parquet" and path.exists():
                                resolved_files.append(path)
                if resolved_files:
                    return sorted(set(resolved_files))

        feature_root = market_path / "features"
        if not feature_root.exists():
            return []
        pattern = "**/*.parquet" if recursive_scan else "*.parquet"
        return sorted(path for path in feature_root.glob(pattern) if path.is_file())

    @staticmethod
    def _extract_market_id_from_prepared_feature_path(path: Path) -> str | None:
        """Infer market_id from prepared feature layout features/<date>/<market_id>/file."""
        parts = list(path.parts)
        try:
            features_idx = parts.index("features")
        except ValueError:
            return None

        market_idx = features_idx + 2
        if market_idx >= len(parts):
            return None
        value = str(parts[market_idx]).strip()
        return value or None

    def load_prepared_feature_market_ids(
        self,
        *,
        limit_files: int | None = None,
        features_manifest_path: Path | str | None = None,
        recursive_scan: bool = True,
    ) -> list[str]:
        """Load distinct market IDs from prepared feature artifacts.

        This avoids fully materializing every shard just to extract market IDs.
        """
        manifest_path: Path | None = None
        if features_manifest_path is None:
            auto_manifest = self.market_path / "manifest.json"
            if auto_manifest.exists():
                manifest_path = auto_manifest
        else:
            manifest_path = Path(features_manifest_path)

        parquet_files = self._resolve_prepared_feature_files(
            market_path=self.market_path,
            manifest_path=manifest_path,
            recursive_scan=recursive_scan,
        )
        if not parquet_files:
            return []

        if limit_files is not None:
            parquet_files = parquet_files[:limit_files]

        market_ids: set[str] = set()
        unresolved_files: list[Path] = []
        for path in parquet_files:
            inferred_market_id = self._extract_market_id_from_prepared_feature_path(path)
            if inferred_market_id is None:
                unresolved_files.append(path)
                continue
            market_ids.add(inferred_market_id)

        # Fallback for non-standard folder structures.
        for path in unresolved_files:
            frame = pd.read_parquet(path, columns=["market_id"])
            if "market_id" not in frame.columns:
                continue
            values = frame["market_id"].dropna().astype(str)
            market_ids.update(values[values.str.len() > 0].tolist())

        return sorted(market_ids)

    def count_prepared_feature_market_batches(
        self,
        *,
        market_batch_size: int,
        limit_files: int | None = None,
        features_manifest_path: Path | str | None = None,
        recursive_scan: bool = True,
    ) -> int:
        """Return the number of market batches implied by a batch size."""
        if market_batch_size <= 0:
            msg = "market_batch_size must be greater than 0"
            raise ValueError(msg)

        market_ids = self.load_prepared_feature_market_ids(
            limit_files=limit_files,
            features_manifest_path=features_manifest_path,
            recursive_scan=recursive_scan,
        )
        if not market_ids:
            return 0
        return ceil(len(market_ids) / market_batch_size)

    def load_prepared_features(  # noqa: C901, PLR0912, PLR0913
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit_files: int | None = None,
        features_manifest_path: Path | str | None = None,
        *,
        recursive_scan: bool = True,
        market_ids: set[str] | list[str] | None = None,
    ) -> pd.DataFrame:
        """Load precomputed orderbook features from prepared dataset shards."""
        manifest_path: Path | None = None
        if features_manifest_path is None:
            auto_manifest = self.market_path / "manifest.json"
            if auto_manifest.exists():
                manifest_path = auto_manifest
        else:
            manifest_path = Path(features_manifest_path)

        parquet_files = self._resolve_prepared_feature_files(
            market_path=self.market_path,
            manifest_path=manifest_path,
            recursive_scan=recursive_scan,
        )
        if not parquet_files:
            logger.info("No prepared feature files found in %s", self.market_path)
            return pd.DataFrame()

        if limit_files is not None:
            parquet_files = parquet_files[:limit_files]

        selected_market_ids: set[str] | None = None
        if market_ids is not None:
            selected_market_ids = {
                str(value).strip() for value in market_ids if str(value).strip()
            }
            if not selected_market_ids:
                return pd.DataFrame()

            filtered_files: list[Path] = []
            unresolved_files: list[Path] = []
            for path in parquet_files:
                inferred_market_id = self._extract_market_id_from_prepared_feature_path(path)
                if inferred_market_id is None:
                    unresolved_files.append(path)
                    continue
                if inferred_market_id in selected_market_ids:
                    filtered_files.append(path)

            # Keep unresolved files for row-level filtering fallback.
            parquet_files = filtered_files + unresolved_files
            if not parquet_files:
                return pd.DataFrame()

        dfs = [pd.read_parquet(path) for path in parquet_files]
        combined = pd.concat(dfs, ignore_index=True)
        if combined.empty:
            return combined

        if selected_market_ids is not None and "market_id" in combined.columns:
            combined = combined.loc[
                combined["market_id"].astype(str).isin(selected_market_ids)
            ].copy()
            if combined.empty:
                return combined

        if "ts_event" in combined.columns:
            combined = prepare_timestamp_index(combined, col="ts_event", sort=True)
            combined = filter_by_time_range(combined, start=start, end=end)

        logger.info("Loaded %d prepared feature rows", len(combined))
        return combined

    @staticmethod
    def _resolve_prepared_resolution_file(
        *,
        market_path: Path,
        manifest_path: Path | None,
    ) -> Path | None:
        if manifest_path is not None and manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None

            if isinstance(payload, dict):
                raw = payload.get("resolution_output_file")
                if isinstance(raw, str) and raw:
                    path = Path(raw)
                    if not path.is_absolute():
                        path = manifest_path.parent / path
                    if path.exists() and path.suffix.lower() == ".parquet":
                        return path

        default_path = market_path / "resolution" / "resolution_frame.parquet"
        if default_path.exists():
            return default_path
        return None

    def load_prepared_resolution_frame(
        self,
        *,
        resolution_manifest_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """Load the prepared resolution frame from manifest or default paths."""
        manifest_path: Path | None = None
        if resolution_manifest_path is None:
            auto_manifest = self.market_path / "manifest.json"
            if auto_manifest.exists():
                manifest_path = auto_manifest
        else:
            manifest_path = Path(resolution_manifest_path)

        resolution_file = self._resolve_prepared_resolution_file(
            market_path=self.market_path,
            manifest_path=manifest_path,
        )
        if resolution_file is None:
            logger.info("No prepared resolution file found in %s", self.market_path)
            return pd.DataFrame()

        frame = pd.read_parquet(resolution_file)
        if frame.empty:
            return frame

        if "market_id" not in frame.columns and frame.index.name == "market_id":
            frame = frame.reset_index()

        if "market_id" not in frame.columns:
            logger.warning("Prepared resolution frame missing market_id column: %s", resolution_file)
            return pd.DataFrame()

        frame["market_id"] = frame["market_id"].astype(str)

        if "resolved_at" in frame.columns:
            frame["resolved_at"] = pd.to_datetime(frame["resolved_at"], utc=True, errors="coerce")
        else:
            frame["resolved_at"] = pd.NaT

        if "feature_date" in frame.columns:
            parsed_feature_dates = pd.to_datetime(frame["feature_date"], utc=True, errors="coerce")
            normalized_feature_dates = parsed_feature_dates.dt.strftime("%Y-%m-%d")
            raw_feature_dates = frame["feature_date"].astype(str).str.strip()
            raw_feature_dates = raw_feature_dates.mask(
                raw_feature_dates.isin({"", "nan", "NaT", "None"})
            )
            frame["feature_date"] = normalized_feature_dates.fillna(raw_feature_dates)
        else:
            frame["feature_date"] = frame["resolved_at"].dt.strftime("%Y-%m-%d")

        dedupe_columns = [
            col
            for col in ("market_id", "feature_date", "resolved_at", "winning_asset_id")
            if col in frame.columns
        ]
        if dedupe_columns:
            frame = frame.drop_duplicates(subset=dedupe_columns, keep="first")

        sort_columns = [col for col in ("market_id", "feature_date", "resolved_at") if col in frame.columns]
        if sort_columns:
            frame = frame.sort_values(sort_columns).reset_index(drop=True)

        return frame

    def _market_event_projection_columns(self, file: Path) -> list[str]:
        """Return the subset of parquet columns needed for market event loading."""
        return market_event_projection_columns_loader(
            file,
            modules=PyArrowModules(pa=pa, pc=pc, ds=ds, pq=pq),
        )

    @staticmethod
    def _filter_market_event_rows(
        df: pd.DataFrame,
        *,
        market_ids_filter: set[str] | None,
        market_slug_prefix: str | None,
    ) -> pd.DataFrame:
        return filter_market_event_rows_loader(
            df,
            market_ids_filter=market_ids_filter,
            market_slug_prefix=market_slug_prefix,
        )

    def _read_market_events_file_arrow(
        self,
        file: Path,
        *,
        max_rows: int | None = None,
        market_ids_filter: set[str] | None = None,
    ) -> pd.DataFrame:
        return read_market_events_file_arrow_loader(
            file,
            max_rows=max_rows,
            market_ids_filter=market_ids_filter,
            modules=PyArrowModules(pa=pa, pc=pc, ds=ds, pq=pq),
        )

    def _read_market_events_file(
        self,
        file: Path,
        *,
        max_rows: int | None = None,
        market_ids_filter: set[str] | None = None,
        market_slug_prefix: str | None = None,
    ) -> pd.DataFrame:
        """Read a single market-event parquet file."""
        return read_market_events_file_loader(
            file,
            max_rows=max_rows,
            market_ids_filter=market_ids_filter,
            market_slug_prefix=market_slug_prefix,
            modules=PyArrowModules(pa=pa, pc=pc, ds=ds, pq=pq),
        )

    def _read_parquet_with_row_limit(
        self,
        file: Path,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        """Read parquet file with optional row limit.

        Uses pyarrow row-group/batch iteration when max_rows is set to avoid
        loading multi-million-row files fully during tests.
        """
        return read_parquet_with_row_limit_loader(
            file,
            max_rows=max_rows,
            modules=PyArrowModules(pa=pa, pc=pc, ds=ds, pq=pq),
        )

    def _compute_cache_key(
        self,
        parquet_files: list[Path],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> str:
        """Compute cache key based on input files and time range.

        Parameters
        ----------
        parquet_files : list[Path]
            List of parquet files being processed.
        start : datetime | None
            Start time filter.
        end : datetime | None
            End time filter.

        Returns
        -------
        str
            Cache key hash.
        """
        # Create hash from file paths, modification times, and time range
        hash_input = []
        for file in sorted(parquet_files):
            hash_input.append(str(file.name))
            hash_input.append(str(file.stat().st_mtime))

        if start:
            hash_input.append(start.isoformat())
        if end:
            hash_input.append(end.isoformat())

        combined = "|".join(hash_input)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def compute_orderbook_features_cached(
        self,
        market_events: pd.DataFrame,
        cache_key: str | None = None,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compute orderbook features with caching support.

        Parameters
        ----------
        market_events : pd.DataFrame
            Market events dataframe.
        cache_key : str | None
            Cache key for this computation. If None, uses hash of event count and time range.
        use_cache : bool
            Whether to use cached features if available.

        Returns
        -------
        pd.DataFrame
            Dataframe with orderbook features.
        """
        return self.feature_generator.generate_orderbook_features_cached(
            market_events,
            cache_key=cache_key,
            use_cache=use_cache,
        )

    def load_crypto_prices(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> pd.DataFrame:
        """Load crypto price events from Parquet partitions.

        Parameters
        ----------
        start : datetime | None
            Start of time range (inclusive). If None, loads all.
        end : datetime | None
            End of time range (inclusive). If None, loads all.

        Returns
        -------
        pd.DataFrame
            Crypto price dataframe with timestamp index.
        """
        return load_crypto_prices_loader(
            self.rtds_path,
            start=start,
            end=end,
            prepare_timestamp_index_fn=prepare_timestamp_index,
            filter_by_time_range_fn=filter_by_time_range,
        )

    def compute_orderbook_features_df(self, market_events: pd.DataFrame) -> pd.DataFrame:
        """Compute orderbook features using book events only.

        Parameters
        ----------
        market_events : pd.DataFrame
            Market events dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with orderbook features.
        """
        return self.feature_generator.generate_orderbook_features(market_events)

    def compute_trade_features_df(
        self, market_events: pd.DataFrame, window: str = "5min"
    ) -> pd.DataFrame:
        """Compute rolling trade flow features.

        Parameters
        ----------
        market_events : pd.DataFrame
            Market events dataframe.
        window : str
            Rolling window for aggregation.

        Returns
        -------
        pd.DataFrame
            Dataframe with trade features indexed by time.
        """
        return self.feature_generator.generate_trade_features(market_events, window=window)

    def build_feature_dataset(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        crypto_window: str = "5min",
    ) -> pd.DataFrame:
        """Build complete feature dataset by joining market and crypto features.

        Parameters
        ----------
        start : datetime | None
            Start time for data loading.
        end : datetime | None
            End time for data loading.
        crypto_window : str
            Lookback window for crypto features.

        Returns
        -------
        pd.DataFrame
            Feature dataset indexed by timestamp.
        """
        logger.info("Building feature dataset from %s to %s", start, end)

        market_events = self.load_market_events(start, end)
        crypto_prices = self.load_crypto_prices(start, end)

        if market_events.empty:
            logger.error("No market events loaded")
            return pd.DataFrame()

        return self.feature_generator.build_feature_dataset_from_frames(
            market_events,
            crypto_prices,
            crypto_window=crypto_window,
        )
