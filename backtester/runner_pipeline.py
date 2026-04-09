from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from itertools import product
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

import pandas as pd

from backtester.config.normalization import coerce_backtest_config, normalize_strategy_output
from backtester.config.types import BacktestConfig, BacktestRunResult, RunMetadata
from backtester.loaders.resolution import DEFAULT_RESOLUTION_COLUMNS
from backtester.simulation.analytics import (
    build_equity_curve,
    build_market_diagnostics,
    build_regime_diagnostics,
    summarize_backtest,
)

logger = logging.getLogger(__name__)

StrategyCallable = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]


class PreparedInputsCacheEntry(TypedDict):
    """Cached prepared inputs reused across equivalent backtest runs."""

    market_events: pd.DataFrame
    features: pd.DataFrame
    resolution_frame: pd.DataFrame
    resolution_diagnostics: pd.DataFrame
    data_quality_report: pd.DataFrame
    feature_health: pd.DataFrame
    prep_error_events: list[dict[str, object]]
    feature_cache_key: str | None
    resolution_repair_audit: list[dict[str, object]]


_RESOLUTION_COLUMNS = DEFAULT_RESOLUTION_COLUMNS


class BacktestPipelineOrchestrator:
    """High-level run orchestration and artifact generation methods."""

    @staticmethod
    def _materialize_market_events_from_features(features: pd.DataFrame) -> pd.DataFrame:
        """Build a lightweight event frame from prepared features.

        The backtest simulation consumes features directly, but downstream quality gates
        and result payloads still expect a market-events frame.
        """
        if features.empty:
            return pd.DataFrame(columns=["market_id", "token_id", "event_type", "data"])

        frame = features.copy(deep=False)
        if "ts_event" in frame.columns:
            ts_event = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
            if ts_event.notna().any():
                valid_mask = ts_event.notna()
                frame = frame.loc[valid_mask].copy()
                frame.index = ts_event.loc[valid_mask]
        else:
            ts_event = pd.to_datetime(frame.index, utc=True, errors="coerce")
            if ts_event.notna().any():
                valid_mask = ts_event.notna()
                frame = frame.loc[valid_mask].copy()
                frame.index = ts_event[valid_mask]

        if frame.empty:
            return pd.DataFrame(columns=["market_id", "token_id", "event_type", "data"])

        market_events = pd.DataFrame(index=frame.index)
        for col in ("market_id", "token_id"):
            if col in frame.columns:
                market_events[col] = frame[col]
        market_events["event_type"] = "feature_observation"
        market_events["data"] = None
        return market_events.sort_index()

    def _build_run_metadata(
        self,
        cfg: BacktestConfig,
        *,
        start: datetime | None,
        end: datetime | None,
        cache_signature: str,
    ) -> RunMetadata:
        config_snapshot = asdict(cfg)
        metadata = RunMetadata(
            run_id=uuid4().hex,
            created_at=datetime.now(tz=UTC),
            config_snapshot=config_snapshot,
            config_hash=self._hash_payload(config_snapshot),
            git_commit=self._resolve_git_commit(),
            data_window={
                "start": start.isoformat() if start is not None else None,
                "end": end.isoformat() if end is not None else None,
            },
            cache_signature=cache_signature,
            mode=cfg.mode,
        )

        required = {
            "run_id": metadata.run_id,
            "created_at": metadata.created_at,
            "config_snapshot": metadata.config_snapshot,
            "config_hash": metadata.config_hash,
            "data_window": metadata.data_window,
            "cache_signature": metadata.cache_signature,
            "mode": metadata.mode,
        }
        missing = [name for name, value in required.items() if value in (None, "", {})]
        if missing:
            missing_text = ", ".join(sorted(missing))
            msg = f"Missing required run metadata fields: {missing_text}"
            raise RuntimeError(msg)

        return metadata

    def _build_empty_run_result(  # noqa: PLR0913
        self,
        *,
        cfg: BacktestConfig,
        start: datetime | None,
        end: datetime | None,
        market_events: pd.DataFrame,
        features: pd.DataFrame,
        resolution_frame: pd.DataFrame,
        resolution_diagnostics: pd.DataFrame,
        data_quality_report: pd.DataFrame,
        feature_health: pd.DataFrame,
        error_events: list[dict[str, object]],
        cache_key: str | None,
    ) -> BacktestRunResult:
        metadata = self._build_run_metadata(
            cfg,
            start=start,
            end=end,
            cache_signature=self._compute_cache_signature(features, cache_key, cfg=cfg),
        )

        error_ledger = pd.DataFrame(error_events)
        if error_ledger.empty:
            error_ledger = pd.DataFrame(
                columns=["stage", "severity", "market_id", "reason", "details"]
            )

        order_ledger = self._attach_run_id(pd.DataFrame(), metadata.run_id)
        trade_ledger = self._attach_run_id(pd.DataFrame(), metadata.run_id)
        diagnostics_by_market = self._attach_run_id(
            build_market_diagnostics(pd.DataFrame()),
            metadata.run_id,
        )
        diagnostics_by_regime = self._attach_run_id(
            build_regime_diagnostics(pd.DataFrame()),
            metadata.run_id,
        )
        backtest_summary = self._attach_run_id(
            summarize_backtest(trade_ledger),
            metadata.run_id,
        )
        equity_curve = self._attach_run_id(
            build_equity_curve(trade_ledger),
            metadata.run_id,
        )

        return BacktestRunResult(
            metadata=metadata,
            market_events=market_events,
            features=features,
            resolution_frame=resolution_frame,
            resolution_diagnostics=self._attach_run_id(
                resolution_diagnostics,
                metadata.run_id,
            ),
            strategy_signals={},
            order_ledger=order_ledger,
            trade_ledger=trade_ledger,
            backtest_summary=backtest_summary,
            equity_curve=equity_curve,
            data_quality_report=self._attach_run_id(data_quality_report, metadata.run_id),
            feature_health=self._attach_run_id(feature_health, metadata.run_id),
            error_ledger=self._attach_run_id(error_ledger, metadata.run_id),
            settlement_repair_audit=self._attach_run_id(
                pd.DataFrame(self._last_resolution_repair_audit),
                metadata.run_id,
            ),
            diagnostics_by_market=diagnostics_by_market,
            diagnostics_by_regime=diagnostics_by_regime,
            cache_metadata={
                "cache_signature": self.feature_generator.cache_signature(),
                "cache_schema_version": cfg.cache_schema_version,
                "cache_computation_signature": cfg.cache_computation_signature,
            },
        )

    def _run_backtest_in_market_batches(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        *,
        cfg: BacktestConfig,
        strategy_map: Mapping[str, StrategyCallable],
        mapping_dir: str | Path,  # noqa: ARG002
        prepared_manifest_path: Path | str | None,
        start: datetime | None,
        end: datetime | None,
        limit_files: int | None,
        market_batch_size: int,
        prepared_feature_market_ids: set[str] | list[str] | None,
        error_events: list[dict[str, object]],
        data_quality_report: pd.DataFrame,
        feature_health: pd.DataFrame,
        run_started: float,
        regime_lookup: dict[pd.Timestamp, str] | None = None,
    ) -> BacktestRunResult:
        if market_batch_size <= 0:
            msg = "market_batch_size must be greater than 0"
            raise ValueError(msg)

        if cfg.metrics_logging_enabled:
            logger.info(
                "Backtest batch mode enabled: strategies=%d batch_size=%d",
                len(strategy_map),
                market_batch_size,
            )

        stage_started = perf_counter()
        resolution_frame = self.load_prepared_resolution_frame(
            resolution_manifest_path=prepared_manifest_path,
        )
        resolution_diagnostics = pd.DataFrame()

        if resolution_frame.empty:
            if cfg.mode == "strict":
                raise RuntimeError(
                    "No prepared resolution rows found. "
                    "Run the standalone dataset preparation script first."
                )
            self._append_error(
                error_events,
                stage="resolution_validation",
                reason="prepared_resolution_empty",
                severity="warning",
            )
            return self._build_empty_run_result(
                cfg=cfg,
                start=start,
                end=end,
                market_events=pd.DataFrame(),
                features=pd.DataFrame(),
                resolution_frame=pd.DataFrame(columns=_RESOLUTION_COLUMNS).set_index("market_id"),
                resolution_diagnostics=pd.DataFrame(),
                data_quality_report=data_quality_report,
                feature_health=feature_health,
                error_events=error_events,
                cache_key=None,
            )

        if cfg.metrics_logging_enabled:
            logger.info(
                "Backtest resolution (batch mode): markets=%d diagnostics_rows=%d elapsed=%.2fs",
                len(resolution_frame),
                len(resolution_diagnostics),
                perf_counter() - stage_started,
            )

        if prepared_feature_market_ids is not None:
            selected_market_ids = sorted(
                {str(value).strip() for value in prepared_feature_market_ids if str(value).strip()}
            )
        else:
            selected_market_ids = self.load_prepared_feature_market_ids(
                limit_files=limit_files,
                features_manifest_path=prepared_manifest_path,
                recursive_scan=True,
            )

        # Batch orchestration must preserve temporal order so incremental continuation
        # state (open positions, risk state, and capital) advances monotonically.
        resolved_at_by_market: dict[str, pd.Timestamp] = {}
        if {"market_id", "resolved_at"}.issubset(resolution_frame.columns):
            resolution_order_frame = resolution_frame[["market_id", "resolved_at"]].copy()
            resolution_order_frame["market_id"] = resolution_order_frame["market_id"].astype(str)
            resolution_order_frame["resolved_at"] = pd.to_datetime(
                resolution_order_frame["resolved_at"],
                utc=True,
                errors="coerce",
            )
            resolution_order_frame = resolution_order_frame.dropna(
                subset=["market_id", "resolved_at"]
            )
            if not resolution_order_frame.empty:
                resolved_at_by_market = (
                    resolution_order_frame.groupby("market_id", sort=False)["resolved_at"]
                    .min()
                    .to_dict()
                )

        resolved_market_ids: list[tuple[pd.Timestamp, str]] = []
        unresolved_market_ids: list[str] = []
        for market_id in selected_market_ids:
            resolved_at_value = resolved_at_by_market.get(market_id)
            if pd.isna(resolved_at_value):
                unresolved_market_ids.append(market_id)
                continue
            resolved_market_ids.append((resolved_at_value, market_id))

        selected_market_ids = [
            market_id
            for _, market_id in sorted(resolved_market_ids, key=lambda item: (item[0], item[1]))
        ] + sorted(unresolved_market_ids)

        if not selected_market_ids:
            if cfg.mode == "strict":
                raise RuntimeError("No prepared feature markets found for batch execution")
            self._append_error(
                error_events,
                stage="features",
                reason="empty_prepared_feature_markets",
                severity="warning",
            )
            return self._build_empty_run_result(
                cfg=cfg,
                start=start,
                end=end,
                market_events=pd.DataFrame(),
                features=pd.DataFrame(),
                resolution_frame=resolution_frame,
                resolution_diagnostics=resolution_diagnostics,
                data_quality_report=data_quality_report,
                feature_health=feature_health,
                error_events=error_events,
                cache_key=None,
            )

        total_batches = ceil(len(selected_market_ids) / market_batch_size)
        batch_progress = self._make_progress_bar(
            enabled=cfg.enable_progress_bars,
            total=total_batches,
            desc="backtest: market batches",
            unit="batch",
        )

        strategy_signal_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in strategy_map}
        strategy_states: dict[str, object | None] = dict.fromkeys(strategy_map)

        order_events: list[dict[str, object]] = []
        trade_frames: list[pd.DataFrame] = []
        retain_features = cfg.retain_full_feature_frames
        retain_signals = cfg.retain_strategy_signals
        feature_batches: list[pd.DataFrame] = []
        feature_health_batches: list[pd.DataFrame] = []
        retained_feature_rows = 0
        observed_feature_rows = 0
        observed_feature_start: pd.Timestamp | None = None
        observed_feature_end: pd.Timestamp | None = None

        regime_labels_by_ts: pd.Series | None = None
        regime_ts_index: pd.DatetimeIndex | None = None
        if regime_lookup is not None:
            regime_labels_by_ts = pd.Series(regime_lookup, dtype="object")
            regime_labels_by_ts.index = pd.to_datetime(
                regime_labels_by_ts.index,
                utc=True,
                errors="coerce",
            )
            regime_labels_by_ts = regime_labels_by_ts[regime_labels_by_ts.index.notna()]
            if not regime_labels_by_ts.empty:
                regime_labels_by_ts = regime_labels_by_ts[
                    ~regime_labels_by_ts.index.duplicated(keep="last")
                ].sort_index()
                regime_ts_index = pd.DatetimeIndex(regime_labels_by_ts.index)

        for batch_idx in range(total_batches):
            start_idx = batch_idx * market_batch_size
            end_idx = min(start_idx + market_batch_size, len(selected_market_ids))
            market_batch_ids = selected_market_ids[start_idx:end_idx]
            market_batch_set = set(market_batch_ids)

            batch_features = self.load_prepared_features(
                start=start,
                end=end,
                limit_files=limit_files,
                features_manifest_path=prepared_manifest_path,
                recursive_scan=True,
                market_ids=market_batch_set,
            )
            if batch_features.empty:
                if batch_progress is not None:
                    batch_progress.update(1)
                    batch_progress.set_postfix(
                        {
                            "markets": f"{end_idx}/{len(selected_market_ids)}",
                            "rows": 0,
                        },
                        refresh=False,
                    )
                continue

            batch_market_events = self._materialize_market_events_from_features(batch_features)
            gate_result = self._apply_feature_quality_gates(
                market_events=batch_market_events,
                features=batch_features,
                resolution_frame=resolution_frame,
                cfg=cfg,
                error_events=error_events,
            )
            batch_market_events, batch_features, batch_feature_health = gate_result

            if not batch_feature_health.empty:
                enriched_health = batch_feature_health.copy()
                enriched_health["batch_index"] = batch_idx
                feature_health_batches.append(enriched_health)

            if batch_features.empty:
                if batch_progress is not None:
                    batch_progress.update(1)
                    batch_progress.set_postfix(
                        {
                            "markets": f"{end_idx}/{len(selected_market_ids)}",
                            "rows": 0,
                        },
                        refresh=False,
                    )
                continue

            observed_feature_rows += len(batch_features)
            batch_feature_ts = pd.to_datetime(batch_features.index, utc=True, errors="coerce")
            if int(pd.Series(batch_feature_ts).notna().sum()) > 0:
                batch_start = batch_feature_ts.min()
                batch_end = batch_feature_ts.max()
                if observed_feature_start is None or batch_start < observed_feature_start:
                    observed_feature_start = batch_start
                if observed_feature_end is None or batch_end > observed_feature_end:
                    observed_feature_end = batch_end

            if retain_features:
                feature_batches.append(batch_features)
                retained_feature_rows += len(batch_features)

            # Filter resolution frame to markets in current batch
            batch_resolution = resolution_frame[
                resolution_frame["market_id"].astype(str).isin(market_batch_set)
            ]

            batch_features_with_timing = batch_features.copy()
            event_ts_values = (
                batch_features_with_timing["ts_event"]
                if "ts_event" in batch_features_with_timing.columns
                else batch_features_with_timing.index
            )
            if (
                "market_id" in batch_features_with_timing.columns
                and "resolved_at" in batch_resolution.columns
            ):
                entry_ts_series = pd.Series(
                    pd.to_datetime(event_ts_values, utc=True, errors="coerce"),
                    index=batch_features_with_timing.index,
                )
                entry_day = entry_ts_series.dt.floor("D")
                market_ids = batch_features_with_timing["market_id"].astype(str)

                resolution_rows = batch_resolution[["market_id", "resolved_at"]].copy()
                resolution_rows["market_id"] = resolution_rows["market_id"].astype(str)
                resolution_rows["resolved_at"] = pd.to_datetime(
                    resolution_rows["resolved_at"],
                    utc=True,
                    errors="coerce",
                )
                resolution_rows["feature_day"] = resolution_rows["resolved_at"].dt.floor("D")

                if "feature_date" in batch_resolution.columns:
                    feature_day_override = pd.to_datetime(
                        batch_resolution["feature_date"],
                        utc=True,
                        errors="coerce",
                    ).dt.floor("D")
                    resolution_rows["feature_day"] = feature_day_override.fillna(
                        resolution_rows["feature_day"]
                    )

                resolution_rows = resolution_rows.dropna(
                    subset=["market_id", "resolved_at", "feature_day"]
                )

                resolution_lookup = (
                    resolution_rows.sort_values("resolved_at")
                    .drop_duplicates(["market_id", "feature_day"], keep="first")
                    [["market_id", "feature_day", "resolved_at"]]
                )

                fallback_counts = resolution_rows.groupby("market_id", observed=True)["resolved_at"].nunique()
                fallback_market_ids = fallback_counts[fallback_counts == 1].index
                market_fallback_lookup = (
                    resolution_rows[resolution_rows["market_id"].isin(fallback_market_ids)]
                    .groupby("market_id", observed=True)["resolved_at"]
                    .min()
                )

                feature_lookup = pd.DataFrame(
                    {
                        "market_id": market_ids,
                        "feature_day": entry_day,
                    },
                    index=batch_features_with_timing.index,
                )
                feature_lookup = feature_lookup.merge(
                    resolution_lookup,
                    on=["market_id", "feature_day"],
                    how="left",
                )

                resolved_at_series = feature_lookup["resolved_at"]
                if not market_fallback_lookup.empty:
                    resolved_at_series = resolved_at_series.fillna(
                        feature_lookup["market_id"].map(market_fallback_lookup)
                    )
                resolved_at_series.index = batch_features_with_timing.index

                time_delta = resolved_at_series - entry_ts_series
                batch_features_with_timing["time_to_resolution_secs"] = (
                    time_delta.dt.total_seconds()
                )

            # Add regime column if regime lookup available.
            if regime_labels_by_ts is not None and regime_ts_index is not None:
                entry_ts = pd.to_datetime(
                    event_ts_values,
                    utc=True,
                    errors="coerce",
                )
                entry_index = pd.DatetimeIndex(entry_ts)
                nearest_positions = regime_ts_index.get_indexer(
                    entry_index,
                    method="nearest",
                    tolerance=pd.Timedelta(minutes=5),
                )
                regimes = pd.Series(
                    data=[None] * len(batch_features_with_timing),
                    index=batch_features_with_timing.index,
                    dtype="object",
                )
                valid_match_mask = nearest_positions >= 0
                if valid_match_mask.any():
                    matched_labels = regime_labels_by_ts.take(
                        nearest_positions[valid_match_mask]
                    )
                    valid_positions = pd.Index(valid_match_mask).to_numpy().nonzero()[0]
                    regimes.iloc[valid_positions] = matched_labels.to_numpy()
                batch_features_with_timing["regime"] = regimes

            for strategy_name, strategy_callable in strategy_map.items():
                strategy_output = strategy_callable(batch_features_with_timing.copy())
                signal_frame = normalize_strategy_output(
                    features=batch_features_with_timing,
                    strategy_name=strategy_name,
                    strategy_output=strategy_output,
                    signal_column="signal",
                )
                if retain_signals:
                    strategy_signal_parts[strategy_name].append(signal_frame)

                sim_result = self.simulate_hold_to_resolution_backtest_incremental(
                    signal_frame,
                    batch_resolution,
                    strategy_name=strategy_name,
                    signal_column="signal",
                    config=cfg,
                    risk_events=error_events,
                    order_events=order_events,
                    continuation_state=strategy_states[strategy_name],
                    finalize=batch_idx == total_batches - 1,
                )
                batch_trades, strategy_state = sim_result
                strategy_states[strategy_name] = strategy_state
                if not batch_trades.empty:
                    trade_frames.append(batch_trades)

            if batch_progress is not None:
                batch_progress.update(1)
                batch_progress.set_postfix(
                    {
                        "markets": f"{end_idx}/{len(selected_market_ids)}",
                        "rows": len(batch_features),
                    },
                    refresh=False,
                )

        if batch_progress is not None:
            batch_progress.close()

        features = pd.DataFrame()
        if retain_features and feature_batches:
            features = pd.concat(feature_batches, ignore_index=False)
            features = features.sort_index()
        market_events = (
            self._materialize_market_events_from_features(features)
            if cfg.retain_market_events
            else pd.DataFrame()
        )
        if feature_health_batches:
            feature_health = pd.concat(feature_health_batches, ignore_index=True)

        if observed_feature_rows == 0:
            if cfg.mode == "strict":
                raise RuntimeError("Feature quality gates removed all feature rows")

            self._append_error(
                error_events,
                stage="feature_quality",
                reason="features_empty_after_batch_filtering",
                severity="warning",
            )
            return self._build_empty_run_result(
                cfg=cfg,
                start=start,
                end=end,
                market_events=market_events,
                features=features,
                resolution_frame=resolution_frame,
                resolution_diagnostics=resolution_diagnostics,
                data_quality_report=data_quality_report,
                feature_health=feature_health,
                error_events=error_events,
                cache_key=None,
            )

        effective_start = start
        effective_end = end

        cache_signature_features = features
        if not retain_features:
            cache_signature_features = pd.DataFrame(
                {
                    "rows": [observed_feature_rows],
                    "retained_rows": [retained_feature_rows],
                }
            )
        metadata = self._build_run_metadata(
            cfg,
            start=effective_start,
            end=effective_end,
            cache_signature=self._compute_cache_signature(cache_signature_features, None, cfg=cfg),
        )

        strategy_signals: dict[str, pd.DataFrame] = {}
        if retain_signals:
            for strategy_name, parts in strategy_signal_parts.items():
                signal_frame = pd.concat(parts, ignore_index=False) if parts else pd.DataFrame()
                strategy_signals[strategy_name] = self._attach_run_id(
                    signal_frame, metadata.run_id
                )
        else:
            for strategy_name in strategy_map:
                strategy_signals[strategy_name] = self._attach_run_id(
                    pd.DataFrame(), metadata.run_id
                )

        order_ledger = pd.DataFrame(order_events)
        if not order_ledger.empty:
            order_ledger = order_ledger.sort_values(
                ["ts_event", "strategy", "market_id", "event_seq"]
            ).reset_index(drop=True)

        trade_ledger = (
            pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
        )
        if not trade_ledger.empty:
            trade_ledger = trade_ledger.sort_values(
                ["resolved_at", "strategy", "market_id", "token_id"]
            ).reset_index(drop=True)

        if cfg.order_lifecycle_enabled:
            reconciliation_issues = self._reconcile_order_trade_ledgers(order_ledger, trade_ledger)
            if reconciliation_issues:
                details = " | ".join(reconciliation_issues)
                if cfg.mode == "strict":
                    msg = f"Order/trade reconciliation failed: {details}"
                    raise ValueError(msg)
                self._append_error(
                    error_events,
                    stage="order_reconciliation",
                    reason="order_trade_reconciliation_failed",
                    severity="warning",
                    details=details,
                )

        order_ledger = self._attach_run_id(order_ledger, metadata.run_id)
        trade_ledger = self._attach_run_id(trade_ledger, metadata.run_id)
        diagnostics_by_market = self._attach_run_id(
            build_market_diagnostics(trade_ledger),
            metadata.run_id,
        )
        diagnostics_by_regime = self._attach_run_id(
            build_regime_diagnostics(trade_ledger),
            metadata.run_id,
        )
        settlement_repair_audit = self._attach_run_id(
            pd.DataFrame(self._last_resolution_repair_audit),
            metadata.run_id,
        )
        resolution_diagnostics = self._attach_run_id(
            resolution_diagnostics,
            metadata.run_id,
        )

        backtest_summary = summarize_backtest(trade_ledger)
        backtest_summary = self._attach_run_id(backtest_summary, metadata.run_id)
        equity_curve = build_equity_curve(trade_ledger)
        equity_curve = self._attach_run_id(equity_curve, metadata.run_id)

        error_ledger = pd.DataFrame(error_events)
        if error_ledger.empty:
            error_ledger = pd.DataFrame(
                columns=["stage", "severity", "market_id", "reason", "details"]
            )

        if cfg.metrics_logging_enabled:
            total_trades = len(trade_ledger)
            total_net_pnl = float(trade_ledger["net_pnl"].sum()) if total_trades else 0.0
            total_cost = (
                float(trade_ledger["total_execution_cost_usdc"].sum()) if total_trades else 0.0
            )
            log_msg = (
                "Backtest batch-mode run complete: run_id=%s strategies=%d batches=%d "
                "trades=%d net_pnl=%.6f execution_cost=%.6f errors=%d elapsed=%.2fs "
                "features_observed=%d features_retained=%d"
            )
            logger.info(
                log_msg,
                metadata.run_id,
                len(strategy_map),
                total_batches,
                total_trades,
                total_net_pnl,
                total_cost,
                len(error_ledger),
                perf_counter() - run_started,
                observed_feature_rows,
                retained_feature_rows,
            )

        return BacktestRunResult(
            metadata=metadata,
            market_events=market_events,
            features=features,
            resolution_frame=resolution_frame,
            resolution_diagnostics=resolution_diagnostics,
            strategy_signals=strategy_signals,
            order_ledger=order_ledger,
            trade_ledger=trade_ledger,
            backtest_summary=backtest_summary,
            equity_curve=equity_curve,
            data_quality_report=self._attach_run_id(data_quality_report, metadata.run_id),
            feature_health=self._attach_run_id(feature_health, metadata.run_id),
            error_ledger=self._attach_run_id(error_ledger, metadata.run_id),
            settlement_repair_audit=settlement_repair_audit,
            diagnostics_by_market=diagnostics_by_market,
            diagnostics_by_regime=diagnostics_by_regime,
            cache_metadata={
                "cache_signature": self.feature_generator.cache_signature(),
                "cache_schema_version": cfg.cache_schema_version,
                "cache_computation_signature": cfg.cache_computation_signature,
            },
        )

    def run_backtest(  # noqa: PLR0913
        self,
        *,
        mapping_dir: str | Path,
        prepared_manifest_path: Path | str | None = None,
        strategies: Mapping[str, StrategyCallable] | None = None,
        strategy: StrategyCallable | None = None,
        strategy_name: str = "strategy",
        start: datetime | None = None,
        end: datetime | None = None,
        limit_files: int | None = None,
        max_rows_per_file: int | None = None,
        market_slug_prefix: str | None = None,
        use_feature_cache: bool = True,
        market_batch_size: int | None = None,
        prepared_feature_market_ids: set[str] | list[str] | None = None,
        market_events: pd.DataFrame | None = None,
        features: pd.DataFrame | None = None,
        config: BacktestConfig | Mapping[str, object] | None = None,
        regime_csv_path: str | Path | None = None,
    ) -> BacktestRunResult:
        """Run end-to-end hold-to-resolution backtest for one or more strategies."""
        del max_rows_per_file, market_slug_prefix, use_feature_cache

        cfg = coerce_backtest_config(config)
        self._last_resolution_repair_audit = []
        self._configure_feature_generator_for_run(cfg)
        error_events: list[dict[str, object]] = []

        # Load regime data if provided
        regime_lookup: dict[pd.Timestamp, str] | None = None
        if regime_csv_path is not None:
            regime_path = Path(regime_csv_path)
            if regime_path.exists():
                try:
                    regime_df = pd.read_csv(regime_path)
                    regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"])
                    regime_lookup = dict(zip(regime_df["timestamp"], regime_df["regime"]))
                    logger.info(f"Loaded regime data from {regime_csv_path}: {len(regime_lookup)} timestamps")
                except Exception as e:
                    logger.warning(f"Failed to load regime data from {regime_csv_path}: {e}")
                    regime_lookup = None
            else:
                logger.warning(f"Regime CSV path does not exist: {regime_csv_path}")

        data_quality_report = pd.DataFrame(
            columns=["row_index", "market_id", "event_type", "column", "reason"]
        )
        feature_health = pd.DataFrame(columns=["metric", "value", "threshold", "passed"])

        run_started = perf_counter()
        input_market_events = market_events
        input_features = features
        prepared_inputs_key: tuple[object, ...] | None = None
        prepared_inputs_cached: PreparedInputsCacheEntry | None = None
        if input_market_events is not None and input_features is not None:
            prepared_inputs_key = self._build_prepared_inputs_cache_key(
                market_events=input_market_events,
                features=input_features,
                mapping_dir=mapping_dir,
                cfg=cfg,
            )
            prepared_inputs_cached = self._prepared_inputs_cache.get(prepared_inputs_key)
            if prepared_inputs_cached is not None:
                self._prepared_inputs_cache.move_to_end(prepared_inputs_key)

        strategy_map: dict[str, StrategyCallable] = {}
        if strategies is not None:
            strategy_map.update(strategies)
        elif strategy is not None:
            strategy_map[strategy_name] = strategy
        else:
            raise ValueError("Provide either 'strategy' or 'strategies' to run_backtest")

        if market_events is not None or features is not None:
            msg = (
                "Legacy in-memory backtest inputs are no longer supported. "
                "Use prepared_manifest_path with batched execution instead."
            )
            raise ValueError(msg)

        effective_market_batch_size = (
            int(market_batch_size) if market_batch_size is not None else 100
        )
        if effective_market_batch_size <= 0:
            msg = "market_batch_size must be greater than 0"
            raise ValueError(msg)

        return self._run_backtest_in_market_batches(
            cfg=cfg,
            strategy_map=strategy_map,
            mapping_dir=mapping_dir,
            prepared_manifest_path=prepared_manifest_path,
            start=start,
            end=end,
            limit_files=limit_files,
            market_batch_size=effective_market_batch_size,
            prepared_feature_market_ids=prepared_feature_market_ids,
            error_events=error_events,
            data_quality_report=data_quality_report,
            feature_health=feature_health,
            run_started=run_started,
            regime_lookup=regime_lookup,
        )

    def write_run_artifact_package(
        self,
        result: BacktestRunResult,
        *,
        output_dir: Path | str,
        artifact_version: str = "v1",
    ) -> Path:
        """Write a self-contained, versioned artifact package for one run."""
        root = Path(output_dir)
        package_dir = root / artifact_version / result.metadata.run_id
        package_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = package_dir / "metadata.json"
        config_snapshot_path = package_dir / "config_snapshot.json"
        cache_metadata_path = package_dir / "cache_metadata.json"
        plots_manifest_path = package_dir / "plots_manifest.json"
        manifest_path = package_dir / "artifact_manifest.json"

        metadata_payload = asdict(result.metadata)
        metadata_path.write_text(
            json.dumps(metadata_payload, sort_keys=True, default=str, indent=2),
            encoding="utf-8",
        )
        config_snapshot_path.write_text(
            json.dumps(result.metadata.config_snapshot, sort_keys=True, default=str, indent=2),
            encoding="utf-8",
        )
        cache_metadata_path.write_text(
            json.dumps(result.cache_metadata, sort_keys=True, default=str, indent=2),
            encoding="utf-8",
        )
        plots_manifest_path.write_text(
            json.dumps({"plots": []}, sort_keys=True, default=str, indent=2),
            encoding="utf-8",
        )

        frames: dict[str, pd.DataFrame] = {
            "trade_ledger": result.trade_ledger,
            "order_ledger": result.order_ledger,
            "backtest_summary": result.backtest_summary,
            "equity_curve": result.equity_curve,
            "resolution_diagnostics": result.resolution_diagnostics,
            "data_quality_report": result.data_quality_report,
            "feature_health": result.feature_health,
            "error_ledger": result.error_ledger,
            "settlement_repair_audit": result.settlement_repair_audit,
            "diagnostics_by_market": result.diagnostics_by_market,
            "diagnostics_by_regime": result.diagnostics_by_regime,
        }
        for name, frame in frames.items():
            (package_dir / f"{name}.csv").write_text(
                frame.to_csv(index=False),
                encoding="utf-8",
            )

        manifest_payload = {
            "artifact_version": artifact_version,
            "run_id": result.metadata.run_id,
            "created_at": datetime.now(tz=UTC).isoformat(),
            "files": sorted(
                [
                    path.name
                    for path in package_dir.glob("*")
                    if path.name != "artifact_manifest.json"
                ]
            ),
            "row_counts": {name: len(frame) for name, frame in frames.items()},
        }
        manifest_path.write_text(
            json.dumps(manifest_payload, sort_keys=True, default=str, indent=2),
            encoding="utf-8",
        )
        result.artifact_manifest = manifest_payload
        return package_dir

    @staticmethod
    def _build_parameter_sweep_grid(
        parameter_sweeps: Mapping[str, list[object]] | None,
    ) -> list[dict[str, object]]:
        if not parameter_sweeps:
            return [{}]

        keys = sorted(parameter_sweeps.keys())
        values = [parameter_sweeps[key] for key in keys]
        grid = [dict(zip(keys, combo, strict=False)) for combo in product(*values)]
        return grid

    def _apply_stress_scenario(
        self,
        *,
        scenario_id: str,
        market_events: pd.DataFrame,
        features: pd.DataFrame,
        config: BacktestConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame, BacktestConfig]:
        stressed_events = market_events.copy()
        stressed_features = features.copy()
        stressed_cfg = config

        if scenario_id == "low_liquidity":
            for col in ("ask_depth_1", "ask_depth_5", "bid_depth_1", "bid_depth_5"):
                if col in stressed_features.columns:
                    stressed_features[col] = (
                        pd.to_numeric(
                            stressed_features[col],
                            errors="coerce",
                        )
                        * 0.2
                    )
        elif scenario_id == "wide_spread":
            if "spread" in stressed_features.columns:
                stressed_features["spread"] = (
                    pd.to_numeric(
                        stressed_features["spread"],
                        errors="coerce",
                    )
                    * 2.0
                )
        elif scenario_id == "fee_increase":
            stressed_cfg = replace(stressed_cfg, fee_rate=stressed_cfg.fee_rate * 1.5)
        elif scenario_id == "settlement_delay":
            events_reset = stressed_events.reset_index()
            if "event_type" in events_reset.columns and "ts_event" in events_reset.columns:
                mask = events_reset["event_type"].astype(str) == "market_resolved"
                events_reset.loc[mask, "ts_event"] = pd.to_datetime(
                    events_reset.loc[mask, "ts_event"],
                    utc=True,
                    errors="coerce",
                ) + pd.to_timedelta(1, unit="h")
                stressed_events = events_reset.sort_values("ts_event").set_index("ts_event")
        elif scenario_id != "baseline":
            msg = f"Unsupported stress scenario: {scenario_id}"
            raise ValueError(msg)

        return stressed_events, stressed_features, stressed_cfg

    def run_sensitivity_scenarios(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        *,
        mapping_dir: str | Path,
        prepared_manifest_path: Path | str | None = None,
        strategies: Mapping[str, StrategyCallable] | None = None,
        strategy: StrategyCallable | None = None,
        strategy_name: str = "strategy",
        start: datetime | None = None,
        end: datetime | None = None,
        limit_files: int | None = None,
        max_rows_per_file: int | None = None,
        market_slug_prefix: str | None = None,
        use_feature_cache: bool = True,
        market_batch_size: int | None = None,
        prepared_feature_market_ids: set[str] | list[str] | None = None,
        market_events: pd.DataFrame | None = None,
        features: pd.DataFrame | None = None,
        base_config: BacktestConfig | Mapping[str, object] | None = None,
        parameter_sweeps: Mapping[str, list[object]] | None = None,
        stress_scenarios: list[str] | None = None,
        parallel_workers: int = 1,
    ) -> pd.DataFrame:
        """Run reproducible parameter sweep and stress scenarios for robustness checks."""
        base_cfg = coerce_backtest_config(base_config)
        if market_events is not None or features is not None:
            msg = (
                "Legacy in-memory sensitivity inputs are no longer supported. "
                "Use prepared_manifest_path with batched execution instead."
            )
            raise ValueError(msg)

        sweep_grid = self._build_parameter_sweep_grid(parameter_sweeps)
        scenarios = stress_scenarios or [
            "baseline",
            "low_liquidity",
            "wide_spread",
            "fee_increase",
            "settlement_delay",
        ]

        total_scenario_runs = len(sweep_grid) * len(scenarios)
        scenario_progress = self._make_progress_bar(
            enabled=base_cfg.enable_progress_bars,
            total=total_scenario_runs,
            desc="sensitivity: scenarios",
            unit="run",
        )
        scenario_started = perf_counter()
        scenario_count = 0
        if base_cfg.metrics_logging_enabled:
            logger.info(
                "Sensitivity run started: parameter_sets=%d scenarios=%d total_runs=%d",
                len(sweep_grid),
                len(scenarios),
                total_scenario_runs,
            )

        rows: list[dict[str, object]] = []
        execution_tasks: list[tuple[int, str, int, dict[str, object]]] = []
        task_id = 0
        for scenario_id in scenarios:
            for set_index, parameter_set in enumerate(sweep_grid):
                execution_tasks.append(
                    (
                        task_id,
                        scenario_id,
                        set_index,
                        dict(parameter_set),
                    )
                )
                task_id += 1

        worker_count = min(max(int(parallel_workers), 1), len(execution_tasks))

        def _execute_task(
            task: tuple[int, str, int, dict[str, object]],
        ) -> tuple[int, str, int, dict[str, object]]:
            (
                task_index,
                scenario_id,
                set_index,
                parameter_set,
            ) = task
            cfg_payload = asdict(base_cfg)
            cfg_payload.update(parameter_set)
            sweep_cfg = coerce_backtest_config(cfg_payload)
            scenario_cfg = (
                replace(sweep_cfg, fee_rate=sweep_cfg.fee_rate * 1.5)
                if scenario_id == "fee_increase"
                else sweep_cfg
            )
            nested_cfg = (
                replace(scenario_cfg, enable_progress_bars=False)
                if base_cfg.enable_progress_bars
                else scenario_cfg
            )

            if worker_count > 1:
                worker_runner = self.__class__(self.storage_path, cache_dir=self.cache_path)
            else:
                worker_runner = self

            run_result = worker_runner.run_backtest(
                mapping_dir=mapping_dir,
                prepared_manifest_path=prepared_manifest_path,
                strategies=strategies,
                strategy=strategy,
                strategy_name=strategy_name,
                start=start,
                end=end,
                limit_files=limit_files,
                max_rows_per_file=max_rows_per_file,
                market_slug_prefix=market_slug_prefix,
                use_feature_cache=use_feature_cache,
                market_batch_size=market_batch_size,
                prepared_feature_market_ids=prepared_feature_market_ids,
                config=nested_cfg,
            )

            scenario_label = f"{scenario_id}__set_{set_index:03d}"
            row = {
                "scenario_id": scenario_label,
                "stress_scenario": scenario_id,
                "parameter_set": json.dumps(parameter_set, sort_keys=True),
                "trades": len(run_result.trade_ledger),
                "net_pnl": float(
                    run_result.trade_ledger["net_pnl"].sum()
                    if not run_result.trade_ledger.empty
                    else 0.0
                ),
                "gross_pnl": float(
                    run_result.trade_ledger["gross_pnl"].sum()
                    if not run_result.trade_ledger.empty
                    else 0.0
                ),
                "total_cost": float(
                    run_result.trade_ledger["total_execution_cost_usdc"].sum()
                    if not run_result.trade_ledger.empty
                    else 0.0
                ),
                "run_id": run_result.metadata.run_id,
            }
            return task_index, scenario_id, set_index, row

        ordered_rows: dict[int, dict[str, object]] = {}
        if worker_count <= 1:
            for task in execution_tasks:
                task_index, scenario_id, set_index, row = _execute_task(task)
                ordered_rows[task_index] = row
                scenario_count += 1
                if scenario_progress is not None:
                    scenario_progress.update(1)
                    scenario_progress.set_postfix(
                        {
                            "scenario": scenario_id,
                            "set": set_index,
                        },
                        refresh=False,
                    )
                if base_cfg.metrics_logging_enabled and (
                    scenario_count % 10 == 0 or scenario_count == total_scenario_runs
                ):
                    logger.info(
                        "Sensitivity progress: %d/%d runs complete",
                        scenario_count,
                        total_scenario_runs,
                    )
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(_execute_task, task): (task[1], task[2])
                    for task in execution_tasks
                }
                for future in as_completed(futures):
                    task_index, scenario_id, set_index, row = future.result()
                    ordered_rows[task_index] = row
                    scenario_count += 1
                    if scenario_progress is not None:
                        scenario_progress.update(1)
                        scenario_progress.set_postfix(
                            {
                                "scenario": scenario_id,
                                "set": set_index,
                            },
                            refresh=False,
                        )
                    if base_cfg.metrics_logging_enabled and (
                        scenario_count % 10 == 0 or scenario_count == total_scenario_runs
                    ):
                        logger.info(
                            "Sensitivity progress: %d/%d runs complete",
                            scenario_count,
                            total_scenario_runs,
                        )

        rows.extend(ordered_rows[idx] for idx in sorted(ordered_rows))

        if scenario_progress is not None:
            scenario_progress.close()

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame

        frame = frame.sort_values(["scenario_id"]).reset_index(drop=True)
        frame["robustness_rank"] = (
            frame["net_pnl"].rank(method="dense", ascending=False).astype(int)
        )

        if base_cfg.metrics_logging_enabled:
            logger.info(
                "Sensitivity run complete: rows=%d best_net_pnl=%.6f elapsed=%.2fs",
                len(frame),
                float(frame["net_pnl"].max()) if not frame.empty else 0.0,
                perf_counter() - scenario_started,
            )

        return frame

    def label_market_outcomes(self, market_events: pd.DataFrame) -> pd.DataFrame:
        """Label market outcomes based on resolution events.

        Parameters
        ----------
        market_events : pd.DataFrame
            Market events dataframe.

        Returns
        -------
        pd.DataFrame
            Labeled dataframe with 'outcome' column indicating contract resolution.
        """
        resolved = market_events[market_events["event_type"] == "market_resolved"]

        labels = []
        for idx, row in resolved.iterrows():
            data = row.get("data", {})
            if isinstance(data, dict):
                outcome = data.get("winning_outcome")
                winning_asset = data.get("winning_asset_id")
                question = data.get("question") or "Unknown"
                logger.debug(
                    "Market resolved: %s, Outcome: %s, Asset: %s",
                    question[:50] if question else "Unknown",
                    outcome,
                    winning_asset,
                )
            else:
                outcome = None
            labels.append(
                {"market_id": row.get("market_id"), "resolved_at": idx, "outcome": outcome}
            )

        labels_df = pd.DataFrame(labels)
        logger.info("Extracted %d market resolution labels", len(labels_df))
        if not labels_df.empty:
            logger.info("Outcome distribution: %s", labels_df["outcome"].value_counts().to_dict())
        return labels_df
