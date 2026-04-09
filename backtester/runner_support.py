from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, replace
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

from backtester.feature_generator import FeatureGenerator, FeatureGeneratorConfig
from backtester.simulation.analytics import build_equity_curve, summarize_backtest
from backtester.simulation.risk_engine import RiskEvaluator, RiskLimits

if TYPE_CHECKING:
    from collections.abc import Mapping

    from backtester.config.types import BacktestConfig, FeatureGatePolicy, ValidationPolicy

logger = logging.getLogger(__name__)

_MIN_VOLATILITY_POINTS = 3

_ORDER_TERMINAL_STATES = {"filled", "cancelled", "expired", "rejected"}
_ORDER_ALLOWED_TRANSITIONS: dict[str | None, set[str]] = {
    None: {"submitted"},
    "submitted": {"partial", "filled", "cancelled", "expired", "rejected"},
    "partial": {"partial", "filled", "cancelled", "expired", "rejected"},
    "filled": set(),
    "cancelled": set(),
    "expired": set(),
    "rejected": set(),
}


class FillExecutionResult(TypedDict):
    """Shape of fill model execution output."""

    filled_qty: float
    avg_fill_price: float
    slippage_bps: float
    reject_reason: str | None
    order_state: str


class BacktestSupportOps:
    """Execution, quality-gate, and metadata helper methods."""

    @staticmethod
    def summarize_backtest(trades: pd.DataFrame) -> pd.DataFrame:
        """Summarize strategy-level backtest metrics from closed trades."""
        return summarize_backtest(trades)

    @staticmethod
    def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
        """Build cumulative net PnL curve per strategy."""
        return build_equity_curve(trades)

    @staticmethod
    def _hash_payload(payload: Mapping[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _resolve_git_commit() -> str | None:
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],  # noqa: S607
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None

        if proc.returncode != 0:
            return None

        commit = proc.stdout.strip()
        return commit or None

    @staticmethod
    def _attach_run_id(frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
        frame = frame.copy()
        frame["run_id"] = run_id
        return frame

    @staticmethod
    def _append_error(  # noqa: PLR0913
        error_events: list[dict[str, object]],
        *,
        stage: str,
        reason: str,
        severity: str = "error",
        market_id: str | None = None,
        details: str | None = None,
    ) -> None:
        error_events.append(
            {
                "stage": stage,
                "severity": severity,
                "market_id": market_id,
                "reason": reason,
                "details": details,
            }
        )

    def _configure_feature_generator_for_run(self, cfg: BacktestConfig) -> None:
        current_config = self.feature_generator.config
        self.feature_generator = FeatureGenerator(
            FeatureGeneratorConfig(
                cache_dir=current_config.cache_dir,
                cache_schema_version=cfg.cache_schema_version,
                cache_computation_signature=cfg.cache_computation_signature,
                book_state_max_age=current_config.book_state_max_age,
            )
        )

    @staticmethod
    def _policy_cache_key(
        policy: ValidationPolicy | FeatureGatePolicy,
    ) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(asdict(policy).items()))

    @staticmethod
    def _frame_cache_signature(frame: pd.DataFrame) -> tuple[object, ...]:
        if frame.empty:
            return (0, tuple(frame.columns), None, None)
        sample = pd.concat([frame.head(8), frame.tail(8)], axis=0)
        sample_json = sample.astype(str).to_json(orient="split", date_format="iso")
        sample_hash = hashlib.sha256(sample_json.encode("utf-8")).hexdigest()[:16]
        return (
            len(frame),
            tuple(frame.columns),
            str(frame.index.min()),
            str(frame.index.max()),
            sample_hash,
        )

    def _build_prepared_inputs_cache_key(
        self,
        *,
        market_events: pd.DataFrame,
        features: pd.DataFrame,
        mapping_dir: str | Path,
        cfg: BacktestConfig,
    ) -> tuple[object, ...]:
        return (
            self._frame_cache_signature(market_events),
            self._frame_cache_signature(features),
            str(Path(mapping_dir).resolve()),
            cfg.mode,
            cfg.confidence_threshold,
            cfg.resolution_repair_dry_run,
            self._policy_cache_key(cfg.validation_policy),
            self._policy_cache_key(cfg.feature_gate_policy),
        )

    def _validate_market_events_with_policy(
        self,
        market_events: pd.DataFrame,
        cfg: BacktestConfig,
        *,
        error_events: list[dict[str, object]],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        validation_policy = cfg.validation_policy
        if cfg.mode == "tolerant":
            validation_policy = replace(validation_policy, quarantine_invalid_rows=True)

        validated_events, report = validate_market_events_rows(
            market_events,
            policy=validation_policy,
        )
        if not report.empty:
            reason_counts = report["reason"].value_counts().to_dict()
            self._append_error(
                error_events,
                stage="schema_validation",
                reason="invalid_rows_detected",
                severity="warning" if cfg.mode == "tolerant" else "error",
                details=str(reason_counts),
            )
        return validated_events, report

    def _trim_rows_after_resolution(
        self,
        frame: pd.DataFrame,
        *,
        resolution_frame: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int, int]:
        """Trim per-market rows that occur after resolved_at timestamps."""
        if frame.empty or resolution_frame.empty or "market_id" not in frame.columns:
            return frame, 0, 0
        if "resolved_at" not in resolution_frame.columns:
            return frame, 0, 0

        resolution_rows = resolution_frame.copy(deep=False)
        if "market_id" not in resolution_rows.columns and resolution_rows.index.name == "market_id":
            resolution_rows = resolution_rows.reset_index()
        if "market_id" not in resolution_rows.columns:
            return frame, 0, 0

        resolution_rows = resolution_rows.copy()
        resolution_rows["market_id"] = resolution_rows["market_id"].astype(str)
        resolution_rows["resolved_at"] = pd.to_datetime(
            resolution_rows["resolved_at"],
            utc=True,
            errors="coerce",
        )
        resolved_day = resolution_rows["resolved_at"].dt.floor("D")
        if "feature_date" in resolution_rows.columns:
            feature_day = pd.to_datetime(
                resolution_rows["feature_date"],
                utc=True,
                errors="coerce",
            ).dt.floor("D")
            resolution_rows["feature_day"] = feature_day.fillna(resolved_day)
        else:
            resolution_rows["feature_day"] = resolved_day

        resolution_rows = resolution_rows.dropna(subset=["market_id", "resolved_at", "feature_day"])
        if resolution_rows.empty:
            return frame, 0, 0

        resolution_lookup = (
            resolution_rows.sort_values("resolved_at")
            .drop_duplicates(["market_id", "feature_day"], keep="first")
            [["market_id", "feature_day", "resolved_at"]]
        )

        fallback_candidates = resolution_rows.groupby("market_id", observed=True)["resolved_at"].nunique()
        fallback_market_ids = fallback_candidates[fallback_candidates == 1].index
        market_fallback_lookup = (
            resolution_rows[resolution_rows["market_id"].isin(fallback_market_ids)]
            .groupby("market_id", observed=True)["resolved_at"]
            .min()
        )

        original_index_name = frame.index.name
        rows = frame.reset_index()
        ts_col = rows.columns[0]
        rows["market_id"] = rows["market_id"].astype(str)
        rows["_ts_event"] = pd.to_datetime(rows[ts_col], utc=True, errors="coerce")
        rows["_feature_day"] = rows["_ts_event"].dt.floor("D")

        rows = rows.merge(
            resolution_lookup.rename(columns={"resolved_at": "_resolved_at"}),
            left_on=["market_id", "_feature_day"],
            right_on=["market_id", "feature_day"],
            how="left",
        )
        rows = rows.drop(columns=["feature_day"], errors="ignore")

        if not market_fallback_lookup.empty:
            fallback_resolved = rows["market_id"].map(market_fallback_lookup)
            rows["_resolved_at"] = rows["_resolved_at"].fillna(fallback_resolved)

        keep_mask = (
            rows["_resolved_at"].isna()
            | rows["_ts_event"].isna()
            | (rows["_ts_event"] <= rows["_resolved_at"])
        )
        trimmed_rows = int((~keep_mask).sum())
        if trimmed_rows == 0:
            return frame, 0, 0

        trimmed_markets = int(rows.loc[~keep_mask, "market_id"].nunique())
        trimmed = rows.loc[keep_mask].drop(columns=["_ts_event", "_feature_day", "_resolved_at"])
        trimmed = trimmed.set_index(ts_col)
        trimmed.index.name = original_index_name
        return trimmed.sort_index(), trimmed_rows, trimmed_markets

    def _evaluate_feature_health(  # noqa: C901
        self,
        features: pd.DataFrame,
        resolution_frame: pd.DataFrame,
        *,
        policy: FeatureGatePolicy,
    ) -> tuple[pd.DataFrame, set[str], list[str]]:
        rows: list[dict[str, object]] = []
        blocking_markets: set[str] = set()
        failures: list[str] = []

        if features.empty:
            rows.append(
                {
                    "metric": "feature_rows",
                    "value": 0,
                    "threshold": ">0",
                    "passed": False,
                }
            )
            failures.append("feature_rows_empty")
            return pd.DataFrame(rows), blocking_markets, failures

        for col in ("mid_price", "spread", "imbalance_1", "bid_depth_1", "ask_depth_1"):
            if col not in features.columns:
                continue
            null_fraction = float(features[col].isna().mean())
            passed = null_fraction <= policy.null_fraction_max
            rows.append(
                {
                    "metric": f"null_fraction:{col}",
                    "value": null_fraction,
                    "threshold": policy.null_fraction_max,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(f"null_fraction_exceeded:{col}")

        if "mid_price" in features.columns:
            invalid_mid_price = ~features["mid_price"].between(
                policy.mid_price_min,
                policy.mid_price_max,
            )
            invalid_count = int(invalid_mid_price.sum())
            rows.append(
                {
                    "metric": "bounds:mid_price",
                    "value": invalid_count,
                    "threshold": 0,
                    "passed": invalid_count == 0,
                }
            )
            if invalid_count:
                failures.append("mid_price_out_of_bounds")

        if "spread" in features.columns:
            invalid_spread = ~features["spread"].between(policy.spread_min, policy.spread_max)
            invalid_count = int(invalid_spread.sum())
            rows.append(
                {
                    "metric": "bounds:spread",
                    "value": invalid_count,
                    "threshold": 0,
                    "passed": invalid_count == 0,
                }
            )
            if invalid_count:
                failures.append("spread_out_of_bounds")

        leakage_count = 0
        if (
            policy.block_on_post_resolution_features
            and not resolution_frame.empty
            and "market_id" in features.columns
        ):
            resolution_rows = resolution_frame.copy(deep=False)
            if "market_id" not in resolution_rows.columns and resolution_rows.index.name == "market_id":
                resolution_rows = resolution_rows.reset_index()

            if "market_id" in resolution_rows.columns:
                resolution_rows = resolution_rows.copy()
                resolution_rows["market_id"] = resolution_rows["market_id"].astype(str)
                resolution_rows["resolved_at"] = pd.to_datetime(
                    resolution_rows["resolved_at"],
                    utc=True,
                    errors="coerce",
                )
                resolved_day = resolution_rows["resolved_at"].dt.floor("D")
                if "feature_date" in resolution_rows.columns:
                    feature_day = pd.to_datetime(
                        resolution_rows["feature_date"],
                        utc=True,
                        errors="coerce",
                    ).dt.floor("D")
                    resolution_rows["feature_day"] = feature_day.fillna(resolved_day)
                else:
                    resolution_rows["feature_day"] = resolved_day

                resolution_rows = resolution_rows.dropna(subset=["market_id", "resolved_at", "feature_day"])

                resolution_lookup = (
                    resolution_rows.sort_values("resolved_at")
                    .drop_duplicates(["market_id", "feature_day"], keep="first")
                    [["market_id", "feature_day", "resolved_at"]]
                    .rename(columns={"resolved_at": "_resolved_at"})
                )

                fallback_candidates = resolution_rows.groupby("market_id", observed=True)["resolved_at"].nunique()
                fallback_market_ids = fallback_candidates[fallback_candidates == 1].index
                market_fallback_lookup = (
                    resolution_rows[resolution_rows["market_id"].isin(fallback_market_ids)]
                    .groupby("market_id", observed=True)["resolved_at"]
                    .min()
                )

                market_ids = features["market_id"].astype(str)
                feature_ts = pd.Series(
                    pd.to_datetime(features.index, utc=True, errors="coerce"),
                    index=features.index,
                )
                feature_days = feature_ts.dt.floor("D")

                feature_keys = pd.DataFrame(
                    {
                        "market_id": market_ids,
                        "feature_day": feature_days,
                    },
                    index=features.index,
                )
                feature_keys = feature_keys.merge(
                    resolution_lookup,
                    on=["market_id", "feature_day"],
                    how="left",
                )

                resolved_at_series = feature_keys["_resolved_at"]
                if not market_fallback_lookup.empty:
                    fallback_resolved = feature_keys["market_id"].map(market_fallback_lookup)
                    resolved_at_series = resolved_at_series.fillna(fallback_resolved)
                resolved_at_series.index = features.index

                leakage_mask = (
                    resolved_at_series.notna()
                    & feature_ts.notna()
                    & (feature_ts > resolved_at_series)
                )

                if leakage_mask.any():
                    leaking_markets = market_ids.loc[leakage_mask].unique().tolist()
                    leakage_count = len(leaking_markets)
                    blocking_markets.update(leaking_markets)

        rows.append(
            {
                "metric": "post_resolution_leakage_markets",
                "value": leakage_count,
                "threshold": 0,
                "passed": leakage_count == 0,
            }
        )
        if leakage_count:
            failures.append("post_resolution_feature_leakage")

        return pd.DataFrame(rows), blocking_markets, failures

    def _apply_feature_quality_gates(  # noqa: C901
        self,
        *,
        market_events: pd.DataFrame,
        features: pd.DataFrame,
        resolution_frame: pd.DataFrame,
        cfg: BacktestConfig,
        error_events: list[dict[str, object]],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if cfg.feature_gate_policy.block_on_post_resolution_features:
            market_events, trimmed_event_rows, trimmed_event_markets = (
                self._trim_rows_after_resolution(
                    market_events,
                    resolution_frame=resolution_frame,
                )
            )
            features, trimmed_feature_rows, trimmed_feature_markets = (
                self._trim_rows_after_resolution(
                    features,
                    resolution_frame=resolution_frame,
                )
            )
            if trimmed_event_rows or trimmed_feature_rows:
                logger.info(
                    "Trimmed post-resolution rows before feature gates: "
                    "events=%d across %d markets, features=%d across %d markets",
                    trimmed_event_rows,
                    trimmed_event_markets,
                    trimmed_feature_rows,
                    trimmed_feature_markets,
                )

        feature_health, blocking_markets, failures = self._evaluate_feature_health(
            features,
            resolution_frame,
            policy=cfg.feature_gate_policy,
        )
        if not failures:
            return market_events, features, feature_health

        failure_msg = ", ".join(sorted(failures))
        if cfg.mode == "strict":
            msg = f"Feature quality gates failed: {failure_msg}"
            raise ValueError(msg)

        self._append_error(
            error_events,
            stage="feature_quality",
            reason="feature_gates_failed_tolerant_mode",
            severity="warning",
            details=failure_msg,
        )

        filtered_features = features.copy()
        filtered_events = market_events.copy()

        # In tolerant mode, prefer row-level pruning for row-level bounds violations.
        # This preserves usable markets instead of dropping whole markets for a few bad rows.
        filtered_row_count = 0
        filtered_reasons: list[str] = []

        if "mid_price_out_of_bounds" in failures and "mid_price" in filtered_features.columns:
            invalid_mid_price = ~filtered_features["mid_price"].between(
                cfg.feature_gate_policy.mid_price_min,
                cfg.feature_gate_policy.mid_price_max,
            )
            mid_removed = int(invalid_mid_price.sum())
            if mid_removed:
                filtered_features = filtered_features.loc[~invalid_mid_price].copy()
                filtered_row_count += mid_removed
                filtered_reasons.append("mid_price_out_of_bounds")

        if "spread_out_of_bounds" in failures and "spread" in filtered_features.columns:
            invalid_spread = ~filtered_features["spread"].between(
                cfg.feature_gate_policy.spread_min,
                cfg.feature_gate_policy.spread_max,
            )
            spread_removed = int(invalid_spread.sum())
            if spread_removed:
                filtered_features = filtered_features.loc[~invalid_spread].copy()
                filtered_row_count += spread_removed
                filtered_reasons.append("spread_out_of_bounds")

        if (
            filtered_row_count
            and "market_id" in filtered_events.columns
            and not filtered_features.empty
        ):
            kept_keys = pd.MultiIndex.from_arrays(
                [
                    pd.Series(filtered_features.index).astype(str),
                    filtered_features["market_id"].astype(str),
                ]
            )
            event_keys = pd.MultiIndex.from_arrays(
                [
                    pd.Series(filtered_events.index).astype(str),
                    filtered_events["market_id"].astype(str),
                ]
            )
            keep_event_mask = event_keys.isin(kept_keys)
            filtered_events = filtered_events.loc[keep_event_mask].copy()

            self._append_error(
                error_events,
                stage="feature_quality",
                reason="feature_rows_filtered_tolerant_mode",
                severity="warning",
                details=(
                    f"rows_filtered={filtered_row_count};"
                    f"reasons={','.join(sorted(set(filtered_reasons)))}"
                ),
            )

        if not blocking_markets or "market_id" not in filtered_features.columns:
            return filtered_events, filtered_features, feature_health

        blocking_set = {str(market_id) for market_id in blocking_markets}
        filtered_features = filtered_features.loc[
            ~filtered_features["market_id"].astype(str).isin(blocking_set)
        ].copy()
        filtered_events = filtered_events.loc[
            ~filtered_events["market_id"].astype(str).isin(blocking_set)
        ].copy()

        self._append_error(
            error_events,
            stage="feature_quality",
            reason="markets_skipped",
            severity="warning",
            details=f"skipped_markets={len(blocking_set)}",
        )
        return filtered_events, filtered_features, feature_health

    def invalidate_feature_cache(self, *, cache_key: str | None = None) -> int:
        """Invalidate cached feature files and return removed count."""
        return self.feature_generator.invalidate_cache(cache_key=cache_key)

    @staticmethod
    def _estimate_market_volatility(
        market_group: pd.DataFrame,
        *,
        entry_ts: pd.Timestamp,
        lookback: int,
    ) -> float | None:
        if "mid_price" not in market_group.columns:
            return None
        history = market_group.loc[market_group.index < entry_ts, "mid_price"]
        history = pd.to_numeric(history, errors="coerce").dropna()
        if len(history) < _MIN_VOLATILITY_POINTS:
            return None
        returns = history.pct_change().dropna().tail(max(lookback, 2))
        if returns.empty:
            return None
        vol = float(returns.std())
        if not isfinite(vol) or vol <= 0:
            return None
        return vol

    def _apply_sizing_policy(  # noqa: C901, PLR0912, PLR0913
        self,
        *,
        signal_value: int,
        signal_abs: float,
        entry_price: float,
        market_group: pd.DataFrame,
        entry_ts: pd.Timestamp,
        cfg: BacktestConfig,
        gross_exposure_used: float,
        capital_remaining: float | None,
    ) -> tuple[float, str, float]:
        del signal_value

        # Size against the selected token's own execution price for both buy/sell actions.
        trade_price = entry_price
        trade_price = max(min(trade_price, 1 - 1e-9), 1e-9)
        base_notional = max(cfg.shares * trade_price, 0.0)
        rationale_parts = [f"sizing_policy={cfg.sizing_policy}"]

        policy = cfg.sizing_policy.strip().lower()
        target_notional = base_notional
        if policy == "fixed_notional":
            if cfg.sizing_fixed_notional is not None:
                target_notional = max(cfg.sizing_fixed_notional, 0.0)
                rationale_parts.append("target=fixed_notional")
            else:
                rationale_parts.append("fallback=fixed_shares")
        elif policy == "risk_budget":
            capital_base = (
                cfg.available_capital if cfg.available_capital is not None else base_notional
            )
            target_notional = max(capital_base * max(cfg.sizing_risk_budget_pct, 0.0), 0.0)
            rationale_parts.append("target=risk_budget")
        elif policy == "volatility_scaled":
            market_vol = self._estimate_market_volatility(
                market_group,
                entry_ts=entry_ts,
                lookback=cfg.sizing_volatility_lookback,
            )
            if market_vol is not None:
                scale = max(cfg.sizing_volatility_target / market_vol, 0.0)
                target_notional = base_notional * scale
                rationale_parts.append(f"target=vol_scaled:vol={market_vol:.6f}")
            else:
                rationale_parts.append("fallback=fixed_shares_no_vol")
        elif policy == "capped_kelly":
            edge = min(max(signal_abs, 0.0), 1.0) * 0.5
            kelly_fraction = min(max(2 * edge, 0.0), cfg.sizing_kelly_fraction_cap)
            # Binary-market payout asymmetry is highest near tails, so scale Kelly by midpoint liquidity.
            price_multiplier = min(max(2.0 * min(trade_price, 1.0 - trade_price), 0.0), 1.0)
            adjusted_kelly_fraction = kelly_fraction * price_multiplier
            capital_base = (
                cfg.available_capital if cfg.available_capital is not None else base_notional
            )
            target_notional = max(capital_base * adjusted_kelly_fraction, 0.0)
            rationale_parts.append(
                f"target=kelly:f={kelly_fraction:.4f}:pm={price_multiplier:.4f}:fa={adjusted_kelly_fraction:.4f}"
            )
        else:
            rationale_parts.append("target=fixed_shares_default")

        if (
            cfg.max_notional_per_market is not None
            and target_notional > cfg.max_notional_per_market
        ):
            target_notional = cfg.max_notional_per_market
            rationale_parts.append("clamped:max_notional_per_market")

        if cfg.max_gross_exposure is not None:
            remaining_gross = max(cfg.max_gross_exposure - gross_exposure_used, 0.0)
            if target_notional > remaining_gross:
                target_notional = remaining_gross
                rationale_parts.append("clamped:max_gross_exposure")

        if capital_remaining is not None and target_notional > capital_remaining:
            target_notional = max(capital_remaining, 0.0)
            rationale_parts.append("clamped:available_capital")

        requested_qty = target_notional / trade_price if trade_price > 0 else 0.0
        return max(requested_qty, 0.0), "|".join(rationale_parts), max(target_notional, 0.0)

    @staticmethod
    def _execute_fill_model(  # noqa: C901, PLR0911, PLR0912
        *,
        requested_qty: float,
        entry_price: float,
        direction: int,
        cfg: BacktestConfig,
        entry_row: pd.Series,
    ) -> FillExecutionResult:
        if requested_qty <= 0:
            return {
                "filled_qty": 0.0,
                "avg_fill_price": entry_price,
                "slippage_bps": 0.0,
                "reject_reason": "non_positive_requested_qty",
                "order_state": "rejected",
            }

        if cfg.fill_model.strip().lower() != "depth_aware":
            return {
                "filled_qty": requested_qty,
                "avg_fill_price": entry_price,
                "slippage_bps": 0.0,
                "reject_reason": None,
                "order_state": "filled",
            }

        if direction > 0:
            depth_top_raw = entry_row.get("ask_depth_1")
            depth_wide_raw = entry_row.get("ask_depth_5")
        else:
            depth_top_raw = entry_row.get("bid_depth_1")
            depth_wide_raw = entry_row.get("bid_depth_5")

        depth_top_parsed = pd.to_numeric(depth_top_raw, errors="coerce")
        depth_wide_parsed = pd.to_numeric(depth_wide_raw, errors="coerce")
        has_depth_observation = pd.notna(depth_top_parsed) or pd.notna(depth_wide_parsed)

        # Preserve backward compatibility for frames that do not yet carry depth fields.
        if not has_depth_observation:
            return {
                "filled_qty": requested_qty,
                "avg_fill_price": entry_price,
                "slippage_bps": 0.0,
                "reject_reason": None,
                "order_state": "filled",
            }

        depth_top = float(depth_top_parsed) if pd.notna(depth_top_parsed) else 0.0
        depth_wide = float(depth_wide_parsed) if pd.notna(depth_wide_parsed) else depth_top
        additional_depth = max(depth_wide - depth_top, 0.0)
        available_depth = depth_top + additional_depth

        if available_depth <= 0:
            return {
                "filled_qty": 0.0,
                "avg_fill_price": entry_price,
                "slippage_bps": 0.0,
                "reject_reason": "insufficient_depth",
                "order_state": "rejected",
            }

        if requested_qty <= depth_top:
            filled_qty = requested_qty
            impact_fraction = 0.0
        elif not cfg.fill_walk_the_book:
            if cfg.fill_allow_partial:
                filled_qty = max(depth_top, 0.0)
                impact_fraction = 0.0
            else:
                return {
                    "filled_qty": 0.0,
                    "avg_fill_price": entry_price,
                    "slippage_bps": 0.0,
                    "reject_reason": "exceeds_top_of_book",
                    "order_state": "rejected",
                }
        else:
            if cfg.fill_allow_partial:
                filled_qty = min(requested_qty, available_depth)
            elif requested_qty > available_depth:
                return {
                    "filled_qty": 0.0,
                    "avg_fill_price": entry_price,
                    "slippage_bps": 0.0,
                    "reject_reason": "insufficient_depth",
                    "order_state": "rejected",
                }
            else:
                filled_qty = requested_qty

            if additional_depth > 0 and filled_qty > depth_top:
                impact_fraction = min((filled_qty - depth_top) / additional_depth, 1.0)
            else:
                impact_fraction = 0.0

        spread_raw = entry_row.get("spread", 0.0)
        spread_parsed = pd.to_numeric(spread_raw, errors="coerce")
        spread = float(spread_parsed) if pd.notna(spread_parsed) else 0.0
        price_shift = (spread / 2.0) * impact_fraction * max(cfg.fill_slippage_factor, 0.0)
        if direction > 0:
            avg_fill_price = min(entry_price + price_shift, 1.0)
        else:
            avg_fill_price = max(entry_price - price_shift, 0.0)

        slippage_bps = abs(avg_fill_price - entry_price) / max(entry_price, 1e-9) * 10_000
        state = "partial" if filled_qty < requested_qty else "filled"

        return {
            "filled_qty": filled_qty,
            "avg_fill_price": avg_fill_price,
            "slippage_bps": slippage_bps,
            "reject_reason": None,
            "order_state": state,
        }

    @staticmethod
    def _decompose_execution_costs(  # noqa: PLR0913
        *,
        fee_usdc: float,
        requested_qty: float,
        filled_qty: float,
        slippage_bps: float,
        avg_fill_price: float,
        entry_price: float,
    ) -> dict[str, float]:
        spread_crossing = max(filled_qty, 0.0) * max(abs(avg_fill_price - entry_price), 0.0)
        slippage_impact = (
            max(filled_qty, 0.0) * max(slippage_bps, 0.0) * max(avg_fill_price, 0.0) / 10_000
        )
        unfilled_penalty = max(requested_qty - filled_qty, 0.0) * max(entry_price, 0.0) * 0.0
        total = fee_usdc + spread_crossing + slippage_impact + unfilled_penalty
        return {
            "taker_fee_usdc": fee_usdc,
            "spread_crossing_usdc": spread_crossing,
            "slippage_impact_usdc": slippage_impact,
            "total_execution_cost_usdc": total,
        }

    @staticmethod
    def _build_risk_evaluator(cfg: BacktestConfig) -> RiskEvaluator:
        limits = RiskLimits(
            max_drawdown_pct=cfg.risk_max_drawdown_pct,
            max_daily_loss=cfg.risk_max_daily_loss,
            max_concentration_pct=cfg.risk_max_concentration_pct,
            max_active_positions=cfg.risk_max_active_positions,
            max_gross_exposure=cfg.risk_max_gross_exposure,
        )
        return RiskEvaluator(limits)

    @staticmethod
    def _build_order_id(
        *,
        strategy_name: str,
        market_id: str,
        token_id: str,
        entry_ts: pd.Timestamp,
    ) -> str:
        payload = "|".join(
            [
                strategy_name,
                market_id,
                token_id,
                str(pd.to_datetime(entry_ts, utc=True)),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    @staticmethod
    def _assert_valid_order_transition(previous_state: str | None, next_state: str) -> None:
        allowed = _ORDER_ALLOWED_TRANSITIONS.get(previous_state, set())
        if next_state not in allowed:
            msg = f"Invalid order state transition: {previous_state!r} -> {next_state!r}"
            raise ValueError(msg)

    @staticmethod
    def _reconcile_order_trade_ledgers(  # noqa: C901, PLR0912
        order_ledger: pd.DataFrame,
        trade_ledger: pd.DataFrame,
        *,
        tolerance: float = 1e-9,
    ) -> list[str]:
        issues: list[str] = []
        if trade_ledger.empty and order_ledger.empty:
            return issues

        if trade_ledger.empty and not order_ledger.empty:
            non_terminal = order_ledger.groupby("order_id", sort=False)["state"].last()
            dangling = non_terminal[~non_terminal.isin(_ORDER_TERMINAL_STATES)]
            if not dangling.empty:
                issues.append(
                    f"non_terminal_orders_without_trades:{sorted(dangling.index.tolist())}"
                )
            return issues

        if order_ledger.empty and not trade_ledger.empty:
            issues.append("missing_order_ledger_for_non_empty_trade_ledger")
            return issues

        order_by_id = {
            order_id: group.sort_values(["ts_event", "event_seq"])
            for order_id, group in order_ledger.groupby("order_id", sort=False)
        }

        trade_by_id = {
            order_id: group
            for order_id, group in trade_ledger.groupby("order_id", sort=False)
        }

        for order_id, trade_group in trade_by_id.items():
            if order_id not in order_by_id:
                issues.append(f"trade_order_id_missing_in_order_ledger:{order_id}")
                continue

            if len(trade_group) != 1:
                issues.append(f"multiple_trade_rows_for_order_id:{order_id}")
                continue

            order_group = order_by_id[order_id]
            states = order_group["state"].astype(str).tolist()
            if not states or states[0] != "submitted":
                issues.append(f"order_missing_submitted_state:{order_id}")

            for prev_state, next_state in pairwise(states):
                allowed = _ORDER_ALLOWED_TRANSITIONS.get(prev_state, set())
                if next_state not in allowed:
                    issues.append(f"invalid_state_sequence:{order_id}:{prev_state}->{next_state}")

            if states and states[-1] not in _ORDER_TERMINAL_STATES:
                issues.append(f"order_not_terminal:{order_id}:{states[-1]}")

            trade_row = trade_group.iloc[0]
            submitted_requested = float(order_group.iloc[0].get("requested_qty", 0.0))
            trade_requested = float(trade_row.get("requested_qty", 0.0))
            if abs(submitted_requested - trade_requested) > tolerance:
                issues.append(
                    f"requested_qty_mismatch:{order_id}:order={submitted_requested},trade={trade_requested}"
                )

            order_filled = float(
                pd.to_numeric(order_group["event_fill_qty"], errors="coerce").fillna(0.0).sum()
            )
            trade_filled = float(trade_row.get("filled_qty", 0.0))
            if abs(order_filled - trade_filled) > tolerance:
                issues.append(
                    f"filled_qty_mismatch:{order_id}:order={order_filled},trade={trade_filled}"
                )

            order_cost = float(
                pd.to_numeric(order_group["event_cost_usdc"], errors="coerce").fillna(0.0).sum()
            )
            trade_cost = float(trade_row.get("total_execution_cost_usdc", 0.0))
            if abs(order_cost - trade_cost) > tolerance:
                issues.append(f"cost_mismatch:{order_id}:order={order_cost},trade={trade_cost}")

        for order_id, order_group in order_by_id.items():
            if order_id in trade_by_id:
                continue
            filled_qty = float(
                pd.to_numeric(order_group["event_fill_qty"], errors="coerce").fillna(0.0).sum()
            )
            if filled_qty > tolerance:
                issues.append(f"filled_order_missing_trade_row:{order_id}:filled={filled_qty}")

        return issues

    def _compute_cache_signature(
        self,
        features: pd.DataFrame,
        cache_key: str | None,
        *,
        cfg: BacktestConfig,
    ) -> str:
        if cache_key:
            return cache_key

        signature_payload: dict[str, object] = {
            "rows": len(features),
            "columns": sorted(features.columns.astype(str).tolist()),
            "feature_cache_signature": self.feature_generator.cache_signature(),
            "cache_schema_version": cfg.cache_schema_version,
            "cache_computation_signature": cfg.cache_computation_signature,
        }
        if not features.empty:
            signature_payload["index_min"] = str(features.index.min())
            signature_payload["index_max"] = str(features.index.max())
        return self._hash_payload(signature_payload)[:16]
