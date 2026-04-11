"""Regime-aware relative-book-strength alpha strategy.

This module wraps the core relative-book-strength strategy with per-regime
parameter overrides. Different regime conditions (risk-on, risk-off, consolidation)
receive optimized parameter sets for each regime independently.

Lookahead-bias-free: Regime classifications come from pre-trained model (Section 4)
and are applied at evaluation time using only regime data from before/at the current
timestamp. Strategy falls back to baseline params if regime is undefined.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from alphas.cumulative_relative_book_strength import (
    StrategyParams,
    build_relative_book_strength_strategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# Baseline parameters from Section 6.6 (grid_006): Sharpe=0.727, OOS Sharpe=0.500
BASELINE_PARAMS = StrategyParams(
    relative_book_score_quantile=0.55,
    spread_bps_narrow_quantile=0.15,
    confidence_score_min=0.78,
    min_liquidity=0.05,
    buy_price_max=0.92,
    max_time_to_resolution_secs=180.0,
    ask_depth_5_max_filter=800.0,
    dynamic_position_sizing=True,
    use_cumulative_signal=True,
    cumulative_signal_mode="sum",
)


@dataclass(frozen=True)
class RegimeAwareParams:
    """Parameter bundle with per-regime overrides.

    Maps regime labels to StrategyParams. Undefined regimes fall back to baseline.
    """

    regime_params: dict[str, StrategyParams]
    baseline_params: StrategyParams = BASELINE_PARAMS

    def get_params_for_regime(self, regime: str | None) -> StrategyParams:
        """Retrieve StrategyParams for given regime, falling back to baseline.

        Args:
            regime: Regime label (e.g., "risk-on", "risk-off", "consolidation")
                or None.

        Returns:
            StrategyParams for the regime, or baseline if regime is not in
            regime_params.
        """
        if regime is None or regime not in self.regime_params:
            return self.baseline_params
        return self.regime_params[regime]

    def validate(self) -> None:
        """Validate all parameter sets."""
        self.baseline_params.validate()
        for regime, params in self.regime_params.items():
            try:
                params.validate()
            except ValueError as e:
                msg = f"Invalid params for regime {regime!r}: {e}"
                raise ValueError(msg) from e


def load_regime_lookup(
    regime_csv_path: str | None,
) -> dict[pd.Timestamp | str, str] | None:
    """Load regime classification from CSV.

    Expected CSV format:
        timestamp,regime,confidence
        2025-12-01 00:30:00,risk-off,0.674
        ...

    Args:
        regime_csv_path: Path to regime CSV, or None to skip loading.

    Returns:
        Dict mapping timestamp (as string or Timestamp) to regime label,
        or None if regime_csv_path is None or file doesn't exist.
    """
    if regime_csv_path is None:
        return None

    path = Path(regime_csv_path)
    if not path.exists():
        msg = f"Warning: regime CSV not found at {regime_csv_path}"
        print(msg)
        return None

    df = pd.read_csv(path)
    if df.empty:
        msg = f"Warning: regime CSV is empty at {regime_csv_path}"
        print(msg)
        return None

    # Load timestamp and regime columns; ensure timestamp is interpretable.
    required_cols = {"timestamp", "regime"}
    if not required_cols.issubset(df.columns):
        cols_str = df.columns.tolist()
        msg = f"Regime CSV must have 'timestamp' and 'regime' columns, got {cols_str}"
        raise ValueError(msg)

    # Convert timestamp strings to UTC and normalize to 5-minute buckets.
    # This keeps regime application causal for features that do not land on an
    # exact regime boundary.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()].copy()
    df["timestamp"] = df["timestamp"].dt.floor("5min")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    regime_lookup = dict(zip(df["timestamp"], df["regime"], strict=True))

    return regime_lookup


def build_regime_aware_strategy(  # noqa: PLR0913
    *,
    regime_aware_params: RegimeAwareParams,
    regime_csv_path: str | None = None,
    enable_spread_gate: bool = False,
    enable_score_gate: bool = True,
    enable_score_gap_gate: bool = True,
    enable_price_cap_gate: bool = True,
    enable_liquidity_gate: bool = True,
    enable_ask_depth_5_cap_gate: bool = True,
    enable_time_gate: bool = True,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build regime-aware strategy callable.

    The returned callable accepts a feature DataFrame and applies per-regime
    parameters based on regime lookup. Timestamps without regime classification
    fall back to baseline.

    Args:
        regime_aware_params: RegimeAwareParams with per-regime overrides.
        regime_csv_path: Optional path to regime CSV file.
        enable_spread_gate: Whether to filter by spread.
        enable_score_gate: Whether to filter by score confidence.
        enable_score_gap_gate: Whether to filter by score gap.
        enable_price_cap_gate: Whether to filter by price cap.
        enable_liquidity_gate: Whether to filter by liquidity.
        enable_ask_depth_5_cap_gate: Whether to filter by ask depth 5.
        enable_time_gate: Whether to filter by time to resolution.

    Returns:
        Strategy callable compatible with BacktestRunner.
    """
    regime_aware_params.validate()
    regime_lookup = load_regime_lookup(regime_csv_path)

    def _strategy(features: pd.DataFrame) -> pd.DataFrame:
        """Apply regime-aware strategy with per-regime gating.

        Logic:
        1. Extract index name (typically "ts_event" or similar UTC timestamp)
        2. For each timestamp in the feature frame, look up its regime
        3. Apply regime-specific parameter set (or baseline if regime not found)
        4. Apply core strategy logic with those parameters
        5. Return signal DataFrame with regime column for post-hoc analysis

        Lookahead-bias-free: Regime lookup uses pre-determined regime
        classifications from Section 4; no data-snooping on features.
        """
        required_cols = {
            "market_id",
            "token_id",
            "mid_price",
            "spread_bps",
            "ask_depth_1",
            "ask_depth_5",
            "bid_depth_1",
            "bid_depth_5",
        }
        missing = [col for col in required_cols if col not in features.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            raise ValueError(msg)

        # Infer index name from features
        index_name = str(features.index.name or "ts_event")

        # Prepare feature frame: reset index, extract required columns
        frame = features[
            [
                "market_id",
                "token_id",
                "mid_price",
                "spread_bps",
                "ask_depth_1",
                "ask_depth_5",
                "bid_depth_1",
                "bid_depth_5",
            ]
        ].copy()

        has_time_col = "time_to_resolution_secs" in features.columns
        if has_time_col:
            frame["time_to_resolution_secs"] = features["time_to_resolution_secs"]

        original_index_column = str(features.index.name or "index")
        frame = frame.reset_index().rename(
            columns={original_index_column: index_name}
        )

        # Extract regime per row. Prefer the runner-attached regime column when
        # present; otherwise fall back to a causal timestamp lookup.
        if "regime" in features.columns:
            # Use positional assignment to avoid index reindexing when ts_event
            # contains duplicate labels.
            frame["_regime"] = pd.Series(
                features["regime"].to_numpy(copy=False),
                index=frame.index,
                dtype="object",
            )
        else:
            frame["_regime"] = frame[index_name].apply(
                lambda ts: _lookup_regime(ts, regime_lookup)
            )

        # Build signal DataFrame by stratifying by regime and applying params
        result_frames = []
        baseline_regime_key = "__baseline__"
        regime_keys = frame["_regime"].fillna(baseline_regime_key)
        unique_regimes = regime_keys.unique()

        for regime_key in unique_regimes:
            regime_mask = regime_keys == regime_key
            regime_frame = frame[regime_mask].copy()
            if regime_frame.empty:
                continue

            regime: str | None = None if regime_key == baseline_regime_key else str(regime_key)

            if regime_frame["spread_bps"].dropna().empty:
                empty_result = regime_frame.set_index(index_name).copy()
                empty_result["signal"] = 0
                empty_result["action_side"] = "buy"
                empty_result["action_score"] = 0.0
                empty_result["signal_abs"] = 0.0
                empty_result["_regime"] = regime
                result_frames.append(empty_result)
                continue

            # Retrieve parameters for this regime (includes fallback to baseline)
            params = regime_aware_params.get_params_for_regime(regime)

            # Apply core strategy with regime-specific params
            regime_frame = regime_frame.set_index(index_name)
            core_strategy = build_relative_book_strength_strategy(
                params=params,
                enable_spread_gate=enable_spread_gate,
                enable_score_gate=enable_score_gate,
                enable_score_gap_gate=enable_score_gap_gate,
                enable_price_cap_gate=enable_price_cap_gate,
                enable_liquidity_gate=enable_liquidity_gate,
                enable_ask_depth_5_cap_gate=enable_ask_depth_5_cap_gate,
                enable_time_gate=enable_time_gate,
            )

            regime_result = core_strategy(regime_frame)

            # Restore regime column in result
            regime_result["_regime"] = regime

            result_frames.append(regime_result)

        # Concatenate all regime results and restore original index order
        if result_frames:
            combined = pd.concat(result_frames, ignore_index=False)
            combined = combined.sort_index()
        else:
            # Edge case: no usable regime buckets; return zero-signal frame.
            combined = frame.set_index(index_name).copy()
            combined["signal"] = 0
            combined["action_side"] = "buy"
            combined["action_score"] = 0.0
            combined["signal_abs"] = 0.0
            combined["_regime"] = None

        return combined

    return _strategy


def _lookup_regime(
    timestamp: pd.Timestamp | str,
    regime_lookup: dict[pd.Timestamp | str, str] | None,
) -> str | None:
    """Look up regime for a given timestamp.

    Handles both pd.Timestamp and string timestamp keys in the lookup dict.
    Returns None if regime_lookup is None or timestamp not found.
    """
    if regime_lookup is None:
        return None

    # Normalize timestamp to the causal 5-minute regime bucket.
    if isinstance(timestamp, str):
        ts = pd.Timestamp(timestamp, tz="UTC")
    else:
        tz_aware = (
            pd.Timestamp(timestamp, tz="UTC")
            if timestamp.tzinfo is None
            else timestamp
        )
        ts = tz_aware

    ts = ts.floor("5min")

    # Try exact match first
    if ts in regime_lookup:
        return regime_lookup[ts]

    return None
