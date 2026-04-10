"""Cumulative relative-book-strength alpha strategy.

This module is the canonical implementation used by diagnostics and notebooks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def recommended_gate_flags() -> dict[str, bool]:
    """Canonical post-ablation gate policy for relative-book strategy.

    The latest 500-market gate ablation recommends disabling only the spread
    gate while keeping all other execution controls enabled.
    """
    return {
        "enable_spread_gate": False,
        "enable_score_gate": True,
        "enable_score_gap_gate": True,
        "enable_price_cap_gate": True,
        "enable_liquidity_gate": True,
        "enable_ask_depth_5_cap_gate": True,
        "enable_time_gate": True,
    }


@dataclass(frozen=True)
class StrategyParams:
    """Parameter bundle for cumulative relative-book-strength strategy."""

    relative_book_score_quantile: float = 0.70
    spread_bps_narrow_quantile: float = 0.15
    confidence_score_min: float = 0.70
    min_liquidity: float = 0.10
    buy_price_max: float | None = 0.88
    min_time_to_resolution_secs: float | None = None
    max_time_to_resolution_secs: float | None = 180.0
    ask_depth_5_max_filter: float | None = 1200.0

    dynamic_position_sizing: bool = True
    dynamic_ask_depth_5_ref: float | None = 1105.44
    dynamic_mid_price_ref: float | None = 0.75

    use_cumulative_signal: bool = True
    cumulative_signal_mode: str = "sum"
    cumulative_signal_alpha: float = 0.20

    pressure_weight: float = 0.45
    spread_weight: float = 0.35
    depth_weight: float = 0.15
    imbalance_weight: float = 0.05

    # Keep enabled gate thresholds static unless explicitly overridden.
    enable_secondary_gate_recalibration: bool = False
    secondary_gate_recalibration_frequency: int = 500  # Recompute thresholds every N markets
    secondary_gate_lookback_window: float | None = None  # If provided, use rolling window of N markets

    def validate(self) -> None:
        total = (
            float(self.pressure_weight)
            + float(self.spread_weight)
            + float(self.depth_weight)
            + float(self.imbalance_weight)
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")
        if self.cumulative_signal_mode not in {"sum", "ewm"}:
            raise ValueError(
                f"cumulative_signal_mode must be 'sum' or 'ewm', got {self.cumulative_signal_mode!r}"
            )
        if not 0.0 < float(self.cumulative_signal_alpha) <= 1.0:
            raise ValueError(
                f"cumulative_signal_alpha must be in (0, 1], got {self.cumulative_signal_alpha}"
            )
        if self.enable_secondary_gate_recalibration:
            if int(self.secondary_gate_recalibration_frequency) < 1:
                raise ValueError(
                    "secondary_gate_recalibration_frequency must be >= 1, "
                    f"got {self.secondary_gate_recalibration_frequency}"
                )
            if self.secondary_gate_lookback_window is not None and int(self.secondary_gate_lookback_window) < 1:
                raise ValueError(
                    "secondary_gate_lookback_window must be >= 1 when provided, "
                    f"got {self.secondary_gate_lookback_window}"
                )


def _add_cumulative_signal_columns(
    frame: pd.DataFrame,
    *,
    index_name: str,
    alpha: float,
    mode: str,
) -> pd.DataFrame:
    """Add cumulative raw and normalized signal columns using token history."""
    sort_frame = frame.sort_values(["market_id", "token_id", index_name]).copy()
    snapshot = sort_frame["relative_book_score"].to_numpy(dtype="float64", copy=False)
    market = sort_frame["market_id"]
    token = sort_frame["token_id"]
    is_new_group = (market.ne(market.shift()) | token.ne(token.shift())).to_numpy()

    cumulative_sum = np.empty_like(snapshot)
    cumulative_ewm = np.empty_like(snapshot)
    normalized_signal = np.empty_like(snapshot)
    running_sum = 0.0
    running_ewm = 0.0
    running_abs_max = 0.0
    one_minus_alpha = 1.0 - float(alpha)

    for idx, value in enumerate(snapshot):
        if is_new_group[idx]:
            running_sum = 0.0
            running_ewm = 0.0
            running_abs_max = 0.0

        running_sum += value
        running_ewm = value if running_abs_max == 0.0 else alpha * value + one_minus_alpha * running_ewm

        raw_cumulative = running_sum if mode == "sum" else running_ewm
        running_abs_max = max(running_abs_max, abs(raw_cumulative))
        cumulative_sum[idx] = running_sum
        cumulative_ewm[idx] = running_ewm
        normalized_signal[idx] = raw_cumulative / running_abs_max if running_abs_max > 0.0 else 0.0

    sort_frame["cumulative_sum_score"] = cumulative_sum
    sort_frame["cumulative_ewm_score"] = cumulative_ewm
    sort_frame["cumulative_signal_raw"] = (
        sort_frame["cumulative_sum_score"] if mode == "sum" else sort_frame["cumulative_ewm_score"]
    )
    sort_frame["cumulative_signal_normalized"] = normalized_signal
    return sort_frame.sort_values([index_name, "market_id", "token_id"]).copy()


def _compute_secondary_threshold(values: np.ndarray, *, param: float) -> float:
    """Compute quantile/absolute threshold from historical values only."""
    clean = values[np.isfinite(values)]
    threshold_param = float(param)
    if clean.size == 0:
        # Bootstrap fallback when there is no history yet.
        return threshold_param
    if 0.0 <= threshold_param <= 1.0:
        return float(np.quantile(clean, threshold_param))
    return threshold_param


def _build_historical_market_threshold_series(
    frame: pd.DataFrame,
    *,
    index_name: str,
    column: str,
    param: float,
) -> pd.Series:
    """Build per-market thresholds from historical markets only.

    Each market receives a threshold computed from earlier markets in chronological
    order (current market excluded).
    """
    first_ts = (
        frame.groupby("market_id", observed=True)[index_name]
        .min()
        .rename("first_ts")
        .reset_index()
        .sort_values(["first_ts", "market_id"], kind="mergesort")
    )
    ordered_market_ids = first_ts["market_id"].tolist()
    if not ordered_market_ids:
        return pd.Series([], dtype="float64")

    market_values: dict[object, np.ndarray] = {}
    for market_id, series in frame.groupby("market_id", observed=True)[column]:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
        market_values[market_id] = values[np.isfinite(values)]

    current_threshold = float(param)
    market_threshold: dict[object, float] = {}
    for idx, market_id in enumerate(ordered_market_ids):
        historical_market_ids = ordered_market_ids[:idx]
        arrays = [
            market_values.get(h, np.empty(0, dtype="float64"))
            for h in historical_market_ids
        ]
        history_values = np.concatenate(arrays) if arrays else np.empty(0, dtype="float64")
        current_threshold = _compute_secondary_threshold(
            history_values,
            param=float(param),
        )
        market_threshold[market_id] = float(current_threshold)

    return frame["market_id"].map(market_threshold).astype("float64")


def _build_secondary_gate_thresholds(
    frame: pd.DataFrame,
    *,
    index_name: str,
    params: StrategyParams,
    need_price_cap: bool,
    need_liquidity: bool,
    need_score: bool,
    score_column: str | None,
) -> tuple[pd.Series | None, pd.Series | None, pd.Series | None]:
    """Build per-row secondary thresholds with market-level causal recalibration.

    Thresholds are recomputed every ``secondary_gate_recalibration_frequency`` markets
    using trailing historical markets only (current market excluded).
    """
    first_ts = (
        frame.groupby("market_id", observed=True)[index_name]
        .min()
        .rename("first_ts")
        .reset_index()
        .sort_values(["first_ts", "market_id"], kind="mergesort")
    )
    ordered_market_ids = first_ts["market_id"].tolist()

    if not ordered_market_ids:
        return None, None, None

    recal_freq = max(1, int(params.secondary_gate_recalibration_frequency))
    lookback = (
        int(params.secondary_gate_lookback_window)
        if params.secondary_gate_lookback_window is not None
        else len(ordered_market_ids)
    )
    lookback = max(1, lookback)

    market_mid_values: dict[object, np.ndarray] = {}
    market_depth_values: dict[object, np.ndarray] = {}
    market_score_values: dict[object, np.ndarray] = {}

    if need_price_cap:
        for market_id, series in frame.groupby("market_id", observed=True)["mid_price"]:
            values = series.to_numpy(dtype="float64", copy=False)
            market_mid_values[market_id] = values[np.isfinite(values)]

    if need_liquidity:
        for market_id, series in frame.groupby("market_id", observed=True)["total_depth_1"]:
            values = series.to_numpy(dtype="float64", copy=False)
            market_depth_values[market_id] = values[np.isfinite(values)]

    if need_score:
        if not score_column:
            raise ValueError("score_column is required when need_score=True")
        for market_id, series in frame.groupby("market_id", observed=True)[score_column]:
            values = series.to_numpy(dtype="float64", copy=False)
            market_score_values[market_id] = values[np.isfinite(values)]

    market_price_threshold: dict[object, float] = {}
    market_liquidity_threshold: dict[object, float] = {}
    market_score_threshold: dict[object, float] = {}
    current_price_threshold = float(params.buy_price_max) if params.buy_price_max is not None else np.nan
    current_liquidity_threshold = float(params.min_liquidity)
    current_score_threshold = float(params.confidence_score_min)

    for idx, market_id in enumerate(ordered_market_ids):
        should_recalibrate = idx == 0 or (idx % recal_freq == 0)
        if should_recalibrate:
            start_idx = max(0, idx - lookback)
            historical_market_ids = ordered_market_ids[start_idx:idx]

            if need_price_cap and params.buy_price_max is not None:
                arrays = [
                    market_mid_values.get(h, np.empty(0, dtype="float64"))
                    for h in historical_market_ids
                ]
                history_values = np.concatenate(arrays) if arrays else np.empty(0, dtype="float64")
                current_price_threshold = _compute_secondary_threshold(
                    history_values,
                    param=float(params.buy_price_max),
                )

            if need_liquidity:
                arrays = [
                    market_depth_values.get(h, np.empty(0, dtype="float64"))
                    for h in historical_market_ids
                ]
                history_values = np.concatenate(arrays) if arrays else np.empty(0, dtype="float64")
                current_liquidity_threshold = _compute_secondary_threshold(
                    history_values,
                    param=float(params.min_liquidity),
                )

            if need_score:
                arrays = [
                    market_score_values.get(h, np.empty(0, dtype="float64"))
                    for h in historical_market_ids
                ]
                history_values = np.concatenate(arrays) if arrays else np.empty(0, dtype="float64")
                current_score_threshold = _compute_secondary_threshold(
                    history_values,
                    param=float(params.confidence_score_min),
                )

        if need_price_cap and params.buy_price_max is not None:
            market_price_threshold[market_id] = float(current_price_threshold)
        if need_liquidity:
            market_liquidity_threshold[market_id] = float(current_liquidity_threshold)
        if need_score:
            market_score_threshold[market_id] = float(current_score_threshold)

    price_threshold_series = (
        frame["market_id"].map(market_price_threshold).astype("float64")
        if need_price_cap and params.buy_price_max is not None
        else None
    )
    liquidity_threshold_series = (
        frame["market_id"].map(market_liquidity_threshold).astype("float64")
        if need_liquidity
        else None
    )
    score_threshold_series = (
        frame["market_id"].map(market_score_threshold).astype("float64")
        if need_score
        else None
    )
    return price_threshold_series, liquidity_threshold_series, score_threshold_series


def build_relative_book_strength_strategy(
    *,
    params: StrategyParams,
    enable_spread_gate: bool,
    enable_score_gate: bool,
    enable_score_gap_gate: bool,
    enable_price_cap_gate: bool,
    enable_liquidity_gate: bool,
    enable_ask_depth_5_cap_gate: bool,
    enable_time_gate: bool,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Build strategy callable compatible with BacktestRunner."""

    params.validate()

    def _strategy(features: pd.DataFrame) -> pd.DataFrame:
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
            raise ValueError(f"Missing required columns: {missing}")

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

        index_name = str(frame.index.name or "ts_event")
        original_index_column = str(frame.index.name or "index")
        frame = frame.reset_index().rename(columns={original_index_column: index_name})

        frame["mid_price"] = pd.to_numeric(frame["mid_price"], errors="coerce")
        frame["spread_bps"] = pd.to_numeric(frame["spread_bps"], errors="coerce")
        frame["ask_depth_1"] = pd.to_numeric(frame["ask_depth_1"], errors="coerce")
        frame["bid_depth_1"] = pd.to_numeric(frame["bid_depth_1"], errors="coerce")
        frame["ask_depth_5"] = pd.to_numeric(frame["ask_depth_5"], errors="coerce")
        frame["bid_depth_5"] = pd.to_numeric(frame["bid_depth_5"], errors="coerce")

        frame["total_depth_1"] = frame["ask_depth_1"].fillna(0.0) + frame["bid_depth_1"].fillna(0.0)
        frame["depth_pressure_1"] = frame["ask_depth_1"].fillna(0.0) - frame["bid_depth_1"].fillna(0.0)
        frame["abs_imbalance_1"] = frame["depth_pressure_1"].abs()

        spread_values = frame["spread_bps"].dropna()
        if spread_values.empty:
            raise ValueError("Cannot build strategy without non-null spread_bps")

        group_keys = ["market_id", index_name]
        group_means = frame.groupby(group_keys, observed=True)[
            ["spread_bps", "total_depth_1", "depth_pressure_1", "abs_imbalance_1"]
        ].transform("mean")

        group_ranges = pd.DataFrame(index=frame.index)
        for col in ["spread_bps", "total_depth_1", "depth_pressure_1", "abs_imbalance_1"]:
            col_range = (
                frame.groupby(group_keys, observed=True)[col].transform("max")
                - frame.groupby(group_keys, observed=True)[col].transform("min")
            )
            group_ranges[col] = col_range.replace(0.0, 1.0)

        frame["relative_pressure"] = (
            frame["depth_pressure_1"] - group_means["depth_pressure_1"]
        ) / group_ranges["depth_pressure_1"]
        frame["relative_spread_tightness"] = (
            group_means["spread_bps"] - frame["spread_bps"]
        ) / group_ranges["spread_bps"]
        frame["relative_depth"] = (
            frame["total_depth_1"] - group_means["total_depth_1"]
        ) / group_ranges["total_depth_1"]
        frame["relative_imbalance"] = (
            frame["abs_imbalance_1"] - group_means["abs_imbalance_1"]
        ) / group_ranges["abs_imbalance_1"]

        frame["relative_book_score"] = (
            params.pressure_weight * frame["relative_pressure"]
            + params.spread_weight * frame["relative_spread_tightness"]
            + params.depth_weight * frame["relative_depth"]
            + params.imbalance_weight * frame["relative_imbalance"]
        )

        if params.use_cumulative_signal:
            frame = _add_cumulative_signal_columns(
                frame,
                index_name=index_name,
                alpha=float(params.cumulative_signal_alpha),
                mode=str(params.cumulative_signal_mode),
            )
            signal_score_col = "cumulative_signal_normalized"
        else:
            signal_score_col = "relative_book_score"

        frame["book_rank"] = (
            frame.groupby(group_keys, observed=True)[signal_score_col]
            .rank(method="first", ascending=False)
            .fillna(2.0)
        )

        frame["score_gap"] = (
            frame.groupby(group_keys, observed=True)[signal_score_col].transform("max")
            - frame.groupby(group_keys, observed=True)[signal_score_col].transform("min")
        ).fillna(0.0)

        spread_cutoff_series = _build_historical_market_threshold_series(
            frame,
            index_name=index_name,
            column="spread_bps",
            param=float(params.spread_bps_narrow_quantile),
        )
        score_cutoff_series = _build_historical_market_threshold_series(
            frame,
            index_name=index_name,
            column=signal_score_col,
            param=float(params.confidence_score_min),
        )
        score_gap_cutoff_series = _build_historical_market_threshold_series(
            frame,
            index_name=index_name,
            column="score_gap",
            param=float(params.relative_book_score_quantile),
        )

        confidence_denominator = score_gap_cutoff_series.clip(lower=1e-9)
        frame["relative_confidence"] = (
            frame["score_gap"] / confidence_denominator
        ).clip(0.0, 1.0)

        qualifying = frame["book_rank"] == 1

        if enable_spread_gate:
            qualifying = qualifying & (frame["spread_bps"] <= spread_cutoff_series)

        if enable_score_gap_gate:
            qualifying = qualifying & (frame["score_gap"] >= score_gap_cutoff_series)

        price_cap_threshold_series: pd.Series | None = None
        liquidity_threshold_series: pd.Series | None = None
        score_threshold_series: pd.Series | None = None
        if params.enable_secondary_gate_recalibration and (
            enable_price_cap_gate or enable_liquidity_gate or enable_score_gate
        ):
            (
                price_cap_threshold_series,
                liquidity_threshold_series,
                score_threshold_series,
            ) = _build_secondary_gate_thresholds(
                frame,
                index_name=index_name,
                params=params,
                need_price_cap=enable_price_cap_gate,
                need_liquidity=enable_liquidity_gate,
                need_score=enable_score_gate,
                score_column=signal_score_col,
            )

        if enable_score_gate:
            if params.enable_secondary_gate_recalibration:
                if score_threshold_series is None:
                    raise RuntimeError("Score recalibration thresholds were not computed")
                qualifying = qualifying & (frame[signal_score_col] >= score_threshold_series)
            else:
                qualifying = qualifying & (frame[signal_score_col] >= score_cutoff_series)

        # Secondary gates with periodic recalibration for sample-dependent parameters
        if enable_price_cap_gate and params.buy_price_max is not None:
            if params.enable_secondary_gate_recalibration:
                if price_cap_threshold_series is None:
                    raise RuntimeError("Price-cap recalibration thresholds were not computed")
                qualifying = qualifying & (frame["mid_price"] <= price_cap_threshold_series)
            else:
                qualifying = qualifying & (frame["mid_price"] <= float(params.buy_price_max))

        if enable_liquidity_gate:
            if params.enable_secondary_gate_recalibration:
                if liquidity_threshold_series is None:
                    raise RuntimeError("Liquidity recalibration thresholds were not computed")
                qualifying = qualifying & (frame["total_depth_1"] >= liquidity_threshold_series)
            else:
                qualifying = qualifying & (frame["total_depth_1"] >= float(params.min_liquidity))

        if enable_ask_depth_5_cap_gate and params.ask_depth_5_max_filter is not None:
            qualifying = qualifying & (
                frame["ask_depth_5"] <= float(params.ask_depth_5_max_filter)
            )

        if enable_time_gate and has_time_col:
            time_col = frame["time_to_resolution_secs"]
            if params.min_time_to_resolution_secs is not None:
                qualifying = qualifying & (time_col >= float(params.min_time_to_resolution_secs))
            if params.max_time_to_resolution_secs is not None:
                qualifying = qualifying & (time_col <= float(params.max_time_to_resolution_secs))

        frame["signal"] = 0
        frame["action_side"] = "buy"
        frame["action_score"] = 0.0
        frame["signal_abs"] = 0.0

        frame.loc[qualifying, "signal"] = 1
        frame.loc[qualifying, "action_score"] = frame.loc[qualifying, signal_score_col]

        signal_abs_scale = 1.0 if params.dynamic_position_sizing else 0.01
        if params.dynamic_position_sizing:
            if params.dynamic_ask_depth_5_ref is not None:
                ask5_ref_series = pd.Series(
                    max(float(params.dynamic_ask_depth_5_ref), 1e-9),
                    index=frame.index,
                    dtype="float64",
                )
            else:
                ask5_ref_series = _build_historical_market_threshold_series(
                    frame,
                    index_name=index_name,
                    column="ask_depth_5",
                    param=0.75,
                )
                # Bootstrap rows (no prior market history) receive fallback param values
                # in (0, 1]; replace them with contemporaneous depth to stay causal.
                ask5_ref_series = ask5_ref_series.where(
                    ask5_ref_series > 1.0,
                    pd.to_numeric(frame["ask_depth_5"], errors="coerce"),
                )
                ask5_ref_series = ask5_ref_series.fillna(1.0).clip(lower=1e-9)

            depth_quality = (1.0 - (frame["ask_depth_5"] / ask5_ref_series)).clip(0.0, 1.0)
            cumulative_strength = frame[signal_score_col].abs().clip(0.0, 1.0)
            signal_strength = (
                0.55 * cumulative_strength
                + 0.25 * frame["relative_confidence"].clip(0.0, 1.0)
                + 0.20 * depth_quality
            ).clip(0.0, 1.0)
            frame.loc[qualifying, "signal_abs"] = (
                signal_strength.loc[qualifying] * signal_abs_scale
            )
        else:
            frame.loc[qualifying, "signal_abs"] = (
                frame.loc[qualifying, signal_score_col].abs().clip(0.0, 1.0)
                * signal_abs_scale
            )

        frame = frame.set_index(index_name)
        return frame

    return _strategy
