"""Cumulative relative-book-strength alpha strategy.

This module is the canonical implementation used by diagnostics and notebooks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


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

        spread_cutoff = float(spread_values.quantile(params.spread_bps_narrow_quantile))
        score_cutoff = (
            float(frame[signal_score_col].quantile(params.confidence_score_min))
            if 0.0 <= params.confidence_score_min <= 1.0
            else float(params.confidence_score_min)
        )
        score_gap_cutoff = (
            float(frame["score_gap"].quantile(params.relative_book_score_quantile))
            if 0.0 <= params.relative_book_score_quantile <= 1.0
            else float(params.relative_book_score_quantile)
        )

        confidence_denominator = max(score_gap_cutoff, 1e-9)
        frame["relative_confidence"] = (
            frame["score_gap"] / confidence_denominator
        ).clip(0.0, 1.0)

        qualifying = frame["book_rank"] == 1

        if enable_spread_gate:
            qualifying = qualifying & (frame["spread_bps"] <= spread_cutoff)

        if enable_score_gate:
            qualifying = qualifying & (frame[signal_score_col] >= score_cutoff)

        if enable_score_gap_gate:
            qualifying = qualifying & (frame["score_gap"] >= score_gap_cutoff)

        if enable_price_cap_gate and params.buy_price_max is not None:
            qualifying = qualifying & (frame["mid_price"] <= float(params.buy_price_max))

        if enable_liquidity_gate:
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
            ask5_ref = (
                float(params.dynamic_ask_depth_5_ref)
                if params.dynamic_ask_depth_5_ref is not None
                else float(frame["ask_depth_5"].quantile(0.75))
            )
            ask5_ref = max(ask5_ref, 1e-9)

            depth_quality = (1.0 - (frame["ask_depth_5"] / ask5_ref)).clip(0.0, 1.0)
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
