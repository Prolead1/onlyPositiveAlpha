from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from alphas.cumulative_relative_book_strength import (
    StrategyParams,
    _build_secondary_gate_thresholds,
    build_relative_book_strength_strategy,
)


def _build_features(mid_prices: list[float]) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for idx, mid in enumerate(mid_prices, start=1):
        rows.append(
            {
                "ts_event": base + timedelta(seconds=idx),
                "market_id": f"m{idx}",
                "token_id": "yes",
                "mid_price": float(mid),
                "spread_bps": 10.0,
                "ask_depth_1": 50.0,
                "ask_depth_5": 100.0,
                "bid_depth_1": 50.0,
                "bid_depth_5": 100.0,
            }
        )
    return pd.DataFrame(rows).set_index("ts_event")


def _run_price_cap_only(features: pd.DataFrame, *, params: StrategyParams) -> pd.DataFrame:
    strategy = build_relative_book_strength_strategy(
        params=params,
        enable_spread_gate=False,
        enable_score_gate=False,
        enable_score_gap_gate=False,
        enable_price_cap_gate=True,
        enable_liquidity_gate=False,
        enable_ask_depth_5_cap_gate=False,
        enable_time_gate=False,
    )
    return strategy(features)


def _run_spread_gate_only(features: pd.DataFrame, *, params: StrategyParams) -> pd.DataFrame:
    strategy = build_relative_book_strength_strategy(
        params=params,
        enable_spread_gate=True,
        enable_score_gate=False,
        enable_score_gap_gate=False,
        enable_price_cap_gate=False,
        enable_liquidity_gate=False,
        enable_ask_depth_5_cap_gate=False,
        enable_time_gate=False,
    )
    return strategy(features)


def test_recalibration_is_not_affected_by_future_markets() -> None:
    base_params = StrategyParams(
        use_cumulative_signal=False,
        buy_price_max=0.5,
        enable_secondary_gate_recalibration=True,
        secondary_gate_recalibration_frequency=1,
        secondary_gate_lookback_window=None,
    )

    features_a = _build_features([0.20, 0.25, 0.30])
    features_b = _build_features([0.20, 0.25, 0.95])

    result_a = _run_price_cap_only(features_a, params=base_params)
    result_b = _run_price_cap_only(features_b, params=base_params)

    early_a = result_a[result_a["market_id"].isin(["m1", "m2"])]["signal"].tolist()
    early_b = result_b[result_b["market_id"].isin(["m1", "m2"])]["signal"].tolist()
    assert early_a == early_b


def test_recalibration_frequency_applies_market_checkpoints() -> None:
    params = StrategyParams(
        use_cumulative_signal=False,
        buy_price_max=0.5,
        enable_secondary_gate_recalibration=True,
        secondary_gate_recalibration_frequency=2,
        secondary_gate_lookback_window=None,
    )
    features = _build_features([0.20, 0.25, 0.30, 0.35])

    result = _run_price_cap_only(features, params=params)
    signals = result.sort_values("market_id")["signal"].tolist()

    # Recompute at m1 (bootstrap) and m3 only when frequency=2.
    assert signals == [1, 1, 0, 0]


def test_recalibration_lookback_window_uses_recent_markets_only() -> None:
    params = StrategyParams(
        use_cumulative_signal=False,
        buy_price_max=0.5,
        enable_secondary_gate_recalibration=True,
        secondary_gate_recalibration_frequency=1,
        secondary_gate_lookback_window=1,
    )
    features = _build_features([0.20, 0.80, 0.30])

    result = _run_price_cap_only(features, params=params)
    signals = result.sort_values("market_id")["signal"].tolist()

    # m2 threshold uses m1 history only, m3 threshold uses m2 history only.
    assert signals == [1, 0, 1]


def test_spread_gate_thresholds_exclude_future_markets() -> None:
    params = StrategyParams(
        use_cumulative_signal=False,
        spread_bps_narrow_quantile=0.5,
        enable_secondary_gate_recalibration=False,
    )

    features_a = _build_features([0.55, 0.55, 0.55])
    features_a["spread_bps"] = [10.0, 50.0, 10.0]
    features_b = _build_features([0.55, 0.55, 0.55])
    features_b["spread_bps"] = [10.0, 50.0, 1000.0]

    result_a = _run_spread_gate_only(features_a, params=params)
    result_b = _run_spread_gate_only(features_b, params=params)

    early_markets = ["m1", "m2"]
    early_a = result_a[result_a["market_id"].isin(early_markets)]["signal"].tolist()
    early_b = result_b[result_b["market_id"].isin(early_markets)]["signal"].tolist()
    assert early_a == early_b


def test_dynamic_depth_reference_excludes_future_markets() -> None:
    params = StrategyParams(
        use_cumulative_signal=False,
        dynamic_position_sizing=True,
        dynamic_ask_depth_5_ref=None,
        enable_secondary_gate_recalibration=False,
    )

    features_a = _build_features([0.55, 0.55, 0.55])
    features_a["ask_depth_5"] = [100.0, 120.0, 140.0]
    features_b = _build_features([0.55, 0.55, 0.55])
    features_b["ask_depth_5"] = [100.0, 120.0, 5000.0]

    result_a = _run_price_cap_only(features_a, params=params)
    result_b = _run_price_cap_only(features_b, params=params)

    first_a = result_a[result_a["market_id"] == "m1"].iloc[0]
    first_b = result_b[result_b["market_id"] == "m1"].iloc[0]
    assert first_a["signal_abs"] == pytest.approx(first_b["signal_abs"])


def test_recalibration_is_deterministic_for_identical_input() -> None:
    params = StrategyParams(
        use_cumulative_signal=False,
        buy_price_max=0.5,
        min_liquidity=0.5,
        enable_secondary_gate_recalibration=True,
        secondary_gate_recalibration_frequency=1,
        secondary_gate_lookback_window=2,
    )
    features = _build_features([0.20, 0.40, 0.60, 0.80])

    strategy = build_relative_book_strength_strategy(
        params=params,
        enable_spread_gate=False,
        enable_score_gate=False,
        enable_score_gap_gate=False,
        enable_price_cap_gate=True,
        enable_liquidity_gate=True,
        enable_ask_depth_5_cap_gate=False,
        enable_time_gate=False,
    )

    first = strategy(features)
    second = strategy(features)

    assert first["signal"].tolist() == second["signal"].tolist()
    assert first["action_score"].tolist() == second["action_score"].tolist()


def _build_score_recalibration_frame(scores: list[float]) -> pd.DataFrame:
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for idx, score in enumerate(scores, start=1):
        rows.append(
            {
                "ts_event": base + timedelta(seconds=idx),
                "market_id": f"m{idx}",
                "score": float(score),
            }
        )
    return pd.DataFrame(rows)


def test_score_recalibration_excludes_future_markets() -> None:
    params = StrategyParams(
        confidence_score_min=0.5,
        enable_secondary_gate_recalibration=True,
        secondary_gate_recalibration_frequency=1,
        secondary_gate_lookback_window=None,
    )
    frame_a = _build_score_recalibration_frame([0.2, 0.25, 0.3])
    frame_b = _build_score_recalibration_frame([0.2, 0.25, 0.95])

    _, _, threshold_a = _build_secondary_gate_thresholds(
        frame_a,
        index_name="ts_event",
        params=params,
        need_price_cap=False,
        need_liquidity=False,
        need_score=True,
        score_column="score",
    )
    _, _, threshold_b = _build_secondary_gate_thresholds(
        frame_b,
        index_name="ts_event",
        params=params,
        need_price_cap=False,
        need_liquidity=False,
        need_score=True,
        score_column="score",
    )

    assert threshold_a is not None
    assert threshold_b is not None
    assert threshold_a.iloc[:2].tolist() == threshold_b.iloc[:2].tolist()


def test_score_recalibration_frequency_applies_market_checkpoints() -> None:
    params = StrategyParams(
        confidence_score_min=0.5,
        enable_secondary_gate_recalibration=True,
        secondary_gate_recalibration_frequency=2,
        secondary_gate_lookback_window=None,
    )
    frame = _build_score_recalibration_frame([0.2, 0.4, 0.6, 0.8])

    _, _, threshold = _build_secondary_gate_thresholds(
        frame,
        index_name="ts_event",
        params=params,
        need_price_cap=False,
        need_liquidity=False,
        need_score=True,
        score_column="score",
    )

    assert threshold is not None
    assert threshold.tolist() == pytest.approx([0.5, 0.5, 0.3, 0.3])


def test_score_recalibration_lookback_uses_recent_markets_only() -> None:
    params = StrategyParams(
        confidence_score_min=0.5,
        enable_secondary_gate_recalibration=True,
        secondary_gate_recalibration_frequency=1,
        secondary_gate_lookback_window=1,
    )
    frame = _build_score_recalibration_frame([0.2, 0.8, 0.3])

    _, _, threshold = _build_secondary_gate_thresholds(
        frame,
        index_name="ts_event",
        params=params,
        need_price_cap=False,
        need_liquidity=False,
        need_score=True,
        score_column="score",
    )

    assert threshold is not None
    assert threshold.tolist() == pytest.approx([0.5, 0.2, 0.8])
