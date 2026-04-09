from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from backtester.config.normalization import (
    coerce_backtest_config,
    generate_spread_imbalance_signals,
    normalize_strategy_output,
)
from backtester.config.types import BacktestConfig
from backtester.runner import BacktestRunner
from backtester.runner_support import BacktestSupportOps


class _Harness(BacktestSupportOps):
    pass


def _make_runner(tmp_path: Path) -> BacktestRunner:
    storage = tmp_path / "pmxt"
    storage.mkdir(parents=True, exist_ok=True)
    return BacktestRunner(storage)


def _sample_features() -> pd.DataFrame:
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "market_id": "m1",
                "token_id": "yes",
                "mid_price": 0.52,
                "spread": 0.02,
                "imbalance_1": 0.8,
                "ask_depth_1": 10.0,
                "ask_depth_5": 20.0,
                "bid_depth_1": 11.0,
                "bid_depth_5": 21.0,
            },
            {
                "ts_event": base_time + timedelta(seconds=1),
                "market_id": "m1",
                "token_id": "no",
                "mid_price": 0.48,
                "spread": 0.04,
                "imbalance_1": -0.9,
                "ask_depth_1": 10.0,
                "ask_depth_5": 20.0,
                "bid_depth_1": 11.0,
                "bid_depth_5": 21.0,
            },
        ]
    ).set_index("ts_event")


def test_core_timestamp_and_float_helpers() -> None:
    assert BacktestRunner._to_utc_timestamp("bad-ts") is None
    parsed = BacktestRunner._to_utc_timestamp("2024-01-01T00:00:00Z")
    assert parsed is not None
    assert BacktestRunner._to_float_or_nan("abc") != BacktestRunner._to_float_or_nan("abc")
    assert BacktestRunner._to_float_or_nan("1.25") == 1.25


def test_compute_cache_key_changes_with_inputs(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("y", encoding="utf-8")

    k1 = runner._compute_cache_key([f1, f2])
    k2 = runner._compute_cache_key([f1, f2], start=datetime(2024, 1, 1, tzinfo=UTC))

    assert k1 != k2


def test_generate_spread_imbalance_signals_and_attrs() -> None:
    features = _sample_features()

    enriched = generate_spread_imbalance_signals(
        features,
        imbalance_threshold=0.5,
        spread_tight_quantile=0.25,
        spread_wide_quantile=0.75,
    )

    assert "imbalance_signal" in enriched.columns
    assert "spread_signal" in enriched.columns
    assert enriched.attrs["spread_tight_cutoff"] <= enriched.attrs["spread_wide_cutoff"]


def test_generate_spread_imbalance_signals_missing_columns_raises() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        generate_spread_imbalance_signals(
            pd.DataFrame({"spread": [0.1]}),
            imbalance_threshold=0.5,
            spread_tight_quantile=0.25,
            spread_wide_quantile=0.75,
        )


def test_generate_spread_imbalance_signals_empty_spread_raises() -> None:
    frame = pd.DataFrame({"spread": [None], "imbalance_1": [0.1]})
    with pytest.raises(ValueError, match="Cannot build spread signals"):
        generate_spread_imbalance_signals(
            frame,
            imbalance_threshold=0.5,
            spread_tight_quantile=0.25,
            spread_wide_quantile=0.75,
        )


def test_coerce_backtest_config_parses_mapping() -> None:
    cfg = coerce_backtest_config(
        {
            "mode": "tolerant",
            "shares": "2.5",
            "fee_rate": "0.01",
            "fees_enabled": "false",
            "order_ttl_seconds": "30",
            "sizing_policy": "fixed_notional",
            "sizing_fixed_notional": "5.0",
            "action_selection_lookahead_seconds": "45",
            "validation_policy": {"quarantine_invalid_rows": "true"},
            "feature_gate_policy": {"spread_max": "0.9"},
        }
    )

    assert cfg.mode == "tolerant"
    assert cfg.shares == 2.5
    assert cfg.fee_rate == 0.01
    assert cfg.fees_enabled is False
    assert cfg.order_ttl_seconds == 30
    assert cfg.sizing_fixed_notional == 5.0
    assert cfg.action_selection_lookahead_seconds == 45
    assert cfg.validation_policy.quarantine_invalid_rows is True
    assert cfg.feature_gate_policy.spread_max == 0.9


def test_normalize_strategy_output_series_and_dataframe_paths() -> None:
    features = _sample_features()

    series_out = pd.Series([1, 0], index=features.index)
    normalized_series = normalize_strategy_output(
        features=features,
        strategy_name="series",
        strategy_output=series_out,
        signal_column="signal",
    )
    assert len(normalized_series) == 1
    assert normalized_series.iloc[0]["signal"] == 1
    assert normalized_series.iloc[0]["action_side"] == "buy"
    assert normalized_series.iloc[0]["action_score"] == pytest.approx(1.0)

    df_out = pd.DataFrame(
        {
            "signal": [1, -1],
            "market_id": ["m1", "m1"],
            "token_id": ["yes", "no"],
            "mid_price": [0.52, 0.48],
        },
        index=features.index,
    )
    normalized_df = normalize_strategy_output(
        features=features,
        strategy_name="df",
        strategy_output=df_out,
        signal_column="signal",
    )
    assert len(normalized_df) == 2
    assert set(normalized_df["action_side"].tolist()) == {"buy", "sell"}
    assert normalized_df["action_score"].tolist() == pytest.approx([1.0, 1.0])


def test_normalize_strategy_output_retains_diagnostic_columns() -> None:
    features = _sample_features()

    df_out = pd.DataFrame(
        {
            "signal": [1, -1],
            "market_id": ["m1", "m1"],
            "token_id": ["yes", "no"],
            "mid_price": [0.52, 0.48],
            "depth_pressure": [1.2, -1.1],
            "depth_pressure_rank": [0.9, 0.95],
            "imbalance_1": [0.8, -0.9],
            "spread_rank": [0.9, 0.95],
            "spread_narrow_rank": [0.1, 0.05],
            "imbalance_rank": [0.8, 0.9],
            "confidence_score": [0.86, 0.92],
            "liquidity": [21.0, 21.0],
            "signal_abs": [0.86, 0.92],
        },
        index=features.index,
    )
    normalized_df = normalize_strategy_output(
        features=features,
        strategy_name="df_with_diagnostics",
        strategy_output=df_out,
        signal_column="signal",
    )

    for col in [
        "depth_pressure",
        "depth_pressure_rank",
        "imbalance_1",
        "spread_rank",
        "spread_narrow_rank",
        "imbalance_rank",
        "confidence_score",
        "liquidity",
        "signal_abs",
    ]:
        assert col in normalized_df.columns


def test_normalize_strategy_output_invalid_type_raises() -> None:
    features = _sample_features()
    with pytest.raises(TypeError, match="must return pandas Series or DataFrame"):
        normalize_strategy_output(
            features=features,
            strategy_name="bad",
            strategy_output=123,
        )


def test_normalize_strategy_output_missing_required_raises() -> None:
    features = _sample_features()
    bad = pd.DataFrame({"signal": [1, -1]}, index=[0, 1])
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_strategy_output(
            features=features,
            strategy_name="bad_df",
            strategy_output=bad,
        )


def test_execute_fill_model_branches() -> None:
    cfg = BacktestConfig(fill_model="depth_aware", fill_allow_partial=True, fill_walk_the_book=True)

    non_positive = BacktestSupportOps._execute_fill_model(
        requested_qty=0.0,
        entry_price=0.5,
        direction=1,
        cfg=cfg,
        entry_row=pd.Series({}),
    )
    assert non_positive["order_state"] == "rejected"

    no_depth_observation = BacktestSupportOps._execute_fill_model(
        requested_qty=2.0,
        entry_price=0.5,
        direction=1,
        cfg=cfg,
        entry_row=pd.Series({"spread": 0.1}),
    )
    assert no_depth_observation["order_state"] == "filled"

    insufficient = BacktestSupportOps._execute_fill_model(
        requested_qty=2.0,
        entry_price=0.5,
        direction=1,
        cfg=cfg,
        entry_row=pd.Series({"ask_depth_1": 0.0, "ask_depth_5": 0.0, "spread": 0.1}),
    )
    assert insufficient["reject_reason"] == "insufficient_depth"


def test_execute_fill_model_partial_and_non_depth_mode() -> None:
    depth_cfg = BacktestConfig(
        fill_model="depth_aware",
        fill_allow_partial=True,
        fill_walk_the_book=True,
        fill_slippage_factor=1.0,
    )
    partial = BacktestSupportOps._execute_fill_model(
        requested_qty=10.0,
        entry_price=0.5,
        direction=1,
        cfg=depth_cfg,
        entry_row=pd.Series({"ask_depth_1": 2.0, "ask_depth_5": 5.0, "spread": 0.1}),
    )
    assert partial["order_state"] == "partial"
    assert partial["filled_qty"] == 5.0

    simple_cfg = BacktestConfig(fill_model="simple")
    simple = BacktestSupportOps._execute_fill_model(
        requested_qty=3.0,
        entry_price=0.5,
        direction=1,
        cfg=simple_cfg,
        entry_row=pd.Series({}),
    )
    assert simple["order_state"] == "filled"
    assert simple["filled_qty"] == 3.0


def test_decompose_execution_costs_and_reconcile_edges() -> None:
    costs = BacktestSupportOps._decompose_execution_costs(
        fee_usdc=0.1,
        requested_qty=10.0,
        filled_qty=5.0,
        slippage_bps=25.0,
        avg_fill_price=0.52,
        entry_price=0.5,
    )
    assert costs["total_execution_cost_usdc"] >= costs["taker_fee_usdc"]

    missing_order = BacktestSupportOps._reconcile_order_trade_ledgers(
        pd.DataFrame(),
        pd.DataFrame([{"order_id": "o1", "requested_qty": 1.0, "filled_qty": 1.0}]),
    )
    assert any("missing_order_ledger" in item for item in missing_order)


def test_estimate_market_volatility_and_sizing() -> None:
    harness = _Harness()
    base = datetime(2024, 1, 1, tzinfo=UTC)
    market_group = pd.DataFrame(
        {
            "mid_price": [0.45, 0.46, 0.48, 0.5, 0.52],
        },
        index=[base + timedelta(seconds=i) for i in range(5)],
    )

    vol = harness._estimate_market_volatility(
        market_group,
        entry_ts=base + timedelta(seconds=5),
        lookback=4,
    )
    assert vol is not None

    qty, rationale, notional = harness._apply_sizing_policy(
        signal_value=1,
        signal_abs=1.0,
        entry_price=0.5,
        market_group=market_group,
        entry_ts=base + timedelta(seconds=5),
        cfg=BacktestConfig(sizing_policy="fixed_notional", sizing_fixed_notional=4.0, shares=1.0),
        gross_exposure_used=0.0,
        capital_remaining=10.0,
    )
    assert qty > 0
    assert notional > 0
    assert "fixed_notional" in rationale


def test_parameter_grid_and_stress_scenarios(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)
    grid = runner._build_parameter_sweep_grid({"shares": [1, 2], "fee_rate": [0.1]})
    assert len(grid) == 2

    base = datetime(2024, 1, 1, tzinfo=UTC)
    events = pd.DataFrame(
        [
            {
                "ts_event": base,
                "event_type": "market_resolved",
                "market_id": "m1",
                "token_id": "yes",
                "data": {},
            }
        ]
    ).set_index("ts_event")
    features = _sample_features()

    stressed_events, stressed_features, stressed_cfg = runner._apply_stress_scenario(
        scenario_id="fee_increase",
        market_events=events,
        features=features,
        config=BacktestConfig(fee_rate=0.01),
    )
    assert len(stressed_events) == len(events)
    assert len(stressed_features) == len(features)
    assert stressed_cfg.fee_rate == pytest.approx(0.015)

    with pytest.raises(ValueError, match="Unsupported stress scenario"):
        runner._apply_stress_scenario(
            scenario_id="unknown",
            market_events=events,
            features=features,
            config=BacktestConfig(),
        )
