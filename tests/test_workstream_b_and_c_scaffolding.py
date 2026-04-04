from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from backtester.config.types import BacktestConfig, FeatureGatePolicy, ValidationPolicy
from backtester.normalize.schema import validate_market_events_rows
from backtester.runner import BacktestRunner
from tests.prepared_helpers import write_prepared_manifest


def _build_sample_inputs(tmp_path: Path) -> tuple[BacktestRunner, Path, str, Path]:
    storage_path = tmp_path / "pmxt"
    storage_path.mkdir(parents=True, exist_ok=True)

    runner = BacktestRunner(storage_path)
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_id = "cond_test"
    token_yes = "token_yes"
    token_no = "token_no"

    market_events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_yes,
                "data": {
                    "asset_id": token_yes,
                    "bids": [{"price": "0.98", "size": "100"}],
                    "asks": [{"price": "0.99", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_no,
                "data": {
                    "asset_id": token_no,
                    "bids": [{"price": "0.01", "size": "100"}],
                    "asks": [{"price": "0.02", "size": "100"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": market_id,
                "token_id": token_yes,
                "data": {
                    "winning_asset_id": token_yes,
                    "winning_outcome": "Yes",
                },
            },
        ]
    ).set_index("ts_event")

    features = runner.compute_orderbook_features_df(market_events)

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(
            {
                "sample-slug": {
                    "conditionId": market_id,
                    "resolvedAt": "2024-01-01T00:00:05Z",
                    "winningAssetId": token_yes,
                    "winningOutcome": "Yes",
                    "clobTokenIds": [token_yes, token_no],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                }
            }
        ),
        encoding="utf-8",
    )

    manifest_path = write_prepared_manifest(
        tmp_path=tmp_path,
        runner=runner,
        features=features,
        market_events=market_events,
        mapping_dir=mapping_dir,
    )

    return runner, mapping_dir, token_yes, manifest_path


def test_validate_market_events_rows_quarantine_policy() -> None:
    frame = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m1",
                "token_id": "t1",
                "data": {"bids": [], "asks": []},
            },
            {
                "ts_event": datetime(2024, 1, 1, tzinfo=UTC),
                "event_type": "book",
                "market_id": "",
                "token_id": "t2",
                "data": {"bids": [], "asks": []},
            },
        ]
    )

    validated, report = validate_market_events_rows(
        frame,
        policy=ValidationPolicy(quarantine_invalid_rows=True),
    )

    assert len(validated) == 1
    assert not report.empty
    assert "missing_market_id" in report["reason"].tolist()


def test_run_backtest_tolerant_mode_quarantines_invalid_rows(tmp_path: Path) -> None:
    runner, mapping_dir, token_yes, manifest_path = _build_sample_inputs(tmp_path)

    def _long_yes(frame: pd.DataFrame) -> pd.Series:
        return (frame["token_id"].astype(str) == token_yes).astype(int)

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes,
        strategy_name="long_yes",
        market_batch_size=1,
        config=BacktestConfig(
            mode="tolerant",
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            validation_policy=ValidationPolicy(quarantine_invalid_rows=True),
        ),
    )

    assert not result.trade_ledger.empty
    assert result.data_quality_report.empty


def test_run_backtest_strict_feature_gate_failure(tmp_path: Path) -> None:
    runner, mapping_dir, token_yes, _ = _build_sample_inputs(tmp_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_id = "cond_test"
    token_no = "token_no"
    market_events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_yes,
                "data": {
                    "asset_id": token_yes,
                    "bids": [{"price": "0.98", "size": "100"}],
                    "asks": [{"price": "0.99", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_no,
                "data": {
                    "asset_id": token_no,
                    "bids": [{"price": "0.01", "size": "100"}],
                    "asks": [{"price": "0.02", "size": "100"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": market_id,
                "token_id": token_yes,
                "data": {
                    "winning_asset_id": token_yes,
                    "winning_outcome": "Yes",
                },
            },
        ]
    ).set_index("ts_event")

    features = runner.compute_orderbook_features_df(market_events)

    features_bad = features.copy()
    features_bad["spread"] = 2.0
    manifest_path = write_prepared_manifest(
        tmp_path=tmp_path,
        runner=runner,
        features=features_bad,
        market_events=market_events,
        mapping_dir=mapping_dir,
    )

    def _long_yes(frame: pd.DataFrame) -> pd.Series:
        return (frame["token_id"].astype(str) == token_yes).astype(int)

    with pytest.raises(ValueError, match="Feature quality gates failed"):
        runner.run_backtest(
            mapping_dir=mapping_dir,
            prepared_manifest_path=manifest_path,
            strategy=_long_yes,
            strategy_name="long_yes",
            market_batch_size=1,
            config=BacktestConfig(
                mode="strict",
                shares=1.0,
                fee_rate=0.0,
                fees_enabled=False,
                feature_gate_policy=FeatureGatePolicy(spread_max=1.0),
            ),
        )


def test_run_backtest_strict_trims_mapping_based_post_resolution_features(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "pmxt"
    storage_path.mkdir(parents=True, exist_ok=True)
    runner = BacktestRunner(storage_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_id = "cond_mapping_trim"
    token_yes = "token_yes"
    token_no = "token_no"

    market_events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_yes,
                "data": {
                    "asset_id": token_yes,
                    "bids": [{"price": "0.98", "size": "100"}],
                    "asks": [{"price": "0.99", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_no,
                "data": {
                    "asset_id": token_no,
                    "bids": [{"price": "0.01", "size": "100"}],
                    "asks": [{"price": "0.02", "size": "100"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=10),
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_yes,
                "data": {
                    "asset_id": token_yes,
                    "bids": [{"price": "0.97", "size": "80"}],
                    "asks": [{"price": "0.98", "size": "80"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=10),
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_no,
                "data": {
                    "asset_id": token_no,
                    "bids": [{"price": "0.02", "size": "80"}],
                    "asks": [{"price": "0.03", "size": "80"}],
                },
            },
        ]
    ).set_index("ts_event")

    features = runner.compute_orderbook_features_df(market_events)
    resolved_at = base_time + timedelta(seconds=5)
    assert (features.index > resolved_at).any()

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(
            {
                "sample-slug": {
                    "conditionId": market_id,
                    "resolvedAt": "2024-01-01T00:00:05Z",
                    "winningAssetId": token_yes,
                    "winningOutcome": "Yes",
                    "clobTokenIds": [token_yes, token_no],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                }
            }
        ),
        encoding="utf-8",
    )

    def _long_yes(frame: pd.DataFrame) -> pd.Series:
        return (frame["token_id"].astype(str) == token_yes).astype(int)

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=write_prepared_manifest(
            tmp_path=tmp_path,
            runner=runner,
            features=features,
            market_events=market_events,
            mapping_dir=mapping_dir,
        ),
        strategy=_long_yes,
        strategy_name="long_yes",
        market_batch_size=1,
        config=BacktestConfig(
            mode="strict",
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            feature_gate_policy=FeatureGatePolicy(
                block_on_post_resolution_features=True,
            ),
        ),
    )

    result_feature_index = pd.to_datetime(result.features.index, utc=True, errors="coerce")
    assert not (result_feature_index > resolved_at).any()
    assert not result.feature_health.empty
    leakage_row = result.feature_health[
        result.feature_health["metric"] == "post_resolution_leakage_markets"
    ]
    assert not leakage_row.empty
    assert int(leakage_row.iloc[0]["value"]) == 0


def test_cache_signature_changes_with_config(tmp_path: Path) -> None:
    runner, _, _, _ = _build_sample_inputs(tmp_path)

    runner._configure_feature_generator_for_run(
        BacktestConfig(cache_computation_signature="sig_a")
    )
    sig_a = runner.feature_generator.cache_signature()

    runner._configure_feature_generator_for_run(
        BacktestConfig(cache_computation_signature="sig_b")
    )
    sig_b = runner.feature_generator.cache_signature()

    assert sig_a != sig_b


def test_invalidate_feature_cache_removes_files(tmp_path: Path) -> None:
    runner, _, _, _ = _build_sample_inputs(tmp_path)

    cache_dir = runner.cache_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "features_anything_key1.parquet").write_text("x", encoding="utf-8")
    (cache_dir / "features_anything_key2.parquet").write_text("x", encoding="utf-8")

    removed = runner.invalidate_feature_cache()
    assert removed >= 2


def test_trade_ledger_contains_workstream_c_stub_columns(tmp_path: Path) -> None:
    runner, mapping_dir, token_yes, manifest_path = _build_sample_inputs(tmp_path)

    def _long_yes(frame: pd.DataFrame) -> pd.Series:
        return (frame["token_id"].astype(str) == token_yes).astype(int)

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes,
        strategy_name="long_yes",
        market_batch_size=1,
        config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
    )

    required_columns = {
        "requested_qty",
        "filled_qty",
        "avg_fill_price",
        "slippage_bps",
        "reject_reason",
        "order_state",
        "sizing_policy",
        "sizing_rationale",
        "taker_fee_usdc",
        "spread_crossing_usdc",
        "slippage_impact_usdc",
        "total_execution_cost_usdc",
    }
    assert required_columns.issubset(set(result.trade_ledger.columns))


def test_run_backtest_is_deterministic_for_identical_prepared_inputs(
    tmp_path: Path,
) -> None:
    runner, mapping_dir, token_yes, manifest_path = _build_sample_inputs(tmp_path)

    def _long_yes(frame: pd.DataFrame) -> pd.Series:
        return (frame["token_id"].astype(str) == token_yes).astype(int)

    config = BacktestConfig(mode="tolerant", shares=1.0, fee_rate=0.0, fees_enabled=False)

    first = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes,
        strategy_name="long_yes",
        market_batch_size=1,
        config=config,
    )
    second = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes,
        strategy_name="long_yes",
        market_batch_size=1,
        config=config,
    )

    pd.testing.assert_frame_equal(
        first.trade_ledger.drop(columns=["run_id"]).reset_index(drop=True),
        second.trade_ledger.drop(columns=["run_id"]).reset_index(drop=True),
        check_dtype=False,
    )
