from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from backtester.benchmarking import (
    BenchmarkThresholds,
    build_rollout_gate_report,
    run_deterministic_benchmark,
)
from backtester.config.types import BacktestConfig
from backtester.loaders.market_events import (
    MarketEventsLoadDeps,
    MarketEventsLoadRequest,
    PyArrowModules,
    load_market_events,
)
from backtester.runner import BacktestRunner
from tests.prepared_helpers import write_prepared_manifest
from utils.dataframes import filter_by_time_range, prepare_timestamp_index

if TYPE_CHECKING:
    from pathlib import Path


def _build_inputs(
    tmp_path: Path,
    *,
    market_id: str = "cond_parity",
    winner: str = "token_yes",
) -> tuple[BacktestRunner, pd.DataFrame, pd.DataFrame, Path, str, Path]:
    storage_path = tmp_path / "pmxt"
    storage_path.mkdir(parents=True, exist_ok=True)

    runner = BacktestRunner(storage_path)
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
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
                    "winning_asset_id": winner,
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
                    "winningAssetId": winner,
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

    return runner, market_events, features, mapping_dir, token_yes, manifest_path


def _long_yes(token_yes: str):
    def _strategy(frame: pd.DataFrame) -> pd.Series:
        return (frame["token_id"].astype(str) == token_yes).astype(int)

    return _strategy


def test_run_backtest_parity_regression_tolerances(tmp_path: Path) -> None:
    runner, _, _, mapping_dir, token_yes, manifest_path = _build_inputs(tmp_path)

    config = BacktestConfig(
        mode="strict",
        shares=1.0,
        fee_rate=0.0,
        fees_enabled=False,
        enable_progress_bars=False,
        metrics_logging_enabled=False,
    )

    first = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes(token_yes),
        strategy_name="parity",
        market_batch_size=1,
        config=config,
    )
    second = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes(token_yes),
        strategy_name="parity",
        market_batch_size=1,
        config=config,
    )

    assert len(first.trade_ledger) == len(second.trade_ledger)
    net_pnl_delta = abs(
        float(first.trade_ledger["net_pnl"].sum()) - float(second.trade_ledger["net_pnl"].sum())
    )
    assert net_pnl_delta <= 1e-10

    feature_delta = abs(
        float(first.features["spread"].fillna(0.0).sum())
        - float(second.features["spread"].fillna(0.0).sum())
    )
    assert feature_delta <= 1e-10


def test_run_sensitivity_parallel_workers_matches_serial(tmp_path: Path) -> None:
    runner, _, _, mapping_dir, token_yes, manifest_path = _build_inputs(tmp_path)

    kwargs = {
        "mapping_dir": mapping_dir,
        "prepared_manifest_path": manifest_path,
        "strategy": _long_yes(token_yes),
        "strategy_name": "sweep_parity",
        "market_batch_size": 1,
        "base_config": BacktestConfig(
            mode="strict",
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            enable_progress_bars=False,
            metrics_logging_enabled=False,
        ),
        "parameter_sweeps": {"shares": [1.0, 2.0]},
        "stress_scenarios": ["baseline", "fee_increase"],
    }

    serial = runner.run_sensitivity_scenarios(parallel_workers=1, **kwargs)
    parallel = runner.run_sensitivity_scenarios(parallel_workers=2, **kwargs)

    left = serial[["scenario_id", "parameter_set", "trades", "net_pnl"]].sort_values(
        ["scenario_id", "parameter_set"]
    )
    right = parallel[["scenario_id", "parameter_set", "trades", "net_pnl"]].sort_values(
        ["scenario_id", "parameter_set"]
    )

    assert left["scenario_id"].tolist() == right["scenario_id"].tolist()
    assert left["parameter_set"].tolist() == right["parameter_set"].tolist()
    assert left["trades"].tolist() == right["trades"].tolist()
    deltas = (left["net_pnl"].to_numpy() - right["net_pnl"].to_numpy())
    assert float(abs(deltas).max()) <= 1e-10


def test_benchmark_harness_and_rollout_gate_report(tmp_path: Path) -> None:
    runner, _, _, mapping_dir, token_yes, manifest_path = _build_inputs(tmp_path)

    benchmark = run_deterministic_benchmark(
        runner=runner,
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        market_batch_size=1,
        strategy=_long_yes(token_yes),
        config=BacktestConfig(
            mode="strict",
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            enable_progress_bars=False,
            metrics_logging_enabled=False,
        ),
        parameter_sweeps={"shares": [1.0]},
        stress_scenarios=["baseline"],
    )

    assert benchmark["load_seconds"] >= 0
    assert benchmark["feature_seconds"] >= 0
    assert benchmark["backtest_seconds"] >= 0
    assert benchmark["scenario_seconds"] >= 0
    assert benchmark["peak_memory_mb"] >= 0

    report = build_rollout_gate_report(
        {
            "9": {
                **benchmark,
                "parity_net_pnl_delta": 0.0,
                "parity_feature_delta": 0.0,
            }
        },
        thresholds=BenchmarkThresholds(
            runtime_sla_seconds=60.0,
            peak_memory_mb=512.0,
            max_net_pnl_delta=1e-6,
            max_feature_delta=1e-6,
        ),
    )
    assert not report.empty
    assert bool(report.iloc[0]["ready"])


def test_manifest_duplicate_paths_are_deduplicated(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    listed_file = prepared_root / "2024-01-01" / "m1" / "a.parquet"
    listed_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"ts_event": datetime(2024, 1, 1, tzinfo=UTC), "event_type": "book", "market_id": "m1"}]
    ).to_parquet(listed_file, index=False)

    manifest_path = prepared_root / "manifest.json"
    manifest_payload = {
        "files": [
            {"output_files": [str(listed_file), str(listed_file)]},
            {"output_files": ["missing.parquet"]},
        ]
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    request = MarketEventsLoadRequest(
        market_path=prepared_root,
        start=None,
        end=None,
        limit_files=None,
        max_rows_per_file=None,
        market_slug_prefix=None,
        is_pmxt_mode=False,
        mapping_path=tmp_path / "mapping",
        manifest_path=manifest_path,
        recursive_scan=False,
    )
    deps = MarketEventsLoadDeps(
        load_condition_ids_for_slug_prefix_fn=lambda _: set(),
        normalize_market_events_schema_fn=lambda frame: frame,
        prepare_timestamp_index_fn=prepare_timestamp_index,
        filter_by_time_range_fn=filter_by_time_range,
    )

    loaded = load_market_events(
        request,
        deps=deps,
        modules=PyArrowModules(pa=None, pc=None, ds=None, pq=None),
    )
    assert len(loaded) == 1
