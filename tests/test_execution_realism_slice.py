from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from backtester.runner import BacktestConfig, BacktestRunner
from backtester.runner_support import BacktestSupportOps


def _resolution_frame(
    base_time: datetime,
    market_ids: list[str],
    winners: list[str],
) -> pd.DataFrame:
    rows = []
    for market_id, winner in zip(market_ids, winners, strict=True):
        rows.append(
            {
                "market_id": market_id,
                "resolved_at": base_time + timedelta(hours=1),
                "winning_asset_id": winner,
                "winning_outcome": "YES",
                "fees_enabled_market": True,
            }
        )
    return pd.DataFrame(rows).set_index("market_id")


def _write_prepared_manifest(
    tmp_path: Path,
    *,
    features: pd.DataFrame,
    resolution_frame: pd.DataFrame,
) -> Path:
    prepared_root = tmp_path / "pmxt_backtest"
    prepared_root.mkdir(parents=True, exist_ok=True)

    feature_entries: list[dict[str, object]] = []
    features_frame = features.copy()
    if "ts_event" not in features_frame.columns:
        features_frame = features_frame.reset_index(names="ts_event")

    for market_id, market_features in features_frame.groupby("market_id", sort=True):
        feature_path = (
            prepared_root
            / "features"
            / "2024-01-01"
            / str(market_id)
            / "features.parquet"
        )
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        market_features.to_parquet(feature_path, index=False)
        feature_entries.append({"feature_output_files": [str(feature_path)]})

    resolution_path = prepared_root / "resolution" / "resolution_frame.parquet"
    resolution_path.parent.mkdir(parents=True, exist_ok=True)
    resolution_frame.reset_index().to_parquet(resolution_path, index=False)

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": feature_entries,
                "resolution_output_file": str(resolution_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_sizing_policy_fixed_notional() -> None:
    runner = BacktestRunner(Path())
    base_time = datetime(2024, 1, 1, tzinfo=UTC)

    signal_frame = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "market_id": "m1",
                "token_id": "yes",
                "mid_price": 0.5,
                "signal": 1,
                "ask_depth_1": 100.0,
                "ask_depth_5": 100.0,
                "spread": 0.02,
            }
        ]
    ).set_index("ts_event")

    resolution_frame = _resolution_frame(base_time, ["m1"], ["yes"])

    trades = runner.simulate_hold_to_resolution_backtest(
        signal_frame,
        resolution_frame,
        strategy_name="fixed_notional",
        config=BacktestConfig(
            sizing_policy="fixed_notional",
            sizing_fixed_notional=5.0,
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            fill_model="depth_aware",
        ),
    )

    assert len(trades) == 1
    assert trades.iloc[0]["requested_qty"] == 10.0
    assert trades.iloc[0]["sizing_policy"] == "fixed_notional"


def test_depth_aware_fill_partial_with_limited_depth() -> None:
    runner = BacktestRunner(Path())
    base_time = datetime(2024, 1, 1, tzinfo=UTC)

    signal_frame = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "market_id": "m1",
                "token_id": "yes",
                "mid_price": 0.5,
                "signal": 1,
                "ask_depth_1": 2.0,
                "ask_depth_5": 5.0,
                "spread": 0.10,
            }
        ]
    ).set_index("ts_event")

    resolution_frame = _resolution_frame(base_time, ["m1"], ["yes"])

    trades = runner.simulate_hold_to_resolution_backtest(
        signal_frame,
        resolution_frame,
        strategy_name="depth_partial",
        config=BacktestConfig(
            shares=10.0,
            fee_rate=0.0,
            fees_enabled=False,
            fill_model="depth_aware",
            fill_allow_partial=True,
            fill_walk_the_book=True,
        ),
    )

    assert len(trades) == 1
    assert trades.iloc[0]["requested_qty"] == 10.0
    assert trades.iloc[0]["filled_qty"] == 5.0
    assert trades.iloc[0]["order_state"] == "partial"
    assert trades.iloc[0]["slippage_bps"] > 0


def test_depth_slippage_monotonicity_with_size() -> None:
    runner = BacktestRunner(Path())
    base_time = datetime(2024, 1, 1, tzinfo=UTC)

    signal_frame = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "market_id": "m1",
                "token_id": "yes",
                "mid_price": 0.5,
                "signal": 1,
                "ask_depth_1": 2.0,
                "ask_depth_5": 5.0,
                "spread": 0.08,
            }
        ]
    ).set_index("ts_event")

    resolution_frame = _resolution_frame(base_time, ["m1"], ["yes"])

    small = runner.simulate_hold_to_resolution_backtest(
        signal_frame,
        resolution_frame,
        strategy_name="small",
        config=BacktestConfig(
            shares=2.0,
            fee_rate=0.0,
            fees_enabled=False,
            fill_model="depth_aware",
        ),
    )
    large = runner.simulate_hold_to_resolution_backtest(
        signal_frame,
        resolution_frame,
        strategy_name="large",
        config=BacktestConfig(
            shares=5.0,
            fee_rate=0.0,
            fees_enabled=False,
            fill_model="depth_aware",
        ),
    )

    assert len(small) == 1
    assert len(large) == 1
    assert float(large.iloc[0]["slippage_bps"]) >= float(small.iloc[0]["slippage_bps"])


def test_tolerant_mode_risk_gate_logs_error_ledger(tmp_path: Path) -> None:
    storage_path = tmp_path / "pmxt"
    storage_path.mkdir(parents=True, exist_ok=True)
    runner = BacktestRunner(storage_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m1",
                "token_id": "yes1",
                "data": {
                    "asset_id": "yes1",
                    "bids": [{"price": "0.4", "size": "100"}],
                    "asks": [{"price": "0.6", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m2",
                "token_id": "yes2",
                "data": {
                    "asset_id": "yes2",
                    "bids": [{"price": "0.4", "size": "100"}],
                    "asks": [{"price": "0.6", "size": "100"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": "m1",
                "token_id": "yes1",
                "data": {"winning_asset_id": "yes1", "winning_outcome": "YES"},
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": "m2",
                "token_id": "yes2",
                "data": {"winning_asset_id": "yes2", "winning_outcome": "YES"},
            },
        ]
    ).set_index("ts_event")

    features = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "market_id": "m1",
                "token_id": "yes1",
                "mid_price": 0.6,
                "spread": 0.1,
                "ask_depth_1": 100.0,
                "ask_depth_5": 100.0,
                "bid_depth_1": 100.0,
                "bid_depth_5": 100.0,
                "imbalance_1": 0.0,
            },
            {
                "ts_event": base_time,
                "market_id": "m2",
                "token_id": "yes2",
                "mid_price": 0.6,
                "spread": 0.1,
                "ask_depth_1": 100.0,
                "ask_depth_5": 100.0,
                "bid_depth_1": 100.0,
                "bid_depth_5": 100.0,
                "imbalance_1": 0.0,
            },
        ]
    ).set_index("ts_event")

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(
            {
                "slug1": {
                    "conditionId": "m1",
                    "resolvedAt": "2024-01-01T01:00:00Z",
                    "winningAssetId": "yes1",
                    "winningOutcome": "YES",
                    "clobTokenIds": ["yes1", "no1"],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                },
                "slug2": {
                    "conditionId": "m2",
                    "resolvedAt": "2024-01-01T01:00:00Z",
                    "winningAssetId": "yes2",
                    "winningOutcome": "YES",
                    "clobTokenIds": ["yes2", "no2"],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                },
            }
        ),
        encoding="utf-8",
    )

    def strategy(frame: pd.DataFrame) -> pd.DataFrame:
        strategy_frame = frame.copy()
        strategy_frame["signal"] = 1
        return strategy_frame

    resolution_frame = _resolution_frame(base_time, ["m1", "m2"], ["yes1", "yes2"])
    manifest_path = _write_prepared_manifest(
        tmp_path,
        features=features,
        resolution_frame=resolution_frame,
    )

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name="risk_shell",
        market_batch_size=1,
        config=BacktestConfig(
            mode="tolerant",
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            risk_max_active_positions=1,
            fill_model="depth_aware",
        ),
    )

    assert not result.error_ledger.empty
    assert "active_positions_limit_exceeded" in result.error_ledger["reason"].tolist()


def test_strict_mode_risk_gate_records_rejection_without_raising(
    tmp_path: Path,
) -> None:
    storage_path = tmp_path / "pmxt"
    storage_path.mkdir(parents=True, exist_ok=True)
    runner = BacktestRunner(storage_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m1",
                "token_id": "yes1",
                "data": {
                    "asset_id": "yes1",
                    "bids": [{"price": "0.4", "size": "100"}],
                    "asks": [{"price": "0.6", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m2",
                "token_id": "yes2",
                "data": {
                    "asset_id": "yes2",
                    "bids": [{"price": "0.4", "size": "100"}],
                    "asks": [{"price": "0.6", "size": "100"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": "m1",
                "token_id": "yes1",
                "data": {"winning_asset_id": "yes1", "winning_outcome": "YES"},
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": "m2",
                "token_id": "yes2",
                "data": {"winning_asset_id": "yes2", "winning_outcome": "YES"},
            },
        ]
    ).set_index("ts_event")

    features = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "market_id": "m1",
                "token_id": "yes1",
                "mid_price": 0.6,
                "spread": 0.1,
                "ask_depth_1": 100.0,
                "ask_depth_5": 100.0,
                "bid_depth_1": 100.0,
                "bid_depth_5": 100.0,
                "imbalance_1": 0.0,
            },
            {
                "ts_event": base_time,
                "market_id": "m2",
                "token_id": "yes2",
                "mid_price": 0.6,
                "spread": 0.1,
                "ask_depth_1": 100.0,
                "ask_depth_5": 100.0,
                "bid_depth_1": 100.0,
                "bid_depth_5": 100.0,
                "imbalance_1": 0.0,
            },
        ]
    ).set_index("ts_event")

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(
            {
                "slug1": {
                    "conditionId": "m1",
                    "resolvedAt": "2024-01-01T01:00:00Z",
                    "winningAssetId": "yes1",
                    "winningOutcome": "YES",
                    "clobTokenIds": ["yes1", "no1"],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                },
                "slug2": {
                    "conditionId": "m2",
                    "resolvedAt": "2024-01-01T01:00:00Z",
                    "winningAssetId": "yes2",
                    "winningOutcome": "YES",
                    "clobTokenIds": ["yes2", "no2"],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                },
            }
        ),
        encoding="utf-8",
    )

    def strategy(frame: pd.DataFrame) -> pd.DataFrame:
        strategy_frame = frame.copy()
        strategy_frame["signal"] = 1
        return strategy_frame

    resolution_frame = _resolution_frame(base_time, ["m1", "m2"], ["yes1", "yes2"])
    manifest_path = _write_prepared_manifest(
        tmp_path,
        features=features,
        resolution_frame=resolution_frame,
    )

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name="risk_shell_strict",
        market_batch_size=1,
        config=BacktestConfig(
            mode="strict",
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            risk_max_active_positions=1,
            fill_model="depth_aware",
        ),
    )

    assert len(result.trade_ledger) == 1
    assert not result.error_ledger.empty
    assert "active_positions_limit_exceeded" in result.error_ledger["reason"].tolist()


def test_batch_mode_releases_capital_for_non_overlapping_markets(tmp_path: Path) -> None:
    storage_path = tmp_path / "pmxt"
    storage_path.mkdir(parents=True, exist_ok=True)
    runner = BacktestRunner(storage_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    features = pd.DataFrame(
        [
            {
                "ts_event": base_time + timedelta(minutes=10),
                "market_id": "a_late",
                "token_id": "yes_late",
                "mid_price": 0.5,
                "spread": 0.02,
                "imbalance_1": 0.4,
                "bid_depth_1": 100.0,
                "bid_depth_5": 100.0,
                "ask_depth_1": 100.0,
                "ask_depth_5": 100.0,
            },
            {
                "ts_event": base_time,
                "market_id": "z_early",
                "token_id": "yes_early",
                "mid_price": 0.5,
                "spread": 0.02,
                "imbalance_1": 0.4,
                "bid_depth_1": 100.0,
                "bid_depth_5": 100.0,
                "ask_depth_1": 100.0,
                "ask_depth_5": 100.0,
            },
        ]
    ).set_index("ts_event")

    resolution_frame = pd.DataFrame(
        [
            {
                "market_id": "a_late",
                "resolved_at": base_time + timedelta(minutes=15),
                "winning_asset_id": "yes_late",
                "winning_outcome": "YES",
                "fees_enabled_market": True,
            },
            {
                "market_id": "z_early",
                "resolved_at": base_time + timedelta(minutes=5),
                "winning_asset_id": "yes_early",
                "winning_outcome": "YES",
                "fees_enabled_market": True,
            },
        ]
    ).set_index("market_id")

    manifest_path = _write_prepared_manifest(
        tmp_path,
        features=features,
        resolution_frame=resolution_frame,
    )

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(
            {
                "early": {
                    "conditionId": "z_early",
                    "resolvedAt": "2024-01-01T00:05:00Z",
                    "winningAssetId": "yes_early",
                    "winningOutcome": "YES",
                    "clobTokenIds": ["yes_early", "no_early"],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                },
                "late": {
                    "conditionId": "a_late",
                    "resolvedAt": "2024-01-01T00:15:00Z",
                    "winningAssetId": "yes_late",
                    "winningOutcome": "YES",
                    "clobTokenIds": ["yes_late", "no_late"],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                },
            }
        ),
        encoding="utf-8",
    )

    def strategy(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["signal"] = 1
        return out

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name="sequential_markets",
        market_batch_size=1,
        config=BacktestConfig(
            mode="tolerant",
            fee_rate=0.0,
            fees_enabled=False,
            sizing_policy="risk_budget",
            sizing_risk_budget_pct=1.0,
            available_capital=1.0,
            risk_max_gross_exposure=1.0,
            fill_model="depth_aware",
        ),
    )

    assert len(result.trade_ledger) == 2
    assert "non_positive_requested_qty" not in result.error_ledger["reason"].tolist()


def test_order_lifecycle_emits_terminal_state_and_reconciles(tmp_path: Path) -> None:
    storage_path = tmp_path / "pmxt"
    storage_path.mkdir(parents=True, exist_ok=True)
    runner = BacktestRunner(storage_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m1",
                "token_id": "yes1",
                "data": {
                    "asset_id": "yes1",
                    "bids": [{"price": "0.4", "size": "100"}],
                    "asks": [{"price": "0.6", "size": "100"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": "m1",
                "token_id": "yes1",
                "data": {"winning_asset_id": "yes1", "winning_outcome": "YES"},
            },
        ]
    ).set_index("ts_event")

    features = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "market_id": "m1",
                "token_id": "yes1",
                "mid_price": 0.6,
                "spread": 0.1,
                "ask_depth_1": 100.0,
                "ask_depth_5": 100.0,
                "bid_depth_1": 100.0,
                "bid_depth_5": 100.0,
                "imbalance_1": 0.0,
            }
        ]
    ).set_index("ts_event")

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(
            {
                "slug1": {
                    "conditionId": "m1",
                    "resolvedAt": "2024-01-01T01:00:00Z",
                    "winningAssetId": "yes1",
                    "winningOutcome": "YES",
                    "clobTokenIds": ["yes1", "no1"],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                }
            }
        ),
        encoding="utf-8",
    )

    def strategy(frame: pd.DataFrame) -> pd.DataFrame:
        strategy_frame = frame.copy()
        strategy_frame["signal"] = 1
        return strategy_frame

    resolution_frame = _resolution_frame(base_time, ["m1"], ["yes1"])
    manifest_path = _write_prepared_manifest(
        tmp_path,
        features=features,
        resolution_frame=resolution_frame,
    )

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name="order_lifecycle",
        market_batch_size=1,
        config=BacktestConfig(
            shares=150.0,
            fee_rate=0.0,
            fees_enabled=False,
            fill_model="depth_aware",
            fill_allow_partial=True,
            fill_walk_the_book=True,
            order_lifecycle_enabled=True,
            order_ttl_seconds=30,
            order_allow_amendments=True,
            order_max_amendments=1,
        ),
    )

    assert len(result.trade_ledger) == 1
    assert not result.order_ledger.empty
    order_id = str(result.trade_ledger.iloc[0]["order_id"])
    order_states = result.order_ledger.loc[
        result.order_ledger["order_id"] == order_id,
        "state",
    ].tolist()
    assert order_states == ["submitted", "partial", "partial", "expired"]
    assert result.trade_ledger.iloc[0]["order_state"] == "expired"
    assert "order_trade_reconciliation_failed" not in result.error_ledger["reason"].tolist()


def test_order_reconciliation_detects_qty_mismatch() -> None:
    issues = BacktestSupportOps._reconcile_order_trade_ledgers(
        pd.DataFrame(
            [
                {
                    "order_id": "o1",
                    "ts_event": datetime(2024, 1, 1, tzinfo=UTC),
                    "event_seq": 1,
                    "state": "submitted",
                    "requested_qty": 10.0,
                    "event_fill_qty": 0.0,
                    "event_cost_usdc": 0.0,
                },
                {
                    "order_id": "o1",
                    "ts_event": datetime(2024, 1, 1, tzinfo=UTC),
                    "event_seq": 2,
                    "state": "filled",
                    "requested_qty": 10.0,
                    "event_fill_qty": 9.0,
                    "event_cost_usdc": 0.5,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "order_id": "o1",
                    "requested_qty": 10.0,
                    "filled_qty": 10.0,
                    "total_execution_cost_usdc": 0.5,
                }
            ]
        ),
    )

    assert any("filled_qty_mismatch" in issue for issue in issues)
