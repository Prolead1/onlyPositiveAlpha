from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from backtester import BacktestConfig, BacktestRunner
from backtester.loaders.crypto_prices import load_crypto_prices
from backtester.loaders.market_events import truncate_post_resolution_events
from backtester.normalize.market_lookup import (
    filter_market_events_by_slug_prefix,
    load_condition_ids_for_slug_prefix,
)
from backtester.normalize.schema import normalize_market_events_schema
from backtester.resolution.parsing import (
    is_missing_resolution_winner,
    parse_clob_token_ids,
    parse_outcome_prices,
    select_winning_asset_id,
)
from backtester.simulation.analytics import build_equity_curve, summarize_backtest
from backtester.simulation.fees import calculate_taker_fee
from utils.dataframes import filter_by_time_range, prepare_timestamp_index


def test_calculate_taker_fee_matches_existing_behavior() -> None:
    fee = calculate_taker_fee(
        0.5,
        shares=1.0,
        fee_rate=0.072,
        fees_enabled=True,
        precision=5,
        minimum_fee=0.00001,
    )
    assert fee == 0.018

    tiny_fee = calculate_taker_fee(
        0.5,
        shares=1.0,
        fee_rate=0.00001,
        fees_enabled=True,
        precision=5,
        minimum_fee=0.00001,
    )
    assert tiny_fee == 0.0


def test_parsing_helpers_cover_token_and_outcome_paths() -> None:
    assert parse_clob_token_ids('["t1", "t2"]') == ["t1", "t2"]
    assert parse_clob_token_ids(["t1", 2]) == ["t1", "2"]
    assert parse_clob_token_ids("not-json") == []

    prices = parse_outcome_prices(["0.1", 0.9, None])
    assert prices[:2] == [0.1, 0.9]
    assert pd.isna(prices[2])

    winner = select_winning_asset_id(["yes", "no"], [0.95, 0.05])
    assert winner == "yes"


def test_missing_winner_detection() -> None:
    assert is_missing_resolution_winner(None)
    assert is_missing_resolution_winner("")
    assert is_missing_resolution_winner(float("nan"))
    assert not is_missing_resolution_winner("token_yes")


def test_analytics_summary_and_equity_curve() -> None:
    trades = pd.DataFrame(
        [
            {
                "strategy": "s1",
                "resolved_at": datetime(2024, 1, 1, tzinfo=UTC),
                "gross_pnl": 0.2,
                "net_pnl": 0.18,
                "fee_usdc": 0.02,
                "gross_notional": 0.5,
                "hold_hours": 1.0,
                "gross_return_pct": 0.4,
                "net_return_pct": 0.36,
            },
            {
                "strategy": "s1",
                "resolved_at": datetime(2024, 1, 2, tzinfo=UTC),
                "gross_pnl": -0.1,
                "net_pnl": -0.11,
                "fee_usdc": 0.01,
                "gross_notional": 0.5,
                "hold_hours": 2.0,
                "gross_return_pct": -0.2,
                "net_return_pct": -0.22,
            },
        ]
    )

    summary = summarize_backtest(trades)
    assert len(summary) == 1
    assert summary.iloc[0]["trades"] == 2
    assert summary.iloc[0]["wins"] == 1
    assert summary.iloc[0]["fees"] == pytest.approx(0.03)

    equity = build_equity_curve(trades)
    assert len(equity) == 2
    assert equity["cumulative_net_pnl"].iloc[-1] == pytest.approx(0.07)


def test_normalize_market_events_schema_handles_fallback_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp_received": datetime(2024, 1, 1, tzinfo=UTC),
                "update_type": "price_change",
                "market_id": "m1",
                "data": json.dumps(
                    {
                        "token_id": "t1",
                        "best_bid": "0.49",
                        "best_ask": "0.51",
                        "change_price": "0.50",
                        "change_side": "BUY",
                        "change_size": "10",
                    }
                ),
            }
        ]
    )

    normalized = normalize_market_events_schema(frame, is_pmxt_mode=False)

    assert list(normalized.columns) == ["ts_event", "event_type", "market_id", "token_id", "data"]
    assert normalized.iloc[0]["event_type"] == "price_change"
    assert normalized.iloc[0]["token_id"] == "t1"
    payload = normalized.iloc[0]["data"]
    assert isinstance(payload, dict)
    assert "price_changes" in payload
    assert payload["price_changes"][0]["asset_id"] == "t1"


def test_market_lookup_and_filter_use_mapping_prefix(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "btc-updown-5m-1": {"conditionId": "0xabc"},
        "eth-updown-5m-1": {"conditionId": "0xdef"},
    }
    (mapping_dir / "gamma_updown_markets_2026-02-21.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    cache: dict[str, set[str]] = {}
    ids = load_condition_ids_for_slug_prefix(mapping_dir, "btc-updown-5m", cache=cache)
    assert ids == {"0xabc"}
    assert cache["btc-updown-5m"] == {"0xabc"}

    market_events = pd.DataFrame(
        [
            {"market_id": "0xabc", "value": 1},
            {"market_id": "0xdef", "value": 2},
        ]
    )

    filtered = filter_market_events_by_slug_prefix(
        market_events,
        "btc-updown-5m",
        is_pmxt_mode=True,
        mapping_path=mapping_dir,
        condition_ids_lookup=lambda prefix: load_condition_ids_for_slug_prefix(
            mapping_dir,
            prefix,
            cache=cache,
        ),
    )
    assert filtered["market_id"].tolist() == ["0xabc"]


def test_crypto_loader_parses_json_data_column(tmp_path: Path) -> None:
    rtds_path = tmp_path / "polymarket_rtds"
    rtds_path.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, tzinfo=UTC),
                "symbol": "BTC",
                "data": json.dumps({"price": 42000.0}),
            }
        ]
    )
    frame.to_parquet(rtds_path / "events.parquet", index=False)

    loaded = load_crypto_prices(
        rtds_path,
        start=None,
        end=None,
        prepare_timestamp_index_fn=prepare_timestamp_index,
        filter_by_time_range_fn=filter_by_time_range,
    )

    assert len(loaded) == 1
    assert loaded.iloc[0]["symbol"] == "BTC"
    assert isinstance(loaded.iloc[0]["data"], dict)
    assert loaded.iloc[0]["data"]["price"] == 42000.0


def test_loader_truncates_post_resolution_rows() -> None:
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m1",
                "token_id": "t1",
                "data": {},
            },
            {
                "ts_event": base_time + pd.Timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": "m1",
                "token_id": "t1",
                "data": {"winning_asset_id": "t1"},
            },
            {
                "ts_event": base_time + pd.Timedelta(seconds=10),
                "event_type": "book",
                "market_id": "m1",
                "token_id": "t1",
                "data": {},
            },
            {
                "ts_event": base_time + pd.Timedelta(seconds=10),
                "event_type": "book",
                "market_id": "m2",
                "token_id": "t2",
                "data": {},
            },
        ]
    ).set_index("ts_event")

    trimmed = truncate_post_resolution_events(frame)

    # m1 rows after its first resolution event are removed.
    assert len(trimmed[(trimmed["market_id"] == "m1") & (trimmed.index > base_time + pd.Timedelta(seconds=5))]) == 0
    # m2 remains untouched because it has no resolution row.
    assert len(trimmed[trimmed["market_id"] == "m2"]) == 1


def test_runner_load_prepared_features_from_manifest(tmp_path: Path) -> None:
    prepared_root = tmp_path / "pmxt_backtest"
    prepared_root.mkdir(parents=True, exist_ok=True)

    feature_file = prepared_root / "features" / "2024-01-01" / "m1" / "f1.parquet"
    feature_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, tzinfo=UTC),
                "market_id": "m1",
                "token_id": "t1",
                "spread": 0.02,
                "mid_price": 0.5,
                "imbalance_1": 0.1,
                "bid_depth_1": 100.0,
                "ask_depth_1": 90.0,
            }
        ]
    ).to_parquet(feature_file, index=False)

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "output_files": [],
                        "feature_output_files": [str(feature_file)],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = BacktestRunner(storage_path=prepared_root)
    loaded = runner.load_prepared_features(features_manifest_path=manifest_path)

    assert not loaded.empty
    assert loaded["market_id"].astype(str).tolist() == ["m1"]
    assert loaded.index.is_monotonic_increasing


def test_runner_load_prepared_resolution_frame_preserves_market_date_rows(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "pmxt_backtest"
    prepared_root.mkdir(parents=True, exist_ok=True)

    resolution_rows = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "feature_date": "2024-01-01",
                "resolved_at": datetime(2024, 1, 1, 1, tzinfo=UTC),
                "winning_asset_id": "yes_day1",
                "winning_outcome": "YES",
                "fees_enabled_market": True,
            },
            {
                "market_id": "m1",
                "feature_date": "2024-01-02",
                "resolved_at": datetime(2024, 1, 2, 1, tzinfo=UTC),
                "winning_asset_id": "yes_day2",
                "winning_outcome": "YES",
                "fees_enabled_market": True,
            },
        ]
    )
    resolution_path = prepared_root / "resolution" / "resolution_frame.parquet"
    resolution_path.parent.mkdir(parents=True, exist_ok=True)
    resolution_rows.to_parquet(resolution_path, index=False)

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [],
                "resolution_output_file": str(resolution_path),
            }
        ),
        encoding="utf-8",
    )

    runner = BacktestRunner(storage_path=prepared_root)
    loaded = runner.load_prepared_resolution_frame(resolution_manifest_path=manifest_path)

    assert len(loaded) == 2
    assert loaded["market_id"].astype(str).tolist() == ["m1", "m1"]
    assert loaded["feature_date"].astype(str).tolist() == ["2024-01-01", "2024-01-02"]
    winners = dict(zip(loaded["feature_date"], loaded["winning_asset_id"], strict=True))
    assert winners == {
        "2024-01-01": "yes_day1",
        "2024-01-02": "yes_day2",
    }


def test_runner_load_prepared_feature_market_ids_and_market_filtering(tmp_path: Path) -> None:
    prepared_root = tmp_path / "pmxt_backtest"
    prepared_root.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    feature_files: list[Path] = []
    for idx, market_id in enumerate(("m1", "m2", "m3"), start=1):
        feature_file = (
            prepared_root
            / "features"
            / "2024-01-01"
            / market_id
            / f"f{idx}.parquet"
        )
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": market_id,
                    "token_id": f"t{idx}",
                    "spread": 0.02,
                    "mid_price": 0.5,
                    "imbalance_1": 0.1,
                    "bid_depth_1": 100.0,
                    "ask_depth_1": 90.0,
                }
            ]
        ).to_parquet(feature_file, index=False)
        feature_files.append(feature_file)

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "output_files": [],
                        "feature_output_files": [str(path) for path in feature_files],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    runner = BacktestRunner(storage_path=prepared_root)

    market_ids = runner.load_prepared_feature_market_ids(features_manifest_path=manifest_path)
    assert market_ids == ["m1", "m2", "m3"]

    batch_count = runner.count_prepared_feature_market_batches(
        market_batch_size=2,
        features_manifest_path=manifest_path,
    )
    assert batch_count == 2

    filtered = runner.load_prepared_features(
        features_manifest_path=manifest_path,
        market_ids={"m2", "m3"},
    )
    assert sorted(filtered["market_id"].astype(str).unique().tolist()) == ["m2", "m3"]


def test_run_backtest_market_batches_preserve_risk_state_across_batches(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "pmxt_backtest"
    prepared_root.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_ids = ["m1", "m2", "m3", "m4"]

    feature_files: list[Path] = []
    resolution_rows: list[dict[str, object]] = []
    for idx, market_id in enumerate(market_ids, start=1):
        token_id = f"yes_{market_id}"
        feature_file = (
            prepared_root
            / "features"
            / "2024-01-01"
            / market_id
            / f"f{idx}.parquet"
        )
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": market_id,
                    "token_id": token_id,
                    "spread": 0.02,
                    "mid_price": 0.6,
                    "imbalance_1": 0.2,
                    "bid_depth_1": 100.0,
                    "ask_depth_1": 100.0,
                    "bid_depth_5": 100.0,
                    "ask_depth_5": 100.0,
                }
            ]
        ).to_parquet(feature_file, index=False)
        feature_files.append(feature_file)
        resolution_rows.append(
            {
                "market_id": market_id,
                "resolved_at": base_time + pd.Timedelta(hours=1),
                "winning_asset_id": token_id,
                "winning_outcome": "YES",
                "fees_enabled_market": True,
            }
        )

    resolution_path = prepared_root / "resolution" / "resolution_frame.parquet"
    resolution_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resolution_rows).to_parquet(resolution_path, index=False)

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "output_files": [],
                        "feature_output_files": [str(path) for path in feature_files],
                    }
                ],
                "resolution_output_file": str(resolution_path),
            }
        ),
        encoding="utf-8",
    )

    runner = BacktestRunner(storage_path=prepared_root)

    def always_long(features: pd.DataFrame) -> pd.DataFrame:
        frame = features.copy()
        frame["signal"] = 1
        return frame

    config = BacktestConfig(
        mode="tolerant",
        shares=1.0,
        fee_rate=0.0,
        fees_enabled=False,
        risk_max_active_positions=1,
        enable_progress_bars=False,
        metrics_logging_enabled=False,
    )

    full_result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=always_long,
        strategy_name="always_long",
        config=config,
        market_batch_size=4,
    )

    batched_result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=always_long,
        strategy_name="always_long",
        config=config,
        market_batch_size=2,
    )

    assert len(full_result.trade_ledger) == 1
    assert len(batched_result.trade_ledger) == 1

    pd.testing.assert_frame_equal(
        full_result.trade_ledger.drop(columns=["run_id"]).reset_index(drop=True),
        batched_result.trade_ledger.drop(columns=["run_id"]).reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        full_result.backtest_summary.drop(columns=["run_id"]).reset_index(drop=True),
        batched_result.backtest_summary.drop(columns=["run_id"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_run_backtest_market_batches_updates_batch_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_root = tmp_path / "pmxt_backtest"
    prepared_root.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_ids = ["m1", "m2", "m3"]
    feature_files: list[Path] = []
    resolution_rows: list[dict[str, object]] = []
    for market_id in market_ids:
        token_id = f"yes_{market_id}"
        feature_file = (
            prepared_root
            / "features"
            / "2024-01-01"
            / market_id
            / "f.parquet"
        )
        feature_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": market_id,
                    "token_id": token_id,
                    "spread": 0.02,
                    "mid_price": 0.6,
                    "imbalance_1": 0.2,
                    "bid_depth_1": 100.0,
                    "ask_depth_1": 100.0,
                    "bid_depth_5": 100.0,
                    "ask_depth_5": 100.0,
                }
            ]
        ).to_parquet(feature_file, index=False)
        feature_files.append(feature_file)
        resolution_rows.append(
            {
                "market_id": market_id,
                "resolved_at": base_time + pd.Timedelta(hours=1),
                "winning_asset_id": token_id,
                "winning_outcome": "YES",
                "fees_enabled_market": True,
            }
        )

    resolution_path = prepared_root / "resolution" / "resolution_frame.parquet"
    resolution_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resolution_rows).to_parquet(resolution_path, index=False)

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "output_files": [],
                        "feature_output_files": [str(path) for path in feature_files],
                    }
                ],
                "resolution_output_file": str(resolution_path),
            }
        ),
        encoding="utf-8",
    )

    runner = BacktestRunner(storage_path=prepared_root)

    class _StubProgress:
        def __init__(self) -> None:
            self.updates = 0
            self.closed = False

        def update(self, n: int = 1) -> None:
            self.updates += n

        def set_postfix(self, ordered_dict=None, *, refresh: bool = True, **kwargs):
            return None

        def close(self) -> None:
            self.closed = True

    progress_stub = _StubProgress()

    def _fake_progress(*, enabled: bool, total: int, desc: str, unit: str):
        if enabled and desc == "backtest: market batches":
            assert total == 2
            assert unit == "batch"
            return progress_stub
        return None

    monkeypatch.setattr(runner, "_make_progress_bar", _fake_progress)

    def always_long(features: pd.DataFrame) -> pd.DataFrame:
        frame = features.copy()
        frame["signal"] = 1
        return frame

    runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=always_long,
        strategy_name="always_long",
        config=BacktestConfig(
            mode="tolerant",
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            enable_progress_bars=True,
            metrics_logging_enabled=False,
        ),
        market_batch_size=2,
    )

    assert progress_stub.updates == 2
    assert progress_stub.closed
