from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.prepare_market_backtest_dataset as prepare_module
from scripts.prepare_market_backtest_dataset import (
    DEFAULT_MAX_WORKERS,
    PrepareOptions,
    _build_pending_source_priority_queue,
    _compute_feature_frames,
    _to_json_compatible,
    collect_market_ids_from_mapping,
    prepare_market_backtest_dataset,
    resolve_feature_queue_size,
    resolve_feature_worker_count,
    resolve_scan_batch_size,
    resolve_scan_engine,
    resolve_worker_count,
)


def test_collect_market_ids_from_mapping_uses_slug_prefix(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_payload = {
        "btc-updown-15m-1704067200": {
            "conditionId": "m_btc",
        },
        "eth-updown-15m-1704067200": {
            "conditionId": "m_eth",
        },
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    market_ids = collect_market_ids_from_mapping(mapping_dir, ["btc-updown-15m"])
    assert market_ids == {"m_btc"}


def test_prepare_market_backtest_dataset_writes_market_isolated_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "pmxt"
    source_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    mapping_payload = {
        "btc-updown-15m-1704067200": {
            "conditionId": "m_btc",
        },
        "eth-updown-15m-1704067200": {
            "conditionId": "m_eth",
        },
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    events = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_btc",
                "token_id": "yes_btc",
                "data": {
                    "asset_id": "yes_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_btc",
                "token_id": "no_btc",
                "data": {
                    "asset_id": "no_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 1, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_eth",
                "token_id": "yes_eth",
                "data": {
                    "asset_id": "yes_eth",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
        ]
    )
    events.to_parquet(source_dir / "polymarket_orderbook_2024-01-01T00.parquet", index=False)

    options = PrepareOptions(
        overwrite=False,
        dry_run=False,
        max_workers=1,
        compression="zstd",
        compression_level=3,
    )
    manifest_path = tmp_path / "manifest.json"

    manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["btc-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    assert manifest["required_market_count"] == 1
    assert manifest["rows_kept"] == 2
    assert manifest["feature_rows"] > 0
    assert manifest["output_files_written"] == 0
    assert manifest["feature_output_files_written"] == 1

    event_files = [
        path
        for path in output_dir.rglob("*.parquet")
        if "features" not in path.parts
    ]
    feature_files = list((output_dir / "runs" / "btc-updown-15m" / "features").rglob("*.parquet"))
    assert len(event_files) == 0
    assert len(feature_files) == 1

    written_features = pd.read_parquet(feature_files[0])
    assert {"ts_event", "market_id", "token_id", "spread", "mid_price", "imbalance_1"}.issubset(
        written_features.columns
    )
    assert written_features["market_id"].astype(str).eq("m_btc").all()


def test_prepare_market_backtest_dataset_reuses_existing_manifest_entries(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "pmxt"
    source_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    mapping_payload = {
        "btc-updown-15m-1704067200": {
            "conditionId": "m_btc",
        }
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    events = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_btc",
                "token_id": "yes_btc",
                "data": {
                    "asset_id": "yes_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_btc",
                "token_id": "no_btc",
                "data": {
                    "asset_id": "no_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            }
        ]
    )
    source_file = source_dir / "polymarket_orderbook_2024-01-01T00.parquet"
    events.to_parquet(source_file, index=False)

    options = PrepareOptions(
        overwrite=False,
        dry_run=False,
        max_workers=1,
        compression="zstd",
        compression_level=3,
    )
    manifest_path = tmp_path / "manifest.json"

    first_manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["btc-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    first_feature_output = Path(first_manifest["files"][0]["feature_output_files"][0])
    first_feature_mtime = first_feature_output.stat().st_mtime_ns

    second_manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["btc-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    assert second_manifest["output_files_written"] == 0
    assert second_manifest["feature_output_files_written"] == 0
    assert second_manifest["output_files_skipped_existing"] == 0
    assert second_manifest["feature_output_files_skipped_existing"] == 1
    assert second_manifest["files"][0]["reused_from_manifest"] is True
    assert second_manifest["files"][0]["output_files"] == []
    assert second_manifest["files"][0]["feature_output_files"] == [str(first_feature_output)]
    assert first_feature_output.stat().st_mtime_ns == first_feature_mtime


def test_prepare_market_backtest_dataset_reuses_feature_cache_by_market_id(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "pmxt"
    source_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    mapping_payload = {
        "btc-updown-15m-1704067200": {
            "conditionId": "m_btc",
        }
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    events = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_btc",
                "token_id": "yes_btc",
                "data": {
                    "asset_id": "yes_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
        ]
    )
    source_file_a = source_dir / "polymarket_orderbook_2024-01-01T00.parquet"
    source_file_b = source_dir / "polymarket_orderbook_2024-01-01T00_alt.parquet"
    events.to_parquet(source_file_a, index=False)

    options = PrepareOptions(
        overwrite=False,
        dry_run=False,
        max_workers=1,
        compression="zstd",
        compression_level=3,
    )
    manifest_path = tmp_path / "manifest.json"

    first_manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["btc-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    first_feature_output = Path(first_manifest["files"][0]["feature_output_files"][0])
    first_feature_mtime = first_feature_output.stat().st_mtime_ns

    events.to_parquet(source_file_b, index=False)

    second_manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["btc-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    assert second_manifest["feature_output_files_written"] == 0
    assert second_manifest["feature_output_files_skipped_existing"] == 2
    assert second_manifest["files"][0]["reused_from_manifest"] is True
    assert second_manifest["files"][1]["reused_from_manifest"] is False
    assert second_manifest["files"][1]["feature_output_files"] == [str(first_feature_output)]
    assert first_feature_output.stat().st_mtime_ns == first_feature_mtime


def test_main_returns_interrupt_exit_code(monkeypatch) -> None:
    def _raise_keyboard_interrupt(**_: object) -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        prepare_module,
        "prepare_market_backtest_dataset",
        _raise_keyboard_interrupt,
    )

    exit_code = prepare_module.main([])
    assert exit_code == 130


def test_resolve_worker_count_auto_and_manual_modes() -> None:
    assert resolve_worker_count(max_workers=0, file_count=1) == 1
    assert resolve_worker_count(max_workers=-1, file_count=10) == min(10, DEFAULT_MAX_WORKERS)
    assert resolve_worker_count(max_workers=3, file_count=10) == 3
    assert resolve_worker_count(max_workers=999, file_count=4) == 4


def test_resolve_feature_worker_count_auto_and_manual_modes() -> None:
    assert resolve_feature_worker_count(feature_workers=0, group_count=1) == 1
    assert resolve_feature_worker_count(feature_workers=-1, group_count=10) == 1
    assert resolve_feature_worker_count(feature_workers=3, group_count=10) == 3
    assert resolve_feature_worker_count(feature_workers=999, group_count=4) == 4


def test_resolve_feature_queue_size_auto_and_manual_modes() -> None:
    assert resolve_feature_queue_size(feature_queue_size=0, feature_workers=1) == 1
    assert resolve_feature_queue_size(feature_queue_size=0, feature_workers=2) == 4
    assert resolve_feature_queue_size(feature_queue_size=1, feature_workers=4) == 4
    assert resolve_feature_queue_size(feature_queue_size=12, feature_workers=3) == 12


def test_resolve_scan_engine_supported_and_fallback_modes() -> None:
    assert resolve_scan_engine(scan_engine="auto") == "auto"
    assert resolve_scan_engine(scan_engine="PYARROW") == "pyarrow"
    assert resolve_scan_engine(scan_engine="duckdb") == "duckdb"
    assert resolve_scan_engine(scan_engine="unknown") == "auto"


def test_resolve_scan_batch_size_auto_and_manual_modes() -> None:
    assert resolve_scan_batch_size(scan_batch_size=0) == 250_000
    assert resolve_scan_batch_size(scan_batch_size=-1) == 250_000
    assert resolve_scan_batch_size(scan_batch_size=1_000) == 10_000
    assert resolve_scan_batch_size(scan_batch_size=500_000) == 500_000


def test_compute_feature_frames_parallel_matches_serial() -> None:
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    group_a = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m_a",
                "token_id": "yes_a",
                "data": {
                    "asset_id": "yes_a",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m_a",
                "token_id": "no_a",
                "data": {
                    "asset_id": "no_a",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
        ]
    )
    group_b = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m_b",
                "token_id": "yes_b",
                "data": {
                    "asset_id": "yes_b",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "m_b",
                "token_id": "no_b",
                "data": {
                    "asset_id": "no_b",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
        ]
    )
    grouped_frames = [
        (0, "2024-01-01", "m_a", group_a),
        (1, "2024-01-01", "m_b", group_b),
    ]

    serial_results = _compute_feature_frames(
        grouped_frames=grouped_frames,
        feature_workers=1,
        feature_queue_size=1,
    )
    parallel_results = _compute_feature_frames(
        grouped_frames=grouped_frames,
        feature_workers=2,
        feature_queue_size=2,
    )

    assert [item[:3] for item in serial_results] == [item[:3] for item in parallel_results]
    for serial_item, parallel_item in zip(serial_results, parallel_results, strict=True):
        serial_frame = serial_item[3].sort_values(["ts_event", "token_id"]).reset_index(drop=True)
        parallel_frame = (
            parallel_item[3]
            .sort_values(["ts_event", "token_id"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(serial_frame, parallel_frame)


def test_build_pending_source_priority_queue_orders_smallest_cost_first(tmp_path: Path) -> None:
    small_source = tmp_path / "polymarket_orderbook_small.parquet"
    large_source = tmp_path / "polymarket_orderbook_large.parquet"
    small_source.write_bytes(b"x" * 16)
    large_source.write_bytes(b"x" * 256)

    per_file_market_ids = {
        str(small_source): {"m1", "m2", "m3"},
        str(large_source): {"m1"},
    }
    queue = _build_pending_source_priority_queue(
        pending_source_files=[small_source, large_source],
        per_file_market_ids=per_file_market_ids,
    )

    _, _, first = queue[0]
    assert first == small_source


def test_prepare_market_backtest_dataset_normalizes_update_type_only_source(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "pmxt"
    source_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    mapping_payload = {
        "btc-updown-15m-1704067200": {
            "conditionId": "m_btc",
        }
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    events = pd.DataFrame(
        [
            {
                "timestamp_received": datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
                "timestamp_created_at": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "market_id": "m_btc",
                "update_type": "book_snapshot",
                "data": {
                    "token_id": "yes_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "timestamp_received": datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
                "timestamp_created_at": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "market_id": "m_btc",
                "update_type": "book_snapshot",
                "data": {
                    "token_id": "no_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            }
        ]
    )
    events.to_parquet(source_dir / "polymarket_orderbook_2024-01-01T00.parquet", index=False)

    options = PrepareOptions(
        overwrite=False,
        dry_run=False,
        max_workers=1,
        compression="zstd",
        compression_level=3,
    )
    manifest_path = tmp_path / "manifest.json"

    manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["btc-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    assert manifest["rows_kept"] == 2
    assert manifest["feature_rows"] > 0
    event_files = [
        path
        for path in output_dir.rglob("*.parquet")
        if "features" not in path.parts
    ]
    feature_files = list((output_dir / "runs" / "btc-updown-15m" / "features").rglob("*.parquet"))
    assert len(event_files) == 0
    assert len(feature_files) == 1

    written_features = pd.read_parquet(feature_files[0])
    assert "spread" in written_features.columns
    assert written_features["market_id"].astype(str).eq("m_btc").all()


def test_prepare_market_backtest_dataset_uses_slug_hour_expectations_to_prune_files(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "pmxt"
    source_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    mapping_payload = {
        "btc-updown-15m-1704067200": {
            "conditionId": "m_btc",
        }
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    in_window = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_btc",
                "token_id": "yes_btc",
                "data": {
                    "asset_id": "yes_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_btc",
                "token_id": "no_btc",
                "data": {
                    "asset_id": "no_btc",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
        ]
    )
    in_window.to_parquet(source_dir / "polymarket_orderbook_2024-01-01T00.parquet", index=False)

    out_of_window = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_eth",
                "token_id": "yes_eth",
                "data": {
                    "asset_id": "yes_eth",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            }
        ]
    )
    out_of_window.to_parquet(
        source_dir / "polymarket_orderbook_2024-01-01T12.parquet",
        index=False,
    )

    options = PrepareOptions(
        overwrite=False,
        dry_run=False,
        max_workers=1,
        compression="zstd",
        compression_level=3,
    )
    manifest_path = tmp_path / "manifest.json"

    manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["btc-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    assert manifest["source_file_count"] == 1
    assert manifest["rows_kept"] == 2


def test_prepare_market_backtest_dataset_applies_deterministic_max_markets(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "pmxt"
    source_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    mapping_payload = {
        "alpha-updown-15m-1704067200": {"conditionId": "m_01"},
        "alpha-updown-15m-1704067300": {"conditionId": "m_02"},
        "alpha-updown-15m-1704067400": {"conditionId": "m_03"},
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    events = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_01",
                "token_id": "yes_01",
                "data": {
                    "asset_id": "yes_01",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_01",
                "token_id": "no_01",
                "data": {
                    "asset_id": "no_01",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_02",
                "token_id": "yes_02",
                "data": {
                    "asset_id": "yes_02",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_02",
                "token_id": "no_02",
                "data": {
                    "asset_id": "no_02",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_03",
                "token_id": "yes_03",
                "data": {
                    "asset_id": "yes_03",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_03",
                "token_id": "no_03",
                "data": {
                    "asset_id": "no_03",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
        ]
    )
    events.to_parquet(source_dir / "polymarket_orderbook_2024-01-01T00.parquet", index=False)

    options = PrepareOptions(
        overwrite=False,
        dry_run=False,
        max_workers=1,
        compression="zstd",
        compression_level=3,
    )
    manifest_path = tmp_path / "manifest.json"

    manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["alpha-updown-15m"],
        max_markets=2,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    assert manifest["selected_mapping_market_ids"] == ["m_01", "m_02"]
    assert manifest["required_market_ids"] == ["m_01", "m_02"]
    assert manifest["selected_mapping_market_count"] == 2


def test_prepare_market_backtest_dataset_reuses_cache_when_max_markets_changes(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "pmxt"
    source_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    mapping_payload = {
        "alpha-updown-15m-1704067200": {"conditionId": "m_01"},
        "alpha-updown-15m-1704067300": {"conditionId": "m_02"},
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    events = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_01",
                "token_id": "yes_01",
                "data": {
                    "asset_id": "yes_01",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_01",
                "token_id": "no_01",
                "data": {
                    "asset_id": "no_01",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_02",
                "token_id": "yes_02",
                "data": {
                    "asset_id": "yes_02",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_02",
                "token_id": "no_02",
                "data": {
                    "asset_id": "no_02",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
        ]
    )
    events.to_parquet(source_dir / "polymarket_orderbook_2024-01-01T00.parquet", index=False)

    options = PrepareOptions(
        overwrite=False,
        dry_run=False,
        max_workers=1,
        compression="zstd",
        compression_level=3,
    )
    manifest_path = tmp_path / "manifest.json"

    first_manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["alpha-updown-15m"],
        max_markets=1,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )
    first_feature_output = Path(first_manifest["files"][0]["feature_output_files"][0])
    first_feature_mtime = first_feature_output.stat().st_mtime_ns

    second_manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["alpha-updown-15m"],
        max_markets=2,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=manifest_path,
    )

    assert second_manifest["required_market_ids"] == ["m_01", "m_02"]
    assert second_manifest["feature_output_files_written"] == 1
    assert second_manifest["feature_output_files_skipped_existing"] == 1
    assert first_feature_output.stat().st_mtime_ns == first_feature_mtime

    combined_feature_outputs = {
        item
        for entry in second_manifest["files"]
        for item in entry["feature_output_files"]
    }
    assert str(first_feature_output) in combined_feature_outputs
    assert any("/m_02/" in path for path in combined_feature_outputs)


def test_prepare_market_backtest_dataset_creates_isolated_prefix_runs(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "pmxt"
    source_dir.mkdir(parents=True, exist_ok=True)
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "prepared"

    mapping_payload = {
        "alpha-updown-15m-1704067200": {"conditionId": "m_alpha"},
        "beta-updown-15m-1704067200": {"conditionId": "m_beta"},
    }
    (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    events = pd.DataFrame(
        [
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_alpha",
                "token_id": "yes_alpha",
                "data": {
                    "asset_id": "yes_alpha",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_alpha",
                "token_id": "no_alpha",
                "data": {
                    "asset_id": "no_alpha",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_beta",
                "token_id": "yes_beta",
                "data": {
                    "asset_id": "yes_beta",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m_beta",
                "token_id": "no_beta",
                "data": {
                    "asset_id": "no_beta",
                    "bids": [{"price": "0.49", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "100"}],
                },
            },
        ]
    )
    events.to_parquet(source_dir / "polymarket_orderbook_2024-01-01T00.parquet", index=False)

    options = PrepareOptions(
        overwrite=False,
        dry_run=False,
        max_workers=1,
        compression="zstd",
        compression_level=3,
    )

    alpha_manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["alpha-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=tmp_path / "alpha_manifest.json",
    )
    beta_manifest = prepare_market_backtest_dataset(
        source_dir=source_dir,
        mapping_dir=mapping_dir,
        output_dir=output_dir,
        market_ids_file=None,
        slug_prefixes=["beta-updown-15m"],
        max_markets=0,
        start_date=None,
        end_date=None,
        options=options,
        manifest_path=tmp_path / "beta_manifest.json",
    )

    alpha_feature_paths = {
        Path(item)
        for entry in alpha_manifest["files"]
        for item in entry["feature_output_files"]
    }
    beta_feature_paths = {
        Path(item)
        for entry in beta_manifest["files"]
        for item in entry["feature_output_files"]
    }

    assert alpha_feature_paths
    assert beta_feature_paths
    assert all("runs/alpha-updown-15m" in str(path) for path in alpha_feature_paths)
    assert all("runs/beta-updown-15m" in str(path) for path in beta_feature_paths)
    assert alpha_feature_paths.isdisjoint(beta_feature_paths)


def test_to_json_compatible_converts_array_payloads_to_lists() -> None:
    payload = {
        "token_id": "yes_btc",
        "bids": np.array([[0.45, 100.0]]),
        "asks": np.array([[0.55, 100.0]]),
    }

    converted = _to_json_compatible(payload)
    assert isinstance(converted, dict)
    assert isinstance(converted["bids"], list)
    assert isinstance(converted["asks"], list)
