from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from backtester.loaders.market_events import (
    MarketEventsLoadDeps,
    MarketEventsLoadRequest,
    PyArrowModules,
    load_market_events,
)
from utils.dataframes import filter_by_time_range, prepare_timestamp_index

if TYPE_CHECKING:
    from pathlib import Path


def _deps() -> MarketEventsLoadDeps:
    return MarketEventsLoadDeps(
        load_condition_ids_for_slug_prefix_fn=lambda _: set(),
        normalize_market_events_schema_fn=lambda frame: frame,
        prepare_timestamp_index_fn=prepare_timestamp_index,
        filter_by_time_range_fn=filter_by_time_range,
    )


def _write_events(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_load_market_events_uses_manifest_file_list(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    listed_file_a = prepared_root / "2024-01-01" / "m1" / "a.parquet"
    listed_file_b = prepared_root / "2024-01-01" / "m2" / "b.parquet"
    not_listed_file = prepared_root / "2024-01-01" / "m3" / "c.parquet"

    base = datetime(2024, 1, 1, tzinfo=UTC)
    _write_events(
        listed_file_a,
        [{"ts_event": base, "event_type": "book", "market_id": "m1", "data": {"asset_id": "m1"}}],
    )
    _write_events(
        listed_file_b,
        [{"ts_event": base, "event_type": "book", "market_id": "m2", "data": {"asset_id": "m2"}}],
    )
    _write_events(
        not_listed_file,
        [{"ts_event": base, "event_type": "book", "market_id": "m3", "data": {"asset_id": "m3"}}],
    )

    manifest_path = prepared_root / "manifest.json"
    manifest_payload = {
        "files": [
            {"output_files": [str(listed_file_a)]},
            {"output_files": [str(listed_file_b)]},
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
        recursive_scan=True,
    )

    loaded = load_market_events(
        request,
        deps=_deps(),
        modules=PyArrowModules(pa=None, pc=None, ds=None, pq=None),
    )

    assert len(loaded) == 2
    assert set(loaded["market_id"].astype(str).tolist()) == {"m1", "m2"}


def test_load_market_events_recurses_when_top_level_empty(tmp_path: Path) -> None:
    prepared_root = tmp_path / "prepared"
    nested_file = prepared_root / "2024-01-01" / "m1" / "nested.parquet"

    _write_events(
        nested_file,
        [
            {
                "ts_event": datetime(2024, 1, 1, tzinfo=UTC),
                "event_type": "book",
                "market_id": "m1",
                "data": {"asset_id": "m1"},
            }
        ],
    )

    request = MarketEventsLoadRequest(
        market_path=prepared_root,
        start=None,
        end=None,
        limit_files=None,
        max_rows_per_file=None,
        market_slug_prefix=None,
        is_pmxt_mode=False,
        mapping_path=tmp_path / "mapping",
        recursive_scan=True,
    )

    loaded = load_market_events(
        request,
        deps=_deps(),
        modules=PyArrowModules(pa=None, pc=None, ds=None, pq=None),
    )

    assert len(loaded) == 1
    assert loaded["market_id"].iloc[0] == "m1"


def test_load_market_events_time_pushdown_honors_window(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pc = pytest.importorskip("pyarrow.compute")
    ds = pytest.importorskip("pyarrow.dataset")
    pq = pytest.importorskip("pyarrow.parquet")

    market_path = tmp_path / "pmxt"
    market_path.mkdir(parents=True, exist_ok=True)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _write_events(
        market_path / "events.parquet",
        [
            {
                "ts_event": base,
                "event_type": "book",
                "market_id": "m1",
                "data": {"asset_id": "m1"},
            },
            {
                "ts_event": base.replace(hour=1),
                "event_type": "book",
                "market_id": "m1",
                "data": {"asset_id": "m1"},
            },
            {
                "ts_event": base.replace(hour=2),
                "event_type": "book",
                "market_id": "m1",
                "data": {"asset_id": "m1"},
            },
        ],
    )

    request = MarketEventsLoadRequest(
        market_path=market_path,
        start=base.replace(hour=1),
        end=base.replace(hour=2),
        limit_files=None,
        max_rows_per_file=None,
        market_slug_prefix=None,
        is_pmxt_mode=False,
        mapping_path=tmp_path / "mapping",
    )

    loaded = load_market_events(
        request,
        deps=_deps(),
        modules=PyArrowModules(pa=pa, pc=pc, ds=ds, pq=pq),
    )

    assert len(loaded) == 2
    assert loaded.index.min() >= pd.Timestamp(base.replace(hour=1))
    assert loaded.index.max() <= pd.Timestamp(base.replace(hour=2))


def test_load_market_events_logs_structured_metrics(tmp_path: Path, caplog) -> None:
    market_path = tmp_path / "pmxt"
    market_path.mkdir(parents=True, exist_ok=True)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    _write_events(
        market_path / "events.parquet",
        [
            {
                "ts_event": base,
                "event_type": "book",
                "market_id": "m1",
                "data": {"asset_id": "m1"},
            }
        ],
    )

    request = MarketEventsLoadRequest(
        market_path=market_path,
        start=None,
        end=None,
        limit_files=None,
        max_rows_per_file=None,
        market_slug_prefix=None,
        is_pmxt_mode=False,
        mapping_path=tmp_path / "mapping",
    )

    with caplog.at_level(logging.INFO, logger="backtester.loaders.market_events"):
        loaded = load_market_events(
            request,
            deps=_deps(),
            modules=PyArrowModules(pa=None, pc=None, ds=None, pq=None),
        )

    assert len(loaded) == 1
    metric_lines = [
        record.message
        for record in caplog.records
        if record.message.startswith("Market events loader metrics:")
    ]
    assert metric_lines

    payload_raw = metric_lines[-1].split("Market events loader metrics:", maxsplit=1)[1].strip()
    payload = json.loads(payload_raw)
    assert payload["files_discovered"] >= 1
    assert payload["files_selected"] >= 1
    assert "elapsed_seconds" in payload
