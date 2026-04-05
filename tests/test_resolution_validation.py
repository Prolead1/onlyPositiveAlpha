from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from backtester.runner import BacktestRunner

if TYPE_CHECKING:
    from pathlib import Path


def _make_runner(tmp_path: Path) -> BacktestRunner:
    storage = tmp_path / "pmxt"
    storage.mkdir(parents=True, exist_ok=True)
    return BacktestRunner(storage)


def test_resolution_validation_fails_on_event_mapping_mismatch(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_id = "cond_123"
    token_yes = "token_yes"

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
                "ts_event": base_time + timedelta(seconds=10),
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

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_payload = {
        "sample-slug": {
            "conditionId": market_id,
            "resolvedAt": "2024-01-01T00:00:10Z",
            "winningAssetId": "token_no",
            "winningOutcome": "No",
            "clobTokenIds": [token_yes, "token_no"],
            "outcomePrices": ["0", "1"],
            "feesEnabledMarket": True,
        }
    }
    (mapping_dir / "gamma_updown_markets_2026-02-21.json").write_text(
        json.dumps(mapping_payload),
        encoding="utf-8",
    )

    resolutions = runner.load_resolution_frame_with_fallback(
        market_events,
        mapping_dir=mapping_dir,
    )

    with pytest.raises(ValueError, match="winner mismatch event/mapping"):
        runner.validate_resolution_consistency(
            market_events,
            resolutions,
            mapping_dir=mapping_dir,
        )


def test_resolution_validation_fails_on_terminal_evidence_mismatch(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_id = "cond_terminal"
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
                    "bids": [{"price": "0.01", "size": "120"}],
                    "asks": [{"price": "0.02", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_no,
                "data": {
                    "asset_id": token_no,
                    "bids": [{"price": "0.98", "size": "120"}],
                    "asks": [{"price": "0.99", "size": "100"}],
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

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    resolutions = runner.load_resolution_frame_with_fallback(
        market_events,
        mapping_dir=mapping_dir,
    )

    with pytest.raises(ValueError, match="terminal evidence mismatch"):
        runner.validate_resolution_consistency(
            market_events,
            resolutions,
            mapping_dir=mapping_dir,
            confidence_threshold=0.95,
        )


def test_resolution_validation_repairs_missing_winner_and_writes_back(tmp_path: Path) -> None:
    runner = _make_runner(tmp_path)

    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    market_id = "cond_repair"
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
                    "winning_asset_id": None,
                    "winning_outcome": "Yes",
                },
            },
        ]
    ).set_index("ts_event")

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = mapping_dir / "gamma_updown_markets_2024-01-01.json"
    mapping_path.write_text(
        json.dumps(
            {
                "sample-slug": {
                    "conditionId": market_id,
                    "resolvedAt": "2024-01-01T00:00:05Z",
                    "winningAssetId": None,
                    "winningOutcome": "Yes",
                    "clobTokenIds": [token_yes, token_no],
                    "outcomePrices": ["0", "1"],
                    "feesEnabledMarket": True,
                }
            }
        ),
        encoding="utf-8",
    )

    resolution_frame, diagnostics = runner.load_and_validate_resolution(
        market_events,
        mapping_dir=mapping_dir,
    )

    assert resolution_frame.loc[market_id, "winning_asset_id"] == token_yes
    assert diagnostics.loc[0, "terminal_winner"] == token_yes

    updated_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert updated_payload["sample-slug"]["winningAssetId"] == token_yes
