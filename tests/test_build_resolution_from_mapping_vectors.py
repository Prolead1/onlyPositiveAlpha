from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

from scripts.build_resolution_from_mapping_vectors import (
    RESOLUTION_COLUMNS,
    build_resolution_frame_from_mapping_vectors,
    collect_market_ids_from_features,
    combine_market_filters,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_resolution_frame_from_mapping_vectors_uses_binary_vectors(tmp_path: Path) -> None:
    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "btc-updown-5m-1": {
            "conditionId": "market_a",
            "clobTokenIds": '["token_a_up", "token_a_down"]',
            "winningOutcome": ["1", "0"],
            "resolvedAt": "2026-02-21T10:00:00Z",
            "feesEnabledMarket": True,
            "winningAssetId": "token_a_up",
        },
        "btc-updown-5m-2": {
            "conditionId": "market_b",
            "clobTokenIds": ["token_b_up", "token_b_down"],
            "winningOutcome": ["0", "1"],
            "resolvedAt": "2026-02-21T11:00:00Z",
            "feesEnabledMarket": False,
            "winningAssetId": "token_b_down",
        },
        "btc-updown-5m-3": {
            "conditionId": "market_c",
            "clobTokenIds": ["token_c_up", "token_c_down"],
            "winningOutcome": ["0.4", "0.6"],
            "resolvedAt": "2026-02-21T12:00:00Z",
        },
        "btc-updown-5m-4": {
            "conditionId": "market_d",
            "clobTokenIds": ["token_d_up", "token_d_mid", "token_d_down"],
            "winningOutcome": ["1", "0"],
            "resolvedAt": "2026-02-21T13:00:00Z",
        },
    }
    (mapping_dir / "gamma_updown_markets_2026-02-21.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    frame, stats = build_resolution_frame_from_mapping_vectors(
        mapping_dir=mapping_dir,
        required_market_ids={"market_a", "market_b", "market_c", "market_d"},
    )

    assert frame.columns.tolist() == RESOLUTION_COLUMNS
    assert len(frame) == 2
    winners = dict(zip(frame["market_id"], frame["winning_asset_id"], strict=True))
    assert winners == {
        "market_a": "token_a_up",
        "market_b": "token_b_down",
    }

    outcomes = dict(zip(frame["market_id"], frame["winning_outcome"], strict=True))
    assert outcomes["market_a"] == '["1", "0"]'
    assert outcomes["market_b"] == '["0", "1"]'

    assert frame["settlement_source"].eq("mapping_binary_vector").all()
    assert frame["settlement_confidence"].eq(1.0).all()
    assert frame["fees_enabled_market"].tolist() == [True, False]

    assert stats.entries_non_binary_outcome_vector == 1
    assert stats.entries_non_binary_token_count == 1
    assert stats.gamma_winner_populated == 2
    assert stats.gamma_winner_mismatches_vector == 0


def test_collect_market_ids_from_features_and_filter_intersection(tmp_path: Path) -> None:
    features_root = tmp_path / "features"
    (features_root / "2026-02-21" / "market_a").mkdir(parents=True, exist_ok=True)
    (features_root / "2026-02-21" / "market_b").mkdir(parents=True, exist_ok=True)
    (features_root / "2026-02-22" / "market_b").mkdir(parents=True, exist_ok=True)
    (features_root / "2026-02-22" / "market_c").mkdir(parents=True, exist_ok=True)

    feature_market_ids = collect_market_ids_from_features(features_root)
    assert feature_market_ids == {"market_a", "market_b", "market_c"}

    combined = combine_market_filters([
        {"market_a", "market_b", "market_x"},
        feature_market_ids,
    ])
    assert combined == {"market_a", "market_b"}

    no_filter = combine_market_filters([set(), set()])
    assert no_filter is None

    empty_frame, _ = build_resolution_frame_from_mapping_vectors(
        mapping_dir=tmp_path / "missing_mapping",
        required_market_ids=combined,
    )
    assert isinstance(empty_frame, pd.DataFrame)
    assert empty_frame.empty
