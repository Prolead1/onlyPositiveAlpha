from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_RESOLUTION_COLUMNS = (
    "market_id",
    "resolved_at",
    "winning_asset_id",
    "winning_outcome",
    "fees_enabled_market",
)


@dataclass(frozen=True)
class ResolutionMappingDeps:
    """Dependency callbacks required to derive mapping-based resolution rows."""

    parse_clob_token_ids_fn: Callable[[object], list[str]]
    parse_outcome_prices_fn: Callable[[object], list[float]]
    select_winning_asset_id_fn: Callable[[list[str], list[float]], str | None]


def _empty_resolution_frame(columns: tuple[str, ...] = DEFAULT_RESOLUTION_COLUMNS) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns)).set_index("market_id")


def load_resolution_frame_from_events(
    market_events: pd.DataFrame,
    *,
    resolution_columns: tuple[str, ...] = DEFAULT_RESOLUTION_COLUMNS,
) -> pd.DataFrame:
    """Extract one resolution row per market from market events."""
    if market_events.empty:
        return _empty_resolution_frame(resolution_columns)

    required_cols = {"market_id", "event_type"}
    missing = [col for col in required_cols if col not in market_events.columns]
    if missing:
        msg = f"Missing required columns for resolution lookup: {missing}"
        raise ValueError(msg)

    resolution_events = market_events[market_events["event_type"] == "market_resolved"].copy()
    if resolution_events.empty:
        return _empty_resolution_frame(resolution_columns)

    rows: list[dict[str, object]] = []
    for idx, row in resolution_events.iterrows():
        payload = row.get("data", {})
        if not isinstance(payload, dict):
            continue
        market_id = row.get("market_id")
        if not market_id:
            continue
        rows.append(
            {
                "market_id": str(market_id),
                "resolved_at": idx,
                "winning_asset_id": payload.get("winning_asset_id"),
                "winning_outcome": payload.get("winning_outcome"),
                "fees_enabled_market": bool(
                    payload.get("fees_enabled_market", payload.get("feesEnabledMarket", True))
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_resolution_frame(resolution_columns)
    frame = frame.sort_values(["market_id", "resolved_at"]).drop_duplicates(
        subset=["market_id"], keep="first"
    )
    return frame.set_index("market_id")


def load_condition_entry_map(
    mapping_dir: str | Path,
) -> dict[str, tuple[Path, str, dict[str, object]]]:
    """Build conditionId -> (shard path, slug, entry) mapping from shards."""
    mapping_path = Path(mapping_dir)
    if not mapping_path.exists():
        return {}

    condition_to_entry: dict[str, tuple[Path, str, dict[str, object]]] = {}
    for shard in sorted(mapping_path.glob("gamma_updown_markets_*.json")):
        try:
            payload = json.loads(shard.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        for slug, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            condition_id = entry.get("conditionId")
            if condition_id:
                condition_to_entry[str(condition_id).lower()] = (shard, str(slug), entry)

    return condition_to_entry


def build_resolution_row_from_mapping_entry(
    market_id: str,
    entry: dict[str, object],
    *,
    deps: ResolutionMappingDeps,
) -> dict[str, object] | None:
    """Build a normalized resolution row from one mapping entry."""
    resolved_raw = entry.get("resolvedAt") or entry.get("endDate")
    if not resolved_raw:
        return None

    resolved_at = pd.to_datetime(str(resolved_raw), utc=True, errors="coerce")
    if pd.isna(resolved_at):
        return None

    _ = deps.parse_clob_token_ids_fn(entry.get("clobTokenIds"))
    raw_outcomes = entry.get("outcomePrices")
    _ = deps.parse_outcome_prices_fn(raw_outcomes)

    winning_asset_id = entry.get("winningAssetId")
    if winning_asset_id is not None:
        winning_asset_id = str(winning_asset_id)

    winning_outcome = entry.get("winningOutcome", raw_outcomes)
    return {
        "market_id": market_id,
        "resolved_at": resolved_at,
        "winning_asset_id": winning_asset_id,
        "winning_outcome": winning_outcome,
        "fees_enabled_market": bool(
            entry.get("feesEnabledMarket", entry.get("feesEnabled", True))
        ),
    }


def load_resolution_frame_from_mapping(
    market_ids: list[str],
    *,
    mapping_dir: str | Path,
    deps: ResolutionMappingDeps,
    resolution_columns: tuple[str, ...] = DEFAULT_RESOLUTION_COLUMNS,
) -> pd.DataFrame:
    """Build resolution table from mapping files for requested markets."""
    condition_to_entry = load_condition_entry_map(mapping_dir)
    if not condition_to_entry:
        return _empty_resolution_frame(resolution_columns)

    rows: list[dict[str, object]] = []
    for market_id in sorted({str(mid) for mid in market_ids if mid is not None}):
        slug_and_entry = condition_to_entry.get(market_id.lower())
        if slug_and_entry is None:
            continue

        _, _, entry = slug_and_entry
        row = build_resolution_row_from_mapping_entry(market_id, entry, deps=deps)
        if row is not None:
            rows.append(row)

    if not rows:
        return _empty_resolution_frame(resolution_columns)

    frame = pd.DataFrame(rows).drop_duplicates(subset=["market_id"], keep="first")
    return frame.set_index("market_id")


def load_resolution_frame_with_fallback(
    market_events: pd.DataFrame,
    *,
    load_resolution_frame_from_events_fn: Callable[[pd.DataFrame], pd.DataFrame],
    load_resolution_frame_from_mapping_fn: Callable[[list[str]], pd.DataFrame],
) -> pd.DataFrame:
    """Resolve market outcomes from events first, then mapping fallback."""
    event_resolutions = load_resolution_frame_from_events_fn(market_events)
    market_ids = market_events["market_id"].dropna().astype(str).unique().tolist()
    mapping_resolutions = load_resolution_frame_from_mapping_fn(market_ids)

    if event_resolutions.empty:
        return mapping_resolutions
    if mapping_resolutions.empty:
        return event_resolutions

    merged = event_resolutions.combine_first(mapping_resolutions)
    return merged.sort_values("resolved_at")
