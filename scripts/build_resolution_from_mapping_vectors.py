"""Build backtester resolution parquet from Gamma mapping binary outcome vectors.

This script infers winners strictly from mapping vectors:
- [1,0] -> first CLOB token wins
- [0,1] -> second CLOB token wins

It writes a backtester-compatible `resolution_frame.parquet` with provenance fields.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Allow direct script execution from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtester.resolution.parsing import parse_clob_token_ids

DEFAULT_MAPPING_DIR = Path("data") / "cached" / "mapping"
DEFAULT_RESOLUTION_FILENAME = "resolution_frame.parquet"
BINARY_VECTOR_LENGTH = 2
FLOAT_EQ_TOLERANCE = 1e-9
RESOLUTION_COLUMNS = [
    "market_id",
    "resolved_at",
    "winning_asset_id",
    "winning_outcome",
    "fees_enabled_market",
    "settlement_source",
    "settlement_confidence",
    "settlement_evidence_ts",
    "feature_date",
]


@dataclass
class BuildStats:
    """Track row selection and skip reasons while building resolution rows."""

    total_entries_seen: int = 0
    entries_not_dict: int = 0
    entries_missing_condition_id: int = 0
    entries_filtered_by_market_ids: int = 0
    entries_missing_resolved_at: int = 0
    entries_invalid_token_ids: int = 0
    entries_non_binary_token_count: int = 0
    entries_non_binary_outcome_vector: int = 0
    rows_selected: int = 0
    duplicate_market_rows: int = 0
    duplicate_market_rows_replaced: int = 0
    gamma_winner_populated: int = 0
    gamma_winner_matches_vector: int = 0
    gamma_winner_mismatches_vector: int = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Create backtester resolution parquet from mapping [1,0]/[0,1] outcome vectors."
        ),
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=DEFAULT_MAPPING_DIR,
        help="Directory containing gamma_updown_markets_*.json mapping shards.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Prepared run root (e.g., data/cached/pmxt_backtest/runs/btc-updown-5m). "
            "If provided, defaults features-root and output-path under this run."
        ),
    )
    parser.add_argument(
        "--features-root",
        type=Path,
        default=None,
        help=(
            "Optional features root used to limit markets to prepared features. "
            "Expected layout: features/YYYY-MM-DD/<market_id>/*.parquet"
        ),
    )
    parser.add_argument(
        "--market-ids-file",
        type=Path,
        default=None,
        help="Optional newline-delimited market_id allowlist file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output parquet path. If omitted with --run-dir, uses "
            "run-dir/resolution/resolution_frame.parquet."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output parquet if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute rows and print stats without writing parquet.",
    )
    return parser.parse_args(argv)


def _normalize_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def _parse_binary_outcome_vector(raw_value: object) -> list[int] | None:  # noqa: PLR0911
    """Parse strict binary vectors [1,0] or [0,1] from list or JSON-list string."""
    candidate: object = raw_value
    if isinstance(raw_value, str):
        try:
            candidate = json.loads(raw_value)
        except json.JSONDecodeError:
            return None

    if not isinstance(candidate, list) or len(candidate) != BINARY_VECTOR_LENGTH:
        return None

    rounded: list[int] = []
    for value in candidate:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        int_value = round(number)
        if abs(number - int_value) > FLOAT_EQ_TOLERANCE:
            return None
        rounded.append(int_value)

    if rounded not in ([1, 0], [0, 1]):
        return None
    return rounded


def _iter_mapping_entries(mapping_dir: Path) -> list[tuple[Path, str, dict[str, object]]]:
    """Load all mapping entries from shards in deterministic order."""
    if not mapping_dir.exists():
        return []

    entries: list[tuple[Path, str, dict[str, object]]] = []
    for shard_path in sorted(mapping_dir.glob("gamma_updown_markets_*.json")):
        try:
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        for slug, entry in sorted(payload.items(), key=lambda item: item[0]):
            if isinstance(entry, dict):
                entries.append((shard_path, str(slug), entry))
            else:
                entries.append((shard_path, str(slug), {}))
    return entries


def collect_market_ids_from_file(path: Path | None) -> set[str]:
    """Load market IDs from newline-delimited file."""
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(path)

    out: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        value = raw_line.strip()
        if value and not value.startswith("#"):
            out.add(value)
    return out


def collect_market_ids_from_features(features_root: Path | None) -> set[str]:
    """Collect market IDs from prepared feature directory layout."""
    if features_root is None:
        return set()
    if not features_root.exists():
        raise FileNotFoundError(features_root)

    market_ids: set[str] = set()
    for day_dir in sorted(features_root.iterdir()):
        if not day_dir.is_dir():
            continue
        for market_dir in sorted(day_dir.iterdir()):
            if market_dir.is_dir():
                market_ids.add(market_dir.name)

    return market_ids


def collect_date_market_pairs_from_features(
    features_root: Path | None,
) -> set[tuple[str, str]]:
    """Collect (date, market_id) pairs from prepared feature directory layout.
    
    Returns set of (date_str, market_id) tuples representing all feature granularity.
    This ensures resolution has one row per unique (date, market_id) combination.
    """
    if features_root is None:
        return set()
    if not features_root.exists():
        raise FileNotFoundError(features_root)

    pairs: set[tuple[str, str]] = set()
    for day_dir in sorted(features_root.iterdir()):
        if not day_dir.is_dir():
            continue
        date_str = day_dir.name
        for market_dir in sorted(day_dir.iterdir()):
            if market_dir.is_dir():
                market_id = market_dir.name
                pairs.add((date_str, market_id))

    return pairs


def combine_market_filters(filters: list[set[str]]) -> set[str] | None:
    """Combine optional market filters by intersection, preserving strict scoping."""
    non_empty_filters = [market_ids for market_ids in filters if market_ids]
    if not non_empty_filters:
        return None

    combined = set(non_empty_filters[0])
    for market_ids in non_empty_filters[1:]:
        combined &= market_ids
    return combined


def combine_date_market_filters(
    filters: list[set[tuple[str, str]]],
) -> set[tuple[str, str]] | None:
    """Combine optional (date, market_id) pair filters by intersection."""
    non_empty_filters = [pairs for pairs in filters if pairs]
    if not non_empty_filters:
        return None

    combined = set(non_empty_filters[0])
    for pairs in non_empty_filters[1:]:
        combined &= pairs
    return combined


def _resolve_market_id(entry: dict[str, object], stats: BuildStats) -> str | None:
    market_id = _normalize_optional_str(entry.get("conditionId"))
    if market_id is None:
        stats.entries_missing_condition_id += 1
        return None
    return market_id


def _resolve_resolved_at(entry: dict[str, object], stats: BuildStats) -> pd.Timestamp | None:
    resolved_raw = entry.get("resolvedAt") or entry.get("endDate")
    resolved_text = _normalize_optional_str(resolved_raw)
    if resolved_text is None:
        stats.entries_missing_resolved_at += 1
        return None

    resolved_at = pd.to_datetime(resolved_text, utc=True, errors="coerce")
    if pd.isna(resolved_at):
        stats.entries_missing_resolved_at += 1
        return None
    return resolved_at


def _resolve_binary_vector(
    entry: dict[str, object],
    stats: BuildStats,
) -> tuple[list[str], list[int]] | None:
    token_ids = parse_clob_token_ids(entry.get("clobTokenIds"))
    if len(token_ids) < BINARY_VECTOR_LENGTH:
        stats.entries_invalid_token_ids += 1
        return None
    if len(token_ids) != BINARY_VECTOR_LENGTH:
        stats.entries_non_binary_token_count += 1
        return None

    vector = _parse_binary_outcome_vector(entry.get("winningOutcome"))
    if vector is None:
        vector = _parse_binary_outcome_vector(entry.get("outcomePrices"))
    if vector is None:
        stats.entries_non_binary_outcome_vector += 1
        return None

    return token_ids, vector


def _build_row_from_entry(
    *,
    market_id: str,
    resolved_at: pd.Timestamp,
    token_ids: list[str],
    vector: list[int],
    entry: dict[str, object],
    stats: BuildStats,
) -> dict[str, object]:
    winner_idx = 0 if vector[0] == 1 else 1
    winning_asset_id = token_ids[winner_idx]

    gamma_winner = _normalize_optional_str(entry.get("winningAssetId"))
    if gamma_winner is not None:
        stats.gamma_winner_populated += 1
        if gamma_winner == winning_asset_id:
            stats.gamma_winner_matches_vector += 1
        else:
            stats.gamma_winner_mismatches_vector += 1

    return {
        "market_id": market_id,
        "resolved_at": resolved_at,
        "winning_asset_id": str(winning_asset_id),
        "winning_outcome": json.dumps([str(vector[0]), str(vector[1])]),
        "fees_enabled_market": bool(entry.get("feesEnabledMarket", entry.get("feesEnabled", True))),
        "settlement_source": "mapping_binary_vector",
        "settlement_confidence": 1.0,
        "settlement_evidence_ts": resolved_at,
    }


def _upsert_market_row(
    *,
    selected_rows: dict[str, dict[str, object]],
    market_id: str,
    row: dict[str, object],
    stats: BuildStats,
) -> None:
    existing = selected_rows.get(market_id)
    if existing is None:
        selected_rows[market_id] = row
        stats.rows_selected += 1
        return

    stats.duplicate_market_rows += 1
    existing_resolved_at = pd.to_datetime(str(existing["resolved_at"]), utc=True, errors="coerce")
    if row["resolved_at"] >= existing_resolved_at:
        selected_rows[market_id] = row
        stats.duplicate_market_rows_replaced += 1


def _select_row_for_feature_date(
    *,
    candidate_rows: list[dict[str, object]],
    feature_date: str,
) -> dict[str, object] | None:
    same_day_rows: list[tuple[pd.Timestamp, dict[str, object]]] = []
    for candidate in candidate_rows:
        resolved_at = pd.to_datetime(candidate.get("resolved_at"), utc=True, errors="coerce")
        if pd.isna(resolved_at):
            continue
        if resolved_at.strftime("%Y-%m-%d") == feature_date:
            same_day_rows.append((resolved_at, candidate))

    if not same_day_rows:
        return None

    same_day_rows.sort(key=lambda item: item[0], reverse=True)
    return same_day_rows[0][1]


def build_resolution_frame_from_mapping_vectors(
    *,
    mapping_dir: Path,
    required_market_ids: set[str] | None,
    required_date_market_pairs: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, BuildStats]:
    """Build resolution frame from mapping binary vectors.
    
    If required_date_market_pairs is provided, creates one resolution row per
    (date, market_id) pair (accounting for market appearing on different dates).
    If only required_market_ids is provided, creates one row per market.
    """
    stats = BuildStats()
    selected_rows: dict[str, dict[str, object]] = {}
    candidate_rows_by_market: dict[str, list[dict[str, object]]] = {}

    for _, _, entry in _iter_mapping_entries(mapping_dir):
        stats.total_entries_seen += 1
        if not entry:
            stats.entries_not_dict += 1
            continue

        market_id = _resolve_market_id(entry, stats)
        if market_id is None:
            continue

        if required_market_ids is not None and market_id not in required_market_ids:
            stats.entries_filtered_by_market_ids += 1
            continue

        resolved_at = _resolve_resolved_at(entry, stats)
        if resolved_at is None:
            continue

        binary_inputs = _resolve_binary_vector(entry, stats)
        if binary_inputs is None:
            continue
        token_ids, vector = binary_inputs

        row = _build_row_from_entry(
            market_id=market_id,
            resolved_at=resolved_at,
            token_ids=token_ids,
            vector=vector,
            entry=entry,
            stats=stats,
        )
        _upsert_market_row(
            selected_rows=selected_rows,
            market_id=market_id,
            row=row,
            stats=stats,
        )
        candidate_rows_by_market.setdefault(market_id, []).append(row)

    # If date-market pairs are provided, choose the row matching that calendar day.
    if required_date_market_pairs is not None:
        replicated_rows: list[dict[str, object]] = []
        for date_str, market_id in sorted(required_date_market_pairs):
            candidate_rows = candidate_rows_by_market.get(market_id, [])
            selected_row = _select_row_for_feature_date(
                candidate_rows=candidate_rows,
                feature_date=date_str,
            )
            if selected_row is None:
                continue

            row_with_date = dict(selected_row)
            row_with_date["feature_date"] = date_str
            replicated_rows.append(row_with_date)
        frame = pd.DataFrame(replicated_rows, columns=RESOLUTION_COLUMNS)
    else:
        # Without date granularity, keep one row per market and derive feature_date
        # from resolved_at for downstream compatibility.
        rows_with_date = []
        for row in selected_rows.values():
            row_with_date = dict(row)
            resolved_at = pd.to_datetime(row_with_date.get("resolved_at"), utc=True, errors="coerce")
            row_with_date["feature_date"] = (
                resolved_at.strftime("%Y-%m-%d") if not pd.isna(resolved_at) else None
            )
            rows_with_date.append(row_with_date)
        frame = pd.DataFrame(rows_with_date, columns=RESOLUTION_COLUMNS)

    if frame.empty:
        return frame, stats

    frame["market_id"] = frame["market_id"].astype(str)
    frame["winning_asset_id"] = frame["winning_asset_id"].astype(str)
    frame["winning_outcome"] = frame["winning_outcome"].astype(str)
    frame["fees_enabled_market"] = frame["fees_enabled_market"].astype(bool)
    frame["settlement_source"] = frame["settlement_source"].astype(str)
    frame["settlement_confidence"] = pd.to_numeric(
        frame["settlement_confidence"],
        errors="coerce",
    ).fillna(1.0)
    frame["resolved_at"] = pd.to_datetime(frame["resolved_at"], utc=True, errors="coerce")
    frame["settlement_evidence_ts"] = pd.to_datetime(
        frame["settlement_evidence_ts"],
        utc=True,
        errors="coerce",
    )
    parsed_feature_dates = pd.to_datetime(frame["feature_date"], utc=True, errors="coerce")
    normalized_feature_dates = parsed_feature_dates.dt.strftime("%Y-%m-%d")
    raw_feature_dates = frame["feature_date"].astype(str).str.strip()
    raw_feature_dates = raw_feature_dates.mask(raw_feature_dates.isin({"", "nan", "NaT", "None"}))
    frame["feature_date"] = normalized_feature_dates.fillna(raw_feature_dates)

    frame = frame.sort_values(["resolved_at", "market_id", "feature_date"]).reset_index(drop=True)
    return frame, stats


def resolve_paths(args: argparse.Namespace) -> tuple[Path | None, Path]:
    """Resolve features root and output path from CLI args."""
    run_dir = args.run_dir
    features_root = args.features_root
    output_path = args.output_path

    if run_dir is not None:
        run_dir = run_dir.resolve()
        if features_root is None:
            features_root = run_dir / "features"
        if output_path is None:
            output_path = run_dir / "resolution" / DEFAULT_RESOLUTION_FILENAME

    if output_path is None:
        msg = "Provide --output-path or --run-dir so output location is known"
        raise ValueError(msg)

    return features_root, output_path


def summarize(
    *,
    frame: pd.DataFrame,
    stats: BuildStats,
    required_market_ids: set[str] | None,
) -> dict[str, Any]:
    """Build JSON-friendly summary payload for logs."""
    required_count = None if required_market_ids is None else len(required_market_ids)
    missing_required_count = None
    if required_market_ids is not None:
        present_market_ids = (
            set(frame["market_id"].astype(str).tolist())
            if not frame.empty
            else set()
        )
        missing_required_count = len(required_market_ids - present_market_ids)

    return {
        "rows_selected": len(frame),
        "required_market_count": required_count,
        "missing_required_market_count": missing_required_count,
        "total_entries_seen": stats.total_entries_seen,
        "entries_not_dict": stats.entries_not_dict,
        "entries_missing_condition_id": stats.entries_missing_condition_id,
        "entries_filtered_by_market_ids": stats.entries_filtered_by_market_ids,
        "entries_missing_resolved_at": stats.entries_missing_resolved_at,
        "entries_invalid_token_ids": stats.entries_invalid_token_ids,
        "entries_non_binary_token_count": stats.entries_non_binary_token_count,
        "entries_non_binary_outcome_vector": stats.entries_non_binary_outcome_vector,
        "duplicate_market_rows": stats.duplicate_market_rows,
        "duplicate_market_rows_replaced": stats.duplicate_market_rows_replaced,
        "gamma_winner_populated": stats.gamma_winner_populated,
        "gamma_winner_matches_vector": stats.gamma_winner_matches_vector,
        "gamma_winner_mismatches_vector": stats.gamma_winner_mismatches_vector,
    }


def main(argv: list[str] | None = None) -> int:
    """Run CLI entrypoint."""
    args = parse_args(argv)

    if not args.mapping_dir.exists():
        msg = f"Mapping directory not found: {args.mapping_dir}"
        raise FileNotFoundError(msg)

    features_root, output_path = resolve_paths(args)

    file_market_ids = collect_market_ids_from_file(args.market_ids_file)
    feature_market_ids = collect_market_ids_from_features(features_root)
    required_market_ids = combine_market_filters([file_market_ids, feature_market_ids])

    # Also collect (date, market_id) pairs from features for proper granularity
    feature_date_market_pairs = collect_date_market_pairs_from_features(features_root)
    # Filter by required market IDs if applicable
    if required_market_ids is not None:
        feature_date_market_pairs = {
            (date, market_id)
            for date, market_id in feature_date_market_pairs
            if market_id in required_market_ids
        }

    frame, stats = build_resolution_frame_from_mapping_vectors(
        mapping_dir=args.mapping_dir,
        required_market_ids=required_market_ids,
        required_date_market_pairs=feature_date_market_pairs or None,
    )
    report = summarize(
        frame=frame,
        stats=stats,
        required_market_ids=required_market_ids,
    )

    print(json.dumps(report, indent=2, sort_keys=True))

    if args.dry_run:
        return 0

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        msg = f"Output already exists (use --overwrite to replace): {output_path}"
        raise FileExistsError(msg)

    frame.to_parquet(output_path, index=False)
    print(f"Wrote resolution parquet: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
