from __future__ import annotations

import json
import logging

import pandas as pd

from backtester.loaders.resolution import ResolutionMappingDeps
from backtester.loaders.resolution import (
    build_resolution_row_from_mapping_entry as build_resolution_row_from_mapping_entry_loader,
)
from backtester.loaders.resolution import (
    load_condition_entry_map as load_condition_entry_map_loader,
)
from backtester.loaders.resolution import (
    load_resolution_frame_from_events as load_resolution_frame_from_events_loader,
)
from backtester.loaders.resolution import (
    load_resolution_frame_from_mapping as load_resolution_frame_from_mapping_loader,
)
from backtester.loaders.resolution import (
    load_resolution_frame_with_fallback as load_resolution_frame_with_fallback_loader,
)
from backtester.resolution.parsing import (
    is_missing_resolution_winner,
    parse_clob_token_ids,
    parse_outcome_prices,
    select_winning_asset_id,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

_RESOLUTION_COLUMNS = [
    "market_id",
    "resolved_at",
    "winning_asset_id",
    "winning_outcome",
    "fees_enabled_market",
]


class BacktestResolutionManager:
    """Resolution loading, repair, provenance, and validation routines."""

    def load_resolution_frame_from_events(self, market_events: pd.DataFrame) -> pd.DataFrame:
        """Extract one resolution row per market from event stream."""
        return load_resolution_frame_from_events_loader(
            market_events,
            resolution_columns=tuple(_RESOLUTION_COLUMNS),
        )

    def _load_condition_entry_map(
        self,
        mapping_dir: str | Path,
    ) -> dict[str, tuple[Path, str, dict[str, object]]]:
        return load_condition_entry_map_loader(mapping_dir)

    def _write_mapping_winner(
        self,
        shard_path: Path,
        slug: str,
        winning_asset_id: str,
    ) -> bool:
        if not shard_path.exists() or not winning_asset_id:
            return False

        try:
            payload = json.loads(shard_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False

        if not isinstance(payload, dict):
            return False

        entry = payload.get(slug)
        if not isinstance(entry, dict):
            return False

        current_winner = entry.get("winningAssetId")
        if not is_missing_resolution_winner(current_winner):
            return False

        updated_entry = dict(entry)
        updated_entry["winningAssetId"] = winning_asset_id
        payload[slug] = updated_entry
        shard_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return True

    def _resolve_settlement_provenance(  # noqa: PLR0913
        self,
        *,
        market_id: str,
        winner: object,
        resolved_at: object,
        event_resolutions: pd.DataFrame,
        mapping_resolutions: pd.DataFrame,
        market_events: pd.DataFrame,
        confidence_threshold: float,
        terminal_feature_lookup: Mapping[str, pd.DataFrame] | None,
    ) -> tuple[str, float | None, pd.Timestamp | None]:
        winner_str = str(winner)
        if market_id in event_resolutions.index:
            event_winner = event_resolutions.loc[market_id].get("winning_asset_id")
            if (
                not self._is_missing_resolution_winner(event_winner)
                and str(event_winner) == winner_str
            ):
                event_resolved_at = self._to_utc_timestamp(
                    event_resolutions.loc[market_id].get("resolved_at")
                )
                return "event", 1.0, event_resolved_at

        if market_id in mapping_resolutions.index:
            mapping_winner = mapping_resolutions.loc[market_id].get("winning_asset_id")
            if (
                not self._is_missing_resolution_winner(mapping_winner)
                and str(mapping_winner) == winner_str
            ):
                mapping_resolved_at = self._to_utc_timestamp(
                    mapping_resolutions.loc[market_id].get("resolved_at")
                )
                return "mapping", 1.0, mapping_resolved_at

        resolved_ts = self._to_utc_timestamp(resolved_at)
        terminal_winner, terminal_price, terminal_ts = self._infer_terminal_winner_from_events(
            market_events,
            market_id,
            resolved_ts,
            terminal_feature_lookup=terminal_feature_lookup,
        )
        if (
            terminal_winner
            and terminal_price is not None
            and terminal_price >= confidence_threshold
            and terminal_winner == winner_str
        ):
            return "inferred", float(terminal_price), terminal_ts

        return "unknown", None, None

    @staticmethod
    def _is_missing_resolution_winner(value: object) -> bool:
        return is_missing_resolution_winner(value)

    def _parse_clob_token_ids(self, raw_value: object) -> list[str]:
        return parse_clob_token_ids(raw_value)

    def _parse_outcome_prices(self, raw_value: object) -> list[float]:
        return parse_outcome_prices(raw_value)

    def _select_winning_asset_id(
        self,
        token_ids: list[str],
        outcome_prices: list[float],
    ) -> str | None:
        return select_winning_asset_id(token_ids, outcome_prices)

    def _build_resolution_row_from_mapping_entry(
        self,
        market_id: str,
        entry: dict[str, object],
    ) -> dict[str, object] | None:
        return build_resolution_row_from_mapping_entry_loader(
            market_id,
            entry,
            deps=ResolutionMappingDeps(
                parse_clob_token_ids_fn=self._parse_clob_token_ids,
                parse_outcome_prices_fn=self._parse_outcome_prices,
                select_winning_asset_id_fn=self._select_winning_asset_id,
            ),
        )

    def load_resolution_frame_from_mapping(
        self,
        market_ids: list[str],
        *,
        mapping_dir: str | Path,
    ) -> pd.DataFrame:
        """Build resolution table from mapping files for requested markets."""
        return load_resolution_frame_from_mapping_loader(
            market_ids,
            mapping_dir=mapping_dir,
            deps=ResolutionMappingDeps(
                parse_clob_token_ids_fn=self._parse_clob_token_ids,
                parse_outcome_prices_fn=self._parse_outcome_prices,
                select_winning_asset_id_fn=self._select_winning_asset_id,
            ),
            resolution_columns=tuple(_RESOLUTION_COLUMNS),
        )

    def load_resolution_frame_with_fallback(
        self,
        market_events: pd.DataFrame,
        *,
        mapping_dir: str | Path,
    ) -> pd.DataFrame:
        """Resolve market outcomes from stream first, then mapping fallback."""
        return load_resolution_frame_with_fallback_loader(
            market_events,
            load_resolution_frame_from_events_fn=self.load_resolution_frame_from_events,
            load_resolution_frame_from_mapping_fn=(
                lambda market_ids: self.load_resolution_frame_from_mapping(
                    market_ids,
                    mapping_dir=mapping_dir,
                )
            ),
        )

    def load_and_validate_resolution(
        self,
        market_events: pd.DataFrame,
        *,
        mapping_dir: str | Path,
        confidence_threshold: float = 0.95,
        features: pd.DataFrame | None = None,
        repair_dry_run: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load, repair, and validate resolution data against mapping and terminal evidence."""
        resolution_frame = self.load_resolution_frame_with_fallback(
            market_events,
            mapping_dir=mapping_dir,
        )
        if resolution_frame.empty:
            raise ValueError("No resolution data available from market events or mapping files")

        terminal_feature_lookup = self._build_terminal_feature_lookup(features)

        market_ids = resolution_frame.index.astype(str).tolist()
        event_resolutions = self.load_resolution_frame_from_events(market_events)
        mapping_resolutions = self.load_resolution_frame_from_mapping(
            market_ids,
            mapping_dir=mapping_dir,
        )

        resolution_frame, repair_audit = self.repair_missing_resolution_winners(
            market_events,
            resolution_frame,
            mapping_dir=mapping_dir,
            confidence_threshold=confidence_threshold,
            terminal_feature_lookup=terminal_feature_lookup,
            dry_run=repair_dry_run,
        )
        self._last_resolution_repair_audit = repair_audit

        diagnostics = self.validate_resolution_consistency(
            market_events,
            resolution_frame,
            mapping_dir=mapping_dir,
            confidence_threshold=confidence_threshold,
            terminal_feature_lookup=terminal_feature_lookup,
        )

        resolution_with_provenance = resolution_frame.copy()
        source_rows: list[dict[str, object]] = []
        for market_id in resolution_with_provenance.index.astype(str):
            row = resolution_with_provenance.loc[market_id]
            source, confidence, evidence_ts = self._resolve_settlement_provenance(
                market_id=market_id,
                winner=row.get("winning_asset_id"),
                resolved_at=row.get("resolved_at"),
                event_resolutions=event_resolutions,
                mapping_resolutions=mapping_resolutions,
                market_events=market_events,
                confidence_threshold=confidence_threshold,
                terminal_feature_lookup=terminal_feature_lookup,
            )
            source_rows.append(
                {
                    "market_id": market_id,
                    "settlement_source": source,
                    "settlement_confidence": confidence,
                    "settlement_evidence_ts": evidence_ts,
                }
            )

        if source_rows:
            source_frame = pd.DataFrame(source_rows).set_index("market_id")
            resolution_with_provenance = resolution_with_provenance.join(source_frame, how="left")

        return resolution_with_provenance, diagnostics

    def _build_terminal_feature_lookup(
        self,
        features: pd.DataFrame | None,
    ) -> dict[str, pd.DataFrame] | None:
        """Build per-market feature lookup for terminal winner inference."""
        if features is None or features.empty:
            return None

        required_cols = {"market_id", "token_id", "mid_price"}
        if not required_cols.issubset(features.columns):
            return None

        frame = features[["market_id", "token_id", "mid_price"]].copy()
        frame = frame.dropna(subset=["market_id", "token_id", "mid_price"])
        if frame.empty:
            return None

        frame["market_id"] = frame["market_id"].astype(str)
        frame["token_id"] = frame["token_id"].astype(str)
        frame = frame.sort_index()

        return {
            str(market_id): group.sort_index()
            for market_id, group in frame.groupby("market_id", sort=False)
        }

    def repair_missing_resolution_winners(  # noqa: PLR0913
        self,
        market_events: pd.DataFrame,
        resolution_frame: pd.DataFrame,
        *,
        mapping_dir: str | Path,
        confidence_threshold: float = 0.95,
        terminal_feature_lookup: Mapping[str, pd.DataFrame] | None = None,
        dry_run: bool = False,
    ) -> tuple[pd.DataFrame, list[dict[str, object]]]:
        """Fill missing winners from terminal evidence and persist them to mapping shards."""
        if resolution_frame.empty:
            return resolution_frame, []

        repaired_frame = resolution_frame.copy()
        mapping_lookup = self._load_condition_entry_map(mapping_dir)
        if not mapping_lookup:
            return repaired_frame, []

        audit_rows: list[dict[str, object]] = []

        for market_id in repaired_frame.index.astype(str):
            row = repaired_frame.loc[market_id]
            if isinstance(row, pd.DataFrame):
                continue

            winner = row.get("winning_asset_id")
            if not self._is_missing_resolution_winner(winner):
                continue

            resolved_at = self._to_utc_timestamp(row.get("resolved_at"))
            terminal_winner, terminal_price, terminal_ts = self._infer_terminal_winner_from_events(
                market_events,
                market_id,
                resolved_at,
                terminal_feature_lookup=terminal_feature_lookup,
            )
            if (
                not terminal_winner
                or terminal_price is None
                or terminal_price < confidence_threshold
            ):
                continue

            repaired_frame.loc[market_id, "winning_asset_id"] = terminal_winner

            lookup = mapping_lookup.get(market_id.lower())
            shard_path = None
            slug = None
            before_winner = None
            write_applied = False
            if lookup is None:
                audit_rows.append(
                    {
                        "market_id": market_id,
                        "shard_path": None,
                        "slug": None,
                        "before_winner": None,
                        "after_winner": terminal_winner,
                        "terminal_price": float(terminal_price),
                        "terminal_evidence_ts": terminal_ts,
                        "confidence_threshold": confidence_threshold,
                        "dry_run": dry_run,
                        "mapping_write_applied": False,
                        "mapping_write_success": False,
                        "reason": "mapping_entry_not_found",
                    }
                )
                continue

            shard_path, slug, entry = lookup
            before_winner = entry.get("winningAssetId") if isinstance(entry, dict) else None
            if not dry_run:
                write_applied = True
                write_success = self._write_mapping_winner(shard_path, slug, terminal_winner)
            else:
                write_success = False

            audit_rows.append(
                {
                    "market_id": market_id,
                    "shard_path": str(shard_path),
                    "slug": slug,
                    "before_winner": before_winner,
                    "after_winner": terminal_winner,
                    "terminal_price": float(terminal_price),
                    "terminal_evidence_ts": terminal_ts,
                    "confidence_threshold": confidence_threshold,
                    "dry_run": dry_run,
                    "mapping_write_applied": write_applied,
                    "mapping_write_success": write_success,
                    "reason": "repaired_from_terminal_evidence",
                }
            )

        return repaired_frame, audit_rows

    @staticmethod
    def _winner_from_feature_frame(
        features: pd.DataFrame,
    ) -> tuple[str | None, float | None, pd.Timestamp | None]:
        if features.empty or "mid_price" not in features.columns:
            return None, None, None

        latest_by_token = (
            features.sort_index()
            .groupby("token_id", sort=False)
            .tail(1)
            .dropna(subset=["mid_price"])
        )
        if latest_by_token.empty:
            return None, None, None

        winner_row = latest_by_token.sort_values("mid_price", ascending=False).iloc[0]
        winner_token = str(winner_row.get("token_id"))
        winner_price_raw = winner_row.get("mid_price")
        if winner_price_raw is None:
            return None, None, None

        winner_price = float(winner_price_raw)
        evidence_ts = latest_by_token.index.max()
        return winner_token, winner_price, evidence_ts

    def _infer_terminal_winner_from_events(
        self,
        market_events: pd.DataFrame,
        market_id: str,
        resolved_at: pd.Timestamp | None,
        *,
        terminal_feature_lookup: Mapping[str, pd.DataFrame] | None = None,
    ) -> tuple[str | None, float | None, pd.Timestamp | None]:
        no_winner = (None, None, None)
        if resolved_at is None:
            return no_winner

        if terminal_feature_lookup is not None:
            market_features = terminal_feature_lookup.get(market_id)
            if market_features is None or market_features.empty:
                return no_winner

            eligible_features = market_features[market_features.index <= resolved_at]
            return self._winner_from_feature_frame(eligible_features)

        market_slice = market_events[market_events["market_id"].astype(str) == market_id]
        if market_slice.empty:
            return no_winner

        market_slice = market_slice[market_slice.index <= resolved_at]
        if market_slice.empty:
            return no_winner

        features = self.compute_orderbook_features_df(market_slice)
        return self._winner_from_feature_frame(features)

    def validate_resolution_consistency(
        self,
        market_events: pd.DataFrame,
        resolution_frame: pd.DataFrame,
        *,
        mapping_dir: str | Path,
        confidence_threshold: float = 0.95,
        terminal_feature_lookup: Mapping[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Strictly validate resolutions against mapping and terminal event evidence.

        Raises ValueError on any mismatch or missing winner information.
        """
        if resolution_frame.empty:
            raise ValueError("Resolution validation failed: resolution frame is empty")

        market_ids = resolution_frame.index.astype(str).tolist()
        mapping_resolutions = self.load_resolution_frame_from_mapping(
            market_ids,
            mapping_dir=mapping_dir,
        )

        diagnostics: list[dict[str, object]] = []
        errors: list[str] = []
        for market_id in market_ids:
            row = resolution_frame.loc[market_id]
            if isinstance(row, pd.DataFrame):
                errors.append(f"{market_id}: duplicate resolution rows detected")
                continue

            resolved_at = self._to_utc_timestamp(row.get("resolved_at"))
            winner = row.get("winning_asset_id")
            if self._is_missing_resolution_winner(winner):
                errors.append(f"{market_id}: missing winning_asset_id")
                continue
            winner_str = str(winner)

            mapping_winner = None
            if market_id in mapping_resolutions.index:
                mapping_winner = mapping_resolutions.loc[market_id].get("winning_asset_id")
                if self._is_missing_resolution_winner(mapping_winner):
                    mapping_winner = None
                else:
                    mapping_winner = str(mapping_winner)

            if mapping_winner and mapping_winner != winner_str:
                errors.append(
                    f"{market_id}: winner mismatch event/mapping "
                    f"({winner_str} vs {mapping_winner})"
                )

            terminal_winner, terminal_price, evidence_ts = self._infer_terminal_winner_from_events(
                market_events,
                market_id,
                resolved_at,
                terminal_feature_lookup=terminal_feature_lookup,
            )
            if (
                terminal_winner
                and terminal_price is not None
                and terminal_price >= confidence_threshold
                and terminal_winner != winner_str
            ):
                errors.append(
                    f"{market_id}: terminal evidence mismatch "
                    f"({terminal_winner} @ {terminal_price:.4f}"
                    f" at {evidence_ts} vs winner {winner_str})"
                )

            diagnostics.append(
                {
                    "market_id": market_id,
                    "resolved_at": resolved_at,
                    "winner": winner_str,
                    "mapping_winner": mapping_winner,
                    "terminal_winner": terminal_winner,
                    "terminal_price": terminal_price,
                    "terminal_evidence_ts": evidence_ts,
                }
            )

        if errors:
            raise ValueError("Resolution validation failed: " + " | ".join(errors))
        return pd.DataFrame(diagnostics)
