from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, MutableMapping
    from pathlib import Path

    import pandas as pd

logger = logging.getLogger(__name__)


def _iter_mapping_payloads(mapping_path: Path) -> Iterable[dict[str, object]]:
    for mapping_file in sorted(mapping_path.glob("gamma_updown_markets_*.json")):
        try:
            payload = json.loads(mapping_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            yield payload


def load_condition_ids_for_slug_prefix(
    mapping_path: Path,
    slug_prefix: str,
    *,
    cache: MutableMapping[str, set[str]] | None = None,
) -> set[str]:
    """Resolve a slug prefix to condition ids from mapping shards."""
    prefix = str(slug_prefix).strip().lower()
    if not prefix or not mapping_path.exists():
        return set()

    if cache is not None:
        cached_ids = cache.get(prefix)
        if cached_ids is not None:
            return set(cached_ids)

    condition_ids: set[str] = set()
    for payload in _iter_mapping_payloads(mapping_path):
        matching_ids = {
            str(entry.get("conditionId")).lower()
            for slug, entry in payload.items()
            if str(slug).lower().startswith(prefix)
            and isinstance(entry, dict)
            and entry.get("conditionId")
        }
        condition_ids.update(matching_ids)

    if cache is not None:
        cache[prefix] = set(condition_ids)
    return condition_ids


def filter_market_events_by_slug_prefix(
    market_events: pd.DataFrame,
    market_slug_prefix: str | None,
    *,
    is_pmxt_mode: bool,
    mapping_path: Path,
    condition_ids_lookup: Callable[[str], set[str]],
) -> pd.DataFrame:
    """Filter market events by slug prefix using PMXT mapping when available."""
    if market_events.empty or market_slug_prefix is None:
        return market_events

    prefix = str(market_slug_prefix).strip().lower()
    if not prefix:
        return market_events

    market_ids = market_events["market_id"].fillna("").astype(str).str.lower()

    if is_pmxt_mode:
        condition_ids = condition_ids_lookup(prefix)
        if condition_ids:
            filtered = market_events.loc[market_ids.isin(condition_ids)].copy()
            logger.info(
                (
                    "Applied PMXT mapping-based market slug filter '%s': "
                    "%d -> %d rows across %d condition IDs"
                ),
                prefix,
                len(market_events),
                len(filtered),
                len(condition_ids),
            )
            return filtered

        logger.warning(
            "No mapping entries found for market_slug_prefix='%s' in %s; "
            "falling back to direct market_id prefix filtering",
            prefix,
            mapping_path,
        )

    filtered = market_events.loc[market_ids.str.startswith(prefix)].copy()
    logger.info(
        "Applied direct market_id prefix filter '%s': %d -> %d rows",
        prefix,
        len(market_events),
        len(filtered),
    )
    return filtered
