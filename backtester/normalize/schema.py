from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from utils.serialization import dataframe_json_column_to_dict

if TYPE_CHECKING:
    from backtester.config.types import ValidationPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SchemaValidationIssue:
    """Single schema validation issue for one row in market events."""

    row_index: int
    market_id: str
    event_type: str
    column: str
    reason: str


def normalize_price_change_payload(payload: dict, token_id: str) -> dict:
    """Normalize PMXT-style price_change payload into canonical schema."""
    if "price_changes" in payload and isinstance(payload["price_changes"], list):
        return payload

    asset_id = str(payload.get("asset_id") or payload.get("token_id") or token_id)
    normalized_change = {
        "asset_id": asset_id,
        "best_bid": payload.get("best_bid"),
        "best_ask": payload.get("best_ask"),
        "price": payload.get("change_price") or payload.get("price"),
        "side": payload.get("change_side") or payload.get("side"),
        "size": payload.get("change_size") or payload.get("size"),
    }
    return {"price_changes": [normalized_change]}


def resolve_market_event_timestamp_column(columns: pd.Index) -> str:
    if "ts_event" in columns:
        return "ts_event"
    for candidate in ("timestamp_received", "timestamp_created_at", "timestamp"):
        if candidate in columns:
            return candidate
    msg = "Unable to find timestamp column for market events"
    raise ValueError(msg)


def resolve_market_event_type_column(columns: pd.Index) -> str:
    if "event_type" in columns:
        return "event_type"
    if "update_type" in columns:
        return "update_type"
    msg = "Unable to find event type column for market events"
    raise ValueError(msg)


def extract_token_from_payload(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("token_id") or payload.get("asset_id") or "")


def normalize_market_event_payload(
    payload: object,
    token_id: str,
    event_type: str,
) -> object:
    if not isinstance(payload, dict):
        return payload

    normalized_payload = payload
    if normalized_payload.get("asset_id") in (None, "") and token_id:
        normalized_payload = dict(normalized_payload)
        normalized_payload["asset_id"] = token_id

    if event_type == "price_change":
        normalized_payload = normalize_price_change_payload(normalized_payload, token_id)
    return normalized_payload


def drop_missing_pmxt_tokens(
    normalized: pd.DataFrame,
    *,
    is_pmxt_mode: bool,
) -> pd.DataFrame:
    if not is_pmxt_mode:
        return normalized

    missing_token_mask = normalized["token_id"].fillna("").str.strip() == ""
    missing_count = int(missing_token_mask.sum())
    if missing_count:
        logger.warning(
            "Dropping %d PMXT rows with missing token_id (unusable for per-side features)",
            missing_count,
        )
        return normalized.loc[~missing_token_mask].copy()
    return normalized


def normalize_market_events_schema(
    df: pd.DataFrame,
    *,
    is_pmxt_mode: bool,
) -> pd.DataFrame:
    """Normalize market events from supported feeds to a single schema."""
    if df.empty:
        return df

    normalized = df.copy()

    timestamp_col = resolve_market_event_timestamp_column(normalized.columns)
    if timestamp_col != "ts_event":
        normalized = normalized.rename(columns={timestamp_col: "ts_event"})

    event_type_col = resolve_market_event_type_column(normalized.columns)
    if event_type_col != "event_type":
        normalized = normalized.rename(columns={event_type_col: "event_type"})

    normalized["event_type"] = normalized["event_type"].replace(
        {"book_snapshot": "book", "book_delta": "book", "book": "book"}
    )

    if "data" in normalized.columns and not normalized.empty:
        normalized = dataframe_json_column_to_dict(normalized, column="data")
    if "data" not in normalized.columns:
        normalized["data"] = pd.Series(
            [{} for _ in range(len(normalized))],
            index=normalized.index,
            dtype="object",
        )

    if "token_id" not in normalized.columns:
        normalized["token_id"] = ""

    token_from_payload = normalized["data"].map(extract_token_from_payload)
    normalized["token_id"] = normalized["token_id"].fillna("").astype(str)
    normalized["token_id"] = normalized["token_id"].mask(
        normalized["token_id"] == "",
        token_from_payload,
    )
    normalized = drop_missing_pmxt_tokens(normalized, is_pmxt_mode=is_pmxt_mode)

    payloads = normalized["data"].tolist()
    token_ids = normalized["token_id"].fillna("").astype(str).tolist()
    event_types = normalized["event_type"].fillna("").astype(str).tolist()
    normalized_payloads = [
        normalize_market_event_payload(payload, token_id, event_type)
        for payload, token_id, event_type in zip(
            payloads,
            token_ids,
            event_types,
            strict=False,
        )
    ]
    normalized["data"] = pd.Series(normalized_payloads, index=normalized.index)

    keep_cols = ["ts_event", "event_type", "market_id", "token_id", "data"]
    existing_cols = [col for col in keep_cols if col in normalized.columns]
    return normalized[existing_cols]


def validate_market_events_rows(
    normalized: pd.DataFrame,
    *,
    policy: ValidationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate normalized market events and optionally quarantine invalid rows."""
    if normalized.empty:
        empty_report = pd.DataFrame(
            columns=["row_index", "market_id", "event_type", "column", "reason"]
        )
        return normalized, empty_report

    issues: list[SchemaValidationIssue] = []
    keep_mask = pd.Series(data=True, index=normalized.index)

    def _add_issue(row_pos: int, row: pd.Series, column: str, reason: str) -> None:
        issues.append(
            SchemaValidationIssue(
                row_index=int(row_pos),
                market_id=str(row.get("market_id", "")),
                event_type=str(row.get("event_type", "")),
                column=column,
                reason=reason,
            )
        )

    for row_pos, (row_index, row) in enumerate(normalized.iterrows()):
        event_type = str(row.get("event_type", ""))
        market_id = str(row.get("market_id", "")).strip()
        token_id = str(row.get("token_id", "")).strip()
        ts_raw = row.get("ts_event", row_index)
        ts_event = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        payload = row.get("data")

        invalid = False
        if not market_id:
            _add_issue(row_pos, row, "market_id", "missing_market_id")
            invalid = True

        if not event_type:
            _add_issue(row_pos, row, "event_type", "missing_event_type")
            invalid = True
        elif event_type not in policy.allowed_event_types:
            _add_issue(row_pos, row, "event_type", "unsupported_event_type")
            invalid = True

        if pd.isna(ts_event):
            _add_issue(row_pos, row, "ts_event", "invalid_timestamp")
            invalid = True

        if event_type in policy.require_token_for_events and not token_id:
            _add_issue(row_pos, row, "token_id", "missing_token_for_event")
            invalid = True

        if not isinstance(payload, dict):
            _add_issue(row_pos, row, "data", "payload_not_dict")
            invalid = True
        elif event_type == "book":
            bids = payload.get("bids")
            asks = payload.get("asks")
            if not isinstance(bids, list) or not isinstance(asks, list):
                _add_issue(row_pos, row, "data", "book_payload_missing_bids_or_asks")
                invalid = True
        elif event_type == "price_change":
            price_changes = payload.get("price_changes")
            if not isinstance(price_changes, list):
                _add_issue(row_pos, row, "data", "price_change_payload_missing_price_changes")
                invalid = True

        if invalid:
            keep_mask.iloc[row_pos] = False

    report = pd.DataFrame(
        [
            {
                "row_index": item.row_index,
                "market_id": item.market_id,
                "event_type": item.event_type,
                "column": item.column,
                "reason": item.reason,
            }
            for item in issues
        ]
    )
    if report.empty:
        return normalized, report

    if policy.quarantine_invalid_rows:
        validated = normalized.loc[keep_mask.to_numpy()].copy()
        logger.warning(
            "Quarantined %d invalid market event rows",
            int((~keep_mask).sum()),
        )
        return validated, report

    reasons = report["reason"].value_counts().to_dict()
    msg = f"Schema validation failed for {len(report)} rows: {reasons}"
    raise ValueError(msg)
