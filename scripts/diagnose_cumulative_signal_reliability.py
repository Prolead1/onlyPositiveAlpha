"""Diagnose whether cumulative signal evolution predicts winners more reliably.

This script compares three winner-selection methods using prepared PMXT features
and resolved winner labels:
1) snapshot_score: latest relative-book score at each timestamp
2) cumulative_sum_score: cumulative sum of relative-book score over time
3) cumulative_ewm_score: exponentially weighted running mean of score

Outputs include:
- timestamp-level accuracy by method
- final-per-market accuracy by method
- accuracy by timeline quartile (early -> late) to test signal evolution
- artifact package with metrics and run metadata
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtester import BacktestRunner
from utils import setup_application_logging


@dataclass(frozen=True)
class SignalWeights:
    pressure_weight: float = 0.45
    spread_weight: float = 0.35
    depth_weight: float = 0.15
    imbalance_weight: float = 0.05

    def validate(self) -> None:
        total = (
            float(self.pressure_weight)
            + float(self.spread_weight)
            + float(self.depth_weight)
            + float(self.imbalance_weight)
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Signal weights must sum to 1.0, got {total:.6f}")


def parse_args() -> argparse.Namespace:
    default_run_dir = PROJECT_ROOT / "data" / "cached" / "pmxt_backtest" / "runs" / "btc-updown-5m"

    parser = argparse.ArgumentParser(
        description="Compare snapshot vs cumulative signal reliability against true winners"
    )
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--market-count",
        type=int,
        default=3000,
        help="Number of markets to evaluate",
    )
    parser.add_argument(
        "--shuffle-markets",
        action="store_true",
        help="Shuffle market IDs before selection",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--market-batch-size",
        type=int,
        default=100,
        help="Number of markets to process per batch.",
    )
    parser.add_argument(
        "--ewm-alpha",
        type=float,
        default=0.20,
        help="EWMA alpha for cumulative_ewm_score",
    )
    parser.add_argument(
        "--include-current-event-in-cumulative",
        action="store_true",
        help=(
            "If set, cumulative features include current event. "
            "Default is strict causal mode (history up to t-1 only)."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "reports" / "artifacts" / "cumulative_signal_diagnostics",
    )
    return parser.parse_args()


def resolve_manifest(run_dir: Path, manifest_arg: Path | None) -> Path | None:
    if manifest_arg is not None:
        return manifest_arg.resolve()
    candidate = run_dir / "manifest.json"
    return candidate if candidate.exists() else None


def select_market_ids(
    all_market_ids: list[str],
    market_count: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[str]:
    if market_count <= 0:
        raise ValueError("--market-count must be > 0")

    ids = list(all_market_ids)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(ids)

    if market_count > len(ids):
        raise RuntimeError(
            f"Requested market_count={market_count} exceeds available markets={len(ids)}"
        )
    return ids[:market_count]


def chunked_market_ids(market_ids: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("--market-batch-size must be > 0")
    return [market_ids[i : i + batch_size] for i in range(0, len(market_ids), batch_size)]


def compute_relative_scores(
    features: pd.DataFrame,
    weights: SignalWeights,
    *,
    ewm_alpha: float,
    include_current_event_in_cumulative: bool,
) -> pd.DataFrame:
    weights.validate()

    required_cols = {
        "market_id",
        "token_id",
        "mid_price",
        "spread_bps",
        "ask_depth_1",
        "bid_depth_1",
    }
    missing = [col for col in required_cols if col not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    frame = features.reset_index().copy()
    if "ts_event" not in frame.columns:
        idx_col = features.index.name or "index"
        frame = frame.rename(columns={idx_col: "ts_event"})

    frame["market_id"] = frame["market_id"].astype(str)
    frame["token_id"] = frame["token_id"].astype(str)
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts_event", "market_id", "token_id"]).copy()

    for col in ["mid_price", "spread_bps", "ask_depth_1", "bid_depth_1"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame["total_depth_1"] = frame["ask_depth_1"].fillna(0.0) + frame["bid_depth_1"].fillna(0.0)
    frame["depth_pressure_1"] = frame["ask_depth_1"].fillna(0.0) - frame["bid_depth_1"].fillna(0.0)
    frame["abs_imbalance_1"] = frame["depth_pressure_1"].abs()

    group_keys = ["market_id", "ts_event"]
    stat_cols = ["spread_bps", "total_depth_1", "depth_pressure_1", "abs_imbalance_1"]

    # Fast path: one grouped aggregation pass, then join back.
    grouped_stats = frame.groupby(group_keys, observed=True)[stat_cols].agg(["mean", "min", "max"])
    grouped_stats.columns = [f"{col}_{stat}" for col, stat in grouped_stats.columns]
    frame = frame.join(grouped_stats, on=group_keys)

    spread_range = (frame["spread_bps_max"] - frame["spread_bps_min"]).replace(0.0, 1.0)
    depth_range = (frame["total_depth_1_max"] - frame["total_depth_1_min"]).replace(0.0, 1.0)
    pressure_range = (
        frame["depth_pressure_1_max"] - frame["depth_pressure_1_min"]
    ).replace(0.0, 1.0)
    imbalance_range = (
        frame["abs_imbalance_1_max"] - frame["abs_imbalance_1_min"]
    ).replace(0.0, 1.0)

    frame["relative_pressure"] = (
        frame["depth_pressure_1"] - frame["depth_pressure_1_mean"]
    ) / pressure_range
    frame["relative_spread_tightness"] = (
        frame["spread_bps_mean"] - frame["spread_bps"]
    ) / spread_range
    frame["relative_depth"] = (
        frame["total_depth_1"] - frame["total_depth_1_mean"]
    ) / depth_range
    frame["relative_imbalance"] = (
        frame["abs_imbalance_1"] - frame["abs_imbalance_1_mean"]
    ) / imbalance_range

    # Drop helper aggregate columns to keep memory footprint manageable downstream.
    helper_cols = [
        f"{col}_{stat}"
        for col in stat_cols
        for stat in ("mean", "min", "max")
    ]
    frame = frame.drop(columns=helper_cols)

    frame["snapshot_score"] = (
        weights.pressure_weight * frame["relative_pressure"]
        + weights.spread_weight * frame["relative_spread_tightness"]
        + weights.depth_weight * frame["relative_depth"]
        + weights.imbalance_weight * frame["relative_imbalance"]
    )

    frame = frame.sort_values(["market_id", "token_id", "ts_event"]).copy()

    # Fast path: compute both cumulative sum and EWM in one sequential pass.
    # Default is strict-causal (output at t uses history up to t-1 only).
    snapshot = frame["snapshot_score"].to_numpy(dtype="float64", copy=False)
    market = frame["market_id"]
    token = frame["token_id"]
    is_new_group = (market.ne(market.shift()) | token.ne(token.shift())).to_numpy()

    cum_values = np.empty_like(snapshot)
    ewm_values = np.empty_like(snapshot)
    alpha = float(ewm_alpha)
    one_minus_alpha = 1.0 - alpha

    running_sum = 0.0
    running_ewm = 0.0
    has_history = False
    for idx, value in enumerate(snapshot):
        if is_new_group[idx]:
            running_sum = 0.0
            running_ewm = 0.0
            has_history = False

        if include_current_event_in_cumulative:
            if not has_history:
                running_sum = value
                running_ewm = value
                has_history = True
            else:
                running_sum += value
                running_ewm = alpha * value + one_minus_alpha * running_ewm

            cum_values[idx] = running_sum
            ewm_values[idx] = running_ewm
            continue

        # Strict-causal output first, then update with current observation.
        cum_values[idx] = running_sum
        ewm_values[idx] = running_ewm if has_history else 0.0

        if not has_history:
            running_sum = value
            running_ewm = value
            has_history = True
        else:
            running_sum += value
            running_ewm = alpha * value + one_minus_alpha * running_ewm

    frame["cumulative_sum_score"] = cum_values
    frame["cumulative_ewm_score"] = ewm_values
    return frame


def winner_predictions(frame: pd.DataFrame, method_col: str) -> pd.DataFrame:
    keep_cols = ["market_id", "ts_event", "token_id", method_col]
    method_df = frame[keep_cols].copy()

    group_keys = ["market_id", "ts_event"]
    method_df["_rank"] = method_df.groupby(group_keys, observed=True)[method_col].rank(
        method="first", ascending=False
    )

    pred = method_df.loc[
        method_df["_rank"] == 1,
        ["market_id", "ts_event", "token_id", method_col],
    ].copy()
    pred = pred.rename(
        columns={
            "token_id": f"pred_token_{method_col}",
            method_col: f"pred_score_{method_col}",
        }
    )
    return pred


def attach_resolution(predictions: pd.DataFrame, resolution_frame: pd.DataFrame) -> pd.DataFrame:
    if "winning_asset_id" not in resolution_frame.columns:
        raise ValueError("Resolution frame missing 'winning_asset_id'")

    resolved = resolution_frame.copy()
    if "market_id" not in resolved.columns:
        resolved = resolved.reset_index()
    resolved = resolved[["market_id", "winning_asset_id"]].copy()
    resolved["market_id"] = resolved["market_id"].astype(str)
    resolved["winning_asset_id"] = resolved["winning_asset_id"].astype(str)

    out = predictions.merge(resolved, on="market_id", how="inner")
    return out


def compute_accuracy_table(scored: pd.DataFrame) -> pd.DataFrame:
    metric_rows: list[dict[str, object]] = []

    methods = [
        "snapshot_score",
        "cumulative_sum_score",
        "cumulative_ewm_score",
    ]

    for method in methods:
        pred_col = f"pred_token_{method}"
        tmp = scored[["market_id", "ts_event", "winning_asset_id", pred_col]].copy()
        tmp["is_correct"] = (
            tmp[pred_col].astype(str) == tmp["winning_asset_id"].astype(str)
        ).astype(int)

        timestamp_acc = float(tmp["is_correct"].mean()) if len(tmp) else 0.0

        final_rows = tmp.sort_values(["market_id", "ts_event"]).groupby(
            "market_id",
            as_index=False,
            observed=True,
        ).tail(1)
        final_acc = float(final_rows["is_correct"].mean()) if len(final_rows) else 0.0

        metric_rows.append(
            {
                "method": method,
                "timestamp_accuracy": timestamp_acc,
                "final_market_accuracy": final_acc,
                "timestamp_rows": len(tmp),
                "markets": int(tmp["market_id"].nunique()),
            }
        )

    table = pd.DataFrame(metric_rows).sort_values("final_market_accuracy", ascending=False)
    return table.reset_index(drop=True)


def compute_timeline_accuracy(scored: pd.DataFrame) -> pd.DataFrame:
    base = scored[["market_id", "ts_event", "winning_asset_id"]].copy()
    base = base.drop_duplicates(["market_id", "ts_event"]).sort_values(["market_id", "ts_event"])

    base["event_idx"] = base.groupby("market_id", observed=True).cumcount()
    base["event_count"] = base.groupby("market_id", observed=True)["event_idx"].transform("max") + 1
    denom = (base["event_count"] - 1).clip(lower=1)
    base["progress"] = base["event_idx"] / denom

    base["progress_bucket"] = pd.cut(
        base["progress"],
        bins=[-1e-9, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1_early", "Q2", "Q3", "Q4_late"],
        include_lowest=True,
    )

    rows: list[dict[str, object]] = []
    methods = ["snapshot_score", "cumulative_sum_score", "cumulative_ewm_score"]
    for method in methods:
        pred_col = f"pred_token_{method}"
        tmp = scored[["market_id", "ts_event", "winning_asset_id", pred_col]].copy()
        tmp["is_correct"] = (tmp[pred_col].astype(str) == tmp["winning_asset_id"].astype(str)).astype(int)
        tmp = tmp.merge(base[["market_id", "ts_event", "progress_bucket"]], on=["market_id", "ts_event"], how="left")

        bucket_acc = (
            tmp.groupby("progress_bucket", observed=True)["is_correct"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy", "count": "rows"})
        )
        for _, row in bucket_acc.iterrows():
            rows.append(
                {
                    "method": method,
                    "progress_bucket": str(row["progress_bucket"]),
                    "accuracy": float(row["accuracy"]),
                    "rows": int(row["rows"]),
                }
            )

    return pd.DataFrame(rows)


def compute_resolution_aligned_signal_evolution(scored_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute average signal evolution for winner vs loser books over time.

    The input frame must contain one row per (market_id, ts_event, token_id) with:
    - winning_asset_id
    - snapshot_score
    - cumulative_sum_score
    - cumulative_ewm_score
    """
    required = {
        "market_id",
        "ts_event",
        "token_id",
        "winning_asset_id",
        "snapshot_score",
        "cumulative_sum_score",
        "cumulative_ewm_score",
    }
    missing = [col for col in required if col not in scored_frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for evolution table: {missing}")

    frame = scored_frame.copy()
    frame["market_id"] = frame["market_id"].astype(str)
    frame["token_id"] = frame["token_id"].astype(str)
    frame["winning_asset_id"] = frame["winning_asset_id"].astype(str)
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts_event"]).copy()

    frame = frame.sort_values(["market_id", "ts_event", "token_id"]).copy()
    frame["event_idx"] = frame.groupby("market_id", observed=True)["ts_event"].rank(
        method="dense",
        ascending=True,
    ) - 1
    frame["event_count"] = frame.groupby("market_id", observed=True)["event_idx"].transform("max") + 1
    denom = (frame["event_count"] - 1).clip(lower=1)
    frame["progress"] = frame["event_idx"] / denom
    frame["progress_bucket"] = pd.cut(
        frame["progress"],
        bins=[-1e-9, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1_early", "Q2", "Q3", "Q4_late"],
        include_lowest=True,
    )

    frame["book_side"] = "loser_book"
    is_winner_book = frame["token_id"] == frame["winning_asset_id"]
    frame.loc[is_winner_book, "book_side"] = "winner_book"

    melt = frame.melt(
        id_vars=["market_id", "ts_event", "progress_bucket", "book_side"],
        value_vars=["snapshot_score", "cumulative_sum_score", "cumulative_ewm_score"],
        var_name="signal_type",
        value_name="signal_value",
    )

    summary = (
        melt.groupby(["signal_type", "progress_bucket", "book_side"], observed=True)[
            "signal_value"
        ]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "avg_signal", "count": "rows"})
    )

    pivot = summary.pivot_table(
        index=["signal_type", "progress_bucket"],
        columns="book_side",
        values="avg_signal",
        aggfunc="first",
    ).reset_index()

    if "winner_book" not in pivot.columns:
        pivot["winner_book"] = pd.NA
    if "loser_book" not in pivot.columns:
        pivot["loser_book"] = pd.NA

    pivot = pivot.rename(
        columns={
            "winner_book": "avg_signal_winner_book",
            "loser_book": "avg_signal_loser_book",
        }
    )
    pivot["winner_minus_loser"] = (
        pd.to_numeric(pivot["avg_signal_winner_book"], errors="coerce")
        - pd.to_numeric(pivot["avg_signal_loser_book"], errors="coerce")
    )
    return pivot.sort_values(["signal_type", "progress_bucket"]).reset_index(drop=True)


def _empty_accuracy_accumulator() -> dict[str, dict[str, float]]:
    methods = ["snapshot_score", "cumulative_sum_score", "cumulative_ewm_score"]
    return {
        method: {
            "timestamp_correct": 0.0,
            "timestamp_rows": 0.0,
            "final_correct": 0.0,
            "final_rows": 0.0,
            "markets": 0.0,
        }
        for method in methods
    }


def _update_accuracy_accumulator(acc: dict[str, dict[str, float]], merged_batch: pd.DataFrame) -> None:
    methods = ["snapshot_score", "cumulative_sum_score", "cumulative_ewm_score"]
    for method in methods:
        pred_col = f"pred_token_{method}"
        tmp = merged_batch[["market_id", "ts_event", "winning_asset_id", pred_col]].copy()
        tmp["is_correct"] = (
            tmp[pred_col].astype(str) == tmp["winning_asset_id"].astype(str)
        ).astype(int)

        acc[method]["timestamp_correct"] += float(tmp["is_correct"].sum())
        acc[method]["timestamp_rows"] += float(len(tmp))
        acc[method]["markets"] += float(tmp["market_id"].nunique())

        final_rows = tmp.sort_values(["market_id", "ts_event"]).groupby(
            "market_id",
            as_index=False,
            observed=True,
        ).tail(1)
        acc[method]["final_correct"] += float(final_rows["is_correct"].sum())
        acc[method]["final_rows"] += float(len(final_rows))


def _accuracy_table_from_accumulator(acc: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, stats in acc.items():
        ts_rows = max(stats["timestamp_rows"], 1.0)
        fin_rows = max(stats["final_rows"], 1.0)
        rows.append(
            {
                "method": method,
                "timestamp_accuracy": stats["timestamp_correct"] / ts_rows,
                "final_market_accuracy": stats["final_correct"] / fin_rows,
                "timestamp_rows": int(stats["timestamp_rows"]),
                "markets": int(stats["final_rows"]),
            }
        )
    return pd.DataFrame(rows).sort_values("final_market_accuracy", ascending=False).reset_index(drop=True)


def _update_timeline_accumulator(
    acc: dict[tuple[str, str], dict[str, float]],
    merged_batch: pd.DataFrame,
) -> None:
    base = merged_batch[["market_id", "ts_event", "winning_asset_id"]].copy()
    base = base.drop_duplicates(["market_id", "ts_event"]).sort_values(["market_id", "ts_event"])
    base["event_idx"] = base.groupby("market_id", observed=True).cumcount()
    base["event_count"] = (
        base.groupby("market_id", observed=True)["event_idx"].transform("max") + 1
    )
    denom = (base["event_count"] - 1).clip(lower=1)
    base["progress"] = base["event_idx"] / denom
    base["progress_bucket"] = pd.cut(
        base["progress"],
        bins=[-1e-9, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1_early", "Q2", "Q3", "Q4_late"],
        include_lowest=True,
    )

    methods = ["snapshot_score", "cumulative_sum_score", "cumulative_ewm_score"]
    for method in methods:
        pred_col = f"pred_token_{method}"
        tmp = merged_batch[["market_id", "ts_event", "winning_asset_id", pred_col]].copy()
        tmp["is_correct"] = (
            tmp[pred_col].astype(str) == tmp["winning_asset_id"].astype(str)
        ).astype(int)
        tmp = tmp.merge(
            base[["market_id", "ts_event", "progress_bucket"]],
            on=["market_id", "ts_event"],
            how="left",
        )
        grouped = (
            tmp.groupby("progress_bucket", observed=True)["is_correct"]
            .agg(["sum", "count"])
            .reset_index()
        )
        for _, row in grouped.iterrows():
            key = (method, str(row["progress_bucket"]))
            if key not in acc:
                acc[key] = {"sum": 0.0, "count": 0.0}
            acc[key]["sum"] += float(row["sum"])
            acc[key]["count"] += float(row["count"])


def _timeline_table_from_accumulator(
    acc: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (method, bucket), stats in acc.items():
        count = max(stats["count"], 1.0)
        rows.append(
            {
                "method": method,
                "progress_bucket": bucket,
                "accuracy": stats["sum"] / count,
                "rows": int(stats["count"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "progress_bucket"]).reset_index(drop=True)


def _update_evolution_accumulator(
    acc: dict[tuple[str, str, str], dict[str, float]],
    scored_with_resolution_batch: pd.DataFrame,
) -> None:
    frame = scored_with_resolution_batch.copy()
    frame["market_id"] = frame["market_id"].astype(str)
    frame["token_id"] = frame["token_id"].astype(str)
    frame["winning_asset_id"] = frame["winning_asset_id"].astype(str)
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts_event"])
    if frame.empty:
        return

    frame = frame.sort_values(["market_id", "ts_event", "token_id"])
    frame["event_idx"] = frame.groupby("market_id", observed=True)["ts_event"].rank(
        method="dense",
        ascending=True,
    ) - 1
    frame["event_count"] = (
        frame.groupby("market_id", observed=True)["event_idx"].transform("max") + 1
    )
    denom = (frame["event_count"] - 1).clip(lower=1)
    frame["progress"] = frame["event_idx"] / denom
    frame["progress_bucket"] = pd.cut(
        frame["progress"],
        bins=[-1e-9, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1_early", "Q2", "Q3", "Q4_late"],
        include_lowest=True,
    )
    frame["book_side"] = "loser_book"
    frame.loc[frame["token_id"] == frame["winning_asset_id"], "book_side"] = "winner_book"

    melt = frame.melt(
        id_vars=["progress_bucket", "book_side"],
        value_vars=["snapshot_score", "cumulative_sum_score", "cumulative_ewm_score"],
        var_name="signal_type",
        value_name="signal_value",
    )

    grouped = (
        melt.groupby(["signal_type", "progress_bucket", "book_side"], observed=True)[
            "signal_value"
        ]
        .agg(["sum", "count"])
        .reset_index()
    )
    for _, row in grouped.iterrows():
        key = (str(row["signal_type"]), str(row["progress_bucket"]), str(row["book_side"]))
        if key not in acc:
            acc[key] = {"sum": 0.0, "count": 0.0}
        acc[key]["sum"] += float(row["sum"])
        acc[key]["count"] += float(row["count"])


def _evolution_table_from_accumulator(
    acc: dict[tuple[str, str, str], dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (signal_type, bucket, side), stats in acc.items():
        count = max(stats["count"], 1.0)
        rows.append(
            {
                "signal_type": signal_type,
                "progress_bucket": bucket,
                "book_side": side,
                "avg_signal": stats["sum"] / count,
                "rows": int(stats["count"]),
            }
        )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    pivot = summary.pivot_table(
        index=["signal_type", "progress_bucket"],
        columns="book_side",
        values="avg_signal",
        aggfunc="first",
    ).reset_index()
    if "winner_book" not in pivot.columns:
        pivot["winner_book"] = pd.NA
    if "loser_book" not in pivot.columns:
        pivot["loser_book"] = pd.NA
    pivot = pivot.rename(
        columns={
            "winner_book": "avg_signal_winner_book",
            "loser_book": "avg_signal_loser_book",
        }
    )
    pivot["winner_minus_loser"] = (
        pd.to_numeric(pivot["avg_signal_winner_book"], errors="coerce")
        - pd.to_numeric(pivot["avg_signal_loser_book"], errors="coerce")
    )
    return pivot.sort_values(["signal_type", "progress_bucket"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    setup_application_logging()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path = resolve_manifest(run_dir, args.manifest)
    if args.ewm_alpha <= 0.0 or args.ewm_alpha > 1.0:
        raise ValueError("--ewm-alpha must be in (0, 1]")

    runner = BacktestRunner(storage_path=run_dir)
    runner.is_pmxt_mode = True

    all_market_ids = runner.load_prepared_feature_market_ids(
        limit_files=None,
        features_manifest_path=manifest_path,
        recursive_scan=True,
    )
    if not all_market_ids:
        raise RuntimeError("No prepared feature market IDs found")

    selected_market_ids = select_market_ids(
        all_market_ids,
        int(args.market_count),
        shuffle=bool(args.shuffle_markets),
        seed=int(args.seed),
    )

    print("=" * 80)
    print("CUMULATIVE SIGNAL RELIABILITY DIAGNOSTIC")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Total markets available: {len(all_market_ids)}")
    print(f"Markets selected: {len(selected_market_ids)}")
    print(f"Market batch size: {int(args.market_batch_size)}")
    print(f"Shuffle markets: {bool(args.shuffle_markets)}")
    print(f"Seed: {int(args.seed)}")
    print(
        "Causal cumulative mode: "
        f"{not bool(args.include_current_event_in_cumulative)}"
    )

    resolution = runner.load_prepared_resolution_frame(resolution_manifest_path=manifest_path)
    if resolution.empty:
        raise RuntimeError("Prepared resolution frame is empty")

    resolution = resolution.reset_index()
    resolution["market_id"] = resolution["market_id"].astype(str)
    resolution = resolution[resolution["market_id"].isin(set(selected_market_ids))].copy()
    resolution = resolution.dropna(subset=["winning_asset_id"]).copy()
    resolution["winning_asset_id"] = resolution["winning_asset_id"].astype(str)

    if resolution.empty:
        raise RuntimeError("No resolved markets with winning_asset_id in selected set")

    resolution_map = resolution[["market_id", "winning_asset_id"]].copy()
    valid_markets = set(resolution_map["market_id"].astype(str).unique().tolist())
    selected_resolved_market_ids = [
        market_id for market_id in selected_market_ids if market_id in valid_markets
    ]
    if not selected_resolved_market_ids:
        raise RuntimeError("No selected markets have resolution labels")

    market_batches = chunked_market_ids(
        selected_resolved_market_ids,
        int(args.market_batch_size),
    )
    print(f"Batches to process: {len(market_batches)}")

    run_id = datetime.now(UTC).strftime("cumdiag_%Y%m%dT%H%M%SZ")
    run_dir_out = args.output_root.resolve() / run_id
    run_dir_out.mkdir(parents=True, exist_ok=True)
    accuracy_csv = run_dir_out / "accuracy_summary.csv"
    timeline_csv = run_dir_out / "timeline_accuracy.csv"
    merged_csv = run_dir_out / "predictions_with_truth.csv"
    evolution_csv = run_dir_out / "signal_evolution_winner_vs_loser.csv"
    meta_json = run_dir_out / "run_metadata.json"

    weights = SignalWeights()
    accuracy_acc = _empty_accuracy_accumulator()
    timeline_acc: dict[tuple[str, str], dict[str, float]] = {}
    evolution_acc: dict[tuple[str, str, str], dict[str, float]] = {}

    for idx, batch_market_ids in enumerate(market_batches, start=1):
        print(f"Processing batch {idx}/{len(market_batches)} (markets={len(batch_market_ids)})")
        features = runner.load_prepared_features(
            features_manifest_path=manifest_path,
            market_ids=set(batch_market_ids),
            recursive_scan=True,
        )
        if features.empty:
            continue

        scored = compute_relative_scores(
            features,
            weights,
            ewm_alpha=float(args.ewm_alpha),
            include_current_event_in_cumulative=bool(
                args.include_current_event_in_cumulative
            ),
        )
        if scored.empty:
            continue

        scored = scored[scored["market_id"].astype(str).isin(set(batch_market_ids))].copy()
        if scored.empty:
            continue

        snapshot_pred = winner_predictions(scored, "snapshot_score")
        sum_pred = winner_predictions(scored, "cumulative_sum_score")
        ewm_pred = winner_predictions(scored, "cumulative_ewm_score")

        merged = snapshot_pred.merge(sum_pred, on=["market_id", "ts_event"], how="inner")
        merged = merged.merge(ewm_pred, on=["market_id", "ts_event"], how="inner")
        merged = attach_resolution(merged, resolution_map)
        if not merged.empty:
            _update_accuracy_accumulator(accuracy_acc, merged)
            _update_timeline_accumulator(timeline_acc, merged)

            write_header = not merged_csv.exists()
            merged.to_csv(
                merged_csv,
                index=False,
                mode="w" if write_header else "a",
                header=write_header,
            )

        scored_with_resolution = scored.merge(
            resolution_map,
            on="market_id",
            how="inner",
        )
        if not scored_with_resolution.empty:
            _update_evolution_accumulator(evolution_acc, scored_with_resolution)

    if not merged_csv.exists():
        raise RuntimeError("No prediction rows produced across batches")

    accuracy_table = _accuracy_table_from_accumulator(accuracy_acc)
    timeline_table = _timeline_table_from_accumulator(timeline_acc)
    evolution_table = _evolution_table_from_accumulator(evolution_acc)

    accuracy_table.to_csv(accuracy_csv, index=False)
    timeline_table.to_csv(timeline_csv, index=False)
    evolution_table.to_csv(evolution_csv, index=False)

    best_method = accuracy_table.iloc[0].to_dict() if not accuracy_table.empty else {}
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "market_count_requested": int(args.market_count),
        "market_count_selected": len(selected_market_ids),
        "market_count_with_resolution": len(selected_resolved_market_ids),
        "shuffle_markets": bool(args.shuffle_markets),
        "seed": int(args.seed),
        "market_batch_size": int(args.market_batch_size),
        "include_current_event_in_cumulative": bool(
            args.include_current_event_in_cumulative
        ),
        "ewm_alpha": float(args.ewm_alpha),
        "weights": {
            "pressure_weight": weights.pressure_weight,
            "spread_weight": weights.spread_weight,
            "depth_weight": weights.depth_weight,
            "imbalance_weight": weights.imbalance_weight,
        },
        "best_method_by_final_market_accuracy": best_method,
        "artifacts": {
            "accuracy_summary_csv": str(accuracy_csv),
            "timeline_accuracy_csv": str(timeline_csv),
            "predictions_with_truth_csv": str(merged_csv),
            "signal_evolution_winner_vs_loser_csv": str(evolution_csv),
        },
    }
    meta_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY")
    print("=" * 80)
    if accuracy_table.empty:
        print("No accuracy rows computed.")
    else:
        print(accuracy_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("TIMELINE ACCURACY (EARLY -> LATE)")
    print("=" * 80)
    if timeline_table.empty:
        print("No timeline rows computed.")
    else:
        print(timeline_table.to_string(index=False))

    print("\n" + "=" * 80)
    print("RESOLUTION-ALIGNED SIGNAL EVOLUTION (WINNER BOOK VS LOSER BOOK)")
    print("=" * 80)
    if evolution_table.empty:
        print("No evolution rows computed.")
    else:
        print(evolution_table.to_string(index=False))

    print("\nArtifacts written:")
    print(f"- {accuracy_csv}")
    print(f"- {timeline_csv}")
    print(f"- {merged_csv}")
    print(f"- {evolution_csv}")
    print(f"- {meta_json}")


if __name__ == "__main__":
    main()
