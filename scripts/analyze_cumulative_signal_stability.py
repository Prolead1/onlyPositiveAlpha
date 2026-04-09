"""Analyze confidence intervals and time stability for cumulative signal diagnostics.

Reads the `predictions_with_truth.csv` artifact produced by
`diagnose_cumulative_signal_reliability.py` and computes:

1) Method-level timestamp and final-per-market accuracy with Wilson 95% CI
2) Weekly and daily stability tables with Wilson 95% CI
3) Edge concentration summary for cumulative_sum_score

Outputs are written into the selected diagnostic run directory:
- confidence_interval_summary.csv
- weekly_accuracy_stability.csv
- daily_accuracy_stability.csv
- stability_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

METHODS = ("snapshot_score", "cumulative_sum_score", "cumulative_ewm_score")
METHOD_PRED_COLS = {method: f"pred_token_{method}" for method in METHODS}


@dataclass
class Accumulator:
    correct: int = 0
    rows: int = 0


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_diag_root = (
        project_root / "reports" / "artifacts" / "cumulative_signal_diagnostics"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Compute confidence intervals and daily/weekly stability from "
            "cumulative signal prediction artifacts"
        )
    )
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=default_diag_root,
        help="Directory containing cumdiag_* run folders",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific cumdiag_* run directory to analyze (default: latest)",
    )
    parser.add_argument(
        "--predictions-file",
        type=Path,
        default=None,
        help="Optional direct path to predictions_with_truth.csv",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="CSV chunk size for streaming aggregation",
    )
    return parser.parse_args()


def pick_run_dir(diag_root: Path, run_dir_arg: Path | None) -> Path:
    if run_dir_arg is not None:
        run_dir = run_dir_arg.resolve()
        if not run_dir.exists() or not run_dir.is_dir():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        return run_dir

    diag_root = diag_root.resolve()
    if not diag_root.exists() or not diag_root.is_dir():
        raise FileNotFoundError(f"Diagnostic root not found: {diag_root}")

    runs = sorted([p for p in diag_root.glob("cumdiag_*") if p.is_dir()])
    if not runs:
        raise RuntimeError(f"No cumdiag_* runs found in {diag_root}")
    return runs[-1]


def resolve_predictions_path(
    run_dir: Path,
    predictions_file_arg: Path | None,
) -> Path:
    if predictions_file_arg is not None:
        pred_path = predictions_file_arg.resolve()
    else:
        pred_path = run_dir / "predictions_with_truth.csv"

    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    return pred_path


def wilson_interval_95(correct: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")

    z = 1.959963984540054
    phat = correct / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half_width = (
        z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n) / denom
    )
    return center - half_width, center + half_width


def add_grouped_counts(
    frame: pd.DataFrame,
    group_col: str,
    is_correct: pd.Series,
    store: dict[tuple[str, str], Accumulator],
    method: str,
) -> None:
    grouped = (
        pd.DataFrame({"group": frame[group_col], "is_correct": is_correct})
        .groupby("group", observed=True)["is_correct"]
        .agg(["sum", "count"])
        .reset_index()
    )

    for _, row in grouped.iterrows():
        key = (method, str(row["group"]))
        acc = store[key]
        acc.correct += int(row["sum"])
        acc.rows += int(row["count"])


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    run_dir = pick_run_dir(args.diagnostic_root, args.run_dir)
    pred_path = resolve_predictions_path(run_dir, args.predictions_file)

    required_columns = [
        "market_id",
        "ts_event",
        "winning_asset_id",
        "pred_token_snapshot_score",
        "pred_token_cumulative_sum_score",
        "pred_token_cumulative_ewm_score",
    ]

    timestamp_acc: dict[str, Accumulator] = {m: Accumulator() for m in METHODS}
    daily_acc: dict[tuple[str, str], Accumulator] = defaultdict(Accumulator)
    weekly_acc: dict[tuple[str, str], Accumulator] = defaultdict(Accumulator)
    # market_id -> (latest_ts_event, correctness_by_method)
    final_by_market: dict[str, tuple[pd.Timestamp, dict[str, int]]] = {}

    reader = pd.read_csv(
        pred_path,
        usecols=required_columns,
        chunksize=int(args.chunk_size),
    )

    total_rows_seen = 0
    chunk_idx = 0
    for chunk in reader:
        chunk_idx += 1
        chunk["ts_event"] = pd.to_datetime(chunk["ts_event"], utc=True, errors="coerce")
        chunk = chunk.dropna(subset=["ts_event", "market_id", "winning_asset_id"]).copy()
        if chunk.empty:
            continue

        chunk["market_id"] = chunk["market_id"].astype(str)
        chunk["winning_asset_id"] = chunk["winning_asset_id"].astype(str)
        chunk["date"] = chunk["ts_event"].dt.date.astype(str)
        # Week bucket as Monday start date, computed without timezone-dropping period conversion.
        week_start = (
            chunk["ts_event"].dt.floor("D")
            - pd.to_timedelta(chunk["ts_event"].dt.weekday, unit="D")
        )
        chunk["week"] = week_start.dt.date.astype(str)

        total_rows_seen += len(chunk)

        for method, pred_col in METHOD_PRED_COLS.items():
            is_correct = (
                chunk[pred_col].astype(str) == chunk["winning_asset_id"]
            ).astype("int8")

            timestamp_acc[method].correct += int(is_correct.sum())
            timestamp_acc[method].rows += len(is_correct)

            add_grouped_counts(chunk, "date", is_correct, daily_acc, method)
            add_grouped_counts(chunk, "week", is_correct, weekly_acc, method)

        # Keep only latest timestamp row per market within this chunk before merging globally.
        latest_in_chunk = (
            chunk.sort_values(["market_id", "ts_event"]).groupby("market_id", as_index=False).tail(1)
        )
        for _, row in latest_in_chunk.iterrows():
            market_id = str(row["market_id"])
            ts_event = row["ts_event"]
            correctness = {
                method: int(str(row[pred_col]) == str(row["winning_asset_id"]))
                for method, pred_col in METHOD_PRED_COLS.items()
            }

            previous = final_by_market.get(market_id)
            if previous is None or ts_event >= previous[0]:
                final_by_market[market_id] = (ts_event, correctness)

        print(
            f"Processed chunks: {chunk_idx} | rows kept so far: {total_rows_seen}",
            flush=True,
        )

    if total_rows_seen == 0:
        raise RuntimeError("No valid rows found in predictions file after timestamp parsing")

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        ts_correct = timestamp_acc[method].correct
        ts_rows = timestamp_acc[method].rows
        final_correct = sum(v[1][method] for v in final_by_market.values())
        final_rows = len(final_by_market)

        ts_lo, ts_hi = wilson_interval_95(ts_correct, ts_rows)
        final_lo, final_hi = wilson_interval_95(final_correct, final_rows)

        summary_rows.append(
            {
                "method": method,
                "final_accuracy": (final_correct / final_rows) if final_rows else float("nan"),
                "final_n_markets": final_rows,
                "final_correct": final_correct,
                "final_ci95_lo": final_lo,
                "final_ci95_hi": final_hi,
                "timestamp_accuracy": (ts_correct / ts_rows) if ts_rows else float("nan"),
                "timestamp_n": ts_rows,
                "timestamp_correct": ts_correct,
                "timestamp_ci95_lo": ts_lo,
                "timestamp_ci95_hi": ts_hi,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("final_accuracy", ascending=False)

    def table_from_acc(store: dict[tuple[str, str], Accumulator], label: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for (method, bucket), acc in store.items():
            lo, hi = wilson_interval_95(acc.correct, acc.rows)
            rows.append(
                {
                    label: bucket,
                    "method": method,
                    "correct": acc.correct,
                    "rows": acc.rows,
                    "accuracy": (acc.correct / acc.rows) if acc.rows else float("nan"),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                }
            )
        if not rows:
            return pd.DataFrame(columns=[label, "method", "correct", "rows", "accuracy", "ci95_lo", "ci95_hi"])
        return pd.DataFrame(rows).sort_values(["method", label]).reset_index(drop=True)

    daily_df = table_from_acc(daily_acc, "date")
    weekly_df = table_from_acc(weekly_acc, "week")

    cumulative_weekly = weekly_df[weekly_df["method"] == "cumulative_sum_score"].copy()
    edge_diag = {
        "weeks_total": len(cumulative_weekly),
        "weeks_above_0_50": int((cumulative_weekly["accuracy"] > 0.50).sum()) if len(cumulative_weekly) else 0,
        "weeks_above_0_55": int((cumulative_weekly["accuracy"] > 0.55).sum()) if len(cumulative_weekly) else 0,
        "weeks_above_0_60": int((cumulative_weekly["accuracy"] > 0.60).sum()) if len(cumulative_weekly) else 0,
        "min_week_accuracy": float(cumulative_weekly["accuracy"].min()) if len(cumulative_weekly) else float("nan"),
        "max_week_accuracy": float(cumulative_weekly["accuracy"].max()) if len(cumulative_weekly) else float("nan"),
        "std_week_accuracy": float(cumulative_weekly["accuracy"].std(ddof=1)) if len(cumulative_weekly) > 1 else float("nan"),
        "worst_3_weeks": cumulative_weekly.nsmallest(3, "accuracy")[["week", "accuracy", "rows"]].to_dict(orient="records") if len(cumulative_weekly) else [],
        "best_3_weeks": cumulative_weekly.nlargest(3, "accuracy")[["week", "accuracy", "rows"]].to_dict(orient="records") if len(cumulative_weekly) else [],
    }

    summary_csv = run_dir / "confidence_interval_summary.csv"
    weekly_csv = run_dir / "weekly_accuracy_stability.csv"
    daily_csv = run_dir / "daily_accuracy_stability.csv"
    edge_json = run_dir / "stability_diagnostics.json"

    summary_df.to_csv(summary_csv, index=False)
    weekly_df.to_csv(weekly_csv, index=False)
    daily_df.to_csv(daily_csv, index=False)
    edge_json.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "predictions_file": str(pred_path),
                "rows_processed": total_rows_seen,
                "market_count_final": len(final_by_market),
                "edge_concentration_cumulative_sum": edge_diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 80)
    print("CUMULATIVE SIGNAL STABILITY ANALYSIS")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Predictions: {pred_path}")
    print(f"Rows processed: {total_rows_seen}")
    print(f"Markets in final snapshot: {len(final_by_market)}")

    print("\nCONFIDENCE SUMMARY")
    print(summary_df.to_string(index=False))

    print("\nEDGE CONCENTRATION (cumulative_sum_score weekly)")
    print(json.dumps(edge_diag, indent=2))

    print("\nArtifacts written:")
    print(f"- {summary_csv}")
    print(f"- {weekly_csv}")
    print(f"- {daily_csv}")
    print(f"- {edge_json}")


if __name__ == "__main__":
    main()
