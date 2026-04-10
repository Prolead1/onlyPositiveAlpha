"""Run strict OOS validation for the best relative-book strategy.

This script assumes hyperparameter optimization was performed on the first
N markets (default 3000) and validates the best configuration on the remaining
unseen markets.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphas.cumulative_relative_book_strength import (
    StrategyParams,
)
from alphas.cumulative_relative_book_strength import (
    build_relative_book_strength_strategy as build_shared_relative_book_strength_strategy,
)
from alphas.cumulative_relative_book_strength import (
    recommended_gate_flags as shared_recommended_gate_flags,
)
from backtester import BacktestConfig, BacktestRunner
from backtester.config.types import FeatureGatePolicy, ValidationPolicy
from backtester.simulation.objective import (
    ObjectiveConfig,
    compute_market_objective_metrics,
    derive_adaptive_drawdown_limit,
)
from utils import setup_application_logging


class MetricsResult(Protocol):
    """Minimal result interface required for metric extraction."""

    backtest_summary: pd.DataFrame
    diagnostics_by_market: pd.DataFrame
    trade_ledger: pd.DataFrame
    metadata: object
    resolution_frame: pd.DataFrame


def build_objective_config(args: argparse.Namespace) -> ObjectiveConfig:
    """Build objective config used by train/OOS comparison."""
    return ObjectiveConfig(
        drawdown_quantile=float(args.objective_drawdown_quantile),
        min_markets=int(args.objective_min_markets),
        min_trades=int(args.objective_min_trades),
        default_initial_capital=100.0,
    )


def best_hpo_strategy_params() -> StrategyParams:
    """Return the best HPO configuration from the first-3000 compact drawdown-aware run (grid_006)."""
    return StrategyParams(
        relative_book_score_quantile=0.55,
        spread_bps_narrow_quantile=0.15,
        confidence_score_min=0.78,
        min_liquidity=0.05,
        buy_price_max=0.92,
        min_time_to_resolution_secs=None,
        max_time_to_resolution_secs=180.0,
        ask_depth_5_max_filter=800.0,
        dynamic_position_sizing=True,
        dynamic_ask_depth_5_ref=1105.44,
        dynamic_mid_price_ref=0.75,
        use_cumulative_signal=True,
        cumulative_signal_mode="sum",
        cumulative_signal_alpha=0.20,
        pressure_weight=0.45,
        spread_weight=0.35,
        depth_weight=0.15,
        imbalance_weight=0.05,
        enable_secondary_gate_recalibration=False,
    )


def best_hpo_gate_flags() -> dict[str, bool]:
    """Return the gate policy used in HPO and recommended diagnostics."""
    return shared_recommended_gate_flags()


def default_backtest_config() -> BacktestConfig:
    """Mirror notebook-like backtest settings for comparability."""
    return BacktestConfig(
        mode="tolerant",
        shares=1.0,
        validation_policy=ValidationPolicy(),
        feature_gate_policy=FeatureGatePolicy(),
        order_lifecycle_enabled=True,
        order_ttl_seconds=5,
        order_allow_amendments=True,
        order_max_amendments=1,
        risk_max_active_positions=100,
        risk_max_concentration_pct=1.0,
        risk_max_gross_exposure=100.0,
        enable_progress_bars=True,
        metrics_logging_enabled=True,
        metrics_log_every_n_markets=50,
        retain_full_feature_frames=False,
        retain_strategy_signals=True,
        retain_market_events=False,
        sizing_policy="capped_kelly",
        sizing_fixed_notional=1.0,
        sizing_risk_budget_pct=0.01,
        sizing_kelly_fraction_cap=0.03,
        available_capital=100.0,
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_run_dir = project_root / "data" / "cached" / "pmxt_backtest" / "runs" / "btc-updown-5m"

    parser = argparse.ArgumentParser(
        description=(
            "Run strict out-of-sample validation for the best relative-book strategy "
            "on markets not used in HPO"
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=default_run_dir,
        help="Prepared backtest run directory containing features/ and manifest.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional explicit manifest path (defaults to <run-dir>/manifest.json if present)",
    )
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=project_root / "data" / "cached" / "mapping",
        help="Mapping directory passed into run_backtest",
    )
    parser.add_argument(
        "--market-batch-size",
        type=int,
        default=100,
        help="Market batch size for backtest runner",
    )
    parser.add_argument(
        "--hpo-train-market-count",
        type=int,
        default=3000,
        help="Number of leading markets assumed used by HPO train phase",
    )
    parser.add_argument(
        "--expected-oos-market-count",
        type=int,
        default=0,
        help="Fail if computed OOS tail size differs from this value (0 disables check)",
    )
    parser.add_argument(
        "--hpo-results-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "gate_grid_search_first3000_grid010_ddaware_compact.csv",
        help="CSV containing first-3000 compact drawdown-aware HPO results for train-side reference metrics",
    )
    parser.add_argument(
        "--hpo-best-scenario",
        type=str,
        default="grid_006",
        help="Scenario label in --hpo-results-csv used as train-side reference",
    )
    parser.add_argument(
        "--strategy-name",
        type=str,
        default="relative_book_oos_grid_006_first3000",
        help="Strategy label for the OOS run",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "relative_book_oos_validation_metrics.csv",
        help="Where to write one-row OOS metrics CSV",
    )
    parser.add_argument(
        "--output-comparison-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "relative_book_oos_train_vs_validation.csv",
        help="Where to write train-vs-OOS comparison CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "relative_book_oos_validation_summary.json",
        help="Where to write validation summary JSON",
    )
    parser.add_argument(
        "--objective-drawdown-quantile",
        type=float,
        default=0.60,
        help="Train-distribution quantile used to derive max drawdown limit.",
    )
    parser.add_argument(
        "--objective-fallback-drawdown-pct",
        type=float,
        default=25.0,
        help="Fallback drawdown limit when train distribution is unavailable.",
    )
    parser.add_argument(
        "--objective-min-markets",
        type=int,
        default=30,
        help="Minimum markets required for objective eligibility.",
    )
    parser.add_argument(
        "--objective-min-trades",
        type=int,
        default=50,
        help="Minimum trades required for objective eligibility.",
    )
    return parser.parse_args()


def extract_metrics(
    result_name: str,
    result: MetricsResult,
    *,
    objective_config: ObjectiveConfig,
    drawdown_limit_pct: float | None = None,
) -> dict[str, float | int | str]:
    """Extract common backtest summary metrics from a strategy run result."""
    summary = result.backtest_summary
    objective = compute_market_objective_metrics(
        result,
        drawdown_limit_pct=drawdown_limit_pct,
        objective_config=objective_config,
    )
    if summary.empty:
        return {
            "scenario": result_name,
            "trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "avg_net_pnl": 0.0,
            "avg_hold_hours": 0.0,
            "fee_drag_pct": 0.0,
            "market_count": objective.market_count,
            "mean_market_log_return": objective.mean_market_log_return,
            "market_log_return_vol": objective.market_log_return_vol,
            "market_sharpe_log": objective.market_sharpe_log,
            "max_drawdown_pct": objective.max_drawdown_pct,
            "objective_eligible": objective.objective_eligible,
        }

    row = summary.iloc[0]
    return {
        "scenario": result_name,
        "trades": int(row.get("trades", 0) or 0),
        "wins": int(row.get("wins", 0) or 0),
        "win_rate": float(row.get("win_rate", 0.0) or 0.0),
        "net_pnl": float(row.get("net_pnl", 0.0) or 0.0),
        "gross_pnl": float(row.get("gross_pnl", 0.0) or 0.0),
        "fees": float(row.get("fees", 0.0) or 0.0),
        "avg_net_pnl": float(row.get("avg_net_pnl", 0.0) or 0.0),
        "avg_hold_hours": float(row.get("avg_hold_hours", 0.0) or 0.0),
        "fee_drag_pct": float(row.get("fee_drag_pct", 0.0) or 0.0),
        "market_count": objective.market_count,
        "mean_market_log_return": objective.mean_market_log_return,
        "market_log_return_vol": objective.market_log_return_vol,
        "market_sharpe_log": objective.market_sharpe_log,
        "max_drawdown_pct": objective.max_drawdown_pct,
        "objective_eligible": objective.objective_eligible,
    }


def objective_score(metrics: dict[str, float | int | str]) -> float:
    """Primary optimization score aligned with HPO objective."""
    return float(metrics.get("market_sharpe_log", 0.0) or 0.0)


def load_train_reference(hpo_csv: Path, scenario: str) -> dict[str, float | int | str] | None:
    """Load train-side metrics for the chosen scenario from HPO artifacts."""
    if not hpo_csv.exists():
        print(f"Train reference CSV not found: {hpo_csv}")
        return None

    df = pd.read_csv(hpo_csv)
    if df.empty or "scenario" not in df.columns:
        print(f"Train reference CSV is empty or missing scenario column: {hpo_csv}")
        return None

    matches = df.loc[df["scenario"].astype(str) == scenario]
    if matches.empty:
        print(f"Scenario '{scenario}' not found in train reference CSV: {hpo_csv}")
        return None

    row = matches.iloc[0]
    return {
        "split": "train_hpo_first_n",
        "scenario": str(row.get("scenario", scenario)),
        "trades": int(row.get("trades", 0) or 0),
        "wins": int(row.get("wins", 0) or 0),
        "win_rate": float(row.get("win_rate", 0.0) or 0.0),
        "net_pnl": float(row.get("net_pnl", 0.0) or 0.0),
        "gross_pnl": float(row.get("gross_pnl", 0.0) or 0.0),
        "fees": float(row.get("fees", 0.0) or 0.0),
        "avg_net_pnl": float(row.get("avg_net_pnl", 0.0) or 0.0),
        "avg_hold_hours": float(row.get("avg_hold_hours", 0.0) or 0.0),
        "fee_drag_pct": float(row.get("fee_drag_pct", 0.0) or 0.0),
        "market_count": int(row.get("market_count", 0) or 0),
        "mean_market_log_return": float(row.get("mean_market_log_return", 0.0) or 0.0),
        "market_log_return_vol": float(row.get("market_log_return_vol", 0.0) or 0.0),
        "market_sharpe_log": float(row.get("market_sharpe_log", 0.0) or 0.0),
        "max_drawdown_pct": float(row.get("max_drawdown_pct", 0.0) or 0.0),
    }


def load_train_drawdown_limit(
    hpo_csv: Path,
    *,
    quantile: float,
    fallback_limit_pct: float,
) -> float:
    """Load adaptive drawdown threshold from full train candidate distribution."""
    if not hpo_csv.exists():
        return float(fallback_limit_pct)

    df = pd.read_csv(hpo_csv)
    if "max_drawdown_pct" not in df.columns:
        return float(fallback_limit_pct)

    return derive_adaptive_drawdown_limit(
        df["max_drawdown_pct"],
        quantile=quantile,
        fallback_limit_pct=fallback_limit_pct,
    )


def main() -> None:
    args = parse_args()
    setup_application_logging()
    objective_config = build_objective_config(args)

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path: Path | None = args.manifest.resolve() if args.manifest else None
    if manifest_path is None:
        candidate = run_dir / "manifest.json"
        manifest_path = candidate if candidate.exists() else None

    mapping_dir: Path = args.mapping_dir.resolve()
    if not mapping_dir.exists():
        raise FileNotFoundError(f"Mapping directory not found: {mapping_dir}")

    runner = BacktestRunner(storage_path=run_dir)
    runner.is_pmxt_mode = True

    all_market_ids = list(
        runner.load_prepared_feature_market_ids(
            limit_files=None,
            features_manifest_path=manifest_path,
            recursive_scan=True,
        )
    )
    if not all_market_ids:
        raise RuntimeError("No prepared feature market IDs found")

    train_count = int(args.hpo_train_market_count)
    if train_count <= 0:
        raise RuntimeError("--hpo-train-market-count must be > 0")
    if train_count >= len(all_market_ids):
        raise RuntimeError(
            "Train cutoff must be smaller than available market count: "
            f"train_cutoff={train_count}, available={len(all_market_ids)}"
        )

    train_market_ids = all_market_ids[:train_count]
    oos_market_ids = all_market_ids[train_count:]
    if not oos_market_ids:
        raise RuntimeError("Computed OOS market set is empty")

    expected_oos = int(args.expected_oos_market_count)
    if expected_oos > 0 and len(oos_market_ids) != expected_oos:
        raise RuntimeError(
            "Computed OOS tail size does not match expectation: "
            f"expected={expected_oos}, actual={len(oos_market_ids)}"
        )

    params = best_hpo_strategy_params()
    gate_flags = best_hpo_gate_flags()
    strategy = build_shared_relative_book_strength_strategy(params=params, **gate_flags)
    cfg = default_backtest_config()

    print("=" * 80)
    print("RELATIVE BOOK STRATEGY OOS VALIDATION")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Mapping dir: {mapping_dir}")
    print(f"Total markets available: {len(all_market_ids)}")
    print(f"HPO train cutoff (prefix): {train_count}")
    print(f"OOS tail markets: {len(oos_market_ids)}")
    print(f"Best scenario reference: {args.hpo_best_scenario}")
    print(f"Executing strategy name: {args.strategy_name}")

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name=str(args.strategy_name),
        market_batch_size=int(args.market_batch_size),
        prepared_feature_market_ids=set(oos_market_ids),
        config=cfg,
    )
    train_ref = load_train_reference(args.hpo_results_csv.resolve(), str(args.hpo_best_scenario))
    train_drawdown_limit = load_train_drawdown_limit(
        args.hpo_results_csv.resolve(),
        quantile=float(args.objective_drawdown_quantile),
        fallback_limit_pct=float(args.objective_fallback_drawdown_pct),
    )

    oos_metrics = extract_metrics(
        str(args.strategy_name),
        result,
        objective_config=objective_config,
        drawdown_limit_pct=train_drawdown_limit,
    )
    oos_metrics["score"] = objective_score(oos_metrics)
    oos_metrics["objective_drawdown_limit_pct"] = train_drawdown_limit
    oos_df = pd.DataFrame([oos_metrics])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    oos_df.to_csv(args.output_csv, index=False)
    comparison_rows: list[dict[str, float | int | str]] = []
    if train_ref is not None:
        train_ref["score"] = objective_score(train_ref)
        train_ref["objective_drawdown_limit_pct"] = train_drawdown_limit
        train_ref["objective_eligible"] = (
            float(train_ref.get("max_drawdown_pct", 0.0) or 0.0) <= train_drawdown_limit
        )
        comparison_rows.append(train_ref)

    comparison_rows.append(
        {
            "split": "oos_remaining",
            **oos_metrics,
        }
    )
    comparison_df = pd.DataFrame(comparison_rows)
    args.output_comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(args.output_comparison_csv, index=False)

    payload: dict[str, object] = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "paths": {
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "mapping_dir": str(mapping_dir),
        },
        "split": {
            "policy": "prefix_train_tail_oos",
            "hpo_train_market_count": train_count,
            "oos_market_count": len(oos_market_ids),
            "total_market_count": len(all_market_ids),
            "train_market_ids_head": train_market_ids[:5],
            "train_market_ids_tail": train_market_ids[-5:],
            "oos_market_ids_head": oos_market_ids[:5],
            "oos_market_ids_tail": oos_market_ids[-5:],
        },
        "best_strategy_reference": {
            "scenario": str(args.hpo_best_scenario),
            "params": {
                "relative_book_score_quantile": params.relative_book_score_quantile,
                "spread_bps_narrow_quantile": params.spread_bps_narrow_quantile,
                "confidence_score_min": params.confidence_score_min,
                "buy_price_max": params.buy_price_max,
                "min_liquidity": params.min_liquidity,
                "ask_depth_5_max_filter": params.ask_depth_5_max_filter,
                "max_time_to_resolution_secs": params.max_time_to_resolution_secs,
                "cumulative_signal_mode": params.cumulative_signal_mode,
                "cumulative_signal_alpha": params.cumulative_signal_alpha,
                "pressure_weight": params.pressure_weight,
                "spread_weight": params.spread_weight,
                "depth_weight": params.depth_weight,
                "imbalance_weight": params.imbalance_weight,
                "enable_secondary_gate_recalibration": params.enable_secondary_gate_recalibration,
                "secondary_gate_recalibration_frequency": params.secondary_gate_recalibration_frequency,
                "secondary_gate_lookback_window": params.secondary_gate_lookback_window,
            },
            "gates": gate_flags,
        },
        "metrics": {
            "train_reference": train_ref,
            "oos_validation": oos_metrics,
        },
        "artifacts": {
            "oos_metrics_csv": str(args.output_csv.resolve()),
            "comparison_csv": str(args.output_comparison_csv.resolve()),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("OOS VALIDATION SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("\nArtifacts written:")
    print(f"- {args.output_csv}")
    print(f"- {args.output_comparison_csv}")
    print(f"- {args.output_json}")


if __name__ == "__main__":
    main()
