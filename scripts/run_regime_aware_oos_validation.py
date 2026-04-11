"""Run strict OOS validation for production regime policy.

This script mirrors the split policy from Section 6.6:
- train: first N markets (default 3000)
- oos: remaining tail markets

Production policy implemented in this script:
- risk-on: baseline parameters (selected for production)
- risk-off: refined parameters from Section 6.7
- consolidation: no-trade (signals dropped)
- unmapped: no-trade (signals dropped)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphas.cumulative_relative_book_strength import StrategyParams
from alphas.relative_book_strength_regime_aware import (
    BASELINE_PARAMS,
    RegimeAwareParams,
    build_regime_aware_strategy,
    load_regime_lookup,
)
from backtester import BacktestConfig, BacktestRunner
from backtester.config.types import FeatureGatePolicy, ValidationPolicy
from backtester.simulation.objective import ObjectiveConfig, compute_market_objective_metrics
from scripts.run_regime_aware_hpo import REGIME_SEQUENCE
from scripts.run_relative_book_oos_validation import write_capital_evolution_figure
from utils import setup_application_logging


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_run_dir = project_root / "data" / "cached" / "pmxt_backtest" / "runs" / "btc-updown-5m"

    parser = argparse.ArgumentParser(
        description=(
            "Run train/OOS validation for regime-aware strategy using optimized "
            "per-regime parameter winners"
        )
    )
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=project_root / "data" / "cached" / "mapping",
    )
    parser.add_argument(
        "--regime-csv",
        type=Path,
        default=default_run_dir / "micro_regimes_2025-12-01_to_2026-03-31.csv",
    )
    parser.add_argument(
        "--allow-consolidation-trading",
        action="store_true",
        help="If set, do not disable consolidation trading (default: disabled).",
    )
    parser.add_argument("--market-batch-size", type=int, default=100)
    parser.add_argument("--hpo-train-market-count", type=int, default=3000)
    parser.add_argument("--expected-oos-market-count", type=int, default=0)
    parser.add_argument(
        "--strategy-name",
        type=str,
        default="regime_aware_oos_optimized_first3000",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_oos_validation_metrics.csv",
    )
    parser.add_argument(
        "--output-comparison-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_oos_train_vs_validation.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_oos_validation_summary.json",
    )
    parser.add_argument(
        "--output-regime-attribution-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_oos_per_regime_attribution.csv",
    )
    parser.add_argument(
        "--output-regime-attribution-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_oos_per_regime_attribution.json",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=project_root / "report" / "figures" / "section6" / "s6_regime_aware_oos_equity_capital_evolution.png",
    )
    parser.add_argument("--objective-min-markets", type=int, default=30)
    parser.add_argument("--objective-min-trades", type=int, default=50)
    return parser.parse_args()


def default_backtest_config() -> BacktestConfig:
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


def default_gate_flags() -> dict[str, bool]:
    return {
        "enable_spread_gate": False,
        "enable_score_gate": True,
        "enable_score_gap_gate": True,
        "enable_price_cap_gate": True,
        "enable_liquidity_gate": True,
        "enable_ask_depth_5_cap_gate": True,
        "enable_time_gate": True,
    }


def objective_config_from_args(args: argparse.Namespace) -> ObjectiveConfig:
    return ObjectiveConfig(
        drawdown_quantile=0.60,
        min_markets=int(args.objective_min_markets),
        min_trades=int(args.objective_min_trades),
        default_initial_capital=100.0,
    )


def build_report_policy_params() -> dict[str, StrategyParams]:
    """Return production regime policy parameters from Section 6.7."""
    risk_on = BASELINE_PARAMS
    risk_off = StrategyParams(
        confidence_score_min=0.82,
        relative_book_score_quantile=0.45,
        buy_price_max=0.88,
        min_liquidity=0.05,
        ask_depth_5_max_filter=800.0,
        max_time_to_resolution_secs=150.0,
        spread_bps_narrow_quantile=BASELINE_PARAMS.spread_bps_narrow_quantile,
        dynamic_position_sizing=BASELINE_PARAMS.dynamic_position_sizing,
        use_cumulative_signal=BASELINE_PARAMS.use_cumulative_signal,
        cumulative_signal_mode=BASELINE_PARAMS.cumulative_signal_mode,
        cumulative_signal_alpha=BASELINE_PARAMS.cumulative_signal_alpha,
        pressure_weight=BASELINE_PARAMS.pressure_weight,
        spread_weight=BASELINE_PARAMS.spread_weight,
        depth_weight=BASELINE_PARAMS.depth_weight,
        imbalance_weight=BASELINE_PARAMS.imbalance_weight,
        enable_secondary_gate_recalibration=BASELINE_PARAMS.enable_secondary_gate_recalibration,
        secondary_gate_recalibration_frequency=BASELINE_PARAMS.secondary_gate_recalibration_frequency,
        secondary_gate_lookback_window=BASELINE_PARAMS.secondary_gate_lookback_window,
    )
    return {
        "risk-on": risk_on,
        "risk-off": risk_off,
        # consolidation intentionally omitted from regime_params and disabled via no-trade wrapper
    }


def build_production_strategy(
    *,
    regime_csv: Path,
    disable_consolidation: bool,
) -> Any:
    """Build production strategy with optional consolidation no-trade policy."""
    regime_params = build_report_policy_params()
    core = build_regime_aware_strategy(
        regime_aware_params=RegimeAwareParams(
            regime_params=regime_params,
            baseline_params=BASELINE_PARAMS,
        ),
        regime_csv_path=str(regime_csv),
        **default_gate_flags(),
    )

    if not disable_consolidation:
        return core

    def _strategy(features: pd.DataFrame) -> pd.DataFrame:
        out = core(features)
        if "_regime" not in out.columns:
            return out
        # Remove blocked regimes entirely so the execution engine never
        # evaluates entries under those states.
        blocked_regimes = {"consolidation", "unmapped"}
        filtered = out[~out["_regime"].astype(str).isin(blocked_regimes)].copy()
        return filtered

    return _strategy


def extract_metrics(
    scenario: str,
    result: Any,
    *,
    objective_config: ObjectiveConfig,
) -> dict[str, Any]:
    objective = compute_market_objective_metrics(
        result,
        drawdown_limit_pct=None,
        objective_config=objective_config,
    )
    summary = result.backtest_summary
    if summary.empty:
        return {
            "scenario": scenario,
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
        "scenario": scenario,
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


def run_split(
    *,
    runner: BacktestRunner,
    cfg: BacktestConfig,
    mapping_dir: Path,
    manifest_path: Path | None,
    strategy: Any,
    strategy_name: str,
    market_batch_size: int,
    market_ids: list[str],
    regime_csv_path: Path,
) -> Any:
    return runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name=strategy_name,
        market_batch_size=market_batch_size,
        prepared_feature_market_ids=market_ids,
        config=cfg,
        regime_csv_path=str(regime_csv_path),
    )


def compute_exact_trade_regime_attribution(
    *,
    trade_ledger: pd.DataFrame,
    regime_lookup: dict[pd.Timestamp, str],
    scenario_prefix: str,
) -> pd.DataFrame:
    """Aggregate trade metrics by regime from one single run trade ledger.

    This attribution is additive by construction for trade-level sums like net PnL.
    """
    if trade_ledger is None or trade_ledger.empty:
        return pd.DataFrame(
            columns=[
                "regime",
                f"scenario_{scenario_prefix}",
                f"trades_{scenario_prefix}",
                f"wins_{scenario_prefix}",
                f"win_rate_{scenario_prefix}",
                f"net_pnl_{scenario_prefix}",
                f"gross_pnl_{scenario_prefix}",
                f"fees_{scenario_prefix}",
                f"avg_net_pnl_{scenario_prefix}",
            ]
        )

    frame = trade_ledger.copy()
    if "market_id" not in frame.columns or "net_pnl" not in frame.columns:
        raise RuntimeError("Trade ledger missing required columns: market_id/net_pnl")

    ts_col = "entry_ts" if "entry_ts" in frame.columns else "resolved_at"
    if ts_col not in frame.columns:
        raise RuntimeError("Trade ledger missing timestamp columns: entry_ts/resolved_at")

    ts = pd.to_datetime(frame[ts_col], utc=True, errors="coerce")
    frame["regime"] = ts.dt.floor("5min").map(regime_lookup).fillna("unmapped")
    frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce").fillna(0.0)
    frame["gross_pnl"] = pd.to_numeric(frame.get("gross_pnl", 0.0), errors="coerce").fillna(0.0)
    frame["fee_usdc"] = pd.to_numeric(frame.get("fee_usdc", 0.0), errors="coerce").fillna(0.0)

    grouped = frame.groupby("regime", sort=False)
    rows: list[dict[str, Any]] = []
    for regime, group in grouped:
        trades = len(group)
        wins = int((group["net_pnl"] > 0).sum())
        rows.append(
            {
                "regime": str(regime),
                f"scenario_{scenario_prefix}": str(scenario_prefix),
                f"trades_{scenario_prefix}": trades,
                f"wins_{scenario_prefix}": wins,
                f"win_rate_{scenario_prefix}": (wins / trades) if trades > 0 else 0.0,
                f"net_pnl_{scenario_prefix}": float(group["net_pnl"].sum()),
                f"gross_pnl_{scenario_prefix}": float(group["gross_pnl"].sum()),
                f"fees_{scenario_prefix}": float(group["fee_usdc"].sum()),
                f"avg_net_pnl_{scenario_prefix}": float(group["net_pnl"].mean()) if trades > 0 else 0.0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    ordered = [r for r in REGIME_SEQUENCE if r in set(out["regime"]) ]
    trailing = [r for r in out["regime"].tolist() if r not in ordered]
    order_map = {name: idx for idx, name in enumerate(ordered + trailing)}
    out["_order"] = out["regime"].map(order_map).fillna(9999)
    out = out.sort_values(["_order", "regime"], kind="stable").drop(columns=["_order"]).reset_index(drop=True)
    return out


def build_exact_oos_regime_attribution(
    *,
    baseline_oos_result: Any,
    regime_aware_oos_result: Any,
    regime_csv: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create exact additive regime attribution from single OOS runs."""
    regime_lookup = load_regime_lookup(str(regime_csv))
    if regime_lookup is None:
        raise RuntimeError(f"Unable to load regime CSV lookup: {regime_csv}")

    baseline_attr = compute_exact_trade_regime_attribution(
        trade_ledger=baseline_oos_result.trade_ledger,
        regime_lookup=regime_lookup,
        scenario_prefix="baseline",
    )
    regime_aware_attr = compute_exact_trade_regime_attribution(
        trade_ledger=regime_aware_oos_result.trade_ledger,
        regime_lookup=regime_lookup,
        scenario_prefix="regime_aware",
    )

    comparison = baseline_attr.merge(regime_aware_attr, on="regime", how="outer")
    comparison = comparison.fillna(0.0)

    for metric in ["trades", "wins", "win_rate", "net_pnl", "gross_pnl", "fees", "avg_net_pnl"]:
        left = f"{metric}_regime_aware"
        right = f"{metric}_baseline"
        if left in comparison.columns and right in comparison.columns:
            comparison[f"delta_{metric}"] = (
                comparison[left].astype(float) - comparison[right].astype(float)
            )

    # Integrity checks for additive attribution.
    baseline_total = float(pd.to_numeric(baseline_oos_result.trade_ledger["net_pnl"], errors="coerce").fillna(0.0).sum())
    regime_aware_total = float(pd.to_numeric(regime_aware_oos_result.trade_ledger["net_pnl"], errors="coerce").fillna(0.0).sum())
    baseline_by_regime = float(comparison.get("net_pnl_baseline", pd.Series(dtype=float)).sum())
    regime_aware_by_regime = float(comparison.get("net_pnl_regime_aware", pd.Series(dtype=float)).sum())

    integrity = {
        "baseline_total_net_pnl": baseline_total,
        "baseline_sum_regime_net_pnl": baseline_by_regime,
        "baseline_sum_diff": baseline_total - baseline_by_regime,
        "regime_aware_total_net_pnl": regime_aware_total,
        "regime_aware_sum_regime_net_pnl": regime_aware_by_regime,
        "regime_aware_sum_diff": regime_aware_total - regime_aware_by_regime,
    }

    return comparison.reset_index(drop=True), integrity


def main() -> None:
    args = parse_args()
    setup_application_logging()
    objective_config = objective_config_from_args(args)

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path: Path | None = args.manifest.resolve() if args.manifest else None
    if manifest_path is None:
        candidate = run_dir / "manifest.json"
        manifest_path = candidate if candidate.exists() else None

    mapping_dir = args.mapping_dir.resolve()
    if not mapping_dir.exists():
        raise FileNotFoundError(f"Mapping directory not found: {mapping_dir}")

    regime_csv = args.regime_csv.resolve()
    if not regime_csv.exists():
        raise FileNotFoundError(f"Regime CSV not found: {regime_csv}")

    disable_consolidation = not bool(args.allow_consolidation_trading)

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

    expected_oos = int(args.expected_oos_market_count)
    if expected_oos > 0 and len(oos_market_ids) != expected_oos:
        raise RuntimeError(
            "Computed OOS tail size does not match expectation: "
            f"expected={expected_oos}, actual={len(oos_market_ids)}"
        )

    strategy = build_production_strategy(
        regime_csv=regime_csv,
        disable_consolidation=disable_consolidation,
    )
    cfg = default_backtest_config()

    print("=" * 80)
    print("REGIME-AWARE OOS VALIDATION")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Mapping dir: {mapping_dir}")
    print(f"Regime CSV: {regime_csv}")
    print("Policy source: Section 6.7 production regime policy")
    print(f"Total markets available: {len(all_market_ids)}")
    print(f"HPO train cutoff (prefix): {train_count}")
    print(f"OOS tail markets: {len(oos_market_ids)}")

    print("\nSelected per-regime policy:")
    print(
        "- risk-on: risk-on_grid_000 "
        "(conf=0.78, gap=0.55, buy_cap=0.92, ask5=800, t=180)"
    )
    print(
        "- risk-off: risk-off_grid_017 "
        "(conf=0.82, gap=0.45, buy_cap=0.88, ask5=800, t=150)"
    )
    print(
        f"- consolidation: {'disabled (no-trade)' if disable_consolidation else 'enabled'}"
    )

    baseline_strategy = build_regime_aware_strategy(
        regime_aware_params=RegimeAwareParams(
            regime_params={},
            baseline_params=BASELINE_PARAMS,
        ),
        regime_csv_path=str(regime_csv),
        **default_gate_flags(),
    )

    train_result = run_split(
        runner=runner,
        cfg=cfg,
        mapping_dir=mapping_dir,
        manifest_path=manifest_path,
        strategy=strategy,
        strategy_name=f"{args.strategy_name}_train",
        market_batch_size=int(args.market_batch_size),
        market_ids=train_market_ids,
        regime_csv_path=regime_csv,
    )
    oos_result = run_split(
        runner=runner,
        cfg=cfg,
        mapping_dir=mapping_dir,
        manifest_path=manifest_path,
        strategy=strategy,
        strategy_name=str(args.strategy_name),
        market_batch_size=int(args.market_batch_size),
        market_ids=oos_market_ids,
        regime_csv_path=regime_csv,
    )
    baseline_oos_result = run_split(
        runner=runner,
        cfg=cfg,
        mapping_dir=mapping_dir,
        manifest_path=manifest_path,
        strategy=baseline_strategy,
        strategy_name="baseline_oos_single_run",
        market_batch_size=int(args.market_batch_size),
        market_ids=oos_market_ids,
        regime_csv_path=regime_csv,
    )

    figure_path = write_capital_evolution_figure(
        oos_result,
        args.output_figure.resolve(),
        initial_capital=float(cfg.available_capital or 100.0),
    )

    train_metrics = extract_metrics(
        "regime_aware_optimized_train_first_n",
        train_result,
        objective_config=objective_config,
    )
    oos_metrics = extract_metrics(
        str(args.strategy_name),
        oos_result,
        objective_config=objective_config,
    )

    oos_df = pd.DataFrame([oos_metrics])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    oos_df.to_csv(args.output_csv, index=False)

    comparison_df = pd.DataFrame(
        [
            {"split": "train_first_n", **train_metrics},
            {"split": "oos_remaining", **oos_metrics},
        ]
    )
    args.output_comparison_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(args.output_comparison_csv, index=False)

    regime_attribution_df, attribution_integrity = build_exact_oos_regime_attribution(
        baseline_oos_result=baseline_oos_result,
        regime_aware_oos_result=oos_result,
        regime_csv=regime_csv,
    )
    args.output_regime_attribution_csv.parent.mkdir(parents=True, exist_ok=True)
    regime_attribution_df.to_csv(args.output_regime_attribution_csv, index=False)

    regime_attribution_payload = {
        "method": "single_run_trade_ledger_additive",
        "rows": regime_attribution_df.to_dict(orient="records"),
        "integrity": attribution_integrity,
        "sorted_by_delta_sharpe": sorted(
            regime_attribution_df.to_dict(orient="records"),
            key=lambda r: float(r.get("delta_market_sharpe_log", 0.0)),
        ),
    }
    args.output_regime_attribution_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_regime_attribution_json.write_text(
        json.dumps(regime_attribution_payload, indent=2),
        encoding="utf-8",
    )

    payload: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "paths": {
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "mapping_dir": str(mapping_dir),
            "regime_csv": str(regime_csv),
        },
        "split": {
            "policy": "prefix_train_tail_oos",
            "hpo_train_market_count": train_count,
            "oos_market_count": len(oos_market_ids),
            "total_market_count": len(all_market_ids),
        },
        "policy": {
            "risk-on": {
                "scenario": "risk-on_grid_000",
                "confidence_score_min": 0.78,
                "relative_book_score_quantile": 0.55,
                "buy_price_max": 0.92,
                "min_liquidity": 0.05,
                "ask_depth_5_max_filter": 800.0,
                "max_time_to_resolution_secs": 180.0,
            },
            "risk-off": {
                "scenario": "risk-off_grid_017",
                "confidence_score_min": 0.82,
                "relative_book_score_quantile": 0.45,
                "buy_price_max": 0.88,
                "min_liquidity": 0.05,
                "ask_depth_5_max_filter": 800.0,
                "max_time_to_resolution_secs": 150.0,
            },
            "consolidation_trading_enabled": not disable_consolidation,
        },
        "metrics": {
            "train_reference": train_metrics,
            "baseline_oos_reference": extract_metrics(
                "baseline_oos_single_run",
                baseline_oos_result,
                objective_config=objective_config,
            ),
            "oos_validation": oos_metrics,
        },
        "attribution_integrity": attribution_integrity,
        "artifacts": {
            "oos_metrics_csv": str(args.output_csv.resolve()),
            "comparison_csv": str(args.output_comparison_csv.resolve()),
            "oos_regime_attribution_csv": str(args.output_regime_attribution_csv.resolve()),
            "oos_regime_attribution_json": str(args.output_regime_attribution_json.resolve()),
            "oos_capital_figure": str(figure_path) if figure_path is not None else None,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("TRAIN VS OOS (REGIME-AWARE)")
    print("=" * 80)
    display_cols = [
        "split",
        "scenario",
        "trades",
        "win_rate",
        "net_pnl",
        "market_sharpe_log",
        "max_drawdown_pct",
    ]
    print(comparison_df[display_cols].to_string(index=False))

    if not regime_attribution_df.empty:
        print("\n" + "=" * 80)
        print("OOS REGIME ATTRIBUTION (BASELINE VS REGIME-AWARE)")
        print("=" * 80)
        attribution_cols = [
            "regime",
            "net_pnl_baseline",
            "net_pnl_regime_aware",
            "delta_net_pnl",
            "trades_baseline",
            "trades_regime_aware",
            "delta_trades",
        ]
        print(regime_attribution_df[attribution_cols].to_string(index=False))
        print("\nAttribution integrity checks:")
        print(
            "- baseline total vs sum(regimes): "
            f"{attribution_integrity['baseline_total_net_pnl']:.6f} vs "
            f"{attribution_integrity['baseline_sum_regime_net_pnl']:.6f} "
            f"(diff={attribution_integrity['baseline_sum_diff']:.12f})"
        )
        print(
            "- regime-aware total vs sum(regimes): "
            f"{attribution_integrity['regime_aware_total_net_pnl']:.6f} vs "
            f"{attribution_integrity['regime_aware_sum_regime_net_pnl']:.6f} "
            f"(diff={attribution_integrity['regime_aware_sum_diff']:.12f})"
        )

    print("\nArtifacts written:")
    print(f"- {args.output_csv.resolve()}")
    print(f"- {args.output_comparison_csv.resolve()}")
    print(f"- {args.output_json.resolve()}")
    print(f"- {args.output_regime_attribution_csv.resolve()}")
    print(f"- {args.output_regime_attribution_json.resolve()}")
    if figure_path is not None:
        print(f"- {figure_path}")


if __name__ == "__main__":
    main()
