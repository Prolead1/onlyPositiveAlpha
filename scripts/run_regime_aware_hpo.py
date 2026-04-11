"""Regime-aware HPO for relative-book-strength strategy.

Patterned after scripts/diagnose_relative_book_strategy.py:
- explicit market selection
- prepared-manifest batch backtests
- extract_metrics + objective ranking on DataFrame rows
- CSV/JSON artifacts
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

type StrategyCallable = Callable[[pd.DataFrame], pd.Series | pd.DataFrame]

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
from backtester import BacktestConfig, BacktestRunner, BacktestRunResult
from backtester.config.types import FeatureGatePolicy, ValidationPolicy
from backtester.simulation.objective import (
    ObjectiveConfig,
    compute_market_objective_metrics,
    derive_adaptive_drawdown_limit,
    rank_by_objective,
)
from utils import setup_application_logging

REGIME_SEQUENCE = ("risk-on", "risk-off", "consolidation")


def selected_regimes(args: argparse.Namespace) -> list[str]:
    """Return validated regime list from CLI arg, preserving input order."""
    raw = str(getattr(args, "target_regimes", "")).strip()
    if not raw:
        return list(REGIME_SEQUENCE)

    items = [part.strip() for part in raw.split(",") if part.strip()]
    if not items:
        return list(REGIME_SEQUENCE)

    invalid = [r for r in items if r not in REGIME_SEQUENCE]
    if invalid:
        msg = (
            f"Invalid regime(s): {invalid}. "
            f"Valid options are: {', '.join(REGIME_SEQUENCE)}"
        )
        raise ValueError(msg)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(items))


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


def default_strategy_params() -> StrategyParams:
    """Return canonical baseline params used for grid construction."""
    return StrategyParams(
        cumulative_signal_mode="sum",
        enable_secondary_gate_recalibration=False,
    )


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
        metrics_logging_enabled=False,
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


def build_objective_config(args: argparse.Namespace) -> ObjectiveConfig:
    return ObjectiveConfig(
        drawdown_quantile=float(args.objective_drawdown_quantile),
        min_markets=int(args.objective_min_markets),
        min_trades=int(args.objective_min_trades),
        default_initial_capital=100.0,
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_run_dir = project_root / "data" / "cached" / "pmxt_backtest" / "runs" / "btc-updown-5m"
    default_regime_csv = default_run_dir / "micro_regimes_2025-12-01_to_2026-03-31.csv"

    parser = argparse.ArgumentParser(description="Regime-aware HPO for relative-book-strength")
    parser.add_argument("--run-dir", type=Path, default=default_run_dir)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument(
        "--mapping-dir",
        type=Path,
        default=project_root / "data" / "cached" / "mapping",
    )
    parser.add_argument("--regime-csv", type=Path, default=default_regime_csv)
    parser.add_argument(
        "--target-regimes",
        type=str,
        default=",".join(REGIME_SEQUENCE),
        help=(
            "Comma-separated regimes to run (default: all). "
            "Example: risk-off,consolidation"
        ),
    )
    parser.add_argument("--market-batch-size", type=int, default=100)
    parser.add_argument(
        "--market-count",
        type=int,
        default=500,
        help="Number of markets to sample for per-regime HPO staging pool.",
    )

    parser.add_argument("--run-per-regime-hpo", action="store_true")
    parser.add_argument("--run-aggregate-validation", action="store_true")
    parser.add_argument(
        "--run-baseline-per-regime",
        action="store_true",
        help="Run the baseline (grid_006) strategy separately on each dominant regime.",
    )
    parser.add_argument(
        "--run-controlled-regime-comparison",
        action="store_true",
        help=(
            "Run baseline and regime-aware strategies on identical per-regime market slices "
            "and write side-by-side comparison artifacts."
        ),
    )

    parser.add_argument("--grid-confidence-min", type=str, default="0.75,0.80,0.85,0.90")
    parser.add_argument("--grid-score-gap-quantiles", type=str, default="0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--grid-buy-price-max", type=str, default="0.80,0.85,0.90,0.95")
    parser.add_argument("--grid-min-liquidity", type=str, default="0.05")
    parser.add_argument("--grid-ask-depth-5-max", type=str, default="500,800,1000,1500")
    parser.add_argument("--grid-max-time-secs", type=str, default="120,150,180")
    parser.add_argument("--grid-max-combos", type=int, default=36)
    parser.add_argument("--grid-seed", type=int, default=42)

    parser.add_argument("--hpo-per-regime-train-markets", type=int, default=300)
    parser.add_argument("--aggregate-market-count", type=int, default=500)
    parser.add_argument("--random-sample-markets", action="store_true")
    parser.add_argument("--sample-seed", type=int, default=42)

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_hpo_per_regime.csv",
        help="Where to write per-regime HPO results CSV.",
    )
    parser.add_argument(
        "--per-regime-output-csv",
        dest="output_csv",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_hpo_per_regime_top.json",
        help="Where to write per-regime top candidates JSON.",
    )
    parser.add_argument(
        "--per-regime-output-json",
        dest="output_json",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aggregate-output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_hpo_aggregate_result.csv",
        help="Where to write aggregate validation CSV.",
    )
    parser.add_argument(
        "--aggregate-output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_hpo_aggregate_result.json",
        help="Where to write aggregate validation JSON.",
    )
    parser.add_argument(
        "--baseline-per-regime-output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_baseline_per_regime.csv",
        help="Where to write baseline-per-regime validation CSV.",
    )
    parser.add_argument(
        "--baseline-per-regime-output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_baseline_per_regime.json",
        help="Where to write baseline-per-regime validation JSON.",
    )
    parser.add_argument(
        "--regime-winners-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_aware_hpo_per_regime_top.json",
        help="Per-regime winners JSON used to build controlled regime-aware comparison.",
    )
    parser.add_argument(
        "--controlled-comparison-output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_controlled_comparison.csv",
        help="Where to write controlled baseline-vs-regime-aware comparison CSV.",
    )
    parser.add_argument(
        "--controlled-comparison-output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "regime_controlled_comparison.json",
        help="Where to write controlled baseline-vs-regime-aware comparison JSON.",
    )

    parser.add_argument("--objective-drawdown-quantile", type=float, default=0.60)
    parser.add_argument("--objective-fallback-drawdown-pct", type=float, default=25.0)
    parser.add_argument(
        "--objective-hard-max-drawdown-pct",
        type=float,
        default=None,
        help="Optional strict max drawdown percent override.",
    )
    parser.add_argument("--objective-min-markets", type=int, default=30)
    parser.add_argument("--objective-min-trades", type=int, default=50)
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return [float(item) for item in values]


def make_grid_param_combinations(args: argparse.Namespace) -> list[StrategyParams]:
    base = default_strategy_params()
    conf = parse_float_list(args.grid_confidence_min)
    gap = parse_float_list(args.grid_score_gap_quantiles)
    buy = parse_float_list(args.grid_buy_price_max)
    liq = parse_float_list(args.grid_min_liquidity)
    ask5 = parse_float_list(args.grid_ask_depth_5_max)
    maxt = parse_float_list(args.grid_max_time_secs)

    combos: list[StrategyParams] = []
    for c, g, b, liq_min, a, t in itertools.product(conf, gap, buy, liq, ask5, maxt):
        combos.append(
            StrategyParams(
                confidence_score_min=float(c),
                relative_book_score_quantile=float(g),
                buy_price_max=float(b),
                min_liquidity=float(liq_min),
                ask_depth_5_max_filter=float(a),
                max_time_to_resolution_secs=float(t),
                spread_bps_narrow_quantile=base.spread_bps_narrow_quantile,
                dynamic_position_sizing=base.dynamic_position_sizing,
                use_cumulative_signal=base.use_cumulative_signal,
                cumulative_signal_mode=base.cumulative_signal_mode,
                pressure_weight=base.pressure_weight,
                spread_weight=base.spread_weight,
                depth_weight=base.depth_weight,
                imbalance_weight=base.imbalance_weight,
            )
        )

    if len(combos) <= int(args.grid_max_combos):
        return combos
    rng = random.Random(int(args.grid_seed))
    return rng.sample(combos, k=int(args.grid_max_combos))


def resolve_drawdown_limit(
    max_drawdown_series: pd.Series,
    *,
    args: argparse.Namespace,
    objective_config: ObjectiveConfig,
) -> float:
    if args.objective_hard_max_drawdown_pct is not None:
        return max(float(args.objective_hard_max_drawdown_pct), 0.0)

    return derive_adaptive_drawdown_limit(
        max_drawdown_series,
        quantile=objective_config.drawdown_quantile,
        fallback_limit_pct=float(args.objective_fallback_drawdown_pct),
    )


def extract_metrics(
    result_name: str,
    result: BacktestRunResult,
    *,
    objective_config: ObjectiveConfig,
    drawdown_limit_pct: float | None = None,
) -> dict[str, float | int | str]:
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


def params_to_metric_columns(params: StrategyParams) -> dict[str, float]:
    return {
        "confidence_score_min": float(params.confidence_score_min),
        "relative_book_score_quantile": float(params.relative_book_score_quantile),
        "buy_price_max": float(params.buy_price_max or 0.0),
        "min_liquidity": float(params.min_liquidity),
        "ask_depth_5_max_filter": float(params.ask_depth_5_max_filter or 0.0),
        "max_time_to_resolution_secs": float(params.max_time_to_resolution_secs or 0.0),
        "spread_bps_narrow_quantile": float(params.spread_bps_narrow_quantile),
    }


def strategy_params_from_metrics_row(row: pd.Series, *, base: StrategyParams) -> StrategyParams:
    return StrategyParams(
        confidence_score_min=float(row["confidence_score_min"]),
        relative_book_score_quantile=float(row["relative_book_score_quantile"]),
        buy_price_max=float(row["buy_price_max"]),
        min_liquidity=float(row["min_liquidity"]),
        ask_depth_5_max_filter=float(row["ask_depth_5_max_filter"]),
        max_time_to_resolution_secs=float(row["max_time_to_resolution_secs"]),
        spread_bps_narrow_quantile=float(row["spread_bps_narrow_quantile"]),
        dynamic_position_sizing=base.dynamic_position_sizing,
        use_cumulative_signal=base.use_cumulative_signal,
        cumulative_signal_mode=base.cumulative_signal_mode,
        cumulative_signal_alpha=base.cumulative_signal_alpha,
        pressure_weight=base.pressure_weight,
        spread_weight=base.spread_weight,
        depth_weight=base.depth_weight,
        imbalance_weight=base.imbalance_weight,
    )


def run_scenario(
    *,
    runner: BacktestRunner,
    cfg: BacktestConfig,
    strategy_name: str,
    strategy: StrategyCallable,
    mapping_dir: Path,
    manifest_path: Path | None,
    market_batch_size: int,
    market_ids: list[str],
    regime_csv_path: Path,
) -> BacktestRunResult:
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


def load_and_stratify_regimes(
    regime_csv_path: Path,
    market_ids: list[str],
    runner: BacktestRunner,
    manifest_path: Path | None,
) -> dict[str, list[str]]:
    regime_lookup = load_regime_lookup(str(regime_csv_path))
    if regime_lookup is None:
        msg = f"Unable to load regime CSV: {regime_csv_path}"
        raise ValueError(msg)

    features = runner.load_prepared_features(
        features_manifest_path=manifest_path,
        market_ids=set(market_ids),
        recursive_scan=True,
    )
    if features.empty:
        return {}

    features_frame = features
    if "market_id" not in features_frame.columns:
        if (
            isinstance(features_frame.index, pd.MultiIndex)
            and "market_id" in features_frame.index.names
        ):
            features_frame = features_frame.reset_index()
        else:
            return {}

    if "ts_event" in features_frame.columns:
        feature_ts = pd.to_datetime(features_frame["ts_event"], utc=True, errors="coerce")
    else:
        feature_ts = pd.Series(
            pd.to_datetime(features_frame.index, utc=True, errors="coerce"),
            index=features_frame.index,
        )

    regime_rows = pd.DataFrame(
        {
            "market_id": features_frame["market_id"].astype(str),
            "regime": feature_ts.dt.floor("5min").map(regime_lookup),
        }
    ).dropna(subset=["regime"])
    if regime_rows.empty:
        return {}

    # Vectorized dominant-regime resolution per market.
    regime_counts = (
        regime_rows.value_counts(["market_id", "regime"])
        .rename("count")
        .reset_index()
    )
    dominant_df = (
        regime_counts.sort_values(
            ["market_id", "count", "regime"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates(subset=["market_id"], keep="first")
    )
    dominant_by_market = dominant_df.set_index("market_id")["regime"]

    ordered_market_df = pd.DataFrame({"market_id": [str(m) for m in market_ids]})
    ordered_market_df["regime"] = ordered_market_df["market_id"].map(dominant_by_market)
    ordered_market_df = ordered_market_df.dropna(subset=["regime"])
    if ordered_market_df.empty:
        return {}

    grouped = ordered_market_df.groupby("regime", sort=False)["market_id"].agg(list)
    return {str(regime): mids for regime, mids in grouped.items()}


def sample_market_ids(
    pool: list[str],
    *,
    sample_size: int,
    rng: random.Random | None,
) -> list[str]:
    if sample_size <= 0 or not pool:
        return []
    k = min(len(pool), sample_size)
    if rng is None:
        return pool[:k]
    return rng.sample(pool, k=k)


def resolve_effective_market_batch_size(
    *,
    requested_batch_size: int,
    market_count: int,
    target_batches: int = 4,
) -> int:
    """Pick an effective batch size that keeps per-run batch progress informative.

    In regime-HPO loops, market slices are often small relative to the default
    batch size, which collapses progress to 1 batch and appears visually stuck.
    """
    if market_count <= 0:
        return max(1, requested_batch_size)

    requested = max(1, int(requested_batch_size))
    target = max(1, int(target_batches))
    adaptive = max(1, market_count // target)
    return min(requested, adaptive)


def run_per_regime_hpo(
    *,
    args: argparse.Namespace,
    runner: BacktestRunner,
    selected_market_ids: list[str],
    manifest_path: Path | None,
    objective_config: ObjectiveConfig,
    cfg: BacktestConfig,
    logger: logging.Logger,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    show_progress = bool(cfg.enable_progress_bars) and sys.stderr.isatty()
    target_regimes = selected_regimes(args)
    gate_flags = default_gate_flags()
    regime_market_map = load_and_stratify_regimes(
        args.regime_csv,
        selected_market_ids,
        runner,
        manifest_path,
    )

    logger.info("Stratified markets by dominant regime:")
    for regime, mids in regime_market_map.items():
        logger.info("  %s: %d markets", regime, len(mids))

    combos = make_grid_param_combinations(args)
    per_regime_rng = random.Random(int(args.sample_seed)) if args.random_sample_markets else None
    sampled_market_map: dict[str, list[str]] = {}
    for regime in target_regimes:
        mids = regime_market_map.get(regime, [])
        sampled_market_map[regime] = sample_market_ids(
            mids,
            sample_size=int(args.hpo_per_regime_train_markets),
            rng=per_regime_rng,
        )

    results_by_regime: dict[str, pd.DataFrame] = {}
    active_regimes = [r for r in target_regimes if sampled_market_map.get(r, [])]
    total_combo_runs = len(active_regimes) * len(combos)

    overall_pbar = None
    if show_progress and total_combo_runs > 0:
        overall_pbar = tqdm(
            total=total_combo_runs,
            desc="Per-regime HPO combos",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.25,
            unit="combo",
            position=2,
        )

    redirect_ctx = logging_redirect_tqdm() if show_progress else None
    if redirect_ctx is not None:
        redirect_ctx.__enter__()

    try:
        for regime_idx, regime in enumerate(target_regimes, start=1):
            sample = sampled_market_map.get(regime, [])
            if not sample:
                results_by_regime[regime] = pd.DataFrame()
                continue

            rows: list[dict[str, Any]] = []
            regime_pbar = None
            if show_progress:
                regime_pbar = tqdm(
                    total=len(combos),
                    desc=f"{regime} ({regime_idx}/{len(target_regimes)})",
                    leave=False,
                    dynamic_ncols=True,
                    mininterval=0.25,
                    unit="combo",
                    position=3,
                )

            for idx, params in enumerate(combos, start=1):
                scenario_name = f"{regime}_grid_{idx:03d}"
                effective_batch_size = resolve_effective_market_batch_size(
                    requested_batch_size=int(args.market_batch_size),
                    market_count=len(sample),
                    target_batches=4,
                )
                strategy = build_regime_aware_strategy(
                    regime_aware_params=RegimeAwareParams(
                        regime_params={regime: params},
                        baseline_params=BASELINE_PARAMS,
                    ),
                    regime_csv_path=str(args.regime_csv),
                    **gate_flags,
                )

                try:
                    result = run_scenario(
                        runner=runner,
                        cfg=cfg,
                        strategy_name=scenario_name,
                        strategy=strategy,
                        mapping_dir=args.mapping_dir,
                        manifest_path=manifest_path,
                        market_batch_size=effective_batch_size,
                        market_ids=sample,
                        regime_csv_path=args.regime_csv,
                    )
                except Exception as exc:
                    logger.warning("%s failed: %s", scenario_name, exc)
                    if regime_pbar is not None:
                        regime_pbar.update(1)
                    if overall_pbar is not None:
                        overall_pbar.update(1)
                    continue

                metric_row = extract_metrics(
                    scenario_name,
                    result,
                    objective_config=objective_config,
                )
                rows.append({**metric_row, **params_to_metric_columns(params)})

                if regime_pbar is not None:
                    regime_pbar.update(1)
                if overall_pbar is not None:
                    overall_pbar.update(1)

            if regime_pbar is not None:
                regime_pbar.close()

            regime_df = pd.DataFrame(rows)
            if regime_df.empty:
                results_by_regime[regime] = regime_df
                continue

            dd_limit = resolve_drawdown_limit(
                regime_df["max_drawdown_pct"],
                args=args,
                objective_config=objective_config,
            )
            regime_df = rank_by_objective(regime_df, drawdown_limit_pct=dd_limit)
            regime_df["objective_drawdown_limit_pct"] = float(dd_limit)
            regime_df["score"] = regime_df["market_sharpe_log"].astype(float)
            regime_df = regime_df.sort_values(
                [
                    "objective_eligible",
                    "score",
                    "market_sharpe_log",
                    "net_pnl",
                    "trades",
                ],
                ascending=[False, False, False, False, False],
            ).reset_index(drop=True)
            results_by_regime[regime] = regime_df

            if overall_pbar is not None:
                overall_pbar.set_postfix_str(f"last_regime={regime}")
    finally:
        if overall_pbar is not None:
            overall_pbar.close()
        if redirect_ctx is not None:
            redirect_ctx.__exit__(None, None, None)

    flat_rows: list[dict[str, Any]] = []
    for regime in target_regimes:
        regime_df = results_by_regime.get(regime, pd.DataFrame())
        if regime_df.empty:
            continue
        with_rank = regime_df.copy()
        with_rank.insert(0, "rank", range(1, len(with_rank) + 1))
        with_rank.insert(0, "regime", regime)
        records = [
            {str(k): v for k, v in rec.items()}
            for rec in with_rank.to_dict(orient="records")
        ]
        flat_rows.extend(records)

    return results_by_regime, pd.DataFrame(flat_rows)


def run_aggregate_validation(
    *,
    args: argparse.Namespace,
    runner: BacktestRunner,
    selected_market_ids: list[str],
    manifest_path: Path | None,
    regime_results: dict[str, pd.DataFrame],
    objective_config: ObjectiveConfig,
    cfg: BacktestConfig,
) -> dict[str, Any]:
    base = default_strategy_params()
    winners: dict[str, StrategyParams] = {}
    for regime in REGIME_SEQUENCE:
        regime_df = regime_results.get(regime, pd.DataFrame())
        if not regime_df.empty:
            winners[regime] = strategy_params_from_metrics_row(regime_df.iloc[0], base=base)
        else:
            winners[regime] = BASELINE_PARAMS

    strategy = build_regime_aware_strategy(
        regime_aware_params=RegimeAwareParams(
            regime_params=winners,
            baseline_params=BASELINE_PARAMS,
        ),
        regime_csv_path=str(args.regime_csv),
        **default_gate_flags(),
    )

    aggregate_rng = random.Random(int(args.sample_seed)) if args.random_sample_markets else None
    markets = sample_market_ids(
        selected_market_ids,
        sample_size=int(args.aggregate_market_count),
        rng=aggregate_rng,
    )

    result = run_scenario(
        runner=runner,
        cfg=cfg,
        strategy_name="aggregate_regime_aware",
        strategy=strategy,
        mapping_dir=args.mapping_dir,
        manifest_path=manifest_path,
        market_batch_size=int(args.market_batch_size),
        market_ids=markets,
        regime_csv_path=args.regime_csv,
    )

    metric_row = extract_metrics(
        "aggregate_validation",
        result,
        objective_config=objective_config,
    )
    metric_df = pd.DataFrame([metric_row])
    dd_limit = resolve_drawdown_limit(
        metric_df["max_drawdown_pct"],
        args=args,
        objective_config=objective_config,
    )
    ranked = rank_by_objective(metric_df, drawdown_limit_pct=dd_limit)
    ranked["objective_drawdown_limit_pct"] = float(dd_limit)
    ranked["score"] = ranked["market_sharpe_log"].astype(float)
    return {str(k): v for k, v in ranked.iloc[0].to_dict().items()}


def run_baseline_per_regime(
    *,
    args: argparse.Namespace,
    runner: BacktestRunner,
    selected_market_ids: list[str],
    manifest_path: Path | None,
    objective_config: ObjectiveConfig,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Run baseline strategy independently on each dominant regime slice."""
    regime_market_map = load_and_stratify_regimes(
        args.regime_csv,
        selected_market_ids,
        runner,
        manifest_path,
    )

    baseline_strategy = build_regime_aware_strategy(
        regime_aware_params=RegimeAwareParams(
            regime_params={},
            baseline_params=BASELINE_PARAMS,
        ),
        regime_csv_path=str(args.regime_csv),
        **default_gate_flags(),
    )

    baseline_df = run_strategy_per_regime(
        args=args,
        runner=runner,
        manifest_path=manifest_path,
        objective_config=objective_config,
        cfg=cfg,
        regime_market_map=regime_market_map,
        strategy=baseline_strategy,
        scenario_prefix="baseline",
    )

    if baseline_df.empty:
        return baseline_df

    dd_limit = resolve_drawdown_limit(
        baseline_df["max_drawdown_pct"],
        args=args,
        objective_config=objective_config,
    )
    baseline_df = rank_by_objective(baseline_df, drawdown_limit_pct=dd_limit)
    baseline_df["objective_drawdown_limit_pct"] = float(dd_limit)
    baseline_df["score"] = baseline_df["market_sharpe_log"].astype(float)
    baseline_df = baseline_df.sort_values("regime", kind="stable").reset_index(drop=True)
    return baseline_df


def run_strategy_per_regime(
    *,
    args: argparse.Namespace,
    runner: BacktestRunner,
    manifest_path: Path | None,
    objective_config: ObjectiveConfig,
    cfg: BacktestConfig,
    regime_market_map: dict[str, list[str]],
    strategy: StrategyCallable,
    scenario_prefix: str,
) -> pd.DataFrame:
    """Run one strategy independently on each dominant-regime market slice."""
    rows: list[dict[str, Any]] = []
    for regime in REGIME_SEQUENCE:
        mids = regime_market_map.get(regime, [])
        if not mids:
            continue

        effective_batch_size = resolve_effective_market_batch_size(
            requested_batch_size=int(args.market_batch_size),
            market_count=len(mids),
            target_batches=4,
        )

        scenario_name = f"{scenario_prefix}_{regime}"
        result = run_scenario(
            runner=runner,
            cfg=cfg,
            strategy_name=scenario_name,
            strategy=strategy,
            mapping_dir=args.mapping_dir,
            manifest_path=manifest_path,
            market_batch_size=effective_batch_size,
            market_ids=mids,
            regime_csv_path=args.regime_csv,
        )

        metric_row = extract_metrics(
            scenario_name,
            result,
            objective_config=objective_config,
        )
        metric_row["regime"] = regime
        metric_row["regime_market_count"] = len(mids)
        rows.append(metric_row)

    return pd.DataFrame(rows)


def load_winner_params_from_json(
    winners_json_path: Path,
) -> dict[str, StrategyParams]:
    """Load top-1 winners per regime and convert to StrategyParams map."""
    if not winners_json_path.exists():
        msg = f"Regime winners JSON not found: {winners_json_path}"
        raise FileNotFoundError(msg)

    payload = json.loads(winners_json_path.read_text(encoding="utf-8"))
    base = default_strategy_params()
    winners: dict[str, StrategyParams] = {}
    for regime in REGIME_SEQUENCE:
        entries = payload.get(regime, [])
        if not entries:
            winners[regime] = BASELINE_PARAMS
            continue

        first_row = pd.Series(entries[0])
        winners[regime] = strategy_params_from_metrics_row(first_row, base=base)
    return winners


def run_controlled_regime_comparison(
    *,
    args: argparse.Namespace,
    runner: BacktestRunner,
    selected_market_ids: list[str],
    manifest_path: Path | None,
    objective_config: ObjectiveConfig,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run baseline and regime-aware on identical regime slices and compute deltas."""
    regime_market_map = load_and_stratify_regimes(
        args.regime_csv,
        selected_market_ids,
        runner,
        manifest_path,
    )

    baseline_strategy = build_regime_aware_strategy(
        regime_aware_params=RegimeAwareParams(
            regime_params={},
            baseline_params=BASELINE_PARAMS,
        ),
        regime_csv_path=str(args.regime_csv),
        **default_gate_flags(),
    )
    baseline_df = run_strategy_per_regime(
        args=args,
        runner=runner,
        manifest_path=manifest_path,
        objective_config=objective_config,
        cfg=cfg,
        regime_market_map=regime_market_map,
        strategy=baseline_strategy,
        scenario_prefix="baseline",
    )

    winners = load_winner_params_from_json(args.regime_winners_json.resolve())
    regime_aware_strategy = build_regime_aware_strategy(
        regime_aware_params=RegimeAwareParams(
            regime_params=winners,
            baseline_params=BASELINE_PARAMS,
        ),
        regime_csv_path=str(args.regime_csv),
        **default_gate_flags(),
    )
    regime_aware_df = run_strategy_per_regime(
        args=args,
        runner=runner,
        manifest_path=manifest_path,
        objective_config=objective_config,
        cfg=cfg,
        regime_market_map=regime_market_map,
        strategy=regime_aware_strategy,
        scenario_prefix="regime_aware",
    )

    merge_keys = ["regime", "regime_market_count"]
    comparison = baseline_df.merge(
        regime_aware_df,
        on=merge_keys,
        how="inner",
        suffixes=("_baseline", "_regime_aware"),
    )

    delta_pairs = [
        ("trades", "delta_trades"),
        ("win_rate", "delta_win_rate"),
        ("net_pnl", "delta_net_pnl"),
        ("market_sharpe_log", "delta_market_sharpe_log"),
        ("max_drawdown_pct", "delta_max_drawdown_pct"),
        ("score", "delta_score"),
    ]
    for metric, delta_name in delta_pairs:
        left = f"{metric}_regime_aware"
        right = f"{metric}_baseline"
        if left in comparison.columns and right in comparison.columns:
            comparison[delta_name] = (
                comparison[left].astype(float) - comparison[right].astype(float)
            )

    comparison = comparison.sort_values("regime", kind="stable").reset_index(drop=True)
    return baseline_df, regime_aware_df, comparison


def main() -> None:
    args = parse_args()
    setup_application_logging()
    logger = logging.getLogger(__name__)

    run_dir = args.run_dir.resolve()
    manifest_path: Path | None = args.manifest.resolve() if args.manifest else None
    if manifest_path is None:
        candidate = run_dir / "manifest.json"
        manifest_path = candidate if candidate.exists() else None

    runner = BacktestRunner(storage_path=run_dir)
    runner.is_pmxt_mode = True

    all_market_ids = runner.load_prepared_feature_market_ids(
        limit_files=None,
        features_manifest_path=manifest_path,
        recursive_scan=True,
    )
    if not all_market_ids:
        raise RuntimeError("No prepared feature market IDs found")

    if (
        not args.run_per_regime_hpo
        and not args.run_aggregate_validation
        and not args.run_baseline_per_regime
        and not args.run_controlled_regime_comparison
    ):
        msg = (
            "Select at least one mode: --run-per-regime-hpo, "
            "--run-aggregate-validation, --run-baseline-per-regime, "
            "or --run-controlled-regime-comparison"
        )
        raise RuntimeError(msg)

    if args.market_count <= 0:
        raise RuntimeError("--market-count must be > 0")

    if args.market_count > len(all_market_ids):
        msg = (
            "Requested market count exceeds available markets: "
            f"market_count={args.market_count}, available={len(all_market_ids)}"
        )
        raise RuntimeError(msg)

    if args.random_sample_markets:
        rng = random.Random(int(args.sample_seed))
        selected_market_ids = rng.sample(
            all_market_ids,
            k=min(int(args.market_count), len(all_market_ids)),
        )
    else:
        selected_market_ids = all_market_ids[: int(args.market_count)]

    objective_config = build_objective_config(args)
    cfg = default_backtest_config()

    print("=" * 80)
    print("REGIME-AWARE RELATIVE BOOK HPO")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Mapping dir: {args.mapping_dir.resolve()}")
    print(f"Regime CSV: {args.regime_csv.resolve()}")
    print(f"Target regimes: {selected_regimes(args)}")
    print(f"Random sampling enabled: {bool(args.random_sample_markets)}")
    print(f"Sampling seed: {int(args.sample_seed)}")
    print(f"Selected markets: {len(selected_market_ids)}")
    print(f"Market batch size: {args.market_batch_size}")

    regime_results: dict[str, pd.DataFrame] = {}
    per_regime_df = pd.DataFrame()
    should_print_per_regime_summary = False
    per_regime_display_cols = [
        "regime",
        "rank",
        "scenario",
        "score",
        "objective_eligible",
        "max_drawdown_pct",
        "trades",
        "market_sharpe_log",
        "net_pnl",
        "confidence_score_min",
        "relative_book_score_quantile",
        "buy_price_max",
        "min_liquidity",
        "ask_depth_5_max_filter",
        "max_time_to_resolution_secs",
    ]
    if args.run_per_regime_hpo:
        print("\n" + "=" * 80)
        print("PER-REGIME HPO")
        print("=" * 80)
        regime_results, per_regime_df = run_per_regime_hpo(
            args=args,
            runner=runner,
            selected_market_ids=selected_market_ids,
            manifest_path=manifest_path,
            objective_config=objective_config,
            cfg=cfg,
            logger=logger,
        )

        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        per_regime_df.to_csv(args.output_csv, index=False)

        top_payload = {
            regime: regime_results.get(regime, pd.DataFrame()).head(3).to_dict(orient="records")
            for regime in selected_regimes(args)
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(top_payload, indent=2), encoding="utf-8")
        should_print_per_regime_summary = True

    if args.run_aggregate_validation:
        print("\n" + "=" * 80)
        print("AGGREGATE VALIDATION")
        print("=" * 80)
        aggregate = run_aggregate_validation(
            args=args,
            runner=runner,
            selected_market_ids=selected_market_ids,
            manifest_path=manifest_path,
            regime_results=regime_results,
            objective_config=objective_config,
            cfg=cfg,
        )

        aggregate_df = pd.DataFrame([aggregate])
        args.aggregate_output_csv.parent.mkdir(parents=True, exist_ok=True)
        aggregate_df.to_csv(args.aggregate_output_csv, index=False)

        args.aggregate_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_output_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

        print("\n" + "=" * 80)
        print("AGGREGATE VALIDATION RESULTS")
        print("=" * 80)
        aggregate_display_cols = [
            "scenario",
            "score",
            "objective_eligible",
            "max_drawdown_pct",
            "trades",
            "market_sharpe_log",
            "net_pnl",
            "win_rate",
            "objective_drawdown_limit_pct",
        ]
        aggregate_existing_cols = [c for c in aggregate_display_cols if c in aggregate_df.columns]
        print(aggregate_df[aggregate_existing_cols].to_string(index=False))
        print("\nAggregate artifacts written:")
        print(f"- {args.aggregate_output_csv}")
        print(f"- {args.aggregate_output_json}")

    if args.run_baseline_per_regime:
        print("\n" + "=" * 80)
        print("BASELINE BY REGIME")
        print("=" * 80)
        baseline_df = run_baseline_per_regime(
            args=args,
            runner=runner,
            selected_market_ids=selected_market_ids,
            manifest_path=manifest_path,
            objective_config=objective_config,
            cfg=cfg,
        )

        args.baseline_per_regime_output_csv.parent.mkdir(parents=True, exist_ok=True)
        baseline_df.to_csv(args.baseline_per_regime_output_csv, index=False)

        args.baseline_per_regime_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_per_regime_output_json.write_text(
            json.dumps(baseline_df.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )

        if baseline_df.empty:
            print("Baseline-per-regime run produced no rows.")
        else:
            baseline_display_cols = [
                "regime",
                "scenario",
                "score",
                "objective_eligible",
                "max_drawdown_pct",
                "trades",
                "market_sharpe_log",
                "net_pnl",
                "win_rate",
                "regime_market_count",
                "objective_drawdown_limit_pct",
            ]
            existing_cols = [c for c in baseline_display_cols if c in baseline_df.columns]
            print(baseline_df[existing_cols].to_string(index=False))

        print("\nBaseline-per-regime artifacts written:")
        print(f"- {args.baseline_per_regime_output_csv}")
        print(f"- {args.baseline_per_regime_output_json}")

    if args.run_controlled_regime_comparison:
        print("\n" + "=" * 80)
        print("CONTROLLED REGIME COMPARISON")
        print("=" * 80)
        baseline_df, regime_aware_df, comparison_df = run_controlled_regime_comparison(
            args=args,
            runner=runner,
            selected_market_ids=selected_market_ids,
            manifest_path=manifest_path,
            objective_config=objective_config,
            cfg=cfg,
        )

        args.controlled_comparison_output_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(args.controlled_comparison_output_csv, index=False)

        args.controlled_comparison_output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_by_regime": baseline_df.to_dict(orient="records"),
            "regime_aware_by_regime": regime_aware_df.to_dict(orient="records"),
            "comparison": comparison_df.to_dict(orient="records"),
        }
        args.controlled_comparison_output_json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        if comparison_df.empty:
            print("Controlled comparison produced no overlapping regime rows.")
        else:
            display_cols = [
                "regime",
                "regime_market_count",
                "scenario_baseline",
                "scenario_regime_aware",
                "market_sharpe_log_baseline",
                "market_sharpe_log_regime_aware",
                "delta_market_sharpe_log",
                "max_drawdown_pct_baseline",
                "max_drawdown_pct_regime_aware",
                "delta_max_drawdown_pct",
                "net_pnl_baseline",
                "net_pnl_regime_aware",
                "delta_net_pnl",
                "trades_baseline",
                "trades_regime_aware",
                "delta_trades",
            ]
            existing_cols = [c for c in display_cols if c in comparison_df.columns]
            print(comparison_df[existing_cols].to_string(index=False))

        print("\nControlled comparison artifacts written:")
        print(f"- {args.controlled_comparison_output_csv}")
        print(f"- {args.controlled_comparison_output_json}")

    if should_print_per_regime_summary:
        print("\n" + "=" * 80)
        print("TOP PER-REGIME CONFIGURATIONS")
        print("=" * 80)
        if per_regime_df.empty:
            print("Per-regime HPO produced no rows.")
        else:
            existing_cols = [c for c in per_regime_display_cols if c in per_regime_df.columns]
            top_by_regime = per_regime_df[existing_cols].groupby("regime", sort=False).head(3)
            print(top_by_regime.to_string(index=False))

        print("\nPer-regime artifacts written:")
        print(f"- {args.output_csv}")
        print(f"- {args.output_json}")


if __name__ == "__main__":
    main()
