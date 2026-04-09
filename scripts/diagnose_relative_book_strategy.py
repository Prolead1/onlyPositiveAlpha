"""Diagnose relative book strength strategy gates with the backtesting framework.

This script compares:
1) Base strategy (token winner only, no gates)
2) Cumulative impact as gates are added one-by-one
3) Leave-one-gate-out impact from the full gated strategy

It then prints quantitative recommendations based on the observed tradeoff
between net PnL, win rate, and trade count.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
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
from backtester import BacktestConfig, BacktestRunner
from backtester.config.types import FeatureGatePolicy, ValidationPolicy
from utils import setup_application_logging

GATE_SEQUENCE = [
    "spread_gate",
    "score_gate",
    "score_gap_gate",
    "price_cap_gate",
    "liquidity_gate",
    "ask_depth_5_cap_gate",
    "time_gate",
]

WIN_RATE_TOLERANCE_FOR_GATE_RELAX = -0.01


def recommended_gate_flags() -> dict[str, bool]:
    """Return the gate policy recommended by the latest diagnostics."""
    return {
        "enable_spread_gate": False,
        "enable_score_gate": True,
        "enable_score_gap_gate": False,
        "enable_price_cap_gate": True,
        "enable_liquidity_gate": True,
        "enable_ask_depth_5_cap_gate": True,
        "enable_time_gate": True,
    }


def default_strategy_params() -> StrategyParams:
    """Return canonical strategy params for diagnostics (cumulative-sum first)."""
    return StrategyParams(cumulative_signal_mode="sum")


class MetricsResult(Protocol):
    """Minimal result interface required for metric extraction."""

    backtest_summary: pd.DataFrame


def hpo_gate_flags() -> dict[str, bool]:
    """Gate policy for HPO based on latest diagnostics."""
    return recommended_gate_flags()


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

    parser = argparse.ArgumentParser(description="Diagnose relative-book strategy gates")
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
        "--market-count",
        type=int,
        default=500,
        help="Number of markets to sample from prepared run for diagnostics",
    )
    parser.add_argument(
        "--sample-pool-market-count",
        type=int,
        default=None,
        help=(
            "Optional prefix pool size for random market sampling in base diagnostics. "
            "If set, sampling is performed from the first N markets only."
        ),
    )
    parser.add_argument(
        "--random-sample-markets",
        action="store_true",
        help="Randomly sample selected markets (without replacement) from the chosen pool.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used when --random-sample-markets is enabled.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "gate_diagnostics.csv",
        help="Where to write scenario metrics CSV",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "gate_recommendations.json",
        help="Where to write recommendation payload JSON",
    )
    parser.add_argument(
        "--run-grid-search",
        action="store_true",
        help="Run threshold grid search using the recommended gated strategy.",
    )
    parser.add_argument(
        "--grid-spread-quantiles",
        type=str,
        default="0.15",
        help="Comma-separated spread quantile candidates.",
    )
    parser.add_argument(
        "--grid-confidence-min",
        type=str,
        default="0.70,0.75",
        help="Comma-separated confidence score threshold candidates.",
    )
    parser.add_argument(
        "--grid-score-gap-quantiles",
        type=str,
        default="0.60",
        help="Comma-separated score-gap quantile candidates.",
    )
    parser.add_argument(
        "--grid-buy-price-max",
        type=str,
        default="0.85,0.88",
        help="Comma-separated buy price cap candidates.",
    )
    parser.add_argument(
        "--grid-max-time-secs",
        type=str,
        default="180,240",
        help="Comma-separated max time-to-resolution candidates.",
    )
    parser.add_argument(
        "--grid-max-combos",
        type=int,
        default=12,
        help="Maximum combinations to evaluate from the Cartesian grid.",
    )
    parser.add_argument(
        "--grid-seed",
        type=int,
        default=42,
        help="Random seed used when downsampling combinations.",
    )
    parser.add_argument(
        "--grid-output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "gate_grid_search.csv",
        help="Where to write grid-search metrics CSV.",
    )
    parser.add_argument(
        "--grid-output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "gate_grid_search_top10.json",
        help="Where to write top grid-search candidates JSON.",
    )
    parser.add_argument(
        "--run-candidate-hpo",
        action="store_true",
        help="Run focused HPO over the 3 shortlisted candidate configurations.",
    )
    parser.add_argument(
        "--hpo-market-count",
        type=int,
        default=500,
        help="Market sample size for candidate HPO mode.",
    )
    parser.add_argument(
        "--hpo-output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "candidate_hpo_results.csv",
        help="Where to write focused candidate HPO results CSV.",
    )
    parser.add_argument(
        "--hpo-output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "candidate_hpo_top.json",
        help="Where to write focused candidate HPO top result JSON.",
    )
    parser.add_argument(
        "--run-train-validate-hpo",
        action="store_true",
        help=(
            "Run grid HPO on a larger training split, then validate top candidates "
            "on a disjoint smaller validation split."
        ),
    )
    parser.add_argument(
        "--hpo-train-market-count",
        type=int,
        default=1500,
        help="Market sample size for the HPO training split.",
    )
    parser.add_argument(
        "--hpo-validation-market-count",
        type=int,
        default=300,
        help="Market sample size for the HPO validation split (disjoint from train).",
    )
    parser.add_argument(
        "--hpo-split-pool-market-count",
        type=int,
        default=2000,
        help=(
            "Use only the first N markets as the split pool for train/validation HPO. "
            "Then shuffle this pool and split into train/validation sets."
        ),
    )
    parser.add_argument(
        "--hpo-top-k",
        type=int,
        default=3,
        help="Number of top train candidates to re-evaluate on validation.",
    )
    parser.add_argument(
        "--hpo-train-validate-output-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "train_validate_hpo_results.csv",
        help="Where to write train+validation HPO comparison results CSV.",
    )
    parser.add_argument(
        "--hpo-train-validate-output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "train_validate_hpo_top.json",
        help="Where to write the top validated configuration JSON.",
    )
    parser.add_argument(
        "--shuffle-markets-before-split",
        action="store_true",
        help=(
            "Shuffle market IDs before train/validation splitting. "
            "Default is a chronological no-shuffle split."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle-markets-before-split is enabled.",
    )
    parser.add_argument(
        "--adaptive-search-space",
        action="store_true",
        help=(
            "Derive grid candidate lists from historical optimization artifacts "
            "instead of broad defaults."
        ),
    )
    parser.add_argument(
        "--adaptive-source-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "train_validate_hpo_results.csv",
        help="Historical optimization CSV used for adaptive search-space derivation.",
    )
    parser.add_argument(
        "--adaptive-top-n",
        type=int,
        default=10,
        help="Use top-N rows from --adaptive-source-csv to derive adaptive search candidates.",
    )
    parser.add_argument(
        "--run-artifacts-root",
        type=Path,
        default=project_root / "reports" / "artifacts" / "hpo_runs",
        help="Root directory where per-run train/validation HPO artifact bundles are written.",
    )
    return parser.parse_args()


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return [float(item) for item in values]


def _format_float_list(values: list[float]) -> str:
    unique_sorted = sorted({float(v) for v in values})
    return ",".join(f"{v:.10g}" for v in unique_sorted)


def derive_adaptive_search_space(args: argparse.Namespace) -> dict[str, str] | None:
    """Build adaptive grid lists from historical optimization artifacts.

    Returns a dict of string overrides for grid-* argparse fields or None when
    adaptive mode is unavailable.
    """
    source = args.adaptive_source_csv.resolve()
    if not source.exists():
        print(f"Adaptive source CSV not found, falling back to explicit grid lists: {source}")
        return None

    df = pd.read_csv(source)
    if df.empty:
        print(f"Adaptive source CSV is empty, falling back to explicit grid lists: {source}")
        return None

    score_col = "validation_score" if "validation_score" in df.columns else "score"
    if score_col not in df.columns:
        print(
            "Adaptive source CSV has no score column (expected 'validation_score' or 'score'); "
            "falling back to explicit grid lists"
        )
        return None

    needed = {
        "spread_bps_narrow_quantile",
        "confidence_score_min",
        "relative_book_score_quantile",
        "buy_price_max",
        "max_time_to_resolution_secs",
    }
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print(
            f"Adaptive source CSV missing required columns {missing}; "
            "falling back to explicit grid lists"
        )
        return None

    top_n = max(1, int(args.adaptive_top_n))
    top = df.sort_values(score_col, ascending=False).head(top_n).copy()

    adaptive = {
        "grid_spread_quantiles": _format_float_list(
            top["spread_bps_narrow_quantile"].astype(float).tolist()
        ),
        "grid_confidence_min": _format_float_list(
            top["confidence_score_min"].astype(float).tolist()
        ),
        "grid_score_gap_quantiles": _format_float_list(
            top["relative_book_score_quantile"].astype(float).tolist()
        ),
        "grid_buy_price_max": _format_float_list(top["buy_price_max"].astype(float).tolist()),
        "grid_max_time_secs": _format_float_list(
            top["max_time_to_resolution_secs"].astype(float).tolist()
        ),
    }

    print("Adaptive search space derived from historical optimization data:")
    print(f"  source_csv={source}")
    print(f"  top_n={top_n}")
    print(f"  spread={adaptive['grid_spread_quantiles']}")
    print(f"  confidence={adaptive['grid_confidence_min']}")
    print(f"  score_gap={adaptive['grid_score_gap_quantiles']}")
    print(f"  buy_price={adaptive['grid_buy_price_max']}")
    print(f"  max_time={adaptive['grid_max_time_secs']}")
    return adaptive


def make_grid_param_combinations(args: argparse.Namespace) -> list[StrategyParams]:
    base = default_strategy_params()
    spread_list = parse_float_list(args.grid_spread_quantiles)
    confidence_list = parse_float_list(args.grid_confidence_min)
    score_gap_list = parse_float_list(args.grid_score_gap_quantiles)
    buy_price_list = parse_float_list(args.grid_buy_price_max)
    max_time_list = parse_float_list(args.grid_max_time_secs)

    combos: list[StrategyParams] = []
    product_iter = itertools.product(
        spread_list,
        confidence_list,
        score_gap_list,
        buy_price_list,
        max_time_list,
    )
    for spread_q, conf_min, score_gap_q, buy_cap, max_time in product_iter:
        combos.append(
            StrategyParams(
                relative_book_score_quantile=float(score_gap_q),
                spread_bps_narrow_quantile=float(spread_q),
                confidence_score_min=float(conf_min),
                min_liquidity=base.min_liquidity,
                buy_price_max=float(buy_cap),
                min_time_to_resolution_secs=base.min_time_to_resolution_secs,
                max_time_to_resolution_secs=float(max_time),
                ask_depth_5_max_filter=base.ask_depth_5_max_filter,
                dynamic_position_sizing=base.dynamic_position_sizing,
                dynamic_ask_depth_5_ref=base.dynamic_ask_depth_5_ref,
                dynamic_mid_price_ref=base.dynamic_mid_price_ref,
                use_cumulative_signal=base.use_cumulative_signal,
                cumulative_signal_mode=base.cumulative_signal_mode,
                cumulative_signal_alpha=base.cumulative_signal_alpha,
                pressure_weight=base.pressure_weight,
                spread_weight=base.spread_weight,
                depth_weight=base.depth_weight,
                imbalance_weight=base.imbalance_weight,
            )
        )

    if len(combos) <= args.grid_max_combos:
        return combos

    rng = random.Random(args.grid_seed)
    return rng.sample(combos, k=int(args.grid_max_combos))


def run_grid_search(
    *,
    runner: BacktestRunner,
    cfg: BacktestConfig,
    selected_market_ids: set[str],
    mapping_dir: Path,
    manifest_path: Path | None,
    market_batch_size: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    combinations = make_grid_param_combinations(args)
    rows: list[dict[str, float | int | str]] = []

    print("\n" + "=" * 80)
    print("GRID SEARCH: FULL GATED THRESHOLD SWEEP")
    print("=" * 80)
    print(f"Combinations to evaluate: {len(combinations)}")

    full_gate_flags = hpo_gate_flags()

    for idx, params in enumerate(combinations, start=1):
        scenario_name = f"grid_{idx:03d}"
        print(
            f"Running {scenario_name} | "
            f"spread_q={params.spread_bps_narrow_quantile:.3f}, "
            f"conf={params.confidence_score_min:.3f}, "
            f"gap_q={params.relative_book_score_quantile:.3f}, "
            f"buy_cap={params.buy_price_max:.3f}, "
            f"max_t={params.max_time_to_resolution_secs:.0f}"
        )

        result = run_scenario(
            runner=runner,
            cfg=cfg,
            params=params,
            scenario_name=scenario_name,
            market_ids=selected_market_ids,
            mapping_dir=mapping_dir,
            manifest_path=manifest_path,
            market_batch_size=market_batch_size,
            gate_flags=full_gate_flags,
        )
        metrics = extract_metrics(scenario_name, result)
        row = {
            **metrics,
            "spread_bps_narrow_quantile": float(params.spread_bps_narrow_quantile),
            "confidence_score_min": float(params.confidence_score_min),
            "relative_book_score_quantile": float(params.relative_book_score_quantile),
            "buy_price_max": float(params.buy_price_max or 0.0),
            "max_time_to_resolution_secs": float(params.max_time_to_resolution_secs or 0.0),
        }
        rows.append(row)

    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        return grid_df

    grid_df["score"] = (
        grid_df["net_pnl"].astype(float)
        + 0.25 * grid_df["win_rate"].astype(float)
        + 0.0025 * grid_df["trades"].astype(float)
    )
    grid_df = grid_df.sort_values(
        ["score", "net_pnl", "win_rate", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return grid_df


def run_train_validate_hpo(
    *,
    runner: BacktestRunner,
    cfg: BacktestConfig,
    train_market_ids: set[str],
    validation_market_ids: set[str],
    mapping_dir: Path,
    manifest_path: Path | None,
    market_batch_size: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run grid HPO on train split and validate top-K candidates on holdout split."""
    base = default_strategy_params()
    train_grid = run_grid_search(
        runner=runner,
        cfg=cfg,
        selected_market_ids=train_market_ids,
        mapping_dir=mapping_dir,
        manifest_path=manifest_path,
        market_batch_size=market_batch_size,
        args=args,
    )
    if train_grid.empty:
        return pd.DataFrame(), pd.DataFrame()

    top_k = max(1, int(args.hpo_top_k))
    candidates = train_grid.head(top_k).copy()

    full_gate_flags = hpo_gate_flags()
    validation_rows: list[dict[str, float | int | str]] = []

    print("\n" + "=" * 80)
    print("TRAIN/VALIDATION HPO: VALIDATING TOP TRAIN CANDIDATES")
    print("=" * 80)
    print(f"Train candidates selected: {len(candidates)}")
    print(f"Validation markets: {len(validation_market_ids)}")

    for _, train_row in candidates.iterrows():
        scenario_name = str(train_row["scenario"])
        params = StrategyParams(
            relative_book_score_quantile=float(train_row["relative_book_score_quantile"]),
            spread_bps_narrow_quantile=float(train_row["spread_bps_narrow_quantile"]),
            confidence_score_min=float(train_row["confidence_score_min"]),
            min_liquidity=base.min_liquidity,
            buy_price_max=float(train_row["buy_price_max"]),
            min_time_to_resolution_secs=None,
            max_time_to_resolution_secs=float(train_row["max_time_to_resolution_secs"]),
            ask_depth_5_max_filter=base.ask_depth_5_max_filter,
            dynamic_position_sizing=True,
            dynamic_ask_depth_5_ref=1105.44,
            dynamic_mid_price_ref=0.75,
            use_cumulative_signal=base.use_cumulative_signal,
            cumulative_signal_mode=base.cumulative_signal_mode,
            cumulative_signal_alpha=base.cumulative_signal_alpha,
            pressure_weight=0.45,
            spread_weight=0.35,
            depth_weight=0.15,
            imbalance_weight=0.05,
        )

        print(
            f"Validating {scenario_name} | "
            f"spread_q={params.spread_bps_narrow_quantile:.3f}, "
            f"conf={params.confidence_score_min:.3f}, "
            f"gap_q={params.relative_book_score_quantile:.3f}, "
            f"buy_cap={params.buy_price_max:.3f}, "
            f"max_t={params.max_time_to_resolution_secs:.0f}"
        )

        result = run_scenario(
            runner=runner,
            cfg=cfg,
            params=params,
            scenario_name=f"{scenario_name}_validation",
            market_ids=validation_market_ids,
            mapping_dir=mapping_dir,
            manifest_path=manifest_path,
            market_batch_size=market_batch_size,
            gate_flags=full_gate_flags,
        )
        val_metrics = extract_metrics(f"{scenario_name}_validation", result)

        train_score = float(train_row["score"])
        val_score = (
            float(val_metrics["net_pnl"])
            + 0.25 * float(val_metrics["win_rate"])
            + 0.0025 * float(val_metrics["trades"])
        )

        validation_rows.append(
            {
                "scenario": scenario_name,
                "train_score": train_score,
                "train_trades": int(train_row["trades"]),
                "train_win_rate": float(train_row["win_rate"]),
                "train_net_pnl": float(train_row["net_pnl"]),
                "validation_score": val_score,
                "validation_trades": int(val_metrics["trades"]),
                "validation_win_rate": float(val_metrics["win_rate"]),
                "validation_net_pnl": float(val_metrics["net_pnl"]),
                "score_delta_validation_minus_train": val_score - train_score,
                "spread_bps_narrow_quantile": float(train_row["spread_bps_narrow_quantile"]),
                "confidence_score_min": float(train_row["confidence_score_min"]),
                "relative_book_score_quantile": float(train_row["relative_book_score_quantile"]),
                "buy_price_max": float(train_row["buy_price_max"]),
                "max_time_to_resolution_secs": float(train_row["max_time_to_resolution_secs"]),
            }
        )

    results = pd.DataFrame(validation_rows)
    if results.empty:
        return train_grid, results

    ranked = results.sort_values(
        ["validation_score", "validation_net_pnl", "validation_win_rate", "validation_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return train_grid, ranked


def shortlisted_candidates() -> list[tuple[str, StrategyParams]]:
    """Return 3 shortlisted candidates based on latest gate diagnostics.

    Signals from the most recent gate run:
    - Relax score-gap gate (handled via ``hpo_gate_flags``)
    - Keep ask-depth cap and time gate
    - Spread relaxation can improve net PnL at some hit-rate cost
    """
    base = default_strategy_params()
    return [
        (
            "candidate_1_balanced_no_score_gap",
            StrategyParams(
                relative_book_score_quantile=base.relative_book_score_quantile,
                spread_bps_narrow_quantile=base.spread_bps_narrow_quantile,
                confidence_score_min=base.confidence_score_min,
                min_liquidity=base.min_liquidity,
                buy_price_max=base.buy_price_max,
                min_time_to_resolution_secs=None,
                max_time_to_resolution_secs=base.max_time_to_resolution_secs,
                ask_depth_5_max_filter=base.ask_depth_5_max_filter,
                dynamic_position_sizing=True,
                dynamic_ask_depth_5_ref=1105.44,
                dynamic_mid_price_ref=0.75,
                use_cumulative_signal=base.use_cumulative_signal,
                cumulative_signal_mode=base.cumulative_signal_mode,
                cumulative_signal_alpha=base.cumulative_signal_alpha,
                pressure_weight=0.45,
                spread_weight=0.35,
                depth_weight=0.15,
                imbalance_weight=0.05,
            ),
        ),
        (
            "candidate_2_spread_relaxed_pnl",
            StrategyParams(
                relative_book_score_quantile=base.relative_book_score_quantile,
                spread_bps_narrow_quantile=0.20,
                confidence_score_min=base.confidence_score_min,
                min_liquidity=base.min_liquidity,
                buy_price_max=base.buy_price_max,
                min_time_to_resolution_secs=None,
                max_time_to_resolution_secs=base.max_time_to_resolution_secs,
                ask_depth_5_max_filter=base.ask_depth_5_max_filter,
                dynamic_position_sizing=True,
                dynamic_ask_depth_5_ref=1105.44,
                dynamic_mid_price_ref=0.75,
                use_cumulative_signal=base.use_cumulative_signal,
                cumulative_signal_mode=base.cumulative_signal_mode,
                cumulative_signal_alpha=base.cumulative_signal_alpha,
                pressure_weight=0.45,
                spread_weight=0.35,
                depth_weight=0.15,
                imbalance_weight=0.05,
            ),
        ),
        (
            "candidate_3_high_winrate_conservative_price",
            StrategyParams(
                relative_book_score_quantile=base.relative_book_score_quantile,
                spread_bps_narrow_quantile=base.spread_bps_narrow_quantile,
                confidence_score_min=0.75,
                min_liquidity=base.min_liquidity,
                buy_price_max=0.85,
                min_time_to_resolution_secs=None,
                max_time_to_resolution_secs=base.max_time_to_resolution_secs,
                ask_depth_5_max_filter=base.ask_depth_5_max_filter,
                dynamic_position_sizing=True,
                dynamic_ask_depth_5_ref=1105.44,
                dynamic_mid_price_ref=0.75,
                use_cumulative_signal=base.use_cumulative_signal,
                cumulative_signal_mode=base.cumulative_signal_mode,
                cumulative_signal_alpha=base.cumulative_signal_alpha,
                pressure_weight=0.45,
                spread_weight=0.35,
                depth_weight=0.15,
                imbalance_weight=0.05,
            ),
        ),
    ]


def run_candidate_hpo(
    *,
    runner: BacktestRunner,
    cfg: BacktestConfig,
    selected_market_ids: set[str],
    mapping_dir: Path,
    manifest_path: Path | None,
    market_batch_size: int,
) -> pd.DataFrame:
    """Evaluate the 3 shortlisted candidates on the same market slice."""
    candidates = shortlisted_candidates()
    full_gate_flags = hpo_gate_flags()
    rows: list[dict[str, float | int | str]] = []

    print("\n" + "=" * 80)
    print("FOCUSED CANDIDATE HPO (3 CONFIGS)")
    print("=" * 80)
    print(f"Candidates: {len(candidates)}")
    print(f"Markets: {len(selected_market_ids)}")

    for scenario_name, params in candidates:
        print(
            f"Running {scenario_name} | "
            f"spread_q={params.spread_bps_narrow_quantile:.3f}, "
            f"conf={params.confidence_score_min:.3f}, "
            f"gap_q={params.relative_book_score_quantile:.3f}, "
            f"buy_cap={params.buy_price_max:.3f}, "
            f"max_t={params.max_time_to_resolution_secs:.0f}"
        )
        result = run_scenario(
            runner=runner,
            cfg=cfg,
            params=params,
            scenario_name=scenario_name,
            market_ids=selected_market_ids,
            mapping_dir=mapping_dir,
            manifest_path=manifest_path,
            market_batch_size=market_batch_size,
            gate_flags=full_gate_flags,
        )
        metrics = extract_metrics(scenario_name, result)
        rows.append(
            {
                **metrics,
                "spread_bps_narrow_quantile": float(params.spread_bps_narrow_quantile),
                "confidence_score_min": float(params.confidence_score_min),
                "relative_book_score_quantile": float(params.relative_book_score_quantile),
                "buy_price_max": float(params.buy_price_max or 0.0),
                "max_time_to_resolution_secs": float(params.max_time_to_resolution_secs or 0.0),
            }
        )

    hpo_df = pd.DataFrame(rows)
    if hpo_df.empty:
        return hpo_df

    hpo_df["score"] = (
        hpo_df["net_pnl"].astype(float)
        + 0.25 * hpo_df["win_rate"].astype(float)
        + 0.0025 * hpo_df["trades"].astype(float)
    )
    return hpo_df.sort_values(
        ["score", "net_pnl", "win_rate", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def extract_metrics(
    result_name: str,
    result: MetricsResult,
) -> dict[str, float | int | str]:
    summary = result.backtest_summary
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
    }


def gate_flags_for_sequence(n_enabled: int) -> dict[str, bool]:
    enabled = set(GATE_SEQUENCE[:n_enabled])
    return {
        "enable_spread_gate": "spread_gate" in enabled,
        "enable_score_gate": "score_gate" in enabled,
        "enable_score_gap_gate": "score_gap_gate" in enabled,
        "enable_price_cap_gate": "price_cap_gate" in enabled,
        "enable_liquidity_gate": "liquidity_gate" in enabled,
        "enable_ask_depth_5_cap_gate": "ask_depth_5_cap_gate" in enabled,
        "enable_time_gate": "time_gate" in enabled,
    }


def build_recommendations(metrics_df: pd.DataFrame) -> dict[str, object]:
    recommendations: list[dict[str, object]] = []

    by_name = metrics_df.set_index("scenario", drop=False)
    if "full_gated" not in by_name.index:
        return {
            "recommendations": recommendations,
            "note": "full_gated scenario missing; no recommendations produced",
        }

    def _row(name: str) -> pd.Series:
        return by_name.loc[[name]].iloc[0]

    full = _row("full_gated")

    def _as_float(series: pd.Series, key: str) -> float:
        return float(series[key])

    def _as_int(series: pd.Series, key: str) -> int:
        return int(series[key])

    # Gate-by-gate leave-one-out impact.
    for gate in GATE_SEQUENCE:
        scenario = f"full_minus_{gate}"
        if scenario not in by_name.index:
            continue

        alt = _row(scenario)
        delta_net = _as_float(alt, "net_pnl") - _as_float(full, "net_pnl")
        delta_win = _as_float(alt, "win_rate") - _as_float(full, "win_rate")
        delta_trades = _as_int(alt, "trades") - _as_int(full, "trades")

        if delta_net > 0 and delta_win >= WIN_RATE_TOLERANCE_FOR_GATE_RELAX:
            recommendations.append(
                {
                    "type": "relax_or_recalibrate_gate",
                    "gate": gate,
                    "reason": (
                        "Removing this gate improved net_pnl without "
                        "meaningful hit-rate damage"
                    ),
                    "expected_delta_net_pnl": round(delta_net, 6),
                    "expected_delta_win_rate": round(delta_win, 6),
                    "expected_delta_trades": delta_trades,
                }
            )
        elif delta_net < 0 and delta_win <= 0:
            recommendations.append(
                {
                    "type": "keep_or_tighten_gate",
                    "gate": gate,
                    "reason": "Removing this gate hurt both net_pnl and win_rate",
                    "expected_delta_net_pnl_if_removed": round(delta_net, 6),
                    "expected_delta_win_rate_if_removed": round(delta_win, 6),
                    "expected_delta_trades_if_removed": delta_trades,
                }
            )

    # Optional baseline-to-full context.
    if "base_no_gates" in by_name.index:
        base = _row("base_no_gates")
        recommendations.append(
            {
                "type": "context",
                "baseline_vs_full": {
                    "delta_net_pnl": round(
                        _as_float(full, "net_pnl") - _as_float(base, "net_pnl"),
                        6,
                    ),
                    "delta_win_rate": round(
                        _as_float(full, "win_rate") - _as_float(base, "win_rate"),
                        6,
                    ),
                    "delta_trades": _as_int(full, "trades") - _as_int(base, "trades"),
                },
            }
        )

    return {
        "recommendations": recommendations,
        "full_gated": {
            "trades": _as_int(full, "trades"),
            "win_rate": _as_float(full, "win_rate"),
            "net_pnl": _as_float(full, "net_pnl"),
        },
    }


def run_scenario(
    *,
    runner: BacktestRunner,
    cfg: BacktestConfig,
    params: StrategyParams,
    scenario_name: str,
    market_ids: set[str],
    mapping_dir: Path,
    manifest_path: Path | None,
    market_batch_size: int,
    gate_flags: dict[str, bool],
) -> MetricsResult:
    strategy = build_shared_relative_book_strength_strategy(params=params, **gate_flags)
    return runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name=scenario_name,
        market_batch_size=market_batch_size,
        prepared_feature_market_ids=market_ids,
        config=cfg,
    )


def main() -> None:
    args = parse_args()
    setup_application_logging()

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.exists():
        msg = f"Run directory not found: {run_dir}"
        raise FileNotFoundError(msg)

    manifest_path: Path | None = args.manifest.resolve() if args.manifest else None
    if manifest_path is None:
        candidate = run_dir / "manifest.json"
        manifest_path = candidate if candidate.exists() else None

    mapping_dir: Path = args.mapping_dir.resolve()
    if not mapping_dir.exists():
        msg = f"Mapping directory not found: {mapping_dir}"
        raise FileNotFoundError(msg)

    runner = BacktestRunner(storage_path=run_dir)
    runner.is_pmxt_mode = True

    all_market_ids = runner.load_prepared_feature_market_ids(
        limit_files=None,
        features_manifest_path=manifest_path,
        recursive_scan=True,
    )
    if not all_market_ids:
        raise RuntimeError("No prepared feature market IDs found")

    if args.run_train_validate_hpo and args.adaptive_search_space:
        adaptive = derive_adaptive_search_space(args)
        if adaptive is not None:
            args.grid_spread_quantiles = adaptive["grid_spread_quantiles"]
            args.grid_confidence_min = adaptive["grid_confidence_min"]
            args.grid_score_gap_quantiles = adaptive["grid_score_gap_quantiles"]
            args.grid_buy_price_max = adaptive["grid_buy_price_max"]
            args.grid_max_time_secs = adaptive["grid_max_time_secs"]

    selected_market_count = (
        int(args.hpo_market_count)
        if args.run_candidate_hpo
        else int(args.market_count)
    )

    pool_count = (
        int(args.sample_pool_market_count)
        if args.sample_pool_market_count is not None
        else len(all_market_ids)
    )
    if pool_count <= 0:
        raise RuntimeError("--sample-pool-market-count must be > 0")
    if pool_count > len(all_market_ids):
        raise RuntimeError(
            "Requested sample pool exceeds available markets: "
            f"pool={pool_count}, available={len(all_market_ids)}"
        )

    market_pool = list(all_market_ids[:pool_count])
    if selected_market_count > len(market_pool):
        raise RuntimeError(
            "Requested market count exceeds sample pool: "
            f"market_count={selected_market_count}, pool={len(market_pool)}"
        )

    if args.random_sample_markets:
        rng = random.Random(int(args.sample_seed))
        selected_market_ids_list = rng.sample(market_pool, k=selected_market_count)
    else:
        selected_market_ids_list = market_pool[:selected_market_count]

    selected_market_ids = set(selected_market_ids_list)
    if not selected_market_ids:
        raise RuntimeError("No market IDs selected; increase --market-count")

    print("=" * 80)
    print("RELATIVE BOOK STRATEGY GATE DIAGNOSTICS")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Mapping dir: {mapping_dir}")
    print(f"Sampling pool size: {len(market_pool)}")
    print(f"Random sampling enabled: {bool(args.random_sample_markets)}")
    print(f"Sampling seed: {int(args.sample_seed)}")
    print(f"Selected markets: {len(selected_market_ids)}")
    print(f"Market batch size: {args.market_batch_size}")

    if args.run_train_validate_hpo:
        split_pool_count = int(args.hpo_split_pool_market_count)
        if split_pool_count <= 0:
            raise RuntimeError("--hpo-split-pool-market-count must be > 0")
        if split_pool_count > len(all_market_ids):
            msg = (
                "Requested split pool exceeds available markets: "
                f"pool={split_pool_count}, available={len(all_market_ids)}"
            )
            raise RuntimeError(msg)

        # Policy: use only the earliest/prefix pool. Shuffle only when explicitly requested.
        market_ids_for_split = list(all_market_ids[:split_pool_count])
        if args.shuffle_markets_before_split:
            rng = random.Random(int(args.split_seed))
            rng.shuffle(market_ids_for_split)

        train_count = int(args.hpo_train_market_count)
        val_count = int(args.hpo_validation_market_count)
        total_needed = train_count + val_count
        if total_needed > len(market_ids_for_split):
            msg = (
                "Requested train+validation markets "
                f"({total_needed}) exceed available ({len(market_ids_for_split)})"
            )
            raise RuntimeError(msg)

        train_ids_list = market_ids_for_split[:train_count]
        validation_ids_list = market_ids_for_split[train_count:train_count + val_count]

        train_market_ids = set(train_ids_list)
        validation_market_ids = set(validation_ids_list)

        if train_market_ids & validation_market_ids:
            raise RuntimeError("Train and validation market sets overlap; expected disjoint sets")

        print("\n" + "=" * 80)
        print("TRAIN/VALIDATION HPO MODE")
        print("=" * 80)
        print(f"Split pool markets (prefix): {split_pool_count}")
        print(f"Train markets: {len(train_market_ids)}")
        print(f"Validation markets: {len(validation_market_ids)}")
        print(f"Top-K validation candidates: {max(1, int(args.hpo_top_k))}")
        print(f"Shuffle before split: {bool(args.shuffle_markets_before_split)}")
        print(f"Split seed: {int(args.split_seed)}")

        run_id = datetime.now(UTC).strftime("tvhpo_%Y%m%dT%H%M%SZ")
        run_artifact_dir = args.run_artifacts_root.resolve() / run_id
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        cfg = default_backtest_config()
        train_grid_df, tv_df = run_train_validate_hpo(
            runner=runner,
            cfg=cfg,
            train_market_ids=train_market_ids,
            validation_market_ids=validation_market_ids,
            mapping_dir=mapping_dir,
            manifest_path=manifest_path,
            market_batch_size=int(args.market_batch_size),
            args=args,
        )
        if tv_df.empty:
            print("Train/validation HPO produced no rows.")
            return

        # Preserve complete per-run artifact bundle.
        split_payload = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "mapping_dir": str(mapping_dir),
            "split_policy": "prefix_pool_then_shuffle",
            "split_pool_market_count": split_pool_count,
            "shuffle_markets_before_split": True,
            "split_seed": int(args.split_seed),
            "train_market_count": len(train_ids_list),
            "validation_market_count": len(validation_ids_list),
            "train_market_ids": train_ids_list,
            "validation_market_ids": validation_ids_list,
        }
        (run_artifact_dir / "split_metadata.json").write_text(
            json.dumps(split_payload, indent=2),
            encoding="utf-8",
        )

        train_grid_df.to_csv(run_artifact_dir / "train_grid_full.csv", index=False)
        tv_df.to_csv(run_artifact_dir / "train_validate_ranked.csv", index=False)

        cli_payload = {
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "resolved_grid_lists": {
                "grid_spread_quantiles": args.grid_spread_quantiles,
                "grid_confidence_min": args.grid_confidence_min,
                "grid_score_gap_quantiles": args.grid_score_gap_quantiles,
                "grid_buy_price_max": args.grid_buy_price_max,
                "grid_max_time_secs": args.grid_max_time_secs,
                "grid_max_combos": int(args.grid_max_combos),
                "grid_seed": int(args.grid_seed),
            },
        }
        (run_artifact_dir / "run_config.json").write_text(
            json.dumps(cli_payload, indent=2),
            encoding="utf-8",
        )

        args.hpo_train_validate_output_csv.parent.mkdir(parents=True, exist_ok=True)
        tv_df.to_csv(args.hpo_train_validate_output_csv, index=False)

        top_row = tv_df.iloc[0].to_dict()
        args.hpo_train_validate_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.hpo_train_validate_output_json.write_text(
            json.dumps(top_row, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 80)
        print("TRAIN/VALIDATION HPO RESULTS")
        print("=" * 80)
        print(tv_df.to_string(index=False))
        print("\nTop validated configuration:")
        print(json.dumps(top_row, indent=2))
        print("\nArtifacts written:")
        print(f"- {args.hpo_train_validate_output_csv}")
        print(f"- {args.hpo_train_validate_output_json}")
        print(f"- per-run bundle: {run_artifact_dir}")
        return

    if args.run_candidate_hpo:
        cfg = default_backtest_config()
        hpo_df = run_candidate_hpo(
            runner=runner,
            cfg=cfg,
            selected_market_ids=selected_market_ids,
            mapping_dir=mapping_dir,
            manifest_path=manifest_path,
            market_batch_size=int(args.market_batch_size),
        )
        if hpo_df.empty:
            print("Focused candidate HPO produced no rows.")
            return

        args.hpo_output_csv.parent.mkdir(parents=True, exist_ok=True)
        hpo_df.to_csv(args.hpo_output_csv, index=False)

        top_row = hpo_df.iloc[0].to_dict()
        args.hpo_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.hpo_output_json.write_text(
            json.dumps(top_row, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 80)
        print("CANDIDATE HPO RESULTS")
        print("=" * 80)
        print(hpo_df.to_string(index=False))
        print("\nWinner:")
        print(json.dumps(top_row, indent=2))
        print("\nHPO artifacts written:")
        print(f"- {args.hpo_output_csv}")
        print(f"- {args.hpo_output_json}")
        return

    cfg = default_backtest_config()
    params = default_strategy_params()

    scenario_defs: list[tuple[str, dict[str, bool]]] = []

    scenario_defs.append(("base_no_gates", gate_flags_for_sequence(0)))

    for idx, gate_name in enumerate(GATE_SEQUENCE, start=1):
        scenario_defs.append(
            (f"cumulative_{idx:02d}_{gate_name}", gate_flags_for_sequence(idx))
        )

    scenario_defs.append(("full_gated", gate_flags_for_sequence(len(GATE_SEQUENCE))))

    full_flags = gate_flags_for_sequence(len(GATE_SEQUENCE))
    for gate_name in GATE_SEQUENCE:
        minus_flags = dict(full_flags)
        key = {
            "spread_gate": "enable_spread_gate",
            "score_gate": "enable_score_gate",
            "score_gap_gate": "enable_score_gap_gate",
            "price_cap_gate": "enable_price_cap_gate",
            "liquidity_gate": "enable_liquidity_gate",
            "ask_depth_5_cap_gate": "enable_ask_depth_5_cap_gate",
            "time_gate": "enable_time_gate",
        }[gate_name]
        minus_flags[key] = False
        scenario_defs.append((f"full_minus_{gate_name}", minus_flags))

    metric_rows: list[dict[str, float | int | str]] = []

    for scenario_name, gate_flags in scenario_defs:
        print(f"\nRunning scenario: {scenario_name}")
        result = run_scenario(
            runner=runner,
            cfg=cfg,
            params=params,
            scenario_name=scenario_name,
            market_ids=selected_market_ids,
            mapping_dir=mapping_dir,
            manifest_path=manifest_path,
            market_batch_size=int(args.market_batch_size),
            gate_flags=gate_flags,
        )
        metrics = extract_metrics(scenario_name, result)
        metric_rows.append(metrics)
        print(
            "  "
            f"trades={metrics['trades']}, "
            f"win_rate={metrics['win_rate']:.2%}, "
            f"net_pnl={metrics['net_pnl']:.6f}, "
            f"avg_net_pnl={metrics['avg_net_pnl']:.6f}"
        )

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values("net_pnl", ascending=False)
        .reset_index(drop=True)
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_csv, index=False)

    recommendations = build_recommendations(metrics_df)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(recommendations, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("TOP SCENARIOS BY NET PNL")
    print("=" * 80)
    print(metrics_df.head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("QUANTITATIVE RECOMMENDATIONS")
    print("=" * 80)
    recommendation_rows = recommendations.get("recommendations", [])
    if not isinstance(recommendation_rows, list):
        recommendation_rows = []
    if recommendation_rows:
        for rec in recommendation_rows:
            print(json.dumps(rec, indent=2))
    else:
        print("No strong recommendation signals produced.")

    print("\nArtifacts written:")
    print(f"- {args.output_csv}")
    print(f"- {args.output_json}")

    if args.run_grid_search:
        grid_df = run_grid_search(
            runner=runner,
            cfg=cfg,
            selected_market_ids=selected_market_ids,
            mapping_dir=mapping_dir,
            manifest_path=manifest_path,
            market_batch_size=int(args.market_batch_size),
            args=args,
        )
        if grid_df.empty:
            print("Grid search produced no rows.")
            return

        args.grid_output_csv.parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_csv(args.grid_output_csv, index=False)

        top10 = grid_df.head(10).copy()
        args.grid_output_json.parent.mkdir(parents=True, exist_ok=True)
        args.grid_output_json.write_text(
            top10.to_json(orient="records", indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 80)
        print("TOP 10 GRID CONFIGURATIONS")
        print("=" * 80)
        display_cols = [
            "scenario",
            "score",
            "trades",
            "win_rate",
            "net_pnl",
            "spread_bps_narrow_quantile",
            "confidence_score_min",
            "relative_book_score_quantile",
            "buy_price_max",
            "max_time_to_resolution_secs",
        ]
        print(top10[display_cols].to_string(index=False))
        print("\nGrid artifacts written:")
        print(f"- {args.grid_output_csv}")
        print(f"- {args.grid_output_json}")


if __name__ == "__main__":
    main()
