"""Diagnose train-vs-OOS degradation for relative-book strategy.

This script runs the same best strategy on:
1) HPO train prefix (first N markets), and
2) strict OOS tail (remaining markets),
then writes quantitative decomposition artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtester import BacktestRunner
from scripts.run_relative_book_oos_validation import (
    best_hpo_gate_flags,
    best_hpo_strategy_params,
    default_backtest_config,
)
from utils import setup_application_logging


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_run_dir = project_root / "data" / "cached" / "pmxt_backtest" / "runs" / "btc-updown-5m"

    parser = argparse.ArgumentParser(
        description="Diagnose why relative-book strategy degrades from train to OOS"
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
        default=4000,
        help="Number of leading markets treated as train/HPO split",
    )
    parser.add_argument(
        "--expected-oos-market-count",
        type=int,
        default=1076,
        help="Fail if computed OOS tail size differs from this value",
    )
    parser.add_argument(
        "--train-strategy-name",
        type=str,
        default="relative_book_train_grid_015",
        help="Strategy name for train split run",
    )
    parser.add_argument(
        "--oos-strategy-name",
        type=str,
        default="relative_book_oos_grid_015",
        help="Strategy name for OOS split run",
    )
    parser.add_argument(
        "--output-split-metrics-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "relative_book_degradation_split_metrics.csv",
        help="Where to write split-level diagnostics CSV",
    )
    parser.add_argument(
        "--output-market-metrics-csv",
        type=Path,
        default=project_root / "reports" / "artifacts" / "relative_book_degradation_market_metrics.csv",
        help="Where to write market-level diagnostics for both splits",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=project_root / "reports" / "artifacts" / "relative_book_degradation_diagnosis.json",
        help="Where to write decomposition payload JSON",
    )
    return parser.parse_args()


def _safe_mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").dropna().mean() or 0.0)


def _safe_median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").dropna().median() or 0.0)


def _safe_quantile(frame: pd.DataFrame, column: str, q: float) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def _split_metrics(
    *,
    split_name: str,
    scenario_name: str,
    market_count: int,
    trade_ledger: pd.DataFrame,
    market_diag: pd.DataFrame,
) -> dict[str, float | int | str]:
    trades = len(trade_ledger)
    wins = int((trade_ledger["net_pnl"] > 0).sum()) if trades else 0
    losses = int((trade_ledger["net_pnl"] <= 0).sum()) if trades else 0

    gross_pnl = float(trade_ledger["gross_pnl"].sum()) if trades else 0.0
    net_pnl = float(trade_ledger["net_pnl"].sum()) if trades else 0.0
    fees = float(trade_ledger["fee_usdc"].sum()) if trades else 0.0

    avg_gross_per_trade = gross_pnl / trades if trades else 0.0
    avg_fees_per_trade = fees / trades if trades else 0.0
    avg_net_per_trade = net_pnl / trades if trades else 0.0

    wins_df = trade_ledger.loc[trade_ledger["net_pnl"] > 0]
    losses_df = trade_ledger.loc[trade_ledger["net_pnl"] <= 0]
    avg_win_net = _safe_mean(wins_df, "net_pnl")
    avg_loss_net = _safe_mean(losses_df, "net_pnl")
    avg_win_gross = _safe_mean(wins_df, "gross_pnl")
    avg_loss_gross = _safe_mean(losses_df, "gross_pnl")

    win_rate = wins / trades if trades else 0.0
    trades_per_market = trades / market_count if market_count else 0.0
    net_pnl_per_market = net_pnl / market_count if market_count else 0.0
    gross_pnl_per_market = gross_pnl / market_count if market_count else 0.0
    fees_per_market = fees / market_count if market_count else 0.0

    gross_notional = float(trade_ledger["gross_notional"].sum()) if trades else 0.0
    avg_gross_notional = gross_notional / trades if trades else 0.0
    fee_drag_pct = fees / gross_notional if gross_notional else 0.0

    payoff_ratio = (avg_win_net / abs(avg_loss_net)) if avg_loss_net < 0 else 0.0
    break_even_wr = (
        abs(avg_loss_net) / (avg_win_net + abs(avg_loss_net))
        if (avg_win_net + abs(avg_loss_net)) > 0
        else 0.0
    )

    profitable_markets = 0
    top10_market_net_share = 0.0
    if not market_diag.empty and "net_pnl" in market_diag.columns:
        market_sorted = market_diag.sort_values("net_pnl", ascending=False).copy()
        profitable_markets = int((market_sorted["net_pnl"] > 0).sum())
        top10 = market_sorted.head(10)
        top10_market_net_share = (
            float(top10["net_pnl"].sum()) / net_pnl if net_pnl else 0.0
        )

    return {
        "split": split_name,
        "scenario": scenario_name,
        "markets": market_count,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "fees": fees,
        "trades_per_market": trades_per_market,
        "gross_pnl_per_market": gross_pnl_per_market,
        "fees_per_market": fees_per_market,
        "net_pnl_per_market": net_pnl_per_market,
        "avg_gross_per_trade": avg_gross_per_trade,
        "avg_fees_per_trade": avg_fees_per_trade,
        "avg_net_per_trade": avg_net_per_trade,
        "avg_win_net": avg_win_net,
        "avg_loss_net": avg_loss_net,
        "avg_win_gross": avg_win_gross,
        "avg_loss_gross": avg_loss_gross,
        "payoff_ratio": payoff_ratio,
        "break_even_win_rate": break_even_wr,
        "avg_entry_price": _safe_mean(trade_ledger, "entry_price"),
        "median_entry_price": _safe_median(trade_ledger, "entry_price"),
        "avg_entry_price_wins": _safe_mean(wins_df, "entry_price"),
        "avg_entry_price_losses": _safe_mean(losses_df, "entry_price"),
        "avg_slippage_bps": _safe_mean(trade_ledger, "slippage_bps"),
        "avg_entry_spread": _safe_mean(trade_ledger, "entry_spread"),
        "avg_entry_liquidity": _safe_mean(trade_ledger, "entry_liquidity"),
        "avg_entry_volatility": _safe_mean(trade_ledger, "entry_volatility"),
        "avg_hold_hours": _safe_mean(trade_ledger, "hold_hours"),
        "avg_gross_notional": avg_gross_notional,
        "fee_drag_pct": fee_drag_pct,
        "net_pnl_p10": _safe_quantile(trade_ledger, "net_pnl", 0.10),
        "net_pnl_p50": _safe_quantile(trade_ledger, "net_pnl", 0.50),
        "net_pnl_p90": _safe_quantile(trade_ledger, "net_pnl", 0.90),
        "profitable_markets": profitable_markets,
        "profitable_market_rate": (profitable_markets / market_count) if market_count else 0.0,
        "top10_market_net_share": top10_market_net_share,
    }


def _per_market_bridge(train: pd.Series, oos: pd.Series) -> dict[str, float]:
    """Decompose delta net PnL per market into participation/gross/fees effects.

    net_per_market = trades_per_market * (avg_gross_per_trade - avg_fees_per_trade)
    """
    tpm_1 = float(train["trades_per_market"])
    tpm_2 = float(oos["trades_per_market"])
    gpt_1 = float(train["avg_gross_per_trade"])
    gpt_2 = float(oos["avg_gross_per_trade"])
    fpt_1 = float(train["avg_fees_per_trade"])
    fpt_2 = float(oos["avg_fees_per_trade"])

    participation_effect = (tpm_2 - tpm_1) * (gpt_1 - fpt_1)
    gross_edge_effect = tpm_2 * (gpt_2 - gpt_1)
    fee_effect = -tpm_2 * (fpt_2 - fpt_1)

    observed_delta = float(oos["net_pnl_per_market"]) - float(train["net_pnl_per_market"])
    reconstructed = participation_effect + gross_edge_effect + fee_effect

    return {
        "observed_delta_net_per_market": observed_delta,
        "reconstructed_delta_net_per_market": reconstructed,
        "participation_effect": participation_effect,
        "gross_edge_effect": gross_edge_effect,
        "fee_effect": fee_effect,
    }


def _expectancy_bridge(train: pd.Series, oos: pd.Series) -> dict[str, float]:
    """Decompose delta net expectancy per trade.

    E = wr * W + (1 - wr) * L where L <= 0.
    """
    wr_1 = float(train["win_rate"])
    wr_2 = float(oos["win_rate"])
    w_1 = float(train["avg_win_net"])
    w_2 = float(oos["avg_win_net"])
    l_1 = float(train["avg_loss_net"])
    l_2 = float(oos["avg_loss_net"])

    win_rate_effect = (wr_2 - wr_1) * (w_1 - l_1)
    win_magnitude_effect = wr_2 * (w_2 - w_1)
    loss_magnitude_effect = (1.0 - wr_2) * (l_2 - l_1)

    observed_delta = float(oos["avg_net_per_trade"]) - float(train["avg_net_per_trade"])
    reconstructed = win_rate_effect + win_magnitude_effect + loss_magnitude_effect

    return {
        "observed_delta_avg_net_per_trade": observed_delta,
        "reconstructed_delta_avg_net_per_trade": reconstructed,
        "win_rate_effect": win_rate_effect,
        "win_magnitude_effect": win_magnitude_effect,
        "loss_magnitude_effect": loss_magnitude_effect,
    }


def main() -> None:
    args = parse_args()
    setup_application_logging()

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
    if train_count <= 0 or train_count >= len(all_market_ids):
        raise RuntimeError(
            "Invalid train split count: "
            f"train_count={train_count}, total={len(all_market_ids)}"
        )

    train_market_ids = all_market_ids[:train_count]
    oos_market_ids = all_market_ids[train_count:]

    expected_oos = int(args.expected_oos_market_count)
    if expected_oos > 0 and len(oos_market_ids) != expected_oos:
        raise RuntimeError(
            "Computed OOS market set size mismatch: "
            f"expected={expected_oos}, actual={len(oos_market_ids)}"
        )

    params = best_hpo_strategy_params()
    gates = best_hpo_gate_flags()
    strategy = __import__(
        "alphas.cumulative_relative_book_strength",
        fromlist=["build_relative_book_strength_strategy"],
    ).build_relative_book_strength_strategy(params=params, **gates)
    cfg = default_backtest_config()

    print("=" * 80)
    print("RELATIVE BOOK TRAIN VS OOS DEGRADATION DIAGNOSTICS")
    print("=" * 80)
    print(f"Total markets: {len(all_market_ids)}")
    print(f"Train markets: {len(train_market_ids)}")
    print(f"OOS markets: {len(oos_market_ids)}")

    train_result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name=str(args.train_strategy_name),
        market_batch_size=int(args.market_batch_size),
        prepared_feature_market_ids=set(train_market_ids),
        config=cfg,
    )

    oos_result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=strategy,
        strategy_name=str(args.oos_strategy_name),
        market_batch_size=int(args.market_batch_size),
        prepared_feature_market_ids=set(oos_market_ids),
        config=cfg,
    )

    train_metrics = _split_metrics(
        split_name="train_hpo_first_n",
        scenario_name=str(args.train_strategy_name),
        market_count=len(train_market_ids),
        trade_ledger=train_result.trade_ledger,
        market_diag=train_result.diagnostics_by_market,
    )
    oos_metrics = _split_metrics(
        split_name="oos_remaining",
        scenario_name=str(args.oos_strategy_name),
        market_count=len(oos_market_ids),
        trade_ledger=oos_result.trade_ledger,
        market_diag=oos_result.diagnostics_by_market,
    )

    split_df = pd.DataFrame([train_metrics, oos_metrics])

    train_row = split_df.loc[split_df["split"] == "train_hpo_first_n"].iloc[0]
    oos_row = split_df.loc[split_df["split"] == "oos_remaining"].iloc[0]

    bridge_per_market = _per_market_bridge(train_row, oos_row)
    bridge_expectancy = _expectancy_bridge(train_row, oos_row)

    market_train = train_result.diagnostics_by_market.copy()
    market_oos = oos_result.diagnostics_by_market.copy()
    market_train["split"] = "train_hpo_first_n"
    market_oos["split"] = "oos_remaining"
    market_metrics = pd.concat([market_train, market_oos], ignore_index=True)

    args.output_split_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(args.output_split_metrics_csv, index=False)

    args.output_market_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    market_metrics.to_csv(args.output_market_metrics_csv, index=False)

    payload = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "paths": {
            "run_dir": str(run_dir),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "mapping_dir": str(mapping_dir),
        },
        "split_policy": {
            "train_market_count": len(train_market_ids),
            "oos_market_count": len(oos_market_ids),
            "total_market_count": len(all_market_ids),
            "expected_oos_market_count": expected_oos,
        },
        "strategy": {
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
            },
            "gates": gates,
        },
        "split_metrics": {
            "train_hpo_first_n": train_metrics,
            "oos_remaining": oos_metrics,
        },
        "decomposition": {
            "net_per_market_bridge": bridge_per_market,
            "net_expectancy_per_trade_bridge": bridge_expectancy,
        },
        "artifacts": {
            "split_metrics_csv": str(args.output_split_metrics_csv.resolve()),
            "market_metrics_csv": str(args.output_market_metrics_csv.resolve()),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("SPLIT METRICS")
    print("=" * 80)
    print(split_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("NET PER MARKET BRIDGE")
    print("=" * 80)
    print(json.dumps(bridge_per_market, indent=2))

    print("\n" + "=" * 80)
    print("NET EXPECTANCY PER TRADE BRIDGE")
    print("=" * 80)
    print(json.dumps(bridge_expectancy, indent=2))

    print("\nArtifacts written:")
    print(f"- {args.output_split_metrics_csv}")
    print(f"- {args.output_market_metrics_csv}")
    print(f"- {args.output_json}")


if __name__ == "__main__":
    main()
