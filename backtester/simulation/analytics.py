from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_MIN_BUCKET_CARDINALITY = 2
_MIN_CAPITAL_PRESERVED = 50.0


@dataclass(frozen=True)
class MetricsTargets:
    """Target thresholds used in consolidated notebook validation checks."""

    min_trades: int = 150
    min_win_rate: float = 0.60
    min_sharpe: float = 0.5
    max_drawdown_pct: float = 25.0


def summarize_backtest(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy-level backtest metrics from closed trades."""
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "trades",
                "wins",
                "win_rate",
                "gross_pnl",
                "net_pnl",
                "fees",
                "avg_gross_pnl",
                "avg_net_pnl",
                "avg_hold_hours",
                "avg_gross_return_pct",
                "avg_net_return_pct",
                "fee_drag_pct",
            ]
        )

    rows: list[dict[str, object]] = []
    for strategy, group in trades.groupby("strategy", sort=False):
        wins = int((group["net_pnl"] > 0).sum())
        gross_pnl = float(group["gross_pnl"].sum())
        net_pnl = float(group["net_pnl"].sum())
        fees = float(group["fee_usdc"].sum())
        gross_notional = float(group["gross_notional"].sum())
        rows.append(
            {
                "strategy": strategy,
                "trades": len(group),
                "wins": wins,
                "win_rate": wins / len(group) if len(group) else 0.0,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "fees": fees,
                "avg_gross_pnl": float(group["gross_pnl"].mean()),
                "avg_net_pnl": float(group["net_pnl"].mean()),
                "avg_hold_hours": float(group["hold_hours"].mean()),
                "avg_gross_return_pct": float(group["gross_return_pct"].mean()),
                "avg_net_return_pct": float(group["net_return_pct"].mean()),
                "fee_drag_pct": fees / gross_notional if gross_notional else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)


def build_equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative net PnL curve per strategy."""
    if trades.empty:
        return pd.DataFrame(columns=["resolved_at", "strategy", "cumulative_net_pnl", "net_pnl"])

    equity = trades.sort_values("resolved_at").copy()
    equity["cumulative_net_pnl"] = equity.groupby("strategy", sort=False)["net_pnl"].cumsum()
    return equity[["resolved_at", "strategy", "cumulative_net_pnl", "net_pnl"]]


def build_market_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate explainability diagnostics at strategy/market granularity."""
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "market_id",
                "trades",
                "hit_rate",
                "expectancy",
                "turnover",
                "avg_hold_hours",
                "avg_slippage_bps",
                "gross_pnl",
                "net_pnl",
            ]
        )

    rows: list[dict[str, object]] = []
    grouped = trades.groupby(["strategy", "market_id"], sort=False)
    for (strategy, market_id), group in grouped:
        rows.append(
            {
                "strategy": strategy,
                "market_id": market_id,
                "trades": len(group),
                "hit_rate": float((group["net_pnl"] > 0).mean()) if len(group) else 0.0,
                "expectancy": float(group["net_pnl"].mean()) if len(group) else 0.0,
                "turnover": float(group["gross_notional"].sum()),
                "avg_hold_hours": float(group["hold_hours"].mean()) if len(group) else 0.0,
                "avg_slippage_bps": float(group["slippage_bps"].mean()) if len(group) else 0.0,
                "gross_pnl": float(group["gross_pnl"].sum()),
                "net_pnl": float(group["net_pnl"].sum()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["strategy", "net_pnl"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def build_regime_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    """Summarize trade behavior by volatility/spread/liquidity regime buckets."""
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "volatility_bucket",
                "spread_bucket",
                "liquidity_bucket",
                "trades",
                "hit_rate",
                "expectancy",
                "turnover",
                "avg_hold_hours",
                "avg_slippage_bps",
                "net_pnl",
            ]
        )

    frame = trades.copy()
    for col in ("entry_volatility", "entry_spread", "entry_liquidity"):
        if col not in frame.columns:
            frame[col] = pd.NA
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    def _bucket(series: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
        valid = series.dropna()
        if valid.nunique() < _MIN_BUCKET_CARDINALITY:
            return pd.Series(["unknown"] * len(series), index=series.index)
        try:
            return pd.qcut(series, q=3, labels=labels, duplicates="drop").astype("object")
        except ValueError:
            return pd.Series(["unknown"] * len(series), index=series.index)

    frame["volatility_bucket"] = _bucket(frame["entry_volatility"], ("low", "mid", "high"))
    frame["spread_bucket"] = _bucket(frame["entry_spread"], ("tight", "mid", "wide"))
    frame["liquidity_bucket"] = _bucket(frame["entry_liquidity"], ("low", "mid", "high"))
    frame[["volatility_bucket", "spread_bucket", "liquidity_bucket"]] = frame[
        ["volatility_bucket", "spread_bucket", "liquidity_bucket"]
    ].fillna("unknown")

    rows: list[dict[str, object]] = []
    grouped = frame.groupby(
        ["strategy", "volatility_bucket", "spread_bucket", "liquidity_bucket"],
        sort=False,
    )
    for (strategy, vol_bucket, spread_bucket, liq_bucket), group in grouped:
        rows.append(
            {
                "strategy": strategy,
                "volatility_bucket": vol_bucket,
                "spread_bucket": spread_bucket,
                "liquidity_bucket": liq_bucket,
                "trades": len(group),
                "hit_rate": float((group["net_pnl"] > 0).mean()) if len(group) else 0.0,
                "expectancy": float(group["net_pnl"].mean()) if len(group) else 0.0,
                "turnover": float(group["gross_notional"].sum()),
                "avg_hold_hours": float(group["hold_hours"].mean()) if len(group) else 0.0,
                "avg_slippage_bps": float(group["slippage_bps"].mean()) if len(group) else 0.0,
                "net_pnl": float(group["net_pnl"].sum()),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["strategy", "net_pnl"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def compute_consolidated_metrics(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    backtest_summary: pd.DataFrame,
    trade_ledger: pd.DataFrame,
    order_ledger: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
    targets: MetricsTargets | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute one consolidated metrics bundle for notebook reporting."""
    target_cfg = targets or MetricsTargets()

    summary = backtest_summary.copy()
    if summary.empty and not trade_ledger.empty and "strategy" in trade_ledger.columns:
        grouped = trade_ledger.groupby("strategy", dropna=False)
        summary = grouped.agg(
            trades=("strategy", "size"),
            wins=("net_pnl", lambda s: int((s > 0).sum())),
            gross_pnl=("gross_pnl", "sum"),
            net_pnl=("net_pnl", "sum"),
            fees=("fees", "sum"),
        ).reset_index()
        summary["win_rate"] = np.where(
            summary["trades"] > 0,
            summary["wins"] / summary["trades"],
            0.0,
        )

    perf_rows: list[dict[str, float | int | str]] = []
    capital_frames: list[pd.DataFrame] = []

    strategies: list[str] = []
    if not summary.empty and "strategy" in summary.columns:
        strategies = [str(x) for x in summary["strategy"].dropna().unique().tolist()]
    elif not trade_ledger.empty and "strategy" in trade_ledger.columns:
        strategies = [str(x) for x in trade_ledger["strategy"].dropna().unique().tolist()]

    for strategy_name in strategies:
        strategy_trades = (
            trade_ledger[trade_ledger["strategy"] == strategy_name].copy()
            if (not trade_ledger.empty and "strategy" in trade_ledger.columns)
            else pd.DataFrame()
        )

        if not strategy_trades.empty:
            if "entry_ts" in strategy_trades.columns:
                strategy_trades = strategy_trades.sort_values("entry_ts")
            elif "exit_ts" in strategy_trades.columns:
                strategy_trades = strategy_trades.sort_values("exit_ts")

            net_pnl = pd.to_numeric(
                strategy_trades.get("net_pnl", pd.Series(0.0, index=strategy_trades.index)),
                errors="coerce",
            ).fillna(0.0)
            net_pnl_values = net_pnl.to_numpy(dtype=float)
            if len(net_pnl_values):
                cumulative_pnl = net_pnl_values.cumsum()
                pre_trade_capital = float(initial_capital) + np.concatenate(
                    ([0.0], cumulative_pnl[:-1])
                )
                post_trade_capital = float(initial_capital) + cumulative_pnl
                capital_change_pct = np.where(
                    pre_trade_capital > 0,
                    (net_pnl_values / pre_trade_capital) * 100.0,
                    0.0,
                )
                capital_frames.append(
                    pd.DataFrame(
                        {
                            "strategy": strategy_name,
                            "trade_num": np.arange(1, len(net_pnl_values) + 1, dtype=int),
                            "net_pnl": net_pnl_values,
                            "pre_trade_capital": pre_trade_capital,
                            "post_trade_capital": post_trade_capital,
                            "capital_change_pct": capital_change_pct,
                        }
                    )
                )

        strategy_equity = (
            equity_curve[equity_curve["strategy"] == strategy_name].copy()
            if (not equity_curve.empty and "strategy" in equity_curve.columns)
            else pd.DataFrame()
        )

        cumulative_pnl: np.ndarray
        if not strategy_equity.empty and "cumulative_net_pnl" in strategy_equity.columns:
            strategy_equity = strategy_equity.sort_values("resolved_at")
            cumulative_pnl = strategy_equity["cumulative_net_pnl"].to_numpy(dtype=float)
            resolved_at = pd.to_datetime(strategy_equity["resolved_at"], utc=True)
        elif not strategy_trades.empty:
            cumulative_pnl = (
                strategy_trades.get("net_pnl", pd.Series(dtype=float))
                .fillna(0.0)
                .cumsum()
                .to_numpy(dtype=float)
            )
            if "exit_ts" in strategy_trades.columns:
                resolved_at = pd.to_datetime(strategy_trades["exit_ts"], utc=True, errors="coerce")
            elif "entry_ts" in strategy_trades.columns:
                resolved_at = pd.to_datetime(
                    strategy_trades["entry_ts"], utc=True, errors="coerce"
                )
            else:
                resolved_at = pd.to_datetime(
                    pd.RangeIndex(start=0, stop=len(cumulative_pnl)),
                    unit="s",
                    utc=True,
                )
        else:
            cumulative_pnl = np.array([], dtype=float)
            resolved_at = pd.to_datetime(pd.Series(dtype="datetime64[ns, UTC]"), utc=True)

        if len(cumulative_pnl) == 0:
            continue

        capital_curve = float(initial_capital) + cumulative_pnl
        final_capital = float(capital_curve[-1])
        total_return = (
            (final_capital - float(initial_capital)) / float(initial_capital)
            if initial_capital > 0
            else 0.0
        )

        equity_ts = pd.Series(capital_curve, index=resolved_at)
        daily_returns = equity_ts.resample("1D").last().pct_change().dropna()
        annualization_factor = 365.0

        if len(daily_returns) > 1:
            annual_vol = float(daily_returns.std() * np.sqrt(annualization_factor))
            downside = daily_returns[daily_returns < 0]
            annual_downside_vol = (
                float(downside.std() * np.sqrt(annualization_factor)) if len(downside) > 1 else 0.0
            )
        else:
            annual_vol = 0.0
            annual_downside_vol = 0.0

        sharpe_ratio = float(total_return / annual_vol) if annual_vol > 0 else 0.0
        if annual_downside_vol > 0:
            sortino_ratio = float(total_return / annual_downside_vol)
        elif total_return > 0:
            sortino_ratio = float("inf")
        else:
            sortino_ratio = 0.0

        running_max = np.maximum.accumulate(capital_curve)
        drawdown = (capital_curve - running_max) / np.maximum(running_max, 1e-9)
        max_drawdown_pct = float(np.min(drawdown) * 100.0) if len(drawdown) else 0.0

        if max_drawdown_pct < 0:
            calmar_ratio = float((total_return * 100.0) / abs(max_drawdown_pct))
        elif total_return > 0:
            calmar_ratio = float("inf")
        else:
            calmar_ratio = 0.0

        num_trades = len(strategy_trades)
        win_rate_pct = (
            float((strategy_trades["net_pnl"] > 0).mean() * 100.0)
            if (num_trades > 0 and "net_pnl" in strategy_trades.columns)
            else 0.0
        )

        if num_trades > 0 and "net_pnl" in strategy_trades.columns:
            gross_wins = float(
                strategy_trades.loc[strategy_trades["net_pnl"] > 0, "net_pnl"].sum()
            )
            gross_losses = float(
                strategy_trades.loc[strategy_trades["net_pnl"] < 0, "net_pnl"].sum()
            )
            profit_factor = (
                float(gross_wins / abs(gross_losses)) if gross_losses < 0 else float("inf")
            )
        else:
            profit_factor = 0.0

        perf_rows.append(
            {
                "strategy": strategy_name,
                "total_return_pct": float(total_return * 100.0),
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "max_drawdown_pct": max_drawdown_pct,
                "calmar_ratio": calmar_ratio,
                "num_trades": num_trades,
                "win_rate_pct": win_rate_pct,
                "profit_factor": profit_factor,
                "initial_capital": float(initial_capital),
                "final_capital": final_capital,
            }
        )

    perf_df = pd.DataFrame(perf_rows)
    capital_df = pd.concat(capital_frames, ignore_index=True) if capital_frames else pd.DataFrame()

    consolidated = summary.copy() if not summary.empty else pd.DataFrame(columns=["strategy"])
    if not perf_df.empty:
        consolidated = (
            perf_df
            if consolidated.empty
            else consolidated.merge(perf_df, on="strategy", how="left")
        )

    order_state_df = pd.DataFrame(columns=["state", "count"])
    if not order_ledger.empty and "state" in order_ledger.columns:
        order_state_df = (
            order_ledger["state"]
            .value_counts(dropna=False)
            .rename_axis("state")
            .reset_index(name="count")
        )

    checks_rows: list[dict[str, object]] = []
    for _, row in consolidated.iterrows() if not consolidated.empty else []:
        trades = int(row.get("num_trades", row.get("trades", 0)) or 0)
        win_rate_ratio = float(row.get("win_rate", (row.get("win_rate_pct", 0.0) / 100.0)) or 0.0)
        sharpe_ratio = float(row.get("sharpe_ratio", 0.0) or 0.0)
        max_dd_pct = float(row.get("max_drawdown_pct", 0.0) or 0.0)
        final_capital = float(row.get("final_capital", initial_capital) or initial_capital)

        checks_rows.extend(
            [
                {
                    "strategy": row.get("strategy", "unknown"),
                    "check": f"trades >= {target_cfg.min_trades}",
                    "passed": trades >= target_cfg.min_trades,
                    "value": trades,
                },
                {
                    "strategy": row.get("strategy", "unknown"),
                    "check": f"win_rate >= {target_cfg.min_win_rate:.0%}",
                    "passed": win_rate_ratio >= target_cfg.min_win_rate,
                    "value": win_rate_ratio,
                },
                {
                    "strategy": row.get("strategy", "unknown"),
                    "check": f"sharpe >= {target_cfg.min_sharpe}",
                    "passed": sharpe_ratio >= target_cfg.min_sharpe,
                    "value": sharpe_ratio,
                },
                {
                    "strategy": row.get("strategy", "unknown"),
                    "check": f"abs(max_drawdown_pct) <= {target_cfg.max_drawdown_pct}",
                    "passed": abs(max_dd_pct) <= target_cfg.max_drawdown_pct,
                    "value": max_dd_pct,
                },
                {
                    "strategy": row.get("strategy", "unknown"),
                    "check": "capital_preserved > 50",
                    "passed": final_capital > _MIN_CAPITAL_PRESERVED,
                    "value": final_capital,
                },
            ]
        )

    checks_df = pd.DataFrame(checks_rows)
    return {
        "consolidated_metrics": consolidated,
        "validation_checks": checks_df,
        "order_state": order_state_df,
        "capital_evolution": capital_df,
    }


def print_consolidated_metrics_report(metrics_bundle: dict[str, pd.DataFrame]) -> None:
    """Print a compact consolidated metrics report for notebook usage."""
    consolidated = metrics_bundle.get("consolidated_metrics", pd.DataFrame())
    checks = metrics_bundle.get("validation_checks", pd.DataFrame())
    order_state = metrics_bundle.get("order_state", pd.DataFrame())

    print("=" * 80)
    print("CONSOLIDATED METRICS SUMMARY")
    print("=" * 80)

    if consolidated.empty:
        print("No consolidated metrics available.")
    else:
        pd.set_option("display.max_columns", None)
        print("\nStrategy metrics:")
        print(consolidated.round(4).to_string(index=False))

    if not order_state.empty:
        print("\nOrder states:")
        print(order_state.to_string(index=False))

    if checks.empty:
        print("\nNo validation checks available.")
    else:
        print("\nValidation checks:")
        for _, row in checks.iterrows():
            status = "PASS" if bool(row.get("passed", False)) else "FAIL"
            print(
                f"[{status}] {row.get('strategy', 'unknown')} :: {row.get('check')} -> {row.get('value')}"
            )

    print("\n" + "=" * 80)
