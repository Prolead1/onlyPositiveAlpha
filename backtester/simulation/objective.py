from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtester.config.types import BacktestRunResult


_MIN_CAPITAL = 1e-6
_MIN_RETURN_FOR_LOG = -0.999999


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for market-level risk-adjusted objective computation."""

    drawdown_quantile: float = 0.6
    min_markets: int = 30
    min_trades: int = 50
    default_initial_capital: float = 100.0


@dataclass(frozen=True)
class ObjectiveMetrics:
    """Risk-adjusted metrics used for strategy/gate/hyperparameter ranking."""

    market_count: int
    trade_count: int
    mean_market_log_return: float
    market_log_return_vol: float
    market_sharpe_log: float
    max_drawdown_pct: float
    objective_eligible: bool


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(parsed):
        return default
    return parsed


def _infer_initial_capital(result: BacktestRunResult, default_capital: float) -> float:
    snapshot = getattr(getattr(result, "metadata", None), "config_snapshot", None)
    if isinstance(snapshot, dict):
        for key in ("available_capital", "max_gross_exposure"):
            if key in snapshot:
                candidate = _safe_float(snapshot.get(key), default_capital)
                if candidate > 0:
                    return candidate
    return default_capital


def _build_market_pnl_series(result: BacktestRunResult) -> pd.DataFrame:
    diagnostics = getattr(result, "diagnostics_by_market", pd.DataFrame())
    if diagnostics.empty or "market_id" not in diagnostics.columns:
        return pd.DataFrame(columns=["market_id", "net_pnl", "resolved_at"])

    frame = diagnostics[["market_id", "net_pnl"]].copy()
    frame["market_id"] = frame["market_id"].astype(str)
    frame["net_pnl"] = pd.to_numeric(frame["net_pnl"], errors="coerce").fillna(0.0)
    frame = frame.groupby("market_id", sort=False, as_index=False)["net_pnl"].sum()

    resolution = getattr(result, "resolution_frame", pd.DataFrame())
    if {
        "market_id",
        "resolved_at",
    }.issubset(resolution.columns):
        ordering = resolution[["market_id", "resolved_at"]].copy()
        ordering["market_id"] = ordering["market_id"].astype(str)
        ordering["resolved_at"] = pd.to_datetime(ordering["resolved_at"], utc=True, errors="coerce")
        ordering = ordering.dropna(subset=["resolved_at"]).groupby(
            "market_id", sort=False, as_index=False
        )["resolved_at"].min()
        frame = frame.merge(ordering, on="market_id", how="left")
    else:
        frame["resolved_at"] = pd.NaT

    frame = frame.sort_values(["resolved_at", "market_id"], ascending=[True, True]).reset_index(
        drop=True
    )
    return frame


def compute_market_objective_metrics(
    result: BacktestRunResult,
    *,
    drawdown_limit_pct: float | None = None,
    objective_config: ObjectiveConfig | None = None,
) -> ObjectiveMetrics:
    """Compute market-level log-return Sharpe and drawdown for objective ranking."""
    cfg = objective_config or ObjectiveConfig()
    initial_capital = _infer_initial_capital(result, default_capital=cfg.default_initial_capital)

    market_pnl = _build_market_pnl_series(result)
    trade_ledger = getattr(result, "trade_ledger", pd.DataFrame())
    trade_count = len(trade_ledger)

    if market_pnl.empty:
        return ObjectiveMetrics(
            market_count=0,
            trade_count=trade_count,
            mean_market_log_return=0.0,
            market_log_return_vol=0.0,
            market_sharpe_log=0.0,
            max_drawdown_pct=0.0,
            objective_eligible=False,
        )

    net_pnl = market_pnl["net_pnl"].to_numpy(dtype=float)
    cumulative_pnl = np.cumsum(net_pnl)
    pre_market_capital = initial_capital + np.concatenate(([0.0], cumulative_pnl[:-1]))
    pre_market_capital = np.maximum(pre_market_capital, _MIN_CAPITAL)

    market_returns = net_pnl / pre_market_capital
    market_returns = np.clip(market_returns, _MIN_RETURN_FOR_LOG, None)
    market_log_returns = np.log1p(market_returns)

    mean_log_return = float(np.mean(market_log_returns)) if len(market_log_returns) else 0.0
    vol_log_return = float(np.std(market_log_returns, ddof=1)) if len(market_log_returns) > 1 else 0.0

    if vol_log_return > 0 and len(market_log_returns) > 1:
        market_sharpe = float((mean_log_return / vol_log_return) * np.sqrt(len(market_log_returns)))
    else:
        market_sharpe = 0.0

    capital_curve = initial_capital + cumulative_pnl
    running_max = np.maximum.accumulate(np.maximum(capital_curve, _MIN_CAPITAL))
    drawdown = (capital_curve - running_max) / running_max
    max_drawdown_pct = float(abs(np.min(drawdown)) * 100.0) if len(drawdown) else 0.0

    passes_sample_floor = len(market_log_returns) >= cfg.min_markets and trade_count >= cfg.min_trades
    passes_drawdown = True if drawdown_limit_pct is None else max_drawdown_pct <= float(drawdown_limit_pct)

    return ObjectiveMetrics(
        market_count=len(market_log_returns),
        trade_count=trade_count,
        mean_market_log_return=mean_log_return,
        market_log_return_vol=vol_log_return,
        market_sharpe_log=market_sharpe,
        max_drawdown_pct=max_drawdown_pct,
        objective_eligible=bool(passes_sample_floor and passes_drawdown),
    )


def derive_adaptive_drawdown_limit(
    max_drawdown_values: pd.Series,
    *,
    quantile: float,
    fallback_limit_pct: float = 25.0,
) -> float:
    """Derive drawdown limit from train-split candidate distribution."""
    numeric = pd.to_numeric(max_drawdown_values, errors="coerce").dropna()
    if numeric.empty:
        return float(fallback_limit_pct)

    q = min(max(float(quantile), 0.05), 0.95)
    limit = float(numeric.quantile(q, interpolation="linear"))
    if not np.isfinite(limit):
        return float(fallback_limit_pct)
    return max(0.0, limit)


def rank_by_objective(
    frame: pd.DataFrame,
    *,
    drawdown_limit_pct: float,
) -> pd.DataFrame:
    """Rank candidates by Sharpe under hard drawdown eligibility constraints."""
    ranked = frame.copy()
    passes_drawdown = (
        pd.to_numeric(ranked["max_drawdown_pct"], errors="coerce").fillna(np.inf)
        <= float(drawdown_limit_pct)
    )
    if "objective_eligible" in ranked.columns:
        prior_eligibility = ranked["objective_eligible"].astype(bool)
    else:
        prior_eligibility = pd.Series(True, index=ranked.index)
    ranked["objective_eligible"] = passes_drawdown & prior_eligibility

    ranked = ranked.sort_values(
        [
            "objective_eligible",
            "market_sharpe_log",
            "mean_market_log_return",
            "trades",
            "net_pnl",
        ],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    ranked["objective_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked
