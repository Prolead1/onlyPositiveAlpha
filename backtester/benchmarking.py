from __future__ import annotations

import tracemalloc
from dataclasses import dataclass, replace
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import pandas as pd

from backtester.config.types import BacktestConfig

if TYPE_CHECKING:
    from pathlib import Path


class BenchmarkRunnerLike(Protocol):
    """Minimal runner protocol used by benchmarking helpers."""

    def load_market_events(self, *args: object, **kwargs: object) -> pd.DataFrame:
        """Load market events for a benchmark run."""
        ...

    def compute_orderbook_features_cached(
        self,
        *args: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Compute cached orderbook features for a benchmark run."""
        ...

    def run_backtest(self, *args: object, **kwargs: object) -> None:
        """Run a benchmark backtest."""
        ...

    def run_sensitivity_scenarios(
        self,
        *args: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Run benchmark sensitivity scenarios."""
        ...


@dataclass(frozen=True)
class BenchmarkThresholds:
    """Target thresholds used to judge benchmark readiness."""

    runtime_sla_seconds: float
    peak_memory_mb: float
    max_net_pnl_delta: float
    max_feature_delta: float


def run_deterministic_benchmark(  # noqa: PLR0913
    *,
    runner: BenchmarkRunnerLike,
    mapping_dir: object,
    strategy: object,
    start: object = None,
    end: object = None,
    prepared_manifest_path: Path | str | None = None,
    market_batch_size: int | None = None,
    limit_files: int | None = None,
    max_rows_per_file: int | None = None,
    market_slug_prefix: str | None = None,
    config: BacktestConfig | None = None,
    parameter_sweeps: dict[str, list[object]] | None = None,
    stress_scenarios: list[str] | None = None,
) -> dict[str, float | int]:
    """Run deterministic stage-timed benchmark and return scalar metrics."""
    del max_rows_per_file, market_slug_prefix

    cfg = config or BacktestConfig(enable_progress_bars=False, metrics_logging_enabled=False)
    if cfg.enable_progress_bars or cfg.metrics_logging_enabled:
        cfg = replace(cfg, enable_progress_bars=False, metrics_logging_enabled=False)

    tracemalloc.start()
    bench_started = perf_counter()

    load_started = perf_counter()
    loaded_features = runner.load_prepared_features(
        start=start,
        end=end,
        limit_files=limit_files,
        features_manifest_path=prepared_manifest_path,
    )
    load_elapsed = perf_counter() - load_started

    feature_started = perf_counter()
    loaded_market_events = loaded_features
    feature_elapsed = perf_counter() - feature_started

    effective_batch_size = int(market_batch_size) if market_batch_size is not None else 100

    backtest_started = perf_counter()
    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=prepared_manifest_path,
        strategy=strategy,
        strategy_name="benchmark_strategy",
        market_batch_size=effective_batch_size,
        config=cfg,
    )
    backtest_elapsed = perf_counter() - backtest_started

    scenario_started = perf_counter()
    sensitivity = runner.run_sensitivity_scenarios(
        mapping_dir=mapping_dir,
        prepared_manifest_path=prepared_manifest_path,
        strategy=strategy,
        strategy_name="benchmark_strategy",
        market_batch_size=effective_batch_size,
        base_config=cfg,
        parameter_sweeps=parameter_sweeps or {"shares": [cfg.shares]},
        stress_scenarios=stress_scenarios or ["baseline"],
    )
    scenario_elapsed = perf_counter() - scenario_started

    _current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "load_seconds": round(load_elapsed, 6),
        "feature_seconds": round(feature_elapsed, 6),
        "backtest_seconds": round(backtest_elapsed, 6),
        "scenario_seconds": round(scenario_elapsed, 6),
        "total_seconds": round(perf_counter() - bench_started, 6),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 6),
        "market_event_rows": len(loaded_market_events),
        "feature_rows": len(loaded_features),
        "trade_rows": len(result.trade_ledger),
        "scenario_rows": len(sensitivity),
    }


def build_rollout_gate_report(
    gate_metrics: dict[str, dict[str, float | int]],
    *,
    thresholds: BenchmarkThresholds,
) -> pd.DataFrame:
    """Build rollout readiness table for 9/100/1000/full gates."""
    preferred_order = ["9", "100", "1000", "full"]
    ordered_gates = [gate for gate in preferred_order if gate in gate_metrics]
    ordered_gates.extend(gate for gate in gate_metrics if gate not in ordered_gates)

    rows: list[dict[str, object]] = []
    for gate in ordered_gates:
        metrics = gate_metrics[gate]
        runtime = float(metrics.get("total_seconds", 0.0))
        peak_mem = float(metrics.get("peak_memory_mb", 0.0))
        net_pnl_delta = float(metrics.get("parity_net_pnl_delta", 0.0))
        feature_delta = float(metrics.get("parity_feature_delta", 0.0))

        runtime_ok = runtime <= thresholds.runtime_sla_seconds
        memory_ok = peak_mem <= thresholds.peak_memory_mb
        pnl_ok = abs(net_pnl_delta) <= thresholds.max_net_pnl_delta
        feature_ok = abs(feature_delta) <= thresholds.max_feature_delta
        rows.append(
            {
                "gate": gate,
                "runtime_seconds": runtime,
                "peak_memory_mb": peak_mem,
                "parity_net_pnl_delta": net_pnl_delta,
                "parity_feature_delta": feature_delta,
                "runtime_ok": runtime_ok,
                "memory_ok": memory_ok,
                "pnl_parity_ok": pnl_ok,
                "feature_parity_ok": feature_ok,
                "ready": runtime_ok and memory_ok and pnl_ok and feature_ok,
            }
        )

    return pd.DataFrame(rows)
