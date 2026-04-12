from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import polars as pl

from backtester import BacktestRunner
import futureWork.analysis.diagnostics as diagnostics_module
from futureWork.core.config import build_config
from futureWork.analysis.diagnostics import run_parameter_stress
from futureWork.analysis.diagnostics import run_exact_backtest
from futureWork.analysis.analysis_runner import _load_cached_parameter_stress
from futureWork.core.utils import combo_output_dir
from futureWork.strategy.current_alpha import (
    _normalize_panel_for_strategy,
    _build_strategy_output,
    build_action_window_panel,
    build_backtest_config,
    run_alpha_backtest,
)


def _make_panel() -> pl.DataFrame:
    market_id = "market_1"
    winning_token_id = "tok_yes"
    market_start_time = pd.Timestamp("2026-04-01T00:00:00Z")
    resolved_at = pd.Timestamp("2026-04-01T00:02:30Z")
    rows = [
        {
            "ts_event": pd.Timestamp("2026-04-01T00:00:00Z"),
            "token_id": "tok_yes",
            "best_bid": 0.39,
            "best_ask": 0.41,
            "bid_depth_1": 2.0,
            "ask_depth_1": 0.2,
            "bid_depth_5": 500.0,
            "ask_depth_5": 120.0,
        },
        {
            "ts_event": pd.Timestamp("2026-04-01T00:00:00Z"),
            "token_id": "tok_no",
            "best_bid": 0.57,
            "best_ask": 0.63,
            "bid_depth_1": 0.2,
            "ask_depth_1": 1.0,
            "bid_depth_5": 120.0,
            "ask_depth_5": 500.0,
        },
        {
            "ts_event": pd.Timestamp("2026-04-01T00:01:00Z"),
            "token_id": "tok_yes",
            "best_bid": 0.42,
            "best_ask": 0.44,
            "bid_depth_1": 2.2,
            "ask_depth_1": 0.2,
            "bid_depth_5": 520.0,
            "ask_depth_5": 110.0,
        },
        {
            "ts_event": pd.Timestamp("2026-04-01T00:01:00Z"),
            "token_id": "tok_no",
            "best_bid": 0.55,
            "best_ask": 0.63,
            "bid_depth_1": 0.2,
            "ask_depth_1": 1.1,
            "bid_depth_5": 110.0,
            "ask_depth_5": 520.0,
        },
    ]

    enriched_rows: list[dict[str, object]] = []
    for row in rows:
        spread = float(row["best_ask"]) - float(row["best_bid"])
        mid_price = (float(row["best_ask"]) + float(row["best_bid"])) / 2.0
        is_winner = row["token_id"] == winning_token_id
        enriched_rows.append(
            {
                "ts_event": row["ts_event"],
                "hour_key": "2026-04-01T00",
                "market_id": market_id,
                "token_id": row["token_id"],
                "coin": "BTC",
                "resolution": "5m",
                "market_start_time": market_start_time,
                "resolved_at": resolved_at,
                "tick_size": 0.01,
                "winning_token_id": winning_token_id,
                "best_bid": row["best_bid"],
                "best_ask": row["best_ask"],
                "spread": spread,
                "spread_bps": spread / mid_price * 10_000.0,
                "mid_price": mid_price,
                "bid_depth_1": row["bid_depth_1"],
                "ask_depth_1": row["ask_depth_1"],
                "bid_depth_5": row["bid_depth_5"],
                "ask_depth_5": row["ask_depth_5"],
                "time_to_resolution_secs": float((resolved_at - row["ts_event"]).total_seconds()),
                "is_winner": int(is_winner),
                "terminal_return": 1.0 - mid_price if is_winner else -mid_price,
            }
        )
    return pl.DataFrame(enriched_rows)


def test_build_action_window_panel_matches_exact_alpha_strategy(tmp_path) -> None:
    config = build_config("subset", repo_root=tmp_path)
    config.paths.data_root.mkdir(parents=True, exist_ok=True)
    panel = _make_panel()

    strategy_panel = build_action_window_panel(
        panel.lazy(),
        config,
        coin="BTC",
        resolution="5m",
    )
    direct = _build_strategy_output(panel.to_pandas(), config)
    direct = direct.reset_index().rename(columns={str(direct.index.name or "index"): "ts_event"})

    compare_cols = [
        "ts_event",
        "market_id",
        "token_id",
        "relative_pressure",
        "relative_spread_tightness",
        "relative_depth",
        "relative_imbalance",
        "relative_book_score",
        "cumulative_signal_raw",
        "cumulative_signal_normalized",
        "book_rank",
        "score_gap",
        "relative_confidence",
        "signal",
        "action_score",
        "signal_abs",
    ]
    actual = strategy_panel.select(compare_cols).sort(["ts_event", "market_id", "token_id"]).to_pandas()
    expected = direct.loc[:, compare_cols].sort_values(["ts_event", "market_id", "token_id"]).reset_index(drop=True)
    actual = actual.reset_index(drop=True)
    pdt.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9)

    cumulative_panel = strategy_panel.sort(["market_id", "token_id", "ts_event"]).to_pandas()
    expected_pressure = cumulative_panel.groupby(["market_id", "token_id"], sort=False)["relative_pressure"].cumsum()
    expected_spread = cumulative_panel.groupby(["market_id", "token_id"], sort=False)["relative_spread_tightness"].cumsum()
    assert np.allclose(cumulative_panel["cum_relative_pressure"], expected_pressure)
    assert np.allclose(cumulative_panel["cum_relative_spread"], expected_spread)


def test_run_alpha_backtest_uses_shared_execution_engine(tmp_path) -> None:
    config = build_config("subset", repo_root=tmp_path)
    config.paths.data_root.mkdir(parents=True, exist_ok=True)
    panel = _make_panel()
    strategy_panel = build_action_window_panel(
        panel.lazy(),
        config,
        coin="BTC",
        resolution="5m",
    )

    actual = run_alpha_backtest(
        strategy_panel,
        config,
        coin="BTC",
        resolution="5m",
        persist=False,
    ).sort(["entry_ts", "market_id", "token_id"])

    runner = BacktestRunner(storage_path=config.paths.data_root)
    resolution_frame = (
        strategy_panel.select(["market_id", "resolved_at", "winning_token_id"])
        .sort(["market_id", "resolved_at"])
        .unique("market_id", keep="last")
        .rename({"winning_token_id": "winning_asset_id"})
        .with_columns(
            pl.lit("unknown").alias("settlement_source"),
            pl.lit(True).alias("fees_enabled_market"),
        )
        .to_pandas()
    )
    direct_pd = runner.simulate_hold_to_resolution_backtest(
        _build_strategy_output(strategy_panel.to_pandas(), config),
        resolution_frame,
        strategy_name=config.analysis_scenario_name,
        config=build_backtest_config(),
    )
    direct = pl.from_pandas(direct_pd, include_index=False).sort(["entry_ts", "market_id", "token_id"])

    assert actual.height == direct.height == 1
    assert actual["market_id"].to_list() == direct["market_id"].to_list()
    assert actual["token_id"].to_list() == direct["token_id"].to_list()
    assert np.allclose(actual["net_pnl"].to_numpy(), direct["net_pnl"].to_numpy())
    assert actual["scenario"].to_list() == [config.analysis_scenario_name]


def test_normalize_panel_for_strategy_collapses_duplicate_signal_rows() -> None:
    panel = _make_panel().to_pandas()
    duplicated = pd.concat([panel, panel.iloc[[0]], panel.iloc[[0]].copy()], ignore_index=True)

    normalized = _normalize_panel_for_strategy(duplicated)

    assert len(normalized) == len(panel)
    assert normalized.duplicated(["ts_event", "market_id", "token_id"]).sum() == 0


def test_build_action_window_panel_ignores_legacy_cached_panel(tmp_path) -> None:
    config = build_config("full", repo_root=tmp_path)
    config.paths.data_root.mkdir(parents=True, exist_ok=True)
    combo_dir = combo_output_dir(config.paths, "BTC", "5m")
    combo_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"legacy_only": [1]}).write_parquet(combo_dir / "full_action_window_panel.parquet")

    strategy_panel = build_action_window_panel(
        _make_panel().lazy(),
        config,
        coin="BTC",
        resolution="5m",
    )

    assert "action_score" in strategy_panel.columns
    assert "score_gap" in strategy_panel.columns


def test_run_alpha_backtest_ignores_legacy_cached_trade_ledger(tmp_path) -> None:
    config = build_config("full", repo_root=tmp_path)
    config.paths.data_root.mkdir(parents=True, exist_ok=True)
    panel = _make_panel()
    strategy_panel = build_action_window_panel(
        panel.lazy(),
        config,
        coin="BTC",
        resolution="5m",
    )

    combo_dir = combo_output_dir(config.paths, "BTC", "5m")
    combo_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "scenario": ["optimistic"],
            "market_id": ["market_1"],
            "token_id": ["tok_yes"],
            "entry_ts": [pd.Timestamp("2026-04-01T00:01:00Z")],
            "resolved_at": [pd.Timestamp("2026-04-01T00:02:30Z")],
            "gross_notional": [1.0],
            "net_pnl": [0.1],
            "candidate_count": [1],
            "action_score": [1.0],
            "score_gap": [0.1],
        }
    ).write_parquet(combo_dir / "full_trade_ledger.parquet")

    actual = run_alpha_backtest(
        strategy_panel,
        config,
        coin="BTC",
        resolution="5m",
        persist=True,
    )

    assert actual["scenario"].to_list() == [config.analysis_scenario_name]


def test_run_parameter_stress_reuses_base_summary_without_rerunning_base(monkeypatch) -> None:
    config = build_config("subset")
    observed_variants: list[str | None] = []

    def _fake_run_alpha_backtest(*args, artifact_suffix: str | None = None, **kwargs) -> pl.DataFrame:
        observed_variants.append(artifact_suffix)
        return pl.DataFrame()

    monkeypatch.setattr(diagnostics_module, "run_alpha_backtest", _fake_run_alpha_backtest)

    base_summary = pl.DataFrame(
        {
            "coin": ["BTC"],
            "resolution": ["5m"],
            "scenario": [config.analysis_scenario_name],
            "trade_count": [1],
            "turnover": [1.0],
            "gross_pnl": [0.1],
            "net_pnl": [0.09],
            "fee_usdc": [0.01],
            "avg_hold_hours": [0.1],
            "hit_rate": [1.0],
            "avg_pnl_per_trade": [0.09],
            "total_return": [0.0009],
            "annualized_return": [None],
            "sharpe": [None],
            "sortino": [None],
            "calmar": [None],
            "max_drawdown_pct": [0.0],
            "exposure_time_hours": [0.1],
            "sample_days": [0.01],
            "cost_drag_pct_turnover": [0.01],
        }
    )

    stress = run_parameter_stress(
        pl.DataFrame({"x": [1]}),
        config,
        coin="BTC",
        resolution="5m",
        base_summary=base_summary,
    )

    assert observed_variants == [
        "score_minus_0p05",
        "score_plus_0p05",
        "gap_minus_0p05",
        "gap_plus_0p05",
    ]
    assert stress.filter(pl.col("variant") == "base")["trade_count"].to_list() == [1]


def test_run_exact_backtest_emits_zero_row_summary_for_no_trade_combo(monkeypatch) -> None:
    config = build_config("subset")

    monkeypatch.setattr(
        diagnostics_module,
        "run_alpha_backtest",
        lambda *args, **kwargs: pl.DataFrame(),
    )

    frames = run_exact_backtest(
        pl.DataFrame({"x": [1]}),
        config,
        coin="BNB",
        resolution="15m",
    )

    assert frames.trades.is_empty()
    assert frames.summary["scenario"].to_list() == [config.analysis_scenario_name]
    assert frames.summary["trade_count"].to_list() == [0]


def test_run_alpha_backtest_persists_schemaful_empty_trade_ledger(tmp_path, monkeypatch) -> None:
    config = build_config("full", repo_root=tmp_path)
    config.paths.data_root.mkdir(parents=True, exist_ok=True)
    strategy_panel = build_action_window_panel(
        _make_panel().lazy(),
        config,
        coin="BTC",
        resolution="5m",
    )

    monkeypatch.setattr(
        BacktestRunner,
        "simulate_hold_to_resolution_backtest",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    actual = run_alpha_backtest(
        strategy_panel,
        config,
        coin="BTC",
        resolution="5m",
        persist=True,
    )
    cached = pl.read_parquet(combo_output_dir(config.paths, "BTC", "5m") / "full_trade_ledger.parquet")

    assert actual.is_empty()
    assert "scenario" in actual.columns
    assert "scenario" in cached.columns


def test_load_cached_parameter_stress_requires_exact_alpha_scenarios(tmp_path) -> None:
    config = build_config("full", repo_root=tmp_path)
    combo_dir = combo_output_dir(config.paths, "BTC", "15m")
    combo_dir.mkdir(parents=True, exist_ok=True)

    legacy = pl.DataFrame(
        {
            "coin": ["BTC"],
            "resolution": ["15m"],
            "scenario": ["optimistic"],
            "variant": ["base"],
        }
    )
    legacy.write_parquet(combo_dir / "full_parameter_stress.parquet")
    assert _load_cached_parameter_stress(config, coin="BTC", resolution="15m") is None

    exact = pl.DataFrame(
        {
            "coin": ["BTC"],
            "resolution": ["15m"],
            "scenario": [config.analysis_scenario_name],
            "variant": ["base"],
        }
    )
    exact.write_parquet(combo_dir / "full_parameter_stress.parquet")
    cached = _load_cached_parameter_stress(config, coin="BTC", resolution="15m")
    assert cached is not None
    assert cached["scenario"].to_list() == [config.analysis_scenario_name]
