from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl
import pytest

from EDA.core.config import RunConfig, RunPaths
from EDA.core.io import HourlyBatch, select_batches
from EDA.pipeline.analysis import build_research_rankings, validate_sample_coverage
from EDA.render import plots as eda_plots


def _make_config(
    tmp_path: Path,
    *,
    sample_hour_limit: int = 12,
    expected_coins: tuple[str, ...] = ("BNB", "BTC", "DOGE", "ETH", "HYPE", "SOL", "XRP"),
    resolution_order: tuple[str, ...] = ("5m", "15m", "4h"),
) -> RunConfig:
    paths = RunPaths(
        repo_root=tmp_path,
        data_root=tmp_path / "data",
        eda_root=tmp_path / "EDA",
        output_root=tmp_path / "EDA" / "output",
        figures_root=tmp_path / "EDA" / "output" / "figures",
        cache_root=tmp_path / "EDA" / "output" / "cache",
    )
    return RunConfig(
        mode="sample",
        paths=paths,
        sample_hour_limit=sample_hour_limit,
        expected_coins=expected_coins,
        resolution_order=resolution_order,
    )


def _make_batch(hour_key: str) -> HourlyBatch:
    return HourlyBatch(
        hour_key=hour_key,
        mapping_paths=(),
        book_snapshot_paths=(),
        price_change_paths=(),
    )


def test_select_batches_prefers_latest_full_universe_hours(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(
        tmp_path,
        sample_hour_limit=2,
        expected_coins=("BTC", "ETH"),
        resolution_order=("5m",),
    )
    batches = [_make_batch(hour_key) for hour_key in ("h1", "h2", "h3", "h4")]
    coverage_by_hour = {
        "h1": {("BTC", "5m")},
        "h2": {("BTC", "5m"), ("ETH", "5m")},
        "h3": {("BTC", "5m"), ("ETH", "5m")},
        "h4": {("BTC", "5m"), ("ETH", "5m")},
    }

    monkeypatch.setattr(
        "EDA.core.io._mapping_coin_resolution_pairs",
        lambda batch: coverage_by_hour[batch.hour_key],
    )

    selected = select_batches(config, batches)

    assert [batch.hour_key for batch in selected] == ["h3", "h4"]


def test_validate_sample_coverage_raises_for_missing_pairs(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path,
        expected_coins=("BTC", "ETH"),
        resolution_order=("5m", "15m"),
    )
    coverage_table = pl.DataFrame(
        [
            {"coin": "BTC", "resolution": "5m", "snapshot_count": 10, "event_count": 10},
            {"coin": "BTC", "resolution": "15m", "snapshot_count": 10, "event_count": 10},
            {"coin": "ETH", "resolution": "5m", "snapshot_count": 8, "event_count": 3},
        ],
    )

    with pytest.raises(RuntimeError, match="ETH 15m"):
        validate_sample_coverage(config, coverage_table)


def test_format_time_axis_uses_compact_date_formatting() -> None:
    fig, ax = plt.subplots()
    ax.plot(
        [date(2026, 4, day) for day in range(1, 8)],
        [0.1 * day for day in range(1, 8)],
    )

    eda_plots._format_time_axis(ax, show_labels=True)
    fig.canvas.draw()

    assert isinstance(ax.xaxis.get_major_locator(), mdates.DayLocator)
    assert isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter)
    assert any(label.get_rotation() == 25 for label in ax.get_xticklabels())

    plt.close(fig)


def test_build_research_rankings_handles_zero_volatility_without_nan() -> None:
    coverage_table = pl.DataFrame(
        [
            {"coin": "BTC", "resolution": "5m", "unique_markets": 10, "snapshot_count": 100, "event_count": 50},
            {"coin": "ETH", "resolution": "5m", "unique_markets": 8, "snapshot_count": 90, "event_count": 45},
        ],
    )
    microstructure_summary = pl.DataFrame(
        [
            {
                "coin": "BTC",
                "resolution": "5m",
                "total_visible_depth_median": 1000.0,
                "relative_spread_median": 0.02,
                "stale_quote_frequency_median": 0.1,
            },
            {
                "coin": "ETH",
                "resolution": "5m",
                "total_visible_depth_median": 800.0,
                "relative_spread_median": 0.03,
                "stale_quote_frequency_median": 0.2,
            },
        ],
    )
    return_summary = pl.DataFrame(
        [
            {
                "coin": "BTC",
                "resolution": "5m",
                "realized_volatility_median": 0.0,
                "abs_midpoint_change_p95": 0.01,
                "jump_rate_large_median": 0.001,
            },
            {
                "coin": "ETH",
                "resolution": "5m",
                "realized_volatility_median": 0.02,
                "abs_midpoint_change_p95": 0.015,
                "jump_rate_large_median": 0.002,
            },
        ],
    )
    stability_table = pl.DataFrame(
        [
            {"coin": "BTC", "resolution": "5m", "spread_cv": 0.1},
            {"coin": "ETH", "resolution": "5m", "spread_cv": 0.2},
        ],
    )

    rankings = build_research_rankings(
        coverage_table,
        microstructure_summary,
        return_summary,
        stability_table,
    )

    assert rankings["research_score"].is_nan().sum() == 0
