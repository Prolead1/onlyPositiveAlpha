from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Mode = Literal["sample", "full"]

DEFAULT_SAMPLE_HOUR_LIMIT = 12
DEFAULT_ACF_LAGS = (1, 2, 5, 10)
DEFAULT_FIG_DPI = 180
DEFAULT_EXPECTED_COINS = ("BNB", "BTC", "DOGE", "ETH", "HYPE", "SOL", "XRP")
DEFAULT_RESOLUTION_ORDER = ("5m", "15m", "4h")
SMALL_NUMBER = 1.0e-12


@dataclass(frozen=True)
class RunPaths:
    """Filesystem layout for the EDA pipeline."""

    repo_root: Path
    data_root: Path
    eda_root: Path
    output_root: Path
    figures_root: Path
    cache_root: Path

    @property
    def insights_path(self) -> Path:
        """Return the canonical markdown insights path."""
        return self.eda_root / "insights.md"

    @property
    def readme_path(self) -> Path:
        """Return the local module README path."""
        return self.eda_root / "README.md"

    def mode_cache_dir(self, mode: Mode) -> Path:
        """Return the cache directory for the selected mode."""
        return self.cache_root / mode

    def mode_figure_dir(self, mode: Mode) -> Path:
        """Return the figure directory for the selected mode."""
        return self.figures_root / mode


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for sample and full EDA execution."""

    mode: Mode
    paths: RunPaths
    sample_hour_limit: int = DEFAULT_SAMPLE_HOUR_LIMIT
    acf_lags: tuple[int, ...] = DEFAULT_ACF_LAGS
    figure_dpi: int = DEFAULT_FIG_DPI
    expected_coins: tuple[str, ...] = DEFAULT_EXPECTED_COINS
    resolution_order: tuple[str, ...] = DEFAULT_RESOLUTION_ORDER
    top_ranking_count: int = 10
    stale_quote_epsilon: float = SMALL_NUMBER
    jump_threshold_small: float = 0.01
    jump_threshold_large: float = 0.05

    @property
    def is_sample(self) -> bool:
        """Whether the pipeline is running in bounded sample mode."""
        return self.mode == "sample"

    @property
    def expected_coin_resolution_pairs(self) -> tuple[tuple[str, str], ...]:
        """Return the expected coin x resolution universe in stable order."""
        return tuple(
            (coin, resolution)
            for coin in self.expected_coins
            for resolution in self.resolution_order
        )


def build_config(mode: Mode, repo_root: Path | None = None) -> RunConfig:
    root = repo_root or Path(__file__).resolve().parents[2]
    paths = RunPaths(
        repo_root=root,
        data_root=root / "data" / "pmxt_polymarket_orderbook",
        eda_root=root / "EDA",
        output_root=root / "EDA" / "output",
        figures_root=root / "EDA" / "output" / "figures",
        cache_root=root / "EDA" / "output" / "cache",
    )
    return RunConfig(mode=mode, paths=paths)
