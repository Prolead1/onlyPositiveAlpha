"""Backtest runner orchestration for stored streaming data workflows.

This module keeps BacktestRunner initialization and composes the full runner
behavior while shifting mixin logic into domain modules.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

from backtester.config.types import (
    BacktestConfig,
    BacktestRunResult,
    FeatureGatePolicy,
    RunMetadata,
    ValidationPolicy,
)
from backtester.feature_generator import FeatureGenerator, FeatureGeneratorConfig
from backtester.loaders.runner_core import BacktestCoreOps
from backtester.resolution.manager import BacktestResolutionManager
from backtester.runner_pipeline import BacktestPipelineOrchestrator
from backtester.runner_support import BacktestSupportOps
from backtester.simulation.execution_engine import BacktestSimulationEngine
from utils.paths import validate_path_exists

logger = logging.getLogger(__name__)


class BacktestRunner(
    BacktestCoreOps,
    BacktestResolutionManager,
    BacktestSimulationEngine,
    BacktestSupportOps,
    BacktestPipelineOrchestrator,
):
    """Load stored streaming data and generate backtest features."""

    def __init__(self, storage_path: Path | str, cache_dir: Path | str | None = None) -> None:
        """Initialize backtest runner.

        Parameters
        ----------
        storage_path : Path | str
            Path to stored stream data root directory.
        cache_dir : Path | str | None
            Path to cache directory for computed features. If None, uses
            storage_path/../feature_cache.
        """
        self.storage_path = Path(storage_path)
        self.is_pmxt_mode = "pmxt" in {part.lower() for part in self.storage_path.parts}
        self.market_path = self.storage_path
        compatibility_market_path = self.storage_path / "polymarket_market"
        if compatibility_market_path.exists():
            self.market_path = compatibility_market_path
        self.rtds_path = self.storage_path / "polymarket_rtds"
        self.mapping_path = self.storage_path.parent / "mapping"

        if cache_dir is None:
            self.cache_path = self.storage_path.parent / "feature_cache"
        else:
            self.cache_path = Path(cache_dir)

        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.feature_generator = FeatureGenerator(
            FeatureGeneratorConfig(cache_dir=self.cache_path)
        )

        validate_path_exists(
            self.storage_path,
            f"Storage path does not exist: {self.storage_path}",
        )

        logger.info("Initialized BacktestRunner with storage path: %s", self.storage_path)
        logger.info("Market event path: %s", self.market_path)
        logger.info("Mapping path: %s", self.mapping_path)
        logger.info("Feature cache directory: %s", self.cache_path)
        logger.info("PMXT mode: %s", self.is_pmxt_mode)

        self._slug_prefix_condition_ids_cache: dict[str, set[str]] = {}
        self._last_resolution_repair_audit: list[dict[str, object]] = []
        self._prepared_inputs_cache_max_entries = 8
        self._prepared_inputs_cache: OrderedDict[
            tuple[object, ...],
            dict[str, object],
        ] = OrderedDict()


__all__ = [
    "BacktestConfig",
    "BacktestRunResult",
    "BacktestRunner",
    "FeatureGatePolicy",
    "RunMetadata",
    "ValidationPolicy",
]
