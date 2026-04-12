from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "output" / "cache" / "mpl"),
)

from EDA.core.config import build_config
from EDA.core.io import (
    build_file_inventory,
    build_schema_summary,
    discover_batches,
    select_batches,
)
from EDA.core.utils import ensure_directories, get_logger, setup_logging
from EDA.pipeline.analysis import build_analysis_outputs
from EDA.pipeline.features import build_hourly_feature_caches, build_market_dimension
from EDA.render.plots import generate_figures
from reporting.organic import write_eda_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Polymarket crypto up/down EDA.")
    parser.add_argument(
        "--mode",
        choices=("sample", "full"),
        default="full",
        help=(
            "Execution mode. Defaults to full; sample uses a small bounded hour subset."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = args.mode
    config = build_config(mode=mode)  # type: ignore[arg-type]
    setup_logging()
    logger = get_logger()

    ensure_directories(
        [
            config.paths.output_root,
            config.paths.figures_root,
            config.paths.cache_root,
            config.paths.mode_cache_dir(config.mode),
            config.paths.mode_figure_dir(config.mode),
        ],
    )

    logger.info("Building file inventory for mode=%s", config.mode)
    inventory = build_file_inventory(config)
    schema_summary = build_schema_summary(config)
    all_batches = discover_batches(inventory)
    batches = select_batches(config, all_batches)
    if not batches:
        msg = (
            "No shared hourly batches were found across mapping, book_snapshot, and price_change."
        )
        raise RuntimeError(msg)

    logger.info("Selected %s / %s hourly batches", len(batches), len(all_batches))
    market_dimension = build_market_dimension(config, batches)
    logger.info("Market dimension rows: %s", market_dimension.height)

    artifacts = build_hourly_feature_caches(config, batches, inventory, market_dimension)
    outputs = build_analysis_outputs(
        config=config,
        inventory=inventory,
        schema_summary=schema_summary,
        batches=batches,
        market_dimension=market_dimension,
        artifacts=artifacts,
    )
    figure_paths = generate_figures(config, outputs)
    report_path = write_eda_report(config, outputs, figure_paths)

    ranking_table = (
        outputs["ranking_table"]
        .head(5)
        .select(
            ["coin", "resolution", "research_score"],
        )
    )
    logger.info("Top research candidates:\n%s", ranking_table)
    logger.info("Insights report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
