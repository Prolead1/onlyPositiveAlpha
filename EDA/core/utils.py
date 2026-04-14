from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path

LOGGER_NAME = "EDA"
ZERO_TOLERANCE = 1.0e-12


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging() -> logging.Logger:
    logger = get_logger()
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"),
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def safe_ratio(numerator: pl.Expr, denominator: pl.Expr, alias: str) -> pl.Expr:
    return (
        pl.when(denominator.abs() > 0).then(numerator / denominator).otherwise(None).alias(alias)
    )


def safe_scalar_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= ZERO_TOLERANCE:
        return None
    return numerator / denominator


def ordered_resolutions(
    observed_resolutions: list[str],
    preferred_order: tuple[str, ...],
) -> list[str]:
    observed = set(observed_resolutions)
    extras = sorted(observed.difference(preferred_order))
    ordered = [resolution for resolution in preferred_order if resolution in observed]
    ordered.extend(extras)
    return ordered


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def format_number(value: object, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:,.{digits}f}"
    return str(value)


def dataframe_to_markdown(frame: pl.DataFrame, digits: int = 3) -> str:
    headers = frame.columns
    rows = [[format_number(value, digits=digits) for value in row] for row in frame.iter_rows()]
    if not rows:
        return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)
