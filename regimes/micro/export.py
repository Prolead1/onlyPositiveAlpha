"""CSV export for micro regime predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def export_micro_regime_csv(
    df: pd.DataFrame,
    regimes: pd.Series,
    confidences: pd.Series,
    output_path: str | Path,
) -> Path:
    """Export 5-minute regime predictions to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Original 5-minute OHLCV data with 'timestamp' column.
    regimes : pd.Series
        Predicted regime labels indexed by row.
    confidences : pd.Series
        Confidence scores [0, 1] indexed by row.
    output_path : str or Path
        Output CSV file path.

    Returns
    -------
    Path
        Path to the exported CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create output DataFrame
    output_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "regime": regimes.values,
        "confidence": confidences.values,
    })

    # Ensure timestamp is string for consistent CSV format
    if pd.api.types.is_datetime64_any_dtype(output_df["timestamp"]):
        output_df["timestamp"] = output_df["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    # Save to CSV
    output_df.to_csv(output_path, index=False)

    # Log statistics
    n_records = len(output_df)
    regime_counts = output_df["regime"].value_counts()
    confidence_stats = output_df["confidence"].describe()

    logger.info(
        f"Exported {n_records} records to {output_path}"
    )
    logger.info(f"Regime distribution:\n{regime_counts}")
    logger.info(f"Confidence statistics:\n{confidence_stats}")

    return output_path
