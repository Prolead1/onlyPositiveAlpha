"""Utility functions for regime analysis and data handling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def validate_ohlcv_data(df: pd.DataFrame) -> bool:
    """Validate that DataFrame has required OHLCV columns.

    Parameters
    ----------
    df : pd.DataFrame
        Data to validate.

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise.
    """
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Index is not DatetimeIndex. Converting...")
        df.index = pd.to_datetime(df.index)

    return True


def ensure_daily_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure data is daily frequency. Fill gaps if necessary.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.

    Returns
    -------
    pd.DataFrame
        Reindexed to daily frequency with NaN filled using ffill.
    """
    df = df.asfreq("D", method="ffill")
    return df


def get_regime_model_dir() -> Path:
    """Get directory for storing regime models.

    Returns
    -------
    Path
        Path to models directory.
    """
    from pathlib import Path

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    return models_dir


def save_regime_backtest_results(
    results_df: pd.DataFrame,
    output_dir: Path | str,
) -> None:
    """Save regime backtest results to CSV.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns: date, regime, returns, volatility, etc.
    output_dir : Path | str
        Directory to save results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file = output_dir / "regime_backtest_results.csv"
    results_df.to_csv(output_file)
    logger.info(f"Results saved to {output_file}")

