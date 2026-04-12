"""Feature engineering for micro regime classification.

Compute rolling features (volatility, momentum, returns) from 5-minute OHLCV data.

CRITICAL: All features are computed with expanding windows to ensure ZERO look-ahead bias.
Each timestamp t uses only data from [t-window, t], never touching future data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from regimes.config import MICRO_ROLLING_WINDOW_SAMPLES

logger = logging.getLogger(__name__)


def compute_rolling_returns(
    close_series: pd.Series,
    window_samples: int = MICRO_ROLLING_WINDOW_SAMPLES,
) -> pd.Series:
    """Compute rolling mean returns (no look-ahead bias).

    Each timestamp t uses data from [t-window, t] only.

    Parameters
    ----------
    close_series : pd.Series
        Close prices indexed by position.
    window_samples : int
        Number of samples in rolling window (default 6 = 30 min).

    Returns
    -------
    pd.Series
        Rolling mean log returns.
    """
    # Compute returns first
    returns = close_series.pct_change()

    # Rolling mean with min_periods=1 for early rows
    rolling_mean_returns = returns.rolling(
        window=window_samples, min_periods=1
    ).mean()

    return rolling_mean_returns.fillna(0.0)


def compute_rolling_volatility(
    close_series: pd.Series,
    window_samples: int = MICRO_ROLLING_WINDOW_SAMPLES,
) -> pd.Series:
    """Compute rolling volatility (standard deviation of returns).

    Each timestamp t uses data from [t-window, t] only.

    Parameters
    ----------
    close_series : pd.Series
        Close prices indexed by position.
    window_samples : int
        Number of samples in rolling window.

    Returns
    -------
    pd.Series
        Rolling volatility (std of returns).
    """
    returns = close_series.pct_change()
    rolling_vol = returns.rolling(
        window=window_samples, min_periods=1
    ).std()

    return rolling_vol.fillna(0.0)


def compute_rolling_momentum(
    close_series: pd.Series,
    short_window: int = MICRO_ROLLING_WINDOW_SAMPLES // 2,
    long_window: int = MICRO_ROLLING_WINDOW_SAMPLES,
) -> pd.Series:
    """Compute rolling momentum as short MA vs long MA.

    Momentum = (MA_short - MA_long) / MA_long
    Each timestamp t uses data from [t-window, t] only.

    Parameters
    ----------
    close_series : pd.Series
        Close prices indexed by position.
    short_window : int
        Short window samples (default 3 = 15 min).
    long_window : int
        Long window samples (default 6 = 30 min).

    Returns
    -------
    pd.Series
        Rolling momentum signal [-1, 1].
    """
    ma_short = close_series.rolling(window=short_window, min_periods=1).mean()
    ma_long = close_series.rolling(window=long_window, min_periods=1).mean()

    # Momentum: difference normalized by long MA
    momentum = (ma_short - ma_long) / (ma_long + 1e-8)

    # Normalize to roughly [-1, 1]
    momentum = np.tanh(momentum * 10)

    return momentum.fillna(0.0)


def compute_rolling_volume_stats(
    volume_series: pd.Series,
    window_samples: int = MICRO_ROLLING_WINDOW_SAMPLES,
) -> pd.DataFrame:
    """Compute rolling volume statistics.

    Each timestamp t uses data from [t-window, t] only.

    Parameters
    ----------
    volume_series : pd.Series
        Volume series indexed by position.
    window_samples : int
        Number of samples in rolling window.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - rolling_volume_mean
        - rolling_volume_std
    """
    result = pd.DataFrame(index=volume_series.index)

    result["rolling_volume_mean"] = volume_series.rolling(
        window=window_samples, min_periods=1
    ).mean()

    result["rolling_volume_std"] = volume_series.rolling(
        window=window_samples, min_periods=1
    ).std()

    return result.fillna(0.0)


def compute_micro_rolling_features(
    df: pd.DataFrame,
    window_samples: int = MICRO_ROLLING_WINDOW_SAMPLES,
) -> pd.DataFrame:
    """Compute all rolling features for micro regime classification.

    Main pipeline: Computes rolling features with zero look-ahead bias.

    Input:  df with ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    Output: df with added rolling feature columns:
            - rolling_returns
            - rolling_volatility
            - rolling_momentum
            - rolling_volume_mean
            - rolling_volume_std

    CRITICAL: Each feature at row t uses only data from [t-window, t].

    Parameters
    ----------
    df : pd.DataFrame
        5-minute OHLCV data.
    window_samples : int
        Rolling window size (default 6 = 30 min).

    Returns
    -------
    pd.DataFrame
        Original df with added feature columns.
    """
    df = df.copy()

    logger.info(
        f"Computing rolling features (window={window_samples} × 5-min = "
        f"{window_samples * 5} minutes) for {len(df)} bars"
    )

    # Rolling features from close price
    df["rolling_returns"] = compute_rolling_returns(df["close"], window_samples)
    df["rolling_volatility"] = compute_rolling_volatility(df["close"], window_samples)
    df["rolling_momentum"] = compute_rolling_momentum(
        df["close"],
        short_window=window_samples // 2,
        long_window=window_samples,
    )

    # Volume statistics
    vol_stats = compute_rolling_volume_stats(df["volume"], window_samples)
    df = pd.concat([df, vol_stats], axis=1)

    # Verify no NaN in key features
    n_missing = df[["rolling_volatility", "rolling_momentum"]].isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"Found {n_missing} NaN values in rolling features")

    logger.info(f"Added rolling features to {len(df)} bars")

    return df


def validate_no_lookahead(
    df: pd.DataFrame,
    window_samples: int = MICRO_ROLLING_WINDOW_SAMPLES,
    sample_rows: int = 10,
) -> bool:
    """Validate that no look-ahead bias exists in rolling features.

    Spot-checks a few rows to ensure features use only past data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with rolling features.
    window_samples : int
        Expected rolling window size.
    sample_rows : int
        Number of rows to spot-check.

    Returns
    -------
    bool
        True if no look-ahead detected, False otherwise.
    """
    logger.info(f"Validating no look-ahead bias (sampling {sample_rows} rows)...")

    n_rows = len(df)
    sampled_indices = np.linspace(
        window_samples, n_rows - 1, sample_rows, dtype=int
    )

    for idx in sampled_indices:
        # Recompute rolling_volatility for this row using data [idx-window, idx]
        window_start = max(0, idx - window_samples)
        window_data = df.iloc[window_start : idx + 1]

        actual_vol = df.iloc[idx]["rolling_volatility"]
        expected_vol = window_data["close"].pct_change().std()

        if not np.isclose(actual_vol, expected_vol, rtol=0.01):
            logger.error(
                f"Row {idx}: actual_vol={actual_vol:.6f}, "
                f"expected_vol={expected_vol:.6f} → LOOK-AHEAD BIAS DETECTED"
            )
            return False

    logger.info("✓ No look-ahead bias detected")
    return True
