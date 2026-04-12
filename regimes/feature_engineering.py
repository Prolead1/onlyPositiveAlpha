"""Feature engineering for regime identification.

Compute volume/volatility regime and sentiment composite scores from OHLCV and orderbook data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from regimes.config import (
    MOMENTUM_MA_WINDOW,
    MOMENTUM_WINDOW,
    SENTIMENT_BEARISH_THRESHOLD,
    SENTIMENT_BULLISH_THRESHOLD,
    VOL_HIGH_THRESHOLD,
    VOL_LOW_THRESHOLD,
    VOL_WINDOW,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute price-based features: returns, volatility, momentum.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'close' and 'volume' columns, daily data.
        Index should be datetime.

    Returns
    -------
    pd.DataFrame
        Original df with added columns:
        - 'returns': daily returns
        - 'volatility': rolling volatility (VOL_WINDOW days)
        - 'momentum': momentum signal (-1, 0, 1)
        - 'price_trend': price above/below MA
    """
    df = df.copy()

    # Returns
    df["returns"] = df["close"].pct_change()

    # Volatility (rolling standard deviation of returns)
    df["volatility"] = df["returns"].rolling(window=VOL_WINDOW).std()

    # Momentum: 5-day vs 20-day MA
    df["ma_short"] = df["close"].rolling(window=MOMENTUM_WINDOW).mean()
    df["ma_long"] = df["close"].rolling(window=MOMENTUM_MA_WINDOW).mean()
    df["momentum"] = np.sign(df["ma_short"] - df["ma_long"])  # -1, 0, 1

    # Price trend: price vs long-term MA
    df["price_above_ma"] = (df["close"] > df["ma_long"]).astype(int)

    return df


def compute_sentiment_composite(df: pd.DataFrame) -> pd.Series:
    """Compute sentiment composite score (0 to 1).

    Simple composite: Blend of technical momentum + price trend + returns.
    1.0 = Fully bullish, 0.0 = Fully bearish, 0.5 = Neutral.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with computed features (returns, momentum, price_above_ma).

    Returns
    -------
    pd.Series
        Sentiment score [0, 1] with dtype float.
    """
    # Normalize components to 0-1 scale
    momentum_score = (df["momentum"] + 1) / 2  # -1 -> 0, 0 -> 0.5, 1 -> 1

    price_score = df["price_above_ma"].astype(float)  # 0 or 1

    # Recent returns (positive = bullish)
    returns_norm = df["returns"].rolling(window=5).mean()
    returns_score = (np.clip(returns_norm, -0.05, 0.05) + 0.05) / 0.10  # Normalize
    returns_score = np.clip(returns_score, 0, 1)

    # Composite: Equal weight blend
    sentiment = (
        0.33 * momentum_score + 0.33 * price_score + 0.34 * returns_score
    )

    return sentiment.fillna(0.5)  # Assume neutral if NaN


def compute_volume_volatility_regime(df: pd.DataFrame) -> pd.Series:
    """Compute volume/volatility regime score (0=low vol, 1=high vol).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'volatility' and 'volume' columns.

    Returns
    -------
    pd.Series
        Regime score [0, 1] where 1 = high volatility.
    """
    # Normalize volatility to 0-1 (using percentile approach)
    vol_percentile = df["volatility"].rank(pct=True)
    return vol_percentile.fillna(0.5)


def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering pipeline.

    Compute all regime-relevant features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV data with columns: open, high, low, close, volume.
        Index should be datetime.

    Returns
    -------
    pd.DataFrame
        Original df with added feature columns:
        - returns, volatility, momentum, price_above_ma
        - sentiment_composite
        - vol_regime_score
    """
    logger.info(f"Computing regime features for {len(df)} days of data")

    # Phase 1: Price features
    df = compute_price_features(df)

    # Phase 2: Sentiment composite
    df["sentiment_composite"] = compute_sentiment_composite(df)

    # Phase 3: Volume/volatility regime
    df["vol_regime_score"] = compute_volume_volatility_regime(df)

    # Feature matrix for clustering: use sentiment + vol_regime
    feature_cols = ["sentiment_composite", "vol_regime_score"]

    logger.info(f"Computed features: {feature_cols}")
    logger.debug(f"\nFeature statistics:\n{df[feature_cols].describe()}")

    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with computed features.

    Returns
    -------
    np.ndarray
        Feature matrix (n_samples, 2) with [sentiment_composite, vol_regime_score].
        NaN values filled with 0.5 (neutral).
    """
    features = df[["sentiment_composite", "vol_regime_score"]].fillna(0.5)
    return features.values
