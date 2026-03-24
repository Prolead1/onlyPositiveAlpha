from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import MIN_PRICES_FOR_FEATURES, MIN_PRICES_FOR_VOLATILITY


@dataclass
class CryptoFeatures:
    """Crypto price features over a window."""

    # Price metrics
    price_return: float | None = None  # Log return over window
    volatility: float | None = None  # Rolling standard deviation of returns

    # Change metrics
    price_change_pct: float | None = None  # Percentage change


def compute_crypto_features(prices_df: pd.DataFrame) -> CryptoFeatures:
    """Compute crypto price features from a time series of prices.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with columns: ['timestamp', 'price'], indexed by timestamp.

    Returns
    -------
    CryptoFeatures
        Computed features.
    """
    features = CryptoFeatures()

    if len(prices_df) < MIN_PRICES_FOR_FEATURES:
        return features

    # Ensure sorted by timestamp
    df = prices_df.sort_index()

    # Compute returns
    df["return"] = np.log(df["price"] / df["price"].shift(1))

    # Price change metrics
    first_price = df["price"].iloc[0]
    last_price = df["price"].iloc[-1]
    if first_price > 0:
        features.price_change_pct = (last_price - first_price) / first_price
        if last_price > 0 and first_price > 0:
            features.price_return = np.log(last_price / first_price)

    # Volatility (rolling std of returns)
    if len(df) > MIN_PRICES_FOR_VOLATILITY:
        features.volatility = df["return"].std()

    return features


def align_features_to_events(
    market_events_df: pd.DataFrame,
    crypto_prices_df: pd.DataFrame,
    window: str = "5min",
) -> pd.DataFrame:
    """Align crypto features to market events using a rolling time window.

    Parameters
    ----------
    market_events_df : pd.DataFrame
        Market events with timestamp index.
    crypto_prices_df : pd.DataFrame
        Crypto prices with timestamp index.
    window : str
        Lookback window for feature alignment.

    Returns
    -------
    pd.DataFrame
        Market events enriched with crypto features.
    """
    # For each market event, compute crypto features over the lookback window
    enriched_rows = []

    for idx, event in market_events_df.iterrows():
        # idx is the timestamp index - treat as timestamp directly
        event_time: pd.Timestamp = idx  # type: ignore[assignment]
        lookback_start = event_time - pd.Timedelta(window)

        # Filter crypto prices in window
        window_prices = crypto_prices_df[
            (crypto_prices_df.index >= lookback_start) & (crypto_prices_df.index <= event_time)
        ]

        # Compute crypto features
        crypto_feats = compute_crypto_features(window_prices)

        # Merge
        enriched = event.to_dict()
        enriched.update(
            {
                "crypto_price_return": crypto_feats.price_return,
                "crypto_volatility": crypto_feats.volatility,
                "crypto_price_change_pct": crypto_feats.price_change_pct,
            }
        )
        enriched_rows.append(enriched)

    return pd.DataFrame(enriched_rows)
