"""Historical market regime identification using K-Means clustering.

This module identifies distinct market regimes from historical OHLCV data using
unsupervised K-Means clustering on computed features.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from regimes.config import (
    CLUSTERING_METHOD,
    N_REGIMES,
    RANDOM_STATE,
    RegimeType,
)
from regimes.feature_engineering import compute_regime_features, get_feature_matrix

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RegimeIdentifier:
    """Identify market regimes from historical data using K-Means clustering."""

    def __init__(self, n_regimes: int = N_REGIMES, random_state: int = RANDOM_STATE):
        """Initialize regime identifier.

        Parameters
        ----------
        n_regimes : int
            Number of regimes to identify (3 or 4).
        random_state : int
            Random seed for reproducibility.
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.kmeans = KMeans(
            n_clusters=n_regimes,
            random_state=random_state,
            n_init=10,
        )
        self.scaler = StandardScaler()
        self.regime_labels = {}  # Mapping: cluster_id -> RegimeType
        self.cluster_centers = None
        self.is_trained = False

    def fit(self, df: pd.DataFrame) -> dict:
        """Fit K-Means on historical data and label regimes.

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV data with columns: open, high, low, close, volume.
            Index should be datetime.

        Returns
        -------
        dict
            Summary of identified regimes.
        """
        logger.info(f"Fitting regime identifier on {len(df)} days of data")

        # Compute features
        df = compute_regime_features(df)

        # Extract feature matrix
        X = get_feature_matrix(df)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit K-Means
        self.kmeans.fit(X_scaled)
        self.cluster_centers = self.kmeans.cluster_centers_

        # Label regimes based on cluster characteristics
        self._label_regimes(df, X)

        self.is_trained = True

        # Return summary
        summary = {
            "n_samples": len(df),
            "n_regimes": self.n_regimes,
            "regime_mapping": {
                int(k): v.value for k, v in self.regime_labels.items()
            },
            "cluster_centers": self.cluster_centers.tolist(),
        }

        logger.info(f"Regime identification complete: {summary}")
        return summary

    def _label_regimes(self, df: pd.DataFrame, X: np.ndarray) -> None:
        """Assign regime labels to clusters based on characteristics.

        Logic:
        - High sentiment + high volatility = RISK_ON
        - Low sentiment + high volatility = RISK_OFF
        - Low volatility = CONSOLIDATION
        - (Optional: Remaining = TRANSITION if 4 regimes)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with computed features.
        X : np.ndarray
            Feature matrix (n_samples, 2) before scaling.
        """
        # Get cluster assignments
        clusters = self.kmeans.predict(self.scaler.transform(X))
        df["cluster"] = clusters

        # Analyze each cluster
        cluster_stats = {}
        for cluster_id in range(self.n_regimes):
            cluster_mask = df["cluster"] == cluster_id
            cluster_data = df[cluster_mask]

            cluster_stats[cluster_id] = {
                "mean_sentiment": cluster_data["sentiment_composite"].mean(),
                "mean_vol_regime": cluster_data["vol_regime_score"].mean(),
                "mean_returns": cluster_data["returns"].mean(),
                "count": cluster_mask.sum(),
            }

        logger.info("Cluster statistics:")
        for cid, stats in cluster_stats.items():
            logger.info(f"  Cluster {cid}: {stats}")

        # Assign regime labels based on characteristics
        for cluster_id, stats in cluster_stats.items():
            sentiment = stats["mean_sentiment"]
            vol_regime = stats["mean_vol_regime"]

            # Decision tree for labeling
            if vol_regime < 0.33:
                # Low volatility -> Consolidation
                regime = RegimeType.CONSOLIDATION
            elif sentiment >= 0.6 and vol_regime >= 0.5:
                # High sentiment + High volatility -> Risk-On
                regime = RegimeType.RISK_ON
            elif sentiment < 0.4 and vol_regime >= 0.5:
                # Low sentiment + High volatility -> Risk-Off
                regime = RegimeType.RISK_OFF
            else:
                # Ambiguous -> Transition (if 4 regimes)
                regime = (
                    RegimeType.TRANSITION
                    if self.n_regimes == 4
                    else RegimeType.CONSOLIDATION
                )

            self.regime_labels[cluster_id] = regime

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels for given data.

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV data.

        Returns
        -------
        pd.Series
            Regime labels (RegimeType values) for each day.
        """
        if not self.is_trained:
            raise ValueError("Regime identifier not trained. Call fit() first.")

        # Compute features
        df = compute_regime_features(df)

        # Extract feature matrix and scale
        X = get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)

        # Predict cluster and map to regime
        clusters = self.kmeans.predict(X_scaled)
        regime_preds = np.array([self.regime_labels[c].value for c in clusters])

        return pd.Series(regime_preds, index=df.index, name="regime")

    def save(self, model_dir: Path | str) -> None:
        """Save trained model to disk.

        Parameters
        ----------
        model_dir : Path | str
            Directory to save model files.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save K-Means model
        with open(model_dir / "kmeans.pkl", "wb") as f:
            pickle.dump(self.kmeans, f)

        # Save scaler
        with open(model_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save regime labels
        regime_labels_dict = {
            int(k): v.value for k, v in self.regime_labels.items()
        }
        with open(model_dir / "regime_labels.json", "w") as f:
            json.dump(regime_labels_dict, f)

        logger.info(f"Model saved to {model_dir}")

    @staticmethod
    def load(model_dir: Path | str) -> RegimeIdentifier:
        """Load trained model from disk.

        Parameters
        ----------
        model_dir : Path | str
            Directory with saved model files.

        Returns
        -------
        RegimeIdentifier
            Loaded regime identifier.
        """
        model_dir = Path(model_dir)

        identifier = RegimeIdentifier()

        # Load K-Means model
        with open(model_dir / "kmeans.pkl", "rb") as f:
            identifier.kmeans = pickle.load(f)

        # Load scaler
        with open(model_dir / "scaler.pkl", "rb") as f:
            identifier.scaler = pickle.load(f)

        # Load regime labels
        with open(model_dir / "regime_labels.json") as f:
            regime_labels_dict = json.load(f)
            identifier.regime_labels = {
                int(k): RegimeType(v) for k, v in regime_labels_dict.items()
            }

        identifier.is_trained = True
        logger.info(f"Model loaded from {model_dir}")

        return identifier
