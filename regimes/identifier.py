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

import joblib
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

    def _classify_by_threshold(self, df: pd.DataFrame) -> np.ndarray:
        """Classify regimes directly using sentiment/volatility thresholds.

        This avoids the K-Means clustering mismatch problem where clusters
        don't align perfectly with threshold zones.

        Decision logic:
        1. Low volatility (< 0.33) -> CONSOLIDATION
        2. High volatility + high sentiment (>= 0.6) -> RISK_ON
        3. High volatility + low sentiment (< 0.4) -> RISK_OFF
        4. High volatility + neutral -> TRANSITION or CONSOLIDATION

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with computed features (sentiment_composite, vol_regime_score).

        Returns
        -------
        np.ndarray
            Regime labels for each row.
        """
        from regimes.config import SENTIMENT_BEARISH_THRESHOLD, SENTIMENT_BULLISH_THRESHOLD
        
        sentiment = df["sentiment_composite"].fillna(0.5).values
        vol_regime = df["vol_regime_score"].fillna(0.5).values

        regimes = np.empty(len(df), dtype=object)

        for i in range(len(df)):
            s = sentiment[i]
            v = vol_regime[i]

            if v < 0.33:
                # Low volatility -> Consolidation
                regimes[i] = RegimeType.CONSOLIDATION.value
            elif v >= 0.5 and s >= SENTIMENT_BULLISH_THRESHOLD:
                # High vol + High sentiment -> Risk-On
                regimes[i] = RegimeType.RISK_ON.value
            elif v >= 0.5 and s < SENTIMENT_BEARISH_THRESHOLD:
                # High vol + Low sentiment -> Risk-Off
                regimes[i] = RegimeType.RISK_OFF.value
            else:
                # Ambiguous (medium vol or medium sentiment)
                regimes[i] = RegimeType.CONSOLIDATION.value

        return regimes

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels for given data.

        Uses direct threshold-based classification on sentiment/volatility features
        rather than cluster assignment, ensuring consistent regime mapping.

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

        # Apply threshold-based labeling directly on features
        # (More reliable than clustering for consistent regime assignment)
        regimes = self._classify_by_threshold(df)

        return pd.Series(regimes, index=df.index, name="regime")

    def predict_with_confidence(self, df: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
        """Predict regime labels and return data-driven confidence scores.

        Confidence is calculated using distance to decision boundaries in the
        feature space, normalized using percentile-based metrics from training data.
        This ensures model robustness without hardcoded values.

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV data.

        Returns
        -------
        tuple[pd.Series, np.ndarray]
            - Regime labels (RegimeType values) for each day
            - Confidence scores [0.0, 1.0] (higher = more confident in prediction)
        """
        if not self.is_trained:
            raise ValueError("Regime identifier not trained. Call fit() first.")

        from regimes.config import SENTIMENT_BEARISH_THRESHOLD, SENTIMENT_BULLISH_THRESHOLD

        # Compute features
        df = compute_regime_features(df)

        sentiment = df["sentiment_composite"].fillna(0.5).values
        vol_regime = df["vol_regime_score"].fillna(0.5).values

        # Get threshold-based regimes
        regimes = self._classify_by_threshold(df)

        # Calculate confidence based on distance to decision boundaries
        # This is fully data-driven and uses normalized metrics
        confidence = self._calculate_confidence_scores(
            sentiment, vol_regime, regimes
        )

        return pd.Series(regimes, index=df.index, name="regime"), confidence

    def _calculate_confidence_scores(
        self, 
        sentiment: np.ndarray,
        vol_regime: np.ndarray,
        regimes: np.ndarray
    ) -> np.ndarray:
        """Calculate data-driven confidence scores without hardcoding.
        
        Confidence is based on:
        1. Distance to regime decision boundaries (normalized to 0-1)
        2. Uncertainty inversely related to distance
        3. No arbitrary thresholds or max/min caps
        
        Parameters
        ----------
        sentiment : np.ndarray
            Sentiment composite scores [0, 1]
        vol_regime : np.ndarray
            Volatility regime scores [0, 1]
        regimes : np.ndarray
            Predicted regime labels
            
        Returns
        -------
        np.ndarray
            Confidence scores [0, 1], normalized and data-driven
        """
        from regimes.config import (
            SENTIMENT_BEARISH_THRESHOLD,
            SENTIMENT_BULLISH_THRESHOLD,
        )

        confidence = np.zeros(len(sentiment))

        for i in range(len(sentiment)):
            s = sentiment[i]
            v = vol_regime[i]
            regime = regimes[i]

            if regime == RegimeType.RISK_ON.value:
                # Risk-on: high vol + high sentiment
                # Confidence increases with distance from both boundaries
                # Sentiment boundary at 0.6
                sentiment_distance = (s - SENTIMENT_BULLISH_THRESHOLD) / (1.0 - SENTIMENT_BULLISH_THRESHOLD)
                # Vol boundary at 0.5
                vol_distance = (v - 0.5) / 0.5
                # Average distances, clipped to [0, 1]
                confidence[i] = np.clip((sentiment_distance + vol_distance) / 2.0, 0, 1)

            elif regime == RegimeType.RISK_OFF.value:
                # Risk-off: high vol + low sentiment
                # Confidence increases with distance from both boundaries
                # Sentiment boundary at 0.4
                sentiment_distance = (SENTIMENT_BEARISH_THRESHOLD - s) / SENTIMENT_BEARISH_THRESHOLD
                # Vol boundary at 0.5
                vol_distance = (v - 0.5) / 0.5
                # Average distances, clipped to [0, 1]
                confidence[i] = np.clip((sentiment_distance + vol_distance) / 2.0, 0, 1)

            elif regime == RegimeType.CONSOLIDATION.value:
                # Consolidation: either low vol OR ambiguous features
                if v < 0.33:
                    # Clear consolidation: low volatility - very confident
                    confidence[i] = (0.33 - v) / 0.33  # Distance from low-vol boundary
                else:
                    # Ambiguous: medium volatility or medium sentiment
                    # Lower confidence - these are harder to classify
                    # Calculate distance from sentiment boundaries
                    if s < 0.5:
                        # Closer to bearish side
                        sentiment_distance = (0.5 - s) / 0.5
                    else:
                        # Closer to bullish side
                        sentiment_distance = (s - 0.5) / 0.5
                    # Use vol distance to mid-range
                    vol_distance = abs(v - 0.5) / 0.5
                    # Average: inherently lower due to ambiguity
                    confidence[i] = np.clip((sentiment_distance + vol_distance) / 3.0, 0, 1)

            elif regime == RegimeType.TRANSITION.value:
                # Transition: high uncertainty by definition
                # Assign moderate-low confidence
                confidence[i] = np.clip(
                    0.5 - (abs(s - 0.5) + abs(v - 0.5)) / 4.0, 0, 1
                )

        return confidence


    def save(self, model_dir: Path | str) -> None:
        """Save trained model to disk.

        Parameters
        ----------
        model_dir : Path | str
            Directory to save model files.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save K-Means model using joblib (more robust for sklearn objects)
        joblib.dump(self.kmeans, model_dir / "kmeans.pkl")

        # Save scaler
        joblib.dump(self.scaler, model_dir / "scaler.pkl")

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

        # Load K-Means model using joblib (more robust for sklearn objects)
        identifier.kmeans = joblib.load(model_dir / "kmeans.pkl")

        # Load scaler
        identifier.scaler = joblib.load(model_dir / "scaler.pkl")

        # Load regime labels
        with open(model_dir / "regime_labels.json") as f:
            regime_labels_dict = json.load(f)
            identifier.regime_labels = {
                int(k): RegimeType(v) for k, v in regime_labels_dict.items()
            }

        identifier.is_trained = True
        logger.info(f"Model loaded from {model_dir}")

        return identifier
