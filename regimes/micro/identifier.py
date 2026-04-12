"""Micro regime identifier using K-Means clustering on rolling features."""

from __future__ import annotations

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
    MICRO_N_REGIMES,
    MICRO_RANDOM_STATE,
    MICRO_N_INIT,
    MICRO_FEATURE_COLS,
    RegimeType,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MicroRegimeIdentifier:
    """K-Means clustering for 5-minute intraday regime identification.

    Trains K-Means model on rolling features and assigns regime labels
    with distance-based confidence scores.
    """

    def __init__(
        self,
        n_regimes: int = MICRO_N_REGIMES,
        random_state: int = MICRO_RANDOM_STATE,
        n_init: int = MICRO_N_INIT,
        feature_cols: list[str] | None = None,
    ):
        """Initialize micro regime identifier.

        Parameters
        ----------
        n_regimes : int
            Number of regimes (typically 3).
        random_state : int
            Random seed for reproducibility.
        n_init : int
            K-Means n_init parameter.
        feature_cols : list[str] or None
            Feature columns to use for clustering.
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.n_init = n_init
        self.feature_cols = feature_cols or MICRO_FEATURE_COLS

        self.kmeans = KMeans(
            n_clusters=n_regimes,
            random_state=random_state,
            n_init=n_init,
        )
        self.scaler = StandardScaler()
        self.regime_labels = {}  # cluster_id -> RegimeType
        self.cluster_centers = None
        self.is_trained = False

    def fit(self, df: pd.DataFrame) -> dict:
        """Train K-Means on rolling features.

        Parameters
        ----------
        df : pd.DataFrame
            5-minute data with rolling features.
            Must have columns specified in self.feature_cols.

        Returns
        -------
        dict
            Training summary with cluster assignments.
        """
        logger.info(
            f"Fitting K-Means on {len(df)} 5-minute bars "
            f"using features {self.feature_cols}"
        )

        # Extract feature matrix
        X = df[self.feature_cols].values

        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        logger.info(f"Feature matrix shape: {X.shape}")

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit K-Means
        self.kmeans.fit(X_scaled)
        self.cluster_centers = self.kmeans.cluster_centers_

        # Get cluster assignments
        cluster_ids = self.kmeans.labels_

        # Label regimes based on cluster characteristics
        self._label_regimes_by_volatility(df, cluster_ids)

        self.is_trained = True

        # Summary
        summary = {
            "n_samples": len(df),
            "n_regimes": self.n_regimes,
            "cluster_inertia": self.kmeans.inertia_,
            "regime_mapping": {int(k): v.value for k, v in self.regime_labels.items()},
            "cluster_counts": {
                int(k): (cluster_ids == k).sum() for k in range(self.n_regimes)
            },
        }

        logger.info(f"Training summary: {summary}")

        return summary

    def _label_regimes_by_volatility(
        self, df: pd.DataFrame, cluster_ids: np.ndarray
    ) -> None:
        """Label clusters based on volatility and momentum characteristics.

        Heuristic: 
        - High vol + High momentum → risk_on
        - Low vol → consolidation
        - High vol + Low momentum → risk_off

        Parameters
        ----------
        df : pd.DataFrame
            5-minute data with features.
        cluster_ids : np.ndarray
            Cluster assignments from K-Means.
        """
        df = df.copy()
        df["_cluster"] = cluster_ids

        cluster_chars = {}
        for cluster_id in range(self.n_regimes):
            cluster_data = df[df["_cluster"] == cluster_id]
            chars = {
                "avg_volatility": cluster_data.get("rolling_volatility", pd.Series()).mean(),
                "avg_momentum": cluster_data.get("rolling_momentum", pd.Series()).mean(),
                "count": len(cluster_data),
            }
            cluster_chars[cluster_id] = chars

        # Sort clusters by volatility: low, medium, high
        sorted_by_vol = sorted(
            cluster_chars.items(),
            key=lambda x: x[1]["avg_volatility"]
        )

        # Assign labels
        # Assume lowest vol = consolidation
        low_vol_cluster = sorted_by_vol[0][0]
        high_vol_cluster_data = {
            k: v for k, v in cluster_chars.items() if k != low_vol_cluster
        }

        if high_vol_cluster_data:
            # Among high-vol clusters, separate by momentum
            by_momentum = sorted(
                high_vol_cluster_data.items(),
                key=lambda x: x[1]["avg_momentum"]
            )
            risk_off_cluster = by_momentum[0][0]
            risk_on_cluster = by_momentum[-1][0]
        else:
            risk_on_cluster = 0 if 0 != low_vol_cluster else 1
            risk_off_cluster = 1 if 1 != low_vol_cluster else 2

        self.regime_labels = {
            risk_on_cluster: RegimeType.RISK_ON,
            low_vol_cluster: RegimeType.CONSOLIDATION,
            risk_off_cluster: RegimeType.RISK_OFF,
        }

        logger.info(f"Regime labels: {self.regime_labels}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels.

        Parameters
        ----------
        df : pd.DataFrame
            5-minute data with rolling features.

        Returns
        -------
        pd.Series
            Regime labels indexed by row.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        X = df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        X_scaled = self.scaler.transform(X)

        cluster_ids = self.kmeans.predict(X_scaled)

        regimes = pd.Series(
            [self.regime_labels[cid].value for cid in cluster_ids],
            index=df.index,
        )

        return regimes

    def predict_with_confidence(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Predict regime labels and confidence scores.

        Confidence = softmax over distances to all cluster centroids.
        Higher confidence = closer to centroid.

        Parameters
        ----------
        df : pd.DataFrame
            5-minute data with rolling features.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            (regimes, confidences) both indexed by row.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        X = df[self.feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        X_scaled = self.scaler.transform(X)

        # Get cluster assignments
        cluster_ids = self.kmeans.predict(X_scaled)

        # Compute distances to all centroids
        distances = self.kmeans.transform(X_scaled)

        # Confidence: softmax over negative distances
        # (closer = lower distance = higher confidence)
        exp_neg_distances = np.exp(-distances / distances.var(axis=0, keepdims=True))
        confidences = exp_neg_distances / exp_neg_distances.sum(axis=1, keepdims=True)

        # Get confidence for assigned cluster
        assigned_confidences = np.array(
            [confidences[i, cluster_ids[i]] for i in range(len(cluster_ids))]
        )

        # Normalize to [0, 1]
        assigned_confidences = (assigned_confidences - assigned_confidences.min()) / (
            assigned_confidences.max() - assigned_confidences.min() + 1e-8
        )

        # Create output series
        regimes = pd.Series(
            [self.regime_labels[cid].value for cid in cluster_ids],
            index=df.index,
        )

        confidences_series = pd.Series(assigned_confidences, index=df.index)

        return regimes, confidences_series

    def save(self, dirpath: str | Path) -> None:
        """Save trained model artifacts.

        Parameters
        ----------
        dirpath : str or Path
            Directory to save model files.
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # Save K-Means
        kmeans_path = dirpath / "kmeans.pkl"
        joblib.dump(self.kmeans, kmeans_path)

        # Save StandardScaler
        scaler_path = dirpath / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)

        # Save regime labels
        labels_path = dirpath / "regime_labels.pkl"
        with open(labels_path, "wb") as f:
            pickle.dump(self.regime_labels, f)

        # Save config
        config_path = dirpath / "config.pkl"
        config = {
            "n_regimes": self.n_regimes,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "feature_cols": self.feature_cols,
            "is_trained": self.is_trained,
        }
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

        logger.info(f"Model saved to {dirpath}")

    @classmethod
    def load(cls, dirpath: str | Path) -> MicroRegimeIdentifier:
        """Load trained model artifacts.

        Parameters
        ----------
        dirpath : str or Path
            Directory containing model files.

        Returns
        -------
        MicroRegimeIdentifier
            Loaded model ready for inference.
        """
        dirpath = Path(dirpath)

        # Load config
        config_path = dirpath / "config.pkl"
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        # Create instance
        identifier = cls(
            n_regimes=config["n_regimes"],
            random_state=config["random_state"],
            n_init=config["n_init"],
            feature_cols=config["feature_cols"],
        )

        # Load K-Means
        kmeans_path = dirpath / "kmeans.pkl"
        identifier.kmeans = joblib.load(kmeans_path)

        # Load StandardScaler
        scaler_path = dirpath / "scaler.pkl"
        identifier.scaler = joblib.load(scaler_path)

        # Load regime labels
        labels_path = dirpath / "regime_labels.pkl"
        with open(labels_path, "rb") as f:
            identifier.regime_labels = pickle.load(f)

        identifier.cluster_centers = identifier.kmeans.cluster_centers_
        identifier.is_trained = True

        logger.info(f"Model loaded from {dirpath}")

        return identifier
