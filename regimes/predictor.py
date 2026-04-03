"""Current market regime prediction (real-time use).

Predicts the current market regime based on recent data and provides
regime probabilities for alpha strategy optimization.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from regimes.identifier import RegimeIdentifier

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RegimePredictor:
    """Predict current market regime in real-time."""

    def __init__(self, model_dir: Path | str):
        """Initialize predictor with trained model.

        Parameters
        ----------
        model_dir : Path | str
            Directory containing trained regime model.
        """
        self.identifier = RegimeIdentifier.load(model_dir)
        logger.info("Regime predictor initialized")

    def predict_current_regime(self, recent_data: pd.DataFrame) -> dict:
        """Predict current regime from recent data.

        Parameters
        ----------
        recent_data : pd.DataFrame
            Recent OHLCV data (minimum ~30 days recommended).
            Index should be datetime.

        Returns
        -------
        dict
            Prediction result with keys:
            - regime: Current regime label
            - probabilities: Dict of regime -> probability
            - confidence: Confidence score (0-1)
            - timestamp: Prediction timestamp
        """
        # Get regime prediction for all days
        regimes = self.identifier.predict(recent_data)

        # Current regime (last day)
        current_regime = regimes.iloc[-1]

        # Simple probability: count of regime in last 10 days
        recent_regimes = regimes.tail(10)
        regime_counts = recent_regimes.value_counts(normalize=True).to_dict()

        # Compute confidence (% of recent days in current regime)
        confidence = regime_counts.get(current_regime, 0.5)

        result = {
            "regime": current_regime,
            "probabilities": regime_counts,
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Predicted regime: {current_regime} (confidence: {confidence:.2%})")
        logger.info(f"Recent regime distribution: {regime_counts}")

        return result

    def get_regime_context(self, regime_label: str) -> dict:
        """Get interpretation and context for a regime.

        Parameters
        ----------
        regime_label : str
            Regime label (e.g., 'risk-on', 'consolidation').

        Returns
        -------
        dict
            Regime characteristics and recommendations.
        """
        from regimes.config import REGIME_CHARACTERISTICS, RegimeType

        try:
            regime_type = RegimeType(regime_label)
            return REGIME_CHARACTERISTICS.get(regime_type, {})
        except ValueError:
            logger.warning(f"Unknown regime: {regime_label}")
            return {}
