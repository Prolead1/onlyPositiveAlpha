"""Market regime identification module.

This module identifies and predicts cryptocurrency market regimes
to complement alpha strategies. It operates independently from the main 
alpha generation pipeline.

Key components:
- feature_engineering: Convert OHLCV data → regime-relevant features
- identifier: Historical regime identification using K-Means clustering
- predictor: Current regime classification in real-time
- config: Regime definitions and parameters
- helpers: Helper functions
"""

from regimes.config import RegimeType, REGIME_CHARACTERISTICS
from regimes.identifier import RegimeIdentifier
from regimes.predictor import RegimePredictor
from regimes.feature_engineering import compute_regime_features, get_feature_matrix

__all__ = [
    "RegimeType",
    "REGIME_CHARACTERISTICS",
    "RegimeIdentifier",
    "RegimePredictor",
    "compute_regime_features",
    "get_feature_matrix",
]
