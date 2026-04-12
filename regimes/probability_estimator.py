"""Regime probability estimation and current regime probability scoring.

PHASE 4: TODO - Implement probability weighting and regime likelihood estimation.
For now, provides simple historical frequency-based probabilities.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def estimate_regime_probabilities(regime_series: pd.Series) -> dict:
    """Estimate stable probability distribution of regimes.

    Simple approach: Use historical frequency of each regime.
    TODO: Replace with more sophisticated Bayesian approach if needed.

    Parameters
    ----------
    regime_series : pd.Series
        Historical regime labels (RegimeType values).

    Returns
    -------
    dict
        Regime -> probability mapping.
    """
    # Simple approach: historical frequency
    return regime_series.value_counts(normalize=True).to_dict()
