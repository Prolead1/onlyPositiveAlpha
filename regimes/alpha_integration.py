"""
Integration of regime predictions with alpha strategies.

Combines market regime identification with imbalance and spread-based alphas
to produce regime-aware trading signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from regimes.alpha_params import get_alpha_params
from regimes.predictor import RegimePredictor

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AlphaSignal:
    """Single alpha signal with metadata."""
    
    signal_value: float        # [-1, 1]: -1=sell, 0=neutral, 1=buy
    signal_name: str
    strength: float           # [0, 1]: confidence in signal
    metadata: dict[str, Any]


@dataclass
class RegimeAwareAlphaResult:
    """Final output: regime-adjusted alpha signals."""
    
    # Regime
    regime: str
    regime_confidence: float
    regime_probabilities: dict[str, float]
    
    # Raw alphas (not adjusted)
    imbalance_signal: AlphaSignal
    spread_signal: AlphaSignal
    
    # Regime-adjusted alphas
    adjusted_imbalance_signal: AlphaSignal
    adjusted_spread_signal: AlphaSignal
    
    # Combined signal
    combined_signal: float      # [-1, 1]
    combined_confidence: float  # [0, 1]
    
    # Recommended parameters
    alpha_params: dict[str, Any]
    
    # Timestamps
    timestamp: pd.Timestamp


def compute_imbalance_alpha(
    imbalance_1: float,
    bid_depth_1: float,
    ask_depth_1: float,
    threshold: float = 0.05,
) -> AlphaSignal:
    """
    Compute imbalance-based alpha signal.
    
    Detects when one side of the book significantly outnumbers the other.
    
    Parameters
    ----------
    imbalance_1 : float
        Bid/ask imbalance at level 1: (bid_size - ask_size) / (bid_size + ask_size)
    bid_depth_1 : float
        Total bid size at level 1
    ask_depth_1 : float
        Total ask size at level 1
    threshold : float
        Threshold for strong signal (default 5%)
    
    Returns
    -------
    AlphaSignal
        Signal value, strength, and metadata
    """
    # Normalize imbalance to [-1, 1]
    signal_value = imbalance_1 * 2  # Scale to [-2, 2] then clip
    signal_value = max(-1, min(1, signal_value))
    
    # Signal strength: how extreme is the imbalance?
    abs_imbalance = abs(imbalance_1)
    strength = max(0, (abs_imbalance - threshold) / (1 - threshold))  # [0, 1]
    
    # Direction
    if imbalance_1 > threshold:
        direction = "buy"
    elif imbalance_1 < -threshold:
        direction = "sell"
    else:
        direction = "neutral"
    
    return AlphaSignal(
        signal_value=signal_value,
        signal_name="imbalance",
        strength=strength,
        metadata={
            "imbalance_1": imbalance_1,
            "bid_depth_1": bid_depth_1,
            "ask_depth_1": ask_depth_1,
            "direction": direction,
            "threshold": threshold,
        },
    )


def compute_spread_alpha(
    spread: float,
    p50_spread: float,
    p75_spread: float,
) -> AlphaSignal:
    """
    Compute spread-based alpha signal.
    
    Tight spreads = good conditions (buy signal).
    Wide spreads = poor conditions (sell signal).
    
    Parameters
    ----------
    spread : float
        Current bid-ask spread
    p50_spread : float
        Median spread (50th percentile)
    p75_spread : float
        75th percentile spread
    
    Returns
    -------
    AlphaSignal
        Signal value, strength, and metadata
    """
    # Normalize spread: tight=1, wide=-1
    if spread < p50_spread:
        # Tight spread (good)
        signal_value = 1.0
        strength = (p50_spread - spread) / p50_spread
    elif spread > p75_spread:
        # Wide spread (bad)
        signal_value = -1.0
        strength = (spread - p75_spread) / p75_spread
    else:
        # Medium spread
        signal_value = 0.0
        strength = 0.0
    
    strength = max(0, min(1, strength))  # Clip to [0, 1]
    
    direction = "buy" if signal_value > 0 else ("sell" if signal_value < 0 else "neutral")
    
    return AlphaSignal(
        signal_value=signal_value,
        signal_name="spread",
        strength=strength,
        metadata={
            "spread": spread,
            "p50_spread": p50_spread,
            "p75_spread": p75_spread,
            "direction": direction,
        },
    )


def adjust_alpha_for_regime(
    alpha_signal: AlphaSignal,
    regime_params: dict[str, Any],
) -> AlphaSignal:
    """
    Adjust alpha signal based on regime parameters.
    
    In Risk-On: amplify signals (1.5x)
    In Consolidation: use normal signals (1.0x)
    In Risk-Off: dampen signals (0.5x)
    """
    param_key = f"{alpha_signal.signal_name}_sensitivity"
    sensitivity = regime_params.get(param_key, 1.0)
    
    adjusted_value = alpha_signal.signal_value * sensitivity
    adjusted_value = max(-1, min(1, adjusted_value))  # Clip to [-1, 1]
    
    adjusted_strength = alpha_signal.strength * sensitivity
    adjusted_strength = max(0, min(1, adjusted_strength))
    
    return AlphaSignal(
        signal_value=adjusted_value,
        signal_name=f"{alpha_signal.signal_name}_adjusted",
        strength=adjusted_strength,
        metadata={
            **alpha_signal.metadata,
            "sensitivity": sensitivity,
            "original_value": alpha_signal.signal_value,
            "original_strength": alpha_signal.strength,
        },
    )


def blend_alpha_signals(
    adjusted_imbalance: AlphaSignal,
    adjusted_spread: AlphaSignal,
    weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    """
    Blend adjusted alpha signals into combined signal.
    
    Parameters
    ----------
    adjusted_imbalance : AlphaSignal
        Regime-adjusted imbalance signal
    adjusted_spread : AlphaSignal
        Regime-adjusted spread signal
    weights : dict or None
        Weights for each signal (default equal weight 0.5/0.5)
    
    Returns
    -------
    tuple[float, float]
        (combined_signal, combined_confidence)
    """
    if weights is None:
        weights = {"imbalance": 0.5, "spread": 0.5}
    
    # Weighted average of signal values
    combined_value = (
        adjusted_imbalance.signal_value * weights.get("imbalance", 0.5) +
        adjusted_spread.signal_value * weights.get("spread", 0.5)
    )
    combined_value = max(-1, min(1, combined_value))
    
    # Weighted average of confidence
    combined_confidence = (
        adjusted_imbalance.strength * weights.get("imbalance", 0.5) +
        adjusted_spread.strength * weights.get("spread", 0.5)
    )
    combined_confidence = max(0, min(1, combined_confidence))
    
    return combined_value, combined_confidence


def integrate_regimes_with_alphas(
    recent_ohlcv: pd.DataFrame,
    current_orderbook_features: dict[str, float],
    regime_model_dir: str | Path,
    alpha_signal_weights: dict[str, float] | None = None,
) -> RegimeAwareAlphaResult:
    """
    Integrate regime predictions with alpha strategies.
    
    End-to-end pipeline:
    1. Predict current market regime
    2. Compute raw alpha signals (imbalance, spread)
    3. Load regime-specific parameters
    4. Adjust alpha signals based on regime
    5. Blend into combined signal
    6. Return with recommended parameters
    
    Parameters
    ----------
    recent_ohlcv : pd.DataFrame
        Recent OHLCV data (for regime prediction)
    current_orderbook_features : dict
        Current orderbook features dict with:
        - imbalance_1, bid_depth_1, ask_depth_1
        - spread, p50_spread, p75_spread
    regime_model_dir : Path or str
        Directory with trained regime model
    alpha_signal_weights : dict or None
        Weights for imbalance and spread alphas
    
    Returns
    -------
    RegimeAwareAlphaResult
        Complete analysis with regime, raw alphas, adjusted alphas, and parameters
    """
    # Step 1: Predict regime
    predictor = RegimePredictor(regime_model_dir)
    regime_result = predictor.predict_current_regime(recent_ohlcv)
    
    regime = regime_result["regime"]
    regime_confidence = regime_result["confidence"]
    regime_probabilities = regime_result["probabilities"]
    
    logger.info(f"Current regime: {regime} (confidence: {regime_confidence:.1%})")
    
    # Step 2: Compute raw alpha signals
    imbalance_signal = compute_imbalance_alpha(
        imbalance_1=current_orderbook_features["imbalance_1"],
        bid_depth_1=current_orderbook_features["bid_depth_1"],
        ask_depth_1=current_orderbook_features["ask_depth_1"],
    )
    
    spread_signal = compute_spread_alpha(
        spread=current_orderbook_features["spread"],
        p50_spread=current_orderbook_features["p50_spread"],
        p75_spread=current_orderbook_features["p75_spread"],
    )
    
    logger.debug(f"Raw imbalance signal: {imbalance_signal.signal_value:.3f} (strength: {imbalance_signal.strength:.1%})")
    logger.debug(f"Raw spread signal: {spread_signal.signal_value:.3f} (strength: {spread_signal.strength:.1%})")
    
    # Step 3: Load regime parameters
    regime_params = get_alpha_params(regime)
    
    # Step 4: Adjust alphas for regime
    adjusted_imbalance = adjust_alpha_for_regime(imbalance_signal, regime_params)
    adjusted_spread = adjust_alpha_for_regime(spread_signal, regime_params)
    
    logger.debug(f"Adjusted imbalance: {adjusted_imbalance.signal_value:.3f} (sensitivity: {regime_params['imbalance_sensitivity']}x)")
    logger.debug(f"Adjusted spread: {adjusted_spread.signal_value:.3f} (sensitivity: {regime_params['spread_sensitivity']}x)")
    
    # Step 5: Blend signals
    combined_signal, combined_confidence = blend_alpha_signals(
        adjusted_imbalance,
        adjusted_spread,
        weights=alpha_signal_weights,
    )
    
    logger.info(f"Combined signal: {combined_signal:.3f} (confidence: {combined_confidence:.1%})")
    
    # Step 6: Package result
    result = RegimeAwareAlphaResult(
        regime=regime,
        regime_confidence=regime_confidence,
        regime_probabilities=regime_probabilities,
        imbalance_signal=imbalance_signal,
        spread_signal=spread_signal,
        adjusted_imbalance_signal=adjusted_imbalance,
        adjusted_spread_signal=adjusted_spread,
        combined_signal=combined_signal,
        combined_confidence=combined_confidence,
        alpha_params=regime_params,
        timestamp=pd.Timestamp.now(tz="UTC"),
    )
    
    return result
