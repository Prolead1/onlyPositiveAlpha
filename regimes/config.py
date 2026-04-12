"""Configuration and constants for regime identification."""

from enum import Enum

# ============================================================================
# REGIME DEFINITIONS (Option A: Market-Centric, No Macro Signals)
# ============================================================================

class RegimeType(Enum):
    """Market regime types (3-4 regimes)."""
    RISK_ON = "risk-on"                    # High vol + positive momentum + strong orderbook
    CONSOLIDATION = "consolidation"        # Low vol + sideways + balanced orderbook
    RISK_OFF = "risk-off"                  # High vol + negative momentum + weak orderbook
    TRANSITION = "transition"              # Uncertain / high uncertainty


# ============================================================================
# REGIME CHARACTERISTICS (for interpretation)
# ============================================================================

REGIME_CHARACTERISTICS = {
    RegimeType.RISK_ON: {
        "description": "Risk appetite, high volatility with positive momentum",
        "volatility": "HIGH",
        "returns": "POSITIVE",
        "orderbook_sentiment": "BULLISH",
        "alpha_opportunity": "HIGH",
    },
    RegimeType.CONSOLIDATION: {
        "description": "Market consolidation, low volatility, sideways movement",
        "volatility": "LOW",
        "returns": "NEUTRAL",
        "orderbook_sentiment": "BALANCED",
        "alpha_opportunity": "MEDIUM",
    },
    RegimeType.RISK_OFF: {
        "description": "Risk aversion, high volatility with negative momentum",
        "volatility": "HIGH",
        "returns": "NEGATIVE",
        "orderbook_sentiment": "BEARISH",
        "alpha_opportunity": "LOW",
    },
    RegimeType.TRANSITION: {
        "description": "Regime transition, high uncertainty",
        "volatility": "VARIABLE",
        "returns": "UNCERTAIN",
        "orderbook_sentiment": "UNCERTAIN",
        "alpha_opportunity": "UNCERTAIN",
    },
}

# ============================================================================
# CLUSTERING PARAMETERS
# ============================================================================

N_REGIMES = 3  # Use 3 or 4 regimes (set to 4 to include TRANSITION)
RANDOM_STATE = 42
CLUSTERING_METHOD = "kmeans"  # Simple K-Means

# ============================================================================
# FEATURE COMPUTATION PARAMETERS
# ============================================================================

# Rolling windows for volatility/momentum calculation
VOL_WINDOW = 20  # 20-day rolling volatility
MOMENTUM_WINDOW = 5  # 5-day momentum
MOMENTUM_MA_WINDOW = 20  # 20-day MA for trend

# Volume/volatility regime thresholds
VOL_HIGH_THRESHOLD = 0.75  # 75th percentile
VOL_LOW_THRESHOLD = 0.25  # 25th percentile

# Sentiment thresholds
SENTIMENT_BULLISH_THRESHOLD = 0.6
SENTIMENT_BEARISH_THRESHOLD = 0.4

# ============================================================================
# DATA PARAMETERS
# ============================================================================

MIN_DAYS_FOR_TRAINING = 60  # Minimum days of data needed to train
HISTORICAL_LOOKBACK_DAYS = 365  # 1 year of historical data by default

# ============================================================================
# MICRO REGIME PARAMETERS (5-minute frequency, intraday)
# ============================================================================

MICRO_TIMEFRAME = "5m"                      # Output 5-minute bars
MICRO_OHLCV_TIMEFRAME = "1m"                # Fetch 1-minute from CCXT
MICRO_ROLLING_WINDOW_SAMPLES = 6            # 6 × 5-min = 30 min rolling window

MICRO_N_REGIMES = 3                         # 3 regimes (risk-on, consolidation, risk-off)
MICRO_RANDOM_STATE = 42
MICRO_N_INIT = 10                           # K-Means n_init

# Features for micro regime clustering
MICRO_FEATURE_COLS = [
    "rolling_volatility",
    "rolling_momentum",
]

# Training & inference split
MICRO_TRAINING_START = "2025-06-01"         # Training period start (CCXT fetch)
MICRO_TRAINING_END = "2025-11-30"           # Training period end
MICRO_INFERENCE_START = "2025-12-01"        # Inference period start
MICRO_INFERENCE_END = "2026-03-31"          # Inference period end

# Data source
MICRO_SYMBOL = "BTC/USDT"                   # Trading pair
MICRO_EXCHANGES = None                      # None = use default exchanges
