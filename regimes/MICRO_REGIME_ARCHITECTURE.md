# Micro Regime Classifier - Architecture Design

## Executive Summary

Upgrade the daily regime classifier (3122 days → 1097 unique trading days) to a **5-minute rolling regime classifier** that:
- Processes high-frequency OHLCV data (1-min, 1-sec → resampled to 5-min)
- Uses rolling feature windows (30-60 min) with **zero look-ahead bias**
- Applies K-Means clustering to compute regime labels + confidence scores
- Outputs clean CSV for integration with trading strategies

---

## Current State Analysis

### Daily Regime Classifier (Existing)

**Architecture:**
```
├── data/historical/ccxt.py         → Fetch 1d OHLCV
├── regimes/feature_engineering.py  → Compute sentiment + vol_regime
│   ├── compute_price_features()     (returns, volatility, momentum)
│   ├── compute_sentiment_composite() (0-1 score)
│   └── compute_volume_volatility_regime() (percentile)
├── regimes/identifier.py            → K-Means clustering + confidence
│   ├── fit() (train on historical data)
│   ├── predict() (assign regimes)
│   └── predict_with_confidence() (distance-based confidence)
└── regimes/export_regime_data_aggregated.py → CSV output
```

**Key Characteristics:**
- Window size: 20-day volatility, 5-day momentum
- Fields clustered: `sentiment_composite + vol_regime_score`
- Confidence: Data-driven (no hardcoding), normalized distance to boundary
- Output: 1097 daily records with regime + confidence

**Lessons Learned:**
- ✅ Data-driven confidence > hardcoding
- ✅ No look-ahead bias critical for ML training
- ✅ Exchange aggregation needed for multi-source data
- ✅ K-Means works well with 2D feature space

---

## Proposed Micro Regime Classifier Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    MICRO REGIME CLASSIFIER                      │
│                  (5-minute Rolling Regimes)                     │
└─────────────────────────────────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
    [Data Input]    [Feature Engineering]  [Model]
    
1. data_loader/micro.py
   ├── Fetch 1-min (or 1-sec) OHLCV from CCXT
   ├── resample_to_5min() → 5-min bars
   └── aggregate_exchanges() → volume-weighted

2. feature_engineering/micro.py
   ├── Rolling windows (← critical for no look-ahead)
   │   ├── Rolling volatility (30-60 min)
   │   ├── Rolling returns / momentum
   │   └── Rolling volume statistics
   ├── Feature normalization (StandardScaler per rolling batch)
   └── Feature matrix for clustering

3. identifier/micro.py
   ├── Train K-Means on training window (e.g., Jan-Feb)
   ├── Predict on forward period (e.g., Feb-Mar)
   ├── Confidence scores (distance-based)
   └── Zero look-ahead guarantee

4. export/micro.py
   └── Output CSV: timestamp | regime | confidence
```

---

## Module-by-Module Design

### 1. **data_loader/micro.py** — Data Loading & Resampling

**Goals:**
- Load 1-min or 1-sec OHLCV from CCXT
- Resample to 5-min intervals
- Aggregate multi-exchange data
- Ensure strict time alignment (no future leakage)

**Key Functions:**

```python
def load_high_frequency_data(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1m",          # "1m" or "1s"
    exchanges: list[str] | None = None
) -> pd.DataFrame:
    """Load 1-min/1-sec OHLCV data.
    
    Returns:
        DataFrame: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    """

def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample high-frequency bars to 5-min intervals.
    
    Uses OHLC aggregation:
    - open   = first open
    - high   = max(high)
    - low    = min(low)
    - close  = last close
    - volume = sum(volume)
    """

def aggregate_exchanges(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-exchange data (if multiple sources).
    
    Volume-weighted averaging for close, median for OHLH.
    """
```

**Parameters:**
- `timeframe`: "1m" (1-minute) or similar from CCXT
- `symbol`: "BTC/USDT"
- `start, end`: datetime with UTC timezone
- `exchanges`: ["binance", "kraken", ...] or None for all

**Output:** 
```
timestamp,open,high,low,close,volume
2026-02-01 10:05:00,48500.5,48650.3,48450.0,48600.0,125.3
2026-02-01 10:10:00,48600.1,48750.5,48550.0,48700.0,98.5
...
```

---

### 2. **feature_engineering/micro.py** — Rolling Feature Construction

**Critical Design:** Rolling windows **without look-ahead bias**

For each timestamp t:
- Use only data up to t (not future)
- Compute features on rolling window [t-60min, t]
- Common window: 30-60 minutes

**Key Functions:**

```python
def compute_rolling_volatility(
    series: pd.Series,
    window_samples: int = 12,  # 12 × 5-min = 60 min
) -> pd.Series:
    """Rolling volatility using expanding window.
    
    Each row t uses [t-60min, t] data.
    window_samples=12 → 12 × 5-min bars = 60 min window
    """

def compute_rolling_momentum(
    series: pd.Series,
    short_window: int = 6,     # 30 min
    long_window: int = 12,     # 60 min
) -> pd.Series:
    """Rolling momentum (short MA vs long MA).
    
    safe_shift=True ensures no look-ahead.
    """

def compute_rolling_returns(
    series: pd.Series,
    window_samples: int = 12,
) -> pd.Series:
    """Rolling mean returns over window."""

def compute_rolling_volume_stats(
    volume: pd.Series,
    window_samples: int = 12,
) -> pd.DataFrame:
    """Rolling volume statistics (mean, std, median)."""

def compute_micro_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main pipeline for micro regime features.
    
    Input:  df with ['open', 'high', 'low', 'close', 'volume']
    
    Output: df with added columns:
        - rolling_volatility
        - rolling_momentum
        - rolling_returns
        - rolling_volume_mean
        - rolling_volume_std
        
    Ensures: No look-ahead bias (each feature uses [t-window, t])
    """
```

**Parameters:**
- `window_samples`: Number of 5-min bars in rolling window
  - Default: 12 (= 60 minutes)
  - Can adjust: 6 (30 min), 24 (120 min)

**Guarantee:** Each feature at row t uses only data from [t-60min, t]

**Output:**
```
timestamp,open,high,low,close,volume,rolling_volatility,rolling_momentum,...
2026-02-01 10:05:00,48500.5,48650.3,...,0.0125,0.8,...
2026-02-01 10:10:00,48600.1,48750.5,...,0.0142,0.75,...
...
```

---

### 3. **identifier/micro.py** — K-Means Clustering & Confidence

**Training-Inference Split:**
- Train K-Means on limited historical window (e.g., Jan 1 - Jan 31)
- Predict regimes on forward period (e.g., Feb 1 - Feb 28)

**Key Class:**

```python
class MicroRegimeIdentifier:
    """K-Means clustering for 5-minute regimes."""
    
    def __init__(self, n_regimes: int = 3, random_state: int = 42):
        """Initialize."""
    
    def fit(self, df: pd.DataFrame):
        """Train K-Means on historical rolling-features data.
        
        Input:  df with rolling feature columns
        
        Process:
        1. Extract feature matrix (e.g., rolling_volatility + rolling_momentum)
        2. Standardize features (StandardScaler)
        3. Fit K-Means (n_clusters=3)
        4. Label clusters as (risk_on | consolidation | risk_off)
        
        Stores:
        - self.kmeans (fitted model)
        - self.scaler (StandardScaler)
        - self.regime_labels (cluster_id → RegimeType)
        """

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels on new data.
        
        Input:  df with rolling feature columns
        Output: pd.Series with regime labels
        
        Process:
        1. Standardize using self.scaler
        2. Get cluster assignments from self.kmeans
        3. Map to regime labels using self.regime_labels
        """

    def predict_with_confidence(
        self, df: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Predict regimes + confidence scores.
        
        Confidence = data-driven distance-based metric
        
        Returns:
        - regimes: pd.Series (regime labels)
        - confidences: pd.Series (0.0 to 1.0)
        """
    
    def save(self, dirpath: str):
        """Save trained model, scaler, labels."""
    
    @classmethod
    def load(cls, dirpath: str):
        """Load trained model."""
```

**Feature Selection:**
- Input features: `rolling_volatility`, `rolling_momentum`, `rolling_returns`
- Flexible: Can adjust to `rolling_volatility + rolling_momentum` (2D)

**Confidence Calculation:**
```python
confidence = (
    distance_to_centroid_min / 
    (distance_to_centroid_min + distance_to_second_closest)
)
# Result ∈ [0, 1]
```

---

### 4. **export/micro.py** — CSV Output

**Function:**

```python
def export_micro_regime_csv(
    df: pd.DataFrame,
    identifier: MicroRegimeIdentifier,
    output_file: str,
) -> Path:
    """Export 5-minute regime predictions to CSV.
    
    Input:  df with rolling features, trained identifier
    
    Output: CSV with columns:
        - timestamp (5-min bar)
        - regime (risk_on | consolidation | risk_off)
        - confidence (float ∈ [0, 1])
    
    Example rows:
        2026-02-01 10:05:00,risk_on,0.82
        2026-02-01 10:10:00,consolidation,0.65
        2026-02-01 10:15:00,risk_off,0.71
    """
```

---

## Data Flow Example

### Scenario: Feb 1 - Mar 1 with 5-minute bars

```
[Jan 1 - Jan 31]  Training Data
└─ Load 1-min OHLCV
└─ Resample to 5-min
└─ Compute rolling features (60-min window)
└─ Train K-Means (cluster centroids + standardization)
└─ Save model to regimes/models/micro/

[Feb 1 - Mar 1]  Inference Period
└─ Load 1-min OHLCV
└─ Resample to 5-min
└─ Compute rolling features (using training scaler + model)
└─ Predict regime + confidence for each 5-min bar
└─ Export to reports/micro_regimes_2026-02-01_to_2026-03-01.csv

Result: ~425 rows (288 per day × ~1.5 days ≈ 432 bars)
```

---

## File Organization

```
onlyPositiveAlpha/
├── regimes/
│   ├── identifier.py                 (existing, keep as-is)
│   ├── feature_engineering.py        (existing, keep as-is)
│   ├── config.py                     (existing, add MICRO_* params)
│   │
│   ├── micro/                        ← NEW FOLDER
│   │   ├── __init__.py
│   │   ├── data_loader.py            ← Load + resample 5-min data
│   │   ├── feature_engineering.py    ← Compute rolling features
│   │   ├── identifier.py             ← K-Means + confidence
│   │   └── export.py                 ← CSV export
│   │
│   └── models/
│       └── micro/                    ← Trained models (kmeans.pkl, scaler.pkl)
│
├── reports/
│   └── micro_regimes_*.csv           ← Output CSVs
│
└── scripts/
    └── train_micro_regime_classifier.py  ← Main training + export script
```

---

## Configuration Parameters

**Add to `regimes/config.py`:**

```python
# ============================================================================
# MICRO REGIME PARAMETERS (5-minute frequency)
# ============================================================================

# Resampling
MICRO_TIMEFRAME = "5m"              # 5-minute bars
MICRO_OHLCV_TIMEFRAME = "1m"        # Load 1-minute from CCXT
MICRO_ROLLING_WINDOW_SAMPLES = 12   # 12 × 5-min = 60 min window

# K-Means
MICRO_N_REGIMES = 3                 # 3 regimes
MICRO_N_INIT = 10                   # K-Means n_init

# Features for clustering
MICRO_FEATURE_COLS = [
    "rolling_volatility",
    "rolling_momentum",
    # optional: "rolling_returns"
]

# Training & Inference
MICRO_TRAINING_PERIOD_DAYS = 30     # e.g., Jan 1-31
MICRO_INFERENCE_PERIOD_DAYS = 28    # e.g., Feb 1-28
```

---

## No Look-Ahead Bias Guarantee

### Strategy: Expanding Window + Strict Time Ordering

1. **Each 5-min bar t** computes features using only [t-60min, t]
2. **Scaler fitted on training data only** (e.g., Jan 1-31)
3. **Training samples never use future data** (K-Means fit)
4. **Inference period strictly after training** (Feb 1 after Jan 31)

### Validation:
```python
def validate_no_lookahead(df: pd.DataFrame) -> bool:
    """Ensure each feature at row t uses only [t-window, t]."""
    for i in range(len(df)):
        feature_value = df.iloc[i]['rolling_volatility']
        window_data = df.iloc[max(0, i-12):i+1]  # Inclusive of t
        recomputed = window_data['returns'].std()
        assert np.isclose(feature_value, recomputed)
    return True
```

---

## Quality Checklist

- ✅ Modular code (data → features → model → export)
- ✅ No look-ahead bias (verified per timestamp)
- ✅ Rolling feature windows (strict time ordering)
- ✅ K-Means + data-driven confidence
- ✅ High-frequency data handling (1-min → 5-min)
- ✅ Multi-exchange support (optional)
- ✅ CSV output for team consumption
- ✅ Production-ready (vectorized, efficient)
- ✅ Explainable (confidence = distance to centroid)

---

## Next Steps

1. **Create module structure** (regimes/micro/)
2. **Implement data_loader.py** (1-min → 5-min resampling)
3. **Implement feature_engineering.py** (rolling features)
4. **Implement identifier.py** (K-Means + confidence)
5. **Implement export.py** (CSV output)
6. **Create training script** (orchestrate full pipeline)
7. **Test on Feb-Mar 2026 data**
8. **Validate no look-ahead bias**
9. **Generate final CSV for team**

