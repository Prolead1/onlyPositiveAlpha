# Market Regime Identification Module

This module identifies and predicts cryptocurrency market regimes to **complement and enhance alpha strategies**.

## Quick Start

### 1. Train Regime Model (One-time setup)

```python
import pandas as pd
from regimes import RegimeIdentifier
from regimes.utils import get_regime_model_dir

# Load historical OHLCV data (e.g., 1-3 years)
# DataFrame should have columns: open, high, low, close, volume
# Index should be datetime
df = pd.read_csv("historical_btc_ohlcv.csv", index_col=0, parse_dates=True)

# Train identifier
identifier = RegimeIdentifier(n_regimes=3)
summary = identifier.fit(df)

# Save trained model
model_dir = get_regime_model_dir()
identifier.save(model_dir)
print(f"Model saved: {model_dir}")
```

### 2. Predict Current Regime (Daily use)

```python
from regimes import RegimePredictor
from regimes.utils import get_regime_model_dir

# Load predictor
model_dir = get_regime_model_dir()
predictor = RegimePredictor(model_dir)

# Get current regime (with ~30+ days of recent data)
recent_data = pd.read_csv("recent_btc_ohlcv.csv", index_col=0, parse_dates=True)
result = predictor.predict_current_regime(recent_data)

print(f"Current regime: {result['regime']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Regime probabilities: {result['probabilities']}")
```

---

## Architecture

### Regime Types (3-4 Discrete Regimes)

**Option A: Market-Centric (Current Implementation)**

1. **Risk-On** (`risk-on`)
   - High volatility + positive momentum
   - Strong market sentiment
   - High alpha opportunity
   - → Alphas should be **aggressive**

2. **Consolidation** (`consolidation`)
   - Low volatility + sideways movement
   - Balanced market sentiment
   - Medium alpha opportunity
   - → Alphas should be **moderate/defensive**

3. **Risk-Off** (`risk-off`)
   - High volatility + negative momentum
   - Weak market sentiment
   - Low alpha opportunity
   - → Alphas should be **conservative**

4. (Optional) **Transition** (`transition`)
   - High uncertainty, regime switching
   - Use only if `n_regimes=4`

---

## Data Flow

### Phase 1-3: Historical Regime Identification ✅ (IMPLEMENTED)

```
Historical OHLCV Data (1-3 years)
    ↓
Feature Engineering
  ├─ Volatility (rolling 20-day)
  ├─ Momentum (5d vs 20d MA)
  ├─ Price trend (above/below MA)
  └─ Returns (daily)
    ↓
Sentiment Composite Score [0,1]
  (blend of momentum + price trend + recent returns)
    ↓
Volume/Volatility Regime Score [0,1]
  (percentile of rolling volatility)
    ↓
K-Means Clustering (K=3 or 4)
    ↓
Regime Labels (Risk-On, Consolidation, Risk-Off, ±Transition)
    ↓
Save Model (kmeans, scaler, regime_labels)
```

### Phase 4-5: Probability & Optimization (SKELETAL - TODO)

```
Current Market Data (last ~30 days)
    ↓
Compute Features (same as training)
    ↓
K-Means Predict Cluster
    ↓
Get Current Regime Label (Risk-On, Consolidation, or Risk-Off)
    ↓
Compute Regime Probabilities (% in each regime over last 10 days)
    ↓
[TODO] Regret Minimization
  - Get optimal alpha params for each regime
  - Compute regret = gap to optimal
  - Blend parameters to minimize weighted regret
    ↓
Output: Current Regime + Recommended Alpha Parameters
```

---

## Module Structure

```
regimes/
├── __init__.py                    # Public API
├── config.py                      # Regime definitions, parameters
├── feature_engineering.py         # Compute features (vol + sentiment)
├── identifier.py                  # K-Means clustering + labeling
├── predictor.py                   # Current regime prediction
├── probability_estimator.py       # (PHASE 4) Probability estimation
├── regret_optimizer.py            # (PHASE 4) Regret minimization
├── utils.py                       # Helper functions
├── models/                        # Trained models dir (auto-created)
│   ├── kmeans.pkl
│   ├── scaler.pkl
│   └── regime_labels.json
└── README.md
```

---

## Feature Engineering Explained

### Sentiment Composite Score [0, 1]

Blends three technical signals:

$$\text{Sentiment} = 0.33 \times \text{Momentum} + 0.33 \times \text{Price Trend} + 0.34 \times \text{Returns}$$

- **Momentum**: 5-day MA vs 20-day MA direction (-1 → 0, 0 → 0.5, 1 → 1)
- **Price Trend**: Is price above 20-day MA? (No → 0, Yes → 1)
- **Recent Returns**: Normalized 5-day average returns (-5% → 0, 0% → 0.5, +5% → 1)

**Result**: 
- 1.0 = Very bullish
- 0.5 = Neutral
- 0.0 = Very bearish

### Volume/Volatility Regime Score [0, 1]

Percentile rank of rolling 20-day volatility:
- 0.0 = Lowest volatility (consolidation)
- 1.0 = Highest volatility (trending/stressed)

---

## Clustering Logic

K-Means groups data into 3-4 clusters based on `[sentiment_composite, vol_regime_score]`.

**Labeling Rules**:

```
If vol_regime < 0.33:
    Regime = CONSOLIDATION
Else if sentiment >= 0.6 AND vol_regime >= 0.5:
    Regime = RISK_ON
Else if sentiment < 0.4 AND vol_regime >= 0.5:
    Regime = RISK_OFF
Else:
    Regime = TRANSITION (if 4 regimes) or CONSOLIDATION (if 3 regimes)
```

---

## Integration with Alpha Strategies

### For Your Teammates

**Output**: Daily regime prediction API

```python
result = predictor.predict_current_regime(recent_data)
# {
#     "regime": "risk-on",
#     "probabilities": {"risk-on": 0.65, "consolidation": 0.25, "risk-off": 0.10},
#     "confidence": 0.87,
#     "timestamp": "2026-04-03T10:30:00",
# }
```

**Recommendation**: 
- If regime is **Risk-On**: Use aggressive alpha parameters
- If regime is **Consolidation**: Use balanced/defensive parameters
- If regime is **Risk-Off**: Use conservative parameters

### Phase 4 (Future): Regret-Minimized Blending

Once your teammates provide:
- Optimal alpha returns for each regime
- Alpha parameter sets

This module will compute:
- "Regret" = gap between optimal and current approach in each regime
- Probability-weighted blend of alpha parameters
- Example: 65% aggressive + 25% balanced + 10% conservative

---

## Usage Examples

### Example 1: Training

```python
import pandas as pd
from regimes import RegimeIdentifier
from regimes.utils import get_regime_model_dir

# Load BTC OHLCV (1 year)
df = pd.read_csv("btc_ohlcv.csv", index_col=0, parse_dates=True)

# Train
identifier = RegimeIdentifier(n_regimes=3)
summary = identifier.fit(df)
# Output:
# {
#     'n_samples': 365,
#     'n_regimes': 3,
#     'regime_mapping': {0: 'consolidation', 1: 'risk-on', 2: 'risk-off'},
#     'cluster_centers': [[0.45, 0.30], [0.75, 0.85], [0.35, 0.80]]
# }

# Save
identifier.save(get_regime_model_dir())
```

### Example 2: Prediction

```python
from regimes import RegimePredictor
from regimes.utils import get_regime_model_dir

# Load recent data (30-90 days)
recent_data = pd.read_csv("btc_recent.csv", index_col=0, parse_dates=True)

# Predict
predictor = RegimePredictor(get_regime_model_dir())
result = predictor.predict_current_regime(recent_data)

print(f"🟢 Regime: {result['regime']}")
print(f"📊 Confidence: {result['confidence']:.1%}")
print(f"📈 Probabilities: {result['probabilities']}")

# Check characteristics
context = predictor.get_regime_context(result['regime'])
print(f"💡 {context['description']}")
print(f"   Alpha Opportunity: {context['alpha_opportunity']}")
```

### Example 3: Backtest Validation

```python
# After training, validate on historical data
from regimes import RegimeIdentifier
from regimes.utils import get_regime_model_dir

identifier = RegimeIdentifier.load(get_regime_model_dir())

# Predict on all historical data
regimes = identifier.predict(df)

# Analyze: Do different regimes have different alpha performance?
# df['regime'] = regimes
# grouped = df.groupby('regime')[['alpha_return']].mean()
# print(grouped)  # Should show meaningful differences
```

---

## Next Steps (TODO - Phases 4-6)

### Phase 4: Integration with Alpha Models
- [ ] Get optimal alpha return for each regime from teammates
- [ ] Get alpha parameter sets per regime
- [ ] Implement regret minimization
- [ ] Output: Probability-weighted alpha parameters

### Phase 5: Real-Time Deployment
- [ ] Set up daily cron job to predict regime
- [ ] Store regime predictions in database
- [ ] Expose API endpoint: `/api/current_regime`

### Phase 6: Backtesting & Validation
- [ ] Compare alpha performance across regimes
- [ ] Verify regime-specific alpha parameters improve returns
- [ ] Document lessons learned + regime transitions

---

## Data Requirements

### For Training
- **Minimum**: 60 days of historical OHLCV data
- **Recommended**: 1-3 years
- **Format**: DataFrame with columns: `open, high, low, close, volume`
- **Index**: Datetime, daily frequency

### For Prediction
- **Minimum**: 30 days of recent OHLCV data
- **Recommended**: 90-365 days
- **Same format** as training data

---

## Troubleshooting

**Q: Model not converging / strange regimes?**
- Try 4 regimes instead of 3 (adds TRANSITION regime)
- Check data quality (any gaps, outliers?)
- Ensure data is sorted by date

**Q: Confidence too low?**
- Use more historical data for training (1+ years)
- Reduce momentum window from 20 to 10 days

**Q: Features not separating well?**
- Could try orderbook sentimentin addition to price (future phase)
- Or add correlation/cross-asset features

---

## References

- **Paper**: GIC/BlackRock "Embracing Uncertainty" - Scenario-based portfolio construction
- **Adaptation**: Simplified market-centric regimes + K-Means clustering
- **Features**: Volume/volatility + sentiment composite (no macro signals for now)

---

**Status**: Phases 1-3 complete ✅. Phases 4-6 skeletal, ready for extension.
