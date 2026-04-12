# Quick-Start Guide: Micro Regime Classifier

## 📌 What You Got

A **production-ready 5-minute regime classifier** for intraday BTC/USDT trading with:
- K-Means clustering on rolling volatility + momentum
- 17,563 predictions (Dec 1, 2025 - Jan 31, 2026)
- Confidence scores for each 5-minute bar
- **Zero look-ahead bias** (validated)

## 🚀 Quick Commands

### 1. View the Output CSV
```bash
cd onlyPositiveAlpha
head -50 reports/micro_regimes_2025-12-01_to_2026-01-31.csv
```

### 2. Analyze the Results
```bash
python3 << 'EOF'
import pandas as pd

df = pd.read_csv("reports/micro_regimes_2025-12-01_to_2026-01-31.csv")

print(f"Total records: {len(df)}")
print(f"\nRegime distribution:")
print(df['regime'].value_counts())
print(f"\nConfidence statistics:")
print(df['confidence'].describe())

# By regime
for regime in df['regime'].unique():
    regime_df = df[df['regime'] == regime]
    print(f"\n{regime.upper()}:")
    print(f"  Count: {len(regime_df)}")
    print(f"  Avg confidence: {regime_df['confidence'].mean():.1%}")
    print(f"  Min/Max confidence: {regime_df['confidence'].min():.1%} / {regime_df['confidence'].max():.1%}")
EOF
```

### 3. Re-run Training with New Data
Edit `regimes/config.py` to change dates:
```python
MICRO_TRAINING_START = "2026-02-01"
MICRO_TRAINING_END = "2026-02-28"
MICRO_INFERENCE_START = "2026-03-01"
MICRO_INFERENCE_END = "2026-03-31"
```

Then run:
```bash
source .venv/bin/activate
PYTHONPATH="${PWD}" python scripts/train_micro_regime_classifier.py
```

### 4. Use the Trained Model in Code
```python
from regimes.micro.identifier import MicroRegimeIdentifier
from regimes.micro.data_loader import load_and_resample_5min
from regimes.micro.feature_engineering import compute_micro_rolling_features

# Load the trained model
identifier = MicroRegimeIdentifier.load("regimes/models/micro")

# Load and prepare new data (e.g., Feb 2026)
df = load_and_resample_5min(
    symbol="BTC/USDT",
    start_date="2026-02-01",
    end_date="2026-02-28"
)
df = compute_micro_rolling_features(df)

# Skip first 6 rows (insufficient window data)
df = df.iloc[6:].reset_index(drop=True)

# Predict
regimes, confidences = identifier.predict_with_confidence(df)

# Results
for ts, regime, conf in zip(df['timestamp'], regimes, confidences):
    print(f"{ts}: {regime} (confidence: {conf:.1%})")
```

## 📊 File Locations

| File | Purpose | Location |
|------|---------|----------|
| Output CSV | 17,563 predictions | `reports/micro_regimes_2025-12-01_to_2026-01-31.csv` |
| Training Script | Main orchestration | `scripts/train_micro_regime_classifier.py` |
| Configuration | Parameters & dates | `regimes/config.py` |
| Data Loader | 1-min → 5-min resampler | `regimes/micro/data_loader.py` |
| Features | Rolling window calculations | `regimes/micro/feature_engineering.py` |
| Model | K-Means classifier | `regimes/micro/identifier.py` |
| Trained Model | Saved artifacts | `regimes/models/micro/` |

## 🎯 Key Metrics

| Metric | Value |
|--------|-------|
| Training period | Nov 1-30, 2025 (8,347 bars) |
| Inference period | Dec 1, 2025 - Jan 31, 2026 (17,563 bars) |
| Feature window | 30 minutes (6 × 5-min bars) |
| Features used | rolling_volatility + rolling_momentum |
| Model | K-Means (3 clusters) |
| Average confidence | 90.6% |
| Consolidation % | 83.7% |
| Risk-Off % | 9.1% |
| Risk-On % | 7.2% |

## 🔍 Understanding Regimes

- **Consolidation (83.7%)**: Low volatility, sideways movement
  - Best for: Range-trading strategies
  - Worst for: Momentum strategies

- **Risk-Off (9.1%)**: High volatility, bearish sentiment
  - Best for: Defensive strategies, short exposure
  - Worst for: Risk-on strategies

- **Risk-On (7.2%)**: High volatility, bullish sentiment
  - Best for: Momentum & trend strategies
  - Worst for: Defensive strategies

## ⚙️ How It Works (High-Level)

```
1. Load 1-minute OHLCV data
   ↓
2. Resample to 5-minute bars
   ↓
3. Compute rolling features (30-min window)
   - rolling_volatility (std of returns)
   - rolling_momentum (short MA vs long MA)
   ↓
4. Train K-Means (Nov data) or predict (Dec-Jan data)
   ↓
5. Assign regime labels (risk-on, consolidation, risk-off)
   ↓
6. Compute confidence scores (distance-based softmax)
   ↓
7. Export to CSV
```

## ✅ What's Guaranteed

- ✅ **No look-ahead bias**: Each bar uses only past 30 minutes of data
- ✅ **Reproducible**: Fixed random seed (42) in K-Means
- ✅ **Efficient**: Vectorized (pandas/numpy), no loops
- ✅ **Persistent**: Trained model saved for future use
- ✅ **Clean**: Multi-exchange data aggregated with volume-weighting

## 📝 Notes

1. **First 6 rows of each run** have NaN values (insufficient window data) → automatically removed
2. **Confidence range** [0, 1] with higher = more certain
3. **Model expires**: Consider retraining if market regime changes significantly
4. **Exchange coverage**: Binance (primary), Coinbase, Bitstamp (Kraken N/A)

## 🆘 Troubleshooting

**Q: CSV looks empty or small?**
- A: Check dates in `regimes/config.py`. Must be valid trading dates.

**Q: Want to re-run on different dates?**
- A: Edit `regimes/config.py` → Set MICRO_TRAINING/INFERENCE dates → Run script

**Q: Can I use this for live trading?**
- A: Yes! Load model once, call `predict_with_confidence()` on each new 5-min bar

**Q: How to share with team?**
- A: Send the CSV from `reports/` folder. Most tools can read CSV directly.

---

**Status**: ✅ Production Ready  
**Last Updated**: 2026-04-04  
**Test Period**: Nov 1, 2025 - Jan 31, 2026 (92 days)
