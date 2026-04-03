#!/usr/bin/env python
"""Quick validation tests for the regime identification module."""

import sys
import pandas as pd
import numpy as np
from datetime import datetime

print("\n" + "=" * 60)
print("REGIME IDENTIFICATION MODULE - VALIDATION TESTS")
print("=" * 60)

# Test 1: Imports
print("\n[TEST 1] Import modules...")
try:
    from regimes import RegimeType, RegimeIdentifier, RegimePredictor
    from regimes import compute_regime_features
    print("✅ PASSED: All imports successful")
except Exception as e:
    print(f"❌ FAILED: Import error: {e}")
    sys.exit(1)

# Test 2: Create sample data
print("\n[TEST 2] Create sample OHLCV data...")
try:
    n = 100
    dates = pd.date_range(end=datetime.now(), periods=n, freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 2),
        'high': 105 + np.cumsum(np.random.randn(n) * 2),
        'low': 95 + np.cumsum(np.random.randn(n) * 2),
        'close': 100 + np.cumsum(np.random.randn(n) * 2),
        'volume': np.random.randint(1000000, 5000000, n),
    }, index=dates)
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    print(f"✅ PASSED: Created {len(df)} days of data")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Feature engineering
print("\n[TEST 3] Compute regime features...")
try:
    df_with_features = compute_regime_features(df)
    sentiment = df_with_features['sentiment_composite'].iloc[-1]
    vol_regime = df_with_features['vol_regime_score'].iloc[-1]
    print(f"✅ PASSED: Features computed")
    print(f"   Last day sentiment: {sentiment:.3f}, vol_regime: {vol_regime:.3f}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 4: Train regime identifier
print("\n[TEST 4] Train RegimeIdentifier...")
try:
    identifier = RegimeIdentifier(n_regimes=3)
    summary = identifier.fit(df)
    print(f"✅ PASSED: Model trained")
    print(f"   Regimes: {summary['regime_mapping']}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 5: Predict regimes
print("\n[TEST 5] Predict regimes...")
try:
    regimes = identifier.predict(df)
    dist = regimes.value_counts()
    print(f"✅ PASSED: Predictions generated")
    print(f"   Regime distribution:")
    for regime, count in dist.items():
        pct = count / len(regimes) * 100
        print(f"     {regime}: {count} days ({pct:.1f}%)")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 6: Save model
print("\n[TEST 6] Save trained model...")
try:
    from regimes.helpers import get_regime_model_dir
    model_dir = get_regime_model_dir()
    identifier.save(model_dir)
    print(f"✅ PASSED: Model saved to {model_dir}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 7: Load model
print("\n[TEST 7] Load trained model...")
try:
    loaded_identifier = RegimeIdentifier.load(model_dir)
    test_pred = loaded_identifier.predict(df.tail(10))
    print(f"✅ PASSED: Model loaded and working")
    print(f"   Last 5 regimes: {list(test_pred.tail(5).values)}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 8: RegimePredictor
print("\n[TEST 8] RegimePredictor...")
try:
    predictor = RegimePredictor(model_dir)
    result = predictor.predict_current_regime(df.tail(30))
    print(f"✅ PASSED: RegimePredictor working")
    print(f"   Current regime: {result['regime']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Probabilities:")
    for regime, prob in result['probabilities'].items():
        print(f"     {regime}: {prob:.1%}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 9: Regime context
print("\n[TEST 9] Get regime context...")
try:
    context = predictor.get_regime_context(result['regime'])
    print(f"✅ PASSED: Got regime context")
    if context:
        print(f"   Description: {context.get('description', 'N/A')}")
        print(f"   Alpha Opportunity: {context.get('alpha_opportunity', 'N/A')}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n🎉 The regime identification module is ready to use!")
print("\nNext steps:")
print("  1. Obtain historical OHLCV data (1-3 years)")
print("  2. Run: identifier.fit(historical_df) to train")
print("  3. Use: predictor.predict_current_regime(recent_df) daily")
print("\nSee regimes/README.md for full documentation.")
print()
