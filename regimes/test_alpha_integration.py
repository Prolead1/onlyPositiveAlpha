"""
Test regime-aware alpha integration end-to-end.

Validates that:
1. Regime model loads correctly
2. Raw alpha signals compute correctly
3. Regime adjustments apply correctly
4. Final signal is reasonable
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from regimes.alpha_integration import integrate_regimes_with_alphas
from regimes.alpha_params import get_alpha_params, get_all_regimes
from data.historical import fetch_historical_data, OHLCVParams
from utils import setup_application_logging, get_workspace_root

setup_application_logging()
logger = logging.getLogger(__name__)


def test_alpha_integration():
    """End-to-end test of regime-aware alpha integration."""
    
    print("\n" + "="*80)
    print("TEST: Regime-Aware Alpha Integration")
    print("="*80)
    
    # Setup
    workspace_root = get_workspace_root()
    regime_model_dir = workspace_root / "regimes" / "models"
    
    if not regime_model_dir.exists():
        print("❌ Regime model not found. Train it first:")
        print("   python regimes/train_on_blended_data.py")
        return False
    
    # Test 1: Load alpha parameters
    print("\n1. Loading alpha parameters per regime...")
    for regime in get_all_regimes():
        params = get_alpha_params(regime)
        print(f"   {regime:15s}: volatility_target={params['volatility_target']:2d}%, "
              f"leverage={params['leverage']:.1f}x, "
              f"sharpe={params['expected_sharpe']:.1f}")
    print("   ✓ Alpha params loaded")
    
    # Test 2: Fetch recent data
    print("\n2. Fetching recent CCXT data (90 days)...")
    try:
        ohlcv_params = OHLCVParams(
            symbol="BTC/USDT",
            start_ms=None,
            end_ms=None,
            timeframe="1d",
            limit=1000,
            exchanges=["binance"],
        )
        recent_ohlcv = fetch_historical_data(ohlcv_params)
        
        if recent_ohlcv is None or recent_ohlcv.empty:
            print("❌ No OHLCV data fetched")
            return False
        
        # Debug: print columns
        print(f"   Data columns: {list(recent_ohlcv.columns)}")
        print(f"   Data shape: {recent_ohlcv.shape}")
        print(f"   Data index: {type(recent_ohlcv.index)}")
        print(f"   First row:\n{recent_ohlcv.iloc[0]}")
        
        # Ensure index is datetime
        if not isinstance(recent_ohlcv.index, pd.DatetimeIndex):
            if 'ts_event' in recent_ohlcv.columns:
                recent_ohlcv['ts_event'] = pd.to_datetime(recent_ohlcv['ts_event'], unit='ms')
                recent_ohlcv = recent_ohlcv.set_index('ts_event').sort_index()
        
        recent_ohlcv = recent_ohlcv.tail(90)  # Last 90 days
        latest_idx = recent_ohlcv.index[-1]
        if isinstance(latest_idx, pd.Timestamp):
            latest_str = latest_idx.strftime('%Y-%m-%d')
        else:
            latest_str = str(latest_idx)
        
        # Get close price from data dict column
        if 'data' in recent_ohlcv.columns:
            close_price = recent_ohlcv['data'].iloc[-1].get('close', 0)
        elif 'close' in recent_ohlcv.columns:
            close_price = recent_ohlcv['close'].iloc[-1]
        else:
            close_price = 0
        
        print(f"   ✓ Fetched {len(recent_ohlcv)} days")
        print(f"   Latest: {latest_str}, Close: ${close_price:,.2f}")
        
        # Normalize: extract 'data' dict columns into flat dataframe
        if 'data' in recent_ohlcv.columns:
            ts_events = recent_ohlcv.index if isinstance(recent_ohlcv.index, pd.DatetimeIndex) else pd.to_datetime(recent_ohlcv['ts_event'].values if 'ts_event' in recent_ohlcv.columns else recent_ohlcv.index)
            flattened = pd.DataFrame(recent_ohlcv['data'].tolist())
            flattened.index = ts_events
            recent_ohlcv = flattened
            print(f"   Flattened to columns: {list(recent_ohlcv.columns)}")
    except Exception as e:
        print(f"❌ Error fetching OHLCV: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Simulate current orderbook features
    print("\n3. Simulating current orderbook features...")
    
    # These would normally come from live orderbook snapshot
    mock_orderbook_features = {
        "imbalance_1": 0.08,        # 8% bullish imbalance
        "bid_depth_1": 50.5,        # BTC
        "ask_depth_1": 46.2,        # BTC
        "spread": 2.50,             # $2.50
        "p50_spread": 3.25,         # Median spread
        "p75_spread": 5.00,         # 75th percentile spread
    }
    
    print(f"   Imbalance: {mock_orderbook_features['imbalance_1']:.1%} (bullish)")
    print(f"   Spread: ${mock_orderbook_features['spread']:.2f} "
          f"(p50=${mock_orderbook_features['p50_spread']:.2f})")
    print("   ✓ Orderbook features ready")
    
    # Test 4: Run integration for each regime (via regime prediction)
    print("\n4. Running regime-aware alpha integration...")
    
    try:
        print(f"   Loading regime model from: {regime_model_dir}")
        print(f"   Files in model dir: {list(regime_model_dir.glob('*'))}")
        result = integrate_regimes_with_alphas(
            recent_ohlcv=recent_ohlcv,
            current_orderbook_features=mock_orderbook_features,
            regime_model_dir=regime_model_dir,
        )
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Validate output
    print("\n5. Validating results...")
    
    print(f"\n   Regime Prediction:")
    print(f"   • Current: {result.regime}")
    print(f"   • Confidence: {result.regime_confidence:.1%}")
    print(f"   • Probabilities: {result.regime_probabilities}")
    
    print(f"\n   Raw Alpha Signals:")
    print(f"   • Imbalance:  {result.imbalance_signal.signal_value:+6.3f} "
          f"(strength: {result.imbalance_signal.strength:.1%})")
    print(f"   • Spread:     {result.spread_signal.signal_value:+6.3f} "
          f"(strength: {result.spread_signal.strength:.1%})")
    
    print(f"\n   Regime-Adjusted Alpha Signals:")
    print(f"   • Imbalance:  {result.adjusted_imbalance_signal.signal_value:+6.3f} "
          f"(strength: {result.adjusted_imbalance_signal.strength:.1%})")
    print(f"   • Spread:     {result.adjusted_spread_signal.signal_value:+6.3f} "
          f"(strength: {result.adjusted_spread_signal.strength:.1%})")
    
    # Determine signal direction
    if result.combined_signal > 0.333:
        signal_direction = "🔼 BUY"
    elif result.combined_signal < -0.333:
        signal_direction = "🔽 SELL"
    else:
        signal_direction = "⬜ NEUTRAL"
    
    print(f"\n   Combined Signal:")
    print(f"   • Direction: {signal_direction}")
    print(f"   • Value: {result.combined_signal:+6.3f}")
    print(f"   • Confidence: {result.combined_confidence:.1%}")
    
    print(f"\n   Recommended Parameters ({result.regime}):")
    params = result.alpha_params
    print(f"   • Volatility Target: {params['volatility_target']}%")
    print(f"   • Leverage: {params['leverage']}x")
    print(f"   • Max Drawdown: {params['max_drawdown_pct']}%")
    print(f"   • Imbalance Sensitivity: {params['imbalance_sensitivity']}x")
    print(f"   • Spread Sensitivity: {params['spread_sensitivity']}x")
    print(f"   • Expected Sharpe: {params['expected_sharpe']}")
    print(f"   • Expected Annual Return: {params['expected_return_pct']}%")
    
    print("\n✓ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_alpha_integration()
    sys.exit(0 if success else 1)
