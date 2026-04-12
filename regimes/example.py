"""Example: Training and using the regime identification module.

Run this script with your OHLCV data to:
1. Train the regime model
2. Save it for production use
3. Validate on historical data
"""

import logging
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import regime module components
from regimes import RegimeIdentifier, RegimePredictor
from regimes.helpers import get_regime_model_dir


def example_train_regime_model(ohlcv_data_path: str) -> None:
    """Example: Train regime model on historical data.
    
    Parameters
    ----------
    ohlcv_data_path : str
        Path to CSV with columns: open, high, low, close, volume
        Index: datetime (daily)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Load Historical Data")
    logger.info("=" * 60)
    
    # Load data
    df = pd.read_csv(ohlcv_data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} days of OHLCV data")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Train Regime Identifier")
    logger.info("=" * 60)
    
    # Train
    identifier = RegimeIdentifier(n_regimes=3)
    summary = identifier.fit(df)
    
    logger.info(f"\nTraining summary:")
    logger.info(f"  N Regimes: {summary['n_regimes']}")
    logger.info(f"  Regime Mapping: {summary['regime_mapping']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Save Model")
    logger.info("=" * 60)
    
    # Save
    model_dir = get_regime_model_dir()
    identifier.save(model_dir)
    logger.info(f"Model saved to: {model_dir}")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Validate - Predict on Training Data")
    logger.info("=" * 60)
    
    # Predict on same data
    regimes = identifier.predict(df)
    logger.info(f"\nRegime sequence (last 10 days):")
    logger.info(regimes.tail(10))
    
    logger.info(f"\nRegime distribution:")
    logger.info(regimes.value_counts())


def example_predict_current_regime(recent_data_path: str) -> None:
    """Example: Use trained model to predict current regime.
    
    Parameters
    ----------
    recent_data_path : str
        Path to CSV with recent OHLCV data (e.g., last 30-90 days)
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Load Recent Data")
    logger.info("=" * 60)
    
    # Load recent data
    df = pd.read_csv(recent_data_path, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} days of recent data")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Load Trained Model")
    logger.info("=" * 60)
    
    model_dir = get_regime_model_dir()
    predictor = RegimePredictor(model_dir)
    logger.info(f"Model loaded from: {model_dir}")
    
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Predict Current Regime")
    logger.info("=" * 60)
    
    result = predictor.predict_current_regime(df)
    
    logger.info(f"\n📊 CURRENT REGIME: {result['regime'].upper()}")
    logger.info(f"🎯 Confidence: {result['confidence']:.1%}")
    logger.info(f"\n📈 Regime Probabilities:")
    for regime, prob in result['probabilities'].items():
        logger.info(f"   {regime:20s}: {prob:6.1%}")
    
    logger.info(f"\n💡 Regime Context:")
    context = predictor.get_regime_context(result['regime'])
    logger.info(f"   Description: {context.get('description', 'N/A')}")
    logger.info(f"   Volatility: {context.get('volatility', 'N/A')}")
    logger.info(f"   Alpha Opportunity: {context.get('alpha_opportunity', 'N/A')}")
    
    return result


if __name__ == "__main__":
    # Example 1: Train model
    # Uncomment if you have historical data file
    # example_train_regime_model("path/to/historical_btc_ohlcv.csv")
    
    # Example 2: Predict current regime
    # Uncomment if you have recent data file
    # result = example_predict_current_regime("path/to/recent_btc_ohlcv.csv")
    
    logger.info("Example script ready. Uncomment the example calls to run.")
    logger.info("\nUsage:")
    logger.info("  1. Prepare historical OHLCV CSV (1-3 years of daily data)")
    logger.info("  2. Uncomment example_train_regime_model() call")
    logger.info("  3. Run this script to train and save model")
    logger.info("  4. Use example_predict_current_regime() for predictions")
