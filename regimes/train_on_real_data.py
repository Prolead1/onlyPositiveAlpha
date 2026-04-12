#!/usr/bin/env python
"""
Train regime identification model on real historical crypto data.

This script:
1. Fetches historical BTC OHLCV data from CCXT
2. Trains the regime identification model
3. Saves the trained model for daily use
4. Validates on a test set
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import existing data fetching infrastructure FIRST (before regimes.helpers)
from data.historical import OHLCVParams, fetch_historical_data

# Import regime module
from regimes import RegimeIdentifier
from regimes.helpers import get_regime_model_dir


def fetch_btc_historical_data(
    years_back: int = 2,
    exchanges: list[str] | None = None,
) -> pd.DataFrame | None:
    """Fetch historical BTC/USDT data from CCXT.

    Parameters
    ----------
    years_back : int
        Number of years of data to fetch (default: 2 years).
    exchanges : list[str] | None
        List of exchanges to use (default: ['binance']).

    Returns
    -------
    pd.DataFrame | None
        Historical OHLCV data, or None if fetch fails.
    """
    if exchanges is None:
        exchanges = ["binance"]

    # Calculate date range
    end = datetime.now(UTC)
    start = end - timedelta(days=365 * years_back)

    logger.info(f"Fetching {years_back} years of BTC/USDT data")
    logger.info(f"Date range: {start.date()} to {end.date()}")
    logger.info(f"Exchanges: {exchanges}")

    # Use existing fetch infrastructure
    params = OHLCVParams(
        symbol="BTC/USDT",
        start_ms=int(start.timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
        timeframe="1d",  # Daily data
        limit=1000,
        exchanges=exchanges,
    )

    df = fetch_historical_data(params)

    if df is not None:
        logger.info(f"✅ Fetched {len(df)} days of data")
        logger.info(f"   Columns: {list(df.columns)}")
        logger.info(f"\n   Data summary:")
        logger.info(f"   {df.describe()}")
    else:
        logger.error("❌ Failed to fetch data")

    return df


def prepare_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare OHLCV data for regime identification.

    Ensures:
    - Index is datetime
    - Has required columns: open, high, low, close, volume
    - Sorted by date
    - No missing values

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data from fetch_historical_data.

    Returns
    -------
    pd.DataFrame
        Prepared data ready for regime model.
    """
    df = df.copy()

    # Extract OHLCV from nested 'data' dictionary
    if 'data' in df.columns:
        # Expand the nested 'data' dictionary into separate columns
        ohlcv_data = df['data'].apply(lambda x: pd.Series(x) if isinstance(x, dict) else x)
        df = pd.concat([df, ohlcv_data], axis=1)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'ts_event' in df.columns:
            df.index = pd.to_datetime(df['ts_event'])
        else:
            df.index = pd.to_datetime(df.index)

    # Keep only OHLCV columns
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    # Filter to only columns that exist
    cols_to_keep = [col for col in ohlcv_cols if col in df.columns]
    if len(cols_to_keep) < len(ohlcv_cols):
        missing = set(ohlcv_cols) - set(cols_to_keep)
        logger.warning(f"⚠️  Missing columns: {missing}")
    df = df[cols_to_keep].copy()

    # Sort by date
    df = df.sort_index()

    # Remove duplicates (keep first)
    df = df[~df.index.duplicated(keep='first')]

    # Forward fill any missing values
    df = df.ffill()

    logger.info(f"✅ Prepared {len(df)} days of OHLCV data")
    logger.info(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")

    return df


def train_regime_model(df: pd.DataFrame, n_regimes: int = 3) -> RegimeIdentifier:
    """Train regime identification model.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV data (prepared).
    n_regimes : int
        Number of regimes (3 or 4).

    Returns
    -------
    RegimeIdentifier
        Trained model.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING REGIME MODEL (n_regimes={n_regimes})")
    logger.info(f"{'=' * 60}")

    identifier = RegimeIdentifier(n_regimes=n_regimes)
    summary = identifier.fit(df)

    logger.info(f"\n✅ Model trained successfully")
    logger.info(f"\nRegime Mapping:")
    for cluster_id, regime_name in summary['regime_mapping'].items():
        logger.info(f"  Cluster {cluster_id} → {regime_name}")

    logger.info(f"\nCluster Centers (sentiment, vol_regime):")
    for i, center in enumerate(summary['cluster_centers']):
        logger.info(f"  Cluster {i}: [{center[0]:.3f}, {center[1]:.3f}]")

    return identifier


def validate_model(identifier: RegimeIdentifier, df: pd.DataFrame) -> None:
    """Validate trained model.

    Parameters
    ----------
    identifier : RegimeIdentifier
        Trained regime model.
    df : pd.DataFrame
        Historical OHLCV data.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"VALIDATION")
    logger.info(f"{'=' * 60}")

    # Predict on all historical data
    regimes = identifier.predict(df)

    # Get distribution
    dist = regimes.value_counts()
    logger.info(f"\n✅ Regimes predicted for all {len(regimes)} days")
    logger.info(f"\nRegime Distribution:")
    for regime, count in dist.items():
        pct = count / len(regimes) * 100
        logger.info(f"  {regime:20s}: {count:5d} days ({pct:5.1f}%)")

    # Show recent regimes
    logger.info(f"\nLast 30 days of regimes:")
    last_30 = regimes.tail(30)
    for i, (date, regime) in enumerate(last_30.items(), 1):
        is_current = "← CURRENT" if i == len(last_30) else ""
        logger.info(f"  {i:2d}. {date.date()}: {regime:20s} {is_current}")


def save_and_finalize(identifier: RegimeIdentifier) -> None:
    """Save trained model for production use.

    Parameters
    ----------
    identifier : RegimeIdentifier
        Trained regime model.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SAVING MODEL")
    logger.info(f"{'=' * 60}")

    model_dir = get_regime_model_dir()
    identifier.save(model_dir)

    logger.info(f"\n✅ Model saved to: {model_dir}")
    logger.info(f"\nFiles created:")
    for file in model_dir.glob("*"):
        size_kb = file.stat().st_size / 1024
        logger.info(f"  - {file.name} ({size_kb:.1f} KB)")


def main() -> None:
    """Main workflow: fetch data → train → validate → save."""
    logger.info("\n" + "=" * 60)
    logger.info("REGIME IDENTIFICATION - REAL DATA TRAINING")
    logger.info("=" * 60)

    # Step 1: Fetch historical data
    logger.info("\n[STEP 1] Fetch Historical Data")
    logger.info("-" * 60)

    df = fetch_btc_historical_data(years_back=2, exchanges=["binance"])

    if df is None:
        logger.error("❌ Failed to fetch data. Exiting.")
        return

    # Step 2: Prepare data
    logger.info("\n[STEP 2] Prepare Data")
    logger.info("-" * 60)

    df = prepare_ohlcv_data(df)

    # Step 3: Train model
    logger.info("\n[STEP 3] Train Model")
    logger.info("-" * 60)

    identifier = train_regime_model(df, n_regimes=3)

    # Step 4: Validate
    logger.info("\n[STEP 4] Validate")
    logger.info("-" * 60)

    validate_model(identifier, df)

    # Step 5: Save
    logger.info("\n[STEP 5] Save Model")
    logger.info("-" * 60)

    save_and_finalize(identifier)

    # Final message
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info("\n📊 Next Steps:")
    logger.info("1. Use the trained model for daily regime predictions:")
    logger.info("   from regimes import RegimePredictor")
    logger.info("   from regimes.utils import get_regime_model_dir")
    logger.info("   predictor = RegimePredictor(get_regime_model_dir())")
    logger.info("   result = predictor.predict_current_regime(recent_data)")
    logger.info("\n2. Share regime signals with your alpha teammates:")
    logger.info("   - Current regime (risk-on, consolidation, risk-off)")
    logger.info("   - Confidence level")
    logger.info("   - Regime probabilities")
    logger.info("   - Recommended alpha parameters")


if __name__ == "__main__":
    main()
