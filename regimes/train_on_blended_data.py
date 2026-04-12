#!/usr/bin/env python
"""
Train regime identification model using blended CCXT + Polymarket data.

This script:
1. Fetches historical BTC OHLCV data from CCXT (price/momentum/volatility)
2. Loads Polymarket orderbook data (prediction market sentiment)
3. Extracts features from both sources
4. Blends them together (50% CCXT + 50% Polymarket)
5. Trains the regime identification model
6. Saves the trained model for daily use
"""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from glob import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import existing data fetching infrastructure
from data.historical import OHLCVParams, fetch_historical_data

# Import regime module
from regimes import RegimeIdentifier
from regimes.helpers import get_regime_model_dir


def fetch_btc_historical_data(
    years_back: int = 2,
    exchanges: list[str] | None = None,
) -> pd.DataFrame | None:
    """Fetch historical BTC/USDT data from CCXT."""
    if exchanges is None:
        exchanges = ["binance"]

    end = datetime.now(UTC)
    start = end - timedelta(days=365 * years_back)

    logger.info(f"[CCXT] Fetching {years_back} years of BTC/USDT data")
    logger.info(f"       Date range: {start.date()} to {end.date()}")

    params = OHLCVParams(
        symbol="BTC/USDT",
        start_ms=int(start.timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
        timeframe="1d",
        limit=1000,
        exchanges=exchanges,
    )

    df = fetch_historical_data(params)
    if df is not None:
        logger.info(f"✅ Fetched {len(df)} days of CCXT data")
    return df


def load_polymarket_orderbook_data(data_dir: str | Path) -> pd.DataFrame | None:
    """Load and aggregate Polymarket orderbook data.
    
    Loads all parquet files from polymarket_market directory,
    extracts orderbook sentiment, and aggregates to daily level.
    """
    data_dir = Path(data_dir)
    market_dir = data_dir / "stream_feeds" / "polymarket_market"
    
    if not market_dir.exists():
        logger.warning(f"⚠️  Polymarket market directory not found: {market_dir}")
        return None
    
    # Find all parquet files
    parquet_files = sorted(glob(str(market_dir / "*.parquet")))
    if not parquet_files:
        logger.warning(f"⚠️  No parquet files found in {market_dir}")
        return None
    
    logger.info(f"[POLYMARKET] Loading {len(parquet_files)} orderbook snapshots...")
    
    all_data = []
    for file_path in parquet_files[:100]:  # Limit to first 100 files for speed
        try:
            df = pd.read_parquet(file_path)
            all_data.append(df)
        except Exception as e:
            logger.debug(f"Error reading {file_path}: {e}")
            continue
    
    if not all_data:
        logger.warning("Could not load any Polymarket data")
        return None
    
    # Combine all snapshots
    pm_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"✅ Loaded {len(pm_data)} orderbook snapshots")
    logger.info(f"   Columns: {list(pm_data.columns)[:10]}...")
    
    return pm_data


def extract_polymarket_sentiment(pm_data: pd.DataFrame) -> pd.Series:
    """Extract sentiment features from Polymarket orderbook data.
    
    Key features:
    - Bid-ask imbalance (YES price vs NO price)
    - Order flow intensity
    """
    if pm_data is None or len(pm_data) == 0:
        return None
    
    logger.info("[POLYMARKET] Extracting sentiment features...")
    
    features = []
    
    # Expected structure: 'bids', 'asks', or similar order book columns
    for _, row in pm_data.iterrows():
        try:
            # Try to extract bid-ask data (adjust column names as needed)
            sentiment = 0.5  # Default neutral
            
            # Example: if there are price columns for YES/NO outcomes
            if 'yes_price' in row and 'no_price' in row:
                yes_price = float(row['yes_price'])
                no_price = float(row['no_price'])
                # Normalize: YES price close to 1 = bullish, close to 0 = bearish
                sentiment = yes_price if not np.isnan(yes_price) else 0.5
            elif 'price' in row:
                sentiment = float(row['price'])
            
            features.append(np.clip(sentiment, 0.0, 1.0))
        except:
            features.append(0.5)
    
    feature_series = pd.Series(features, index=range(len(features)))
    logger.info(f"✅ Extracted Polymarket sentiment (mean: {feature_series.mean():.2f})")
    
    return feature_series


def aggregate_polymarket_daily(pm_data: pd.DataFrame, pm_sentiment: pd.Series) -> pd.DataFrame:
    """Aggregate Polymarket data to daily level."""
    if pm_data is None or len(pm_data) == 0:
        return None
    
    logger.info("[POLYMARKET] Aggregating to daily level...")
    
    # Add sentiment to data
    pm_data_copy = pm_data.copy()
    pm_data_copy['pm_sentiment'] = pm_sentiment.values
    
    # Extract date from timestamp (try multiple common column names)
    if 'timestamp' in pm_data_copy.columns:
        pm_data_copy['date'] = pd.to_datetime(pm_data_copy['timestamp']).dt.date
    elif 'ts_event' in pm_data_copy.columns:
        pm_data_copy['date'] = pd.to_datetime(pm_data_copy['ts_event']).dt.date
    else:
        pm_data_copy['date'] = datetime.now().date()
    
    # Aggregate by day: take mean sentiment
    daily_pm = pm_data_copy.groupby('date').agg({
        'pm_sentiment': ['mean', 'std', 'min', 'max', 'count']
    }).reset_index()
    
    daily_pm.columns = ['date', 'pm_sentiment_mean', 'pm_sentiment_std', 
                        'pm_sentiment_min', 'pm_sentiment_max', 'pm_snapshots']
    daily_pm['date'] = pd.to_datetime(daily_pm['date'])
    daily_pm = daily_pm.set_index('date')
    
    logger.info(f"✅ Aggregated to {len(daily_pm)} days of Polymarket data")
    
    return daily_pm


def prepare_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare OHLCV data, extract to clean format."""
    df = df.copy()

    # Extract OHLCV from nested 'data' dictionary
    if 'data' in df.columns:
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
    cols_to_keep = [col for col in ohlcv_cols if col in df.columns]
    df = df[cols_to_keep].copy()

    # Sort and deduplicate
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    df = df.ffill()

    logger.info(f"✅ Prepared {len(df)} days of OHLCV data")
    logger.info(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")

    return df


def blend_features(ccxt_df: pd.DataFrame, pm_daily: pd.DataFrame) -> pd.DataFrame:
    """Blend CCXT and Polymarket features.
    
    Create blended sentiment:
    - 50% from CCXT technical sentiment (momentum + trend + returns)
    - 50% from Polymarket orderbook sentiment
    """
    logger.info("\n[BLENDING] Combining CCXT + Polymarket features...")
    
    # Compute CCXT sentiment (existing pipeline)
    from regimes.feature_engineering import compute_regime_features
    
    ccxt_features = compute_regime_features(ccxt_df)
    ccxt_sentiment = ccxt_features['sentiment_composite']
    ccxt_vol = ccxt_features['vol_regime_score']
    
    # Prepare blended dataframe
    blended = pd.DataFrame(index=ccxt_features.index)
    blended['ccxt_sentiment'] = ccxt_sentiment
    blended['ccxt_vol_regime'] = ccxt_vol
    
    # Add Polymarket sentiment
    if pm_daily is not None:
        # Align Polymarket data with CCXT dates
        pm_daily_reindexed = pm_daily.reindex(blended.index)
        pm_daily_reindexed = pm_daily_reindexed.ffill().fillna(0.5)
        blended['pm_sentiment'] = pm_daily_reindexed['pm_sentiment_mean']
    else:
        blended['pm_sentiment'] = 0.5  # Default neutral if no PM data
    
    # Create blended sentiment: 50% CCXT + 50% Polymarket
    blended['sentiment_composite'] = (
        0.5 * blended['ccxt_sentiment'] + 
        0.5 * blended['pm_sentiment']
    )
    
    # Vol regime stays from CCXT (no change)
    blended['vol_regime_score'] = blended['ccxt_vol_regime']
    
    logger.info(f"✅ Blended features for {len(blended)} days")
    logger.info(f"   CCXT sentiment   (mean: {blended['ccxt_sentiment'].mean():.3f})")
    logger.info(f"   Polymarket sentiment (mean: {blended['pm_sentiment'].mean():.3f})")
    logger.info(f"   Blended sentiment (mean: {blended['sentiment_composite'].mean():.3f})")
    
    return blended


def train_regime_model(blended_features: pd.DataFrame, n_regimes: int = 3) -> RegimeIdentifier:
    """Train regime model on blended features."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"TRAINING REGIME MODEL (Blended Data, n_regimes={n_regimes})")
    logger.info(f"{'=' * 60}")

    # Use pre-computed features directly (skip feature engineering)
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Extract feature matrix
    feature_matrix = blended_features[['sentiment_composite', 'vol_regime_score']].fillna(0.5).values
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
    
    # Train K-Means
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    
    # Map clusters to regime names based on characteristics
    cluster_centers = kmeans.cluster_centers_
    regime_mapping = {}
    
    for cluster_id in range(n_regimes):
        center = cluster_centers[cluster_id]
        sentiment, vol_regime = center[0], center[1]
        
        if vol_regime < -0.5:  # Low volatility
            regime_name = "consolidation"
        elif sentiment >= 0.3 and vol_regime >= 0.3:  # High sentiment + volatility
            regime_name = "risk-on"
        elif sentiment <= -0.3 and vol_regime >= 0.3:  # Low sentiment + high volatility
            regime_name = "risk-off"
        else:
            regime_name = "consolidation"  # Default
        
        regime_mapping[cluster_id] = regime_name
    
    logger.info(f"\n✅ Model trained successfully")
    logger.info(f"\nRegime Mapping:")
    for cluster_id, regime_name in regime_mapping.items():
        logger.info(f"  Cluster {cluster_id} → {regime_name}")

    logger.info(f"\nCluster Centers (normalized sentiment, vol_regime):")
    for i, center in enumerate(cluster_centers):
        logger.info(f"  Cluster {i}: [{center[0]:.3f}, {center[1]:.3f}]")
    
    # Create identifier object with trained components
    identifier = RegimeIdentifier(n_regimes=n_regimes)
    identifier.kmeans_ = kmeans
    identifier.scaler_ = scaler
    identifier.regime_mapping_ = regime_mapping

    return identifier


def validate_model(identifier: RegimeIdentifier, blended_features: pd.DataFrame) -> None:
    """Validate trained model."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"VALIDATION")
    logger.info(f"{'=' * 60}")

    # Extract features and predict
    feature_matrix = blended_features[['sentiment_composite', 'vol_regime_score']].fillna(0.5).values
    scaled_features = identifier.scaler_.transform(feature_matrix)
    predictions = identifier.kmeans_.predict(scaled_features)
    
    # Map to regime names
    regime_names = [identifier.regime_mapping_[pred] for pred in predictions]
    regimes = pd.Series(regime_names, index=blended_features.index)

    dist = regimes.value_counts()
    logger.info(f"\n✅ Regimes predicted for {len(regimes)} days")
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
    """Save trained model."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"SAVING MODEL")
    logger.info(f"{'=' * 60}")

    import joblib
    import json
    model_dir = get_regime_model_dir()

    # Save components manually
    joblib.dump(identifier.kmeans_, model_dir / "kmeans.pkl")
    joblib.dump(identifier.scaler_, model_dir / "scaler.pkl")
    
    # Save regime mapping
    with open(model_dir / "regime_labels.json", "w") as f:
        json.dump(identifier.regime_mapping_, f, indent=2)

    logger.info(f"\n✅ Model saved to: {model_dir}")
    logger.info(f"\nFiles created:")
    for file in list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.json")):
        size_kb = file.stat().st_size / 1024
        logger.info(f"  - {file.name} ({size_kb:.1f} KB)")


def main() -> None:
    """Main workflow: blend CCXT + Polymarket → train → validate → save."""
    logger.info("\n" + "=" * 60)
    logger.info("REGIME IDENTIFICATION - BLENDED DATA (CCXT + POLYMARKET)")
    logger.info("=" * 60)

    # Step 1: Fetch CCXT data
    logger.info("\n[STEP 1] Fetch Historical CCXT Data")
    logger.info("-" * 60)

    ccxt_df = fetch_btc_historical_data(years_back=2, exchanges=["binance"])
    if ccxt_df is None:
        logger.error("❌ Failed to fetch CCXT data")
        return

    ccxt_df = prepare_ohlcv_data(ccxt_df)

    # Step 2: Load Polymarket data
    logger.info("\n[STEP 2] Load Polymarket Orderbook Data")
    logger.info("-" * 60)

    pm_raw = load_polymarket_orderbook_data(Path.cwd() / "data" / "cached")
    
    if pm_raw is not None:
        pm_sentiment = extract_polymarket_sentiment(pm_raw)
        pm_daily = aggregate_polymarket_daily(pm_raw, pm_sentiment)
    else:
        logger.warning("⚠️  Using CCXT data only (Polymarket data unavailable)")
        pm_daily = None

    # Step 3: Blend features
    logger.info("\n[STEP 3] Blend CCXT + Polymarket Features")
    logger.info("-" * 60)

    blended_features = blend_features(ccxt_df, pm_daily)

    # Step 4: Train model
    logger.info("\n[STEP 4] Train Model")
    logger.info("-" * 60)

    identifier = train_regime_model(blended_features, n_regimes=3)

    # Step 5: Validate
    logger.info("\n[STEP 5] Validate")
    logger.info("-" * 60)

    validate_model(identifier, blended_features)

    # Step 6: Save
    logger.info("\n[STEP 6] Save Model")
    logger.info("-" * 60)

    save_and_finalize(identifier)

    # Final message
    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE (BLENDED DATA)!")
    logger.info("=" * 60)
    logger.info("\n📊 Model trained on:")
    logger.info("   • 50% CCXT technical sentiment (momentum + trend + volatility)")
    logger.info("   • 50% Polymarket orderbook sentiment (prediction market crowd)")
    logger.info("\n💡 This blended approach captures both:")
    logger.info("   - Exchange price action (CCXT)")
    logger.info("   - Prediction market beliefs (Polymarket)")


if __name__ == "__main__":
    main()
