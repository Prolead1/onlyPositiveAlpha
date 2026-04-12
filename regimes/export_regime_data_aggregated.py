#!/usr/bin/env python
"""Export daily regime labels with properly aggregated exchange data to CSV."""

import logging
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import numpy as np

from regimes.identifier import RegimeIdentifier
from regimes.feature_engineering import compute_regime_features, get_feature_matrix
from data.historical import fetch_historical_data, OHLCVParams
from utils.paths import get_workspace_root

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_regime_csv_aggregated():
    """Export daily regime labels with properly aggregated data to CSV.
    
    Aggregates OHLCV data from multiple exchanges by:
    - Taking volume-weighted average of close prices
    - Summing volumes across exchanges
    - Using median of open, high, low prices
    
    Returns
    -------
    Path
        Path to the exported CSV file.
    """
    
    # Fetch 3 years of historical data (2023-01-01 to 2026-01-01)
    start = datetime(2023, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 1, tzinfo=UTC)
    
    logger.info(f"Fetching data from {start.date()} to {end.date()}")
    
    params = OHLCVParams(
        symbol="BTC/USDT",
        start_ms=int(start.timestamp() * 1000),
        end_ms=int(end.timestamp() * 1000),
        timeframe="1d",
        limit=1000,
    )
    
    df_raw = fetch_historical_data(params)
    
    # Flatten OHLCV data
    flat_data = []
    for _, row in df_raw.iterrows():
        flat_row = row.to_dict()
        if isinstance(flat_row.get('data'), dict):
            flat_row.update(flat_row.pop('data'))
        flat_data.append(flat_row)
    
    df_raw = pd.DataFrame(flat_data)
    
    logger.info(f"Loaded {len(df_raw)} raw records from multiple exchanges")
    logger.info(f"Unique dates: {df_raw['ts_event'].dt.date.nunique()}")
    
    # Extract date (without time)
    df_raw['date'] = df_raw['ts_event'].dt.date
    
    # Aggregate by date using volume-weighting
    agg_data = []
    for date, group in df_raw.groupby('date'):
        # Prices: volume-weighted average for close
        total_volume = group['volume'].sum()
        if total_volume > 0:
            weighted_close = (group['close'] * group['volume']).sum() / total_volume
        else:
            weighted_close = group['close'].mean()
        
        # Use median for OHLH (more robust to outliers)
        agg_row = {
            'date': date,
            'open': group['open'].median(),
            'high': group['high'].median(),
            'low': group['low'].median(),
            'close': weighted_close,
            'volume': total_volume,
            'num_exchanges': group['source'].nunique(),
        }
        agg_data.append(agg_row)
    
    df = pd.DataFrame(agg_data).sort_values('date').reset_index(drop=True)
    df.set_index('date', inplace=True)
    
    logger.info(f"Aggregated to {len(df)} unique daily records")
    
    # Load the trained regime model
    regime_model_dir = "regimes/models"
    identifier = RegimeIdentifier.load(regime_model_dir)
    
    # Compute regime features on aggregated data
    df = compute_regime_features(df)
    
    # Predict regimes
    regimes, confidence_scores = identifier.predict_with_confidence(df)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'date': df.index,
        'regime': regimes.values,
        'confidence': np.round(confidence_scores, 4),  # Round to 4 decimals
        'close': df['close'].values,
        'volume': df['volume'].values,
        'num_exchanges': df['num_exchanges'].values,
    })
    
    # Save to CSV
    csv_dir = Path(get_workspace_root()) / "reports"
    csv_dir.mkdir(exist_ok=True)
    
    csv_file = csv_dir / f"regime_data_aggregated_{start.date()}_to_{end.date()}.csv"
    
    output_df.to_csv(csv_file, index=False)
    
    logger.info(f"Exported {len(output_df)} aggregated records to CSV")
    logger.info(f"CSV file saved to: {csv_file}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"AGGREGATED REGIME DATA EXPORT SUMMARY")
    print(f"{'='*80}\n")
    print(f"File: {csv_file}")
    print(f"Total records: {len(output_df)}")
    print(f"Date range: {output_df['date'].min()} to {output_df['date'].max()}")
    print(f"All dates should be unique now: {output_df['date'].nunique() == len(output_df)}")
    
    print(f"\nRegime distribution:")
    regime_counts = output_df['regime'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        pct = 100 * count / len(output_df)
        print(f"  {regime:15} : {count:4d} days ({pct:5.1f}%)")
    
    print(f"\nConfidence statistics:")
    for regime in ['risk-on', 'consolidation', 'risk-off']:
        mask = output_df['regime'] == regime
        if mask.any():
            conf_values = output_df.loc[mask, 'confidence']
            print(f"\n  {regime.upper()}:")
            print(f"    Mean:   {conf_values.mean():.1%}")
            print(f"    Median: {conf_values.median():.1%}")
            print(f"    Std:    {conf_values.std():.1%}")
            print(f"    Min:    {conf_values.min():.1%}")
            print(f"    Max:    {conf_values.max():.1%}")
    
    print(f"\nExchange coverage (for sanity check):")
    print(f"  Avg exchanges per date: {output_df['num_exchanges'].mean():.2f}")
    print(f"  Min exchanges per date: {output_df['num_exchanges'].min()}")
    print(f"  Max exchanges per date: {output_df['num_exchanges'].max()}")
    
    print(f"\n{'='*80}\n")
    
    return csv_file


if __name__ == "__main__":
    csv_file = export_regime_csv_aggregated()
    print(f"✓ Aggregated CSV ready for alpha strategy finetuning: {csv_file}")
