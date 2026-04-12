#!/usr/bin/env python
"""Export daily regime labels and confidence scores to CSV for model finetuning."""

import logging
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import numpy as np

from regimes.identifier import RegimeIdentifier
from data.historical import fetch_historical_data, OHLCVParams
from utils.paths import get_workspace_root

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_regime_csv():
    """Export daily regime labels and confidence scores to CSV.
    
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
    
    df = fetch_historical_data(params)
    
    # Flatten OHLCV data
    flat_data = []
    for _, row in df.iterrows():
        flat_row = row.to_dict()
        if isinstance(flat_row.get('data'), dict):
            flat_row.update(flat_row.pop('data'))
        flat_data.append(flat_row)
    
    df = pd.DataFrame(flat_data)
    df.set_index('ts_event', inplace=True)
    
    logger.info(f"Loaded {len(df)} days of OHLCV data")
    
    # Load the trained regime model
    regime_model_dir = "regimes/models"
    identifier = RegimeIdentifier.load(regime_model_dir)
    
    # Predict regimes for all days
    regimes, confidence_scores = identifier.predict_with_confidence(df)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'date': df.index.date,
        'regime': regimes.values,
        'confidence': confidence_scores,
    })
    
    # Add price data for context
    output_df['close'] = df['close'].values
    output_df['volume'] = df['volume'].values
    
    # Only add returns if it exists in the dataframe
    if 'returns' in df.columns:
        output_df['returns'] = df['returns'].values
    
    # Reorder columns for better readability
    available_cols = ['date', 'regime', 'confidence', 'close']
    if 'returns' in output_df.columns:
        available_cols.append('returns')
    available_cols.append('volume')
    
    output_df = output_df[available_cols]
    
    # Save to CSV
    csv_dir = Path(get_workspace_root()) / "reports"
    csv_dir.mkdir(exist_ok=True)
    
    csv_file = csv_dir / f"regime_data_{start.date()}_to_{end.date()}.csv"
    
    output_df.to_csv(csv_file, index=False)
    
    logger.info(f"Exported {len(output_df)} records to CSV")
    logger.info(f"CSV file saved to: {csv_file}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print(f"REGIME DATA EXPORT SUMMARY")
    print(f"{'='*80}\n")
    print(f"File: {csv_file}")
    print(f"Total records: {len(output_df)}")
    print(f"Date range: {output_df['date'].min()} to {output_df['date'].max()}")
    
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
    
    print(f"\n{'='*80}\n")
    
    return csv_file


if __name__ == "__main__":
    csv_file = export_regime_csv()
    print(f"✓ CSV data ready for alpha strategy finetuning: {csv_file}")
