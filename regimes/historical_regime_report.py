#!/usr/bin/env python
"""Generate a comprehensive historical regime report."""

import logging
from datetime import datetime, UTC, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO

from regimes.identifier import RegimeIdentifier
from data.historical import fetch_historical_data, OHLCVParams
from utils.paths import get_workspace_root

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_report():
    """Generate historical regime report and return as string."""
    
    report = StringIO()
    
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
    
    report.write(f"\n{'='*80}\n")
    report.write(f"HISTORICAL REGIME REPORT: {df.index[0].date()} to {df.index[-1].date()}\n")
    report.write(f"{'='*80}\n")
    report.write(f"Total days analyzed: {len(df)}\n\n")
    
    # Load the trained regime model
    regime_model_dir = "regimes/models"
    identifier = RegimeIdentifier.load(regime_model_dir)
    
    # Predict regimes for all days
    regimes, confidence_scores = identifier.predict_with_confidence(df)
    
    report.write(f"{'REGIME DISTRIBUTION:':-^80}\n\n")
    
    # Overall distribution
    regime_counts = regimes.value_counts()
    regime_pcts = regimes.value_counts(normalize=True) * 100
    
    for regime in ['risk-on', 'consolidation', 'risk-off']:
        count = regime_counts.get(regime, 0)
        pct = regime_pcts.get(regime, 0)
        bar_length = int(pct / 2)
        bar = "█" * bar_length
        report.write(f"  {regime:15} : {count:4d} days ({pct:5.1f}%)  {bar}\n")
    
    
    report.write(f"\n{'REGIME TRANSITIONS:':-^80}\n\n")
    
    # Find transitions
    transitions = []
    for i in range(1, len(regimes)):
        if regimes.iloc[i] != regimes.iloc[i-1]:
            transitions.append({
                'date': regimes.index[i].date(),
                'from': regimes.iloc[i-1],
                'to': regimes.iloc[i],
                'confidence_before': confidence_scores[i-1],
                'confidence_after': confidence_scores[i],
            })
    
    report.write(f"Total transitions: {len(transitions)}\n\n")
    report.write("Recent transitions (last 15):\n")
    for trans in transitions[-15:]:
        report.write(f"  {trans['date']}: {trans['from']:15} → {trans['to']:15} "
                     f"(conf: {trans['confidence_before']:.1%} → {trans['confidence_after']:.1%})\n")
    
    report.write(f"\n{'-'*78}REGIME PERIODS (LONGEST STREAKS){'-'*78}\n\n")
    
    # Calculate duration of each regime period
    current_regime = regimes.iloc[0]
    regime_start_idx = 0
    regime_periods = []
    
    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime:
            regime_periods.append({
                'regime': current_regime,
                'start': regimes.index[regime_start_idx].date(),
                'end': regimes.index[i-1].date(),
                'days': i - regime_start_idx,
                'avg_confidence': confidence_scores[regime_start_idx:i].mean(),
                'min_confidence': confidence_scores[regime_start_idx:i].min(),
            })
            current_regime = regimes.iloc[i]
            regime_start_idx = i
    
    # Add last period
    regime_periods.append({
        'regime': current_regime,
        'start': regimes.index[regime_start_idx].date(),
        'end': regimes.index[-1].date(),
        'days': len(regimes) - regime_start_idx,
        'avg_confidence': confidence_scores[regime_start_idx:].mean(),
        'min_confidence': confidence_scores[regime_start_idx:].min(),
    })
    
    # Sort by duration descending
    regime_periods_sorted = sorted(regime_periods, key=lambda x: x['days'], reverse=True)
    
    for idx, period in enumerate(regime_periods_sorted[:20], 1):
        report.write(f"{idx:2d}. {period['regime']:15} : {period['start']} to {period['end']} "
                     f"({period['days']:3d} days) "
                     f"avg_conf={period['avg_confidence']:.1%}\n")
    
    report.write(f"\n{'-'*78}REGIME STATISTICS{'-'*78}\n\n")
    
    for regime_type in ['risk-on', 'consolidation', 'risk-off']:
        periods = [p for p in regime_periods if p['regime'] == regime_type]
        if periods:
            total_days = sum(p['days'] for p in periods)
            avg_duration = total_days / len(periods)
            avg_conf = np.mean([p['avg_confidence'] for p in periods])
            min_conf = min([p['min_confidence'] for p in periods])
            max_duration = max([p['days'] for p in periods])
            
            report.write(f"{regime_type.upper()}:\n")
            report.write(f"  Number of periods:    {len(periods):3d}\n")
            report.write(f"  Total days:           {total_days:3d} ({total_days/len(df)*100:.1f}%)\n")
            report.write(f"  Avg duration/period:  {avg_duration:5.1f} days\n")
            report.write(f"  Longest streak:       {max_duration:3d} days\n")
            report.write(f"  Avg confidence:       {avg_conf:5.1%}\n")
            report.write(f"  Min confidence:       {min_conf:5.1%}\n\n")
    
    report.write(f"{'-'*78}CONFIDENCE DISTRIBUTION{'-'*78}\n\n")
    
    for regime in ['risk-on', 'consolidation', 'risk-off']:
        mask = regimes == regime
        if mask.any():
            conf_values = confidence_scores[mask]
            report.write(f"{regime.upper()}:\n")
            report.write(f"  Mean:         {conf_values.mean():.1%}\n")
            report.write(f"  Median:       {np.median(conf_values):.1%}\n")
            report.write(f"  Std Dev:      {conf_values.std():.1%}\n")
            report.write(f"  Min:          {conf_values.min():.1%}\n")
            report.write(f"  Max:          {conf_values.max():.1%}\n")
            report.write(f"  25th pct:     {np.percentile(conf_values, 25):.1%}\n")
            report.write(f"  75th pct:     {np.percentile(conf_values, 75):.1%}\n\n")
    
    report.write(f"{'='*80}\n\n")
    
    return report.getvalue(), start, end


def main():
    """Generate historical regime report and save to file."""
    
    # Generate report
    report_content, start, end = generate_report()
    
    # Print to console
    print(report_content)
    
    # Save to file
    report_dir = Path(get_workspace_root()) / "reports"
    report_dir.mkdir(exist_ok=True)
    
    report_file = report_dir / f"regime_report_{start.date()}_to_{end.date()}.txt"
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to: {report_file}")
    print(f"\n{'='*80}")
    print(f"✓ Report saved to: {report_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

