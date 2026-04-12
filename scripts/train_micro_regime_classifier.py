#!/usr/bin/env python
"""Train and export micro regime classifier with zero look-ahead bias.

This script orchestrates the complete pipeline:
1. Load high-frequency OHLCV data
2. Resample to 5-minute bars
3. Compute rolling features (30-min window)
4. Train K-Means model on training period
5. Predict on inference period
6. Export CSV with regime labels and confidence scores
7. Validate no look-ahead bias

Usage:
    python scripts/train_micro_regime_classifier.py
"""

import logging
from datetime import datetime, UTC
from pathlib import Path

import pandas as pd
import numpy as np

from regimes.micro.data_loader import load_and_resample_5min
from regimes.micro.feature_engineering import (
    compute_micro_rolling_features,
    validate_no_lookahead,
)
from regimes.micro.identifier import MicroRegimeIdentifier
from regimes.micro.export import export_micro_regime_csv
from regimes.config import (
    MICRO_SYMBOL,
    MICRO_EXCHANGES,
    MICRO_TRAINING_START,
    MICRO_TRAINING_END,
    MICRO_INFERENCE_START,
    MICRO_INFERENCE_END,
    MICRO_ROLLING_WINDOW_SAMPLES,
    MICRO_FEATURE_COLS,
)
from utils.paths import get_workspace_root

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete micro regime classifier pipeline."""

    logger.info("="*80)
    logger.info("MICRO REGIME CLASSIFIER - TRAINING & INFERENCE PIPELINE")
    logger.info("="*80)

    workspace_root = get_workspace_root()
    model_dir = workspace_root / "regimes" / "models" / "micro"
    reports_dir = workspace_root / "reports"

    # =========================================================================
    # STEP 1: Load and prepare TRAINING data
    # =========================================================================
    logger.info(f"\n[STEP 1] Loading training data ({MICRO_TRAINING_START} to {MICRO_TRAINING_END})")

    try:
        df_train = load_and_resample_5min(
            symbol=MICRO_SYMBOL,
            start_date=MICRO_TRAINING_START,
            end_date=MICRO_TRAINING_END,
            exchanges=MICRO_EXCHANGES,
        )
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise

    logger.info(f"Loaded {len(df_train)} 5-minute bars for training")

    # =========================================================================
    # STEP 2: Compute rolling features on TRAINING data
    # =========================================================================
    logger.info(f"\n[STEP 2] Computing rolling features (window={MICRO_ROLLING_WINDOW_SAMPLES} × 5-min)")

    df_train = compute_micro_rolling_features(
        df_train,
        window_samples=MICRO_ROLLING_WINDOW_SAMPLES,
    )

    # Remove rows with insufficient data (< window size)
    df_train_clean = df_train.iloc[MICRO_ROLLING_WINDOW_SAMPLES:].reset_index(drop=True)
    logger.info(f"After removing NaN window rows: {len(df_train_clean)} bars")

    # =========================================================================
    # STEP 3: Train K-Means model
    # =========================================================================
    logger.info(f"\n[STEP 3] Training K-Means model")

    identifier = MicroRegimeIdentifier(
        feature_cols=MICRO_FEATURE_COLS,
    )

    training_summary = identifier.fit(df_train_clean)
    logger.info(f"Training summary: {training_summary}")

    # Save trained model
    model_dir.mkdir(parents=True, exist_ok=True)
    identifier.save(model_dir)
    logger.info(f"Model saved to {model_dir}")

    # =========================================================================
    # STEP 4: Load and prepare INFERENCE data
    # =========================================================================
    logger.info(f"\n[STEP 4] Loading inference data ({MICRO_INFERENCE_START} to {MICRO_INFERENCE_END})")

    try:
        df_inference = load_and_resample_5min(
            symbol=MICRO_SYMBOL,
            start_date=MICRO_INFERENCE_START,
            end_date=MICRO_INFERENCE_END,
            exchanges=MICRO_EXCHANGES,
        )
    except Exception as e:
        logger.error(f"Failed to load inference data: {e}")
        raise

    logger.info(f"Loaded {len(df_inference)} 5-minute bars for inference")

    # =========================================================================
    # STEP 5: Compute rolling features on INFERENCE data
    # =========================================================================
    logger.info(f"\n[STEP 5] Computing rolling features on inference data")

    df_inference = compute_micro_rolling_features(
        df_inference,
        window_samples=MICRO_ROLLING_WINDOW_SAMPLES,
    )

    # Remove rows with insufficient data
    df_inference_clean = df_inference.iloc[MICRO_ROLLING_WINDOW_SAMPLES:].reset_index(drop=True)
    logger.info(f"After removing NaN window rows: {len(df_inference_clean)} bars")

    # =========================================================================
    # STEP 6: Predict regimes + confidence
    # =========================================================================
    logger.info(f"\n[STEP 6] Predicting regimes and confidence scores")

    regimes, confidences = identifier.predict_with_confidence(df_inference_clean)

    logger.info(f"Predicted {len(regimes)} regimes")
    logger.info(f"Regime distribution:\n{regimes.value_counts()}")
    logger.info(f"Confidence statistics:\n{confidences.describe()}")

    # =========================================================================
    # STEP 7: Validate NO look-ahead bias
    # =========================================================================
    logger.info(f"\n[STEP 7] Validating NO look-ahead bias")

    no_lookahead_valid = validate_no_lookahead(
        df_inference_clean,
        window_samples=MICRO_ROLLING_WINDOW_SAMPLES,
        sample_rows=10,
    )

    if not no_lookahead_valid:
        logger.error("LOOK-AHEAD BIAS DETECTED - ABORTING")
        raise RuntimeError("Look-ahead bias validation failed")

    # =========================================================================
    # STEP 8: Export to CSV
    # =========================================================================
    logger.info(f"\n[STEP 8] Exporting to CSV")

    # Generate output filename
    start_date = pd.to_datetime(MICRO_INFERENCE_START).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(MICRO_INFERENCE_END).strftime("%Y-%m-%d")
    output_filename = f"micro_regimes_{start_date}_to_{end_date}.csv"
    output_path = reports_dir / output_filename

    csv_path = export_micro_regime_csv(
        df_inference_clean,
        regimes,
        confidences,
        output_path,
    )

    logger.info(f"CSV exported to {csv_path}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"\nTraining Period:  {MICRO_TRAINING_START} to {MICRO_TRAINING_END}")
    logger.info(f"  Samples:        {len(df_train_clean)} 5-min bars")
    logger.info(f"  Model saved to: {model_dir}")
    logger.info(f"\nInference Period: {MICRO_INFERENCE_START} to {MICRO_INFERENCE_END}")
    logger.info(f"  Samples:        {len(df_inference_clean)} 5-min bars")
    logger.info(f"  Output CSV:     {output_path}")
    logger.info(f"\nRegimes Distribution:")
    for regime, count in regimes.value_counts().items():
        pct = 100 * count / len(regimes)
        logger.info(f"  {regime:20s}: {count:6d} ({pct:5.1f}%)")
    logger.info(f"\nConfidence Statistics:")
    logger.info(f"  Mean:   {confidences.mean():.1%}")
    logger.info(f"  Median: {confidences.median():.1%}")
    logger.info(f"  Min:    {confidences.min():.1%}")
    logger.info(f"  Max:    {confidences.max():.1%}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
