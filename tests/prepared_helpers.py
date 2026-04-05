from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from backtester.runner import BacktestRunner


def write_prepared_manifest(
    *,
    tmp_path: Path,
    runner: BacktestRunner,
    features: pd.DataFrame,
    market_events: pd.DataFrame,
    mapping_dir: Path,
) -> Path:
    """Materialize prepared feature and resolution artifacts for batched tests."""
    prepared_root = tmp_path / "prepared_backtest"
    feature_root = prepared_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)

    frame = features.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.sort_index()

    feature_files: list[str] = []
    grouped = frame.groupby(frame["market_id"].astype(str), sort=True)
    for market_id, market_frame in grouped:
        feature_path = feature_root / f"{market_id}.parquet"
        market_payload = market_frame.reset_index().rename(columns={"index": "ts_event"})
        if "ts_event" not in market_payload.columns:
            market_payload = market_payload.rename(columns={market_payload.columns[0]: "ts_event"})
        market_payload.to_parquet(feature_path, index=False)
        feature_files.append(str(feature_path))

    resolution_frame, _ = runner.load_and_validate_resolution(
        market_events,
        mapping_dir=mapping_dir,
        confidence_threshold=0.95,
        features=features,
        repair_dry_run=True,
    )
    resolution_path = prepared_root / "resolution" / "resolution_frame.parquet"
    resolution_path.parent.mkdir(parents=True, exist_ok=True)
    resolution_payload = resolution_frame.reset_index()
    resolution_payload.to_parquet(resolution_path, index=False)

    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "output_files": [],
                        "feature_output_files": feature_files,
                    }
                ],
                "resolution_output_file": str(resolution_path),
            }
        ),
        encoding="utf-8",
    )
    return manifest_path
