from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from backtester.config.types import BacktestConfig
from backtester.runner import BacktestRunner
from tests.prepared_helpers import write_prepared_manifest


def _build_sample_run_inputs(
    tmp_path: Path,
    *,
    market_id: str = "cond_req",
    winner: str | None = "token_yes",
) -> tuple[BacktestRunner, Path, str, Path, Path]:
    storage_path = tmp_path / "pmxt"
    storage_path.mkdir(parents=True, exist_ok=True)

    runner = BacktestRunner(storage_path)
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    token_yes = "token_yes"
    token_no = "token_no"

    market_events = pd.DataFrame(
        [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_yes,
                "data": {
                    "asset_id": token_yes,
                    "bids": [{"price": "0.98", "size": "100"}],
                    "asks": [{"price": "0.99", "size": "100"}],
                },
            },
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": market_id,
                "token_id": token_no,
                "data": {
                    "asset_id": token_no,
                    "bids": [{"price": "0.01", "size": "100"}],
                    "asks": [{"price": "0.02", "size": "100"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=5),
                "event_type": "market_resolved",
                "market_id": market_id,
                "token_id": token_yes,
                "data": {
                    "winning_asset_id": winner,
                    "winning_outcome": "Yes",
                },
            },
        ]
    ).set_index("ts_event")
    features = runner.compute_orderbook_features_df(market_events)

    mapping_dir = tmp_path / "mapping"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = mapping_dir / "gamma_updown_markets_2024-01-01.json"
    mapping_path.write_text(
        json.dumps(
            {
                "sample-slug": {
                    "conditionId": market_id,
                    "resolvedAt": "2024-01-01T00:00:05Z",
                    "winningAssetId": winner,
                    "winningOutcome": "Yes",
                    "clobTokenIds": [token_yes, token_no],
                    "outcomePrices": ["1", "0"],
                    "feesEnabledMarket": True,
                }
            }
        ),
        encoding="utf-8",
    )

    manifest_path = write_prepared_manifest(
        tmp_path=tmp_path,
        runner=runner,
        features=features,
        market_events=market_events,
        mapping_dir=mapping_dir,
    )

    return runner, mapping_dir, token_yes, mapping_path, manifest_path


def _long_yes(token_yes: str):
    def _strategy(frame: pd.DataFrame) -> pd.Series:
        return (frame["token_id"].astype(str) == token_yes).astype(int)

    return _strategy


def test_settlement_provenance_and_repair_audit_dry_run(tmp_path: Path) -> None:
    runner, mapping_dir, token_yes, mapping_path, manifest_path = (
        _build_sample_run_inputs(
        tmp_path,
        market_id="cond_dry",
        winner=None,
        )
    )

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes(token_yes),
        strategy_name="long_yes",
        market_batch_size=1,
        config=BacktestConfig(
            shares=1.0,
            fee_rate=0.0,
            fees_enabled=False,
            resolution_repair_dry_run=True,
        ),
    )

    assert not result.trade_ledger.empty
    assert "settlement_source" in result.trade_ledger.columns
    assert result.trade_ledger.iloc[0]["settlement_source"] == "inferred"
    assert result.trade_ledger.iloc[0]["settlement_confidence"] >= 0.95

    assert result.settlement_repair_audit.empty

    updated_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert updated_payload["sample-slug"]["winningAssetId"] is None


def test_run_backtest_strategy_output_validation_errors(tmp_path: Path) -> None:
    runner, mapping_dir, _, _, manifest_path = _build_sample_run_inputs(tmp_path)

    def malformed(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"signal": [1, 1, 1]})

    with pytest.raises(ValueError, match="missing required columns"):
        runner.run_backtest(
            mapping_dir=mapping_dir,
            prepared_manifest_path=manifest_path,
            strategy=malformed,
            strategy_name="bad",
            market_batch_size=1,
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )


def test_run_backtest_failure_contracts(tmp_path: Path) -> None:
    runner, mapping_dir, token_yes, _, manifest_path = _build_sample_run_inputs(tmp_path)

    with pytest.raises(
        RuntimeError,
        match="No prepared feature markets found|Feature quality gates removed all feature rows",
    ):
        runner.run_backtest(
            mapping_dir=mapping_dir,
            prepared_manifest_path=manifest_path,
            strategy=_long_yes(token_yes),
            strategy_name="strict_empty",
            market_batch_size=1,
            prepared_feature_market_ids={"missing_market"},
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )

    with pytest.raises(ValueError, match="Legacy in-memory backtest inputs are no longer supported"):
        runner.run_backtest(
            mapping_dir=mapping_dir,
            strategy=_long_yes(token_yes),
            strategy_name="strict_empty_features",
            market_events=pd.DataFrame(),
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )


def test_run_sensitivity_scenarios_outputs_ranked_rows(tmp_path: Path) -> None:
    runner, mapping_dir, token_yes, _, manifest_path = _build_sample_run_inputs(tmp_path)

    result = runner.run_sensitivity_scenarios(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes(token_yes),
        strategy_name="sweep",
        market_batch_size=1,
        base_config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        parameter_sweeps={"shares": [1.0, 2.0]},
        stress_scenarios=["baseline", "fee_increase"],
    )

    assert len(result) == 4
    assert {"scenario_id", "parameter_set", "robustness_rank"}.issubset(result.columns)

    result_two = runner.run_sensitivity_scenarios(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes(token_yes),
        strategy_name="sweep",
        market_batch_size=1,
        base_config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        parameter_sweeps={"shares": [1.0, 2.0]},
        stress_scenarios=["baseline", "fee_increase"],
    )
    pd.testing.assert_series_equal(result["scenario_id"], result_two["scenario_id"])
    pd.testing.assert_series_equal(result["parameter_set"], result_two["parameter_set"])


def test_run_backtest_emits_market_and_regime_diagnostics(tmp_path: Path) -> None:
    runner, mapping_dir, token_yes, _, manifest_path = _build_sample_run_inputs(tmp_path)

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes(token_yes),
        strategy_name="diag",
        market_batch_size=1,
        config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
    )

    assert not result.diagnostics_by_market.empty
    assert not result.diagnostics_by_regime.empty
    run_id = result.metadata.run_id
    assert set(result.diagnostics_by_market["run_id"].unique()) == {run_id}
    assert set(result.diagnostics_by_regime["run_id"].unique()) == {run_id}


def test_write_run_artifact_package_completeness(tmp_path: Path) -> None:
    runner, mapping_dir, token_yes, _, manifest_path = _build_sample_run_inputs(tmp_path)

    result = runner.run_backtest(
        mapping_dir=mapping_dir,
        prepared_manifest_path=manifest_path,
        strategy=_long_yes(token_yes),
        strategy_name="artifact",
        market_batch_size=1,
        config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
    )

    package_dir = runner.write_run_artifact_package(
        result,
        output_dir=tmp_path / "artifacts",
        artifact_version="v1",
    )

    required_files = {
        "metadata.json",
        "config_snapshot.json",
        "cache_metadata.json",
        "plots_manifest.json",
        "artifact_manifest.json",
        "trade_ledger.csv",
        "order_ledger.csv",
        "backtest_summary.csv",
        "equity_curve.csv",
        "resolution_diagnostics.csv",
        "data_quality_report.csv",
        "feature_health.csv",
        "error_ledger.csv",
        "settlement_repair_audit.csv",
        "diagnostics_by_market.csv",
        "diagnostics_by_regime.csv",
    }
    package_files = {path.name for path in package_dir.glob("*")}
    assert required_files.issubset(package_files)

    manifest_payload = json.loads(
        (package_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_payload["artifact_version"] == "v1"
    assert manifest_payload["run_id"] == result.metadata.run_id


def test_notebook_smoke_contract_file_present() -> None:
    notebook_path = Path("alphas/spread_alpha.ipynb")
    assert notebook_path.exists()

    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    assert code_cells, "Notebook should contain executable code cells"

    notebook_code = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
    assert "run_backtest(" in notebook_code or "BacktestRunner(" in notebook_code
