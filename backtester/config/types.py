from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime


@dataclass(frozen=True)
class RunMetadata:
    """Metadata required to reproduce and audit a backtest run."""

    run_id: str
    created_at: datetime
    config_snapshot: dict[str, object]
    config_hash: str
    git_commit: str | None
    data_window: dict[str, str | None]
    cache_signature: str
    mode: str


@dataclass(frozen=True)
class ValidationPolicy:
    """Schema validation policy for market event input handling."""

    quarantine_invalid_rows: bool = False
    allowed_event_types: tuple[str, ...] = (
        "book",
        "price_change",
        "market_resolved",
        "last_trade_price",
    )
    require_token_for_events: tuple[str, ...] = (
        "book",
        "price_change",
        "last_trade_price",
    )


@dataclass(frozen=True)
class FeatureGatePolicy:
    """Feature quality gate policy for strict or tolerant execution modes."""

    null_fraction_max: float = 0.2
    mid_price_min: float = 0.0
    mid_price_max: float = 1.0
    spread_min: float = 0.0
    spread_max: float = 1.0
    block_on_post_resolution_features: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for hold-to-resolution backtests."""

    mode: str = "strict"
    shares: float = 1.0
    fee_rate: float = 0.072
    fees_enabled: bool = True
    fee_precision: int = 5
    min_fee: float = 0.00001
    confidence_threshold: float = 0.95
    resolution_repair_dry_run: bool = False
    validation_policy: ValidationPolicy = field(default_factory=ValidationPolicy)
    feature_gate_policy: FeatureGatePolicy = field(default_factory=FeatureGatePolicy)
    cache_schema_version: str = "v1"
    cache_computation_signature: str = "orderbook_core_v1"

    # Workstream C execution realism fields.
    sizing_policy: str = "fixed_shares"
    sizing_fixed_notional: float | None = None
    sizing_risk_budget_pct: float = 0.02
    sizing_volatility_target: float = 0.02
    sizing_volatility_lookback: int = 20
    sizing_kelly_fraction_cap: float = 0.25
    fill_model: str = "depth_aware"
    fill_allow_partial: bool = True
    fill_walk_the_book: bool = True
    fill_slippage_factor: float = 1.0
    order_lifecycle_enabled: bool = False
    order_ttl_seconds: int | None = None
    order_allow_amendments: bool = False
    order_max_amendments: int = 0
    max_notional_per_market: float | None = None
    max_gross_exposure: float | None = None
    available_capital: float | None = None

    # Workstream D risk-gate shell fields.
    risk_max_drawdown_pct: float | None = None
    risk_max_daily_loss: float | None = None
    risk_max_concentration_pct: float | None = None
    risk_max_active_positions: int | None = None
    risk_max_gross_exposure: float | None = None

    # Runtime observability controls.
    enable_progress_bars: bool = False
    metrics_logging_enabled: bool = True
    metrics_log_every_n_markets: int = 200

    # Memory/performance controls for large batch runs.
    retain_full_feature_frames: bool = True
    retain_strategy_signals: bool = True
    retain_market_events: bool = True

    # Action selection controls for explicit UP/DOWN token actions.
    action_selection_lookahead_seconds: int = 0


@dataclass
class BacktestRunResult:
    """Structured result from a high-level backtest run."""

    metadata: RunMetadata
    market_events: pd.DataFrame
    features: pd.DataFrame
    resolution_frame: pd.DataFrame
    resolution_diagnostics: pd.DataFrame
    strategy_signals: dict[str, pd.DataFrame] = field(default_factory=dict)
    order_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    data_quality_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_health: pd.DataFrame = field(default_factory=pd.DataFrame)
    error_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    settlement_repair_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics_by_market: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics_by_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    cache_metadata: dict[str, object] = field(default_factory=dict)
    artifact_manifest: dict[str, object] = field(default_factory=dict)
