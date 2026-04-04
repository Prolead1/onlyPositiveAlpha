from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from backtester.config.types import BacktestConfig, FeatureGatePolicy, ValidationPolicy
from backtester.simulation.fees import calculate_taker_fee as calculate_taker_fee_model


def generate_spread_imbalance_signals(
    features: pd.DataFrame,
    *,
    imbalance_threshold: float,
    spread_tight_quantile: float,
    spread_wide_quantile: float,
) -> pd.DataFrame:
    """Add spread/imbalance signal columns to a feature frame."""
    required_cols = {"spread", "imbalance_1"}
    missing = [col for col in required_cols if col not in features.columns]
    if missing:
        msg = f"Missing required columns for signal generation: {missing}"
        raise ValueError(msg)

    enriched = features.copy()
    enriched["imbalance_signal"] = 0
    enriched.loc[enriched["imbalance_1"] > imbalance_threshold, "imbalance_signal"] = 1
    enriched.loc[enriched["imbalance_1"] < -imbalance_threshold, "imbalance_signal"] = -1

    spread_values = enriched["spread"].dropna()
    if spread_values.empty:
        raise ValueError("Cannot build spread signals without non-null spread values")

    tight_cutoff = spread_values.quantile(spread_tight_quantile)
    wide_cutoff = spread_values.quantile(spread_wide_quantile)

    enriched["spread_signal"] = 0
    enriched.loc[enriched["spread"] <= tight_cutoff, "spread_signal"] = 1
    enriched.loc[enriched["spread"] >= wide_cutoff, "spread_signal"] = -1
    enriched.attrs["spread_tight_cutoff"] = float(tight_cutoff)
    enriched.attrs["spread_wide_cutoff"] = float(wide_cutoff)
    return enriched


def calculate_taker_fee(  # noqa: PLR0913
    price: float,
    *,
    shares: float,
    fee_rate: float,
    fees_enabled: bool = True,
    precision: int = 5,
    minimum_fee: float = 0.00001,
) -> float:
    """Calculate taker fee using the Polymarket fee model."""
    return calculate_taker_fee_model(
        price,
        shares=shares,
        fee_rate=fee_rate,
        fees_enabled=fees_enabled,
        precision=precision,
        minimum_fee=minimum_fee,
    )


def coerce_backtest_config(  # noqa: C901
    config: BacktestConfig | Mapping[str, object] | None,
) -> BacktestConfig:
    """Coerce a mapping into BacktestConfig while preserving defaults."""
    if config is None:
        return BacktestConfig()
    if isinstance(config, BacktestConfig):
        return config

    def _to_float(value: object, default: float) -> float:
        if value is None:
            return default
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return default

    def _to_int(value: object, default: int) -> int:
        if value is None:
            return default
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    def _to_bool(value: object, default: bool) -> bool:  # noqa: FBT001
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def _to_opt_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return None

    validation_cfg_raw = config.get("validation_policy")
    validation_policy = ValidationPolicy()
    if isinstance(validation_cfg_raw, Mapping):
        validation_policy = ValidationPolicy(
            quarantine_invalid_rows=_to_bool(
                validation_cfg_raw.get("quarantine_invalid_rows"),
                default=validation_policy.quarantine_invalid_rows,
            ),
            allowed_event_types=tuple(
                str(item)
                for item in validation_cfg_raw.get(
                    "allowed_event_types",
                    validation_policy.allowed_event_types,
                )
            ),
            require_token_for_events=tuple(
                str(item)
                for item in validation_cfg_raw.get(
                    "require_token_for_events",
                    validation_policy.require_token_for_events,
                )
            ),
        )

    feature_gate_cfg_raw = config.get("feature_gate_policy")
    feature_gate_policy = FeatureGatePolicy()
    if isinstance(feature_gate_cfg_raw, Mapping):
        feature_gate_policy = FeatureGatePolicy(
            null_fraction_max=_to_float(
                feature_gate_cfg_raw.get("null_fraction_max"),
                feature_gate_policy.null_fraction_max,
            ),
            mid_price_min=_to_float(
                feature_gate_cfg_raw.get("mid_price_min"),
                feature_gate_policy.mid_price_min,
            ),
            mid_price_max=_to_float(
                feature_gate_cfg_raw.get("mid_price_max"),
                feature_gate_policy.mid_price_max,
            ),
            spread_min=_to_float(
                feature_gate_cfg_raw.get("spread_min"),
                feature_gate_policy.spread_min,
            ),
            spread_max=_to_float(
                feature_gate_cfg_raw.get("spread_max"),
                feature_gate_policy.spread_max,
            ),
            block_on_post_resolution_features=_to_bool(
                feature_gate_cfg_raw.get("block_on_post_resolution_features"),
                default=feature_gate_policy.block_on_post_resolution_features,
            ),
        )

    mode_raw = str(config.get("mode", "strict")).strip().lower()
    mode = "tolerant" if mode_raw == "tolerant" else "strict"

    return BacktestConfig(
        mode=mode,
        shares=_to_float(config.get("shares", 1.0), 1.0),
        fee_rate=_to_float(config.get("fee_rate", 0.072), 0.072),
        fees_enabled=_to_bool(config.get("fees_enabled", True), default=True),
        fee_precision=_to_int(config.get("fee_precision", 5), 5),
        min_fee=_to_float(config.get("min_fee", 0.00001), 0.00001),
        confidence_threshold=_to_float(config.get("confidence_threshold", 0.95), 0.95),
        resolution_repair_dry_run=_to_bool(
            config.get("resolution_repair_dry_run", False),
            default=False,
        ),
        validation_policy=validation_policy,
        feature_gate_policy=feature_gate_policy,
        cache_schema_version=str(config.get("cache_schema_version", "v1")),
        cache_computation_signature=str(
            config.get("cache_computation_signature", "orderbook_core_v1")
        ),
        sizing_policy=str(config.get("sizing_policy", "fixed_shares")),
        sizing_fixed_notional=_to_opt_float(config.get("sizing_fixed_notional")),
        sizing_risk_budget_pct=_to_float(config.get("sizing_risk_budget_pct", 0.02), 0.02),
        sizing_volatility_target=_to_float(
            config.get("sizing_volatility_target", 0.02),
            0.02,
        ),
        sizing_volatility_lookback=_to_int(
            config.get("sizing_volatility_lookback", 20),
            20,
        ),
        sizing_kelly_fraction_cap=_to_float(
            config.get("sizing_kelly_fraction_cap", 0.25),
            0.25,
        ),
        fill_model=str(config.get("fill_model", "depth_aware")),
        fill_allow_partial=_to_bool(config.get("fill_allow_partial", True), default=True),
        fill_walk_the_book=_to_bool(config.get("fill_walk_the_book", True), default=True),
        fill_slippage_factor=_to_float(config.get("fill_slippage_factor", 1.0), 1.0),
        order_lifecycle_enabled=_to_bool(
            config.get("order_lifecycle_enabled", False),
            default=False,
        ),
        order_ttl_seconds=(
            _to_int(config.get("order_ttl_seconds"), 0)
            if config.get("order_ttl_seconds") is not None
            else None
        ),
        order_allow_amendments=_to_bool(
            config.get("order_allow_amendments", False),
            default=False,
        ),
        order_max_amendments=_to_int(config.get("order_max_amendments", 0), 0),
        max_notional_per_market=_to_opt_float(config.get("max_notional_per_market")),
        max_gross_exposure=_to_opt_float(config.get("max_gross_exposure")),
        available_capital=_to_opt_float(config.get("available_capital")),
        risk_max_drawdown_pct=_to_opt_float(config.get("risk_max_drawdown_pct")),
        risk_max_daily_loss=_to_opt_float(config.get("risk_max_daily_loss")),
        risk_max_concentration_pct=_to_opt_float(config.get("risk_max_concentration_pct")),
        risk_max_active_positions=(
            _to_int(config.get("risk_max_active_positions"), 0)
            if config.get("risk_max_active_positions") is not None
            else None
        ),
        risk_max_gross_exposure=_to_opt_float(config.get("risk_max_gross_exposure")),
        retain_full_feature_frames=_to_bool(
            config.get("retain_full_feature_frames", True),
            default=True,
        ),
        retain_strategy_signals=_to_bool(
            config.get("retain_strategy_signals", True),
            default=True,
        ),
        retain_market_events=_to_bool(
            config.get("retain_market_events", True),
            default=True,
        ),
        action_selection_lookahead_seconds=_to_int(
            config.get("action_selection_lookahead_seconds", 0),
            0,
        ),
    )


def normalize_strategy_output(  # noqa: C901
    *,
    features: pd.DataFrame,
    strategy_name: str,
    strategy_output: pd.Series | pd.DataFrame,
    signal_column: str = "signal",
) -> pd.DataFrame:
    """Normalize strategy output to runner simulation input schema."""
    required_cols = ["market_id", "token_id", "mid_price"]

    if isinstance(strategy_output, pd.Series):
        base = features[required_cols].copy()
        base[signal_column] = strategy_output
        normalized = base
    elif isinstance(strategy_output, pd.DataFrame):
        normalized = strategy_output.copy()
        if signal_column not in normalized.columns and "signal" in normalized.columns:
            normalized = normalized.rename(columns={"signal": signal_column})
        missing_required = [col for col in required_cols if col not in normalized.columns]
        if missing_required:
            if normalized.index.equals(features.index):
                for col in missing_required:
                    normalized[col] = features[col]
            else:
                msg = (
                    f"Strategy '{strategy_name}' output missing required "
                    f"columns: {missing_required}"
                )
                raise ValueError(msg)
    else:
        msg = (
            f"Strategy '{strategy_name}' must return pandas Series or DataFrame, "
            f"got {type(strategy_output).__name__}"
        )
        raise TypeError(msg)

    if signal_column not in normalized.columns:
        msg = f"Strategy '{strategy_name}' output missing signal column '{signal_column}'"
        raise ValueError(msg)

    normalized[signal_column] = pd.to_numeric(
        normalized[signal_column],
        errors="coerce",
    ).fillna(0.0)

    valid_action_sides = {"buy", "sell"}
    if "action_side" in normalized.columns:
        raw_action_side = normalized["action_side"].astype(str).str.strip().str.lower()
        normalized["action_side"] = raw_action_side.where(
            raw_action_side.isin(valid_action_sides),
            pd.NA,
        )
    else:
        normalized["action_side"] = pd.Series(index=normalized.index, dtype="object")

    derived_action_side = pd.Series("buy", index=normalized.index, dtype="object")
    derived_action_side = derived_action_side.where(normalized[signal_column] > 0, "sell")
    normalized["action_side"] = normalized["action_side"].fillna(derived_action_side)

    if "action_score" in normalized.columns:
        normalized["action_score"] = pd.to_numeric(
            normalized["action_score"],
            errors="coerce",
        )
    else:
        normalized["action_score"] = pd.Series(index=normalized.index, dtype="float64")
    normalized["action_score"] = normalized["action_score"].fillna(
        normalized[signal_column].abs()
    )

    for col in required_cols:
        if col not in normalized.columns:
            msg = f"Strategy '{strategy_name}' missing required column '{col}'"
            raise ValueError(msg)

    optional_execution_cols = [
        "spread",
        "ask_depth_1",
        "ask_depth_5",
        "bid_depth_1",
        "bid_depth_5",
    ]
    optional_diagnostic_cols = [
        "depth_pressure",
        "depth_pressure_rank",
        "imbalance_1",
        "spread_rank",
        "spread_narrow_rank",
        "imbalance_rank",
        "confidence_score",
        "liquidity",
        "signal_abs",
        "time_to_resolution_secs",
    ]
    keep_cols = [*required_cols, signal_column, "action_side", "action_score"]
    keep_cols.extend(
        col
        for col in optional_execution_cols
        if col in normalized.columns and col not in keep_cols
    )
    keep_cols.extend(
        col
        for col in optional_diagnostic_cols
        if col in normalized.columns and col not in keep_cols
    )

    subset = normalized[keep_cols].copy()
    subset = subset[subset[signal_column] != 0]
    if subset.empty:
        return subset

    subset = subset.sort_index()
    subset["market_id"] = subset["market_id"].astype(str)
    subset["token_id"] = subset["token_id"].astype(str)
    subset["action_side"] = subset["action_side"].astype(str).str.strip().str.lower()
    subset["action_score"] = pd.to_numeric(subset["action_score"], errors="coerce").fillna(
        subset[signal_column].abs()
    )
    subset = subset.dropna(subset=["mid_price"])
    return subset


__all__ = [
    "calculate_taker_fee",
    "coerce_backtest_config",
    "generate_spread_imbalance_signals",
    "normalize_strategy_output",
]
