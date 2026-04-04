from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import pandas as pd

from backtester.config.normalization import coerce_backtest_config
from backtester.simulation.fees import calculate_taker_fee
from backtester.simulation.risk_engine import RiskState

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from backtester.config.types import BacktestConfig
    from backtester.simulation.risk_engine import RiskEvaluator


class FillExecutionResult(TypedDict):
    """Shape of fill-model execution output."""

    filled_qty: float
    avg_fill_price: float
    slippage_bps: float
    reject_reason: str | None
    order_state: str


class PendingTrade(TypedDict):
    """Carry-over entry for a trade that has not yet been flushed to output."""

    row: Mapping[str, object]
    effective_notional: float
    net_pnl: float
    resolved_at: pd.Timestamp
    market_id: str
    token_id: str


if TYPE_CHECKING:
    class _BacktestSimulationHost(Protocol):
        def _to_float_or_nan(self, value: object) -> float: ...

        def _iter_market_groups_by_id(
            self,
            signal_frame: pd.DataFrame,
        ) -> Iterator[tuple[str, pd.DataFrame]]: ...

        def simulate_hold_to_resolution_backtest_incremental(  # noqa: PLR0913
            self,
            signal_frame: pd.DataFrame,
            resolution_frame: pd.DataFrame,
            *,
            strategy_name: str,
            signal_column: str = "signal",
            config: BacktestConfig | Mapping[str, object] | None = None,
            risk_events: list[dict[str, object]] | None = None,
            order_events: list[dict[str, object]] | None = None,
            continuation_state: SimulationContinuationState | None = None,
            finalize: bool = True,
        ) -> tuple[pd.DataFrame, SimulationContinuationState]: ...

        def _estimate_market_volatility(
            self,
            market_group: pd.DataFrame,
            *,
            entry_ts: pd.Timestamp,
            lookback: int,
        ) -> float | None: ...

        def _apply_sizing_policy(  # noqa: PLR0913
            self,
            *,
            signal_value: int,
            signal_abs: float,
            entry_price: float,
            market_group: pd.DataFrame,
            entry_ts: pd.Timestamp,
            cfg: BacktestConfig,
            gross_exposure_used: float,
            capital_remaining: float | None,
        ) -> tuple[float, str, float]: ...

        def _execute_fill_model(
            self,
            *,
            requested_qty: float,
            entry_price: float,
            direction: int,
            cfg: BacktestConfig,
            entry_row: pd.Series,
        ) -> FillExecutionResult: ...

        def _decompose_execution_costs(  # noqa: PLR0913
            self,
            *,
            fee_usdc: float,
            requested_qty: float,
            filled_qty: float,
            slippage_bps: float,
            avg_fill_price: float,
            entry_price: float,
        ) -> dict[str, float]: ...

        def _build_risk_evaluator(self, cfg: BacktestConfig) -> RiskEvaluator: ...

        def _build_order_id(
            self,
            *,
            strategy_name: str,
            market_id: str,
            token_id: str,
            entry_ts: pd.Timestamp,
        ) -> str: ...

        def _assert_valid_order_transition(
            self,
            previous_state: str | None,
            next_state: str,
        ) -> None: ...

logger = logging.getLogger(__name__)


@dataclass
class SimulationContinuationState:
    """Mutable simulation carry-over state used for batch-wise execution."""

    risk_state: RiskState
    gross_exposure_used: float
    capital_remaining: float | None
    pending_trades: list[PendingTrade] = field(default_factory=list)


class BacktestSimulationEngine:
    """Core trade simulation loop for hold-to-resolution execution."""

    @staticmethod
    def _iter_market_groups_by_id(
        signal_frame: pd.DataFrame,
    ) -> Iterator[tuple[str, pd.DataFrame]]:
        """Yield per-market slices ordered by first actionable timestamp."""
        market_groups: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
        for market_id, market_group in signal_frame.groupby("market_id", sort=False):
            group_start = pd.to_datetime(
                market_group.index.min(),
                utc=True,
                errors="coerce",
            )
            if pd.isna(group_start):
                group_start = pd.Timestamp.max.tz_localize("UTC")
            market_groups.append((group_start, str(market_id), market_group.sort_index()))

        for _, market_id, market_group in sorted(
            market_groups,
            key=lambda item: (item[0], item[1]),
        ):
            yield market_id, market_group

    def simulate_hold_to_resolution_backtest(  # noqa: PLR0913
        self: _BacktestSimulationHost,
        signal_frame: pd.DataFrame,
        resolution_frame: pd.DataFrame,
        *,
        strategy_name: str,
        signal_column: str = "signal",
        config: BacktestConfig | Mapping[str, object] | None = None,
        risk_events: list[dict[str, object]] | None = None,
        order_events: list[dict[str, object]] | None = None,
    ) -> pd.DataFrame:
        """Simulate first-signal entry and hold-to-resolution exits for one strategy."""
        required_cols = {"market_id", "token_id", "mid_price", signal_column}
        missing = [col for col in required_cols if col not in signal_frame.columns]
        if missing:
            msg = f"Missing required columns for backtest simulation: {missing}"
            raise ValueError(msg)

        if signal_frame.empty or resolution_frame.empty:
            return pd.DataFrame()

        trade_frame, _ = self.simulate_hold_to_resolution_backtest_incremental(
            signal_frame,
            resolution_frame,
            strategy_name=strategy_name,
            signal_column=signal_column,
            config=config,
            risk_events=risk_events,
            order_events=order_events,
        )
        return trade_frame
    def simulate_hold_to_resolution_backtest_incremental(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self: _BacktestSimulationHost,
        signal_frame: pd.DataFrame,
        resolution_frame: pd.DataFrame,
        *,
        strategy_name: str,
        signal_column: str = "signal",
        config: BacktestConfig | Mapping[str, object] | None = None,
        risk_events: list[dict[str, object]] | None = None,
        order_events: list[dict[str, object]] | None = None,
        continuation_state: SimulationContinuationState | None = None,
        finalize: bool = True,
    ) -> tuple[pd.DataFrame, SimulationContinuationState]:
        """Stateful variant of hold-to-resolution simulation for batch-wise execution."""
        required_cols = {"market_id", "token_id", "mid_price", signal_column}
        missing = [col for col in required_cols if col not in signal_frame.columns]
        if missing:
            msg = f"Missing required columns for backtest simulation: {missing}"
            raise ValueError(msg)

        cfg = coerce_backtest_config(config)
        signal_frame = signal_frame.sort_index()
        resolution_frame = resolution_frame.copy(deep=False)
        resolution_frame.index = resolution_frame.index.astype(str)
        risk_evaluator = self._build_risk_evaluator(cfg)

        if continuation_state is None:
            risk_state = RiskState()
            gross_exposure_used = 0.0
            capital_remaining = cfg.available_capital
            pending_trades = []
        else:
            risk_state = continuation_state.risk_state
            gross_exposure_used = float(continuation_state.gross_exposure_used)
            capital_remaining = continuation_state.capital_remaining
            pending_trades = list(continuation_state.pending_trades)

        if signal_frame.empty or resolution_frame.empty:
            return (
                pd.DataFrame(),
                SimulationContinuationState(
                    risk_state=risk_state,
                    gross_exposure_used=gross_exposure_used,
                    capital_remaining=capital_remaining,
                    pending_trades=pending_trades,
                ),
            )

        # Batch orchestration owns the high-level progress bar; suppress nested bars here.
        progress = None
        stage_started = perf_counter()

        processed_markets = 0
        markets_with_resolution = 0
        markets_with_eligible_signal = 0
        markets_with_sizing_reject = 0
        markets_with_risk_reject = 0
        markets_with_fill_reject = 0
        markets_with_trade = 0
        total_requested_qty = 0.0
        total_filled_qty = 0.0
        total_gross_notional = 0.0
        total_net_pnl = 0.0
        total_execution_cost = 0.0
        total_markets = int(signal_frame["market_id"].astype(str).nunique())
        metrics_every = max(int(cfg.metrics_log_every_n_markets), 1)
        def _update_progress_and_metrics() -> None:
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    {
                        "trades": markets_with_trade,
                        "fill_rate": f"{(total_filled_qty / total_requested_qty):.2%}"
                        if total_requested_qty > 0
                        else "0.00%",
                    },
                    refresh=False,
                )

            if not cfg.metrics_logging_enabled:
                return
            if processed_markets % metrics_every != 0 and processed_markets != total_markets:
                return

            logger.info(
                "Backtest market progress [%s]: %d/%d markets | trades=%d | "
                "eligible=%d | risk_reject=%d | fill_reject=%d | fill_rate=%.2f%% | net_pnl=%.6f",
                strategy_name,
                processed_markets,
                total_markets,
                markets_with_trade,
                markets_with_eligible_signal,
                markets_with_risk_reject,
                markets_with_fill_reject,
                (
                    (total_filled_qty / total_requested_qty * 100.0)
                    if total_requested_qty > 0
                    else 0.0
                ),
                total_net_pnl,
            )

        def _flush_pending_trades(
            *,
            cutoff_ts: pd.Timestamp | None = None,
            force: bool = False,
        ) -> None:
            nonlocal gross_exposure_used
            nonlocal capital_remaining
            nonlocal total_gross_notional
            nonlocal total_net_pnl
            nonlocal total_execution_cost

            if not pending_trades:
                return

            if force:
                ready_trades = sorted(
                    pending_trades,
                    key=lambda trade: (
                        trade["resolved_at"],
                        trade["market_id"],
                        trade["token_id"],
                    ),
                )
                pending_trades.clear()
            else:
                if cutoff_ts is None or pd.isna(cutoff_ts):
                    return
                ready_trades = [
                    trade
                    for trade in pending_trades
                    if trade["resolved_at"] <= cutoff_ts
                ]
                if not ready_trades:
                    return
                pending_trades[:] = [
                    trade
                    for trade in pending_trades
                    if trade["resolved_at"] > cutoff_ts
                ]

            for trade in ready_trades:
                trade_row = dict(trade["row"])
                effective_notional = float(trade["effective_notional"])
                net_pnl = float(trade["net_pnl"])

                risk_evaluator.register_close(
                    market_id=str(trade["market_id"]),
                    resolved_at=trade["resolved_at"].to_pydatetime(),
                    net_pnl=net_pnl,
                    state=risk_state,
                )
                gross_exposure_used = max(gross_exposure_used - effective_notional, 0.0)
                if capital_remaining is not None:
                    capital_remaining = capital_remaining + effective_notional + net_pnl

                total_gross_notional += float(cast("float", trade_row["gross_notional"]))
                total_net_pnl += net_pnl
                total_execution_cost += float(
                    cast("float", trade_row["total_execution_cost_usdc"])
                )
                trade_rows.append(trade_row)

        trade_rows: list[dict[str, object]] = []
        for market_id_str, market_group in self._iter_market_groups_by_id(signal_frame):
            processed_markets += 1
            if market_id_str not in resolution_frame.index:
                _update_progress_and_metrics()
                continue
            markets_with_resolution += 1

            resolution_row = resolution_frame.loc[market_id_str]
            if isinstance(resolution_row, pd.DataFrame):
                msg = f"Duplicate resolution rows for market {market_id_str}"
                raise TypeError(msg)

            resolved_at = resolution_row.get("resolved_at")
            if pd.isna(resolved_at):
                _update_progress_and_metrics()
                continue

            eligible = market_group[market_group.index < resolved_at]
            if eligible.empty:
                _update_progress_and_metrics()
                continue
            markets_with_eligible_signal += 1

            entry_ts = eligible.index.min()
            # Realize PnL and release capital for any positions resolved by this entry time.
            _flush_pending_trades(cutoff_ts=pd.to_datetime(entry_ts, utc=True), force=False)

            entry_candidates = eligible.loc[eligible.index == entry_ts].copy()
            if isinstance(entry_candidates, pd.Series):
                entry_candidates = entry_candidates.to_frame().T

            if entry_candidates.empty:
                _update_progress_and_metrics()
                continue

            entry_candidates["_signal_abs"] = entry_candidates[signal_column].abs()
            entry_candidates["_mid_price"] = pd.to_numeric(
                entry_candidates["mid_price"],
                errors="coerce",
            )
            entry_candidates["_distance_from_mid"] = (
                entry_candidates["_mid_price"] - 0.5
            ).abs()
            entry_candidates = entry_candidates.sort_values(
                ["_signal_abs", "_distance_from_mid", "_mid_price", "token_id"],
                ascending=[False, False, False, True],
            )

            entry_row = entry_candidates.iloc[0]
            entry_candidate_count = len(entry_candidates)
            entry_signal_abs = float(entry_row.get("_signal_abs", 0.0))
            entry_distance_from_mid = float(entry_row.get("_distance_from_mid", 0.0))
            entry_selection_reason = (
                "max_abs_signal_then_distance_from_0p5_then_mid_price_then_token_id"
            )
            token_id_str = str(entry_row.get("token_id"))
            signal_value = int(entry_row[signal_column])
            direction = 1 if signal_value > 0 else -1
            order_id = self._build_order_id(
                strategy_name=strategy_name,
                market_id=market_id_str,
                token_id=token_id_str,
                entry_ts=entry_ts,
            )

            entry_price_raw = entry_row.get("mid_price")
            if entry_price_raw is None:
                _update_progress_and_metrics()
                continue
            entry_price = float(entry_price_raw)
            if not isfinite(entry_price) or entry_price <= 0 or entry_price >= 1:
                _update_progress_and_metrics()
                continue

            winning_asset_id = resolution_row.get("winning_asset_id")
            winning_outcome = resolution_row.get("winning_outcome")
            settlement_source = resolution_row.get("settlement_source", "unknown")
            settlement_confidence = resolution_row.get("settlement_confidence")
            settlement_evidence_ts = resolution_row.get("settlement_evidence_ts")
            exit_price = 1.0 if str(winning_asset_id) == token_id_str else 0.0
            trade_price = entry_price if direction > 0 else 1 - entry_price
            exit_trade_price = exit_price if direction > 0 else 1 - exit_price

            entry_volatility = self._estimate_market_volatility(
                market_group,
                entry_ts=entry_ts,
                lookback=cfg.sizing_volatility_lookback,
            )
            entry_spread_raw = self._to_float_or_nan(entry_row.get("spread"))
            entry_spread = float(entry_spread_raw) if pd.notna(entry_spread_raw) else 0.0

            bid_depth_raw = self._to_float_or_nan(entry_row.get("bid_depth_1"))
            ask_depth_raw = self._to_float_or_nan(entry_row.get("ask_depth_1"))
            bid_depth = float(bid_depth_raw) if pd.notna(bid_depth_raw) else 0.0
            ask_depth = float(ask_depth_raw) if pd.notna(ask_depth_raw) else 0.0
            entry_liquidity = max(bid_depth, 0.0) + max(ask_depth, 0.0)

            requested_qty, sizing_rationale, requested_notional = self._apply_sizing_policy(
                signal_value=signal_value,
                signal_abs=entry_signal_abs,
                entry_price=entry_price,
                market_group=market_group,
                entry_ts=entry_ts,
                cfg=cfg,
                gross_exposure_used=gross_exposure_used,
                capital_remaining=capital_remaining,
            )

            order_events_sink: list[dict[str, object]] = (
                order_events if order_events is not None else []
            )
            lifecycle_enabled = bool(cfg.order_lifecycle_enabled and order_events is not None)
            order_last_state: str | None = None
            order_event_seq = 0
            order_event_context = {
                "lifecycle_enabled": lifecycle_enabled,
                "order_events_sink": order_events_sink,
                "order_id_value": order_id,
                "market_id_value": market_id_str,
                "token_id_value": token_id_str,
            }

            def _record_order_event(  # noqa: PLR0913
                state: str,
                *,
                event_ts: pd.Timestamp,
                event_fill_qty: float,
                cumulative_filled_qty: float,
                requested: float,
                avg_price: float | None,
                event_cost_usdc: float,
                reject: str | None,
                details: str | None = None,
                lifecycle_enabled: bool,
                order_events_sink: list[dict[str, object]],
                order_id_value: str,
                market_id_value: str,
                token_id_value: str,
            ) -> None:
                nonlocal order_last_state
                nonlocal order_event_seq
                if not lifecycle_enabled:
                    return
                self._assert_valid_order_transition(order_last_state, state)
                order_event_seq += 1
                order_last_state = state
                order_events_sink.append(
                    {
                        "order_id": order_id_value,
                        "strategy": strategy_name,
                        "market_id": market_id_value,
                        "token_id": token_id_value,
                        "ts_event": pd.to_datetime(event_ts, utc=True),
                        "event_seq": order_event_seq,
                        "state": state,
                        "requested_qty": max(float(requested), 0.0),
                        "event_fill_qty": max(float(event_fill_qty), 0.0),
                        "cumulative_filled_qty": max(float(cumulative_filled_qty), 0.0),
                        "remaining_qty": max(float(requested) - float(cumulative_filled_qty), 0.0),
                        "avg_fill_price": avg_price,
                        "event_cost_usdc": max(float(event_cost_usdc), 0.0),
                        "reject_reason": reject,
                        "details": details,
                    }
                )

            _record_order_event(
                "submitted",
                event_ts=entry_ts,
                event_fill_qty=0.0,
                cumulative_filled_qty=0.0,
                requested=requested_qty,
                avg_price=None,
                event_cost_usdc=0.0,
                reject=None,
                details="order_submitted",
                **order_event_context,
            )

            if requested_qty <= 0:
                markets_with_sizing_reject += 1
                _record_order_event(
                    "rejected",
                    event_ts=entry_ts,
                    event_fill_qty=0.0,
                    cumulative_filled_qty=0.0,
                    requested=requested_qty,
                    avg_price=None,
                    event_cost_usdc=0.0,
                    reject="non_positive_requested_qty",
                    details=sizing_rationale,
                    **order_event_context,
                )
                if risk_events is not None and cfg.mode == "tolerant":
                    risk_events.append(
                        {
                            "stage": "sizing",
                            "severity": "warning",
                            "market_id": market_id_str,
                            "reason": "non_positive_requested_qty",
                            "details": sizing_rationale,
                        }
                    )
                _update_progress_and_metrics()
                continue

            total_requested_qty += requested_qty

            allowed, gate_events = risk_evaluator.evaluate_entry(
                timestamp=pd.to_datetime(entry_ts, utc=True).to_pydatetime(),
                market_id=market_id_str,
                requested_notional=requested_notional,
                state=risk_state,
            )
            if not allowed:
                markets_with_risk_reject += 1
                gate_reason = ", ".join(event.reason for event in gate_events)
                _record_order_event(
                    "rejected",
                    event_ts=entry_ts,
                    event_fill_qty=0.0,
                    cumulative_filled_qty=0.0,
                    requested=requested_qty,
                    avg_price=None,
                    event_cost_usdc=0.0,
                    reject="risk_gate_blocked",
                    details=gate_reason,
                    **order_event_context,
                )
                if risk_events is not None:
                    event_severity = "warning" if cfg.mode == "tolerant" else "info"
                    for event in gate_events:
                        risk_events.append(
                            {
                                "stage": "risk_gate",
                                "severity": event_severity,
                                "market_id": event.market_id or market_id_str,
                                "reason": event.reason,
                                "details": (
                                    f"gate={event.gate},current={event.current_value},"
                                    f"limit={event.limit_value}"
                                ),
                            }
                        )
                _update_progress_and_metrics()
                continue

            fill_result = self._execute_fill_model(
                requested_qty=requested_qty,
                entry_price=entry_price,
                direction=direction,
                cfg=cfg,
                entry_row=entry_row,
            )
            filled_qty = float(fill_result["filled_qty"])
            avg_fill_price = float(fill_result["avg_fill_price"])
            slippage_bps = float(fill_result["slippage_bps"])
            reject_reason = fill_result["reject_reason"]
            order_state = fill_result["order_state"]

            if filled_qty <= 0:
                markets_with_fill_reject += 1
                _record_order_event(
                    "rejected",
                    event_ts=entry_ts,
                    event_fill_qty=0.0,
                    cumulative_filled_qty=0.0,
                    requested=requested_qty,
                    avg_price=None,
                    event_cost_usdc=0.0,
                    reject=str(reject_reason or "fill_rejected"),
                    details=f"fill_model={cfg.fill_model}",
                    **order_event_context,
                )
                if risk_events is not None and cfg.mode == "tolerant":
                    risk_events.append(
                        {
                            "stage": "fill_engine",
                            "severity": "warning",
                            "market_id": market_id_str,
                            "reason": str(reject_reason or "fill_rejected"),
                            "details": (
                                f"requested_qty={requested_qty:.8f},"
                                f"order_state={order_state}"
                            ),
                        }
                    )
                _update_progress_and_metrics()
                continue

            total_filled_qty += filled_qty

            effective_notional = filled_qty * trade_price
            risk_evaluator.register_fill(
                market_id=market_id_str,
                notional=effective_notional,
                state=risk_state,
            )
            gross_exposure_used += effective_notional
            if capital_remaining is not None:
                capital_remaining = max(capital_remaining - effective_notional, 0.0)

            gross_pnl = filled_qty * (exit_trade_price - trade_price)
            market_fees_enabled = bool(resolution_row.get("fees_enabled_market", True))
            fee_usdc = calculate_taker_fee(
                trade_price,
                shares=filled_qty,
                fee_rate=cfg.fee_rate,
                fees_enabled=cfg.fees_enabled and market_fees_enabled,
                precision=cfg.fee_precision,
                minimum_fee=cfg.min_fee,
            )
            execution_costs = self._decompose_execution_costs(
                fee_usdc=fee_usdc,
                requested_qty=requested_qty,
                filled_qty=filled_qty,
                slippage_bps=slippage_bps,
                avg_fill_price=avg_fill_price,
                entry_price=entry_price,
            )

            if filled_qty < requested_qty:
                if lifecycle_enabled:
                    _record_order_event(
                        "partial",
                        event_ts=entry_ts,
                        event_fill_qty=filled_qty,
                        cumulative_filled_qty=filled_qty,
                        requested=requested_qty,
                        avg_price=avg_fill_price,
                        event_cost_usdc=0.0,
                        reject=reject_reason,
                        details=f"fill_model={cfg.fill_model}",
                        **order_event_context,
                    )
                    remaining_qty = requested_qty - filled_qty
                    for attempt in range(max(cfg.order_max_amendments, 0)):
                        if not cfg.order_allow_amendments or remaining_qty <= 0:
                            break
                        _record_order_event(
                            "partial",
                            event_ts=entry_ts,
                            event_fill_qty=0.0,
                            cumulative_filled_qty=filled_qty,
                            requested=requested_qty,
                            avg_price=avg_fill_price,
                            event_cost_usdc=0.0,
                            reject=reject_reason,
                            details=f"amendment_attempt={attempt + 1}",
                            **order_event_context,
                        )

                    if cfg.order_ttl_seconds is not None and cfg.order_ttl_seconds > 0:
                        terminal_state = "expired"
                        terminal_ts = entry_ts + pd.to_timedelta(cfg.order_ttl_seconds, unit="s")
                    else:
                        terminal_state = "cancelled"
                        terminal_ts = entry_ts

                    order_state = terminal_state
                    if reject_reason is None:
                        reject_reason = f"unfilled_{terminal_state}"

                    _record_order_event(
                        terminal_state,
                        event_ts=terminal_ts,
                        event_fill_qty=0.0,
                        cumulative_filled_qty=filled_qty,
                        requested=requested_qty,
                        avg_price=avg_fill_price,
                        event_cost_usdc=execution_costs["total_execution_cost_usdc"],
                        reject=reject_reason,
                        details="terminal_unfilled_quantity",
                        **order_event_context,
                    )
                else:
                    order_state = "partial"
            else:
                order_state = "filled"
                _record_order_event(
                    "filled",
                    event_ts=entry_ts,
                    event_fill_qty=filled_qty,
                    cumulative_filled_qty=filled_qty,
                    requested=requested_qty,
                    avg_price=avg_fill_price,
                    event_cost_usdc=execution_costs["total_execution_cost_usdc"],
                    reject=reject_reason,
                    details="terminal_fill_complete",
                    **order_event_context,
                )

            net_pnl = gross_pnl - execution_costs["total_execution_cost_usdc"]
            gross_notional = filled_qty * trade_price
            hold_hours = (resolved_at - entry_ts) / pd.Timedelta(hours=1)

            pending_trades.append(
                {
                    "row": {
                        "strategy": strategy_name,
                        "order_id": order_id,
                        "market_id": market_id_str,
                        "token_id": token_id_str,
                        "entry_ts": entry_ts,
                        "resolved_at": resolved_at,
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "trade_price": trade_price,
                        "requested_qty": requested_qty,
                        "filled_qty": filled_qty,
                        "avg_fill_price": avg_fill_price,
                        "slippage_bps": slippage_bps,
                        "reject_reason": reject_reason,
                        "order_state": order_state,
                        "sizing_policy": cfg.sizing_policy,
                        "sizing_rationale": sizing_rationale,
                        "exit_trade_price": exit_trade_price,
                        "gross_pnl": gross_pnl,
                        "fee_usdc": fee_usdc,
                        "taker_fee_usdc": execution_costs["taker_fee_usdc"],
                        "spread_crossing_usdc": execution_costs["spread_crossing_usdc"],
                        "slippage_impact_usdc": execution_costs["slippage_impact_usdc"],
                        "total_execution_cost_usdc": execution_costs["total_execution_cost_usdc"],
                        "net_pnl": net_pnl,
                        "gross_return_pct": gross_pnl / gross_notional if gross_notional else 0.0,
                        "net_return_pct": net_pnl / gross_notional if gross_notional else 0.0,
                        "hold_hours": float(hold_hours),
                        "signal_value": signal_value,
                        "entry_candidate_count": entry_candidate_count,
                        "entry_signal_abs": entry_signal_abs,
                        "entry_distance_from_mid": entry_distance_from_mid,
                        "entry_selection_reason": entry_selection_reason,
                        "winning_asset_id": winning_asset_id,
                        "winning_outcome": winning_outcome,
                        "settlement_source": settlement_source,
                        "settlement_confidence": settlement_confidence,
                        "settlement_evidence_ts": settlement_evidence_ts,
                        "entry_volatility": entry_volatility,
                        "entry_spread": entry_spread,
                        "entry_liquidity": entry_liquidity,
                        "gross_notional": gross_notional,
                    },
                    "effective_notional": effective_notional,
                    "net_pnl": net_pnl,
                    "resolved_at": pd.to_datetime(resolved_at, utc=True),
                    "market_id": market_id_str,
                    "token_id": token_id_str,
                }
            )

            markets_with_trade += 1
            _update_progress_and_metrics()

        if progress is not None:
            progress.close()

        if finalize:
            _flush_pending_trades(force=True)

        trade_frame = pd.DataFrame(trade_rows)
        if cfg.metrics_logging_enabled:
            elapsed_seconds = perf_counter() - stage_started
            logger.info(
                "Backtest market summary [%s]: markets=%d resolved=%d eligible=%d trades=%d "
                "sizing_reject=%d risk_reject=%d fill_reject=%d gross_notional=%.6f "
                "execution_cost=%.6f net_pnl=%.6f fill_rate=%.2f%% elapsed=%.2fs",
                strategy_name,
                processed_markets,
                markets_with_resolution,
                markets_with_eligible_signal,
                markets_with_trade,
                markets_with_sizing_reject,
                markets_with_risk_reject,
                markets_with_fill_reject,
                total_gross_notional,
                total_execution_cost,
                total_net_pnl,
                (
                    (total_filled_qty / total_requested_qty * 100.0)
                    if total_requested_qty > 0
                    else 0.0
                ),
                elapsed_seconds,
            )

        if trade_frame.empty:
            return (
                trade_frame,
                SimulationContinuationState(
                    risk_state=risk_state,
                    gross_exposure_used=gross_exposure_used,
                    capital_remaining=capital_remaining,
                    pending_trades=pending_trades,
                ),
            )

        return (
            trade_frame.sort_values(["resolved_at", "market_id", "token_id"]).reset_index(
                drop=True
            ),
            SimulationContinuationState(
                risk_state=risk_state,
                gross_exposure_used=gross_exposure_used,
                capital_remaining=capital_remaining,
                pending_trades=pending_trades,
            ),
        )

