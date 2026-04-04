from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class RiskLimits:
    """Configurable risk hard-limits used by the shell evaluator."""

    max_drawdown_pct: float | None = None
    max_daily_loss: float | None = None
    max_concentration_pct: float | None = None
    max_active_positions: int | None = None
    max_gross_exposure: float | None = None


@dataclass(frozen=True)
class RiskEvent:
    """Single risk decision event emitted by the evaluator."""

    timestamp: datetime
    gate: str
    reason: str
    current_value: float | int
    limit_value: float | int
    market_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a plain dictionary representation of the event."""
        return {
            "timestamp": self.timestamp,
            "gate": self.gate,
            "reason": self.reason,
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "market_id": self.market_id,
        }


@dataclass
class RiskState:
    """Mutable portfolio risk state tracked during simulation."""

    gross_exposure: float = 0.0
    market_exposure: dict[str, float] = field(default_factory=dict)
    active_positions: set[str] = field(default_factory=set)
    realized_pnl_by_day: dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    peak_equity: float = 0.0

    def day_key(self, timestamp: datetime) -> str:
        """Return the UTC day bucket used for realized PnL tracking."""
        ts = timestamp.astimezone(UTC)
        return ts.date().isoformat()


class RiskEvaluator:
    """Minimal shell evaluator for hard risk gates."""

    def __init__(self, limits: RiskLimits) -> None:
        """Initialize the evaluator with fixed risk limits."""
        self.limits = limits

    def evaluate_entry(
        self,
        *,
        timestamp: datetime,
        market_id: str,
        requested_notional: float,
        state: RiskState,
    ) -> tuple[bool, list[RiskEvent]]:
        """Evaluate whether a new position can be opened."""
        events: list[RiskEvent] = []
        next_gross = state.gross_exposure + requested_notional
        next_market_exposure = state.market_exposure.get(market_id, 0.0) + requested_notional
        total_for_concentration = max(next_gross, 1e-12)

        if (
            self.limits.max_gross_exposure is not None
            and next_gross > self.limits.max_gross_exposure
        ):
            events.append(
                RiskEvent(
                    timestamp=timestamp,
                    gate="gross_exposure",
                    reason="gross_exposure_limit_exceeded",
                    current_value=next_gross,
                    limit_value=self.limits.max_gross_exposure,
                )
            )

        if self.limits.max_active_positions is not None:
            next_active = len(state.active_positions | {market_id})
            if next_active > self.limits.max_active_positions:
                events.append(
                    RiskEvent(
                        timestamp=timestamp,
                        gate="active_positions",
                        reason="active_positions_limit_exceeded",
                        current_value=next_active,
                        limit_value=self.limits.max_active_positions,
                    )
                )

        if self.limits.max_concentration_pct is not None:
            concentration = next_market_exposure / total_for_concentration
            if concentration > self.limits.max_concentration_pct:
                events.append(
                    RiskEvent(
                        timestamp=timestamp,
                        gate="concentration",
                        reason="market_concentration_limit_exceeded",
                        current_value=concentration,
                        limit_value=self.limits.max_concentration_pct,
                        market_id=market_id,
                    )
                )

        if self.limits.max_drawdown_pct is not None and state.peak_equity > 0:
            drawdown = (state.peak_equity - state.equity) / state.peak_equity
            if drawdown > self.limits.max_drawdown_pct:
                events.append(
                    RiskEvent(
                        timestamp=timestamp,
                        gate="drawdown",
                        reason="drawdown_limit_exceeded",
                        current_value=drawdown,
                        limit_value=self.limits.max_drawdown_pct,
                    )
                )

        if self.limits.max_daily_loss is not None:
            day_pnl = state.realized_pnl_by_day.get(state.day_key(timestamp), 0.0)
            if day_pnl < -self.limits.max_daily_loss:
                events.append(
                    RiskEvent(
                        timestamp=timestamp,
                        gate="daily_loss",
                        reason="daily_loss_limit_exceeded",
                        current_value=abs(day_pnl),
                        limit_value=self.limits.max_daily_loss,
                    )
                )

        return len(events) == 0, events

    def register_fill(
        self,
        *,
        market_id: str,
        notional: float,
        state: RiskState,
    ) -> None:
        """Register an opened position in the risk state."""
        state.gross_exposure += notional
        state.market_exposure[market_id] = state.market_exposure.get(market_id, 0.0) + notional
        state.active_positions.add(market_id)

    def register_close(
        self,
        *,
        market_id: str,
        resolved_at: datetime,
        net_pnl: float,
        state: RiskState,
    ) -> None:
        """Register a closed position and realized PnL."""
        day_key = state.day_key(resolved_at)
        state.realized_pnl_by_day[day_key] = state.realized_pnl_by_day.get(day_key, 0.0) + net_pnl
        state.equity += net_pnl
        state.peak_equity = max(state.peak_equity, state.equity)

        notional = state.market_exposure.pop(market_id, 0.0)
        state.gross_exposure = max(state.gross_exposure - notional, 0.0)
        state.active_positions.discard(market_id)

    def register_realized_pnl(
        self,
        *,
        resolved_at: datetime,
        net_pnl: float,
        state: RiskState,
    ) -> None:
        """Update equity and daily PnL while preserving open exposure state."""
        day_key = state.day_key(resolved_at)
        state.realized_pnl_by_day[day_key] = state.realized_pnl_by_day.get(day_key, 0.0) + net_pnl
        state.equity += net_pnl
        state.peak_equity = max(state.peak_equity, state.equity)
