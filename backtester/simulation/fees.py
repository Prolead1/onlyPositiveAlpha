from __future__ import annotations

from math import isfinite


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
    if not fees_enabled or fee_rate <= 0:
        return 0.0
    if not isfinite(price) or price <= 0 or price >= 1:
        return 0.0

    fee = shares * fee_rate * price * (1 - price)
    fee = round(fee, precision)
    if fee < minimum_fee:
        return 0.0
    return fee
