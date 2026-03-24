"""Feature generators for orderbook microstructure and trade flow.

This module provides functions to compute alpha features from streaming market data,
focusing on orderbook depth, spread, imbalance, and trade characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from utils import parse_orderbook_level

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class OrderbookFeatures:
    """Orderbook microstructure features at a point in time."""

    # Spread metrics
    spread: float | None = None  # Absolute spread (best_ask - best_bid)
    spread_bps: float | None = None  # Spread in basis points relative to mid
    mid_price: float | None = None  # Mid price (best_bid + best_ask) / 2

    # Depth metrics
    bid_depth_1: float | None = None  # Size at best bid
    ask_depth_1: float | None = None  # Size at best ask
    bid_depth_5: float | None = None  # Cumulative depth in top 5 bid levels
    ask_depth_5: float | None = None  # Cumulative depth in top 5 ask levels

    # Imbalance metrics
    imbalance_1: float | None = None  # (bid_depth_1 - ask_depth_1) / (bid_depth_1 + ask_depth_1)
    imbalance_5: float | None = None  # Same as above but for top 5 levels

    # Pressure metrics
    bid_ask_ratio: float | None = None  # bid_depth_5 / ask_depth_5


@dataclass
class TradeFeatures:
    """Trade flow features over a window."""

    # Trade direction
    buy_volume: float = 0.0  # Volume of buy trades
    sell_volume: float = 0.0  # Volume of sell trades
    trade_imbalance: float | None = None  # (buy - sell) / (buy + sell)

    # Trade intensity
    trade_count: int = 0  # Number of trades in window
    avg_trade_size: float | None = None  # Average trade size

    # Price impact
    vwap: float | None = None  # Volume-weighted average price
    price_change: float | None = None  # Change from first to last trade


def compute_orderbook_features(
    bids: list[tuple[float, float]] | list[dict[str, Any]],
    asks: list[tuple[float, float]] | list[dict[str, Any]],
) -> OrderbookFeatures:
    """Compute orderbook microstructure features from bid/ask levels.

    Parameters
    ----------
    bids : list
        List of (price, size) tuples or dicts with 'price' and 'size' keys,
        sorted by price descending.
    asks : list
        List of (price, size) tuples or dicts with 'price' and 'size' keys,
        sorted by price ascending.

    Returns
    -------
    OrderbookFeatures
        Computed features.
    """
    features = OrderbookFeatures()

    if not bids or not asks:
        return features

    # Parse levels using utility function
    bid_levels = [parse_orderbook_level(b) for b in bids]
    ask_levels = [parse_orderbook_level(a) for a in asks]

    # Ensure correct book ordering before selecting top levels:
    # bids highest->lowest, asks lowest->highest
    bid_levels = sorted(bid_levels, key=lambda level: level[0], reverse=True)[:5]
    ask_levels = sorted(ask_levels, key=lambda level: level[0])[:5]

    best_bid, bid_size_1 = bid_levels[0]
    best_ask, ask_size_1 = ask_levels[0]

    # Spread metrics
    features.spread = best_ask - best_bid
    features.mid_price = (best_bid + best_ask) / 2.0
    if features.mid_price > 0:
        features.spread_bps = (features.spread / features.mid_price) * 10000

    # Depth metrics
    features.bid_depth_1 = bid_size_1
    features.ask_depth_1 = ask_size_1
    features.bid_depth_5 = sum(size for _, size in bid_levels)
    features.ask_depth_5 = sum(size for _, size in ask_levels)

    # Imbalance metrics
    if (bid_size_1 + ask_size_1) > 0:
        features.imbalance_1 = (bid_size_1 - ask_size_1) / (bid_size_1 + ask_size_1)
    if (features.bid_depth_5 + features.ask_depth_5) > 0:
        features.imbalance_5 = (features.bid_depth_5 - features.ask_depth_5) / (
            features.bid_depth_5 + features.ask_depth_5
        )

    # Pressure metrics
    if features.ask_depth_5 > 0:
        features.bid_ask_ratio = features.bid_depth_5 / features.ask_depth_5

    return features


def compute_trade_features(trades_df: pd.DataFrame) -> TradeFeatures:
    """Compute trade flow features from a window of trades.

    Parameters
    ----------
    trades_df : pd.DataFrame
        DataFrame with columns: ['price', 'size', 'side'] where side is 'BUY' or 'SELL'.

    Returns
    -------
    TradeFeatures
        Computed features.
    """
    features = TradeFeatures()

    if trades_df.empty:
        return features

    features.trade_count = len(trades_df)

    # Separate buy/sell volumes
    buy_trades = trades_df[trades_df["side"] == "BUY"]
    sell_trades = trades_df[trades_df["side"] == "SELL"]

    features.buy_volume = buy_trades["size"].sum() if not buy_trades.empty else 0.0
    features.sell_volume = sell_trades["size"].sum() if not sell_trades.empty else 0.0

    total_volume = features.buy_volume + features.sell_volume
    if total_volume > 0:
        features.trade_imbalance = (features.buy_volume - features.sell_volume) / total_volume
        features.avg_trade_size = total_volume / features.trade_count

    # VWAP and price change
    if "price" in trades_df.columns and "size" in trades_df.columns:
        total_notional = (trades_df["price"] * trades_df["size"]).sum()
        if total_volume > 0:
            features.vwap = total_notional / total_volume

        if len(trades_df) > 1:
            first_price = trades_df.iloc[0]["price"]
            last_price = trades_df.iloc[-1]["price"]
            if first_price > 0:
                features.price_change = (last_price - first_price) / first_price

    return features


def compute_features_from_price_change(price_change_data: dict[str, Any]) -> OrderbookFeatures:
    """Compute orderbook features from Polymarket price_change event.

    Price change events have the structure:
    {
        "price_changes": [
            {
                "asset_id": "...",
                "best_bid": "0.67",
                "best_ask": "0.71",
                "price": "0.29",
                "side": "BUY",
                "size": "12"
            },
            ...
        ]
    }

    Parameters
    ----------
    price_change_data : dict
        The 'price_changes' data from a price_change event.

    Returns
    -------
    OrderbookFeatures
        Computed features using best_bid and best_ask from the event.
    """
    features = OrderbookFeatures()

    price_changes = price_change_data.get("price_changes", [])
    if not price_changes:
        return features

    # Use the first price change (there can be multiple assets in binary markets)
    change = price_changes[0]

    try:
        best_bid = float(change.get("best_bid", 0))
        best_ask = float(change.get("best_ask", 0))

        if best_bid > 0 and best_ask > 0:
            # Spread metrics
            features.spread = best_ask - best_bid
            features.mid_price = (best_bid + best_ask) / 2.0

            if features.mid_price > 0:
                features.spread_bps = (features.spread / features.mid_price) * 10000

            # price_change events do not provide true bid/ask depth snapshots.
            # Depth and pressure metrics are sourced from book events and forward-filled.

    except (ValueError, TypeError, KeyError):
        pass

    return features
