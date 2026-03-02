"""Typed event models for Polymarket streaming data."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from decimal import Decimal
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OrderbookLevel(BaseModel):
    """Single level in orderbook (bid or ask)."""

    price: Decimal = Field(description="Price level")
    size: Decimal = Field(description="Size at this level")

    @field_validator("price", "size", mode="before")
    @classmethod
    def parse_decimal(cls, v: str | float | Decimal) -> Decimal:
        """Parse string/float to Decimal."""
        return Decimal(str(v))


class BookEvent(BaseModel):
    """L2 orderbook snapshot from Polymarket market channel."""

    event_type: Literal["book"] = "book"
    asset_id: str = Field(description="Token ID for the outcome")
    market: str = Field(description="Market identifier")
    timestamp: int = Field(description="Server timestamp (milliseconds)")
    hash: str | None = Field(default=None, description="Hash of orderbook state")
    bids: list[OrderbookLevel] = Field(default_factory=list)
    asks: list[OrderbookLevel] = Field(default_factory=list)

    @field_validator("bids", "asks", mode="before")
    @classmethod
    def parse_levels(
        cls,
        v: Sequence[OrderbookLevel] | Sequence[dict[str, str]] | Sequence[Sequence[str]] | None,
    ) -> list[OrderbookLevel]:
        """Parse orderbook levels from nested lists or dicts."""
        if not v:
            return []
        first = v[0]
        if isinstance(first, OrderbookLevel):
            return list(cast("Sequence[OrderbookLevel]", v))
        if isinstance(first, dict):
            levels = cast("Sequence[dict[str, str]]", v)
            return [
                OrderbookLevel(
                    price=Decimal(str(level["price"])),
                    size=Decimal(str(level["size"])),
                )
                for level in levels
            ]
        levels = cast("Sequence[Sequence[str]]", v)
        return [
            OrderbookLevel(price=Decimal(str(level[0])), size=Decimal(str(level[1])))
            for level in levels
        ]

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | float) -> int:
        """Parse timestamp into milliseconds as int."""
        return int(float(v))


class PriceChangeEntry(BaseModel):
    """Single price change entry within a price_change event."""

    asset_id: str
    price: Decimal
    size: Decimal
    side: str | None = None
    hash: str | None = None
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None

    @field_validator("price", "size", "best_bid", "best_ask", mode="before")
    @classmethod
    def parse_decimal(cls, v: str | float | Decimal | None) -> Decimal | None:
        """Parse string/float to Decimal."""
        if v is None:
            return None
        return Decimal(str(v))


class PriceChangeEvent(BaseModel):
    """Price change event from Polymarket market channel."""

    event_type: Literal["price_change"] = "price_change"
    market: str
    timestamp: int
    price_changes: list[PriceChangeEntry] = Field(default_factory=list)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | float) -> int:
        """Parse timestamp into milliseconds as int."""
        return int(float(v))


class LastTradePriceEvent(BaseModel):
    """Last trade price event from Polymarket market channel."""

    event_type: Literal["last_trade_price"] = "last_trade_price"
    asset_id: str
    market: str
    timestamp: int
    price: Decimal
    size: Decimal | None = None
    side: str | None = None  # 'BUY' or 'SELL'
    fee_rate_bps: str | None = None

    @field_validator("price", "size", mode="before")
    @classmethod
    def parse_decimal(cls, v: str | float | Decimal | None) -> Decimal | None:
        """Parse string/float to Decimal."""
        if v is None:
            return None
        return Decimal(str(v))

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | float) -> int:
        """Parse timestamp into milliseconds as int."""
        return int(float(v))


class BestBidAskEvent(BaseModel):
    """Best bid/ask prices from Polymarket market channel."""

    event_type: Literal["best_bid_ask"] = "best_bid_ask"
    asset_id: str
    market: str
    timestamp: int
    best_bid: Decimal | None = None
    best_ask: Decimal | None = None
    spread: Decimal | None = None

    @field_validator("best_bid", "best_ask", mode="before")
    @classmethod
    def parse_decimal(cls, v: str | float | Decimal | None) -> Decimal | None:
        """Parse string/float to Decimal."""
        if v is None:
            return None
        return Decimal(str(v))

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | float) -> int:
        """Parse timestamp into milliseconds as int."""
        return int(float(v))


class TickSizeChangeEvent(BaseModel):
    """Tick size change event from Polymarket market channel."""

    event_type: Literal["tick_size_change"] = "tick_size_change"
    asset_id: str
    market: str
    timestamp: int
    old_tick_size: Decimal
    new_tick_size: Decimal

    @field_validator("old_tick_size", "new_tick_size", mode="before")
    @classmethod
    def parse_decimal(cls, v: str | float | Decimal) -> Decimal:
        """Parse string/float to Decimal."""
        return Decimal(str(v))

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | float) -> int:
        """Parse timestamp into milliseconds as int."""
        return int(float(v))


class NewMarketEvent(BaseModel):
    """New market created event from Polymarket market channel."""

    event_type: Literal["new_market"] = "new_market"
    market: str
    timestamp: int
    id: str | None = None
    question: str | None = None
    slug: str | None = None
    description: str | None = None
    assets_ids: list[str] | None = None
    outcomes: list[str] | None = None
    event_message: dict | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | float) -> int:
        """Parse timestamp into milliseconds as int."""
        return int(float(v))


class MarketResolvedEvent(BaseModel):
    """Market resolution event from Polymarket market channel."""

    event_type: Literal["market_resolved"] = "market_resolved"
    market: str
    timestamp: int
    id: str | None = None
    question: str | None = None
    slug: str | None = None
    description: str | None = None
    assets_ids: list[str] | None = None
    outcomes: list[str] | None = None
    winning_asset_id: str | None = None
    winning_outcome: str | None = None
    event_message: dict | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: str | float) -> int:
        """Parse timestamp into milliseconds as int."""
        return int(float(v))


# Union of all market events
MarketEvent = (
    BookEvent
    | PriceChangeEvent
    | LastTradePriceEvent
    | BestBidAskEvent
    | TickSizeChangeEvent
    | NewMarketEvent
    | MarketResolvedEvent
)


class CryptoPriceEvent(BaseModel):
    """Crypto price update from Polymarket RTDS stream."""

    symbol: str = Field(description="Trading pair symbol (e.g., 'btc/usd', 'btcusdt')")
    price: Decimal = Field(description="Current price")
    timestamp: int = Field(description="Server timestamp (milliseconds)")
    source: Literal["binance", "chainlink"] = Field(description="Price data source")
    volume_24h: Decimal | None = Field(default=None, description="24h volume if available")
    change_24h: Decimal | None = Field(default=None, description="24h price change if available")

    @field_validator("price", "volume_24h", "change_24h", mode="before")
    @classmethod
    def parse_decimal(cls, v: str | float | Decimal | None) -> Decimal | None:
        """Parse string/float to Decimal."""
        if v is None:
            return None
        return Decimal(str(v))


class StoredPolymarketEvent(BaseModel):
    """Stored Polymarket market data event.

    Covers all polymarket market data types:
    - book: L2 orderbook snapshot
    - price_change: Aggregated price changes
    - last_trade_price: Trade execution
    - best_bid_ask: Best bid/ask snapshot
    - tick_size_change: Tick size update
    - new_market: Market creation
    - market_resolved: Market resolution
    """

    ts_event: datetime = Field(description="Event timestamp (UTC)")
    ts_ingest: datetime = Field(description="Ingestion timestamp (UTC)")
    source: Literal["polymarket"] = Field(description="Event source (Polymarket)")
    event_type: str = Field(description="Type of market event")
    market_id: str = Field(description="Market identifier")
    token_id: str = Field(description="Token/asset identifier")
    market: str | None = Field(default=None, description="Market slug for reference")
    data: dict = Field(description="Raw event data as dict")

    model_config = ConfigDict()


class StoredCryptoEvent(BaseModel):
    """Stored crypto price data event.

    Covers crypto price data from multiple sources:
    - polymarket_rtds: Real-time crypto prices from Polymarket RTDS stream
    - exchange names (e.g., 'binance', 'coinbase', 'upbit'): Historical OHLCV candles
    """

    ts_event: datetime = Field(description="Event timestamp (UTC)")
    ts_ingest: datetime = Field(description="Ingestion timestamp (UTC)")
    source: str = Field(
        description="Event source (exchange name or 'polymarket_rtds')"
    )
    event_type: str = Field(description="Type of crypto event")
    symbol: str = Field(description="Symbol for crypto prices or trading pairs")
    timeframe: str | None = Field(
        default=None, description="Candle timeframe for OHLCV data"
    )
    data: dict = Field(description="Raw event data as dict")

    model_config = ConfigDict()


class StoredEvent(BaseModel):
    """[DEPRECATED] Use StoredPolymarketEvent or StoredCryptoEvent instead.

    This class is kept for backward compatibility but should not be used for new code.
    """

    ts_event: datetime = Field(description="Event timestamp (UTC)")
    ts_ingest: datetime = Field(description="Ingestion timestamp (UTC)")
    source: Literal["polymarket_market", "polymarket_rtds", "historical_ohlcv"] = (
        Field(description="Event source")
    )
    event_type: str = Field(description="Type of event")
    market_id: str | None = Field(default=None, description="Market identifier")
    token_id: str | None = Field(default=None, description="Token/asset identifier")
    symbol: str | None = Field(
        default=None, description="Symbol for crypto prices or trading pairs"
    )
    timeframe: str | None = Field(
        default=None, description="Candle timeframe for OHLCV data"
    )
    data: dict = Field(description="Raw event data as dict")

    model_config = ConfigDict()
