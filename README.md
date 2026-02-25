# FE5214 Intro to Quantitative Investing Project

## Overview
This project pulls and streams crypto market data using Polymarket sources, plus historical OHLCV data via ccxt. The code is split into small modules so each data source and the shared WebSocket logic are easy to reuse.

## Modules

### main.py
Entry point for running the project and switching between workflows. It wires configuration, logging, and the high-level streaming or historical tasks.
Key pieces:
- `get_historical()` fetches historical OHLCV data using ccxt parameters.
- `get_crypto_stream()` streams live crypto prices from Polymarket RTDS.
- `stream_bitcoin_updown()` resolves the BTC up-down market via the reference module and streams order book/trade events.

### data/historical/cctx.py
Historical data engine built on ccxt. It is responsible for pulling, aggregating, and caching OHLCV data across exchanges.
Key pieces:
- `OHLCVParams` defines the query inputs (symbol, time range, timeframe, exchanges).
- `fetch_historical_data()` collects data from each exchange, aggregates via volume-weighted averages, and caches results under data/cached/.

### data/stream/websocket.py
Shared WebSocket infrastructure used by all streaming clients. It handles connection setup, retries, rate-limiting, and dispatching parsed messages.
Key pieces:
- `WebSocketConfig` stores connection and SSL settings.
- `BaseWebSocketClient` implements connect/subscribe/stream/run lifecycle and common error handling.

### data/stream/rtds.py
Polymarket RTDS streaming client for crypto prices. It builds the correct subscription messages for Binance or Chainlink and formats price updates.
Key pieces:
- `CryptoPriceConfig` defines symbols, data source, and retry settings.
- `PolymarketCryptoStream` sends subscriptions and logs price updates.
- `stream_crypto_prices()` is the helper to start streaming with a callback.

### data/stream/polymarket.py
Polymarket market channel client for order book and trade events. It streams order book/trade updates once asset IDs are provided.
Key pieces:
- `MarketChannelConfig` and `PolymarketMarketChannel` configure and run the market stream.
- `stream_polymarket_data()` is the helper to start streaming with a callback.

### data/reference/gamma.py
Reference lookup helpers for Polymarket markets. It resolves time-based BTC up-down market IDs using the Gamma API.
Key pieces:
- `get_updown_asset_ids()` builds a time-based slug, calls the Gamma API, and extracts the CLOB token IDs.
- `_get_btc_slug()` and `_resolution_to_seconds()` keep the time slug logic consistent.

## Setup

In VSCode, download the ["Ruff" extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).
Run the below to download the required dependencies.

```
uv sync -U
```
