# FE5214 Intro to Quantitative Investing Project

## Overview
This project pulls and streams crypto market data using Polymarket sources, plus historical OHLCV data via ccxt. The code is split into small modules so each data source and the shared WebSocket logic are easy to reuse.

## Modules

### main.py
Entry point for running the project. It contains three workflows:
- `get_historical()` fetches historical OHLCV data using ccxt.
- `get_crypto_stream()` streams live crypto prices from Polymarket RTDS.
- `stream_bitcoin_updown()` streams Polymarket order book and trades for Bitcoin up-down markets.

### data/historical/cctx.py
Historical data engine built on ccxt. It:
- Fetches OHLCV data across multiple exchanges.
- Aggregates the data into volume-weighted averages.
- Caches results under data/cached/.

### data/stream/websocket.py
Shared WebSocket client with retry logic, rate-limit handling, and message dispatch. It defines:
- `WebSocketConfig` for connection settings.
- `BaseWebSocketClient` for connect/subscribe/stream/run lifecycle.

### data/stream/rtds.py
Polymarket RTDS streaming client for crypto prices. It supports:
- Binance or Chainlink sources.
- Symbol filters (or all symbols).
- `stream_crypto_prices()` helper to start streaming.

### data/stream/polymarket.py
Polymarket market channel client for order book and trade events. It includes:
- `get_btc_asset_id()` to resolve Bitcoin up-down market asset IDs.
- `PolymarketMarketChannel` WebSocket client.
- `stream_polymarket_data()` helper to start streaming.

## Setup

In VSCode, download the ["Ruff" extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).
Run the below to download the required dependencies.

```
uv sync -U
```
