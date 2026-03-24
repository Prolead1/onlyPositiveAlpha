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

### data/historical/ccxt.py
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

### models/events.py
Typed event models for Polymarket streaming data using Pydantic. Validates incoming events and provides type safety.
Key pieces:
- `BookEvent`, `PriceChangeEvent`, `LastTradePriceEvent`, `BestBidAskEvent` - typed market events
- `CryptoPriceEvent` - crypto price updates from RTDS
- `StoredPolymarketEvent` - persisted Polymarket market data with metadata
- `StoredCryptoEvent` - persisted crypto price data (RTDS and historical OHLCV)
- `StoredEvent` - (deprecated) legacy wrapper for backward compatibility

### data/stream/storage.py
Time-partitioned Parquet storage sink for streaming data. Writes validated events to hourly/daily partitions.
Key pieces:
- `StreamStorageSink` - manages in-memory buffers and writes to Parquet files
- `write_market_event()` - validates and stores Polymarket market events
- `write_crypto_price()` - validates and stores crypto price updates
- Storage structure:
  - `data/stream_storage/polymarket_market/` - market orderbook/trade events
  - `data/stream_storage/polymarket_rtds/` - crypto price updates
  - Files named by timestamp: `YYYY-MM-DD-HH.parquet` (hourly) or `YYYY-MM-DD.parquet` (daily)

### data/utils.py
Utilities for normalizing timestamps, symbols, and identifiers across data sources.
Key pieces:
- `normalize_symbol()` - converts symbols between formats (BTC/USD, btcusdt)
- `ts_to_datetime()` and `datetime_to_ts()` - timestamp conversion helpers
- `build_market_identifier()` - creates composite market:asset_id keys

### features/crypto.py and features/order_book.py
Feature generation for orderbook microstructure and trade flow analysis.
Key pieces:
- `compute_orderbook_features()` - spread, depth, imbalance, bid/ask ratio
- `compute_trade_features()` - buy/sell volume, trade imbalance, VWAP
- `compute_crypto_features()` - price returns, volatility
- `align_features_to_events()` - joins crypto features to market events using time windows

### backtester/runner.py
Backtest runner for loading stored data and generating feature datasets for modeling.
Key pieces:
- `BacktestRunner` - main class for feature generation pipeline
- `load_market_events()` and `load_crypto_prices()` - load from Parquet partitions
- `compute_orderbook_features_df()` - batch orderbook feature computation
- `compute_trade_features_df()` - rolling window trade flow features
- `build_feature_dataset()` - joins all features into a single dataset
- `label_market_outcomes()` - extracts market resolution labels for supervised learning

### visualization/
Visualization module for diagnostic reporting and analysis plots.
Key pieces:
- `plots.py` - all plotting functions for orderbook, trades, correlations
- `create_diagnostic_report()` - generates comprehensive diagnostic report with visualizations
- Plots include: spread timeseries, orderbook depth, imbalance, trade flow, crypto correlation, market timeline
- Reports saved to `reports/diagnostics/` with PNG plots and JSON summary

### alphas/
Alpha strategies and example implementations.
Key pieces:
- `example_alpha.py` - example script showing full alpha pipeline usage
  - Loads streaming data from storage
  - Computes features and labels
  - Generates diagnostic visualizations
  - Saves features for model training

## Usage

### Streaming and Storage

Run streaming workflows from `main.py`:

```bash
# Stream Polymarket orderbook data and crypto prices and store to Parquet
python main.py
```

Stored data is written to:
- `data/stream_storage/polymarket_market/` - Orderbook, trades, market events
- `data/stream_storage/polymarket_rtds/` - Crypto price updates

Files are partitioned by hour (default) or day, with schema validation via Pydantic models.

### Feature Generation and Alpha Research

Use `BacktestRunner` to load stored data and compute features. See `alphas/example_alpha.py` for a complete example:

```python
# Initialize runner
runner = BacktestRunner(storage_path=Path("data/stream_storage"))

# Load data for a time range
start = datetime(2026, 2, 25, 0, 0, 0)
end = datetime(2026, 2, 25, 23, 59, 59)

# Build feature dataset
features = runner.build_feature_dataset(start=start, end=end, crypto_window="5min")

# Extract labels for supervised learning
labels = runner.label_market_outcomes(runner.load_market_events(start, end))

# Generate diagnostic visualizations
from visualization import create_diagnostic_report
create_diagnostic_report(runner, start, end, output_dir=Path("reports/diagnostics"))

# Join features with labels and build models
# ... your ML pipeline here ...
```

Run the example alpha script:
```bash
uv run alphas/example_alpha.py
```

This will:
- Load streaming data from `data/cached/stream_feeds/`
- Compute orderbook and trade features
- Extract market resolution labels
- Generate diagnostic visualizations in `reports/diagnostics/`
- Save feature dataset to `data/cached/features.parquet`

Features include:
- **Orderbook microstructure**: spread, depth, imbalance, bid/ask ratio
- **Trade flow**: buy/sell volume, trade imbalance, VWAP
- **Crypto signals**: price returns, volatility aligned to market events