"""Centralized configuration and constants for onlyPositiveAlpha project."""

from __future__ import annotations

from pathlib import Path

# ============================================================================
# Path Configuration
# ============================================================================

# Workspace root directory
WORKSPACE_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = WORKSPACE_ROOT / "data"
CACHE_DIR = DATA_DIR / "cached"
STREAM_FEEDS_DIR = CACHE_DIR / "stream_feeds"
PMXT_DIR = CACHE_DIR / "pmxt"
HISTORICAL_DIR = DATA_DIR / "historical"
REFERENCE_DIR = DATA_DIR / "reference"

# Output directories
REPORTS_DIR = WORKSPACE_ROOT / "reports"
DIAGNOSTICS_DIR = REPORTS_DIR / "diagnostics"

# ============================================================================
# WebSocket Endpoints
# ============================================================================

# Polymarket WebSocket URLs
POLYMARKET_MARKET_CHANNEL_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
POLYMARKET_RTDS_URL = "wss://ws-live-data.polymarket.com"

# ============================================================================
# Feature Computation Constants
# ============================================================================

# Crypto feature computation
MIN_PRICES_FOR_FEATURES = 2
MIN_PRICES_FOR_VOLATILITY = 2

# Visualization and correlation
ROLLING_CORRELATION_WINDOW = 20
ROLLING_CORRELATION_MIN_PERIODS = 10
INVALID_TIMESTAMP_THRESHOLD = 1e8

# Orderbook feature limits
MAX_ORDERBOOK_DEPTH = 5  # Number of levels to analyze

# ============================================================================
# Validation Thresholds
# ============================================================================

# Timestamp validation
MIN_TIMESTAMP_MS = 1_000_000_000_000  # Jan 2001 in milliseconds
MAX_TIMESTAMP_MS = 4_000_000_000_000  # Year 2096 in milliseconds

# Price validation
MIN_PRICE_VALUE = 0.0
MAX_PRICE_VALUE = 1_000_000_000.0  # Sanity check for unreasonable prices

# ============================================================================
# Storage Configuration
# ============================================================================

# Parquet storage settings
DEFAULT_BUFFER_SIZE = 100  # Events to buffer before flush
PARQUET_COMPRESSION = "snappy"
PARTITION_FREQUENCY = "hour"  # Options: "hour", "day"

# ============================================================================
# Streaming Configuration
# ============================================================================

# WebSocket settings
WS_RECONNECT_DELAY_SEC = 5
WS_PING_INTERVAL_SEC = 30
WS_PING_TIMEOUT_SEC = 10

# ============================================================================
# Historical Data Configuration
# ============================================================================

# CCXT settings
HISTORICAL_CACHE_EXPIRY_DAYS = 7
DEFAULT_OHLCV_LIMIT = 1000
