"""Backtest runner for generating predictions from stored streaming data.

This module loads time-partitioned Parquet files, computes orderbook microstructure
and crypto price features, and generates a labeled dataset for model training or backtesting.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime

from features import (
    OrderbookFeatures,
    align_features_to_events,
    compute_features_from_price_change,
    compute_orderbook_features,
    compute_trade_features,
)
from utils import (
    dataframe_json_column_to_dict,
    filter_by_time_range,
    prepare_timestamp_index,
    validate_path_exists,
)

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Load stored streaming data and generate backtest features."""

    def __init__(self, storage_path: Path | str, cache_dir: Path | str | None = None) -> None:
        """Initialize backtest runner.

        Parameters
        ----------
        storage_path : Path | str
            Path to stored stream data root directory.
        cache_dir : Path | str | None
            Path to cache directory for computed features. If None, uses
            storage_path/../feature_cache.
        """
        self.storage_path = Path(storage_path)
        self.market_path = self.storage_path / "polymarket_market"
        self.rtds_path = self.storage_path / "polymarket_rtds"

        if cache_dir is None:
            self.cache_path = self.storage_path.parent / "feature_cache"
        else:
            self.cache_path = Path(cache_dir)

        self.cache_path.mkdir(parents=True, exist_ok=True)

        validate_path_exists(
            self.storage_path,
            f"Storage path does not exist: {self.storage_path}",
        )

        logger.info("Initialized BacktestRunner with storage path: %s", self.storage_path)
        logger.info("Feature cache directory: %s", self.cache_path)

    def load_market_events(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit_files: int | None = None,
    ) -> pd.DataFrame:
        """Load market events from Parquet partitions.

        Parameters
        ----------
        start : datetime | None
            Start of time range (inclusive). If None, loads all.
        end : datetime | None
            End of time range (inclusive). If None, loads all.
        limit_files : int | None
            Limit number of parquet files to load. If None, loads all files.

        Returns
        -------
        pd.DataFrame
            Market events dataframe with timestamp index.
        """
        parquet_files = sorted(self.market_path.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No market event files found in %s", self.market_path)
            return pd.DataFrame()

        # Limit number of files if specified
        if limit_files is not None:
            parquet_files = parquet_files[:limit_files]

        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        combined = prepare_timestamp_index(combined, col="ts_event", sort=True)

        # Parse JSON data column if it's stored as string
        if "data" in combined.columns and not combined.empty:
            combined = dataframe_json_column_to_dict(combined, column="data")

        # Filter by time range
        combined = filter_by_time_range(combined, start=start, end=end)

        logger.info("Loaded %d market events", len(combined))
        return combined

    def _compute_cache_key(
        self,
        parquet_files: list[Path],
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> str:
        """Compute cache key based on input files and time range.

        Parameters
        ----------
        parquet_files : list[Path]
            List of parquet files being processed.
        start : datetime | None
            Start time filter.
        end : datetime | None
            End time filter.

        Returns
        -------
        str
            Cache key hash.
        """
        # Create hash from file paths, modification times, and time range
        hash_input = []
        for file in sorted(parquet_files):
            hash_input.append(str(file.name))
            hash_input.append(str(file.stat().st_mtime))

        if start:
            hash_input.append(start.isoformat())
        if end:
            hash_input.append(end.isoformat())

        combined = "|".join(hash_input)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def compute_orderbook_features_cached(
        self,
        market_events: pd.DataFrame,
        cache_key: str | None = None,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compute orderbook features with caching support.

        Parameters
        ----------
        market_events : pd.DataFrame
            Market events dataframe.
        cache_key : str | None
            Cache key for this computation. If None, uses hash of event count and time range.
        use_cache : bool
            Whether to use cached features if available.

        Returns
        -------
        pd.DataFrame
            Dataframe with orderbook features.
        """
        # Generate cache key if not provided
        if cache_key is None:
            if market_events.empty:
                cache_key = "empty"
            else:
                hash_parts = [
                    str(len(market_events)),
                    str(market_events.index.min()),
                    str(market_events.index.max()),
                ]
                cache_key = hashlib.sha256("|".join(hash_parts).encode()).hexdigest()[:16]

        cache_file = self.cache_path / f"features_{cache_key}.parquet"

        # Try to load from cache
        if use_cache and cache_file.exists():
            logger.info("Loading cached features from %s", cache_file)
            try:
                df = pd.read_parquet(cache_file)
            except Exception as e:
                logger.warning("Failed to load cache, recomputing: %s", e)
            else:
                logger.info("Loaded %d cached feature vectors", len(df))
                return df

        # Compute features
        logger.info("Computing features (no cache found)...")
        df = self.compute_orderbook_features_df(market_events)

        # Save to cache
        if not df.empty:
            try:
                df.to_parquet(cache_file)
                logger.info("Saved %d features to cache: %s", len(df), cache_file)
            except Exception as e:
                logger.warning("Failed to save cache: %s", e)

        return df

    def load_crypto_prices(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> pd.DataFrame:
        """Load crypto price events from Parquet partitions.

        Parameters
        ----------
        start : datetime | None
            Start of time range (inclusive). If None, loads all.
        end : datetime | None
            End of time range (inclusive). If None, loads all.

        Returns
        -------
        pd.DataFrame
            Crypto price dataframe with timestamp index.
        """
        parquet_files = sorted(self.rtds_path.glob("*.parquet"))
        if not parquet_files:
            logger.warning("No crypto price files found in %s", self.rtds_path)
            return pd.DataFrame()

        dfs = []
        for file in parquet_files:
            df = pd.read_parquet(file)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        combined = prepare_timestamp_index(combined, col="ts_event", sort=True)

        # Parse JSON data column if it's stored as string
        if "data" in combined.columns and not combined.empty:
            combined = dataframe_json_column_to_dict(combined, column="data")

        # Filter by time range
        combined = filter_by_time_range(combined, start=start, end=end)

        logger.info("Loaded %d crypto price events", len(combined))
        return combined

    def _postprocess_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process orderbook features dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Raw features dataframe with ts_event, market_id, token_id columns.

        Returns
        -------
        pd.DataFrame
            Processed features dataframe with filled values and recomputed metrics.
        """
        if df.empty:
            return df

        df["token_id"] = df["token_id"].fillna("")
        df = df.set_index("ts_event")
        df = df.sort_index()

        # Forward-fill depth/imbalance only within market+token streams
        fill_cols = [
            "imbalance_1",
            "imbalance_5",
            "bid_depth_1",
            "ask_depth_1",
            "bid_depth_5",
            "ask_depth_5",
            "bid_ask_ratio",
        ]
        group_keys = ["market_id", "token_id"]
        df[fill_cols] = df.groupby(group_keys, sort=False)[fill_cols].ffill()

        # Recompute pressure metrics from depths to keep columns internally consistent
        depth1_sum = df["bid_depth_1"] + df["ask_depth_1"]
        valid_depth1 = depth1_sum > 0
        df.loc[valid_depth1, "imbalance_1"] = (
            (df.loc[valid_depth1, "bid_depth_1"] - df.loc[valid_depth1, "ask_depth_1"])
            / depth1_sum.loc[valid_depth1]
        )

        depth5_sum = df["bid_depth_5"] + df["ask_depth_5"]
        valid_depth5 = depth5_sum > 0
        df.loc[valid_depth5, "imbalance_5"] = (
            (df.loc[valid_depth5, "bid_depth_5"] - df.loc[valid_depth5, "ask_depth_5"])
            / depth5_sum.loc[valid_depth5]
        )

        valid_ratio = df["ask_depth_5"] > 0
        df.loc[valid_ratio, "bid_ask_ratio"] = (
            df.loc[valid_ratio, "bid_depth_5"] / df.loc[valid_ratio, "ask_depth_5"]
        )

        return df

    def _resolve_token_id(self, row_token_id: object, asset_id: str) -> str:
        """Resolve token_id, preferring concrete asset_id over placeholders."""
        if isinstance(row_token_id, str):
            token = row_token_id.strip()
            if token and token.lower() != "unknown":
                return token
        return asset_id

    def _create_feature_dict(
        self,
        idx: Any,  # noqa: ANN401
        row: pd.Series,
        features: OrderbookFeatures,
        token_id: str,
    ) -> dict:
        """Create feature dictionary from computation results."""
        return {
            "ts_event": idx,
            "market_id": row["market_id"],
            "token_id": token_id,
            "spread": features.spread,
            "spread_bps": features.spread_bps,
            "mid_price": features.mid_price,
            "bid_depth_1": features.bid_depth_1,
            "ask_depth_1": features.ask_depth_1,
            "bid_depth_5": features.bid_depth_5,
            "ask_depth_5": features.ask_depth_5,
            "imbalance_1": features.imbalance_1,
            "imbalance_5": features.imbalance_5,
            "bid_ask_ratio": features.bid_ask_ratio,
        }

    def _process_book_events(self, book_events: pd.DataFrame) -> list:
        """Process book events and extract features."""
        features_list = []
        for idx, row in book_events.iterrows():
            try:
                data = row["data"]
                if not isinstance(data, dict):
                    continue

                bids = data.get("bids", [])
                asks = data.get("asks", [])
                if not (bids and asks):
                    continue

                features = compute_orderbook_features(bids, asks)
                token_id = self._resolve_token_id(
                    row.get("token_id"), data.get("asset_id", "")
                )
                features_list.append(
                    self._create_feature_dict(idx, row, features, token_id)
                )
            except (KeyError, TypeError):
                continue

        return features_list

    def _process_price_change_events(self, price_change_events: pd.DataFrame) -> list:
        """Process price_change events and extract features."""
        features_list = []
        for idx, row in price_change_events.iterrows():
            try:
                data = row["data"]
                if not isinstance(data, dict):
                    continue

                price_changes = data.get("price_changes", [])
                if not isinstance(price_changes, list):
                    continue

                # Process each asset update independently
                for change in price_changes:
                    if not isinstance(change, dict):
                        continue

                    features = compute_features_from_price_change(
                        {"price_changes": [change]}
                    )
                    asset_id = str(change.get("asset_id", ""))
                    token_id = self._resolve_token_id(row.get("token_id"), asset_id)
                    features_list.append(
                        self._create_feature_dict(idx, row, features, token_id)
                    )
            except (KeyError, TypeError):
                continue

        return features_list

    def compute_orderbook_features_df(self, market_events: pd.DataFrame) -> pd.DataFrame:
        """Compute orderbook features combining book and price_change events.

        Parameters
        ----------
        market_events : pd.DataFrame
            Market events dataframe.

        Returns
        -------
        pd.DataFrame
            Dataframe with orderbook features.
        """
        # Process book events for depth and imbalance
        book_events = market_events[market_events["event_type"] == "book"]
        book_features = self._process_book_events(book_events)

        # Process price_change events for high-frequency price updates
        price_change_events = market_events[market_events["event_type"] == "price_change"]
        price_features = self._process_price_change_events(price_change_events)

        # Combine all features
        features_list = book_features + price_features
        df = pd.DataFrame(features_list)
        df = self._postprocess_orderbook_features(df)

        logger.info(
            "Computed orderbook features for %d events (%d book, %d price_change)",
            len(df),
            len(book_events),
            len(price_change_events),
        )
        return df

    def compute_trade_features_df(
        self, market_events: pd.DataFrame, window: str = "5min"
    ) -> pd.DataFrame:
        """Compute rolling trade flow features.

        Parameters
        ----------
        market_events : pd.DataFrame
            Market events dataframe.
        window : str
            Rolling window for aggregation.

        Returns
        -------
        pd.DataFrame
            Dataframe with trade features indexed by time.
        """
        trade_events = market_events[market_events["event_type"] == "last_trade_price"]

        if trade_events.empty:
            logger.warning("No trade events found")
            return pd.DataFrame()

        # Extract trade data
        trades = []
        for idx, row in trade_events.iterrows():
            try:
                data = row["data"]
                if isinstance(data, dict):
                    trades.append(
                        {
                            "timestamp": idx,
                            "price": float(data.get("price", 0)),
                            "size": float(data.get("size", 1.0)),
                            "side": data.get("side", "UNKNOWN"),
                            "market_id": row["market_id"],
                            "token_id": row["token_id"],
                        }
                    )
            except (KeyError, TypeError, ValueError):
                continue

        trades_df = pd.DataFrame(trades).set_index("timestamp").sort_index()

        # Compute rolling features
        features_list = []
        for end_time in trades_df.index.unique():
            start_time = end_time - pd.Timedelta(window)
            window_trades = trades_df[
                (trades_df.index >= start_time) & (trades_df.index <= end_time)
            ]

            features = compute_trade_features(window_trades)
            features_list.append(
                {
                    "ts_event": end_time,
                    "buy_volume": features.buy_volume,
                    "sell_volume": features.sell_volume,
                    "trade_imbalance": features.trade_imbalance,
                    "trade_count": features.trade_count,
                    "avg_trade_size": features.avg_trade_size,
                    "vwap": features.vwap,
                    "price_change": features.price_change,
                }
            )

        df = pd.DataFrame(features_list).set_index("ts_event")
        logger.info("Computed trade features for %d windows", len(df))
        return df

    def build_feature_dataset(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        crypto_window: str = "5min",
    ) -> pd.DataFrame:
        """Build complete feature dataset by joining market and crypto features.

        Parameters
        ----------
        start : datetime | None
            Start time for data loading.
        end : datetime | None
            End time for data loading.
        crypto_window : str
            Lookback window for crypto features.

        Returns
        -------
        pd.DataFrame
            Feature dataset indexed by timestamp.
        """
        logger.info("Building feature dataset from %s to %s", start, end)

        # Load data
        market_events = self.load_market_events(start, end)
        crypto_prices = self.load_crypto_prices(start, end)

        if market_events.empty:
            logger.error("No market events loaded")
            return pd.DataFrame()

        # Compute orderbook features
        orderbook_features = self.compute_orderbook_features_df(market_events)

        # Compute trade features
        trade_features = self.compute_trade_features_df(market_events, window=crypto_window)

        # Parse crypto prices and compute features
        if not crypto_prices.empty:
            crypto_df = []
            for idx, row in crypto_prices.iterrows():
                data = row.get("data", {})
                if isinstance(data, dict):
                    crypto_df.append(
                        {
                            "timestamp": idx,
                            "price": float(data.get("price", 0)),
                            "symbol": row.get("symbol"),
                        }
                    )

            crypto_df = pd.DataFrame(crypto_df).set_index("timestamp").sort_index()

            # Align crypto features to orderbook events
            if not orderbook_features.empty and not crypto_df.empty:
                aligned = align_features_to_events(
                    orderbook_features, crypto_df, window=crypto_window
                )
            else:
                aligned = orderbook_features
        else:
            logger.warning("No crypto prices loaded, skipping crypto features")
            aligned = orderbook_features

        # Merge trade features
        if not trade_features.empty and not aligned.empty:
            aligned = aligned.join(trade_features, how="left", rsuffix="_trade")

        logger.info(
            "Built feature dataset with %d rows and %d columns",
            len(aligned),
            len(aligned.columns),
        )
        return aligned

    def label_market_outcomes(self, market_events: pd.DataFrame) -> pd.DataFrame:
        """Label market outcomes based on resolution events.

        Parameters
        ----------
        market_events : pd.DataFrame
            Market events dataframe.

        Returns
        -------
        pd.DataFrame
            Labeled dataframe with 'outcome' column indicating contract resolution.
        """
        resolved = market_events[market_events["event_type"] == "market_resolved"]

        labels = []
        for idx, row in resolved.iterrows():
            data = row.get("data", {})
            if isinstance(data, dict):
                outcome = data.get("winning_outcome")
                winning_asset = data.get("winning_asset_id")
                question = data.get("question") or "Unknown"
                logger.debug(
                    "Market resolved: %s, Outcome: %s, Asset: %s",
                    question[:50] if question else "Unknown",
                    outcome,
                    winning_asset,
                )
            else:
                outcome = None
            labels.append(
                {"market_id": row.get("market_id"), "resolved_at": idx, "outcome": outcome}
            )

        labels_df = pd.DataFrame(labels)
        logger.info("Extracted %d market resolution labels", len(labels_df))
        if not labels_df.empty:
            logger.info("Outcome distribution: %s", labels_df["outcome"].value_counts().to_dict())
        return labels_df
