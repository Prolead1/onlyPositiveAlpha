"""Feature generation pipeline for backtesting.

This module contains the feature-only computation path extracted from the
runner: orderbook microstructure features, trade features, and optional cache
loading/saving.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

from features import (
    OrderbookFeatures,
    align_features_to_events,
    compute_orderbook_features,
    compute_trade_features,
)

logger = logging.getLogger(__name__)
FEATURE_CACHE_VERSION = "book_only_v1"


@dataclass(frozen=True)
class FeatureGeneratorConfig:
    """Configuration for feature generation behavior."""

    cache_dir: Path | None = None
    cache_schema_version: str = "v1"
    cache_computation_signature: str = "orderbook_core_v1"
    book_state_max_age: pd.Timedelta | str | None = field(
        default_factory=lambda: pd.Timedelta("5min")
    )


class FeatureGenerator:
    """Generate orderbook, trade, and joined feature datasets."""

    def __init__(self, config: FeatureGeneratorConfig | None = None) -> None:
        """Create a feature generator with optional cache configuration."""
        self.config = config or FeatureGeneratorConfig()

    def cache_signature(self) -> str:
        """Return stable cache signature derived from version + computation config."""
        payload = {
            "feature_cache_version": FEATURE_CACHE_VERSION,
            "cache_schema_version": self.config.cache_schema_version,
            "cache_computation_signature": self.config.cache_computation_signature,
            "book_state_max_age": str(self.config.book_state_max_age),
        }
        return hashlib.sha256(
            "|".join(f"{k}:{payload[k]}" for k in sorted(payload)).encode("utf-8")
        ).hexdigest()[:16]

    def invalidate_cache(self, *, cache_key: str | None = None) -> int:
        """Invalidate cached feature files and return number of deleted files."""
        cache_dir = self.config.cache_dir
        if cache_dir is None or not cache_dir.exists():
            return 0

        pattern = f"features_*_{cache_key}.parquet" if cache_key else "features_*.parquet"

        removed = 0
        for path in cache_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1
        return removed

    def _postprocess_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill sparse fields and keep derived columns internally consistent."""
        if df.empty:
            return df

        df = df.copy()
        df["token_id"] = df["token_id"].fillna("").astype(str)
        df = df.sort_values(["market_id", "token_id", "ts_event"])

        if "_source_event_type" not in df.columns:
            df["_source_event_type"] = "book"

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
        df = df.set_index("ts_event")

        book_state_ts = pd.Series(df.index, index=df.index).where(
            df["_source_event_type"] == "book"
        )
        book_state_ts = book_state_ts.groupby(
            [df["market_id"], df["token_id"]],
            sort=False,
        ).ffill()

        known_token_mask = df["token_id"].str.len() > 0
        if known_token_mask.any():
            df.loc[known_token_mask, fill_cols] = (
                df.loc[known_token_mask].groupby(group_keys, sort=False)[fill_cols].ffill()
            )

        max_age = self.config.book_state_max_age
        if max_age is not None:
            max_age = pd.Timedelta(max_age)
            stale_mask = (
                book_state_ts.notna()
                & ((df.index.to_series() - book_state_ts) > max_age)
                & (df["_source_event_type"] != "book")
            )
            if stale_mask.any():
                df.loc[stale_mask, fill_cols] = np.nan

        depth1_sum = df["bid_depth_1"] + df["ask_depth_1"]
        valid_depth1 = depth1_sum > 0
        df.loc[valid_depth1, "imbalance_1"] = (
            df.loc[valid_depth1, "bid_depth_1"] - df.loc[valid_depth1, "ask_depth_1"]
        ) / depth1_sum.loc[valid_depth1]

        depth5_sum = df["bid_depth_5"] + df["ask_depth_5"]
        valid_depth5 = depth5_sum > 0
        df.loc[valid_depth5, "imbalance_5"] = (
            df.loc[valid_depth5, "bid_depth_5"] - df.loc[valid_depth5, "ask_depth_5"]
        ) / depth5_sum.loc[valid_depth5]

        valid_ratio = df["ask_depth_5"] > 0
        df.loc[valid_ratio, "bid_ask_ratio"] = (
            df.loc[valid_ratio, "bid_depth_5"] / df.loc[valid_ratio, "ask_depth_5"]
        )

        return df.sort_index().drop(columns=["_source_event_type"], errors="ignore")

    def _resolve_token_id(self, row_token_id: object, asset_id: str) -> str:
        if isinstance(row_token_id, str):
            token = row_token_id.strip()
            if token and token.lower() != "unknown":
                return token
        return asset_id

    def _create_feature_dict(
        self,
        idx: object,
        market_id: object,
        features: OrderbookFeatures,
        token_id: str,
        source_event_type: str,
    ) -> dict[str, object]:
        return {
            "ts_event": idx,
            "market_id": market_id,
            "token_id": token_id,
            "_source_event_type": source_event_type,
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

    def _process_book_events(self, book_events: pd.DataFrame) -> list[dict[str, object]]:
        features_list: list[dict[str, object]] = []
        for row in book_events.itertuples():
            try:
                data = getattr(row, "data", None)
                if not isinstance(data, dict):
                    continue

                bids = data.get("bids", [])
                asks = data.get("asks", [])
                if hasattr(bids, "tolist") and not isinstance(bids, list):
                    bids = bids.tolist()
                if hasattr(asks, "tolist") and not isinstance(asks, list):
                    asks = asks.tolist()

                if len(bids) == 0 or len(asks) == 0:
                    continue

                features = compute_orderbook_features(bids, asks)
                token_id = self._resolve_token_id(
                    getattr(row, "token_id", None),
                    data.get("asset_id", ""),
                )
                features_list.append(
                    self._create_feature_dict(
                        row.Index,
                        getattr(row, "market_id", ""),
                        features,
                        token_id,
                        "book",
                    )
                )
            except (KeyError, TypeError):
                continue

        return features_list

    def generate_orderbook_features(self, market_events: pd.DataFrame) -> pd.DataFrame:
        """Compute orderbook features from book events only."""
        if market_events.empty:
            return pd.DataFrame()

        book_events = market_events[market_events["event_type"] == "book"]
        book_features = self._process_book_events(book_events)

        features_list = book_features
        df = pd.DataFrame(features_list)
        df = self._postprocess_orderbook_features(df)

        logger.info(
            "Computed orderbook features for %d rows from %d book events",
            len(df),
            len(book_events),
        )
        return df

    def generate_orderbook_features_cached(
        self,
        market_events: pd.DataFrame,
        *,
        cache_key: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compute orderbook features with optional cache read/write."""
        cache_dir = self.config.cache_dir
        if cache_dir is None:
            return self.generate_orderbook_features(market_events)

        cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_key is None:
            if market_events.empty:
                cache_key = "empty"
            else:
                hash_parts = [
                    self.cache_signature(),
                    str(len(market_events)),
                    str(market_events.index.min()),
                    str(market_events.index.max()),
                ]
                cache_key = hashlib.sha256("|".join(hash_parts).encode()).hexdigest()[:16]

        cache_file = cache_dir / f"features_{self.cache_signature()}_{cache_key}.parquet"

        if use_cache and cache_file.exists():
            logger.info("Loading cached features from %s", cache_file)
            try:
                df = pd.read_parquet(cache_file)
            except Exception as exc:
                logger.warning("Failed to load cache, recomputing: %s", exc)
            else:
                logger.info("Loaded %d cached feature vectors", len(df))
                return df

        df = self.generate_orderbook_features(market_events)
        if not df.empty:
            try:
                df.to_parquet(cache_file)
                logger.info("Saved %d features to cache: %s", len(df), cache_file)
            except Exception as exc:
                logger.warning("Failed to save cache: %s", exc)
        return df

    def generate_trade_features(
        self,
        market_events: pd.DataFrame,
        window: str = "5min",
    ) -> pd.DataFrame:
        """Compute rolling trade flow features."""
        trade_events = market_events[market_events["event_type"] == "last_trade_price"]

        if trade_events.empty:
            logger.warning("No trade events found")
            return pd.DataFrame()

        payload = trade_events["data"].map(lambda value: value if isinstance(value, dict) else {})
        trades_df = pd.DataFrame(index=trade_events.index)
        trades_df["price"] = pd.to_numeric(
            payload.map(lambda item: item.get("price")),
            errors="coerce",
        )
        trades_df["size"] = pd.to_numeric(
            payload.map(lambda item: item.get("size", 1.0)),
            errors="coerce",
        ).fillna(1.0)
        trades_df["side"] = payload.map(lambda item: item.get("side", "UNKNOWN")).astype(str)
        trades_df["market_id"] = trade_events["market_id"].fillna("").astype(str)
        trades_df["token_id"] = trade_events["token_id"].fillna("").astype(str)
        trades_df = trades_df.dropna(subset=["price"])

        if trades_df.empty:
            return pd.DataFrame()

        trades_df = trades_df.sort_index()

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

    def build_feature_dataset_from_frames(
        self,
        market_events: pd.DataFrame,
        crypto_prices: pd.DataFrame,
        *,
        crypto_window: str = "5min",
    ) -> pd.DataFrame:
        """Join orderbook, crypto, and trade features into one dataset."""
        if market_events.empty:
            return pd.DataFrame()

        orderbook_features = self.generate_orderbook_features(market_events)
        trade_features = self.generate_trade_features(market_events, window=crypto_window)

        if not crypto_prices.empty:
            payload = crypto_prices["data"].map(
                lambda value: value if isinstance(value, dict) else {}
            )
            crypto_df = pd.DataFrame(index=crypto_prices.index)
            crypto_df["price"] = pd.to_numeric(
                payload.map(lambda item: item.get("price")),
                errors="coerce",
            )
            crypto_df["symbol"] = (
                crypto_prices["symbol"] if "symbol" in crypto_prices.columns else None
            )
            crypto_df = crypto_df.dropna(subset=["price"]).sort_index()

            if not orderbook_features.empty and not crypto_df.empty:
                aligned = align_features_to_events(
                    orderbook_features,
                    crypto_df,
                    window=crypto_window,
                )
            else:
                aligned = orderbook_features
        else:
            logger.warning("No crypto prices loaded, skipping crypto features")
            aligned = orderbook_features

        if not trade_features.empty and not aligned.empty:
            aligned = aligned.join(trade_features, how="left", rsuffix="_trade")

        logger.info(
            "Built feature dataset with %d rows and %d columns",
            len(aligned),
            len(aligned.columns),
        )
        return aligned
