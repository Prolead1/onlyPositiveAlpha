"""Test suite for backtester functionality.

These tests ensure compute_orderbook_features_df correctly processes events
and prevents regressions like missing imbalance data.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.runner import BacktestRunner


class TestComputeOrderbookFeatures:
    """Tests for compute_orderbook_features_df method."""

    def test_book_events_compute_imbalance(self):
        """Verify book events produce valid imbalance data."""
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        events = [
            {
                "ts_event": base_time + timedelta(seconds=i),
                "event_type": "book",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "asset_id": "token1",
                    "bids": [
                        {"price": "0.50", "size": "100"},
                        {"price": "0.49", "size": "50"},
                    ],
                    "asks": [
                        {"price": "0.51", "size": "80"},
                        {"price": "0.52", "size": "40"},
                    ],
                }
            }
            for i in range(3)
        ]

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        assert len(features_df) == 3
        assert "imbalance_1" in features_df.columns
        assert features_df["imbalance_1"].notna().all(), "All book events should have imbalance_1"

        # Verify calculation: (100-80)/(100+80) = 20/180 ≈ 0.111
        expected = (100 - 80) / (100 + 80)
        assert np.isclose(features_df["imbalance_1"].iloc[0], expected, atol=0.001)

    def test_price_change_events_compute_spread(self):
        """Verify price_change events produce valid spread/mid_price."""
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        events = [
            {
                "ts_event": base_time + timedelta(seconds=i),
                "event_type": "price_change",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "price_changes": [{
                        "asset_id": "token1",
                        "best_bid": "0.50",
                        "best_ask": "0.52",
                        "price": "0.51",
                        "side": "BUY",
                        "size": "10"
                    }]
                }
            }
            for i in range(3)
        ]

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        assert len(features_df) == 3
        assert features_df["spread"].notna().all()
        assert features_df["mid_price"].notna().all()

        # Verify spread: 0.52 - 0.50 = 0.02
        assert np.isclose(features_df["spread"].iloc[0], 0.02, atol=0.001)
        # Verify mid_price: (0.50 + 0.52) / 2 = 0.51
        assert np.isclose(features_df["mid_price"].iloc[0], 0.51, atol=0.001)

    def test_hybrid_approach_forward_fills_imbalance(self):
        """Verify imbalance forward-fills from book to price_change events."""
        events = []
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        # Book event with imbalance
        events.append({
            "ts_event": base_time,
            "event_type": "book",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "asset_id": "token1",
                "bids": [{"price": "0.50", "size": "100"}],
                "asks": [{"price": "0.51", "size": "80"}],
            }
        })

        # Price_change events (should get forward-filled imbalance)
        events.extend([
            {
                "ts_event": base_time + timedelta(seconds=i),
                "event_type": "price_change",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "price_changes": [{
                        "asset_id": "token1",
                        "best_bid": "0.50",
                        "best_ask": "0.51",
                        "price": "0.505",
                        "side": "BUY",
                        "size": "10"
                    }]
                }
            }
            for i in range(1, 5)
        ])

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        assert len(features_df) == 5
        # All rows should have imbalance (forward-filled)
        assert features_df["imbalance_1"].notna().all(), \
            "Forward-fill should propagate imbalance to all events"

        # All price_change events should have same imbalance as book event
        book_imbalance = features_df["imbalance_1"].iloc[0]
        for i in range(1, 5):
            assert np.isclose(features_df["imbalance_1"].iloc[i], book_imbalance, atol=0.001)

    def test_multiple_book_events_update_imbalance(self):
        """Verify imbalance updates when new book events arrive."""
        events = []
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        # First book event: imbalance from bid=100, ask=80
        events.append({
            "ts_event": base_time,
            "event_type": "book",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "asset_id": "token1",
                "bids": [{"price": "0.50", "size": "100"}],
                "asks": [{"price": "0.51", "size": "80"}],
            }
        })

        # Price_change event (gets first imbalance)
        events.append({
            "ts_event": base_time + timedelta(seconds=1),
            "event_type": "price_change",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "price_changes": [{
                    "asset_id": "token1",
                    "best_bid": "0.50",
                    "best_ask": "0.51",
                    "price": "0.505",
                    "side": "BUY",
                    "size": "10"
                }]
            }
        })

        # Second book event: imbalance from bid=60, ask=140 (negative)
        events.append({
            "ts_event": base_time + timedelta(seconds=2),
            "event_type": "book",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "asset_id": "token1",
                "bids": [{"price": "0.50", "size": "60"}],
                "asks": [{"price": "0.51", "size": "140"}],
            }
        })

        # Another price_change event (gets updated imbalance)
        events.append({
            "ts_event": base_time + timedelta(seconds=3),
            "event_type": "price_change",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "price_changes": [{
                    "asset_id": "token1",
                    "best_bid": "0.50",
                    "best_ask": "0.51",
                    "price": "0.505",
                    "side": "BUY",
                    "size": "10"
                }]
            }
        })

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        # First imbalance: (100-80)/(100+80) = 0.111
        first_imb = (100 - 80) / (100 + 80)
        # Second imbalance: (60-140)/(60+140) = -0.4
        second_imb = (60 - 140) / (60 + 140)

        assert np.isclose(features_df["imbalance_1"].iloc[0], first_imb, atol=0.001)
        assert np.isclose(features_df["imbalance_1"].iloc[1], first_imb, atol=0.001)
        assert np.isclose(features_df["imbalance_1"].iloc[2], second_imb, atol=0.001)
        assert np.isclose(features_df["imbalance_1"].iloc[3], second_imb, atol=0.001)

    def test_empty_events_returns_empty_dataframe(self):
        """Verify empty events return empty features dataframe."""
        market_events = pd.DataFrame(columns=["ts_event", "event_type", "market_id", "data"])
        market_events = market_events.set_index("ts_event")

        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        assert features_df.empty

    def test_malformed_book_event_skipped(self):
        """Verify malformed book events are skipped without crashing."""
        events = [
            # Valid book event
            {
                "ts_event": datetime(2024, 1, 1, tzinfo=UTC),
                "event_type": "book",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "asset_id": "token1",
                    "bids": [{"price": "0.50", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "80"}],
                }
            },
            # Malformed book event (missing bids)
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
                "event_type": "book",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "asset_id": "token1",
                    "asks": [{"price": "0.51", "size": "80"}],
                }
            },
            # Valid book event
            {
                "ts_event": datetime(2024, 1, 1, 0, 0, 2, tzinfo=UTC),
                "event_type": "book",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "asset_id": "token1",
                    "bids": [{"price": "0.50", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "80"}],
                }
            },
        ]

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        # Should have 2 valid events (malformed one skipped)
        assert len(features_df) == 2
        assert features_df["imbalance_1"].notna().all()

    def test_depth_columns_forward_filled(self):
        """Verify depth columns are also forward-filled."""
        events = []
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        # Book event with depth
        events.append({
            "ts_event": base_time,
            "event_type": "book",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "asset_id": "token1",
                "bids": [
                    {"price": "0.50", "size": "100"},
                    {"price": "0.49", "size": "50"},
                ],
                "asks": [
                    {"price": "0.51", "size": "80"},
                    {"price": "0.52", "size": "40"},
                ],
            }
        })

        # Price_change events
        events.extend([
            {
                "ts_event": base_time + timedelta(seconds=i),
                "event_type": "price_change",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "price_changes": [{
                        "asset_id": "token1",
                        "best_bid": "0.50",
                        "best_ask": "0.51",
                        "price": "0.505",
                        "side": "BUY",
                        "size": "10"
                    }]
                }
            }
            for i in range(1, 3)
        ])

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        # Check depth columns are forward-filled (all non-null after forward fill)
        assert features_df["bid_depth_1"].notna().all()
        assert features_df["ask_depth_1"].notna().all()
        assert features_df["bid_depth_5"].notna().all()
        assert features_df["ask_depth_5"].notna().all()

        # Verify book event values
        assert features_df["bid_depth_1"].iloc[0] == 100
        assert features_df["ask_depth_1"].iloc[0] == 80
        assert features_df["bid_depth_5"].iloc[0] == 150  # 100 + 50
        assert features_df["ask_depth_5"].iloc[0] == 120  # 80 + 40

        # Note: price_change events have partial depth from trade side,
        # but forward fill ensures no NaN values remain
        assert all(features_df["ask_depth_5"] == 120)  # 80 + 40

    def test_both_event_types_sorted_by_time(self):
        """Verify mixed event types are properly time-sorted."""
        events = []
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        # Interleaved book and price_change events
        for i in range(5):
            if i % 2 == 0:
                events.append({
                    "ts_event": base_time + timedelta(seconds=i),
                    "event_type": "book",
                    "market_id": "market1",
                    "token_id": "token1",
                    "data": {
                        "asset_id": "token1",
                        "bids": [{"price": "0.50", "size": "100"}],
                        "asks": [{"price": "0.51", "size": "80"}],
                    }
                })
            else:
                events.append({
                    "ts_event": base_time + timedelta(seconds=i),
                    "event_type": "price_change",
                    "market_id": "market1",
                    "token_id": "token1",
                    "data": {
                        "price_changes": [{
                            "asset_id": "token1",
                            "best_bid": "0.50",
                            "best_ask": "0.51",
                            "price": "0.505",
                            "side": "BUY",
                            "size": "10"
                        }]
                    }
                })

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        # Verify time ordering
        assert features_df.index.is_monotonic_increasing
        assert len(features_df) == 5

    def test_imbalance_5_also_computed(self):
        """Verify 5-level imbalance is also computed and forward-filled."""
        events = []
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        # Book event with 5 levels
        events.append({
            "ts_event": base_time,
            "event_type": "book",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "asset_id": "token1",
                "bids": [
                    {"price": "0.50", "size": "100"},
                    {"price": "0.49", "size": "50"},
                    {"price": "0.48", "size": "30"},
                    {"price": "0.47", "size": "20"},
                    {"price": "0.46", "size": "10"},
                ],
                "asks": [
                    {"price": "0.51", "size": "80"},
                    {"price": "0.52", "size": "40"},
                    {"price": "0.53", "size": "20"},
                    {"price": "0.54", "size": "10"},
                    {"price": "0.55", "size": "10"},
                ],
            }
        })

        # Price_change event
        events.append({
            "ts_event": base_time + timedelta(seconds=1),
            "event_type": "price_change",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "price_changes": [{
                    "asset_id": "token1",
                    "best_bid": "0.50",
                    "best_ask": "0.51",
                    "price": "0.505",
                    "side": "BUY",
                    "size": "10"
                }]
            }
        })

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        # Check both imbalance levels exist and are forward-filled
        assert "imbalance_5" in features_df.columns
        assert features_df["imbalance_5"].notna().all()

        # Verify calculation for 5-level
        # bid_depth_5 = 100+50+30+20+10 = 210
        # ask_depth_5 = 80+40+20+10+10 = 160
        # imbalance_5 = (210-160)/(210+160) = 50/370 ≈ 0.135
        expected = (210 - 160) / (210 + 160)
        assert np.isclose(features_df["imbalance_5"].iloc[0], expected, atol=0.001)
        assert np.isclose(features_df["imbalance_5"].iloc[1], expected, atol=0.001)


class TestBacktestIntegration:
    """Integration tests for full backtest workflow."""

    def test_realistic_event_stream_produces_complete_features(self):
        """Test with realistic event mix produces all required features."""
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        # Simulate realistic event stream: few book events, many price_change events
        # Book event every 10 seconds
        events = [
            {
                "ts_event": base_time + timedelta(seconds=book_idx * 10),
                "event_type": "book",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "asset_id": "token1",
                    "bids": [{"price": "0.50", "size": f"{100 + book_idx * 10}"}],
                    "asks": [{"price": "0.51", "size": f"{80 + book_idx * 5}"}],
                }
            }
            for book_idx in range(3)
        ]

        # 9 price_change events between each book event
        events.extend([
            {
                "ts_event": base_time + timedelta(seconds=book_idx * 10 + pc_idx),
                "event_type": "price_change",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "price_changes": [{
                        "asset_id": "token1",
                        "best_bid": "0.50",
                        "best_ask": "0.51",
                        "price": "0.505",
                        "side": "BUY",
                        "size": "10"
                    }]
                }
            }
            for book_idx in range(3)
            for pc_idx in range(1, 10)
        ])

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        # Should have all events processed
        assert len(features_df) == 30  # 3 book + 27 price_change

        # All required columns should exist and have valid data
        required_cols = [
            "market_id", "token_id", "spread", "mid_price",
            "bid_depth_1", "ask_depth_1", "imbalance_1"
        ]
        for col in required_cols:
            assert col in features_df.columns
            assert features_df[col].notna().all(), f"{col} should not have NaN values"


class TestBacktestWithRealData:
    """Integration tests using real cached parquet data."""

    @pytest.fixture
    def data_path(self) -> Path:
        """Get path to cached market data."""
        return Path("data/cached/stream_feeds/polymarket_market")

    def test_real_data_imbalance_not_all_nan(self, data_path: Path):
        """Test that real data produces non-NaN imbalance values.

        This is a regression test for the imbalance bug where all imbalance
        values were NaN because only price_change events were processed.
        """
        if not data_path.exists() or not any(data_path.glob("*.parquet")):
            pytest.skip("No cached market data available")

        # Load real data using runner (handles JSON parsing)
        # data_path is data/cached/stream_feeds/polymarket_market
        # BacktestRunner expects storage_path that contains polymarket_market subfolder
        # So we need to pass data_path.parent (data/cached/stream_feeds)
        storage_path = data_path.parent
        runner = BacktestRunner(storage_path)
        # Load only 1 parquet file for faster test execution
        market_events = runner.load_market_events(limit_files=1)

        if market_events.empty:
            pytest.skip("No market events loaded")

        # Compute features
        features_df = runner.compute_orderbook_features_df(market_events)

        # Assertions
        assert not features_df.empty, "Features dataframe should not be empty"
        assert "imbalance_1" in features_df.columns, "Should have imbalance_1 column"

        # Key assertion: imbalance should have SOME valid values (not all NaN)
        imbalance_valid_count = features_df["imbalance_1"].notna().sum()
        assert imbalance_valid_count > 0, \
            "imbalance_1 should have at least some non-NaN values (regression check)"

        # Should have a reasonable percentage of valid imbalance data
        # (depends on book event frequency, but at least 1% should have imbalance)
        imbalance_valid_pct = imbalance_valid_count / len(features_df) * 100
        assert imbalance_valid_pct >= 1.0, \
            f"At least 1% of rows should have imbalance data, got {imbalance_valid_pct:.2f}%"

        print("\n✓ Real data test passed:")
        print(f"  - Total events: {len(market_events)}")
        print(f"  - Features computed: {len(features_df)}")
        print(f"  - Valid imbalance_1: {imbalance_valid_count} ({imbalance_valid_pct:.1f}%)")
        print(f"  - Event types: {market_events['event_type'].value_counts().to_dict()}")

    def test_real_data_has_both_event_types(self, data_path: Path):
        """Verify real data contains both book and price_change events."""
        if not data_path.exists() or not any(data_path.glob("*.parquet")):
            pytest.skip("No cached market data available")

        storage_path = data_path.parent
        runner = BacktestRunner(storage_path)
        # Load only 1 parquet file for faster test execution
        market_events = runner.load_market_events(limit_files=1)

        if market_events.empty:
            pytest.skip("No market events loaded")

        event_types = market_events["event_type"].value_counts().to_dict()

        # Should have book events (source of imbalance)
        assert "book" in event_types, "Real data should contain 'book' events"
        assert event_types["book"] > 0, "Should have at least one book event"

        # Usually also has price_change events
        if "price_change" in event_types:
            assert event_types["price_change"] > 0

    def test_real_data_forward_fill_increases_imbalance_coverage(
        self, data_path: Path
    ):
        """Verify forward-fill increases imbalance data coverage.

        Tests that the hybrid approach (book + price_change with forward-fill)
        provides more complete imbalance data than just book events alone.
        """
        if not data_path.exists() or not any(data_path.glob("*.parquet")):
            pytest.skip("No cached market data available")

        storage_path = data_path.parent
        runner = BacktestRunner(storage_path)
        # Load only 1 parquet file for faster test execution
        market_events = runner.load_market_events(limit_files=1)

        if market_events.empty:
            pytest.skip("No market events loaded")

        # Count book events (original source of imbalance)
        book_event_count = (market_events["event_type"] == "book").sum()

        # Compute features with forward-fill
        features_df = runner.compute_orderbook_features_df(market_events)

        if features_df.empty:
            pytest.skip("No features computed from data")

        # Count rows with valid imbalance
        imbalance_valid_count = features_df["imbalance_1"].notna().sum()

        # Forward-fill should provide imbalance for more rows than just book events
        # (because price_change events between book events get forward-filled values)
        assert imbalance_valid_count >= book_event_count, (
            f"Forward-fill should provide imbalance for at least {book_event_count} rows "
            f"(book event count), but only {imbalance_valid_count} have valid imbalance"
        )

        print("\n✓ Forward-fill effectiveness:")
        print(f"  - Book events: {book_event_count}")
        print(f"  - Rows with imbalance: {imbalance_valid_count}")
        print(f"  - Coverage increase: {imbalance_valid_count - book_event_count} additional rows")

    def test_real_data_all_required_columns_present(
        self, data_path: Path
    ):
        """Verify all required feature columns are present in real data output."""
        if not data_path.exists() or not any(data_path.glob("*.parquet")):
            pytest.skip("No cached market data available")

        storage_path = data_path.parent
        runner = BacktestRunner(storage_path)
        # Load only 1 parquet file for faster test execution
        market_events = runner.load_market_events(limit_files=1)

        if market_events.empty:
            pytest.skip("No market events loaded")

        features_df = runner.compute_orderbook_features_df(market_events)

        if features_df.empty:
            pytest.skip("No features computed from data")

        # Check all required columns exist
        required_cols = [
            "market_id", "token_id", "spread", "spread_bps", "mid_price",
            "bid_depth_1", "ask_depth_1", "bid_depth_5", "ask_depth_5",
            "imbalance_1", "imbalance_5", "bid_ask_ratio"
        ]

        for col in required_cols:
            assert col in features_df.columns, f"Missing required column: {col}"

        # Verify at least some data in key columns
        # (spread and mid_price should always have values from price_change events)
        assert features_df["spread"].notna().any(), "spread should have some valid values"
        assert features_df["mid_price"].notna().any(), "mid_price should have some valid values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
