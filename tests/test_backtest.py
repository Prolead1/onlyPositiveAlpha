"""Test suite for backtester functionality.

These tests ensure compute_orderbook_features_df correctly processes book events
and prevents regressions like missing imbalance data.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtester.runner import BacktestConfig, BacktestRunner
from tests.prepared_helpers import write_prepared_manifest


class TestComputeOrderbookFeatures:
    """Tests for compute_orderbook_features_df method."""

    def test_load_market_events_filters_by_market_slug_prefix(self, tmp_path: Path):
        """Verify market slug prefix filtering keeps only matching rows."""
        storage_path = tmp_path / "storage"
        storage_path.mkdir(parents=True, exist_ok=True)

        base_time = datetime(2024, 1, 1, tzinfo=UTC)
        events = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "event_type": "book",
                    "market_id": "btc-updown-5m-123456",
                    "token_id": "token-btc",
                    "data": '{"asset_id": "token-btc"}',
                },
                {
                    "ts_event": base_time + timedelta(seconds=1),
                    "event_type": "book",
                    "market_id": "eth-updown-5m-999999",
                    "token_id": "token-eth",
                    "data": '{"asset_id": "token-eth"}',
                },
            ]
        )
        events.to_parquet(storage_path / "events.parquet", index=False)

        runner = BacktestRunner(storage_path)
        filtered = runner.load_market_events(market_slug_prefix="btc-updown-5m")

        assert len(filtered) == 1
        assert filtered["market_id"].nunique() == 1
        assert filtered["market_id"].iloc[0].startswith("btc-updown-5m")

    def test_load_market_events_pmxt_slug_filter_uses_mapping_by_default(self, tmp_path: Path):
        """Verify PMXT slug filtering resolves slug->conditionId via mapping shards."""
        storage_path = tmp_path / "pmxt"
        storage_path.mkdir(parents=True, exist_ok=True)

        mapping_dir = tmp_path / "mapping"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        mapping_payload = {
            "btc-updown-5m-123456": {
                "conditionId": "0xabc123",
                "clobTokenIds": ["token-btc"],
            },
            "eth-updown-5m-654321": {
                "conditionId": "0xdef456",
                "clobTokenIds": ["token-eth"],
            },
        }
        (mapping_dir / "gamma_updown_markets_2026-01-01.json").write_text(
            json.dumps(mapping_payload),
            encoding="utf-8",
        )

        base_time = datetime(2024, 1, 1, tzinfo=UTC)
        events = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "event_type": "book",
                    "market_id": "0xabc123",
                    "token_id": "token-btc",
                    "data": '{"asset_id": "token-btc"}',
                },
                {
                    "ts_event": base_time + timedelta(seconds=1),
                    "event_type": "book",
                    "market_id": "0xdef456",
                    "token_id": "token-eth",
                    "data": '{"asset_id": "token-eth"}',
                },
            ]
        )
        events.to_parquet(storage_path / "events.parquet", index=False)

        runner = BacktestRunner(storage_path)
        filtered = runner.load_market_events(market_slug_prefix="btc-updown-5m")

        assert len(filtered) == 1
        assert filtered["market_id"].nunique() == 1
        assert filtered["market_id"].iloc[0] == "0xabc123"

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

    def test_price_change_events_are_ignored_for_orderbook_features(self):
        """Verify non-book events do not produce orderbook feature rows."""
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

        assert features_df.empty

    def test_only_book_events_are_retained_from_mixed_event_stream(self):
        """Verify mixed streams only emit feature rows for book events."""
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

        # Price_change events should be ignored by feature generation.
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

        assert len(features_df) == 1
        assert features_df["imbalance_1"].notna().all()

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

        # Second book event: imbalance from bid=60, ask=140 (negative)
        events.append({
            "ts_event": base_time + timedelta(seconds=1),
            "event_type": "book",
            "market_id": "market1",
            "token_id": "token1",
            "data": {
                "asset_id": "token1",
                "bids": [{"price": "0.50", "size": "60"}],
                "asks": [{"price": "0.51", "size": "140"}],
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
        assert np.isclose(features_df["imbalance_1"].iloc[1], second_imb, atol=0.001)

    def test_book_features_are_isolated_per_token_within_market(self):
        """Verify book feature calculations do not leak across market sides."""
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        events = [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "market1",
                "token_id": "token_yes",
                "data": {
                    "asset_id": "token_yes",
                    "bids": [{"price": "0.60", "size": "100"}],
                    "asks": [{"price": "0.61", "size": "20"}],
                },
            },
            {
                "ts_event": base_time + timedelta(seconds=1),
                "event_type": "book",
                "market_id": "market1",
                "token_id": "token_no",
                "data": {
                    "asset_id": "token_no",
                    "bids": [{"price": "0.39", "size": "20"}],
                    "asks": [{"price": "0.40", "size": "100"}],
                },
            },
        ]

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        yes_imb = (100 - 20) / (100 + 20)
        no_imb = (20 - 100) / (20 + 100)

        token_yes = features_df[features_df["token_id"] == "token_yes"]
        token_no = features_df[features_df["token_id"] == "token_no"]

        assert token_yes["imbalance_1"].notna().all()
        assert token_no["imbalance_1"].notna().all()
        assert np.isclose(token_yes["imbalance_1"].iloc[0], yes_imb, atol=0.001)
        assert np.isclose(token_no["imbalance_1"].iloc[0], no_imb, atol=0.001)

    def test_stale_non_book_events_do_not_create_feature_rows(self):
        """Verify non-book events are ignored even when far from book updates."""
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        events = [
            {
                "ts_event": base_time,
                "event_type": "book",
                "market_id": "market1",
                "token_id": "token1",
                "data": {
                    "asset_id": "token1",
                    "bids": [{"price": "0.50", "size": "100"}],
                    "asks": [{"price": "0.51", "size": "80"}],
                },
            },
            {
                "ts_event": base_time + timedelta(minutes=10),
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
                },
            },
        ]

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        assert len(features_df) == 1
        assert features_df["imbalance_1"].iloc[0] == pytest.approx((100 - 80) / (100 + 80), rel=0, abs=0.001)

    def test_empty_events_returns_empty_dataframe(self):
        """Verify empty events return empty features dataframe."""
        market_events = pd.DataFrame(columns=["ts_event", "event_type", "market_id", "data"])
        market_events = market_events.set_index("ts_event")

        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        assert features_df.empty

    def test_pmxt_mode_drops_rows_without_token_id(self, tmp_path: Path):
        """Verify PMXT mode removes events that do not resolve to a token_id."""
        pmxt_dir = tmp_path / "pmxt"
        pmxt_dir.mkdir(parents=True, exist_ok=True)
        runner = BacktestRunner(pmxt_dir)

        df = pd.DataFrame(
            [
                {
                    "timestamp_received": datetime(2024, 1, 1, tzinfo=UTC),
                    "update_type": "book",
                    "market_id": "market1",
                    "data": {"bids": [{"price": "0.5", "size": "1"}], "asks": [{"price": "0.6", "size": "1"}]},
                },
                {
                    "timestamp_received": datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC),
                    "update_type": "book",
                    "market_id": "market1",
                    "token_id": "token_yes",
                    "data": {
                        "asset_id": "token_yes",
                        "bids": [{"price": "0.5", "size": "2"}],
                        "asks": [{"price": "0.6", "size": "2"}],
                    },
                },
            ]
        )

        normalized = runner._normalize_market_events_schema(df)

        assert len(normalized) == 1
        assert normalized["token_id"].iloc[0] == "token_yes"

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

    def test_depth_columns_computed_from_book_events(self):
        """Verify depth columns are computed correctly from book snapshots."""
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

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        # Check depth columns are populated for book rows.
        assert features_df["bid_depth_1"].notna().all()
        assert features_df["ask_depth_1"].notna().all()
        assert features_df["bid_depth_5"].notna().all()
        assert features_df["ask_depth_5"].notna().all()

        # Verify book event values
        assert features_df["bid_depth_1"].iloc[0] == 100
        assert features_df["ask_depth_1"].iloc[0] == 80
        assert features_df["bid_depth_5"].iloc[0] == 150  # 100 + 50
        assert features_df["ask_depth_5"].iloc[0] == 120  # 80 + 40

    def test_mixed_event_types_keep_book_rows_time_sorted(self):
        """Verify mixed event streams emit book rows in chronological order."""
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
        assert len(features_df) == 3

    def test_imbalance_5_also_computed(self):
        """Verify 5-level imbalance is computed from book snapshots."""
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

        market_events = pd.DataFrame(events).set_index("ts_event")
        runner = BacktestRunner(Path())
        features_df = runner.compute_orderbook_features_df(market_events)

        # Check both imbalance levels exist.
        assert "imbalance_5" in features_df.columns
        assert features_df["imbalance_5"].notna().all()

        # Verify calculation for 5-level
        # bid_depth_5 = 100+50+30+20+10 = 210
        # ask_depth_5 = 80+40+20+10+10 = 160
        # imbalance_5 = (210-160)/(210+160) = 50/370 ≈ 0.135
        expected = (210 - 160) / (210 + 160)
        assert np.isclose(features_df["imbalance_5"].iloc[0], expected, atol=0.001)


class TestBacktestIntegration:
    """Integration tests for full backtest workflow."""

    def test_realistic_event_stream_produces_complete_features(self):
        """Test mixed streams still produce complete features from book updates."""
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

        # Only book events should produce feature rows.
        assert len(features_df) == 3

        # All required columns should exist and have valid data
        required_cols = [
            "market_id", "token_id", "spread", "mid_price",
            "bid_depth_1", "ask_depth_1", "imbalance_1"
        ]
        for col in required_cols:
            assert col in features_df.columns
            assert features_df[col].notna().all(), f"{col} should not have NaN values"


class TestBacktestWithPMXTData:
    """Integration tests using PMXT cached parquet data."""

    MAX_ROWS_PER_FILE = 10000

    @pytest.fixture
    def data_path(self) -> Path:
        """Get path to cached PMXT orderbook data."""
        return Path("data/cached/pmxt")

    def test_pmxt_data_imbalance_not_all_nan(self, data_path: Path):
        """Test that PMXT data produces non-NaN imbalance values.

        This is a regression test for the imbalance bug where all imbalance
        values were NaN because only price_change events were processed.
        """
        if not data_path.exists() or not any(data_path.glob("*.parquet")):
            pytest.skip("No cached PMXT data available")

        # Load real data using runner (handles JSON parsing)
        runner = BacktestRunner(data_path)
        # Load only 1 parquet file for faster test execution
        market_events = runner.load_market_events(
            limit_files=1,
            max_rows_per_file=self.MAX_ROWS_PER_FILE,
        )

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

        print("\n✓ PMXT data test passed:")
        print(f"  - Total events: {len(market_events)}")
        print(f"  - Features computed: {len(features_df)}")
        print(f"  - Valid imbalance_1: {imbalance_valid_count} ({imbalance_valid_pct:.1f}%)")
        print(f"  - Event types: {market_events['event_type'].value_counts().to_dict()}")

    def test_pmxt_data_has_both_event_types(self, data_path: Path):
        """Verify PMXT data contains both book and price_change events."""
        if not data_path.exists() or not any(data_path.glob("*.parquet")):
            pytest.skip("No cached PMXT data available")

        runner = BacktestRunner(data_path)
        # Load only 1 parquet file for faster test execution
        market_events = runner.load_market_events(
            limit_files=1,
            max_rows_per_file=self.MAX_ROWS_PER_FILE,
        )

        if market_events.empty:
            pytest.skip("No market events loaded")

        event_types = market_events["event_type"].value_counts().to_dict()

        # Should have book events (source of imbalance)
        assert "book" in event_types, "Real data should contain 'book' events"
        assert event_types["book"] > 0, "Should have at least one book event"

        # Usually also has price_change events
        if "price_change" in event_types:
            assert event_types["price_change"] > 0

    def test_pmxt_data_book_only_features_align_with_book_event_coverage(
        self, data_path: Path
    ):
        """Verify book-only features do not exceed book event coverage."""
        if not data_path.exists() or not any(data_path.glob("*.parquet")):
            pytest.skip("No cached PMXT data available")

        runner = BacktestRunner(data_path)
        # Load only 1 parquet file for faster test execution
        market_events = runner.load_market_events(
            limit_files=1,
            max_rows_per_file=self.MAX_ROWS_PER_FILE,
        )

        if market_events.empty:
            pytest.skip("No market events loaded")

        # Count book events (original source of imbalance)
        book_event_count = (market_events["event_type"] == "book").sum()

        # Compute features from book events only.
        features_df = runner.compute_orderbook_features_df(market_events)

        if features_df.empty:
            pytest.skip("No features computed from data")

        # Count rows with valid imbalance
        imbalance_valid_count = features_df["imbalance_1"].notna().sum()

        # Book-only features should not exceed the number of book events.
        assert len(features_df) <= book_event_count
        assert imbalance_valid_count <= book_event_count

        print("\n✓ Book-only feature coverage:")
        print(f"  - Book events: {book_event_count}")
        print(f"  - Feature rows: {len(features_df)}")
        print(f"  - Rows with imbalance: {imbalance_valid_count}")

    def test_pmxt_data_all_required_columns_present(
        self, data_path: Path
    ):
        """Verify all required feature columns are present in PMXT data output."""
        if not data_path.exists() or not any(data_path.glob("*.parquet")):
            pytest.skip("No cached PMXT data available")

        runner = BacktestRunner(data_path)
        # Load only 1 parquet file for faster test execution
        market_events = runner.load_market_events(
            limit_files=1,
            max_rows_per_file=self.MAX_ROWS_PER_FILE,
        )

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

        # Verify at least some data in key columns from book updates.
        assert features_df["spread"].notna().any(), "spread should have some valid values"
        assert features_df["mid_price"].notna().any(), "mid_price should have some valid values"


class TestRunnerBacktestUtilities:
    """Tests for runner-native backtest simulation and summaries."""

    def test_simulate_hold_to_resolution_backtest_basic(self):
        runner = BacktestRunner(Path())
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_yes",
                    "mid_price": 0.6,
                    "signal": 1,
                }
            ]
        ).set_index("ts_event")

        resolution_frame = pd.DataFrame(
            [
                {
                    "market_id": "market1",
                    "resolved_at": base_time + timedelta(hours=2),
                    "winning_asset_id": "token_yes",
                    "winning_outcome": "YES",
                    "fees_enabled_market": True,
                }
            ]
        ).set_index("market_id")

        trades = runner.simulate_hold_to_resolution_backtest(
            signal_frame,
            resolution_frame,
            strategy_name="spread",
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )

        assert len(trades) == 1
        assert trades.iloc[0]["strategy"] == "spread"
        assert trades.iloc[0]["gross_pnl"] == pytest.approx(0.4)
        assert trades.iloc[0]["net_pnl"] == pytest.approx(0.4)
        assert trades.iloc[0]["hold_hours"] == pytest.approx(2.0)
        assert trades.iloc[0]["entry_candidate_count"] == 1
        assert trades.iloc[0]["entry_signal_abs"] == pytest.approx(1.0)
        assert trades.iloc[0]["entry_distance_from_mid"] == pytest.approx(0.1)
        assert "max_action_score" in trades.iloc[0]["entry_selection_reason"]

    def test_simulate_hold_to_resolution_backtest_takes_single_side_per_market(self):
        runner = BacktestRunner(Path())
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_yes",
                    "mid_price": 0.62,
                    "signal": 1,
                },
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_no",
                    "mid_price": 0.38,
                    "signal": 1,
                },
            ]
        ).set_index("ts_event")

        resolution_frame = pd.DataFrame(
            [
                {
                    "market_id": "market1",
                    "resolved_at": base_time + timedelta(hours=1),
                    "winning_asset_id": "token_yes",
                    "winning_outcome": "YES",
                    "fees_enabled_market": True,
                }
            ]
        ).set_index("market_id")

        trades = runner.simulate_hold_to_resolution_backtest(
            signal_frame,
            resolution_frame,
            strategy_name="single_side",
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )

        assert len(trades) == 1
        assert trades.iloc[0]["market_id"] == "market1"
        assert trades.iloc[0]["token_id"] == "token_yes"
        assert trades.iloc[0]["entry_candidate_count"] == 2
        assert trades.iloc[0]["entry_distance_from_mid"] == pytest.approx(0.12)

    def test_simulate_hold_to_resolution_backtest_buy_up_equals_sell_down_when_up_wins(self):
        runner = BacktestRunner(Path())
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        resolution_frame = pd.DataFrame(
            [
                {
                    "market_id": "market1",
                    "resolved_at": base_time + timedelta(hours=1),
                    "winning_asset_id": "token_up",
                    "winning_outcome": "UP",
                    "fees_enabled_market": True,
                }
            ]
        ).set_index("market_id")

        buy_up_signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_up",
                    "mid_price": 0.6,
                    "signal": 1,
                    "action_side": "buy",
                }
            ]
        ).set_index("ts_event")

        sell_down_signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_down",
                    "mid_price": 0.4,
                    "signal": -1,
                    "action_side": "sell",
                }
            ]
        ).set_index("ts_event")

        config = BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False)

        buy_up_trades = runner.simulate_hold_to_resolution_backtest(
            buy_up_signal_frame,
            resolution_frame,
            strategy_name="buy_up",
            config=config,
        )
        sell_down_trades = runner.simulate_hold_to_resolution_backtest(
            sell_down_signal_frame,
            resolution_frame,
            strategy_name="sell_down",
            config=config,
        )

        assert len(buy_up_trades) == 1
        assert len(sell_down_trades) == 1
        assert buy_up_trades.iloc[0]["gross_pnl"] == pytest.approx(0.4)
        assert sell_down_trades.iloc[0]["gross_pnl"] == pytest.approx(0.4)
        assert buy_up_trades.iloc[0]["net_pnl"] == pytest.approx(sell_down_trades.iloc[0]["net_pnl"])
        assert buy_up_trades.iloc[0]["net_pnl"] > 0
        assert sell_down_trades.iloc[0]["net_pnl"] > 0

    def test_simulate_hold_to_resolution_backtest_prefers_highest_action_score_within_lookahead(self):
        runner = BacktestRunner(Path())
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_up",
                    "mid_price": 0.70,
                    "signal": -1,
                    "action_side": "sell",
                    "action_score": 1.0,
                },
                {
                    "ts_event": base_time + timedelta(seconds=30),
                    "market_id": "market1",
                    "token_id": "token_down",
                    "mid_price": 0.20,
                    "signal": 1,
                    "action_side": "buy",
                    "action_score": 2.0,
                },
            ]
        ).set_index("ts_event")

        resolution_frame = pd.DataFrame(
            [
                {
                    "market_id": "market1",
                    "resolved_at": base_time + timedelta(hours=1),
                    "winning_asset_id": "token_down",
                    "winning_outcome": "DOWN",
                    "fees_enabled_market": True,
                }
            ]
        ).set_index("market_id")

        trades = runner.simulate_hold_to_resolution_backtest(
            signal_frame,
            resolution_frame,
            strategy_name="lookahead_rank",
            config=BacktestConfig(
                shares=1.0,
                fee_rate=0.0,
                fees_enabled=False,
                action_selection_lookahead_seconds=60,
            ),
        )

        assert len(trades) == 1
        assert trades.iloc[0]["token_id"] == "token_down"
        assert trades.iloc[0]["action_side"] == "buy"
        assert trades.iloc[0]["entry_action_score"] == pytest.approx(2.0)
        assert trades.iloc[0]["entry_candidate_count"] == 2
        assert "max_action_score" in trades.iloc[0]["entry_selection_reason"]

    def test_simulate_hold_to_resolution_backtest_sell_action_uses_selected_token_payout(self):
        runner = BacktestRunner(Path())
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_up",
                    "mid_price": 0.8,
                    "signal": -1,
                    "action_side": "sell",
                    "action_score": 1.0,
                }
            ]
        ).set_index("ts_event")

        resolution_frame = pd.DataFrame(
            [
                {
                    "market_id": "market1",
                    "resolved_at": base_time + timedelta(hours=1),
                    "winning_asset_id": "token_up",
                    "winning_outcome": "UP",
                    "fees_enabled_market": True,
                }
            ]
        ).set_index("market_id")

        trades = runner.simulate_hold_to_resolution_backtest(
            signal_frame,
            resolution_frame,
            strategy_name="sell_up",
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )

        assert len(trades) == 1
        assert trades.iloc[0]["action_side"] == "sell"
        assert trades.iloc[0]["trade_price"] == pytest.approx(0.8)
        assert trades.iloc[0]["exit_trade_price"] == pytest.approx(1.0)
        assert trades.iloc[0]["gross_pnl"] == pytest.approx(-0.2)
        assert trades.iloc[0]["gross_notional"] == pytest.approx(0.8)

    def test_simulate_hold_to_resolution_backtest_uses_signal_abs_for_capped_kelly_sizing(self):
        runner = BacktestRunner(Path())
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        resolution_frame = pd.DataFrame(
            [
                {
                    "market_id": "market1",
                    "resolved_at": base_time + timedelta(hours=1),
                    "winning_asset_id": "token_up",
                    "winning_outcome": "UP",
                    "fees_enabled_market": True,
                }
            ]
        ).set_index("market_id")

        low_confidence_signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_up",
                    "mid_price": 0.5,
                    "signal": 1,
                    "signal_abs": 0.01,
                    "action_side": "buy",
                }
            ]
        ).set_index("ts_event")

        high_confidence_signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_up",
                    "mid_price": 0.5,
                    "signal": 1,
                    "signal_abs": 0.04,
                    "action_side": "buy",
                }
            ]
        ).set_index("ts_event")

        config = BacktestConfig(
            sizing_policy="capped_kelly",
            sizing_kelly_fraction_cap=0.05,
            available_capital=100.0,
            fee_rate=0.0,
            fees_enabled=False,
        )

        low_trades = runner.simulate_hold_to_resolution_backtest(
            low_confidence_signal_frame,
            resolution_frame,
            strategy_name="low_confidence",
            config=config,
        )
        high_trades = runner.simulate_hold_to_resolution_backtest(
            high_confidence_signal_frame,
            resolution_frame,
            strategy_name="high_confidence",
            config=config,
        )

        assert len(low_trades) == 1
        assert len(high_trades) == 1
        assert low_trades.iloc[0]["gross_notional"] == pytest.approx(1.0)
        assert high_trades.iloc[0]["gross_notional"] == pytest.approx(4.0)
        assert high_trades.iloc[0]["gross_notional"] > low_trades.iloc[0]["gross_notional"]

    def test_simulate_hold_to_resolution_backtest_applies_price_aware_capped_kelly_scaling(self):
        runner = BacktestRunner(Path())
        base_time = datetime(2024, 1, 1, tzinfo=UTC)

        resolution_frame = pd.DataFrame(
            [
                {
                    "market_id": "market1",
                    "resolved_at": base_time + timedelta(hours=1),
                    "winning_asset_id": "token_up",
                    "winning_outcome": "UP",
                    "fees_enabled_market": True,
                }
            ]
        ).set_index("market_id")

        midpoint_signal_frame = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "market_id": "market1",
                    "token_id": "token_up",
                    "mid_price": 0.5,
                    "signal": 1,
                    "signal_abs": 0.04,
                    "action_side": "buy",
                }
            ]
        ).set_index("ts_event")

        tail_low_signal_frame = midpoint_signal_frame.copy()
        tail_low_signal_frame["mid_price"] = 0.1

        tail_high_signal_frame = midpoint_signal_frame.copy()
        tail_high_signal_frame["mid_price"] = 0.9

        config = BacktestConfig(
            sizing_policy="capped_kelly",
            sizing_kelly_fraction_cap=0.05,
            available_capital=100.0,
            fee_rate=0.0,
            fees_enabled=False,
        )

        midpoint_trades = runner.simulate_hold_to_resolution_backtest(
            midpoint_signal_frame,
            resolution_frame,
            strategy_name="midpoint_confidence",
            config=config,
        )
        tail_low_trades = runner.simulate_hold_to_resolution_backtest(
            tail_low_signal_frame,
            resolution_frame,
            strategy_name="tail_low_confidence",
            config=config,
        )
        tail_high_trades = runner.simulate_hold_to_resolution_backtest(
            tail_high_signal_frame,
            resolution_frame,
            strategy_name="tail_high_confidence",
            config=config,
        )

        assert len(midpoint_trades) == 1
        assert len(tail_low_trades) == 1
        assert len(tail_high_trades) == 1
        assert midpoint_trades.iloc[0]["gross_notional"] == pytest.approx(4.0)
        assert tail_low_trades.iloc[0]["gross_notional"] == pytest.approx(0.8)
        assert tail_high_trades.iloc[0]["gross_notional"] == pytest.approx(0.8)
        assert midpoint_trades.iloc[0]["gross_notional"] > tail_low_trades.iloc[0]["gross_notional"]
        assert midpoint_trades.iloc[0]["gross_notional"] > tail_high_trades.iloc[0]["gross_notional"]

    def test_run_backtest_repairs_mapping_and_completes(self, tmp_path: Path):
        storage_path = tmp_path / "pmxt"
        storage_path.mkdir(parents=True, exist_ok=True)

        runner = BacktestRunner(storage_path)
        base_time = datetime(2024, 1, 1, tzinfo=UTC)
        market_id = "cond_repair"
        token_yes = "token_yes"
        token_no = "token_no"

        market_events = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "event_type": "book",
                    "market_id": market_id,
                    "token_id": token_yes,
                    "data": {
                        "asset_id": token_yes,
                        "bids": [{"price": "0.98", "size": "100"}],
                        "asks": [{"price": "0.99", "size": "100"}],
                    },
                },
                {
                    "ts_event": base_time,
                    "event_type": "book",
                    "market_id": market_id,
                    "token_id": token_no,
                    "data": {
                        "asset_id": token_no,
                        "bids": [{"price": "0.01", "size": "100"}],
                        "asks": [{"price": "0.02", "size": "100"}],
                    },
                },
                {
                    "ts_event": base_time + timedelta(seconds=5),
                    "event_type": "market_resolved",
                    "market_id": market_id,
                    "token_id": token_yes,
                    "data": {
                        "winning_asset_id": None,
                        "winning_outcome": "Yes",
                    },
                },
            ]
        ).set_index("ts_event")
        features = runner.compute_orderbook_features_df(market_events)

        mapping_dir = tmp_path / "mapping"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        mapping_path = mapping_dir / "gamma_updown_markets_2024-01-01.json"
        mapping_path.write_text(
            json.dumps(
                {
                    "sample-slug": {
                        "conditionId": market_id,
                        "resolvedAt": "2024-01-01T00:00:05Z",
                        "winningAssetId": None,
                        "winningOutcome": "Yes",
                        "clobTokenIds": [token_yes, token_no],
                        "outcomePrices": ["0", "1"],
                        "feesEnabledMarket": True,
                    }
                }
            ),
            encoding="utf-8",
        )

        manifest_path = write_prepared_manifest(
            tmp_path=tmp_path,
            runner=runner,
            features=features,
            market_events=market_events,
            mapping_dir=mapping_dir,
        )

        def _long_yes(frame: pd.DataFrame) -> pd.Series:
            return (frame["token_id"].astype(str) == token_yes).astype(int)

        result = runner.run_backtest(
            mapping_dir=mapping_dir,
            prepared_manifest_path=manifest_path,
            strategy=_long_yes,
            strategy_name="long_yes",
            market_batch_size=1,
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )

        assert not result.trade_ledger.empty
        assert result.trade_ledger.iloc[0]["winning_asset_id"] == token_yes
        assert result.backtest_summary.iloc[0]["trades"] == 1

        updated_payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        assert updated_payload["sample-slug"]["winningAssetId"] is None

    def test_run_backtest_emits_metadata_and_propagates_run_id(
        self,
        tmp_path: Path,
    ):
        storage_path = tmp_path / "pmxt"
        storage_path.mkdir(parents=True, exist_ok=True)

        runner = BacktestRunner(storage_path)
        base_time = datetime(2024, 1, 1, tzinfo=UTC)
        market_id = "cond_meta"
        token_yes = "token_yes"
        token_no = "token_no"

        market_events = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "event_type": "book",
                    "market_id": market_id,
                    "token_id": token_yes,
                    "data": {
                        "asset_id": token_yes,
                        "bids": [{"price": "0.98", "size": "100"}],
                        "asks": [{"price": "0.99", "size": "100"}],
                    },
                },
                {
                    "ts_event": base_time,
                    "event_type": "book",
                    "market_id": market_id,
                    "token_id": token_no,
                    "data": {
                        "asset_id": token_no,
                        "bids": [{"price": "0.01", "size": "100"}],
                        "asks": [{"price": "0.02", "size": "100"}],
                    },
                },
                {
                    "ts_event": base_time + timedelta(seconds=5),
                    "event_type": "market_resolved",
                    "market_id": market_id,
                    "token_id": token_yes,
                    "data": {
                        "winning_asset_id": token_yes,
                        "winning_outcome": "Yes",
                    },
                },
            ]
        ).set_index("ts_event")
        features = runner.compute_orderbook_features_df(market_events)

        mapping_dir = tmp_path / "mapping"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
            json.dumps(
                {
                    "sample-slug": {
                        "conditionId": market_id,
                        "resolvedAt": "2024-01-01T00:00:05Z",
                        "winningAssetId": token_yes,
                        "winningOutcome": "Yes",
                        "clobTokenIds": [token_yes, token_no],
                        "outcomePrices": ["1", "0"],
                        "feesEnabledMarket": True,
                    }
                }
            ),
            encoding="utf-8",
        )

        manifest_path = write_prepared_manifest(
            tmp_path=tmp_path,
            runner=runner,
            features=features,
            market_events=market_events,
            mapping_dir=mapping_dir,
        )

        def _long_yes(frame: pd.DataFrame) -> pd.Series:
            return (frame["token_id"].astype(str) == token_yes).astype(int)

        result = runner.run_backtest(
            mapping_dir=mapping_dir,
            prepared_manifest_path=manifest_path,
            strategy=_long_yes,
            strategy_name="long_yes",
            market_batch_size=1,
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )

        assert result.metadata.run_id
        assert result.metadata.created_at is not None
        assert result.metadata.config_snapshot
        assert result.metadata.config_hash
        assert result.metadata.data_window == {"start": None, "end": None}
        assert result.metadata.cache_signature

        run_id = result.metadata.run_id
        assert set(result.trade_ledger["run_id"].unique()) == {run_id}
        assert set(result.backtest_summary["run_id"].unique()) == {run_id}
        assert set(result.equity_curve["run_id"].unique()) == {run_id}
        assert result.resolution_diagnostics.empty
        assert set(result.strategy_signals["long_yes"]["run_id"].unique()) == {run_id}

    def test_run_backtest_deterministic_outputs_for_identical_inputs(
        self,
        tmp_path: Path,
    ):
        storage_path = tmp_path / "pmxt"
        storage_path.mkdir(parents=True, exist_ok=True)

        runner = BacktestRunner(storage_path)
        base_time = datetime(2024, 1, 1, tzinfo=UTC)
        market_id = "cond_det"
        token_yes = "token_yes"
        token_no = "token_no"

        market_events = pd.DataFrame(
            [
                {
                    "ts_event": base_time,
                    "event_type": "book",
                    "market_id": market_id,
                    "token_id": token_yes,
                    "data": {
                        "asset_id": token_yes,
                        "bids": [{"price": "0.98", "size": "100"}],
                        "asks": [{"price": "0.99", "size": "100"}],
                    },
                },
                {
                    "ts_event": base_time,
                    "event_type": "book",
                    "market_id": market_id,
                    "token_id": token_no,
                    "data": {
                        "asset_id": token_no,
                        "bids": [{"price": "0.01", "size": "100"}],
                        "asks": [{"price": "0.02", "size": "100"}],
                    },
                },
                {
                    "ts_event": base_time + timedelta(seconds=5),
                    "event_type": "market_resolved",
                    "market_id": market_id,
                    "token_id": token_yes,
                    "data": {
                        "winning_asset_id": token_yes,
                        "winning_outcome": "Yes",
                    },
                },
            ]
        ).set_index("ts_event")
        features = runner.compute_orderbook_features_df(market_events)

        mapping_dir = tmp_path / "mapping"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        (mapping_dir / "gamma_updown_markets_2024-01-01.json").write_text(
            json.dumps(
                {
                    "sample-slug": {
                        "conditionId": market_id,
                        "resolvedAt": "2024-01-01T00:00:05Z",
                        "winningAssetId": token_yes,
                        "winningOutcome": "Yes",
                        "clobTokenIds": [token_yes, token_no],
                        "outcomePrices": ["1", "0"],
                        "feesEnabledMarket": True,
                    }
                }
            ),
            encoding="utf-8",
        )

        manifest_path = write_prepared_manifest(
            tmp_path=tmp_path,
            runner=runner,
            features=features,
            market_events=market_events,
            mapping_dir=mapping_dir,
        )

        def _long_yes(frame: pd.DataFrame) -> pd.Series:
            return (frame["token_id"].astype(str) == token_yes).astype(int)

        run_one = runner.run_backtest(
            mapping_dir=mapping_dir,
            prepared_manifest_path=manifest_path,
            strategy=_long_yes,
            strategy_name="long_yes",
            market_batch_size=1,
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )
        run_two = runner.run_backtest(
            mapping_dir=mapping_dir,
            prepared_manifest_path=manifest_path,
            strategy=_long_yes,
            strategy_name="long_yes",
            market_batch_size=1,
            config=BacktestConfig(shares=1.0, fee_rate=0.0, fees_enabled=False),
        )

        pd.testing.assert_frame_equal(
            run_one.trade_ledger.drop(columns=["run_id"]),
            run_two.trade_ledger.drop(columns=["run_id"]),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            run_one.backtest_summary.drop(columns=["run_id"]),
            run_two.backtest_summary.drop(columns=["run_id"]),
            check_dtype=False,
        )
        pd.testing.assert_frame_equal(
            run_one.equity_curve.drop(columns=["run_id"]),
            run_two.equity_curve.drop(columns=["run_id"]),
            check_dtype=False,
        )

        assert run_one.metadata.config_hash == run_two.metadata.config_hash
        assert run_one.metadata.cache_signature == run_two.metadata.cache_signature


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
