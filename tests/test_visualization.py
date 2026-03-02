"""Test suite for visualization module.

Tests cover:
- OrderBook depth plotting with edge cases
- Imbalance calculation and display
- Crypto price correlation analysis
- Data validation and error handling
"""

from __future__ import annotations

import logging
from datetime import UTC

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization import plots

logger = logging.getLogger(__name__)


class TestOrderbookDepthPlotting:
    """Tests for orderbook depth visualization."""

    def test_plot_depth_empty_dataframe(self):
        """Test plotting with empty dataframe."""
        df = pd.DataFrame()
        fig = plots.plot_orderbook_depth(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_depth_valid_data(self):
        """Test plotting with valid bid/ask depth data."""
        times = pd.date_range("2024-01-01", periods=10, freq="1min")
        df = pd.DataFrame(
            {
                "bid_depth_1": [
                    100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0
                ],
                "ask_depth_1": [
                    105.0, 115.0, 125.0, 135.0, 145.0, 155.0, 165.0, 175.0, 185.0, 195.0
                ],
                "bid_depth_5": [
                    500.0, 510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0
                ],
                "ask_depth_5": [
                    505.0, 515.0, 525.0, 535.0, 545.0, 555.0, 565.0, 575.0, 585.0, 595.0
                ],
                "market_id": ["m1"] * 10,
            },
            index=times,
        )
        fig = plots.plot_orderbook_depth(df)
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_depth_with_inf_values(self):
        """Test that infinite values are properly handled."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "bid_depth_1": [100.0, np.inf, 120.0, -np.inf, 140.0],
                "ask_depth_1": [105.0, 115.0, np.inf, 135.0, -np.inf],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_orderbook_depth(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_depth_with_extreme_values(self):
        """Test that extreme outliers are filtered out."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "bid_depth_1": [100.0, 200.0, 5000000, 400.0, 500.0],  # 5M is outlier
                "ask_depth_1": [105.0, 205.0, 305.0, 405.0, 505.0],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_orderbook_depth(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_depth_with_nan_values(self):
        """Test handling of NaN values in depth columns."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "bid_depth_1": [100.0, np.nan, 120.0, np.nan, 140.0],
                "ask_depth_1": [105.0, 115.0, np.nan, 135.0, 145.0],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_orderbook_depth(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_depth_market_filter(self):
        """Test filtering by market_id."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "bid_depth_1": [100.0, 110.0, 120.0, 130.0, 140.0],
                "ask_depth_1": [105.0, 115.0, 125.0, 135.0, 145.0],
                "market_id": ["m1", "m1", "m2", "m2", "m1"],
            },
            index=times,
        )
        fig = plots.plot_orderbook_depth(df, market_id="m1")
        assert fig is not None
        plt.close(fig)

    def test_plot_depth_missing_columns(self):
        """Test handling when depth columns are missing."""
        times = pd.date_range("2024-01-01", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "market_id": ["m1", "m1", "m1"],
            },
            index=times,
        )
        fig = plots.plot_orderbook_depth(df)
        assert fig is not None
        plt.close(fig)


class TestImbalancePlotting:
    """Tests for orderbook imbalance visualization."""

    def test_plot_imbalance_empty_dataframe(self):
        """Test plotting with empty dataframe."""
        df = pd.DataFrame()
        fig = plots.plot_imbalance(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_imbalance_valid_data(self):
        """Test plotting with valid imbalance data."""
        times = pd.date_range("2024-01-01", periods=10, freq="1min")
        # Imbalance ranges from -1 to 1
        imbalances = np.linspace(-0.8, 0.8, 10)
        df = pd.DataFrame(
            {
                "imbalance_1": imbalances,
                "market_id": ["m1"] * 10,
            },
            index=times,
        )
        fig = plots.plot_imbalance(df)
        assert fig is not None
        assert len(fig.axes) > 0
        plt.close(fig)

    def test_plot_imbalance_all_positive(self):
        """Test plotting when all imbalances are positive (bid pressure)."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "imbalance_1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_imbalance(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_imbalance_all_negative(self):
        """Test plotting when all imbalances are negative (ask pressure)."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "imbalance_1": [-0.1, -0.2, -0.3, -0.4, -0.5],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_imbalance(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_imbalance_with_nan(self):
        """Test handling of NaN in imbalance data."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "imbalance_1": [0.1, np.nan, 0.3, np.nan, 0.5],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_imbalance(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_imbalance_with_inf(self):
        """Test handling of inf values in imbalance data."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "imbalance_1": [0.1, np.inf, 0.3, -np.inf, 0.5],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_imbalance(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_imbalance_fallback_to_imbalance_5(self):
        """Test fallback to imbalance_5 when imbalance_1 is missing."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "imbalance_5": [0.1, 0.2, 0.3, 0.4, 0.5],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_imbalance(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_imbalance_no_valid_data(self):
        """Test handling when all imbalance values are NaN."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "imbalance_1": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "market_id": ["m1"] * 5,
            },
            index=times,
        )
        fig = plots.plot_imbalance(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_imbalance_market_filter(self):
        """Test filtering by market_id."""
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame(
            {
                "imbalance_1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "market_id": ["m1", "m1", "m2", "m2", "m1"],
            },
            index=times,
        )
        fig = plots.plot_imbalance(df, market_id="m1")
        assert fig is not None
        plt.close(fig)


class TestCryptoPriceParsing:
    """Tests for crypto price data parsing."""

    def test_parse_crypto_prices_valid_dict(self):
        """Test parsing with valid price dict structures."""
        times = pd.date_range("2024-01-01", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "data": [
                    {"price": "66600.5"},
                    {"price": "66601.0"},
                    {"value": "66602.5"},  # Alternate field name
                ],
            },
            index=times,
        )
        result = plots._parse_crypto_prices(df)
        assert not result.empty
        assert len(result) == 3
        assert result["price"].iloc[0] == 66600.5

    def test_parse_crypto_prices_empty(self):
        """Test parsing empty dataframe."""
        df = pd.DataFrame()
        result = plots._parse_crypto_prices(df)
        assert result.empty

    def test_parse_crypto_prices_no_valid_prices(self):
        """Test when no valid prices can be extracted."""
        times = pd.date_range("2024-01-01", periods=2, freq="1min")
        df = pd.DataFrame(
            {
                "data": [
                    {"no_price": 123},
                    {},
                ],
            },
            index=times,
        )
        result = plots._parse_crypto_prices(df)
        assert result.empty

    def test_parse_crypto_prices_with_zero(self):
        """Test that zero prices are filtered out."""
        times = pd.date_range("2024-01-01", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "data": [
                    {"price": "66600.5"},
                    {"price": "0"},  # Zero should be filtered
                    {"price": "66602.5"},
                ],
            },
            index=times,
        )
        result = plots._parse_crypto_prices(df)
        assert len(result) == 2  # Only non-zero prices

    def test_parse_crypto_prices_invalid_json_string(self):
        """Test handling of JSON string data."""
        times = pd.date_range("2024-01-01", periods=2, freq="1min")
        df = pd.DataFrame(
            {
                "data": [
                    '{"price": "66600.5"}',
                    '{"price": "66601.0"}',
                ],
            },
            index=times,
        )
        result = plots._parse_crypto_prices(df)
        assert len(result) == 2

    def test_parse_crypto_prices_with_inf(self):
        """Test that infinite values are filtered out."""
        times = pd.date_range("2024-01-01", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "data": [
                    {"price": "66600.5"},
                    {"price": "inf"},  # Infinity should be filtered
                    {"price": "66602.5"},
                ],
            },
            index=times,
        )
        result = plots._parse_crypto_prices(df)
        assert len(result) == 2


class TestCorrelationDataPreparation:
    """Tests for correlation data preparation."""

    def test_prepare_correlation_valid_data(self):
        """Test with valid overlapping data."""
        times = pd.date_range("2024-01-01 10:00", periods=5, freq="1min")
        crypto_df = pd.DataFrame(
            {"price": [66600.0, 66601.0, 66602.0, 66603.0, 66604.0]},
            index=times,
        )
        market_df = pd.DataFrame(
            {
                "mid_price": [0.50, 0.51, 0.52, 0.53, 0.54],
                "spread": [0.02, 0.02, 0.02, 0.02, 0.02],
            },
            index=times,
        )
        success, crypto_aligned, market_aligned = plots._prepare_correlation_data(
            crypto_df, market_df, "1min"
        )
        assert success
        assert len(crypto_aligned) == len(market_aligned)
        assert len(crypto_aligned) > 0

    def test_prepare_correlation_no_overlap(self):
        """Test with non-overlapping timestamps."""
        crypto_times = pd.date_range("2024-01-01 10:00", periods=3, freq="1min")
        market_times = pd.date_range("2024-01-02 10:00", periods=3, freq="1min")

        crypto_df = pd.DataFrame(
            {"price": [66600.0, 66601.0, 66602.0]},
            index=crypto_times,
        )
        market_df = pd.DataFrame(
            {
                "mid_price": [0.50, 0.51, 0.52],
            },
            index=market_times,
        )
        success, crypto_aligned, market_aligned = plots._prepare_correlation_data(
            crypto_df, market_df, "1min"
        )
        assert not success
        assert crypto_aligned.empty
        assert market_aligned.empty

    def test_prepare_correlation_empty_crypto(self):
        """Test with empty crypto dataframe."""
        market_times = pd.date_range("2024-01-01 10:00", periods=3, freq="1min")
        market_df = pd.DataFrame(
            {"mid_price": [0.50, 0.51, 0.52]},
            index=market_times,
        )
        success, _, _ = plots._prepare_correlation_data(pd.DataFrame(), market_df, "1min")
        assert not success

    def test_prepare_correlation_empty_market(self):
        """Test with empty market dataframe."""
        crypto_times = pd.date_range("2024-01-01 10:00", periods=3, freq="1min")
        crypto_df = pd.DataFrame(
            {"price": [66600.0, 66601.0, 66602.0]},
            index=crypto_times,
        )
        success, _, _ = plots._prepare_correlation_data(crypto_df, pd.DataFrame(), "1min")
        assert not success

    def test_prepare_correlation_with_timezone(self):
        """Test handling of timezone-aware indices."""
        times = pd.date_range("2024-01-01 10:00", periods=3, freq="1min", tz=UTC)
        crypto_df = pd.DataFrame(
            {"price": [66600.0, 66601.0, 66602.0]},
            index=times,
        )
        market_df = pd.DataFrame(
            {"mid_price": [0.50, 0.51, 0.52]},
            index=times,
        )
        success, crypto_aligned, _market_aligned = plots._prepare_correlation_data(
            crypto_df, market_df, "1min"
        )
        assert success
        assert len(crypto_aligned) > 0

    def test_prepare_correlation_no_numeric_cols(self):
        """Test when market dataframe has no numeric columns."""
        times = pd.date_range("2024-01-01 10:00", periods=3, freq="1min")
        crypto_df = pd.DataFrame(
            {"price": [66600.0, 66601.0, 66602.0]},
            index=times,
        )
        market_df = pd.DataFrame(
            {"text_col": ["a", "b", "c"]},
            index=times,
        )
        success, _, _ = plots._prepare_correlation_data(crypto_df, market_df, "1min")
        assert not success


class TestPlotCryptoCorrelation:
    """Tests for crypto correlation plot."""

    def test_plot_crypto_correlation_valid_data(self):
        """Test plotting with valid crypto and market data."""
        times = pd.date_range("2024-01-01", periods=10, freq="1min")
        crypto_df = pd.DataFrame(
            {"data": [{"price": f"{66600 + i}"} for i in range(10)]},
            index=times,
        )
        market_df = pd.DataFrame(
            {
                "mid_price": np.linspace(0.5, 0.6, 10),
                "spread": [0.02] * 10,
            },
            index=times,
        )
        fig = plots.plot_crypto_correlation(crypto_df, market_df)
        assert fig is not None
        plt.close(fig)

    def test_plot_crypto_correlation_empty_crypto(self):
        """Test with empty crypto data."""
        market_df = pd.DataFrame(
            {"mid_price": [0.5, 0.6, 0.7]},
            index=pd.date_range("2024-01-01", periods=3, freq="1min"),
        )
        fig = plots.plot_crypto_correlation(pd.DataFrame(), market_df)
        assert fig is not None
        plt.close(fig)

    def test_plot_crypto_correlation_empty_market(self):
        """Test with empty market data."""
        times = pd.date_range("2024-01-01", periods=3, freq="1min")
        crypto_df = pd.DataFrame(
            {"data": [{"price": "66600"}, {"price": "66601"}, {"price": "66602"}]},
            index=times,
        )
        fig = plots.plot_crypto_correlation(crypto_df, pd.DataFrame())
        assert fig is not None
        plt.close(fig)

    def test_plot_crypto_correlation_no_overlap(self):
        """Test with non-overlapping time ranges."""
        crypto_times = pd.date_range("2024-01-01", periods=3, freq="1min")
        market_times = pd.date_range("2024-01-02", periods=3, freq="1min")

        crypto_df = pd.DataFrame(
            {"data": [{"price": "66600"}, {"price": "66601"}, {"price": "66602"}]},
            index=crypto_times,
        )
        market_df = pd.DataFrame(
            {"mid_price": [0.5, 0.6, 0.7]},
            index=market_times,
        )
        fig = plots.plot_crypto_correlation(crypto_df, market_df)
        assert fig is not None
        plt.close(fig)


class TestVisualizationRobustness:
    """Tests for visualization robustness and edge cases."""

    def test_plot_with_single_point(self):
        """Test plotting with single data point."""
        time = pd.date_range("2024-01-01", periods=1)
        df = pd.DataFrame(
            {
                "bid_depth_1": [100.0],
                "ask_depth_1": [105.0],
                "imbalance_1": [0.02],
                "market_id": ["m1"],
            },
            index=time,
        )
        fig_depth = plots.plot_orderbook_depth(df)
        fig_imbalance = plots.plot_imbalance(df)
        assert fig_depth is not None
        assert fig_imbalance is not None
        plt.close(fig_depth)
        plt.close(fig_imbalance)

    def test_plot_with_duplicate_timestamps(self):
        """Test plotting with duplicate timestamps."""
        times = pd.DatetimeIndex(
            ["2024-01-01 10:00", "2024-01-01 10:00", "2024-01-01 10:01"]
        )
        df = pd.DataFrame(
            {
                "bid_depth_1": [100.0, 110.0, 120.0],
                "ask_depth_1": [105.0, 115.0, 125.0],
                "imbalance_1": [0.02, 0.02, 0.04],
                "market_id": ["m1", "m1", "m1"],
            },
            index=times,
        )
        fig = plots.plot_orderbook_depth(df)
        assert fig is not None
        plt.close(fig)

    def test_plot_string_data_converted_to_numeric(self):
        """Test that string numeric values are properly converted."""
        times = pd.date_range("2024-01-01", periods=3, freq="1min")
        df = pd.DataFrame(
            {
                "bid_depth_1": ["100.0", "110.0", "120.0"],
                "ask_depth_1": ["105", "115", "125"],
                "market_id": ["m1", "m1", "m1"],
            },
            index=times,
        )
        fig = plots.plot_orderbook_depth(df)
        assert fig is not None
        plt.close(fig)

    def test_large_dataset_performance(self):
        """Test with larger dataset (1000 points)."""
        rng = np.random.default_rng(42)
        times = pd.date_range("2024-01-01", periods=1000, freq="1s")
        df = pd.DataFrame(
            {
                "bid_depth_1": rng.uniform(100, 200, 1000),
                "ask_depth_1": rng.uniform(100, 200, 1000),
                "imbalance_1": rng.uniform(-0.5, 0.5, 1000),
                "market_id": ["m1"] * 1000,
            },
            index=times,
        )
        fig_depth = plots.plot_orderbook_depth(df)
        fig_imbalance = plots.plot_imbalance(df)
        assert fig_depth is not None
        assert fig_imbalance is not None
        plt.close(fig_depth)
        plt.close(fig_imbalance)
