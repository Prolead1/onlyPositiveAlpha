"""Visualization module for streaming data diagnostics and analysis."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.axes
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    INVALID_TIMESTAMP_THRESHOLD,
    ROLLING_CORRELATION_MIN_PERIODS,
    ROLLING_CORRELATION_WINDOW,
)
from utils import (
    ensure_directory,
    parse_orderbook_level,
)
from utils.dataframes import get_numeric_columns

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from matplotlib.figure import Figure

    from backtester.runner import BacktestRunner


class ReportData(NamedTuple):
    """Container for report data dataframes."""

    orderbook_features: pd.DataFrame
    trade_features: pd.DataFrame
    crypto_prices: pd.DataFrame
    market_events: pd.DataFrame


# Set style
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# Constants
MAX_DEPTH_THRESHOLD = 1000000
DEPTH_PLOT_PERCENTILE_CAP = 0.995
DEPTH_PLOT_MIN_RATIO_FOR_CAP = 3.0


def _format_datetime_axis(ax: matplotlib.axes.Axes) -> None:
    """Format x-axis for datetime data with appropriate tick locators and formatters.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to format.
    """
    try:
        # Only apply formatters if the axis data actually contains valid dates
        if hasattr(ax, "collections") or len(ax.get_lines()) > 0:
            # Check if x-data looks like valid dates (not huge numbers from pandas)
            for line in ax.get_lines():
                xdata = np.asarray(line.get_xdata())
                if len(xdata) > 0 and np.all(xdata > INVALID_TIMESTAMP_THRESHOLD):
                    # If all x values are > 10^8, they're likely invalid (timestamps)
                    # Skip formatting for this axis
                    return

            # Automatically choose appropriate locator and formatter based on time span
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    except Exception as exc:
        logger.debug("Failed to format datetime axis: %s", exc)


def plot_orderbook_snapshot(
    bids: list[dict[str, str]] | list[list[str]],
    asks: list[dict[str, str]] | list[list[str]],
    depth: int = 10,
    title: str = "Order Book Snapshot",
) -> Figure:
    """Plot a single orderbook snapshot.

    Parameters
    ----------
    bids : list
        List of bid levels [price, size] or dicts with 'price' and 'size'.
    asks : list
        List of ask levels [price, size] or dicts with 'price' and 'size'.
    depth : int
        Number of levels to display.
    title : str
        Plot title.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Parse bid/ask levels using utility function
    def parse_levels_list(levels: list) -> tuple[list[float], list[float]]:
        prices, sizes = [], []
        for level in levels[:depth]:
            price, size = parse_orderbook_level(level)
            prices.append(price)
            sizes.append(size)
        return prices, sizes

    bid_prices, bid_sizes = parse_levels_list(bids)
    ask_prices, ask_sizes = parse_levels_list(asks)

    # Plot bids (green) and asks (red)
    ax.barh(range(len(bid_prices)), bid_sizes, color="green", alpha=0.6, label="Bids")
    ax.barh(
        range(len(bid_prices), len(bid_prices) + len(ask_prices)),
        ask_sizes,
        color="red",
        alpha=0.6,
        label="Asks",
    )

    # Set y-axis labels to prices
    all_prices = bid_prices + ask_prices
    ax.set_yticks(range(len(all_prices)))
    ax.set_yticklabels([f"${p:.3f}" for p in all_prices])

    ax.set_xlabel("Size")
    ax.set_ylabel("Price Level")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    return fig


def plot_spread_timeseries(features_df: pd.DataFrame, market_id: str | None = None) -> Figure:
    """Plot spread and mid-price over time.

    Parameters
    ----------
    features_df : pd.DataFrame
        Orderbook features dataframe with 'spread' and 'mid_price' columns.
    market_id : str | None
        Filter by market ID if provided.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if market_id:
        df = features_df[features_df["market_id"] == market_id].copy()
    else:
        df = features_df.copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot spread
    axes[0].plot(df.index, df["spread"], color="blue", linewidth=1)
    axes[0].fill_between(df.index, 0, df["spread"], alpha=0.3, color="blue")
    axes[0].set_ylabel("Spread")
    axes[0].set_title("Bid-Ask Spread Over Time")
    axes[0].grid(visible=True, alpha=0.3)

    # Plot mid-price
    axes[1].plot(df.index, df["mid_price"], color="purple", linewidth=1.5)
    axes[1].set_ylabel("Mid Price")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Mid Price Over Time")
    axes[1].grid(visible=True, alpha=0.3)

    # Format x-axis for both plots
    _format_datetime_axis(axes[1])

    plt.tight_layout()
    return fig


def _prepare_depth_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare depth dataframe by cleaning and validating data."""
    df = df.copy()

    # Convert to numeric and clean
    for col in ["bid_depth_1", "ask_depth_1", "bid_depth_5", "ask_depth_5"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df.loc[df[col] > MAX_DEPTH_THRESHOLD, col] = np.nan

    return df


def _plot_depth_side(
    ax: matplotlib.axes.Axes,
    valid_data: pd.DataFrame,
    df: pd.DataFrame,
    side: str,
) -> None:
    """Plot depth for one side (bid or ask)."""
    is_bid = side == "bid"
    color = "green" if is_bid else "red"
    dark_color = "darkgreen" if is_bid else "darkred"
    label_prefix = "Bid" if is_bid else "Ask"

    timestamps = valid_data.index
    depth_1_col = f"{side}_depth_1"
    depth_5_col = f"{side}_depth_5"

    # Plot L1 depth
    values_1 = valid_data[depth_1_col].fillna(0).to_numpy()
    if values_1.max() > 0:
        ax.fill_between(
            timestamps,
            0,
            values_1,
            color=color,
            alpha=0.3,
            label=f"{label_prefix} Depth (L1)",
            interpolate=True,
        )
        ax.plot(timestamps, values_1, color=dark_color, linewidth=0.5, alpha=0.7)

    # Plot L5 depth if available
    if depth_5_col in valid_data.columns:
        values_5 = valid_data[depth_5_col].fillna(0).to_numpy()
        if values_5.max() > 0:
            ax.plot(
                timestamps,
                values_5,
                color=color,
                linewidth=1.5,
                alpha=0.8,
                label=f"{label_prefix} Depth (L5)",
            )

    # Apply robust y-axis cap
    depth_series = []
    for col in [depth_1_col, depth_5_col]:
        if col in df.columns:
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            valid_col = numeric_col[(numeric_col > 0) & np.isfinite(numeric_col)]
            if not valid_col.empty:
                depth_series.append(valid_col)

    if depth_series:
        all_values = pd.concat(depth_series, ignore_index=True)
        robust_upper = float(all_values.quantile(DEPTH_PLOT_PERCENTILE_CAP))
        max_depth = float(all_values.max())

        if (
            np.isfinite(robust_upper)
            and robust_upper > 0
            and max_depth > robust_upper * DEPTH_PLOT_MIN_RATIO_FOR_CAP
        ):
            ax.set_ylim(0, robust_upper)
            ax.text(
                0.01,
                0.98,
                f"Y-axis capped at {DEPTH_PLOT_PERCENTILE_CAP:.1%} percentile",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                alpha=0.8,
            )

    ax.set_ylabel(f"{label_prefix} Depth")
    ax.legend(loc="upper right")
    ax.grid(visible=True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")


def plot_orderbook_depth(features_df: pd.DataFrame, market_id: str | None = None) -> Figure:
    """Plot bid/ask depth over time.

    Parameters
    ----------
    features_df : pd.DataFrame
        Orderbook features with depth columns.
    market_id : str | None
        Filter by market ID if provided.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    # Filter by market if specified
    if market_id:
        df = features_df[features_df["market_id"] == market_id].copy()
    else:
        df = features_df.copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    # Check if required columns exist
    if "bid_depth_1" not in df.columns or "ask_depth_1" not in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Missing depth columns", ha="center", va="center")
        return fig

    # Create two subplots: one for bids (top), one for asks (bottom)
    fig, (ax_bid, ax_ask) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Prepare data
    df = _prepare_depth_dataframe(df)

    # Filter to rows with valid data
    valid_data = df[(df["bid_depth_1"].notna()) | (df["ask_depth_1"].notna())].copy()

    if valid_data.empty:
        ax_bid.text(0.5, 0.5, "No valid depth data", ha="center", va="center")
        return fig

    # Plot bid and ask sides
    ax_bid.set_title("Order Book Depth Over Time")
    _plot_depth_side(ax_bid, valid_data, df, "bid")
    _plot_depth_side(ax_ask, valid_data, df, "ask")

    # Format x-axis for datetime (only on bottom subplot)
    ax_ask.set_xlabel("Time")
    _format_datetime_axis(ax_ask)

    plt.tight_layout()

    return fig


def plot_imbalance(features_df: pd.DataFrame, market_id: str | None = None) -> Figure:
    """Plot order book imbalance over time.

    Parameters
    ----------
    features_df : pd.DataFrame
        Orderbook features with imbalance columns.
    market_id : str | None
        Filter by market ID if provided.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if market_id:
        df = features_df[features_df["market_id"] == market_id].copy()
    else:
        df = features_df.copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(12, 6))

    # Try imbalance_1 first, fall back to imbalance_5 if needed
    imbalance_col = "imbalance_1" if "imbalance_1" in df.columns else "imbalance_5"
    if imbalance_col not in df.columns:
        ax.text(0.5, 0.5, "No imbalance data available", ha="center", va="center")
        return fig

    # Convert None values to NaN for plotting
    df = df.copy()
    df[imbalance_col] = pd.to_numeric(df[imbalance_col], errors="coerce")

    # Remove inf/nan values
    df[imbalance_col] = df[imbalance_col].replace([np.inf, -np.inf], np.nan)

    # Filter out rows where imbalance is NaN
    valid_data = df[df[imbalance_col].notna()].copy()

    if valid_data.empty:
        logger.warning("No valid imbalance data after filtering NaN values")
        ax.text(0.5, 0.5, "No valid imbalance data", ha="center", va="center")
        return fig

    logger.info(
        "Plotting %d valid imbalance points (min=%.4f, max=%.4f)",
        len(valid_data),
        valid_data[imbalance_col].min(),
        valid_data[imbalance_col].max(),
    )

    # Plot imbalance with color-coded areas
    imb_values = valid_data[imbalance_col].to_numpy()
    timestamps = valid_data.index

    # Create separate fills for positive and negative using the full dataset
    # This ensures proper visualization without interpolation artifacts
    positive_vals = np.where(imb_values >= 0, imb_values, 0)  # type: ignore[arg-type]
    negative_vals = np.where(imb_values < 0, imb_values, 0)  # type: ignore[arg-type]

    # Plot positive (bid pressure) - only show in legend if there are positive values
    if positive_vals.max() > 0:
        ax.fill_between(
            timestamps,
            0,
            positive_vals,
            color="green",
            alpha=0.4,
            label="Bid Pressure",
            interpolate=True,
        )

    # Plot negative (ask pressure) - only show in legend if there are negative values
    if negative_vals.min() < 0:
        ax.fill_between(
            timestamps,
            0,
            negative_vals,
            color="red",
            alpha=0.4,
            label="Ask Pressure",
            interpolate=True,
        )

    # Also plot line for clarity
    ax.plot(timestamps, imb_values, color="blue", linewidth=0.5, alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Imbalance")
    ax.set_xlabel("Time")
    ax.set_title(f"Order Book Imbalance ({imbalance_col})")
    ax.legend()
    ax.grid(visible=True, alpha=0.3)

    # Format x-axis for datetime
    _format_datetime_axis(ax)

    plt.tight_layout()

    return fig


def plot_trade_flow(trade_features_df: pd.DataFrame) -> Figure:
    """Plot trade volume and imbalance over time.

    Parameters
    ----------
    trade_features_df : pd.DataFrame
        Trade features dataframe.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if trade_features_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No trade data available", ha="center", va="center")
        return fig

    df = trade_features_df.copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Buy vs sell volume
    axes[0].bar(
        df.index, df["buy_volume"], width=0.001, color="green", alpha=0.6, label="Buy Volume"
    )
    axes[0].bar(
        df.index, -df["sell_volume"], width=0.001, color="red", alpha=0.6, label="Sell Volume"
    )
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Volume")
    axes[0].set_title("Trade Volume (Buy vs Sell)")
    axes[0].legend()
    axes[0].grid(visible=True, alpha=0.3)

    # Trade imbalance
    axes[1].plot(df.index, df["trade_imbalance"], color="purple", linewidth=1.5)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Imbalance")
    axes[1].set_title("Trade Imbalance (Buy - Sell)")
    axes[1].grid(visible=True, alpha=0.3)

    # Trade count
    axes[2].plot(df.index, df["trade_count"], color="orange", linewidth=1.5, marker="o")
    axes[2].set_ylabel("Count")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Trade Count")
    axes[2].grid(visible=True, alpha=0.3)

    # Format x-axis for datetime (applies to all subplots due to sharex=True)
    _format_datetime_axis(axes[2])

    plt.tight_layout()
    return fig


def _extract_price_from_data(data: dict) -> float | None:
    """Extract price value from data dict.

    Parameters
    ----------
    data : dict
        Data dictionary potentially containing price information.

    Returns
    -------
    float | None
        Extracted price value or None if not found.
    """
    if not isinstance(data, dict):
        return None
    if "price" in data:
        try:
            return float(data["price"])
        except (ValueError, TypeError):
            return None
    if "value" in data:
        try:
            return float(data["value"])
        except (ValueError, TypeError):
            return None
    return None


def _parse_crypto_prices(crypto_prices_df: pd.DataFrame) -> pd.DataFrame:
    """Parse crypto prices from dataframe with data column.

    Handles nested dict structures with price information.
    """
    crypto_data = []
    for idx, row in crypto_prices_df.iterrows():
        try:
            data = row.get("data") if isinstance(row, dict) else row["data"]

            # Handle case where data is already stored as dict
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    continue

            if data is None:
                continue

            price = _extract_price_from_data(data)
            if price is not None and price != 0 and np.isfinite(price):
                crypto_data.append({"timestamp": idx, "price": float(price)})
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            logger.debug("Failed to parse crypto price: %s", e)
            continue

    if not crypto_data:
        logger.warning("No valid crypto prices extracted from %d rows", len(crypto_prices_df))
        return pd.DataFrame()

    df = pd.DataFrame(crypto_data).set_index("timestamp")
    logger.info(
        "Parsed %d crypto prices (range: %.2f - %.2f)",
        len(df),
        df["price"].min(),
        df["price"].max(),
    )
    return df


def _validate_crypto_correlation_data(
    crypto_prices_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
) -> tuple[bool, pd.DataFrame | None]:
    """Validate and prepare data for correlation plot.

    Parameters
    ----------
    crypto_prices_df : pd.DataFrame
        Crypto price dataframe.
    market_features_df : pd.DataFrame
        Market features dataframe.

    Returns
    -------
    tuple[bool, pd.DataFrame | None]
        (is_valid, crypto_df) where crypto_df is parsed prices or None if invalid.
    """
    if crypto_prices_df.empty or market_features_df.empty:
        return False, None
    crypto_df = _parse_crypto_prices(crypto_prices_df)
    if crypto_df.empty:
        return False, None
    if "mid_price" not in market_features_df.columns:
        return False, None
    return True, crypto_df


def _prepare_correlation_data(
    crypto_df: pd.DataFrame,
    market_features_df: pd.DataFrame,
    window: str,
) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    """Resample and align dataframes for correlation analysis.

    Parameters
    ----------
    crypto_df : pd.DataFrame
        Parsed crypto prices.
    market_features_df : pd.DataFrame
        Market features dataframe.
    window : str
        Resampling window.

    Returns
    -------
    tuple[bool, pd.DataFrame, pd.DataFrame]
        (success, crypto_aligned, market_aligned) or (False, empty, empty) if failed.
    """
    if crypto_df.empty or market_features_df.empty:
        return False, pd.DataFrame(), pd.DataFrame()

    try:
        crypto_resampled = crypto_df.resample(window).last().ffill()
        market_numeric_cols = get_numeric_columns(market_features_df)

        if not market_numeric_cols:
            logger.warning("No numeric columns found in market features")
            return False, pd.DataFrame(), pd.DataFrame()

        market_numeric = market_features_df[market_numeric_cols]
        # Use mid_price if available, otherwise use mean of all numeric columns
        if "mid_price" in market_numeric.columns:
            market_resampled = market_numeric[["mid_price"]].resample(window).mean()
        else:
            market_resampled = market_numeric.resample(window).mean()

        # Remove timezone info if present for proper alignment
        if (
            isinstance(crypto_resampled.index, pd.DatetimeIndex)
            and crypto_resampled.index.tz is not None
        ):
            crypto_resampled.index = crypto_resampled.index.tz_localize(None)
        if (
            isinstance(market_resampled.index, pd.DatetimeIndex)
            and market_resampled.index.tz is not None
        ):
            market_resampled.index = market_resampled.index.tz_localize(None)

        # Find overlapping time range
        common_idx = crypto_resampled.index.intersection(market_resampled.index)  # type: ignore[arg-type]
        if len(common_idx) == 0:
            logger.warning(
                "No overlapping timestamps. Crypto: %s-%s, Market: %s-%s",
                crypto_resampled.index.min(),
                crypto_resampled.index.max(),
                market_resampled.index.min(),
                market_resampled.index.max(),
            )
            return False, pd.DataFrame(), pd.DataFrame()

        crypto_aligned = crypto_resampled.loc[common_idx]
        market_aligned = market_resampled.loc[common_idx]

        logger.info("Aligned %d common timestamps for correlation", len(common_idx))
    except Exception:
        logger.exception("Failed to prepare correlation data")
        return False, pd.DataFrame(), pd.DataFrame()
    else:
        return True, crypto_aligned, market_aligned


def _create_empty_figure(message: str) -> Figure:
    """Create empty figure with centered text message.

    Parameters
    ----------
    message : str
        Message to display.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_crypto_correlation(
    crypto_prices_df: pd.DataFrame, market_features_df: pd.DataFrame, window: str = "5min"
) -> Figure:
    """Plot crypto price correlation with market features.

    Parameters
    ----------
    crypto_prices_df : pd.DataFrame
        Crypto price dataframe.
    market_features_df : pd.DataFrame
        Market features dataframe.
    window : str
        Resampling window.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    # Validate data
    is_valid, crypto_df = _validate_crypto_correlation_data(
        crypto_prices_df, market_features_df
    )
    if not is_valid or crypto_df is None:
        return _create_empty_figure("Insufficient data for correlation")

    # Prepare aligned data
    success, crypto_aligned, market_aligned = _prepare_correlation_data(
        crypto_df, market_features_df, window
    )
    if not success:
        return _create_empty_figure("No overlapping timestamps")

    # Create figure with flexible subplots
    has_enough_data = len(market_aligned) > ROLLING_CORRELATION_WINDOW + 1
    n_subplots = 2 if has_enough_data else 1

    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 6 * n_subplots))
    if n_subplots == 1:
        axes = [axes]  # Make it iterable

    # Plot crypto price and market mid price on same axis
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    (line1,) = ax1.plot(
        crypto_aligned.index, crypto_aligned["price"], color="blue", linewidth=2
    )
    ax1.set_ylabel("Crypto Price", color="blue", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.set_title("Crypto Price vs Market Mid Price", fontsize=12)
    ax1.grid(visible=True, alpha=0.3)

    (line2,) = ax1_twin.plot(
        market_aligned.index, market_aligned["mid_price"], color="green", linewidth=2
    )
    ax1_twin.set_ylabel("Market Mid Price", color="green", fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor="green")

    # Add legend
    lines = [line1, line2]
    labels = ["Crypto Price", "Market Mid Price"]
    ax1.legend(lines, labels, loc="upper left")

    # Plot correlation over rolling window if enough data
    if has_enough_data:
        ax2 = axes[1]
        rolling_corr = (
            crypto_aligned["price"]
            .rolling(
                window=ROLLING_CORRELATION_WINDOW,
                min_periods=ROLLING_CORRELATION_MIN_PERIODS,
            )
            .corr(market_aligned["mid_price"])
        )
        ax2.plot(rolling_corr.index, rolling_corr, color="purple", linewidth=1.5)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.set_ylabel("Correlation", fontsize=11)
        ax2.set_xlabel("Time", fontsize=11)
        ax2.set_title(
            f"Rolling Correlation ({ROLLING_CORRELATION_WINDOW}-period window)",
            fontsize=12,
        )
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(visible=True, alpha=0.3)
        _format_datetime_axis(ax2)
    else:
        # Format single plot's x-axis
        _format_datetime_axis(ax1)

    plt.tight_layout()
    return fig


def plot_market_timeline(
    market_events_df: pd.DataFrame,
    market_id: str | None = None,
    *,
    log_scale: bool = False,
) -> Figure:
    """Plot timeline of market events (trades, resolutions, etc.).

    Parameters
    ----------
    market_events_df : pd.DataFrame
        Market events dataframe.
    market_id : str | None
        Filter by market ID if provided.
    log_scale : bool
        If True, render event counts on a log-scaled y-axis.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if market_id:
        df = market_events_df[market_events_df["market_id"] == market_id].copy()
    else:
        df = market_events_df.copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No market events available", ha="center", va="center")
        return fig

    # Count events by type
    event_counts = df.groupby(["event_type"]).size().reset_index(name="count")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Event type distribution
    axes[0].bar(event_counts["event_type"], event_counts["count"], color="steelblue", alpha=0.7)
    axes[0].set_ylabel("Count (log scale)" if log_scale else "Count")
    axes[0].set_title("Event Type Distribution")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(visible=True, alpha=0.3, axis="y")
    if log_scale:
        axes[0].set_yscale("log")

    # Events over time - group by 1min bins and event type
    try:
        event_df = df.reset_index()  # Convert index to column
        event_df["time_bin"] = pd.to_datetime(event_df["ts_event"]).dt.floor("1min")

        event_timeline = (
            event_df
            .groupby(["time_bin", "event_type"])
            .size()
            .reset_index(name="count")
            .pivot_table(index="time_bin", columns="event_type", values="count", fill_value=0)
        )

        if not event_timeline.empty and len(event_timeline) > 0:
            timeline_to_plot = event_timeline
            if log_scale:
                timeline_to_plot = event_timeline.replace(0, np.nan)

            timeline_to_plot.plot(ax=axes[1], marker="o", linewidth=1.5, alpha=0.7)
            ylabel = (
                "Events per Minute (log scale)"
                if log_scale
                else "Events per Minute"
            )
            axes[1].set_ylabel(ylabel)
            axes[1].set_xlabel("Time")
            axes[1].set_title("Event Timeline")
            axes[1].legend(title="Event Type", bbox_to_anchor=(1.05, 1), loc="upper left")
            axes[1].grid(visible=True, alpha=0.3)
            if log_scale:
                axes[1].set_yscale("log")

            # Format ticks without setting explicit formatter to avoid date conversion errors
            ax = axes[1]
            locs = ax.get_xticks()
            if len(locs) > 0 and hasattr(ax.xaxis, "set_major_locator"):
                # Just rotate labels without triggering date formatter
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    except Exception:
        # Fallback if timeline binning fails
        axes[1].text(0.5, 0.5, "Could not generate event timeline",
                    ha="center", va="center")

    plt.tight_layout()
    return fig


def _save_visualization(fig: Figure, filename: str, output_dir: Path, label: str) -> None:
    """Save a visualization to file and close the figure.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    filename : str
        Output filename.
    output_dir : Path
        Output directory.
    label : str
        Label to print.
    """
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {label}")


def _generate_report_visualizations(
    data: ReportData,
    output_dir: Path,
    market_id: str | None = None,
) -> None:
    """Generate and save all visualization plots.

    Parameters
    ----------
    data : ReportData
        Container with orderbook_features, trade_features, crypto_prices, and market_events.
    output_dir : Path
        Output directory.
    market_id : str | None
        Filter by market ID if provided.
    """
    if not data.orderbook_features.empty:
        _save_visualization(plot_spread_timeseries(data.orderbook_features, market_id),
                           "spread_timeseries.png", output_dir, "Spread timeseries")
        _save_visualization(plot_orderbook_depth(data.orderbook_features, market_id),
                           "orderbook_depth.png", output_dir, "Orderbook depth")
        _save_visualization(plot_imbalance(data.orderbook_features, market_id),
                           "imbalance.png", output_dir, "Order imbalance")

    if not data.trade_features.empty:
        _save_visualization(plot_trade_flow(data.trade_features),
                           "trade_flow.png", output_dir, "Trade flow")

    if not data.crypto_prices.empty and not data.orderbook_features.empty:
        _save_visualization(plot_crypto_correlation(data.crypto_prices, data.orderbook_features),
                           "crypto_correlation.png", output_dir, "Crypto correlation")

    _save_visualization(plot_market_timeline(data.market_events, market_id),
                       "market_timeline.png", output_dir, "Market timeline")


def create_diagnostic_report(
    runner: BacktestRunner,
    start: datetime,
    end: datetime,
    output_dir: Path | str,
    market_id: str | None = None,
) -> None:
    """Generate a complete diagnostic report with all visualizations.

    Parameters
    ----------
    runner : BacktestRunner
        BacktestRunner instance with loaded data.
    start : datetime
        Start time.
    end : datetime
        End time.
    output_dir : Path | str
        Directory to save plots.
    market_id : str | None
        Filter by market ID if provided.
    """
    output_dir = ensure_directory(output_dir)

    print(f"\nGenerating diagnostic report from {start} to {end}...")
    print(f"Output directory: {output_dir}")

    # Load data
    market_events = runner.load_market_events(start, end)
    crypto_prices = runner.load_crypto_prices(start, end)

    if market_events.empty:
        print("No market events found. Cannot generate report.")
        return

    # Compute features
    print("\nComputing features...")
    orderbook_features = runner.compute_orderbook_features_df(market_events)
    trade_features = runner.compute_trade_features_df(market_events, window="5min")

    # Generate plots
    print("\nGenerating visualizations...")
    report_data = ReportData(orderbook_features, trade_features, crypto_prices, market_events)
    _generate_report_visualizations(report_data, output_dir, market_id)

    # Create summary statistics
    print("\nGenerating summary statistics...")
    summary = {
        "total_events": len(market_events),
        "event_types": market_events["event_type"].value_counts().to_dict(),
        "unique_markets": market_events["market_id"].nunique(),
        "time_range": f"{market_events.index.min()} to {market_events.index.max()}",
    }

    if not orderbook_features.empty:
        summary["avg_spread"] = float(orderbook_features["spread"].mean())
        summary["avg_mid_price"] = float(orderbook_features["mid_price"].mean())

    if not trade_features.empty:
        summary["total_trades"] = int(trade_features["trade_count"].sum())

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  ✓ Summary saved to {summary_path}")

    print(f"\n✅ Diagnostic report complete! Check {output_dir}/ for results.")


def plot_feature_correlations(features_df: pd.DataFrame) -> Figure:
    """Plot correlation heatmap of all features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Features dataframe.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if features_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "No features available", ha="center", va="center")
        return fig

    # Select numeric columns
    numeric_cols = get_numeric_columns(features_df)
    corr_matrix = features_df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()

    return fig
