from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from EDA.core.utils import ensure_directories, format_number, get_logger, ordered_resolutions

if TYPE_CHECKING:
    from pathlib import Path

    from EDA.core.config import RunConfig

mpl.use("Agg")
sns.set_theme(style="white", context="talk")

HEATMAP_TEXT_SWITCH = 0.55
DIVERGING_TEXT_SWITCH = 0.35
MISSING_CELL_COLOR = "#E9EDF2"
BAR_COLOR = "#1F4E79"
NEGATIVE_BAR_COLOR = "#B24C63"
MISSING_BAR_COLOR = "#BFC6CF"


def generate_figures(config: RunConfig, outputs: dict[str, pl.DataFrame]) -> list[Path]:
    logger = get_logger()
    figure_dir = config.paths.mode_figure_dir(config.mode)
    ensure_directories([figure_dir])
    figure_paths: list[Path] = []

    coverage = outputs["coverage_table"]
    micro = outputs["microstructure_summary"]
    returns = outputs["return_summary"]
    contract_book = outputs["contract_book_summary"]
    autocorr = outputs["autocorr_long"]
    daily_panel = outputs["daily_stability_panel"]
    rankings = outputs["ranking_table"]

    coins = coverage["coin"].unique().sort().to_list()
    resolutions = ordered_resolutions(
        coverage["resolution"].unique().to_list(),
        config.resolution_order,
    )
    coin_resolution_grid = build_coin_resolution_grid(coins, resolutions)

    figure_paths.append(
        plot_heatmap(
            figure_dir / "coverage_coin_resolution.png",
            frame=coverage,
            coin_resolution_grid=coin_resolution_grid,
            coins=coins,
            resolutions=resolutions,
            value_column="unique_markets",
            title="Unique Markets by Coin and Resolution",
            colorbar_label="Unique markets",
        ),
    )
    figure_paths.append(
        plot_bar_counts(
            figure_dir / "observation_counts_by_coin.png",
            coverage.group_by("coin").agg(pl.col("snapshot_count").sum().alias("snapshot_count")),
            category_column="coin",
            value_column="snapshot_count",
            title="Snapshot Observation Count by Coin",
            ylabel="Snapshots",
            category_order=coins,
            descending=True,
        ),
    )
    figure_paths.append(
        plot_bar_counts(
            figure_dir / "observation_counts_by_resolution.png",
            coverage.group_by("resolution").agg(
                pl.col("snapshot_count").sum().alias("snapshot_count"),
            ),
            category_column="resolution",
            value_column="snapshot_count",
            title="Snapshot Observation Count by Resolution",
            ylabel="Snapshots",
            category_order=resolutions,
        ),
    )
    figure_paths.append(
        plot_heatmap(
            figure_dir / "spread_by_coin_resolution.png",
            frame=micro,
            coin_resolution_grid=coin_resolution_grid,
            coins=coins,
            resolutions=resolutions,
            value_column="relative_spread_median",
            title="Median Relative Spread by Coin and Resolution",
            colorbar_label="Relative spread",
            value_digits=4,
        ),
    )
    figure_paths.append(
        plot_heatmap(
            figure_dir / "depth_by_coin_resolution.png",
            frame=micro,
            coin_resolution_grid=coin_resolution_grid,
            coins=coins,
            resolutions=resolutions,
            value_column="total_visible_depth_median",
            title="Median Visible Depth by Coin and Resolution",
            colorbar_label="Visible depth",
        ),
    )
    figure_paths.append(
        plot_imbalance_distributions(
            figure_dir / "imbalance_distributions_by_coin.png",
            contract_book=contract_book,
            coins=coins,
        ),
    )
    figure_paths.append(
        plot_heatmap(
            figure_dir / "volatility_by_coin_resolution.png",
            frame=returns,
            coin_resolution_grid=coin_resolution_grid,
            coins=coins,
            resolutions=resolutions,
            value_column="realized_volatility_median",
            title="Median Realized Volatility by Coin and Resolution",
            colorbar_label="Realized volatility",
            value_digits=4,
        ),
    )
    figure_paths.append(
        plot_autocorr_heatmaps(
            figure_dir / "return_autocorrelation_heatmap.png",
            autocorr=autocorr,
            coins=coins,
            resolutions=resolutions,
            lags=config.acf_lags,
        ),
    )
    figure_paths.append(
        plot_heatmap(
            figure_dir / "tail_risk_by_coin_resolution.png",
            frame=returns,
            coin_resolution_grid=coin_resolution_grid,
            coins=coins,
            resolutions=resolutions,
            value_column="jump_rate_large_median",
            title="Tail-Risk Proxy by Coin and Resolution",
            colorbar_label="Jump rate (>= 5c)",
            value_digits=4,
        ),
    )
    figure_paths.append(
        plot_time_stability(
            figure_dir / "time_stability_major_features.png",
            daily_panel=daily_panel,
            coins=coins,
            resolutions=resolutions,
        ),
    )
    figure_paths.append(
        plot_rankings(
            figure_dir / "research_rankings.png",
            coin_resolution_grid=coin_resolution_grid,
            rankings=rankings,
        ),
    )

    logger.info("Generated %s figures under %s", len(figure_paths), figure_dir)
    return figure_paths


def build_coin_resolution_grid(coins: list[str], resolutions: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"coin": coin, "resolution": resolution}
            for coin in coins
            for resolution in resolutions
        ],
    )


def plot_heatmap(
    path: Path,
    *,
    frame: pl.DataFrame,
    coin_resolution_grid: pl.DataFrame,
    coins: list[str],
    resolutions: list[str],
    value_column: str,
    title: str,
    colorbar_label: str,
    value_digits: int = 2,
) -> Path:
    panel = coin_resolution_grid.join(
        frame.select(["coin", "resolution", value_column]),
        on=["coin", "resolution"],
        how="left",
    )
    matrix = build_coin_resolution_matrix(panel, value_column, coins, resolutions)

    fig, ax = plt.subplots(figsize=(1.9 * len(resolutions) + 4, 0.9 * len(coins) + 3))
    image = render_heatmap_axis(
        ax=ax,
        matrix=matrix,
        x_labels=resolutions,
        y_labels=coins,
        title=title,
        cmap_name="YlGnBu",
        value_digits=value_digits,
    )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.03)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bar_counts(
    path: Path,
    frame: pl.DataFrame,
    *,
    category_column: str,
    value_column: str,
    title: str,
    ylabel: str,
    category_order: list[str] | None = None,
    descending: bool = False,
) -> Path:
    if category_order is not None:
        ordered = (
            pl.DataFrame({category_column: category_order})
            .join(frame.select([category_column, value_column]), on=category_column, how="left")
            .with_columns(pl.col(value_column).fill_null(0))
        )
        if descending:
            ordered = ordered.sort(value_column, descending=True)
    else:
        ordered = frame.sort(value_column, descending=descending)

    categories = ordered[category_column].to_list()
    values = ordered[value_column].to_list()
    ymax = max(float(max(values, default=0.0)), 1.0)

    fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.5), 5.2))
    bars = ax.bar(categories, values, color=BAR_COLOR, width=0.7)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax * 1.12)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            format_number(value, digits=0),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_imbalance_distributions(
    path: Path,
    *,
    contract_book: pl.DataFrame,
    coins: list[str],
) -> Path:
    figure_data = contract_book.select(["coin", "order_book_imbalance_mean"]).drop_nulls()
    positions = np.arange(1, len(coins) + 1)
    distributions: list[np.ndarray] = []
    distribution_positions: list[int] = []
    missing_positions: list[int] = []

    for position, coin in zip(positions, coins, strict=True):
        values = (
            figure_data.filter(pl.col("coin") == coin)["order_book_imbalance_mean"]
            .cast(pl.Float64)
            .to_numpy()
        )
        if values.size:
            distributions.append(values)
            distribution_positions.append(int(position))
        else:
            missing_positions.append(int(position))

    fig, ax = plt.subplots(figsize=(11, 6.4))
    if distributions:
        boxplot = ax.boxplot(
            distributions,
            positions=distribution_positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
        )
        for box in boxplot["boxes"]:
            box.set_facecolor("#7BB6D9")
            box.set_edgecolor(BAR_COLOR)
            box.set_alpha(0.9)
        for median in boxplot["medians"]:
            median.set_color("#17324D")
            median.set_linewidth(1.5)

    ax.axhline(0.0, color="#69707A", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(coins)
    ax.set_title("Order-Book Imbalance Distribution by Coin", fontsize=16, pad=12)
    ax.set_xlabel("Coin")
    ax.set_ylabel("Average imbalance")
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for position in missing_positions:
        ax.text(
            position,
            0.03,
            "No data",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#6C757D",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_autocorr_heatmaps(
    path: Path,
    *,
    autocorr: pl.DataFrame,
    coins: list[str],
    resolutions: list[str],
    lags: tuple[int, ...],
) -> Path:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(resolutions),
        figsize=(4.7 * len(resolutions), 0.85 * len(coins) + 4.2),
        squeeze=False,
    )

    last_image = None
    for axis, resolution in zip(axes.ravel(), resolutions, strict=True):
        subset = autocorr.filter(pl.col("resolution") == resolution)
        matrix = build_autocorr_matrix(subset, coins, lags)
        last_image = render_heatmap_axis(
            ax=axis,
            matrix=matrix,
            x_labels=[str(lag) for lag in lags],
            y_labels=coins,
            title=f"{resolution} Hourly-Return ACF",
            cmap_name="RdBu_r",
            value_digits=3,
            center=0.0,
        )
        axis.set_xlabel("Lag")
        axis.set_ylabel("Coin")

    if last_image is not None:
        colorbar_axis = figure.add_axes([0.905, 0.18, 0.018, 0.64])
        colorbar = figure.colorbar(last_image, cax=colorbar_axis)
        colorbar.set_label("Autocorrelation")

    figure.suptitle(
        "Return Autocorrelation by Lag, Coin, and Resolution",
        fontsize=16,
        y=1.02,
    )
    figure.subplots_adjust(top=0.82, right=0.88, wspace=0.32)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_time_stability(
    path: Path,
    *,
    daily_panel: pl.DataFrame,
    coins: list[str],
    resolutions: list[str],
) -> Path:
    metrics = [
        ("spread_mean", "Average Spread"),
        ("total_visible_depth_mean", "Average Visible Depth"),
        ("realized_volatility", "Average Realized Volatility"),
    ]
    palette_values = sns.color_palette("tab10", n_colors=max(len(coins), 3))
    palette = {coin: palette_values[index] for index, coin in enumerate(coins)}

    fig, axes = plt.subplots(
        nrows=len(metrics),
        ncols=len(resolutions),
        figsize=(5.0 * len(resolutions), 3.7 * len(metrics) + 1.5),
        squeeze=False,
        sharex="col",
    )

    for row_index, (metric, label) in enumerate(metrics):
        for col_index, resolution in enumerate(resolutions):
            axis = axes[row_index, col_index]
            resolution_panel = daily_panel.filter(pl.col("resolution") == resolution).sort(
                ["observation_day", "coin"],
            )
            missing_coins: list[str] = []

            for coin in coins:
                series = resolution_panel.filter(pl.col("coin") == coin).sort("observation_day")
                if series.height == 0:
                    missing_coins.append(coin)
                    continue
                axis.plot(
                    series["observation_day"].to_list(),
                    series[metric].to_list(),
                    color=palette[coin],
                    linewidth=1.6,
                    alpha=0.9,
                )

            if missing_coins:
                axis.text(
                    0.02,
                    0.98,
                    f"Missing: {', '.join(missing_coins)}",
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.5,
                    color="#6C757D",
                )

            axis.set_title(f"{resolution} | {label}", fontsize=12, pad=8)
            if col_index == 0:
                axis.set_ylabel(label)
            if row_index == len(metrics) - 1:
                axis.set_xlabel("Observation day")
            _format_time_axis(axis, show_labels=row_index == len(metrics) - 1)
            axis.grid(axis="y", alpha=0.22)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

    legend_handles = [
        Line2D([0], [0], color=palette[coin], lw=2, label=coin)
        for coin in coins
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=min(4, len(coins)),
        frameon=False,
        fontsize=10,
    )
    fig.suptitle(
        "Time Stability of Major Features Across Coins and Resolutions",
        fontsize=16,
        y=1.08,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _format_time_axis(axis: plt.Axes, *, show_labels: bool) -> None:
    if not axis.has_data():
        if not show_labels:
            axis.tick_params(axis="x", labelbottom=False)
        return

    x_min, x_max = axis.get_xlim()
    span_days = max(float(x_max - x_min), 1.0)
    locator = mdates.DayLocator(interval=max(1, int(np.ceil(span_days / 4.0))))
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    axis.margins(x=0.04)
    if show_labels:
        axis.tick_params(axis="x", labelrotation=25, labelsize=9)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
    else:
        axis.tick_params(axis="x", labelbottom=False)


def plot_rankings(
    path: Path,
    *,
    coin_resolution_grid: pl.DataFrame,
    rankings: pl.DataFrame,
) -> Path:
    ranking_panel = (
        coin_resolution_grid.join(
            rankings.select(["coin", "resolution", "research_score"]),
            on=["coin", "resolution"],
            how="left",
        )
        .with_columns(
            pl.concat_str(["coin", pl.lit("-"), "resolution"]).alias("label"),
            pl.col("research_score").is_not_null().alias("has_score"),
        )
        .sort(
            ["has_score", "research_score", "coin", "resolution"],
            descending=[True, True, False, False],
            nulls_last=True,
        )
    )

    labels = ranking_panel["label"].to_list()
    scores = ranking_panel["research_score"].fill_null(0.0).to_list()
    has_score = ranking_panel["has_score"].to_list()

    valid_scores = [score for score, valid in zip(scores, has_score, strict=True) if valid]
    max_abs_score = max((abs(score) for score in valid_scores), default=1.0)
    x_margin = max_abs_score * 0.12 + 0.2

    colors = [
        MISSING_BAR_COLOR if not valid else (BAR_COLOR if score >= 0 else NEGATIVE_BAR_COLOR)
        for score, valid in zip(scores, has_score, strict=True)
    ]

    fig, ax = plt.subplots(figsize=(11, max(7, 0.45 * len(labels) + 2.2)))
    bars = ax.barh(labels, scores, color=colors, alpha=0.92)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#69707A", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-(max_abs_score + x_margin), max_abs_score + x_margin)
    ax.set_title(
        "Research Candidate Ranking Across the Full Coin-Resolution Universe",
        fontsize=16,
        pad=16,
    )
    ax.set_xlabel("Research score")
    ax.grid(axis="x", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0,
        0.985,
        "Gray rows lack enough processed observations in the current run.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#6C757D",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.5},
    )

    for bar, score, valid in zip(bars, scores, has_score, strict=True):
        label = format_number(score, digits=2) if valid else "NA"
        offset = 0.04 * max_abs_score + 0.04
        x_position = score + offset if score >= 0 else score - offset
        ax.text(
            x_position if valid else offset,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left" if (not valid or score >= 0) else "right",
            fontsize=9.5,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_coin_resolution_matrix(
    panel: pl.DataFrame,
    value_column: str,
    coins: list[str],
    resolutions: list[str],
) -> np.ndarray:
    lookup = {
        (row["coin"], row["resolution"]): row[value_column]
        for row in panel.iter_rows(named=True)
    }
    matrix = np.full((len(coins), len(resolutions)), np.nan)
    for row_index, coin in enumerate(coins):
        for col_index, resolution in enumerate(resolutions):
            value = lookup.get((coin, resolution))
            matrix[row_index, col_index] = np.nan if value is None else float(value)
    return matrix


def build_autocorr_matrix(
    autocorr: pl.DataFrame,
    coins: list[str],
    lags: tuple[int, ...],
) -> np.ndarray:
    lookup = {
        (row["coin"], int(row["lag"])): row["autocorrelation"]
        for row in autocorr.iter_rows(named=True)
    }
    matrix = np.full((len(coins), len(lags)), np.nan)
    for row_index, coin in enumerate(coins):
        for col_index, lag in enumerate(lags):
            value = lookup.get((coin, lag))
            matrix[row_index, col_index] = np.nan if value is None else float(value)
    return matrix


def render_heatmap_axis(
    *,
    ax: plt.Axes,
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    cmap_name: str,
    value_digits: int,
    center: float | None = None,
) -> mpl.image.AxesImage:
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad(MISSING_CELL_COLOR)
    valid = matrix[~np.isnan(matrix)]

    if valid.size:
        if center is None:
            vmin = float(np.nanmin(matrix))
            vmax = float(np.nanmax(matrix))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0
            norm: mcolors.Normalize = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            max_distance = max(
                abs(float(np.nanmin(matrix)) - center),
                abs(float(np.nanmax(matrix)) - center),
            )
            if np.isclose(max_distance, 0.0):
                max_distance = 1.0
            norm = mcolors.TwoSlopeNorm(
                vmin=center - max_distance,
                vcenter=center,
                vmax=center + max_distance,
            )
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        aspect="auto",
        norm=norm,
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_facecolor(MISSING_CELL_COLOR)
    ax.grid(visible=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_index, _y_label in enumerate(y_labels):
        for col_index, _x_label in enumerate(x_labels):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                annotation = "NA"
                color = "#26313C"
            else:
                annotation = format_number(float(value), digits=value_digits)
                color = heatmap_text_color(value, norm, diverging=center is not None)
            ax.text(
                col_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                color=color,
                fontsize=10.5,
            )
    return image


def heatmap_text_color(
    value: float,
    norm: mcolors.Normalize,
    *,
    diverging: bool,
) -> str:
    normalized_value = float(norm(value))
    if diverging:
        contrast = abs(normalized_value - 0.5) * 2.0
        return "white" if contrast > DIVERGING_TEXT_SWITCH else "black"
    return "white" if normalized_value > HEATMAP_TEXT_SWITCH else "black"
