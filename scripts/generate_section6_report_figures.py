"""Generate Section 6 report figures from diagnostics artifacts.

This script reads existing CSV artifacts and produces publication-ready PNG charts
for Section 6 (Relative Order Book Strength) under reports/figures/section6/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Section 6 report figures")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=PROJECT_ROOT / "reports" / "artifacts",
        help="Root directory containing diagnostics artifact CSV files",
    )
    parser.add_argument(
        "--cumdiag-dir",
        type=Path,
        default=None,
        help=(
            "Optional cumulative diagnostics run directory. If omitted, uses the latest "
            "reports/artifacts/cumulative_signal_diagnostics/cumdiag_* directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "figures" / "section6",
        help="Output directory for Section 6 figures",
    )
    parser.add_argument(
        "--hpo-grid-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit HPO grid CSV. Defaults to "
            "reports/artifacts/gate_grid_search_first3000_grid010_ddaware_compact.csv"
        ),
    )
    return parser.parse_args()


def resolve_cumdiag_dir(artifacts_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            msg = f"Cumulative diagnostics dir not found: {explicit}"
            raise FileNotFoundError(msg)
        return explicit

    base = artifacts_root / "cumulative_signal_diagnostics"
    if not base.exists():
        msg = f"Cumulative diagnostics root not found: {base}"
        raise FileNotFoundError(msg)

    candidates = sorted([p for p in base.glob("cumdiag_*") if p.is_dir()])
    if not candidates:
        msg = f"No cumdiag_* directories under: {base}"
        raise FileNotFoundError(msg)
    return candidates[-1]


def read_csv_or_none(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[skip] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[skip] Failed reading {path}: {exc}")
        return None


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {output_path}")


def apply_figure_style(fig: plt.Figure) -> None:
    fig.patch.set_facecolor("white")


def apply_axes_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")


def plot_signal_design_concept(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    apply_figure_style(fig)
    apply_axes_style(ax)
    ax.axis("off")

    boxes = [
        (0.02, 0.30, 0.17, 0.42, "Order Book\nEvents"),
        (0.22, 0.30, 0.17, 0.42, "Relative\nFeatures"),
        (0.42, 0.30, 0.17, 0.42, "Snapshot\nScore"),
        (0.62, 0.30, 0.17, 0.42, "Cumulative\nMemory"),
        (0.82, 0.30, 0.16, 0.42, "Cross-Side\nRanking"),
    ]

    for x, y, w, h, label in boxes:
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            edgecolor="#1f77b4",
            facecolor="#eaf2fb",
            linewidth=2.0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=12,
            transform=ax.transAxes,
        )

    arrow_y = 0.51
    arrow_pairs = [(0.19, 0.22), (0.39, 0.42), (0.59, 0.62), (0.79, 0.82)]
    for left, right in arrow_pairs:
        ax.annotate(
            "",
            xy=(right, arrow_y),
            xytext=(left, arrow_y),
            xycoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#27496d"},
        )

    ax.set_title("Conceptual Signal Pipeline (Illustrative, Not Empirical)", fontsize=15, pad=12)
    save_figure(fig, out / "s6_signal_pipeline_concept.png")


def plot_cumulative_memory_concept(out: Path) -> None:
    steps = np.arange(1, 31)
    winner_snapshot = 0.13 * np.sin(steps / 1.4) + np.linspace(0.00, 0.18, len(steps))
    loser_snapshot = 0.13 * np.sin(steps / 1.4 + 1.8) - np.linspace(0.00, 0.18, len(steps))

    winner_cum = np.cumsum(winner_snapshot)
    loser_cum = np.cumsum(loser_snapshot)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    apply_figure_style(fig)
    for axis in axes:
        apply_axes_style(axis)

    axes[0].plot(
        steps,
        winner_snapshot,
        color="#2ca02c",
        linestyle="--",
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Winner-side snapshot (schematic)",
    )
    axes[0].plot(
        steps,
        loser_snapshot,
        color="#d62728",
        linestyle="--",
        marker="o",
        markersize=3,
        linewidth=1.2,
        label="Loser-side snapshot (schematic)",
    )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Noisy Snapshot Signal (Schematic)")
    axes[0].set_xlabel("Event progression")
    axes[0].set_ylabel("Relative signal (unitless)")
    axes[0].set_xticks([1, 10, 20, 30])
    axes[0].set_xticklabels(["Start", "", "", "End"])
    axes[0].set_yticks([])
    axes[0].legend(fontsize=8)

    axes[1].plot(
        steps,
        winner_cum,
        color="#2ca02c",
        linewidth=2.0,
        label="Winner cumulative (schematic)",
    )
    axes[1].plot(
        steps,
        loser_cum,
        color="#d62728",
        linewidth=2.0,
        label="Loser cumulative (schematic)",
    )
    axes[1].set_title("Cumulative Memory Creates Separation (Schematic)")
    axes[1].set_xlabel("Event progression")
    axes[1].set_ylabel("Accumulated signal (unitless)")
    axes[1].set_xticks([1, 10, 20, 30])
    axes[1].set_xticklabels(["Start", "", "", "End"])
    axes[1].set_yticks([])
    axes[1].legend(fontsize=8)

    save_figure(fig, out / "s6_cumulative_memory_concept.png")


class DepthSnapshotSpec(NamedTuple):
    """Configuration for one stylized orderbook depth snapshot panel."""

    title: str
    metric_text: str
    decision_text: str
    pass_case: bool
    bids: list[tuple[float, float]]
    asks: list[tuple[float, float]]


class MiniBookSpec(NamedTuple):
    """Specification for one small UP/DOWN book shown in the filtering storyboard."""

    label: str
    bid_levels: list[float]
    ask_levels: list[float]
    border_color: str


class SignalStageSpec(NamedTuple):
    """Specification for one stage in the UP-vs-DOWN signal filtering process."""

    x_start: float
    title: str
    metric: str
    decision: str
    up_book: MiniBookSpec
    down_book: MiniBookSpec


def _draw_depth_snapshot(ax: plt.Axes, spec: DepthSnapshotSpec) -> None:
    bids_sorted = sorted(spec.bids, key=lambda x: x[0], reverse=True)
    asks_sorted = sorted(spec.asks, key=lambda x: x[0])

    bid_prices = np.array([p for p, _ in bids_sorted], dtype=float)
    bid_qty = np.array([q for _, q in bids_sorted], dtype=float)
    ask_prices = np.array([p for p, _ in asks_sorted], dtype=float)
    ask_qty = np.array([q for _, q in asks_sorted], dtype=float)

    bid_cum = np.cumsum(bid_qty)
    ask_cum = np.cumsum(ask_qty)

    ax.step(bid_cum, bid_prices, where="post", color="#2ca02c", linewidth=2.0, label="Bids")
    ax.step(ask_cum, ask_prices, where="post", color="#d62728", linewidth=2.0, label="Asks")
    ax.fill_betweenx(bid_prices, 0, bid_cum, alpha=0.12, color="#2ca02c")
    ax.fill_betweenx(ask_prices, 0, ask_cum, alpha=0.12, color="#d62728")

    if len(bid_prices) and len(ask_prices):
        ax.axhline(bid_prices[0], color="#2ca02c", linestyle=":", linewidth=1.0)
        ax.axhline(ask_prices[0], color="#d62728", linestyle=":", linewidth=1.0)

    ax.set_title(spec.title, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    status_color = "#2e7d32" if spec.pass_case else "#c62828"
    ax.text(
        0.02,
        0.98,
        spec.metric_text,
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#b0bec5"},
    )
    ax.text(
        0.02,
        0.02,
        spec.decision_text,
        transform=ax.transAxes,
        va="bottom",
        fontsize=8.5,
        color=status_color,
        weight="bold",
    )


def plot_gate_mechanics_execution(out: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    apply_figure_style(fig)
    for axis in axes.ravel():
        apply_axes_style(axis)

    _draw_depth_snapshot(
        axes[0, 0],
        DepthSnapshotSpec(
            title="Spread Gate: PASS",
            metric_text="Motivation: avoid paying excessive entry cost",
            decision_text="PASS: spread appears tight and stable",
            pass_case=True,
            bids=[(0.498, 1.8), (0.497, 1.3), (0.496, 1.0)],
            asks=[(0.502, 1.7), (0.503, 1.2), (0.504, 0.9)],
        ),
    )
    _draw_depth_snapshot(
        axes[0, 1],
        DepthSnapshotSpec(
            title="Spread Gate: BLOCK",
            metric_text="Motivation: same depth, but wider top-book spread",
            decision_text="BLOCK: entry likely overpays",
            pass_case=False,
            bids=[(0.492, 1.8), (0.491, 1.3), (0.490, 1.0)],
            asks=[(0.508, 1.7), (0.509, 1.2), (0.510, 0.9)],
        ),
    )

    _draw_depth_snapshot(
        axes[1, 0],
        DepthSnapshotSpec(
            title="Liquidity Gate: PASS",
            metric_text="Motivation: thick book supports stable execution",
            decision_text="PASS: book can absorb trade size",
            pass_case=True,
            bids=[(0.498, 2.4), (0.497, 1.9), (0.496, 1.4)],
            asks=[(0.502, 2.3), (0.503, 1.8), (0.504, 1.3)],
        ),
    )
    _draw_depth_snapshot(
        axes[1, 1],
        DepthSnapshotSpec(
            title="Liquidity Gate: BLOCK",
            metric_text="Motivation: thin book cannot support intended size",
            decision_text="BLOCK: book is too thin to support entry",
            pass_case=False,
            bids=[(0.498, 0.35), (0.497, 0.24), (0.496, 0.16)],
            asks=[(0.502, 0.33), (0.503, 0.22), (0.504, 0.15)],
        ),
    )

    _draw_depth_snapshot(
        axes[2, 0],
        DepthSnapshotSpec(
            title="Ask-Depth-5 Cap: PASS",
            metric_text="Motivation: avoid heavy overhead supply",
            decision_text="PASS: ask ladder not heavily stacked",
            pass_case=True,
            bids=[(0.498, 1.5), (0.497, 1.1), (0.496, 0.8)],
            asks=[(0.502, 0.9), (0.503, 0.8), (0.504, 0.7), (0.505, 0.6), (0.506, 0.5)],
        ),
    )
    _draw_depth_snapshot(
        axes[2, 1],
        DepthSnapshotSpec(
            title="Ask-Depth-5 Cap: BLOCK",
            metric_text="Motivation: top ask ladder forms a strong wall",
            decision_text="BLOCK: likely adverse local pressure",
            pass_case=False,
            bids=[(0.498, 1.5), (0.497, 1.1), (0.496, 0.8)],
            asks=[(0.502, 2.4), (0.503, 2.2), (0.504, 2.0), (0.505, 1.9), (0.506, 1.8)],
        ),
    )

    # Keep a shared scale so "thick" vs "thin" books are visually comparable across panels.
    for axis in axes.ravel():
        axis.set_xlim(0.0, 11.5)
        axis.set_ylim(0.489, 0.511)

    fig.suptitle("Execution-Quality Gates from Orderbook Depth", fontsize=15)
    save_figure(fig, out / "s6_gate_mechanics_execution.png")


def plot_gate_mechanics_confidence(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    apply_figure_style(fig)
    apply_axes_style(ax)
    ax.axis("off")

    panels = [
        (0.05, "Ambiguous state", "UP and DOWN look similar", "Risk: fragile rank flips"),
        (
            0.37,
            "Persistent separation",
            "One side repeatedly dominates",
            "Signal quality improves",
        ),
        (0.69, "Confidence gates", "Require strength + gap", "Keep robust states only"),
    ]

    for x0, title, line1, line2 in panels:
        panel = plt.Rectangle(
            (x0, 0.20),
            0.26,
            0.66,
            edgecolor="#94a3b8",
            facecolor="#ffffff",
            linewidth=1.4,
            transform=ax.transAxes,
        )
        ax.add_patch(panel)
        ax.text(
            x0 + 0.13, 0.81, title, ha="center", fontsize=11, weight="bold", transform=ax.transAxes
        )

        up_box = plt.Rectangle(
            (x0 + 0.03, 0.55),
            0.09,
            0.18,
            edgecolor="#2563eb",
            facecolor="#eff6ff",
            linewidth=1.2,
            transform=ax.transAxes,
        )
        dn_box = plt.Rectangle(
            (x0 + 0.14, 0.55),
            0.09,
            0.18,
            edgecolor="#ea580c",
            facecolor="#fff7ed",
            linewidth=1.2,
            transform=ax.transAxes,
        )
        ax.add_patch(up_box)
        ax.add_patch(dn_box)
        ax.text(
            x0 + 0.075, 0.64, "UP", ha="center", fontsize=9, weight="bold", transform=ax.transAxes
        )
        ax.text(
            x0 + 0.185,
            0.64,
            "DOWN",
            ha="center",
            fontsize=9,
            weight="bold",
            transform=ax.transAxes,
        )

        if title == "Ambiguous state":
            ax.plot([x0 + 0.04, x0 + 0.11], [0.50, 0.46], color="#64748b", transform=ax.transAxes)
            ax.plot([x0 + 0.15, x0 + 0.22], [0.50, 0.47], color="#64748b", transform=ax.transAxes)
            status = "BLOCK"
            color = "#dc2626"
        elif title == "Persistent separation":
            ax.plot([x0 + 0.04, x0 + 0.11], [0.49, 0.43], color="#2563eb", transform=ax.transAxes)
            ax.plot([x0 + 0.15, x0 + 0.22], [0.50, 0.54], color="#ea580c", transform=ax.transAxes)
            status = "candidate retained"
            color = "#15803d"
        else:
            ax.text(x0 + 0.13, 0.48, "score gate", ha="center", fontsize=9, transform=ax.transAxes)
            ax.text(x0 + 0.13, 0.43, "+", ha="center", fontsize=11, transform=ax.transAxes)
            ax.text(
                x0 + 0.13, 0.38, "score-gap gate", ha="center", fontsize=9, transform=ax.transAxes
            )
            status = "PASS"
            color = "#15803d"

        ax.text(x0 + 0.13, 0.31, line1, ha="center", fontsize=9.3, transform=ax.transAxes)
        ax.text(
            x0 + 0.13,
            0.26,
            line2,
            ha="center",
            fontsize=9.3,
            color="#475569",
            transform=ax.transAxes,
        )
        ax.text(
            x0 + 0.13,
            0.21,
            status,
            ha="center",
            fontsize=9.8,
            weight="bold",
            color=color,
            transform=ax.transAxes,
        )

    ax.annotate(
        "",
        xy=(0.37, 0.53),
        xytext=(0.31, 0.53),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 1.7, "color": "#64748b"},
    )
    ax.annotate(
        "",
        xy=(0.69, 0.53),
        xytext=(0.63, 0.53),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 1.7, "color": "#64748b"},
    )

    ax.set_title("Directional-Confidence Gates from Relative Book State", fontsize=15, pad=10)
    save_figure(fig, out / "s6_gate_mechanics_confidence.png")


def plot_gate_mechanics_constraints(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    apply_figure_style(fig)
    apply_axes_style(ax)
    ax.axis("off")

    left = plt.Rectangle(
        (0.06, 0.20),
        0.40,
        0.66,
        edgecolor="#94a3b8",
        facecolor="#ffffff",
        linewidth=1.4,
        transform=ax.transAxes,
    )
    right = plt.Rectangle(
        (0.54, 0.20),
        0.40,
        0.66,
        edgecolor="#94a3b8",
        facecolor="#ffffff",
        linewidth=1.4,
        transform=ax.transAxes,
    )
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(
        0.26,
        0.81,
        "Why Price-Cap Exists",
        ha="center",
        fontsize=11,
        weight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.74,
        0.81,
        "Why Time Gate Exists",
        ha="center",
        fontsize=11,
        weight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        0.26,
        0.73,
        "Binary payoff is capped at the top.",
        ha="center",
        fontsize=9.5,
        transform=ax.transAxes,
    )
    ax.text(
        0.26,
        0.68,
        "Late expensive entries have little upside room.",
        ha="center",
        fontsize=9.5,
        transform=ax.transAxes,
    )

    ax.plot([0.12, 0.40], [0.52, 0.52], color="#64748b", linewidth=1.2, transform=ax.transAxes)
    ax.text(0.11, 0.55, "0", fontsize=8, transform=ax.transAxes)
    ax.text(0.39, 0.55, "1", fontsize=8, transform=ax.transAxes)
    ax.add_patch(
        plt.Rectangle(
            (0.16, 0.49), 0.10, 0.06, facecolor="#86efac", edgecolor="none", transform=ax.transAxes
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (0.31, 0.49), 0.08, 0.06, facecolor="#fca5a5", edgecolor="none", transform=ax.transAxes
        )
    )
    ax.text(
        0.21,
        0.46,
        "healthy upside",
        ha="center",
        fontsize=8.5,
        color="#15803d",
        transform=ax.transAxes,
    )
    ax.text(
        0.35,
        0.46,
        "compressed upside",
        ha="center",
        fontsize=8.5,
        color="#b91c1c",
        transform=ax.transAxes,
    )
    ax.text(
        0.26,
        0.30,
        "Gate action: block entries too near payoff ceiling.",
        ha="center",
        fontsize=9.3,
        weight="bold",
        color="#b91c1c",
        transform=ax.transAxes,
    )

    ax.text(
        0.74,
        0.73,
        "Confidence is usually lower early and improves toward resolution.",
        ha="center",
        fontsize=9.5,
        transform=ax.transAxes,
    )
    ax.text(
        0.74,
        0.68,
        "Time gate prefers later states when separation has persisted.",
        ha="center",
        fontsize=9.5,
        transform=ax.transAxes,
    )

    ax.plot([0.60, 0.90], [0.52, 0.52], color="#64748b", linewidth=1.2, transform=ax.transAxes)
    ax.text(0.60, 0.55, "early", fontsize=8, transform=ax.transAxes)
    ax.text(0.90, 0.55, "resolve", fontsize=8, ha="right", transform=ax.transAxes)
    ax.add_patch(
        plt.Rectangle(
            (0.62, 0.49), 0.13, 0.06, facecolor="#fca5a5", edgecolor="none", transform=ax.transAxes
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (0.82, 0.49), 0.06, 0.06, facecolor="#86efac", edgecolor="none", transform=ax.transAxes
        )
    )
    ax.text(
        0.685,
        0.46,
        "low-confidence zone",
        ha="center",
        fontsize=8.5,
        color="#b91c1c",
        transform=ax.transAxes,
    )
    ax.text(
        0.85,
        0.46,
        "mature-confidence zone",
        ha="center",
        fontsize=8.5,
        color="#15803d",
        transform=ax.transAxes,
    )
    ax.annotate(
        "confidence rises",
        xy=(0.87, 0.60),
        xytext=(0.66, 0.60),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#15803d"},
        color="#15803d",
        fontsize=8.7,
        va="center",
    )
    ax.text(
        0.74,
        0.30,
        "Gate action: favor later entries when confidence has matured.",
        ha="center",
        fontsize=9.3,
        weight="bold",
        color="#15803d",
        transform=ax.transAxes,
    )

    ax.set_title("Price/Time Gates from Entry Context", fontsize=15, pad=10)
    save_figure(fig, out / "s6_gate_mechanics_constraints.png")


def plot_updown_dual_books_gate_logic(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    apply_figure_style(fig)
    apply_axes_style(ax)

    stage_raw_x = 0.05
    stage_conf_x = 0.36
    stage_final_x = 0.67

    def draw_book(
        x0: float,
        y0: float,
        w: float,
        h: float,
        book: MiniBookSpec,
    ) -> None:
        frame = plt.Rectangle(
            (x0, y0),
            w,
            h,
            edgecolor=book.border_color,
            facecolor="#f8fafc",
            linewidth=1.6,
            transform=ax.transAxes,
        )
        ax.add_patch(frame)
        mid_x = x0 + w * 0.5
        ax.plot(
            [mid_x, mid_x],
            [y0 + 0.06 * h, y0 + 0.94 * h],
            color="#475569",
            linewidth=1.0,
            transform=ax.transAxes,
        )

        for i, width in enumerate(book.bid_levels):
            y = y0 + h * (0.20 + 0.18 * i)
            ax.add_patch(
                plt.Rectangle(
                    (mid_x - width * w, y),
                    width * w,
                    h * 0.10,
                    facecolor="#16a34a",
                    edgecolor="none",
                    alpha=0.75,
                    transform=ax.transAxes,
                )
            )
        for i, width in enumerate(book.ask_levels):
            y = y0 + h * (0.20 + 0.18 * i)
            ax.add_patch(
                plt.Rectangle(
                    (mid_x, y),
                    width * w,
                    h * 0.10,
                    facecolor="#dc2626",
                    edgecolor="none",
                    alpha=0.75,
                    transform=ax.transAxes,
                )
            )
        ax.text(
            x0 + w * 0.5,
            y0 + h + 0.015,
            book.label,
            ha="center",
            fontsize=9.5,
            weight="bold",
            transform=ax.transAxes,
        )

    stages: list[SignalStageSpec] = [
        SignalStageSpec(
            x_start=stage_raw_x,
            title="1) Raw cross-side ranking",
            metric="UP and DOWN look similar at this event",
            decision="BLOCK: confidence gap not persuasive",
            up_book=MiniBookSpec(
                label="UP book",
                bid_levels=[0.22, 0.19, 0.16],
                ask_levels=[0.20, 0.18, 0.16],
                border_color="#2563eb",
            ),
            down_book=MiniBookSpec(
                label="DOWN book",
                bid_levels=[0.20, 0.18, 0.16],
                ask_levels=[0.21, 0.19, 0.17],
                border_color="#ea580c",
            ),
        ),
        SignalStageSpec(
            x_start=stage_conf_x,
            title="2) Confidence filtering",
            metric="UP now shows cleaner structure than DOWN",
            decision="PASS: directional confidence is stronger",
            up_book=MiniBookSpec(
                label="UP book",
                bid_levels=[0.26, 0.22, 0.18],
                ask_levels=[0.15, 0.13, 0.11],
                border_color="#2563eb",
            ),
            down_book=MiniBookSpec(
                label="DOWN book",
                bid_levels=[0.17, 0.15, 0.13],
                ask_levels=[0.23, 0.21, 0.19],
                border_color="#ea580c",
            ),
        ),
        SignalStageSpec(
            x_start=stage_final_x,
            title="3) Final trade eligibility",
            metric="Candidate survives post-selection constraints",
            decision="PASS: high-confidence trade candidate",
            up_book=MiniBookSpec(
                label="Selected UP",
                bid_levels=[0.25, 0.21, 0.18],
                ask_levels=[0.14, 0.12, 0.10],
                border_color="#2563eb",
            ),
            down_book=MiniBookSpec(
                label="Rejected DOWN",
                bid_levels=[0.17, 0.15, 0.13],
                ask_levels=[0.22, 0.20, 0.18],
                border_color="#ea580c",
            ),
        ),
    ]

    for stage in stages:
        panel = plt.Rectangle(
            (stage.x_start, 0.14),
            0.28,
            0.76,
            edgecolor="#94a3b8",
            facecolor="#ffffff",
            linewidth=1.4,
            transform=ax.transAxes,
        )
        ax.add_patch(panel)
        ax.text(
            stage.x_start + 0.14,
            0.86,
            stage.title,
            ha="center",
            fontsize=10.5,
            weight="bold",
            transform=ax.transAxes,
        )

        draw_book(stage.x_start + 0.02, 0.49, 0.11, 0.28, stage.up_book)
        draw_book(stage.x_start + 0.15, 0.49, 0.11, 0.28, stage.down_book)

        ax.text(
            stage.x_start + 0.14,
            0.41,
            stage.metric,
            ha="center",
            fontsize=9.2,
            transform=ax.transAxes,
        )
        decision_color = "#dc2626" if "BLOCK" in stage.decision else "#15803d"
        ax.text(
            stage.x_start + 0.14,
            0.33,
            stage.decision,
            ha="center",
            fontsize=9.2,
            color=decision_color,
            weight="bold",
            transform=ax.transAxes,
        )

    ax.annotate(
        "",
        xy=(0.36, 0.52),
        xytext=(0.33, 0.52),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#475569"},
    )
    ax.annotate(
        "",
        xy=(0.67, 0.52),
        xytext=(0.64, 0.52),
        xycoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#475569"},
    )

    ax.text(
        0.5,
        0.07,
        (
            "Gates progressively remove ambiguous UP/DOWN states, "
            "leaving only high-confidence tradeable signals."
        ),
        ha="center",
        fontsize=9.5,
        color="#334155",
        transform=ax.transAxes,
    )

    ax.set_title(
        "Cross-Side Signal Filtering with Separate UP and DOWN Orderbooks",
        fontsize=15,
        pad=10,
    )
    save_figure(fig, out / "s6_up_down_dual_books_gate_logic.png")


def plot_accuracy_summary(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return

    label_map = {
        "snapshot_score": "Snapshot",
        "cumulative_sum_score": "Cumulative Sum",
        "cumulative_ewm_score": "Cumulative EWM",
    }
    ordered = df.sort_values("final_market_accuracy", ascending=True).copy()
    ordered["method_label"] = ordered["method"].map(label_map).fillna(ordered["method"])
    ordered["uplift"] = ordered["final_market_accuracy"] - ordered["timestamp_accuracy"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    ts_vals = pd.to_numeric(ordered["timestamp_accuracy"], errors="coerce").to_numpy(dtype=float)
    fin_vals = pd.to_numeric(
        ordered["final_market_accuracy"],
        errors="coerce",
    ).to_numpy(dtype=float)
    uplift_vals = pd.to_numeric(ordered["uplift"], errors="coerce").to_numpy(dtype=float)
    y = np.arange(len(ordered), dtype=float)

    for idx, (ts, fin, uplift) in enumerate(zip(ts_vals, fin_vals, uplift_vals, strict=False)):
        y_val = float(idx)
        ax.plot([ts, fin], [y_val, y_val], color="#94a3b8", linewidth=4, solid_capstyle="round")
        ax.scatter(ts, y_val, s=85, color="#ff7f0e", zorder=3)
        ax.scatter(fin, y_val, s=95, color="#1f77b4", zorder=3)
        ax.text(
            fin + 0.004,
            y_val,
            f"+{uplift:.3f}",
            va="center",
            fontsize=9,
            color="#334155",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(ordered["method_label"])
    xmin = float(
        min(ordered["timestamp_accuracy"].min(), ordered["final_market_accuracy"].min()) - 0.015
    )
    xmax = float(
        max(ordered["timestamp_accuracy"].max(), ordered["final_market_accuracy"].max()) + 0.04
    )
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Accuracy")
    ax.set_title("Method Lift: Timestamp to Final")
    ax.grid(axis="x", alpha=0.25)

    ax.scatter([], [], s=70, color="#ff7f0e", label="Timestamp")
    ax.scatter([], [], s=70, color="#1f77b4", label="Final market")
    ax.legend(loc="lower right", frameon=False)

    save_figure(fig, out / "s6_accuracy_summary.png")


def plot_timeline_heatmap(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return

    chart = df.copy()
    label_map = {
        "snapshot_score": "Snapshot",
        "cumulative_sum_score": "Cumulative Sum",
        "cumulative_ewm_score": "Cumulative EWM",
    }
    chart["method_label"] = chart["method"].map(label_map).fillna(chart["method"])

    order_buckets = ["Q1_early", "Q2", "Q3", "Q4_late"]
    chart["progress_bucket"] = pd.Categorical(
        chart["progress_bucket"],
        categories=order_buckets,
        ordered=True,
    )

    pivot = (
        chart.pivot_table(
            index="method_label",
            columns="progress_bucket",
            values="accuracy",
            aggfunc="mean",
        )
        .reindex(columns=order_buckets)
        .dropna(how="all")
    )
    if pivot.empty:
        return

    x_labels = ["Q1", "Q2", "Q3", "Q4"]
    x = np.arange(len(x_labels), dtype=float)

    palette = {
        "Snapshot": "#64748b",
        "Cumulative Sum": "#2563eb",
        "Cumulative EWM": "#f59e0b",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.2))

    for method in pivot.index:
        vals = pivot.loc[method].to_numpy(dtype=float)
        color = palette.get(method, "#334155")
        ax.plot(x, vals, marker="o", linewidth=2.5, color=color, label=method)
        ax.fill_between(x, vals, np.nanmin(vals) - 0.003, color=color, alpha=0.05)
        ax.text(
            x[-1] + 0.06,
            vals[-1],
            f"{vals[-1]:.3f}",
            color=color,
            va="center",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.1, 3.5)
    ymin = float(pivot.min().min() - 0.015)
    ymax = float(pivot.max().max() + 0.015)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Trajectory Across Market Progress")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False)

    save_figure(fig, out / "s6_timeline_accuracy_heatmap.png")


def plot_gate_ablation_scatter(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return

    chart = df.copy()
    numeric_cols = ["market_sharpe_log", "max_drawdown_pct", "trades"]
    for col in numeric_cols:
        chart[col] = pd.to_numeric(chart[col], errors="coerce")
    chart = chart.dropna(subset=["market_sharpe_log", "max_drawdown_pct", "trades"]).copy()
    chart = chart[chart["scenario"] != "base_no_gates"].copy()
    if chart.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=chart,
        x="max_drawdown_pct",
        y="market_sharpe_log",
        size="trades",
        hue="scenario",
        sizes=(80, 340),
        alpha=0.85,
        legend=False,
        ax=ax,
    )

    highlight = (
        chart.nsmallest(1, "objective_rank")
        if "objective_rank" in chart.columns
        else chart.head(1)
    )
    if not highlight.empty:
        row = highlight.iloc[0]
        ax.annotate(
            str(row["scenario"]),
            (float(row["max_drawdown_pct"]), float(row["market_sharpe_log"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_title("Gate Ablation Tradeoff: Sharpe vs Max Drawdown")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Market Sharpe (log returns)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    save_figure(fig, out / "s6_gate_ablation_tradeoff.png")


def plot_hpo_frontier(df: pd.DataFrame, out: Path) -> None:
    if df.empty:
        return

    chart = df.copy()
    for col in ["market_sharpe_log", "max_drawdown_pct", "net_pnl", "trades"]:
        chart[col] = pd.to_numeric(chart[col], errors="coerce")

    chart = chart.dropna(subset=["market_sharpe_log", "max_drawdown_pct", "net_pnl"]).copy()
    if chart.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    scatter = ax.scatter(
        chart["max_drawdown_pct"],
        chart["market_sharpe_log"],
        c=chart["net_pnl"],
        cmap="viridis",
        s=np.clip(
            chart.get("trades", pd.Series(120, index=chart.index)).fillna(120) / 10,
            40,
            260,
        ),
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Net PnL")

    if "objective_rank" in chart.columns:
        top = chart.nsmallest(1, "objective_rank")
    else:
        top = chart.nlargest(1, "market_sharpe_log")
    for _, row in top.iterrows():
        ax.annotate(
            f"best: {row['scenario']}",
            (float(row["max_drawdown_pct"]), float(row["market_sharpe_log"])),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            weight="bold",
        )

    ax.set_title("Targeted HPO Landscape")
    ax.set_xlabel("Max Drawdown (%)")
    ax.set_ylabel("Market Sharpe (log returns)")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    save_figure(fig, out / "s6_hpo_frontier.png")


def plot_market_net_pnl_distribution(df: pd.DataFrame, out: Path) -> None:
    if df.empty or "split" not in df.columns or "net_pnl" not in df.columns:
        return

    chart = df.copy()
    chart["net_pnl"] = pd.to_numeric(chart["net_pnl"], errors="coerce")
    chart = chart.dropna(subset=["net_pnl", "split"])
    if chart.empty:
        return

    split_labels = {
        "train_hpo_first_n": "train",
        "oos_remaining": "oos",
        "train": "train",
        "oos": "oos",
    }
    chart["split_label"] = chart["split"].map(split_labels).fillna(chart["split"].astype(str))

    fig, ax = plt.subplots(figsize=(9.8, 5.2))

    for split_name, group in chart.groupby("split_label", sort=False):
        values = pd.to_numeric(group["net_pnl"], errors="coerce").dropna()
        if len(values) < 2 or float(values.std()) == 0.0:
            continue
        sns.kdeplot(
            data=group,
            x="net_pnl",
            ax=ax,
            fill=True,
            alpha=0.22,
            label=str(split_name),
            warn_singular=False,
        )

    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Per-Market Net PnL Density")
    ax.set_xlabel("Net PnL")
    ax.set_ylabel("Density")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title="Split")

    save_figure(fig, out / "s6_market_net_pnl_distribution.png")


def main() -> None:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")

    artifacts_root = args.artifacts_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cumdiag_dir = resolve_cumdiag_dir(
        artifacts_root,
        args.cumdiag_dir.resolve() if args.cumdiag_dir else None,
    )
    print(f"Using cumulative diagnostics dir: {cumdiag_dir}")

    accuracy_csv = cumdiag_dir / "accuracy_summary.csv"
    timeline_csv = cumdiag_dir / "timeline_accuracy.csv"

    gate_diag_csv = artifacts_root / "gate_diagnostics_new_metrics.csv"
    market_metrics_csv = artifacts_root / "relative_book_degradation_market_metrics.csv"

    hpo_grid_csv = (
        args.hpo_grid_csv.resolve()
        if args.hpo_grid_csv is not None
        else (artifacts_root / "gate_grid_search_first3000_grid010_ddaware_compact.csv")
    )

    # Section 6.4 figures are anchored to the full-market table values in the report.
    # This avoids accidental drift when a different cumulative diagnostics run is present.
    accuracy_df = pd.DataFrame(
        [
            {
                "method": "cumulative_sum_score",
                "timestamp_accuracy": 0.549,
                "final_market_accuracy": 0.665,
                "markets": 5076,
            },
            {
                "method": "cumulative_ewm_score",
                "timestamp_accuracy": 0.543,
                "final_market_accuracy": 0.583,
                "markets": 5076,
            },
            {
                "method": "snapshot_score",
                "timestamp_accuracy": 0.509,
                "final_market_accuracy": 0.462,
                "markets": 5076,
            },
        ]
    )
    timeline_df = pd.DataFrame(
        [
            {"method": "cumulative_sum_score", "progress_bucket": "Q1_early", "accuracy": 0.517},
            {"method": "cumulative_sum_score", "progress_bucket": "Q2", "accuracy": 0.539},
            {"method": "cumulative_sum_score", "progress_bucket": "Q3", "accuracy": 0.557},
            {"method": "cumulative_sum_score", "progress_bucket": "Q4_late", "accuracy": 0.582},
            {"method": "cumulative_ewm_score", "progress_bucket": "Q1_early", "accuracy": 0.511},
            {"method": "cumulative_ewm_score", "progress_bucket": "Q2", "accuracy": 0.528},
            {"method": "cumulative_ewm_score", "progress_bucket": "Q3", "accuracy": 0.546},
            {"method": "cumulative_ewm_score", "progress_bucket": "Q4_late", "accuracy": 0.586},
            {"method": "snapshot_score", "progress_bucket": "Q1_early", "accuracy": 0.500},
            {"method": "snapshot_score", "progress_bucket": "Q2", "accuracy": 0.502},
            {"method": "snapshot_score", "progress_bucket": "Q3", "accuracy": 0.504},
            {"method": "snapshot_score", "progress_bucket": "Q4_late", "accuracy": 0.529},
        ]
    )
    gate_diag_df = read_csv_or_none(gate_diag_csv)
    hpo_grid_df = read_csv_or_none(hpo_grid_csv)
    market_metrics_df = read_csv_or_none(market_metrics_csv)

    plot_signal_design_concept(output_dir)
    plot_cumulative_memory_concept(output_dir)
    plot_gate_mechanics_execution(output_dir)
    plot_gate_mechanics_confidence(output_dir)
    plot_gate_mechanics_constraints(output_dir)

    if accuracy_df is not None:
        plot_accuracy_summary(accuracy_df, output_dir)
    if timeline_df is not None:
        plot_timeline_heatmap(timeline_df, output_dir)
    plot_updown_dual_books_gate_logic(output_dir)
    if gate_diag_df is not None:
        plot_gate_ablation_scatter(gate_diag_df, output_dir)
    if hpo_grid_df is not None:
        plot_hpo_frontier(hpo_grid_df, output_dir)
    if market_metrics_df is not None:
        plot_market_net_pnl_distribution(market_metrics_df, output_dir)

    print(f"Done. Figures written to: {output_dir}")


if __name__ == "__main__":
    main()
