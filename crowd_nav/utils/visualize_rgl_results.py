#!/usr/bin/env python
"""
Plot RGL / ICRA benchmark results (24 cells) — paper-style axes, all line plots.

Example:
  cd crowd_nav
  python utils/visualize_rgl_results.py "results-rgl Jun 06.json"
  python utils/visualize_rgl_results.py "results-rgl Jun 06.json" --save-dir plots/rgl_jun06
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.visualize_benchmark_colab import build_dataframe, load_results

# Y-axis: (column, label, tick values)
METRICS = [
    ("success_rate", "Success rate", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("avg_nav_time", "Average time (s)", [0, 7, 14, 21, 28, 35, 42]),
    ("avg_min_dist", "Min. dist. to human (m)", [0.0, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]),
    ("avg_path_irregularity", "Path irregularity (rad/m)", [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]),
]

GEOMETRY_X_TICKS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

# Paper order left → right on X
DIRECTIONALITY_ORDER = [
    ("passing", "P"),
    ("crossing", "C"),
    ("passing_crossing", "P,C"),
    ("circle_crossing", "CC"),
    ("random", "R"),
]

POLICY_MIX_ORDER = [
    (0, "SF"),      # O0/SF15/L0/S0
    (1, "ORCA"),    # O15/SF0/L0/S0
    (2, "Mix1"),    # O8/SF7/L0/S0
    (3, "Mix2"),    # O5/SF5/L2/S3
    (4, "Mix3"),    # O4/SF4/L4/S3
    # Custom mixes (human_mix_presets PL_only / PLD_only) use powerlaw / pledestrians counts
]

SWEEPS = [
    ("density", "Density (agents/m²)"),
    ("geometry", "Environment width (m)"),
    ("directionality", "Directionality"),
    ("policy_mixture", "Policy mixture"),
]


def add_paper_axes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pl = df["powerlaw"] if "powerlaw" in df.columns else 0
    pld = df["pledestrians"] if "pledestrians" in df.columns else 0
    df["density_x"] = (df["orca"] + df["sfm"] + pl + pld) / 100.0
    df["geometry_x"] = 2.0 * df["dx_hi"]
    return df


def prepare_sweep(df: pd.DataFrame, sweep: str) -> tuple[pd.DataFrame, str, list | None, list | None]:
    """Return (sorted rows, x column, optional fixed xticks, optional x tick labels)."""
    d = df[df.sweep == sweep].copy()

    if sweep == "density":
        d = d.sort_values("density_x")
        return d, "density_x", None, None

    if sweep == "geometry":
        d = d.sort_values("geometry_x")
        return d, "geometry_x", GEOMETRY_X_TICKS, None

    if sweep == "directionality":
        rows = []
        labels = []
        for scenario, label in DIRECTIONALITY_ORDER:
            match = d[d.scenario == scenario]
            if match.empty:
                continue
            rows.append(match.iloc[0])
            labels.append(label)
        d = pd.DataFrame(rows)
        d["x_pos"] = np.arange(len(d))
        return d, "x_pos", list(range(len(d))), labels

    if sweep == "policy_mixture":
        rows = []
        labels = []
        for se, label in POLICY_MIX_ORDER:
            match = d[d.se == se]
            if match.empty:
                continue
            rows.append(match.iloc[0])
            labels.append(label)
        d = pd.DataFrame(rows)
        d["x_pos"] = np.arange(len(d))
        return d, "x_pos", list(range(len(d))), labels

    raise ValueError(f"Unknown sweep: {sweep}")


def apply_y_axis(ax, y_ticks: list):
    ax.set_yticks(y_ticks)
    ax.set_ylim(y_ticks[0], y_ticks[-1])


def apply_x_axis(ax, x_ticks: list | None, x_labels: list | None, xlim: tuple | None = None):
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if x_labels is not None:
        ax.set_xticklabels(x_labels, fontsize=8)
    if xlim is not None:
        ax.set_xlim(*xlim)


def plot_sweep(ax, d: pd.DataFrame, x_col: str, metric_col: str, ylabel: str, y_ticks: list,
               x_ticks: list | None, x_labels: list | None, xlim: tuple | None = None,
               show_ylabel: bool = True):
    xs = d[x_col].values
    ys = d[metric_col].values
    ax.plot(xs, ys, "o-", linewidth=2, markersize=7, color="steelblue")
    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    apply_y_axis(ax, y_ticks)
    apply_x_axis(ax, x_ticks, x_labels, xlim)
    ax.grid(True, alpha=0.3)


def _sweep_xlim(sweep: str, d: pd.DataFrame):
    if sweep == "geometry":
        return (GEOMETRY_X_TICKS[0], GEOMETRY_X_TICKS[-1])
    if sweep in ("directionality", "policy_mixture") and len(d) > 1:
        return (-0.15, len(d) - 0.85)
    return None


def plot_all(df: pd.DataFrame, model_name: str, save_dir: Path | None):
    df = add_paper_axes(df)

    # Rows = sweeps, columns = metrics
    fig, axes = plt.subplots(len(SWEEPS), len(METRICS), figsize=(16, 12))
    fig.suptitle(f"{model_name} — benchmark metrics", fontsize=14, y=1.01)

    for row, (sweep, sweep_title) in enumerate(SWEEPS):
        d, x_col, x_ticks, x_labels = prepare_sweep(df, sweep)
        xlim = _sweep_xlim(sweep, d)
        for col, (metric_col, metric_ylabel, y_ticks) in enumerate(METRICS):
            ax = axes[row, col]
            plot_sweep(
                ax, d, x_col, metric_col, metric_ylabel, y_ticks,
                x_ticks, x_labels, xlim,
                show_ylabel=(col == 0),
            )
            if row == 0:
                ax.set_title(metric_ylabel, fontsize=10)
            if col == 0:
                ax.text(
                    -0.32, 0.5, sweep_title, transform=ax.transAxes,
                    va="center", ha="center", rotation=90, fontsize=10,
                )
            if row == len(SWEEPS) - 1:
                ax.set_xlabel(sweep_title, fontsize=9)

    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "rgl_benchmark_4x4.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        df.to_csv(save_dir / "rgl_benchmark_parsed.csv", index=False)
        print(f"Saved: {save_dir / 'rgl_benchmark_parsed.csv'}")
    plt.show()


def plot_per_sweep(df: pd.DataFrame, model_name: str, save_dir: Path | None):
    df = add_paper_axes(df)
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for sweep, x_title in SWEEPS:
        d, x_col, x_ticks, x_labels = prepare_sweep(df, sweep)
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f"{model_name} — {x_title}", fontsize=12)
        for ax, (metric_col, metric_ylabel, y_ticks) in zip(axes.flat, METRICS):
            xlim = _sweep_xlim(sweep, d)
            plot_sweep(
                ax, d, x_col, metric_col, metric_ylabel, y_ticks,
                x_ticks, x_labels, xlim,
            )
            ax.set_xlabel(sweep_title, fontsize=9)
        plt.tight_layout()
        if save_dir:
            fname = save_dir / f"rgl_{sweep}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"Saved: {fname}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot RGL benchmark JSON (4 metrics × 4 sweeps)")
    parser.add_argument(
        "results_path",
        nargs="?",
        default=str(_ROOT / "results-rgl Jun 06.json"),
    )
    parser.add_argument("--model-name", default="RGL")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--separate-sweeps", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    results = load_results(args.results_path)
    df = build_dataframe(results, model_name=args.model_name)
    save_dir = Path(args.save_dir) if args.save_dir else None

    if args.no_show:
        plt.ioff()

    plot_all(df, args.model_name, save_dir)
    if args.separate_sweeps and save_dir:
        plot_per_sweep(df, args.model_name, save_dir)


if __name__ == "__main__":
    main()
