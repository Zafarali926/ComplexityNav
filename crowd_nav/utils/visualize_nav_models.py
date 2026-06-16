#!/usr/bin/env python
"""
Compare multiple navigation algorithms on the ICRA benchmark grid.

Default inputs: crowd_nav/Nav-models/
  - rgl_results.json      → RGL
  - SGAN-MPC.log          → MPC-SGAN
  - mpc-cv_results.json   → MPC-CV
  - RP_results.json       → RP
  - orca_results.json     → ORCA
  - sfm_results.json      → SFM
  - powerlaw_results.json → Power Law
  - pledestrians_results.json → PLEdestrians

Layout: rows = sweeps (Density, Env width, Directionality, Policy mix)
        columns = metrics (Success rate, Avg time, Min dist, Path irregularity)

Example:
  cd crowd_nav
  python utils/visualize_nav_models.py
  python utils/visualize_nav_models.py --results-dir Nav-models --save-dir plots/nav_compare
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.visualize_benchmark_colab import build_dataframe
from utils.visualize_rgl_results import (
    METRICS,
    SWEEPS,
    add_paper_axes,
    apply_x_axis,
    apply_y_axis,
    prepare_sweep,
    _sweep_xlim,
)

# (display name, filename in results-dir, color)
DEFAULT_ALGORITHMS = [
    ("RGL", "rgl_results.json", "#E41A1C"),       # Red
    ("MPC-SGAN", "SGAN-MPC.log", "#FF7F00"),      # Orange
    ("MPC-CV", "mpc-cv_results.json", "#FFD700"), # Gold / light yellow
    ("RP", "RP_results.json", "#90EE90"),         # Light green
    ("ORCA", "orca_results.json", "#008000"),     # Green
    ("CV", "cv_results.json", "#A98AD6"),         # Light Purple / Lavender
    ("SFM", "sfm_results.json", "#377EB8"),       # Blue
    #("Power Law", "powerlaw_results.json", "#984EA3"),       # Purple
    #("PLEdestrians", "pledestrians_results.json", "#4DAF4A"),  # Yellow-green
]


def load_results_robust(path: Path) -> list:
    """Load benchmark rows from JSON or a log file containing a Python list."""
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: root must be a list")
        return data

    text = path.read_text()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            data = ast.literal_eval(match.group(0))
            if isinstance(data, list):
                return data
        except (SyntaxError, ValueError):
            pass

    tuple_re = re.compile(r"\(([\d\.\-eE\+]+(?:,\s*[\d\.\-eE\+]+){15})\)")
    rows = []
    for m in tuple_re.finditer(text):
        rows.append([float(x.strip()) for x in m.group(1).split(",")])
    if not rows:
        raise ValueError(f"Could not parse any result rows from {path}")
    if len(rows) != 24:
        print(f"Warning: {path.name} has {len(rows)} rows (expected 24); using available rows")
    return rows


def load_algorithm_results(results_dir: Path, name: str, filename: str) -> pd.DataFrame:
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing results for {name}: {path}")
    results = load_results_robust(path)
    df = build_dataframe(results, model_name=name)
    return add_paper_axes(df)


def load_all_algorithms(results_dir: Path, algorithms=None) -> list[tuple[str, pd.DataFrame, str]]:
    specs = algorithms or DEFAULT_ALGORITHMS
    loaded = []
    for name, filename, color in specs:
        df = load_algorithm_results(results_dir, name, filename)
        loaded.append((name, df, color))
        print(f"Loaded {name}: {results_dir / filename} ({len(df)} cells)")
    return loaded


def plot_multi_sweep(
    ax,
    models: list[tuple[str, pd.DataFrame, str]],
    sweep: str,
    metric_col: str,
    y_ticks: list,
    show_ylabel: bool,
    metric_ylabel: str,
):
    x_ticks = x_labels = None
    xlim = None

    for name, df, color in models:
        d, x_col, xt, xl = prepare_sweep(df, sweep)
        if x_ticks is None:
            x_ticks, x_labels = xt, xl
            xlim = _sweep_xlim(sweep, d)
        ax.plot(
            d[x_col].values,
            d[metric_col].values,
            "o-",
            linewidth=2,
            markersize=5,
            color=color,
            label=name,
        )

    if show_ylabel and metric_ylabel:
        ax.set_ylabel(metric_ylabel, fontsize=9)
    apply_y_axis(ax, y_ticks)
    apply_x_axis(ax, x_ticks, x_labels, xlim)
    ax.grid(True, alpha=0.3)


def plot_comparison(models: list[tuple[str, pd.DataFrame, str]], save_dir: Path | None):
    fig, axes = plt.subplots(len(SWEEPS), len(METRICS), figsize=(18, 12))

    for row, (sweep, sweep_title) in enumerate(SWEEPS):
        for col, (metric_col, metric_ylabel, y_ticks) in enumerate(METRICS):
            ax = axes[row, col]
            plot_multi_sweep(
                ax, models, sweep, metric_col, y_ticks,
                show_ylabel=(col == 0),
                metric_ylabel=metric_ylabel,
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

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(models),
        bbox_to_anchor=(0.5, 0.98),
        fontsize=9,
        frameon=True,
    )
    fig.suptitle("Navigation algorithms — benchmark comparison", fontsize=14, y=1.0)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / "nav_models_comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", bbox_extra_artists=[legend])
        print(f"Saved: {out}")
        combined = pd.concat([df.assign(algorithm=name) for name, df, _ in models], ignore_index=True)
        combined.to_csv(save_dir / "nav_models_combined.csv", index=False)
        print(f"Saved: {save_dir / 'nav_models_combined.csv'}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare Nav-models benchmark results")
    parser.add_argument(
        "--results-dir",
        default=str(_ROOT / "Nav-models"),
        help="Directory containing per-algorithm JSON/log files",
    )
    parser.add_argument("--save-dir", default=None, help="Output directory for PNG/CSV")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.no_show:
        plt.ioff()

    results_dir = Path(args.results_dir)
    models = load_all_algorithms(results_dir)
    save_dir = Path(args.save_dir) if args.save_dir else None
    plot_comparison(models, save_dir)


if __name__ == "__main__":
    main()
