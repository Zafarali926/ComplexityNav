"""
Visualize ComplexityNav / ICRA benchmark test results in Google Colab.

Usage in Colab:
  1. Upload `results.json` OR `RGL_Test_result.log` when prompted.
  2. Run all cells (or: !python visualize_benchmark_colab.py after copying this file).

Each row = one (e, se) benchmark cell (500 test episodes aggregated into 16 metrics).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Benchmark grid metadata (BaseExperimentsConfig in configs/icra_benchmark/config.py)
# Order must match test.py loops: for e in range(4): for se in range(...)
# ---------------------------------------------------------------------------

BENCHMARK_SPEC = [
    # e, se, orca, sfm, linear, static, scenario, dx_lo, dx_hi, dy_lo, dy_hi, sweep_name
    (0, 0, 2, 3, 0, 0, "passing_crossing", -5, 5, -5, 5, "density"),
    (0, 1, 5, 5, 0, 0, "passing_crossing", -5, 5, -5, 5, "density"),
    (0, 2, 7, 8, 0, 0, "passing_crossing", -5, 5, -5, 5, "density"),
    (0, 3, 10, 10, 0, 0, "passing_crossing", -5, 5, -5, 5, "density"),
    (0, 4, 12, 13, 0, 0, "passing_crossing", -5, 5, -5, 5, "density"),
    (0, 5, 15, 15, 0, 0, "passing_crossing", -5, 5, -5, 5, "density"),
    (0, 6, 17, 18, 0, 0, "passing_crossing", -5, 5, -5, 5, "density"),
    (1, 0, 1, 2, 0, 0, "passing_crossing", -0.75, 0.75, -5, 5, "geometry"),
    (1, 1, 2, 2, 0, 0, "passing_crossing", -1, 1, -5, 5, "geometry"),
    (1, 2, 2, 3, 0, 0, "passing_crossing", -1.25, 1.25, -5, 5, "geometry"),
    (1, 3, 3, 3, 0, 0, "passing_crossing", -1.5, 1.5, -5, 5, "geometry"),
    (1, 4, 3, 4, 0, 0, "passing_crossing", -1.75, 1.75, -5, 5, "geometry"),
    (1, 5, 4, 4, 0, 0, "passing_crossing", -2, 2, -5, 5, "geometry"),
    (1, 6, 4, 5, 0, 0, "passing_crossing", -2.25, 2.25, -5, 5, "geometry"),
    (2, 0, 7, 8, 0, 0, "passing", -5, 5, -5, 5, "directionality"),
    (2, 1, 7, 8, 0, 0, "crossing", -5, 5, -5, 5, "directionality"),
    (2, 2, 7, 8, 0, 0, "passing_crossing", -5, 5, -5, 5, "directionality"),
    (2, 3, 7, 8, 0, 0, "random", -5, 5, -5, 5, "directionality"),
    (2, 4, 7, 8, 0, 0, "circle_crossing", -5, 5, -6, 6, "directionality"),
    (3, 0, 0, 15, 0, 0, "passing_crossing", -5, 5, -5, 5, "policy_mixture"),
    (3, 1, 15, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "policy_mixture"),
    (3, 2, 7, 8, 0, 0, "passing_crossing", -5, 5, -5, 5, "policy_mixture"),
    (3, 3, 5, 5, 2, 3, "passing_crossing", -5, 5, -5, 5, "policy_mixture"),
    (3, 4, 4, 4, 4, 3, "passing_crossing", -5, 5, -5, 5, "policy_mixture"),
    (4, 0, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_policy_mixture"),
    (4, 1, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_policy_mixture"),
    (4, 2, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_policy_mixture"),
    (4, 3, 0, 0, 2, 3, "passing_crossing", -5, 5, -5, 5, "pl_policy_mixture"),
    (4, 4, 0, 0, 4, 3, "passing_crossing", -5, 5, -5, 5, "pl_policy_mixture"),
    (5, 0, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_density"),
    (5, 1, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_density"),
    (5, 2, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_density"),
    (5, 3, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_density"),
    (5, 4, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_density"),
    (5, 5, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_density"),
    (5, 6, 0, 0, 0, 0, "passing_crossing", -5, 5, -5, 5, "pl_density"),
]

EXP_STATS_COLUMNS = [
    "success_rate",
    "success_std",
    "collision_rate",
    "collision_std",
    "timeout_rate",
    "timeout_std",
    "avg_nav_time",
    "nav_time_std",
    "avg_min_dist",
    "min_dist_std",
    "avg_accel",
    "avg_accel_std",
    "avg_min_dist_overall",
    "min_dist_overall_std",
    "avg_path_irregularity",
    "avg_path_irregularity_std",
]


def load_results_from_json(path: str | Path) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of rows")
    return data


def load_results_from_log(path: str | Path) -> list:
    text = Path(path).read_text()
    # Find first '[' ... ']' block (printed exp_stats_list)
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No Python list found in log file")
    return ast.literal_eval(match.group(0))


def load_results(path: str | Path) -> list:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_results_from_json(path)
    return load_results_from_log(path)


def build_dataframe(results: list, model_name: str = "RGL") -> pd.DataFrame:
    rows = []
    n = min(len(results), len(BENCHMARK_SPEC))
    if len(results) != len(BENCHMARK_SPEC):
        print(
            f"Warning: got {len(results)} result rows, expected {len(BENCHMARK_SPEC)}. "
            f"Using first {n} rows only."
        )

    for i in range(n):
        spec = BENCHMARK_SPEC[i]
        e, se, orca, sfm, linear, static, scenario, dx_lo, dx_hi, dy_lo, dy_hi, sweep = spec
        metrics = list(results[i])
        if len(metrics) < len(EXP_STATS_COLUMNS):
            metrics.extend([np.nan] * (len(EXP_STATS_COLUMNS) - len(metrics)))
        elif len(metrics) > len(EXP_STATS_COLUMNS):
            metrics = metrics[: len(EXP_STATS_COLUMNS)]

        row = {
            "idx": i,
            "model": model_name,
            "e": e,
            "se": se,
            "sweep": sweep,
            "orca": orca,
            "sfm": sfm,
            "linear": linear,
            "static": static,
            "scenario": scenario,
            "dx_lo": dx_lo,
            "dx_hi": dx_hi,
            "dy_lo": dy_lo,
            "dy_hi": dy_hi,
            "spawn_width_x": dx_hi - dx_lo,
            "spawn_width_y": dy_hi - dy_lo,
            "powerlaw": 0,
            "pledestrians": 0,
            "human_count": orca + sfm + linear + static,
            "mix_label": f"O{orca}/SF{sfm}/L{linear}/S{static}",
        }
        row.update(dict(zip(EXP_STATS_COLUMNS, metrics)))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_density_sweep(df: pd.DataFrame, metric: str = "success_rate", out_dir: Path | None = None):
    d = df[df.sweep == "density"].sort_values("human_count")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d["human_count"], d[metric], "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Human count (ORCA + SFM + Power Law + PLEdestrians)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Density sweep (e=0) — {metric}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05 if "rate" in metric else None)
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / f"density_{metric}.png", dpi=150)
    plt.show()


def plot_geometry_sweep(df: pd.DataFrame, metric: str = "success_rate", out_dir: Path | None = None):
    d = df[df.sweep == "geometry"].sort_values("dx_hi")
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d["dx_hi"], d[metric], "s-", linewidth=2, markersize=8, color="C1")
    ax.set_xlabel("Spawn half-width dx_hi (m)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Environment width / geometry (e=1) — {metric}")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05 if "rate" in metric else None)
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / f"geometry_{metric}.png", dpi=150)
    plt.show()


def plot_directionality(df: pd.DataFrame, metric: str = "success_rate", out_dir: Path | None = None):
    d = df[df.sweep == "directionality"].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(d))
    ax.bar(x, d[metric], color="C2", edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(d["scenario"], rotation=25, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Directionality / scenario type (e=2) — {metric}")
    ax.set_ylim(0, 1.05 if "rate" in metric else None)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / f"directionality_{metric}.png", dpi=150)
    plt.show()


def plot_policy_mixture(df: pd.DataFrame, metric: str = "success_rate", out_dir: Path | None = None):
    d = df[df.sweep == "policy_mixture"].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(d))
    ax.bar(x, d[metric], color="C3", edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(d["mix_label"], rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Policy mixture (e=3) — {metric}")
    ax.set_ylim(0, 1.05 if "rate" in metric else None)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / f"policy_mixture_{metric}.png", dpi=150)
    plt.show()


def plot_heatmap_all(df: pd.DataFrame, metric: str = "success_rate", out_dir: Path | None = None):
    """24 cells: rows=e, columns=se (variable width per row — padded with NaN)."""
    pivot = df.pivot_table(index="e", columns="se", values=metric, aggfunc="first")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"e={i}" for i in pivot.index])
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"se={j}" for j in pivot.columns])
    ax.set_title(f"All benchmark cells — {metric}")
    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / f"heatmap_{metric}.png", dpi=150)
    plt.show()


def plot_rates_stacked(df: pd.DataFrame, out_dir: Path | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    sweeps = ["density", "geometry", "directionality", "policy_mixture"]
    titles = ["Density (e=0)", "Geometry (e=1)", "Directionality (e=2)", "Policy mix (e=3)"]

    for ax, sweep, title in zip(axes.flat, sweeps, titles):
        d = df[df.sweep == sweep]
        if sweep == "density":
            x, xl = d["human_count"], "Human count"
        elif sweep == "geometry":
            x, xl = d["dx_hi"], "dx_hi (m)"
        elif sweep == "directionality":
            x, xl = np.arange(len(d)), "Scenario"
            ax.set_xticks(x)
            ax.set_xticklabels(d["scenario"], rotation=25, ha="right")
        else:
            x, xl = np.arange(len(d)), "Mix"
            ax.set_xticks(x)
            ax.set_xticklabels(d["mix_label"], rotation=20, ha="right")

        ax.plot(x, d["success_rate"], "o-", label="Success", linewidth=2)
        ax.plot(x, d["collision_rate"], "s-", label="Collision", linewidth=2)
        ax.plot(x, d["timeout_rate"], "^-", label="Timeout", linewidth=2)
        if sweep not in ("directionality", "policy_mixture"):
            ax.set_xlabel(xl)
        ax.set_ylabel("Rate")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Outcome rates across benchmark sweeps", y=1.02)
    plt.tight_layout()
    if out_dir:
        fig.savefig(out_dir / "rates_overview.png", dpi=150, bbox_inches="tight")
    plt.show()


def run_visualization(results_path: str | Path, model_name: str = "RGL", save_dir: str | Path | None = None):
    results_path = Path(results_path)
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)
    df = build_dataframe(results, model_name=model_name)

    print(df[["idx", "e", "se", "sweep", "human_count", "scenario", "mix_label",
              "success_rate", "collision_rate", "timeout_rate", "avg_nav_time"]].to_string())

    if save_dir:
        csv_path = save_dir / "benchmark_parsed.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved table: {csv_path}")

    for metric in ("success_rate", "collision_rate", "avg_nav_time"):
        plot_density_sweep(df, metric, save_dir)
        plot_geometry_sweep(df, metric, save_dir)
        plot_directionality(df, metric, save_dir)
        plot_policy_mixture(df, metric, save_dir)

    plot_heatmap_all(df, "success_rate", save_dir)
    plot_rates_stacked(df, save_dir)

    return df


# ---------------------------------------------------------------------------
# Colab entry point
# ---------------------------------------------------------------------------

def colab_main():
    try:
        from google.colab import files  # type: ignore
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        print("Upload results.json or RGL_Test_result.log")
        uploaded = files.upload()
        if not uploaded:
            raise SystemExit("No file uploaded")
        path = list(uploaded.keys())[0]
        model_name = "RGL" if "rgl" in path.lower() else "Model"
    else:
        import argparse

        parser = argparse.ArgumentParser(description="Visualize ICRA benchmark test results")
        parser.add_argument("results_path", nargs="?", default="results.json")
        parser.add_argument("--model-name", default="RGL")
        parser.add_argument("--save-dir", default="benchmark_plots")
        args = parser.parse_args()
        path = args.results_path
        model_name = args.model_name
        save_dir = args.save_dir
        df = run_visualization(path, model_name=model_name, save_dir=save_dir)
        return df

    save_dir = Path("/content/benchmark_plots")
    df = run_visualization(path, model_name=model_name, save_dir=save_dir)

    # Offer zip download of plots + CSV
    import shutil

    zip_path = "/content/benchmark_plots.zip"
    shutil.make_archive("/content/benchmark_plots", "zip", save_dir)
    files.download(zip_path)
    return df


if __name__ == "__main__":
    colab_main()
