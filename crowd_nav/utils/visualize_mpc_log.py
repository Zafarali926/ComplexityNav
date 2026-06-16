#!/usr/bin/env python3
"""
Replay / plot MPC-SGAN (vecMPC) episode logs written under REAL_WORLD/REAL_WORLD/.

Each completed episode K produces:
  K.npy           — executed trajectories, shape (T, 1+H, 5)  [px, py, vx, vy, radius]
  K_ballbot.npy   — per-step robot pose extras (optional)
  K.pkl           — per planning step: state, action_set, predictions, cost, goal, action

Examples (from crowd_nav/):
  python utils/visualize_mpc_log.py --episode 42
  python utils/visualize_mpc_log.py --episode 42 --step 10
  python utils/visualize_mpc_log.py --episode 42 --animate --save plots/ep42.gif
  python utils/visualize_mpc_log.py --list
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter


DEFAULT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "REAL_WORLD",
    "REAL_WORLD",
)


def list_episodes(log_dir: str) -> list:
    ids = []
    for path in glob.glob(os.path.join(log_dir, "*.npy")):
        base = os.path.basename(path)
        if base.endswith("_ballbot.npy"):
            continue
        stem = base.replace(".npy", "")
        if stem.isdigit():
            ids.append(int(stem))
    return sorted(ids)


def load_episode(log_dir: str, episode: int):
    traj_path = os.path.join(log_dir, f"{episode}.npy")
    pkl_path = os.path.join(log_dir, f"{episode}.pkl")
    if not os.path.isfile(traj_path):
        raise FileNotFoundError(traj_path)
    trajectory = np.load(traj_path)
    predictions_log = None
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            predictions_log = pickle.load(f)
    return trajectory, predictions_log


def _agent_xy(traj: np.ndarray, agent_idx: int):
    return traj[:, agent_idx, 0], traj[:, agent_idx, 1]


def plot_executed_paths(traj: np.ndarray, ax, title: str = "Executed paths", goal=None):
    """traj: (T, 1+H, 5) — index 0 = robot, 1..H = humans."""
    n_agents = traj.shape[1]
    rx, ry = _agent_xy(traj, 0)
    ax.plot(rx, ry, "k-", linewidth=2, label="robot")
    ax.plot(rx[0], ry[0], "ko", markersize=6)
    ax.plot(rx[-1], ry[-1], "k*", markersize=10)

    for h in range(1, n_agents):
        hx, hy = _agent_xy(traj, h)
        ax.plot(hx, hy, "-", linewidth=1, alpha=0.8, label=f"human {h}")
        r = float(traj[0, h, 4])
        ax.add_patch(Circle((hx[-1], hy[-1]), r, fill=False, linestyle="--", alpha=0.5))

    if goal is not None:
        goal = np.asarray(goal).reshape(-1)
        ax.plot(goal[0], goal[1], "g*", markersize=14, label="goal")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def plot_mpc_step(traj: np.ndarray, step_log: dict, ax, step_idx: int):
    """Overlay SGAN rollouts for one planning step (best-cost sample)."""
    goal = step_log.get("goal")
    plot_executed_paths(traj[: step_idx + 1], ax, title=f"Through step {step_idx}", goal=goal)

    state = np.asarray(step_log["state"])
    preds = step_log.get("predictions")
    costs = step_log.get("cost")
    action = step_log.get("action")

    if preds is not None:
        preds = np.asarray(preds)
        # Typical SGAN: (N, S, T', H, 4) — px, py, vx, vy per human on horizon
        if preds.ndim == 5:
            if costs is not None:
                costs = np.asarray(costs)
                if costs.ndim >= 2:
                    best_s = int(np.argmin(costs.reshape(costs.shape[0], -1).mean(axis=1)))
                else:
                    best_s = int(np.argmin(costs))
            else:
                best_s = 0
            n_h = min(preds.shape[3], state.shape[0] - 1)
            for h in range(1, n_h + 1):
                px = preds[0, best_s, :, h - 1, 0]
                py = preds[0, best_s, :, h - 1, 1]
                ax.plot(px, py, "c--", alpha=0.5, linewidth=1)
            ax.plot([], [], "c--", label="SGAN pred (best sample)")

    if action is not None:
        action = np.asarray(action).reshape(-1)
        if state.shape[0] > 0 and action.size >= 2:
            x0, y0 = state[0, 0], state[0, 1]
            ax.arrow(x0, y0, action[0], action[1], head_width=0.08, color="red",
                     length_includes_head=True, label="chosen action")

    ax.legend(loc="upper right", fontsize=7)


def animate_episode(traj: np.ndarray, save_path: str | None, fps: int = 8):
    fig, ax = plt.subplots(figsize=(7, 7))
    n_agents = traj.shape[1]
    lines = []
    for i in range(n_agents):
        color = "k" if i == 0 else None
        (ln,) = ax.plot([], [], "-" if i == 0 else "-", color=color, linewidth=2 if i == 0 else 1)
        lines.append(ln)
    goal = None
    if os.path.isfile(save_path or ""):
        pass

    def init():
        ax.set_xlim(traj[:, :, 0].min() - 1, traj[:, :, 0].max() + 1)
        ax.set_ylim(traj[:, :, 1].min() - 1, traj[:, :, 1].max() + 1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        return lines

    def update(frame):
        for i, ln in enumerate(lines):
            ln.set_data(traj[: frame + 1, i, 0], traj[: frame + 1, i, 1])
        return lines

    anim = FuncAnimation(fig, update, frames=traj.shape[0], init_func=init, blit=True)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=fps))
        print("Saved animation:", save_path)
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize vecMPC REAL_WORLD logs")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--episode", type=int, default=None, help="Episode id (matches K.npy)")
    parser.add_argument("--step", type=int, default=None, help="Planning step index inside .pkl")
    parser.add_argument("--list", action="store_true", help="List available episode ids")
    parser.add_argument("--animate", action="store_true", help="Animate executed trajectories")
    parser.add_argument("--save", type=str, default=None, help="Save figure or .gif path")
    args = parser.parse_args()

    if args.list:
        eps = list_episodes(args.log_dir)
        print(f"Found {len(eps)} episodes in {args.log_dir}")
        if eps:
            print("  min:", eps[0], " max:", eps[-1])
            print("  last 10:", eps[-10:])
        return

    if args.episode is None:
        eps = list_episodes(args.log_dir)
        if not eps:
            raise SystemExit(f"No .npy logs in {args.log_dir}")
        args.episode = eps[-1]
        print(f"No --episode given; using latest: {args.episode}")

    traj, pred_log = load_episode(args.log_dir, args.episode)
    print(f"Episode {args.episode}: trajectory shape {traj.shape} (T, 1+H, 5)")
    if pred_log is not None:
        print(f"  planning steps in .pkl: {len(pred_log)}")
    else:
        print("  no .pkl (trajectory-only plot)")

    if args.animate:
        out = args.save or None
        animate_episode(traj, out)
        return

    fig, axes = plt.subplots(1, 2 if args.step is not None and pred_log else 1,
                             figsize=(14 if args.step is not None and pred_log else 7, 6),
                             squeeze=False)
    ax = axes[0, 0]
    last_goal = pred_log[-1].get("goal") if pred_log else None
    plot_executed_paths(traj, ax, title=f"Episode {args.episode} — executed", goal=last_goal)

    if args.step is not None:
        if pred_log is None:
            raise SystemExit("No .pkl for this episode; cannot plot --step")
        if args.step < 0 or args.step >= len(pred_log):
            raise SystemExit(f"--step must be in [0, {len(pred_log) - 1}]")
        plot_mpc_step(traj, pred_log[args.step], axes[0, 1], args.step)

    plt.tight_layout()
    if args.save and not args.animate:
        plt.savefig(args.save, dpi=150)
        print("Saved:", args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
