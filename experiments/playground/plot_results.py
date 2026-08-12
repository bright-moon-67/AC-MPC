"""Plot existing Playground training curves and final evaluations on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = ("CartpoleSwingup", "ReacherHard", "HopperHop", "WalkerRun", "HumanoidRun")
METHODS = ("PPO", "KMPC", "AB-PQ", "AC-MPC-MPVE")
COLORS = {
    "PPO": "#1f77b4",
    "KMPC": "#ff7f0e",
    "AB-PQ": "#2ca02c",
    "AC-MPC-MPVE": "#d62728",
}
LABELS = {"AC-MPC-MPVE": "MPVE", "AB-PQ": "AB-PQ", "KMPC": "KMPC", "PPO": "PPO"}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step")
        score = row.get("eval/episode_reward")
        if isinstance(step, (int, float)) and isinstance(score, (int, float)):
            rows.append(row)
    return rows


def plot_training(root: Path, output: Path, seed: int) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.ravel()
    for task_index, task in enumerate(TASKS):
        axis = axes_flat[task_index]
        any_curve = False
        for method in METHODS:
            rows = _metrics(
                root / "train" / task / f"seed_{seed}" / method / "metrics.jsonl"
            )
            if not rows:
                continue
            any_curve = True
            steps = np.asarray([float(row["step"]) / 1e6 for row in rows])
            scores = np.asarray([float(row["eval/episode_reward"]) for row in rows])
            deviations = np.asarray(
                [float(row.get("eval/episode_reward_std", 0.0)) for row in rows]
            )
            axis.plot(
                steps,
                scores,
                marker="o",
                linewidth=2,
                markersize=4,
                color=COLORS[method],
                label=LABELS[method],
            )
            axis.fill_between(
                steps,
                scores - deviations,
                scores + deviations,
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
        axis.set_title(task)
        axis.set_xlabel("Environment steps (million)")
        axis.set_ylabel("Evaluation return / 1000")
        axis.grid(alpha=0.25)
        if any_curve:
            axis.legend(fontsize=8, loc="best")
    axes_flat[-1].axis("off")
    figure.suptitle(
        "MuJoCo Playground training evaluation curves (seed 20260812)\n"
        "Shading: evaluator episode standard deviation; Humanoid structured runs not started",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_final(root: Path, output: Path, seed: int) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes_flat = axes.ravel()
    for task_index, task in enumerate(TASKS):
        axis = axes_flat[task_index]
        labels: list[str] = []
        means: list[float] = []
        deviations: list[float] = []
        colors: list[str] = []
        for method in METHODS:
            report = _load_json(
                root
                / "train"
                / task
                / f"seed_{seed}"
                / method
                / "eval_latest_128.json"
            )
            if not report:
                continue
            labels.append(LABELS[method])
            means.append(float(report["return_mean"]))
            deviations.append(float(report["return_std_population"]))
            colors.append(COLORS[method])
        positions = np.arange(len(labels))
        bars = axis.bar(
            positions,
            means,
            yerr=deviations,
            capsize=4,
            color=colors,
            alpha=0.85,
        )
        axis.set_xticks(positions, labels, rotation=18)
        axis.set_title(task)
        axis.set_ylabel("Final deterministic return / 1000")
        axis.grid(axis="y", alpha=0.25)
        for bar, mean in zip(bars, means, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{mean:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes_flat[-1].axis("off")
    figure.suptitle(
        "Final latest-checkpoint evaluation (128 deterministic episodes)\n"
        "Error bars: population standard deviation across evaluation episodes",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/playground"))
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/playground/plots")
    )
    args = parser.parse_args()
    plot_training(
        args.root,
        args.output_dir / "training_returns_seed_20260812.png",
        args.seed,
    )
    plot_final(
        args.root,
        args.output_dir / "final_eval_128_seed_20260812.png",
        args.seed,
    )


if __name__ == "__main__":
    main()
