"""Summarize the active formal Hopper Stand matrix and make its plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


METHODS = ("AWAC", "Cal-QL", "Cal-RLPD", "IQL", "RLPD")
SEEDS = tuple(range(20260851, 20260856))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    if not rows:
        raise ValueError(f"empty metrics: {path}")
    return rows


def offline_curve(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    points: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("phase") not in {"initial", "offline_diagnostic", "offline_evaluation"}:
            continue
        if row.get("online_step") != 0:
            continue
        step = row.get("offline_update")
        value = row.get("return_mean")
        if isinstance(step, bool) or not isinstance(step, int) or value is None:
            continue
        if not math.isfinite(float(value)):
            raise ValueError("non-finite offline return")
        points[step] = row
    return [
        {
            "step": float(step),
            "return_mean": float(points[step]["return_mean"]),
            "return_std_population": float(points[step].get("return_std_population", 0.0)),
        }
        for step in sorted(points)
    ]


def online_curve(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    points: dict[int, dict[str, Any]] = {}
    for row in rows:
        phase = row.get("phase")
        step = row.get("online_step")
        if phase in {"initial", "offline_evaluation"} and step == 0:
            points[0] = row
        elif phase == "online_evaluation" and isinstance(step, int) and step > 0:
            points[step] = row
    return [
        {
            "step": float(step),
            "return_mean": float(points[step]["return_mean"]),
            "return_std_population": float(points[step].get("return_std_population", 0.0)),
        }
        for step in sorted(points)
    ]


def stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
        "n": int(len(array)),
    }


def aggregate_curve(curves: list[list[dict[str, float]]]) -> list[dict[str, Any]]:
    grids = [tuple(int(p["step"]) for p in curve) for curve in curves]
    if len(set(grids)) != 1:
        raise ValueError(f"evaluation grids differ: {set(grids)}")
    result = []
    for index, step in enumerate(grids[0]):
        means = [curve[index]["return_mean"] for curve in curves]
        result.append({"step": step, "return": stats(means)})
    return result


def final10x10(run_dir: Path) -> dict[str, Any]:
    result = {}
    for checkpoint in ("online_000000", "online_020000"):
        value = read_json(run_dir / f"evaluation_10x10_{checkpoint}.json")
        result[checkpoint] = {
            "return_mean": float(value["return_mean"]),
            "return_std_population": float(value["return_std_population"]),
            "return_min": float(value["return_min"]),
            "return_max": float(value["return_max"]),
            "total_episodes": int(value["evaluation_protocol"]["total_episodes"]),
        }
    return result


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_seed: dict[str, Any] = {}
    offline_by_method: dict[str, list[list[dict[str, float]]]] = {}
    online_by_method: dict[str, list[list[dict[str, float]]]] = {}
    final_by_method: dict[str, dict[str, Any]] = {}
    run_dirs = []
    for method in METHODS:
        offline_curves = []
        online_curves = []
        method_final: dict[str, Any] = {}
        for seed in SEEDS:
            run_dir = root / f"seed_{seed}" / method
            run = read_json(run_dir / "run.json")
            if run.get("completed") is not True:
                raise ValueError(f"incomplete run: {run_dir}")
            rows = read_metrics(run_dir / "metrics.jsonl")
            off = offline_curve(rows)
            on = online_curve(rows)
            if not off or not on or int(on[-1]["step"]) != 20_000:
                raise ValueError(f"incomplete evaluation curve: {run_dir}")
            offline_curves.append(off)
            online_curves.append(on)
            method_final[str(seed)] = final10x10(run_dir)
            run_dirs.append(str(run_dir))
            per_seed.setdefault(str(seed), {})[method] = {
                "run_dir": str(run_dir),
                "offline_curve": off,
                "online_curve": on,
                "final_10x10": method_final[str(seed)],
            }
        offline_by_method[method] = offline_curves
        online_by_method[method] = online_curves
        final_by_method[method] = method_final

    summary: dict[str, Any] = {
        "kind": "acmpc_hopper_stand_active_matrix_summary_v1",
        "root": str(root),
        "task": "hopper_stand",
        "methods": list(METHODS),
        "training_seeds": list(SEEDS),
        "excluded_from_summary": {
            "old_seed_20260851_methods": ["Cal-RLPD-KMPC", "Cal-RLPD-Lift"],
            "location": str(root.parent / "hopper_stand_proto200k_offline50k_online20k_excluded"),
            "reason": "temporarily moved out by user request",
        },
        "protocol": {
            "offline_updates": 50_000,
            "online_transitions": 20_000,
            "diagnostic_episodes": 10,
            "final_evaluation_episodes": 100,
        },
        "offline_curves": {
            method: aggregate_curve(curves) for method, curves in offline_by_method.items()
        },
        "online_curves": {
            method: aggregate_curve(curves) for method, curves in online_by_method.items()
        },
        "final_10x10": {
            method: {
                checkpoint: stats(
                    [final_by_method[method][str(seed)][checkpoint]["return_mean"] for seed in SEEDS]
                )
                for checkpoint in ("online_000000", "online_020000")
            }
            for method in METHODS
        },
        "per_seed": per_seed,
        "run_dirs": run_dirs,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    offline_rows = []
    for method in METHODS:
        for point in summary["offline_curves"][method]:
            offline_rows.append([method, point["step"], point["return"]["mean"], point["return"]["std_population"]])
    write_csv(output / "offline_curve.csv", ["method", "offline_update", "return_mean_across_seeds", "return_std_across_seeds"], offline_rows)
    online_rows = []
    for method in METHODS:
        for point in summary["online_curves"][method]:
            online_rows.append([method, point["step"], point["return"]["mean"], point["return"]["std_population"]])
    write_csv(output / "online_curve.csv", ["method", "online_step", "return_mean_across_seeds", "return_std_across_seeds"], online_rows)

    table_rows = []
    for method in METHODS:
        for seed in SEEDS:
            values = final_by_method[method][str(seed)]
            table_rows.append([method, seed, values["online_000000"]["return_mean"], values["online_000000"]["return_std_population"], values["online_020000"]["return_mean"], values["online_020000"]["return_std_population"]])
    write_csv(output / "final_10x10_by_seed.csv", ["method", "training_seed", "online0_mean", "online0_population_std", "online20k_mean", "online20k_population_std"], table_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {method: f"C{index}" for index, method in enumerate(METHODS)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    for method in METHODS:
        for axis, key, xlabel in (
            (axes[0], "offline_curves", "Offline gradient updates (thousands)"),
            (axes[1], "online_curves", "Online environment steps (thousands)"),
        ):
            points = summary[key][method]
            x = np.asarray([p["step"] for p in points]) / 1000.0
            y = np.asarray([p["return"]["mean"] for p in points])
            s = np.asarray([p["return"]["std_population"] for p in points])
            axis.plot(x, y, marker="o", linewidth=2, color=colors[method], label=method)
            axis.fill_between(x, np.clip(y - s, 0, 1000), np.clip(y + s, 0, 1000), color=colors[method], alpha=0.12)
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Hopper Stand return")
            axis.set_ylim(0, 1000)
            axis.grid(alpha=0.25)
    axes[0].set_title("Offline diagnostic evaluation")
    axes[1].set_title("Online diagnostic evaluation")
    axes[1].legend(fontsize=8)
    fig.suptitle("Hopper Stand active formal matrix: 5 seeds, mean ± seed population std")
    fig.savefig(output / "offline_online_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "offline_online_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Hopper Stand active formal matrix summary",
        "",
        "The old seed `20260851` KMPC/Lift directories are excluded because they were temporarily moved out by request. The quality-mixed Koopman campaign is a separate experiment and is not merged here.",
        "",
        "## Final 10×10 results",
        "",
        "| Method | Seed | Online 0 | Online 20k |",
        "|---|---:|---:|---:|",
    ]
    for row in table_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]:.2f} ± {row[3]:.2f} | {row[4]:.2f} ± {row[5]:.2f} |")
    lines += ["", "### 5-seed mean of final 10×10 episode means", "", "| Method | Online 0 | Online 20k |", "|---|---:|---:|"]
    for method in METHODS:
        z = summary["final_10x10"][method]
        lines.append(f"| {method} | {z['online_000000']['mean']:.2f} ± {z['online_000000']['std_population']:.2f} | {z['online_020000']['mean']:.2f} ± {z['online_020000']['std_population']:.2f} |")
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(output / "summary.json"), "markdown": str(output / "SUMMARY.md"), "plot": str(output / "offline_online_curves.png")}, indent=2))


if __name__ == "__main__":
    main()
