"""Validate and render one completed formal Walker Run training seed."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dmc.o2o.formal_walker import FORMAL_METHODS
from experiments.dmc.o2o.formal_walker_results import (
    OFFLINE_POINTS,
    ONLINE_POINTS,
    _diagnostic,
    _final_10x10,
    _read_json,
)
from experiments.dmc.o2o.koopman import file_sha256
from experiments.dmc.o2o.plot import COLORS, MAX_EPISODE_RETURN, _save_atomic


REPORT_KIND = "acmpc_walker_formal_single_seed_report_v1"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty formal seed CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _diagnostic_point(run_dir: Path, stage: str, counter: int) -> dict[str, Any]:
    mean = _diagnostic(run_dir, stage, counter)
    stem = f"{stage}_{counter:06d}"
    payload = _read_json(run_dir / f"evaluation_{stem}.json")
    evaluation = payload["evaluation"]
    std = float(evaluation["return_std_population"])
    if not math.isfinite(std) or std < 0.0:
        raise ValueError(f"Invalid diagnostic population std: {run_dir / stem}")
    checkpoint = run_dir / f"{stem}.pt"
    return {
        "counter": counter,
        "return_mean": mean,
        "return_std_population": std,
        "episodes": len(evaluation["returns"]),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": payload["checkpoint_sha256"],
    }


def _final_point(
    run_dir: Path, counter: int, training_seed: int, method: str
) -> dict[str, Any]:
    mean = _final_10x10(run_dir, counter, training_seed, method)
    checkpoint = f"online_{counter:06d}"
    payload = _read_json(run_dir / f"evaluation_10x10_{checkpoint}.json")
    std = float(payload["return_std_population"])
    if not math.isfinite(std) or std < 0.0:
        raise ValueError(f"Invalid final population std: {run_dir / checkpoint}")
    return {
        "online_step": counter,
        "return_mean": mean,
        "return_std_population": std,
        "return_median": float(payload["return_median"]),
        "return_min": float(payload["return_min"]),
        "return_max": float(payload["return_max"]),
        "episodes": len(payload["returns"]),
        "checkpoint_path": payload["checkpoint_path"],
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "evaluation_protocol": payload["evaluation_protocol"],
    }


def build_report(seed_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    seed_dir = seed_dir.resolve()
    protocol = _read_json(seed_dir / "protocol.json")
    training_seed = protocol.get("training_seed")
    if (
        protocol.get("kind") != "acmpc_walker_formal_single_seed_protocol_v1"
        or not isinstance(training_seed, int)
        or protocol.get("methods") != list(FORMAL_METHODS)
    ):
        raise ValueError(f"Invalid formal seed protocol: {seed_dir}")
    dataset = protocol.get("dataset")
    koopman = protocol.get("koopman")
    if not isinstance(dataset, dict) or not isinstance(koopman, dict):
        raise ValueError("Formal dataset/Koopman identity is missing")
    dataset_sha = dataset.get("sha256")
    koopman_sha = koopman.get("sha256")
    if not isinstance(dataset_sha, str) or not isinstance(koopman_sha, str):
        raise ValueError("Formal dataset/Koopman SHA is invalid")

    curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    curve_rows: list[dict[str, Any]] = []
    final: dict[str, dict[str, Any]] = {}
    final_rows: list[dict[str, Any]] = []
    for method in FORMAL_METHODS:
        run_dir = seed_dir / method
        run = _read_json(run_dir / "run.json")
        config = run.get("config")
        expected_offline = 0 if method == "RLPD" else 50_000
        if (
            not isinstance(config, dict)
            or config.get("task") != "walker_run"
            or config.get("method") != method
            or config.get("seed") != training_seed
            or config.get("offline_updates") != 50_000
            or config.get("online_steps") != 20_000
            or config.get("offline_eval_interval_updates") != 5_000
            or config.get("eval_interval_online_steps") != 2_500
            or run.get("completed") is not True
            or run.get("offline_updates_completed") != expected_offline
            or run.get("online_steps_completed") != 20_000
        ):
            raise ValueError(f"Incomplete or incompatible formal run: {run_dir}")
        if run.get("dataset", {}).get("sha256") != dataset_sha:
            raise ValueError(f"Dataset SHA differs: {run_dir}")
        run_koopman = run.get("koopman")
        if method in {"Cal-RLPD-KMPC", "Cal-RLPD-Lift"}:
            if not isinstance(run_koopman, dict) or run_koopman.get("sha256") != koopman_sha:
                raise ValueError(f"Seed-paired Koopman differs: {run_dir}")
        elif run_koopman is not None:
            raise ValueError(f"Baseline unexpectedly loads Koopman: {run_dir}")

        method_curves: dict[str, list[dict[str, Any]]] = {"offline": [], "online": []}
        offline_counters = (0,) if method == "RLPD" else OFFLINE_POINTS
        for counter in offline_counters:
            source_stage = "online" if method == "RLPD" else "offline"
            point = _diagnostic_point(run_dir, source_stage, counter)
            method_curves["offline"].append(point)
            curve_rows.append({"training_seed": training_seed, "method": method, "stage": "offline", **point})
        for counter in ONLINE_POINTS:
            point = _diagnostic_point(run_dir, "online", counter)
            method_curves["online"].append(point)
            curve_rows.append({"training_seed": training_seed, "method": method, "stage": "online", **point})
        curves[method] = method_curves

        online_zero = _final_point(run_dir, 0, training_seed, method)
        online_final = _final_point(run_dir, 20_000, training_seed, method)
        absolute_gain = online_final["return_mean"] - online_zero["return_mean"]
        relative_gain = (
            100.0 * absolute_gain / abs(online_zero["return_mean"])
            if online_zero["return_mean"] != 0.0
            else None
        )
        final[method] = {"online_000000": online_zero, "online_020000": online_final, "absolute_gain": absolute_gain, "relative_gain_percent": relative_gain}
        final_rows.append(
            {
                "training_seed": training_seed,
                "method": method,
                "online_0_mean": online_zero["return_mean"],
                "online_0_std_population": online_zero["return_std_population"],
                "online_20k_mean": online_final["return_mean"],
                "online_20k_std_population": online_final["return_std_population"],
                "absolute_gain": absolute_gain,
                "relative_gain_percent": relative_gain,
                "online_0_checkpoint_sha256": online_zero["checkpoint_sha256"],
                "online_20k_checkpoint_sha256": online_final["checkpoint_sha256"],
                "episodes_per_checkpoint": 100,
            }
        )

    report = {
        "kind": REPORT_KIND,
        "task": "walker_run",
        "training_seed": training_seed,
        "dataset": dataset,
        "koopman": koopman,
        "methods": list(FORMAL_METHODS),
        "curves": curves,
        "final_10x10": final,
        "notes": {
            "curve_evaluation": "10 deterministic episodes per milestone",
            "final_evaluation": "10 evaluation seeds x 10 deterministic episodes",
            "offline_axis": "gradient updates",
            "online_axis": "environment transitions",
        },
    }
    return report, curve_rows, final_rows


def _plot_curves(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for method in FORMAL_METHODS:
        color = COLORS[method]
        for axis, stage in zip(axes, ("offline", "online"), strict=True):
            points = report["curves"][method][stage]
            x = np.asarray([point["counter"] for point in points], dtype=np.float64) / 1000.0
            mean = np.asarray([point["return_mean"] for point in points], dtype=np.float64)
            std = np.asarray([point["return_std_population"] for point in points], dtype=np.float64)
            axis.plot(x, mean, color=color, linewidth=2.0, marker="o", markersize=3.5, label=method)
            axis.fill_between(x, np.clip(mean - std, 0.0, None), np.clip(mean + std, None, MAX_EPISODE_RETURN), color=color, alpha=0.09, linewidth=0)
    axes[0].set_title("Offline deterministic evaluation")
    axes[0].set_xlabel("Offline gradient updates (thousands)")
    axes[1].set_title("Online deterministic evaluation")
    axes[1].set_xlabel("Online environment transitions (thousands)")
    for axis in axes:
        axis.set_ylabel("Walker Run episode return (maximum 1000)")
        axis.set_ylim(0.0, MAX_EPISODE_RETURN * 1.02)
        axis.axhline(MAX_EPISODE_RETURN, color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5, loc="best")
    figure.suptitle(
        f"Walker Run offline-to-online evaluation — training seed {report['training_seed']}\n"
        "Each milestone uses 10 deterministic episodes; shading is episode population std",
        fontsize=13,
    )
    png = output_dir / "offline_online_return_curves.png"
    pdf = output_dir / "offline_online_return_curves.pdf"
    _save_atomic(figure, png, image_format="png")
    _save_atomic(figure, pdf, image_format="pdf")
    plt.close(figure)
    return png, pdf


def _markdown_table(report: dict[str, Any]) -> str:
    lines = [
        f"# Walker Run seed {report['training_seed']} final 10×10 evaluation",
        "",
        "| Method | Online 0 | Online 20k | Absolute gain | Relative gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in FORMAL_METHODS:
        value = report["final_10x10"][method]
        start = value["online_000000"]
        end = value["online_020000"]
        relative = value["relative_gain_percent"]
        relative_text = "n/a" if relative is None else f"{relative:.2f}%"
        lines.append(
            f"| {method} | {start['return_mean']:.2f} ± {start['return_std_population']:.2f} "
            f"| {end['return_mean']:.2f} ± {end['return_std_population']:.2f} "
            f"| {value['absolute_gain']:+.2f} | {relative_text} |"
        )
    lines.extend(
        [
            "",
            "Each checkpoint is evaluated with 10 evaluation seeds × 10 deterministic episodes. ",
            "Checkpoint paths and SHA256 identities are stored in `seed_report.json` and `final_10x10.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(seed_dir: Path, output_dir: Path) -> dict[str, str]:
    report, curve_rows, final_rows = build_report(seed_dir)
    output_dir = output_dir.resolve()
    report_json = output_dir / "seed_report.json"
    curves_csv = output_dir / "offline_online_return_curves.csv"
    final_csv = output_dir / "final_10x10.csv"
    final_md = output_dir / "final_10x10.md"
    _atomic_json(report_json, report)
    _atomic_csv(curves_csv, curve_rows)
    _atomic_csv(final_csv, final_rows)
    _atomic_text(final_md, _markdown_table(report))
    png, pdf = _plot_curves(report, output_dir)
    return {"report_json": str(report_json), "curves_csv": str(curves_csv), "curves_png": str(png), "curves_pdf": str(pdf), "final_csv": str(final_csv), "final_markdown": str(final_md)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(args.seed_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
