"""Plot cross-method O2O training and fixed-seed evaluation curves."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from experiments.dmc.o2o.aggregate import AGGREGATE_KIND, MAX_EPISODE_RETURN


COLORS = {
    "Cal-QL-Raw": "#4c78a8",
    "Cal-QL": "#4c78a8",
    "Cal-QL-AC-KMPC": "#b279a2",
    "RLPD-Raw": "#72b7b2",
    "RLPD": "#72b7b2",
    "Cal-RLPD-Raw": "#54a24b",
    "Cal-RLPD": "#54a24b",
    "Cal-QL-KMPC": "#f58518",
    "Cal-RLPD-KMPC": "#b279a2",
    "Cal-QL-MPVE": "#e45756",
    "Cal-RLPD-MPVE": "#8c564b",
    # Historical internal names for the same structured methods.
    "Cal-RLPD-AC-KMPC": "#f58518",
    "Cal-RLPD-AC-KMPC-MPVE": "#e45756",
    "Cal-RLPD-Lift": "#ff9da6",
    "AWAC": "#9c755f",
    "IQL": "#bab0ac",
}


def _read_aggregate(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid aggregate JSON: {path}") from exc
    if not isinstance(value, dict) or value.get("kind") != AGGREGATE_KIND:
        raise ValueError("Unsupported O2O aggregate")
    methods = value.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Aggregate contains no methods")
    return value


def _curve_arrays(points: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    steps = np.asarray([point["online_step"] for point in points], dtype=np.float64)
    means = np.asarray([point["return"]["mean"] for point in points], dtype=np.float64)
    lower = []
    upper = []
    for point, mean in zip(points, means, strict=True):
        half = point["return"].get("ci95_half_width")
        half = 0.0 if half is None else float(half)
        lower.append(max(0.0, mean - half))
        upper.append(min(MAX_EPISODE_RETURN, mean + half))
    return steps, means, np.asarray(lower), np.asarray(upper)


def _save_atomic(figure: Any, path: Path, *, image_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        figure.savefig(
            temporary,
            format=image_format,
            dpi=180 if image_format == "png" else None,
            bbox_inches="tight",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# Phases that carry a fixed-seed offline diagnostic evaluation.  The initial
# step-0 row is `initial`, the intermediate 10k/20k/... rows are
# `offline_diagnostic`, and the final approved-budget row is
# `offline_evaluation`; all of them keep `online_step == 0`.
OFFLINE_PHASES = ("initial", "offline_diagnostic", "offline_evaluation")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Metrics row {path}:{line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"Metrics file is empty: {path}")
    return rows


def _display_method(raw: str) -> str:
    """Normalize historical internal names for display.

    Older checkpoints keep the `AC` marker (e.g. `Cal-QL-AC-KMPC`); the
    paper plots drop it so the structured methods read `Cal-QL-KMPC` /
    `Cal-RLPD-KMPC` / `*-MPVE`.
    """

    aliases = {
        "Cal-QL-Raw": "Cal-QL",
        "RLPD-Raw": "RLPD",
        "Cal-RLPD-Raw": "Cal-RLPD",
    }
    return aliases.get(raw, raw).replace("-AC-KMPC", "-KMPC")


def _method_name(run_dir: Path) -> str:
    """Resolve the display method label from run.json, falling back to folder name."""

    run_path = run_dir / "run.json"
    if run_path.is_file():
        value = _read_json(run_path)
        config = value.get("config")
        if isinstance(config, dict) and isinstance(config.get("method"), str):
            return _display_method(config["method"])
        if isinstance(value.get("method"), str):
            return _display_method(value["method"])
    return _display_method(run_dir.name)


def _offline_curve(run_dir: Path) -> list[dict[str, Any]]:
    """Collect existing offline diagnostic points, ordered by gradient updates.

    Only rows that actually exist in metrics.jsonl are kept; missing points
    (for example a step-0 row that was never persisted) are skipped without
    interpolation or zero-filling.
    """

    rows = _read_jsonl(run_dir / "metrics.jsonl")
    points: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("phase") not in OFFLINE_PHASES:
            continue
        if row.get("online_step") != 0:
            continue
        offline_update = row.get("offline_update")
        if (
            isinstance(offline_update, bool)
            or not isinstance(offline_update, int)
            or offline_update < 0
        ):
            raise ValueError(f"Invalid offline_update in {run_dir}/metrics.jsonl")
        value = row.get("return_mean")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Invalid return_mean in {run_dir}/metrics.jsonl")
        std = row.get("return_std_population")
        std = 0.0 if std is None else float(std)
        returns = row.get("returns")
        point = {
            "offline_update": offline_update,
            "return_mean": float(value),
            "return_std_population": std,
            "returns": returns if isinstance(returns, list) else [],
        }
        if offline_update in points:
            raise ValueError(
                f"Duplicate offline evaluation at {offline_update} in {run_dir}"
            )
        points[offline_update] = point
    if not points:
        raise ValueError(f"No offline evaluation rows in {run_dir}/metrics.jsonl")
    return [points[step] for step in sorted(points)]


def _write_offline_csv(curves: Mapping[str, list[dict[str, Any]]], path: Path) -> None:
    """Atomically persist the raw points used for the offline plot as CSV."""

    lines = ["method,offline_update,return_mean,return_std_population"]
    for method in sorted(curves):
        for point in curves[method]:
            lines.append(
                f"{method},{point['offline_update']},{point['return_mean']!r},"
                f"{point['return_std_population']!r}"
            )
    payload = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _online_curves(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read raw training episodes and fixed-seed online evaluations."""

    rows = _read_jsonl(run_dir / "metrics.jsonl")
    training: list[dict[str, Any]] = []
    evaluations: dict[int, dict[str, Any]] = {}
    for row in rows:
        phase = row.get("phase")
        step = row.get("online_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            continue
        if phase == "online_episode":
            value = row.get("episode_return")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                training.append(
                    {
                        "online_step": step,
                        "return_mean": float(value),
                        "return_std_population": 0.0,
                    }
                )
        elif (
            (phase in ("initial", "offline_evaluation") and step == 0)
            or (phase == "online_evaluation" and step > 0)
        ):
            value = row.get("return_mean")
            std = row.get("return_std_population", 0.0)
            if (
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and isinstance(std, (int, float))
                and math.isfinite(float(std))
            ):
                # A resumed run may repeat an idempotent evaluation.  The
                # latest durable row at a step is authoritative.
                evaluations[step] = {
                    "online_step": step,
                    "return_mean": float(value),
                    "return_std_population": float(std),
                }
    if not training and not evaluations:
        raise ValueError(f"No online rows in {run_dir}/metrics.jsonl")
    return {
        "training": training,
        "evaluation": [evaluations[step] for step in sorted(evaluations)],
    }


def _write_online_csv(
    curves: Mapping[str, dict[str, list[dict[str, Any]]]], path: Path
) -> None:
    lines = ["series,method,online_step,return_mean,return_std_population"]
    for method in sorted(curves):
        for series in ("training", "evaluation"):
            for point in curves[method][series]:
                lines.append(
                    f"{series},{method},{point['online_step']},"
                    f"{point['return_mean']!r},"
                    f"{point['return_std_population']!r}"
                )
    payload = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def plot_online_run_curves(
    run_dirs: Iterable[Path],
    output_prefix: Path,
    *,
    task_label: str,
    protocol_label: str,
) -> tuple[Path, Path]:
    """Plot single-training-seed online curves directly from run artifacts."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved = [Path(path).resolve() for path in run_dirs]
    if not resolved:
        raise ValueError("At least one online run directory is required")
    curves: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for run_dir in resolved:
        method = _method_name(run_dir)
        if method in curves:
            raise ValueError(f"Duplicate method {method}")
        curves[method] = _online_curves(run_dir)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    training_axis, evaluation_axis = axes
    for index, (method, method_curves) in enumerate(sorted(curves.items())):
        color = COLORS.get(method, f"C{index % 10}")
        training = method_curves["training"]
        if training:
            x = np.asarray([point["online_step"] for point in training]) / 1000.0
            y = np.asarray([point["return_mean"] for point in training])
            training_axis.plot(
                x, y, color=color, linewidth=1.5, marker="o", markersize=3,
                label=method,
            )
        evaluation = method_curves["evaluation"]
        if evaluation:
            x = np.asarray([point["online_step"] for point in evaluation]) / 1000.0
            y = np.asarray([point["return_mean"] for point in evaluation])
            std = np.asarray(
                [point["return_std_population"] for point in evaluation]
            )
            evaluation_axis.plot(
                x, y, color=color, linewidth=2, marker="o", markersize=3.5,
                label=method,
            )
            evaluation_axis.fill_between(
                x,
                np.clip(y - std, 0.0, None),
                np.clip(y + std, None, MAX_EPISODE_RETURN),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
    training_axis.set_title("Online training episode return")
    evaluation_axis.set_title("Deterministic evaluation return (10 episodes)")
    for axis in axes:
        axis.set_xlabel("Online environment steps (thousands)")
        axis.set_ylabel(f"{task_label} episode return (maximum 1000)")
        axis.set_ylim(0.0, MAX_EPISODE_RETURN * 1.02)
        axis.axhline(MAX_EPISODE_RETURN, color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        f"DMC {task_label} online learning\n"
        f"{protocol_label}; single training seed; evaluation shading is population std",
        fontsize=13,
    )
    output_prefix = output_prefix.resolve()
    png_path = output_prefix.with_suffix(".png")
    csv_path = output_prefix.with_suffix(".csv")
    _save_atomic(figure, png_path, image_format="png")
    plt.close(figure)
    _write_online_csv(curves, csv_path)
    return png_path, csv_path


def plot_aggregate(aggregate_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = _read_aggregate(aggregate_path)
    online_budget = int(aggregate["protocol"]["primary_online_budget"])
    early_step = int(aggregate["protocol"]["early_auc_diagnostic_step"])
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    training_axis, evaluation_axis = axes
    for index, (method, summary) in enumerate(sorted(aggregate["methods"].items())):
        color = COLORS.get(method, f"C{index % 10}")
        training = summary.get("training_curve", [])
        if training:
            steps, means, lower, upper = _curve_arrays(training)
            training_axis.plot(
                steps / 1000.0, means, color=color, linewidth=1.8, label=method
            )
            training_axis.fill_between(
                steps / 1000.0, lower, upper, color=color, alpha=0.12, linewidth=0
            )
        evaluation = summary.get("evaluation_curve", [])
        if evaluation:
            steps, means, lower, upper = _curve_arrays(evaluation)
            evaluation_axis.plot(
                steps / 1000.0,
                means,
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=3.5,
                label=method,
            )
            evaluation_axis.fill_between(
                steps / 1000.0, lower, upper, color=color, alpha=0.15, linewidth=0
            )

    training_axis.set_title("Online training episode return")
    evaluation_axis.set_title("Deterministic evaluation return (10 episodes)")
    for axis in axes:
        axis.set_xlabel("Online environment steps (thousands)")
        axis.set_ylabel("Cartpole Swingup episode return (maximum 1000)")
        axis.set_ylim(0.0, MAX_EPISODE_RETURN * 1.02)
        axis.axhline(MAX_EPISODE_RETURN, color="black", linestyle=":", linewidth=1)
        if early_step < online_budget:
            axis.axvline(
                early_step / 1000.0,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.65,
            )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        "DMC Cartpole offline-to-online learning\n"
        f"common online budget={online_budget / 1000:g}k; lines are training-seed "
        "means and shading is 95% Student-t CI",
        fontsize=13,
    )

    output_prefix = output_prefix.resolve()
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    _save_atomic(figure, png_path, image_format="png")
    _save_atomic(figure, pdf_path, image_format="pdf")
    plt.close(figure)
    return png_path, pdf_path


def plot_offline_curves(
    run_dirs: Iterable[Path],
    output_prefix: Path,
    *,
    task_label: str = "Cartpole Swingup",
    protocol_label: str = "dataset-specific offline protocol",
) -> tuple[Path, Path]:
    """Plot offline return vs gradient updates and export the raw points as CSV.

    This is the offline-only round entry: it recursively discovers each
    method's metrics.jsonl, draws only the evaluation rows that actually
    exist (no zero-fill or interpolation), and delivers PNG plus CSV (the
    CSV stores the exact points used for drawing).
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved = [Path(path).resolve() for path in run_dirs]
    if not resolved:
        raise ValueError("At least one run directory is required")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Run directories must be unique")
    curves: dict[str, list[dict[str, Any]]] = {}
    for run_dir in resolved:
        method = _method_name(run_dir)
        if method in curves:
            raise ValueError(f"Duplicate method {method}")
        curves[method] = _offline_curve(run_dir)

    figure, axis = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    for index, (method, points) in enumerate(sorted(curves.items())):
        color = COLORS.get(method, f"C{index % 10}")
        steps = np.asarray(
            [point["offline_update"] for point in points], dtype=np.float64
        )
        means = np.asarray(
            [point["return_mean"] for point in points], dtype=np.float64
        )
        stds = np.asarray(
            [point["return_std_population"] for point in points], dtype=np.float64
        )
        axis.plot(
            steps / 1000.0,
            means,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label=method,
        )
        axis.fill_between(
            steps / 1000.0,
            np.clip(means - stds, 0.0, None),
            np.clip(means + stds, None, MAX_EPISODE_RETURN),
            color=color,
            alpha=0.12,
            linewidth=0,
        )
    axis.set_title("Offline diagnostic return (10 episodes)")
    axis.set_xlabel("Offline gradient updates (thousands)")
    axis.set_ylabel(f"{task_label} episode return (maximum 1000)")
    axis.set_ylim(0.0, MAX_EPISODE_RETURN * 1.02)
    axis.axhline(MAX_EPISODE_RETURN, color="black", linestyle=":", linewidth=1)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        f"DMC {task_label} offline-only development\n"
        f"({protocol_label}; shading is 10-episode population std)",
        fontsize=13,
    )

    output_prefix = output_prefix.resolve()
    png_path = output_prefix.with_suffix(".png")
    csv_path = output_prefix.with_suffix(".csv")
    _save_atomic(figure, png_path, image_format="png")
    plt.close(figure)
    _write_offline_csv(curves, csv_path)
    return png_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--aggregate", type=Path, help="online aggregate JSON to plot"
    )
    mode.add_argument(
        "--offline-root",
        type=Path,
        help="root that recursively contains offline-only method runs",
    )
    mode.add_argument(
        "--online-root",
        type=Path,
        help="root that recursively contains online method runs",
    )
    parser.add_argument(
        "--offline-run-dir",
        type=Path,
        action="append",
        default=[],
        help="explicit offline-only run directory (repeatable)",
    )
    parser.add_argument(
        "--online-run-dir",
        type=Path,
        action="append",
        default=[],
        help="explicit online run directory (repeatable)",
    )
    parser.add_argument(
        "--task-label",
        default="Cartpole Swingup",
        help="Task label used in the offline plot title and y-axis",
    )
    parser.add_argument(
        "--protocol-label",
        default="dataset-specific offline protocol",
        help="Dataset/update protocol label used in the offline plot subtitle",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.aggregate is not None:
        png_path, pdf_path = plot_aggregate(args.aggregate, args.output_prefix)
        print(json.dumps({"png": str(png_path), "pdf": str(pdf_path)}, indent=2))
        return
    if args.online_root is not None:
        run_dirs = list(args.online_run_dir)
        run_dirs.extend(
            sorted(path.parent for path in args.online_root.rglob("run.json"))
        )
        png_path, csv_path = plot_online_run_curves(
            run_dirs,
            args.output_prefix,
            task_label=args.task_label,
            protocol_label=args.protocol_label,
        )
        print(json.dumps({"png": str(png_path), "csv": str(csv_path)}, indent=2))
        return
    run_dirs = list(args.offline_run_dir)
    if args.offline_root is not None:
        run_dirs.extend(
            sorted(path.parent for path in args.offline_root.rglob("run.json"))
        )
    png_path, csv_path = plot_offline_curves(
        run_dirs,
        args.output_prefix,
        task_label=args.task_label,
        protocol_label=args.protocol_label,
    )
    print(json.dumps({"png": str(png_path), "csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
