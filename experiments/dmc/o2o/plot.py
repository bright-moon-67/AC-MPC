"""Plot cross-method O2O training and fixed-seed evaluation curves."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dmc.o2o.aggregate import AGGREGATE_KIND, MAX_EPISODE_RETURN


COLORS = {
    "REDQ-Online": "#4c78a8",
    "RLPD-MLP": "#72b7b2",
    "Cal-RLPD-MLP": "#54a24b",
    "Cal-RLPD-AC-KMPC": "#f58518",
    "Cal-RLPD-AC-KMPC-MPVE": "#e45756",
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


def plot_aggregate(aggregate_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregate = _read_aggregate(aggregate_path)
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
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        "DMC Cartpole offline-to-online learning\n"
        "Lines are means across training seeds; shading is 95% Student-t CI over seeds",
        fontsize=13,
    )

    output_prefix = output_prefix.resolve()
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    _save_atomic(figure, png_path, image_format="png")
    _save_atomic(figure, pdf_path, image_format="pdf")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    png_path, pdf_path = plot_aggregate(args.aggregate, args.output_prefix)
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
