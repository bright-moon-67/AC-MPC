"""Plot complete offline-to-online evaluation curves from O2O run metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            values.append(json.loads(line))
    if not values:
        raise ValueError(f"No metrics in {path}")
    return values


def _curve(rows: list[dict[str, Any]], phase: str, x_key: str) -> list[dict[str, float]]:
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("phase") != phase:
            continue
        x = row.get(x_key)
        value = row.get("return_mean")
        if isinstance(x, bool) or not isinstance(x, int) or value is None:
            continue
        if not np.isfinite(float(value)):
            raise ValueError(f"Non-finite return in {phase}")
        selected[int(x)] = row
    if not selected:
        raise ValueError(f"No {phase} rows")
    return [
        {"step": float(step), "return_mean": float(selected[step]["return_mean"])}
        for step in sorted(selected)
    ]


def _offline_curve(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Combine initial, periodic offline diagnostics, and final offline eval."""
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        phase = row.get("phase")
        if phase not in ("initial", "offline_diagnostic", "offline_evaluation"):
            continue
        step = row.get("offline_update")
        value = row.get("return_mean")
        if isinstance(step, bool) or not isinstance(step, int) or value is None:
            continue
        if not np.isfinite(float(value)):
            raise ValueError("Non-finite offline return")
        selected[int(step)] = row
    if not selected:
        raise ValueError("No offline/initial evaluation rows")
    return [
        {"step": float(step), "return_mean": float(selected[step]["return_mean"])}
        for step in sorted(selected)
    ]


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", action="append", required=True,
        help="METHOD=run directory; may be repeated",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    runs: dict[str, Path] = {}
    for item in args.run_dir:
        if "=" not in item:
            raise ValueError("--run-dir must use METHOD=PATH")
        method, value = item.split("=", 1)
        if not method or method in runs:
            raise ValueError("Duplicate or empty method name")
        runs[method] = Path(value).resolve()

    report: dict[str, Any] = {"kind": "acmpc_dmc_o2o_phase_curves_v1", "methods": {}}
    for method, run_dir in sorted(runs.items()):
        rows = _rows(run_dir)
        offline = _offline_curve(rows)
        online = _curve(rows, "online_evaluation", "online_step")
        report["methods"][method] = {
            "run_dir": str(run_dir),
            "offline_evaluation": offline,
            "online_evaluation": online,
            "final_online": online[-1],
            "final_offline": offline[-1],
        }

    prefix = args.output_prefix.resolve()
    json_path = prefix.with_suffix(".json")
    _atomic_json(json_path, report)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [f"C{i}" for i in range(len(report["methods"]))]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for color, (method, value) in zip(colors, sorted(report["methods"].items())):
        offline = value["offline_evaluation"]
        online = value["online_evaluation"]
        axes[0].plot(
            [p["step"] / 1000.0 for p in offline],
            [p["return_mean"] for p in offline],
            marker="o", linewidth=2, color=color, label=method,
        )
        axes[1].plot(
            [p["step"] / 1000.0 for p in online],
            [p["return_mean"] for p in online],
            marker="o", linewidth=2, color=color, label=method,
        )
    axes[0].set_title("Offline evaluation")
    axes[0].set_xlabel("Offline gradient updates (thousands)")
    axes[1].set_title("Online evaluation")
    axes[1].set_xlabel("Online environment steps (thousands)")
    for axis in axes:
        axis.set_ylabel("Deterministic return (10 episodes)")
        axis.set_ylim(0, 1000)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("DMC Cartpole Swingup: complete offline → online evaluation")
    png_path = prefix.with_suffix(".png")
    pdf_path = prefix.with_suffix(".pdf")
    for path, fmt in ((png_path, "png"), (pdf_path, "pdf")):
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        figure.savefig(temporary, format=fmt, dpi=180 if fmt == "png" else None,
                       bbox_inches="tight")
        os.replace(temporary, path)
    plt.close(figure)
    print(json.dumps({"json": str(json_path), "png": str(png_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
    main()
