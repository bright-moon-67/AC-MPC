#!/usr/bin/env python
"""Aggregate formal 100-episode evaluations across seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "success_mean",
    "return_mean",
    "d4rl_normalized_score",
    "length_mean",
    "delta_action_energy_mean",
    "action_energy_mean",
    "saturation_rate_mean",
    "dare_residual_max",
    "dare_relative_residual_max",
    "dare_condition_max",
    "closed_loop_spectral_radius_max",
    "dare_failure_count",
    "dare_retry_count",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluations", nargs="+")
    parser.add_argument("--minimum-episodes", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.evaluations
    ]
    if len(reports) < 5:
        raise ValueError(f"Formal aggregate requires at least 5 seeds, got {len(reports)}")
    methods = {report["method"] for report in reports}
    if len(methods) != 1:
        raise ValueError(f"Cannot aggregate mixed methods: {sorted(methods)}")
    training_seeds = [report.get("training_seed") for report in reports]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError(f"Training seeds are not unique: {training_seeds}")
    for report in reports:
        if int(report["episodes"]) < args.minimum_episodes:
            raise ValueError(
                f"{report['checkpoint']} has only {report['episodes']} evaluation episodes"
            )

    aggregate = {}
    for metric in METRICS:
        values = [
            float(report[metric])
            for report in reports
            if report.get(metric) is not None
        ]
        aggregate[metric] = (
            {
                "mean_across_seeds": float(np.mean(values)),
                "std_across_seeds": float(np.std(values)),
                "per_seed": values,
            }
            if values
            else None
        )
    output = {
        "method": reports[0]["method"],
        "seed_count": len(reports),
        "training_seeds": training_seeds,
        "episodes_per_seed": [report["episodes"] for report in reports],
        "evaluation_files": [str(Path(path).resolve()) for path in args.evaluations],
        "aggregate": aggregate,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
