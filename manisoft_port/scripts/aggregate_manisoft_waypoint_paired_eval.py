#!/usr/bin/env python
"""Aggregate matched-seed ManiSoft waypoint evaluations.

Each variant directory must contain JSON reports emitted by
``evaluate_manisoft_waypoint_sac.py``.  Pairing by episode seed makes small
policy changes much easier to distinguish from path/reset variation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Directory containing one subdirectory per variant")
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="Variant subdirectory names, including the baseline",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_variant(root: Path, name: str) -> dict[int, dict[str, Any]]:
    reports = sorted((root / name).glob("*.json"))
    if not reports:
        raise FileNotFoundError(f"no JSON reports found for variant {name!r}")
    episodes: dict[int, dict[str, Any]] = {}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        for episode in report["episodes"]:
            seed = int(episode["seed"])
            if seed in episodes:
                raise ValueError(f"duplicate seed {seed} in variant {name!r}")
            episodes[seed] = episode
    return episodes


def exact_mcnemar_p_value(gains: int, losses: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant outcomes."""

    discordant = gains + losses
    if discordant == 0:
        return 1.0
    smaller = min(gains, losses)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in episodes if bool(row["success"])]
    failures = [row for row in episodes if not bool(row["success"])]

    def mean(key: str, rows: list[dict[str, Any]] = episodes) -> float | None:
        return None if not rows else float(fmean(float(row[key]) for row in rows))

    entries = sorted({int(row["entry_index"]) for row in episodes})
    speeds = sorted({float(row["desired_speed"]) for row in episodes})
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "success_rate": len(successes) / len(episodes),
        "mean_rmse_distance_m": mean("rmse_distance"),
        "successful_mean_rmse_distance_m": mean("rmse_distance", successes),
        "mean_cross_track_distance_m": mean("mean_cross_track_distance"),
        "successful_mean_cross_track_distance_m": mean(
            "mean_cross_track_distance", successes
        ),
        "mean_final_progress": mean("final_progress"),
        "mean_internal_waypoints_completed": mean(
            "internal_waypoints_completed"
        ),
        "mean_return": mean("return"),
        "mean_steps": mean("steps"),
        "table_violations": sum(bool(row["table_violation"]) for row in episodes),
        "dynamics_violations": sum(
            bool(row["dynamics_violation"]) for row in episodes
        ),
        "terminal_timeouts": sum(
            bool(row["terminal_timeout"]) for row in episodes
        ),
        "entry_success": {
            str(entry): {
                "successes": sum(
                    bool(row["success"])
                    for row in episodes
                    if int(row["entry_index"]) == entry
                ),
                "episodes": sum(
                    int(row["entry_index"]) == entry for row in episodes
                ),
            }
            for entry in entries
        },
        "speed_success": {
            str(speed): {
                "successes": sum(
                    bool(row["success"])
                    for row in episodes
                    if float(row["desired_speed"]) == speed
                ),
                "episodes": sum(
                    float(row["desired_speed"]) == speed for row in episodes
                ),
            }
            for speed in speeds
        },
        "failure_internal_waypoint_histogram": {
            str(completed): sum(
                int(row["internal_waypoints_completed"]) == completed
                for row in failures
            )
            for completed in sorted(
                {int(row["internal_waypoints_completed"]) for row in failures}
            )
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    names = list(dict.fromkeys(args.variants))
    if args.baseline not in names:
        raise ValueError("baseline must be present in --variants")
    loaded = {name: load_variant(root, name) for name in names}
    common_seeds = sorted(set.intersection(*(set(rows) for rows in loaded.values())))
    if not common_seeds:
        raise ValueError("variants have no common episode seeds")
    for name, rows in loaded.items():
        missing = sorted(set(common_seeds) - set(rows))
        if missing:
            raise ValueError(f"variant {name!r} is missing common seeds: {missing}")
    baseline = loaded[args.baseline]
    variants: dict[str, Any] = {}
    for name in names:
        episodes = [loaded[name][seed] for seed in common_seeds]
        result = summarize(episodes)
        if name != args.baseline:
            gains = [
                seed
                for seed in common_seeds
                if bool(loaded[name][seed]["success"])
                and not bool(baseline[seed]["success"])
            ]
            losses = [
                seed
                for seed in common_seeds
                if bool(baseline[seed]["success"])
                and not bool(loaded[name][seed]["success"])
            ]
            rmse_deltas = [
                float(loaded[name][seed]["rmse_distance"])
                - float(baseline[seed]["rmse_distance"])
                for seed in common_seeds
            ]
            result["paired_vs_baseline"] = {
                "success_gains": len(gains),
                "success_losses": len(losses),
                "net_success_gain": len(gains) - len(losses),
                "gain_seeds": gains,
                "loss_seeds": losses,
                "exact_mcnemar_p_value": exact_mcnemar_p_value(
                    len(gains), len(losses)
                ),
                "mean_rmse_delta_m": float(fmean(rmse_deltas)),
                "median_rmse_delta_m": float(median(rmse_deltas)),
            }
        variants[name] = result
    output = {
        "kind": "manisoft_waypoint_matched_seed_evaluation_aggregate",
        "root": str(root),
        "baseline": args.baseline,
        "common_episode_count": len(common_seeds),
        "seed_min": min(common_seeds),
        "seed_max": max(common_seeds),
        "variants": variants,
    }
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
