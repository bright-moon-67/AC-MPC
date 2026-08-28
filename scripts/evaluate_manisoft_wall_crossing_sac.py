#!/usr/bin/env python
"""Deterministically evaluate a wall-crossing SAC checkpoint by reset bin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from antmaze_ac.envs.manisoft_wall_crossing_sac_env import (
    ManiSoftWallCrossingSACEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnormalize", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=820000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _stratified_indices(env: ManiSoftWallCrossingSACEnv, count: int) -> list[int]:
    eligible = env.eligible_snapshot_indices
    cells: dict[tuple[int, float], list[int]] = {}
    for index in eligible:
        key = (
            int(env.snapshot_bank.route_sides[index]),
            round(float(env.snapshot_bank.crossed_fractions[index]), 3),
        )
        cells.setdefault(key, []).append(int(index))
    ordered_cells = sorted(cells)
    cursors = {key: 0 for key in ordered_cells}
    selected: list[int] = []
    while len(selected) < min(count, len(eligible)):
        advanced = False
        for key in ordered_cells:
            cursor = cursors[key]
            if cursor < len(cells[key]) and len(selected) < count:
                selected.append(cells[key][cursor])
                cursors[key] += 1
                advanced = True
        if not advanced:
            break
    return selected


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    run_config_path = Path(args.run_config).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    vecnormalize_path = Path(args.vecnormalize).expanduser().resolve()
    for path in (run_config_path, model_path, vecnormalize_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    run = json.loads(run_config_path.read_text(encoding="utf-8"))

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def make_env():
        return ManiSoftWallCrossingSACEnv(
            run["scenario"],
            task_config_path=run["task_config"],
            snapshot_bank_path=run["snapshot_bank"],
            **run["environment"],
        )

    env = make_env()
    normalization = VecNormalize.load(
        str(vecnormalize_path), DummyVecEnv([make_env])
    )
    normalization.training = False
    normalization.norm_reward = False
    model = SAC.load(str(model_path), device=args.device)
    rows = []
    for episode, index in enumerate(_stratified_indices(env, args.episodes)):
        observation, reset_info = env.reset(
            seed=args.seed + episode, options={"snapshot_index": index}
        )
        episode_return = 0.0
        minimum_wall_clearance = float(reset_info["wall_clearance"])
        minimum_ground_clearance = float(reset_info["ground_clearance"])
        minimum_target_plane_distance = float(
            reset_info["target_plane_distance"]
        )
        maximum_distal_fraction = float(
            reset_info["distal_crossed_fraction"]
        )
        maximum_tip_speed = float(reset_info["tip_speed"])
        while True:
            normalized = normalization.normalize_obs(observation[None, :])[0]
            action, _ = model.predict(normalized, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            minimum_wall_clearance = min(
                minimum_wall_clearance, float(info["wall_clearance"])
            )
            minimum_ground_clearance = min(
                minimum_ground_clearance, float(info["ground_clearance"])
            )
            minimum_target_plane_distance = min(
                minimum_target_plane_distance,
                float(info["target_plane_distance"]),
            )
            maximum_distal_fraction = max(
                maximum_distal_fraction,
                float(info["distal_crossed_fraction"]),
            )
            maximum_tip_speed = max(maximum_tip_speed, float(info["tip_speed"]))
            if terminated or truncated:
                break
        rows.append(
            {
                "snapshot_index": index,
                "side": int(reset_info["route_side"]),
                "source_fraction": float(reset_info["source_crossed_fraction"]),
                "success": bool(info["is_success"]),
                "termination_reason": info["termination_reason"] or "episode_limit",
                "steps": int(env.step_count),
                "return": float(episode_return),
                "final_distal_fraction": float(info["distal_crossed_fraction"]),
                "maximum_distal_fraction": maximum_distal_fraction,
                "final_target_plane_distance": float(
                    info["target_plane_distance"]
                ),
                "minimum_target_plane_distance": minimum_target_plane_distance,
                "maximum_tip_speed": maximum_tip_speed,
                "minimum_wall_clearance": minimum_wall_clearance,
                "minimum_ground_clearance": minimum_ground_clearance,
            }
        )
    env.close()
    normalization.close()
    cells = sorted({(row["side"], row["source_fraction"]) for row in rows})
    summary = {
        "model": str(model_path),
        "episode_count": len(rows),
        "success_count": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_count": sum(
            row["termination_reason"] == "virtual_wall_collision" for row in rows
        ),
        "ground_violation_count": sum(
            row["termination_reason"] == "ground_violation" for row in rows
        ),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_final_target_plane_distance": float(
            np.mean([row["final_target_plane_distance"] for row in rows])
        ),
        "minimum_target_plane_distance": float(
            np.min([row["minimum_target_plane_distance"] for row in rows])
        ),
        "maximum_distal_fraction": float(
            np.max([row["maximum_distal_fraction"] for row in rows])
        ),
        "by_side_and_source_fraction": {
            f"side_{side:+d}_fraction_{fraction:.2f}": {
                "episodes": len(cell_rows),
                "successes": sum(row["success"] for row in cell_rows),
            }
            for side, fraction in cells
            for cell_rows in [
                [
                    row
                    for row in rows
                    if row["side"] == side
                    and np.isclose(row["source_fraction"], fraction)
                ]
            ]
        },
        "episodes": rows,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
