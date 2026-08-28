#!/usr/bin/env python
"""Evaluate frozen Phase-A crossing followed by a Phase-B SAC controller."""

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
    parser.add_argument("--phase-a-run-config", required=True)
    parser.add_argument("--phase-b-model", required=True)
    parser.add_argument("--phase-b-vecnormalize", required=True)
    parser.add_argument("--switch-fraction", type=float, default=0.40)
    parser.add_argument("--return-tip-x-tolerance", type=float, default=0.20)
    parser.add_argument("--success-streak", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_path = Path(args.phase_a_run_config).expanduser().resolve()
    model_path = Path(args.phase_b_model).expanduser().resolve()
    normalizer_path = Path(args.phase_b_vecnormalize).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (run_path, model_path, normalizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 < args.switch_fraction <= 1:
        raise ValueError("switch-fraction must lie in (0,1]")
    if args.return_tip_x_tolerance <= 0 or args.success_streak < 1:
        raise ValueError("invalid return success settings")

    run = json.loads(run_path.read_text(encoding="utf-8"))
    environment = dict(run["environment"])
    environment.update(
        {
            "success_crossed_fraction": float(args.switch_fraction),
            "success_streak": int(args.success_streak),
            "return_tip_x_tolerance": float(args.return_tip_x_tolerance),
            "return_reward_minimum_fraction": float(args.switch_fraction),
            "base_policy_latch_off_fraction": float(args.switch_fraction),
            "latched_residual_action_scale": 1.0,
        }
    )
    env = ManiSoftWallCrossingSACEnv(
        run["scenario"],
        task_config_path=run["task_config"],
        snapshot_bank_path=run["snapshot_bank"],
        **environment,
    )

    from stable_baselines3 import SAC
    from stable_baselines3.common.save_util import load_from_pkl

    phase_b = SAC.load(str(model_path), device=args.device)
    normalizer = load_from_pkl(normalizer_path)

    def normalize(observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32)
        if bool(normalizer.norm_obs):
            value = np.clip(
                (value - normalizer.obs_rms.mean)
                / np.sqrt(normalizer.obs_rms.var + normalizer.epsilon),
                -normalizer.clip_obs,
                normalizer.clip_obs,
            ).astype(np.float32)
        return value

    rows = []
    for ordinal, index in enumerate(env.eligible_snapshot_indices):
        observation, info = env.reset(
            seed=950000 + ordinal, options={"snapshot_index": int(index)}
        )
        minimum_target_plane_distance = float(info["target_plane_distance"])
        minimum_wall_clearance = float(info["wall_clearance"])
        minimum_ground_clearance = float(info["ground_clearance"])
        maximum_distal_fraction = float(info["distal_crossed_fraction"])
        switch_step = None
        while True:
            use_phase_b = (
                info["distal_crossed_fraction"]
                >= args.switch_fraction - 1e-8
            )
            if use_phase_b:
                if switch_step is None:
                    switch_step = env.step_count
                action, _ = phase_b.predict(
                    normalize(observation), deterministic=True
                )
            else:
                # Before the latch, a zero residual lets the environment's
                # frozen Phase-A controller supply the complete command.
                action = np.zeros(18, dtype=np.float32)
            observation, _, terminated, truncated, info = env.step(action)
            minimum_target_plane_distance = min(
                minimum_target_plane_distance,
                float(info["target_plane_distance"]),
            )
            minimum_wall_clearance = min(
                minimum_wall_clearance, float(info["wall_clearance"])
            )
            minimum_ground_clearance = min(
                minimum_ground_clearance, float(info["ground_clearance"])
            )
            maximum_distal_fraction = max(
                maximum_distal_fraction,
                float(info["distal_crossed_fraction"]),
            )
            if terminated or truncated:
                break
        rows.append(
            {
                "snapshot_index": int(index),
                "success": bool(info["is_success"]),
                "termination_reason": info["termination_reason"]
                or "episode_limit",
                "steps": int(env.step_count),
                "switch_step": switch_step,
                "final_distal_fraction": float(
                    info["distal_crossed_fraction"]
                ),
                "maximum_distal_fraction": maximum_distal_fraction,
                "final_target_plane_distance": float(
                    info["target_plane_distance"]
                ),
                "minimum_target_plane_distance": (
                    minimum_target_plane_distance
                ),
                "minimum_wall_clearance": minimum_wall_clearance,
                "minimum_ground_clearance": minimum_ground_clearance,
            }
        )
    env.close()
    summary = {
        "phase_a_run_config": str(run_path),
        "phase_b_model": str(model_path),
        "phase_b_vecnormalize": str(normalizer_path),
        "switch_fraction": float(args.switch_fraction),
        "return_tip_x_tolerance": float(args.return_tip_x_tolerance),
        "success_streak": int(args.success_streak),
        "episode_count": len(rows),
        "success_count": sum(row["success"] for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "minimum_target_plane_distance": float(
            min(row["minimum_target_plane_distance"] for row in rows)
        ),
        "minimum_wall_clearance": float(
            min(row["minimum_wall_clearance"] for row in rows)
        ),
        "minimum_ground_clearance": float(
            min(row["minimum_ground_clearance"] for row in rows)
        ),
        "episodes": rows,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
