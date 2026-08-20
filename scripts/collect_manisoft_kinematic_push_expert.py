#!/usr/bin/env python
"""Collect fixed-Koopman-MPC demonstrations for the kinematic push task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.control.history_reference_mpc import FixedCostHistoryKoopmanMPC
from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_kinematic_push_env import ManiSoftKinematicPushEnv
from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256
from antmaze_ac.koopman.history_model import HistoryDeepKoopman


TIP_INDICES = (30, 31, 32)


def _device(specification: str) -> torch.device:
    if specification == "auto":
        specification = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(specification)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--state-weight", type=float, default=800.0)
    parser.add_argument("--action-weight", type=float, default=200.0)
    parser.add_argument("--control-weight", type=float, default=1.0)
    parser.add_argument(
        "--tip-axis-scales",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 5.0),
        metavar=("X", "Y", "Z"),
        help="Per-axis tip tracking scales; high Z weight prevents table dives.",
    )
    parser.add_argument(
        "--middle-axis-scales",
        type=float,
        nargs=3,
        default=(3.0, 2.0, 2.0),
        metavar=("X", "Y", "Z"),
        help="Node-14 position scales used to keep the arm body around the box.",
    )
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=0.01,
        help="Maximum per-control-frame change of each expert action component.",
    )
    parser.add_argument("--rollout-noise-std", type=float, default=0.0)
    parser.add_argument("--qp-max-iterations", type=int, default=4000)
    parser.add_argument("--minimum-contact-fraction", type=float, default=0.0)
    parser.add_argument("--minimum-success-fraction", type=float, default=0.0)
    parser.add_argument(
        "--route-sides",
        choices=("both", "left", "right"),
        default="both",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.episodes, args.episode_steps, args.horizon) < 1:
        parser.error("episodes, episode-steps and horizon must be positive")
    if args.rollout_noise_std < 0:
        parser.error("rollout-noise-std must be non-negative")
    if args.max_action_delta <= 0:
        parser.error("max-action-delta must be positive")
    if not (
        0.0 <= args.minimum_contact_fraction <= 1.0
        and 0.0 <= args.minimum_success_fraction <= 1.0
    ):
        parser.error("minimum coverage fractions must lie in [0,1]")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.koopman_checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for name, path in (("Koopman checkpoint", checkpoint), ("scenario", scenario)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    model, payload = load_checkpoint(checkpoint, map_location=device)
    if not isinstance(model, HistoryDeepKoopman):
        raise ValueError("Expert collection requires a history-context Koopman model")
    model = model.to(device).freeze_dynamics()
    state_stats = payload["normalizers"]["state"]
    state_mean = torch.as_tensor(state_stats["mean"], device=device)
    state_std = torch.as_tensor(state_stats["std"], device=device)
    action_low = np.full(model.action_dim, -args.absolute_action_limit)
    action_high = np.full(model.action_dim, args.absolute_action_limit)
    physical_state_scales = np.zeros(model.state_dim, dtype=np.float64)
    physical_state_scales[np.asarray((15, 16, 17))] = args.middle_axis_scales
    physical_state_scales[np.asarray(TIP_INDICES)] = args.tip_axis_scales
    expert = FixedCostHistoryKoopmanMPC(
        model=model,
        state_mean=state_mean,
        state_std=state_std,
        action_low=action_low,
        action_high=action_high,
        horizon=args.horizon,
        state_weight=args.state_weight,
        action_weight=args.action_weight,
        control_weight=args.control_weight,
        physical_state_scales=physical_state_scales,
        qp_max_iterations=args.qp_max_iterations,
    )
    base_env = ManiSoftKinematicPushEnv(
        scenario,
        episode_steps=args.episode_steps,
        absolute_action_limit=args.absolute_action_limit,
    )
    env = HistoryContextTrackingWrapper(
        base_env,
        history_steps=model.history_steps,
        state_mean=state_stats["mean"],
        state_std=state_stats["std"],
        tip_indices=TIP_INDICES,
    )
    rng = np.random.default_rng(args.seed)
    sides = {
        "left": (-1,),
        "right": (1,),
        "both": (-1, 1),
    }[args.route_sides]
    arrays: dict[str, list] = {
        name: []
        for name in (
            "observation",
            "expert_action",
            "applied_action",
            "episode_id",
            "step_index",
            "phase",
            "target_center",
            "goal_center",
            "contact_locked",
            "collision",
            "robot_collision",
            "tip_collision",
            "whole_arm_collision",
            "target_collision",
            "is_success",
            "expert_cost",
            "qp_iterations",
            "route_side",
            "softrobot_positions",
        )
    }
    episode_summaries = []
    try:
        for episode in range(args.episodes):
            route_side = sides[episode % len(sides)]
            observation, _ = env.reset(
                seed=args.seed + episode,
                options={"route_side": route_side},
            )
            warm_start = None
            episode_return = 0.0
            terminated = truncated = False
            info = {}
            for step in range(args.episode_steps):
                state = observation[: model.state_dim]
                context = observation[
                    model.state_dim : model.state_dim + model.context_dim
                ]
                reference_state = state.copy()
                reference_state[np.asarray((15, 16, 17))] = (
                    base_env.task.middle_section_target
                )
                reference_state[np.asarray(TIP_INDICES)] = base_env.active_target_tip
                reference_action = env.previous_action.copy()
                plan = expert.solve(
                    state=state,
                    context=context,
                    reference_state=reference_state,
                    reference_action=reference_action,
                    initial_actions=warm_start,
                )
                planned_action = np.asarray(plan["action"], dtype=np.float32)
                previous_action = env.previous_action.copy()
                expert_action = np.clip(
                    planned_action,
                    previous_action - args.max_action_delta,
                    previous_action + args.max_action_delta,
                ).astype(np.float32)
                rollout_action = expert_action.copy()
                if args.rollout_noise_std:
                    rollout_action += rng.normal(
                        0.0,
                        args.rollout_noise_std,
                        size=model.action_dim,
                    ).astype(np.float32)
                phase_before = int(base_env.task.phase)
                target_before = base_env.target_center.copy()
                goal_before = base_env.goal_center.copy()
                contact_before = bool(base_env.task.contact_locked)
                next_observation, reward, terminated, truncated, info = env.step(
                    rollout_action
                )
                frame = base_env.trajectory_frame()
                for name, value in (
                    ("observation", observation.copy()),
                    ("expert_action", expert_action),
                    ("applied_action", info["applied_action"]),
                    ("episode_id", episode),
                    ("step_index", step),
                    ("phase", phase_before),
                    ("target_center", target_before),
                    ("goal_center", goal_before),
                    ("contact_locked", contact_before),
                    ("collision", info["collision"]),
                    ("robot_collision", info["robot_collision"]),
                    ("tip_collision", info["tip_collision"]),
                    ("whole_arm_collision", info["whole_arm_collision"]),
                    ("target_collision", info["target_collision"]),
                    ("is_success", info["is_success"]),
                    ("expert_cost", plan["cost"]),
                    ("qp_iterations", plan["qp_iterations"]),
                    ("route_side", route_side),
                    ("softrobot_positions", frame["softrobot_positions"]),
                ):
                    arrays[name].append(value)
                planned = np.asarray(plan["actions"], dtype=np.float32)
                warm_start = np.concatenate((planned[1:], planned[-1:]), axis=0)
                observation = next_observation
                episode_return += float(reward)
                if terminated or truncated:
                    break
            summary = {
                "episode": episode,
                "steps": step + 1,
                "return": episode_return,
                "phase": int(base_env.task.phase),
                "route_side": route_side,
                "contact_locked": bool(base_env.task.contact_locked),
                "success": bool(info.get("is_success", False)),
                "collision": bool(info.get("collision", False)),
                "goal_distance": float(info.get("goal_distance", np.nan)),
            }
            episode_summaries.append(summary)
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        env.close()

    typed = {
        "observation": np.float32,
        "expert_action": np.float32,
        "applied_action": np.float32,
        "episode_id": np.int64,
        "step_index": np.int64,
        "phase": np.int64,
        "target_center": np.float32,
        "goal_center": np.float32,
        "contact_locked": np.bool_,
        "collision": np.bool_,
        "robot_collision": np.bool_,
        "tip_collision": np.bool_,
        "whole_arm_collision": np.bool_,
        "target_collision": np.bool_,
        "is_success": np.bool_,
        "expert_cost": np.float32,
        "qp_iterations": np.int32,
        "route_side": np.int8,
        "softrobot_positions": np.float32,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{
                name: np.asarray(values, dtype=typed[name])
                for name, values in arrays.items()
            },
        )
    temporary.replace(output)
    report = {
        "schema_version": 1,
        "kind": "manisoft_kinematic_push_fixed_mpc_expert",
        "samples": len(arrays["observation"]),
        "episodes": args.episodes,
        "successful_episodes": int(sum(row["success"] for row in episode_summaries)),
        "contact_episodes": int(
            sum(row["contact_locked"] for row in episode_summaries)
        ),
        "collision_episodes": int(sum(row["collision"] for row in episode_summaries)),
        "observation_dim": int(env.observation_space.shape[0]),
        "action_dim": int(model.action_dim),
        "history_steps": int(model.history_steps),
        "task_mode": "kinematic_push",
        "koopman_checkpoint": str(checkpoint),
        "koopman_checkpoint_sha256": sha256(checkpoint),
        "scenario": str(scenario),
        "scenario_sha256": sha256(scenario),
        "runtime": vars(args),
        "episode_summaries": episode_summaries,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "episode_summaries"}, indent=2, default=str))
    contact_fraction = report["contact_episodes"] / args.episodes
    success_fraction = report["successful_episodes"] / args.episodes
    if contact_fraction < args.minimum_contact_fraction:
        raise RuntimeError(
            f"Expert contact coverage {contact_fraction:.3f} is below "
            f"{args.minimum_contact_fraction:.3f}; do not start BC"
        )
    if success_fraction < args.minimum_success_fraction:
        raise RuntimeError(
            f"Expert success coverage {success_fraction:.3f} is below "
            f"{args.minimum_success_fraction:.3f}; do not start BC"
        )


if __name__ == "__main__":
    main()
