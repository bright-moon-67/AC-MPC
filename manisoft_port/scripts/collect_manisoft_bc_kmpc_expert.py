#!/usr/bin/env python
"""Collect history-MPC demonstrations for soft-robot BC-KMPC pretraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.control.history_reference_mpc import FixedCostHistoryKoopmanMPC
from antmaze_ac.envs.history_context_wrapper import HistoryContextTrackingWrapper
from antmaze_ac.envs.manisoft_tracking_env import (
    ManiSoftThreeWaypointTrackingEnv,
    load_manisoft_waypoint_reference_bank,
)
from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256
from antmaze_ac.koopman.history_model import HistoryDeepKoopman
from antmaze_ac.rl.serialization import load_history_mpc_checkpoint


TIP_INDICES = (30, 31, 32)


def _minimum_jerk(alpha: np.ndarray) -> np.ndarray:
    """Match the certified waypoint-bank action ramp."""

    return 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5


def _start_stage_schedule(
    episodes: int,
    start_stages: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    cycles = [
        rng.permutation(start_stages)
        for _ in range((episodes + len(start_stages) - 1) // len(start_stages))
    ]
    return np.concatenate(cycles)[:episodes].astype(np.int64, copy=False)


def _warm_start_stage(
    env: HistoryContextTrackingWrapper,
    base_env: ManiSoftThreeWaypointTrackingEnv,
    *,
    start_stage: int,
    equilibrium_action: np.ndarray,
    reference_state: np.ndarray,
    ramp_steps: int,
    hold_steps: int,
) -> tuple[np.ndarray, float]:
    """Reach a certified physical equilibrium before starting DAgger.

    A compact 45-D waypoint state cannot be teleported into the simulator.
    Replaying its certified action with the original minimum-jerk ramp gives
    a dynamically valid full simulator state and a real Koopman history.
    Warm-up steps are excluded from the collected episode.
    """

    if start_stage == 0:
        state = base_env._physical_state()
        return env._observation(state), 0.0
    if start_stage not in (1, 2):
        raise ValueError("start_stage must be 0, 1 or 2")

    action = np.asarray(equilibrium_action, dtype=np.float32).reshape(-1)
    ramp = _minimum_jerk(np.linspace(0.0, 1.0, ramp_steps))
    observation = None
    for alpha in ramp:
        observation, _, _, _, _ = env.step(action * np.float32(alpha))
    for _ in range(hold_steps):
        observation, _, _, _, _ = env.step(action)
    if observation is None:
        raise RuntimeError("warm start did not execute any simulator step")

    physical_state = np.asarray(
        observation[: base_env.observation_space.shape[0]],
        dtype=np.float32,
    )
    reference_tip = np.asarray(reference_state, dtype=np.float32)[
        np.asarray(TIP_INDICES)
    ]
    tip_error = float(
        np.linalg.norm(physical_state[np.asarray(TIP_INDICES)] - reference_tip)
    )

    base_env.active_waypoint_index = int(start_stage)
    base_env.waypoints_completed = int(start_stage)
    base_env.target_tip = base_env.fixed_waypoints[start_stage].copy()
    base_env.success_count = 0
    base_env.step_count = 0
    distance = float(
        np.linalg.norm(
            physical_state[np.asarray(TIP_INDICES)] - base_env.target_tip
        )
    )
    base_env.previous_distance = distance
    base_env.target_scale = max(distance, np.finfo(np.float32).eps)
    return env._observation(physical_state), tip_error


def _device(specification: str) -> torch.device:
    return torch.device(
        "cuda"
        if specification == "auto" and torch.cuda.is_available()
        else ("cpu" if specification == "auto" else specification)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--koopman-checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--waypoint-root",
        required=True,
        help="Directory containing the certified waypoint-bank manifest.json.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--base-dataset",
        default=None,
        help="Existing expert dataset to prepend to the newly collected samples.",
    )
    parser.add_argument(
        "--rollout-checkpoint",
        default=None,
        help=(
            "Optional BC-KMPC checkpoint used to drive the simulator while the "
            "fixed MPC expert labels the visited states (DAgger collection)."
        ),
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=300)
    parser.add_argument(
        "--start-stages",
        type=int,
        nargs="+",
        choices=(0, 1, 2),
        default=(0,),
        help="Episode starting stages. Stages 1/2 are reached by replaying "
        "the preceding certified waypoint action before collection.",
    )
    parser.add_argument("--warmup-ramp-steps", type=int, default=100)
    parser.add_argument("--warmup-hold-steps", type=int, default=400)
    parser.add_argument("--warmup-tip-tolerance", type=float, default=0.003)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--state-weight", type=float, default=200.0)
    parser.add_argument(
        "--tip-state-scale",
        type=float,
        default=5.0,
        help="Tip-state multiplier tuned for the no-max-delta absolute-action MPC.",
    )
    parser.add_argument(
        "--action-weight",
        type=float,
        default=8000.0,
        help="Reference-action weight tuned to keep plans in Koopman support.",
    )
    parser.add_argument("--control-weight", type=float, default=1.0)
    parser.add_argument("--absolute-action-limit", type=float, default=0.30)
    parser.add_argument("--rollout-noise-std", type=float, default=0.0002)
    parser.add_argument("--qp-max-iterations", type=int, default=4000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(
        args.episodes,
        args.episode_steps,
        args.horizon,
        args.warmup_ramp_steps,
        args.warmup_hold_steps,
    ) < 1:
        parser.error("episodes, episode-steps and horizon must be positive")
    if args.rollout_noise_std < 0:
        parser.error("rollout-noise-std must be non-negative")
    if args.warmup_tip_tolerance <= 0:
        parser.error("warmup-tip-tolerance must be positive")
    args.start_stages = tuple(dict.fromkeys(args.start_stages))
    return args


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.koopman_checkpoint).expanduser().resolve()
    scenario = Path(args.scenario).expanduser().resolve()
    waypoint_root = Path(args.waypoint_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for name, path in (
        ("koopman checkpoint", checkpoint),
        ("scenario", scenario),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    base_dataset_path = (
        None
        if args.base_dataset is None
        else Path(args.base_dataset).expanduser().resolve()
    )
    rollout_checkpoint_path = (
        None
        if args.rollout_checkpoint is None
        else Path(args.rollout_checkpoint).expanduser().resolve()
    )
    for name, path in (
        ("base dataset", base_dataset_path),
        ("rollout checkpoint", rollout_checkpoint_path),
    ):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")

    waypoint_bank = load_manisoft_waypoint_reference_bank(waypoint_root)
    reference_states = waypoint_bank.states
    reference_actions = waypoint_bank.actions
    if waypoint_bank.scenario_sha256 != sha256(scenario):
        raise ValueError("Waypoint bank was certified with another scenario")
    if base_dataset_path is not None:
        base_report_path = base_dataset_path.with_suffix(".json")
        if not base_report_path.is_file():
            raise FileNotFoundError(
                f"Missing base dataset metadata: {base_report_path}"
            )
        base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
        if int(base_report.get("schema_version", 0)) < 5:
            raise ValueError(
                "Base dataset predates randomized waypoint-bank BC-KMPC"
            )
        if base_report.get("waypoint_bank_sha256") != waypoint_bank.manifest_sha256:
            raise ValueError("Base dataset references another waypoint set")

    device = _device(args.device)
    model, payload = load_checkpoint(checkpoint, map_location=device)
    if not isinstance(model, HistoryDeepKoopman):
        raise ValueError("BC-KMPC collection requires HistoryDeepKoopman")
    model = model.to(device).freeze_dynamics()
    rollout_policy = None
    rollout_payload = None
    if rollout_checkpoint_path is not None:
        rollout_policy, rollout_payload, _ = load_history_mpc_checkpoint(
            rollout_checkpoint_path,
            device,
        )
        if rollout_payload.get("koopman_checkpoint_sha256") != sha256(checkpoint):
            raise ValueError("Rollout checkpoint references another Koopman model")
        if rollout_policy.history_steps != model.history_steps:
            raise ValueError("Rollout policy history length is incompatible")
        if rollout_policy.waypoint_count != 3:
            raise ValueError("Rollout policy is not a three-waypoint policy")
        if rollout_payload.get("waypoint_bank_sha256") != waypoint_bank.manifest_sha256:
            raise ValueError("Rollout checkpoint references another waypoint set")
        rollout_policy.eval()
    state_stats = payload["normalizers"]["state"]
    state_mean = torch.as_tensor(
        state_stats["mean"],
        dtype=torch.float32,
        device=device,
    )
    state_std = torch.as_tensor(
        state_stats["std"],
        dtype=torch.float32,
        device=device,
    )
    action_low = np.full(18, -args.absolute_action_limit, dtype=np.float64)
    action_high = np.full(18, args.absolute_action_limit, dtype=np.float64)
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
        tip_state_scale=args.tip_state_scale,
        qp_max_iterations=args.qp_max_iterations,
    )
    base_env = ManiSoftThreeWaypointTrackingEnv(
        scenario,
        waypoint_tips=reference_states[:, :, np.asarray(TIP_INDICES)],
        episode_steps=(
            args.episode_steps
            + args.warmup_ramp_steps
            + args.warmup_hold_steps
        ),
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
    waypoint_schedule = np.concatenate(
        [
            rng.permutation(waypoint_bank.triplet_count)
            for _ in range(
                (args.episodes + waypoint_bank.triplet_count - 1)
                // waypoint_bank.triplet_count
            )
        ]
    )[: args.episodes]
    start_stage_schedule = _start_stage_schedule(
        args.episodes,
        args.start_stages,
        rng,
    )
    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    step_indices: list[int] = []
    expert_costs: list[float] = []
    qp_iterations: list[int] = []
    active_waypoint_indices: list[int] = []
    waypoint_triplet_indices: list[int] = []
    waypoint_passed: list[bool] = []
    waypoints_completed: list[int] = []
    if base_dataset_path is not None:
        with np.load(base_dataset_path, allow_pickle=False) as archive:
            base_arrays = {
                name: np.asarray(archive[name])
                for name in (
                    "observation",
                    "expert_action",
                    "applied_action",
                    "episode_id",
                    "step_index",
                    "expert_cost",
                    "qp_iterations",
                    "active_waypoint_index",
                    "waypoint_triplet_index",
                    "waypoint_passed",
                    "waypoints_completed",
                )
            }
        base_count = len(base_arrays["observation"])
        if any(len(value) != base_count for value in base_arrays.values()):
            raise ValueError("Base dataset arrays have inconsistent lengths")
        if base_arrays["observation"].shape[1:] != env.observation_space.shape:
            raise ValueError("Base dataset observation dimension is incompatible")
        if base_arrays["expert_action"].shape[1:] != (model.action_dim,):
            raise ValueError("Base dataset action dimension is incompatible")
        observations.extend(base_arrays["observation"].astype(np.float32))
        expert_actions.extend(base_arrays["expert_action"].astype(np.float32))
        applied_actions.extend(base_arrays["applied_action"].astype(np.float32))
        episode_ids.extend(base_arrays["episode_id"].astype(np.int64).tolist())
        step_indices.extend(base_arrays["step_index"].astype(np.int64).tolist())
        expert_costs.extend(base_arrays["expert_cost"].astype(np.float32).tolist())
        qp_iterations.extend(base_arrays["qp_iterations"].astype(np.int32).tolist())
        active_waypoint_indices.extend(
            base_arrays["active_waypoint_index"].astype(np.int64).tolist()
        )
        waypoint_triplet_indices.extend(
            base_arrays["waypoint_triplet_index"].astype(np.int64).tolist()
        )
        waypoint_passed.extend(
            base_arrays["waypoint_passed"].astype(np.bool_).tolist()
        )
        waypoints_completed.extend(
            base_arrays["waypoints_completed"].astype(np.int64).tolist()
        )
    base_samples = len(observations)
    episode_offset = max(episode_ids, default=-1) + 1
    episode_returns: list[float] = []
    episode_start_stages: list[int] = []
    warmup_tip_errors: list[float] = []
    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset(
                seed=args.seed + episode,
                options={
                    "waypoint_triplet_index": int(waypoint_schedule[episode])
                },
            )
            waypoint_triplet_index = int(reset_info["waypoint_triplet_index"])
            start_stage = int(start_stage_schedule[episode])
            warmup_tip_error = 0.0
            if start_stage:
                equilibrium_index = start_stage - 1
                observation, warmup_tip_error = _warm_start_stage(
                    env,
                    base_env,
                    start_stage=start_stage,
                    equilibrium_action=reference_actions[
                        waypoint_triplet_index, equilibrium_index
                    ],
                    reference_state=reference_states[
                        waypoint_triplet_index, equilibrium_index
                    ],
                    ramp_steps=args.warmup_ramp_steps,
                    hold_steps=args.warmup_hold_steps,
                )
                if warmup_tip_error > args.warmup_tip_tolerance:
                    raise RuntimeError(
                        "Certified warm start tip error "
                        f"{1000.0 * warmup_tip_error:.3f} mm exceeds "
                        f"{1000.0 * args.warmup_tip_tolerance:.3f} mm"
                    )
                warm_start = np.repeat(
                    reference_actions[
                        waypoint_triplet_index, equilibrium_index
                    ][None, :],
                    args.horizon,
                    axis=0,
                )
            else:
                warm_start = None
            episode_return = 0.0
            executed_steps = 0
            for step in range(args.episode_steps):
                active_waypoint_index = int(base_env.active_waypoint_index)
                state = observation[: model.state_dim]
                context_start = model.state_dim
                context_stop = context_start + model.context_dim
                context = observation[context_start:context_stop]
                plan = expert.solve(
                    state=state,
                    context=context,
                    reference_state=reference_states[
                        waypoint_triplet_index, active_waypoint_index
                    ],
                    reference_action=reference_actions[
                        waypoint_triplet_index, active_waypoint_index
                    ],
                    initial_actions=warm_start,
                )
                target_action = np.asarray(plan["action"], dtype=np.float32)
                if rollout_policy is None:
                    rollout_action = target_action.copy()
                else:
                    with torch.no_grad():
                        rollout_action, _, _ = rollout_policy.act(
                            torch.as_tensor(
                                observation,
                                dtype=torch.float32,
                                device=device,
                            ),
                            deterministic=True,
                        )
                    rollout_action = rollout_action.detach().cpu().numpy()
                if args.rollout_noise_std:
                    rollout_action += rng.normal(
                        0.0,
                        args.rollout_noise_std,
                        size=model.action_dim,
                    ).astype(np.float32)
                next_observation, reward, terminated, truncated, info = env.step(
                    rollout_action
                )
                observations.append(observation.copy())
                expert_actions.append(target_action)
                applied_actions.append(
                    np.asarray(info["applied_action"], dtype=np.float32)
                )
                episode_ids.append(episode_offset + episode)
                step_indices.append(step)
                expert_costs.append(float(plan["cost"]))
                qp_iterations.append(int(plan["qp_iterations"]))
                active_waypoint_indices.append(active_waypoint_index)
                waypoint_triplet_indices.append(waypoint_triplet_index)
                waypoint_passed.append(bool(info["waypoint_passed"]))
                waypoints_completed.append(int(info["waypoints_completed"]))
                episode_return += float(reward)
                executed_steps += 1
                observation = next_observation
                actions = np.asarray(plan["actions"], dtype=np.float32)
                warm_start = np.concatenate((actions[1:], actions[-1:]), axis=0)
                if terminated or truncated:
                    break
            episode_returns.append(episode_return)
            episode_start_stages.append(start_stage)
            warmup_tip_errors.append(warmup_tip_error)
            print(
                json.dumps(
                    {
                        "episode": episode,
                        "steps": executed_steps,
                        "return": episode_return,
                        "new_samples": len(observations) - base_samples,
                        "samples": len(observations),
                        "waypoints_completed": int(base_env.waypoints_completed),
                        "waypoint_triplet_index": waypoint_triplet_index,
                        "success": bool(terminated),
                        "start_stage": start_stage,
                        "warmup_tip_error_mm": 1000.0 * warmup_tip_error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        env.close()

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            observation=np.asarray(observations, dtype=np.float32),
            expert_action=np.asarray(expert_actions, dtype=np.float32),
            applied_action=np.asarray(applied_actions, dtype=np.float32),
            episode_id=np.asarray(episode_ids, dtype=np.int64),
            step_index=np.asarray(step_indices, dtype=np.int64),
            expert_cost=np.asarray(expert_costs, dtype=np.float32),
            qp_iterations=np.asarray(qp_iterations, dtype=np.int32),
            active_waypoint_index=np.asarray(
                active_waypoint_indices, dtype=np.int64
            ),
            waypoint_triplet_index=np.asarray(
                waypoint_triplet_indices, dtype=np.int64
            ),
            waypoint_passed=np.asarray(waypoint_passed, dtype=np.bool_),
            waypoints_completed=np.asarray(waypoints_completed, dtype=np.int64),
        )
    temporary.replace(output)
    report = {
        "schema_version": 5,
        "fixed_smoothness": False,
        "action_constraints": "absolute_box",
        "kind": "manisoft_history_bc_kmpc_three_waypoint_expert",
        "output": str(output),
        "samples": len(observations),
        "base_samples": base_samples,
        "new_samples": len(observations) - base_samples,
        "episodes": args.episodes,
        "episode_return_mean": float(np.mean(episode_returns)),
        "start_stage_episode_counts": {
            str(stage): int(np.sum(np.asarray(episode_start_stages) == stage))
            for stage in args.start_stages
        },
        "warmup_tip_error_mm_mean": float(
            1000.0 * np.mean(warmup_tip_errors)
        ),
        "warmup_tip_error_mm_max": float(
            1000.0 * np.max(warmup_tip_errors)
        ),
        "observation_dim": int(env.observation_space.shape[0]),
        "action_dim": model.action_dim,
        "history_steps": model.history_steps,
        "koopman_checkpoint": str(checkpoint),
        "koopman_checkpoint_sha256": sha256(checkpoint),
        "waypoint_root": str(waypoint_root),
        "waypoint_bank_manifest": str(waypoint_bank.manifest_path),
        "waypoint_bank_sha256": waypoint_bank.manifest_sha256,
        "waypoint_triplet_count": waypoint_bank.triplet_count,
        "waypoint_sampling": "shuffled_without_replacement_per_cycle",
        "references": [str(path) for path in waypoint_bank.reference_paths],
        "scenario": str(scenario),
        "base_dataset": (
            None if base_dataset_path is None else str(base_dataset_path)
        ),
        "base_dataset_sha256": (
            None if base_dataset_path is None else sha256(base_dataset_path)
        ),
        "rollout_checkpoint": (
            None
            if rollout_checkpoint_path is None
            else str(rollout_checkpoint_path)
        ),
        "rollout_checkpoint_sha256": (
            None
            if rollout_checkpoint_path is None
            else sha256(rollout_checkpoint_path)
        ),
        "runtime": vars(args),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
