"""Collect visual ordered three-waypoint PandaReach demonstrations.

Same DLS expert and causal rollout as the state-space collector, but the
environment is ``ACMPC-VisualPandaReach3-v0`` (goal spheres visible to the
camera) and each transition additionally stores the RGB-D observation.  The
numerical waypoint positions are stored as privileged side data for the
training-only ``pos_branch`` supervision; they never enter the model at
inference time.

This collector is fully additive: it reuses the existing DLS expert, episode
splits and the ``q_qdot_tcp`` state convention, and does not modify the
state-space collector.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from experiments.state_only_feasibility.collect_pandareach import (
    ObservableDLSExpert,
    _episode_splits,
    _first,
    _minimum_jerk,
    _numpy,
)
from experiments.state_only_feasibility.collect_pandareach_threewaypoint import (
    _state_from_observation,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
)
from experiments.state_only_feasibility.visual_pandareach_env import (
    VisualPandaReachThreeWaypointEnv,
)
from experiments.state_only_feasibility.visual_pandareach_single_goal import (
    VisualPandaReachSingleGoalEnv,
)


@dataclass(frozen=True)
class CollectionConfig:
    env_id: str = "ACMPC-VisualPandaReach3-v0"
    episodes: int = 100
    seed_start: int = 20_260_804
    split_seed: int = 31
    steps_per_waypoint: int = 45
    max_episode_steps: int = 200
    goal_threshold: float = 0.01
    waypoint_joint_jitter: float = 0.02
    dls_damping: float = 0.04
    dls_gain: float = 1.0
    velocity_damping: float = 0.002
    action_radians: float = 0.1
    # ManiSkill minimal-shader depth is in millimeters (uint16).
    depth_scale: float = 2500.0
    # Workspace-space goal sampling box for the single-goal env; None keeps the
    # FK-certified waypoint path used by the three-waypoint collector.
    goal_region_radius: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.env_id not in {
            "ACMPC-VisualPandaReach3-v0",
            "ACMPC-VisualPandaReach1-v0",
        }:
            raise ValueError(f"Unsupported visual env id {self.env_id!r}")
        if self.episodes < 3:
            raise ValueError("At least three episodes are required")
        if self.steps_per_waypoint < 1:
            raise ValueError("steps_per_waypoint must be positive")
        if self.max_episode_steps < 3 * self.steps_per_waypoint:
            raise ValueError("max_episode_steps is too short for three segments")
        if self.goal_threshold <= 0:
            raise ValueError("goal_threshold must be positive")
        if self.waypoint_joint_jitter <= 0:
            raise ValueError("waypoint_joint_jitter must be positive")
        if self.dls_damping <= 0 or self.dls_gain <= 0:
            raise ValueError("DLS damping and gain must be positive")
        if self.action_radians <= 0:
            raise ValueError("action_radians must be positive")
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        if self.goal_region_radius is not None and (
            len(self.goal_region_radius) != 3
            or any(value <= 0 for value in self.goal_region_radius)
        ):
            raise ValueError(
                "goal_region_radius must be three positive values or None"
            )


def _unbatch(value: Any, *, name: str) -> np.ndarray:
    array = _numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite values in observation leaf {name}")
    return array


def _extract_observation(
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (robot[17], rgb HWC uint8, depth HW1 uint16)."""

    robot = _state_from_observation(observation)
    camera = observation["sensor_data"]["base_camera"]
    rgb = _unbatch(camera["rgb"], name="sensor_data/base_camera/rgb")
    depth = _unbatch(camera["depth"], name="sensor_data/base_camera/depth")
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise RuntimeError(f"Expected HWC RGB, got {rgb.shape}")
    if depth.ndim == 2:
        depth = depth[..., None]
    if depth.ndim != 3 or depth.shape[-1] != 1:
        raise RuntimeError(f"Expected HW1 depth, got {depth.shape}")
    if rgb.shape[:2] != depth.shape[:2]:
        raise RuntimeError("RGB and depth resolutions differ")
    return (
        robot.astype(np.float32),
        rgb.astype(np.uint8, copy=False),
        depth.astype(np.uint16, copy=False),
    )


def collect(config: CollectionConfig, output_path: Path) -> dict[str, Any]:
    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        num_envs=1,
        obs_mode="rgb+depth",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode=None,
        max_episode_steps=config.max_episode_steps,
        goal_threshold=config.goal_threshold,
        waypoint_joint_jitter=config.waypoint_joint_jitter,
    )
    region = config.goal_region_radius
    if config.env_id == "ACMPC-VisualPandaReach1-v0":
        region = region if region is not None else (0.06, 0.06, 0.03)
        env_kwargs["goal_region_radius"] = region
        env_kwargs["goal_marker_scale"] = 5.0
    base_env = gym.make(config.env_id, **env_kwargs)
    env = PandaArmOnlyActionWrapper(base_env)
    env.reset(seed=config.seed_start)
    unwrapped = env.unwrapped
    expert = ObservableDLSExpert(
        unwrapped,
        damping=config.dls_damping,
        gain=config.dls_gain,
        velocity_damping=config.velocity_damping,
        action_radians=config.action_radians,
    )

    if (
        int(unwrapped.control_freq),
        int(unwrapped.sim_freq),
        int(unwrapped._sim_steps_per_control),
    ) != (20, 100, 5):
        raise RuntimeError("Unexpected PandaReach control timing")

    names = (
        "state",
        "action",
        "env_action",
        "next_state",
        "reward",
        "terminal",
        "timeout",
        "success",
        "waypoint_passed",
        "episode_id",
        "step_index",
        "active_waypoint_index",
        "next_active_waypoint_index",
        "waypoints_completed",
        "active_goal_position",
        "next_active_goal_position",
        "desired_tcp_position",
        "tcp_position",
        "next_tcp_position",
    )
    transitions: dict[str, list[Any]] = {name: [] for name in names}
    rgb_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    episode_seed: list[int] = []
    episode_length: list[int] = []
    episode_waypoints: list[np.ndarray] = []
    episode_fk_residual: list[np.ndarray] = []
    episode_final_distance: list[float] = []

    try:
        for episode_id in range(config.episodes):
            seed = config.seed_start + episode_id
            observation, _ = env.reset(seed=seed)
            waypoint_count = int(unwrapped.waypoint_count)
            waypoints = _numpy(unwrapped.waypoints).reshape(
                waypoint_count, 3
            ).copy()
            oracle_qpos = _numpy(unwrapped.oracle_waypoint_qpos).reshape(
                waypoint_count, 7
            ).copy()
            full_qpos = _numpy(unwrapped.agent.robot.get_qpos()).reshape(9)
            if region is not None:
                # Waypoints are deliberately offset in workspace space after
                # FK, so the oracle-qpos certificate no longer matches them.
                fk_residual = np.zeros(waypoint_count, dtype=np.float32)
            else:
                fk_residual = np.empty(waypoint_count, dtype=np.float32)
                for waypoint_index in range(waypoint_count):
                    certificate = full_qpos.copy()
                    certificate[:7] = oracle_qpos[waypoint_index]
                    certified = expert.fk_tcp_world(certificate)
                    fk_residual[waypoint_index] = np.linalg.norm(
                        certified - waypoints[waypoint_index]
                    )
            if not np.isfinite(waypoints).all() or float(fk_residual.max()) > 2e-4:
                raise RuntimeError(f"Episode {episode_id} failed FK certification")

            robot, rgb, depth = _extract_observation(observation)
            segment_start = robot[-3:].copy()
            segment_step = 0
            success = False
            final_distance = float("inf")
            for step_index in range(config.max_episode_steps):
                active_index = int(
                    _first(observation["extra"]["active_waypoint_index"])
                )
                active_goal = _numpy(
                    observation["extra"]["active_goal"]
                ).reshape(3).copy()
                progress = _minimum_jerk(
                    segment_step + 1, config.steps_per_waypoint
                )
                desired_tcp = segment_start + progress * (
                    active_goal - segment_start
                )
                state = robot
                tcp = state[-3:].copy()
                env_action = expert.action(desired_tcp)
                action = config.action_radians * env_action
                next_observation, reward, terminated, truncated, info = env.step(
                    env_action
                )
                next_robot, next_rgb, next_depth = _extract_observation(
                    next_observation
                )
                next_state = next_robot
                next_goal = _numpy(
                    next_observation["extra"]["active_goal"]
                ).reshape(3).copy()
                next_index = int(
                    _first(next_observation["extra"]["active_waypoint_index"])
                )
                passed = bool(_first(info["waypoint_passed"]))
                transition_success = bool(_first(info["success"]))
                transition_terminal = bool(_first(terminated))
                transition_timeout = bool(_first(truncated))
                final_distance = float(_first(info["active_waypoint_distance"]))

                values = {
                    "state": state,
                    "action": action,
                    "env_action": env_action,
                    "next_state": next_state,
                    "reward": float(_first(reward)),
                    "terminal": transition_terminal,
                    "timeout": transition_timeout,
                    "success": transition_success,
                    "waypoint_passed": passed,
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "active_waypoint_index": active_index,
                    "next_active_waypoint_index": next_index,
                    "waypoints_completed": int(_first(info["waypoints_completed"])),
                    "active_goal_position": active_goal,
                    "next_active_goal_position": next_goal,
                    "desired_tcp_position": desired_tcp.astype(np.float32),
                    "tcp_position": tcp,
                    "next_tcp_position": next_state[-3:].copy(),
                }
                for name, value in values.items():
                    transitions[name].append(value)
                rgb_frames.append(rgb)
                depth_frames.append(depth)

                observation = next_observation
                robot = next_robot
                if passed:
                    segment_start = next_state[-3:].copy()
                    segment_step = 0
                else:
                    segment_step += 1
                if transition_success:
                    success = True
                    break
                if transition_terminal or transition_timeout:
                    break

            if not success:
                completed = int(_first(info["waypoints_completed"]))
                raise RuntimeError(
                    f"DLS expert failed seed {seed}: completed={completed}/3, "
                    f"steps={step_index + 1}, distance={final_distance:.6f} m"
                )
            episode_seed.append(seed)
            episode_length.append(step_index + 1)
            episode_waypoints.append(waypoints)
            episode_fk_residual.append(fk_residual)
            episode_final_distance.append(final_distance)
    finally:
        env.close()

    if len(rgb_frames) != len(transitions["state"]):
        raise RuntimeError("RGB/depth frames and transitions are misaligned")

    train_ids, validation_ids, test_ids = _episode_splits(
        config.episodes, config.split_seed
    )
    float_vectors = {
        "state",
        "action",
        "env_action",
        "next_state",
        "active_goal_position",
        "next_active_goal_position",
        "desired_tcp_position",
        "tcp_position",
        "next_tcp_position",
    }
    bool_scalars = {"terminal", "timeout", "success", "waypoint_passed"}
    int_scalars = {
        "episode_id",
        "step_index",
        "active_waypoint_index",
        "next_active_waypoint_index",
        "waypoints_completed",
    }
    arrays: dict[str, np.ndarray] = {}
    for name, values in transitions.items():
        if name in float_vectors:
            arrays[name] = np.asarray(values, dtype=np.float32)
        elif name in bool_scalars:
            arrays[name] = np.asarray(values, dtype=np.bool_)
        elif name in int_scalars:
            arrays[name] = np.asarray(values, dtype=np.int64)
        else:
            arrays[name] = np.asarray(values, dtype=np.float32)
    arrays.update(
        {
            "rgb": np.stack(rgb_frames).astype(np.uint8),
            "depth": np.stack(depth_frames).astype(np.uint16),
            "state_kind": np.asarray("q_qdot_tcp"),
            "image_size": np.asarray(rgb_frames[0].shape[:2], dtype=np.int64),
            "depth_scale": np.asarray(config.depth_scale, dtype=np.float32),
            "episode_seed": np.asarray(episode_seed, dtype=np.int64),
            "episode_length": np.asarray(episode_length, dtype=np.int64),
            "episode_waypoints": np.asarray(episode_waypoints, dtype=np.float32),
            "episode_fk_residual": np.asarray(
                episode_fk_residual, dtype=np.float32
            ),
            "episode_final_distance": np.asarray(
                episode_final_distance, dtype=np.float32
            ),
            "train_episode_ids": train_ids,
            "validation_episode_ids": validation_ids,
            "test_episode_ids": test_ids,
        }
    )
    for name, value in arrays.items():
        if value.dtype.kind not in {"U", "S"} and not np.isfinite(value).all():
            raise RuntimeError(f"Dataset field {name} contains NaN or Inf")
    np.savez_compressed(output_path, **arrays)

    lengths = arrays["episode_length"]
    summary = {
        "dataset_path": str(output_path.resolve()),
        "episodes": config.episodes,
        "transitions": int(len(arrays["state"])),
        "expert_success_rate": 1.0,
        "image_size": arrays["image_size"].tolist(),
        "episode_length": {
            "mean": float(lengths.mean()),
            "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
        },
        "waypoint_region_tcp": {
            str(index + 1): {
                "minimum": arrays["episode_waypoints"][:, index].min(0).tolist(),
                "maximum": arrays["episode_waypoints"][:, index].max(0).tolist(),
            }
            for index in range(int(arrays["episode_waypoints"].shape[1]))
        },
        "config": asdict(config),
    }
    with output_path.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--env-id",
        default=CollectionConfig.env_id,
        choices=["ACMPC-VisualPandaReach3-v0", "ACMPC-VisualPandaReach1-v0"],
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--steps-per-waypoint", type=int, default=None)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument(
        "--goal-region-radius",
        type=str,
        default=None,
        help="comma-separated x,y,z workspace half-extents for the single-goal "
        "env, e.g. 0.06,0.06,0.03 (skips FK certification)",
    )
    args = parser.parse_args()

    config = CollectionConfig(
        env_id=args.env_id,
        episodes=args.episodes if args.episodes is not None else CollectionConfig.episodes,
        seed_start=(
            args.seed_start if args.seed_start is not None else CollectionConfig.seed_start
        ),
        steps_per_waypoint=(
            args.steps_per_waypoint
            if args.steps_per_waypoint is not None
            else CollectionConfig.steps_per_waypoint
        ),
        depth_scale=(
            args.depth_scale if args.depth_scale is not None else CollectionConfig.depth_scale
        ),
        goal_region_radius=(
            tuple(float(v) for v in args.goal_region_radius.split(","))
            if args.goal_region_radius is not None
            else None
        ),
    )
    collect(config, args.output)
    print(f"collected visual dataset -> {args.output}")


if __name__ == "__main__":
    main()
