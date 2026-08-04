"""Collect ordered three-waypoint PandaReach demonstrations.

The expert observes only robot state and the active absolute Cartesian goal.
Each segment uses a fresh minimum-jerk reference tracked by DLS.  Hidden joint
configurations from the waypoint sampler are saved only as FK certificates.
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
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
    PandaReachThreeWaypointEnv,
)


@dataclass(frozen=True)
class CollectionConfig:
    episodes: int = 100
    seed_start: int = 20_260_804
    split_seed: int = 31
    steps_per_waypoint: int = 45
    max_episode_steps: int = 200
    goal_threshold: float = 0.01
    waypoint_joint_jitter: float = 0.02
    waypoint_event_reward: float = 0.2
    dls_damping: float = 0.04
    dls_gain: float = 1.0
    velocity_damping: float = 0.002
    action_radians: float = 0.1

    def validate(self) -> None:
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


def _state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    qpos = _numpy(observation["agent"]["qpos"]).reshape(-1, 7)[0]
    qvel = _numpy(observation["agent"]["qvel"]).reshape(-1, 7)[0]
    tcp = _numpy(observation["extra"]["tcp_pos"]).reshape(-1, 3)[0]
    state = np.concatenate((qpos, qvel, tcp)).astype(np.float32)
    if state.shape != (17,) or not np.isfinite(state).all():
        raise RuntimeError("Invalid robot-only [q, qdot, tcp] state")
    return state


def collect(config: CollectionConfig, output_path: Path) -> dict[str, Any]:
    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_env = gym.make(
        "ACMPC-PandaReach3-v0",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        reward_mode="sparse",
        render_mode=None,
        render_backend="none",
        max_episode_steps=config.max_episode_steps,
        goal_threshold=config.goal_threshold,
        waypoint_joint_jitter=config.waypoint_joint_jitter,
        waypoint_event_reward=config.waypoint_event_reward,
    )
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
    episode_seed: list[int] = []
    episode_length: list[int] = []
    episode_waypoints: list[np.ndarray] = []
    episode_oracle_qpos: list[np.ndarray] = []
    episode_fk_residual: list[np.ndarray] = []
    episode_final_distance: list[float] = []

    try:
        for episode_id in range(config.episodes):
            seed = config.seed_start + episode_id
            observation, _ = env.reset(seed=seed)
            waypoints = _numpy(unwrapped.waypoints).reshape(3, 3).copy()
            oracle_qpos = _numpy(unwrapped.oracle_waypoint_qpos).reshape(
                3, 7
            ).copy()
            full_qpos = _numpy(unwrapped.agent.robot.get_qpos()).reshape(9)
            fk_residual = np.empty(3, dtype=np.float32)
            for waypoint_index in range(3):
                certificate = full_qpos.copy()
                certificate[:7] = oracle_qpos[waypoint_index]
                certified = expert.fk_tcp_world(certificate)
                fk_residual[waypoint_index] = np.linalg.norm(
                    certified - waypoints[waypoint_index]
                )
            if not np.isfinite(waypoints).all() or float(fk_residual.max()) > 2e-4:
                raise RuntimeError(f"Episode {episode_id} failed FK certification")

            segment_start = _numpy(
                unwrapped.agent.tcp_pose.p
            ).reshape(3).copy()
            segment_step = 0
            success = False
            final_distance = float("inf")
            for step_index in range(config.max_episode_steps):
                active_index = int(
                    _numpy(observation["extra"]["active_waypoint_index"])
                    .reshape(-1)[0]
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
                state = _state_from_observation(observation)
                tcp = state[-3:].copy()
                env_action = expert.action(desired_tcp)
                action = config.action_radians * env_action
                next_observation, reward, terminated, truncated, info = env.step(
                    env_action
                )
                next_state = _state_from_observation(next_observation)
                next_goal = _numpy(
                    next_observation["extra"]["active_goal"]
                ).reshape(3).copy()
                next_index = int(
                    _numpy(
                        next_observation["extra"]["active_waypoint_index"]
                    ).reshape(-1)[0]
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
                observation = next_observation

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
            episode_oracle_qpos.append(oracle_qpos)
            episode_fk_residual.append(fk_residual)
            episode_final_distance.append(final_distance)
    finally:
        env.close()

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
            "state_kind": np.asarray("q_qdot_tcp"),
            "episode_seed": np.asarray(episode_seed, dtype=np.int64),
            "episode_length": np.asarray(episode_length, dtype=np.int64),
            "episode_waypoints": np.asarray(episode_waypoints, dtype=np.float32),
            "episode_oracle_waypoint_qpos": np.asarray(
                episode_oracle_qpos, dtype=np.float32
            ),
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
        "episode_length": {
            "mean": float(lengths.mean()),
            "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
        },
        "waypoint_region_tcp": {
            str(index + 1): {
                "minimum": arrays["episode_waypoints"][:, index].min(0).tolist(),
                "maximum": arrays["episode_waypoints"][:, index].max(0).tolist(),
                "mean": arrays["episode_waypoints"][:, index].mean(0).tolist(),
            }
            for index in range(3)
        },
        "fk_certificate_residual_m_max": float(
            arrays["episode_fk_residual"].max()
        ),
        "timing": {
            "control_frequency_hz": 20,
            "simulation_frequency_hz": 100,
            "physics_steps_per_action": 5,
            "control_dt_seconds": 0.05,
        },
        "config": asdict(config),
        "state_definition": "x=[q(7), qdot(7), tcp_xyz(3)]",
        "task_context": "c=active absolute waypoint G_j in world xyz",
        "actor_input": "h=[psi(normalize(x)), normalize(G_j)]",
        "action_definition": (
            "u=joint_delta in radians with hard support [-0.1,0.1]; "
            "env_action=u/0.1 is retained separately"
        ),
        "reward": "1 on final success; beta on first passage of waypoint 1/2; 0 otherwise",
        "expert": "per-segment minimum jerk Cartesian reference + observable DLS",
        "scope": "ordered three-waypoint state-space dataset",
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/pandareach_threewaypoint/data/pandareach_dls_100.npz"
        ),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=20_260_804)
    parser.add_argument("--steps-per-waypoint", type=int, default=45)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--goal-threshold", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CollectionConfig(
        episodes=args.episodes,
        seed_start=args.seed_start,
        steps_per_waypoint=args.steps_per_waypoint,
        max_episode_steps=args.max_episode_steps,
        goal_threshold=args.goal_threshold,
    )
    print(json.dumps(collect(config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
