"""Collect FK-certified PandaReach demonstrations with an observable DLS expert.

The sampler's hidden joint configuration is retained only as an audit
certificate. Expert actions depend on the measured joints, joint velocities,
TCP position, and the Cartesian waypoint; the hidden configuration is never an
expert or policy input.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
)


@dataclass(frozen=True)
class CollectionConfig:
    episodes: int = 100
    seed_start: int = 20_260_731
    split_seed: int = 31
    trajectory_steps: int = 120
    max_episode_steps: int = 160
    collection_goal_threshold: float = 0.005
    dls_damping: float = 0.04
    dls_gain: float = 1.0
    velocity_damping: float = 0.002
    action_radians: float = 0.1
    goal_joint_delta: float = 0.15
    goal_min_tcp_distance: float = 0.04
    goal_max_tcp_distance: float = 0.18
    goal_min_height: float = 0.10

    def validate(self) -> None:
        if self.episodes < 3:
            raise ValueError("At least three episodes are required")
        if self.trajectory_steps < 1:
            raise ValueError("trajectory_steps must be positive")
        if self.max_episode_steps < self.trajectory_steps:
            raise ValueError(
                "max_episode_steps must allow the nominal trajectory"
            )
        if self.collection_goal_threshold <= 0:
            raise ValueError("collection_goal_threshold must be positive")
        if self.dls_damping <= 0 or self.dls_gain <= 0:
            raise ValueError("DLS damping and gain must be positive")
        if self.action_radians <= 0:
            raise ValueError("action_radians must be positive")


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first(value: Any) -> Any:
    return _numpy(value).reshape(-1)[0]


def _state_from_observation(observation: dict[str, Any]) -> np.ndarray:
    qpos = _numpy(observation["agent"]["qpos"]).reshape(-1, 7)[0]
    qvel = _numpy(observation["agent"]["qvel"]).reshape(-1, 7)[0]
    error = _numpy(observation["extra"]["tcp_to_goal"]).reshape(-1, 3)[0]
    state = np.concatenate((qpos, qvel, error)).astype(np.float32)
    if state.shape != (17,) or not np.isfinite(state).all():
        raise RuntimeError("Invalid [q, qdot, tcp-goal] observation")
    return state


def _minimum_jerk(step: int, trajectory_steps: int) -> float:
    tau = min(float(step) / float(trajectory_steps), 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


class ObservableDLSExpert:
    """Track Cartesian waypoints without using the sampler's hidden q goal."""

    def __init__(
        self,
        unwrapped_env: Any,
        *,
        damping: float,
        gain: float,
        velocity_damping: float,
        action_radians: float,
    ) -> None:
        self.env = unwrapped_env
        self.agent = unwrapped_env.agent
        self.damping = float(damping)
        self.gain = float(gain)
        self.velocity_damping = float(velocity_damping)
        self.action_radians = float(action_radians)

        # ManiSkill constructs the same kinematics object used by its
        # end-effector controller. Switch back before collecting actions.
        original_mode = self.agent.control_mode
        self.agent.set_control_mode("pd_ee_delta_pos")
        self.kinematics = self.agent.controller.controllers["arm"].kinematics
        self.agent.set_control_mode(original_mode)
        self.model_indices = _numpy(
            self.kinematics.pmodel_active_joint_indices
        ).astype(np.int64)

    def _position_jacobian(self, full_qpos: np.ndarray) -> np.ndarray:
        model_qpos = full_qpos[self.model_indices]
        model = self.kinematics.pmodel
        model.compute_forward_kinematics(model_qpos)
        link_pose = model.get_link_pose(self.kinematics.end_link_idx)
        model.compute_full_jacobian(model_qpos)
        local_jacobian = model.get_link_jacobian(
            self.kinematics.end_link_idx,
            local=True,
        )[:3, :7]
        rotation = link_pose.to_transformation_matrix()[:3, :3]
        return np.asarray(rotation @ local_jacobian, dtype=np.float64)

    def fk_tcp_world(self, full_qpos: np.ndarray) -> np.ndarray:
        model_qpos = full_qpos[self.model_indices]
        model = self.kinematics.pmodel
        model.compute_forward_kinematics(model_qpos)
        local_pose = model.get_link_pose(self.kinematics.end_link_idx)
        local_tcp = local_pose.to_transformation_matrix()[:3, 3]
        world_from_base = _numpy(
            self.agent.robot.pose.to_transformation_matrix()
        ).reshape(-1, 4, 4)[0]
        return (
            world_from_base[:3, :3] @ local_tcp
            + world_from_base[:3, 3]
        ).astype(np.float32)

    def action(self, desired_tcp: np.ndarray) -> np.ndarray:
        full_qpos = _numpy(self.agent.robot.get_qpos()).reshape(-1, 9)[0]
        qvel = _numpy(self.agent.robot.get_qvel()).reshape(-1, 9)[0, :7]
        tcp = _numpy(self.agent.tcp_pose.p).reshape(-1, 3)[0]
        jacobian = self._position_jacobian(full_qpos)
        regularized = (
            jacobian @ jacobian.T
            + self.damping**2 * np.eye(3, dtype=np.float64)
        )
        error = np.asarray(desired_tcp, dtype=np.float64) - tcp
        joint_delta = (
            self.gain
            * jacobian.T
            @ np.linalg.solve(regularized, error)
            - self.velocity_damping * qvel
        )
        action = np.clip(
            joint_delta / self.action_radians,
            -1.0,
            1.0,
        ).astype(np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError("DLS expert produced an invalid action")
        return action


def _episode_splits(
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(count).astype(np.int64)
    train_count = int(np.floor(0.8 * count))
    validation_count = int(np.floor(0.1 * count))
    return (
        np.sort(order[:train_count]),
        np.sort(order[train_count : train_count + validation_count]),
        np.sort(order[train_count + validation_count :]),
    )


def collect(
    config: CollectionConfig,
    output_path: Path,
) -> dict[str, Any]:
    config.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_env = gym.make(
        "ACMPC-PandaReach-v0",
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        reward_mode="dense",
        max_episode_steps=config.max_episode_steps,
        goal_threshold=config.collection_goal_threshold,
        goal_joint_delta=config.goal_joint_delta,
        goal_min_tcp_distance=config.goal_min_tcp_distance,
        goal_max_tcp_distance=config.goal_max_tcp_distance,
        goal_min_height=config.goal_min_height,
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

    control_frequency = int(unwrapped.control_freq)
    simulation_frequency = int(unwrapped.sim_freq)
    simulation_steps = int(unwrapped._sim_steps_per_control)
    control_dt = float(unwrapped.control_timestep)
    if (control_frequency, simulation_frequency, simulation_steps) != (
        20,
        100,
        5,
    ) or not np.isclose(control_dt, 0.05):
        raise RuntimeError("Unexpected PandaReach control timing")

    transitions: dict[str, list[np.ndarray | float | int | bool]] = {
        "state": [],
        "action": [],
        "action_rad": [],
        "next_state": [],
        "reward": [],
        "terminal": [],
        "timeout": [],
        "success": [],
        "episode_id": [],
        "step_index": [],
        "goal_position": [],
        "desired_tcp_position": [],
        "tcp_position": [],
        "next_tcp_position": [],
    }
    episode_seed: list[int] = []
    episode_length: list[int] = []
    episode_goal: list[np.ndarray] = []
    episode_start_tcp: list[np.ndarray] = []
    episode_oracle_qpos: list[np.ndarray] = []
    episode_sampling_attempts: list[int] = []
    episode_fk_residual: list[float] = []
    episode_initial_distance: list[float] = []
    episode_final_distance: list[float] = []

    try:
        for episode_id in range(config.episodes):
            seed = config.seed_start + episode_id
            observation, _ = env.reset(seed=seed)
            start_tcp = _numpy(unwrapped.agent.tcp_pose.p).reshape(-1, 3)[0]
            goal = _numpy(unwrapped.goal_site.pose.p).reshape(-1, 3)[0]
            oracle_qpos = _numpy(unwrapped.oracle_goal_qpos).reshape(-1, 7)[0]
            full_qpos = _numpy(
                unwrapped.agent.robot.get_qpos()
            ).reshape(-1, 9)[0]
            certificate_qpos = full_qpos.copy()
            certificate_qpos[:7] = oracle_qpos
            certified_goal = expert.fk_tcp_world(certificate_qpos)
            fk_residual = float(np.linalg.norm(certified_goal - goal))
            initial_distance = float(np.linalg.norm(start_tcp - goal))
            attempts = int(_first(unwrapped.goal_sampling_attempts_used))

            if (
                not np.isfinite(oracle_qpos).all()
                or fk_residual > 2e-4
                or not (
                    config.goal_min_tcp_distance
                    <= initial_distance
                    <= config.goal_max_tcp_distance
                )
                or goal[2] < config.goal_min_height
            ):
                raise RuntimeError(
                    f"Episode {episode_id} failed its FK/range certificate"
                )

            success = False
            final_distance = initial_distance
            for step_index in range(config.max_episode_steps):
                progress = _minimum_jerk(
                    step_index + 1,
                    config.trajectory_steps,
                )
                desired_tcp = start_tcp + progress * (goal - start_tcp)
                state = _state_from_observation(observation)
                tcp = _numpy(unwrapped.agent.tcp_pose.p).reshape(-1, 3)[0]
                action = expert.action(desired_tcp)
                (
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)
                next_state = _state_from_observation(next_observation)
                next_tcp = _numpy(
                    unwrapped.agent.tcp_pose.p
                ).reshape(-1, 3)[0]
                transition_success = bool(_first(info["success"]))
                transition_terminal = bool(_first(terminated))
                transition_timeout = bool(_first(truncated))
                final_distance = float(
                    _first(info["tcp_to_goal_distance"])
                )

                transitions["state"].append(state)
                transitions["action"].append(action)
                transitions["action_rad"].append(
                    config.action_radians * action
                )
                transitions["next_state"].append(next_state)
                transitions["reward"].append(float(_first(reward)))
                transitions["terminal"].append(transition_terminal)
                transitions["timeout"].append(transition_timeout)
                transitions["success"].append(transition_success)
                transitions["episode_id"].append(episode_id)
                transitions["step_index"].append(step_index)
                transitions["goal_position"].append(goal.copy())
                transitions["desired_tcp_position"].append(
                    desired_tcp.astype(np.float32)
                )
                transitions["tcp_position"].append(tcp.copy())
                transitions["next_tcp_position"].append(next_tcp.copy())
                observation = next_observation

                if transition_success:
                    success = True
                    break
                if transition_terminal or transition_timeout:
                    break

            if not success:
                raise RuntimeError(
                    f"DLS expert failed seed {seed} after "
                    f"{step_index + 1} steps; final distance "
                    f"{final_distance:.6f} m"
                )

            episode_seed.append(seed)
            episode_length.append(step_index + 1)
            episode_goal.append(goal.copy())
            episode_start_tcp.append(start_tcp.copy())
            episode_oracle_qpos.append(oracle_qpos.copy())
            episode_sampling_attempts.append(attempts)
            episode_fk_residual.append(fk_residual)
            episode_initial_distance.append(initial_distance)
            episode_final_distance.append(final_distance)
    finally:
        env.close()

    train_ids, validation_ids, test_ids = _episode_splits(
        config.episodes,
        config.split_seed,
    )
    arrays: dict[str, np.ndarray] = {}
    float_vectors = {
        "state",
        "action",
        "action_rad",
        "next_state",
        "goal_position",
        "desired_tcp_position",
        "tcp_position",
        "next_tcp_position",
    }
    bool_scalars = {"terminal", "timeout", "success"}
    int_scalars = {"episode_id", "step_index"}
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
            "episode_seed": np.asarray(episode_seed, dtype=np.int64),
            "episode_length": np.asarray(episode_length, dtype=np.int64),
            "episode_goal": np.asarray(episode_goal, dtype=np.float32),
            "episode_start_tcp": np.asarray(
                episode_start_tcp,
                dtype=np.float32,
            ),
            # Audit-only reachability certificate. It is explicitly excluded
            # from state and expert action computation.
            "episode_oracle_goal_qpos": np.asarray(
                episode_oracle_qpos,
                dtype=np.float32,
            ),
            "episode_sampling_attempts": np.asarray(
                episode_sampling_attempts,
                dtype=np.int64,
            ),
            "episode_fk_residual": np.asarray(
                episode_fk_residual,
                dtype=np.float32,
            ),
            "episode_initial_distance": np.asarray(
                episode_initial_distance,
                dtype=np.float32,
            ),
            "episode_final_distance": np.asarray(
                episode_final_distance,
                dtype=np.float32,
            ),
            "train_episode_ids": train_ids,
            "validation_episode_ids": validation_ids,
            "test_episode_ids": test_ids,
        }
    )

    if arrays["state"].shape[1:] != (17,):
        raise RuntimeError("Collected state schema is not [N,17]")
    if arrays["action"].shape[1:] != (7,):
        raise RuntimeError("Collected action schema is not [N,7]")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError("Collected dataset contains NaN or Inf")

    np.savez_compressed(output_path, **arrays)
    lengths = arrays["episode_length"]
    summary = {
        "dataset_path": str(output_path.resolve()),
        "episodes": int(config.episodes),
        "transitions": int(len(arrays["state"])),
        "successes": int(config.episodes),
        "episode_length": {
            "mean": float(lengths.mean()),
            "minimum": int(lengths.min()),
            "p50": float(np.percentile(lengths, 50)),
            "p95": float(np.percentile(lengths, 95)),
            "maximum": int(lengths.max()),
        },
        "initial_goal_distance_m": {
            "mean": float(arrays["episode_initial_distance"].mean()),
            "minimum": float(arrays["episode_initial_distance"].min()),
            "maximum": float(arrays["episode_initial_distance"].max()),
        },
        "final_goal_distance_m": {
            "mean": float(arrays["episode_final_distance"].mean()),
            "maximum": float(arrays["episode_final_distance"].max()),
        },
        "sampler_attempts": {
            "mean": float(arrays["episode_sampling_attempts"].mean()),
            "maximum": int(arrays["episode_sampling_attempts"].max()),
        },
        "fk_certificate_residual_m": {
            "mean": float(arrays["episode_fk_residual"].mean()),
            "maximum": float(arrays["episode_fk_residual"].max()),
        },
        "timing": {
            "control_frequency_hz": control_frequency,
            "simulation_frequency_hz": simulation_frequency,
            "physics_steps_per_action": simulation_steps,
            "control_dt_seconds": control_dt,
        },
        "split_episode_counts": {
            "train": int(len(train_ids)),
            "validation": int(len(validation_ids)),
            "test": int(len(test_ids)),
        },
        "config": asdict(config),
        "state_definition": "[q(7), qdot(7), tcp_xyz-goal_xyz(3)]",
        "action_definition": (
            "actual normalized pd_joint_delta_pos command; "
            "action_rad = 0.1 * action"
        ),
        "expert_definition": (
            "minimum-jerk Cartesian waypoints tracked by deterministic DLS; "
            "hidden oracle q_goal is audit-only"
        ),
        "goal_schedule": "one FK-certified goal per reset; never switched mid-episode",
        "scope": "100-episode state-only pilot dataset",
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/pandareach_small/data/pandareach_dls_100.npz"
        ),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=20_260_731)
    parser.add_argument("--split-seed", type=int, default=31)
    parser.add_argument("--trajectory-steps", type=int, default=120)
    parser.add_argument("--max-episode-steps", type=int, default=160)
    parser.add_argument(
        "--collection-goal-threshold",
        type=float,
        default=0.005,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CollectionConfig(
        episodes=args.episodes,
        seed_start=args.seed_start,
        split_seed=args.split_seed,
        trajectory_steps=args.trajectory_steps,
        max_episode_steps=args.max_episode_steps,
        collection_goal_threshold=args.collection_goal_threshold,
    )
    summary = collect(config, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
