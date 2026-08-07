"""Smoke test for the visual PandaReach3 environment and observation extraction."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from experiments.state_only_feasibility.collect_visual_pandareach_threewaypoint import (
    _extract_observation,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
)
from experiments.state_only_feasibility.visual_pandareach_env import (
    VisualPandaReachThreeWaypointEnv,
)


def main() -> None:
    env = PandaArmOnlyActionWrapper(
        gym.make(
            "ACMPC-VisualPandaReach3-v0",
            num_envs=1,
            obs_mode="rgb+depth",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode=None,
            max_episode_steps=220,
            goal_threshold=0.01,
            waypoint_joint_jitter=0.02,
        )
    )
    try:
        observation, _ = env.reset(seed=0)
        print("obs keys:", sorted(observation.keys()))
        print("extra keys:", sorted(observation["extra"].keys()))
        print(
            "sensor keys:",
            sorted(observation["sensor_data"]["base_camera"].keys()),
        )
        robot, rgb, depth = _extract_observation(observation)
        print(f"robot {robot.shape} {robot.dtype}")
        print(f"rgb {rgb.shape} {rgb.dtype} depth {depth.shape} {depth.dtype}")
        print(f"depth range: {depth.min()}..{depth.max()} (mm)")
        active_goal = np.asarray(observation["extra"]["active_goal"]).reshape(-1)
        print("active_goal:", active_goal)
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        print("step ok; info keys:", sorted(info.keys()))
        print("active_waypoint_distance:", float(np.asarray(info["active_waypoint_distance"]).reshape(-1)[0]))
    finally:
        env.close()
    print("ENV SMOKE OK")


if __name__ == "__main__":
    main()
