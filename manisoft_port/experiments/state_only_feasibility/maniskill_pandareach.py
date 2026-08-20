from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("ACMPC-PandaReach-v0", max_episode_steps=100)
class PandaReachEnv(BaseEnv):
    """Contact-free Panda end-effector reaching task.

    The physical scene contains only a fixed-base Franka Panda. The floor-like
    platform and green goal sphere are visual-only actors with no collision
    geometry. State observations contain the seven arm joint positions, seven
    arm joint velocities, and the Cartesian goal position.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    goal_threshold = 0.03
    goal_joint_delta = 0.35
    goal_min_tcp_distance = 0.12
    goal_max_tcp_distance = 0.45
    goal_min_height = 0.10
    goal_joint_limit_margin = 0.05
    goal_sampling_attempts = 32
    rest_qpos = np.array(
        [
            0.0,
            np.pi / 8,
            0.0,
            -5 * np.pi / 8,
            0.0,
            3 * np.pi / 4,
            np.pi / 4,
            0.04,
            0.04,
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        *args: Any,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.01,
        **kwargs: Any,
    ) -> None:
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.3, 0.0, 0.65],
            target=[-0.05, 0.0, 0.25],
        )
        return [
            CameraConfig(
                "base_camera",
                pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.6, 0.7, 0.6],
            target=[0.0, 0.0, 0.35],
        )
        return CameraConfig(
            "render_camera",
            pose,
            width=512,
            height=512,
            fov=1.0,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict) -> None:
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.0, 0.0]))

    def _load_scene(self, options: dict) -> None:
        # This platform is a visual reference only. In particular, it has no
        # collision shape and therefore cannot inject contact dynamics.
        self.visual_platform = actors.build_box(
            self.scene,
            half_sizes=[0.85, 0.62, 0.02],
            color=[0.32, 0.36, 0.42, 1.0],
            name="visual_platform",
            body_type="static",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-0.10, 0.0, -0.03]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_threshold,
            color=[0.05, 1.0, 0.15, 1.0],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.12, -0.12, 0.36]),
        )
        # The goal is visible in human/rgb_array renders but hidden from future
        # visual-policy sensors. State policies receive goal_pos explicitly.
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ) -> None:
        with torch.device(self.device):
            batch_size = len(env_idx)
            qpos = torch.as_tensor(self.rest_qpos).repeat(batch_size, 1)
            if self.robot_init_qpos_noise:
                qpos[:, :7] += (
                    torch.rand((batch_size, 7)) * 2.0 - 1.0
                ) * self.robot_init_qpos_noise
            qpos[:, -2:] = 0.04
            self.agent.reset(qpos)

            if "goal_pos" in options:
                goal = torch.as_tensor(
                    options["goal_pos"],
                    dtype=torch.float32,
                )
                if goal.shape == (3,):
                    goal = goal.repeat(batch_size, 1)
                if goal.shape != (batch_size, 3):
                    raise ValueError(
                        "goal_pos must have shape [3] or [num_reset_envs, 3]"
                    )
            else:
                goal = self._sample_reachable_goal(env_idx, qpos)
            self.goal_site.set_pose(Pose.create_from_pq(p=goal))

    def _sample_reachable_goal(
        self,
        env_idx: torch.Tensor,
        start_qpos: torch.Tensor,
    ) -> torch.Tensor:
        """Generate goal positions by forward kinematics from valid qpos.

        A Cartesian box includes unreachable points and makes an environment
        failure look like a policy failure. Instead, this sampler perturbs the
        reset configuration inside the Panda joint limits, obtains the TCP
        position through simulator FK, and restores the actual reset state.
        The sampled joint configuration is deliberately not part of the policy
        observation; it only certifies geometric reachability of goal_pos.
        """

        batch_size = len(env_idx)
        tcp_index = env_idx.to(
            device=self.agent.tcp_pose.p.device,
            dtype=torch.long,
        )
        start_tcp = self.agent.tcp_pose.p[tcp_index].clone()
        qlimits = self.agent.robot.get_qlimits()[tcp_index, :7]
        lower = qlimits[..., 0] + self.goal_joint_limit_margin
        upper = qlimits[..., 1] - self.goal_joint_limit_margin

        goal_qpos = start_qpos.clone()
        goal = start_tcp.clone()
        valid = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=start_qpos.device,
        )

        for _ in range(self.goal_sampling_attempts):
            candidate = start_qpos.clone()
            candidate[:, :7] += (
                2.0 * torch.rand((batch_size, 7)) - 1.0
            ) * self.goal_joint_delta
            candidate[:, :7] = torch.maximum(
                torch.minimum(candidate[:, :7], upper),
                lower,
            )
            candidate[:, -2:] = 0.04
            trial_qpos = torch.where(
                valid[:, None],
                goal_qpos,
                candidate,
            )

            self.agent.reset(trial_qpos)
            trial_goal = self.agent.tcp_pose.p[tcp_index].clone()
            distance = torch.linalg.norm(trial_goal - start_tcp, dim=-1)
            trial_valid = (
                (distance >= self.goal_min_tcp_distance)
                & (distance <= self.goal_max_tcp_distance)
                & (trial_goal[:, 2] >= self.goal_min_height)
            )
            accepted = (~valid) & trial_valid
            goal_qpos = torch.where(
                accepted[:, None],
                trial_qpos,
                goal_qpos,
            )
            goal = torch.where(accepted[:, None], trial_goal, goal)
            valid |= trial_valid
            if bool(torch.all(valid)):
                break

        if not bool(torch.all(valid)):
            # A deterministic, limit-safe fallback prevents a rare rejection
            # streak from returning the reset TCP itself as the target.
            fallback_delta = torch.tensor(
                [0.25, -0.20, 0.15, 0.25, -0.15, 0.20, -0.25],
                dtype=start_qpos.dtype,
                device=start_qpos.device,
            )
            fallback = start_qpos.clone()
            fallback[:, :7] = torch.maximum(
                torch.minimum(
                    fallback[:, :7] + fallback_delta,
                    upper,
                ),
                lower,
            )
            fallback[:, -2:] = 0.04
            goal_qpos = torch.where(valid[:, None], goal_qpos, fallback)
            self.agent.reset(goal_qpos)
            fallback_goal = self.agent.tcp_pose.p[tcp_index].clone()
            goal = torch.where(valid[:, None], goal, fallback_goal)

        self.agent.reset(start_qpos)
        return goal

    def _get_obs_agent(self):
        return {
            "qpos": self.agent.robot.get_qpos()[..., :7],
            "qvel": self.agent.robot.get_qvel()[..., :7],
        }

    def _get_obs_extra(self, info: dict):
        return {"goal_pos": self.goal_site.pose.p}

    def evaluate(self):
        distance = torch.linalg.norm(
            self.agent.tcp_pose.p - self.goal_site.pose.p,
            dim=-1,
        )
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": (distance <= self.goal_threshold) & is_robot_static,
            "tcp_to_goal_distance": distance,
            "is_robot_static": is_robot_static,
        }

    def compute_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        distance = info["tcp_to_goal_distance"]
        reaching_reward = 1.0 - torch.tanh(5.0 * distance)
        arm_speed = torch.linalg.norm(
            self.agent.robot.get_qvel()[..., :7],
            dim=-1,
        )
        settling_reward = 1.0 - torch.tanh(5.0 * arm_speed)
        near_goal = distance <= 2.0 * self.goal_threshold
        reward = reaching_reward + near_goal.float() * settling_reward
        reward[info["success"]] = 2.0
        return reward

    def compute_normalized_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return self.compute_dense_reward(obs, action, info) / 2.0


class PandaArmOnlyActionWrapper(gym.ActionWrapper):
    """Expose only the seven arm actions and keep the gripper fully open."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("PandaReach requires a continuous Box action space")
        if env.action_space.shape != (8,):
            raise ValueError(
                "Expected Panda pd_joint_delta_pos action shape (8,), "
                f"got {env.action_space.shape}"
            )
        self.action_space = gym.spaces.Box(
            low=np.asarray(env.action_space.low[:-1], dtype=np.float32),
            high=np.asarray(env.action_space.high[:-1], dtype=np.float32),
            dtype=np.float32,
        )

    def action(self, action):
        if isinstance(action, torch.Tensor):
            gripper = torch.ones_like(action[..., :1])
            return torch.cat((action, gripper), dim=-1)
        arm_action = np.asarray(action, dtype=np.float32)
        if arm_action.shape[-1] != 7:
            raise ValueError("Arm-only action must have trailing dimension 7")
        gripper = np.ones((*arm_action.shape[:-1], 1), dtype=np.float32)
        return np.concatenate((arm_action, gripper), axis=-1)
