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
    arm joint velocities, and ``tcp_xyz - goal_xyz``.
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    goal_threshold = 0.03
    goal_joint_delta = 0.15
    goal_min_tcp_distance = 0.04
    goal_max_tcp_distance = 0.18
    goal_min_height = 0.10
    goal_joint_limit_margin = 0.05
    goal_sampling_attempts = 128
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
        goal_threshold: float | None = None,
        goal_joint_delta: float | None = None,
        goal_min_tcp_distance: float | None = None,
        goal_max_tcp_distance: float | None = None,
        goal_min_height: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.robot_init_qpos_noise = float(robot_init_qpos_noise)
        if goal_threshold is not None:
            self.goal_threshold = float(goal_threshold)
        if goal_joint_delta is not None:
            self.goal_joint_delta = float(goal_joint_delta)
        if goal_min_tcp_distance is not None:
            self.goal_min_tcp_distance = float(goal_min_tcp_distance)
        if goal_max_tcp_distance is not None:
            self.goal_max_tcp_distance = float(goal_max_tcp_distance)
        if goal_min_height is not None:
            self.goal_min_height = float(goal_min_height)
        if not 0 < self.goal_min_tcp_distance < self.goal_max_tcp_distance:
            raise ValueError(
                "goal distances must satisfy 0 < minimum < maximum"
            )
        if self.goal_joint_delta <= 0:
            raise ValueError("goal_joint_delta must be positive")
        if self.goal_threshold <= 0:
            raise ValueError("goal_threshold must be positive")
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

            if not hasattr(self, "_oracle_goal_qpos"):
                self._oracle_goal_qpos = torch.full_like(
                    self.agent.robot.get_qpos(),
                    torch.nan,
                )
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
                self._oracle_goal_qpos[env_idx] = torch.nan
            else:
                goal, goal_qpos = self._sample_reachable_goal(env_idx, qpos)
                self._oracle_goal_qpos[env_idx] = goal_qpos
            self.goal_site.set_pose(Pose.create_from_pq(p=goal))

    def _sample_reachable_goal(
        self,
        env_idx: torch.Tensor,
        start_qpos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        attempts_used = torch.zeros(
            batch_size,
            dtype=torch.int64,
            device=start_qpos.device,
        )

        for attempt in range(1, self.goal_sampling_attempts + 1):
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
            self._sync_gpu_kinematics()
            trial_goal = self.agent.tcp_pose.p[tcp_index].clone()
            distance = torch.linalg.norm(trial_goal - start_tcp, dim=-1)
            trial_valid = (
                (distance >= self.goal_min_tcp_distance)
                & (distance <= self.goal_max_tcp_distance)
                & (trial_goal[:, 2] >= self.goal_min_height)
            )
            accepted = (~valid) & trial_valid
            attempts_used = torch.where(
                accepted,
                torch.full_like(attempts_used, attempt),
                attempts_used,
            )
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
            failed = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
            self.agent.reset(start_qpos)
            raise RuntimeError(
                "Failed to sample a range-valid FK-certified goal for "
                f"batch indices {failed} after "
                f"{self.goal_sampling_attempts} attempts"
            )

        self.agent.reset(start_qpos)
        self._sync_gpu_kinematics()
        self._goal_sampling_attempts_used = attempts_used
        return goal, goal_qpos

    def _sync_gpu_kinematics(self) -> None:
        """Make link poses reflect qpos writes during GPU episode setup."""

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

    @property
    def oracle_goal_qpos(self) -> torch.Tensor:
        """Joint-space reachability certificate for expert generation only."""

        if not hasattr(self, "_oracle_goal_qpos"):
            raise RuntimeError("Call reset before requesting oracle_goal_qpos")
        return self._oracle_goal_qpos[..., :7]

    @property
    def goal_sampling_attempts_used(self) -> torch.Tensor:
        """Rejection-sampling attempts for each current FK-certified goal."""

        if not hasattr(self, "_goal_sampling_attempts_used"):
            raise RuntimeError("Call reset with a sampled goal first")
        return self._goal_sampling_attempts_used

    def _get_obs_agent(self):
        return {
            "qpos": self.agent.robot.get_qpos()[..., :7],
            "qvel": self.agent.robot.get_qvel()[..., :7],
        }

    def _get_obs_extra(self, info: dict):
        return {
            "tcp_to_goal": self.agent.tcp_pose.p - self.goal_site.pose.p,
        }

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


@register_env("ACMPC-PandaReach3-v0", max_episode_steps=220)
class PandaReachThreeWaypointEnv(PandaReachEnv):
    """Contact-free ordered three-waypoint task with local goal regions.

    Koopman identification should use only ``[q, qdot, tcp_xyz]``.  The
    active absolute Cartesian waypoint is task context and is intentionally
    exposed separately rather than being folded into the dynamics state.
    Each waypoint is sampled by applying a small joint-space perturbation to
    a fixed, FK-verified nominal configuration.  This gives three compact,
    distinct Cartesian regions while retaining a reachability certificate.
    """

    SUPPORTED_REWARD_MODES = ("sparse", "dense", "none")
    waypoint_count = 3
    waypoint_joint_jitter = 0.02
    waypoint_event_reward = 0.2
    dense_distance_penalty_scale = 0.05
    dense_waypoint_completion_reward = 1.0
    # Final-success radius (defaults to goal_threshold) and whether success
    # additionally requires the robot to be static. Both are configurable so
    # the success criterion can be relaxed independently of waypoint passing.
    success_goal_threshold = 0.01
    require_robot_static = True
    waypoint_qpos_centers = np.array(
        [
            [
                -0.13095243,
                0.28832006,
                -0.15331894,
                -2.0354195,
                0.07943689,
                2.2703905,
                0.66720384,
            ],
            [
                -0.15071833,
                0.29406947,
                0.14641319,
                -1.8554804,
                -0.12025512,
                2.3638175,
                0.8467025,
            ],
            [
                0.08170456,
                0.32289994,
                0.14957078,
                -2.0403616,
                0.01322,
                2.3765316,
                0.7515145,
            ],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        *args: Any,
        waypoint_joint_jitter: float | None = None,
        waypoint_event_reward: float | None = None,
        dense_distance_penalty_scale: float | None = None,
        dense_waypoint_completion_reward: float | None = None,
        success_goal_threshold: float | None = None,
        require_robot_static: bool | None = None,
        **kwargs: Any,
    ) -> None:
        if waypoint_joint_jitter is not None:
            self.waypoint_joint_jitter = float(waypoint_joint_jitter)
        if waypoint_event_reward is not None:
            self.waypoint_event_reward = float(waypoint_event_reward)
        if dense_distance_penalty_scale is not None:
            self.dense_distance_penalty_scale = float(
                dense_distance_penalty_scale
            )
        if dense_waypoint_completion_reward is not None:
            self.dense_waypoint_completion_reward = float(
                dense_waypoint_completion_reward
            )
        if self.waypoint_joint_jitter <= 0:
            raise ValueError("waypoint_joint_jitter must be positive")
        if self.waypoint_event_reward < 0:
            raise ValueError("waypoint_event_reward must be non-negative")
        if self.dense_distance_penalty_scale < 0:
            raise ValueError(
                "dense_distance_penalty_scale must be non-negative"
            )
        if self.dense_waypoint_completion_reward <= 0:
            raise ValueError(
                "dense_waypoint_completion_reward must be positive"
            )
        super().__init__(*args, **kwargs)
        # The final-success radius defaults to the waypoint-passing threshold.
        # Set it after the base class because goal_threshold is set there.
        if success_goal_threshold is not None:
            if success_goal_threshold <= 0:
                raise ValueError("success_goal_threshold must be positive")
            self.success_goal_threshold = float(success_goal_threshold)
        else:
            self.success_goal_threshold = self.goal_threshold
        self.require_robot_static = (
            True if require_robot_static is None else bool(require_robot_static)
        )

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)
        self.goal_site_1 = actors.build_sphere(
            self.scene,
            radius=self.goal_threshold,
            color=[1.0, 0.65, 0.05, 1.0],
            name="goal_site_1",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.0, 0.0, 0.2]),
        )
        self.goal_site_2 = actors.build_sphere(
            self.scene,
            radius=self.goal_threshold,
            color=[0.15, 0.55, 1.0, 1.0],
            name="goal_site_2",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.0, 0.0, 0.25]),
        )
        self._waypoint_sites = (
            self.goal_site,
            self.goal_site_1,
            self.goal_site_2,
        )
        self._hidden_objects.extend((self.goal_site_1, self.goal_site_2))

    def _initialize_episode(
        self,
        env_idx: torch.Tensor,
        options: dict,
    ) -> None:
        if "goal_pos" in options:
            raise ValueError(
                "PandaReachThreeWaypointEnv uses three local waypoint regions; "
                "goal_pos overrides are not supported"
            )
        with torch.device(self.device):
            batch_size = len(env_idx)
            start_qpos = torch.as_tensor(self.rest_qpos).repeat(batch_size, 1)
            if self.robot_init_qpos_noise:
                start_qpos[:, :7] += (
                    2.0 * torch.rand((batch_size, 7)) - 1.0
                ) * self.robot_init_qpos_noise
            start_qpos[:, -2:] = 0.04
            self.agent.reset(start_qpos)

            qlimits = self.agent.robot.get_qlimits()[env_idx, :7]
            lower = qlimits[..., 0] + self.goal_joint_limit_margin
            upper = qlimits[..., 1] - self.goal_joint_limit_margin
            centers = torch.as_tensor(
                self.waypoint_qpos_centers,
                dtype=start_qpos.dtype,
            )
            candidates = centers.unsqueeze(0).repeat(batch_size, 1, 1)
            candidates += (
                2.0 * torch.rand((batch_size, self.waypoint_count, 7))
                - 1.0
            ) * self.waypoint_joint_jitter
            candidates = torch.maximum(
                torch.minimum(candidates, upper[:, None, :]),
                lower[:, None, :],
            )

            waypoints = start_qpos.new_empty(
                batch_size, self.waypoint_count, 3
            )
            trial_qpos = start_qpos.clone()
            tcp_index = env_idx.to(
                device=self.agent.tcp_pose.p.device,
                dtype=torch.long,
            )
            for waypoint_index in range(self.waypoint_count):
                trial_qpos[:, :7] = candidates[:, waypoint_index]
                self.agent.reset(trial_qpos)
                self._sync_gpu_kinematics()
                waypoints[:, waypoint_index] = self.agent.tcp_pose.p[
                    tcp_index
                ].clone()
            self.agent.reset(start_qpos)
            self._sync_gpu_kinematics()

            if not hasattr(self, "_waypoints"):
                self._waypoints = torch.full(
                    (self.num_envs, self.waypoint_count, 3),
                    torch.nan,
                    device=start_qpos.device,
                    dtype=start_qpos.dtype,
                )
                self._oracle_waypoint_qpos = torch.full(
                    (self.num_envs, self.waypoint_count, 7),
                    torch.nan,
                    device=start_qpos.device,
                    dtype=start_qpos.dtype,
                )
                self._active_waypoint_index = torch.zeros(
                    self.num_envs,
                    device=start_qpos.device,
                    dtype=torch.long,
                )
                self._last_stage_update_step = torch.full(
                    (self.num_envs,),
                    -1,
                    device=start_qpos.device,
                    dtype=torch.long,
                )
            self._waypoints[env_idx] = waypoints
            self._oracle_waypoint_qpos[env_idx] = candidates
            self._active_waypoint_index[env_idx] = 0
            self._last_stage_update_step[env_idx] = -1
            for waypoint_index, site in enumerate(self._waypoint_sites):
                site.set_pose(
                    Pose.create_from_pq(p=waypoints[:, waypoint_index])
                )

    @property
    def waypoints(self) -> torch.Tensor:
        return self._waypoints

    @property
    def oracle_waypoint_qpos(self) -> torch.Tensor:
        return self._oracle_waypoint_qpos

    @property
    def active_waypoint_index(self) -> torch.Tensor:
        return self._active_waypoint_index

    @property
    def active_waypoint(self) -> torch.Tensor:
        rows = torch.arange(self.num_envs, device=self._waypoints.device)
        return self._waypoints[rows, self._active_waypoint_index]

    def _get_obs_extra(self, info: dict):
        return {
            "tcp_pos": self.agent.tcp_pose.p,
            "active_goal": self.active_waypoint,
            "active_waypoint_index": self._active_waypoint_index,
            "waypoints": self._waypoints,
        }

    def evaluate(self):
        tcp = self.agent.tcp_pose.p
        active_before = self.active_waypoint
        distance_before = torch.linalg.norm(tcp - active_before, dim=-1)
        intermediate = self._active_waypoint_index < self.waypoint_count - 1
        new_step = self._last_stage_update_step != self._elapsed_steps
        waypoint_passed = (
            new_step & intermediate & (distance_before <= self.goal_threshold)
        )
        self._active_waypoint_index = torch.where(
            waypoint_passed,
            self._active_waypoint_index + 1,
            self._active_waypoint_index,
        )
        self._last_stage_update_step = torch.where(
            new_step,
            self._elapsed_steps,
            self._last_stage_update_step,
        )

        active_after = self.active_waypoint
        active_distance = torch.linalg.norm(tcp - active_after, dim=-1)
        final_stage = self._active_waypoint_index == self.waypoint_count - 1
        if self.require_robot_static:
            is_robot_static = self.agent.is_static(0.2)
            static_ok = is_robot_static
        else:
            is_robot_static = torch.ones_like(
                final_stage, dtype=torch.bool
            )
            static_ok = True
        success = (
            final_stage
            & (active_distance <= self.success_goal_threshold)
            & static_ok
        )
        return {
            "success": success,
            "waypoint_passed": waypoint_passed,
            "active_waypoint_index": self._active_waypoint_index.clone(),
            "waypoints_completed": self._active_waypoint_index
            + success.to(torch.long),
            "active_waypoint_distance": active_distance,
            "reached_waypoint_distance": distance_before,
            "is_robot_static": is_robot_static,
        }

    def compute_sparse_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        return (
            info["success"].float()
            + self.waypoint_event_reward * info["waypoint_passed"].float()
        )

    def compute_dense_reward(
        self,
        obs: Any,
        action: torch.Tensor,
        info: dict,
    ):
        """Penalize current-goal distance and reward every passed waypoint.

        ``waypoint_passed`` covers the first two stages and ``success`` covers
        the final stage, so each ordered waypoint produces one positive event.
        Distance is evaluated after a stage transition and always refers to
        the currently active goal.
        """

        completion = (
            info["waypoint_passed"].float() + info["success"].float()
        )
        return (
            -self.dense_distance_penalty_scale
            * info["active_waypoint_distance"]
            + self.dense_waypoint_completion_reward * completion
        )


class PandaArmOnlyActionWrapper(gym.ActionWrapper):
    """Expose only the seven arm actions and keep the gripper fully open."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("PandaReach requires a continuous Box action space")
        if not env.action_space.shape or env.action_space.shape[-1] != 8:
            raise ValueError(
                "Expected Panda pd_joint_delta_pos action trailing shape 8, "
                f"got {env.action_space.shape}"
            )
        self.action_space = gym.spaces.Box(
            low=np.asarray(env.action_space.low[..., :-1], dtype=np.float32),
            high=np.asarray(env.action_space.high[..., :-1], dtype=np.float32),
            dtype=np.float32,
        )
        single_action_space = env.get_wrapper_attr("single_action_space")
        self.single_action_space = gym.spaces.Box(
            low=np.asarray(single_action_space.low[:-1], dtype=np.float32),
            high=np.asarray(single_action_space.high[:-1], dtype=np.float32),
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
