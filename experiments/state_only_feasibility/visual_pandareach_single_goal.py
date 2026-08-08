"""Single-goal visual PandaReach environment for isolating the target channel.

A single local waypoint region, sensor-visible and enlarged like the
three-waypoint visual variant, so a visual policy must estimate the goal from
the camera.  Reuses the three-waypoint machinery with ``waypoint_count=1``;
the robot state stays ``q_qdot_tcp`` (no goal in the state), so the frozen
``koopman_coverage`` model transfers unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaReachEnv,
    PandaReachThreeWaypointEnv,
)


@register_env("ACMPC-VisualPandaReach1-v0", max_episode_steps=220)
class VisualPandaReachSingleGoalEnv(PandaReachThreeWaypointEnv):
    """PandaReach3 machinery with a single local goal region.

    Physics/reward/success are unchanged from the three-waypoint family
    (TCP-distance + static), the goal marker is sensor-visible and enlarged,
    and the agent camera is configurable (default 256x256).
    """

    waypoint_count = 1
    waypoint_qpos_centers = PandaReachThreeWaypointEnv.waypoint_qpos_centers[:1]

    def __init__(
        self,
        *args,
        goal_marker_scale: float | None = None,
        camera_size: int | None = None,
        goal_region_radius: tuple[float, float, float] | None = None,
        **kwargs,
    ) -> None:
        self.goal_marker_scale = (
            4.0 if goal_marker_scale is None else float(goal_marker_scale)
        )
        self.camera_size = int(camera_size or 256)
        # Workspace-space goal sampling box (metres, per axis).  Enlarging the
        # region makes the goal the decisive variable, so a policy must read
        # the target channel instead of converging to a nearly fixed point.
        self.goal_region_radius = (
            (0.06, 0.06, 0.03)
            if goal_region_radius is None
            else tuple(float(value) for value in goal_region_radius)
        )
        if self.goal_marker_scale <= 0:
            raise ValueError("goal_marker_scale must be positive")
        if self.camera_size < 64:
            raise ValueError("camera_size must be at least 64")
        if len(self.goal_region_radius) != 3 or any(
            value <= 0 for value in self.goal_region_radius
        ):
            raise ValueError("goal_region_radius must be three positive values")
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        configs = super()._default_sensor_configs
        return tuple(
            replace(config, width=self.camera_size, height=self.camera_size)
            for config in configs
        )

    def _load_scene(self, options: dict) -> None:
        # Build only the base single-goal scene (PandaReachEnv hides goal_site).
        PandaReachEnv._load_scene(self, options)
        if not hasattr(self, "goal_site"):
            raise RuntimeError("Base scene did not build goal_site")
        # Replace the small hidden marker with a larger visible one.  The
        # original stays in _hidden_objects (invisible); the replacement is
        # wired into _waypoint_sites so _initialize_episode repositions it.
        radius = self.goal_threshold * self.goal_marker_scale
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=radius,
            color=(0.05, 1.0, 0.15, 1.0),
            name="goal_site_vis",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.0, 0.0, 0.2]),
        )
        self._waypoint_sites = (self.goal_site,)
        if any(
            self.goal_site is hidden for hidden in self._hidden_objects
        ):
            raise RuntimeError("replacement goal site must not be hidden")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            batch_size = len(env_idx)
            offset = (
                2.0 * torch.rand((batch_size, self.waypoint_count, 3)) - 1.0
            ) * torch.as_tensor(
                self.goal_region_radius,
                dtype=torch.float32,
                device=self.device,
            )
            self._waypoints[env_idx] = self._waypoints[env_idx] + offset
            for waypoint_index, site in enumerate(self._waypoint_sites):
                site.set_pose(
                    Pose.create_from_pq(
                        p=self._waypoints[env_idx][:, waypoint_index]
                    )
                )


__all__ = ["VisualPandaReachSingleGoalEnv"]
