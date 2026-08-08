"""Visual PandaReach three-waypoint environment with sensor-visible goal markers.

The upstream ``PandaReachThreeWaypointEnv`` hides the three goal spheres from
agent cameras (state policies receive goal positions as privileged context).
This variant keeps those originals hidden and instead builds enlarged,
sensor-visible replacement markers at a higher camera resolution, so a visual
policy can encode the active waypoint directly from the image.  The numerical
goal positions in ``observation["extra"]`` remain available for training-only
supervision (the visual ``pos_branch``), but never enter the model at
inference time.

Physics, reward and success are unchanged: the markers have no collision
geometry and every success / waypoint decision uses the TCP distance to the
numerical waypoint position, never the visual sphere.
"""

from __future__ import annotations

from dataclasses import replace

import sapien
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env

from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaReachThreeWaypointEnv,
)

_GOAL_COLORS = (
    (0.05, 1.0, 0.15, 1.0),  # goal_site   (green)
    (1.0, 0.65, 0.05, 1.0),  # goal_site_1 (orange)
    (0.15, 0.55, 1.0, 1.0),  # goal_site_2 (blue)
)


@register_env("ACMPC-VisualPandaReach3-v0", max_episode_steps=220)
class VisualPandaReachThreeWaypointEnv(PandaReachThreeWaypointEnv):
    """PandaReach3 with unchanged physics and visible enlarged goal markers.

    ``goal_marker_scale`` enlarges only the rendered spheres (no collision;
    success still uses the 1 cm TCP-distance threshold) and ``camera_size``
    raises the agent-camera resolution.  Both are visual-only: physics, reward
    and success semantics are identical to the base environment.
    """

    def __init__(
        self,
        *args,
        goal_marker_scale: float | None = None,
        camera_size: int | None = None,
        **kwargs,
    ) -> None:
        self.goal_marker_scale = (
            3.0 if goal_marker_scale is None else float(goal_marker_scale)
        )
        self.camera_size = int(camera_size or 256)
        if self.goal_marker_scale <= 0:
            raise ValueError("goal_marker_scale must be positive")
        if self.camera_size < 64:
            raise ValueError("camera_size must be at least 64")
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        configs = super()._default_sensor_configs
        return tuple(
            replace(config, width=self.camera_size, height=self.camera_size)
            for config in configs
        )

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)
        original_sites = tuple(getattr(self, "_waypoint_sites", ()))
        if len(original_sites) != 3:
            raise RuntimeError("Expected three waypoint goal sites after _load_scene")
        # The base class already appended the originals to _hidden_objects, so
        # they stay invisible.  Build enlarged visible replacements and wire
        # them into the attribute names the episode initializer uses.
        radius = self.goal_threshold * self.goal_marker_scale
        names = ("goal_site_vis", "goal_site_1_vis", "goal_site_2_vis")
        replacements = [
            actors.build_sphere(
                self.scene,
                radius=radius,
                color=color,
                name=name,
                body_type="kinematic",
                add_collision=False,
                # Harmless placeholder; _initialize_episode repositions these
                # sites to the sampled waypoints every episode.
                initial_pose=sapien.Pose(p=[0.0, 0.0, 0.2]),
            )
            for name, color in zip(names, _GOAL_COLORS)
        ]
        self.goal_site, self.goal_site_1, self.goal_site_2 = replacements
        self._waypoint_sites = tuple(replacements)
        for site in replacements:
            if any(site is hidden for hidden in self._hidden_objects):
                raise RuntimeError("replacement goal sites must not be hidden")


__all__ = ["VisualPandaReachThreeWaypointEnv"]
