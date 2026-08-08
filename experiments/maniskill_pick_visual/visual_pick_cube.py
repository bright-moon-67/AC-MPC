"""A PickCube variant whose goal marker is visible to agent cameras."""

from __future__ import annotations

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env


VISUAL_PICK_CUBE_ENV_ID = "ACMPC-VisualPickCube-v1"


@register_env(VISUAL_PICK_CUBE_ENV_ID, max_episode_steps=50)
class VisualPickCubeEnv(PickCubeEnv):
    """PickCube with unchanged physics and a sensor-visible goal sphere.

    Upstream ``PickCube-v1`` intentionally appends ``goal_site`` to
    ``_hidden_objects``.  That makes the numerical ``extra/goal_pos`` the only
    source of goal information.  This variant removes only that rendering
    exclusion, allowing a visual latent to encode the cube and goal while the
    task dynamics, randomization, reward, and success condition remain
    unchanged.
    """

    def _load_scene(self, options: dict) -> None:
        super()._load_scene(options)
        self._hidden_objects = [
            actor for actor in self._hidden_objects if actor is not self.goal_site
        ]
        if any(actor is self.goal_site for actor in self._hidden_objects):
            raise RuntimeError("goal_site must be visible to agent sensors")


__all__ = ["VISUAL_PICK_CUBE_ENV_ID", "VisualPickCubeEnv"]

