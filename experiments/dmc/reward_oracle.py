"""Exact reward oracles for MPVE rollouts on DMC observations.

The dm_control public task API evaluates rewards from a live ``Physics``
instance.  Some tasks nevertheless expose every reward-relevant quantity in
their canonical observation.  For those tasks we reproduce the official task
formula directly and require transition-level parity against the live suite
before formal training is approved.

The learned ``TransitionRewardModel`` remains available as an explicit
alternative for tasks whose observation is not a sufficient reward statistic
and for later offline/real-world experiments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn


OFFICIAL_OBSERVATION_ORACLE = "dmc_official_observation_oracle_v1"
LEARNED_TRANSITION_REWARD = "learned_transition_model_v1"
MPVE_REWARD_SOURCES = frozenset(
    {OFFICIAL_OBSERVATION_ORACLE, LEARNED_TRANSITION_REWARD}
)

EXACT_REWARD_ORACLE_TASKS = frozenset({"cartpole_swingup", "walker_run"})
ORACLE_PARITY_MAX_ABS_ERROR = 2e-7


def _gaussian_tolerance_zero_bound(
    value: torch.Tensor,
    *,
    margin: float,
    value_at_margin: float = 0.1,
) -> torch.Tensor:
    """Torch equivalent of DMC ``rewards.tolerance`` for this exact case."""

    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    if not 0.0 < value_at_margin < 1.0:
        raise ValueError("value_at_margin must lie strictly between zero and one")
    # For bounds=(0, 0), DMC's Gaussian sigmoid simplifies to
    # exp(log(value_at_margin) * (abs(x) / margin) ** 2).
    return torch.exp(
        math.log(value_at_margin) * (value.abs() / float(margin)).square()
    )


def cartpole_swingup_official_reward(
    next_observation: torch.Tensor,
    applied_action: torch.Tensor,
) -> torch.Tensor:
    """Official dense Cartpole Swingup reward from canonical next observation.

    Canonical observation layout is
    ``[cart_position, pole_cos, pole_sin, cart_velocity, angular_velocity]``.
    dm_control evaluates the task reward after stepping physics, hence the
    reward-state terms use ``next_observation`` while ``small_control`` uses the
    action applied on that transition.
    """

    if next_observation.ndim < 1 or next_observation.shape[-1] != 5:
        raise ValueError("Cartpole reward expects next_observation[..., 5]")
    if applied_action.ndim < 1 or applied_action.shape[-1] != 1:
        raise ValueError("Cartpole reward expects applied_action[..., 1]")
    if next_observation.shape[:-1] != applied_action.shape[:-1]:
        raise ValueError("Cartpole reward observation/action batch shapes disagree")
    if not torch.isfinite(next_observation).all() or not torch.isfinite(
        applied_action
    ).all():
        raise FloatingPointError("Cartpole reward inputs contain NaN or Inf")

    cart_position = next_observation[..., 0]
    pole_cosine = next_observation[..., 1]
    angular_velocity = next_observation[..., 4]
    action = applied_action[..., 0]

    upright = (pole_cosine + 1.0) / 2.0
    centered = (
        1.0
        + _gaussian_tolerance_zero_bound(cart_position, margin=2.0)
    ) / 2.0
    # Official actions are clipped to [-1, 1].  DMC's quadratic tolerance with
    # margin=1 and value_at_margin=0 is max(0, 1 - action**2).
    small_control = (4.0 + torch.clamp(1.0 - action.square(), min=0.0)) / 5.0
    small_velocity = (
        1.0
        + _gaussian_tolerance_zero_bound(angular_velocity, margin=5.0)
    ) / 2.0
    return upright * small_control * small_velocity * centered


# Exact mass/geometry coefficients for dm_control's planar Walker model.
# They are the mass-weighted horizontal COM-velocity contributions of the
# torso-to-hip offset, thigh COM/knee, leg COM/ankle, and foot COM offset.
# Expanding the planar forward kinematics gives the same value as
# ``physics.named.data.subtree_linvel["torso"][0]`` without a MuJoCo forward.
_WALKER_TORSO_ANGLE_COEFFICIENT = -0.1878109635282894
_WALKER_THIGH_ANGLE_COEFFICIENT = -0.10886750568723857
_WALKER_LEG_ANGLE_COEFFICIENT = -0.06105525794378807
_WALKER_FOOT_ANGLE_COEFFICIENT = -0.004403023409407794


def _walker_subtree_horizontal_velocity_numpy(observation: Any) -> np.ndarray:
    """Vectorized exact torso-subtree COM horizontal velocity."""

    obs = np.asarray(observation)
    velocity = obs[..., 15:24]
    torso_rate = velocity[..., 2]
    right_thigh_rate = torso_rate - velocity[..., 3]
    right_leg_rate = right_thigh_rate - velocity[..., 4]
    right_foot_rate = right_leg_rate - velocity[..., 5]
    left_thigh_rate = torso_rate - velocity[..., 6]
    left_leg_rate = left_thigh_rate - velocity[..., 7]
    left_foot_rate = left_leg_rate - velocity[..., 8]
    return (
        velocity[..., 1]
        + _WALKER_TORSO_ANGLE_COEFFICIENT * obs[..., 0] * torso_rate
        + _WALKER_THIGH_ANGLE_COEFFICIENT
        * (
            obs[..., 2] * right_thigh_rate
            + obs[..., 8] * left_thigh_rate
        )
        + _WALKER_LEG_ANGLE_COEFFICIENT
        * (obs[..., 4] * right_leg_rate + obs[..., 10] * left_leg_rate)
        + _WALKER_FOOT_ANGLE_COEFFICIENT
        * (obs[..., 7] * right_foot_rate + obs[..., 13] * left_foot_rate)
    )


def _walker_subtree_horizontal_velocity_torch(
    observation: torch.Tensor,
) -> torch.Tensor:
    """Torch counterpart of the exact vectorized Walker COM velocity."""

    velocity = observation[..., 15:24]
    torso_rate = velocity[..., 2]
    right_thigh_rate = torso_rate - velocity[..., 3]
    right_leg_rate = right_thigh_rate - velocity[..., 4]
    right_foot_rate = right_leg_rate - velocity[..., 5]
    left_thigh_rate = torso_rate - velocity[..., 6]
    left_leg_rate = left_thigh_rate - velocity[..., 7]
    left_foot_rate = left_leg_rate - velocity[..., 8]
    return (
        velocity[..., 1]
        + _WALKER_TORSO_ANGLE_COEFFICIENT * observation[..., 0] * torso_rate
        + _WALKER_THIGH_ANGLE_COEFFICIENT
        * (
            observation[..., 2] * right_thigh_rate
            + observation[..., 8] * left_thigh_rate
        )
        + _WALKER_LEG_ANGLE_COEFFICIENT
        * (
            observation[..., 4] * right_leg_rate
            + observation[..., 10] * left_leg_rate
        )
        + _WALKER_FOOT_ANGLE_COEFFICIENT
        * (
            observation[..., 7] * right_foot_rate
            + observation[..., 13] * left_foot_rate
        )
    )


def walker_run_exact_reward_numpy(
    next_observation: Any,
    applied_action: Any,
) -> Any:
    """Vectorized official PlanarWalker run reward (no MuJoCo forward)."""

    obs = np.asarray(next_observation, dtype=np.float64)
    action = np.asarray(applied_action)
    if obs.ndim < 1 or obs.shape[-1] != 24:
        raise ValueError("Walker reward expects next_observation[..., 24]")
    if action.ndim < 1 or action.shape[-1] != 6:
        raise ValueError("Walker reward expects applied_action[..., 6]")
    if obs.shape[:-1] != action.shape[:-1]:
        raise ValueError("Walker reward observation/action batch shapes disagree")
    if not np.isfinite(obs).all() or not np.isfinite(action).all():
        raise FloatingPointError("Walker reward inputs contain NaN or Inf")
    flat = obs.reshape(-1, 24)
    height = flat[:, 14]
    upright = (1.0 + flat[:, 0]) / 2.0
    standing_distance = np.maximum(1.2 - height, 0.0) / 0.6
    standing = np.exp(np.log(0.1) * standing_distance**2)
    speed = _walker_subtree_horizontal_velocity_numpy(flat)
    move = np.clip(speed / 8.0, 0.0, 1.0)
    reward = ((3.0 * standing + upright) / 4.0) * ((5.0 * move + 1.0) / 6.0)
    return reward.astype(np.float32).reshape(obs.shape[:-1])


def walker_run_official_reward(
    next_observation: torch.Tensor,
    applied_action: torch.Tensor,
) -> torch.Tensor:
    """Official PlanarWalker run reward from canonical next observation.

    The canonical walker observation ``[orientations(14), height(1),
    velocity(9)]`` does not expose the ``torso_subtreelinvel`` sensor used by
    the official reward, but it does contain enough information to reconstruct
    the full mechanical state:

    * ``height`` fixes ``rootz`` (``rootz = height - 1.3``);
    * the seven body planar angles (``cos``/``sin`` pairs) fix ``rooty`` and
      the six joint angles.  Walker joints rotate about ``(0, -1, 0)``, hence
      ``joint = parent_angle - child_angle``;
    * ``velocity`` is the complete 9-dof qvel (order ``rootz, rootx, rooty,
      right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle``).

    Expanding the official model's planar mass-weighted COM kinematics yields
    the torso-subtree horizontal velocity directly.  This is algebraically
    equivalent to MuJoCo's ``subtree_linvel`` but remains a batched Torch
    operation on the learner device.
    """

    if next_observation.ndim < 1 or next_observation.shape[-1] != 24:
        raise ValueError("Walker reward expects next_observation[..., 24]")
    if applied_action.ndim < 1 or applied_action.shape[-1] != 6:
        raise ValueError("Walker reward expects applied_action[..., 6]")
    if next_observation.shape[:-1] != applied_action.shape[:-1]:
        raise ValueError("Walker reward observation/action batch shapes disagree")
    if not torch.isfinite(next_observation).all() or not torch.isfinite(
        applied_action
    ).all():
        raise FloatingPointError("Walker reward inputs contain NaN or Inf")

    height = next_observation[..., 14]
    upright = (1.0 + next_observation[..., 0]) / 2.0
    standing_distance = torch.clamp(1.2 - height, min=0.0) / 0.6
    standing = torch.exp(math.log(0.1) * standing_distance.square())
    speed = _walker_subtree_horizontal_velocity_torch(next_observation)
    move = torch.clamp(speed / 8.0, min=0.0, max=1.0)
    return ((3.0 * standing + upright) / 4.0) * ((5.0 * move + 1.0) / 6.0)


def official_reward_for_task(task_name: str):
    """Return the parity-verified exact observation reward callable."""
    if task_name == "cartpole_swingup":
        return cartpole_swingup_official_reward
    if task_name == "walker_run":
        return walker_run_official_reward
    raise ValueError(f"No exact observation reward oracle for {task_name!r}")


class ExactObservationRewardOracle(nn.Module):
    """Frozen exact reward evaluator accepting normalized Koopman states."""

    def __init__(
        self,
        task_name: str,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        super().__init__()
        if task_name not in EXACT_REWARD_ORACLE_TASKS:
            raise ValueError(
                f"No verified exact observation reward oracle for {task_name!r}"
            )
        center = torch.as_tensor(center, dtype=torch.float32).detach().clone()
        scale = torch.as_tensor(scale, dtype=torch.float32).detach().clone()
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("Reward oracle center/scale must be matching vectors")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise FloatingPointError("Reward oracle normalizer contains NaN or Inf")
        if torch.any(scale <= 0):
            raise ValueError("Reward oracle scale must be strictly positive")
        self.task_name = task_name
        self.register_buffer("center", center)
        self.register_buffer("scale", scale)
        self.requires_grad_(False)

    def forward(
        self,
        normalized_state: torch.Tensor,
        applied_action: torch.Tensor,
        normalized_next_state: torch.Tensor,
    ) -> torch.Tensor:
        del normalized_state  # Dense rewards are evaluated post-transition.
        next_observation = normalized_next_state * self.scale + self.center
        if self.task_name == "cartpole_swingup":
            return cartpole_swingup_official_reward(
                next_observation, applied_action
            )
        if self.task_name == "walker_run":
            return walker_run_official_reward(next_observation, applied_action)
        raise AssertionError("Validated exact reward task dispatch drifted")

    def metadata(self) -> dict[str, Any]:
        return exact_reward_oracle_metadata(self.task_name)


def exact_reward_oracle_metadata(task_name: str) -> dict[str, Any]:
    if task_name not in EXACT_REWARD_ORACLE_TASKS:
        raise ValueError(f"No verified exact reward oracle for {task_name!r}")
    formula = {
        "cartpole_swingup": "dm_control.suite.cartpole.Balance._get_reward_dense_v1",
        "walker_run": (
            "dm_control.suite.walker.PlanarWalker.get_reward_v1 "
            "(torso-subtree COM velocity reconstructed analytically from "
            "observation using official-model mass/geometry coefficients)"
        ),
    }[task_name]
    return {
        "source": OFFICIAL_OBSERVATION_ORACLE,
        "task": task_name,
        "formula": formula,
        "state_timing": "post_transition_next_observation",
        "action": "applied_action",
        "input_state": "canonical_observation_unnormalized_from_koopman_reconstruction",
        "parity_contract": {
            "reference": "live_dm_control_TimeStep.reward",
            "max_abs_error": ORACLE_PARITY_MAX_ABS_ERROR,
        },
    }


def validate_mpve_reward_source(task_name: str, source: str) -> None:
    if source not in MPVE_REWARD_SOURCES:
        raise ValueError(
            f"Unknown MPVE reward source {source!r}; expected "
            f"{sorted(MPVE_REWARD_SOURCES)}"
        )
    if source == OFFICIAL_OBSERVATION_ORACLE and task_name not in (
        EXACT_REWARD_ORACLE_TASKS
    ):
        raise ValueError(
            f"Task {task_name!r} has no parity-verified exact reward oracle"
        )


def validate_value_expansion_reward_metadata(
    metadata: Mapping[str, Any],
    *,
    task_name: str,
    expected_source: str,
) -> None:
    """Fail closed on formal checkpoint reward-source lineage."""

    validate_mpve_reward_source(task_name, expected_source)
    if not isinstance(metadata, Mapping):
        raise ValueError("value_expansion reward metadata must be a mapping")
    if metadata.get("source") != expected_source:
        raise ValueError("value_expansion reward source does not match config")
    if expected_source == OFFICIAL_OBSERVATION_ORACLE:
        if dict(metadata) != exact_reward_oracle_metadata(task_name):
            raise ValueError("Exact reward-oracle metadata contract mismatch")
    elif metadata.get("model_input_contract") is None:
        raise ValueError("Learned reward source is missing its input contract")
