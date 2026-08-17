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

import torch
from torch import nn


OFFICIAL_OBSERVATION_ORACLE = "dmc_official_observation_oracle_v1"
LEARNED_TRANSITION_REWARD = "learned_transition_model_v1"
MPVE_REWARD_SOURCES = frozenset(
    {OFFICIAL_OBSERVATION_ORACLE, LEARNED_TRANSITION_REWARD}
)

EXACT_REWARD_ORACLE_TASKS = frozenset({"cartpole_swingup"})
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
        del normalized_state  # Dense Cartpole reward is evaluated post-transition.
        next_observation = normalized_next_state * self.scale + self.center
        if self.task_name == "cartpole_swingup":
            return cartpole_swingup_official_reward(
                next_observation, applied_action
            )
        raise AssertionError("Validated exact reward task dispatch drifted")

    def metadata(self) -> dict[str, Any]:
        return exact_reward_oracle_metadata(self.task_name)


def exact_reward_oracle_metadata(task_name: str) -> dict[str, Any]:
    if task_name not in EXACT_REWARD_ORACLE_TASKS:
        raise ValueError(f"No verified exact reward oracle for {task_name!r}")
    return {
        "source": OFFICIAL_OBSERVATION_ORACLE,
        "task": task_name,
        "formula": "dm_control.suite.cartpole.Balance._get_reward_dense_v1",
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
