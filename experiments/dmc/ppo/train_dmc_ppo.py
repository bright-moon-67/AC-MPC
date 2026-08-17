"""Train the five PPO actor variants on a state-based DMC task.

The plain PPO actor operates directly on the raw task observation and does not
need a Koopman checkpoint.  The structured actors use a frozen normalized Deep
Koopman lift and therefore require ``--koopman``.  ``AC-MPC-MPVE`` shares the
exact KMPC actor and adds only the detached model-predictive TD-k critic loss
from Romero et al., TRO 2025, Eq. (8)-(9).

The plain PPO reference follows Acme's state-dependent diagonal Normal followed
by ``tanh`` and evaluates with ``tanh(loc)``.  Structured variants retain their
validated exploration wrapper around each controller action.  At a DMC time
limit, the TD residual bootstraps from the final transition observation using
DMC's discount of one, but GAE never propagates across the automatic reset
boundary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from experiments.dmc.approval import (
    validate_training_approval,
    validate_training_preflight,
)
from experiments.dmc.actors import (
    ACTOR_TYPES,
    HAIKU_DEFAULT_LINEAR_INITIALIZATION,
    ActorConfig,
    StandardPPOActor,
    actor_mean,
    build_actor,
    checkpoint_protocol_fingerprint,
    load_koopman,
    normalizer_arrays,
    initialize_haiku_default_linear,
)
from experiments.dmc.config import (
    ExperimentConfig,
    PROFILE_NAMES,
    load_experiment_config,
    resolve_execution_spec,
    resolve_ppo_config,
)
from experiments.dmc.ppo.vector_env import EnvFactory, VectorStep, make_dmc_vector_env
from experiments.dmc.protocol import canonical_json, protocol_fingerprint
from experiments.dmc.reward_model import (
    reward_model_from_checkpoint,
    transition_reward_input_contract,
)
from experiments.dmc.reward_oracle import (
    LEARNED_TRANSITION_REWARD,
    MPVE_REWARD_SOURCES,
    OFFICIAL_OBSERVATION_ORACLE,
    ExactObservationRewardOracle,
    exact_reward_oracle_metadata,
    validate_mpve_reward_source,
)
from experiments.dmc.tasks.registry import TASK_SPECS, get_task_spec


TRAINING_SPEC_VERSION = "dmc_ppo_v4_raw_observation_critic"
CHECKPOINT_FORMAT_VERSION = 3
_TEST_ONLY_AUTHORIZATION = object()


@dataclass(frozen=True)
class PPOConfig:
    """Optimization settings independent of actor architecture and task."""

    num_envs: int = 8
    rollout_steps: int = 128
    minibatch_size: int = 256
    update_epochs: int = 4
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    anneal_learning_rate: bool = True
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 1e-3
    initial_std: float = 1.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03
    checkpoint_interval_updates: int = 10
    max_wall_time_seconds: float | None = None
    critic_hidden_dim: int = 256
    seed: int = 20_240_101
    collect_flush_transitions: int = 50_000
    collect_max_transitions: int | None = None
    normalize_observation: bool = True
    normalize_advantage: bool = True
    normalize_value: bool = True
    normalization_ema_tau: float = 0.995
    value_clip: bool = False
    value_clipping_epsilon: float = 0.2
    max_abs_reward: float | None = None
    adam_epsilon: float = 1e-7
    mpve_horizon: int = 10
    mpve_value_loss_coefficient: float = 1.0
    mpve_reward_source: str = LEARNED_TRANSITION_REWARD

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def number_updates(self) -> int:
        return math.ceil(self.total_timesteps / self.batch_size)

    def validate(self) -> None:
        integers = {
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "minibatch_size": self.minibatch_size,
            "update_epochs": self.update_epochs,
            "total_timesteps": self.total_timesteps,
            "checkpoint_interval_updates": self.checkpoint_interval_updates,
            "critic_hidden_dim": self.critic_hidden_dim,
            "collect_flush_transitions": self.collect_flush_transitions,
            "mpve_horizon": self.mpve_horizon,
        }
        for name, value in integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_rate <= 0 or self.initial_std <= 0:
            raise ValueError("learning_rate and initial_std must be positive")
        if not 0 <= self.discount <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("discount and gae_lambda must lie in [0, 1]")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must lie in (0, 1)")
        if self.value_coefficient < 0 or self.entropy_coefficient < 0:
            raise ValueError("loss coefficients must be non-negative")
        for name in (
            "anneal_learning_rate",
            "normalize_observation",
            "normalize_advantage",
            "normalize_value",
            "value_clip",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not 0.0 < self.normalization_ema_tau < 1.0:
            raise ValueError("normalization_ema_tau must lie in (0, 1)")
        if self.value_clipping_epsilon <= 0 or not math.isfinite(
            float(self.value_clipping_epsilon)
        ):
            raise ValueError("value_clipping_epsilon must be finite and positive")
        if self.adam_epsilon <= 0 or not math.isfinite(float(self.adam_epsilon)):
            raise ValueError("adam_epsilon must be finite and positive")
        if self.max_abs_reward is not None and (
            self.max_abs_reward <= 0
            or not math.isfinite(float(self.max_abs_reward))
        ):
            raise ValueError("max_abs_reward must be finite and positive")
        if (
            isinstance(self.mpve_value_loss_coefficient, bool)
            or not isinstance(self.mpve_value_loss_coefficient, (int, float))
            or not math.isfinite(float(self.mpve_value_loss_coefficient))
            or self.mpve_value_loss_coefficient <= 0
        ):
            raise ValueError(
                "mpve_value_loss_coefficient must be finite and positive"
            )
        if self.mpve_reward_source not in MPVE_REWARD_SOURCES:
            raise ValueError(
                "mpve_reward_source must be one of "
                f"{sorted(MPVE_REWARD_SOURCES)}"
            )
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive when provided")
        if self.max_wall_time_seconds is not None and self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive when provided")
        if (
            self.collect_max_transitions is not None
            and self.collect_max_transitions < 1
        ):
            raise ValueError("collect_max_transitions must be positive when provided")


def ppo_config_from_experiment(
    experiment: ExperimentConfig,
    profile: str,
    *,
    train_seed_index: int = 0,
) -> PPOConfig:
    """Resolve the approval-bound YAML fields consumed by ``PPOConfig``."""

    resolved = resolve_ppo_config(
        experiment,
        profile,
        train_seed_index=train_seed_index,
    )
    resolved.setdefault(
        "collect_max_transitions",
        int(experiment.raw["data"]["max_transitions_per_train_seed"]),
    )
    config = PPOConfig(**resolved)
    config.validate()
    return config


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class ValueNetwork(nn.Module):
    """Three-layer ReLU critic used by the Acme continuous-PPO baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or hidden_layers < 1:
            raise ValueError("value-network dimensions must be positive")
        layers: list[nn.Module] = []
        following_input = int(input_dim)
        for _ in range(hidden_layers):
            layer = nn.Linear(following_input, hidden_dim)
            initialize_haiku_default_linear(layer)
            layers.extend((layer, nn.ReLU()))
            following_input = hidden_dim
        output = nn.Linear(following_input, 1)
        initialize_haiku_default_linear(output)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation).squeeze(-1)


class ObservationRunningMeanStd:
    """Acme-style count-based Welford observation normalizer."""

    def __init__(self, dimension: int, *, epsilon: float = 1e-6) -> None:
        if dimension < 1 or epsilon <= 0:
            raise ValueError("normalizer dimension/epsilon must be positive")
        self.dimension = int(dimension)
        self.epsilon = float(epsilon)
        self.count = 0
        self.mean = torch.zeros(self.dimension, dtype=torch.float64)
        self.summed_variance = torch.zeros(self.dimension, dtype=torch.float64)

    @property
    def std(self) -> torch.Tensor:
        if self.count == 0:
            return torch.ones_like(self.mean)
        return torch.sqrt(
            torch.clamp(self.summed_variance / self.count, min=0.0)
        ).clamp(min=self.epsilon, max=1e6)

    def update(self, batch: torch.Tensor) -> None:
        if batch.shape[-1] != self.dimension:
            raise ValueError("Observation normalizer received the wrong dimension")
        flat = batch.detach().reshape(-1, self.dimension).to(
            device="cpu", dtype=torch.float64
        )
        if not len(flat):
            raise ValueError("Observation normalizer cannot update from an empty batch")
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Observation normalizer input contains NaN or Inf")
        new_count = self.count + len(flat)
        difference = flat - self.mean
        mean_update = difference.sum(dim=0) / new_count
        following_mean = self.mean + mean_update
        self.summed_variance += (
            difference * (flat - following_mean)
        ).sum(dim=0)
        self.mean = following_mean
        self.count = int(new_count)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.dimension:
            raise ValueError("Observation normalizer received the wrong dimension")
        if self.count == 0:
            return value
        mean = self.mean.to(device=value.device, dtype=value.dtype)
        std = self.std.to(device=value.device, dtype=value.dtype)
        return (value - mean) / std

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "acme_welford_observation_normalizer_v1",
            "dimension": self.dimension,
            "epsilon": self.epsilon,
            "count": self.count,
            "mean": self.mean.clone(),
            "summed_variance": self.summed_variance.clone(),
            "std": self.std.clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("kind") != (
            "acme_welford_observation_normalizer_v1"
        ):
            raise ValueError("Invalid observation normalizer checkpoint")
        if state.get("dimension") != self.dimension or state.get("epsilon") != (
            self.epsilon
        ):
            raise ValueError("Observation normalizer checkpoint metadata mismatch")
        count = state.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Observation normalizer count is invalid")
        mean = torch.as_tensor(state.get("mean"), dtype=torch.float64).cpu()
        summed = torch.as_tensor(
            state.get("summed_variance"), dtype=torch.float64
        ).cpu()
        if mean.shape != (self.dimension,) or summed.shape != (self.dimension,):
            raise ValueError("Observation normalizer tensor shape mismatch")
        if not torch.isfinite(mean).all() or not torch.isfinite(summed).all():
            raise FloatingPointError("Observation normalizer state is non-finite")
        if bool((summed < -1e-9).any()):
            raise ValueError("Observation normalizer variance is negative")
        self.count = count
        self.mean = mean.clone()
        self.summed_variance = summed.clamp(min=0.0).clone()
        saved_std = torch.as_tensor(state.get("std"), dtype=torch.float64)
        if saved_std.shape != (self.dimension,) or not torch.allclose(
            saved_std, self.std, rtol=1e-7, atol=1e-9
        ):
            raise ValueError("Observation normalizer saved std is inconsistent")


class AcmeReturnNormalizer:
    """Zero-debiased EMA statistics for Acme advantage/value normalization."""

    def __init__(self, tau: float) -> None:
        if not 0.0 < tau < 1.0:
            raise ValueError("EMA tau must lie in (0, 1)")
        self.tau = float(tau)
        self.ema_counter = 0
        self.biased_advantage_scale = 0.0
        self.advantage_scale = 0.0
        self.biased_value_first_moment = 0.0
        self.biased_value_second_moment = 0.0
        self.value_mean = 0.0
        self.value_std = 0.0

    def begin_update(self) -> None:
        self.ema_counter += 1

    def _zero_debias(self) -> float:
        if self.ema_counter < 1:
            raise RuntimeError("EMA update counter has not advanced")
        return 1.0 / (1.0 - self.tau**self.ema_counter)

    def update_value(self, behavior_values: torch.Tensor) -> None:
        if not torch.isfinite(behavior_values).all():
            raise FloatingPointError("Value-normalizer input contains NaN or Inf")
        zero_debias = self._zero_debias()
        first = float(behavior_values.detach().mean())
        second = float(behavior_values.detach().square().mean())
        self.biased_value_first_moment = (
            self.tau * self.biased_value_first_moment + (1.0 - self.tau) * first
        )
        self.biased_value_second_moment = (
            self.tau * self.biased_value_second_moment
            + (1.0 - self.tau) * second
        )
        self.value_mean = self.biased_value_first_moment * zero_debias
        value_second = self.biased_value_second_moment * zero_debias
        self.value_std = math.sqrt(max(value_second - self.value_mean**2, 0.0))

    def update_advantage(self, advantages: torch.Tensor) -> None:
        if not torch.isfinite(advantages).all():
            raise FloatingPointError("Advantage normalizer input contains NaN or Inf")
        zero_debias = self._zero_debias()
        batch_scale = float(advantages.detach().abs().mean())
        self.biased_advantage_scale = (
            self.tau * self.biased_advantage_scale
            + (1.0 - self.tau) * batch_scale
        )
        self.advantage_scale = self.biased_advantage_scale * zero_debias

    def denormalize_value(self, value: torch.Tensor) -> torch.Tensor:
        scale = max(self.value_std, 1e-6)
        return value * scale + self.value_mean

    def normalize_value(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.value_mean) / max(self.value_std, 1e-6)

    def normalize_advantage(self, value: torch.Tensor) -> torch.Tensor:
        return value / max(self.advantage_scale, 1e-6)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "acme_ppo_zero_debiased_ema_v1",
            "tau": self.tau,
            "ema_counter": self.ema_counter,
            "biased_advantage_scale": self.biased_advantage_scale,
            "advantage_scale": self.advantage_scale,
            "biased_value_first_moment": self.biased_value_first_moment,
            "biased_value_second_moment": self.biased_value_second_moment,
            "value_mean": self.value_mean,
            "value_std": self.value_std,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict) or state.get("kind") != (
            "acme_ppo_zero_debiased_ema_v1"
        ):
            raise ValueError("Invalid Acme EMA normalization checkpoint")
        if state.get("tau") != self.tau:
            raise ValueError("Acme EMA tau changed on resume")
        counter = state.get("ema_counter")
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise ValueError("Acme EMA counter is invalid")
        names = (
            "biased_advantage_scale",
            "advantage_scale",
            "biased_value_first_moment",
            "biased_value_second_moment",
            "value_mean",
            "value_std",
        )
        values = {name: float(state[name]) for name in names}
        non_negative = {
            "biased_advantage_scale",
            "advantage_scale",
            "biased_value_second_moment",
            "value_std",
        }
        if not all(math.isfinite(value) for value in values.values()) or not all(
            values[name] >= 0.0 for name in non_negative
        ):
            raise ValueError("Acme EMA normalization state is invalid")
        self.ema_counter = counter
        for name, value in values.items():
            setattr(self, name, value)


_TANH_LOG_PROB_THRESHOLD = 0.999


def tanh_normal_sample(
    location: torch.Tensor,
    scale: torch.Tensor,
    *,
    action_limit: float = 1.0,
) -> torch.Tensor:
    """Sample the Acme tanh-transformed diagonal Normal policy."""

    return float(action_limit) * torch.tanh(Normal(location, scale).sample())


def tanh_normal_log_prob(
    location: torch.Tensor,
    scale: torch.Tensor,
    action: torch.Tensor,
    *,
    action_limit: float = 1.0,
) -> torch.Tensor:
    """Stable TFP-compatible log probability for a bounded action batch."""

    if location.shape != scale.shape or action.shape != location.shape:
        raise ValueError("Tanh-Normal parameter/action shapes must match")
    limit = float(action_limit)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("action_limit must be finite and positive")
    normalized = action / limit
    threshold = _TANH_LOG_PROB_THRESHOLD
    clipped = normalized.clamp(min=-threshold, max=threshold)
    inverse = torch.atanh(clipped)
    distribution = Normal(location, scale)
    middle = distribution.log_prob(inverse) - torch.log1p(-clipped.square())

    inverse_threshold = math.atanh(threshold)
    log_epsilon = math.log(1.0 - threshold)
    left_standardized = (-inverse_threshold - location) / scale
    right_standardized = (location - inverse_threshold) / scale
    left = torch.special.log_ndtr(left_standardized) - log_epsilon
    right = torch.special.log_ndtr(right_standardized) - log_epsilon
    component = torch.where(
        normalized <= -threshold,
        left,
        torch.where(normalized >= threshold, right, middle),
    )
    return (component - math.log(limit)).sum(dim=-1)


def tanh_normal_entropy(
    location: torch.Tensor,
    scale: torch.Tensor,
    *,
    action_limit: float = 1.0,
) -> torch.Tensor:
    """Single-sample transformed entropy estimate used by Acme PPO."""

    distribution = Normal(location, scale)
    latent = distribution.rsample()
    log_jacobian = math.log(float(action_limit)) + 2.0 * (
        math.log(2.0)
        - latent
        - torch.nn.functional.softplus(-2.0 * latent)
    )
    return (distribution.entropy() + log_jacobian).sum(dim=-1)


@dataclass(frozen=True)
class MPVERolloutPrediction:
    """Detached MPC prediction data saved alongside one real rollout state."""

    action: torch.Tensor
    action_sequence: torch.Tensor
    value_observations: torch.Tensor
    predicted_rewards: torch.Tensor
    terminal_value: torch.Tensor
    td_k_targets: torch.Tensor


def compute_mpve_td_k_targets(
    predicted_rewards: torch.Tensor,
    terminal_value: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    r"""Return all TD-k targets in TRO25 Eq. (8)-(9).

    For every predicted state ``s_hat[t]``, the target is

    ``sum(k=t..H-1) gamma**(k-t) r_hat[k] + gamma**(H-t) V(s_hat[H])``.

    Rewards and the terminal bootstrap are target data and are detached here,
    making the gradient boundary explicit even for direct unit-test callers.
    """

    if predicted_rewards.ndim < 1:
        raise ValueError("predicted_rewards must have a horizon dimension")
    horizon = predicted_rewards.shape[-1]
    if horizon < 1:
        raise ValueError("MPVE horizon must be positive")
    if terminal_value.shape != predicted_rewards.shape[:-1]:
        raise ValueError("terminal_value shape must match the reward batch shape")
    if not math.isfinite(float(gamma)) or not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    if not torch.isfinite(predicted_rewards).all() or not torch.isfinite(
        terminal_value
    ).all():
        raise FloatingPointError("MPVE target inputs contain NaN or Inf")

    rewards = predicted_rewards.detach()
    following = terminal_value.detach()
    reversed_targets: list[torch.Tensor] = []
    for index in range(horizon - 1, -1, -1):
        following = rewards[..., index] + float(gamma) * following
        reversed_targets.append(following)
    return torch.stack(list(reversed(reversed_targets)), dim=-1).detach()


def collect_mpve_prediction(
    actor: nn.Module,
    value_network: ValueNetwork,
    koopman: nn.Module,
    reward_model: nn.Module,
    lifted_state: torch.Tensor,
    *,
    horizon: int,
    gamma: float,
) -> MPVERolloutPrediction:
    """Reuse KMPC's action plan to build one frozen Koopman prediction path.

    This function is intentionally a no-gradient boundary.  MPC predictions,
    exact-or-learned rewards and the terminal value become immutable rollout targets;
    the later Eq. (9) regression can therefore update only the critic.
    """

    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("MPVE horizon must be a positive integer")
    with torch.no_grad():
        actor_output = actor(lifted_state)
        action_sequence = actor_output.action_sequence[..., :horizon, :]
        if action_sequence.shape[-2] != horizon:
            raise ValueError(
                f"MPVE horizon {horizon} exceeds MPC horizon "
                f"{actor_output.action_sequence.shape[-2]}"
            )
        current = lifted_state.detach()
        value_observations: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        for index in range(horizon):
            # The controller plans in lifted coordinates, but every method's
            # critic consumes the normalized physical observation.  With the
            # identity-skip Koopman model, reconstruct() returns that canonical
            # observation without exposing learned lift features to the critic.
            value_observations.append(koopman.reconstruct(current))
            following = koopman.linear_step(current, action_sequence[..., index, :])
            normalized_state = koopman.reconstruct(current)
            normalized_following = koopman.reconstruct(following)
            rewards.append(
                reward_model(
                    normalized_state,
                    action_sequence[..., index, :],
                    normalized_following,
                )
            )
            current = following
        predicted_rewards = torch.stack(rewards, dim=-1)
        terminal_value = value_network(koopman.reconstruct(current))
        targets = compute_mpve_td_k_targets(
            predicted_rewards,
            terminal_value,
            gamma=gamma,
        )
        return MPVERolloutPrediction(
            action=actor_output.action.detach(),
            action_sequence=action_sequence.detach(),
            value_observations=torch.stack(value_observations, dim=-2).detach(),
            predicted_rewards=predicted_rewards.detach(),
            terminal_value=terminal_value.detach(),
            td_k_targets=targets,
        )


def mpve_value_loss(
    value_network: ValueNetwork,
    predicted_observations: torch.Tensor,
    td_k_targets: torch.Tensor,
) -> torch.Tensor:
    """Eq. (9): mean squared TD-k regression over all prediction depths."""

    if predicted_observations.shape[:-1] != td_k_targets.shape:
        raise ValueError("MPVE predicted-state and target shapes disagree")
    predicted_values = value_network(predicted_observations.detach())
    return (predicted_values - td_k_targets.detach()).square().mean()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_values: torch.Tensor,
    discounts: torch.Tensor,
    reset_boundaries: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute timeout-correct GAE for ``[time, env]`` rollout tensors.

    ``discounts`` affects the one-step bootstrap, so a DMC timeout with discount
    one uses the value of the final transition observation.  Independently,
    ``reset_boundaries`` stops the lambda trace, preventing advantages from the
    autoreset episode leaking backward into the episode that just ended.
    """

    expected = rewards.shape
    tensors = {
        "values": values,
        "bootstrap_values": bootstrap_values,
        "discounts": discounts,
        "reset_boundaries": reset_boundaries,
    }
    if rewards.ndim != 2:
        raise ValueError("GAE tensors must have shape [time, env]")
    for name, value in tensors.items():
        if value.shape != expected:
            raise ValueError(f"{name} shape {value.shape} does not match {expected}")
    if not all(
        torch.isfinite(value).all()
        for value in (rewards, values, bootstrap_values, discounts)
    ):
        raise FloatingPointError("GAE input contains NaN or Inf")
    if bool(((discounts < 0) | (discounts > 1)).any()):
        raise ValueError("Environment discounts must lie in [0, 1]")

    boundaries = reset_boundaries.to(dtype=torch.bool)
    advantages = torch.zeros_like(rewards)
    following_advantage = torch.zeros_like(rewards[0])
    for index in range(rewards.shape[0] - 1, -1, -1):
        delta = (
            rewards[index]
            + float(gamma) * discounts[index] * bootstrap_values[index]
            - values[index]
        )
        trace = discounts[index] * (~boundaries[index]).to(rewards.dtype)
        following_advantage = (
            delta
            + float(gamma)
            * float(gae_lambda)
            * trace
            * following_advantage
        )
        advantages[index] = following_advantage
    return advantages, advantages + values


def prepare_acme_ppo_learner_values(
    value_network: nn.Module,
    observation_normalizer: ObservationRunningMeanStd,
    states: torch.Tensor,
    transition_states: torch.Tensor,
    *,
    normalize_observation: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update learner observation stats, then recompute all behavior values.

    Acme updates observation statistics from the learner's ``T+1`` sequence
    before applying the current critic.  The actions' behavior log-probabilities
    remain those recorded during collection, but behavior values are deliberately
    recomputed under the newly updated observation statistics.  In this
    synchronous autoreset implementation, ``transition_states[-1]`` is the
    bootstrap observation completing the ``T+1`` statistics sequence; all
    transition states are still evaluated so timeout-correct GAE can use each
    final pre-reset observation.
    """

    if states.ndim != 3 or transition_states.shape != states.shape:
        raise ValueError(
            "PPO learner states and transition_states must share [time, env, obs]"
        )
    if states.shape[-1] != observation_normalizer.dimension:
        raise ValueError("PPO learner state dimension is inconsistent")
    if not torch.isfinite(states).all() or not torch.isfinite(
        transition_states
    ).all():
        raise FloatingPointError("PPO learner observations contain NaN or Inf")

    with torch.no_grad():
        if normalize_observation:
            observation_normalizer.update(
                torch.cat((states, transition_states[-1:]), dim=0)
            )
            learner_states = observation_normalizer.normalize(states)
            learner_transition_states = observation_normalizer.normalize(
                transition_states
            )
        else:
            learner_states = states
            learner_transition_states = transition_states
        normalized_values = value_network(learner_states).detach()
        normalized_bootstrap_values = value_network(
            learner_transition_states
        ).detach()
    if normalized_values.shape != states.shape[:2] or (
        normalized_bootstrap_values.shape != states.shape[:2]
    ):
        raise ValueError("PPO value network returned the wrong learner shape")
    return (
        learner_states.detach(),
        learner_transition_states.detach(),
        normalized_values,
        normalized_bootstrap_values,
    )


def crossed_diagnostic_milestones(
    previous_step: int,
    current_step: int,
    interval: int | None,
) -> list[int]:
    """Return fixed diagnostic milestones crossed by an aligned PPO update."""

    if (
        isinstance(previous_step, bool)
        or isinstance(current_step, bool)
        or not isinstance(previous_step, int)
        or not isinstance(current_step, int)
        or previous_step < 0
        or current_step < previous_step
    ):
        raise ValueError("Diagnostic step bounds must be ordered non-negative ints")
    if interval is None:
        return []
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
        raise ValueError("Diagnostic interval must be a positive integer")
    first = (previous_step // interval + 1) * interval
    return list(range(first, current_step + 1, interval))


def clip_ppo_gradients(
    policy_parameters: list[torch.Tensor],
    critic_parameters: list[torch.Tensor],
    *,
    max_grad_norm: float,
    separate_policy_and_critic: bool,
) -> torch.Tensor:
    """Clip PPO gradients while preserving the KMPC/MPVE actor ablation.

    Plain PPO retains Acme's single global norm.  KMPC and AC-MPC-MPVE use the
    same policy/log-std group clip and an independent critic clip, so adding the
    detached MPVE critic target cannot indirectly rescale the actor gradient.
    """

    if not policy_parameters or not critic_parameters:
        raise ValueError("PPO gradient groups must both be non-empty")
    if not math.isfinite(float(max_grad_norm)) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    if separate_policy_and_critic:
        policy_norm = nn.utils.clip_grad_norm_(policy_parameters, max_grad_norm)
        critic_norm = nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm)
        return torch.sqrt(policy_norm.square() + critic_norm.square())
    return nn.utils.clip_grad_norm_(
        [*policy_parameters, *critic_parameters], max_grad_norm
    )


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _truncate_metrics_to_checkpoint(path: Path, checkpoint_update: int) -> None:
    """Atomically discard metric rows newer than the authoritative checkpoint."""

    if checkpoint_update < 0:
        raise ValueError("checkpoint_update must be non-negative")
    if not path.exists():
        if checkpoint_update:
            raise ValueError("Resume checkpoint exists but metrics.jsonl is missing")
        return
    retained: list[dict[str, Any]] = []
    previous_update = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            # A crash while appending the first row after the checkpoint may
            # leave only that final line torn.  It is safe to discard because
            # the checkpoint update is already completely represented.
            if line_number == len(lines) and previous_update == checkpoint_update:
                break
            raise ValueError(
                f"metrics.jsonl line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"metrics.jsonl line {line_number} is not an object")
        update = value.get("update")
        if isinstance(update, bool) or not isinstance(update, int):
            raise ValueError(
                f"metrics.jsonl line {line_number} has an invalid update"
            )
        if update <= checkpoint_update:
            if update <= previous_update:
                raise ValueError("metrics.jsonl retained updates are not increasing")
            retained.append(value)
            previous_update = update
    if checkpoint_update and previous_update != checkpoint_update:
        raise ValueError(
            "metrics.jsonl does not contain the authoritative checkpoint update"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for value in retained:
                stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Recursively convert RNG/metadata values to plain JSON values."""

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _capture_rng_state() -> dict[str, Any]:
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().cpu(),
        "cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(torch.as_tensor(state["torch"]).cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.as_tensor(item).cpu() for item in state["cuda"]]
        )


class OnPolicyEpisodeCollector:
    """Persist complete on-policy episodes in the Koopman/offline-RL schema."""

    FIELDS = (
        "state",
        "requested_action",
        "action",
        "next_state",
        "reward",
        "discount",
        "terminated",
        "truncated",
        "reset_seed",
        "update",
        "global_step",
    )
    COLLECTION_SCHEMA_VERSION = 4
    EPISODE_NAMESPACE_SIZE = 1_000_000
    STAGE_NAMES = ("early", "mid", "late")
    SELECTION_STRATEGY = "stage_quota_first_complete_episode_v2"

    def __init__(
        self,
        output_dir: Path,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        *,
        flush_transitions: int = 50_000,
        metadata: dict[str, Any] | None = None,
        seed_index: int = 0,
        seed_dir: str | None = None,
        max_transitions: int,
        total_updates: int,
    ) -> None:
        if num_envs < 1 or obs_dim < 1 or action_dim < 1:
            raise ValueError("collector dimensions must be positive")
        if flush_transitions < 1:
            raise ValueError("flush_transitions must be positive")
        if max_transitions < 1:
            raise ValueError("max_transitions must be positive")
        if total_updates < 3:
            raise ValueError(
                "staged collection requires at least three PPO updates"
            )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.flush_transitions = int(flush_transitions)
        self.max_transitions = int(max_transitions)
        self.total_updates = int(total_updates)
        self.metadata = copy.deepcopy(metadata or {})
        if seed_index < 0:
            raise ValueError("seed_index must be non-negative")
        self.seed_index = int(seed_index)
        self.seed_dir = str(seed_dir or self.output_dir.name)
        if not self.seed_dir:
            raise ValueError("seed_dir must be non-empty")
        try:
            self.task_name = str(self.metadata["task"])
            self.environment_protocol = copy.deepcopy(self.metadata["protocol"])
        except KeyError as exc:
            raise ValueError("collector metadata requires task and protocol") from exc
        if not isinstance(self.environment_protocol, dict):
            raise TypeError("collector protocol metadata must be a mapping")
        self.environment_protocol_json = canonical_json(self.environment_protocol)
        self.protocol_fingerprint = protocol_fingerprint(
            self.environment_protocol
        )
        if "step_limit" not in self.environment_protocol:
            raise ValueError("collector environment protocol requires step_limit")
        self.expected_episode_steps = int(self.environment_protocol["step_limit"])
        if self.expected_episode_steps < 1:
            raise ValueError("collector step_limit must be positive")
        if self.max_transitions < 3 * self.expected_episode_steps:
            raise ValueError(
                "collection budget must fit at least one complete expected "
                "episode in each of early/mid/late"
            )
        self.collection_protocol = {
            **self.environment_protocol,
            "collector_max_episode_steps": int(
                self.environment_protocol["step_limit"]
            ),
            "collector_truncates_episodes": False,
        }
        self.protocol_json = canonical_json(self.collection_protocol)
        supplied_fingerprint = self.metadata.get("protocol_fingerprint")
        if (
            supplied_fingerprint is not None
            and supplied_fingerprint != self.protocol_fingerprint
        ):
            raise ValueError("collector protocol fingerprint does not match protocol")
        if self.environment_protocol.get("task") != self.task_name:
            raise ValueError("collector task does not match protocol task")
        if int(self.environment_protocol.get("obs_dim", -1)) != self.obs_dim:
            raise ValueError("collector obs_dim does not match protocol")
        if int(self.environment_protocol.get("action_dim", -1)) != self.action_dim:
            raise ValueError("collector action_dim does not match protocol")
        lineage_fields = (
            "training_seed",
            "actor_type",
            "training_approved",
            "config_fingerprint",
            "approval_profile",
            "approval_file_sha256",
            "preflight_report_sha256",
            "authorization_kind",
            "train_seed_index",
        )
        missing_lineage = [
            name for name in lineage_fields if name not in self.metadata
        ]
        if missing_lineage:
            raise ValueError(
                f"collector metadata is missing lineage {missing_lineage}"
            )
        if self.metadata["actor_type"] != "PPO":
            raise ValueError("primary on-policy collection requires actor_type=PPO")
        if self.metadata["training_approved"] is not True:
            raise ValueError("on-policy collection requires approved training")
        if int(self.metadata["train_seed_index"]) != self.seed_index:
            raise ValueError("collector train_seed_index/seed_index mismatch")
        self.lineage = {
            name: (
                copy.deepcopy(self.metadata[name])
                if self.metadata[name] is not None
                else ""
            )
            for name in lineage_fields
        }
        self._active = [self._empty_episode() for _ in range(num_envs)]
        self._complete: list[dict[str, np.ndarray]] = []
        self._pending_transitions = 0
        self._last_update = 0
        self._last_training_global_step = 0
        quota_base, quota_remainder = divmod(self.max_transitions, 3)
        self._stage_quotas = [
            quota_base + (1 if index < quota_remainder else 0)
            for index in range(3)
        ]
        stage_ends = [
            self.total_updates // 3,
            2 * self.total_updates // 3,
            self.total_updates,
        ]
        self._stage_ranges = [
            (1, stage_ends[0]),
            (stage_ends[0] + 1, stage_ends[1]),
            (stage_ends[1] + 1, stage_ends[2]),
        ]
        self._stage_target_episodes = [
            max(1, quota // self.expected_episode_steps)
            for quota in self._stage_quotas
        ]
        self._stage_transition_counts = [0, 0, 0]
        self._stage_episode_counts = [0, 0, 0]
        self._stage_skipped_episodes = [0, 0, 0]

        existing = sorted(self.output_dir.glob("coverage_*.npz"))
        indices: list[int] = []
        maximum_episode = -1
        maximum_global_step = -1
        maximum_update = 0
        total_transitions = 0
        seen_episode_ids: set[int] = set()
        seen_global_steps: set[int] = set()
        for path in existing:
            try:
                indices.append(int(path.stem.split("_")[-1]))
            except ValueError:
                raise ValueError(f"Invalid collection chunk filename {path.name!r}")
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    *self.FIELDS,
                    "done",
                    "collector_truncated",
                    "episode_id",
                    "step_index",
                    "update",
                    "global_step",
                    "collection_schema_version",
                    "protocol_json",
                    "environment_protocol_json",
                    "protocol_fingerprint",
                    "task",
                    "seed_index",
                    "seed_dir",
                    "rng_state_after_json",
                    "collection_stage",
                    "collection_selection_strategy",
                    "collection_max_transitions",
                    "collection_total_updates",
                    *self.lineage.keys(),
                }
                missing = required - set(archive.files)
                if missing:
                    raise ValueError(
                        f"Existing chunk {path} has obsolete schema; missing "
                        f"{sorted(missing)}"
                    )
                scalar_checks = {
                    "collection_schema_version": self.COLLECTION_SCHEMA_VERSION,
                    "protocol_json": self.protocol_json,
                    "environment_protocol_json": self.environment_protocol_json,
                    "protocol_fingerprint": self.protocol_fingerprint,
                    "task": self.task_name,
                    "seed_index": self.seed_index,
                    "seed_dir": self.seed_dir,
                    "collection_selection_strategy": self.SELECTION_STRATEGY,
                    "collection_max_transitions": self.max_transitions,
                    "collection_total_updates": self.total_updates,
                    **self.lineage,
                }
                mismatches: dict[str, tuple[Any, Any]] = {}
                for name, expected in scalar_checks.items():
                    stored = np.asarray(archive[name])
                    actual = (
                        stored.item()
                        if stored.shape == ()
                        else {"invalid_shape": list(stored.shape)}
                    )
                    if actual != expected:
                        mismatches[name] = (actual, expected)
                if mismatches:
                    raise ValueError(
                        f"Existing chunk {path} metadata mismatch: {mismatches}"
                    )
                total_transitions += int(len(archive["state"]))
                if len(archive["episode_id"]):
                    episode_ids = np.asarray(archive["episode_id"], dtype=np.int64)
                    global_steps = np.asarray(archive["global_step"], dtype=np.int64)
                    chunk_episode_ids = {
                        int(value) for value in np.unique(episode_ids)
                    }
                    duplicate_episodes = seen_episode_ids & chunk_episode_ids
                    if duplicate_episodes:
                        raise ValueError(
                            f"Collection repeats episode ids {duplicate_episodes}"
                        )
                    chunk_global_steps = {int(value) for value in global_steps}
                    if len(chunk_global_steps) != len(global_steps):
                        raise ValueError(
                            f"Collection chunk {path} repeats global steps"
                        )
                    duplicate_steps = seen_global_steps & chunk_global_steps
                    if duplicate_steps:
                        raise ValueError(
                            f"Collection repeats global steps {duplicate_steps}"
                        )
                    seen_episode_ids.update(chunk_episode_ids)
                    seen_global_steps.update(chunk_global_steps)
                    for episode_id in chunk_episode_ids:
                        indices_for_episode = np.flatnonzero(
                            episode_ids == episode_id
                        )
                        labels = np.unique(
                            np.asarray(archive["collection_stage"])[
                                indices_for_episode
                            ]
                        )
                        if len(labels) != 1 or str(labels[0]) not in self.STAGE_NAMES:
                            raise ValueError(
                                f"Episode {episode_id} has invalid collection stage"
                            )
                        stage_index = self.STAGE_NAMES.index(str(labels[0]))
                        self._stage_episode_counts[stage_index] += 1
                        self._stage_transition_counts[stage_index] += len(
                            indices_for_episode
                        )
                    maximum_episode = max(
                        maximum_episode, int(np.max(archive["episode_id"]))
                    )
                    maximum_global_step = max(
                        maximum_global_step, int(np.max(archive["global_step"]))
                    )
                    maximum_update = max(
                        maximum_update, int(np.max(archive["update"]))
                    )
        if total_transitions > self.max_transitions:
            raise ValueError("Existing durable collection exceeds its hard cap")
        for index, count in enumerate(self._stage_transition_counts):
            if count > self._stage_quotas[index]:
                raise ValueError(
                    f"Existing {self.STAGE_NAMES[index]} stage exceeds its quota"
                )
        self._chunk_index = max(indices, default=-1) + 1
        namespace_start = self.EPISODE_NAMESPACE_SIZE * self.seed_index
        namespace_end = namespace_start + self.EPISODE_NAMESPACE_SIZE
        if any(
            episode_id < namespace_start or episode_id >= namespace_end
            for episode_id in seen_episode_ids
        ):
            raise ValueError("Existing episode id is outside its seed namespace")
        self._next_episode_id = (
            namespace_start if maximum_episode < 0 else maximum_episode + 1
        )
        if not namespace_start <= self._next_episode_id < namespace_end:
            raise ValueError(
                f"Episode id {self._next_episode_id} is outside seed "
                f"{self.seed_index}'s namespace"
            )
        self._next_global_step = maximum_global_step + 1
        self._last_seen_global_step = maximum_global_step
        self._last_update = maximum_update
        self.total_transitions = total_transitions
        self.chunks_written = len(existing)

    @staticmethod
    def _empty_episode() -> dict[str, list[Any]]:
        return {name: [] for name in OnPolicyEpisodeCollector.FIELDS}

    def _stage_index(self, update: int) -> int:
        if not 1 <= update <= self.total_updates:
            raise ValueError(
                f"PPO update {update} lies outside [1, {self.total_updates}]"
            )
        for index, (_start, end) in enumerate(self._stage_ranges):
            if update <= end:
                return index
        raise AssertionError("stage partition does not cover the final update")

    def _select_episode(
        self, stage_index: int, completion_update: int, episode_length: int
    ) -> bool:
        # Selection is quota-driven rather than progress-gated.  With Acme's
        # 256 actors and eight-step unroll, a 1000-step DMC task completes 256
        # episodes in one synchronized burst every 125 updates.  A progress
        # gate applied sequentially inside that burst permanently underfilled
        # every stage.  Taking the first complete episodes up to each fixed
        # stage quota is deterministic, resume-safe, never cuts an episode and
        # fills 100/100/100 Cartpole episodes under the 300k transition cap.
        if stage_index != self._stage_index(completion_update):
            raise ValueError("Episode completion update does not match its stage")
        if (
            self._stage_episode_counts[stage_index]
            >= self._stage_target_episodes[stage_index]
        ):
            return False
        if (
            self._stage_transition_counts[stage_index] + episode_length
            > self._stage_quotas[stage_index]
        ):
            return False
        return True

    def budget_report(self) -> dict[str, Any]:
        """Resolved hard cap and deterministic early/mid/late selection state."""

        return {
            "selection_strategy": self.SELECTION_STRATEGY,
            "configured_max_transitions": self.max_transitions,
            "effective_durable_upper_bound": self.max_transitions,
            "expected_episode_steps": self.expected_episode_steps,
            "total_updates": self.total_updates,
            "durable_transitions": self.total_transitions,
            "pending_complete_episode_transitions": self._pending_transitions,
            "stages": [
                {
                    "name": name,
                    "start_update": self._stage_ranges[index][0],
                    "end_update": self._stage_ranges[index][1],
                    "transition_quota": self._stage_quotas[index],
                    "target_episodes": self._stage_target_episodes[index],
                    "selected_episodes": self._stage_episode_counts[index],
                    "selected_transitions": self._stage_transition_counts[index],
                    "skipped_complete_episodes": self._stage_skipped_episodes[index],
                }
                for index, name in enumerate(self.STAGE_NAMES)
            ],
        }

    def checkpoint_state(self) -> dict[str, Any]:
        """State needed to prove durable collector continuity on resume."""

        return {
            "kind": "dmc_on_policy_collector_v4",
            "output_dir": str(self.output_dir.resolve()),
            "total_transitions": self.total_transitions,
            "chunks_written": self.chunks_written,
            "next_episode_id": self._next_episode_id,
            "next_global_step": self._next_global_step,
            "last_update": self._last_update,
            "budget": self.budget_report(),
            "active_episode_lengths": [
                len(episode["state"]) for episode in self._active
            ],
            "incomplete_episode_resume_policy": "discard_after_env_reset",
        }

    def validate_resume_state(self, saved: dict[str, Any]) -> None:
        """Cross-check checkpoint state against atomically durable chunks."""

        current = self.checkpoint_state()
        comparable = (
            "kind",
            "output_dir",
            "total_transitions",
            "chunks_written",
            "next_episode_id",
        )
        mismatches = {
            name: (saved.get(name), current[name])
            for name in comparable
            if saved.get(name) != current[name]
        }
        saved_next_global = saved.get("next_global_step")
        if (
            not isinstance(saved_next_global, int)
            or saved_next_global < current["next_global_step"]
        ):
            mismatches["next_global_step"] = (
                saved_next_global,
                f">={current['next_global_step']}",
            )
        saved_budget = saved.get("budget")
        if not isinstance(saved_budget, dict):
            mismatches["budget"] = (saved_budget, current["budget"])
        else:
            for name in (
                "selection_strategy",
                "configured_max_transitions",
                "effective_durable_upper_bound",
                "total_updates",
            ):
                if saved_budget.get(name) != current["budget"].get(name):
                    mismatches[f"budget.{name}"] = (
                        saved_budget.get(name),
                        current["budget"].get(name),
                    )
            saved_stages = saved_budget.get("stages")
            current_stages = current["budget"]["stages"]
            if not isinstance(saved_stages, list) or len(saved_stages) != 3:
                mismatches["budget.stages"] = (saved_stages, current_stages)
            else:
                for index, (saved_stage, current_stage) in enumerate(
                    zip(saved_stages, current_stages)
                ):
                    for name in (
                        "name",
                        "start_update",
                        "end_update",
                        "transition_quota",
                        "target_episodes",
                        "selected_episodes",
                        "selected_transitions",
                    ):
                        if saved_stage.get(name) != current_stage.get(name):
                            mismatches[f"budget.stages[{index}].{name}"] = (
                                saved_stage.get(name),
                                current_stage.get(name),
                            )
        if mismatches:
            raise ValueError(f"Collector resume state mismatch: {mismatches}")

    def record(
        self,
        state: np.ndarray,
        requested_action: np.ndarray,
        transition: VectorStep,
        *,
        update: int,
        global_step_start: int,
    ) -> None:
        states = np.asarray(state, dtype=np.float32)
        if states.shape != (self.num_envs, self.obs_dim):
            raise ValueError("collector state batch has the wrong shape")
        requested_actions = np.asarray(requested_action, dtype=np.float32)
        if requested_actions.shape != (self.num_envs, self.action_dim):
            raise ValueError("collector requested-action batch has the wrong shape")
        if self._last_update and int(update) < self._last_update:
            raise ValueError("collector update tags must be monotonic")
        self._stage_index(int(update))
        batch_global_steps = np.arange(
            int(global_step_start),
            int(global_step_start) + self.num_envs,
            dtype=np.int64,
        )
        if int(batch_global_steps[0]) <= self._last_seen_global_step:
            raise ValueError("collector global steps must be globally increasing")
        self._last_update = int(update)
        self._last_training_global_step = int(global_step_start + self.num_envs)
        self._last_seen_global_step = int(batch_global_steps[-1])
        self._next_global_step = self._last_seen_global_step + 1
        for index in range(self.num_envs):
            episode = self._active[index]
            episode["state"].append(states[index].copy())
            episode["requested_action"].append(requested_actions[index].copy())
            episode["action"].append(transition.applied_action[index].copy())
            episode["next_state"].append(
                transition.transition_observation[index].copy()
            )
            episode["reward"].append(float(transition.reward[index]))
            episode["discount"].append(float(transition.discount[index]))
            episode["terminated"].append(bool(transition.terminated[index]))
            episode["truncated"].append(bool(transition.truncated[index]))
            episode["reset_seed"].append(int(transition.reset_seed[index]))
            episode["update"].append(int(update))
            episode["global_step"].append(int(batch_global_steps[index]))
            if bool(transition.reset_boundary[index]):
                episode_length = len(episode["state"])
                reset_seeds = np.asarray(episode["reset_seed"], dtype=np.int64)
                if len(np.unique(reset_seeds)) != 1:
                    raise RuntimeError("An episode contains multiple reset seeds")
                updates = np.asarray(episode["update"], dtype=np.int64)
                if len(updates) > 1 and not bool(np.all(np.diff(updates) >= 0)):
                    raise RuntimeError("An episode has non-monotonic update tags")
                global_steps = np.asarray(episode["global_step"], dtype=np.int64)
                if len(global_steps) > 1 and not bool(
                    np.all(np.diff(global_steps) > 0)
                ):
                    raise RuntimeError("An episode has non-increasing global steps")
                stage_index = self._stage_index(int(update))
                selected = self._select_episode(
                    stage_index, int(update), episode_length
                )
                finalized = {
                    "state": np.asarray(episode["state"], dtype=np.float32),
                    "requested_action": np.asarray(
                        episode["requested_action"], dtype=np.float32
                    ),
                    "action": np.asarray(episode["action"], dtype=np.float32),
                    "next_state": np.asarray(
                        episode["next_state"], dtype=np.float32
                    ),
                    "reward": np.asarray(episode["reward"], dtype=np.float32),
                    "discount": np.asarray(
                        episode["discount"], dtype=np.float32
                    ),
                    "terminated": np.asarray(
                        episode["terminated"], dtype=np.bool_
                    ),
                    "truncated": np.asarray(
                        episode["truncated"], dtype=np.bool_
                    ),
                    "reset_seed": reset_seeds,
                    "update": updates,
                    "global_step": global_steps,
                }
                self._active[index] = self._empty_episode()
                if selected:
                    namespace_end = (
                        self.seed_index + 1
                    ) * self.EPISODE_NAMESPACE_SIZE
                    if self._next_episode_id >= namespace_end:
                        raise RuntimeError(
                            "Collection exhausted its episode-id namespace"
                        )
                    finalized["episode_id"] = np.full(
                        episode_length,
                        self._next_episode_id,
                        dtype=np.int64,
                    )
                    finalized["step_index"] = np.arange(
                        episode_length, dtype=np.int64
                    )
                    finalized["collection_stage"] = np.full(
                        episode_length,
                        self.STAGE_NAMES[stage_index],
                    )
                    self._next_episode_id += 1
                    self._complete.append(finalized)
                    self._pending_transitions += episode_length
                    self._stage_episode_counts[stage_index] += 1
                    self._stage_transition_counts[stage_index] += episode_length
                else:
                    self._stage_skipped_episodes[stage_index] += 1
    def flush(self, *, force: bool = True) -> Path | None:
        if not self._complete:
            return None
        if not force and self._pending_transitions < self.flush_transitions:
            return None
        fields = (
            *self.FIELDS,
            "episode_id",
            "step_index",
            "collection_stage",
        )
        chunk = {
            name: np.concatenate([episode[name] for episode in self._complete])
            for name in fields
        }
        chunk["done"] = np.logical_or(chunk["terminated"], chunk["truncated"])
        # These are genuine environment boundaries, never collector-imposed
        # cuts.  Keep both causes explicit for collection schema v4.
        chunk["collector_truncated"] = np.zeros(
            len(chunk["state"]), dtype=np.bool_
        )
        chunk["collection_schema_version"] = np.asarray(
            self.COLLECTION_SCHEMA_VERSION, dtype=np.int64
        )
        chunk["protocol_json"] = np.asarray(self.protocol_json)
        chunk["environment_protocol_json"] = np.asarray(
            self.environment_protocol_json
        )
        chunk["protocol_fingerprint"] = np.asarray(self.protocol_fingerprint)
        chunk["task"] = np.asarray(self.task_name)
        chunk["seed_index"] = np.asarray(self.seed_index, dtype=np.int64)
        chunk["seed_dir"] = np.asarray(self.seed_dir)
        chunk["collection_selection_strategy"] = np.asarray(
            self.SELECTION_STRATEGY
        )
        chunk["collection_max_transitions"] = np.asarray(
            self.max_transitions, dtype=np.int64
        )
        chunk["collection_total_updates"] = np.asarray(
            self.total_updates, dtype=np.int64
        )
        for name, value in self.lineage.items():
            chunk[name] = np.asarray(value)
        chunk["rng_state_after_json"] = np.asarray(
            json.dumps(
                {
                    "python": _json_safe(random.getstate()),
                    "numpy": _json_safe(np.random.get_state()),
                    "torch": _json_safe(torch.random.get_rng_state()),
                    "collector": {
                        "next_episode_id": self._next_episode_id,
                        "next_global_step": self._next_global_step,
                        "training_update": self._last_update,
                        "training_global_step": self._last_training_global_step,
                    },
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if chunk["state"].shape[1:] != (self.obs_dim,) or chunk[
            "action"
        ].shape[1:] != (self.action_dim,):
            raise RuntimeError("collector assembled an invalid chunk")
        path = self.output_dir / f"coverage_{self._chunk_index:06d}.npz"
        temporary = path.with_name(f".{path.stem}.tmp.npz")
        try:
            np.savez_compressed(temporary, **chunk)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._chunk_index += 1
        self.chunks_written += 1
        self.total_transitions += len(chunk["state"])
        self._complete.clear()
        self._pending_transitions = 0
        _atomic_json(
            self.output_dir / "collection_status.json",
            {
                **self.metadata,
                "collection_schema_version": self.COLLECTION_SCHEMA_VERSION,
                "environment_protocol_json": self.environment_protocol_json,
                "collection_protocol": self.collection_protocol,
                "protocol_fingerprint": self.protocol_fingerprint,
                "seed_index": self.seed_index,
                "seed_dir": self.seed_dir,
                "total_transitions": self.total_transitions,
                "chunks_written": self.chunks_written,
                "next_episode_id": self._next_episode_id,
                "next_global_step": self._next_global_step,
                "last_chunk": path.name,
                "budget": self.budget_report(),
            },
        )
        return path


def _model_inputs(
    actor_type: str,
    state: torch.Tensor,
    koopman: nn.Module | None,
    center: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if actor_type == "PPO":
        return state, None
    if koopman is None or center is None or scale is None:
        raise ValueError(f"{actor_type} requires Koopman features and normalizers")
    with torch.no_grad():
        normalized_observation = (state - center) / scale
        lifted = koopman.lift(normalized_observation)
    return normalized_observation, lifted


def _actor_value(
    actor_type: str,
    actor: nn.Module,
    value_network: ValueNetwork,
    state: torch.Tensor,
    koopman: nn.Module | None,
    center: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_observation, lifted = _model_inputs(
        actor_type, state, koopman, center, scale
    )
    mean = actor_mean(actor_type, actor, normalized_observation, lifted)
    return mean, value_network(normalized_observation)


def _resume_config_compatible(
    previous: dict[str, Any], current: dict[str, Any]
) -> bool:
    # YAML-bound training semantics (including total_timesteps) are immutable.
    # Only persistence cadence and a test/operational wall-time stop may vary;
    # formal runs are additionally bound by the config fingerprint.
    mutable = {
        "max_wall_time_seconds",
        "checkpoint_interval_updates",
        "collect_flush_transitions",
    }
    return (
        {key: value for key, value in previous.items() if key not in mutable}
        == {key: value for key, value in current.items() if key not in mutable}
    )


def _optional_mean(values: deque[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _coerce_experiment_config(
    value: ExperimentConfig | Path | str | None,
) -> ExperimentConfig:
    if isinstance(value, ExperimentConfig):
        return value
    if value is None:
        raise PermissionError(
            "Approval-bound training requires an experiment config path"
        )
    return load_experiment_config(value)


def _validate_yaml_binding(
    experiment: ExperimentConfig,
    profile: str | None,
    train_seed_index: int,
    task_name: str,
    config: PPOConfig,
    actor_config: ActorConfig,
) -> str:
    if profile is None:
        raise PermissionError("Approval-bound training requires --profile")
    if experiment.task != task_name:
        raise ValueError(
            f"CLI task {task_name!r} does not match YAML task {experiment.task!r}"
        )
    expected_ppo = ppo_config_from_experiment(
        experiment,
        profile,
        train_seed_index=train_seed_index,
    )
    if asdict(config) != asdict(expected_ppo):
        raise ValueError(
            "PPOConfig does not exactly match the resolved approval-bound YAML"
        )
    if actor_config.to_dict() != experiment.actor_config.to_dict():
        raise ValueError(
            "ActorConfig does not exactly match the approval-bound YAML"
        )
    return profile


def _koopman_checkpoint_field(payload: dict[str, Any], name: str) -> Any:
    if name in payload:
        return payload[name]
    training_state = payload.get("training_state")
    if isinstance(training_state, dict):
        return training_state.get(name)
    return None


def _validate_koopman_authorization_lineage(
    payload: dict[str, Any],
    *,
    task_name: str,
    authorization_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Validate the frozen model's own lineage independently of actor approval.

    A Koopman model is a fixed input artifact for structured policy training.
    Its data/model seed and approval therefore need not equal the policy
    training seed and approval.  We still fail closed on the model's task,
    protocol (checked by the caller), approval status, and complete immutable
    provenance.  ``authorization_metadata`` is intentionally retained in the
    private signature for compatibility with older callers, but actor identity
    is not compared to model identity.
    """

    del authorization_metadata

    checkpoint_task = (
        _koopman_checkpoint_field(payload, "task")
        or _koopman_checkpoint_field(payload, "task_name")
        or _koopman_checkpoint_field(payload, "state_kind")
    )
    expected = {"task": task_name, "training_approved": True}
    actual = {
        "task": checkpoint_task,
        **{
            name: _koopman_checkpoint_field(payload, name)
            for name in expected
            if name != "task"
        },
    }
    mismatches = {
        name: (actual.get(name), value)
        for name, value in expected.items()
        if actual.get(name) != value
    }
    lineage = {
        name: _koopman_checkpoint_field(payload, name)
        for name in (
            "dataset_sha256",
            "config_fingerprint",
            "approval_profile",
            "approval_file_sha256",
            "preflight_report_sha256",
        )
    }
    dataset_sha256 = lineage["dataset_sha256"]
    sha_fields = (
        "dataset_sha256",
        "approval_file_sha256",
        "preflight_report_sha256",
    )
    for name in sha_fields:
        value = lineage[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            mismatches[name] = (value, "64 lowercase hex")
    config_fingerprint = lineage["config_fingerprint"]
    if (
        not isinstance(config_fingerprint, str)
        or not config_fingerprint.startswith("sha256:")
        or len(config_fingerprint) != 71
        or any(
            character not in "0123456789abcdef"
            for character in config_fingerprint.removeprefix("sha256:")
        )
    ):
        mismatches["config_fingerprint"] = (
            config_fingerprint,
            "sha256: followed by 64 lowercase hex",
        )
    if lineage["approval_profile"] not in {"development", "benchmark"}:
        mismatches["approval_profile"] = (
            lineage["approval_profile"],
            "development or benchmark",
        )
    authorization_kind = _koopman_checkpoint_field(payload, "authorization_kind")
    if authorization_kind != "dmc_training_approval_v1":
        mismatches["authorization_kind"] = (
            authorization_kind,
            "dmc_training_approval_v1",
        )
    if mismatches:
        raise ValueError(
            f"Koopman checkpoint authorization lineage mismatch: {mismatches}"
        )
    return lineage


def train(
    task_name: str,
    actor_type: str,
    output_dir: Path,
    config: PPOConfig,
    *,
    actor_config: ActorConfig | None = None,
    koopman_path: Path | None = None,
    collect_dir: Path | None = None,
    control_timestep: float | None = None,
    time_limit: float | None = None,
    device_name: str = "auto",
    env_workers: int | None = None,
    resume: bool = True,
    env_factory: EnvFactory | None = None,
    collection_seed_index: int = 0,
    collection_seed_dir: str | None = None,
    experiment_config: ExperimentConfig | Path | str | None = None,
    profile: str | None = None,
    train_seed_index: int = 0,
    approval_file: Path | None = None,
    preflight_file: Path | None = None,
    dry_run: bool = False,
    _test_authorization: object | None = None,
    _test_interrupt_after_update: int | None = None,
) -> dict[str, Any]:
    """Run PPO and return a compact final report.

    A real optimization run is fail-closed through the shared DMC approval
    validator.  The private test token exists only for synthetic unit tests;
    CLI and public callers must bind the resolved YAML, profile, reviewed
    preflight bytes and approval artifact.  ``dry_run`` never authorizes or
    performs optimization.
    """

    if task_name not in TASK_SPECS:
        raise ValueError(f"Unknown DMC task {task_name!r}")
    if actor_type not in ACTOR_TYPES:
        raise ValueError(f"Unknown actor type {actor_type!r}")
    if actor_type != "PPO" and koopman_path is None:
        raise ValueError(f"{actor_type} requires --koopman")
    config.validate()
    actor_config = actor_config or ActorConfig()
    actor_config.validate()
    if (
        actor_type == "AC-MPC-MPVE"
        and config.mpve_horizon > actor_config.kmpc_horizon
    ):
        raise ValueError(
            "mpve_horizon cannot exceed ActorConfig.kmpc_horizon"
        )
    if collect_dir is not None and config.collect_max_transitions is None:
        raise ValueError(
            "collect_dir requires YAML-bound collect_max_transitions; flush "
            "cadence is not a collection budget"
        )
    if collect_dir is not None and actor_type != "PPO":
        raise ValueError(
            "primary ppo_training_stages collection requires actor_type=PPO"
        )
    if _test_authorization is not None and (
        _test_authorization is not _TEST_ONLY_AUTHORIZATION
    ):
        raise PermissionError("Invalid private test authorization token")
    test_authorized = _test_authorization is _TEST_ONLY_AUTHORIZATION
    if _test_interrupt_after_update is not None:
        if not test_authorized:
            raise PermissionError("Test interruption hook requires private test auth")
        if _test_interrupt_after_update < 1:
            raise ValueError("_test_interrupt_after_update must be positive")
    experiment: ExperimentConfig | None = None
    bound_profile: str | None = None
    if not test_authorized:
        if control_timestep is not None or time_limit is not None:
            raise ValueError(
                "Approval-bound DMC training forbids protocol overrides; use "
                "the native protocol declared by the YAML"
            )
        experiment = _coerce_experiment_config(experiment_config)
        bound_profile = _validate_yaml_binding(
            experiment,
            profile,
            train_seed_index,
            task_name,
            config,
            actor_config,
        )
        if collection_seed_index != train_seed_index:
            raise ValueError(
                "collection_seed_index must equal the approved train_seed_index"
            )
        if collect_dir is not None:
            expected_suffix = (
                task_name,
                str(bound_profile),
                f"seed_{config.seed}",
            )
            collect_parts = Path(collect_dir).parts
            if tuple(collect_parts[-3:]) != expected_suffix:
                raise ValueError(
                    "Formal PPO collect_dir must end in "
                    f"{Path(*expected_suffix)} to isolate task/profile/seed"
                )
        if preflight_file is None or (not dry_run and approval_file is None):
            raise PermissionError(
                "Formal runs require --preflight-file; training also requires "
                "--approval-file"
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"

    resume_payload: dict[str, Any] | None = None
    start_update = 0
    if resume and latest_path.exists():
        resume_payload = torch.load(latest_path, map_location="cpu", weights_only=False)
        start_update = int(resume_payload.get("update", 0))

    _seed_all(config.seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    spec = get_task_spec(task_name)

    koopman = None
    koopman_payload: dict[str, Any] | None = None
    koopman_sha256: str | None = None
    center: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    normalizer_metadata: dict[str, list[float]] | None = None
    mpve_reward_predictor: nn.Module | None = None
    if koopman_path is not None:
        koopman_path = Path(koopman_path)
        koopman, koopman_payload = load_koopman(koopman_path, task_name, device)
        center_array, scale_array = normalizer_arrays(koopman_payload, task_name)
        center = torch.as_tensor(center_array, device=device, dtype=torch.float32)
        scale = torch.as_tensor(scale_array, device=device, dtype=torch.float32)
        normalizer_metadata = {
            "center": center_array.tolist(),
            "scale": scale_array.tolist(),
        }
        koopman_sha256 = _sha256(koopman_path)
        if actor_type == "AC-MPC-MPVE":
            validate_mpve_reward_source(task_name, config.mpve_reward_source)
            if config.mpve_reward_source == OFFICIAL_OBSERVATION_ORACLE:
                mpve_reward_predictor = ExactObservationRewardOracle(
                    task_name, center, scale
                ).to(device)
            elif config.mpve_reward_source == LEARNED_TRANSITION_REWARD:
                learned_reward_model = reward_model_from_checkpoint(
                    koopman_payload,
                    device=device,
                )
                if (
                    learned_reward_model.state_dim != spec.obs_dim
                    or learned_reward_model.action_dim != spec.action_dim
                ):
                    raise ValueError(
                        "Reward-model dimensions do not match the DMC task"
                    )
                mpve_reward_predictor = learned_reward_model
            else:  # pragma: no cover - guarded by strict config validation.
                raise AssertionError("Validated MPVE reward source drifted")
            mpve_reward_predictor.requires_grad_(False)
            mpve_reward_predictor.eval()

    # Frozen model reconstruction may allocate randomly initialized modules
    # before loading their state (MPVE additionally reconstructs a reward MLP).
    # Re-seed after every auxiliary load so KMPC and AC-MPC-MPVE have identical
    # actor/critic initialization and subsequent action RNG under the same seed.
    _seed_all(config.seed)
    actor = build_actor(
        actor_type,
        task_name,
        device,
        koopman=koopman,
        config=actor_config,
    )
    # Actor architectures deliberately use method-specific initialization.
    # Re-seed the common raw-observation critic independently so actor module
    # construction cannot perturb its initial weights across methods.
    _seed_all(config.seed)
    value_network = ValueNetwork(spec.obs_dim, config.critic_hidden_dim).to(device)
    observation_normalizer = ObservationRunningMeanStd(spec.obs_dim)
    return_normalizer = AcmeReturnNormalizer(config.normalization_ema_tau)
    log_std: nn.Parameter | None = None
    policy_trainable = list(actor.parameters())
    critic_trainable = list(value_network.parameters())
    if actor_type != "PPO":
        log_std = nn.Parameter(
            torch.full(
                (spec.action_dim,),
                math.log(config.initial_std),
                dtype=torch.float32,
                device=device,
            )
        )
        policy_trainable.append(log_std)
    trainable = [*policy_trainable, *critic_trainable]

    # A resumed run starts fresh environments at a deterministic disjoint seed
    # range.  Optimizer/policy/RNG state resumes exactly; MuJoCo physics state is
    # intentionally not serialized at update boundaries.
    env_seed = config.seed + start_update * config.num_envs * 1009
    vector_env = make_dmc_vector_env(
        task_name,
        config.num_envs,
        env_seed,
        workers=env_workers,
        control_timestep=control_timestep,
        time_limit=time_limit,
        env_factory=env_factory,
    )
    resolved_env_workers = int(getattr(vector_env, "workers", 1))
    protocol = vector_env.protocol
    environment_protocol_json = canonical_json(protocol)
    environment_fingerprint = protocol_fingerprint(protocol)
    if test_authorized:
        authorization_metadata: dict[str, Any] = {
            "authorization_kind": "private_test_only",
            "training_approved": True,
            "config_fingerprint": "test-only",
            "approval_profile": None,
            "approval_file_sha256": None,
            "preflight_report_sha256": None,
        }
    elif dry_run:
        if experiment is None or bound_profile is None or preflight_file is None:
            vector_env.close()
            raise AssertionError("resolved dry-run approval inputs are missing")
        try:
            validate_training_preflight(
                experiment,
                bound_profile,
                preflight_file,
                runtime_protocol_fingerprint=environment_fingerprint,
            )
        except BaseException:
            vector_env.close()
            raise
        authorization_metadata = {
            "authorization_kind": "training_free_run_manifest",
            "training_approved": False,
            "config_fingerprint": experiment.fingerprint,
            "approval_profile": bound_profile,
            "approval_file_sha256": None,
            "preflight_report_sha256": _sha256(Path(preflight_file)),
        }
    else:
        if (
            experiment is None
            or bound_profile is None
            or approval_file is None
            or preflight_file is None
        ):
            vector_env.close()
            raise AssertionError("approval inputs disappeared after validation")
        try:
            approval_payload = validate_training_approval(
                experiment,
                bound_profile,
                approval_file,
                preflight_file,
                runtime_protocol_fingerprint=environment_fingerprint,
            )
        except BaseException:
            vector_env.close()
            raise
        authorization_metadata = {
            "authorization_kind": approval_payload["kind"],
            "training_approved": True,
            "config_fingerprint": experiment.fingerprint,
            "approval_profile": bound_profile,
            "approval_file_sha256": _sha256(Path(approval_file)),
            "preflight_report_sha256": _sha256(Path(preflight_file)),
        }
    koopman_protocol_fingerprint = (
        checkpoint_protocol_fingerprint(koopman_payload)
        if koopman_payload is not None
        else None
    )
    if (
        koopman_protocol_fingerprint is not None
        and koopman_protocol_fingerprint != environment_fingerprint
    ):
        vector_env.close()
        raise ValueError(
            "Koopman checkpoint protocol does not match the training environment"
        )
    koopman_lineage: dict[str, Any] | None = None
    if koopman_payload is not None:
        if not test_authorized and not dry_run:
            koopman_lineage = _validate_koopman_authorization_lineage(
                koopman_payload,
                task_name=task_name,
                authorization_metadata=authorization_metadata,
            )
        else:
            koopman_lineage = {
                "dataset_sha256": _koopman_checkpoint_field(
                    koopman_payload, "dataset_sha256"
                ),
                "config_fingerprint": _koopman_checkpoint_field(
                    koopman_payload, "config_fingerprint"
                ),
                "approval_profile": _koopman_checkpoint_field(
                    koopman_payload, "approval_profile"
                ),
                "approval_file_sha256": _koopman_checkpoint_field(
                    koopman_payload, "approval_file_sha256"
                ),
                "preflight_report_sha256": _koopman_checkpoint_field(
                    koopman_payload, "preflight_report_sha256"
                ),
            }
    koopman_dataset_sha256 = (
        koopman_lineage.get("dataset_sha256")
        if koopman_lineage is not None
        else None
    )
    koopman_config_fingerprint = (
        koopman_lineage.get("config_fingerprint")
        if koopman_lineage is not None
        else None
    )
    ppo_config_mapping = asdict(config)
    actor_config_mapping = actor_config.to_dict()
    resolved_execution_spec = (
        resolve_execution_spec(experiment, bound_profile)
        if experiment is not None and bound_profile is not None
        else None
    )
    evaluation_seeds = (
        list(resolved_execution_spec["evaluation_seeds"])
        if resolved_execution_spec is not None
        else None
    )
    evaluation_episodes_per_seed = (
        int(resolved_execution_spec["evaluation"]["episodes_per_seed"])
        if resolved_execution_spec is not None
        else None
    )
    evaluation_reference_episodes_per_seed = (
        int(
            resolved_execution_spec["evaluation"][
                "reference_episodes_per_seed"
            ]
        )
        if resolved_execution_spec is not None
        else None
    )
    diagnostic_every_steps = (
        int(resolved_execution_spec["evaluation"]["diagnostic_every_steps"])
        if resolved_execution_spec is not None
        else None
    )
    gradient_clip_contract = (
        "separate_policy_log_std_and_critic_global_norm_v1"
        if actor_type in ("KMPC", "AC-MPC-MPVE")
        else "acme_single_global_norm_v1"
    )
    structured_initialization = {
        "KLQR": "koopman_lqr_cost_map_module_defaults_v1",
        "AB-PQ": "low_rank_quadratic_value_module_defaults_v1",
        "KMPC": "koopman_mpc_cost_map_module_defaults_v1",
        "AC-MPC-MPVE": "koopman_mpc_cost_map_module_defaults_v1",
    }
    network_initialization_contract = {
        "plain_ppo_policy_torso": (
            HAIKU_DEFAULT_LINEAR_INITIALIZATION
            if actor_type == "PPO"
            else None
        ),
        "plain_ppo_policy_location_scale_heads": (
            "variance_scaling_scale1_fan_in_uniform_v1"
            if actor_type == "PPO"
            else None
        ),
        "critic": HAIKU_DEFAULT_LINEAR_INITIALIZATION,
        "critic_input": "normalized_raw_task_observation_v1",
        "critic_seed": int(config.seed),
        "critic_seed_independent_of_actor_construction": True,
        "structured_actor": (
            structured_initialization[actor_type]
            if actor_type != "PPO"
            else None
        ),
        "seed_semantics": (
            "same_seed_method_specific_initializers_not_equal_parameters_v1"
        ),
        "strict_parameter_pairing_group": (
            "KMPC_AC-MPC-MPVE"
            if actor_type in ("KMPC", "AC-MPC-MPVE")
            else None
        ),
        "alignment": "acme_aligned_pytorch_synchronous_reference",
    }
    value_expansion_metadata = {
        "enabled": actor_type == "AC-MPC-MPVE",
        "kind": (
            "mpve_td_k_tro25_eq8_eq9_v1"
            if actor_type == "AC-MPC-MPVE"
            else None
        ),
        "actor_shared_with": (
            "KMPC" if actor_type == "AC-MPC-MPVE" else None
        ),
        "horizon": (
            config.mpve_horizon if actor_type == "AC-MPC-MPVE" else None
        ),
        "value_loss_coefficient": (
            config.mpve_value_loss_coefficient
            if actor_type == "AC-MPC-MPVE"
            else None
        ),
        "prediction_gradient": "detached",
        "terminal_target_gradient": "detached",
        "standard_gae_value_loss_retained": True,
        "reward": (
            exact_reward_oracle_metadata(task_name)
            if actor_type == "AC-MPC-MPVE"
            and config.mpve_reward_source == OFFICIAL_OBSERVATION_ORACLE
            else {
                "source": LEARNED_TRANSITION_REWARD,
                "model_input_contract": transition_reward_input_contract(),
                "checkpoint_field": "reward_model_state",
            }
            if actor_type == "AC-MPC-MPVE"
            else None
        ),
    }

    if resume_payload is not None:
        expected = {
            "kind": "dmc_ppo_actor",
            "training_spec_version": TRAINING_SPEC_VERSION,
            "task": task_name,
            "actor_type": actor_type,
            "policy_initialization": "from_scratch",
            "training_seed": config.seed,
            "actor_config": actor_config_mapping,
            "protocol": protocol,
            "environment_protocol_json": environment_protocol_json,
            "protocol_fingerprint": environment_fingerprint,
            "koopman_sha256": koopman_sha256,
            "koopman_lineage": koopman_lineage,
            "koopman_dataset_sha256": koopman_dataset_sha256,
            "koopman_config_fingerprint": koopman_config_fingerprint,
            "resolved_execution_spec": resolved_execution_spec,
            "train_seed_index": train_seed_index,
            "evaluation_seeds": evaluation_seeds,
            "evaluation_episodes_per_seed": evaluation_episodes_per_seed,
            "evaluation_reference_episodes_per_seed": (
                evaluation_reference_episodes_per_seed
            ),
            "diagnostic_every_steps": diagnostic_every_steps,
            "running_episode_resume_policy": "discard_after_env_reset",
            "value_expansion": value_expansion_metadata,
            "gradient_clip_contract": gradient_clip_contract,
            "network_initialization_contract": network_initialization_contract,
            **authorization_metadata,
        }
        mismatches = {
            key: (resume_payload.get(key), value)
            for key, value in expected.items()
            if resume_payload.get(key) != value
        }
        if not _resume_config_compatible(
            dict(resume_payload.get("ppo_config", {})), ppo_config_mapping
        ):
            mismatches["ppo_config"] = (
                resume_payload.get("ppo_config"),
                ppo_config_mapping,
            )
        if mismatches:
            vector_env.close()
            raise ValueError(f"Resume checkpoint metadata mismatch: {mismatches}")
    metadata: dict[str, Any] = {
        "kind": "dmc_ppo_actor",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_spec_version": TRAINING_SPEC_VERSION,
        "task": task_name,
        "actor_type": actor_type,
        "actor_name": actor_type,
        "policy_initialization": "from_scratch",
        "protocol": protocol,
        "environment_protocol_json": environment_protocol_json,
        "protocol_fingerprint": environment_fingerprint,
        "training_seed": config.seed,
        "actor_config": actor_config_mapping,
        "ppo_config": ppo_config_mapping,
        "koopman_path": (
            str(koopman_path.resolve()) if koopman_path is not None else None
        ),
        "koopman_sha256": koopman_sha256,
        "koopman_protocol_fingerprint": koopman_protocol_fingerprint,
        "koopman_lineage": koopman_lineage,
        "koopman_dataset_sha256": koopman_dataset_sha256,
        "koopman_config_fingerprint": koopman_config_fingerprint,
        "normalizer": normalizer_metadata,
        "resolved_execution_spec": resolved_execution_spec,
        "train_seed_index": train_seed_index,
        "evaluation_seeds": evaluation_seeds,
        "evaluation_episodes_per_seed": evaluation_episodes_per_seed,
        "evaluation_reference_episodes_per_seed": (
            evaluation_reference_episodes_per_seed
        ),
        "diagnostic_every_steps": diagnostic_every_steps,
        "running_episode_resume_policy": "discard_after_env_reset",
        "value_expansion": value_expansion_metadata,
        "gradient_clip_contract": gradient_clip_contract,
        "network_initialization_contract": network_initialization_contract,
        "environment_runner": {
            "kind": (
                "spawn_process_vector_v1"
                if resolved_env_workers > 1
                else "synchronous_vector_v1"
            ),
            "workers": resolved_env_workers,
            "num_envs": config.num_envs,
            "identity_scope": "execution_only",
        },
        "normalization_contract": {
            "observation": (
                "acme_welford_running_mean_std_v1"
                if actor_type == "PPO" and config.normalize_observation
                else "frozen_koopman_checkpoint_center_scale_v1"
                if actor_type != "PPO"
                else None
            ),
            "observation_scope": (
                "plain_ppo_raw_task_state"
                if actor_type == "PPO"
                else "structured_raw_task_state_shared_by_critic_and_controller_lift"
            ),
            "critic_input": "normalized_raw_task_observation_v1",
            "controller_input": (
                "normalized_raw_task_observation_v1"
                if actor_type == "PPO"
                else "frozen_koopman_lifted_state_v1"
            ),
            "advantage": (
                "zero_debiased_ema_mean_absolute_v1"
                if config.normalize_advantage
                else None
            ),
            "value": (
                "zero_debiased_ema_first_second_moment_v1"
                if config.normalize_value
                else None
            ),
        },
        **authorization_metadata,
    }
    if dry_run:
        manifest_path = output_dir / "run_manifest.json"
        manifest = {
            **metadata,
            "kind": "dmc_ppo_training_free_run_manifest",
            "training_approved": False,
            "device": str(device),
            "would_resume_from_update": start_update,
            "optimization_steps": 0,
            "environment_steps": 0,
            "notice": (
                "This manifest is not an approval artifact. Use the shared "
                "experiments.dmc.preflight and experiments.dmc.approval flow."
            ),
        }
        _atomic_json(manifest_path, manifest)
        vector_env.close()
        return {
            "kind": "dmc_ppo_training_free_run_manifest",
            "task": task_name,
            "actor_type": actor_type,
            "protocol_fingerprint": environment_fingerprint,
            "config_fingerprint": authorization_metadata["config_fingerprint"],
            "run_manifest": str(manifest_path.resolve()),
            "would_resume_from_update": start_update,
            "optimization_steps": 0,
            "environment_steps": 0,
        }
    # Optimizer construction is intentionally below both the shared approval
    # validation and the training-free manifest return.  Thus an unapproved or
    # dry-run invocation cannot allocate optimizer state, let alone update it.
    optimizer = torch.optim.Adam(
        trainable,
        lr=config.learning_rate,
        eps=config.adam_epsilon,
    )
    if resume_payload is not None:
        actor.load_state_dict(resume_payload["actor_state"])
        value_network.load_state_dict(resume_payload["value_state"])
        if log_std is None:
            if resume_payload.get("log_std") is not None:
                raise ValueError("State-dependent PPO must not save global log_std")
        else:
            saved_log_std = resume_payload.get("log_std")
            if not isinstance(saved_log_std, torch.Tensor):
                raise ValueError("Structured actor checkpoint is missing log_std")
            log_std.data.copy_(saved_log_std.to(device))
        optimizer.load_state_dict(resume_payload["optimizer_state"])
        observation_normalizer.load_state_dict(
            resume_payload.get("observation_normalizer_state")
        )
        return_normalizer.load_state_dict(
            resume_payload.get("return_normalizer_state")
        )
        _restore_rng_state(resume_payload.get("rng_state"))
    _atomic_json(
        output_dir / "run_config.json",
        {
            **metadata,
            "device": str(device),
            "resume_from_update": start_update,
        },
    )
    collector = (
        OnPolicyEpisodeCollector(
            Path(collect_dir),
            config.num_envs,
            spec.obs_dim,
            spec.action_dim,
            flush_transitions=config.collect_flush_transitions,
            metadata={
                "task": task_name,
                "protocol": protocol,
                "protocol_fingerprint": environment_fingerprint,
                "training_seed": config.seed,
                "actor_type": actor_type,
                "training_approved": authorization_metadata[
                    "training_approved"
                ],
                "config_fingerprint": authorization_metadata[
                    "config_fingerprint"
                ],
                "approval_profile": authorization_metadata["approval_profile"],
                "approval_file_sha256": authorization_metadata[
                    "approval_file_sha256"
                ],
                "preflight_report_sha256": authorization_metadata[
                    "preflight_report_sha256"
                ],
                "authorization_kind": authorization_metadata[
                    "authorization_kind"
                ],
                "train_seed_index": train_seed_index,
            },
            seed_index=collection_seed_index,
            seed_dir=collection_seed_dir,
            max_transitions=int(config.collect_max_transitions),
            total_updates=config.number_updates,
        )
        if collect_dir is not None
        else None
    )
    if resume_payload is not None:
        saved_collector_state = resume_payload.get("collector_state")
        if collector is None and saved_collector_state is not None:
            vector_env.close()
            raise ValueError(
                "Resume checkpoint used data collection but collect_dir is absent"
            )
        if collector is not None:
            if not isinstance(saved_collector_state, dict):
                vector_env.close()
                raise ValueError(
                    "Resume checkpoint is missing strict collector_state"
                )
            try:
                collector.validate_resume_state(saved_collector_state)
            except BaseException:
                vector_env.close()
                raise

    global_step = int(
        resume_payload.get("global_step", start_update * config.batch_size)
        if resume_payload is not None
        else 0
    )
    best_return = float(
        resume_payload.get("best_return", -float("inf"))
        if resume_payload is not None
        else -float("inf")
    )
    best_return_update = int(
        resume_payload.get("best_return_update", 0)
        if resume_payload is not None
        else 0
    )
    def resume_history(name: str) -> list[float]:
        if resume_payload is None:
            return []
        raw = resume_payload.get(name)
        if not isinstance(raw, list):
            raise ValueError(f"Resume checkpoint is missing {name}")
        values = [float(value) for value in raw]
        if len(values) > 100 or not all(np.isfinite(values)):
            raise ValueError(f"Resume checkpoint has invalid {name}")
        return values

    recent_returns: deque[float] = deque(
        resume_history("recent_episode_returns"), maxlen=100
    )
    recent_lengths: deque[float] = deque(
        resume_history("recent_episode_lengths"), maxlen=100
    )
    running_returns = np.zeros(config.num_envs, dtype=np.float64)
    running_lengths = np.zeros(config.num_envs, dtype=np.int64)
    metrics_path = output_dir / "metrics.jsonl"
    if resume_payload is not None:
        _truncate_metrics_to_checkpoint(metrics_path, start_update)
    observations = vector_env.reset()
    started = time.perf_counter()
    last_report: dict[str, Any] = dict(
        resume_payload.get("last_report", {}) if resume_payload else {}
    )
    completed_update = start_update

    def checkpoint_payload(update: int) -> dict[str, Any]:
        return {
            **metadata,
            "actor_state": actor.state_dict(),
            "value_state": value_network.state_dict(),
            "log_std": None if log_std is None else log_std.detach(),
            "observation_normalizer_state": observation_normalizer.state_dict(),
            "return_normalizer_state": return_normalizer.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "update": int(update),
            "global_step": int(global_step),
            "best_return": float(best_return),
            "best_return_update": int(best_return_update),
            "rng_state": _capture_rng_state(),
            "recent_episode_returns": list(recent_returns),
            "recent_episode_lengths": list(recent_lengths),
            "last_report": last_report,
            "collector_state": (
                collector.checkpoint_state() if collector is not None else None
            ),
        }

    try:
        for update in range(start_update + 1, config.number_updates + 1):
            if config.anneal_learning_rate:
                fraction = 1.0 - (update - 1.0) / config.number_updates
                for group in optimizer.param_groups:
                    group["lr"] = fraction * config.learning_rate

            rollout_states: list[torch.Tensor] = []
            rollout_transition_states: list[torch.Tensor] = []
            rollout_actions: list[torch.Tensor] = []
            rollout_old_log_probs: list[torch.Tensor] = []
            rollout_rewards: list[torch.Tensor] = []
            rollout_discounts: list[torch.Tensor] = []
            rollout_boundaries: list[torch.Tensor] = []
            rollout_values: list[torch.Tensor] = []
            rollout_bootstrap_values: list[torch.Tensor] = []
            rollout_mpve_value_observations: list[torch.Tensor] = []
            rollout_mpve_terminal_values: list[torch.Tensor] = []
            rollout_mpve_rewards: list[torch.Tensor] = []
            rollout_ppo_scales: list[torch.Tensor] = []
            applied_bound_count = 0
            applied_action_count = 0

            for _ in range(config.rollout_steps):
                state = torch.as_tensor(
                    observations, device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    if actor_type == "PPO":
                        if not isinstance(actor, StandardPPOActor):
                            raise AssertionError("PPO actor has the wrong architecture")
                        policy_state = (
                            observation_normalizer.normalize(state)
                            if config.normalize_observation
                            else state
                        )
                        location, policy_scale = actor.distribution_parameters(
                            policy_state
                        )
                        rollout_ppo_scales.append(policy_scale.detach())
                        action = tanh_normal_sample(
                            location,
                            policy_scale,
                            action_limit=actor_config.action_limit,
                        )
                        old_log_prob = tanh_normal_log_prob(
                            location,
                            policy_scale,
                            action,
                            action_limit=actor_config.action_limit,
                        )
                        value = value_network(policy_state)
                    elif actor_type == "AC-MPC-MPVE":
                        if (
                            koopman is None
                            or center is None
                            or scale is None
                            or mpve_reward_predictor is None
                        ):
                            raise AssertionError("MPVE models are missing")
                        lifted = koopman.lift((state - center) / scale)
                        mpve_prediction = collect_mpve_prediction(
                            actor,
                            value_network,
                            koopman,
                            mpve_reward_predictor,
                            lifted,
                            horizon=config.mpve_horizon,
                            gamma=config.discount,
                        )
                        mean = mpve_prediction.action
                        value = value_network(koopman.reconstruct(lifted))
                        rollout_mpve_value_observations.append(
                            mpve_prediction.value_observations
                        )
                        rollout_mpve_terminal_values.append(
                            mpve_prediction.terminal_value
                        )
                        rollout_mpve_rewards.append(
                            mpve_prediction.predicted_rewards
                        )
                        if log_std is None:
                            raise AssertionError("Structured log_std is missing")
                        distribution = Normal(
                            mean, log_std.exp().expand_as(mean)
                        )
                        action = distribution.sample()
                        old_log_prob = distribution.log_prob(action).sum(-1)
                    else:
                        mean, value = _actor_value(
                            actor_type,
                            actor,
                            value_network,
                            state,
                            koopman,
                            center,
                            scale,
                        )
                        if log_std is None:
                            raise AssertionError("Structured log_std is missing")
                        distribution = Normal(
                            mean, log_std.exp().expand_as(mean)
                        )
                        action = distribution.sample()
                        old_log_prob = distribution.log_prob(action).sum(-1)

                transition_global_start = global_step
                action_array = action.detach().cpu().numpy()
                transition = vector_env.step(action_array)
                transition_state = torch.as_tensor(
                    transition.transition_observation,
                    device=device,
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    if actor_type == "PPO":
                        normalized_transition_state = (
                            observation_normalizer.normalize(transition_state)
                            if config.normalize_observation
                            else transition_state
                        )
                        bootstrap_value = value_network(
                            normalized_transition_state
                        )
                    else:
                        _unused_mean, bootstrap_value = _actor_value(
                            actor_type,
                            actor,
                            value_network,
                            transition_state,
                            koopman,
                            center,
                            scale,
                        )

                rollout_states.append(state)
                rollout_transition_states.append(transition_state)
                rollout_actions.append(action)
                rollout_old_log_probs.append(old_log_prob)
                rollout_rewards.append(
                    torch.as_tensor(transition.reward, device=device)
                )
                rollout_discounts.append(
                    torch.as_tensor(transition.discount, device=device)
                )
                rollout_boundaries.append(
                    torch.as_tensor(transition.reset_boundary, device=device)
                )
                rollout_values.append(value)
                rollout_bootstrap_values.append(bootstrap_value)

                if collector is not None:
                    collector.record(
                        observations,
                        action_array,
                        transition,
                        update=update,
                        global_step_start=transition_global_start,
                    )
                running_returns += transition.reward
                running_lengths += 1
                for index in np.flatnonzero(transition.reset_boundary):
                    recent_returns.append(float(running_returns[index]))
                    recent_lengths.append(float(running_lengths[index]))
                    running_returns[index] = 0.0
                    running_lengths[index] = 0
                applied_bound_count += int(
                    (
                        np.isclose(
                            transition.applied_action,
                            vector_env.action_low,
                            atol=1e-6,
                        )
                        | np.isclose(
                            transition.applied_action,
                            vector_env.action_high,
                            atol=1e-6,
                        )
                    ).sum()
                )
                applied_action_count += transition.applied_action.size
                observations = transition.observation
                global_step += config.num_envs

            state_batch = torch.stack(rollout_states)
            transition_state_batch = torch.stack(rollout_transition_states)
            action_batch = torch.stack(rollout_actions)
            old_log_prob_batch = torch.stack(rollout_old_log_probs)
            reward_batch = torch.stack(rollout_rewards)
            discount_batch = torch.stack(rollout_discounts)
            boundary_batch = torch.stack(rollout_boundaries).bool()
            normalized_value_batch = torch.stack(rollout_values)
            normalized_bootstrap_value_batch = torch.stack(
                rollout_bootstrap_values
            )
            with torch.no_grad():
                if actor_type == "PPO":
                    (
                        learning_state_batch,
                        _learning_transition_state_batch,
                        normalized_value_batch,
                        normalized_bootstrap_value_batch,
                    ) = prepare_acme_ppo_learner_values(
                        value_network,
                        observation_normalizer,
                        state_batch,
                        transition_state_batch,
                        normalize_observation=config.normalize_observation,
                    )
                else:
                    learning_state_batch = state_batch

                if config.normalize_advantage or config.normalize_value:
                    return_normalizer.begin_update()
                if config.normalize_value:
                    # Acme estimates value moments over the complete T+1
                    # sequence, including the final bootstrap value.
                    return_normalizer.update_value(
                        torch.cat(
                            (
                                normalized_value_batch,
                                normalized_bootstrap_value_batch[-1:],
                            ),
                            dim=0,
                        )
                    )
                    value_batch = return_normalizer.denormalize_value(
                        normalized_value_batch
                    )
                    bootstrap_value_batch = return_normalizer.denormalize_value(
                        normalized_bootstrap_value_batch
                    )
                else:
                    value_batch = normalized_value_batch
                    bootstrap_value_batch = normalized_bootstrap_value_batch
                if config.max_abs_reward is not None:
                    reward_batch = reward_batch.clamp(
                        min=-config.max_abs_reward,
                        max=config.max_abs_reward,
                    )
                advantages, returns = compute_gae(
                    reward_batch,
                    value_batch,
                    bootstrap_value_batch,
                    discount_batch,
                    boundary_batch,
                    gamma=config.discount,
                    gae_lambda=config.gae_lambda,
                )
                if config.normalize_advantage:
                    return_normalizer.update_advantage(advantages)
                    advantages = return_normalizer.normalize_advantage(
                        advantages
                    )
                value_targets = (
                    return_normalizer.normalize_value(returns)
                    if config.normalize_value
                    else returns
                )

            flat_state = learning_state_batch.flatten(0, 1)
            flat_action = action_batch.flatten(0, 1)
            flat_old_log_prob = old_log_prob_batch.flatten()
            flat_advantage = advantages.flatten()
            flat_value_target = value_targets.flatten()
            flat_behavior_value = normalized_value_batch.flatten()
            flat_mpve_value_observations: torch.Tensor | None = None
            flat_mpve_targets: torch.Tensor | None = None
            if actor_type == "AC-MPC-MPVE":
                if (
                    len(rollout_mpve_value_observations) != config.rollout_steps
                    or len(rollout_mpve_terminal_values)
                    != config.rollout_steps
                ):
                    raise RuntimeError("MPVE rollout predictions are incomplete")
                flat_mpve_value_observations = torch.stack(
                    rollout_mpve_value_observations
                ).flatten(0, 1)
                mpve_rewards = torch.stack(rollout_mpve_rewards).flatten(0, 1)
                mpve_terminal = torch.stack(
                    rollout_mpve_terminal_values
                ).flatten(0, 1)
                if config.normalize_value:
                    mpve_terminal = return_normalizer.denormalize_value(
                        mpve_terminal
                    )
                raw_mpve_targets = compute_mpve_td_k_targets(
                    mpve_rewards,
                    mpve_terminal,
                    gamma=config.discount,
                )
                flat_mpve_targets = (
                    return_normalizer.normalize_value(raw_mpve_targets)
                    if config.normalize_value
                    else raw_mpve_targets
                )

            policy_losses: list[float] = []
            value_losses: list[float] = []
            mpve_value_losses: list[float] = []
            entropies: list[float] = []
            approximate_kls: list[float] = []
            clip_fractions: list[float] = []
            early_stopped = False
            for _epoch in range(config.update_epochs):
                order = torch.randperm(config.batch_size, device=device)
                for start in range(0, config.batch_size, config.minibatch_size):
                    index = order[start : start + config.minibatch_size]
                    if actor_type == "PPO":
                        if not isinstance(actor, StandardPPOActor):
                            raise AssertionError("PPO actor has the wrong architecture")
                        location, policy_scale = actor.distribution_parameters(
                            flat_state[index]
                        )
                        predicted_value = value_network(flat_state[index])
                        new_log_prob = tanh_normal_log_prob(
                            location,
                            policy_scale,
                            flat_action[index],
                            action_limit=actor_config.action_limit,
                        )
                        entropy = tanh_normal_entropy(
                            location,
                            policy_scale,
                            action_limit=actor_config.action_limit,
                        ).mean()
                    else:
                        mean, predicted_value = _actor_value(
                            actor_type,
                            actor,
                            value_network,
                            flat_state[index],
                            koopman,
                            center,
                            scale,
                        )
                        if log_std is None:
                            raise AssertionError("Structured log_std is missing")
                        distribution = Normal(
                            mean, log_std.exp().expand_as(mean)
                        )
                        new_log_prob = distribution.log_prob(
                            flat_action[index]
                        ).sum(-1)
                        entropy = distribution.entropy().sum(-1).mean()
                    log_ratio = new_log_prob - flat_old_log_prob[index]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    if (
                        config.target_kl is not None
                        and float(approximate_kl) > config.target_kl
                    ):
                        early_stopped = True
                        break
                    advantage = flat_advantage[index]
                    policy_loss = -torch.minimum(
                        ratio * advantage,
                        torch.clamp(
                            ratio,
                            1.0 - config.clip_ratio,
                            1.0 + config.clip_ratio,
                        )
                        * advantage,
                    ).mean()
                    unclipped_value_loss = (
                        predicted_value - flat_value_target[index]
                    ).square()
                    if config.value_clip:
                        clipped_value = flat_behavior_value[index] + (
                            predicted_value - flat_behavior_value[index]
                        ).clamp(
                            min=-config.value_clipping_epsilon,
                            max=config.value_clipping_epsilon,
                        )
                        clipped_value_loss = (
                            clipped_value - flat_value_target[index]
                        ).square()
                        value_loss = torch.maximum(
                            unclipped_value_loss, clipped_value_loss
                        ).mean()
                    else:
                        value_loss = unclipped_value_loss.mean()
                    extra_mpve_loss = predicted_value.new_zeros(())
                    if actor_type == "AC-MPC-MPVE":
                        if (
                            flat_mpve_value_observations is None
                            or flat_mpve_targets is None
                        ):
                            raise AssertionError("MPVE minibatch targets are missing")
                        extra_mpve_loss = mpve_value_loss(
                            value_network,
                            flat_mpve_value_observations[index],
                            flat_mpve_targets[index],
                        )
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        + config.mpve_value_loss_coefficient * extra_mpve_loss
                        - config.entropy_coefficient * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = clip_ppo_gradients(
                        policy_trainable,
                        critic_trainable,
                        max_grad_norm=config.max_grad_norm,
                        separate_policy_and_critic=actor_type
                        in ("KMPC", "AC-MPC-MPVE"),
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("PPO produced a non-finite gradient")
                    optimizer.step()
                    policy_losses.append(float(policy_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                    if actor_type == "AC-MPC-MPVE":
                        mpve_value_losses.append(float(extra_mpve_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    approximate_kls.append(float(approximate_kl.detach()))
                    clip_fractions.append(
                        float(
                            ((ratio - 1.0).abs() > config.clip_ratio)
                            .float()
                            .mean()
                        )
                    )
                if early_stopped:
                    break

            completed_update = update
            recent_return = _optional_mean(recent_returns)
            last_report = {
                "update": update,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "recent_episode_return": recent_return,
                "online_stochastic_last_100_episode_return_mean": recent_return,
                "online_stochastic_last_100_episode_count": len(recent_returns),
                "recent_episode_length": _optional_mean(recent_lengths),
                "policy_loss": (
                    float(np.mean(policy_losses)) if policy_losses else None
                ),
                "value_loss": (
                    float(np.mean(value_losses)) if value_losses else None
                ),
                "mpve_value_loss": (
                    float(np.mean(mpve_value_losses))
                    if mpve_value_losses
                    else None
                ),
                "mpve_predicted_reward_mean": (
                    float(torch.stack(rollout_mpve_rewards).mean().cpu())
                    if rollout_mpve_rewards
                    else None
                ),
                "entropy": float(np.mean(entropies)) if entropies else None,
                "approximate_kl": (
                    float(np.mean(approximate_kls)) if approximate_kls else None
                ),
                "clip_fraction": (
                    float(np.mean(clip_fractions)) if clip_fractions else None
                ),
                "early_stopped_kl": early_stopped,
                "log_std": (
                    None if log_std is None else log_std.detach().cpu().tolist()
                ),
                "state_dependent_policy_scale_mean": (
                    float(torch.stack(rollout_ppo_scales).mean().cpu())
                    if rollout_ppo_scales
                    else None
                ),
                "observation_normalizer_count": observation_normalizer.count,
                "advantage_scale": (
                    return_normalizer.advantage_scale
                    if config.normalize_advantage
                    else None
                ),
                "value_mean": (
                    return_normalizer.value_mean
                    if config.normalize_value
                    else None
                ),
                "value_std": (
                    return_normalizer.value_std
                    if config.normalize_value
                    else None
                ),
                "applied_action_bound_fraction": (
                    applied_bound_count / max(applied_action_count, 1)
                ),
                "elapsed_seconds": time.perf_counter() - started,
                "collection_budget": (
                    collector.budget_report() if collector is not None else None
                ),
            }
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(last_report, sort_keys=True) + "\n")

            is_best = recent_return is not None and recent_return > best_return
            if is_best:
                best_return = float(recent_return)
                best_return_update = update
            if collector is not None:
                # Persist every complete episode at the same update boundary as
                # the policy checkpoint.  Incomplete episodes remain in memory
                # and are intentionally discarded if a resumed run resets envs.
                collector.flush(force=True)
            should_stop = (
                config.max_wall_time_seconds is not None
                and last_report["elapsed_seconds"] >= config.max_wall_time_seconds
            )
            scheduled_checkpoint = (
                update % config.checkpoint_interval_updates == 0
                or update == config.number_updates
                or should_stop
            )
            # Collection chunks are committed once per completed PPO update.
            # Keep latest.pt at the same update even when the ordinary
            # checkpoint interval is larger, so a restart never observes a
            # durable collection that is many updates ahead of optimizer/RNG.
            if collector is not None or scheduled_checkpoint:
                _atomic_torch_save(latest_path, checkpoint_payload(update))
            # Save best after latest: if a process dies between the two, the
            # resume checkpoint still agrees with any newly durable chunks.
            # Its collector state is post-flush rather than stale in-memory
            # state in either file.
            if is_best:
                _atomic_torch_save(
                    output_dir / "best.pt", checkpoint_payload(update)
                )
            for milestone in crossed_diagnostic_milestones(
                global_step - config.batch_size,
                global_step,
                diagnostic_every_steps,
            ):
                diagnostic_payload = checkpoint_payload(update)
                diagnostic_payload["diagnostic_snapshot"] = {
                    "kind": "acme_training_curve_checkpoint_v1",
                    "requested_milestone_step": int(milestone),
                    "actual_aligned_environment_step": int(global_step),
                    "alignment_batch_size": int(config.batch_size),
                    "performs_environment_evaluation": False,
                }
                _atomic_torch_save(
                    output_dir
                    / "diagnostics"
                    / f"step_{milestone:07d}.pt",
                    diagnostic_payload,
                )
            _atomic_json(
                output_dir / "status.json",
                {"state": "running", **last_report, "pid": os.getpid()},
            )
            print(json.dumps(last_report, sort_keys=True), flush=True)
            if _test_interrupt_after_update == update:
                raise RuntimeError(
                    f"test-only interruption after PPO update {update}"
                )
            if should_stop:
                break
    finally:
        # Complete episodes are committed only at a successfully completed
        # update boundary above.  Flushing here after an exception could put
        # collection chunks ahead of the authoritative optimizer checkpoint.
        vector_env.close()

    # Ensure a one-update smoke and runs ending between checkpoint intervals are
    # always resumable.
    final_checkpoint = checkpoint_payload(completed_update)
    _atomic_torch_save(latest_path, final_checkpoint)
    final = {
        "kind": "dmc_ppo_result",
        "task": task_name,
        "actor_type": actor_type,
        "update": completed_update,
        "global_step": global_step,
        "best_return": best_return if math.isfinite(best_return) else None,
        "best_return_update": best_return_update,
        "elapsed_seconds": time.perf_counter() - started,
        "last_report": last_report,
        "online_stochastic_last_100_episode_return_mean": _optional_mean(
            recent_returns
        ),
        "online_stochastic_last_100_episode_count": len(recent_returns),
        "checkpoint": str(latest_path.resolve()),
        "diagnostic_checkpoint_count": len(
            list((output_dir / "diagnostics").glob("step_*.pt"))
        ),
        "collected_transitions": (
            collector.total_transitions if collector is not None else 0
        ),
        "collection_budget": (
            collector.budget_report() if collector is not None else None
        ),
    }
    _atomic_json(output_dir / "final.json", final)
    _atomic_json(output_dir / "status.json", {"state": "complete", **final})
    return final


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_NAMES, required=True)
    parser.add_argument("--train-seed-index", type=int, required=True)
    parser.add_argument("--preflight-file", type=Path, required=True)
    parser.add_argument("--approval-file", type=Path, default=None)
    parser.add_argument("--actor", choices=ACTOR_TYPES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--koopman", type=Path, default=None)
    parser.add_argument("--collect-dir", type=Path, default=None)
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="disable PPO transition collection for actor-comparison runs",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write a non-authorizing run manifest; never construct an optimizer",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--env-workers",
        type=int,
        default=None,
        help=(
            "CPU processes for DMC stepping; execution-only and safe to "
            "change without editing YAML (default: DMC_ENV_WORKERS or 16)"
        ),
    )
    args = parser.parse_args(argv)
    if not args.dry_run and args.approval_file is None:
        parser.error("--approval-file is required unless --dry-run is explicit")
    if args.actor != "PPO" and args.koopman is None:
        parser.error(f"--koopman is required for actor {args.actor}")
    if args.no_collect and args.collect_dir is not None:
        parser.error("--no-collect and --collect-dir are mutually exclusive")
    if args.no_collect and args.actor != "PPO":
        parser.error("--no-collect is only meaningful for the plain PPO actor")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    experiment = load_experiment_config(args.config)
    config = ppo_config_from_experiment(
        experiment,
        args.profile,
        train_seed_index=args.train_seed_index,
    )
    collect_dir = args.collect_dir
    if args.actor == "PPO" and collect_dir is None and not args.no_collect:
        collect_dir = (
            Path("runs/dmc/data")
            / experiment.task
            / args.profile
            / f"seed_{config.seed}"
        )
    final = train(
        experiment.task,
        args.actor,
        args.output_dir,
        config,
        actor_config=experiment.actor_config,
        koopman_path=args.koopman,
        collect_dir=collect_dir,
        device_name=args.device,
        env_workers=args.env_workers,
        resume=not args.no_resume and not args.dry_run,
        collection_seed_index=args.train_seed_index,
        collection_seed_dir=f"seed_{config.seed}",
        experiment_config=experiment,
        profile=args.profile,
        train_seed_index=args.train_seed_index,
        approval_file=args.approval_file,
        preflight_file=args.preflight_file,
        dry_run=args.dry_run,
    )
    print(json.dumps(final, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
