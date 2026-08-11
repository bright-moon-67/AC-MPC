from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from .critic import Critic
from .history_koopman_mpc_policy import RateLimitedSquashedNormal


def _activation(name: str) -> type[nn.Module]:
    choices = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported MLP activation {name!r}") from exc


class HistoryMLPObservation(NamedTuple):
    physical_state: torch.Tensor
    history_context: torch.Tensor
    target_tip: torch.Tensor
    previous_action: torch.Tensor


@dataclass
class HistoryMLPPolicyOutput:
    distribution: RateLimitedSquashedNormal
    mean: torch.Tensor
    value: torch.Tensor
    features: torch.Tensor
    raw_mean: torch.Tensor


class HistoryMLPActor(nn.Module):
    """History-conditioned neural baseline with exact absolute/rate bounds."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "tanh",
        *,
        action_low: float | torch.Tensor = -0.30,
        action_high: float | torch.Tensor = 0.30,
        max_delta: float | torch.Tensor = 0.001,
    ) -> None:
        super().__init__()
        if input_dim < 1 or action_dim < 1 or not hidden_dims:
            raise ValueError("MLP dimensions must be positive and non-empty")
        activation_type = _activation(activation)
        dimensions = [int(input_dim), *map(int, hidden_dims), int(action_dim)]
        layers: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layer = nn.Linear(in_dim, out_dim)
            layers.append(layer)
            if index < len(dimensions) - 2:
                layers.append(activation_type())
        self.network = nn.Sequential(*layers)
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Expected a linear final actor layer")
        nn.init.orthogonal_(final.weight, gain=0.01)
        nn.init.zeros_(final.bias)

        low = torch.broadcast_to(
            torch.as_tensor(action_low, dtype=torch.float32),
            (action_dim,),
        ).clone()
        high = torch.broadcast_to(
            torch.as_tensor(action_high, dtype=torch.float32),
            (action_dim,),
        ).clone()
        delta = torch.broadcast_to(
            torch.as_tensor(max_delta, dtype=torch.float32),
            (action_dim,),
        ).clone()
        if bool((low >= high).any()):
            raise ValueError("Every action lower bound must be below its upper bound")
        if not torch.isfinite(delta).all() or bool((delta <= 0).any()):
            raise ValueError("max_delta must be finite and positive")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("max_delta", delta)
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)

    def feasible_interval(
        self,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_action.shape[-1] != self.action_dim:
            raise ValueError("Wrong previous-action dimension")
        lower = torch.maximum(self.action_low, previous_action - self.max_delta)
        upper = torch.minimum(self.action_high, previous_action + self.max_delta)
        return lower, upper

    def forward(
        self,
        features: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected actor input dimension {self.input_dim}, "
                f"got {features.shape[-1]}"
            )
        raw_mean = self.network(features)
        lower, upper = self.feasible_interval(previous_action)
        center = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        mean = center + half_width * torch.tanh(raw_mean)
        return mean, raw_mean


class HistoryMLPPolicy(nn.Module):
    """PPO baseline for 45-D state, 18-D absolute action and explicit history.

    The actor and critic receive the complete normalized history context plus
    normalized target-tip position and tip error. No Koopman lift or learned
    dynamics are used by this policy.
    """

    TASK_CONTEXT_DIM = 6
    ACTION_DISTRIBUTION = "rate_limited_squashed_normal_v1"

    def __init__(
        self,
        actor: HistoryMLPActor,
        critic: Critic,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        *,
        state_dim: int,
        action_dim: int,
        history_steps: int,
        tip_indices: tuple[int, int, int] = (30, 31, 32),
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        if min(state_dim, action_dim, history_steps) < 1:
            raise ValueError("State, action and history dimensions must be positive")
        if actor.action_dim != action_dim:
            raise ValueError("Actor action dimension does not match policy")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.history_steps = int(history_steps)
        self.history_context_dim = self.history_steps * (
            self.state_dim + self.action_dim
        )
        self.feature_dim = self.history_context_dim + self.TASK_CONTEXT_DIM
        self.observation_dim = self.state_dim + self.history_context_dim + 3
        if actor.input_dim != self.feature_dim:
            raise ValueError("Actor input must equal history_context_dim + 6")
        if state_mean.shape != (self.state_dim,) or state_std.shape != (
            self.state_dim,
        ):
            raise ValueError("State normalizer shape does not match policy")
        first_linear = next(
            (layer for layer in critic.network if isinstance(layer, nn.Linear)),
            None,
        )
        if first_linear is None or first_linear.in_features != self.feature_dim:
            raise ValueError("Critic input must equal history_context_dim + 6")

        tip_index_tensor = torch.as_tensor(tip_indices, dtype=torch.long)
        if tip_index_tensor.shape != (3,):
            raise ValueError("tip_indices must contain exactly three indices")
        if bool((tip_index_tensor < 0).any()) or bool(
            (tip_index_tensor >= self.state_dim).any()
        ):
            raise ValueError("tip_indices are outside the physical state")

        self.actor = actor
        self.critic = critic
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(log_std_init))
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer(
            "state_std",
            state_std.detach().clone().clamp_min(1e-6),
        )
        self.register_buffer("tip_indices", tip_index_tensor)

    def split_observation(
        self,
        observation: torch.Tensor,
    ) -> HistoryMLPObservation:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"got {observation.shape[-1]}"
            )
        state_stop = self.state_dim
        context_stop = state_stop + self.history_context_dim
        physical_state = observation[..., :state_stop]
        history_context = observation[..., state_stop:context_stop]
        target_tip = observation[..., context_stop:]
        previous_action = history_context[..., -self.action_dim :]
        return HistoryMLPObservation(
            physical_state,
            history_context,
            target_tip,
            previous_action,
        )

    def policy_features(
        self,
        observation: torch.Tensor,
    ) -> tuple[HistoryMLPObservation, torch.Tensor]:
        split = self.split_observation(observation)
        normalized_state = (
            split.physical_state - self.state_mean
        ) / self.state_std
        tip_mean = self.state_mean[self.tip_indices]
        tip_std = self.state_std[self.tip_indices]
        normalized_target = (split.target_tip - tip_mean) / tip_std
        normalized_tip = normalized_state[..., self.tip_indices]
        target_error = normalized_target - normalized_tip
        task_context = torch.cat((normalized_target, target_error), dim=-1)
        features = torch.cat((split.history_context, task_context), dim=-1)
        return split, features

    def actor_mean(self, observation: torch.Tensor) -> torch.Tensor:
        split, features = self.policy_features(observation)
        mean, _ = self.actor(features, split.previous_action)
        return mean

    def forward(self, observation: torch.Tensor) -> HistoryMLPPolicyOutput:
        single = observation.ndim == 1
        observation_batch = observation.unsqueeze(0) if single else observation
        split, features = self.policy_features(observation_batch)
        mean_batch, raw_mean_batch = self.actor(
            features,
            split.previous_action,
        )
        if not torch.isfinite(mean_batch).all():
            raise FloatingPointError("History-MLP actor produced NaN or Inf")
        distribution = RateLimitedSquashedNormal(
            mean_batch,
            split.previous_action,
            self.actor.action_low,
            self.actor.action_high,
            self.actor.max_delta,
            self.log_std,
        )
        value_batch = self.critic(features)
        return HistoryMLPPolicyOutput(
            distribution=distribution,
            mean=mean_batch[0] if single else mean_batch,
            value=value_batch[0] if single else value_batch,
            features=features[0] if single else features,
            raw_mean=raw_mean_batch[0] if single else raw_mean_batch,
        )

    def act(
        self,
        observation: torch.Tensor,
        deterministic: bool = False,
        return_output: bool = False,
    ):
        output = self(observation)
        action = output.mean if deterministic else output.distribution.sample()
        if observation.ndim == 1 and action.ndim == 2:
            action = action[0]
        distribution_action = action.unsqueeze(0) if action.ndim == 1 else action
        log_prob = output.distribution.log_prob(distribution_action)
        if observation.ndim == 1:
            log_prob = log_prob[0]
        result = (action, log_prob, output.value)
        return (*result, output) if return_output else result

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        output = self(observations)
        return (
            output.distribution.log_prob(actions),
            output.distribution.entropy(),
            output.value,
            output,
        )
