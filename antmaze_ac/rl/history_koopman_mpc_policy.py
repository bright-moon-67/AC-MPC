from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Normal

from antmaze_ac.koopman.history_model import HistoryDeepKoopman

from .critic import Critic
from .koopman_mpc_actor import KoopmanMPCActor, KoopmanMPCActorOutput


class HistoryMPCObservation(NamedTuple):
    physical_state: torch.Tensor
    history_context: torch.Tensor
    target_tip: torch.Tensor
    previous_action: torch.Tensor


@dataclass
class HistoryKoopmanMPCPolicyOutput:
    distribution: "RateLimitedSquashedNormal"
    mean: torch.Tensor
    value: torch.Tensor
    lifted_state: torch.Tensor
    actor_context: torch.Tensor
    mpc: KoopmanMPCActorOutput

    @property
    def stage_hessian_diag(self) -> torch.Tensor:
        """Compatibility alias used by existing actor diagnostics."""

        return self.mpc.quadratic_diagonal

    @property
    def stage_linear(self) -> torch.Tensor:
        """Compatibility alias used by existing actor diagnostics."""

        return self.mpc.linear_term


class RateLimitedSquashedNormal:
    """Gaussian policy transformed into the state-dependent action-rate box.

    Sampling an unconstrained Gaussian and clipping it in the environment
    changes the executed action without changing the stored PPO log-probability.
    This distribution instead maps a latent Gaussian through tanh into the exact
    interval allowed by both the absolute and rate constraints, so rollout and
    update probabilities describe the action that is actually executed.
    """

    def __init__(
        self,
        median: torch.Tensor,
        previous_action: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        max_delta: torch.Tensor,
        log_std: torch.Tensor,
        boundary_margin: float = 0.02,
        inverse_epsilon: float = 1e-6,
    ) -> None:
        lower = torch.maximum(action_low, previous_action - max_delta)
        upper = torch.minimum(action_high, previous_action + max_delta)
        self.center = 0.5 * (lower + upper)
        self.half_width = (0.5 * (upper - lower)).clamp_min(inverse_epsilon)
        normalized_median = ((median - self.center) / self.half_width).clamp(
            -1.0 + boundary_margin,
            1.0 - boundary_margin,
        )
        self.location = torch.atanh(normalized_median)
        self.scale = log_std.exp().expand_as(median)
        self.base = Normal(self.location, self.scale)
        self.inverse_epsilon = float(inverse_epsilon)

    def _transform(self, latent: torch.Tensor) -> torch.Tensor:
        return self.center + self.half_width * torch.tanh(latent)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._transform(self.base.sample(sample_shape))

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self._transform(self.base.rsample(sample_shape))

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        normalized = ((action - self.center) / self.half_width).clamp(
            -1.0 + self.inverse_epsilon,
            1.0 - self.inverse_epsilon,
        )
        latent = torch.atanh(normalized)
        log_jacobian = torch.log(self.half_width) + torch.log1p(
            -normalized.square() + self.inverse_epsilon
        )
        return (self.base.log_prob(latent) - log_jacobian).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        # One-sample reparameterized estimate is sufficient for the PPO entropy
        # diagnostic and remains valid if entropy regularization is enabled.
        action = self.rsample()
        return -self.log_prob(action)


class HistoryKoopmanMPCPolicy(nn.Module):
    """Actor-critic policy for history-context, absolute-action BC-KMPC.

    The environment observation contains all history needed by
    :class:`HistoryDeepKoopman`, so shuffled PPO minibatches reconstruct the
    same policy without hidden controller state.  The actor receives the
    frozen Koopman lift plus a six-dimensional task context consisting of the
    normalized target tip and normalized target-minus-current tip error.
    """

    TASK_CONTEXT_DIM = 6
    ACTION_DISTRIBUTION = "rate_limited_squashed_normal_v1"

    def __init__(
        self,
        koopman: HistoryDeepKoopman,
        actor: KoopmanMPCActor,
        critic: Critic,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        *,
        tip_indices: tuple[int, int, int] = (30, 31, 32),
        log_std_init: float = -7.0,
    ) -> None:
        super().__init__()
        if not isinstance(koopman, HistoryDeepKoopman):
            raise TypeError("HistoryKoopmanMPCPolicy requires HistoryDeepKoopman")
        if actor.lifted_dim != koopman.lifted_dim:
            raise ValueError("Actor and Koopman lifted dimensions do not match")
        if actor.physical_dim != koopman.state_dim:
            raise ValueError("Actor physical dimension does not match Koopman state")
        if actor.action_dim != koopman.action_dim:
            raise ValueError("Actor action dimension does not match Koopman action")
        if actor.context_dim != self.TASK_CONTEXT_DIM:
            raise ValueError(
                f"Actor context_dim must be {self.TASK_CONTEXT_DIM}"
            )
        if state_mean.shape != (koopman.state_dim,) or state_std.shape != (
            koopman.state_dim,
        ):
            raise ValueError("State normalizer shape does not match Koopman state")
        tip_index_tensor = torch.as_tensor(tip_indices, dtype=torch.long)
        if tip_index_tensor.shape != (3,):
            raise ValueError("tip_indices must contain exactly three indices")
        if bool((tip_index_tensor < 0).any()) or bool(
            (tip_index_tensor >= koopman.state_dim).any()
        ):
            raise ValueError("tip_indices are outside the physical state")

        self.koopman = koopman.freeze_dynamics()
        self.actor = actor
        self.critic = critic
        self.log_std = nn.Parameter(
            torch.full((koopman.action_dim,), float(log_std_init))
        )
        self.register_buffer("state_mean", state_mean.detach().clone())
        self.register_buffer(
            "state_std",
            state_std.detach().clone().clamp_min(1e-6),
        )
        self.register_buffer("tip_indices", tip_index_tensor)

        self.state_dim = int(koopman.state_dim)
        self.action_dim = int(koopman.action_dim)
        self.history_steps = int(koopman.history_steps)
        self.history_context_dim = int(koopman.context_dim)
        self.observation_dim = self.state_dim + self.history_context_dim + 3
        expected_critic_input = koopman.lifted_dim + self.TASK_CONTEXT_DIM
        first_linear = next(
            (layer for layer in critic.network if isinstance(layer, nn.Linear)),
            None,
        )
        if first_linear is None or first_linear.in_features != expected_critic_input:
            raise ValueError(
                "Critic input dimension must equal lifted_dim + task_context_dim"
            )

    def split_observation(
        self,
        observation: torch.Tensor,
    ) -> HistoryMPCObservation:
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
        return HistoryMPCObservation(
            physical_state,
            history_context,
            target_tip,
            previous_action,
        )

    def features(
        self,
        observation: torch.Tensor,
    ) -> tuple[HistoryMPCObservation, torch.Tensor, torch.Tensor]:
        split = self.split_observation(observation)
        normalized_state = (
            split.physical_state - self.state_mean
        ) / self.state_std
        lifted = self.koopman.lift(normalized_state, split.history_context)
        tip_mean = self.state_mean[self.tip_indices]
        tip_std = self.state_std[self.tip_indices]
        normalized_target = (split.target_tip - tip_mean) / tip_std
        normalized_tip = normalized_state[..., self.tip_indices]
        target_error = normalized_target - normalized_tip
        actor_context = torch.cat((normalized_target, target_error), dim=-1)
        return split, lifted, actor_context

    def actor_mean(self, observation: torch.Tensor) -> KoopmanMPCActorOutput:
        split, lifted, actor_context = self.features(observation)
        return self.actor(
            lifted,
            split.previous_action,
            actor_context,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> HistoryKoopmanMPCPolicyOutput:
        single = observation.ndim == 1
        observation_batch = observation.unsqueeze(0) if single else observation
        split, lifted, actor_context = self.features(observation_batch)
        mpc = self.actor(lifted, split.previous_action, actor_context)
        if not (
            torch.isfinite(mpc.action).all()
            and torch.isfinite(mpc.quadratic_diagonal).all()
            and torch.isfinite(mpc.linear_term).all()
        ):
            raise FloatingPointError("BC-KMPC actor produced NaN or Inf")
        mean_batch = mpc.action
        distribution = RateLimitedSquashedNormal(
            mean_batch,
            split.previous_action,
            self.actor.action_low,
            self.actor.action_high,
            self.actor.max_delta,
            self.log_std,
        )
        value_batch = self.critic(torch.cat((lifted, actor_context), dim=-1))
        mean = mean_batch[0] if single else mean_batch
        value = value_batch[0] if single else value_batch
        lifted_output = lifted[0] if single else lifted
        actor_context_output = actor_context[0] if single else actor_context
        return HistoryKoopmanMPCPolicyOutput(
            distribution=distribution,
            mean=mean,
            value=value,
            lifted_state=lifted_output,
            actor_context=actor_context_output,
            mpc=mpc,
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
