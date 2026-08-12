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
    task_context: torch.Tensor


@dataclass
class HistoryKoopmanMPCPolicyOutput:
    distribution: Normal
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


class HistoryKoopmanMPCPolicy(nn.Module):
    """Actor-critic policy for history-context, absolute-action BC-KMPC.

    The environment observation contains all history needed by
    :class:`HistoryDeepKoopman`, so shuffled PPO minibatches reconstruct the
    same policy without hidden controller state.  The actor receives the
    frozen Koopman lift plus task context.  The legacy single-target context is
    ``[normalized_target, normalized_target-current_tip]``.  Ordered waypoint
    tracking uses ``[normalized_G1,G2,G3,one_hot(active_stage)]``.
    """

    TASK_CONTEXT_DIM = 6
    ACTION_DISTRIBUTION = "diagonal_normal_v1"

    def __init__(
        self,
        koopman: HistoryDeepKoopman,
        actor: KoopmanMPCActor,
        critic: Critic,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        *,
        waypoint_count: int = 1,
        tip_indices: tuple[int, int, int] = (30, 31, 32),
        log_std_init: float = -3.0,
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
        if waypoint_count < 1:
            raise ValueError("waypoint_count must be positive")
        self.waypoint_count = int(waypoint_count)
        self.task_observation_dim = (
            3 if self.waypoint_count == 1 else 4 * self.waypoint_count
        )
        self.task_context_dim = (
            self.TASK_CONTEXT_DIM
            if self.waypoint_count == 1
            else 4 * self.waypoint_count
        )
        if actor.context_dim != self.task_context_dim:
            raise ValueError(
                f"Actor context_dim must be {self.task_context_dim}"
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
        self.observation_dim = (
            self.state_dim
            + self.history_context_dim
            + self.task_observation_dim
        )
        expected_critic_input = koopman.lifted_dim + self.task_context_dim
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
        task_context = observation[..., context_stop:]
        return HistoryMPCObservation(
            physical_state,
            history_context,
            task_context,
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
        if self.waypoint_count == 1:
            normalized_target = (split.task_context - tip_mean) / tip_std
            normalized_tip = normalized_state[..., self.tip_indices]
            target_error = normalized_target - normalized_tip
            actor_context = torch.cat(
                (normalized_target, target_error), dim=-1
            )
        else:
            waypoint_stop = 3 * self.waypoint_count
            waypoints = split.task_context[..., :waypoint_stop].reshape(
                *split.task_context.shape[:-1], self.waypoint_count, 3
            )
            stage = split.task_context[..., waypoint_stop:]
            normalized_waypoints = (waypoints - tip_mean) / tip_std
            actor_context = torch.cat(
                (normalized_waypoints.flatten(start_dim=-2), stage), dim=-1
            )
        return split, lifted, actor_context

    def actor_mean(self, observation: torch.Tensor) -> KoopmanMPCActorOutput:
        _, lifted, actor_context = self.features(observation)
        return self.actor(lifted, actor_context)

    def forward(
        self,
        observation: torch.Tensor,
    ) -> HistoryKoopmanMPCPolicyOutput:
        single = observation.ndim == 1
        observation_batch = observation.unsqueeze(0) if single else observation
        _, lifted, actor_context = self.features(observation_batch)
        mpc = self.actor(lifted, actor_context)
        if not (
            torch.isfinite(mpc.action).all()
            and torch.isfinite(mpc.quadratic_diagonal).all()
            and torch.isfinite(mpc.linear_term).all()
        ):
            raise FloatingPointError("BC-KMPC actor produced NaN or Inf")
        mean_batch = mpc.action
        distribution = Normal(
            mean_batch,
            self.log_std.exp().expand_as(mean_batch),
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
        log_prob = output.distribution.log_prob(distribution_action).sum(dim=-1)
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
            output.distribution.log_prob(actions).sum(dim=-1),
            output.distribution.entropy().sum(dim=-1),
            output.value,
            output,
        )
