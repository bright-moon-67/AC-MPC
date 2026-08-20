from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Normal

from antmaze_ac.koopman.history_model import HistoryDeepKoopman

from .critic import Critic
from .koopman_mpc_actor import KoopmanMPCActor, KoopmanMPCActorOutput


class _TruncatedNormal:
    """State-dependent truncated Normal with exact density and entropy."""

    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> None:
        if not (mean.shape == std.shape == low.shape == high.shape):
            raise ValueError("Bounded distribution tensors must share a shape")
        self.low = low
        self.high = high
        if bool((high <= low).any()):
            raise ValueError("Every bounded action interval must be non-empty")
        self.location = torch.clamp(mean, min=low, max=high)
        self.scale = std
        self.base = Normal(self.location, std)
        self.standard = Normal(
            torch.zeros_like(mean), torch.ones_like(mean)
        )
        self.alpha = (low - self.location) / std
        self.beta = (high - self.location) / std
        self.cdf_low = self.standard.cdf(self.alpha)
        self.cdf_high = self.standard.cdf(self.beta)
        self.normalizer = (self.cdf_high - self.cdf_low).clamp_min(
            torch.finfo(mean.dtype).tiny
        )
        self._epsilon = torch.finfo(mean.dtype).eps * 8

    def _sample(self) -> torch.Tensor:
        uniform = torch.rand_like(self.location)
        probability = self.cdf_low + uniform * self.normalizer
        probability = torch.clamp(
            probability,
            min=self._epsilon,
            max=1.0 - self._epsilon,
        )
        return torch.clamp(
            self.location + self.scale * self.standard.icdf(probability),
            min=self.low,
            max=self.high,
        )

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            return self._sample()

    def rsample(self) -> torch.Tensor:
        return self._sample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.base.log_prob(value) - torch.log(self.normalizer)

    def entropy(self) -> torch.Tensor:
        inverse_sqrt_two_pi = 1.0 / (2.0 * torch.pi) ** 0.5
        alpha_density = inverse_sqrt_two_pi * torch.exp(
            -0.5 * self.alpha.square()
        )
        beta_density = inverse_sqrt_two_pi * torch.exp(
            -0.5 * self.beta.square()
        )
        correction = (
            self.alpha * alpha_density - self.beta * beta_density
        ) / (2.0 * self.normalizer)
        return (
            0.5 * torch.log(2.0 * torch.pi * torch.e * self.scale.square())
            + torch.log(self.normalizer)
            + correction
        )


class HistoryMPCObservation(NamedTuple):
    physical_state: torch.Tensor
    history_context: torch.Tensor
    task_context: torch.Tensor


@dataclass
class HistoryKoopmanMPCPolicyOutput:
    distribution: Normal | _TruncatedNormal
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
    KINEMATIC_PUSH_TASK_CONTEXT_DIM = 33
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
        task_mode: str = "tracking",
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
        if task_mode not in {"tracking", "kinematic_push"}:
            raise ValueError(f"Unsupported history MPC task_mode: {task_mode}")
        if waypoint_count < 1:
            raise ValueError("waypoint_count must be positive")
        self.task_mode = str(task_mode)
        self.waypoint_count = int(waypoint_count)
        if self.task_mode == "kinematic_push":
            self.task_observation_dim = self.KINEMATIC_PUSH_TASK_CONTEXT_DIM
            self.task_context_dim = self.KINEMATIC_PUSH_TASK_CONTEXT_DIM
        else:
            self.task_observation_dim = (
                3 if self.waypoint_count == 1 else 4 * self.waypoint_count
            )
            self.task_context_dim = (
                self.TASK_CONTEXT_DIM
                if self.waypoint_count == 1
                else self.task_observation_dim
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
        if self.actor.max_delta is not None:
            self.ACTION_DISTRIBUTION = (
                "state_dependent_truncated_normalized_delta_v1"
            )
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
    ) -> tuple[
        HistoryMPCObservation,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        split = self.split_observation(observation)
        normalized_state = (
            split.physical_state - self.state_mean
        ) / self.state_std
        lifted = self.koopman.lift(normalized_state, split.history_context)
        tip_mean = self.state_mean[self.tip_indices]
        tip_std = self.state_std[self.tip_indices]
        if self.task_mode == "kinematic_push":
            # Raw layout is documented in KinematicPushTask.context.  Keep the
            # actor input at the same fixed 32-D size but put heterogeneous
            # coordinates on useful scales.  Only the active tip target enters
            # the explicit MPC reference; all remaining object/obstacle/phase
            # features condition the learned Q/p/R cost map.
            task = split.task_context
            actor_context = task.clone()
            for start in (0, 3, 9, 12):
                actor_context[..., start : start + 3] = (
                    task[..., start : start + 3] - tip_mean
                ) / tip_std
            actor_context[..., 6:9] = task[..., 6:9] / 0.5
            actor_context[..., 15:18] = task[..., 15:18] / 0.5
            actor_context[..., 18:24] = task[..., 18:24] / 0.5
            actor_context[..., 24:25] = task[..., 24:25] / 0.5
            active_target = actor_context[..., 0:3]
            action_reference = normalized_state.new_zeros(
                *normalized_state.shape[:-1],
                self.action_dim,
            )
        elif self.waypoint_count == 1:
            normalized_target = (split.task_context - tip_mean) / tip_std
            normalized_tip = normalized_state[..., self.tip_indices]
            target_error = normalized_target - normalized_tip
            actor_context = torch.cat(
                (normalized_target, target_error), dim=-1
            )
            active_target = normalized_target
            action_reference = normalized_state.new_zeros(
                *normalized_state.shape[:-1],
                self.action_dim,
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
            active_target = torch.sum(
                normalized_waypoints * stage.unsqueeze(-1),
                dim=-2,
            )
            action_reference = normalized_state.new_zeros(
                *normalized_state.shape[:-1],
                self.action_dim,
            )

        # Standard reference-tracking initialization for PPO without BC:
        # preserve the current normalized non-tip state and move only the
        # physical tip coordinates toward the active waypoint.  The learned
        # cost map remains free to reshape this reference cost during PPO.
        physical_reference = normalized_state.clone()
        physical_reference[..., self.tip_indices] = active_target
        zero_reference_indices = getattr(
            self.actor,
            "zero_physical_reference_indices",
            None,
        )
        if zero_reference_indices is not None:
            normalized_physical_zero = (
                -self.state_mean[zero_reference_indices]
                / self.state_std[zero_reference_indices]
            )
            physical_reference[..., zero_reference_indices] = (
                normalized_physical_zero
            )
        return (
            split,
            lifted,
            actor_context,
            physical_reference,
            action_reference,
        )

    def actor_mean(self, observation: torch.Tensor) -> KoopmanMPCActorOutput:
        split, lifted, actor_context, physical_reference, action_reference = (
            self.features(observation)
        )
        if getattr(self.actor, "reference_mode", "explicit") == "implicit":
            physical_reference = None
            action_reference = None
        return self.actor(
            lifted,
            actor_context,
            physical_reference,
            action_reference,
            previous_action=self.previous_action(split),
        )

    def previous_action(
        self,
        split: HistoryMPCObservation | torch.Tensor,
    ) -> torch.Tensor:
        """Extract the latest applied absolute action from history context."""

        if isinstance(split, torch.Tensor):
            split = self.split_observation(split)
        action_history_start = self.history_steps * self.state_dim
        actions = split.history_context[..., action_history_start:].reshape(
            *split.history_context.shape[:-1],
            self.history_steps,
            self.action_dim,
        )
        return actions[..., -1, :]

    def forward(
        self,
        observation: torch.Tensor,
    ) -> HistoryKoopmanMPCPolicyOutput:
        single = observation.ndim == 1
        observation_batch = observation.unsqueeze(0) if single else observation
        split, lifted, actor_context, physical_reference, action_reference = (
            self.features(observation_batch)
        )
        previous_action = self.previous_action(split)
        if getattr(self.actor, "reference_mode", "explicit") == "implicit":
            physical_reference = None
            action_reference = None
        mpc = self.actor(
            lifted,
            actor_context,
            physical_reference,
            action_reference,
            previous_action=previous_action,
        )
        if not (
            torch.isfinite(mpc.action).all()
            and torch.isfinite(mpc.quadratic_diagonal).all()
            and torch.isfinite(mpc.linear_term).all()
        ):
            raise FloatingPointError("BC-KMPC actor produced NaN or Inf")
        if self.actor.max_delta is None:
            mean_batch = mpc.action
            distribution: Normal | _TruncatedNormal = Normal(
                mean_batch,
                self.log_std.exp().expand_as(mean_batch),
            )
        else:
            if mpc.normalized_delta is None:
                raise RuntimeError("Normalized-delta MPC returned no delta action")
            mean_batch = mpc.normalized_delta
            lower, upper = self.actor.normalized_delta_bounds(previous_action)
            distribution = _TruncatedNormal(
                mean_batch,
                self.log_std.exp().expand_as(mean_batch),
                lower,
                upper,
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
