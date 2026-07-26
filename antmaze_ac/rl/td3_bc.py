from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from antmaze_ac.data.build_sequences import AugmentedDataset

from .ac_koopman_policy import KoopmanLQRPolicy


def _activation(name: str) -> type[nn.Module]:
    choices = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported action-value activation {name!r}") from exc


def _q_network(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: str,
) -> nn.Sequential:
    activation_type = _activation(activation)
    dimensions = [input_dim, *map(int, hidden_dims), 1]
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dimensions) - 2:
            layers.append(activation_type())
    return nn.Sequential(*layers)


class TwinActionValueCritic(nn.Module):
    """Twin TD3 critics for Q_value(state, delta_action).

    These action-value functions are intentionally named separately from the
    CostActor's local quadratic stage Hessian.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        *,
        hidden_dims: Sequence[int] = (512, 512),
        activation: str = "relu",
        action_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if state_mean.shape != (state_dim,) or state_std.shape != (state_dim,):
            raise ValueError("State normalization shape does not match critic")
        if action_scale <= 0:
            raise ValueError("action_scale must be positive")
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone().clamp_min(1e-6))
        self.action_scale = float(action_scale)
        input_dim = int(state_dim) + int(action_dim)
        self.q_value_1 = _q_network(input_dim, hidden_dims, activation)
        self.q_value_2 = _q_network(input_dim, hidden_dims, activation)

    def _input(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.shape[:-1] != action.shape[:-1]:
            raise ValueError("State and action batch shapes must match")
        normalized_state = (state - self.state_mean) / self.state_std
        normalized_action = action / self.action_scale
        return torch.cat((normalized_state, normalized_action), dim=-1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self._input(state, action)
        return (
            self.q_value_1(inputs).squeeze(-1),
            self.q_value_2(inputs).squeeze(-1),
        )

    def first(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q_value_1(self._input(state, action)).squeeze(-1)


@dataclass
class OfflineTransitionBatch:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


def sample_transition_batch(
    dataset: AugmentedDataset,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> OfflineTransitionBatch:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = rng.integers(0, len(dataset), size=batch_size)

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(values[indices]),
            dtype=torch.float32,
            device=device,
        )

    return OfflineTransitionBatch(
        state=tensor(dataset.state),
        action=tensor(dataset.action),
        next_state=tensor(dataset.next_state),
        reward=tensor(dataset.reward),
        done=tensor(dataset.done),
    )


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0,1]")
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(),
            source.parameters(),
            strict=True,
        ):
            target_parameter.lerp_(source_parameter, tau)
        for target_buffer, source_buffer in zip(
            target.buffers(),
            source.buffers(),
            strict=True,
        ):
            if torch.is_floating_point(target_buffer):
                target_buffer.lerp_(source_buffer, tau)
            else:
                target_buffer.copy_(source_buffer)


class TD3BCTrainer:
    """One-step updater for deterministic Koopman-LQR TD3+BC."""

    def __init__(
        self,
        policy: KoopmanLQRPolicy,
        target_policy: KoopmanLQRPolicy,
        critic: TwinActionValueCritic,
        target_critic: TwinActionValueCritic,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        *,
        discount: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_frequency: int,
        alpha: float,
        bc_weight: float,
        bc_warmup_steps: int,
        max_delta_action: float,
        reward_scale: float,
        reward_bias: float,
        max_grad_norm: float,
    ) -> None:
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0,1]")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0,1]")
        if policy_noise < 0 or noise_clip < 0:
            raise ValueError("TD3 target noise values must be non-negative")
        if policy_frequency < 1:
            raise ValueError("policy_frequency must be positive")
        if alpha < 0 or bc_weight < 0:
            raise ValueError("TD3+BC loss weights must be non-negative")
        if bc_warmup_steps < 0:
            raise ValueError("bc_warmup_steps must be non-negative")
        if max_delta_action <= 0 or max_grad_norm <= 0:
            raise ValueError("Action limit and gradient norm must be positive")
        self.policy = policy
        self.target_policy = target_policy
        self.critic = critic
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.discount = float(discount)
        self.tau = float(tau)
        self.policy_noise = float(policy_noise)
        self.noise_clip = float(noise_clip)
        self.policy_frequency = int(policy_frequency)
        self.alpha = float(alpha)
        self.bc_weight = float(bc_weight)
        self.bc_warmup_steps = int(bc_warmup_steps)
        self.max_delta_action = float(max_delta_action)
        self.reward_scale = float(reward_scale)
        self.reward_bias = float(reward_bias)
        self.max_grad_norm = float(max_grad_norm)

    def update(
        self,
        batch: OfflineTransitionBatch,
        gradient_step: int,
    ) -> dict[str, torch.Tensor | int | None]:
        self.policy.actor.train()
        self.critic.train()
        with torch.no_grad():
            transformed_reward = (
                self.reward_scale * batch.reward + self.reward_bias
            )
            bootstrap = torch.zeros_like(transformed_reward)
            continuing = batch.done < 0.5
            # Legacy D4RL HDF5 defines next observations by shifting rows.
            # Boundary next states can therefore belong to the next episode.
            # Do not even evaluate the target policy on those masked rows.
            if bool(torch.any(continuing)):
                continuing_next_state = batch.next_state[continuing]
                target_output = self.target_policy(continuing_next_state)
                noise = (
                    torch.randn_like(target_output.mean) * self.policy_noise
                )
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                target_action = (target_output.mean + noise).clamp(
                    -self.max_delta_action,
                    self.max_delta_action,
                )
                target_q_1, target_q_2 = self.target_critic(
                    continuing_next_state,
                    target_action,
                )
                bootstrap[continuing] = torch.minimum(
                    target_q_1,
                    target_q_2,
                )
            q_target = transformed_reward + self.discount * bootstrap

        current_q_1, current_q_2 = self.critic(batch.state, batch.action)
        critic_loss = (
            (current_q_1 - q_target).square().mean()
            + (current_q_2 - q_target).square().mean()
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.max_grad_norm,
        )
        if not torch.isfinite(critic_grad_norm):
            raise FloatingPointError("TD3+BC critic gradient is NaN or Inf")
        self.critic_optimizer.step()

        # Keep per-step scalars on the GPU. The training script reduces them
        # once per log interval, avoiding repeated device synchronization.
        metrics: dict[str, torch.Tensor | int | None] = {
            "critic_loss": critic_loss.detach(),
            "critic_grad_norm": critic_grad_norm.detach(),
            "q_value_1_mean": current_q_1.detach().mean(),
            "q_value_2_mean": current_q_2.detach().mean(),
            "q_target_mean": q_target.detach().mean(),
            "dataset_reward_mean": batch.reward.mean().detach(),
            "dataset_success_fraction": (
                (batch.reward > 0).float().mean().detach()
            ),
            "actor_updated": 0,
            "actor_loss": None,
            "actor_q_loss": None,
            "behavior_cloning_loss": None,
            "actor_grad_norm": None,
            "td3_bc_lambda": None,
            "dare_retry_fraction": None,
            "dare_fallback_fraction": None,
        }

        if gradient_step % self.policy_frequency:
            return metrics

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        actor_output = self.policy(batch.state)
        policy_action = actor_output.mean
        behavior_cloning_loss = (
            policy_action - batch.action
        ).square().mean()
        action_value = self.critic.first(batch.state, policy_action)
        if gradient_step <= self.bc_warmup_steps:
            td3_bc_lambda = torch.zeros((), device=action_value.device)
        else:
            td3_bc_lambda = self.alpha / (
                action_value.detach().abs().mean() + 1e-6
            )
        actor_q_loss = -td3_bc_lambda * action_value.mean()
        actor_loss = actor_q_loss + self.bc_weight * behavior_cloning_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.actor.parameters(),
            self.max_grad_norm,
        )
        if not torch.isfinite(actor_grad_norm):
            raise FloatingPointError("TD3+BC actor gradient is NaN or Inf")
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        soft_update(self.target_policy.actor, self.policy.actor, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
        metrics.update(
            {
                "actor_updated": 1,
                "actor_loss": actor_loss.detach(),
                "actor_q_loss": actor_q_loss.detach(),
                "behavior_cloning_loss": behavior_cloning_loss.detach(),
                "actor_grad_norm": actor_grad_norm.detach(),
                "td3_bc_lambda": td3_bc_lambda.detach(),
                "dare_retry_fraction": (
                    actor_output.solver_retry_used.float().mean().detach()
                ),
                "dare_fallback_fraction": (
                    actor_output.solver_fallback_used.float().mean().detach()
                ),
            }
        )
        return metrics


@torch.no_grad()
def offline_validation_metrics(
    policy: KoopmanLQRPolicy,
    critic: TwinActionValueCritic,
    batch: OfflineTransitionBatch,
) -> dict[str, float]:
    policy.eval()
    critic.eval()
    # Training may defer the O(batch * n^3) eigendecomposition. Validation is
    # deliberately infrequent and restores the full closed-loop stability
    # check, then the context manager restores the hot-path setting.
    with policy.full_dare_diagnostics():
        output = policy(batch.state)
    q_value_1, q_value_2 = critic(batch.state, output.mean)
    return {
        "behavior_cloning_loss": float(
            (output.mean - batch.action).square().mean()
        ),
        "policy_delta_abs_mean": float(output.mean.abs().mean()),
        "dataset_delta_abs_mean": float(batch.action.abs().mean()),
        "q_value_min_mean": float(torch.minimum(q_value_1, q_value_2).mean()),
        "dare_retry_fraction": float(
            output.solver_retry_used.float().mean()
        ),
        "dare_fallback_fraction": float(
            output.solver_fallback_used.float().mean()
        ),
        "dare_relative_residual_max": float(
            output.lqr.dare.relative_residual.max()
        ),
        "closed_loop_spectral_radius_max": float(
            output.lqr.dare.closed_loop_spectral_radius.max()
        ),
    }
