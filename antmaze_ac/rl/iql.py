"""Implicit Q-Learning components for the history Koopman-MPC policy.

The update equations follow rlkit's PyTorch IQL trainer while keeping this
project's differentiable KMPC actor and frozen history-Koopman features.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .history_koopman_mpc_policy import HistoryKoopmanMPCPolicy


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    activation_types = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        activation_type = activation_types[activation.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported IQL activation {activation!r}") from error
    dimensions = [int(input_dim), *map(int, hidden_dims), int(output_dim)]
    if min(dimensions) < 1 or len(hidden_dims) < 1:
        raise ValueError("IQL network dimensions must be positive and non-empty")
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dimensions) - 2:
            layers.append(activation_type())
    return nn.Sequential(*layers)


class IQLActionValue(nn.Module):
    """Scalar Q(s,a) over frozen Koopman/task features and policy actions."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        activation: str = "relu",
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if action_scale <= 0:
            raise ValueError("action_scale must be positive")
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.action_scale = float(action_scale)
        self.network = _mlp(
            self.feature_dim + self.action_dim,
            hidden_dims,
            1,
            activation,
        )

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if features.shape[:-1] != actions.shape[:-1]:
            raise ValueError("Feature and action batch shapes must match")
        if features.shape[-1] != self.feature_dim:
            raise ValueError("IQL feature dimension does not match Q network")
        if actions.shape[-1] != self.action_dim:
            raise ValueError("IQL action dimension does not match Q network")
        inputs = torch.cat((features, actions / self.action_scale), dim=-1)
        return self.network(inputs).squeeze(-1)


class IQLValue(nn.Module):
    """Scalar expectile value V(s) over frozen Koopman/task features."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.network = _mlp(feature_dim, hidden_dims, 1, activation)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.feature_dim:
            raise ValueError("IQL feature dimension does not match V network")
        return self.network(features).squeeze(-1)


@dataclass
class IQLTransitionBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    terminal: torch.Tensor
    behavior_action_mean: torch.Tensor | None = None


def policy_features(
    policy: HistoryKoopmanMPCPolicy,
    observations: torch.Tensor,
) -> torch.Tensor:
    """Return the same frozen lifted/task features used by PPO-KMPC."""

    split, lifted, actor_context, _, _ = policy.features(observations)
    previous_action = policy.previous_action(split)
    center = 0.5 * (policy.actor.action_low + policy.actor.action_high)
    half_range = 0.5 * (policy.actor.action_high - policy.actor.action_low)
    normalized_previous_action = (previous_action - center) / half_range
    return torch.cat(
        (lifted, actor_context, normalized_previous_action), dim=-1
    )


def expectile_loss(
    difference: torch.Tensor,
    expectile: float,
) -> torch.Tensor:
    """Asymmetric squared loss for difference ``target - prediction``."""

    if not 0.0 < expectile < 1.0:
        raise ValueError("expectile must be in (0,1)")
    weight = torch.where(
        difference > 0,
        expectile,
        1.0 - expectile,
    )
    return (weight * difference.square()).mean()


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    if not 0.0 < tau <= 1.0:
        raise ValueError("tau must be in (0,1]")
    with torch.no_grad():
        for target_parameter, source_parameter in zip(
            target.parameters(), source.parameters(), strict=True
        ):
            target_parameter.lerp_(source_parameter, tau)
        for target_buffer, source_buffer in zip(
            target.buffers(), source.buffers(), strict=True
        ):
            if torch.is_floating_point(target_buffer):
                target_buffer.lerp_(source_buffer, tau)
            else:
                target_buffer.copy_(source_buffer)


def build_iql_candidate_policy(
    initial_checkpoint: str | Path,
    device: torch.device,
    *,
    cost_parameterization: str = "checkpoint",
    solver_iterations: int | None = None,
    structured_shape_weight: float = 1e-3,
    structured_linear_velocity_weight: float = 1e-2,
    structured_angular_velocity_weight: float = 1e-2,
    structured_normalized_delta_weight: float = 1e-4,
) -> tuple[HistoryKoopmanMPCPolicy, dict, dict]:
    """Load the behavior actor or derive a structured-v2 IQL candidate."""

    from .manisoft_ppo_policies import (
        load_manisoft_ppo_checkpoint,
        make_manisoft_ppo_policy,
    )

    source, initial_payload, koopman_payload = load_manisoft_ppo_checkpoint(
        initial_checkpoint, device
    )
    if not isinstance(source, HistoryKoopmanMPCPolicy):
        raise ValueError("IQL requires a history Koopman-MPC checkpoint")
    if cost_parameterization == "checkpoint":
        if solver_iterations is not None:
            source.actor.solver_iterations = int(solver_iterations)
        return source, initial_payload, koopman_payload
    if cost_parameterization != "structured_v2":
        raise ValueError("IQL candidate must be checkpoint or structured_v2")

    runtime = initial_payload["runtime"]
    candidate, _ = make_manisoft_ppo_policy(
        "ppo_kmpc",
        initial_payload["koopman_checkpoint"],
        device,
        absolute_action_limit=float(runtime["absolute_action_limit"]),
        initial_action_std=float(runtime["initial_action_std"]),
        waypoint_count=int(runtime["waypoint_count"]),
        mlp_hidden_dims=tuple(runtime["mlp_hidden_dims"]),
        kmpc_hidden_dims=tuple(runtime["kmpc_hidden_dims"]),
        horizon=int(runtime["horizon"]),
        solver_iterations=int(
            runtime["solver_iterations"]
            if solver_iterations is None
            else solver_iterations
        ),
        quadratic_log_scale=float(runtime["quadratic_log_scale"]),
        linear_scale=float(runtime["linear_scale"]),
        action_quadratic_scale=float(runtime["action_quadratic_scale"]),
        tip_weight=float(runtime.get("tip_weight", 1.0)),
        max_delta=float(runtime["max_delta"]),
        normalized_delta_curvature=float(
            runtime.get("normalized_delta_curvature", 0.0)
        ),
        kmpc_cost_parameterization="structured_v2",
        structured_log_scale=float(runtime["structured_log_scale"]),
        structured_shape_weight=structured_shape_weight,
        structured_linear_velocity_weight=structured_linear_velocity_weight,
        structured_angular_velocity_weight=structured_angular_velocity_weight,
        structured_normalized_delta_weight=(
            structured_normalized_delta_weight
        ),
        structured_terminal_multiplier=True,
    )
    if not isinstance(candidate, HistoryKoopmanMPCPolicy):
        raise TypeError("Structured-v2 builder returned the wrong policy type")

    # Preserve learned v15e features.  The first six v2 rows are state groups;
    # v1 rows 0:3 are tip xyz, row 3 is shared R, and row 4 is terminal.
    source_layers = [m for m in source.actor.network if isinstance(m, nn.Linear)]
    candidate_layers = [
        m for m in candidate.actor.network if isinstance(m, nn.Linear)
    ]
    if len(source_layers) != len(candidate_layers):
        raise ValueError("Candidate and behavior actor depths differ")
    if source_layers[-1].out_features != 5:
        raise ValueError(
            "Structured-v2 transfer currently requires the five-output "
            "structured-v1 behavior actor"
        )
    with torch.no_grad():
        for source_layer, candidate_layer in zip(
            source_layers[:-1], candidate_layers[:-1], strict=True
        ):
            candidate_layer.weight.copy_(source_layer.weight)
            candidate_layer.bias.copy_(source_layer.bias)
        source_final, candidate_final = source_layers[-1], candidate_layers[-1]
        candidate_final.weight[0:3].copy_(source_final.weight[0:3])
        candidate_final.bias[0:3].copy_(source_final.bias[0:3])
        for row in range(6, 9):
            candidate_final.weight[row].copy_(source_final.weight[3])
            candidate_final.bias[row].copy_(source_final.bias[3])
        candidate_final.weight[9].copy_(source_final.weight[4])
        candidate_final.bias[9].copy_(source_final.bias[4])
        candidate.log_std.copy_(source.log_std)
        candidate.critic.load_state_dict(source.critic.state_dict())
    return candidate, initial_payload, koopman_payload


def distill_behavior_means(
    policy: HistoryKoopmanMPCPolicy,
    batch: IQLTransitionBatch,
    optimizer: torch.optim.Optimizer,
    *,
    max_grad_norm: float,
) -> dict[str, torch.Tensor]:
    """One supervised candidate step against stored behavior policy means."""

    if batch.behavior_action_mean is None:
        raise ValueError("V2 distillation requires behavior_action_means")
    output = policy.actor_mean(batch.observation)
    mean = (
        output.normalized_delta
        if output.normalized_delta is not None
        else output.action
    )
    loss = (mean - batch.behavior_action_mean).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group["params"]],
        max_grad_norm,
    )
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("V2 distillation gradient is NaN or Inf")
    optimizer.step()
    return {"distillation_loss": loss.detach(), "distillation_grad_norm": grad_norm.detach()}


class IQLTrainer:
    """One-gradient-step rlkit-style IQL updater for KMPC."""

    def __init__(
        self,
        policy: HistoryKoopmanMPCPolicy,
        qf1: IQLActionValue,
        qf2: IQLActionValue,
        target_qf1: IQLActionValue,
        target_qf2: IQLActionValue,
        vf: IQLValue,
        policy_optimizer: torch.optim.Optimizer,
        qf_optimizer: torch.optim.Optimizer,
        vf_optimizer: torch.optim.Optimizer,
        *,
        discount: float = 0.99,
        expectile: float = 0.9,
        temperature: float = 0.1,
        max_advantage_weight: float = 100.0,
        reward_scale: float = 1.0,
        reward_bias: float = 0.0,
        target_tau: float = 0.01,
        max_grad_norm: float = 1.0,
        minimum_log_std: float | None = None,
        maximum_log_std: float | None = None,
    ) -> None:
        if not 0.0 <= discount <= 1.0:
            raise ValueError("discount must be in [0,1]")
        if not 0.0 < expectile < 1.0:
            raise ValueError("expectile must be in (0,1)")
        if temperature <= 0 or max_advantage_weight <= 0:
            raise ValueError("IQL temperature and weight clip must be positive")
        if not 0.0 < target_tau <= 1.0 or max_grad_norm <= 0:
            raise ValueError("Target tau and gradient norm must be positive")
        if (
            minimum_log_std is not None
            and maximum_log_std is not None
            and minimum_log_std > maximum_log_std
        ):
            raise ValueError("minimum_log_std must not exceed maximum_log_std")
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vf = vf
        self.policy_optimizer = policy_optimizer
        self.qf_optimizer = qf_optimizer
        self.vf_optimizer = vf_optimizer
        self.discount = float(discount)
        self.expectile = float(expectile)
        self.temperature = float(temperature)
        self.max_advantage_weight = float(max_advantage_weight)
        self.reward_scale = float(reward_scale)
        self.reward_bias = float(reward_bias)
        self.target_tau = float(target_tau)
        self.max_grad_norm = float(max_grad_norm)
        self.minimum_log_std = minimum_log_std
        self.maximum_log_std = maximum_log_std

    def update(
        self,
        batch: IQLTransitionBatch,
        *,
        update_policy: bool = True,
    ) -> dict[str, torch.Tensor]:
        self.policy.actor.train()
        self.qf1.train()
        self.qf2.train()
        self.vf.train()
        with torch.no_grad():
            features = policy_features(self.policy, batch.observation)
            next_features = policy_features(self.policy, batch.next_observation)
            transformed_reward = self.reward_scale * batch.reward + self.reward_bias
            q_target = transformed_reward + (
                self.discount
                * (1.0 - batch.terminal)
                * self.vf(next_features)
            )

        q1_prediction = self.qf1(features, batch.action)
        q2_prediction = self.qf2(features, batch.action)
        qf1_loss = (q1_prediction - q_target).square().mean()
        qf2_loss = (q2_prediction - q_target).square().mean()
        qf_loss = qf1_loss + qf2_loss
        self.qf_optimizer.zero_grad(set_to_none=True)
        qf_loss.backward()
        qf_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            self.max_grad_norm,
        )
        if not torch.isfinite(qf_grad_norm):
            raise FloatingPointError("IQL Q gradient is NaN or Inf")
        self.qf_optimizer.step()

        with torch.no_grad():
            target_q = torch.minimum(
                self.target_qf1(features, batch.action),
                self.target_qf2(features, batch.action),
            )
        value_prediction = self.vf(features)
        advantage_before_value_update = target_q - value_prediction
        vf_loss = expectile_loss(
            advantage_before_value_update,
            self.expectile,
        )
        self.vf_optimizer.zero_grad(set_to_none=True)
        vf_loss.backward()
        vf_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.vf.parameters(), self.max_grad_norm
        )
        if not torch.isfinite(vf_grad_norm):
            raise FloatingPointError("IQL V gradient is NaN or Inf")
        self.vf_optimizer.step()

        with torch.no_grad():
            advantage = target_q - self.vf(features)
            advantage_weight = torch.exp(
                advantage / self.temperature
            ).clamp(max=self.max_advantage_weight)
        policy_loss = features.new_zeros(())
        policy_grad_norm = features.new_zeros(())
        log_probability = features.new_zeros(features.shape[:-1])
        if update_policy:
            log_probability, _, _, _ = self.policy.evaluate_actions(
                batch.observation,
                batch.action,
            )
            policy_loss = -(advantage_weight * log_probability).mean()
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            policy_parameters = [
                parameter
                for group in self.policy_optimizer.param_groups
                for parameter in group["params"]
            ]
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                policy_parameters, self.max_grad_norm
            )
            if not torch.isfinite(policy_grad_norm):
                raise FloatingPointError("IQL policy gradient is NaN or Inf")
            self.policy_optimizer.step()
            with torch.no_grad():
                if self.minimum_log_std is not None:
                    self.policy.log_std.clamp_(min=self.minimum_log_std)
                if self.maximum_log_std is not None:
                    self.policy.log_std.clamp_(max=self.maximum_log_std)

        soft_update(self.target_qf1, self.qf1, self.target_tau)
        soft_update(self.target_qf2, self.qf2, self.target_tau)
        return {
            "qf1_loss": qf1_loss.detach(),
            "qf2_loss": qf2_loss.detach(),
            "vf_loss": vf_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "qf_grad_norm": qf_grad_norm.detach(),
            "vf_grad_norm": vf_grad_norm.detach(),
            "policy_grad_norm": policy_grad_norm.detach(),
            "q1_mean": q1_prediction.detach().mean(),
            "q2_mean": q2_prediction.detach().mean(),
            "q_target_mean": q_target.detach().mean(),
            "value_mean": value_prediction.detach().mean(),
            "advantage_mean": advantage.detach().mean(),
            "advantage_weight_mean": advantage_weight.detach().mean(),
            "advantage_weight_max": advantage_weight.detach().max(),
            "log_probability_mean": log_probability.detach().mean(),
            "reward_mean": batch.reward.detach().mean(),
            "terminal_fraction": batch.terminal.detach().mean(),
            "policy_updated": features.new_tensor(float(update_policy)),
        }


@torch.no_grad()
def offline_validation_metrics(
    policy: HistoryKoopmanMPCPolicy,
    qf1: IQLActionValue,
    qf2: IQLActionValue,
    vf: IQLValue,
    batch: IQLTransitionBatch,
    *,
    discount: float,
    expectile: float,
    temperature: float,
    max_advantage_weight: float,
    reward_scale: float,
    reward_bias: float,
) -> dict[str, float]:
    policy.eval()
    qf1.eval()
    qf2.eval()
    vf.eval()
    features = policy_features(policy, batch.observation)
    next_features = policy_features(policy, batch.next_observation)
    q1 = qf1(features, batch.action)
    q2 = qf2(features, batch.action)
    target_q = torch.minimum(q1, q2)
    value = vf(features)
    advantage = target_q - value
    weight = torch.exp(advantage / temperature).clamp(
        max=max_advantage_weight
    )
    transformed_reward = reward_scale * batch.reward + reward_bias
    bellman_target = transformed_reward + (
        discount * (1.0 - batch.terminal) * vf(next_features)
    )
    log_probability, _, _, output = policy.evaluate_actions(
        batch.observation, batch.action
    )
    result = {
        "qf1_loss": float((q1 - bellman_target).square().mean()),
        "qf2_loss": float((q2 - bellman_target).square().mean()),
        "vf_loss": float(expectile_loss(advantage, expectile)),
        "policy_loss": float(-(weight * log_probability).mean()),
        "behavior_negative_log_likelihood": float(-log_probability.mean()),
        "behavior_action_mse": float(
            (output.mean - batch.action).square().mean()
        ),
        "q_mean": float(target_q.mean()),
        "value_mean": float(value.mean()),
        "advantage_mean": float(advantage.mean()),
        "advantage_weight_mean": float(weight.mean()),
        "advantage_weight_max": float(weight.max()),
        "bellman_target_mean": float(bellman_target.mean()),
    }
    if batch.behavior_action_mean is not None:
        result["behavior_mean_mse"] = float(
            (output.mean - batch.behavior_action_mean).square().mean()
        )
    if output.mpc.normalized_delta is not None:
        result["projected_gradient_residual_mean"] = float(
            output.mpc.projected_gradient_residual.mean()
        )
    return result


@torch.no_grad()
def fixed_critic_selection_metrics(
    policy: HistoryKoopmanMPCPolicy,
    selection_qf1: IQLActionValue,
    selection_qf2: IQLActionValue,
    batch: IQLTransitionBatch,
    *,
    behavior_mse_penalty: float,
) -> dict[str, float]:
    """Comparable offline score using critics frozen after warm-up."""

    if batch.behavior_action_mean is None:
        raise ValueError("Checkpoint selection requires behavior_action_means")
    features = policy_features(policy, batch.observation)
    candidate = policy.actor_mean(batch.observation)
    candidate_action = (
        candidate.normalized_delta
        if candidate.normalized_delta is not None
        else candidate.action
    )
    behavior = batch.behavior_action_mean
    candidate_q = torch.minimum(
        selection_qf1(features, candidate_action),
        selection_qf2(features, candidate_action),
    )
    behavior_q = torch.minimum(
        selection_qf1(features, behavior),
        selection_qf2(features, behavior),
    )
    behavior_mse = (candidate_action - behavior).square().mean()
    q_improvement = (candidate_q - behavior_q).mean()
    score = q_improvement - float(behavior_mse_penalty) * behavior_mse
    return {
        "score": float(score),
        "candidate_q_mean": float(candidate_q.mean()),
        "behavior_q_mean": float(behavior_q.mean()),
        "q_improvement": float(q_improvement),
        "behavior_mean_mse": float(behavior_mse),
    }


def load_manisoft_iql_checkpoint(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[HistoryKoopmanMPCPolicy, dict, dict]:
    """Reconstruct the deployment policy stored in a ManiSoft KMPC-IQL run."""

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("method") != "manisoft_kmpc_iql":
        raise ValueError(f"{path} is not a ManiSoft KMPC-IQL checkpoint")
    initialization_path = Path(payload["initial_policy_checkpoint"])
    if not initialization_path.is_file():
        raise FileNotFoundError(
            f"Missing IQL initialization policy: {initialization_path}"
        )
    candidate = payload.get("candidate", {})
    policy, initialization_payload, koopman_payload = build_iql_candidate_policy(
        initialization_path,
        device,
        cost_parameterization=candidate.get("cost_parameterization", "checkpoint"),
        solver_iterations=candidate.get("solver_iterations"),
        structured_shape_weight=float(candidate.get("structured_shape_weight", 1e-3)),
        structured_linear_velocity_weight=float(
            candidate.get("structured_linear_velocity_weight", 1e-2)
        ),
        structured_angular_velocity_weight=float(
            candidate.get("structured_angular_velocity_weight", 1e-2)
        ),
        structured_normalized_delta_weight=float(
            candidate.get("structured_normalized_delta_weight", 1e-4)
        ),
    )
    if initialization_payload.get("actor_name") != "ppo_kmpc":
        raise ValueError("IQL initialization is not a PPO-KMPC policy")
    policy.load_state_dict(payload["policy"])
    return policy, payload, koopman_payload
