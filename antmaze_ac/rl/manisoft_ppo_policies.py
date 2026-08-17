from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torch.distributions import Normal

from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256
from antmaze_ac.koopman.history_model import HistoryDeepKoopman

from .critic import Critic
from .history_koopman_mpc_policy import HistoryKoopmanMPCPolicy
from .koopman_mpc_actor import KoopmanMPCActor, StructuredKoopmanMPCActor


PPO_ACTOR_NAMES = ("ppo_mlp", "ppo_kmpc")


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _tanh_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    output_gain: float,
) -> nn.Sequential:
    if input_dim < 1 or output_dim < 1 or not hidden_dims:
        raise ValueError("MLP dimensions must be positive and non-empty")
    dimensions = [int(input_dim), *map(int, hidden_dims), int(output_dim)]
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:])
    ):
        layer = nn.Linear(in_dim, out_dim)
        _orthogonal(
            layer,
            output_gain if index == len(dimensions) - 2 else math.sqrt(2.0),
        )
        layers.append(layer)
        if index < len(dimensions) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _initialize_critic(network: nn.Sequential) -> None:
    linear_layers = [
        layer for layer in network if isinstance(layer, nn.Linear)
    ]
    for index, layer in enumerate(linear_layers):
        _orthogonal(
            layer,
            1.0 if index == len(linear_layers) - 1 else math.sqrt(2.0),
        )


class ManiSoftPPOObservation(NamedTuple):
    physical_state: torch.Tensor
    history_context: torch.Tensor
    task_context: torch.Tensor


class StandardPPOActor(nn.Module):
    """Conventional MLP Gaussian-mean actor without a Koopman model."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        *,
        action_limit: float = 0.30,
    ) -> None:
        super().__init__()
        if action_limit <= 0:
            raise ValueError("action_limit must be positive")
        self.network = _tanh_mlp(
            input_dim,
            hidden_dims,
            action_dim,
            output_gain=0.01,
        )
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.action_limit = float(action_limit)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected actor input dimension {self.input_dim}, "
                f"got {features.shape[-1]}"
            )
        # Match the reference repository's standard PPO route: the Gaussian
        # mean is linear, while the environment clips sampled physical actions.
        return self.action_limit * self.network(features)


@dataclass
class StandardPPOPolicyOutput:
    distribution: Normal
    mean: torch.Tensor
    value: torch.Tensor
    features: torch.Tensor


class StandardHistoryPPOPolicy(nn.Module):
    """MLP actor-critic on the frozen history-Koopman lift and task context.

    The feature extractor is identical to PPO-KMPC: normalize the current
    physical state, lift it together with the finite state/action history, and
    append normalized waypoint tips plus the active-stage one-hot vector. Only
    the actor head differs: it maps these features directly to an action mean
    instead of constructing and solving an MPC problem.
    """

    ACTION_DISTRIBUTION = "diagonal_normal_v1"

    def __init__(
        self,
        koopman: HistoryDeepKoopman,
        actor: StandardPPOActor,
        critic: nn.Module,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        *,
        waypoint_count: int = 3,
        tip_indices: tuple[int, int, int] = (30, 31, 32),
        log_std_init: float = math.log(0.015),
    ) -> None:
        super().__init__()
        if not isinstance(koopman, HistoryDeepKoopman):
            raise TypeError("StandardHistoryPPOPolicy requires HistoryDeepKoopman")
        if waypoint_count < 1:
            raise ValueError("Policy dimensions must be positive")
        self.koopman = koopman.freeze_dynamics()
        self.state_dim = int(koopman.state_dim)
        self.action_dim = int(koopman.action_dim)
        self.history_steps = int(koopman.history_steps)
        self.waypoint_count = int(waypoint_count)
        self.history_context_dim = self.history_steps * (
            self.state_dim + self.action_dim
        )
        self.task_observation_dim = 4 * self.waypoint_count
        self.observation_dim = (
            self.state_dim
            + self.history_context_dim
            + self.task_observation_dim
        )
        self.feature_dim = int(koopman.lifted_dim + self.task_observation_dim)
        if actor.input_dim != self.feature_dim or actor.action_dim != self.action_dim:
            raise ValueError("Actor dimensions do not match the policy")
        first_critic_layer = next(
            (layer for layer in critic.modules() if isinstance(layer, nn.Linear)),
            None,
        )
        if (
            first_critic_layer is None
            or first_critic_layer.in_features != self.feature_dim
        ):
            raise ValueError("Critic input dimension does not match the policy")
        if state_mean.shape != (self.state_dim,) or state_std.shape != (
            self.state_dim,
        ):
            raise ValueError("State normalizer shape does not match the policy")

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
    ) -> ManiSoftPPOObservation:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dimension {self.observation_dim}, "
                f"got {observation.shape[-1]}"
            )
        state_stop = self.state_dim
        history_stop = state_stop + self.history_context_dim
        return ManiSoftPPOObservation(
            observation[..., :state_stop],
            observation[..., state_stop:history_stop],
            observation[..., history_stop:],
        )

    def policy_features(self, observation: torch.Tensor) -> torch.Tensor:
        split = self.split_observation(observation)
        normalized_state = (
            split.physical_state - self.state_mean
        ) / self.state_std
        lifted = self.koopman.lift(normalized_state, split.history_context)
        waypoint_stop = 3 * self.waypoint_count
        waypoints = split.task_context[..., :waypoint_stop].reshape(
            *split.task_context.shape[:-1], self.waypoint_count, 3
        )
        stage = split.task_context[..., waypoint_stop:]
        tip_mean = self.state_mean[self.tip_indices]
        tip_std = self.state_std[self.tip_indices]
        task_features = torch.cat(
            (((waypoints - tip_mean) / tip_std).flatten(start_dim=-2), stage),
            dim=-1,
        )
        return torch.cat(
            (
                lifted,
                task_features,
            ),
            dim=-1,
        )

    def actor_mean(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(self.policy_features(observation))

    def forward(self, observation: torch.Tensor) -> StandardPPOPolicyOutput:
        single = observation.ndim == 1
        batch = observation.unsqueeze(0) if single else observation
        features = self.policy_features(batch)
        mean_batch = self.actor(features)
        if not torch.isfinite(mean_batch).all():
            raise FloatingPointError("PPO-MLP actor produced NaN or Inf")
        distribution = Normal(
            mean_batch,
            self.log_std.exp().expand_as(mean_batch),
        )
        value_batch = self.critic(features)
        return StandardPPOPolicyOutput(
            distribution=distribution,
            mean=mean_batch[0] if single else mean_batch,
            value=value_batch[0] if single else value_batch,
            features=features[0] if single else features,
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
        evaluated_action = action.unsqueeze(0) if action.ndim == 1 else action
        log_prob = output.distribution.log_prob(evaluated_action).sum(dim=-1)
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


def make_manisoft_ppo_policy(
    actor_name: str,
    koopman_checkpoint: str | Path,
    device: torch.device,
    *,
    absolute_action_limit: float = 0.30,
    initial_action_std: float = 0.015,
    waypoint_count: int = 3,
    mlp_hidden_dims: Sequence[int] = (256, 256),
    kmpc_hidden_dims: Sequence[int] = (128,),
    horizon: int = 10,
    solver_iterations: int = 20,
    quadratic_log_scale: float = 1.5,
    linear_scale: float = 10.0,
    action_quadratic_scale: float = 1.0,
    tip_weight: float = 1.0,
    max_delta: float | None = 0.001,
    normalized_delta_curvature: float = 0.0,
    kmpc_cost_parameterization: str = "full",
    structured_log_scale: float = math.log(2.0),
) -> tuple[StandardHistoryPPOPolicy | HistoryKoopmanMPCPolicy, dict]:
    """Build one of the two from-scratch PPO comparison policies."""

    if actor_name not in PPO_ACTOR_NAMES:
        raise ValueError(f"Unsupported actor_name {actor_name!r}")
    if (
        absolute_action_limit <= 0
        or initial_action_std <= 0
        or tip_weight <= 0
    ):
        raise ValueError("Action limit and initial standard deviation must be positive")
    if max_delta is not None and max_delta <= 0:
        raise ValueError("max_delta must be positive when configured")
    if kmpc_cost_parameterization not in ("full", "structured"):
        raise ValueError(
            "kmpc_cost_parameterization must be 'full' or 'structured'"
        )
    if structured_log_scale <= 0:
        raise ValueError("structured_log_scale must be positive")
    checkpoint = Path(koopman_checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    architecture = payload.get("architecture", {})
    if architecture.get("architecture") != "fullA_history_context_v1":
        raise ValueError(
            "ManiSoft PPO comparison requires a history Koopman checkpoint"
        )
    state_dim = int(architecture["state_dim"])
    action_dim = int(architecture["action_dim"])
    history_steps = int(architecture["history_steps"])
    state_stats = payload["normalizers"]["state"]
    state_mean = torch.as_tensor(state_stats["mean"], dtype=torch.float32)
    state_std = torch.as_tensor(state_stats["std"], dtype=torch.float32)
    log_std_init = math.log(float(initial_action_std))
    koopman, loaded_payload = load_checkpoint(checkpoint, map_location=device)
    if not isinstance(koopman, HistoryDeepKoopman):
        raise ValueError("ManiSoft PPO requires HistoryDeepKoopman")

    if actor_name == "ppo_mlp":
        task_dim = 4 * int(waypoint_count)
        feature_dim = int(koopman.lifted_dim + task_dim)
        actor = StandardPPOActor(
            feature_dim,
            action_dim,
            mlp_hidden_dims,
            action_limit=absolute_action_limit,
        )
        critic = _tanh_mlp(
            feature_dim,
            mlp_hidden_dims,
            1,
            output_gain=1.0,
        )

        class _ScalarCritic(nn.Module):
            def __init__(self, network: nn.Sequential) -> None:
                super().__init__()
                self.network = network

            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return self.network(features).squeeze(-1)

        policy: StandardHistoryPPOPolicy | HistoryKoopmanMPCPolicy = (
            StandardHistoryPPOPolicy(
                koopman,
                actor,
                _ScalarCritic(critic),
                state_mean,
                state_std,
                waypoint_count=waypoint_count,
                log_std_init=log_std_init,
            )
        )
        return policy.to(device), loaded_payload

    context_dim = 4 * int(waypoint_count)
    physical_quadratic_scale = torch.full_like(state_std, 1e-8)
    tip_indices = torch.as_tensor((30, 31, 32), dtype=torch.long)
    # Tip-only initialization: the only non-negligible state penalty is the
    # three-dimensional active-goal error available at deployment time.
    physical_quadratic_scale[tip_indices] = float(tip_weight)
    actor_class = (
        StructuredKoopmanMPCActor
        if kmpc_cost_parameterization == "structured"
        else KoopmanMPCActor
    )
    actor_kwargs = {}
    if kmpc_cost_parameterization == "structured":
        actor_kwargs["structured_log_scale"] = structured_log_scale
        actor_kwargs["structured_tip_indices"] = (30, 31, 32)
    actor = actor_class(
        koopman.A,
        koopman.B,
        koopman.C[: koopman.state_dim],
        horizon=horizon,
        context_dim=context_dim,
        hidden_dims=kmpc_hidden_dims,
        activation="gelu",
        action_low=-absolute_action_limit,
        action_high=absolute_action_limit,
        physical_quadratic_scale=physical_quadratic_scale.to(device),
        quadratic_log_scale=quadratic_log_scale,
        linear_scale=linear_scale,
        action_quadratic_scale=action_quadratic_scale,
        max_delta=max_delta,
        normalized_delta_curvature=normalized_delta_curvature,
        solver_iterations=solver_iterations,
        **actor_kwargs,
    )
    critic = Critic(
        koopman.lifted_dim + context_dim,
        hidden_dims=(256, 256),
        activation="tanh",
    )
    _initialize_critic(critic.network)
    policy = HistoryKoopmanMPCPolicy(
        koopman,
        actor,
        critic,
        state_mean.to(device),
        state_std.to(device),
        waypoint_count=waypoint_count,
        log_std_init=log_std_init,
    )
    policy.cost_initialization = (
        "structured_reference_weights_v1"
        if kmpc_cost_parameterization == "structured"
        else "active_waypoint_tip_only_v1"
    )
    return policy.to(device), loaded_payload


def load_manisoft_ppo_checkpoint(
    checkpoint: str | Path,
    device: torch.device,
) -> tuple[StandardHistoryPPOPolicy | HistoryKoopmanMPCPolicy, dict, dict]:
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("method") != "manisoft_ppo_from_scratch":
        raise ValueError(f"{path} is not a from-scratch ManiSoft PPO checkpoint")
    koopman_path = Path(payload["koopman_checkpoint"])
    if not koopman_path.is_file():
        raise FileNotFoundError(f"Missing Koopman metadata/model: {koopman_path}")
    if sha256(koopman_path) != payload["koopman_checkpoint_sha256"]:
        raise ValueError("Koopman checkpoint SHA256 does not match")
    runtime = payload["runtime"]
    policy, koopman_payload = make_manisoft_ppo_policy(
        payload["actor_name"],
        koopman_path,
        device,
        absolute_action_limit=float(runtime["absolute_action_limit"]),
        initial_action_std=float(runtime["initial_action_std"]),
        waypoint_count=int(runtime["waypoint_count"]),
        mlp_hidden_dims=tuple(runtime["mlp_hidden_dims"]),
        kmpc_hidden_dims=tuple(runtime["kmpc_hidden_dims"]),
        horizon=int(runtime["horizon"]),
        solver_iterations=int(runtime["solver_iterations"]),
        quadratic_log_scale=float(runtime["quadratic_log_scale"]),
        linear_scale=float(runtime["linear_scale"]),
        action_quadratic_scale=float(runtime["action_quadratic_scale"]),
        tip_weight=float(runtime.get("tip_weight", 1.0)),
        max_delta=(
            None
            if payload["actor_name"] != "ppo_kmpc"
            or runtime.get("max_delta") is None
            else float(runtime["max_delta"])
        ),
        normalized_delta_curvature=float(
            runtime.get("normalized_delta_curvature", 0.0)
        ),
        kmpc_cost_parameterization=str(
            runtime.get("kmpc_cost_parameterization") or "full"
        ),
        structured_log_scale=float(
            runtime.get("structured_log_scale") or math.log(2.0)
        ),
    )
    policy.load_state_dict(payload["policy"])
    return policy, payload, koopman_payload
