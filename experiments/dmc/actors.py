"""Shared actor and Koopman-checkpoint utilities for the DMC sub-project.

The validated Hopper experiments used four peer policies: a plain PPO MLP,
KLQR, AB-PQ and finite-horizon Koopman MPC.  DMC additionally evaluates an
``AC-MPC-MPVE`` critic-training ablation whose actor is exactly KMPC.  Keeping
their construction in one module prevents the collector, PPO trainer and
evaluator from silently using different architectures or checkpoint
conventions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from antmaze_ac.koopman.model import DeepKoopman
from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor
from antmaze_ac.rl.quadratic_actors import KoopmanLQRActor, LowRankValueActor
from experiments.dmc.protocol import protocol_fingerprint_from_json
from experiments.dmc.tasks.registry import get_task_spec


ACTOR_TYPES = ("PPO", "KLQR", "AB-PQ", "KMPC", "AC-MPC-MPVE")
HAIKU_DEFAULT_LINEAR_INITIALIZATION = (
    "truncated_normal_stddev_inverse_sqrt_fan_in_bounds_2sigma_v1"
)


@dataclass(frozen=True)
class ActorConfig:
    """Architecture-only settings saved into every actor checkpoint."""

    hidden_dim: int = 128
    ppo_hidden_dim: int = 256
    ppo_hidden_layers: int = 3
    ppo_activation: str = "relu"
    ppo_distribution: str = "tanh_squashed_state_dependent_gaussian"
    ab_rank: int = 4
    kmpc_horizon: int = 10
    kmpc_solver_iterations: int = 20
    action_limit: float = 1.0

    def validate(self) -> None:
        integer_fields = {
            "hidden_dim": self.hidden_dim,
            "ppo_hidden_dim": self.ppo_hidden_dim,
            "ppo_hidden_layers": self.ppo_hidden_layers,
            "ab_rank": self.ab_rank,
            "kmpc_horizon": self.kmpc_horizon,
            "kmpc_solver_iterations": self.kmpc_solver_iterations,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim < 1 or self.ppo_hidden_dim < 1:
            raise ValueError("actor hidden dimensions must be positive")
        if self.ppo_hidden_layers != 3:
            raise ValueError("Primary DMC PPO requires ppo_hidden_layers=3")
        if self.ppo_activation != "relu":
            raise ValueError("Primary DMC PPO requires ppo_activation='relu'")
        if self.ppo_distribution != "tanh_squashed_state_dependent_gaussian":
            raise ValueError(
                "Primary DMC PPO requires the Acme tanh-squashed "
                "state-dependent Gaussian"
            )
        if not isinstance(self.action_limit, (int, float)) or isinstance(
            self.action_limit, bool
        ):
            raise ValueError("action_limit must be a finite positive number")
        if not math.isfinite(float(self.action_limit)) or self.action_limit <= 0:
            raise ValueError("action_limit must be a finite positive number")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "ActorConfig":
        if value is None:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        unknown = set(value) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown actor config fields: {sorted(unknown)}")
        config = cls(**{name: value[name] for name in allowed if name in value})
        config.validate()
        return config


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def initialize_haiku_default_linear(layer: nn.Linear) -> None:
    """Match Haiku ``Linear``'s default truncated-Normal initializer."""

    fan_in = int(layer.weight.shape[1])
    if fan_in < 1:
        raise ValueError("Linear fan-in must be positive")
    stddev = 1.0 / math.sqrt(fan_in)
    nn.init.trunc_normal_(
        layer.weight,
        mean=0.0,
        std=stddev,
        a=-2.0 * stddev,
        b=2.0 * stddev,
    )
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class StandardPPOActor(nn.Module):
    """Acme continuous-control PPO policy network.

    A three-layer ReLU torso produces state-dependent location and scale
    parameters.  Training samples a diagonal Normal followed by ``tanh``;
    deterministic evaluation uses ``tanh(loc)``.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        action_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            input_dim < 1
            or action_dim < 1
            or hidden_dim < 1
            or hidden_layers < 1
        ):
            raise ValueError("actor dimensions must be positive")
        if not math.isfinite(float(action_limit)) or action_limit <= 0:
            raise ValueError("action_limit must be finite and positive")
        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.action_limit = float(action_limit)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        layers: list[nn.Module] = []
        following_input = self.input_dim
        for _ in range(self.hidden_layers):
            layer = nn.Linear(following_input, self.hidden_dim)
            initialize_haiku_default_linear(layer)
            layers.extend((layer, nn.ReLU()))
            following_input = self.hidden_dim
        self.network = nn.Sequential(*layers)
        self.loc_layer = nn.Linear(self.hidden_dim, self.action_dim)
        self.scale_layer = nn.Linear(self.hidden_dim, self.action_dim)
        output_bound = math.sqrt(3.0 / self.hidden_dim)
        nn.init.uniform_(self.loc_layer.weight, -output_bound, output_bound)
        nn.init.uniform_(self.scale_layer.weight, -output_bound, output_bound)
        nn.init.zeros_(self.loc_layer.bias)
        nn.init.zeros_(self.scale_layer.bias)

    def distribution_parameters(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected state dimension {self.input_dim}, got {state.shape[-1]}"
            )
        features = self.network(state)
        location = self.loc_layer(features)
        scale = torch.nn.functional.softplus(self.scale_layer(features)) + 1e-3
        return location, scale

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        location, _scale = self.distribution_parameters(state)
        return self.action_limit * torch.tanh(location)


def _checkpoint_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "model" in checkpoint:
        return checkpoint["model"]
    raise ValueError("Koopman checkpoint has neither 'model_state' nor 'model'")


def _checkpoint_normalizer(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "normalizer" in checkpoint:
        return checkpoint["normalizer"]
    if "normalizers" in checkpoint:
        return checkpoint["normalizers"]
    raise ValueError("Koopman checkpoint has no state normalizer")


def checkpoint_protocol_fingerprint(checkpoint: dict[str, Any]) -> str | None:
    """Read and validate the environment protocol identity from best/latest files."""

    training_state = checkpoint.get("training_state") or {}
    fingerprint = checkpoint.get(
        "protocol_fingerprint", training_state.get("protocol_fingerprint")
    )
    environment_json = checkpoint.get(
        "environment_protocol_json",
        training_state.get("environment_protocol_json"),
    )
    is_dmc_checkpoint = (
        checkpoint.get("kind") == "dmc_k_step_koopman"
        or training_state.get("task_name") is not None
    )
    if fingerprint is None and environment_json is None:
        if is_dmc_checkpoint:
            raise ValueError("DMC Koopman checkpoint is missing protocol identity")
        return None
    if not isinstance(fingerprint, str) or not isinstance(environment_json, str):
        raise ValueError(
            "Koopman checkpoint must save both protocol_fingerprint and "
            "environment_protocol_json"
        )
    try:
        expected = protocol_fingerprint_from_json(environment_json)
    except (ValueError, TypeError) as exc:
        raise ValueError("Koopman checkpoint has invalid environment protocol") from exc
    if fingerprint != expected:
        raise ValueError("Koopman checkpoint protocol fingerprint is invalid")
    return fingerprint


def load_koopman(
    path: Path,
    task_name: str,
    device: torch.device,
) -> tuple[DeepKoopman, dict[str, Any]]:
    """Load either a DMC ``best.pt`` or resumable ``latest.pt`` checkpoint."""

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Koopman checkpoint {path} must contain a mapping")
    if "architecture" not in checkpoint:
        raise ValueError(f"Koopman checkpoint {path} has no architecture metadata")
    if not isinstance(checkpoint["architecture"], dict):
        raise TypeError("Koopman checkpoint architecture must be a mapping")
    architecture = dict(checkpoint["architecture"])
    architecture_name = architecture.pop("architecture", None)
    if architecture_name not in (None, "fullA_history_v2_adapted"):
        raise ValueError(
            f"Unsupported Koopman architecture {architecture_name!r}"
        )
    model = DeepKoopman(
        state_dim=int(architecture["state_dim"]),
        action_dim=int(architecture["action_dim"]),
        lift_dim=int(architecture["lift_dim"]),
        hidden_dims=tuple(architecture["hidden_dims"]),
        activation=str(architecture.get("activation", "silu")),
    ).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint))
    model.freeze_dynamics()

    spec = get_task_spec(task_name)
    training_state = checkpoint.get("training_state") or {}
    state_kind = checkpoint.get(
        "state_kind", training_state.get("state_kind", training_state.get("task_name"))
    )
    if state_kind is not None and str(state_kind) != task_name:
        raise ValueError(
            f"Koopman checkpoint state_kind {state_kind!r} does not match "
            f"task {task_name!r}"
        )
    if model.state_dim != spec.obs_dim or model.action_dim != spec.action_dim:
        raise ValueError(
            f"Koopman architecture ({model.state_dim}, {model.action_dim}) does "
            f"not match {task_name} ({spec.obs_dim}, {spec.action_dim})"
        )
    checkpoint_protocol_fingerprint(checkpoint)
    return model, checkpoint


def normalizer_arrays(
    checkpoint: dict[str, Any], task_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated float32 center/scale arrays from a Koopman checkpoint."""

    def as_array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    normalizer = _checkpoint_normalizer(checkpoint)
    center = as_array(normalizer["center"])
    scale = as_array(normalizer["scale"])
    expected = (get_task_spec(task_name).obs_dim,)
    if center.shape != expected or scale.shape != expected:
        raise ValueError(
            f"Normalizer shapes {center.shape}/{scale.shape} do not match {expected}"
        )
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise FloatingPointError("Koopman normalizer contains NaN or Inf")
    if np.any(scale <= 0):
        raise ValueError("Koopman normalizer scale must be strictly positive")
    return center, scale


def build_actor(
    actor_type: str,
    task_name: str,
    device: torch.device,
    *,
    koopman: DeepKoopman | None = None,
    config: ActorConfig | None = None,
) -> nn.Module:
    """Build one of five methods; AC-MPC-MPVE deliberately shares KMPC's actor."""

    if actor_type not in ACTOR_TYPES:
        raise ValueError(f"Unknown actor type {actor_type!r}; expected {ACTOR_TYPES}")
    config = config or ActorConfig()
    config.validate()
    spec = get_task_spec(task_name)
    if actor_type == "PPO":
        return StandardPPOActor(
            spec.obs_dim,
            spec.action_dim,
            hidden_dim=config.ppo_hidden_dim,
            hidden_layers=config.ppo_hidden_layers,
            action_limit=config.action_limit,
        ).to(device)
    if koopman is None:
        raise ValueError(f"{actor_type} requires a frozen Koopman model")
    if koopman.state_dim != spec.obs_dim or koopman.action_dim != spec.action_dim:
        raise ValueError("Koopman model dimensions do not match the DMC task")

    lifted_dim = koopman.lifted_dim
    if actor_type == "KLQR":
        actor: nn.Module = KoopmanLQRActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            context_dim=0,
            hidden_dims=(config.hidden_dim,),
            max_action=config.action_limit,
        )
    elif actor_type == "AB-PQ":
        actor = LowRankValueActor(
            observation_dim=lifted_dim,
            A=koopman.A,
            B=koopman.B,
            R=torch.eye(spec.action_dim, device=device, dtype=koopman.A.dtype),
            base_hessian=torch.eye(
                lifted_dim, device=device, dtype=koopman.A.dtype
            ),
            rank=config.ab_rank,
            hidden_dims=(config.hidden_dim,),
            max_action=config.action_limit,
        )
    else:  # KMPC and AC-MPC-MPVE deliberately share the exact same actor.
        actor = KoopmanMPCActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            horizon=config.kmpc_horizon,
            context_dim=0,
            hidden_dims=(config.hidden_dim,),
            action_low=-config.action_limit,
            action_high=config.action_limit,
            solver_iterations=config.kmpc_solver_iterations,
        )
    return actor.to(device)


def actor_mean(
    actor_type: str,
    actor: nn.Module,
    state: torch.Tensor,
    lifted: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the deterministic action mean for a batch of task states."""

    if actor_type == "PPO":
        return actor(state)
    if lifted is None:
        raise ValueError(f"{actor_type} requires lifted state features")
    if actor_type == "KLQR":
        return actor(lifted).action
    if actor_type == "AB-PQ":
        return actor(lifted, lifted).action
    if actor_type in ("KMPC", "AC-MPC-MPVE"):
        return actor(lifted).action
    raise ValueError(f"Unknown actor type {actor_type!r}")


def actor_config_from_checkpoint(payload: dict[str, Any]) -> ActorConfig:
    """Load saved architecture settings, rejecting ambiguous old payloads."""

    value = payload.get("actor_config")
    if value is None:
        raise ValueError("Actor checkpoint is missing 'actor_config' metadata")
    if not isinstance(value, dict):
        raise TypeError("actor_config metadata must be a mapping")
    return ActorConfig.from_mapping(value)
