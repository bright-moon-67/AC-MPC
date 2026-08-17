"""Stochastic raw-state/AC-KMPC actors and REDQ/Cal-QL critics."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.dmc.o2o.koopman import FrozenKoopman


# Match the state-based RLPD TanhNormal policy.  In particular, do not use a
# tanh remapping here: a raw output near zero should mean an initial std near
# one, for both the MLP and AC-KMPC actors.
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
EXORL_CQL_LOG_STD_MIN = -10.0


class FrozenObservationNormalizer(nn.Module):
    """Dataset-bound raw-observation transform for non-Koopman baselines."""

    KIND = "acmpc_offline_observation_normalizer_v1"

    def __init__(
        self,
        center: np.ndarray | torch.Tensor,
        scale: np.ndarray | torch.Tensor,
        *,
        dataset_sha256: str,
    ) -> None:
        super().__init__()
        center_tensor = torch.as_tensor(center, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        if center_tensor.ndim != 1 or scale_tensor.shape != center_tensor.shape:
            raise ValueError("Raw observation center/scale shapes disagree")
        if center_tensor.numel() != 5:
            raise ValueError("Cartpole raw normalizer must have dimension five")
        if not torch.isfinite(center_tensor).all() or not torch.isfinite(scale_tensor).all():
            raise FloatingPointError("Raw observation normalizer is non-finite")
        if torch.any(scale_tensor <= 0):
            raise ValueError("Raw observation scales must be positive")
        if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64:
            raise ValueError("Raw normalizer requires the source dataset SHA256")
        self.register_buffer("center", center_tensor.clone())
        self.register_buffer("scale", scale_tensor.clone())
        self.dataset_sha256 = dataset_sha256

    @classmethod
    def from_offline_observations(
        cls, observations: np.ndarray, *, dataset_sha256: str
    ) -> "FrozenObservationNormalizer":
        observations = np.asarray(observations)
        if observations.ndim != 2 or observations.shape[1] != 5:
            raise ValueError("Offline observations must have shape [N,5]")
        if observations.shape[0] < 1 or not np.isfinite(observations).all():
            raise ValueError("Offline observations must be non-empty and finite")
        # Accumulate in float64 for a stable, dataset-independent artifact;
        # training/evaluation buffers are stored in float32 like the networks.
        center = observations.astype(np.float64).mean(axis=0)
        scale = observations.astype(np.float64).std(axis=0)
        scale = np.maximum(scale, 1e-6)
        return cls(center, scale, dataset_sha256=dataset_sha256)

    @property
    def observation_dim(self) -> int:
        return int(self.center.numel())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self.center) / self.scale

    def identity(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.KIND,
            "source": "offline_dataset_observation_only",
            "dataset_sha256": self.dataset_sha256,
            "estimator": "population_mean_std_float64_then_float32_v1",
            "minimum_scale": 1e-6,
            "center": self.center.detach().cpu().tolist(),
            "scale": self.scale.detach().cpu().tolist(),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def atanh_clipped(value: torch.Tensor) -> torch.Tensor:
    return torch.atanh(value.clamp(-0.999, 0.999))


def tanh_normal_sample(
    location: torch.Tensor,
    log_std: torch.Tensor,
    *,
    deterministic: bool,
    sample_shape: tuple[int, ...] = (),
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterized tanh-Normal action and corrected log probability."""

    log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
    expanded_location = location.expand(*sample_shape, *location.shape)
    expanded_log_std = log_std.expand_as(expanded_location)
    if deterministic:
        pre_tanh = expanded_location
    else:
        noise = torch.randn(
            expanded_location.shape,
            dtype=expanded_location.dtype,
            device=expanded_location.device,
            generator=generator,
        )
        pre_tanh = expanded_location + expanded_log_std.exp() * noise
    action = torch.tanh(pre_tanh)
    normal_log_prob = -0.5 * (
        ((pre_tanh - expanded_location) / expanded_log_std.exp()).square()
        + 2.0 * expanded_log_std
        + math.log(2.0 * math.pi)
    )
    correction = 2.0 * (
        math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh)
    )
    log_prob = (normal_log_prob - correction).sum(dim=-1)
    return action, log_prob


class MLPActor(nn.Module):
    """Tanh Gaussian actor over an explicitly selected state representation."""

    def __init__(
        self,
        state_dim: int | None = None,
        action_dim: int = 1,
        hidden_dim: int = 256,
        *,
        lifted_dim: int | None = None,
    ):
        super().__init__()
        # ``lifted_dim`` is retained solely as a construction alias for older
        # unit fixtures.  Method identity, not this generic module, enforces
        # whether the tensor represents raw observations or Koopman features.
        if state_dim is None:
            state_dim = lifted_dim
        elif lifted_dim is not None and lifted_dim != state_dim:
            raise ValueError("state_dim and lifted_dim aliases disagree")
        if state_dim is None:
            raise ValueError("MLPActor requires state_dim")
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )
        self.action_dim = action_dim
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.uniform_(self.net[-1].bias, -1e-3, 1e-3)

    def distribution(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        location, raw_log_std = self.net(state).chunk(2, dim=-1)
        log_std = raw_log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return location, log_std

    def sample(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
        samples: int = 1,
        return_plan: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        location, log_std = self.distribution(state)
        sample_shape = () if samples == 1 else (samples,)
        action, log_prob = tanh_normal_sample(
            location,
            log_std,
            deterministic=deterministic,
            sample_shape=sample_shape,
            generator=generator,
        )
        return action, log_prob, None


def _orthogonal_linear(module: nn.Module) -> None:
    """Match ExORL's ``utils.weight_init`` for fully connected layers."""

    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ExORLCQLActor(nn.Module):
    """Task-matched ExORL CQL MLP with the project's SAC distribution contract.

    ExORL applies ``tanh`` to the Gaussian location and then applies a second
    tanh through ``SquashedNormal``.  We retain its layer layout and orthogonal
    initialization, but deliberately use one standard tanh-Gaussian transform;
    this avoids the double-tanh compatibility quirk while preserving the
    Cal-QL/ExORL loss semantics.
    """

    DISTRIBUTION_PROFILE = "standard_single_tanh_gaussian_exorl_compat_v1"

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )
        self.action_dim = action_dim
        self.apply(_orthogonal_linear)

    def distribution(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        location, raw_log_std = self.policy(state).chunk(2, dim=-1)
        return location, raw_log_std.clamp(EXORL_CQL_LOG_STD_MIN, LOG_STD_MAX)

    def sample(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
        samples: int = 1,
        return_plan: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        del return_plan
        location, log_std = self.distribution(state)
        sample_shape = () if samples == 1 else (samples,)
        action, log_prob = tanh_normal_sample(
            location,
            log_std,
            deterministic=deterministic,
            sample_shape=sample_shape,
            generator=generator,
        )
        return action, log_prob, None


def _condense_dynamics(
    koopman: FrozenKoopman, horizon: int
) -> tuple[torch.Tensor, torch.Tensor]:
    lifted_dim, action_dim = koopman.B.shape
    lifted_power = torch.eye(lifted_dim, device=koopman.A.device)
    lifted_action = torch.zeros(
        lifted_dim, horizon * action_dim, device=koopman.A.device
    )
    state_rows = []
    action_rows = []
    for step in range(horizon):
        lifted_power = koopman.A @ lifted_power
        lifted_action = koopman.A @ lifted_action
        lifted_action[:, step * action_dim : (step + 1) * action_dim] += koopman.B
        state_rows.append(koopman.C @ lifted_power)
        action_rows.append(koopman.C @ lifted_action)
    return torch.cat(state_rows, dim=0), torch.cat(action_rows, dim=0)


class KMPCTanhGaussianActor(nn.Module):
    """Differentiable diagonal-cost MPC mean with a learned exploration scale."""

    def __init__(
        self,
        koopman: FrozenKoopman,
        *,
        horizon: int = 20,
        solver_iterations: int = 20,
        hidden_dim: int = 128,
        log_std_min: float = LOG_STD_MIN,
    ) -> None:
        super().__init__()
        self.koopman = koopman
        self.horizon = int(horizon)
        self.solver_iterations = int(solver_iterations)
        self.log_std_min = float(log_std_min)
        if not math.isfinite(self.log_std_min) or self.log_std_min >= LOG_STD_MAX:
            raise ValueError("KMPC log-std lower bound is invalid")
        physical_dim = koopman.state_dim
        action_dim = koopman.action_dim
        output_dim = 2 * horizon * (physical_dim + action_dim)
        self.controller = nn.Sequential(
            nn.Linear(koopman.lifted_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Zero cost head is the successful, neutral controller initialization.
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        state_map, action_map = _condense_dynamics(koopman, horizon)
        self.register_buffer("state_map", state_map)
        self.register_buffer("action_map", action_map)

    def plan(self, lifted_state: torch.Tensor) -> torch.Tensor:
        batch_shape = lifted_state.shape[:-1]
        physical_dim = self.koopman.state_dim
        action_dim = self.koopman.action_dim
        augmented_dim = physical_dim + action_dim
        raw = self.controller(lifted_state).reshape(
            *batch_shape, 2, self.horizon, augmented_dim
        )
        raw_quadratic = torch.tanh(raw[..., 0, :, :])
        centered = raw_quadratic - raw_quadratic.mean(dim=-1, keepdim=True)
        quadratic = torch.exp(1.5 * centered)
        linear = 10.0 * torch.tanh(raw[..., 1, :, :])
        free_physical = F.linear(lifted_state, self.state_map)
        q_state = quadratic[..., :physical_dim].reshape(
            *batch_shape, self.horizon * physical_dim
        )
        q_action = quadratic[..., physical_dim:].reshape(
            *batch_shape, self.horizon * action_dim
        )
        p_state = linear[..., :physical_dim].reshape(
            *batch_shape, self.horizon * physical_dim
        )
        p_action = linear[..., physical_dim:].reshape(
            *batch_shape, self.horizon * action_dim
        )
        weighted_map = self.action_map * q_state.unsqueeze(-1)
        hessian = self.action_map.T @ weighted_map
        hessian = hessian + torch.diag_embed(q_action)
        qp_linear = torch.einsum(
            "...p,pi->...i", q_state * free_physical + p_state, self.action_map
        ) + p_action
        lipschitz = hessian.abs().sum(dim=-1).amax(dim=-1)
        step_size = 0.95 / (lipschitz + 1e-6)
        current = torch.zeros_like(qp_linear)
        extrapolated = current
        momentum = 1.0
        for _ in range(self.solver_iterations):
            gradient = torch.einsum("...ij,...j->...i", hessian, extrapolated)
            following = (extrapolated - step_size.unsqueeze(-1) * (gradient + qp_linear)).clamp(
                -1.0, 1.0
            )
            next_momentum = 0.5 * (
                1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum)
            )
            extrapolated = following + ((momentum - 1.0) / next_momentum) * (
                following - current
            )
            current = following
            momentum = next_momentum
        return current.reshape(*batch_shape, self.horizon, action_dim)

    def distribution(
        self, lifted_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        plan = self.plan(lifted_state)
        location = atanh_clipped(plan[..., 0, :])
        log_std = self.log_std.clamp(self.log_std_min, LOG_STD_MAX).expand_as(location)
        return location, log_std, plan

    def sample(
        self,
        lifted_state: torch.Tensor,
        *,
        deterministic: bool = False,
        samples: int = 1,
        return_plan: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        location, log_std, plan = self.distribution(lifted_state)
        sample_shape = () if samples == 1 else (samples,)
        action, log_prob = tanh_normal_sample(
            location,
            log_std,
            deterministic=deterministic,
            sample_shape=sample_shape,
            generator=generator,
        )
        return action, log_prob, plan if return_plan else None


class EnsembleLinear(nn.Module):
    def __init__(self, ensemble: int, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(ensemble, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble, output_dim))
        for member in self.weight:
            nn.init.xavier_uniform_(member)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2:
            return torch.einsum("bi,eio->ebo", value, self.weight) + self.bias[:, None]
        if value.ndim == 3:
            return torch.einsum("ebi,eio->ebo", value, self.weight) + self.bias[:, None]
        raise ValueError("EnsembleLinear expects [B,D] or [E,B,D]")


class EnsembleLayerNorm(nn.Module):
    def __init__(self, ensemble: int, hidden_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ensemble, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(ensemble, hidden_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(value, (value.shape[-1],))
        return normalized * self.weight[:, None] + self.bias[:, None]


class QEnsemble(nn.Module):
    """Vectorized Q ensemble with RLPD-style per-head LayerNorm."""

    def __init__(
        self,
        state_dim: int | None = None,
        action_dim: int = 1,
        *,
        lifted_dim: int | None = None,
        ensemble_size: int = 10,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if state_dim is None:
            state_dim = lifted_dim
        elif lifted_dim is not None and lifted_dim != state_dim:
            raise ValueError("state_dim and lifted_dim aliases disagree")
        if state_dim is None:
            raise ValueError("QEnsemble requires state_dim")
        if hidden_layers < 1:
            raise ValueError("Q ensemble needs at least one hidden layer")
        self.ensemble_size = ensemble_size
        dimensions = [state_dim + action_dim] + [hidden_dim] * hidden_layers
        self.layers = nn.ModuleList(
            EnsembleLinear(ensemble_size, left, right)
            for left, right in zip(dimensions[:-1], dimensions[1:], strict=True)
        )
        self.norms = nn.ModuleList(
            EnsembleLayerNorm(ensemble_size, hidden_dim) for _ in range(hidden_layers)
        )
        self.output = EnsembleLinear(ensemble_size, hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat((state, action), dim=-1)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            value = F.relu(norm(layer(value)))
        return self.output(value)[..., 0]


class ExORLCQLQEnsemble(nn.Module):
    """Vectorized two-Q version of ExORL's DMC CQL critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        ensemble_size: int,
        hidden_dim: int,
        hidden_layers: int,
    ) -> None:
        super().__init__()
        if ensemble_size != 2:
            raise ValueError("The ExORL CQL critic profile requires exactly two Q heads")
        if hidden_layers != 2:
            raise ValueError("The ExORL CQL critic profile requires two hidden layers")
        self.ensemble_size = ensemble_size
        self.q_nets = nn.ModuleList(
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(ensemble_size)
        )
        self.apply(_orthogonal_linear)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        value = torch.cat((state, action), dim=-1)
        return torch.stack([network(value)[..., 0] for network in self.q_nets], dim=0)


def build_actor(
    method: str,
    koopman: FrozenKoopman | None,
    *,
    network_profile: str = "rlpd",
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    controller_hidden_dim: int,
    kmpc_horizon: int,
    kmpc_solver_iterations: int,
) -> nn.Module:
    if "AC-KMPC" in method or method == "Cal-QL-MPVE":
        if koopman is None:
            raise ValueError("AC-KMPC actor requires a Koopman model")
        return KMPCTanhGaussianActor(
            koopman,
            horizon=kmpc_horizon,
            solver_iterations=kmpc_solver_iterations,
            hidden_dim=controller_hidden_dim,
            log_std_min=(
                EXORL_CQL_LOG_STD_MIN
                if network_profile == "exorl_cql"
                else LOG_STD_MIN
            ),
        )
    if koopman is not None:
        raise ValueError("Raw MLP actor must not receive a Koopman model")
    if network_profile == "exorl_cql":
        return ExORLCQLActor(state_dim, action_dim, hidden_dim)
    if network_profile != "rlpd":
        raise ValueError(f"Unknown actor network profile {network_profile!r}")
    return MLPActor(state_dim, action_dim, hidden_dim)


def build_critic(
    *,
    network_profile: str,
    state_dim: int,
    action_dim: int,
    ensemble_size: int,
    hidden_dim: int,
    hidden_layers: int,
) -> nn.Module:
    if network_profile == "exorl_cql":
        return ExORLCQLQEnsemble(
            state_dim,
            action_dim,
            ensemble_size=ensemble_size,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
    if network_profile != "rlpd":
        raise ValueError(f"Unknown critic network profile {network_profile!r}")
    return QEnsemble(
        state_dim,
        action_dim,
        ensemble_size=ensemble_size,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
    )
