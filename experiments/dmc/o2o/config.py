"""Resolved configuration for Cartpole offline-to-online experiments."""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any


METHODS = (
    "REDQ-Online",
    "RLPD-MLP",
    "Cal-RLPD-MLP",
    "Cal-RLPD-AC-KMPC",
    "Cal-RLPD-AC-KMPC-MPVE",
)


@dataclass(frozen=True)
class O2OConfig:
    task: str = "cartpole_swingup"
    method: str = "Cal-RLPD-MLP"
    seed: int = 20260821
    device: str = "cuda"

    # Shared SAC / REDQ / RLPD learner.
    batch_size: int = 256
    hidden_dim: int = 256
    critic_hidden_layers: int = 2
    critic_ensemble_size: int = 10
    target_critic_subset: int = 2
    discount: float = 0.99
    target_tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    initial_temperature: float = 1.0
    target_entropy: float = -0.5
    gradient_clip_norm: float = 10.0

    # Offline Cal-QL phase.  500k matches the public ExORL training budget.
    offline_updates: int = 500_000
    cql_actions: int = 10
    cql_temperature: float = 1.0
    cql_weight: float = 0.01

    # Online RLPD phase.  Every real transition triggers one fused update call
    # containing ``online_utd`` critic minibatches and one actor update.
    online_steps: int = 100_000
    online_utd: int = 20
    offline_replay_ratio: float = 0.5
    online_warmup_steps: int = 5_000
    replay_capacity: int = 200_000
    # Five CPU simulators keep the 5k evaluation cadence aligned with the
    # 1000-control-step Cartpole episode boundary.  This is execution-only
    # parallelism: ``online_steps`` still counts individual transitions.
    num_envs: int = 5
    env_workers: int = 5

    # Structured actor and model-predictive value expansion.
    kmpc_horizon: int = 20
    kmpc_solver_iterations: int = 20
    controller_hidden_dim: int = 128
    mpve_total_horizon: int = 10
    mpve_loss_weight: float = 1.0

    eval_interval_online_steps: int = 5_000
    eval_episodes: int = 10
    checkpoint_interval_updates: int = 10_000
    log_interval_updates: int = 1_000

    def validate(self) -> None:
        if self.task != "cartpole_swingup":
            raise ValueError("The first O2O protocol is frozen to cartpole_swingup")
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected {METHODS}")
        integer_fields = (
            "batch_size",
            "hidden_dim",
            "critic_hidden_layers",
            "critic_ensemble_size",
            "target_critic_subset",
            "offline_updates",
            "cql_actions",
            "online_steps",
            "online_utd",
            "online_warmup_steps",
            "replay_capacity",
            "num_envs",
            "env_workers",
            "kmpc_horizon",
            "kmpc_solver_iterations",
            "controller_hidden_dim",
            "mpve_total_horizon",
            "eval_interval_online_steps",
            "eval_episodes",
            "checkpoint_interval_updates",
            "log_interval_updates",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.target_critic_subset > self.critic_ensemble_size:
            raise ValueError("target_critic_subset cannot exceed critic ensemble size")
        if self.env_workers > self.num_envs:
            raise ValueError("env_workers cannot exceed num_envs")
        for name in (
            "online_steps",
            "online_warmup_steps",
            "eval_interval_online_steps",
        ):
            if getattr(self, name) % self.num_envs:
                raise ValueError(f"{name} must be divisible by num_envs")
        if self.mpve_total_horizon > self.kmpc_horizon:
            raise ValueError("MPVE total horizon cannot exceed KMPC horizon")
        finite_positive = (
            "discount",
            "target_tau",
            "actor_learning_rate",
            "critic_learning_rate",
            "temperature_learning_rate",
            "initial_temperature",
            "gradient_clip_norm",
            "cql_temperature",
            "mpve_loss_weight",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must lie in (0, 1]")
        if not 0 < self.target_tau <= 1:
            raise ValueError("target_tau must lie in (0, 1]")
        if not 0 <= self.offline_replay_ratio <= 1:
            raise ValueError("offline_replay_ratio must lie in [0, 1]")
        if not math.isfinite(self.cql_weight) or self.cql_weight < 0:
            raise ValueError("cql_weight must be finite and nonnegative")
        if not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")

    @property
    def uses_offline_pretraining(self) -> bool:
        return self.method.startswith("Cal-RLPD-")

    @property
    def requires_own_offline_pretraining(self) -> bool:
        # The MPVE ablation forks the completed AC-KMPC offline checkpoint so
        # both structured methods enter online learning with bit-identical
        # actor/critic/temperature/optimizer state.
        return self.uses_offline_pretraining and not self.uses_mpve

    @property
    def uses_offline_replay_online(self) -> bool:
        return self.method != "REDQ-Online"

    @property
    def uses_calql(self) -> bool:
        return self.method.startswith("Cal-RLPD-")

    @property
    def uses_kmpc(self) -> bool:
        return "AC-KMPC" in self.method

    @property
    def uses_mpve(self) -> bool:
        return self.method == "Cal-RLPD-AC-KMPC-MPVE"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dataclasses.asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return sha256(payload).hexdigest()
