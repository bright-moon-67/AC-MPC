"""Cal-QL/RLPD-style learner with MLP or AC-KMPC stochastic actors."""

from __future__ import annotations

import copy
import hashlib
import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.dmc.o2o.config import O2OConfig
from experiments.dmc.o2o.koopman import FrozenKoopman
from experiments.dmc.o2o.networks import QEnsemble, build_actor
from experiments.dmc.reward_oracle import cartpole_swingup_official_reward


RNG_SUBSTREAM_VERSION = "o2o_torch_rng_substreams_v1"
_RNG_SUBSTREAM_NAMES = ("actor_init", "critic_init", "training_sampling")


def _substream_seed(base_seed: int, name: str) -> int:
    """Derive a stable, method-independent Torch seed for one RNG purpose."""

    if name not in _RNG_SUBSTREAM_NAMES:
        raise ValueError(f"Unknown RNG substream {name!r}")
    payload = f"{RNG_SUBSTREAM_VERSION}:{int(base_seed)}:{name}".encode("utf-8")
    # Torch accepts a signed-64-bit seed.  A cryptographic derivation avoids
    # accidental overlap without relying on Python's process-randomized hash.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


@contextmanager
def _cpu_initialization_stream(seed: int) -> Iterator[None]:
    """Temporarily select a CPU initialization stream without global leakage.

    Modules are constructed on CPU and moved to the requested learner device
    afterwards, so only the CPU default generator is involved in initialization.
    Restoring it here means actor architecture cannot perturb critic weights or
    the caller's global Torch RNG state.
    """

    previous = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(previous)


def _optimizer(parameters: Any, learning_rate: float, clip_norm: float):
    # Adam is deliberately exposed through a tiny helper so checkpoint loading
    # never silently changes its hyperparameters.
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    optimizer._acmpc_clip_norm = float(clip_norm)  # type: ignore[attr-defined]
    return optimizer


def _clip(parameters: Any, optimizer: torch.optim.Optimizer) -> float:
    value = nn.utils.clip_grad_norm_(
        parameters, float(optimizer._acmpc_clip_norm)  # type: ignore[attr-defined]
    )
    return float(value.detach())


def _optimizer_to(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    """Move restored Adam moments to the learner device.

    Checkpoints are deliberately loaded through CPU for portability.  Module
    ``load_state_dict`` handles the model device, whereas optimizer state needs
    this explicit migration before a CUDA resume can take its next step.
    """

    if any(
        parameter.device != device
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        raise ValueError("Optimizer parameters do not live on the learner device")
    for group in optimizer.param_groups:
        scalar_step_on_parameter = bool(
            group.get("capturable", False) or group.get("fused", False)
        )
        for parameter in group["params"]:
            for key, value in optimizer.state.get(parameter, {}).items():
                if not isinstance(value, torch.Tensor):
                    continue
                # Adam deliberately keeps its scalar step counter on CPU when
                # it is neither fused nor capturable.  Moving that tensor to
                # CUDA would undo PyTorch's checkpoint-loading policy.
                target_device = (
                    parameter.device
                    if key != "step" or scalar_step_on_parameter
                    else torch.device("cpu")
                )
                optimizer.state[parameter][key] = value.to(device=target_device)


@contextmanager
def _frozen_parameters(module: nn.Module) -> Iterator[None]:
    """Freeze module weights while preserving gradients through its inputs."""

    parameters = tuple(module.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad, strict=True):
            parameter.requires_grad_(enabled)


@dataclass
class TensorBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    discount: torch.Tensor
    next_observation: torch.Tensor
    mc_return: torch.Tensor
    offline_mask: torch.Tensor

    @classmethod
    def from_numpy(cls, batch: dict[str, np.ndarray], device: torch.device):
        return cls(
            **{
                key: torch.as_tensor(batch[key], dtype=torch.float32, device=device)
                for key in cls.__dataclass_fields__
            }
        )

    def slice(self, index: slice) -> "TensorBatch":
        return TensorBatch(
            **{key: getattr(self, key)[index] for key in self.__dataclass_fields__}
        )


@dataclass(frozen=True)
class CriticProposalCache:
    """Frozen actor proposals for one fused UTD batch.

    Only states, actions, and action log-probabilities are cached.  Target-Q
    and current-Q values must be evaluated again after every critic update.
    """

    lifted: torch.Tensor
    next_lifted: torch.Tensor
    target_next_action: torch.Tensor
    target_next_log_prob: torch.Tensor
    cql_current_actions: torch.Tensor | None = None
    cql_current_log_prob: torch.Tensor | None = None
    cql_next_actions: torch.Tensor | None = None
    cql_next_log_prob: torch.Tensor | None = None

    def slice(self, index: slice) -> "CriticProposalCache":
        def slice_samples(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value[:, index]

        return CriticProposalCache(
            lifted=self.lifted[index],
            next_lifted=self.next_lifted[index],
            target_next_action=self.target_next_action[index],
            target_next_log_prob=self.target_next_log_prob[index],
            cql_current_actions=slice_samples(self.cql_current_actions),
            cql_current_log_prob=slice_samples(self.cql_current_log_prob),
            cql_next_actions=slice_samples(self.cql_next_actions),
            cql_next_log_prob=slice_samples(self.cql_next_log_prob),
        )


class O2OLearner:
    """Shared off-policy Q core; only the actor and optional MPVE term differ."""

    def __init__(
        self,
        config: O2OConfig,
        koopman: FrozenKoopman,
        device: torch.device,
    ) -> None:
        config.validate()
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            # Keep optimizer-resume validation and the private CUDA generator
            # on the same explicit device as parameters created via ``.to``.
            device = torch.device("cuda", torch.cuda.current_device())
        self.config = config
        self.device = device
        self.koopman = koopman.to(device).eval()
        self.rng_substream_seeds = {
            name: _substream_seed(config.seed, name)
            for name in _RNG_SUBSTREAM_NAMES
        }
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        self.training_generator = torch.Generator(device=generator_device)
        self.training_generator.manual_seed(
            self.rng_substream_seeds["training_sampling"]
        )

        # Actor and critic initialization use independent, method-invariant
        # streams.  In particular, changing from MLP to KMPC cannot change the
        # shared critic ensemble merely by consuming a different number of
        # default-generator draws while constructing the actor.
        with _cpu_initialization_stream(self.rng_substream_seeds["actor_init"]):
            self.actor = build_actor(
                config.method,
                self.koopman,
                hidden_dim=config.hidden_dim,
                controller_hidden_dim=config.controller_hidden_dim,
                kmpc_horizon=config.kmpc_horizon,
                kmpc_solver_iterations=config.kmpc_solver_iterations,
            ).to(device)
        with _cpu_initialization_stream(self.rng_substream_seeds["critic_init"]):
            self.critic = QEnsemble(
                koopman.lifted_dim,
                koopman.action_dim,
                ensemble_size=config.critic_ensemble_size,
                hidden_dim=config.hidden_dim,
                hidden_layers=config.critic_hidden_layers,
            ).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device).eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(config.initial_temperature), device=device)
        )
        self.actor_optimizer = _optimizer(
            self.actor.parameters(), config.actor_learning_rate, config.gradient_clip_norm
        )
        self.critic_optimizer = _optimizer(
            self.critic.parameters(),
            config.critic_learning_rate,
            config.gradient_clip_norm,
        )
        self.temperature_optimizer = _optimizer(
            [self.log_temperature],
            config.temperature_learning_rate,
            config.gradient_clip_norm,
        )
        self.gradient_updates = 0
        self.actor_updates = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        lifted = self.koopman.lift(tensor)
        action, _, _ = self.actor.sample(
            lifted,
            deterministic=deterministic,
            generator=self.training_generator,
        )
        return action.cpu().numpy()

    @torch.no_grad()
    def _prepare_critic_cache(self, batch: TensorBatch) -> CriticProposalCache:
        """Sample fixed proposals once while the actor is fixed across UTD."""

        lifted = self.koopman.lift(batch.observation)
        next_lifted = self.koopman.lift(batch.next_observation)
        if self.config.uses_calql:
            sample_count = self.config.cql_actions
            current_actions, current_log_prob, _ = self.actor.sample(
                lifted,
                samples=sample_count,
                generator=self.training_generator,
            )
            # Preserve the independent target-vs-CQL proposal semantics of
            # separate actor calls while sharing the expensive KMPC plan.
            # Sample zero is target-only; the remaining K samples are CQL-only.
            next_actions, next_log_prob, _ = self.actor.sample(
                next_lifted,
                samples=sample_count + 1,
                generator=self.training_generator,
            )
            # ``actor.sample(samples=1)`` intentionally returns the ordinary
            # [B,A]/[B] shape.  Canonicalize it so cache slicing is uniform.
            if sample_count == 1:
                current_actions = current_actions.unsqueeze(0)
                current_log_prob = current_log_prob.unsqueeze(0)
            return CriticProposalCache(
                lifted=lifted.detach(),
                next_lifted=next_lifted.detach(),
                target_next_action=next_actions[0].detach(),
                target_next_log_prob=next_log_prob[0].detach(),
                cql_current_actions=current_actions.detach(),
                cql_current_log_prob=current_log_prob.detach(),
                cql_next_actions=next_actions[1:].detach(),
                cql_next_log_prob=next_log_prob[1:].detach(),
            )

        next_action, next_log_prob, _ = self.actor.sample(
            next_lifted,
            generator=self.training_generator,
        )
        return CriticProposalCache(
            lifted=lifted.detach(),
            next_lifted=next_lifted.detach(),
            target_next_action=next_action.detach(),
            target_next_log_prob=next_log_prob.detach(),
        )

    def _target_q(
        self, batch: TensorBatch, cache: CriticProposalCache
    ) -> torch.Tensor:
        with torch.no_grad():
            # Target critic parameters evolve after every REDQ update, so only
            # the fixed actor proposal is reused here; Q is always recomputed.
            next_q = self._minimum_target_q(
                cache.next_lifted, cache.target_next_action
            )
            next_q = (
                next_q
                - self.temperature.detach() * cache.target_next_log_prob
            )
            return batch.reward + self.config.discount * batch.discount * next_q

    def _minimum_target_q(
        self, lifted_state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """REDQ target: minimum over one freshly sampled critic subset."""

        target_heads = self.target_critic(lifted_state, action)
        if self.config.target_critic_subset < target_heads.shape[0]:
            choice = torch.randperm(
                target_heads.shape[0],
                device=target_heads.device,
                generator=self.training_generator,
            )[: self.config.target_critic_subset]
            target_heads = target_heads[choice]
        return target_heads.amin(dim=0)

    def _cql_calibrated_penalty(
        self,
        batch: TensorBatch,
        cache: CriticProposalCache,
        data_q: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        offline = batch.offline_mask > 0.5
        if not self.config.uses_calql or not torch.any(offline):
            zero = data_q.sum() * 0.0
            return zero, {"cql_penalty": 0.0, "calibration_bound_rate": 0.0}
        lifted_offline = cache.lifted[offline]
        batch_size = lifted_offline.shape[0]
        sample_count = self.config.cql_actions
        expanded_lifted = lifted_offline.unsqueeze(0).expand(sample_count, -1, -1)

        random_actions = torch.empty(
            sample_count, batch_size, self.koopman.action_dim, device=self.device
        ).uniform_(-1.0, 1.0, generator=self.training_generator)
        proposals = (
            cache.cql_current_actions,
            cache.cql_current_log_prob,
            cache.cql_next_actions,
            cache.cql_next_log_prob,
        )
        if any(value is None for value in proposals):
            raise RuntimeError("Cal-QL critic cache is missing policy proposals")
        current_actions = cache.cql_current_actions[:, offline]
        current_log_prob = cache.cql_current_log_prob[:, offline]
        next_actions = cache.cql_next_actions[:, offline]
        next_log_prob = cache.cql_next_log_prob[:, offline]

        def evaluate(actions: torch.Tensor) -> torch.Tensor:
            flat_lifted = expanded_lifted.reshape(-1, cache.lifted.shape[-1])
            flat_action = actions.reshape(-1, actions.shape[-1])
            values = self.critic(flat_lifted, flat_action)
            return values.reshape(values.shape[0], sample_count, batch_size)

        q_random = evaluate(random_actions)
        q_current = evaluate(current_actions)
        q_next = evaluate(next_actions)
        lower_bound = batch.mc_return[offline].view(1, 1, batch_size)
        bound_rate = 0.5 * (
            (q_current < lower_bound).float().mean()
            + (q_next < lower_bound).float().mean()
        )
        # Cal-QL modifies only the OOD/policy push-down side of CQL.  Q(data)
        # and the Bellman target are never clamped.
        q_current = torch.maximum(q_current, lower_bound)
        q_next = torch.maximum(q_next, lower_bound)
        random_density = -self.koopman.action_dim * math.log(2.0)
        candidates = torch.cat(
            (
                q_random - random_density,
                q_current - current_log_prob.unsqueeze(0),
                q_next - next_log_prob.unsqueeze(0),
            ),
            dim=1,
        )
        ood = torch.logsumexp(
            candidates / self.config.cql_temperature, dim=1
        ) * self.config.cql_temperature
        penalty = (ood - data_q[:, offline]).mean()
        return penalty, {
            "cql_penalty": float(penalty.detach()),
            "calibration_bound_rate": float(bound_rate.detach()),
        }

    def _mpve_target(
        self,
        batch: TensorBatch,
        real_target: torch.Tensor,
        *,
        next_lifted: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Total H=10 target: one real transition followed by nine model steps."""

        if not self.config.uses_mpve:
            return real_target.detach()
        with torch.no_grad():
            current = (
                self.koopman.lift(batch.next_observation)
                if next_lifted is None
                else next_lifted
            )
            total = batch.reward.clone()
            continuation = self.config.discount * batch.discount
            # The real transition above is step one.  Add H-1 imagined rewards.
            for _ in range(1, self.config.mpve_total_horizon):
                action, log_prob, _ = self.actor.sample(
                    current,
                    generator=self.training_generator,
                )
                following = self.koopman.step(current, action)
                normalized_following = self.koopman.reconstruct_normalized(following)
                reward = cartpole_swingup_official_reward(
                    self.koopman.denormalize(normalized_following), action
                )
                total = total + continuation * (
                    reward - self.temperature.detach() * log_prob
                )
                continuation = continuation * self.config.discount
                current = following
            terminal_action, terminal_log_prob, _ = self.actor.sample(
                current,
                generator=self.training_generator,
            )
            terminal_q = self._minimum_target_q(current, terminal_action)
            terminal_q = terminal_q - self.temperature.detach() * terminal_log_prob
            return (total + continuation * terminal_q).detach()

    def update_critic(
        self,
        batch: TensorBatch,
        *,
        apply_mpve: bool,
        cache: CriticProposalCache | None = None,
    ) -> dict[str, float]:
        if apply_mpve and not self.config.uses_mpve:
            raise ValueError("MPVE auxiliary requested for a non-MPVE method")
        if cache is None:
            cache = self._prepare_critic_cache(batch)
        lifted = cache.lifted
        target = self._target_q(batch, cache)
        q = self.critic(lifted, batch.action)
        bellman_loss = (q - target.unsqueeze(0)).square().mean()
        cql_penalty, cql_metrics = self._cql_calibrated_penalty(batch, cache, q)
        loss = bellman_loss + self.config.cql_weight * cql_penalty
        mpve_loss = q.sum() * 0.0
        mpve_target_mean = 0.0
        if apply_mpve:
            model_target = self._mpve_target(
                batch,
                target,
                next_lifted=cache.next_lifted,
            )
            mpve_loss = (
                q - model_target.unsqueeze(0)
            ).square().mean()
            loss = loss + self.config.mpve_loss_weight * mpve_loss
            mpve_target_mean = float(model_target.mean())
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = _clip(self.critic.parameters(), self.critic_optimizer)
        self.critic_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, self.config.target_tau)
        self.gradient_updates += 1
        return {
            "critic_loss": float(loss.detach()),
            "bellman_loss": float(bellman_loss.detach()),
            "critic_grad_norm": grad_norm,
            "q_mean": float(q.detach().mean()),
            "target_q_mean": float(target.mean()),
            "mpve_applied": float(apply_mpve),
            "mpve_loss": float(mpve_loss.detach()),
            "mpve_target_mean": mpve_target_mean,
            **cql_metrics,
        }

    def update_actor_and_temperature(
        self,
        batch: TensorBatch,
        *,
        lifted: torch.Tensor | None = None,
    ) -> dict[str, float]:
        lifted = (
            self.koopman.lift(batch.observation).detach()
            if lifted is None
            else lifted
        )
        action, log_prob, _ = self.actor.sample(
            lifted,
            generator=self.training_generator,
        )
        # The SAC policy needs dQ/da, but the actor step must neither calculate
        # nor retain gradients for critic parameters.
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.actor_optimizer.zero_grad(set_to_none=True)
        with _frozen_parameters(self.critic):
            q = self.critic(lifted, action).mean(dim=0)
            actor_loss = (self.temperature.detach() * log_prob - q).mean()
            actor_loss.backward()
        actor_grad_norm = _clip(self.actor.parameters(), self.actor_optimizer)
        self.actor_optimizer.step()

        # Match RLPD's Temperature module exactly: optimize alpha=exp(log_alpha)
        # against entropy-target_entropy, rather than optimizing log_alpha
        # directly.  The two gradients agree only at alpha == 1.
        entropy = (-log_prob).detach()
        temperature_loss = (
            self.temperature * (entropy - self.config.target_entropy)
        ).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.actor_updates += 1
        return {
            "actor_loss": float(actor_loss.detach()),
            "actor_grad_norm": actor_grad_norm,
            "entropy": float((-log_prob).detach().mean()),
            "temperature": float(self.temperature.detach()),
            "temperature_loss": float(temperature_loss.detach()),
        }

    def update(
        self,
        batch: TensorBatch,
        utd: int,
        *,
        phase: Literal["offline", "online"],
    ) -> dict[str, float]:
        if phase not in ("offline", "online"):
            raise ValueError("phase must be exactly 'offline' or 'online'")
        if batch.reward.shape[0] % utd:
            raise ValueError("Fused batch must divide evenly by UTD")
        size = batch.reward.shape[0] // utd
        cache = self._prepare_critic_cache(batch)
        metrics: dict[str, float] = {}
        mini_batch = batch
        for index in range(utd):
            batch_slice = slice(index * size, (index + 1) * size)
            mini_batch = batch.slice(batch_slice)
            mini_cache = cache.slice(batch_slice)
            # MPVE is an online-only auxiliary and runs once per real
            # environment step, independent of REDQ's critic UTD.  The first
            # UTD-1 minibatches remain pure real-transition TD updates.
            apply_mpve = (
                self.config.uses_mpve
                and phase == "online"
                and index + 1 == utd
            )
            metrics = self.update_critic(
                mini_batch,
                apply_mpve=apply_mpve,
                cache=mini_cache,
            )
        metrics.update(
            self.update_actor_and_temperature(
                mini_batch,
                lifted=mini_cache.lifted,
            )
        )
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "gradient_updates": self.gradient_updates,
            "actor_updates": self.actor_updates,
            # This is part of the scientific resume contract: initialization
            # seeds document cross-method pairing, while the private sampling
            # state makes the next stochastic update exactly reproducible.
            "rng_substreams": {
                "version": RNG_SUBSTREAM_VERSION,
                "base_seed": int(self.config.seed),
                "seeds": dict(self.rng_substream_seeds),
                "training_sampling_device": str(self.device),
                "training_sampling_state": self.training_generator.get_state().cpu(),
            },
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        restore_sampling_rng: bool = True,
    ) -> None:
        """Restore learner state.

        Training resume/fork keeps the default and restores the private
        generator exactly on the configured device.  Deterministic evaluation
        may explicitly skip that device-specific CUDA/CPU generator state;
        it never draws policy noise, while model parameters remain identical.
        """
        rng_state = state.get("rng_substreams")
        if not isinstance(rng_state, dict):
            raise ValueError("Learner checkpoint is missing RNG substream state")
        expected_identity = {
            "version": RNG_SUBSTREAM_VERSION,
            "base_seed": int(self.config.seed),
            "seeds": self.rng_substream_seeds,
        }
        actual_identity = {
            key: rng_state.get(key) for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                "Learner checkpoint RNG substreams do not match this configuration"
            )
        training_sampling_state = rng_state.get("training_sampling_state")
        if not isinstance(training_sampling_state, torch.Tensor):
            raise ValueError("Learner checkpoint has no Torch sampling-generator state")
        sampling_device = rng_state.get("training_sampling_device")
        if not isinstance(sampling_device, str) or not sampling_device:
            raise ValueError("Learner checkpoint has no sampling-generator device")
        if restore_sampling_rng and sampling_device != str(self.device):
            raise ValueError(
                "Learner sampling RNG device differs; training resume must use "
                "the checkpoint device"
            )
        self.actor.load_state_dict(state["actor"], strict=True)
        self.critic.load_state_dict(state["critic"], strict=True)
        self.target_critic.load_state_dict(state["target_critic"], strict=True)
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"].to(self.device))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        _optimizer_to(self.actor_optimizer, self.device)
        _optimizer_to(self.critic_optimizer, self.device)
        _optimizer_to(self.temperature_optimizer, self.device)
        self.target_critic.eval()
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.gradient_updates = int(state["gradient_updates"])
        self.actor_updates = int(state["actor_updates"])
        if restore_sampling_rng:
            try:
                self.training_generator.set_state(training_sampling_state.cpu())
            except RuntimeError as exc:
                raise ValueError(
                    "Learner sampling RNG is incompatible with the restore device; "
                    "training resume must use the checkpoint device"
                ) from exc
