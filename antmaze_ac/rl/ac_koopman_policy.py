from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Independent, Normal

from antmaze_ac.control.quadratic_cost import physical_to_lifted_cost
from antmaze_ac.control.differentiable_dare import (
    DAREResult,
    detectability_diagnostic,
    stabilizability_diagnostic,
)
from antmaze_ac.control.steady_state_lqr import AffineLQRResult, affine_lqr
from antmaze_ac.koopman.model import DeepKoopman

from .cost_actor import CostActor
from .critic import Critic


@dataclass
class PolicyOutput:
    distribution: Independent
    mean: torch.Tensor
    value: torch.Tensor
    stage_hessian_diag: torch.Tensor
    stage_linear: torch.Tensor
    lqr: AffineLQRResult
    solver_valid: torch.Tensor
    solver_retry_used: torch.Tensor
    solver_fallback_used: torch.Tensor


class KoopmanLQRPolicy(nn.Module):
    """Actor cost -> differentiable affine DARE-LQR -> Gaussian delta action."""

    def __init__(
        self,
        koopman: DeepKoopman,
        actor: CostActor,
        critic: Critic,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        *,
        log_std_init: float = -1.0,
        dare_tolerance: float = 1e-7,
        dare_max_iterations: int = 500,
        dare_jitter: float = 1e-9,
        fail_on_nonconvergence: bool = True,
        retry_max_iterations: int = 1000,
        retry_jitter_multiplier: float = 100.0,
        fallback_state_cost: float = 1.0,
        fallback_control_cost: float = 1.0,
        fallback_delta_limit: float = 1.0,
        mean_action_limit: float | None = None,
    ) -> None:
        super().__init__()
        if state_mean.shape != (koopman.state_dim,) or state_std.shape != (koopman.state_dim,):
            raise ValueError("State normalization shape does not match Koopman state dimension")
        self.koopman = koopman.freeze_dynamics()
        self.actor = actor
        self.critic = critic
        self.log_std = nn.Parameter(torch.full((koopman.action_dim,), float(log_std_init)))
        self.register_buffer("state_mean", state_mean.clone())
        self.register_buffer("state_std", state_std.clone().clamp_min(1e-6))
        failures = stabilizability_diagnostic(self.koopman.A, self.koopman.B)
        if failures:
            raise RuntimeError(
                "Frozen Koopman dynamics are not stabilizable: " + "; ".join(failures)
            )
        failures = detectability_diagnostic(self.koopman.A, self.koopman.C)
        if failures:
            raise RuntimeError(
                "Frozen Koopman physical readout cannot detect unstable modes: "
                + "; ".join(failures)
            )
        if retry_max_iterations < dare_max_iterations:
            raise ValueError("retry_max_iterations must be >= dare_max_iterations")
        if retry_jitter_multiplier < 1.0:
            raise ValueError("retry_jitter_multiplier must be >= 1")
        if fallback_state_cost <= 0 or fallback_control_cost <= 0:
            raise ValueError("Fallback LQR costs must be strictly positive")
        if fallback_delta_limit <= 0:
            raise ValueError("fallback_delta_limit must be positive")
        if mean_action_limit is not None and mean_action_limit <= 0:
            raise ValueError("mean_action_limit must be positive when provided")
        self.dare_kwargs = {
            "tolerance": dare_tolerance,
            "max_iterations": dare_max_iterations,
            "jitter": dare_jitter,
            # The invariant frozen (A,B) pair was checked once above.
            "check_stabilizable": False,
            # Q_z=C'Q_xC with strictly positive Q_x has the same observable
            # subspace as C, checked once above.
            "check_detectable": False,
            "fail_on_nonconvergence": fail_on_nonconvergence,
        }
        self.retry_dare_kwargs = {
            **self.dare_kwargs,
            "max_iterations": int(retry_max_iterations),
            "jitter": max(
                float(dare_jitter) * float(retry_jitter_multiplier),
                torch.finfo(torch.float64).eps,
            ),
            # A retry must return per-sample validity so invalid samples can
            # take the deterministic fallback instead of aborting PPO.
            "fail_on_nonconvergence": False,
        }
        self.fallback_delta_limit = float(fallback_delta_limit)
        # TD3+BC uses a deterministic, bounded delta action. PPO keeps this
        # unset so existing Gaussian-policy semantics and checkpoints are
        # unchanged. This is deliberately non-persistent: the TD3+BC
        # checkpoint records the limit in its runtime/config metadata.
        self.mean_action_limit = (
            None if mean_action_limit is None else float(mean_action_limit)
        )

        # The fallback is fixed and observation-reconstructable: unlike reusing
        # the last successful gain, it introduces no hidden policy state. Its
        # structural feasibility is checked once at construction time.
        lifted_dim = self.koopman.lifted_dim
        safe_q_x = torch.full(
            (self.koopman.state_dim,),
            float(fallback_state_cost),
            dtype=self.koopman.A.dtype,
            device=self.koopman.A.device,
        )
        safe_q = self.koopman.C.mT @ torch.diag(safe_q_x) @ self.koopman.C
        safe_r = (
            torch.eye(
                self.koopman.action_dim,
                dtype=self.koopman.A.dtype,
                device=self.koopman.A.device,
            )
            * float(fallback_control_cost)
        )
        safe_lqr = affine_lqr(
            self.koopman.A,
            self.koopman.B,
            safe_q,
            safe_r,
            torch.zeros(lifted_dim, dtype=safe_q.dtype, device=safe_q.device),
            torch.zeros(
                self.koopman.action_dim,
                dtype=safe_q.dtype,
                device=safe_q.device,
            ),
            **{
                **self.retry_dare_kwargs,
                "tolerance": max(float(dare_tolerance), 1e-10),
                "max_iterations": max(int(retry_max_iterations), 1000),
                "fail_on_nonconvergence": True,
            },
        )
        self.register_buffer("fallback_gain", safe_lqr.gain.detach().clone())
        self.register_buffer("fallback_value_hessian", safe_lqr.value_hessian.detach().clone())
        self.register_buffer(
            "fallback_closed_loop_spectral_radius",
            safe_lqr.dare.closed_loop_spectral_radius.detach().clone(),
        )
        self.register_buffer(
            "fallback_condition_number",
            safe_lqr.dare.condition_number.detach().clone(),
        )

    @staticmethod
    def _batch_valid(lqr: AffineLQRResult) -> torch.Tensor:
        converged = lqr.dare.converged
        converged = converged.unsqueeze(0) if converged.ndim == 0 else converged
        gain = lqr.gain.unsqueeze(0) if lqr.gain.ndim == 2 else lqr.gain
        feedforward = (
            lqr.feedforward.unsqueeze(0)
            if lqr.feedforward.ndim == 1
            else lqr.feedforward
        )
        return (
            converged
            & torch.isfinite(gain).flatten(1).all(dim=1)
            & torch.isfinite(feedforward).flatten(1).all(dim=1)
        )

    def _fallback_lqr(self, batch: int) -> AffineLQRResult:
        gain = self.fallback_gain.unsqueeze(0).expand(batch, -1, -1)
        value_hessian = self.fallback_value_hessian.unsqueeze(0).expand(batch, -1, -1)
        feedforward = torch.zeros(
            batch,
            self.koopman.action_dim,
            dtype=gain.dtype,
            device=gain.device,
        )
        value_linear_half = torch.zeros(
            batch,
            self.koopman.lifted_dim,
            dtype=gain.dtype,
            device=gain.device,
        )
        failed = torch.zeros(batch, dtype=torch.bool, device=gain.device)
        infinity = torch.full(
            (batch,),
            float("inf"),
            dtype=gain.dtype,
            device=gain.device,
        )
        dare = DAREResult(
            P=value_hessian,
            gain=gain,
            converged=failed,
            iterations=0,
            residual=infinity,
            relative_residual=infinity,
            condition_number=self.fallback_condition_number.expand(batch),
            closed_loop_spectral_radius=(
                self.fallback_closed_loop_spectral_radius.expand(batch)
            ),
        )
        return AffineLQRResult(
            gain=gain,
            feedforward=feedforward,
            value_hessian=value_hessian,
            value_linear_half=value_linear_half,
            dare=dare,
        )

    def _solve_with_recovery(self, cost) -> tuple[AffineLQRResult, torch.Tensor, torch.Tensor]:
        batch = cost.state_hessian.shape[0]
        primary: AffineLQRResult | None = None
        try:
            primary = affine_lqr(
                self.koopman.A,
                self.koopman.B,
                cost.state_hessian,
                cost.control_hessian,
                cost.state_linear,
                cost.control_linear,
                **self.dare_kwargs,
            )
        except (RuntimeError, FloatingPointError):
            pass
        if primary is not None and bool(torch.all(self._batch_valid(primary))):
            valid = self._batch_valid(primary)
            return primary, valid, torch.zeros_like(valid)

        retry_used = torch.ones(batch, dtype=torch.bool, device=cost.state_hessian.device)
        try:
            recovered = affine_lqr(
                self.koopman.A,
                self.koopman.B,
                cost.state_hessian,
                cost.control_hessian,
                cost.state_linear,
                cost.control_linear,
                **self.retry_dare_kwargs,
            )
        except (RuntimeError, FloatingPointError):
            fallback = self._fallback_lqr(batch)
            return fallback, torch.zeros_like(retry_used), retry_used

        valid = self._batch_valid(recovered)
        if bool(torch.all(valid)):
            return recovered, valid, retry_used

        fallback_gain = self.fallback_gain.unsqueeze(0).expand(batch, -1, -1)
        fallback_feedforward = torch.zeros_like(recovered.feedforward)
        gain = torch.where(valid[:, None, None], recovered.gain, fallback_gain)
        feedforward = torch.where(
            valid[:, None],
            recovered.feedforward,
            fallback_feedforward,
        )
        recovered = AffineLQRResult(
            gain=gain,
            feedforward=feedforward,
            value_hessian=recovered.value_hessian,
            value_linear_half=recovered.value_linear_half,
            dare=recovered.dare,
        )
        return recovered, valid, retry_used

    def normalize(self, observation: torch.Tensor) -> torch.Tensor:
        return (observation - self.state_mean) / self.state_std

    def transform_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """Apply the optional smooth TD3+BC delta-action support bound."""

        if self.mean_action_limit is None:
            return mean
        limit = self.mean_action_limit
        return limit * torch.tanh(mean / limit)

    def forward(self, observation: torch.Tensor) -> PolicyOutput:
        single = observation.ndim == 1
        observation = observation.unsqueeze(0) if single else observation
        normalized = self.normalize(observation)
        stage_hessian_diag, stage_linear = self.actor(normalized)
        if not (
            torch.isfinite(stage_hessian_diag).all()
            and torch.isfinite(stage_linear).all()
        ):
            raise FloatingPointError("Cost actor produced NaN or Inf")
        cost = physical_to_lifted_cost(
            self.koopman.C,
            stage_hessian_diag,
            stage_linear,
            self.koopman.state_dim,
        )
        lqr, solver_valid, solver_retry_used = self._solve_with_recovery(cost)
        solver_fallback_used = ~solver_valid
        lifted = self.koopman.lift(normalized)
        gain = lqr.gain.unsqueeze(0) if lqr.gain.ndim == 2 else lqr.gain
        feedforward = lqr.feedforward.unsqueeze(0) if lqr.feedforward.ndim == 1 else lqr.feedforward
        lifted_for_control = lifted.to(gain.dtype)
        mean = (
            -(gain @ lifted_for_control.unsqueeze(-1)).squeeze(-1) - feedforward
        ).to(normalized.dtype)
        fallback_mean = mean.clamp(
            min=-self.fallback_delta_limit,
            max=self.fallback_delta_limit,
        )
        mean = torch.where(solver_fallback_used[:, None], fallback_mean, mean)
        mean = self.transform_mean(mean)
        std = self.log_std.exp().expand_as(mean)
        distribution = Independent(Normal(mean, std), 1)
        value = self.critic(normalized)
        if single:
            # Distribution retains a batch dimension to keep sampling/evaluation consistent.
            mean = mean[0]
            value = value[0]
            stage_hessian_diag = stage_hessian_diag[0]
            stage_linear = stage_linear[0]
        return PolicyOutput(
            distribution,
            mean,
            value,
            stage_hessian_diag,
            stage_linear,
            lqr,
            solver_valid,
            solver_retry_used,
            solver_fallback_used,
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
        log_prob = output.distribution.log_prob(action.unsqueeze(0) if action.ndim == 1 else action)
        if observation.ndim == 1:
            log_prob = log_prob[0]
        result = (action, log_prob, output.value)
        return (*result, output) if return_output else result

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        output = self(observations)
        return output.distribution.log_prob(actions), output.distribution.entropy(), output.value, output


class GainHoldController:
    """Evaluation-only gain scheduling for intervals greater than one.

    PPO training deliberately does not use this stateful helper. Training uses
    ``gain_update_interval=1``, so ``evaluate_actions`` can reconstruct every
    sampled action from its observation.
    """

    def __init__(self, policy: KoopmanLQRPolicy, gain_update_interval: int = 1) -> None:
        if gain_update_interval < 1:
            raise ValueError("gain_update_interval must be >= 1")
        self.policy = policy
        self.interval = gain_update_interval
        self.phase = 0
        self.gain: torch.Tensor | None = None
        self.feedforward: torch.Tensor | None = None
        self.last_lqr: AffineLQRResult | None = None
        self.last_solver_retry = False
        self.last_solver_fallback = False
        self.last_gain_recomputed = False

    def reset(self) -> None:
        self.phase = 0
        self.gain = None
        self.feedforward = None
        self.last_lqr = None
        self.last_solver_retry = False
        self.last_solver_fallback = False
        self.last_gain_recomputed = False

    @torch.no_grad()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = self.policy.normalize(observation)
        self.last_gain_recomputed = False
        if self.gain is None or self.phase % self.interval == 0:
            output = self.policy(observation)
            self.last_lqr = output.lqr
            self.last_solver_retry = bool(torch.any(output.solver_retry_used))
            self.last_solver_fallback = bool(torch.any(output.solver_fallback_used))
            self.last_gain_recomputed = True
            self.gain = output.lqr.gain
            self.feedforward = output.lqr.feedforward
            if observation.ndim == 1 and self.gain.ndim == 3:
                self.gain = self.gain[0]
                self.feedforward = self.feedforward[0]
        assert self.gain is not None and self.feedforward is not None
        lifted = self.policy.koopman.lift(normalized)
        action = (
            -(self.gain @ lifted.to(self.gain.dtype).unsqueeze(-1)).squeeze(-1)
            - self.feedforward
        ).to(normalized.dtype)
        if self.last_solver_fallback:
            action = action.clamp(
                min=-self.policy.fallback_delta_limit,
                max=self.policy.fallback_delta_limit,
            )
        action = self.policy.transform_mean(action)
        self.phase += 1
        return action
