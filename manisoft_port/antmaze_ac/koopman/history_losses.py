from __future__ import annotations

import torch
import torch.nn.functional as F

from .history_model import HistoryDeepKoopman
from .losses import KoopmanLoss, _controllability_svd_loss, _weighted_mean


def history_koopman_loss(
    model: HistoryDeepKoopman,
    contexts: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    *,
    rollout_discount: float = 0.99,
    linear_weight: float = 10.0,
    rollout_weight: float = 1.0,
    stability_weight: float = 0.01,
    latent_std_weight: float = 0.1,
    identity_weight: float = 1e-4,
    controllability_svd_weight: float = 0.0,
    augmentation_weight: float = 0.0,
    reconstruction_weight: float = 1.0,
    tip_position_weight: float = 0.0,
    tip_position_slice: tuple[int, int] | None = None,
    state_std: torch.Tensor | None = None,
    spectral_radius_limit: float = 1.0,
    target_latent_std: float = 1.0,
    svd_min_singular_value: float = 0.0,
) -> KoopmanLoss:
    """The existing AC-MPC Koopman loss with history-conditioned lifting."""

    if contexts.ndim != 3 or states.ndim != 3 or actions.ndim != 3:
        raise ValueError("contexts, states and actions must be rank-3 tensors")
    if states.shape[1] != actions.shape[1] + 1 or contexts.shape[:2] != states.shape[:2]:
        raise ValueError("contexts/states must have K+1 steps and actions must have K")
    if not (
        torch.isfinite(contexts).all()
        and torch.isfinite(states).all()
        and torch.isfinite(actions).all()
    ):
        raise FloatingPointError("Input contains NaN or Inf")

    predicted_states, predicted_lifts = model.rollout(
        states[:, 0],
        actions,
        contexts[:, 0],
    )
    true_lifts = model.lift(states[:, 1:], contexts[:, 1:])
    lifted_step_errors = (predicted_lifts - true_lifts).square().mean(dim=(0, 2))
    state_step_errors = (predicted_states - states[:, 1:]).square().mean(dim=(0, 2))
    linear = _weighted_mean(lifted_step_errors, rollout_discount)
    rollout = _weighted_mean(state_step_errors, rollout_discount)

    if tip_position_weight > 0:
        if tip_position_slice is None or state_std is None:
            raise ValueError(
                "tip_position_slice and state_std are required when "
                "tip_position_weight is positive"
            )
        tip_start, tip_stop = map(int, tip_position_slice)
        if not (0 <= tip_start < tip_stop <= states.shape[-1]):
            raise ValueError(
                f"invalid tip position slice {tip_position_slice} for "
                f"state dimension {states.shape[-1]}"
            )
        physical_scale = state_std.to(
            device=states.device,
            dtype=states.dtype,
        )[tip_start:tip_stop]
        physical_tip_error = (
            predicted_states[..., tip_start:tip_stop]
            - states[:, 1:, tip_start:tip_stop]
        ) * physical_scale
        tip_step_errors = physical_tip_error.square().mean(dim=(0, 2))
        tip_position = _weighted_mean(tip_step_errors, rollout_discount)
        tip_position_h1 = tip_step_errors[0]
    else:
        tip_position = states.new_zeros(())
        tip_position_h1 = states.new_zeros(())

    all_contexts = contexts.reshape(-1, model.context_dim)
    phi_std = model.encoder(all_contexts).std(dim=0, unbiased=False)
    latent_std = torch.relu(float(target_latent_std) - phi_std).square().mean()
    eigenvalue_abs = torch.linalg.eigvals(model.A).abs()
    stability = torch.relu(eigenvalue_abs - float(spectral_radius_limit)).square().mean()
    spectral_radius = eigenvalue_abs.max().real
    identity = F.mse_loss(
        model.A,
        torch.eye(model.lifted_dim, dtype=model.A.dtype, device=model.A.device),
    )
    controllability_svd = (
        _controllability_svd_loss(model, svd_min_singular_value)
        if controllability_svd_weight > 0
        else states.new_zeros(())
    )
    if augmentation_weight > 0:
        relifted = model.lift(model.reconstruct(predicted_lifts), contexts[:, 1:])
        augmentation = F.mse_loss(relifted, predicted_lifts)
    else:
        augmentation = states.new_zeros(())
    reconstruction = F.mse_loss(model.reconstruct(model.lift(states, contexts)), states)

    total = (
        linear_weight * linear
        + rollout_weight * rollout
        + stability_weight * stability
        + latent_std_weight * latent_std
        + identity_weight * identity
        + controllability_svd_weight * controllability_svd
        + augmentation_weight * augmentation
        + reconstruction_weight * reconstruction
        + tip_position_weight * tip_position
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Koopman loss is NaN or Inf")
    return KoopmanLoss(
        total=total,
        linear=linear,
        rollout=rollout,
        stability=stability,
        latent_std=latent_std,
        identity=identity,
        controllability_svd=controllability_svd,
        augmentation=augmentation,
        reconstruction=reconstruction,
        tip_position=tip_position,
        tip_position_h1=tip_position_h1,
        spectral_radius=spectral_radius,
        latent_std_min=phi_std.min(),
        latent_std_mean=phi_std.mean(),
    )
