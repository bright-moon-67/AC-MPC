from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import DeepKoopman


@dataclass
class KoopmanLoss:
    total: torch.Tensor
    linear: torch.Tensor
    rollout: torch.Tensor
    stability: torch.Tensor
    latent_std: torch.Tensor
    identity: torch.Tensor
    controllability_svd: torch.Tensor
    augmentation: torch.Tensor
    reconstruction: torch.Tensor
    spectral_radius: torch.Tensor
    latent_std_min: torch.Tensor
    latent_std_mean: torch.Tensor

    def scalars(self) -> dict[str, float]:
        values = {name: float(value.detach()) for name, value in self.__dict__.items()}
        # Naming bridge: fullA_history_v2 calls the physical open-loop term
        # pred_loss, while the prompt calls it rollout loss.
        values["pred_loss"] = values["rollout"]
        values["linear_loss"] = values["linear"]
        return values


def _weighted_mean(step_errors: torch.Tensor, gamma: float) -> torch.Tensor:
    weights = torch.pow(
        step_errors.new_tensor(gamma),
        torch.arange(len(step_errors), device=step_errors.device, dtype=step_errors.dtype),
    )
    return (step_errors * weights).sum() / weights.sum()


def _controllability_svd_loss(
    model: DeepKoopman,
    minimum_singular_value: float,
) -> torch.Tensor:
    dimension = model.lifted_dim
    blocks = []
    current = model.B
    for _ in range(dimension):
        blocks.append(current)
        current = model.A @ current
    controllability = torch.cat(blocks, dim=-1)
    singular_values = torch.linalg.svdvals(controllability)
    return torch.relu(float(minimum_singular_value) - singular_values).square().mean()


def koopman_loss(
    model: DeepKoopman,
    states: torch.Tensor,
    delta_actions: torch.Tensor,
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
    spectral_radius_limit: float = 1.0,
    target_latent_std: float = 1.0,
    svd_min_singular_value: float = 0.0,
) -> KoopmanLoss:
    """fullA_history_v2 open-loop loss adapted to prompt history=1.

    Only the initial true state is lifted for prediction. Future lifted states
    are generated recursively by the full-A linear model, while true future
    states are encoded solely to construct lifted alignment targets.
    """

    if states.ndim != 3 or delta_actions.ndim != 3:
        raise ValueError("states and delta_actions must have shapes [batch,K+1,nx] and [batch,K,nu]")
    if states.shape[1] != delta_actions.shape[1] + 1:
        raise ValueError("states must contain one more time step than delta_actions")
    if not (torch.isfinite(states).all() and torch.isfinite(delta_actions).all()):
        raise FloatingPointError("Input contains NaN or Inf")

    predicted_states, predicted_lifts = model.rollout(states[:, 0], delta_actions)
    true_lifts = model.lift(states[:, 1:])
    lifted_step_errors = (predicted_lifts - true_lifts).square().mean(dim=(0, 2))
    state_step_errors = (predicted_states - states[:, 1:]).square().mean(dim=(0, 2))
    linear = _weighted_mean(lifted_step_errors, rollout_discount)
    rollout = _weighted_mean(state_step_errors, rollout_discount)

    all_true_phi = model.encoder(states.reshape(-1, model.state_dim))
    phi_std = all_true_phi.std(dim=0, unbiased=False)
    latent_std = torch.relu(float(target_latent_std) - phi_std).square().mean()

    eigenvalue_abs = torch.linalg.eigvals(model.A).abs()
    stability = torch.relu(eigenvalue_abs - float(spectral_radius_limit)).square().mean()
    spectral_radius = eigenvalue_abs.max().real
    identity = F.mse_loss(
        model.A,
        torch.eye(model.lifted_dim, dtype=model.A.dtype, device=model.A.device),
    )
    if controllability_svd_weight > 0:
        controllability_svd = _controllability_svd_loss(model, svd_min_singular_value)
    else:
        controllability_svd = states.new_zeros(())
    if augmentation_weight > 0:
        relifted_predictions = model.lift(model.reconstruct(predicted_lifts))
        augmentation = F.mse_loss(relifted_predictions, predicted_lifts)
    else:
        augmentation = states.new_zeros(())
    # With identity skip C=[I,0], this should be exactly zero and serves as an
    # invariant check required by the prompt.
    reconstruction = F.mse_loss(model.reconstruct(model.lift(states)), states)

    total = (
        linear_weight * linear
        + rollout_weight * rollout
        + stability_weight * stability
        + latent_std_weight * latent_std
        + identity_weight * identity
        + controllability_svd_weight * controllability_svd
        + augmentation_weight * augmentation
        + reconstruction_weight * reconstruction
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
        spectral_radius=spectral_radius,
        latent_std_min=phi_std.min(),
        latent_std_mean=phi_std.mean(),
    )
