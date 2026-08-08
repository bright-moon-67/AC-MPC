from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .visual_model import VisualLinearKoopman


@dataclass
class VisualKoopmanLoss:
    total: torch.Tensor
    linear: torch.Tensor
    robot_rollout: torch.Tensor
    feature_reconstruction: torch.Tensor
    future_feature_reconstruction: torch.Tensor
    latent_variance: torch.Tensor
    stability: torch.Tensor
    identity: torch.Tensor
    transform_reconstruction: torch.Tensor
    transform_condition: torch.Tensor
    transform_singular_value: torch.Tensor
    spectral_radius: torch.Tensor
    latent_std_min: torch.Tensor
    latent_std_mean: torch.Tensor
    transform_min_singular_value: torch.Tensor
    transform_max_singular_value: torch.Tensor
    transform_condition_number: torch.Tensor

    def scalars(self) -> dict[str, float]:
        return {
            name: float(value.detach())
            for name, value in self.__dict__.items()
        }


def _weighted_mean(step_errors: torch.Tensor, discount: float) -> torch.Tensor:
    weights = torch.pow(
        step_errors.new_tensor(discount),
        torch.arange(
            len(step_errors),
            dtype=step_errors.dtype,
            device=step_errors.device,
        ),
    )
    return (step_errors * weights).sum() / weights.sum()


def visual_koopman_loss(
    model: VisualLinearKoopman,
    robot_states: torch.Tensor,
    visual_features: torch.Tensor,
    actions: torch.Tensor,
    *,
    rollout_discount: float = 0.99,
    linear_weight: float = 10.0,
    robot_rollout_weight: float = 1.0,
    feature_reconstruction_weight: float = 0.1,
    future_feature_reconstruction_weight: float = 0.1,
    latent_variance_weight: float = 0.1,
    stability_weight: float = 0.01,
    identity_weight: float = 1e-4,
    transform_reconstruction_weight: float = 1.0,
    transform_condition_weight: float = 1e-3,
    transform_singular_value_weight: float = 1e-2,
    transform_minimum_singular_value: float = 0.25,
    target_latent_std: float = 1.0,
    spectral_radius_limit: float = 1.0,
    detach_linear_targets: bool = True,
) -> VisualKoopmanLoss:
    """Multi-step loss for cached frozen-vision features.

    Args:
        robot_states: Normalized robot states with shape ``[batch,H+1,dr]``.
        visual_features: Normalized frozen-backbone features with shape
            ``[batch,H+1,db]``.
        actions: Applied controls with shape ``[batch,H,du]``.  Actions are
            external inputs to the dynamics and are never predicted.
    """

    if robot_states.ndim != 3 or visual_features.ndim != 3 or actions.ndim != 3:
        raise ValueError(
            "robot_states, visual_features and actions must be rank-3 tensors"
        )
    if robot_states.shape[:2] != visual_features.shape[:2]:
        raise ValueError("Robot-state and visual-feature windows differ")
    if robot_states.shape[1] != actions.shape[1] + 1:
        raise ValueError("State windows must contain one more step than actions")
    if robot_states.shape[0] != actions.shape[0]:
        raise ValueError("State and action batch sizes differ")
    if robot_states.shape[-1] != model.robot_dim:
        raise ValueError("Wrong robot-state dimension")
    if visual_features.shape[-1] != model.visual_feature_dim:
        raise ValueError("Wrong visual-feature dimension")
    if actions.shape[-1] != model.action_dim:
        raise ValueError("Wrong action dimension")
    if not 0.0 < rollout_discount <= 1.0:
        raise ValueError("rollout_discount must lie in (0, 1]")
    if target_latent_std < 0.0:
        raise ValueError("target_latent_std must be non-negative")
    if transform_minimum_singular_value <= 0.0:
        raise ValueError("transform_minimum_singular_value must be positive")
    tensors = (robot_states, visual_features, actions)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise FloatingPointError("Input contains NaN or Inf")

    states = model.make_state(robot_states, visual_features)
    # Materialize a derived transform once.  In learned_orthogonal mode this
    # avoids repeating matrix_exp in lift, rollout, reconstruction, and the
    # transform diagnostics below.
    transform = model.transform_matrix()
    true_lifts = model.lift(states, transform=transform)
    predicted_states, predicted_lifts = model.rollout(
        states[:, 0],
        actions,
        transform=transform,
    )

    if detach_linear_targets:
        # Stop gradients through the visual target encoder while retaining the
        # target-side transform derivative in ||z_hat - T s_next||.  Detaching
        # true_lifts directly would also (incorrectly) detach T.
        target_lifts = model.lift(
            states[:, 1:].detach(),
            transform=transform,
        )
    else:
        target_lifts = true_lifts[:, 1:]
    lifted_step_errors = (
        predicted_lifts - target_lifts
    ).square().mean(dim=(0, 2))
    linear = _weighted_mean(lifted_step_errors, rollout_discount)

    robot_step_errors = (
        predicted_states[..., : model.robot_dim] - robot_states[:, 1:]
    ).square().mean(dim=(0, 2))
    robot_rollout = _weighted_mean(robot_step_errors, rollout_discount)

    true_visual_latents = model.visual_from_state(states)
    reconstructed_features = model.decode_visual(true_visual_latents)
    feature_reconstruction = F.mse_loss(
        reconstructed_features,
        visual_features.detach(),
    )

    predicted_visual_latents = model.visual_from_state(predicted_states)
    predicted_future_features = model.decode_visual(predicted_visual_latents)
    future_feature_step_errors = (
        predicted_future_features - visual_features[:, 1:].detach()
    ).square().mean(dim=(0, 2))
    future_feature_reconstruction = _weighted_mean(
        future_feature_step_errors,
        rollout_discount,
    )

    flattened_latents = true_visual_latents.reshape(
        -1,
        model.visual_latent_dim,
    )
    latent_std = flattened_latents.std(dim=0, unbiased=False)
    latent_variance = torch.relu(
        float(target_latent_std) - latent_std
    ).square().mean()

    eigenvalue_abs = torch.linalg.eigvals(model.A).abs()
    stability = torch.relu(
        eigenvalue_abs - float(spectral_radius_limit)
    ).square().mean()
    spectral_radius = eigenvalue_abs.max().real

    state_identity = torch.eye(
        model.lifted_dim,
        dtype=model.A.dtype,
        device=model.A.device,
    )
    identity = F.mse_loss(model.A, state_identity)

    reconstructed_states = model.reconstruct(
        true_lifts,
        transform=transform,
    )
    transform_reconstruction = F.mse_loss(
        reconstructed_states,
        states.detach(),
    )
    transform_identity = torch.eye(
        model.state_dim,
        dtype=transform.dtype,
        device=transform.device,
    )
    transform_condition = F.mse_loss(
        transform.mT @ transform,
        transform_identity,
    )
    transform_singular_values = torch.linalg.svdvals(transform)
    transform_min_singular_value = transform_singular_values.min()
    transform_max_singular_value = transform_singular_values.max()
    transform_singular_value = torch.relu(
        float(transform_minimum_singular_value) - transform_singular_values
    ).square().mean()
    transform_condition_number = (
        transform_max_singular_value
        / transform_min_singular_value.clamp_min(torch.finfo(transform.dtype).eps)
    )

    total = (
        linear_weight * linear
        + robot_rollout_weight * robot_rollout
        + feature_reconstruction_weight * feature_reconstruction
        + future_feature_reconstruction_weight * future_feature_reconstruction
        + latent_variance_weight * latent_variance
        + stability_weight * stability
        + identity_weight * identity
        + transform_reconstruction_weight * transform_reconstruction
        + transform_condition_weight * transform_condition
        + transform_singular_value_weight * transform_singular_value
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Visual Koopman loss is NaN or Inf")
    return VisualKoopmanLoss(
        total=total,
        linear=linear,
        robot_rollout=robot_rollout,
        feature_reconstruction=feature_reconstruction,
        future_feature_reconstruction=future_feature_reconstruction,
        latent_variance=latent_variance,
        stability=stability,
        identity=identity,
        transform_reconstruction=transform_reconstruction,
        transform_condition=transform_condition,
        transform_singular_value=transform_singular_value,
        spectral_radius=spectral_radius,
        latent_std_min=latent_std.min(),
        latent_std_mean=latent_std.mean(),
        transform_min_singular_value=transform_min_singular_value,
        transform_max_singular_value=transform_max_singular_value,
        transform_condition_number=transform_condition_number,
    )
