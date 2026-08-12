"""JAX inference for framework-neutral Koopman exports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np


class KoopmanParameters(NamedTuple):
    A: Any
    B: Any
    C: Any
    center: Any
    scale: Any
    encoder_weights: tuple[Any, ...]
    encoder_biases: tuple[Any, ...]
    reward_weights: tuple[Any, ...]
    reward_biases: tuple[Any, ...]


def load_export(path: Path) -> tuple[KoopmanParameters, dict[str, Any]]:
    import jax.numpy as jp

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("kind") != "playground_koopman_export_v1":
            raise ValueError("Unsupported Koopman export kind")
        encoder_count = int(metadata["encoder_layer_count"])
        reward_count = int(metadata["reward_layer_count"])
        arrays = {
            name: jp.asarray(archive[name], dtype=jp.float32)
            for name in archive.files
            if name != "metadata_json"
        }
    parameters = KoopmanParameters(
        A=arrays["A"],
        B=arrays["B"],
        C=arrays["C"],
        center=arrays["center"],
        scale=arrays["scale"],
        encoder_weights=tuple(arrays[f"encoder_{i}_weight"] for i in range(encoder_count)),
        encoder_biases=tuple(arrays[f"encoder_{i}_bias"] for i in range(encoder_count)),
        reward_weights=tuple(arrays[f"reward_{i}_weight"] for i in range(reward_count)),
        reward_biases=tuple(arrays[f"reward_{i}_bias"] for i in range(reward_count)),
    )
    state_dim = int(metadata["architecture"]["state_dim"])
    action_dim = int(metadata["architecture"]["action_dim"])
    lifted_dim = state_dim + int(metadata["architecture"]["lift_dim"])
    if parameters.A.shape != (lifted_dim, lifted_dim):
        raise ValueError("Koopman A has the wrong shape")
    if parameters.B.shape != (lifted_dim, action_dim):
        raise ValueError("Koopman B has the wrong shape")
    if parameters.center.shape != (state_dim,) or parameters.scale.shape != (state_dim,):
        raise ValueError("Koopman normalizer has the wrong shape")
    if not bool(jp.all(jp.isfinite(parameters.scale))) or not bool(
        jp.all(parameters.scale > 0)
    ):
        raise ValueError("Koopman normalizer scale must be finite and positive")
    return parameters, metadata


def _mlp(value: Any, weights: tuple[Any, ...], biases: tuple[Any, ...]) -> Any:
    import jax.nn as jnn

    for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        value = value @ weight.T + bias
        if index + 1 < len(weights):
            value = jnn.silu(value)
    return value


def normalize(parameters: KoopmanParameters, observation: Any) -> Any:
    return (observation - parameters.center) / parameters.scale


def denormalize(parameters: KoopmanParameters, normalized_state: Any) -> Any:
    return normalized_state * parameters.scale + parameters.center


def lift(parameters: KoopmanParameters, normalized_state: Any) -> Any:
    import jax.numpy as jp

    encoded = _mlp(
        normalized_state, parameters.encoder_weights, parameters.encoder_biases
    )
    return jp.concatenate((normalized_state, encoded), axis=-1)


def linear_step(parameters: KoopmanParameters, lifted_state: Any, action: Any) -> Any:
    return lifted_state @ parameters.A.T + action @ parameters.B.T


def reconstruct(parameters: KoopmanParameters, lifted_state: Any) -> Any:
    return lifted_state @ parameters.C.T


def learned_reward(
    parameters: KoopmanParameters,
    state: Any,
    action: Any,
    next_state: Any,
) -> Any:
    import jax.nn as jnn
    import jax.numpy as jp

    features = jp.concatenate((state, action, next_state), axis=-1)
    logits = _mlp(features, parameters.reward_weights, parameters.reward_biases)
    return jnn.sigmoid(logits[..., 0])


def cartpole_exact_reward(
    parameters: KoopmanParameters,
    action: Any,
    normalized_next_state: Any,
) -> Any:
    """Official dense reward from predicted normalized Cartpole observation."""

    import jax.numpy as jp

    observation = denormalize(parameters, normalized_next_state)
    position = observation[..., 0]
    # Linear Koopman reconstruction does not exactly preserve the physical
    # cos^2+sin^2=1 manifold.  Project only that orientation pair before
    # applying the otherwise exact official reward formula.
    orientation_norm = jp.maximum(
        jp.sqrt(observation[..., 1] ** 2 + observation[..., 2] ** 2), 1e-6
    )
    pole_cosine = observation[..., 1] / orientation_norm
    upright = (pole_cosine + 1.0) / 2.0
    angular_velocity = observation[..., 4]
    centered = (1.0 + jp.exp(jp.log(0.1) * (jp.abs(position) / 2.0) ** 2)) / 2.0
    small_control = (4.0 + jp.maximum(0.0, 1.0 - action[..., 0] ** 2)) / 5.0
    small_velocity = (
        1.0
        + jp.exp(jp.log(0.1) * (jp.abs(angular_velocity) / 5.0) ** 2)
    ) / 2.0
    return upright * centered * small_control * small_velocity


def reacher_hard_exact_reward(
    parameters: KoopmanParameters,
    action: Any,
    normalized_next_state: Any,
) -> Any:
    """Official ReacherHard sparse reward from its ``to_target`` observation."""

    del action
    import jax.numpy as jp

    observation = denormalize(parameters, normalized_next_state)
    # Playground's hard target has radius 0.015 and the finger geom radius is
    # 0.010, matching the environment's private ``_radii == 0.025`` contract.
    return (jp.linalg.norm(observation[..., 2:4], axis=-1) <= 0.025).astype(
        observation.dtype
    )


def humanoid_run_exact_reward(
    parameters: KoopmanParameters,
    action: Any,
    normalized_next_state: Any,
) -> Any:
    """Official HumanoidRun reward reconstructed from the 67-D observation."""

    import jax.numpy as jp
    from mujoco_playground._src import reward as playground_reward

    observation = denormalize(parameters, normalized_next_state)
    # Observation layout: 21 joint angles, head height, 12 extremities,
    # torso vertical orientation (3), COM velocity (3), qvel (27).
    head_height = observation[..., 21]
    torso_upright = observation[..., 36]
    horizontal_velocity = observation[..., 37:39]
    standing = playground_reward.tolerance(
        head_height, bounds=(1.4, float("inf")), margin=1.4 / 4
    )
    upright = playground_reward.tolerance(
        torso_upright,
        bounds=(0.9, float("inf")),
        sigmoid="linear",
        margin=1.9,
        value_at_margin=0,
    )
    control = playground_reward.tolerance(
        action, margin=1, value_at_margin=0, sigmoid="quadratic"
    ).mean(axis=-1)
    small_control = (4 + control) / 5
    speed = jp.linalg.norm(horizontal_velocity, axis=-1)
    move = playground_reward.tolerance(
        speed,
        bounds=(10.0, float("inf")),
        margin=10.0,
        value_at_margin=0,
        sigmoid="linear",
    )
    move = (5 * move + 1) / 6
    return standing * upright * move * small_control


def exact_reward(
    source: str,
    parameters: KoopmanParameters,
    action: Any,
    normalized_next_state: Any,
) -> Any:
    """Dispatch an observation-sufficient official Playground reward oracle."""

    if source == "exact_cartpole":
        return cartpole_exact_reward(parameters, action, normalized_next_state)
    if source == "exact_reacher_hard":
        return reacher_hard_exact_reward(parameters, action, normalized_next_state)
    if source == "exact_humanoid_run":
        return humanoid_run_exact_reward(parameters, action, normalized_next_state)
    raise ValueError(f"Unknown exact reward source {source!r}")
