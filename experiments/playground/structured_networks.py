"""JAX policy heads for AB-PQ and Koopman MPC PPO peers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from experiments.playground.koopman import (
    KoopmanParameters,
    lift,
    load_export,
    normalize,
)


STRUCTURED_METHODS = ("AB-PQ", "KMPC", "AC-MPC-MPVE")


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _init_mlp(
    key: Any,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> tuple[dict[str, Any], ...]:
    """PyTorch-Linear-style hidden init with a zero final controller head."""

    import jax
    import jax.numpy as jp

    keys = jax.random.split(key, 4)
    hidden_bound = 1.0 / math.sqrt(input_dim)
    hidden_weight = jax.random.uniform(
        keys[0], (hidden_dim, input_dim), minval=-hidden_bound, maxval=hidden_bound
    )
    hidden_bias = jax.random.uniform(
        keys[1], (hidden_dim,), minval=-hidden_bound, maxval=hidden_bound
    )
    return (
        {"weight": hidden_weight, "bias": hidden_bias},
        {
            "weight": jp.zeros((output_dim, hidden_dim), dtype=jp.float32),
            "bias": jp.zeros((output_dim,), dtype=jp.float32),
        },
    )


def _apply_mlp(layers: tuple[dict[str, Any], ...], value: Any) -> Any:
    import jax.nn as jnn

    value = jnn.gelu(value @ layers[0]["weight"].T + layers[0]["bias"])
    return value @ layers[1]["weight"].T + layers[1]["bias"]


def _condense_dynamics(
    parameters: KoopmanParameters, horizon: int
) -> tuple[Any, Any]:
    import jax.numpy as jp

    lifted_dim, action_dim = parameters.B.shape
    lifted_power = jp.eye(lifted_dim, dtype=parameters.A.dtype)
    lifted_action = jp.zeros(
        (lifted_dim, horizon * action_dim), dtype=parameters.A.dtype
    )
    state_rows = []
    action_rows = []
    for step in range(horizon):
        lifted_power = parameters.A @ lifted_power
        lifted_action = parameters.A @ lifted_action
        lifted_action = lifted_action.at[
            :, step * action_dim : (step + 1) * action_dim
        ].add(parameters.B)
        state_rows.append(parameters.C @ lifted_power)
        action_rows.append(parameters.C @ lifted_action)
    return jp.concatenate(state_rows, axis=0), jp.concatenate(action_rows, axis=0)


def _abpq_action(
    koopman: KoopmanParameters,
    layers: tuple[dict[str, Any], ...],
    lifted_state: Any,
    *,
    rank: int,
) -> Any:
    import jax.nn as jnn
    import jax.numpy as jp

    lifted_dim, action_dim = koopman.B.shape
    raw = _apply_mlp(layers, lifted_state)
    diagonal = jnn.softplus(raw[..., :lifted_dim] - 6.0)
    factor_stop = lifted_dim + lifted_dim * rank
    factors = 0.1 * jp.tanh(raw[..., lifted_dim:factor_stop]).reshape(
        *raw.shape[:-1], lifted_dim, rank
    )
    linear = 10.0 * jp.tanh(raw[..., factor_stop:])

    diagonal_b = diagonal[..., :, None] * koopman.B
    diagonal_a = diagonal[..., :, None] * koopman.A
    ut_b = jp.einsum("...lr,la->...ra", factors, koopman.B)
    ut_a = jp.einsum("...lr,lk->...rk", factors, koopman.A)
    hessian = (
        jp.eye(action_dim, dtype=lifted_state.dtype)
        + koopman.B.T @ koopman.B
        + jp.einsum("al,...lb->...ab", koopman.B.T, diagonal_b)
        + jp.einsum("...ra,...rb->...ab", ut_b, ut_b)
    )
    state_action = (
        koopman.B.T @ koopman.A
        + jp.einsum("al,...lk->...ak", koopman.B.T, diagonal_a)
        + jp.einsum("...ra,...rk->...ak", ut_b, ut_a)
    )
    action_linear = jp.einsum("la,...l->...a", koopman.B, linear)
    rhs = jp.einsum("...al,...l->...a", state_action, lifted_state) + action_linear
    raw_action = -jp.linalg.solve(
        hessian + 1e-6 * jp.eye(action_dim), rhs[..., None]
    )[..., 0]
    return jp.tanh(raw_action)


def _kmpc_plan(
    koopman: KoopmanParameters,
    layers: tuple[dict[str, Any], ...],
    lifted_state: Any,
    state_map: Any,
    action_map: Any,
    *,
    horizon: int,
    solver_iterations: int,
) -> Any:
    import jax.numpy as jp

    physical_dim = koopman.C.shape[0]
    action_dim = koopman.B.shape[1]
    augmented_dim = physical_dim + action_dim
    raw = _apply_mlp(layers, lifted_state).reshape(
        *lifted_state.shape[:-1], 2, horizon, augmented_dim
    )
    raw_quadratic = jp.tanh(raw[..., 0, :, :])
    centered = raw_quadratic - jp.mean(raw_quadratic, axis=-1, keepdims=True)
    quadratic = jp.exp(1.5 * centered)
    linear = 10.0 * jp.tanh(raw[..., 1, :, :])
    free_physical = lifted_state @ state_map.T
    q_state = quadratic[..., :physical_dim].reshape(
        *lifted_state.shape[:-1], horizon * physical_dim
    )
    q_action = quadratic[..., physical_dim:].reshape(
        *lifted_state.shape[:-1], horizon * action_dim
    )
    p_state = linear[..., :physical_dim].reshape(
        *lifted_state.shape[:-1], horizon * physical_dim
    )
    p_action = linear[..., physical_dim:].reshape(
        *lifted_state.shape[:-1], horizon * action_dim
    )
    hessian = (
        jp.einsum("pi,...p,pj->...ij", action_map, q_state, action_map)
        + jax_vmap_diag(q_action)
    )
    qp_linear = (
        jp.einsum("...p,pi->...i", q_state * free_physical + p_state, action_map)
        + p_action
    )
    lipschitz = jp.max(jp.sum(jp.abs(hessian), axis=-1), axis=-1)
    step = 0.95 / (lipschitz + 1e-6)
    current = jp.zeros_like(qp_linear)
    extrapolated = current
    momentum = 1.0
    for _ in range(solver_iterations):
        gradient = jp.einsum("...ij,...j->...i", hessian, extrapolated) + qp_linear
        following = jp.clip(extrapolated - step[..., None] * gradient, -1.0, 1.0)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        extrapolated = following + ((momentum - 1.0) / next_momentum) * (
            following - current
        )
        current = following
        momentum = next_momentum
    return current.reshape(*lifted_state.shape[:-1], horizon, action_dim)


def jax_vmap_diag(value: Any) -> Any:
    """Batch-compatible diagonal embedding without a Python batch loop."""

    import jax.numpy as jp

    size = value.shape[-1]
    return value[..., :, None] * jp.eye(size, dtype=value.dtype)


def _controller_outputs(
    method: str,
    koopman: KoopmanParameters,
    layers: tuple[dict[str, Any], ...],
    observation: Any,
    *,
    ab_rank: int,
    kmpc_horizon: int,
    kmpc_solver_iterations: int,
    state_map: Any,
    action_map: Any,
) -> tuple[Any, Any | None]:
    normalized_state = normalize(koopman, observation)
    lifted_state = lift(koopman, normalized_state)
    if method == "AB-PQ":
        return _abpq_action(
            koopman, layers, lifted_state, rank=ab_rank
        ), None
    plan = _kmpc_plan(
        koopman,
        layers,
        lifted_state,
        state_map,
        action_map,
        horizon=kmpc_horizon,
        solver_iterations=kmpc_solver_iterations,
    )
    return plan[..., 0, :], plan


def make_structured_ppo_networks(
    observation_size: Any,
    action_size: int,
    preprocess_observations_fn: Any,
    *,
    method: str,
    koopman_path: str,
    hidden_dim: int = 128,
    ab_rank: int = 4,
    kmpc_horizon: int = 20,
    kmpc_solver_iterations: int = 20,
    critic_input: str = "raw_observation",
):
    """Brax network factory with a selectable critic and structured actor."""

    import jax.numpy as jp
    from brax.training import distribution, networks
    from brax.training.agents.ppo import networks as ppo_networks

    if method not in STRUCTURED_METHODS:
        raise ValueError(f"Unknown structured method {method!r}")
    if critic_input not in {"raw_observation", "lifted_state"}:
        raise ValueError(f"Unknown critic input {critic_input!r}")
    if not isinstance(observation_size, tuple) or len(observation_size) != 1:
        raise ValueError("Structured actors require a flat observation vector")
    koopman, metadata = load_export(Path(koopman_path))
    architecture = metadata["architecture"]
    if int(architecture["state_dim"]) != observation_size[0]:
        raise ValueError("Koopman observation dimension does not match environment")
    if int(architecture["action_dim"]) != action_size:
        raise ValueError("Koopman action dimension does not match environment")
    lifted_dim = koopman.A.shape[0]
    if method == "AB-PQ":
        output_dim = lifted_dim * (ab_rank + 2)
        state_map = jp.empty((0, 0), dtype=jp.float32)
        action_map = jp.empty((0, 0), dtype=jp.float32)
        plan_size = 0
    else:
        output_dim = 2 * kmpc_horizon * (observation_size[0] + action_size)
        state_map, action_map = _condense_dynamics(koopman, kmpc_horizon)
        plan_size = kmpc_horizon * action_size if method == "AC-MPC-MPVE" else 0

    def policy_init(key: Any) -> dict[str, Any]:
        return {
            "controller": _init_mlp(key, lifted_dim, hidden_dim, output_dim),
            "raw_scale": jp.full(
                (action_size,), _inverse_softplus(1.0 - 0.001), dtype=jp.float32
            ),
        }

    def policy_apply(
        normalizer_params: Any, policy_params: dict[str, Any], observation: Any
    ) -> Any:
        del normalizer_params  # Koopman uses its own frozen data normalizer.
        action, plan = _controller_outputs(
            method,
            koopman,
            policy_params["controller"],
            observation,
            ab_rank=ab_rank,
            kmpc_horizon=kmpc_horizon,
            kmpc_solver_iterations=kmpc_solver_iterations,
            state_map=state_map,
            action_map=action_map,
        )
        location = jp.arctanh(jp.clip(action, -0.999, 0.999))
        raw_scale = jp.broadcast_to(policy_params["raw_scale"], location.shape)
        values = [location, raw_scale]
        if plan_size:
            assert plan is not None
            values.append(plan.reshape(*plan.shape[:-2], plan_size))
        return jp.concatenate(values, axis=-1)

    base = ppo_networks.make_ppo_networks(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
    )
    if critic_input == "lifted_state":
        import jax.nn as jnn

        lifted_value = networks.make_value_network(
            lifted_dim,
            hidden_layer_sizes=(256,) * 5,
            activation=jnn.swish,
        )

        def value_apply(
            normalizer_params: Any, value_params: Any, observation: Any
        ) -> Any:
            del normalizer_params
            lifted = lift(koopman, normalize(koopman, observation))
            return lifted_value.apply(None, value_params, lifted)

        value_network = networks.FeedForwardNetwork(
            init=lifted_value.init,
            apply=value_apply,
        )
    else:
        value_network = base.value_network
    policy_network = networks.FeedForwardNetwork(
        init=policy_init, apply=policy_apply
    )

    class PlanTanhDistribution(distribution.NormalTanhDistribution):
        def __init__(self) -> None:
            super().__init__(event_size=action_size)
            self._param_size = 2 * action_size + plan_size

        def create_dist(self, parameters: Any):
            return super().create_dist(parameters[..., : 2 * action_size])

    return ppo_networks.PPONetworks(
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=PlanTanhDistribution(),
    )
