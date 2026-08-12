"""Detached model-predictive value expansion for the Playground PPO critic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from experiments.playground.koopman import (
    denormalize,
    exact_reward,
    learned_reward,
    lift,
    linear_step,
    load_export,
    normalize,
    reconstruct,
)


def make_mpve_inference_fn(
    base_make_inference_fn: Callable[..., Any],
    *,
    koopman_path: str,
    action_size: int,
    horizon: int,
) -> Callable[..., Any]:
    """Save behavior-critic terminal values while collecting each rollout."""

    import jax

    koopman, _metadata = load_export(Path(koopman_path))

    def make_inference_fn(ppo_network: Any, *args: Any, **kwargs: Any):
        base_factory = base_make_inference_fn(ppo_network, *args, **kwargs)

        def make_policy(params: Any, deterministic: bool = False):
            base_policy = base_factory(params, deterministic=deterministic)

            def policy(observation: Any, key: Any):
                action, extras = base_policy(observation, key)
                # Deterministic evaluation intentionally has no rollout extras.
                if "distribution_params" not in extras:
                    return action, extras
                logits = extras["distribution_params"]
                plan = logits[..., 2 * action_size :]
                expected = horizon * action_size
                plan = jax.lax.stop_gradient(
                    plan[..., :expected].reshape(
                        *plan.shape[:-1], horizon, action_size
                    )
                )
                current = lift(koopman, normalize(koopman, observation))
                for depth in range(horizon):
                    current = linear_step(koopman, current, plan[..., depth, :])
                terminal_observation = denormalize(
                    koopman, reconstruct(koopman, current)
                )
                terminal_value = ppo_network.value_network.apply(
                    params[0], params[2], terminal_observation
                )
                extras = {
                    **extras,
                    "mpve_terminal_value": jax.lax.stop_gradient(terminal_value),
                }
                return action, extras

            return policy

        return make_policy

    return make_inference_fn


def make_mpve_ppo_loss(
    base_loss: Callable[..., Any],
    *,
    koopman_path: str,
    action_size: int,
    horizon: int = 10,
    coefficient: float = 1.0,
    reward_source: str = "exact_cartpole",
) -> Callable[..., Any]:
    """Wrap Brax PPO loss with detached TD-k regression on MPC plans."""

    import jax
    import jax.numpy as jnp

    if horizon < 1 or coefficient < 0:
        raise ValueError("MPVE horizon must be positive and coefficient non-negative")
    if reward_source not in {
        "exact_cartpole",
        "exact_reacher_hard",
        "exact_humanoid_run",
        "learned",
    }:
        raise ValueError("Unknown MPVE reward source")
    koopman, _metadata = load_export(Path(koopman_path))

    def loss(
        params: Any,
        normalizer_params: Any,
        data: Any,
        rng: Any,
        ppo_network: Any,
        entropy_cost: float = 1e-4,
        discounting: float = 0.9,
        reward_scaling: float = 1.0,
        gae_lambda: float = 0.95,
        clipping_epsilon: float = 0.3,
        normalize_advantage: bool = True,
        vf_coefficient: float = 0.5,
        clipping_epsilon_value: float | None = None,
        use_distributional_critic: bool = False,
    ):
        base_total, metrics = base_loss(
            params,
            normalizer_params,
            data,
            rng,
            ppo_network,
            entropy_cost=entropy_cost,
            discounting=discounting,
            reward_scaling=reward_scaling,
            gae_lambda=gae_lambda,
            clipping_epsilon=clipping_epsilon,
            normalize_advantage=normalize_advantage,
            vf_coefficient=vf_coefficient,
            clipping_epsilon_value=clipping_epsilon_value,
            use_distributional_critic=use_distributional_critic,
        )
        if use_distributional_critic:
            raise ValueError("MPVE does not support the distributional critic")
        # Brax data arrives [batch, time, ...].  Distribution params contain
        # [loc, raw_scale, full MPC plan] from the behavior policy.
        observation = jnp.swapaxes(data.observation, 0, 1)
        behavior_logits = jnp.swapaxes(
            data.extras["policy_extras"]["distribution_params"], 0, 1
        )
        plan = behavior_logits[..., 2 * action_size :]
        expected_plan = horizon * action_size
        if plan.shape[-1] < expected_plan:
            raise ValueError("Stored MPC plan is shorter than the MPVE horizon")
        plan = jax.lax.stop_gradient(
            plan[..., :expected_plan].reshape(
                *plan.shape[:-1], horizon, action_size
            )
        )
        current = jax.lax.stop_gradient(lift(koopman, normalize(koopman, observation)))
        imagined_observations = []
        imagined_rewards = []
        for depth in range(horizon):
            normalized_state = reconstruct(koopman, current)
            action = plan[..., depth, :]
            following = linear_step(koopman, current, action)
            normalized_following = reconstruct(koopman, following)
            imagined_observations.append(denormalize(koopman, normalized_state))
            if reward_source == "learned":
                reward = learned_reward(
                    koopman, normalized_state, action, normalized_following
                )
            else:
                reward = exact_reward(
                    reward_source, koopman, action, normalized_following
                )
            imagined_rewards.append(reward * reward_scaling)
            current = following
        imagined_observations = jnp.stack(imagined_observations, axis=-2)
        imagined_rewards = jnp.stack(imagined_rewards, axis=-1)
        value_apply = ppo_network.value_network.apply
        terminal_value = jax.lax.stop_gradient(
            jnp.swapaxes(
                data.extras["policy_extras"]["mpve_terminal_value"], 0, 1
            )
        )
        following_target = terminal_value
        reversed_targets = []
        for depth in range(horizon - 1, -1, -1):
            following_target = (
                imagined_rewards[..., depth] + discounting * following_target
            )
            reversed_targets.append(following_target)
        targets = jax.lax.stop_gradient(
            jnp.stack(tuple(reversed(reversed_targets)), axis=-1)
        )
        predicted_values = value_apply(
            normalizer_params,
            params.value,
            jax.lax.stop_gradient(imagined_observations),
        )
        mpve_loss = jnp.mean(jnp.square(predicted_values - targets))
        total = base_total + coefficient * mpve_loss
        return total, {
            **metrics,
            "total_loss": total,
            "mpve_value_loss": mpve_loss,
            "mpve_predicted_reward_mean": jnp.mean(imagined_rewards),
            "mpve_target_mean": jnp.mean(targets),
        }

    return loss
