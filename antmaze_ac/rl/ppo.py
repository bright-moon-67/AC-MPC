from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import copy

import numpy as np
import torch

from .ac_koopman_policy import KoopmanLQRPolicy


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    rewards: np.ndarray
    saturation: np.ndarray
    action_bound: np.ndarray
    applied_action_abs_mean: np.ndarray
    applied_delta_action_l2: np.ndarray
    applied_delta_action_abs_max: np.ndarray
    dare_retry: np.ndarray
    dare_fallback: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    distances: np.ndarray
    episode_successes: np.ndarray
    episode_waypoints_completed: np.ndarray


@torch.no_grad()
def collect_rollout(env, policy: KoopmanLQRPolicy, steps: int, gamma: float, gae_lambda: float, device):
    if not hasattr(env, "_ppo_observation"):
        observation, _ = env.reset()
    else:
        observation = env._ppo_observation
    observations, actions, log_probs, values = [], [], [], []
    rewards, dones, saturation, action_bound = [], [], [], []
    applied_action_abs_mean, applied_delta_action_l2 = [], []
    applied_delta_action_abs_max = []
    dare_retry, dare_fallback = [], []
    completed_returns, completed_lengths, completed_successes = [], [], []
    completed_waypoints = []
    distances = []
    episode_return = float(getattr(env, "_ppo_episode_return", 0.0))
    episode_length = int(getattr(env, "_ppo_episode_length", 0))
    for _ in range(steps):
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
        if isinstance(policy, KoopmanLQRPolicy):
            action, log_prob, value, policy_output = policy.act(
                observation_tensor,
                return_output=True,
            )
            solver_retry = float(torch.any(policy_output.solver_retry_used))
            solver_fallback = float(torch.any(policy_output.solver_fallback_used))
        else:
            action, log_prob, value = policy.act(observation_tensor)
            solver_retry = 0.0
            solver_fallback = 0.0
        next_observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        done = terminated or truncated
        observations.append(observation_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(float(reward))
        distances.append(float(info.get("distance", np.nan)))
        episode_return += float(reward)
        episode_length += 1
        dones.append(float(done))
        saturation.append(float(info.get("action_saturation_ratio", 0.0)))
        action_bound.append(float(info.get("action_bound_ratio", 0.0)))
        applied = np.asarray(
            info.get("applied_action", action.detach().cpu().numpy()),
            dtype=np.float32,
        )
        delta = np.asarray(
            info.get("applied_delta_action", np.zeros_like(applied)),
            dtype=np.float32,
        )
        applied_action_abs_mean.append(float(np.mean(np.abs(applied))))
        applied_delta_action_l2.append(float(np.linalg.norm(delta)))
        applied_delta_action_abs_max.append(float(np.max(np.abs(delta))))
        dare_retry.append(solver_retry)
        dare_fallback.append(solver_fallback)
        if done:
            completed_returns.append(episode_return)
            completed_lengths.append(episode_length)
            completed_successes.append(bool(info.get("is_success", terminated)))
            completed_waypoints.append(
                int(info.get("waypoints_completed", int(terminated)))
            )
            episode_return = 0.0
            episode_length = 0
            next_observation, _ = env.reset()
        observation = next_observation
    env._ppo_observation = observation
    env._ppo_episode_return = episode_return
    env._ppo_episode_length = episode_length
    next_value = policy(torch.as_tensor(observation, dtype=torch.float32, device=device)).value
    value_tensor = torch.stack(values)
    advantages = torch.zeros(steps, device=device)
    gae = torch.zeros((), device=device)
    for index in reversed(range(steps)):
        mask = 1.0 - dones[index]
        following_value = next_value if index == steps - 1 else value_tensor[index + 1]
        delta = rewards[index] + gamma * following_value * mask - value_tensor[index]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[index] = gae
    returns = advantages + value_tensor
    return Rollout(
        observations=torch.stack(observations),
        actions=torch.stack(actions),
        old_log_probs=torch.stack(log_probs),
        returns=returns,
        advantages=advantages,
        rewards=np.asarray(rewards),
        saturation=np.asarray(saturation),
        action_bound=np.asarray(action_bound),
        applied_action_abs_mean=np.asarray(applied_action_abs_mean),
        applied_delta_action_l2=np.asarray(applied_delta_action_l2),
        applied_delta_action_abs_max=np.asarray(applied_delta_action_abs_max),
        dare_retry=np.asarray(dare_retry),
        dare_fallback=np.asarray(dare_fallback),
        episode_returns=np.asarray(completed_returns),
        episode_lengths=np.asarray(completed_lengths),
        distances=np.asarray(distances, dtype=np.float32),
        episode_successes=np.asarray(completed_successes, dtype=np.bool_),
        episode_waypoints_completed=np.asarray(
            completed_waypoints, dtype=np.int64
        ),
    )


@torch.no_grad()
def collect_vector_rollout(
    envs: Sequence,
    policy: KoopmanLQRPolicy,
    steps: int,
    gamma: float,
    gae_lambda: float,
    device,
) -> Rollout:
    """Collect ``steps`` total transitions with batched policy/DARE calls.

    Environments are stepped independently and reset explicitly. This keeps
    each DeltaActionWrapper's previous action isolated while batching the
    expensive actor and DARE inference across environments. ``envs`` may also
    be a process-vector pool exposing batched ``step`` and ``reset_indices``
    methods; in that case physics simulation executes concurrently.
    """

    environment_count = len(envs)
    if environment_count < 2:
        raise ValueError("collect_vector_rollout requires at least two environments")
    if steps < environment_count or steps % environment_count:
        raise ValueError(
            f"steps={steps} must be divisible by num_envs={environment_count}"
        )
    time_steps = steps // environment_count
    process_pool = hasattr(envs, "reset_indices") and hasattr(
        envs, "current_observations"
    )
    if process_pool:
        if envs.current_observations is None:
            raise RuntimeError("Process vector environments must be reset first")
        current_observations = list(envs.current_observations)
        episode_returns = envs.episode_returns.tolist()
        episode_lengths = envs.episode_lengths.tolist()
    else:
        current_observations = []
        episode_returns = []
        episode_lengths = []
        for env in envs:
            if not hasattr(env, "_ppo_observation"):
                observation, _ = env.reset()
            else:
                observation = env._ppo_observation
            current_observations.append(
                np.asarray(observation, dtype=np.float32)
            )
            episode_returns.append(
                float(getattr(env, "_ppo_episode_return", 0.0))
            )
            episode_lengths.append(
                int(getattr(env, "_ppo_episode_length", 0))
            )

    observations, actions, log_probs, values = [], [], [], []
    rewards, dones, saturation, action_bound = [], [], [], []
    applied_action_abs_mean, applied_delta_action_l2 = [], []
    applied_delta_action_abs_max = []
    dare_retry, dare_fallback = [], []
    completed_returns, completed_lengths, completed_successes = [], [], []
    completed_waypoints = []
    distances = []
    for _ in range(time_steps):
        observation_tensor = torch.as_tensor(
            np.stack(current_observations),
            dtype=torch.float32,
            device=device,
        )
        if isinstance(policy, KoopmanLQRPolicy):
            action, log_prob, value, policy_output = policy.act(
                observation_tensor,
                return_output=True,
            )
            solver_retry = (
                policy_output.solver_retry_used.detach().float().cpu().numpy()
            )
            solver_fallback = (
                policy_output.solver_fallback_used.detach().float().cpu().numpy()
            )
        else:
            action, log_prob, value = policy.act(observation_tensor)
            solver_retry = np.zeros(environment_count, dtype=np.float32)
            solver_fallback = np.zeros(environment_count, dtype=np.float32)
        action_array = action.detach().cpu().numpy()
        next_observations: list[np.ndarray | None] = [None] * environment_count
        step_rewards, step_dones, step_saturation, step_action_bound = [], [], [], []
        step_action_abs_mean, step_delta_action_l2 = [], []
        step_delta_action_abs_max = []
        step_distances = []
        step_results = (
            envs.step(action_array)
            if process_pool
            else [
                env.step(action_array[index])
                for index, env in enumerate(envs)
            ]
        )
        reset_indices = []
        for index, result in enumerate(step_results):
            next_observation, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            episode_returns[index] += float(reward)
            episode_lengths[index] += 1
            if done:
                completed_returns.append(episode_returns[index])
                completed_lengths.append(episode_lengths[index])
                completed_successes.append(
                    bool(info.get("is_success", terminated))
                )
                completed_waypoints.append(
                    int(info.get("waypoints_completed", int(terminated)))
                )
                episode_returns[index] = 0.0
                episode_lengths[index] = 0
                if process_pool:
                    reset_indices.append(index)
                else:
                    next_observation, _ = envs[index].reset()
            next_observations[index] = np.asarray(
                next_observation, dtype=np.float32
            )
            step_rewards.append(float(reward))
            step_dones.append(float(done))
            step_saturation.append(
                float(info.get("action_saturation_ratio", 0.0))
            )
            step_action_bound.append(
                float(info.get("action_bound_ratio", 0.0))
            )
            applied = np.asarray(
                info.get("applied_action", action_array[index]),
                dtype=np.float32,
            )
            delta = np.asarray(
                info.get("applied_delta_action", np.zeros_like(applied)),
                dtype=np.float32,
            )
            step_action_abs_mean.append(float(np.mean(np.abs(applied))))
            step_delta_action_l2.append(float(np.linalg.norm(delta)))
            step_delta_action_abs_max.append(float(np.max(np.abs(delta))))
            step_distances.append(float(info.get("distance", np.nan)))
        if process_pool and reset_indices:
            reset_results = envs.reset_indices(reset_indices)
            for index, (reset_observation, _) in reset_results.items():
                next_observations[index] = np.asarray(
                    reset_observation, dtype=np.float32
                )
        if any(observation is None for observation in next_observations):
            raise RuntimeError("Vector rollout produced a missing observation")
        observations.append(observation_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(step_rewards)
        dones.append(step_dones)
        saturation.append(step_saturation)
        action_bound.append(step_action_bound)
        applied_action_abs_mean.append(step_action_abs_mean)
        applied_delta_action_l2.append(step_delta_action_l2)
        applied_delta_action_abs_max.append(step_delta_action_abs_max)
        distances.append(step_distances)
        dare_retry.append(solver_retry)
        dare_fallback.append(solver_fallback)
        current_observations = next_observations

    if process_pool:
        envs.current_observations = current_observations
        envs.episode_returns = np.asarray(episode_returns, dtype=np.float64)
        envs.episode_lengths = np.asarray(episode_lengths, dtype=np.int64)
    else:
        for index, env in enumerate(envs):
            env._ppo_observation = current_observations[index]
            env._ppo_episode_return = episode_returns[index]
            env._ppo_episode_length = episode_lengths[index]

    next_observation_tensor = torch.as_tensor(
        np.stack(current_observations),
        dtype=torch.float32,
        device=device,
    )
    next_values = policy(next_observation_tensor).value
    value_tensor = torch.stack(values)
    reward_array = np.asarray(rewards, dtype=np.float32)
    done_array = np.asarray(dones, dtype=np.float32)
    advantages = torch.zeros(
        time_steps,
        environment_count,
        dtype=value_tensor.dtype,
        device=device,
    )
    gae = torch.zeros(
        environment_count,
        dtype=value_tensor.dtype,
        device=device,
    )
    reward_tensor = torch.as_tensor(reward_array, device=device)
    done_tensor = torch.as_tensor(done_array, device=device)
    for index in reversed(range(time_steps)):
        mask = 1.0 - done_tensor[index]
        following_value = (
            next_values if index == time_steps - 1 else value_tensor[index + 1]
        )
        delta = (
            reward_tensor[index]
            + gamma * following_value * mask
            - value_tensor[index]
        )
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[index] = gae
    returns = advantages + value_tensor

    def flatten_tensor(tensors: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(tensors)
        return stacked.reshape(steps, *stacked.shape[2:])

    return Rollout(
        observations=flatten_tensor(observations),
        actions=flatten_tensor(actions),
        old_log_probs=flatten_tensor(log_probs),
        returns=returns.reshape(steps),
        advantages=advantages.reshape(steps),
        rewards=reward_array.reshape(steps),
        saturation=np.asarray(saturation).reshape(steps),
        action_bound=np.asarray(action_bound).reshape(steps),
        applied_action_abs_mean=np.asarray(applied_action_abs_mean).reshape(steps),
        applied_delta_action_l2=np.asarray(applied_delta_action_l2).reshape(steps),
        applied_delta_action_abs_max=np.asarray(
            applied_delta_action_abs_max
        ).reshape(steps),
        dare_retry=np.asarray(dare_retry).reshape(steps),
        dare_fallback=np.asarray(dare_fallback).reshape(steps),
        episode_returns=np.asarray(completed_returns),
        episode_lengths=np.asarray(completed_lengths),
        distances=np.asarray(distances, dtype=np.float32).reshape(steps),
        episode_successes=np.asarray(completed_successes, dtype=np.bool_),
        episode_waypoints_completed=np.asarray(
            completed_waypoints, dtype=np.int64
        ),
    )


def ppo_update(
    policy: KoopmanLQRPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    *,
    update_epochs: int,
    minibatch_size: int,
    clip_range: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_grad_norm: float,
    target_kl: float | None = None,
    clip_value_loss: bool = False,
    minimum_log_std: float | None = None,
    maximum_log_std: float | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
    std_optimizer: torch.optim.Optimizer | None = None,
    kl_soft_stop_multiplier: float = 1.5,
    kl_hard_rollback_multiplier: float = 3.0,
    normalize_advantages_globally: bool = True,
) -> dict[str, float]:
    if kl_soft_stop_multiplier <= 0:
        raise ValueError("kl_soft_stop_multiplier must be positive")
    if kl_hard_rollback_multiplier < kl_soft_stop_multiplier:
        raise ValueError(
            "kl_hard_rollback_multiplier must be >= "
            "kl_soft_stop_multiplier"
        )
    if std_optimizer is not None and critic_optimizer is None:
        raise ValueError(
            "std_optimizer requires the separated critic_optimizer path"
        )
    if (
        minimum_log_std is not None
        and maximum_log_std is not None
        and minimum_log_std > maximum_log_std
    ):
        raise ValueError("minimum_log_std must not exceed maximum_log_std")
    log_std_parameter = getattr(policy, "log_std", None)
    if log_std_parameter is not None and (
        minimum_log_std is not None or maximum_log_std is not None
    ):
        with torch.no_grad():
            log_std_parameter.clamp_(
                min=(
                    -float("inf")
                    if minimum_log_std is None
                    else minimum_log_std
                ),
                max=(
                    float("inf")
                    if maximum_log_std is None
                    else maximum_log_std
                ),
            )
    metrics: dict[str, list[torch.Tensor]] = {
        "policy": [],
        "value": [],
        "entropy": [],
        "grad_norm": [],
        "approx_kl": [],
        "clip_fraction": [],
        "update_dare_retry_fraction": [],
        "update_dare_fallback_fraction": [],
    }
    count = len(rollout.observations)
    old_values = rollout.returns - rollout.advantages
    normalized_advantages = rollout.advantages
    if normalize_advantages_globally and count > 1:
        normalized_advantages = (
            normalized_advantages - normalized_advantages.mean()
        ) / (normalized_advantages.std() + 1e-8)
    optimizer_steps = 0
    critic_optimizer_steps = 0
    std_optimizer_steps = 0
    actor_grad_norms: list[torch.Tensor] = []
    critic_grad_norms: list[torch.Tensor] = []
    std_grad_norms: list[torch.Tensor] = []
    early_stopped = False
    soft_stopped = False
    hard_rollbacks = 0
    early_stop_kl = torch.zeros((), device=rollout.observations.device)
    separated_optimizers = critic_optimizer is not None
    actor_updates_enabled = True

    def optimizer_parameters(current_optimizer):
        return [
            parameter
            for group in current_optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]

    def snapshot_optimizer_step(current_optimizer):
        parameters = [
            parameter
            for group in current_optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        parameter_values = [parameter.detach().clone() for parameter in parameters]
        optimizer_values = {}
        for parameter in parameters:
            optimizer_values[parameter] = {
                key: (
                    value.detach().clone()
                    if torch.is_tensor(value)
                    else copy.deepcopy(value)
                )
                for key, value in current_optimizer.state.get(parameter, {}).items()
            }
        return parameters, parameter_values, optimizer_values

    @torch.no_grad()
    def restore_optimizer_step(current_optimizer, snapshot) -> None:
        parameters, parameter_values, optimizer_values = snapshot
        for parameter, value in zip(parameters, parameter_values):
            parameter.copy_(value)
            state = current_optimizer.state[parameter]
            state.clear()
            state.update(optimizer_values[parameter])

    def clip_optimizer_gradients(current_optimizer) -> torch.Tensor:
        parameters = optimizer_parameters(current_optimizer)
        if not parameters:
            return torch.zeros((), device=rollout.observations.device)
        return torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)

    soft_kl_limit = (
        None
        if target_kl is None
        else float(target_kl) * float(kl_soft_stop_multiplier)
    )
    hard_kl_limit = (
        None
        if target_kl is None
        else float(target_kl) * float(kl_hard_rollback_multiplier)
    )
    actor_update_snapshot = (
        snapshot_optimizer_step(optimizer)
        if separated_optimizers and hard_kl_limit is not None
        else None
    )
    std_update_snapshot = (
        snapshot_optimizer_step(std_optimizer)
        if std_optimizer is not None and hard_kl_limit is not None
        else None
    )
    for _ in range(update_epochs):
        permutation = torch.randperm(count, device=rollout.observations.device)
        for start in range(0, count, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            log_prob, entropy, values, policy_output = policy.evaluate_actions(
                rollout.observations[indices], rollout.actions[indices]
            )
            solver_retry = getattr(policy_output, "solver_retry_used", None)
            solver_fallback = getattr(policy_output, "solver_fallback_used", None)
            advantages = normalized_advantages[indices]
            if not normalize_advantages_globally and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = torch.exp(log_prob - rollout.old_log_probs[indices])
            objective = ratio * advantages
            clipped = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
            policy_loss = -torch.minimum(objective, clipped).mean()
            # Match the referenced SB3 PPO: vf_coef multiplies the plain MSE.
            value_error = (values - rollout.returns[indices]).square()
            if clip_value_loss:
                clipped_values = old_values[indices] + torch.clamp(
                    values - old_values[indices],
                    -clip_range,
                    clip_range,
                )
                clipped_error = (
                    clipped_values - rollout.returns[indices]
                ).square()
                value_error = torch.maximum(value_error, clipped_error)
            value_loss = value_error.mean()
            entropy_mean = entropy.mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_mean
            with torch.no_grad():
                log_ratio = log_prob - rollout.old_log_probs[indices]
                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()
            if (
                separated_optimizers
                and optimizer_steps > 0
                and hard_kl_limit is not None
                and float(approx_kl) > hard_kl_limit
            ):
                # A step can look safe on its own minibatch yet generalize to a
                # catastrophic KL on the next one. In that case restore only
                # this update's policy parameters/moments; critic learning is
                # deliberately retained.
                restore_optimizer_step(optimizer, actor_update_snapshot)
                if std_optimizer is not None:
                    restore_optimizer_step(std_optimizer, std_update_snapshot)
                optimizer_steps = 0
                std_optimizer_steps = 0
                hard_rollbacks += 1
                early_stopped = True
                actor_updates_enabled = False
                early_stop_kl = approx_kl.detach()
            elif (
                actor_updates_enabled
                and soft_kl_limit is not None
                and float(approx_kl) > soft_kl_limit
            ):
                early_stopped = True
                soft_stopped = True
                early_stop_kl = approx_kl.detach()
                actor_updates_enabled = False

            actor_grad_norm = torch.zeros((), device=values.device)
            critic_grad_norm = torch.zeros((), device=values.device)
            std_grad_norm = torch.zeros((), device=values.device)
            following_kl = approx_kl
            following_clip_fraction = clip_fraction

            if separated_optimizers:
                # Actor and log-std are the only policy-distribution parameters.
                # A KL event must not discard critic parameters or Adam moments.
                if actor_updates_enabled:
                    optimizer.zero_grad(set_to_none=True)
                    if std_optimizer is not None:
                        std_optimizer.zero_grad(set_to_none=True)
                    actor_loss = policy_loss - entropy_coefficient * entropy_mean
                    actor_loss.backward(retain_graph=True)
                    actor_grad_norm = clip_optimizer_gradients(optimizer)
                    actor_grad_norms.append(actor_grad_norm.detach())
                    if std_optimizer is not None:
                        std_grad_norm = clip_optimizer_gradients(std_optimizer)
                        std_grad_norms.append(std_grad_norm.detach())
                    if not torch.isfinite(actor_grad_norm) or not torch.isfinite(
                        std_grad_norm
                    ):
                        raise FloatingPointError("PPO policy gradient is NaN or Inf")
                    optimizer.step()
                    if std_optimizer is not None:
                        std_optimizer.step()
                    log_std = getattr(policy, "log_std", None)
                    if log_std is not None and log_std.requires_grad and (
                        minimum_log_std is not None or maximum_log_std is not None
                    ):
                        with torch.no_grad():
                            log_std.clamp_(
                                min=(
                                    -float("inf")
                                    if minimum_log_std is None
                                    else minimum_log_std
                                ),
                                max=(
                                    float("inf")
                                    if maximum_log_std is None
                                    else maximum_log_std
                                ),
                            )
                    with torch.no_grad():
                        following_log_prob = policy.evaluate_actions(
                            rollout.observations[indices],
                            rollout.actions[indices],
                        )[0]
                        following_log_ratio = (
                            following_log_prob - rollout.old_log_probs[indices]
                        )
                        following_ratio = torch.exp(following_log_ratio)
                        following_kl = (
                            (following_ratio - 1.0) - following_log_ratio
                        ).mean()
                        following_clip_fraction = (
                            torch.abs(following_ratio - 1.0) > clip_range
                        ).float().mean()
                    if (
                        hard_kl_limit is not None
                        and float(following_kl) > hard_kl_limit
                    ):
                        restore_optimizer_step(optimizer, actor_update_snapshot)
                        if std_optimizer is not None:
                            restore_optimizer_step(
                                std_optimizer, std_update_snapshot
                            )
                        optimizer_steps = 0
                        std_optimizer_steps = 0
                        hard_rollbacks += 1
                        early_stopped = True
                        actor_updates_enabled = False
                        early_stop_kl = following_kl.detach()
                    else:
                        optimizer_steps += 1
                        if std_optimizer is not None:
                            std_optimizer_steps += 1
                        if (
                            soft_kl_limit is not None
                            and float(following_kl) > soft_kl_limit
                        ):
                            # Keep this reasonable policy step, but do not reuse
                            # the rollout for any further actor updates.
                            early_stopped = True
                            soft_stopped = True
                            actor_updates_enabled = False
                            early_stop_kl = following_kl.detach()

                critic_optimizer.zero_grad(set_to_none=True)
                (value_coefficient * value_loss).backward()
                critic_grad_norm = clip_optimizer_gradients(critic_optimizer)
                critic_grad_norms.append(critic_grad_norm.detach())
                if not torch.isfinite(critic_grad_norm):
                    raise FloatingPointError("PPO critic gradient is NaN or Inf")
                critic_optimizer.step()
                critic_optimizer_steps += 1
                grad_norm = actor_grad_norm
            else:
                # Backward-compatible shared-optimizer route. Mild KL excesses
                # are retained; only a catastrophic post-step jump is reverted.
                if not actor_updates_enabled:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_optimizer_gradients(optimizer)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("PPO gradient is NaN or Inf")
                step_snapshot = (
                    snapshot_optimizer_step(optimizer)
                    if hard_kl_limit is not None
                    else None
                )
                optimizer.step()
                log_std = getattr(policy, "log_std", None)
                if log_std is not None and (
                    minimum_log_std is not None or maximum_log_std is not None
                ):
                    with torch.no_grad():
                        log_std.clamp_(
                            min=(
                                -float("inf")
                                if minimum_log_std is None
                                else minimum_log_std
                            ),
                            max=(
                                float("inf")
                                if maximum_log_std is None
                                else maximum_log_std
                            ),
                        )
                with torch.no_grad():
                    following_log_prob = policy.evaluate_actions(
                        rollout.observations[indices], rollout.actions[indices]
                    )[0]
                    following_log_ratio = (
                        following_log_prob - rollout.old_log_probs[indices]
                    )
                    following_ratio = torch.exp(following_log_ratio)
                    following_kl = (
                        (following_ratio - 1.0) - following_log_ratio
                    ).mean()
                    following_clip_fraction = (
                        torch.abs(following_ratio - 1.0) > clip_range
                    ).float().mean()
                if (
                    hard_kl_limit is not None
                    and float(following_kl) > hard_kl_limit
                ):
                    restore_optimizer_step(optimizer, step_snapshot)
                    hard_rollbacks += 1
                    early_stopped = True
                    actor_updates_enabled = False
                    early_stop_kl = following_kl.detach()
                else:
                    optimizer_steps += 1
                    if (
                        soft_kl_limit is not None
                        and float(following_kl) > soft_kl_limit
                    ):
                        early_stopped = True
                        soft_stopped = True
                        actor_updates_enabled = False
                        early_stop_kl = following_kl.detach()
            for key, value in {
                "policy": policy_loss,
                "value": value_loss,
                "entropy": entropy_mean,
                "grad_norm": grad_norm,
                "approx_kl": following_kl,
                "clip_fraction": following_clip_fraction,
                "update_dare_retry_fraction": (
                    solver_retry.float().mean()
                    if solver_retry is not None
                    else torch.zeros((), device=values.device)
                ),
                "update_dare_fallback_fraction": (
                    solver_fallback.float().mean()
                    if solver_fallback is not None
                    else torch.zeros((), device=values.device)
                ),
            }.items():
                # Keep scalar diagnostics on-device during the update. Calling
                # float() for every metric/minibatch forces repeated GPU/CPU
                # synchronization and materially slows small-matrix DARE PPO.
                metrics[key].append(value.detach())
            if early_stopped:
                if not separated_optimizers:
                    break
        if early_stopped and not separated_optimizers:
            break
    result = {
        key: (
            float(torch.stack(values).mean().cpu())
            if values
            else 0.0
        )
        for key, values in metrics.items()
    }
    result["ppo_optimizer_steps"] = float(optimizer_steps)
    result["ppo_early_stopped"] = float(early_stopped)
    result["ppo_early_stop_kl"] = float(early_stop_kl.cpu())
    if separated_optimizers:
        result["actor_grad_norm"] = (
            float(torch.stack(actor_grad_norms).mean().cpu())
            if actor_grad_norms
            else 0.0
        )
        result["critic_grad_norm"] = (
            float(torch.stack(critic_grad_norms).mean().cpu())
            if critic_grad_norms
            else 0.0
        )
        result["std_grad_norm"] = (
            float(torch.stack(std_grad_norms).mean().cpu())
            if std_grad_norms
            else 0.0
        )
        result["ppo_actor_optimizer_steps"] = float(optimizer_steps)
        result["ppo_critic_optimizer_steps"] = float(critic_optimizer_steps)
        result["ppo_std_optimizer_steps"] = float(std_optimizer_steps)
        result["ppo_kl_soft_stopped"] = float(soft_stopped)
        result["ppo_kl_hard_rollbacks"] = float(hard_rollbacks)
    if early_stopped:
        # Report the KL that actually activated the guard; the minibatches that
        # preceded it can all have near-zero pre-update KL.
        result["approx_kl"] = result["ppo_early_stop_kl"]
    return result
