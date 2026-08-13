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
    expensive actor and DARE inference across environments.
    """

    environment_count = len(envs)
    if environment_count < 2:
        raise ValueError("collect_vector_rollout requires at least two environments")
    if steps < environment_count or steps % environment_count:
        raise ValueError(
            f"steps={steps} must be divisible by num_envs={environment_count}"
        )
    time_steps = steps // environment_count
    current_observations = []
    episode_returns = []
    episode_lengths = []
    for env in envs:
        if not hasattr(env, "_ppo_observation"):
            observation, _ = env.reset()
        else:
            observation = env._ppo_observation
        current_observations.append(np.asarray(observation, dtype=np.float32))
        episode_returns.append(float(getattr(env, "_ppo_episode_return", 0.0)))
        episode_lengths.append(int(getattr(env, "_ppo_episode_length", 0)))

    observations, actions, log_probs, values = [], [], [], []
    rewards, dones, saturation, action_bound = [], [], [], []
    applied_action_abs_mean, applied_delta_action_l2 = [], []
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
        next_observations = []
        step_rewards, step_dones, step_saturation, step_action_bound = [], [], [], []
        step_action_abs_mean, step_delta_action_l2 = [], []
        step_distances = []
        for index, env in enumerate(envs):
            next_observation, reward, terminated, truncated, info = env.step(
                action_array[index]
            )
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
                next_observation, _ = env.reset()
            next_observations.append(
                np.asarray(next_observation, dtype=np.float32)
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
            step_distances.append(float(info.get("distance", np.nan)))
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
        distances.append(step_distances)
        dare_retry.append(solver_retry)
        dare_fallback.append(solver_fallback)
        current_observations = next_observations

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
) -> dict[str, float]:
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
    optimizer_steps = 0
    early_stopped = False
    early_stop_kl = torch.zeros((), device=rollout.observations.device)

    def snapshot_optimizer_step():
        parameters = [
            parameter
            for group in optimizer.param_groups
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
                for key, value in optimizer.state.get(parameter, {}).items()
            }
        return parameters, parameter_values, optimizer_values

    @torch.no_grad()
    def restore_optimizer_step(snapshot) -> None:
        parameters, parameter_values, optimizer_values = snapshot
        for parameter, value in zip(parameters, parameter_values):
            parameter.copy_(value)
            state = optimizer.state[parameter]
            state.clear()
            state.update(optimizer_values[parameter])
    for _ in range(update_epochs):
        permutation = torch.randperm(count, device=rollout.observations.device)
        for start in range(0, count, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            log_prob, entropy, values, policy_output = policy.evaluate_actions(
                rollout.observations[indices], rollout.actions[indices]
            )
            solver_retry = getattr(policy_output, "solver_retry_used", None)
            solver_fallback = getattr(policy_output, "solver_fallback_used", None)
            advantages = rollout.advantages[indices]
            if len(advantages) > 1:
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
            if target_kl is not None and float(approx_kl) > target_kl:
                early_stopped = True
                early_stop_kl = approx_kl.detach()
                break
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("PPO gradient is NaN or Inf")
            step_snapshot = (
                snapshot_optimizer_step() if target_kl is not None else None
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
            # A pre-step KL guard cannot detect the jump caused by the current
            # Adam step.  Re-evaluate the same minibatch transactionally and
            # roll back both parameters and Adam moments when that jump exceeds
            # the trust-region budget.  This is essential for sensitive
            # differentiable MPC actors with small exploration variance.
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
            if target_kl is not None and float(following_kl) > target_kl:
                restore_optimizer_step(step_snapshot)
                early_stopped = True
                early_stop_kl = following_kl.detach()
            else:
                optimizer_steps += 1
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
                break
        if early_stopped:
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
    if early_stopped:
        # Report the KL that actually activated the guard; the minibatches that
        # preceded it can all have near-zero pre-update KL.
        result["approx_kl"] = result["ppo_early_stop_kl"]
    return result
