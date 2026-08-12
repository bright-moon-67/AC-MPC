"""Training-free preflight for the AC-MPC DMC benchmark sub-project.

This command deliberately performs no optimizer step.  It validates the frozen
experiment config, compares every adapter against ``dm_control.suite.load``,
checks timeout semantics and actor construction, and writes a review artifact.
Formal training remains behind the user's explicit approval gate.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from antmaze_ac.koopman.model import DeepKoopman
from experiments.dmc.actors import ACTOR_TYPES, actor_mean, build_actor
from experiments.dmc.config import (
    PROFILE_NAMES,
    default_config_path,
    load_experiment_config,
    resolve_execution_spec,
)
from experiments.dmc.protocol import protocol_fingerprint
from experiments.dmc.reward_model import TransitionRewardModel
from experiments.dmc.reward_oracle import (
    EXACT_REWARD_ORACLE_TASKS,
    LEARNED_TRANSITION_REWARD,
    OFFICIAL_OBSERVATION_ORACLE,
    ORACLE_PARITY_MAX_ABS_ERROR,
    ExactObservationRewardOracle,
    cartpole_swingup_official_reward,
)
from experiments.dmc.source_identity import source_identity
from experiments.dmc.tasks.registry import ALL_TASK_ORDER, get_task_spec, verify_task_spec


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _flatten(observation: dict[str, Any], task_name: str) -> np.ndarray:
    spec = get_task_spec(task_name)
    return np.concatenate(
        [
            np.asarray(observation[key], dtype=np.float64).reshape(-1)
            for key, _ in spec.obs_layout
        ]
    ).astype(np.float32)


def _suite_parity(task_name: str, *, seed: int, steps: int) -> dict[str, Any]:
    from dm_control import suite

    from experiments.dmc.tasks.adapter import make_dmc_adapter

    spec = get_task_spec(task_name)
    official = suite.load(
        spec.domain,
        spec.task,
        task_kwargs={"random": seed},
    )
    adapter = make_dmc_adapter(task_name, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    max_observation_error = 0.0
    max_reward_error = 0.0
    max_probe_error = 0.0
    max_discount_error = 0.0
    try:
        official_step = official.reset()
        adapted_observation = adapter.reset(seed=seed)
        max_observation_error = float(
            np.max(np.abs(_flatten(official_step.observation, task_name) - adapted_observation))
        )
        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0, size=spec.action_dim)
            official_step = official.step(action)
            adapted_observation, reward, done, info = adapter.step(action)
            max_observation_error = max(
                max_observation_error,
                float(
                    np.max(
                        np.abs(
                            _flatten(official_step.observation, task_name)
                            - adapted_observation
                        )
                    )
                ),
            )
            max_reward_error = max(
                max_reward_error, abs(float(official_step.reward) - float(reward))
            )
            max_probe_error = max(
                max_probe_error,
                abs(float(reward) - float(info["reward_components"]["reward"])),
            )
            official_discount = float(official_step.discount)
            adapted_discount = float(info.get("discount", official_discount))
            max_discount_error = max(
                max_discount_error, abs(official_discount - adapted_discount)
            )
            if bool(official_step.last()) != bool(done):
                raise AssertionError(f"done mismatch for {task_name}")
        first = adapter.reset(seed=seed).copy()
        second = adapter.reset(seed=seed).copy()
        reset_reproducible = bool(np.array_equal(first, second))
    finally:
        official.close()
        adapter.close()
    passed = bool(
        max_observation_error <= 1e-7
        and max_reward_error <= 1e-12
        and max_probe_error <= 1e-12
        and max_discount_error <= 1e-12
        and reset_reproducible
    )
    return {
        "passed": passed,
        "steps": steps,
        "max_observation_abs_error": max_observation_error,
        "max_reward_abs_error": max_reward_error,
        "max_reward_probe_abs_error": max_probe_error,
        "max_discount_abs_error": max_discount_error,
        "reset_reproducible": reset_reproducible,
    }


def _mpve_reward_source_parity(
    task_name: str,
    *,
    source: str,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    """Verify an exact observation oracle against live transition rewards."""

    if source == LEARNED_TRANSITION_REWARD:
        return {
            "passed": True,
            "source": source,
            "task": task_name,
            "exact_observation_oracle_available": task_name
            in EXACT_REWARD_ORACLE_TASKS,
            "status": "explicit_learned_reward_fallback_or_ablation",
            "steps": 0,
            "max_reward_abs_error": None,
        }
    if source != OFFICIAL_OBSERVATION_ORACLE:
        raise ValueError(f"Unsupported MPVE reward source {source!r}")
    if task_name != "cartpole_swingup":
        raise ValueError(f"No exact observation reward oracle for {task_name!r}")

    from experiments.dmc.tasks.adapter import make_dmc_adapter

    spec = get_task_spec(task_name)
    env = make_dmc_adapter(task_name, seed=seed)
    rng = np.random.default_rng(seed + 20_000)
    max_error = 0.0
    try:
        env.reset(seed=seed)
        for _ in range(steps):
            action = rng.uniform(-1.0, 1.0, size=spec.action_dim)
            next_observation, reward, done, info = env.step(action)
            applied_action = np.asarray(info["applied_action"], dtype=np.float32)
            predicted = cartpole_swingup_official_reward(
                torch.as_tensor(next_observation).unsqueeze(0),
                torch.as_tensor(applied_action).unsqueeze(0),
            )
            max_error = max(max_error, abs(float(predicted.item()) - reward))
            if done:
                env.reset(seed=seed)
    finally:
        env.close()
    return {
        "passed": bool(max_error <= ORACLE_PARITY_MAX_ABS_ERROR),
        "source": source,
        "task": task_name,
        "exact_observation_oracle_available": True,
        "status": "verified_against_live_dm_control_transition_reward",
        "steps": steps,
        "max_reward_abs_error": max_error,
        "required_max_abs_error": ORACLE_PARITY_MAX_ABS_ERROR,
        "state_timing": "post_transition_next_observation",
        "action": "applied_action",
    }


def _timeout_check(task_name: str) -> dict[str, Any]:
    from experiments.dmc.tasks.adapter import make_dmc_adapter

    spec = get_task_spec(task_name)
    env = make_dmc_adapter(
        task_name,
        seed=91,
        time_limit=3 * spec.native_control_dt,
    )
    try:
        env.reset(seed=91)
        final_info: dict[str, Any] = {}
        done = False
        steps = 0
        while not done and steps < 5:
            _, _, done, final_info = env.step(np.zeros(spec.action_dim))
            steps += 1
    finally:
        env.close()
    discount = float(final_info.get("discount", float("nan")))
    truncated = bool(final_info.get("truncated", done))
    terminated = bool(final_info.get("terminated", False))
    return {
        "passed": bool(done and steps == 3 and discount == 1.0 and truncated and not terminated),
        "steps": steps,
        "done": bool(done),
        "discount": discount,
        "terminated": terminated,
        "truncated": truncated,
    }


def _actor_forward_check(
    task_name: str,
    config,
    *,
    profile: str,
) -> dict[str, Any]:
    spec = get_task_spec(task_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synthetic_batch_size = int(
        config.raw["profiles"][profile]["minibatch_size"]
    )
    model = DeepKoopman(
        state_dim=spec.obs_dim,
        action_dim=spec.action_dim,
        lift_dim=int(config.raw["koopman"]["lift_dim"]),
        hidden_dims=tuple(config.raw["koopman"]["hidden_dims"]),
        activation=str(config.raw["koopman"]["activation"]),
    ).to(device)
    with torch.no_grad():
        model.A.copy_(0.9 * torch.eye(model.lifted_dim))
        model.B.normal_(mean=0.0, std=0.01)
    model.freeze_dynamics()
    states = torch.zeros(synthetic_batch_size, spec.obs_dim, device=device)
    with torch.no_grad():
        lifted = model.lift(states)
    results: dict[str, Any] = {}
    for actor_type in ACTOR_TYPES:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        actor = build_actor(
            actor_type,
            task_name,
            device,
            koopman=model if actor_type != "PPO" else None,
            config=config.actor_config,
        )
        actor.train()
        action = actor_mean(
            actor_type,
            actor,
            states,
            lifted if actor_type != "PPO" else None,
        )
        # Synthetic gradient probe only: no data, loss history, or optimizer
        # exists, and no parameter update is performed.
        objective = action.square().mean() + 1e-3 * action.mean()
        objective.backward()
        gradients = [
            parameter.grad
            for parameter in actor.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        finite = bool(torch.isfinite(action).all())
        bounded = bool((action.abs() <= config.actor_config.action_limit + 1e-6).all())
        finite_gradients = bool(
            gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        )
        results[actor_type] = {
            "passed": (
                finite
                and bounded
                and finite_gradients
                and tuple(action.shape)
                == (synthetic_batch_size, spec.action_dim)
            ),
            "shape": list(action.shape),
            "synthetic_batch_size": synthetic_batch_size,
            "device": str(device),
            "finite": finite,
            "bounded": bounded,
            "finite_gradients": finite_gradients,
            "forward_backward_seconds": time.perf_counter() - started,
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
            "trainable_parameters": sum(
                parameter.numel() for parameter in actor.parameters() if parameter.requires_grad
            ),
        }
        del actor, action, objective, gradients
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def _mpve_forward_check(
    task_name: str,
    config,
    *,
    profile: str,
) -> dict[str, Any]:
    """Exercise the real MPC-prediction/reward-source critic-only boundary.

    This is a synthetic shape/gradient probe.  It constructs no optimizer,
    reads no training data and performs no parameter update.
    """

    from experiments.dmc.ppo.train_dmc_ppo import (
        ValueNetwork,
        collect_mpve_prediction,
        mpve_value_loss,
    )

    spec = get_task_spec(task_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config.raw["profiles"][profile]["minibatch_size"])
    horizon = int(config.raw["ppo"]["mpve_horizon"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = DeepKoopman(
        state_dim=spec.obs_dim,
        action_dim=spec.action_dim,
        lift_dim=int(config.raw["koopman"]["lift_dim"]),
        hidden_dims=tuple(config.raw["koopman"]["hidden_dims"]),
        activation=str(config.raw["koopman"]["activation"]),
    ).to(device)
    with torch.no_grad():
        model.A.copy_(0.9 * torch.eye(model.lifted_dim, device=device))
        model.B.normal_(mean=0.0, std=0.01)
    actor = build_actor(
        "AC-MPC-MPVE",
        task_name,
        device,
        koopman=model,
        config=config.actor_config,
    )
    reward_source = str(config.raw["ppo"]["mpve_reward_source"])
    if reward_source == OFFICIAL_OBSERVATION_ORACLE:
        reward_predictor: torch.nn.Module = ExactObservationRewardOracle(
            task_name,
            torch.zeros(spec.obs_dim, device=device),
            torch.ones(spec.obs_dim, device=device),
        ).to(device)
    else:
        reward_predictor = TransitionRewardModel(
            state_dim=spec.obs_dim,
            action_dim=spec.action_dim,
            hidden_dims=tuple(config.raw["koopman"]["reward_hidden_dims"]),
            activation=str(config.raw["koopman"]["activation"]),
        ).to(device)
    critic = ValueNetwork(
        spec.obs_dim,
        int(config.raw["ppo"]["critic_hidden_dim"]),
        int(config.raw["actors"]["architecture"]["ppo_hidden_layers"]),
    ).to(device)
    states = torch.linspace(
        -0.25,
        0.25,
        steps=batch_size * spec.obs_dim,
        dtype=torch.float32,
        device=device,
    ).reshape(batch_size, spec.obs_dim)
    with torch.no_grad():
        lifted = model.lift(states)
    prediction = collect_mpve_prediction(
        actor,
        critic,
        model,
        reward_predictor,
        lifted,
        horizon=horizon,
        gamma=float(config.raw["ppo"]["discount"]),
    )
    loss = mpve_value_loss(
        critic,
        prediction.value_observations,
        prediction.td_k_targets,
    )
    loss.backward()
    critic_gradients = [
        parameter.grad
        for parameter in critic.parameters()
        if parameter.grad is not None
    ]
    frozen_modules = (actor, model, reward_predictor)
    frozen_has_gradient = any(
        parameter.grad is not None
        for module in frozen_modules
        for parameter in module.parameters()
    )
    expected_observation_shape = (batch_size, horizon, spec.obs_dim)
    expected_scalar_shape = (batch_size, horizon)
    critic_gradients_finite = bool(
        critic_gradients
        and all(torch.isfinite(gradient).all() for gradient in critic_gradients)
    )
    critic_gradient_nonzero = bool(
        critic_gradients
        and any(bool((gradient != 0).any()) for gradient in critic_gradients)
    )
    reward_bounded = bool(
        (prediction.predicted_rewards >= 0).all()
        and (prediction.predicted_rewards <= 1).all()
    )
    detached = bool(
        not prediction.value_observations.requires_grad
        and not prediction.predicted_rewards.requires_grad
        and not prediction.terminal_value.requires_grad
        and not prediction.td_k_targets.requires_grad
    )
    passed = bool(
        tuple(prediction.action.shape) == (batch_size, spec.action_dim)
        and tuple(prediction.value_observations.shape)
        == expected_observation_shape
        and tuple(prediction.predicted_rewards.shape) == expected_scalar_shape
        and tuple(prediction.td_k_targets.shape) == expected_scalar_shape
        and torch.isfinite(loss)
        and torch.isfinite(prediction.td_k_targets).all()
        and reward_bounded
        and detached
        and critic_gradients_finite
        and critic_gradient_nonzero
        and not frozen_has_gradient
    )
    result = {
        "passed": passed,
        "device": str(device),
        "synthetic_batch_size": batch_size,
        "horizon": horizon,
        "reward_source": reward_source,
        "predicted_value_observation_shape": list(
            prediction.value_observations.shape
        ),
        "critic_input": "normalized_raw_task_observation_v1",
        "td_k_target_shape": list(prediction.td_k_targets.shape),
        "reward_in_closed_interval_0_1": reward_bounded,
        "prediction_and_targets_detached": detached,
        "critic_gradients_finite": critic_gradients_finite,
        "critic_gradient_nonzero": critic_gradient_nonzero,
        "actor_koopman_reward_gradient_absent": not frozen_has_gradient,
        "loss": float(loss.detach()),
        "forward_backward_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "optimizer_steps": 0,
        "environment_steps": 0,
    }
    del actor, critic, reward_predictor, model, prediction, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _vector_rollout_probe(
    task_name: str,
    config,
    *,
    profile: str,
    env_workers: int | None = None,
) -> dict[str, Any]:
    """Measure one exact configured vector rollout without policy optimization."""

    from experiments.dmc.ppo.vector_env import make_dmc_vector_env

    profile_config = config.raw["profiles"][profile]
    num_envs = int(profile_config["num_envs"])
    rollout_steps = int(profile_config["rollout_steps"])
    spec = get_task_spec(task_name)
    before_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    started = time.perf_counter()
    env = make_dmc_vector_env(
        task_name, num_envs, seed=83, workers=env_workers
    )
    reset_seconds = 0.0
    step_seconds = 0.0
    transitions = 0
    finite = True
    try:
        reset_started = time.perf_counter()
        observation = env.reset()
        reset_seconds = time.perf_counter() - reset_started
        finite = finite and bool(np.isfinite(observation).all())
        rng = np.random.default_rng(84)
        step_started = time.perf_counter()
        for _ in range(rollout_steps):
            actions = rng.uniform(
                -1.0, 1.0, size=(num_envs, spec.action_dim)
            ).astype(np.float32)
            transition = env.step(actions)
            transitions += num_envs
            finite = finite and bool(
                np.isfinite(transition.observation).all()
                and np.isfinite(transition.reward).all()
                and np.isfinite(transition.discount).all()
            )
        step_seconds = time.perf_counter() - step_started
        protocol = dict(env.protocol)
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    after_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    expected = num_envs * rollout_steps
    passed = bool(
        transitions == expected
        and finite
        and step_seconds > 0
        and np.isfinite(step_seconds)
    )
    return {
        "passed": passed,
        "num_envs": num_envs,
        "env_workers": int(getattr(env, "workers", 1)),
        "rollout_steps": rollout_steps,
        "environment_transitions": transitions,
        "reset_seconds": reset_seconds,
        "step_seconds": step_seconds,
        "total_seconds_including_construction": elapsed,
        "environment_transitions_per_step_second": (
            transitions / max(step_seconds, 1e-12)
        ),
        "process_peak_rss_before_kib": before_rss_kib,
        "process_peak_rss_after_kib": after_rss_kib,
        "process_peak_rss_increase_kib": max(0, after_rss_kib - before_rss_kib),
        "protocol_fingerprint": protocol_fingerprint(protocol),
        "optimizer_steps": 0,
    }


def _storage_upper_bound(config, *, profile: str) -> dict[str, Any]:
    """Conservative uncompressed collector-array size (not an NPZ promise)."""

    spec = get_task_spec(config.task)
    transitions = int(config.raw["data"]["max_transitions_per_train_seed"])
    float32_values = 2 * spec.obs_dim + 2 * spec.action_dim + 2
    bool_values = 4
    int64_values = 5
    stage_unicode_bytes = 5 * 4
    bytes_per_transition = (
        4 * float32_values
        + bool_values
        + 8 * int64_values
        + stage_unicode_bytes
    )
    seed_count = int(config.raw["profiles"][profile]["train_seed_count"])
    return {
        "kind": "uncompressed_collector_arrays_upper_bound_v1",
        "excludes": [
            "npz_compression_ratio",
            "filesystem_overhead",
            "checkpoints",
            "temporary_files",
        ],
        "bytes_per_transition": bytes_per_transition,
        "transitions_per_training_seed": transitions,
        "training_seed_count": seed_count,
        "total_bytes": bytes_per_transition * transitions * seed_count,
    }


def _hardware_inventory() -> dict[str, Any]:
    """Return review-only capacity data without reserving any resources."""

    memory: dict[str, int | None] = {
        "system_total_bytes": None,
        "system_available_bytes": None,
        "process_current_rss_bytes": None,
        "process_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        * 1024,
    }
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        memory["system_total_bytes"] = pages * page_size
    except (OSError, ValueError, TypeError):
        pass
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", maxsplit=1)
            fields = raw.strip().split()
            if fields:
                values[name] = int(fields[0]) * 1024
        memory["system_available_bytes"] = values.get("MemAvailable")
    except (OSError, ValueError):
        pass
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                memory["process_current_rss_bytes"] = int(line.split()[1]) * 1024
                break
    except (OSError, ValueError):
        pass

    usage = shutil.disk_usage(Path.cwd())
    cuda_available = torch.cuda.is_available()
    cuda_free: int | None = None
    cuda_total: int | None = None
    if cuda_available:
        cuda_free, cuda_total = (int(value) for value in torch.cuda.mem_get_info(0))
    return {
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "workspace_filesystem": {
            "path": str(Path.cwd().resolve()),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        },
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_free_bytes_at_report": cuda_free,
        "cuda_total_bytes": cuda_total,
    }


def _throughput(task_name: str, steps: int, seed: int) -> dict[str, Any]:
    from experiments.dmc.tasks.adapter import make_dmc_adapter

    spec = get_task_spec(task_name)
    env = make_dmc_adapter(task_name, seed=seed)
    rng = np.random.default_rng(seed)
    completed = 0
    started = time.perf_counter()
    try:
        env.reset(seed=seed)
        for index in range(steps):
            _, _, done, _ = env.step(
                rng.uniform(-1.0, 1.0, size=spec.action_dim)
            )
            completed += 1
            if done and index + 1 < steps:
                env.reset(seed=seed + index + 1)
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    return {
        "steps": completed,
        "elapsed_seconds": elapsed,
        "environment_steps_per_second": completed / max(elapsed, 1e-12),
    }


def run_preflight(
    config_path: Path,
    *,
    profile: str = "development",
    parity_steps: int = 5,
    throughput_steps: int = 200,
    env_workers: int | None = None,
) -> dict[str, Any]:
    preflight_started = time.perf_counter()
    if isinstance(parity_steps, bool) or not isinstance(parity_steps, int) or parity_steps < 1:
        raise ValueError("parity_steps must be a positive integer")
    if (
        isinstance(throughput_steps, bool)
        or not isinstance(throughput_steps, int)
        or throughput_steps < 1
    ):
        raise ValueError("throughput_steps must be a positive integer")
    maximum_parity_steps = min(
        get_task_spec(task_name).native_step_limit for task_name in ALL_TASK_ORDER
    )
    if parity_steps > maximum_parity_steps:
        raise ValueError(
            f"parity_steps must not exceed {maximum_parity_steps} without "
            "explicit reset handling"
        )
    config = load_experiment_config(config_path)
    if profile not in PROFILE_NAMES:
        raise ValueError(f"profile must be one of {PROFILE_NAMES}")
    execution_spec = resolve_execution_spec(config, profile)
    live_specs: dict[str, Any] = {}
    parity: dict[str, Any] = {}
    for task_name in ALL_TASK_ORDER:
        live_specs[task_name] = verify_task_spec(task_name)
        parity[task_name] = _suite_parity(task_name, seed=73, steps=parity_steps)
    timeout = _timeout_check(config.task)
    actors = _actor_forward_check(config.task, config, profile=profile)
    mpve = _mpve_forward_check(config.task, config, profile=profile)
    reward_source_parity = _mpve_reward_source_parity(
        config.task,
        source=str(config.raw["ppo"]["mpve_reward_source"]),
        seed=77,
        steps=parity_steps,
    )
    from experiments.dmc.tasks.adapter import make_dmc_adapter

    selected_env = make_dmc_adapter(config.task, seed=79)
    try:
        environment_protocol = selected_env.protocol_metadata()
    finally:
        selected_env.close()
    throughput = _throughput(config.task, throughput_steps, seed=81)
    throughput_valid = bool(
        throughput["steps"] == throughput_steps
        and np.isfinite(throughput["elapsed_seconds"])
        and throughput["elapsed_seconds"] > 0
        and np.isfinite(throughput["environment_steps_per_second"])
        and throughput["environment_steps_per_second"] > 0
    )
    vector_rollout = _vector_rollout_probe(
        config.task,
        config,
        profile=profile,
        env_workers=env_workers,
    )
    selected_protocol_fingerprint = protocol_fingerprint(environment_protocol)
    vector_protocol_matches = bool(
        vector_rollout["protocol_fingerprint"] == selected_protocol_fingerprint
    )
    vector_rollout = {
        **vector_rollout,
        "protocol_matches_selected_environment": vector_protocol_matches,
        "passed": bool(vector_rollout["passed"] and vector_protocol_matches),
    }
    storage = _storage_upper_bound(config, profile=profile)
    total_training_steps = int(
        config.raw["profiles"][profile]["total_timesteps"]
    )
    vector_rate = float(
        vector_rollout["environment_transitions_per_step_second"]
    )
    environment_only_seconds = total_training_steps / max(vector_rate, 1e-12)
    environment_only_lower_bound = {
        "kind": "environment_step_time_lower_bound_v1",
        "description": (
            "Configured vector-environment stepping only; excludes policy, "
            "critic, Koopman/MPC, backpropagation, evaluation and I/O."
        ),
        "training_steps_per_seed": total_training_steps,
        "measured_environment_transitions_per_second": vector_rate,
        "estimated_seconds_per_training_seed": environment_only_seconds,
    }
    checks_passed = bool(
        all(value["passed"] for value in parity.values())
        and timeout["passed"]
        and all(value["passed"] for value in actors.values())
        and mpve["passed"]
        and reward_source_parity["passed"]
        and throughput_valid
        and vector_rollout["passed"]
    )
    hardware = _hardware_inventory()
    reviewed_source_identity = source_identity()
    preflight_wall_seconds = time.perf_counter() - preflight_started
    return {
        "kind": "dmc_training_free_preflight",
        "ready_for_user_review": checks_passed,
        "training_approved": False,
        "config_path": str(config_path.resolve()),
        "config_fingerprint": config.fingerprint,
        "task": config.task,
        "profile": profile,
        "resolved_execution_spec": execution_spec,
        "protocol": config.protocol,
        "environment_protocol": environment_protocol,
        "protocol_fingerprint": selected_protocol_fingerprint,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "dm_control": _version("dm-control"),
            "mujoco": _version("mujoco"),
        },
        "hardware": hardware,
        "preflight_wall_seconds": preflight_wall_seconds,
        "source_identity": reviewed_source_identity,
        "live_specs": live_specs,
        "suite_parity": parity,
        "timeout_semantics": timeout,
        "actor_forward": actors,
        "mpve_critic_only_forward": mpve,
        "mpve_reward_source_parity": reward_source_parity,
        "single_env_throughput": {**throughput, "passed": throughput_valid},
        "configured_vector_rollout": vector_rollout,
        "environment_only_wall_time_lower_bound": environment_only_lower_bound,
        "storage_upper_bound": storage,
        "review_required_for": {
            "selected_profile": config.raw["profiles"][profile],
            "train_seeds": config.raw["seeds"]["train"],
            "evaluation_seeds": config.raw["seeds"]["evaluation"],
            "data": config.raw["data"],
            "koopman": config.raw["koopman"],
            "actors": config.raw["actors"],
            "evaluation": config.raw["evaluation"],
            "proposed_gates": config.raw["proposed_gates"],
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="cartpole_swingup")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="development")
    parser.add_argument("--parity-steps", type=int, default=5)
    parser.add_argument("--throughput-steps", type=int, default=200)
    parser.add_argument(
        "--env-workers",
        type=int,
        default=None,
        help="execution-only DMC CPU worker count (default: env or 16)",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = parse_args()
    config_path = args.config or default_config_path(args.task)
    report = run_preflight(
        config_path,
        profile=args.profile,
        parity_steps=args.parity_steps,
        throughput_steps=args.throughput_steps,
        env_workers=args.env_workers,
    )
    output = args.output or Path("runs/dmc/preflight") / f"{report['task']}.json"
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"[preflight] wrote {output}")
    if not report["ready_for_user_review"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
