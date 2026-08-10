from __future__ import annotations

from pathlib import Path

import torch

from antmaze_ac.koopman.checkpoint import load_checkpoint

from .ac_koopman_policy import KoopmanLQRPolicy
from .cost_actor import CostActor
from .critic import Critic
from .delta_policy import DeltaPolicy


def make_policy(
    koopman_checkpoint: str | Path,
    device: torch.device,
    *,
    mean_action_limit: float | None = None,
    policy_observation_dim: int | None = None,
    implicit_dare_backward: bool = False,
    training_dare_spectral_radius_diagnostics: bool = True,
) -> tuple[KoopmanLQRPolicy, dict]:
    koopman, payload = load_checkpoint(koopman_checkpoint, map_location=device)
    config = payload["config"]
    actor_config = config["actor"]
    control = config["control"]
    # Enrich checkpoints produced before solver recovery was added. The
    # resulting Actor-Critic checkpoint records the effective values explicitly.
    recovery_defaults = {
        "dare_retry_max_iterations": 1000,
        "dare_retry_jitter_multiplier": 100.0,
        "dare_fallback_state_cost": 1.0,
        "dare_fallback_control_cost": 1.0,
        "dare_fallback_delta_limit": 1.0,
        "dare_max_fallback_fraction_per_rollout": 0.05,
        "dare_max_consecutive_failure_rollouts": 3,
    }
    for key, value in recovery_defaults.items():
        control.setdefault(key, value)
    state_stats = payload["normalizers"]["state"]
    policy_observation_dim = int(
        koopman.state_dim
        if policy_observation_dim is None
        else policy_observation_dim
    )
    if policy_observation_dim < koopman.state_dim:
        raise ValueError(
            "policy_observation_dim must be at least the Koopman state dimension"
        )
    extra_observation_dim = policy_observation_dim - koopman.state_dim
    actor = CostActor(
        koopman.state_dim,
        koopman.action_dim,
        actor_config["hidden_dims"],
        control["stage_cost_epsilon"],
        control["q_max"],
        control["p_max"],
        previous_action_dim=koopman.action_dim,
        previous_action_cost_scale=control["previous_action_cost_scale"],
        delta_action_cost_scale=control["delta_action_cost_scale"],
        activation=actor_config["activation"],
        observation_dim=policy_observation_dim,
    )
    critic = Critic(
        policy_observation_dim,
        actor_config["critic_hidden_dims"],
        actor_config["critic_activation"],
    )
    state_mean = torch.tensor(state_stats["mean"], dtype=torch.float32)
    state_std = torch.tensor(state_stats["std"], dtype=torch.float32)
    if extra_observation_dim:
        # Optional extra observation features are expected to be normalized.
        state_mean = torch.cat((state_mean, torch.zeros(extra_observation_dim)))
        state_std = torch.cat((state_std, torch.ones(extra_observation_dim)))
    policy = KoopmanLQRPolicy(
        koopman,
        actor,
        critic,
        state_mean,
        state_std,
        log_std_init=actor_config["log_std_init"],
        dare_tolerance=control["dare_tolerance"],
        dare_max_iterations=control["dare_max_iterations"],
        dare_jitter=control["dare_jitter"],
        fail_on_nonconvergence=control["dare_fail_on_nonconvergence"],
        retry_max_iterations=control["dare_retry_max_iterations"],
        retry_jitter_multiplier=control["dare_retry_jitter_multiplier"],
        fallback_state_cost=control["dare_fallback_state_cost"],
        fallback_control_cost=control["dare_fallback_control_cost"],
        fallback_delta_limit=control["dare_fallback_delta_limit"],
        mean_action_limit=mean_action_limit,
        implicit_dare_backward=implicit_dare_backward,
        training_dare_spectral_radius_diagnostics=(
            training_dare_spectral_radius_diagnostics
        ),
    ).to(device)
    return policy, payload


def load_actor_checkpoint(path: str | Path, device: torch.device):
    actor_payload = torch.load(path, map_location=device, weights_only=False)
    runtime = actor_payload.get("runtime", {})
    policy, koopman_payload = make_policy(
        actor_payload["koopman_checkpoint"],
        device,
        policy_observation_dim=runtime.get("policy_observation_dim"),
    )
    policy.load_state_dict(actor_payload["policy"])
    return policy, actor_payload, koopman_payload


def load_delta_checkpoint(path: str | Path, device: torch.device):
    delta_payload = torch.load(path, map_location=device, weights_only=False)
    if delta_payload.get("method") != "delta_ppo":
        raise ValueError(f"{path} is not a Delta-PPO checkpoint")
    koopman_payload = torch.load(
        delta_payload["koopman_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    architecture = koopman_payload["architecture"]
    state_stats = koopman_payload["normalizers"]["state"]
    baseline = delta_payload["config"]["delta_ppo_baseline"]
    policy = DeltaPolicy(
        architecture["state_dim"],
        architecture["action_dim"],
        torch.tensor(state_stats["mean"], dtype=torch.float32),
        torch.tensor(state_stats["std"], dtype=torch.float32),
        baseline["hidden_dims"],
        baseline["log_std_init"],
        baseline["activation"],
    ).to(device)
    policy.load_state_dict(delta_payload["policy"])
    return policy, delta_payload, koopman_payload


def load_td3_bc_checkpoint(path: str | Path, device: torch.device):
    td3_payload = torch.load(path, map_location=device, weights_only=False)
    if td3_payload.get("method") != "td3_bc_koopman_lqr":
        raise ValueError(f"{path} is not a Koopman-LQR TD3+BC checkpoint")
    action_limit = float(td3_payload["runtime"]["max_delta_action"])
    runtime = td3_payload.get("runtime", {})
    policy, koopman_payload = make_policy(
        td3_payload["koopman_checkpoint"],
        device,
        mean_action_limit=action_limit,
        # Checkpoints predating the optimized solver used explicit backward and
        # full eigvalue diagnostics, so those are the compatibility defaults.
        implicit_dare_backward=bool(
            runtime.get("implicit_dare_backward", False)
        ),
        training_dare_spectral_radius_diagnostics=bool(
            runtime.get(
                "training_dare_spectral_radius_diagnostics",
                True,
            )
        ),
    )
    policy.load_state_dict(td3_payload["policy"])
    return policy, td3_payload, koopman_payload
