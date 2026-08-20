from __future__ import annotations

from pathlib import Path

import torch

from antmaze_ac.koopman.checkpoint import load_checkpoint, sha256

from .ac_koopman_policy import KoopmanLQRPolicy
from .cost_actor import CostActor
from .critic import Critic
from .delta_policy import DeltaPolicy
from .history_mlp_policy import HistoryMLPActor, HistoryMLPPolicy
from .history_koopman_mpc_policy import HistoryKoopmanMPCPolicy
from .koopman_mpc_actor import KoopmanMPCActor


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


def make_history_mpc_policy(
    koopman_checkpoint: str | Path,
    device: torch.device,
    *,
    horizon: int | None = None,
    absolute_action_limit: float | None = None,
    solver_iterations: int | None = None,
    quadratic_log_scale: float | None = None,
    linear_scale: float | None = None,
    action_quadratic_scale: float | None = None,
    waypoint_count: int = 1,
    task_mode: str = "tracking",
) -> tuple[HistoryKoopmanMPCPolicy, dict]:
    """Build the soft-robot BC-KMPC policy from a history checkpoint."""

    koopman, payload = load_checkpoint(koopman_checkpoint, map_location=device)
    from antmaze_ac.koopman.history_model import HistoryDeepKoopman

    if not isinstance(koopman, HistoryDeepKoopman):
        raise ValueError(
            "BC-KMPC requires a fullA_history_context_v1 Koopman checkpoint"
        )
    config = payload["config"]
    settings = {
        "horizon": 10,
        "hidden_dims": [256, 256],
        "activation": "gelu",
        "quadratic_log_scale": 1.5,
        "linear_scale": 10.0,
        "action_quadratic_scale": 1.0,
        "solver_iterations": 20,
        "step_fraction": 0.95,
        "solver_epsilon": 1e-6,
        "absolute_action_limit": 0.30,
        # Match the reference PPO's physical-action standard deviation ~= 0.05.
        "log_std_init": -3.0,
        "critic_hidden_dims": [256, 256],
        "critic_activation": "gelu",
    }
    settings.update(config.get("bc_kmpc", {}))
    # Older Koopman checkpoints may embed the retired BC-KMPC setting. It is
    # deliberately ignored: the current actor contains no fixed smoothness.
    settings.pop("smoothness_weight", None)
    settings.pop("max_delta", None)
    if horizon is not None:
        settings["horizon"] = int(horizon)
    if absolute_action_limit is not None:
        settings["absolute_action_limit"] = float(absolute_action_limit)
    if solver_iterations is not None:
        settings["solver_iterations"] = int(solver_iterations)
    if quadratic_log_scale is not None:
        settings["quadratic_log_scale"] = float(quadratic_log_scale)
    if linear_scale is not None:
        settings["linear_scale"] = float(linear_scale)
    if action_quadratic_scale is not None:
        settings["action_quadratic_scale"] = float(action_quadratic_scale)

    if task_mode not in {"tracking", "kinematic_push"}:
        raise ValueError(f"Unsupported history MPC task_mode: {task_mode}")
    if waypoint_count < 1:
        raise ValueError("waypoint_count must be positive")
    task_context_dim = (
        HistoryKoopmanMPCPolicy.KINEMATIC_PUSH_TASK_CONTEXT_DIM
        if task_mode == "kinematic_push"
        else (
            HistoryKoopmanMPCPolicy.TASK_CONTEXT_DIM
            if waypoint_count == 1
            else 4 * int(waypoint_count)
        )
    )
    limit = float(settings["absolute_action_limit"])
    actor = KoopmanMPCActor(
        koopman.A,
        koopman.B,
        koopman.C[: koopman.state_dim],
        horizon=int(settings["horizon"]),
        context_dim=task_context_dim,
        hidden_dims=settings["hidden_dims"],
        activation=str(settings["activation"]),
        action_low=-limit,
        action_high=limit,
        quadratic_log_scale=float(settings["quadratic_log_scale"]),
        linear_scale=float(settings["linear_scale"]),
        action_quadratic_scale=float(settings["action_quadratic_scale"]),
        solver_iterations=int(settings["solver_iterations"]),
        step_fraction=float(settings["step_fraction"]),
        solver_epsilon=float(settings["solver_epsilon"]),
    )
    critic = Critic(
        koopman.lifted_dim + task_context_dim,
        settings["critic_hidden_dims"],
        str(settings["critic_activation"]),
    )
    state_stats = payload["normalizers"]["state"]
    policy = HistoryKoopmanMPCPolicy(
        koopman,
        actor,
        critic,
        torch.as_tensor(state_stats["mean"], dtype=torch.float32),
        torch.as_tensor(state_stats["std"], dtype=torch.float32),
        waypoint_count=int(waypoint_count),
        task_mode=task_mode,
        log_std_init=float(settings["log_std_init"]),
    ).to(device)
    return policy, payload


def load_history_mpc_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[HistoryKoopmanMPCPolicy, dict, dict]:
    """Load either a BC initialization or PPO-fine-tuned BC-KMPC policy."""

    policy_payload = torch.load(path, map_location=device, weights_only=False)
    method = policy_payload.get("method")
    if method not in {
        "bc_kmpc_bc",
        "actor_critic_bc_kmpc",
        "kinematic_push_bc_kmpc",
        "actor_critic_kinematic_push",
    }:
        raise ValueError(f"{path} is not a BC-KMPC checkpoint")
    if int(policy_payload.get("format_version", 0)) < 4:
        raise ValueError(
            "BC-KMPC checkpoint predates absolute-action box FISTA and is incompatible"
        )
    koopman_checkpoint = Path(policy_payload["koopman_checkpoint"])
    if not koopman_checkpoint.is_file():
        raise FileNotFoundError(
            f"BC-KMPC Koopman checkpoint is missing: {koopman_checkpoint}"
        )
    expected_sha = policy_payload.get("koopman_checkpoint_sha256")
    if expected_sha is not None and sha256(koopman_checkpoint) != expected_sha:
        raise ValueError("BC-KMPC Koopman checkpoint SHA256 does not match")
    runtime = policy_payload.get("runtime", {})
    policy, koopman_payload = make_history_mpc_policy(
        koopman_checkpoint,
        device,
        horizon=runtime.get("horizon"),
        absolute_action_limit=runtime.get("absolute_action_limit"),
        solver_iterations=runtime.get("solver_iterations"),
        quadratic_log_scale=runtime.get("quadratic_log_scale"),
        linear_scale=runtime.get("linear_scale"),
        action_quadratic_scale=runtime.get("action_quadratic_scale"),
        waypoint_count=int(runtime.get("waypoint_count", 1)),
        task_mode=str(runtime.get("task_mode", "tracking")),
    )
    if method in {"bc_kmpc_bc", "kinematic_push_bc_kmpc"}:
        policy.actor.load_state_dict(policy_payload["actor"])
    else:
        policy.load_state_dict(policy_payload["policy"])
    return policy, policy_payload, koopman_payload


def make_history_mlp_policy(
    koopman_checkpoint: str | Path,
    device: torch.device,
    *,
    absolute_action_limit: float | None = None,
    max_delta: float | None = None,
) -> tuple[HistoryMLPPolicy, dict]:
    """Build the Koopman-free history MLP baseline from shared metadata."""

    koopman, payload = load_checkpoint(koopman_checkpoint, map_location=device)
    from antmaze_ac.koopman.history_model import HistoryDeepKoopman

    if not isinstance(koopman, HistoryDeepKoopman):
        raise ValueError(
            "History-MLP requires a fullA_history_context_v1 checkpoint "
            "for dimensions and train-split normalization"
        )
    settings = {
        "hidden_dims": [256, 256],
        "activation": "tanh",
        "absolute_action_limit": 0.30,
        "max_delta": 0.001,
        "log_std_init": -0.5,
        "critic_hidden_dims": [256, 256],
        "critic_activation": "tanh",
    }
    settings.update(payload["config"].get("history_mlp_baseline", {}))
    if absolute_action_limit is not None:
        settings["absolute_action_limit"] = float(absolute_action_limit)
    if max_delta is not None:
        settings["max_delta"] = float(max_delta)

    feature_dim = koopman.context_dim + HistoryMLPPolicy.TASK_CONTEXT_DIM
    limit = float(settings["absolute_action_limit"])
    actor = HistoryMLPActor(
        feature_dim,
        koopman.action_dim,
        settings["hidden_dims"],
        str(settings["activation"]),
        action_low=-limit,
        action_high=limit,
        max_delta=float(settings["max_delta"]),
    )
    critic = Critic(
        feature_dim,
        settings["critic_hidden_dims"],
        str(settings["critic_activation"]),
    )
    state_stats = payload["normalizers"]["state"]
    policy = HistoryMLPPolicy(
        actor,
        critic,
        torch.as_tensor(state_stats["mean"], dtype=torch.float32),
        torch.as_tensor(state_stats["std"], dtype=torch.float32),
        state_dim=koopman.state_dim,
        action_dim=koopman.action_dim,
        history_steps=koopman.history_steps,
        log_std_init=float(settings["log_std_init"]),
    ).to(device)
    return policy, payload


def load_history_mlp_checkpoint(
    path: str | Path,
    device: torch.device,
) -> tuple[HistoryMLPPolicy, dict, dict]:
    """Load a BC-pretrained or PPO-fine-tuned history MLP baseline."""

    policy_payload = torch.load(path, map_location=device, weights_only=False)
    method = policy_payload.get("method")
    if method not in {"history_mlp_bc", "actor_critic_history_mlp"}:
        raise ValueError(f"{path} is not a History-MLP checkpoint")
    koopman_checkpoint = Path(policy_payload["koopman_checkpoint"])
    if not koopman_checkpoint.is_file():
        raise FileNotFoundError(
            f"History-MLP metadata checkpoint is missing: {koopman_checkpoint}"
        )
    expected_sha = policy_payload.get("koopman_checkpoint_sha256")
    if expected_sha is not None and sha256(koopman_checkpoint) != expected_sha:
        raise ValueError("History-MLP metadata checkpoint SHA256 does not match")
    runtime = policy_payload.get("runtime", {})
    policy, koopman_payload = make_history_mlp_policy(
        koopman_checkpoint,
        device,
        absolute_action_limit=runtime.get("absolute_action_limit"),
        max_delta=runtime.get("max_delta"),
    )
    if method == "history_mlp_bc":
        policy.actor.load_state_dict(policy_payload["actor"])
    else:
        policy.load_state_dict(policy_payload["policy"])
    return policy, policy_payload, koopman_payload
