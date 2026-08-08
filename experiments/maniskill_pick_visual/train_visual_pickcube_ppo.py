"""Vectorized PPO for the visual PickCube task with four actor routes.

The standard ``PPO`` route mirrors the official ManiSkill3 PPO RGB baseline
(``examples/baselines/ppo/ppo_rgb.py``): a NatureCNN visual encoder, rgb
normalized by ``/255``, a ``state`` observation that is the full flattened
``agent`` + ``extra`` (including the privileged goal), orthogonal init, an
actor log-std initialized to ``-0.5``, ``normalized_dense`` reward,
``update_epochs=8`` and ``batch / 8`` minibatches.  Only hardware-bound knobs
(``num_envs``, ``minibatch_size``, ``total_timesteps``) are expected to be
lowered when running on a single GPU.

The AC-MPC routes share the same NatureCNN rgb features but feed them through
a context head into structured actors on the frozen robot Koopman:
  * ``KLQR`` : KoopmanLQRActor (closed-form DARE gain).
  * ``KMPC`` : KoopmanMPCActor (differentiable condensed box-QP).
  * ``AB-PQ``  : LowRankValueActor (low-rank quadratic value -> box-QP).
``extra/goal_pos`` is used only for a training-time pos_branch auxiliary loss
on the context (privileged, never an input for the AC-MPC routes).

Mid-training artifacts (project convention): ``recovery_update_XXXXX.pt``
every ``checkpoint_interval_updates`` updates and a ``metrics.jsonl`` appended
and flushed every update.

Run a smoke (tiny, verifies the pipeline end to end):
  python -m experiments.maniskill_pick_visual.train_visual_pickcube_ppo \
      --actor PPO --koopman runs/pickcube_robot_koopman/best.pt \
      --output-dir runs/visual_pickcube_ppo_smoke/PPO --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor
from antmaze_ac.rl.quadratic_actors import KoopmanLQRActor, LowRankValueActor
from antmaze_ac.koopman.model import DeepKoopman
from experiments.maniskill_pick_visual.visual_pick_cube import (
    VisualPickCubeEnv,  # registers ACMPC-VisualPickCube-v1
)

ACTOR_NAMES = ("PPO", "KLQR", "KMPC", "AB-PQ")
ROBOT_DIM = 21
ACTION_DIM = 8
# Official PPO "state" = qpos9 + qvel9 + is_grasped1 + tcp_pose7 + goal_pos3.
STATE_DIM = 29


# --------------------------------------------------------------------------- #
# Official NatureCNN + PPO agent (mirrors examples/baselines/ppo/ppo_rgb.py)
# --------------------------------------------------------------------------- #
def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """RGB -> NatureCNN (conv 3->32 k8s4, 32->64 k4s2, 64->64 k3s1, fc 256).

    rgb is normalized by /255 like the official baseline.  A separate state
    extractor maps the flattened state to 256.  ``forward`` returns the
    concatenated rgb+state features; ``forward_rgb`` returns only the rgb part
    (used by the AC-MPC context head).
    """

    def __init__(self, rgb_channels: int = 3, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, rgb_channels, 128, 128)
            n_flatten = self.cnn(dummy).shape[1]
        self.rgb_fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 256)), nn.ReLU()
        )
        self.state_mlp = layer_init(nn.Linear(state_dim, 256))
        self.out_features = 256 + 256

    def forward_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim == 4 and rgb.shape[-1] == 3:
            image = rgb.float().permute(0, 3, 1, 2)
        else:
            image = rgb.float()
        image = image / 255.0
        return self.rgb_fc(self.cnn(image))

    def forward(
        self, rgb: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        rgb_feature = self.forward_rgb(rgb)
        return torch.cat((rgb_feature, self.state_mlp(state)), dim=-1)


class PPORoute(nn.Module):
    """Official PPO Agent: NatureCNN -> actor_mean/critic, log-std init -0.5."""

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.feature_net = NatureCNN(rgb_channels=3, state_dim=state_dim)
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, ACTION_DIM), std=0.01 * np.sqrt(2)
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, ACTION_DIM) * -0.5
        )

    def get_action_and_value(
        self,
        rgb: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        features = self.feature_net(rgb, state)
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(-1),
            probs.entropy().sum(-1),
            self.critic(features).squeeze(-1),
        )

    def get_value(self, rgb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.critic(self.feature_net(rgb, state)).squeeze(-1)


class ContextHead(nn.Module):
    """NatureCNN rgb features -> task context (goal estimate) + pos_branch."""

    def __init__(
        self,
        feature_dim: int = 256,
        v_dim: int = 3,
        pos_dim: int = 3,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, v_dim),
        )
        self.pos_branch = nn.Linear(v_dim, pos_dim)
        nn.init.zeros_(self.pos_branch.weight)
        nn.init.zeros_(self.pos_branch.bias)

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.context(features)
        return context, self.pos_branch(context)


class ValueNetwork(nn.Module):
    """Value on the frozen Koopman lift + context (KLQR / KMPC / AB-PQ)."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, lifted: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((lifted, context), dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 10_000_000
    num_envs: int = 32
    rollout_steps: int = 16
    minibatch_size: int | None = None  # default batch_size // 8 (official)
    update_epochs: int = 8
    learning_rate: float = 3e-4
    anneal_learning_rate: bool = True
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    clip_value_loss: bool = True
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = None  # official PPO does not early-stop on KL
    checkpoint_interval_updates: int = 10
    max_wall_time_seconds: float | None = None
    max_episode_steps: int = 50
    reward_mode: str = "normalized_dense"
    v_dim: int = 3
    context_hidden_dim: int = 128
    kmpc_horizon: int = 10
    kmpc_solver_iterations: int = 20
    ab_rank: int = 16
    # Auxiliary pos_branch supervision on extra/goal_pos (privileged, train-only).
    pos_weight: float = 0.5
    seed: int = 20_280_804

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    def effective_minibatch_size(self) -> int:
        if self.minibatch_size is not None:
            return self.minibatch_size
        return max(1, self.batch_size // 8)

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.num_envs <= 0 or self.rollout_steps <= 0:
            raise ValueError("num_envs and rollout_steps must be positive")
        mini = self.effective_minibatch_size()
        if not 0 < mini <= self.batch_size:
            raise ValueError("minibatch_size must be in [1, batch_size]")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.checkpoint_interval_updates <= 0:
            raise ValueError("checkpoint_interval_updates must be positive")
        if self.pos_weight < 0:
            raise ValueError("pos_weight must be non-negative")
        if self.reward_mode not in {"dense", "normalized_dense", "sparse"}:
            raise ValueError(f"Unsupported reward_mode {self.reward_mode!r}")


# --------------------------------------------------------------------------- #
# Environment + feature extraction
# --------------------------------------------------------------------------- #
def _as_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _make_env(config: PPOConfig):
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    base = gym.make(
        "ACMPC-VisualPickCube-v1",
        num_envs=config.num_envs,
        sim_backend="gpu" if torch.cuda.is_available() else "cpu",
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode=config.reward_mode,
        render_mode=None,
        max_episode_steps=config.max_episode_steps,
    )
    return ManiSkillVectorEnv(
        base,
        config.num_envs,
        auto_reset=True,
        ignore_terminations=False,
        record_metrics=True,
    )


def _load_koopman(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    kind = str(payload.get("kind", ""))
    if kind != "pickcube_robot_k_step_koopman":
        raise ValueError(f"Expected pickcube robot Koopman, got {kind!r}")
    model = DeepKoopman(
        state_dim=int(payload["robot_dim"]),
        action_dim=int(payload["action_dim"]),
        lift_dim=int(payload["config"]["lift_dim"]),
        hidden_dims=tuple(payload["config"]["hidden_dims"]),
        activation="silu",
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    center = _as_tensor(payload["normalizer"]["center"], device).float()
    scale = _as_tensor(payload["normalizer"]["scale"], device).float()
    return model, center, scale


def _extract(
    observation: dict[str, Any],
    koopman: nn.Module,
    state_center: torch.Tensor,
    state_scale: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (rgb, robot21 normalized, state29, lifted).

    robot21 = qpos9 + qvel9 + tcp_xyz3 (normalized by the Koopman normalizer).
    state29 = qpos9 + qvel9 + is_grasped + tcp_pose7 + goal_pos3 (raw floats,
    matching the official flattened observation; used only by the PPO route).
    """
    rgb = _as_tensor(observation["sensor_data"]["base_camera"]["rgb"], device)
    qpos = _as_tensor(observation["agent"]["qpos"], device).float()[..., :9]
    qvel = _as_tensor(observation["agent"]["qvel"], device).float()[..., :9]
    tcp_pose = _as_tensor(observation["extra"]["tcp_pose"], device).float()
    is_grasped = _as_tensor(
        observation["extra"]["is_grasped"], device
    ).float().reshape(-1, 1)
    goal_pos = _as_tensor(observation["extra"]["goal_pos"], device).float()[..., :3]
    robot = torch.cat((qpos, qvel, tcp_pose[..., :3]), dim=-1)
    normalized_robot = (robot - state_center) / state_scale
    state29 = torch.cat((qpos, qvel, is_grasped, tcp_pose, goal_pos), dim=-1)
    with torch.no_grad():
        lifted = koopman.lift(normalized_robot)
    return rgb, normalized_robot, state29, lifted


# --------------------------------------------------------------------------- #
# Actor / value / context dispatch
# --------------------------------------------------------------------------- #
def _build_actor(
    actor_name: str,
    koopman: nn.Module,
    config: PPOConfig,
    device: torch.device,
) -> nn.Module:
    if actor_name == "PPO":
        return PPORoute(state_dim=STATE_DIM).to(device)
    if actor_name == "KLQR":
        return KoopmanLQRActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            context_dim=config.v_dim,
            hidden_dims=(config.context_hidden_dim,),
            max_action=1.0,
        ).to(device)
    if actor_name == "KMPC":
        return KoopmanMPCActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            horizon=config.kmpc_horizon,
            context_dim=config.v_dim,
            hidden_dims=(config.context_hidden_dim,),
            action_low=-1.0,
            action_high=1.0,
            solver_iterations=config.kmpc_solver_iterations,
        ).to(device)
    if actor_name == "AB-PQ":
        return LowRankValueActor(
            observation_dim=koopman.lifted_dim + config.v_dim,
            A=koopman.A,
            B=koopman.B,
            R=torch.eye(ACTION_DIM, device=device, dtype=koopman.A.dtype),
            base_hessian=torch.eye(
                koopman.lifted_dim, device=device, dtype=koopman.A.dtype
            ),
            rank=config.ab_rank,
            hidden_dims=(config.context_hidden_dim,),
            value_linear_scale=1.0,
            max_action=1.0,
        ).to(device)
    raise ValueError(f"Unsupported actor {actor_name!r}")


def _build_value(
    actor_name: str,
    koopman: nn.Module,
    config: PPOConfig,
) -> nn.Module | None:
    if actor_name == "PPO":
        return None  # value lives inside PPORoute
    return ValueNetwork(koopman.lifted_dim + config.v_dim)


def _structured_mean(
    actor_name: str,
    actor: nn.Module,
    lifted: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    if actor_name in ("KLQR", "KMPC"):
        return actor(lifted, context).action
    return actor(torch.cat((lifted, context), dim=-1), lifted).action


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _optional_mean(values: deque[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _fmt(value: float | None) -> str:
    return "nan" if value is None else f"{value:.4g}"


def train(
    actor_name: str,
    koopman_path: Path,
    output_dir: Path,
    config: PPOConfig,
    device_name: str = "auto",
    smoke: bool = False,
) -> dict[str, Any]:
    config_used = config
    if smoke:
        config_used = PPOConfig(
            total_timesteps=4 * 128,
            num_envs=4,
            rollout_steps=32,
            minibatch_size=32,
            update_epochs=2,
            learning_rate=3e-4,
            entropy_coefficient=0.0,
            target_kl=None,
            checkpoint_interval_updates=2,
            seed=config.seed,
            max_episode_steps=config.max_episode_steps,
            reward_mode=config.reward_mode,
            v_dim=config.v_dim,
            context_hidden_dim=config.context_hidden_dim,
            kmpc_horizon=config.kmpc_horizon,
            kmpc_solver_iterations=config.kmpc_solver_iterations,
            ab_rank=config.ab_rank,
            pos_weight=config.pos_weight,
        )
    config_used.validate()
    if actor_name not in ACTOR_NAMES:
        raise ValueError(f"Unsupported actor {actor_name!r}")
    random.seed(config_used.seed)
    np.random.seed(config_used.seed)
    torch.manual_seed(config_used.seed)
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config_used.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    koopman, state_center, state_scale = _load_koopman(koopman_path, device)
    actor = _build_actor(actor_name, koopman, config_used, device)
    value = _build_value(actor_name, koopman, config_used)
    if value is not None:
        value = value.to(device)
    if actor_name == "PPO":
        context_head: ContextHead | None = None
        log_std: nn.Parameter | None = None
        encoder = actor.feature_net
        encoder_parameters: list[nn.Parameter] = []
        auxiliary_parameters: list[nn.Parameter] = []
    else:
        context_head = ContextHead(
            256, config_used.v_dim, hidden_dim=config_used.context_hidden_dim
        ).to(device)
        encoder = NatureCNN(rgb_channels=3, state_dim=STATE_DIM).to(device)
        log_std = nn.Parameter(
            torch.full((ACTION_DIM,), math.log(0.3), device=device)
        )
        assert value is not None
        encoder_parameters = list(encoder.parameters()) + list(
            context_head.parameters()
        )
        auxiliary_parameters = [*value.parameters(), log_std]
    actor_parameters = list(actor.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": actor_parameters, "lr": config_used.learning_rate},
            {
                "params": encoder_parameters + auxiliary_parameters,
                "lr": config_used.learning_rate,
            },
        ],
        eps=1e-5,
    )

    env = _make_env(config_used)
    observation, _ = env.reset(seed=config_used.seed)
    number_updates = math.ceil(config_used.total_timesteps / config_used.batch_size)
    minibatch_size = config_used.effective_minibatch_size()
    metrics_path = output_dir / "metrics.jsonl"
    latest_path = output_dir / "latest.pt"
    episode_returns: deque[float] = deque(maxlen=100)
    episode_lengths: deque[float] = deque(maxlen=100)
    episode_successes: deque[float] = deque(maxlen=100)
    started = time.perf_counter()
    wall_time_reached = False
    metadata = {
        "kind": "visual_pickcube_ppo",
        "actor_name": actor_name,
        "smoke": bool(smoke),
        "official_baseline_alignment": actor_name == "PPO",
        "koopman_path": str(koopman_path.resolve()),
        "device": str(device),
        "config": asdict(config_used),
        "trainable_parameters": sum(
            p.numel()
            for p in encoder_parameters + actor_parameters + auxiliary_parameters
            if p.requires_grad
        ),
    }
    _atomic_json(output_dir / "run_config.json", metadata)

    try:
        for update in range(1, number_updates + 1):
            if config_used.anneal_learning_rate:
                fraction = 1.0 - (update - 1.0) / number_updates
                optimizer.param_groups[0]["lr"] = (
                    fraction * config_used.learning_rate
                )
                optimizer.param_groups[1]["lr"] = (
                    fraction * config_used.learning_rate
                )
            robot_states: list[torch.Tensor] = []
            lifted_states: list[torch.Tensor] = []
            states29: list[torch.Tensor] = []
            rgb_list: list[torch.Tensor] = []
            contexts: list[torch.Tensor] = []
            pos_targets: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            log_probabilities: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            dones: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for _ in range(config_used.rollout_steps):
                rgb, normalized_robot, state29, lifted = _extract(
                    observation,
                    koopman,
                    state_center,
                    state_scale,
                    device,
                )
                with torch.no_grad():
                    if actor_name == "PPO":
                        action, log_probability, _, state_value = (
                            actor.get_action_and_value(rgb, state29)
                        )
                        pos = torch.zeros(rgb.shape[0], 3, device=device)
                        context = torch.zeros(
                            rgb.shape[0], config_used.v_dim, device=device
                        )
                    else:
                        assert context_head is not None and log_std is not None
                        features = encoder.forward_rgb(rgb)
                        context, pos = context_head(features)
                        mean = _structured_mean(
                            actor_name, actor, lifted, context
                        )
                        distribution = Normal(
                            mean, log_std.exp().expand_as(mean)
                        )
                        action = distribution.sample()
                        log_probability = distribution.log_prob(action).sum(-1)
                        assert value is not None
                        state_value = value(lifted, context)
                next_observation, reward, terminated, truncated, info = env.step(
                    action
                    if actor_name == "PPO"
                    else torch.clamp(action, -1.0, 1.0)
                )
                reward_tensor = _as_tensor(reward, device).float().reshape(-1)
                done = torch.logical_or(
                    _as_tensor(terminated, device).bool().reshape(-1),
                    _as_tensor(truncated, device).bool().reshape(-1),
                )
                if bool(done.any()):
                    final_info = info.get("final_info", info)
                    episode = final_info.get("episode", {})
                    for item in _as_tensor(
                        episode.get("return", reward_tensor), device
                    ).reshape(-1)[done].tolist():
                        episode_returns.append(float(item))
                    for item in _as_tensor(
                        episode.get(
                            "episode_len",
                            torch.ones(config_used.num_envs, device=device),
                        ),
                        device,
                    ).reshape(-1)[done].tolist():
                        episode_lengths.append(float(item))
                    success = final_info.get(
                        "success", torch.zeros(config_used.num_envs, device=device)
                    )
                    for item in _as_tensor(success, device).float().reshape(-1)[done].tolist():
                        episode_successes.append(float(item))
                robot_states.append(normalized_robot.detach())
                lifted_states.append(lifted.detach())
                states29.append(state29.detach())
                rgb_list.append(rgb.detach())
                contexts.append(context.detach())
                if actor_name == "PPO":
                    pos_targets.append(
                        torch.zeros(config_used.num_envs, 3, device=device)
                    )
                else:
                    goal = _as_tensor(
                        observation["extra"]["goal_pos"], device
                    ).float()[..., :3]
                    pos_targets.append(goal)
                actions.append(action.detach())
                log_probabilities.append(log_probability.detach())
                rewards.append(reward_tensor)
                dones.append(done.float())
                values.append(state_value.detach())
                observation = next_observation

            with torch.no_grad():
                next_rgb, _, next_state29, next_lifted = _extract(
                    observation,
                    koopman,
                    state_center,
                    state_scale,
                    device,
                )
                if actor_name == "PPO":
                    next_value = actor.get_value(next_rgb, next_state29)
                else:
                    assert context_head is not None and value is not None
                    next_context = context_head(encoder.forward_rgb(next_rgb))[1]
                    next_value = value(next_lifted, next_context)

            rgb_batch = torch.stack(rgb_list)
            state29_batch = torch.stack(states29)
            lifted_batch = torch.stack(lifted_states)
            context_batch = torch.stack(contexts)
            pos_target_batch = torch.stack(pos_targets)
            action_batch = torch.stack(actions)
            old_log_prob = torch.stack(log_probabilities)
            reward_batch = torch.stack(rewards)
            done_batch = torch.stack(dones)
            old_value = torch.stack(values)
            advantages = torch.zeros_like(reward_batch)
            last_advantage = torch.zeros(config_used.num_envs, device=device)
            for step in range(config_used.rollout_steps - 1, -1, -1):
                following_value = (
                    next_value
                    if step == config_used.rollout_steps - 1
                    else old_value[step + 1]
                )
                nonterminal = 1.0 - done_batch[step]
                delta = (
                    reward_batch[step]
                    + config_used.discount * following_value * nonterminal
                    - old_value[step]
                )
                last_advantage = delta + (
                    config_used.discount
                    * config_used.gae_lambda
                    * nonterminal
                    * last_advantage
                )
                advantages[step] = last_advantage
            returns = advantages + old_value

            flat_rgb = rgb_batch.flatten(0, 1)
            flat_state29 = state29_batch.flatten(0, 1)
            flat_lifted = lifted_batch.flatten(0, 1)
            flat_context = context_batch.flatten(0, 1)
            flat_pos_target = pos_target_batch.flatten(0, 1)
            flat_action = action_batch.flatten(0, 1)
            flat_old_log_prob = old_log_prob.flatten()
            flat_old_value = old_value.flatten()
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()

            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            pos_losses: list[float] = []
            stopped_early = False
            stop_kl: float | None = None
            for _ in range(config_used.update_epochs):
                order = torch.randperm(config_used.batch_size, device=device)
                for start in range(0, config_used.batch_size, minibatch_size):
                    index = order[start : start + minibatch_size]
                    advantage = flat_advantages[index]
                    advantage = (advantage - advantage.mean()) / (
                        advantage.std(unbiased=False) + 1e-8
                    )
                    if actor_name == "PPO":
                        _, new_log_probability, entropy, current_value = (
                            actor.get_action_and_value(
                                flat_rgb[index],
                                flat_state29[index],
                                flat_action[index],
                            )
                        )
                        pos = torch.zeros(len(index), 3, device=device)
                    else:
                        assert context_head is not None and log_std is not None
                        features = encoder.forward_rgb(flat_rgb[index])
                        context, pos = context_head(features)
                        mean = _structured_mean(
                            actor_name, actor, flat_lifted[index], context
                        )
                        distribution = Normal(
                            mean, log_std.exp().expand_as(mean)
                        )
                        new_log_probability = distribution.log_prob(
                            flat_action[index]
                        ).sum(-1)
                        entropy = distribution.entropy().sum(-1)
                        assert value is not None
                        current_value = value(flat_lifted[index], context)
                    log_ratio = new_log_probability - flat_old_log_prob[index]
                    ratio = log_ratio.exp()
                    if config_used.target_kl is not None:
                        with torch.no_grad():
                            pre_update_kl = ((ratio - 1.0) - log_ratio).mean()
                        if pre_update_kl > config_used.target_kl:
                            stopped_early = True
                            stop_kl = float(pre_update_kl)
                            break
                    policy_loss = -torch.minimum(
                        ratio * advantage,
                        torch.clamp(
                            ratio,
                            1.0 - config_used.clip_ratio,
                            1.0 + config_used.clip_ratio,
                        )
                        * advantage,
                    ).mean()
                    if config_used.clip_value_loss:
                        clipped_value = flat_old_value[index] + torch.clamp(
                            current_value - flat_old_value[index],
                            -config_used.clip_ratio,
                            config_used.clip_ratio,
                        )
                        value_loss = torch.maximum(
                            (current_value - flat_returns[index]).square(),
                            (clipped_value - flat_returns[index]).square(),
                        ).mean()
                    else:
                        value_loss = (
                            (current_value - flat_returns[index]).square().mean()
                        )
                    pos_loss = torch.zeros((), device=device)
                    if actor_name != "PPO" and config_used.pos_weight > 0:
                        pos_loss = config_used.pos_weight * (
                            (pos - flat_pos_target[index]).square().mean()
                        )
                    loss = (
                        policy_loss
                        - config_used.entropy_coefficient * entropy.mean()
                        + config_used.value_coefficient * value_loss
                        + pos_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        [
                            *encoder_parameters,
                            *actor_parameters,
                            *auxiliary_parameters,
                        ],
                        config_used.max_grad_norm,
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError("Non-finite PPO gradient")
                    optimizer.step()
                    policy_losses.append(float(policy_loss))
                    value_losses.append(float(value_loss))
                    entropies.append(float(entropy.mean()))
                    pos_losses.append(float(pos_loss))
                if stopped_early:
                    break

            record = {
                "update": update,
                "global_step": update * config_used.batch_size,
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else float("nan"),
                "value_loss": float(np.mean(value_losses)) if value_losses else float("nan"),
                "entropy": float(np.mean(entropies)) if entropies else float("nan"),
                "pos_loss": float(np.mean(pos_losses)) if pos_losses else float("nan"),
                "episode_return": _optional_mean(episode_returns),
                "episode_length": _optional_mean(episode_lengths),
                "episode_success": _optional_mean(episode_successes),
                "log_std": (
                    float(log_std.detach().mean())
                    if log_std is not None
                    else float(actor.actor_logstd.detach().mean())
                ),
                "early_stopped_kl": stop_kl,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            print(
                f"update={update:04d} step={record['global_step']} "
                f"ret={_fmt(record['episode_return'])} succ={_fmt(record['episode_success'])} "
                f"len={_fmt(record['episode_length'])} ppo_loss={record['policy_loss']:.4g} "
                f"entropy={record['entropy']:.4g}",
                flush=True,
            )
            if update % config_used.checkpoint_interval_updates == 0:
                payload = {
                    "kind": "visual_pickcube_ppo",
                    "actor_name": actor_name,
                    "config": asdict(config_used),
                    "update": update,
                    "global_step": record["global_step"],
                    "encoder_state": (
                        encoder.state_dict() if actor_name != "PPO" else None
                    ),
                    "context_head_state": (
                        context_head.state_dict() if context_head is not None else None
                    ),
                    "actor_state": actor.state_dict(),
                    "value_state": (
                        value.state_dict() if value is not None else None
                    ),
                    "log_std": (
                        log_std.detach().cpu() if log_std is not None else None
                    ),
                    "optimizer_state": optimizer.state_dict(),
                }
                _atomic_torch_save(
                    output_dir / f"recovery_update_{update:05d}.pt", payload
                )
                _atomic_torch_save(latest_path, payload)
            if (
                config_used.max_wall_time_seconds is not None
                and time.perf_counter() - started
                > config_used.max_wall_time_seconds
            ):
                wall_time_reached = True
                break
    finally:
        env.close()

    summary = {
        "kind": "visual_pickcube_ppo",
        "actor_name": actor_name,
        "completed_updates": update,
        "completed_global_steps": update * config_used.batch_size,
        "wall_time_reached": wall_time_reached,
        "episode_return": _optional_mean(episode_returns),
        "episode_success": _optional_mean(episode_successes),
        "elapsed_seconds": time.perf_counter() - started,
        "config": asdict(config_used),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor",
        required=True,
        choices=list(ACTOR_NAMES),
    )
    parser.add_argument("--koopman", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    # Expose the official ManiSkill PPO RGB knobs so the PPO route can be run
    # with the recommended parameters (gamma=0.8, gae=0.9, update_epochs=4,
    # rollout_steps=50, minibatch=batch/32, target_kl=0.2, no lr anneal).
    # Structured routes (KLQR/KMPC/AB-PQ) share the same base PPO hyperparams.
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--anneal-lr", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny run that verifies the pipeline end to end",
    )
    args = parser.parse_args()
    base = PPOConfig()
    overrides = {
        "total_timesteps": args.total_timesteps,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
        "learning_rate": args.learning_rate,
        "discount": args.discount,
        "gae_lambda": args.gae_lambda,
        "target_kl": args.target_kl,
        "anneal_learning_rate": args.anneal_lr,
        "max_episode_steps": args.max_episode_steps,
        "seed": args.seed,
    }
    config = PPOConfig(
        **{
            name: value
            for name, value in overrides.items()
            if value is not None
        }
    )
    summary = train(
        args.actor,
        args.koopman,
        args.output_dir,
        config,
        args.device,
        smoke=args.smoke,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
