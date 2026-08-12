"""Fair, restart-safe visual PickCube PPO comparison.

The standard ``PPO`` route mirrors the official ManiSkill3 PPO RGB baseline
(``examples/baselines/ppo/ppo_rgb.py``): a NatureCNN visual encoder, rgb
normalized by ``/255``, a ``state`` observation that is the full flattened
``agent`` + ``extra`` (including the privileged goal), orthogonal init, an
actor log-std initialized to ``-0.5``, ``normalized_dense`` reward,
``update_epochs=8`` and ``batch / 8`` minibatches.  Only hardware-bound knobs
(``num_envs``, ``minibatch_size``, ``total_timesteps``) are expected to be
lowered when running on a single GPU.

The AC-MPC routes give their cost-map / value networks the EXACT same input
as the PPO actor: the 512-d NatureCNN feature over (rgb normalized by
``/255``, state29).  The frozen robot Koopman lift is used only inside the
bottom-level solver, never as an actor input:
  * ``KLQR`` : KoopmanLQRActor (closed-form DARE gain).
  * ``KMPC`` : KoopmanMPCActor (differentiable condensed box-QP).
  * ``AB-PQ``  : LowRankValueActor (low-rank quadratic value -> box-QP).
Each structured actor maps perception(512) -> cost-map parameters consumed
by the underlying Koopman-based solver on z0 = lift(robot state).  There is
no context head and no auxiliary loss; the goal signal reaches the actor
through state29 -> state_mlp, exactly as in the PPO route.

The published ManiSkill v3.0.1 PickCube RGB runs use a 16,384-transition
update batch (1024 envs x 16 steps), 8 epochs, 32 minibatches and 50M total
transitions.  This host can create at most 128 visual envs reliably, so one
update collects eight independent 128 x 16 chunks under the same frozen
policy.  This preserves the published batch/minibatch/update arithmetic and
the 16-step GAE horizon without exceeding SAPIEN's camera-group limit.

``AC-MPC-MPVE`` deliberately uses the exact KMPC actor.  Its only distinction
is a detached, critic-only model-predictive value-expansion loss.  Because the
available PickCube robot Koopman artifact predates reward-model training, this
visual trainer learns a separate online transition-reward model from real
rollout transitions; that model is never optimized by PPO/MPVE gradients.

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

ACTOR_NAMES = ("PPO", "KLQR", "KMPC", "AB-PQ", "AC-MPC-MPVE")
ROBOT_DIM = 21
ACTION_DIM = 8
# Official PPO "state" = qpos9 + qvel9 + is_grasped1 + tcp_pose7 + goal_pos3.
STATE_DIM = 29
# Structured routes feed their cost-map / value networks the EXACT same
# 512-d perception feature as the PPO route: NatureCNN(rgb) -> 256 plus
# state_mlp(state29) -> 256 (including the privileged goal).  The frozen
# Koopman lift is used only by the bottom-level solver (KMPC QP / DARE /
# AB-PQ greedy), never as an actor input.
PERCEPTION_DIM = 512


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
        # These two layers intentionally retain PyTorch's default
        # initialization, matching ManiSkill v3.0.1 ppo_rgb.py.
        self.rgb_fc = nn.Sequential(nn.Linear(n_flatten, 256), nn.ReLU())
        self.state_mlp = nn.Linear(state_dim, 256)
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


def make_critic(input_dim: int) -> nn.Module:
    """The shared critic head, identical across all four actor routes.

    Mirrors the official ManiSkill PPO critic: ``Linear(512) -> ReLU ->
    Linear(1)`` with orthogonal init (sqrt(2)) and a final std of 1.0.
    """
    return nn.Sequential(
        layer_init(nn.Linear(input_dim, 512)),
        nn.ReLU(inplace=True),
        layer_init(nn.Linear(512, 1)),
    )


class PPORoute(nn.Module):
    """Official PPO Agent: NatureCNN -> actor_mean/critic, log-std init -0.5."""

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.feature_net = NatureCNN(rgb_channels=3, state_dim=state_dim)
        latent_size = self.feature_net.out_features
        self.critic = make_critic(latent_size)
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


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 50_000_000
    num_envs: int = 128
    rollout_steps: int = 16
    collection_chunks: int = 8
    minibatch_size: int | None = 512
    update_epochs: int = 8
    learning_rate: float = 3e-4
    anneal_learning_rate: bool = False
    discount: float = 0.8
    gae_lambda: float = 0.9
    clip_ratio: float = 0.2
    # official ManiSkill ppo_rgb.py sets clip_vloss=False
    clip_value_loss: bool = False
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.2
    checkpoint_interval_updates: int = 10
    recovery_checkpoints_to_keep: int = 2
    milestone_interval_updates: int = 250
    eval_interval_updates: int = 25
    num_eval_envs: int = 16
    num_eval_steps: int = 50
    max_wall_time_seconds: float | None = None
    max_episode_steps: int = 50
    reward_mode: str = "normalized_dense"
    context_hidden_dim: int = 128
    kmpc_horizon: int = 10
    kmpc_solver_iterations: int = 20
    ab_rank: int = 16
    mpve_horizon: int = 5
    mpve_value_loss_coefficient: float = 1.0
    reward_model_learning_rate: float = 3e-4
    seed: int = 20_280_804

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps * self.collection_chunks

    def effective_minibatch_size(self) -> int:
        if self.minibatch_size is not None:
            return self.minibatch_size
        return max(1, self.batch_size // 32)

    def validate(self) -> None:
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.num_envs <= 0 or self.rollout_steps <= 0 or self.collection_chunks <= 0:
            raise ValueError("num_envs, rollout_steps and collection_chunks must be positive")
        mini = self.effective_minibatch_size()
        if not 0 < mini <= self.batch_size:
            raise ValueError("minibatch_size must be in [1, batch_size]")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.checkpoint_interval_updates <= 0:
            raise ValueError("checkpoint_interval_updates must be positive")
        if self.recovery_checkpoints_to_keep < 1:
            raise ValueError("recovery_checkpoints_to_keep must be positive")
        if self.eval_interval_updates <= 0 or self.num_eval_envs <= 0:
            raise ValueError("evaluation intervals/env counts must be positive")
        if not 1 <= self.mpve_horizon <= self.kmpc_horizon:
            raise ValueError("mpve_horizon must lie in [1, kmpc_horizon]")
        if self.reward_mode not in {"dense", "normalized_dense", "sparse"}:
            raise ValueError(f"Unsupported reward_mode {self.reward_mode!r}")


# --------------------------------------------------------------------------- #
# Environment + feature extraction
# --------------------------------------------------------------------------- #
def _as_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _make_env(
    config: PPOConfig,
    *,
    num_envs: int | None = None,
    evaluation: bool = False,
):
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    count = config.num_envs if num_envs is None else int(num_envs)
    base = gym.make(
        "ACMPC-VisualPickCube-v1",
        num_envs=count,
        sim_backend="gpu" if torch.cuda.is_available() else "cpu",
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode=config.reward_mode,
        render_mode=None,
        max_episode_steps=config.max_episode_steps,
    )
    return ManiSkillVectorEnv(
        base,
        count,
        auto_reset=True,
        ignore_terminations=evaluation,
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
            context_dim=PERCEPTION_DIM,
            hidden_dims=(config.context_hidden_dim,),
            max_action=1.0,
            perception_only_network=True,
        ).to(device)
    if actor_name in ("KMPC", "AC-MPC-MPVE"):
        return KoopmanMPCActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            horizon=config.kmpc_horizon,
            context_dim=PERCEPTION_DIM,
            hidden_dims=(config.context_hidden_dim,),
            action_low=-1.0,
            action_high=1.0,
            solver_iterations=config.kmpc_solver_iterations,
            perception_only_network=True,
        ).to(device)
    if actor_name == "AB-PQ":
        return LowRankValueActor(
            observation_dim=PERCEPTION_DIM,
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
    # Critic identical to the PPO route: make_critic over the 512-d
    # perception feature (the exact same NatureCNN output the PPO critic sees).
    return make_critic(PERCEPTION_DIM)


def _structured_mean(
    actor_name: str,
    actor: nn.Module,
    lifted: torch.Tensor,
    perception: torch.Tensor,
) -> torch.Tensor:
    # Actor input is IDENTICAL to the PPO route: the 512-d NatureCNN feature
    # over (rgb, state29).  The Koopman lift feeds only the bottom-level
    # solver (KMPC QP / DARE / AB-PQ greedy), never the cost-map network.
    if actor_name in ("KLQR", "KMPC", "AC-MPC-MPVE"):
        return actor(lifted, perception).action
    return actor(perception, lifted).action


class VisualTransitionRewardModel(nn.Module):
    """Online reward predictor used only to form detached MPVE targets."""

    def __init__(self) -> None:
        super().__init__()
        input_dim = PERCEPTION_DIM + 2 * ROBOT_DIM + ACTION_DIM
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        perception: torch.Tensor,
        robot: torch.Tensor,
        action: torch.Tensor,
        next_robot: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat((perception, robot, action, next_robot), dim=-1)
        return self.network(features).squeeze(-1)


def compute_mpve_td_k_targets(
    predicted_rewards: torch.Tensor,
    terminal_value: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Detached TD-k targets for every depth on an imagined trajectory."""

    if predicted_rewards.ndim < 1 or predicted_rewards.shape[-1] < 1:
        raise ValueError("predicted_rewards must have a horizon dimension")
    if terminal_value.shape != predicted_rewards.shape[:-1]:
        raise ValueError("terminal_value shape does not match predicted rewards")
    following = terminal_value.detach()
    reversed_targets: list[torch.Tensor] = []
    for index in range(predicted_rewards.shape[-1] - 1, -1, -1):
        following = predicted_rewards[..., index].detach() + gamma * following
        reversed_targets.append(following)
    return torch.stack(list(reversed(reversed_targets)), dim=-1).detach()


def _predicted_state29(
    base_state29: torch.Tensor,
    normalized_robot: torch.Tensor,
    state_center: torch.Tensor,
    state_scale: torch.Tensor,
) -> torch.Tensor:
    """Insert a predicted robot state while retaining visual task context."""

    robot = normalized_robot * state_scale + state_center
    predicted = base_state29.clone()
    predicted[..., :9] = robot[..., :9]
    predicted[..., 9:18] = robot[..., 9:18]
    predicted[..., 19:22] = robot[..., 18:21]
    return predicted


def compute_timeout_correct_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    following_values: torch.Tensor,
    reset_boundaries: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with final-observation bootstrap and no trace across autoresets."""

    if not (rewards.shape == values.shape == following_values.shape == reset_boundaries.shape):
        raise ValueError("GAE tensors must have identical [time, env] shapes")
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[step] + gamma * following_values[step] - values[step]
        trace_mask = 1.0 - reset_boundaries[step]
        last_advantage = delta + gamma * gae_lambda * trace_mask * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


def ppo_value_loss(
    current_value: torch.Tensor,
    target: torch.Tensor,
    behavior_value: torch.Tensor,
    *,
    clip_value_loss: bool,
    clip_ratio: float,
) -> torch.Tensor:
    """Official ManiSkill PPO value loss, including its leading 0.5."""

    error = (current_value - target).square()
    if clip_value_loss:
        clipped = behavior_value + torch.clamp(
            current_value - behavior_value, -clip_ratio, clip_ratio
        )
        error = torch.maximum(error, (clipped - target).square())
    return 0.5 * error.mean()


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


def _masked_tree(tree: Any, mask: torch.Tensor) -> Any:
    if isinstance(tree, dict):
        return {key: _masked_tree(value, mask) for key, value in tree.items()}
    return tree[mask]


def _route_value(
    actor_name: str,
    actor: nn.Module,
    encoder: nn.Module,
    value: nn.Module | None,
    rgb: torch.Tensor,
    state29: torch.Tensor,
) -> torch.Tensor:
    if actor_name == "PPO":
        return actor.get_value(rgb, state29)
    if value is None:
        raise AssertionError("Structured route critic is missing")
    return value(encoder(rgb, state29)).squeeze(-1)


def _route_mean(
    actor_name: str,
    actor: nn.Module,
    encoder: nn.Module,
    rgb: torch.Tensor,
    state29: torch.Tensor,
    lifted: torch.Tensor,
) -> torch.Tensor:
    if actor_name == "PPO":
        return actor.actor_mean(actor.feature_net(rgb, state29))
    perception = encoder(rgb, state29)
    return _structured_mean(actor_name, actor, lifted, perception)


def _evaluate(
    actor_name: str,
    actor: nn.Module,
    encoder: nn.Module,
    value: nn.Module | None,
    koopman: nn.Module,
    state_center: torch.Tensor,
    state_scale: torch.Tensor,
    config: PPOConfig,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """Run the official-style deterministic, full-horizon evaluation."""

    del value  # evaluation only needs the deterministic actor
    eval_env = _make_env(
        config, num_envs=config.num_eval_envs, evaluation=True
    )
    was_actor_training = actor.training
    was_encoder_training = encoder.training
    actor.eval()
    encoder.eval()
    metrics: dict[str, list[torch.Tensor]] = {}
    try:
        observation, _ = eval_env.reset(seed=seed)
        for _ in range(config.num_eval_steps):
            rgb, _, state29, lifted = _extract(
                observation, koopman, state_center, state_scale, device
            )
            with torch.no_grad():
                mean = _route_mean(
                    actor_name, actor, encoder, rgb, state29, lifted
                )
            observation, _, _, _, info = eval_env.step(
                mean if actor_name == "PPO" else mean.clamp(-1.0, 1.0)
            )
            if "final_info" in info:
                mask = _as_tensor(info["_final_info"], device).bool()
                episode = info["final_info"].get("episode", {})
                for key, value_item in episode.items():
                    tensor = _as_tensor(value_item, device).float().reshape(-1)
                    metrics.setdefault(key, []).append(tensor[mask].detach().cpu())
    finally:
        eval_env.close()
        actor.train(was_actor_training)
        encoder.train(was_encoder_training)
    result: dict[str, float] = {}
    for key, pieces in metrics.items():
        if pieces:
            result[key] = float(torch.cat(pieces).mean())
    result["episodes"] = float(
        sum(piece.numel() for piece in metrics.get("return", []))
    )
    return result


def _prune_recovery_checkpoints(output_dir: Path, keep: int) -> None:
    paths = sorted(output_dir.glob("recovery_update_*.pt"))
    for path in paths[:-keep]:
        path.unlink(missing_ok=True)


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
            total_timesteps=16,
            num_envs=4,
            rollout_steps=4,
            collection_chunks=1,
            minibatch_size=8,
            update_epochs=1,
            learning_rate=3e-4,
            entropy_coefficient=0.0,
            target_kl=0.2,
            checkpoint_interval_updates=1,
            recovery_checkpoints_to_keep=1,
            milestone_interval_updates=100,
            eval_interval_updates=1,
            num_eval_envs=2,
            num_eval_steps=50,
            seed=config.seed,
            max_episode_steps=config.max_episode_steps,
            reward_mode=config.reward_mode,
            context_hidden_dim=config.context_hidden_dim,
            kmpc_horizon=max(2, config.mpve_horizon),
            kmpc_solver_iterations=2,
            ab_rank=config.ab_rank,
            mpve_horizon=min(2, config.mpve_horizon),
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
        log_std: nn.Parameter | None = None
        encoder = actor.feature_net
        encoder_parameters: list[nn.Parameter] = []
        auxiliary_parameters: list[nn.Parameter] = []
    else:
        # The SAME NatureCNN as the PPO route, used in full: the structured
        # cost-map / value networks consume the identical 512-d perception
        # feature NatureCNN(rgb, state29) -> [rgb_branch 256; state_mlp 256].
        encoder = NatureCNN(rgb_channels=3, state_dim=STATE_DIM).to(device)
        # official PPO route inits actor log_std at -0.5; match it exactly so
        # the structured routes start at the same exploration scale.
        log_std = nn.Parameter(
            torch.full((ACTION_DIM,), -0.5, device=device)
        )
        assert value is not None
        encoder_parameters = list(encoder.parameters())
        auxiliary_parameters = [*value.parameters(), log_std]
    actor_parameters = list(actor.parameters())
    ppo_parameters = [
        *encoder_parameters,
        *actor_parameters,
        *auxiliary_parameters,
    ]
    policy_parameters = ppo_parameters
    critic_parameters: list[nn.Parameter] = []
    if actor_name != "PPO":
        assert value is not None and log_std is not None
        policy_parameters = [
            *encoder_parameters,
            *actor_parameters,
            log_std,
        ]
        critic_parameters = list(value.parameters())
    optimizer = torch.optim.Adam(
        ppo_parameters, lr=config_used.learning_rate, eps=1e-5
    )
    reward_model: VisualTransitionRewardModel | None = None
    reward_optimizer: torch.optim.Optimizer | None = None
    if actor_name == "AC-MPC-MPVE":
        reward_model = VisualTransitionRewardModel().to(device)
        reward_optimizer = torch.optim.Adam(
            reward_model.parameters(),
            lr=config_used.reward_model_learning_rate,
            eps=1e-5,
        )

    # Auxiliary construction must not perturb the common rollout RNG stream:
    # KMPC and its MPVE critic ablation start with identical actors, encoders,
    # critics, environment seeds and exploration samples.
    random.seed(config_used.seed)
    np.random.seed(config_used.seed)
    torch.manual_seed(config_used.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config_used.seed)

    env = _make_env(config_used)
    observation, _ = env.reset(seed=config_used.seed)
    number_updates = config_used.total_timesteps // config_used.batch_size
    if number_updates < 1:
        raise ValueError("total_timesteps is smaller than one update batch")
    minibatch_size = config_used.effective_minibatch_size()
    metrics_path = output_dir / "metrics.jsonl"
    latest_path = output_dir / "latest.pt"

    # Auto-resume: if a previous run left latest.pt, continue from the saved
    # update instead of restarting from scratch (crash / reboot protection).
    start_update = 1
    if latest_path.exists():
        payload = torch.load(
            latest_path, map_location=device, weights_only=False
        )
        if payload.get("kind") != "visual_pickcube_ppo_v2":
            raise ValueError(
                f"Refusing to resume: latest.pt is not a visual PPO "
                f"checkpoint (kind={payload.get('kind')!r})"
            )
        saved_actor = payload.get("actor_name")
        if saved_actor != actor_name:
            raise ValueError(
                f"Refusing to resume: latest.pt belongs to actor "
                f"{saved_actor!r}, not {actor_name!r}"
            )
        saved_config = payload.get("config")
        if saved_config is not None and saved_config != asdict(config_used):
            print(
                f"[resume] WARNING: checkpoint config differs from the "
                f"requested config; loading weights regardless.",
                flush=True,
            )
        if actor_name == "PPO":
            actor.load_state_dict(payload["actor_state"])
        else:
            assert value is not None
            assert log_std is not None
            encoder.load_state_dict(payload["encoder_state"])
            actor.load_state_dict(payload["actor_state"])
            value.load_state_dict(payload["value_state"])
            log_std.data.copy_(payload["log_std"].to(device))
        optimizer.load_state_dict(payload["optimizer_state"])
        if reward_model is not None:
            if payload.get("reward_model_state") is None:
                raise ValueError("MPVE checkpoint is missing its reward model")
            reward_model.load_state_dict(payload["reward_model_state"])
            assert reward_optimizer is not None
            reward_optimizer.load_state_dict(payload["reward_optimizer_state"])
        start_update = int(payload["update"]) + 1
        print(
            f"[resume] {actor_name} resumed from update "
            f"{int(payload['update'])} step "
            f"{int(payload['global_step'])} -> continues at update "
            f"{start_update}",
            flush=True,
        )
    episode_returns: deque[float] = deque(maxlen=100)
    episode_lengths: deque[float] = deque(maxlen=100)
    episode_successes: deque[float] = deque(maxlen=100)
    started = time.perf_counter()
    wall_time_reached = False
    metadata = {
        "kind": "visual_pickcube_ppo_v2",
        "actor_name": actor_name,
        "smoke": bool(smoke),
        "official_baseline_alignment": True,
        "published_batch_size": 16_384,
        "hardware_adaptation": (
            f"{config_used.num_envs} envs x {config_used.rollout_steps} steps x "
            f"{config_used.collection_chunks} frozen-policy chunks"
        ),
        "mpve_actor_equivalent_to": (
            "KMPC" if actor_name == "AC-MPC-MPVE" else None
        ),
        "mpve_reward_model_source": (
            "online real rollout transitions" if reward_model is not None else None
        ),
        "koopman_path": str(koopman_path.resolve()),
        "device": str(device),
        "config": asdict(config_used),
        "trainable_parameters": sum(
            p.numel()
            for p in encoder_parameters + actor_parameters + auxiliary_parameters
            if p.requires_grad
        ),
        "reward_model_parameters": (
            sum(p.numel() for p in reward_model.parameters())
            if reward_model is not None
            else 0
        ),
    }
    _atomic_json(output_dir / "run_config.json", metadata)

    def checkpoint_payload(current_update: int) -> dict[str, Any]:
        return {
            "kind": "visual_pickcube_ppo_v2",
            "actor_name": actor_name,
            "config": asdict(config_used),
            "update": current_update,
            "global_step": current_update * config_used.batch_size,
            "encoder_state": encoder.state_dict() if actor_name != "PPO" else None,
            "actor_state": actor.state_dict(),
            "value_state": value.state_dict() if value is not None else None,
            "log_std": log_std.detach().cpu() if log_std is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "reward_model_state": (
                reward_model.state_dict() if reward_model is not None else None
            ),
            "reward_optimizer_state": (
                reward_optimizer.state_dict() if reward_optimizer is not None else None
            ),
        }

    update = start_update - 1
    try:
        for update in range(start_update, number_updates + 1):
            if config_used.anneal_learning_rate:
                fraction = 1.0 - (update - 1.0) / number_updates
                for group in optimizer.param_groups:
                    group["lr"] = fraction * config_used.learning_rate
            robot_states: list[torch.Tensor] = []
            lifted_states: list[torch.Tensor] = []
            states29: list[torch.Tensor] = []
            rgb_list: list[torch.Tensor] = []
            actions: list[torch.Tensor] = []
            log_probabilities: list[torch.Tensor] = []
            rewards: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            next_robot_states: list[torch.Tensor] = []
            applied_actions: list[torch.Tensor] = []
            advantage_chunks: list[torch.Tensor] = []
            return_chunks: list[torch.Tensor] = []
            for _chunk in range(config_used.collection_chunks):
                chunk_values: list[torch.Tensor] = []
                chunk_rewards: list[torch.Tensor] = []
                chunk_boundaries: list[torch.Tensor] = []
                chunk_final_values: list[torch.Tensor] = []
                for _step in range(config_used.rollout_steps):
                    rgb, normalized_robot, state29, lifted = _extract(
                        observation, koopman, state_center, state_scale, device
                    )
                    with torch.no_grad():
                        if actor_name == "PPO":
                            action, log_probability, _, state_value = (
                                actor.get_action_and_value(rgb, state29)
                            )
                        else:
                            assert log_std is not None and value is not None
                            perception = encoder(rgb, state29)
                            mean = _structured_mean(
                                actor_name, actor, lifted, perception
                            )
                            distribution = Normal(
                                mean, log_std.exp().expand_as(mean)
                            )
                            action = distribution.sample()
                            log_probability = distribution.log_prob(action).sum(-1)
                            state_value = value(perception).squeeze(-1)
                    applied_action = (
                        action if actor_name == "PPO" else action.clamp(-1.0, 1.0)
                    )
                    next_observation, reward, terminated, truncated, info = env.step(
                        applied_action
                    )
                    reward_tensor = _as_tensor(reward, device).float().reshape(-1)
                    boundary = torch.logical_or(
                        _as_tensor(terminated, device).bool().reshape(-1),
                        _as_tensor(truncated, device).bool().reshape(-1),
                    )
                    with torch.no_grad():
                        _, next_robot, _, _ = _extract(
                            next_observation, koopman, state_center, state_scale, device
                        )
                    final_value = torch.zeros(config_used.num_envs, device=device)
                    if bool(boundary.any()) and "final_observation" in info:
                        final_observation = _masked_tree(
                            info["final_observation"], boundary
                        )
                        with torch.no_grad():
                            final_rgb, final_robot, final_state29, _ = _extract(
                                final_observation,
                                koopman,
                                state_center,
                                state_scale,
                                device,
                            )
                            final_value[boundary] = _route_value(
                                actor_name,
                                actor,
                                encoder,
                                value,
                                final_rgb,
                                final_state29,
                            )
                            next_robot = next_robot.clone()
                            next_robot[boundary] = final_robot
                    if "final_info" in info:
                        mask = _as_tensor(info["_final_info"], device).bool()
                        episode = info["final_info"].get("episode", {})
                        for key, target in (
                            ("return", episode_returns),
                            ("episode_len", episode_lengths),
                            ("success_once", episode_successes),
                        ):
                            fallback = (
                                reward_tensor
                                if key == "return"
                                else torch.zeros(config_used.num_envs, device=device)
                            )
                            item = episode.get(key, episode.get("success", fallback))
                            for scalar in _as_tensor(item, device).float().reshape(-1)[mask].tolist():
                                target.append(float(scalar))
                    robot_states.append(normalized_robot.detach())
                    next_robot_states.append(next_robot.detach())
                    lifted_states.append(lifted.detach())
                    states29.append(state29.detach())
                    rgb_list.append(rgb.detach())
                    actions.append(action.detach())
                    applied_actions.append(applied_action.detach())
                    log_probabilities.append(log_probability.detach())
                    rewards.append(reward_tensor)
                    values.append(state_value.detach())
                    chunk_values.append(state_value.detach())
                    chunk_rewards.append(reward_tensor)
                    chunk_boundaries.append(boundary.float())
                    chunk_final_values.append(final_value)
                    observation = next_observation

                with torch.no_grad():
                    next_rgb, _, next_state29, _ = _extract(
                        observation, koopman, state_center, state_scale, device
                    )
                    chunk_next_value = _route_value(
                        actor_name, actor, encoder, value, next_rgb, next_state29
                    )
                value_chunk = torch.stack(chunk_values)
                reward_chunk = torch.stack(chunk_rewards)
                boundary_chunk = torch.stack(chunk_boundaries)
                final_value_chunk = torch.stack(chunk_final_values)
                ordinary_following = torch.cat(
                    (value_chunk[1:], chunk_next_value.unsqueeze(0)), dim=0
                )
                following_chunk = torch.where(
                    boundary_chunk.bool(), final_value_chunk, ordinary_following
                )
                chunk_advantage, chunk_return = compute_timeout_correct_gae(
                    reward_chunk,
                    value_chunk,
                    following_chunk,
                    boundary_chunk,
                    gamma=config_used.discount,
                    gae_lambda=config_used.gae_lambda,
                )
                advantage_chunks.append(chunk_advantage)
                return_chunks.append(chunk_return)

            rgb_batch = torch.stack(rgb_list)
            state29_batch = torch.stack(states29)
            lifted_batch = torch.stack(lifted_states)
            robot_batch = torch.stack(robot_states)
            next_robot_batch = torch.stack(next_robot_states)
            applied_action_batch = torch.stack(applied_actions)
            action_batch = torch.stack(actions)
            old_log_prob = torch.stack(log_probabilities)
            old_value = torch.stack(values)
            advantages = torch.cat(advantage_chunks, dim=0)
            returns = torch.cat(return_chunks, dim=0)

            flat_rgb = rgb_batch.flatten(0, 1)
            flat_state29 = state29_batch.flatten(0, 1)
            flat_lifted = lifted_batch.flatten(0, 1)
            flat_action = action_batch.flatten(0, 1)
            flat_old_log_prob = old_log_prob.flatten()
            flat_old_value = old_value.flatten()
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()
            flat_rewards = torch.stack(rewards).flatten()

            reward_model_losses: list[float] = []
            flat_mpve_features_cpu: torch.Tensor | None = None
            flat_mpve_targets_cpu: torch.Tensor | None = None
            if actor_name == "AC-MPC-MPVE":
                assert reward_model is not None and reward_optimizer is not None
                flat_robot = robot_batch.flatten(0, 1)
                flat_next_robot = next_robot_batch.flatten(0, 1)
                flat_applied_action = applied_action_batch.flatten(0, 1)
                reward_order = torch.randperm(config_used.batch_size, device=device)
                for start in range(0, config_used.batch_size, minibatch_size):
                    index = reward_order[start : start + minibatch_size]
                    with torch.no_grad():
                        reward_perception = encoder(
                            flat_rgb[index], flat_state29[index]
                        )
                    predicted_reward = reward_model(
                        reward_perception.detach(),
                        flat_robot[index],
                        flat_applied_action[index],
                        flat_next_robot[index],
                    )
                    reward_loss = 0.5 * (
                        predicted_reward - flat_rewards[index]
                    ).square().mean()
                    reward_optimizer.zero_grad(set_to_none=True)
                    reward_loss.backward()
                    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
                    reward_optimizer.step()
                    reward_model_losses.append(float(reward_loss.detach()))

                imagined_features: list[torch.Tensor] = []
                imagined_targets: list[torch.Tensor] = []
                with torch.no_grad():
                    for start in range(0, config_used.batch_size, minibatch_size):
                        stop = min(start + minibatch_size, config_used.batch_size)
                        rgb_part = encoder.forward_rgb(flat_rgb[start:stop])
                        base_state = flat_state29[start:stop]
                        base_perception = torch.cat(
                            (rgb_part, encoder.state_mlp(base_state)), dim=-1
                        )
                        actor_output = actor(
                            flat_lifted[start:stop], base_perception
                        )
                        action_sequence = actor_output.action_sequence[
                            ..., : config_used.mpve_horizon, :
                        ]
                        current = flat_lifted[start:stop]
                        step_features: list[torch.Tensor] = []
                        predicted_rewards: list[torch.Tensor] = []
                        for depth in range(config_used.mpve_horizon):
                            current_robot = koopman.reconstruct(current)
                            predicted_state = _predicted_state29(
                                base_state, current_robot, state_center, state_scale
                            )
                            current_perception = torch.cat(
                                (rgb_part, encoder.state_mlp(predicted_state)), dim=-1
                            )
                            step_features.append(current_perception)
                            following = koopman.linear_step(
                                current, action_sequence[..., depth, :]
                            )
                            following_robot = koopman.reconstruct(following)
                            predicted_rewards.append(
                                reward_model(
                                    current_perception,
                                    current_robot,
                                    action_sequence[..., depth, :],
                                    following_robot,
                                )
                            )
                            current = following
                        terminal_state = _predicted_state29(
                            base_state,
                            koopman.reconstruct(current),
                            state_center,
                            state_scale,
                        )
                        terminal_perception = torch.cat(
                            (rgb_part, encoder.state_mlp(terminal_state)), dim=-1
                        )
                        assert value is not None
                        terminal_value = value(terminal_perception).squeeze(-1)
                        target = compute_mpve_td_k_targets(
                            torch.stack(predicted_rewards, dim=-1),
                            terminal_value,
                            config_used.discount,
                        )
                        imagined_features.append(
                            torch.stack(step_features, dim=-2).cpu().to(torch.float16)
                        )
                        imagined_targets.append(target.cpu())
                flat_mpve_features_cpu = torch.cat(imagined_features, dim=0)
                flat_mpve_targets_cpu = torch.cat(imagined_targets, dim=0)

            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            mpve_value_losses: list[float] = []
            stopped_early = False
            stop_kl: float | None = None
            for _ in range(config_used.update_epochs):
                order = torch.randperm(config_used.batch_size, device=device)
                for start in range(0, config_used.batch_size, minibatch_size):
                    index = order[start : start + minibatch_size]
                    advantage = flat_advantages[index]
                    advantage = (advantage - advantage.mean()) / (
                        advantage.std() + 1e-8
                    )
                    if actor_name == "PPO":
                        _, new_log_probability, entropy, current_value = (
                            actor.get_action_and_value(
                                flat_rgb[index],
                                flat_state29[index],
                                flat_action[index],
                            )
                        )
                    else:
                        assert log_std is not None
                        perception = encoder(
                            flat_rgb[index], flat_state29[index]
                        )
                        mean = _structured_mean(
                            actor_name, actor, flat_lifted[index], perception
                        )
                        distribution = Normal(
                            mean, log_std.exp().expand_as(mean)
                        )
                        new_log_probability = distribution.log_prob(
                            flat_action[index]
                        ).sum(-1)
                        entropy = distribution.entropy().sum(-1)
                        assert value is not None
                        current_value = value(perception).squeeze(-1)
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
                    value_loss = ppo_value_loss(
                        current_value,
                        flat_returns[index],
                        flat_old_value[index],
                        clip_value_loss=config_used.clip_value_loss,
                        clip_ratio=config_used.clip_ratio,
                    )
                    mpve_value_loss = current_value.new_zeros(())
                    if actor_name == "AC-MPC-MPVE":
                        assert value is not None
                        assert flat_mpve_features_cpu is not None
                        assert flat_mpve_targets_cpu is not None
                        cpu_index = index.detach().cpu()
                        mpve_features = flat_mpve_features_cpu[cpu_index].to(
                            device=device, dtype=torch.float32
                        )
                        mpve_targets = flat_mpve_targets_cpu[cpu_index].to(device)
                        mpve_value_loss = (
                            value(mpve_features).squeeze(-1) - mpve_targets
                        ).square().mean()
                    loss = (
                        policy_loss
                        - config_used.entropy_coefficient * entropy.mean()
                        + config_used.value_coefficient * value_loss
                        + config_used.mpve_value_loss_coefficient * mpve_value_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if actor_name == "PPO":
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            ppo_parameters, config_used.max_grad_norm
                        )
                        finite_gradient = torch.isfinite(gradient_norm)
                    else:
                        policy_norm = torch.nn.utils.clip_grad_norm_(
                            policy_parameters, config_used.max_grad_norm
                        )
                        critic_norm = torch.nn.utils.clip_grad_norm_(
                            critic_parameters, config_used.max_grad_norm
                        )
                        finite_gradient = torch.logical_and(
                            torch.isfinite(policy_norm), torch.isfinite(critic_norm)
                        )
                    if not bool(finite_gradient):
                        raise FloatingPointError("Non-finite PPO gradient")
                    optimizer.step()
                    policy_losses.append(float(policy_loss))
                    value_losses.append(float(value_loss))
                    entropies.append(float(entropy.mean()))
                    if actor_name == "AC-MPC-MPVE":
                        mpve_value_losses.append(float(mpve_value_loss.detach()))
                if stopped_early:
                    break

            record = {
                "update": update,
                "global_step": update * config_used.batch_size,
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else float("nan"),
                "value_loss": float(np.mean(value_losses)) if value_losses else float("nan"),
                "mpve_value_loss": (
                    float(np.mean(mpve_value_losses)) if mpve_value_losses else None
                ),
                "reward_model_loss": (
                    float(np.mean(reward_model_losses)) if reward_model_losses else None
                ),
                "entropy": float(np.mean(entropies)) if entropies else float("nan"),
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
            if update == 1 or update % config_used.eval_interval_updates == 1:
                eval_result = _evaluate(
                    actor_name,
                    actor,
                    encoder,
                    value,
                    koopman,
                    state_center,
                    state_scale,
                    config_used,
                    device,
                    config_used.seed + 10_000 + update,
                )
                for key, item in eval_result.items():
                    record[f"eval_{key}"] = item
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
                payload = checkpoint_payload(update)
                _atomic_torch_save(
                    output_dir / f"recovery_update_{update:05d}.pt", payload
                )
                _atomic_torch_save(latest_path, payload)
                _prune_recovery_checkpoints(
                    output_dir, config_used.recovery_checkpoints_to_keep
                )
            if update % config_used.milestone_interval_updates == 0:
                _atomic_torch_save(
                    output_dir / f"milestone_update_{update:05d}.pt",
                    checkpoint_payload(update),
                )
            if (
                config_used.max_wall_time_seconds is not None
                and time.perf_counter() - started
                > config_used.max_wall_time_seconds
            ):
                wall_time_reached = True
                break
    finally:
        env.close()

    if update >= start_update:
        final_payload = checkpoint_payload(update)
        _atomic_torch_save(latest_path, final_payload)
        _atomic_torch_save(output_dir / "final.pt", final_payload)

    summary = {
        "kind": "visual_pickcube_ppo_v2",
        "actor_name": actor_name,
        "completed_updates": update,
        "completed_global_steps": update * config_used.batch_size,
        "wall_time_reached": wall_time_reached,
        "episode_return": _optional_mean(episode_returns),
        "episode_success": _optional_mean(episode_successes),
        "elapsed_seconds": time.perf_counter() - started,
        "config": asdict(config_used),
    }
    _atomic_json(output_dir / "summary.json", summary)
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
    parser.add_argument("--collection-chunks", type=int, default=None)
    parser.add_argument("--update-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--discount", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--anneal-lr", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--clip-vloss", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--eval-interval-updates", type=int, default=None)
    parser.add_argument("--num-eval-envs", type=int, default=None)
    parser.add_argument("--num-eval-steps", type=int, default=None)
    parser.add_argument("--checkpoint-interval-updates", type=int, default=None)
    parser.add_argument("--recovery-checkpoints-to-keep", type=int, default=None)
    parser.add_argument("--milestone-interval-updates", type=int, default=None)
    parser.add_argument("--mpve-horizon", type=int, default=None)
    parser.add_argument("--mpve-value-loss-coefficient", type=float, default=None)
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
        "collection_chunks": args.collection_chunks,
        "update_epochs": args.update_epochs,
        "minibatch_size": args.minibatch_size,
        "learning_rate": args.learning_rate,
        "discount": args.discount,
        "gae_lambda": args.gae_lambda,
        "target_kl": args.target_kl,
        "anneal_learning_rate": args.anneal_lr,
        "clip_value_loss": args.clip_vloss,
        "max_episode_steps": args.max_episode_steps,
        "eval_interval_updates": args.eval_interval_updates,
        "num_eval_envs": args.num_eval_envs,
        "num_eval_steps": args.num_eval_steps,
        "checkpoint_interval_updates": args.checkpoint_interval_updates,
        "recovery_checkpoints_to_keep": args.recovery_checkpoints_to_keep,
        "milestone_interval_updates": args.milestone_interval_updates,
        "mpve_horizon": args.mpve_horizon,
        "mpve_value_loss_coefficient": args.mpve_value_loss_coefficient,
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
