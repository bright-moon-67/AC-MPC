"""Short PPO interface/gradient smoke test for all PandaReach3 actors."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor
from antmaze_ac.rl.quadratic_actors import (
    KoopmanLQRActor,
    LowRankValueActor,
)
from experiments.state_only_feasibility.collect_pandareach_threewaypoint import (
    _state_from_observation,
    _numpy,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
    PandaReachThreeWaypointEnv,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_bc import (
    BCConfig,
    StandardPPOActor,
    TASK_CONTEXT_DIM,
    WAYPOINT_COUNT,
    load_koopman,
)


@dataclass(frozen=True)
class PPOSmokeConfig:
    rollout_steps: int = 256
    updates: int = 2
    optimization_epochs: int = 4
    minibatch_size: int = 64
    learning_rate: float = 1e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    initial_std_rad: float = 0.005
    max_grad_norm: float = 1.0
    seed: int = 20_280_804


class ValueNetwork(nn.Module):
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


def _load_actor(
    actor_path: Path,
    koopman_path: Path,
    device: torch.device,
) -> tuple[str, nn.Module, Any, dict[str, np.ndarray], BCConfig]:
    payload = torch.load(actor_path, map_location=device, weights_only=False)
    actor_name = str(payload.get("name"))
    if actor_name not in {"PPO", "KLQR", "AB-PQ", "BC-KMPC"}:
        raise ValueError(f"Unsupported PPO smoke actor {actor_name!r}")
    config = BCConfig(**payload["config"])
    koopman, _ = load_koopman(koopman_path, device)
    if actor_name == "PPO":
        actor = StandardPPOActor(
            koopman.state_dim + TASK_CONTEXT_DIM,
            config.ppo_hidden_dim,
            config.action_limit_rad,
        )
    elif actor_name == "KLQR":
        actor = KoopmanLQRActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            context_dim=TASK_CONTEXT_DIM,
            hidden_dims=(config.hidden_dim,),
            max_action=config.action_limit_rad,
        )
    elif actor_name == "AB-PQ":
        actor = LowRankValueActor(
            observation_dim=koopman.lifted_dim + TASK_CONTEXT_DIM,
            A=koopman.A,
            B=koopman.B,
            R=torch.eye(7, device=device, dtype=koopman.A.dtype),
            base_hessian=torch.eye(
                koopman.lifted_dim, device=device, dtype=koopman.A.dtype
            ),
            rank=config.ab_rank,
            hidden_dims=(config.hidden_dim,),
            value_linear_scale=1.0,
            max_action=config.action_limit_rad,
        )
    else:
        actor = KoopmanMPCActor(
            A=koopman.A,
            B=koopman.B,
            C=koopman.C,
            horizon=config.kmpc_horizon,
            context_dim=TASK_CONTEXT_DIM,
            hidden_dims=(config.hidden_dim,),
            action_low=-config.action_limit_rad,
            action_high=config.action_limit_rad,
            solver_iterations=config.kmpc_solver_iterations,
        )
    actor = actor.to(device)
    actor.load_state_dict(payload["actor_state"])
    normalizer = {
        key: np.asarray(
            value.detach().cpu().numpy()
            if isinstance(value, torch.Tensor)
            else value,
            dtype=np.float32,
        )
        for key, value in payload["normalizer"].items()
    }
    return actor_name, actor, koopman, normalizer, config


def _features(
    observation: dict[str, Any],
    koopman: Any,
    normalizer: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = _state_from_observation(observation)
    waypoints = _numpy(observation["extra"]["waypoints"]).reshape(
        WAYPOINT_COUNT, 3
    )
    active_index = int(
        _numpy(observation["extra"]["active_waypoint_index"])
        .reshape(-1)[0]
    )
    context_raw = np.concatenate(
        (
            waypoints.reshape(-1),
            np.eye(WAYPOINT_COUNT, dtype=np.float32)[active_index],
        )
    )
    normalized_state = torch.as_tensor(
        (state - normalizer["state_center"]) / normalizer["state_scale"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    context = torch.as_tensor(
        (context_raw - normalizer["context_center"])
        / normalizer["context_scale"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        lifted = koopman.lift(normalized_state)
    return normalized_state, lifted, context


def _actor_mean(
    actor_name: str,
    actor: nn.Module,
    normalized_state: torch.Tensor,
    lifted: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    if actor_name == "PPO":
        return actor(normalized_state, context)
    if actor_name == "KLQR":
        return actor(lifted, context).action
    if actor_name == "AB-PQ":
        return actor(torch.cat((lifted, context), dim=-1), lifted).action
    return actor(lifted, context).action


def run_smoke(
    actor_path: Path,
    koopman_path: Path,
    output_path: Path,
    config: PPOSmokeConfig,
    device_name: str = "auto",
) -> dict[str, Any]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    actor_name, actor, koopman, normalizer, bc_config = _load_actor(
        actor_path, koopman_path, device
    )
    value = ValueNetwork(koopman.lifted_dim + TASK_CONTEXT_DIM).to(device)
    log_std = nn.Parameter(
        torch.full(
            (7,), np.log(config.initial_std_rad), dtype=torch.float32, device=device
        )
    )
    optimizer = torch.optim.Adam(
        [*actor.parameters(), *value.parameters(), log_std],
        lr=config.learning_rate,
    )
    env = PandaArmOnlyActionWrapper(
        gym.make(
            "ACMPC-PandaReach3-v0",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode=None,
            render_backend="none",
            max_episode_steps=bc_config.max_episode_steps,
            goal_threshold=bc_config.goal_threshold,
        )
    )
    observation, _ = env.reset(seed=config.seed)
    episode_offset = 0
    update_reports: list[dict[str, Any]] = []
    try:
        for update in range(1, config.updates + 1):
            normalized_state_items: list[torch.Tensor] = []
            lifted_items: list[torch.Tensor] = []
            context_items: list[torch.Tensor] = []
            action_items: list[torch.Tensor] = []
            log_prob_items: list[torch.Tensor] = []
            reward_items: list[float] = []
            done_items: list[float] = []
            value_items: list[torch.Tensor] = []
            next_value_items: list[torch.Tensor] = []
            completed_events = 0
            episode_completions = 0
            for _ in range(config.rollout_steps):
                normalized_state, lifted, context = _features(
                    observation, koopman, normalizer, device
                )
                with torch.no_grad():
                    mean = _actor_mean(
                        actor_name, actor, normalized_state, lifted, context
                    )
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    sampled_action = distribution.sample()
                    log_probability = distribution.log_prob(sampled_action).sum(-1)
                    state_value = value(lifted, context)
                next_observation, reward, terminated, truncated, info = env.step(
                    torch.clamp(
                        sampled_action.squeeze(0) / bc_config.action_limit_rad,
                        -1.0,
                        1.0,
                    ).cpu().numpy()
                )
                done = bool(_numpy(terminated).reshape(-1)[0]) or bool(
                    _numpy(truncated).reshape(-1)[0]
                )
                scalar_reward = float(_numpy(reward).reshape(-1)[0])
                completed_events += int(
                    bool(_numpy(info["waypoint_passed"]).reshape(-1)[0])
                )
                episode_completions += int(
                    bool(_numpy(info["success"]).reshape(-1)[0])
                )
                if done:
                    next_state_value = state_value.new_zeros(1)
                else:
                    _, next_lifted, next_context = _features(
                        next_observation, koopman, normalizer, device
                    )
                    with torch.no_grad():
                        next_state_value = value(next_lifted, next_context)
                lifted_items.append(lifted.squeeze(0))
                normalized_state_items.append(normalized_state.squeeze(0))
                context_items.append(context.squeeze(0))
                action_items.append(sampled_action.squeeze(0))
                log_prob_items.append(log_probability.squeeze(0))
                reward_items.append(scalar_reward)
                done_items.append(float(done))
                value_items.append(state_value.squeeze(0))
                next_value_items.append(next_state_value.squeeze(0))
                observation = next_observation
                if done:
                    episode_offset += 1
                    observation, _ = env.reset(
                        seed=config.seed + episode_offset
                    )

            normalized_state_batch = torch.stack(normalized_state_items)
            lifted_batch = torch.stack(lifted_items)
            context_batch = torch.stack(context_items)
            action_batch = torch.stack(action_items)
            old_log_prob = torch.stack(log_prob_items)
            old_value = torch.stack(value_items)
            next_value = torch.stack(next_value_items)
            rewards = torch.tensor(reward_items, device=device)
            dones = torch.tensor(done_items, device=device)
            deltas = rewards + config.discount * (1.0 - dones) * next_value - old_value
            advantages = torch.zeros_like(deltas)
            following_advantage = deltas.new_zeros(())
            for index in range(config.rollout_steps - 1, -1, -1):
                following_advantage = deltas[index] + (
                    config.discount
                    * config.gae_lambda
                    * (1.0 - dones[index])
                    * following_advantage
                )
                advantages[index] = following_advantage
            returns = advantages + old_value
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

            actor_gradient_norms: list[float] = []
            policy_losses: list[float] = []
            value_losses: list[float] = []
            generator = torch.Generator(device=device).manual_seed(
                config.seed + update
            )
            for _ in range(config.optimization_epochs):
                order = torch.randperm(
                    config.rollout_steps, generator=generator, device=device
                )
                for start in range(0, config.rollout_steps, config.minibatch_size):
                    index = order[start : start + config.minibatch_size]
                    mean = _actor_mean(
                        actor_name,
                        actor,
                        normalized_state_batch[index],
                        lifted_batch[index],
                        context_batch[index],
                    )
                    distribution = Normal(mean, log_std.exp().expand_as(mean))
                    log_probability = distribution.log_prob(
                        action_batch[index]
                    ).sum(-1)
                    ratio = torch.exp(log_probability - old_log_prob[index])
                    unclipped = ratio * advantages[index]
                    clipped = torch.clamp(
                        ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio
                    ) * advantages[index]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    predicted_value = value(
                        lifted_batch[index], context_batch[index]
                    )
                    value_loss = (predicted_value - returns[index]).square().mean()
                    entropy = distribution.entropy().sum(-1).mean()
                    loss = (
                        policy_loss
                        + config.value_coefficient * value_loss
                        - config.entropy_coefficient * entropy
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    actor_gradient = torch.sqrt(
                        sum(
                            parameter.grad.square().sum()
                            for parameter in actor.parameters()
                            if parameter.grad is not None
                        )
                    )
                    if not torch.isfinite(actor_gradient):
                        raise FloatingPointError("Non-finite PPO cost-map gradient")
                    actor_gradient_norms.append(float(actor_gradient.detach()))
                    torch.nn.utils.clip_grad_norm_(
                        [*actor.parameters(), *value.parameters(), log_std],
                        config.max_grad_norm,
                    )
                    optimizer.step()
                    policy_losses.append(float(policy_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
            update_reports.append(
                {
                    "update": update,
                    "rollout_reward_sum": float(rewards.sum()),
                    "waypoint_pass_events": completed_events,
                    "full_success_events": episode_completions,
                    "advantage_std_before_normalization": float(
                        (returns - old_value).std(unbiased=False)
                    ),
                    "mean_policy_loss": float(np.mean(policy_losses)),
                    "mean_value_loss": float(np.mean(value_losses)),
                    "mean_actor_gradient_norm": float(
                        np.mean(actor_gradient_norms)
                    ),
                    "minimum_actor_gradient_norm": float(
                        np.min(actor_gradient_norms)
                    ),
                    "maximum_actor_gradient_norm": float(
                        np.max(actor_gradient_norms)
                    ),
                    "log_std": log_std.detach().cpu().tolist(),
                }
            )
    finally:
        env.close()

    report = {
        "kind": "pandareach_threewaypoint_ppo_smoke",
        "actor_name": actor_name,
        "scope": (
            "short interface/numerical/gradient smoke only; not a PPO "
            "convergence or performance result"
        ),
        "actor_path": str(actor_path),
        "koopman_path": str(koopman_path),
        "device": str(device),
        "config": asdict(config),
        "updates": update_reports,
        "finite_nonzero_cost_map_gradients": all(
            np.isfinite(row["maximum_actor_gradient_norm"])
            and row["maximum_actor_gradient_norm"] > 0
            for row in update_reports
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {
            "kind": "pandareach_threewaypoint_ppo_smoke_checkpoint",
            "actor_state": actor.state_dict(),
            "value_state": value.state_dict(),
            "log_std": log_std.detach(),
            "config": asdict(config),
        },
        output_path.with_suffix(".pt"),
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor",
        type=Path,
        default=Path(
            "runs/pandareach_threewaypoint/bc_context12/BC-KMPC.pt"
        ),
    )
    parser.add_argument(
        "--koopman",
        type=Path,
        default=Path("runs/pandareach_threewaypoint/koopman/best.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/pandareach_threewaypoint/ppo_smoke/report.json"),
    )
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_smoke(
        args.actor,
        args.koopman,
        args.output,
        PPOSmokeConfig(
            rollout_steps=args.rollout_steps,
            updates=args.updates,
        ),
        args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
