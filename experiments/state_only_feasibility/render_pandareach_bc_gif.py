"""Render a BC checkpoint deployment as GIFs (deterministic mean policy)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_bc import (
    BCConfig,
    _batch_features,
    _make_builders,
    load_koopman,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo import (
    _actor_mean,
)


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(np.asarray(value).reshape(-1)[0])


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _np_array(value: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _annotate(frame: np.ndarray, label: str, row: dict[str, Any]) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 74), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        (
            f"step={row['step']:03d}  waypoint={row['active_waypoint'] + 1}/3  "
            f"distance={row['distance_m']:.3f}m"
        ),
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 230, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        result,
        (
            f"completed={row['waypoints_completed']}  "
            f"sparse return={row['return']:.1f}"
        ),
        (12, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (120, 255, 120),
        1,
        cv2.LINE_AA,
    )
    return result


def _rollout(
    env: gym.Env,
    actor_name: str,
    actor: torch.nn.Module,
    koopman: torch.nn.Module,
    center: np.ndarray,
    scale: np.ndarray,
    context_center: np.ndarray,
    context_scale: np.ndarray,
    action_limit: float,
    seed: int,
    max_steps: int,
    device: torch.device,
    label: str = "",
) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    observation, _ = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    episode_return = 0.0
    waypoints_completed = 0
    actor.eval()
    for step in range(max_steps):
        frame = env.render()
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        tcp = _numpy(observation["extra"]["tcp_pos"]).reshape(-1, 3)[0]
        active_goal = _numpy(observation["extra"]["active_goal"]).reshape(
            -1, 3
        )[0]
        active_waypoint = int(
            _numpy(observation["extra"]["active_waypoint_index"]).reshape(-1)[0]
        )
        distance_m = float(np.linalg.norm(tcp - active_goal))
        trace.append(
            {
                "step": step,
                "active_waypoint": active_waypoint,
                "distance_m": distance_m,
                "waypoints_completed": waypoints_completed,
                "return": episode_return,
            }
        )
        frames.append(
            _annotate(frame, label or f"{actor_name} - deployment", trace[-1])
        )

        normalized, lifted, context = _batch_features(
            observation,
            koopman,
            center,
            scale,
            context_center,
            context_scale,
            device,
        )
        with torch.no_grad():
            action = _actor_mean(actor_name, actor, normalized, lifted, context)
        action_rad = action.squeeze(0).cpu().numpy()
        observation, reward, terminated, truncated, info = env.step(
            np.clip(action_rad / action_limit, -1.0, 1.0).astype(np.float32)
        )
        episode_return += _scalar(reward)
        waypoints_completed = int(
            _numpy(info["waypoints_completed"]).reshape(-1)[0]
        )
        if bool(_scalar(terminated)) or bool(_scalar(truncated)):
            break
    return frames, trace, {
        "seed": seed,
        "steps": len(trace),
        "success": bool(_scalar(info["success"])),
        "waypoints_completed": waypoints_completed,
        "return": episode_return,
        "final_distance_m": _scalar(info["active_waypoint_distance"]),
    }


def _load_policy(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[str, torch.nn.Module, torch.nn.Module, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, int, float, float, str]:
    """Load a BC or PPO-trained actor for deterministic-mean deployment.

    Returns ``(actor_name, actor, koopman, center, scale, context_center,
    context_scale, max_steps, action_limit, goal_threshold, label)``.
    """

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    kind = payload.get("kind", "")
    if kind == "pandareach_threewaypoint_bc_actor":
        actor_name = str(payload["name"])
        config = BCConfig(**payload["config"])
        koopman, _ = load_koopman(Path(payload["koopman_path"]), device)
        actor = _make_builders(koopman, config, device)[actor_name]().to(device)
        actor.load_state_dict(payload["actor_state"])
        actor.eval()
        center = np.asarray(
            payload["normalizer"]["state_center"], dtype=np.float32
        )
        scale = np.asarray(
            payload["normalizer"]["state_scale"], dtype=np.float32
        )
        context_center = np.asarray(
            payload["normalizer"]["context_center"], dtype=np.float32
        )
        context_scale = np.asarray(
            payload["normalizer"]["context_scale"], dtype=np.float32
        )
        return (
            actor_name, actor, koopman, center, scale, context_center,
            context_scale, config.max_episode_steps, config.action_limit_rad,
            config.goal_threshold, f"{actor_name} - BC",
        )
    # PPO checkpoint (BC-finetune or from-scratch): actor_name/config fields
    # live in the training metadata, normalizers come from the Koopman payload
    # and the run's context_center/scale.
    from experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo import (
        _build_actor,
    )

    actor_name = str(payload["actor_name"])
    actor_config = BCConfig(**payload["actor_config"])
    koopman, koopman_payload = load_koopman(
        Path(payload["koopman_path"]), device
    )
    actor = _build_actor(actor_name, koopman, actor_config, device)
    actor.load_state_dict(payload["actor_state"])
    actor.eval()
    center = _np_array(koopman_payload["normalizer"]["center"])
    scale = _np_array(koopman_payload["normalizer"]["scale"])
    context_center = _np_array(payload["context_center"])
    context_scale = _np_array(payload["context_scale"])
    return (
        actor_name,
        actor,
        koopman,
        center,
        scale,
        context_center,
        context_scale,
        int(payload["config"]["max_episode_steps"]),
        actor_config.action_limit_rad,
        float(payload["config"]["goal_threshold"]),
        f"{actor_name} - PPO {int(payload['global_step']):,} steps",
    )


def render_deployment(
    checkpoint_path: Path,
    output_dir: Path,
    episodes: int,
    seed_start: int,
    device_name: str = "auto",
) -> dict[str, Any]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    (
        actor_name,
        actor,
        koopman,
        center,
        scale,
        context_center,
        context_scale,
        max_steps,
        action_limit,
        goal_threshold,
        label,
    ) = _load_policy(checkpoint_path, device)
    env = PandaArmOnlyActionWrapper(
        gym.make(
            "ACMPC-PandaReach3-v0",
            num_envs=1,
            sim_backend="gpu" if device.type == "cuda" else "cpu",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode="rgb_array",
            max_episode_steps=max_steps,
            goal_threshold=goal_threshold,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            episode_seed = seed_start + episode
            frames, trace, report = _rollout(
                env,
                actor_name,
                actor,
                koopman,
                center,
                scale,
                context_center,
                context_scale,
                action_limit,
                episode_seed,
                max_steps,
                device,
                label,
            )
            reports.append(report)
            # Stream half-resolution frames to keep GIF size/memory sane.
            gif_path = output_dir / (
                f"{actor_name}_ep{episode:02d}_seed{episode_seed}.gif"
            )
            with imageio.get_writer(
                gif_path, mode="I", duration=1.0 / 20.0
            ) as writer:
                for frame in frames:
                    half = cv2.resize(
                        frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                    )
                    writer.append_data(half.astype(np.uint8))
            print(
                json.dumps(
                    {"actor": actor_name, "gif": str(gif_path), **report}
                ),
                flush=True,
            )
    finally:
        env.close()
    summary = {
        "kind": "pandareach_threewaypoint_bc_deployment",
        "actor_name": actor_name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "policy": "deterministic mean action",
        "episodes": reports,
    }
    (output_dir / f"{actor_name}.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=20_290_804)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_deployment(
        args.checkpoint,
        args.output_dir,
        args.episodes,
        args.seed_start,
        args.device,
    )


if __name__ == "__main__":
    main()
