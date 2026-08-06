"""Render initial-versus-trained deterministic PandaReach3 PPO policies."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
    PandaReachThreeWaypointEnv,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_bc import (
    B0Actor,
    BCConfig,
    TASK_CONTEXT_DIM,
    load_koopman,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_ppo import (
    H1_MIN_FINAL_GAIN,
    _actor_mean,
    _build_actor,
    _build_value,
    _features,
    _initialize_ppo_modules,
)


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    return float(np.asarray(value).reshape(-1)[0])


def _frame(env: gym.Env) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return frame.astype(np.uint8, copy=False)


def _annotate(
    frame: np.ndarray, label: str, row: dict[str, Any]
) -> np.ndarray:
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
    state_center: torch.Tensor,
    state_scale: torch.Tensor,
    context_center: torch.Tensor,
    context_scale: torch.Tensor,
    action_limit: float,
    seed: int,
    max_steps: int,
    device: torch.device,
) -> tuple[list[bytes], list[dict[str, Any]], dict[str, Any]]:
    observation, _ = env.reset(seed=seed)
    initial_distance = float(
        torch.linalg.vector_norm(
            observation["extra"]["tcp_pos"]
            - observation["extra"]["active_goal"],
            dim=-1,
        )[0]
    )
    encoded_frames: list[bytes] = []
    trace: list[dict[str, Any]] = []
    distances: list[float] = []
    absolute_actions: list[np.ndarray] = []
    sparse_return = 0.0
    final_info: dict[str, Any] = {}
    actor.eval()
    for step in range(1, max_steps + 1):
        normalized_state, lifted, context = _features(
            observation,
            koopman,
            state_center,
            state_scale,
            context_center,
            context_scale,
            device,
        )
        with torch.no_grad():
            action = _actor_mean(
                actor_name, actor, normalized_state, lifted, context
            )
        absolute_actions.append(action.detach().abs().cpu().numpy().reshape(-1))
        observation, reward, terminated, truncated, info = env.step(
            torch.clamp(action / action_limit, -1.0, 1.0)
        )
        sparse_return += _scalar(reward)
        distance = _scalar(info["active_waypoint_distance"])
        distances.append(distance)
        row = {
            "step": step,
            "active_waypoint": int(_scalar(info["active_waypoint_index"])),
            "waypoints_completed": int(_scalar(info["waypoints_completed"])),
            "distance_m": distance,
            "return": sparse_return,
        }
        trace.append(row)
        ok, encoded = cv2.imencode(
            ".jpg", _frame(env)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok:
            raise RuntimeError("Could not encode deployment frame")
        encoded_frames.append(encoded.tobytes())
        final_info = info
        if bool(_scalar(terminated)) or bool(_scalar(truncated)):
            break
    actions = np.concatenate(absolute_actions) if absolute_actions else np.zeros(1)
    return encoded_frames, trace, {
        "seed": seed,
        "steps": len(trace),
        "sparse_return": sparse_return,
        "success": bool(_scalar(final_info.get("success", False))),
        "waypoints_completed": int(
            _scalar(final_info.get("waypoints_completed", 0))
        ),
        "initial_active_distance_m": initial_distance,
        "minimum_active_distance_m": float(np.min(distances)),
        "final_active_distance_m": float(distances[-1]),
        "mean_absolute_action_rad": float(actions.mean()),
        "action_bound_fraction": float(np.mean(actions >= 0.99 * action_limit)),
    }


def render_comparison(
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
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_name = str(payload["actor_name"])
    actor_config = BCConfig(**payload["actor_config"])
    koopman_path = Path(payload["koopman_path"])
    koopman, koopman_payload = load_koopman(koopman_path, device)
    seed = int(payload["config"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    legacy_b0 = (
        actor_name == "B0"
        and payload.get("architecture_version")
        != "standard_raw_mlp_256x256_v1"
    )
    if legacy_b0:
        initial_actor = B0Actor(
            koopman.state_dim + TASK_CONTEXT_DIM,
            actor_config.b0_hidden_dim,
            actor_config.action_limit_rad,
        ).to(device)
        trained_actor = B0Actor(
            koopman.state_dim + TASK_CONTEXT_DIM,
            actor_config.b0_hidden_dim,
            actor_config.action_limit_rad,
        ).to(device)
    else:
        initial_actor = _build_actor(actor_name, koopman, actor_config, device)
        initial_value = _build_value(actor_name, koopman, device)
        h1_min_final_gain = H1_MIN_FINAL_GAIN
        if payload.get("architecture_version") == (
            "minimal_h_orthogonal_nonzero_v2"
        ):
            h1_min_final_gain = 0.001
        _initialize_ppo_modules(
            actor_name,
            initial_actor,
            initial_value,
            h1_min_final_gain=h1_min_final_gain,
        )
        trained_actor = _build_actor(actor_name, koopman, actor_config, device)
    trained_actor.load_state_dict(payload["actor_state"])
    state_center = torch.as_tensor(
        koopman_payload["normalizer"]["center"], device=device
    )
    state_scale = torch.as_tensor(
        koopman_payload["normalizer"]["scale"], device=device
    )
    context_center = payload["context_center"].to(device)
    context_scale = payload["context_scale"].to(device)
    env = PandaArmOnlyActionWrapper(
        gym.make(
            "ACMPC-PandaReach3-v0",
            num_envs=1,
            sim_backend="gpu" if device.type == "cuda" else "cpu",
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode="rgb_array",
            max_episode_steps=int(payload["config"]["max_episode_steps"]),
            goal_threshold=float(payload["config"]["goal_threshold"]),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_reports: list[dict[str, Any]] = []
    trained_reports: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            episode_seed = seed_start + episode
            initial_frames, initial_trace, initial_report = _rollout(
                env,
                actor_name,
                initial_actor,
                koopman,
                state_center,
                state_scale,
                context_center,
                context_scale,
                actor_config.action_limit_rad,
                episode_seed,
                int(payload["config"]["max_episode_steps"]),
                device,
            )
            trained_frames, trained_trace, trained_report = _rollout(
                env,
                actor_name,
                trained_actor,
                koopman,
                state_center,
                state_scale,
                context_center,
                context_scale,
                actor_config.action_limit_rad,
                episode_seed,
                int(payload["config"]["max_episode_steps"]),
                device,
            )
            video_path = output_dir / f"episode_{episode:02d}_seed_{episode_seed}.mp4"
            frame_count = max(len(initial_frames), len(trained_frames))
            with imageio.get_writer(
                video_path,
                fps=20,
                codec="libx264",
                quality=8,
                macro_block_size=None,
            ) as writer:
                for index in range(frame_count):
                    initial_index = min(index, len(initial_frames) - 1)
                    trained_index = min(index, len(trained_frames) - 1)
                    initial_frame = cv2.imdecode(
                        np.frombuffer(initial_frames[initial_index], np.uint8),
                        cv2.IMREAD_COLOR,
                    )[..., ::-1]
                    trained_frame = cv2.imdecode(
                        np.frombuffer(trained_frames[trained_index], np.uint8),
                        cv2.IMREAD_COLOR,
                    )[..., ::-1]
                    comparison = np.concatenate(
                        (
                            _annotate(
                                initial_frame,
                                f"{actor_name} - initialization",
                                initial_trace[initial_index],
                            ),
                            _annotate(
                                trained_frame,
                                f"{actor_name} - checkpoint {payload['global_step']:,} steps",
                                trained_trace[trained_index],
                            ),
                        ),
                        axis=1,
                    )
                    writer.append_data(comparison)
            initial_reports.append(initial_report)
            trained_reports.append(trained_report)
    finally:
        env.close()
    report = {
        "kind": "pandareach_threewaypoint_ppo_deployment_comparison",
        "actor_name": actor_name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_update": int(payload["update"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "policy": "deterministic mean action",
        "episodes": episodes,
        "initialization": initial_reports,
        "trained": trained_reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20_290_804)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = render_comparison(
        args.checkpoint,
        args.output_dir,
        args.episodes,
        args.seed_start,
        args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
