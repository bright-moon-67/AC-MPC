"""Causally replay official PickCube actions into a visual trajectory file.

Only the initial simulator state of each source episode is restored.  Every
subsequent observation is produced by an actual ``env.step(action)`` call, so
the saved triples are valid samples of ``(s_t, u_t, s_{t+1})``.  In
particular, this collector deliberately does not emulate replay modes that
force the recorded environment state between actions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import h5py
import numpy as np
import torch

# Importing the module registers the custom Gymnasium environment.
from experiments.maniskill_pick_visual.visual_pick_cube import (
    VISUAL_PICK_CUBE_ENV_ID,
)


FORMAT_NAME = "acmpc.visual_pickcube.causal.v2"
SOURCE_CONTROL_MODE = "pd_joint_delta_pos"


@dataclass(frozen=True)
class CollectionConfig:
    source_h5: Path
    output_h5: Path
    source_json: Path | None = None
    num_episodes: int | None = None
    seed: int = 0
    device: str = "auto"
    sim_backend: str = "physx_cpu"

    def validated(self) -> "CollectionConfig":
        source_h5 = self.source_h5.expanduser().resolve()
        source_json = (
            self.source_json.expanduser().resolve()
            if self.source_json is not None
            else source_h5.with_suffix(".json")
        )
        output_h5 = self.output_h5.expanduser().resolve()
        if source_h5.suffix != ".h5" or output_h5.suffix != ".h5":
            raise ValueError("source_h5 and output_h5 must use the .h5 suffix")
        if not source_h5.is_file():
            raise FileNotFoundError(source_h5)
        if not source_json.is_file():
            raise FileNotFoundError(source_json)
        if source_h5 == output_h5:
            raise ValueError("The output must not replace the source trajectory")
        if self.num_episodes is not None and self.num_episodes < 1:
            raise ValueError("num_episodes must be positive when provided")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda")
        if self.sim_backend not in {"auto", "physx_cpu", "physx_cuda"}:
            raise ValueError(
                "sim_backend must be one of: auto, physx_cpu, physx_cuda"
            )
        if self.device == "cpu" and self.sim_backend == "physx_cuda":
            raise ValueError("device=cpu conflicts with sim_backend=physx_cuda")
        if self.device == "cuda" and self.sim_backend == "physx_cpu":
            raise ValueError("device=cuda conflicts with sim_backend=physx_cpu")
        return CollectionConfig(
            source_h5=source_h5,
            source_json=source_json,
            output_h5=output_h5,
            num_episodes=self.num_episodes,
            seed=int(self.seed),
            device=self.device,
            sim_backend=self.sim_backend,
        )


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _unbatch(value: Any, *, name: str) -> np.ndarray:
    array = _numpy(value)
    if array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite values in observation leaf {name}")
    return array


def _extract_observation(observation: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
    """Extract the strict non-privileged model inputs from an RGB-D observation."""

    agent = observation["agent"]
    extra = observation["extra"]
    camera = observation["sensor_data"]["base_camera"]

    qpos = _unbatch(agent["qpos"], name="agent/qpos").reshape(-1)
    qvel = _unbatch(agent["qvel"], name="agent/qvel").reshape(-1)
    tcp_pose = _unbatch(extra["tcp_pose"], name="extra/tcp_pose").reshape(-1)
    if qpos.shape != (9,) or qvel.shape != (9,) or tcp_pose.shape != (7,):
        raise RuntimeError(
            "Expected Panda qpos/qvel/tcp_pose dimensions (9, 9, 7), got "
            f"{qpos.shape}, {qvel.shape}, {tcp_pose.shape}"
        )
    robot = np.concatenate((qpos, qvel, tcp_pose[:3])).astype(np.float32)

    rgb = _unbatch(camera["rgb"], name="sensor_data/base_camera/rgb")
    depth = _unbatch(camera["depth"], name="sensor_data/base_camera/depth")
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise RuntimeError(f"Expected HWC RGB, got {rgb.shape}")
    if depth.ndim == 2:
        depth = depth[..., None]
    if depth.ndim != 3 or depth.shape[-1] != 1:
        raise RuntimeError(f"Expected HW1 depth, got {depth.shape}")
    if rgb.shape[:2] != depth.shape[:2]:
        raise RuntimeError("RGB and depth resolutions differ")
    return robot, rgb.astype(np.uint8, copy=False), depth.astype(np.uint16)


def _read_state_at(group: h5py.Group, index: int) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            state[key] = _read_state_at(value, index)
        else:
            # Preserve the environment batch dimension expected by ManiSkill.
            state[key] = value[index : index + 1]
    return state


def _load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    env_info = metadata.get("env_info", {})
    if env_info.get("env_id") != "PickCube-v1":
        raise ValueError("Source metadata must describe PickCube-v1")
    if not isinstance(metadata.get("episodes"), list):
        raise ValueError("Source metadata does not contain an episodes list")
    return metadata


def _select_episodes(
    episodes: list[dict[str, Any]],
    *,
    num_episodes: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        episode
        for episode in episodes
        if episode.get("control_mode") == SOURCE_CONTROL_MODE
    ]
    if not eligible:
        raise ValueError(f"No {SOURCE_CONTROL_MODE} episodes were found")
    if num_episodes is None or num_episodes >= len(eligible):
        return eligible
    indices = np.random.default_rng(seed).choice(
        len(eligible), size=num_episodes, replace=False
    )
    return [eligible[index] for index in sorted(indices.tolist())]


def _resolve_backend(config: CollectionConfig) -> str:
    if config.sim_backend != "auto":
        return config.sim_backend
    return "physx_cuda" if config.device == "cuda" else "physx_cpu"


def _write_episode(
    output: h5py.File,
    name: str,
    *,
    robots: list[np.ndarray],
    rgbs: list[np.ndarray],
    depths: list[np.ndarray],
    actions: np.ndarray,
    raw_actions: np.ndarray,
    terminated: list[bool],
    truncated: list[bool],
    success: list[bool],
    source_episode: Mapping[str, Any],
) -> None:
    transition_count = len(actions)
    if not (
        len(robots) == len(rgbs) == len(depths) == transition_count + 1
        and len(raw_actions) == transition_count
        and len(terminated) == len(truncated) == len(success) == transition_count
    ):
        raise RuntimeError(f"T/T+1 alignment failed for {name}")
    group = output.create_group(name, track_order=True)
    group.attrs["source_episode_id"] = int(source_episode["episode_id"])
    group.attrs["episode_seed"] = int(source_episode.get("episode_seed", -1))
    group.create_dataset("robot", data=np.stack(robots), dtype=np.float32)
    group.create_dataset(
        "rgb",
        data=np.stack(rgbs),
        dtype=np.uint8,
        compression="gzip",
        compression_opts=5,
    )
    group.create_dataset(
        "depth",
        data=np.stack(depths),
        dtype=np.uint16,
        compression="gzip",
        compression_opts=5,
    )
    group.create_dataset("actions", data=actions, dtype=np.float32)
    group.create_dataset("raw_actions", data=raw_actions, dtype=np.float32)
    group.create_dataset("terminated", data=terminated, dtype=np.bool_)
    group.create_dataset("truncated", data=truncated, dtype=np.bool_)
    group.create_dataset("success", data=success, dtype=np.bool_)


def collect(config: CollectionConfig) -> dict[str, Any]:
    config = config.validated()
    output_json = config.output_h5.with_suffix(".json")
    if config.output_h5.exists() or output_json.exists():
        raise FileExistsError(
            f"Refusing to overwrite {config.output_h5} or {output_json}"
        )
    config.output_h5.parent.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(config.source_json)  # type: ignore[arg-type]
    episodes = _select_episodes(
        metadata["episodes"],
        num_episodes=config.num_episodes,
        seed=config.seed,
    )
    backend = _resolve_backend(config)

    env = gym.make(
        VISUAL_PICK_CUBE_ENV_ID,
        num_envs=1,
        # Use the explicit spelling accepted by ManiSkill's observation-mode
        # parser; it produces sensor_data/base_camera/{rgb, depth}.
        obs_mode="rgb+depth",
        control_mode=SOURCE_CONTROL_MODE,
        reward_mode="none",
        render_mode=None,
        sim_backend=backend,
    )
    actual_device = env.unwrapped.device.type
    if config.device != "auto" and config.device != actual_device:
        env.close()
        raise RuntimeError(
            f"Requested device={config.device}, environment uses {actual_device}"
        )

    written: list[dict[str, Any]] = []
    try:
        with h5py.File(config.source_h5, "r") as source, h5py.File(
            config.output_h5, "x"
        ) as output:
            output.attrs["format"] = FORMAT_NAME
            output.attrs["causal_replay"] = True
            output.attrs["env_id"] = VISUAL_PICK_CUBE_ENV_ID
            output.attrs["control_mode"] = SOURCE_CONTROL_MODE
            output.attrs["sim_backend"] = backend
            output.attrs["device"] = actual_device
            output.attrs["goal_visible"] = True
            output.attrs["source_h5"] = str(config.source_h5)
            output.attrs["actions_are_applied"] = True
            output.attrs["action_semantics"] = (
                "actions are clipped to normalized environment bounds; "
                "raw_actions preserve source PPO policy outputs"
            )

            expected_action_shape = tuple(env.unwrapped.single_action_space.shape)
            action_low = np.asarray(
                env.unwrapped.single_action_space.low, dtype=np.float32
            )
            action_high = np.asarray(
                env.unwrapped.single_action_space.high, dtype=np.float32
            )
            output.attrs["action_low"] = action_low
            output.attrs["action_high"] = action_high
            for output_index, episode in enumerate(episodes):
                source_name = f"traj_{int(episode['episode_id'])}"
                if source_name not in source:
                    raise KeyError(f"Missing source group {source_name}")
                source_group = source[source_name]
                if "env_states" not in source_group:
                    raise ValueError(f"{source_name} has no env_states")
                raw_actions = np.asarray(source_group["actions"], dtype=np.float32)
                if raw_actions.ndim != 2 or tuple(raw_actions.shape[1:]) != expected_action_shape:
                    raise ValueError(
                        f"{source_name} action shape {raw_actions.shape} does not match "
                        f"{expected_action_shape}"
                    )
                if len(raw_actions) == 0 or not np.isfinite(raw_actions).all():
                    raise ValueError(f"{source_name} actions must be non-empty and finite")
                # The recorded PPO distribution is unbounded. ManiSkill clips
                # it in controller preprocessing; system identification must
                # therefore save the command that actually reached the plant.
                actions = np.clip(raw_actions, action_low, action_high).astype(
                    np.float32, copy=False
                )
                if len(actions) != int(episode["elapsed_steps"]):
                    raise ValueError(f"Metadata/action length mismatch in {source_name}")
                for leaf_name, leaf in _iter_h5_leaves(source_group["env_states"]):
                    if leaf.shape[0] != len(actions) + 1:
                        raise ValueError(
                            f"{source_name}/{leaf_name} must have T+1 states"
                        )

                episode_seed = int(episode.get("episode_seed", config.seed))
                env.reset(seed=episode_seed)
                initial_state = _read_state_at(source_group["env_states"], 0)
                env.unwrapped.set_state_dict(initial_state)
                observation = env.unwrapped.get_obs()
                robot, rgb, depth = _extract_observation(observation)
                robots, rgbs, depths = [robot], [rgb], [depth]
                terminated_values: list[bool] = []
                truncated_values: list[bool] = []
                success_values: list[bool] = []

                for action in actions:
                    observation, _, terminated, truncated, info = env.step(action)
                    robot, rgb, depth = _extract_observation(observation)
                    robots.append(robot)
                    rgbs.append(rgb)
                    depths.append(depth)
                    terminated_values.append(bool(_unbatch(terminated, name="terminated")))
                    truncated_values.append(bool(_unbatch(truncated, name="truncated")))
                    success_values.append(
                        bool(_unbatch(info.get("success", False), name="success"))
                    )

                output_name = f"traj_{output_index}"
                _write_episode(
                    output,
                    output_name,
                    robots=robots,
                    rgbs=rgbs,
                    depths=depths,
                    actions=actions,
                    raw_actions=raw_actions,
                    terminated=terminated_values,
                    truncated=truncated_values,
                    success=success_values,
                    source_episode=episode,
                )
                written.append(
                    {
                        "episode_name": output_name,
                        "source_episode_id": int(episode["episode_id"]),
                        "episode_seed": episode_seed,
                        "elapsed_steps": len(actions),
                        "success": bool(success_values[-1]),
                        "clipped_action_scalars": int(
                            np.count_nonzero(actions != raw_actions)
                        ),
                    }
                )
    finally:
        env.close()

    result = {
        "format": FORMAT_NAME,
        "config": {
            **asdict(config),
            "source_h5": str(config.source_h5),
            "source_json": str(config.source_json),
            "output_h5": str(config.output_h5),
        },
        "env_id": VISUAL_PICK_CUBE_ENV_ID,
        "control_mode": SOURCE_CONTROL_MODE,
        "sim_backend": backend,
        "episodes": written,
    }
    with output_json.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def _iter_h5_leaves(
    group: h5py.Group, prefix: str = ""
) -> list[tuple[str, h5py.Dataset]]:
    leaves: list[tuple[str, h5py.Dataset]] = []
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Group):
            leaves.extend(_iter_h5_leaves(value, path))
        else:
            leaves.append((path, value))
    return leaves


def _parse_args() -> CollectionConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--sim-backend",
        choices=("auto", "physx_cpu", "physx_cuda"),
        default="physx_cpu",
    )
    args = parser.parse_args()
    return CollectionConfig(
        source_h5=args.source_h5,
        source_json=args.source_json,
        output_h5=args.output_h5,
        num_episodes=args.num_episodes,
        seed=args.seed,
        device=args.device,
        sim_backend=args.sim_backend,
    )


if __name__ == "__main__":
    summary = collect(_parse_args())
    print(json.dumps(summary, indent=2))
