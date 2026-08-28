from __future__ import annotations

import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from manisoft.envs.vlm_env import VisionLanguageManipulationEnvironment
from manisoft.muscle import SplineMuscle
from manisoft.utils import (
    KOOPMAN_PHYSICAL_STATE_DIM,
    KOOPMAN_TIP_POSITION_SLICE,
    koopman_section_state,
    load_yaml,
)

from .delta_action_wrapper import DeltaActionWrapper


MANISOFT_WAYPOINT_REFERENCE_FILES = (
    "ref_4cm/reference.npz",
    "ref_8cm/reference.npz",
    "ref_12cm/reference.npz",
)
MANISOFT_WAYPOINT_ACTION_FILES = (
    "actions/u_scale_0p25.json",
    "actions/u_scale_0p50.json",
    "actions/u_scale_0p75.json",
)

# A waypoint is passed on the first simulation step whose tip enters this
# ball.  There is deliberately no dwell/streak requirement: reaching 5 mm
# immediately advances to the next waypoint (or terminates at waypoint 3).
MANISOFT_WAYPOINT_SUCCESS_THRESHOLD = 0.005
MANISOFT_WAYPOINT_SUCCESS_STREAK = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManiSoftWaypointReferenceBank:
    """Validated three-waypoint state/action reference bank."""

    states: np.ndarray
    actions: np.ndarray
    reference_paths: tuple[Path, ...]
    manifest_path: Path
    manifest_sha256: str
    scenario_sha256: str

    @property
    def triplet_count(self) -> int:
        return int(self.states.shape[0])


def load_manisoft_waypoint_reference_bank(
    root: str | Path,
) -> ManiSoftWaypointReferenceBank:
    """Load a certified bank shaped ``[triplet, waypoint, feature]``."""

    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing waypoint-bank manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("kind") != (
        "manisoft_certified_three_waypoint_reference_bank"
    ):
        raise ValueError(f"Unsupported waypoint-bank manifest: {manifest_path}")
    rows = manifest.get("triplets")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Waypoint-bank manifest must contain at least one triplet")
    if manifest.get("triplet_count") != len(rows):
        raise ValueError("Waypoint-bank triplet_count does not match its contents")
    if manifest.get("waypoint_count") != 3:
        raise ValueError("Waypoint-bank must contain exactly three waypoints per triplet")

    state_triplets: list[np.ndarray] = []
    action_triplets: list[np.ndarray] = []
    reference_paths: list[Path] = []
    for triplet_index, row in enumerate(rows):
        if row.get("index") != triplet_index:
            raise ValueError("Waypoint-bank triplet indices must be contiguous")
        waypoints = row.get("waypoints")
        if not isinstance(waypoints, list) or len(waypoints) != 3:
            raise ValueError(f"Triplet {triplet_index} must contain three waypoints")
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        for waypoint_index, waypoint in enumerate(waypoints):
            if waypoint.get("index") != waypoint_index:
                raise ValueError(
                    f"Triplet {triplet_index} waypoint indices must be contiguous"
                )
            relative = Path(str(waypoint.get("reference", "")))
            path = (root / relative).resolve()
            if root not in path.parents or not path.is_file():
                raise FileNotFoundError(f"Missing or invalid waypoint reference: {path}")
            if _sha256(path) != waypoint.get("sha256"):
                raise ValueError(f"Waypoint reference hash mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                state = np.asarray(archive["reference_state"], dtype=np.float32).reshape(-1)
                action = np.asarray(archive["reference_action"], dtype=np.float32).reshape(-1)
                stored_tip = np.asarray(
                    archive["reference_tip_position"], dtype=np.float32
                ).reshape(-1)
            if state.shape != (KOOPMAN_PHYSICAL_STATE_DIM,) or action.shape != (18,):
                raise ValueError(f"{path} must contain a 45-D state and 18-D action")
            if stored_tip.shape != (3,) or not np.allclose(
                stored_tip, state[KOOPMAN_TIP_POSITION_SLICE], atol=1e-7, rtol=1e-6
            ):
                raise ValueError(f"Reference tip does not match reference state: {path}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"{path} contains NaN or Inf")
            if np.max(np.abs(action)) > 0.30 + 1e-7:
                raise ValueError(f"{path} exceeds the absolute action limit 0.30")
            states.append(state)
            actions.append(action)
            reference_paths.append(path)
        state_triplets.append(np.stack(states))
        action_triplets.append(np.stack(actions))
    return ManiSoftWaypointReferenceBank(
        states=np.stack(state_triplets),
        actions=np.stack(action_triplets),
        reference_paths=tuple(reference_paths),
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        scenario_sha256=str(manifest.get("scenario_sha256", "")),
    )


def load_manisoft_waypoint_references(
    root: str | Path,
) -> tuple[np.ndarray, np.ndarray, tuple[Path, ...], tuple[Path, ...]]:
    """Load and cross-check the fixed 4/8/12 cm waypoint references."""

    root = Path(root).expanduser().resolve()
    reference_paths = tuple(root / name for name in MANISOFT_WAYPOINT_REFERENCE_FILES)
    action_paths = tuple(root / name for name in MANISOFT_WAYPOINT_ACTION_FILES)
    missing = [path for path in (*reference_paths, *action_paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing ManiSoft waypoint reference files: "
            + ", ".join(map(str, missing))
        )

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for reference_path, action_path in zip(reference_paths, action_paths):
        with np.load(reference_path, allow_pickle=False) as archive:
            state = np.asarray(archive["reference_state"], dtype=np.float32).reshape(-1)
            action = np.asarray(archive["reference_action"], dtype=np.float32).reshape(-1)
        if state.shape != (KOOPMAN_PHYSICAL_STATE_DIM,) or action.shape != (18,):
            raise ValueError(
                f"{reference_path} must contain a 45-D state and 18-D action"
            )
        action_payload = json.loads(action_path.read_text(encoding="utf-8"))
        reproducible_action = np.asarray(
            action_payload.get("u"), dtype=np.float32
        ).reshape(-1)
        if reproducible_action.shape != (18,):
            raise ValueError(f"{action_path} must contain an 18-D 'u' action")
        if not np.allclose(action, reproducible_action, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"Reference action in {reference_path} does not match {action_path}"
            )
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{reference_path} contains NaN or Inf")
        states.append(state)
        actions.append(action)
    return (
        np.stack(states),
        np.stack(actions),
        reference_paths,
        action_paths,
    )


class ManiSoftTipTrackingEnv(gym.Env):
    """固定目标的ManiSoft末端跟踪环境，输出45维物理状态。"""

    def __init__(
        self,
        scenario_path: str | Path,
        target_offset: Sequence[float] = (0.0, 0.005, 0.0),
        target_tip: Sequence[float] | None = None,
        episode_steps: int = 300,
        success_threshold: float = 0.0015,
        success_streak: int = 5,
        absolute_action_limit: float = 0.30,
        muscle_torque_scale: float = 30.0,
        progress_reward_scale: float = 1.0,
    ):
        super().__init__()

        self.scenario_path = Path(scenario_path).resolve()
        self.target_offset = np.asarray(target_offset, dtype=np.float32)
        self.fixed_target_tip = (
            None
            if target_tip is None
            else np.asarray(target_tip, dtype=np.float32)
        )

        if self.target_offset.shape != (3,):
            raise ValueError("target_offset必须是3维")
        if self.fixed_target_tip is not None and self.fixed_target_tip.shape != (3,):
            raise ValueError("target_tip必须是3维")
        if self.fixed_target_tip is None and np.linalg.norm(self.target_offset) <= 0:
            raise ValueError("target_offset不能为零")

        self.episode_steps = int(episode_steps)
        self.success_threshold = float(success_threshold)
        self.required_success_streak = int(success_streak)
        self.absolute_action_limit = float(absolute_action_limit)
        self.muscle_torque_scale = float(muscle_torque_scale)
        self.progress_reward_scale = float(progress_reward_scale)
        if self.absolute_action_limit <= 0:
            raise ValueError("absolute_action_limit必须为正")
        if self.muscle_torque_scale <= 0:
            raise ValueError("muscle_torque_scale必须为正")
        if self.progress_reward_scale <= 0:
            raise ValueError("progress_reward_scale必须为正")

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(KOOPMAN_PHYSICAL_STATE_DIM,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-self.absolute_action_limit,
            high=self.absolute_action_limit,
            shape=(18,),
            dtype=np.float32,
        )

        self.sim = None
        self.muscle = None
        self.target_tip = None
        self.previous_distance = None
        self.target_scale = None
        self.step_count = 0
        self.success_count = 0

    def _physical_state(self) -> np.ndarray:
        soft = self.sim._backend.softrobot_state
        state = koopman_section_state(soft)

        if state.shape != (KOOPMAN_PHYSICAL_STATE_DIM,):
            raise RuntimeError(f"错误状态维度：{state.shape}")
        if not np.isfinite(state).all():
            raise FloatingPointError("ManiSoft状态出现NaN或Inf")

        return state

    @staticmethod
    def _tip_position(state: np.ndarray) -> np.ndarray:
        return state[KOOPMAN_TIP_POSITION_SLICE]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)

        self.sim = None
        self.muscle = None
        gc.collect()

        config = load_yaml(self.scenario_path)
        config.pop("renderer", None)

        self.sim = VisionLanguageManipulationEnvironment.from_dict(config)

        soft = self.sim._backend.softrobot_state
        self.muscle = SplineMuscle(
            robot_length=float(np.sum(soft.element_lengths)),
            robot_num_elements=int(soft.num_elements),
            number_of_control_points=6,
            muscle_torque_scale=self.muscle_torque_scale,
        )

        # Settle with zero activation for 1 s at 50 Hz, matching the Koopman
        # data collector (settle-seconds=1).  This relaxes the rod to the
        # gravitational equilibrium the model was trained on; without it the
        # closed-loop starts from the upright initial pose (out of
        # distribution) and the MPC diverges.
        self.muscle.set_activation(np.zeros((6, 3), dtype=np.float64))

        def zero_torque(element_lengths: np.ndarray) -> np.ndarray:
            return self.muscle.evaluate(element_lengths)

        for _ in range(50):
            self.sim.step_with_torque_callback(zero_torque)

        observation = self._physical_state()
        self.target_tip = (
            self._tip_position(observation) + self.target_offset
            if self.fixed_target_tip is None
            else self.fixed_target_tip
        ).astype(np.float32)

        self.previous_distance = float(
            np.linalg.norm(
                self._tip_position(observation) - self.target_tip
            )
        )
        # Offset tasks retain their original normalization.  An explicit
        # reference tip may be much farther away, so normalize progress and
        # distance reward by its actual reset distance instead of the default
        # 5 mm offset.
        self.target_scale = max(
            float(np.linalg.norm(self.target_offset))
            if self.fixed_target_tip is None
            else self.previous_distance,
            np.finfo(np.float32).eps,
        )
        self.step_count = 0
        self.success_count = 0

        return observation, {
            "target_tip": self.target_tip.copy(),
            "distance": self.previous_distance,
        }

    def step(self, absolute_action: np.ndarray):
        action = np.asarray(
            absolute_action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape != (18,):
            raise ValueError(f"动作维度错误：{action.shape}")
        if not np.isfinite(action).all():
            raise FloatingPointError("动作出现NaN或Inf")

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        self.muscle.set_activation(action.reshape(6, 3))

        def current_torque(element_lengths: np.ndarray) -> np.ndarray:
            return self.muscle.evaluate(element_lengths)

        # Match the 50 Hz Koopman data collector: keep the activation fixed
        # while refreshing the length-dependent distributed torque at every
        # physics substep.
        self.sim.step_with_torque_callback(current_torque)

        observation = self._physical_state()
        distance = float(
            np.linalg.norm(
                self._tip_position(observation) - self.target_tip
            )
        )

        if self.target_scale is None:
            raise RuntimeError("Environment must be reset before step")
        target_scale = self.target_scale
        progress = (self.previous_distance - distance) / target_scale

        normalized_action = action / self.absolute_action_limit

        reward = (
            self.progress_reward_scale * float(progress)
            - 0.01
            - 0.001 * float(np.mean(normalized_action ** 2))
        )

        self.step_count += 1

        if distance <= self.success_threshold:
            self.success_count += 1
        else:
            self.success_count = 0

        terminated = (
            self.success_count >= self.required_success_streak
        )
        truncated = (
            self.step_count >= self.episode_steps and not terminated
        )

        if terminated:
            reward += 5.0

        self.previous_distance = distance

        info = {
            "distance": distance,
            "target_tip": self.target_tip.copy(),
            "is_success": bool(terminated),
            "success_streak": self.success_count,
        }

        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.sim = None
        self.muscle = None
        gc.collect()


class ManiSoftThreeWaypointTrackingEnv(ManiSoftTipTrackingEnv):
    """Track one coherently sampled three-waypoint reference triplet."""

    waypoint_count = 3

    def __init__(
        self,
        scenario_path: str | Path,
        waypoint_tips: Sequence[Sequence[float]] | np.ndarray,
        episode_steps: int = 300,
        success_threshold: float = MANISOFT_WAYPOINT_SUCCESS_THRESHOLD,
        success_streak: int = MANISOFT_WAYPOINT_SUCCESS_STREAK,
        waypoint_event_reward: float = 3.0,
        absolute_action_limit: float = 0.30,
        progress_reward_scale: float = 1.0,
    ) -> None:
        waypoint_bank = np.asarray(waypoint_tips, dtype=np.float32)
        if waypoint_bank.shape == (self.waypoint_count, 3):
            waypoint_bank = waypoint_bank[None, ...]
        if (
            waypoint_bank.ndim != 3
            or waypoint_bank.shape[1:] != (self.waypoint_count, 3)
            or waypoint_bank.shape[0] < 1
        ):
            raise ValueError("waypoint_tips必须是[3,3]或[N,3,3]")
        if not np.isfinite(waypoint_bank).all():
            raise ValueError("waypoint_tips出现NaN或Inf")
        if waypoint_event_reward < 0:
            raise ValueError("waypoint_event_reward必须非负")
        super().__init__(
            scenario_path,
            target_tip=waypoint_bank[0, 0],
            episode_steps=episode_steps,
            success_threshold=success_threshold,
            success_streak=success_streak,
            absolute_action_limit=absolute_action_limit,
            progress_reward_scale=progress_reward_scale,
        )
        self.waypoint_tip_bank = waypoint_bank.copy()
        self.fixed_waypoints = self.waypoint_tip_bank[0].copy()
        self.active_waypoint_triplet_index = 0
        self.waypoint_event_reward = float(waypoint_event_reward)
        self.active_waypoint_index = 0
        self.waypoints_completed = 0

    @property
    def waypoints(self) -> np.ndarray:
        return self.fixed_waypoints

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = super().reset(seed=seed, options=options)
        forced_index = None if options is None else options.get("waypoint_triplet_index")
        if forced_index is None:
            triplet_index = int(self.np_random.integers(len(self.waypoint_tip_bank)))
        else:
            if not isinstance(forced_index, (int, np.integer)):
                raise ValueError("waypoint_triplet_index必须是整数")
            triplet_index = int(forced_index)
            if not 0 <= triplet_index < len(self.waypoint_tip_bank):
                raise ValueError("waypoint_triplet_index超出参考库范围")
        self.active_waypoint_triplet_index = triplet_index
        self.fixed_waypoints = self.waypoint_tip_bank[triplet_index].copy()
        self.active_waypoint_index = 0
        self.waypoints_completed = 0
        self.target_tip = self.fixed_waypoints[0].copy()
        tip = self._tip_position(observation)
        distances = np.linalg.norm(self.fixed_waypoints - tip[None, :], axis=1)
        self.previous_distance = float(distances[0])
        self.target_scale = max(self.previous_distance, np.finfo(np.float32).eps)
        info.update(
            {
                "target_tip": self.target_tip.copy(),
                "waypoints": self.fixed_waypoints.copy(),
                "waypoint_triplet_index": self.active_waypoint_triplet_index,
                "active_waypoint_index": self.active_waypoint_index,
                "waypoints_completed": self.waypoints_completed,
                "waypoint_passed": False,
                "all_waypoint_distances": distances.astype(np.float32),
            }
        )
        return observation, info

    def step(self, absolute_action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(
            absolute_action
        )
        reached_distance = float(info["distance"])
        waypoint_passed = False

        # Entering the 5 mm target ball immediately advances the stage; the
        # final target terminates the episode on that same simulation step.
        if terminated and self.active_waypoint_index < self.waypoint_count - 1:
            terminated = False
            waypoint_passed = True
            self.waypoints_completed += 1
            self.active_waypoint_index += 1
            self.success_count = 0
            self.target_tip = self.fixed_waypoints[
                self.active_waypoint_index
            ].copy()
            next_distance = float(
                np.linalg.norm(self._tip_position(observation) - self.target_tip)
            )
            self.previous_distance = next_distance
            self.target_scale = max(next_distance, np.finfo(np.float32).eps)
            # Replace the single-task terminal bonus with the reward for
            # passing one waypoint. The final waypoint receives both this
            # event reward and the extra all-waypoints completion reward.
            reward = reward - 5.0 + self.waypoint_event_reward
            truncated = self.step_count >= self.episode_steps

        if terminated:
            self.waypoints_completed = self.waypoint_count
            reward += self.waypoint_event_reward

        tip = self._tip_position(observation)
        distances = np.linalg.norm(self.fixed_waypoints - tip[None, :], axis=1)
        active_distance = float(distances[self.active_waypoint_index])
        info.update(
            {
                "distance": active_distance,
                "target_tip": self.target_tip.copy(),
                "waypoints": self.fixed_waypoints.copy(),
                "waypoint_triplet_index": self.active_waypoint_triplet_index,
                "active_waypoint_index": self.active_waypoint_index,
                "waypoints_completed": self.waypoints_completed,
                "waypoint_passed": waypoint_passed,
                "success_streak": self.success_count,
                "reached_waypoint_distance": reached_distance,
                "all_waypoint_distances": distances.astype(np.float32),
                "is_success": bool(terminated),
            }
        )
        return observation, float(reward), terminated, truncated, info


def make_manisoft_tracking_env(
    scenario_path: str | Path,
    *,
    target_offset=(0.0, 0.005, 0.0),
    target_tip=None,
    episode_steps: int = 300,
    absolute_action_limit: float = 0.30,
):
    base_env = ManiSoftTipTrackingEnv(
        scenario_path,
        target_offset=target_offset,
        target_tip=target_tip,
        episode_steps=episode_steps,
        absolute_action_limit=absolute_action_limit,
    )
    return DeltaActionWrapper(
        base_env,
        expected_observation_dim=KOOPMAN_PHYSICAL_STATE_DIM,
    )
