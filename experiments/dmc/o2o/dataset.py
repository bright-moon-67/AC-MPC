"""ExORL conversion and replay sampling for offline-to-online learning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.dmc.reward_oracle import walker_run_exact_reward_numpy
from experiments.dmc.tasks.registry import get_task_spec


DATASET_KIND = "acmpc_exorl_cartpole_transitions_v1"
WALKER_DATASET_KIND = "acmpc_exorl_walker_run_transitions_v1"
TDMPC2_DATASET_KIND = "acmpc_tdmpc2_dmc_transitions_v1"
DATASET_KEYS = (
    "observation",
    "action",
    "reward",
    "discount",
    "next_observation",
    "episode_id",
    "episode_step",
    "mc_return",
)
BOUNDARY_KEYS = ("terminated", "truncated")
CANONICAL_DATASET_KEYS = (
    "observation",
    "action",
    "reward",
    "discount",
    "next_observation",
    "episode_id",
    "episode_step",
    *BOUNDARY_KEYS,
    "mc_return",
)

_RELEASE_EPISODE_NAME = re.compile(
    r"episode_(?P<index>[0-9]+)_(?P<transitions>[0-9]+)\.npz"
)
_LEGACY_EPISODE_NAME = re.compile(
    r"[0-9]{8}T[0-9]{6}_(?P<index>[0-9]+)_(?P<transitions>[0-9]+)\.npz"
)


@dataclass(frozen=True)
class _EpisodeSource:
    path: Path
    index: int
    transitions: int
    release_schema: bool


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def temporal_stratified_episode_indices(
    *,
    source_total_episodes: int = 10_000,
    temporal_deciles: int = 10,
    episodes_per_decile: int = 100,
) -> tuple[int, ...]:
    """Return deterministic, evenly spaced episode indices in every time block.

    Each temporal block is divided into ``episodes_per_decile`` equal-width
    micro-strata and the first episode of every micro-stratum is selected.
    For the public ExORL Proto10M release this yields indices 0, 10, ..., 990
    in the first decile and the analogous 100 indices in each later decile.
    This avoids both the early-prefix bias and any RNG-dependent dataset
    identity while retaining exactly one million complete transitions.
    """

    for name, value in (
        ("source_total_episodes", source_total_episodes),
        ("temporal_deciles", temporal_deciles),
        ("episodes_per_decile", episodes_per_decile),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if source_total_episodes % temporal_deciles:
        raise ValueError("source_total_episodes must divide evenly into temporal blocks")
    episodes_per_block = source_total_episodes // temporal_deciles
    if episodes_per_block % episodes_per_decile:
        raise ValueError("Each temporal block must divide evenly into micro-strata")
    micro_width = episodes_per_block // episodes_per_decile
    if micro_width < 1:
        raise ValueError("episodes_per_decile exceeds the temporal block size")
    offset = 0
    return tuple(
        block * episodes_per_block + sample * micro_width + offset
        for block in range(temporal_deciles)
        for sample in range(episodes_per_decile)
    )


def _episode_index_identity(indices: Sequence[int]) -> str:
    payload = json.dumps(
        [int(index) for index in indices], separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _cartpole_reward(next_observation: np.ndarray, action: np.ndarray) -> np.ndarray:
    if next_observation.shape[-1] != 5 or action.shape[-1] != 1:
        raise ValueError("Cartpole ExORL data must have obs=5 and action=1")
    position = next_observation[..., 0]
    pole_cosine = next_observation[..., 1]
    angular_velocity = next_observation[..., 4]
    control = action[..., 0]
    upright = (pole_cosine + 1.0) / 2.0
    centered = (1.0 + np.exp(np.log(0.1) * (np.abs(position) / 2.0) ** 2)) / 2.0
    small_control = (4.0 + np.maximum(0.0, 1.0 - control**2)) / 5.0
    small_velocity = (
        1.0 + np.exp(np.log(0.1) * (np.abs(angular_velocity) / 5.0) ** 2)
    ) / 2.0
    return (upright * centered * small_control * small_velocity).astype(np.float32)


def _episode_files(root: Path) -> list[_EpisodeSource]:
    """Return only canonical ExORL episode files in episode-index order.

    The public release uses ``episode_000000_1000.npz``.  Older URLB replay
    buffers used ``YYYYMMDDTHHMMSS_index_length.npz``; retaining that naming
    variant also keeps small format fixtures useful.  Arbitrary ``.npz``
    files are deliberately ignored so a converted archive or Koopman stage
    can never be mistaken for one giant source episode.
    """

    episodes: list[_EpisodeSource] = []
    for path in root.rglob("*.npz"):
        release_match = _RELEASE_EPISODE_NAME.fullmatch(path.name)
        legacy_match = _LEGACY_EPISODE_NAME.fullmatch(path.name)
        match = release_match or legacy_match
        if match is None:
            continue
        episodes.append(
            _EpisodeSource(
                path=path,
                index=int(match.group("index")),
                transitions=int(match.group("transitions")),
                release_schema=release_match is not None,
            )
        )
    episodes.sort(key=lambda episode: (episode.index, str(episode.path)))
    indices = [episode.index for episode in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError("ExORL source contains duplicate episode indices")
    return episodes


def _load_official_episode(
    source: _EpisodeSource, *, task: str = "cartpole_swingup"
) -> dict[str, np.ndarray]:
    path = source.path
    with np.load(path, allow_pickle=False) as archive:
        missing = {"observation", "action", "reward", "discount"} - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing ExORL fields {sorted(missing)}")
        episode = {
            key: np.asarray(archive[key])
            for key in ("observation", "action", "reward", "discount")
        }
        if "physics" in archive.files:
            episode["physics"] = np.asarray(archive["physics"])
    length = episode["observation"].shape[0]
    if length != source.transitions + 1:
        raise ValueError(
            f"{path} filename declares {source.transitions} transitions but "
            f"contains {length - 1}"
        )
    spec = get_task_spec(task)
    if episode["action"].shape != (length, spec.action_dim):
        raise ValueError(f"{path} has an invalid action array")
    if episode["observation"].shape != (length, spec.obs_dim):
        raise ValueError(f"{path} has an invalid observation array")
    if episode["discount"].shape not in {(length,), (length, 1)}:
        raise ValueError(f"{path} has an invalid discount array")
    if episode["reward"].shape not in {(length,), (length, 1)}:
        raise ValueError(f"{path} has an invalid reward array")
    if not all(np.isfinite(value).all() for value in episode.values()):
        raise FloatingPointError(f"{path} contains NaN or Inf")

    if source.release_schema:
        # Fail closed on the schema actually distributed by ExORL.  Index zero
        # is produced by ExtendedTimeStepWrapper.reset and is never a real
        # transition.
        if source.transitions != 1000 or length != 1001:
            raise ValueError(f"{path} is not a 1000-step ExORL release episode")
        if (
            episode["observation"].dtype != np.float32
            or episode["action"].dtype != np.float32
            or episode["reward"].dtype != np.float32
            or episode["discount"].dtype != np.float32
        ):
            raise ValueError(f"{path} has non-canonical ExORL dtypes")
        physics = episode.get("physics")
        if task == "cartpole_swingup" and (
            physics is None or physics.shape != (length, 4) or physics.dtype != np.float64
        ):
            raise ValueError(f"{path} is missing canonical float64 physics state")
        if (
            not np.array_equal(
                episode["action"][0], np.zeros(spec.action_dim, np.float32)
            )
            or float(np.asarray(episode["reward"][0]).item()) != 0.0
            or float(np.asarray(episode["discount"][0]).item()) != 1.0
        ):
            raise ValueError(f"{path} has an invalid dummy reset record")
        if task == "cartpole_swingup":
            physics_observation = np.column_stack(
                (
                    physics[:, 0],
                    np.cos(physics[:, 1]),
                    np.sin(physics[:, 1]),
                    physics[:, 2],
                    physics[:, 3],
                )
            ).astype(np.float32)
            np.testing.assert_allclose(
                episode["observation"],
                physics_observation,
                rtol=0.0,
                atol=5e-6,
                err_msg=f"Observation/physics parity failed for {path}",
            )
    # Index zero is a dummy reset record.  ExORL's own replay buffer samples
    # (obs[i-1], action[i], reward[i], discount[i], obs[i]) for i >= 1.
    return episode


def _mc_returns(
    reward: np.ndarray, environment_discount: np.ndarray, gamma: float
) -> np.ndarray:
    result = np.empty_like(reward, dtype=np.float32)
    running = 0.0
    for index in range(reward.shape[0] - 1, -1, -1):
        # Recursion always stops at the recorded episode boundary.  The DMC
        # timeout discount is nevertheless retained separately as one for
        # Bellman bootstrapping during learning.
        if index + 1 == reward.shape[0]:
            running = float(reward[index])
        else:
            running = float(reward[index]) + gamma * float(
                environment_discount[index]
            ) * running
        result[index] = running
    return result


def convert_exorl(
    source_dir: Path,
    output_path: Path,
    *,
    max_transitions: int = 1_000_000,
    gamma: float = 0.99,
    selected_episode_indices: Sequence[int] | None = None,
    selection_metadata: dict[str, Any] | None = None,
    source_archive: Path | None = None,
    task: str = "cartpole_swingup",
    reward_source: str = "oracle",
    allow_unselected_episode_files: bool = False,
) -> dict[str, Any]:
    """Convert official ExORL episodes to one strict transition archive."""

    source_dir = source_dir.resolve()
    spec = get_task_spec(task)
    if task not in {"cartpole_swingup", "walker_run"}:
        raise ValueError(f"Unsupported ExORL O2O conversion task: {task}")
    if reward_source not in {"oracle", "recorded", "zero"}:
        raise ValueError("reward_source must be oracle, recorded, or zero")
    output_path = output_path.resolve()
    if max_transitions < 1:
        raise ValueError("max_transitions must be positive")
    if not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError("gamma must lie in (0, 1]")
    sources = _episode_files(source_dir)
    if not sources:
        raise FileNotFoundError(f"No ExORL episode files under {source_dir}")

    selected_indices: tuple[int, ...] | None = None
    if selected_episode_indices is not None:
        selected_indices = tuple(int(index) for index in selected_episode_indices)
        if (
            not selected_indices
            or any(index < 0 for index in selected_indices)
            or tuple(sorted(set(selected_indices))) != selected_indices
        ):
            raise ValueError(
                "selected_episode_indices must be non-empty, unique, non-negative, "
                "and strictly increasing"
            )
        available_by_index = {source.index: source for source in sources}
        missing_indices = [
            index for index in selected_indices if index not in available_by_index
        ]
        unexpected_indices = sorted(set(available_by_index) - set(selected_indices))
        if missing_indices or (unexpected_indices and not allow_unselected_episode_files):
            raise ValueError(
                "ExORL source files differ from the selected episode identity: "
                f"missing={missing_indices[:10]}, unexpected={unexpected_indices[:10]}"
            )
        sources = [available_by_index[index] for index in selected_indices]

    release_sources = [source for source in sources if source.release_schema]
    if release_sources and len(release_sources) != len(sources):
        raise ValueError("Cannot mix release and legacy ExORL episode schemas")
    if release_sources and selected_indices is None:
        expected_indices = list(range(len(release_sources)))
        actual_indices = [source.index for source in release_sources]
        if actual_indices != expected_indices:
            raise ValueError(
                "ExORL release episodes must be the contiguous prefix starting at zero"
            )

    parts: dict[str, list[np.ndarray]] = {
        key: [] for key in CANONICAL_DATASET_KEYS
    }
    source_hash = hashlib.sha256()
    total = 0
    episode_count = 0
    reward_max_abs_error = 0.0
    for source in sources:
        if total >= max_transitions:
            break
        path = source.path
        episode = _load_official_episode(source, task=task)
        available = episode["observation"].shape[0] - 1
        remaining = max_transitions - total
        if available > remaining:
            raise ValueError(
                "max_transitions would cut through an ExORL episode; "
                "only complete episodes are permitted"
            )
        take = available
        observation = episode["observation"][:take].astype(np.float32, copy=False)
        next_observation = episode["observation"][1 : take + 1].astype(
            np.float32, copy=False
        )
        action = episode["action"][1 : take + 1].astype(np.float32, copy=False)
        discount = np.asarray(episode["discount"][1 : take + 1]).reshape(-1).astype(
            np.float32, copy=False
        )
        recorded_reward = np.asarray(episode["reward"][1 : take + 1]).reshape(-1)
        oracle_reward = None
        if reward_source == "oracle":
            oracle_reward = (
                _cartpole_reward(next_observation, action)
                if task == "cartpole_swingup"
                else walker_run_exact_reward_numpy(next_observation, action)
            )
        reward = {
            "oracle": oracle_reward,
            "recorded": recorded_reward.astype(np.float32),
            "zero": np.zeros(take, dtype=np.float32),
        }[reward_source]
        episode_reward_error = 0.0
        if oracle_reward is not None:
            episode_reward_error = float(
                np.max(
                    np.abs(
                        recorded_reward.astype(np.float64)
                        - oracle_reward.astype(np.float64)
                    )
                )
            )
        reward_max_abs_error = max(reward_max_abs_error, episode_reward_error)
        if task == "cartpole_swingup" and reward_source == "oracle":
            np.testing.assert_allclose(
                recorded_reward,
                reward,
                rtol=0.0,
                atol=2e-7,
                err_msg=f"Official reward parity failed for {path}",
            )
        mc_return = _mc_returns(reward, discount, gamma)
        parts["observation"].append(observation)
        parts["action"].append(action)
        parts["reward"].append(reward)
        parts["discount"].append(discount)
        parts["next_observation"].append(next_observation)
        parts["episode_id"].append(
            np.full(take, episode_count, dtype=np.int64)
        )
        parts["episode_step"].append(np.arange(take, dtype=np.int32))
        terminated = discount == 0.0
        truncated = np.zeros(take, dtype=np.bool_)
        truncated[-1] = not bool(terminated[-1])
        parts["terminated"].append(terminated.astype(np.bool_, copy=False))
        parts["truncated"].append(truncated)
        parts["mc_return"].append(mc_return)
        identity = {
            "index": source.index,
            "name": path.name,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
            "transitions": take,
        }
        source_hash.update(
            (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
        total += take
        episode_count += 1

    if total != max_transitions:
        raise ValueError(
            f"Requested {max_transitions} transitions but source provided only {total} "
            "complete transitions"
        )

    arrays = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    validate_dataset_arrays(
        arrays,
        gamma_for_mc_return=gamma,
        expected_episode_count=episode_count,
        task=task,
        reward_source=reward_source,
    )
    metadata = {
        "kind": DATASET_KIND if task == "cartpole_swingup" else WALKER_DATASET_KIND,
        "task": task,
        "source": f"ExORL {task} exploratory dataset",
        "observation_dim": spec.obs_dim,
        "action_dim": spec.action_dim,
        "source_directory": str(source_dir),
        "source_episode_identity_sha256": source_hash.hexdigest(),
        "source_schema": (
            "exorl_public_release_episode_v1"
            if release_sources
            else "exorl_legacy_replay_episode_v1"
        ),
        "transitions": total,
        "episodes": episode_count,
        "transitions_per_episode": total // episode_count,
        "source_episode_index_first": sources[0].index,
        "source_episode_index_last": sources[episode_count - 1].index,
        "dummy_records_skipped": episode_count,
        "gamma_for_mc_return": gamma,
        "alignment": "obs[i-1],action[i],official_reward(obs[i],action[i]),discount[i],obs[i]",
        "reward": (
            "dm_control_cartpole_swingup_dense_observation_oracle_v1"
            if task == "cartpole_swingup"
            else "dm_control_walker_run_exact_observation_oracle_v1"
        ),
        "reward_oracle": (
            "dm_control_cartpole_swingup_dense_observation_oracle_v1"
            if task == "cartpole_swingup"
            else "dm_control_walker_run_exact_observation_oracle_v1"
        ),
        "recorded_reward_max_abs_error": reward_max_abs_error,
        "recorded_reward_parity_atol": 2e-7,
        "recorded_reward_is_training_target": task == "cartpole_swingup",
        "reward_source": reward_source,
        "timeout_bootstrap": "preserve_environment_discount",
        "boundary_semantics": (
            "terminated iff environment discount is zero; otherwise the final "
            "transition of each complete 1000-step episode is truncated"
        ),
        "environment_discount_values": sorted(
            float(value) for value in np.unique(arrays["discount"])
        ),
    }
    if selected_indices is not None:
        metadata.update(
            source_episode_indices=list(selected_indices),
            source_episode_indices_sha256=_episode_index_identity(selected_indices),
            source_episode_index_first=selected_indices[0],
            source_episode_index_last=selected_indices[-1],
            selection=selection_metadata,
        )
    if source_archive is not None:
        resolved_archive = source_archive.resolve()
        if not resolved_archive.is_file():
            raise FileNotFoundError(f"Source archive does not exist: {resolved_archive}")
        metadata.update(
            source_archive=str(resolved_archive),
            source_archive_sha256=_sha256_file(resolved_archive),
            source_archive_size_bytes=resolved_archive.stat().st_size,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    metadata["output_path"] = str(output_path)
    metadata["output_sha256"] = _sha256_file(output_path)
    return metadata


def convert_exorl_cartpole(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Backward-compatible Cartpole conversion entry point."""

    kwargs.setdefault("task", "cartpole_swingup")
    return convert_exorl(*args, **kwargs)


def validate_dataset_arrays(
    arrays: dict[str, np.ndarray],
    *,
    gamma_for_mc_return: float | None = None,
    expected_episode_count: int | None = None,
    task: str = "cartpole_swingup",
    reward_source: str = "oracle",
) -> None:
    missing = set(CANONICAL_DATASET_KEYS) - set(arrays)
    if missing:
        raise ValueError(f"Dataset is missing arrays {sorted(missing)}")
    count = arrays["reward"].shape[0]
    spec = get_task_spec(task)
    expected_shapes = {
        "observation": (count, spec.obs_dim),
        "action": (count, spec.action_dim),
        "reward": (count,),
        "discount": (count,),
        "next_observation": (count, spec.obs_dim),
        "episode_id": (count,),
        "episode_step": (count,),
        "terminated": (count,),
        "truncated": (count,),
        "mc_return": (count,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key} has shape {arrays[key].shape}, expected {shape}")
        if not np.isfinite(arrays[key]).all():
            raise FloatingPointError(f"{key} contains NaN or Inf")
    if np.any(np.abs(arrays["action"]) > 1.00001):
        raise ValueError("Dataset actions exceed [-1, 1]")
    if np.any((arrays["discount"] < 0) | (arrays["discount"] > 1)):
        raise ValueError("Dataset discounts must lie in [0, 1]")
    if not np.issubdtype(arrays["episode_id"].dtype, np.integer):
        raise ValueError("episode_id must have an integer dtype")
    if not np.issubdtype(arrays["episode_step"].dtype, np.integer):
        raise ValueError("episode_step must have an integer dtype")
    for key in BOUNDARY_KEYS:
        if arrays[key].dtype != np.bool_:
            raise ValueError(f"{key} must have boolean dtype")
    if np.any(arrays["terminated"] & arrays["truncated"]):
        raise ValueError("A transition cannot be both terminated and truncated")
    if reward_source == "oracle":
        recomputed = (
            _cartpole_reward(arrays["next_observation"], arrays["action"])
            if task == "cartpole_swingup"
            else walker_run_exact_reward_numpy(
                arrays["next_observation"], arrays["action"]
            )
        )
        np.testing.assert_allclose(
            arrays["reward"], recomputed, rtol=0.0, atol=2e-7
        )
    elif reward_source not in {"recorded", "zero"}:
        raise ValueError("Unsupported dataset reward_source")
    episode_id = arrays["episode_id"]
    episode_step = arrays["episode_step"]
    if count and (
        episode_id[0] != 0
        or episode_step[0] != 0
        or np.any(np.diff(episode_id) < 0)
    ):
        raise ValueError("Episode IDs/steps are not in canonical order")
    same = episode_id[1:] == episode_id[:-1]
    if np.any(episode_step[1:][same] != episode_step[:-1][same] + 1):
        raise ValueError("Episode steps are not contiguous")
    if np.any(episode_step[1:][~same] != 0):
        raise ValueError("New episodes must begin at step zero")
    unique_episode_id = np.unique(episode_id)
    if not np.array_equal(
        unique_episode_id, np.arange(unique_episode_id.size, dtype=episode_id.dtype)
    ):
        raise ValueError("Episode IDs must be contiguous from zero")
    if expected_episode_count is not None and unique_episode_id.size != int(
        expected_episode_count
    ):
        raise ValueError("Episode count disagrees with the conversion contract")
    # Physical continuity and MC returns are episode-local.  In particular,
    # neither target is allowed to cross a DMC timeout boundary where the
    # environment discount remains one.
    starts = np.flatnonzero(np.r_[True, episode_id[1:] != episode_id[:-1]])
    boundary = np.r_[starts, count]
    for left, right in zip(boundary[:-1], boundary[1:], strict=True):
        length = right - left
        if not np.array_equal(
            episode_step[left:right], np.arange(length, dtype=episode_step.dtype)
        ):
            raise ValueError("Episode steps are not a zero-based contiguous sequence")
        if length > 1 and not np.array_equal(
            arrays["next_observation"][left : right - 1],
            arrays["observation"][left + 1 : right],
        ):
            raise ValueError("Observation transitions are discontinuous within an episode")
        if np.any(arrays["terminated"][left : right - 1]) or np.any(
            arrays["truncated"][left : right - 1]
        ):
            raise ValueError("Episode boundary flags may only appear on the final step")
        if not bool(
            arrays["terminated"][right - 1] or arrays["truncated"][right - 1]
        ):
            raise ValueError("Every complete episode must end in a boundary flag")
        if bool(arrays["terminated"][right - 1]) != bool(
            arrays["discount"][right - 1] == 0.0
        ):
            raise ValueError("Terminal flags disagree with environment discount")
        if gamma_for_mc_return is not None:
            expected_return = _mc_returns(
                arrays["reward"][left:right],
                arrays["discount"][left:right],
                gamma_for_mc_return,
            )
            np.testing.assert_allclose(
                arrays["mc_return"][left:right],
                expected_return,
                rtol=2e-6,
                atol=2e-5,
                err_msg="Stored MC returns disagree with episode-local recursion",
            )


@dataclass
class OfflineDataset:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: Path) -> "OfflineDataset":
        path = path.resolve()
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise ValueError("Dataset is missing metadata_json")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {key: np.asarray(archive[key]) for key in DATASET_KEYS}
            if all(key in archive.files for key in BOUNDARY_KEYS):
                arrays.update(
                    {key: np.asarray(archive[key]) for key in BOUNDARY_KEYS}
                )
            elif any(key in archive.files for key in BOUNDARY_KEYS):
                raise ValueError("Dataset contains only one episode boundary array")
            else:
                # Compatibility for the completed Cartpole artifact created
                # before the canonical terminated/truncated fields were added.
                episode_id = arrays["episode_id"]
                count = episode_id.shape[0]
                episode_end = np.r_[episode_id[1:] != episode_id[:-1], True]
                terminated = arrays["discount"] == 0.0
                if np.any(terminated & ~episode_end):
                    raise ValueError("Legacy dataset terminates inside an episode")
                arrays["terminated"] = terminated.astype(np.bool_, copy=False)
                arrays["truncated"] = (
                    episode_end & ~terminated
                ).astype(np.bool_, copy=False)
        task = str(metadata.get("task", "cartpole_swingup"))
        expected_kind = {
            "cartpole_swingup": DATASET_KIND,
            "walker_run": WALKER_DATASET_KIND,
            "hopper_stand": TDMPC2_DATASET_KIND,
            "hopper_hop": TDMPC2_DATASET_KIND,
        }.get(task)
        if expected_kind is None:
            raise ValueError(f"Unsupported offline dataset task {task!r}")
        if metadata.get("kind") != expected_kind:
            raise ValueError("Unsupported offline dataset kind")
        gamma = float(metadata.get("gamma_for_mc_return", float("nan")))
        if not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("Dataset has an invalid MC-return discount")
        episodes = int(metadata.get("episodes", -1))
        if episodes < 1:
            raise ValueError("Dataset has an invalid episode count")
        validate_dataset_arrays(
            arrays,
            gamma_for_mc_return=gamma,
            expected_episode_count=episodes,
            task=task,
            reward_source=str(metadata.get("reward_source", "oracle")),
        )
        if int(metadata.get("transitions", -1)) != arrays["reward"].shape[0]:
            raise ValueError("Dataset transition count disagrees with metadata")
        return cls(arrays, metadata, path, _sha256_file(path))

    def __len__(self) -> int:
        return int(self.arrays["reward"].shape[0])

    def sample(self, size: int, generator: np.random.Generator) -> dict[str, np.ndarray]:
        if size < 1:
            raise ValueError("sample size must be positive")
        index = generator.integers(0, len(self), size=size)
        return {key: value[index] for key, value in self.arrays.items()}


class OnlineReplay:
    """Fixed-capacity NumPy ring buffer with checkpointable RNG-independent state."""

    def __init__(self, capacity: int, obs_dim: int = 5, action_dim: int = 1) -> None:
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.arrays = {
            # Zero initialization prevents unwritten process memory from being
            # serialized when a checkpoint is saved before the replay fills.
            "observation": np.zeros((capacity, obs_dim), dtype=np.float32),
            "action": np.zeros((capacity, action_dim), dtype=np.float32),
            "reward": np.zeros(capacity, dtype=np.float32),
            "discount": np.zeros(capacity, dtype=np.float32),
            "next_observation": np.zeros((capacity, obs_dim), dtype=np.float32),
            "mc_return": np.zeros(capacity, dtype=np.float32),
            "offline_mask": np.zeros(capacity, dtype=np.float32),
        }
        self.size = 0
        self.cursor = 0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        discount: float,
        next_observation: np.ndarray,
        *,
        mc_return: float = 0.0,
    ) -> None:
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        expected_observation_shape = self.arrays["observation"].shape[1:]
        expected_action_shape = self.arrays["action"].shape[1:]
        if observation.shape != expected_observation_shape:
            raise ValueError("Replay observation has the wrong shape")
        if action.shape != expected_action_shape:
            raise ValueError("Replay action has the wrong shape")
        if next_observation.shape != expected_observation_shape:
            raise ValueError("Replay next_observation has the wrong shape")
        scalars = np.asarray((reward, discount, mc_return), dtype=np.float32)
        if not all(np.isfinite(value).all() for value in (
            observation, action, next_observation, scalars
        )):
            raise FloatingPointError("Replay transition contains NaN or Inf")
        if not 0.0 <= float(discount) <= 1.0:
            raise ValueError("Replay discount must lie in [0, 1]")
        index = self.cursor
        self.arrays["observation"][index] = observation
        self.arrays["action"][index] = action
        self.arrays["reward"][index] = reward
        self.arrays["discount"][index] = discount
        self.arrays["next_observation"][index] = next_observation
        self.arrays["mc_return"][index] = mc_return
        self.arrays["offline_mask"][index] = 0.0
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_episode(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        discount: np.ndarray,
        next_observation: np.ndarray,
        *,
        gamma: float,
    ) -> None:
        """Atomically validate an episode, compute RTG, then append in order.

        Validation and return computation finish before replay is mutated.  If
        an episode exceeds replay capacity, ordinary ring-buffer semantics are
        retained: after insertion replay contains the final ``capacity``
        transitions in chronological order modulo ``cursor``.
        """

        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32)
        discount = np.asarray(discount, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if isinstance(gamma, bool) or not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        length = reward.shape[0] if reward.ndim == 1 else -1
        expected_shapes = {
            "observation": (length, self.arrays["observation"].shape[1]),
            "action": (length, self.arrays["action"].shape[1]),
            "reward": (length,),
            "discount": (length,),
            "next_observation": (length, self.arrays["next_observation"].shape[1]),
        }
        values = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "discount": discount,
            "next_observation": next_observation,
        }
        if length < 1:
            raise ValueError("Replay episode must contain at least one transition")
        for name, value in values.items():
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"Replay episode {name} has shape {value.shape}, "
                    f"expected {expected_shapes[name]}"
                )
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Replay episode {name} contains NaN or Inf")
        if np.any((discount < 0.0) | (discount > 1.0)):
            raise ValueError("Replay episode discounts must lie in [0, 1]")
        if length > 1 and not np.array_equal(
            observation[1:], next_observation[:-1]
        ):
            raise ValueError("Replay episode transitions are not contiguous")
        returns = _mc_returns(reward, discount, float(gamma))
        if not np.isfinite(returns).all():
            raise FloatingPointError("Replay episode produced non-finite MC returns")

        for index in range(length):
            self.add(
                observation[index],
                action[index],
                float(reward[index]),
                float(discount[index]),
                next_observation[index],
                mc_return=float(returns[index]),
            )

    def sample(self, size: int, generator: np.random.Generator) -> dict[str, np.ndarray]:
        if self.size < 1:
            raise RuntimeError("Cannot sample an empty online replay")
        index = generator.integers(0, self.size, size=size)
        return {key: value[index] for key, value in self.arrays.items()}

    def state_dict(self) -> dict[str, Any]:
        arrays = {key: value.copy() for key, value in self.arrays.items()}
        if self.size < self.capacity:
            # The unwritten suffix is not part of replay state.  Canonicalize
            # it even when loading an older checkpoint that may have stored
            # arbitrary bytes there.
            for value in arrays.values():
                value[self.size :] = 0
        return {
            "capacity": self.capacity,
            "size": self.size,
            "cursor": self.cursor,
            "arrays": arrays,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Online replay capacity changed across resume")
        size = int(state["size"])
        cursor = int(state["cursor"])
        if not 0 <= size <= self.capacity or not 0 <= cursor < self.capacity:
            raise ValueError("Invalid online replay checkpoint indices")
        if size < self.capacity and cursor != size:
            raise ValueError("Partially filled replay cursor must equal its size")
        if not isinstance(state.get("arrays"), dict):
            raise ValueError("Online replay checkpoint arrays are missing")
        if set(state["arrays"]) != set(self.arrays):
            raise ValueError("Online replay checkpoint array keys differ")
        for key in self.arrays:
            restored = np.asarray(state["arrays"][key])
            if restored.shape != self.arrays[key].shape:
                raise ValueError(f"Online replay array {key!r} has the wrong shape")
            if restored.dtype != self.arrays[key].dtype:
                raise ValueError(f"Online replay array {key!r} has the wrong dtype")
            if not np.isfinite(restored).all():
                raise FloatingPointError(f"Online replay array {key!r} is non-finite")
            np.copyto(self.arrays[key], restored)
            if size < self.capacity:
                self.arrays[key][size:] = 0
        self.size = size
        self.cursor = cursor


def mark_offline(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {
        key: value
        for key, value in batch.items()
        if key in {"observation", "action", "reward", "discount", "next_observation", "mc_return"}
    }
    result["offline_mask"] = np.ones(result["reward"].shape[0], dtype=np.float32)
    return result


def mixed_batch(
    offline: OfflineDataset,
    online: OnlineReplay,
    *,
    batch_size: int,
    utd: int,
    offline_ratio: float,
    generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(utd, bool) or not isinstance(utd, int) or utd < 1:
        raise ValueError("utd must be a positive integer")
    if not math.isfinite(offline_ratio) or not 0 <= offline_ratio <= 1:
        raise ValueError("offline_ratio must lie in [0, 1]")
    requested_offline = batch_size * offline_ratio
    offline_per_update = int(round(requested_offline))
    if not math.isclose(
        requested_offline, offline_per_update, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("offline_ratio * batch_size must be an integer")
    online_per_update = batch_size - offline_per_update
    if online_per_update and online.size < 1:
        raise RuntimeError("Online portion requested before replay has data")

    # RLPD's replay ratio is a per-critic-update contract, not merely a ratio
    # over the fused UTD batch.  Keep UTD slices contiguous for
    # ``O2OLearner.update`` and independently shuffle within each slice.
    fused: dict[str, list[np.ndarray]] = {}
    for _update in range(utd):
        pieces = []
        if offline_per_update:
            pieces.append(
                mark_offline(offline.sample(offline_per_update, generator))
            )
        if online_per_update:
            pieces.append(online.sample(online_per_update, generator))
        keys = tuple(pieces[0])
        if any(tuple(piece) != keys for piece in pieces[1:]):
            raise AssertionError("Offline and online replay schemas disagree")
        update_batch = {
            key: np.concatenate([piece[key] for piece in pieces], axis=0)
            for key in keys
        }
        permutation = generator.permutation(batch_size)
        for key, value in update_batch.items():
            fused.setdefault(key, []).append(value[permutation])
    return {key: np.concatenate(values, axis=0) for key, values in fused.items()}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task", choices=("cartpole_swingup", "walker_run", "hopper_stand", "hopper_hop"), default="cartpole_swingup"
    )
    parser.add_argument(
        "--reward-source", choices=("oracle", "recorded", "zero"), default="oracle",
        help="Reward target for the archive; Koopman training ignores this field.",
    )
    parser.add_argument("--max-transitions", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--temporal-stratified",
        action="store_true",
        help=(
            "Select deterministic micro-stratum starts from every temporal block; "
            "the source directory must contain exactly those episode files"
        ),
    )
    parser.add_argument("--source-total-episodes", type=int, default=10_000)
    parser.add_argument("--temporal-deciles", type=int, default=10)
    parser.add_argument("--episodes-per-decile", type=int, default=100)
    parser.add_argument(
        "--source-archive",
        type=Path,
        help="Optional immutable release archive whose SHA256 is bound into metadata",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Conversion manifest (default: <output stem>.manifest.json)",
    )
    args = parser.parse_args()
    selected_indices = None
    selection_metadata = None
    if args.temporal_stratified:
        selected_indices = temporal_stratified_episode_indices(
            source_total_episodes=args.source_total_episodes,
            temporal_deciles=args.temporal_deciles,
            episodes_per_decile=args.episodes_per_decile,
        )
        episodes_per_block = args.source_total_episodes // args.temporal_deciles
        micro_width = episodes_per_block // args.episodes_per_decile
        selection_metadata = {
            "kind": "temporal_block_microstratum_start_v1",
            "source_total_episodes": args.source_total_episodes,
            "temporal_blocks": args.temporal_deciles,
            "episodes_per_block": episodes_per_block,
            "selected_episodes_per_block": args.episodes_per_decile,
            "microstratum_width_episodes": micro_width,
            "microstratum_offset": 0,
        }
    metadata = convert_exorl(
        args.source_dir,
        args.output,
        max_transitions=args.max_transitions,
        gamma=args.gamma,
        selected_episode_indices=selected_indices,
        selection_metadata=selection_metadata,
        source_archive=args.source_archive,
        task=args.task,
        reward_source=args.reward_source,
    )
    manifest_path = (
        args.manifest.resolve()
        if args.manifest is not None
        else args.output.resolve().with_suffix(".manifest.json")
    )
    metadata["converter_cli"] = "python -m experiments.dmc.o2o.dataset"
    metadata["conversion_arguments"] = {
        "source_dir": str(args.source_dir.resolve()),
        "output": str(args.output.resolve()),
        "max_transitions": args.max_transitions,
        "gamma": args.gamma,
        "temporal_stratified": args.temporal_stratified,
        "source_total_episodes": args.source_total_episodes,
        "temporal_deciles": args.temporal_deciles,
        "episodes_per_decile": args.episodes_per_decile,
        "source_archive": (
            str(args.source_archive.resolve()) if args.source_archive is not None else None
        ),
        "task": args.task,
        "reward_source": args.reward_source,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
    metadata["conversion_manifest"] = str(manifest_path)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
