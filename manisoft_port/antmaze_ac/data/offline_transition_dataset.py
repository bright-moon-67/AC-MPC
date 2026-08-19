"""Memory-mapped access to transition-complete offline NPZ datasets."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import shutil
from typing import Any
import zipfile

import numpy as np
import torch

from antmaze_ac.rl.iql import IQLTransitionBatch


IQL_DATASET_KEYS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
    "timeouts",
    "episode_ids",
)
IQL_OPTIONAL_DATASET_KEYS = ("behavior_action_means",)


def _source_signature(path: Path, members: dict[str, zipfile.ZipInfo]) -> dict:
    stat = path.stat()
    return {
        "source": str(path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "members": {
            key: {
                "filename": members[key].filename,
                "file_size": members[key].file_size,
                "crc": members[key].CRC,
            }
            for key in members
        },
    }


def prepare_npz_memmap_cache(
    dataset_path: str | Path,
    cache_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Extract selected NPY members once so large observations can be mmap'd."""

    source = Path(dataset_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else source.parent / f".{source.stem}_iql_memmap"
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    with zipfile.ZipFile(source) as archive:
        entries = {entry.filename: entry for entry in archive.infolist()}
        members: dict[str, zipfile.ZipInfo] = {}
        for key in IQL_DATASET_KEYS:
            filename = f"{key}.npy"
            if filename not in entries:
                raise KeyError(f"Offline dataset is missing {filename}")
            members[key] = entries[filename]
        for key in IQL_OPTIONAL_DATASET_KEYS:
            filename = f"{key}.npy"
            if filename in entries:
                members[key] = entries[filename]
        signature = _source_signature(source, members)
        existing = None
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = None
        complete = existing == signature and all(
            (destination / f"{key}.npy").is_file()
            and (destination / f"{key}.npy").stat().st_size
            == members[key].file_size
            for key in members
        )
        if not complete:
            for key in members:
                output = destination / f"{key}.npy"
                if (
                    output.is_file()
                    and output.stat().st_size == members[key].file_size
                    and existing is not None
                    and existing.get("members", {}).get(key)
                    == signature["members"][key]
                ):
                    continue
                temporary = destination / f".{key}.npy.tmp"
                temporary.unlink(missing_ok=True)
                print(
                    f"Extracting {key}.npy to IQL memmap cache "
                    f"({members[key].file_size / 2**30:.2f} GiB)...",
                    flush=True,
                )
                try:
                    with archive.open(members[key]) as source_stream:
                        with temporary.open("wb") as output_stream:
                            shutil.copyfileobj(
                                source_stream,
                                output_stream,
                                length=16 * 1024 * 1024,
                            )
                            output_stream.flush()
                            os.fsync(output_stream.fileno())
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
            temporary_manifest = destination / ".manifest.json.tmp"
            temporary_manifest.write_text(
                json.dumps(signature, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary_manifest, manifest_path)
    return destination, signature


class OfflineTransitionDataset:
    """Random-batch transition dataset backed by read-only NPY memory maps."""

    def __init__(
        self,
        dataset_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        treat_timeouts_as_terminal: bool = False,
    ) -> None:
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.cache_dir, self.source_signature = prepare_npz_memmap_cache(
            self.dataset_path,
            cache_dir,
        )
        cached_keys = tuple(self.source_signature["members"])
        self.arrays = {
            key: np.load(
                self.cache_dir / f"{key}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for key in cached_keys
        }
        lengths = {key: len(array) for key, array in self.arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Offline transition arrays are misaligned: {lengths}")
        self.transition_count = lengths["observations"]
        observations = self.arrays["observations"]
        actions = self.arrays["actions"]
        if observations.ndim != 2:
            raise ValueError("observations must have shape [N,D]")
        if self.arrays["next_observations"].shape != observations.shape:
            raise ValueError("next_observations must match observations")
        if actions.ndim != 2:
            raise ValueError("actions must have shape [N,A]")
        behavior_means = self.arrays.get("behavior_action_means")
        if behavior_means is not None and behavior_means.shape != actions.shape:
            raise ValueError("behavior_action_means must match actions")
        if self.arrays["rewards"].shape != (self.transition_count,):
            raise ValueError("rewards must have shape [N]")
        for key in ("terminals", "timeouts", "episode_ids"):
            if self.arrays[key].shape != (self.transition_count,):
                raise ValueError(f"{key} must have shape [N]")
        self.observation_dim = int(observations.shape[1])
        self.action_dim = int(actions.shape[1])
        self.treat_timeouts_as_terminal = bool(treat_timeouts_as_terminal)

    def __len__(self) -> int:
        return self.transition_count

    def split_by_episode(
        self,
        validation_fraction: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0,1)")
        episode_ids = np.asarray(self.arrays["episode_ids"])
        episodes = np.unique(episode_ids)
        if len(episodes) < 2:
            raise ValueError("Episode-level split requires at least two episodes")
        rng = np.random.default_rng(seed)
        rng.shuffle(episodes)
        validation_count = max(1, round(len(episodes) * validation_fraction))
        validation_count = min(validation_count, len(episodes) - 1)
        validation_episodes = episodes[:validation_count]
        validation_mask = np.isin(episode_ids, validation_episodes)
        return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)

    def sample_batch(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
        *,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> IQLTransitionBatch:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if indices is None:
            selected = rng.integers(
                0, self.transition_count, size=batch_size
            )
        else:
            population = np.asarray(indices, dtype=np.int64)
            if population.ndim != 1 or len(population) < 1:
                raise ValueError("Sampling indices must be a non-empty vector")
            selected = population[
                rng.integers(0, len(population), size=batch_size)
            ]

        def tensor(key: str) -> torch.Tensor:
            values = np.array(self.arrays[key][selected], copy=True, order="C")
            return torch.as_tensor(values, dtype=torch.float32, device=device)

        terminal = tensor("terminals")
        if self.treat_timeouts_as_terminal:
            terminal = torch.maximum(terminal, tensor("timeouts"))
        return IQLTransitionBatch(
            observation=tensor("observations"),
            action=tensor("actions"),
            reward=tensor("rewards"),
            next_observation=tensor("next_observations"),
            terminal=terminal,
            behavior_action_mean=(
                tensor("behavior_action_means")
                if "behavior_action_means" in self.arrays
                else None
            ),
        )

    def metadata(self) -> dict[str, Any]:
        terminals = np.asarray(self.arrays["terminals"])
        timeouts = np.asarray(self.arrays["timeouts"])
        return {
            "dataset": str(self.dataset_path),
            "cache_dir": str(self.cache_dir),
            "transitions": self.transition_count,
            "episodes": int(len(np.unique(self.arrays["episode_ids"]))),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "has_behavior_action_means": (
                "behavior_action_means" in self.arrays
            ),
            "terminal_transitions": int(terminals.sum()),
            "timeout_transitions": int(timeouts.sum()),
            "treat_timeouts_as_terminal": self.treat_timeouts_as_terminal,
            "source_signature": self.source_signature,
        }
