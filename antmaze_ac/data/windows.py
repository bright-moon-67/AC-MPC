from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .build_sequences import AugmentedDataset, Normalizer, valid_window_starts


def load_npz_dataset(path: str | Path) -> AugmentedDataset:
    with np.load(path) as archive:
        dataset = AugmentedDataset.from_mapping(
            {name: archive[name] for name in archive.files}
        )
    dataset.validate()
    return dataset


class KoopmanWindowDataset(Dataset):
    def __init__(
        self,
        dataset: AugmentedDataset,
        transitions: int,
        normalizer: Normalizer,
        starts: np.ndarray | None = None,
    ) -> None:
        self.dataset = dataset
        self.transitions = int(transitions)
        self.normalizer = normalizer
        self.starts = valid_window_starts(dataset, transitions) if starts is None else starts
        if not len(self.starts):
            raise ValueError(f"No valid {transitions}-transition windows")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        stop = start + self.transitions
        initial = self.dataset.x[start : start + 1]
        successors = self.dataset.next_x[start:stop]
        states = np.concatenate((initial, successors), axis=0)
        states = self.normalizer.normalize(states).astype(np.float32)
        actions = self.dataset.delta_action[start:stop].astype(np.float32)
        return torch.from_numpy(states), torch.from_numpy(actions)
