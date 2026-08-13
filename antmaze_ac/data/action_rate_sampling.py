"""Action-rate diagnostics and stratified sampling for Koopman windows."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


def transition_action_rates(
    actions: np.ndarray,
    episode_ids: np.ndarray,
) -> np.ndarray:
    """Return max-component ``|u[t]-u[t-1]|``, with zero at boundaries."""

    actions = np.asarray(actions, dtype=np.float32)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    if actions.ndim != 2 or len(actions) != len(episode_ids):
        raise ValueError("actions must be [T,A] and match episode_ids")
    previous = np.zeros_like(actions)
    same_episode = episode_ids[1:] == episode_ids[:-1]
    continuing_rows = np.flatnonzero(same_episode) + 1
    previous[continuing_rows] = actions[continuing_rows - 1]
    return np.max(np.abs(actions - previous), axis=1)


def window_action_rates(
    actions: np.ndarray,
    episode_ids: np.ndarray,
    starts: np.ndarray,
    transitions: int,
    history_steps: int,
) -> np.ndarray:
    """Return peak action rate in each history-plus-rollout window."""

    if transitions < 1 or history_steps < 1:
        raise ValueError("transitions and history_steps must be positive")
    starts = np.asarray(starts, dtype=np.int64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    rates = transition_action_rates(actions, episode_ids)
    peaks = np.zeros(len(starts), dtype=np.float32)
    start_episode_ids = episode_ids[starts]
    for offset in range(-history_steps + 1, transitions):
        indices = starts + offset
        valid = (indices >= 0) & (indices < len(rates))
        valid &= episode_ids[np.clip(indices, 0, len(rates) - 1)] == start_episode_ids
        if np.any(valid):
            peaks[valid] = np.maximum(peaks[valid], rates[indices[valid]])
    return peaks


def rate_bin_indices(rates: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Assign rates to ``len(edges)+1`` bins, including edges on the left."""

    edges_array = np.asarray(edges, dtype=np.float64)
    if np.any(edges_array <= 0) or np.any(np.diff(edges_array) <= 0):
        raise ValueError("rate bin edges must be positive and strictly increasing")
    return np.digitize(np.asarray(rates), edges_array, right=True)


def stratified_sample_indices(
    rates: np.ndarray,
    edges: Sequence[float],
    fractions: Sequence[float],
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample window indices at requested rate proportions.

    Empty bins are removed and their probability is redistributed over bins
    that exist. Sampling within a bin uses replacement only when necessary.
    """

    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    bins = rate_bin_indices(rates, edges)
    fractions_array = np.asarray(fractions, dtype=np.float64)
    expected_bins = len(edges) + 1
    if fractions_array.shape != (expected_bins,):
        raise ValueError(
            f"expected {expected_bins} fractions for {len(edges)} edges, "
            f"got {fractions_array.shape}"
        )
    if np.any(fractions_array < 0) or fractions_array.sum() <= 0:
        raise ValueError("fractions must be non-negative and sum to > 0")
    groups = [np.flatnonzero(bins == index) for index in range(expected_bins)]
    available = np.array([len(group) > 0 for group in groups])
    if not np.any(available):
        raise ValueError("no windows are available for stratified sampling")
    weights = fractions_array * available
    if weights.sum() <= 0:
        weights = available.astype(np.float64)
    weights /= weights.sum()
    exact_counts = weights * num_samples
    counts = np.floor(exact_counts).astype(np.int64)
    remainder_order = np.argsort(-(exact_counts - counts))
    counts[remainder_order[: num_samples - int(counts.sum())]] += 1

    sampled = []
    for group, count in zip(groups, counts):
        if count:
            sampled.append(
                rng.choice(group, size=int(count), replace=count > len(group))
            )
    result = np.concatenate(sampled).astype(np.int64, copy=False)
    rng.shuffle(result)
    return result


class ActionRateStratifiedSampler(Sampler[int]):
    """Deterministically resample rate strata once per training epoch."""

    def __init__(
        self,
        rates: np.ndarray,
        edges: Sequence[float],
        fractions: Sequence[float],
        num_samples: int,
        seed: int,
    ) -> None:
        self.rates = np.asarray(rates, dtype=np.float32)
        self.edges = tuple(float(value) for value in edges)
        self.fractions = tuple(float(value) for value in fractions)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        # Validate immediately instead of failing after training starts.
        stratified_sample_indices(
            self.rates,
            self.edges,
            self.fractions,
            self.num_samples,
            np.random.default_rng(self.seed),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        indices = stratified_sample_indices(
            self.rates,
            self.edges,
            self.fractions,
            self.num_samples,
            np.random.default_rng(self.seed + self.epoch),
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples
