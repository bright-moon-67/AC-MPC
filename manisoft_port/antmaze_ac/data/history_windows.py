from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .build_sequences import Normalizer


class AbsoluteActionHistoryWindowDataset(Dataset):
    """Raw ManiSoft windows for ``z[t+1] = A z[t] + B u[t]``.

    Context at ``t`` is ``[s[t-H+1:t+1], u[t-H:t]]``. The current absolute
    action ``u[t]`` is excluded from that context and supplied only to the
    transition. Before enough history exists, missing states are left-padded
    with the episode's first state and missing actions are padded with zeros.
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        episode_ids: np.ndarray,
        step_indices: np.ndarray,
        normalizer: Normalizer,
        transitions: int,
        history_steps: int,
        starts: np.ndarray | None = None,
    ) -> None:
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self.step_indices = np.asarray(step_indices, dtype=np.int64)
        self.normalizer = normalizer
        self.transitions = int(transitions)
        self.history_steps = int(history_steps)
        if self.history_steps < 1 or self.transitions < 1:
            raise ValueError("history_steps and transitions must be positive")
        if len(self.states) != len(self.actions) or len(self.states) != len(self.episode_ids):
            raise ValueError("states, actions and episode ids must have matching rows")

        if starts is None:
            candidates = np.arange(max(0, len(self.states) - self.transitions), dtype=np.int64)
            stops = candidates + self.transitions
            valid = (
                (self.episode_ids[candidates] == self.episode_ids[stops])
                & (self.step_indices[stops] == self.step_indices[candidates] + self.transitions)
            )
            starts = candidates[valid]
        self.starts = np.asarray(starts, dtype=np.int64)
        if not len(self.starts):
            raise ValueError(
                f"No valid {self.transitions}-transition windows with "
                f"{self.history_steps} history steps"
            )

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.starts[index])
        stop = start + self.transitions
        episode_step = int(self.step_indices[start])
        episode_start = start - episode_step

        state_sequence = self.normalizer.normalize(self.states[start : stop + 1]).astype(np.float32)
        state_padding = max(0, self.history_steps - 1 - episode_step)
        state_history_start = max(episode_start, start - self.history_steps + 1)
        context_states = self.states[state_history_start : stop + 1]
        if state_padding:
            context_states = np.concatenate(
                (
                    np.repeat(self.states[episode_start : episode_start + 1], state_padding, axis=0),
                    context_states,
                ),
                axis=0,
            )
        context_states = self.normalizer.normalize(context_states).astype(np.float32)

        action_padding = max(0, self.history_steps - episode_step)
        action_history_start = max(episode_start, start - self.history_steps)
        context_actions = self.actions[action_history_start:stop]
        if action_padding:
            context_actions = np.concatenate(
                (
                    np.zeros((action_padding, self.actions.shape[1]), dtype=np.float32),
                    context_actions,
                ),
                axis=0,
            )
        context_actions = context_actions.astype(np.float32, copy=False)
        state_windows = np.lib.stride_tricks.sliding_window_view(
            context_states,
            window_shape=self.history_steps,
            axis=0,
        ).swapaxes(-1, -2)
        action_windows = np.lib.stride_tricks.sliding_window_view(
            context_actions,
            window_shape=self.history_steps,
            axis=0,
        ).swapaxes(-1, -2)
        contexts = np.concatenate(
            (
                state_windows.reshape(self.transitions + 1, -1),
                action_windows.reshape(self.transitions + 1, -1),
            ),
            axis=1,
        ).astype(np.float32)
        controls = self.actions[start:stop].astype(np.float32)
        return (
            torch.from_numpy(contexts),
            torch.from_numpy(state_sequence),
            torch.from_numpy(controls),
        )
