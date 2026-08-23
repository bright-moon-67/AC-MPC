"""Deterministic reset panel used for comparable periodic policy evaluation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


class FixedSeedPanelWrapper(gym.Wrapper):
    """Cycle through the same consecutive reset seeds after every rewind."""

    def __init__(self, env: gym.Env, seed_start: int) -> None:
        super().__init__(env)
        self.seed_start = int(seed_start)
        self.panel_index = 0

    def rewind_evaluation_panel(self) -> None:
        self.panel_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        del seed
        panel_seed = self.seed_start + self.panel_index
        self.panel_index += 1
        return self.env.reset(seed=panel_seed, options=options)
