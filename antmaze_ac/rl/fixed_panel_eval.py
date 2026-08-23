"""Stable-Baselines evaluation callback with a repeatable episode panel."""

from __future__ import annotations

from stable_baselines3.common.callbacks import EvalCallback


class FixedPanelEvalCallback(EvalCallback):
    """Rewind a fixed-seed reset wrapper immediately before every evaluation."""

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.env_method("rewind_evaluation_panel")
        return super()._on_step()
