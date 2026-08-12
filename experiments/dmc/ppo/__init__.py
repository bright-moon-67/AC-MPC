"""PPO training and vector-environment utilities for DMC.

The public names are imported lazily so ``python -m ...train_dmc_ppo`` does
not preload its own module while Python is preparing to execute it.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PPOConfig",
    "ProcessDMCVectorEnv",
    "SyncDMCVectorEnv",
    "VectorStep",
    "compute_gae",
    "ppo_config_from_experiment",
    "train",
]


def __getattr__(name: str) -> Any:
    if name in {
        "PPOConfig",
        "compute_gae",
        "ppo_config_from_experiment",
        "train",
    }:
        from . import train_dmc_ppo

        return getattr(train_dmc_ppo, name)
    if name in {"ProcessDMCVectorEnv", "SyncDMCVectorEnv", "VectorStep"}:
        from . import vector_env

        return getattr(vector_env, name)
    raise AttributeError(name)
