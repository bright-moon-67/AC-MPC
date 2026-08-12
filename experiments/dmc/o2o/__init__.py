"""Offline-to-online RL experiments for DMC and AC-KMPC.

The package is intentionally independent from the on-policy PPO approval
pipeline.  It has one small, explicit experiment contract: a fixed offline
dataset, a frozen Koopman representation, a learner configuration, and an
exact count of subsequent online environment interactions.
"""

from experiments.dmc.o2o.config import O2OConfig

__all__ = ["O2OConfig"]
