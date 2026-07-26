from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["data"]["history"] != 1:
        raise ValueError("This implementation defines history=1 as (x_t, delta_u_t) -> x_{t+1}")
    if "lqr_horizon" in str(config):
        raise ValueError("lqr_horizon is intentionally unsupported; use steady-state DARE")
    koopman = config["koopman"]
    prompt_fixed = {
        "lift_dim": 32,
        "K_step": 20,
        "encoder_hidden_dims": [256, 256],
        "encoder_activation": "silu",
        "max_epochs": 1000,
        "max_wall_time_hours": 5,
        "stop_condition": "first_reached",
    }
    for key, expected in prompt_fixed.items():
        if koopman.get(key) != expected:
            raise ValueError(f"Prompt fixes koopman.{key}={expected!r}, got {koopman.get(key)!r}")
    if koopman.get("architecture") != "fullA_history_v2_adapted":
        raise ValueError("Koopman architecture must adapt the referenced fullA_history_v2")
    if koopman.get("batch_size") != 4096 or koopman.get("eval_batch_size") != 4096:
        raise ValueError("Formal Koopman train/eval batch sizes are fixed to 4096")
    reference_loss_weights = {
        "linear": 10.0,
        "rollout": 1.0,
        "stability": 0.01,
        "latent_std": 0.1,
        "identity": 0.0001,
        "controllability_svd": 0.0,
        "augmentation": 0.0,
    }
    for key, expected in reference_loss_weights.items():
        actual = koopman["loss_weights"].get(key)
        if actual != expected:
            raise ValueError(
                f"fullA_history_v2 alignment fixes loss_weights.{key}={expected}, got {actual}"
            )
    if config["control"]["gain_update_interval"] < 1:
        raise ValueError("gain_update_interval must be >= 1")
    td3_bc = config.get("td3_bc")
    if td3_bc is None:
        raise ValueError("Config must define the independent td3_bc section")
    positive_td3_values = (
        "gradient_steps",
        "batch_size",
        "validation_batch_size",
        "actor_learning_rate",
        "critic_learning_rate",
        "tau",
        "policy_frequency",
        "max_delta_action",
        "max_grad_norm",
        "log_interval",
        "validation_interval",
        "checkpoint_interval",
    )
    for key in positive_td3_values:
        if float(td3_bc[key]) <= 0:
            raise ValueError(f"td3_bc.{key} must be positive")
    if int(td3_bc["bc_warmup_steps"]) < 0:
        raise ValueError("td3_bc.bc_warmup_steps must be non-negative")
    max_wall_time_hours = td3_bc.get("max_wall_time_hours")
    if max_wall_time_hours is not None and float(max_wall_time_hours) <= 0:
        raise ValueError("td3_bc.max_wall_time_hours must be null or positive")
    if not 0.0 <= float(td3_bc["discount"]) <= 1.0:
        raise ValueError("td3_bc.discount must be in [0,1]")
    if not 0.0 < float(td3_bc["tau"]) <= 1.0:
        raise ValueError("td3_bc.tau must be in (0,1]")
    return config
