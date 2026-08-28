#!/usr/bin/env python
"""Train the stage-1 SAC policy that pulls the distal arm past the wall."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any

import numpy as np
import torch
import yaml

from antmaze_ac.envs.manisoft_wall_crossing_sac_env import (
    ManiSoftWallCrossingSACEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--snapshot-bank", required=True)
    parser.add_argument(
        "--config", default="configs/manisoft_wall_crossing_sac.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--initialize-from-model",
        default=None,
        help=(
            "Initialize actor/critic weights from an SAC checkpoint while "
            "starting a fresh replay buffer and stage-local timestep count."
        ),
    )
    parser.add_argument(
        "--initialize-from-vecnormalize",
        default=None,
        help="Continue updating observation statistics from this VecNormalize file.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _make_env(
    scenario: str,
    task_config: str,
    snapshot_bank: str,
    env_config: dict[str, Any],
    seed: int,
):
    from stable_baselines3.common.monitor import Monitor

    env = ManiSoftWallCrossingSACEnv(
        scenario,
        task_config_path=task_config,
        snapshot_bank_path=snapshot_bank,
        **env_config,
    )
    return Monitor(
        env,
        info_keywords=(
            "is_success",
            "termination_reason",
            "distal_crossed_fraction",
            "wall_clearance",
            "ground_clearance",
            "target_plane_distance",
            "source_crossed_fraction",
        ),
    )


class CrossingStatusCallback:
    """Factory keeps Stable-Baselines imports out of module import smoke tests."""

    @staticmethod
    def create(output: Path, status_freq: int):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.recent = deque(maxlen=100)
                self.next_report = status_freq

            def _on_step(self) -> bool:
                for info, done in zip(
                    self.locals.get("infos", ()), self.locals.get("dones", ())
                ):
                    if done and "episode" in info:
                        episode = info["episode"]
                        self.recent.append(
                            {
                                "success": float(info.get("is_success", False)),
                                "return": float(episode["r"]),
                                "length": float(episode["l"]),
                                "distal_fraction": float(
                                    info.get("distal_crossed_fraction", np.nan)
                                ),
                                "wall_clearance": float(
                                    info.get("wall_clearance", np.nan)
                                ),
                                "target_plane_distance": float(
                                    info.get("target_plane_distance", np.nan)
                                ),
                            }
                        )
                if self.num_timesteps >= self.next_report:
                    rows = list(self.recent)
                    status = {
                        "timesteps": int(self.num_timesteps),
                        "recent_episode_count": len(rows),
                        "recent_success_rate": (
                            float(np.mean([row["success"] for row in rows]))
                            if rows
                            else None
                        ),
                        "recent_mean_return": (
                            float(np.mean([row["return"] for row in rows]))
                            if rows
                            else None
                        ),
                        "recent_mean_length": (
                            float(np.mean([row["length"] for row in rows]))
                            if rows
                            else None
                        ),
                        "recent_mean_final_distal_fraction": (
                            float(np.nanmean([row["distal_fraction"] for row in rows]))
                            if rows
                            else None
                        ),
                        "recent_min_wall_clearance": (
                            float(np.nanmin([row["wall_clearance"] for row in rows]))
                            if rows
                            else None
                        ),
                        "recent_mean_final_target_plane_distance": (
                            float(
                                np.nanmean(
                                    [row["target_plane_distance"] for row in rows]
                                )
                            )
                            if rows
                            else None
                        ),
                    }
                    temporary = output / "training_status.json.tmp"
                    temporary.write_text(
                        json.dumps(status, indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                    temporary.replace(output / "training_status.json")
                    print(json.dumps(status, sort_keys=True), flush=True)
                    self.next_report += status_freq
                return True

        return _Callback()


def main() -> None:
    args = parse_args()
    scenario = Path(args.scenario).expanduser().resolve()
    task_config = Path(args.task_config).expanduser().resolve()
    snapshot_bank = Path(args.snapshot_bank).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    initialization_model = (
        None
        if args.initialize_from_model is None
        else Path(args.initialize_from_model).expanduser().resolve()
    )
    initialization_vecnormalize = (
        None
        if args.initialize_from_vecnormalize is None
        else Path(args.initialize_from_vecnormalize).expanduser().resolve()
    )
    if (initialization_model is None) != (initialization_vecnormalize is None):
        raise ValueError(
            "initialize-from-model and initialize-from-vecnormalize must be supplied together"
        )
    required_paths = [scenario, task_config, snapshot_bank, config_path]
    if initialization_model is not None:
        required_paths.extend((initialization_model, initialization_vecnormalize))
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"training output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("wall-crossing SAC config schema_version must be 1")
    env_config = dict(payload["environment"])
    training = dict(payload["training"])
    total_timesteps = int(
        training["total_timesteps"]
        if args.total_timesteps is None
        else args.total_timesteps
    )
    num_envs = int(training["num_envs"] if args.num_envs is None else args.num_envs)
    seed = int(training["seed"])
    if total_timesteps < 1 or num_envs < 1:
        raise ValueError("total_timesteps and num_envs must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.utils import get_schedule_fn
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    factories = [
        lambda rank=rank: _make_env(
            str(scenario),
            str(task_config),
            str(snapshot_bank),
            env_config,
            seed + rank,
        )
        for rank in range(num_envs)
    ]
    vector = (
        DummyVecEnv(factories)
        if num_envs == 1
        else SubprocVecEnv(factories, start_method="forkserver")
    )
    vector.seed(seed)
    if initialization_vecnormalize is None:
        vector = VecNormalize(
            vector,
            norm_obs=bool(training.get("norm_obs", True)),
            norm_reward=bool(training.get("norm_reward", True)),
            clip_obs=float(training.get("clip_obs", 10.0)),
        )
    else:
        vector = VecNormalize.load(str(initialization_vecnormalize), vector)
        vector.training = True
        vector.norm_obs = bool(training.get("norm_obs", True))
        vector.norm_reward = bool(training.get("norm_reward", True))
        vector.clip_obs = float(training.get("clip_obs", 10.0))
    run_config = {
        "scenario": str(scenario),
        "task_config": str(task_config),
        "snapshot_bank": str(snapshot_bank),
        "training_config": str(config_path),
        "environment": env_config,
        "training": {**training, "total_timesteps": total_timesteps, "num_envs": num_envs},
        "device": args.device,
        "provenance": {
            "git_head": _git_head(Path(__file__).resolve().parents[1]),
            "scenario_sha256": _sha256(scenario),
            "task_config_sha256": _sha256(task_config),
            "snapshot_bank_sha256": _sha256(snapshot_bank),
            "training_config_sha256": _sha256(config_path),
            "training_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "initialization": (
            None
            if initialization_model is None
            else {
                "model": str(initialization_model),
                "vecnormalize": str(initialization_vecnormalize),
                "replay_buffer": None,
                "reset_num_timesteps": True,
            }
        ),
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    if initialization_model is None:
        model = SAC(
            "MlpPolicy",
            vector,
            learning_rate=float(training["learning_rate"]),
            buffer_size=int(training["buffer_size"]),
            learning_starts=int(training["learning_starts"]),
            batch_size=int(training["batch_size"]),
            tau=float(training["tau"]),
            gamma=float(training["gamma"]),
            train_freq=int(training["train_freq"]),
            gradient_steps=int(training["gradient_steps"]),
            ent_coef=training.get("ent_coef", "auto"),
            target_entropy=training.get("target_entropy", "auto"),
            policy_kwargs={"net_arch": list(training["net_arch"])},
            seed=seed,
            device=args.device,
            verbose=1,
        )
    else:
        continuation_learning_rate = float(training["learning_rate"])
        model = SAC.load(
            str(initialization_model),
            env=vector,
            device=args.device,
            custom_objects={
                "learning_rate": continuation_learning_rate,
                "lr_schedule": get_schedule_fn(continuation_learning_rate),
            },
            print_system_info=False,
        )
        # A new curriculum changes rewards and termination labels.  Keeping the
        # old replay buffer would incorrectly mark 30% states as terminal, so
        # only network/optimizer weights and normalization statistics transfer.
        model.learning_starts = int(training["learning_starts"])
        model.batch_size = int(training["batch_size"])
        model.buffer_size = int(training["buffer_size"])
        model.learning_rate = continuation_learning_rate
        model.lr_schedule = get_schedule_fn(continuation_learning_rate)
        model.gamma = float(training["gamma"])
        model.tau = float(training["tau"])
        model.train_freq = int(training["train_freq"])
        model._convert_train_freq()
        model.gradient_steps = int(training["gradient_steps"])
        model.target_entropy = float(training["target_entropy"])
        requested_ent_coef = training.get("ent_coef", "auto")
        if isinstance(requested_ent_coef, (int, float)):
            model.ent_coef = float(requested_ent_coef)
            model.ent_coef_tensor = torch.tensor(
                float(requested_ent_coef), device=model.device
            )
            model.ent_coef_optimizer = None
        log_std_bias_delta = float(
            training.get("continuation_log_std_bias_delta", 0.0)
        )
        if log_std_bias_delta != 0.0:
            with torch.no_grad():
                model.actor.log_std.bias.add_(log_std_bias_delta)
        optimizers = [model.actor.optimizer, model.critic.optimizer]
        if model.ent_coef_optimizer is not None:
            optimizers.append(model.ent_coef_optimizer)
        for optimizer in optimizers:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = continuation_learning_rate
    checkpoint_freq = max(
        1, int(training["checkpoint_freq"]) // max(num_envs, 1)
    )
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(output / "checkpoints"),
                name_prefix="wall_crossing_sac",
                save_replay_buffer=True,
                save_vecnormalize=True,
            ),
            CrossingStatusCallback.create(
                output, int(training.get("status_freq", 2000))
            ),
        ]
    )
    interrupted = False
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
        )
    except KeyboardInterrupt:
        interrupted = True
        model.save(output / "interrupted_model")
        vector.save(output / "interrupted_vecnormalize.pkl")
        print(
            json.dumps(
                {
                    "interrupted": True,
                    "timesteps": int(model.num_timesteps),
                    "model": str(output / "interrupted_model.zip"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        model.save(output / "final_model")
        vector.save(output / "vecnormalize.pkl")
    finally:
        try:
            vector.close()
        except (BrokenPipeError, EOFError):
            # Ctrl-C may reach a SubprocVecEnv worker before the parent.  The
            # model and normalizer have already been saved above.
            if not interrupted:
                raise


if __name__ == "__main__":
    main()
