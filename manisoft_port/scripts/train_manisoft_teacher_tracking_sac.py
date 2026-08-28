#!/usr/bin/env python
"""Behavior-clone the smooth wall teacher, then train one unified SAC actor."""

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
from torch.nn import functional as F
import yaml

from antmaze_ac.envs.manisoft_teacher_tracking_sac_env import (
    ManiSoftTeacherTrackingSACEnv,
)
from antmaze_ac.rl.anchored_sac import AnchoredSAC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--teacher-episode", required=True)
    parser.add_argument(
        "--config", default="configs/manisoft_teacher_tracking_sac.yaml"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--bc-epochs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
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
    teacher_episode: str,
    environment: dict[str, Any],
):
    from stable_baselines3.common.monitor import Monitor

    env = ManiSoftTeacherTrackingSACEnv(
        scenario,
        task_config_path=task_config,
        teacher_episode_path=teacher_episode,
        **environment,
    )
    return Monitor(
        env,
        info_keywords=(
            "is_success",
            "termination_reason",
            "start_index",
            "reference_progress",
            "node_tracking_rmse",
            "tip_tracking_error",
            "teacher_action_error",
            "distal_crossed_fraction",
            "wall_clearance",
            "ground_clearance",
            "target_plane_distance",
            "arch_height",
        ),
    )


class TeacherWarmupSACMixin:
    """Use the behavior-cloned actor instead of uniform actions during warm-up."""

    teacher_policy_warmup = True

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.teacher_policy_warmup and self.num_timesteps < learning_starts:
            if self._last_obs is None:
                raise RuntimeError("last observation is unavailable during warm-up")
            unscaled_action, _ = self.predict(self._last_obs, deterministic=True)
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1.0, 1.0)
            return self.policy.unscale_action(scaled_action), scaled_action
        return super()._sample_action(learning_starts, action_noise, n_envs)


def _behavior_clone_actor(
    model,
    observations: np.ndarray,
    target_actions: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    output: Path,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    device = model.device
    observations_tensor = torch.as_tensor(
        observations, dtype=torch.float32, device=device
    )
    targets_tensor = torch.as_tensor(
        target_actions, dtype=torch.float32, device=device
    )
    losses: list[float] = []
    for epoch in range(epochs):
        order = rng.permutation(len(observations))
        epoch_losses = []
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(
                order[start : start + batch_size], dtype=torch.long, device=device
            )
            predicted = model.actor(
                observations_tensor.index_select(0, indices), deterministic=True
            )
            loss = F.mse_loss(predicted, targets_tensor.index_select(0, indices))
            model.actor.optimizer.zero_grad()
            loss.backward()
            model.actor.optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                json.dumps(
                    {
                        "phase": "behavior_cloning",
                        "epoch": epoch + 1,
                        "epochs": epochs,
                        "loss": losses[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    with torch.no_grad():
        predicted = model.actor(observations_tensor, deterministic=True)
        error = predicted - targets_tensor
        final_mse = float(torch.mean(error**2).item())
        final_max_abs = float(torch.max(torch.abs(error)).item())
    result = {
        "epochs": int(epochs),
        "sample_count": int(len(observations)),
        "final_mse": final_mse,
        "final_rmse": float(np.sqrt(final_mse)),
        "final_maximum_absolute_error": final_max_abs,
        "first_epoch_loss": losses[0],
        "last_epoch_loss": losses[-1],
    }
    (output / "behavior_cloning.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _evaluate_upright(
    model,
    normalizer,
    *,
    scenario: Path,
    task_config: Path,
    teacher_episode: Path,
    environment: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    eval_environment = dict(environment)
    with np.load(teacher_episode, allow_pickle=False) as archive:
        transition_count = int(np.asarray(archive["actions"]).shape[0])
    eval_environment.update(
        {
            "reset_start_mode": "upright",
            "upright_start_probability": 1.0,
            "episode_steps": transition_count + 50,
        }
    )
    env = ManiSoftTeacherTrackingSACEnv(
        scenario,
        task_config_path=task_config,
        teacher_episode_path=teacher_episode,
        **eval_environment,
    )
    observation, info = env.reset(seed=seed, options={"start_index": 0})
    total_reward = 0.0
    minimum_wall = float(info["wall_clearance"])
    minimum_ground = float(info["ground_clearance"])
    maximum_node_error = 0.0
    maximum_tip_error = 0.0
    enforced_arch_heights: list[float] = []
    terminated = truncated = False
    final_info = info
    while not (terminated or truncated):
        value = observation[None].astype(np.float32)
        if bool(normalizer.norm_obs):
            value = normalizer.normalize_obs(value)
        action, _ = model.predict(value, deterministic=True)
        observation, reward, terminated, truncated, final_info = env.step(action[0])
        total_reward += reward
        minimum_wall = min(minimum_wall, float(final_info.get("wall_clearance", np.inf)))
        minimum_ground = min(
            minimum_ground, float(final_info.get("ground_clearance", np.inf))
        )
        maximum_node_error = max(
            maximum_node_error, float(final_info.get("node_tracking_rmse", 0.0))
        )
        maximum_tip_error = max(
            maximum_tip_error, float(final_info.get("tip_tracking_error", 0.0))
        )
        if (
            float(final_info.get("reference_progress", 0.0))
            >= env.arch_enforcement_start_progress
            and np.isfinite(float(final_info.get("arch_height", np.nan)))
        ):
            enforced_arch_heights.append(float(final_info["arch_height"]))
    result = {
        "is_success": bool(final_info.get("is_success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "steps": int(env.step_count),
        "return": float(total_reward),
        "reference_progress": float(final_info.get("reference_progress", 0.0)),
        "final_node_tracking_rmse_m": float(
            final_info.get("node_tracking_rmse", np.nan)
        ),
        "final_tip_tracking_error_m": float(
            final_info.get("tip_tracking_error", np.nan)
        ),
        "maximum_node_tracking_rmse_m": maximum_node_error,
        "maximum_tip_tracking_error_m": maximum_tip_error,
        "minimum_wall_clearance_m": minimum_wall,
        "minimum_ground_clearance_m": minimum_ground,
        "final_distal_crossed_fraction": float(
            final_info.get("distal_crossed_fraction", np.nan)
        ),
        "final_target_plane_distance_m": float(
            final_info.get("target_plane_distance", np.nan)
        ),
        "final_tip_speed_mps": float(final_info.get("tip_speed", np.nan)),
        "final_tip_xyz_m": env._rod_arrays()[0][-1].astype(float).tolist(),
        "final_arch_height_m": float(final_info.get("arch_height", np.nan)),
        "minimum_enforced_arch_height_m": (
            float(np.min(enforced_arch_heights))
            if enforced_arch_heights
            else None
        ),
    }
    env.close()
    return result


class TeacherStatusCallback:
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
                        self.recent.append(
                            {
                                "success": float(info.get("is_success", False)),
                                "return": float(info["episode"]["r"]),
                                "length": float(info["episode"]["l"]),
                                "progress": float(info.get("reference_progress", np.nan)),
                                "node_error": float(info.get("node_tracking_rmse", np.nan)),
                                "tip_error": float(info.get("tip_tracking_error", np.nan)),
                                "action_error": float(info.get("teacher_action_error", np.nan)),
                                "wall": float(info.get("wall_clearance", np.nan)),
                            }
                        )
                if self.num_timesteps >= self.next_report:
                    rows = list(self.recent)
                    mean = lambda name: (
                        float(np.nanmean([row[name] for row in rows])) if rows else None
                    )
                    status = {
                        "timesteps": int(self.num_timesteps),
                        "recent_episode_count": len(rows),
                        "recent_success_rate": mean("success"),
                        "recent_mean_return": mean("return"),
                        "recent_mean_length": mean("length"),
                        "recent_mean_final_progress": mean("progress"),
                        "recent_mean_final_node_error": mean("node_error"),
                        "recent_mean_final_tip_error": mean("tip_error"),
                        "recent_mean_teacher_action_error": mean("action_error"),
                        "recent_min_wall_clearance": (
                            float(np.nanmin([row["wall"] for row in rows]))
                            if rows
                            else None
                        ),
                    }
                    temporary = output / "training_status.json.tmp"
                    temporary.write_text(
                        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
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
    teacher_episode = Path(args.teacher_episode).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    for path in (scenario, task_config, teacher_episode, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"training output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("teacher SAC config schema_version must be 1")
    for section in ("environment", "sac", "behavior_cloning", "training"):
        if not isinstance(payload.get(section), dict):
            raise ValueError(f"teacher SAC config is missing section {section!r}")
    environment = dict(payload["environment"])
    sac = dict(payload["sac"])
    bc = dict(payload["behavior_cloning"])
    training = dict(payload["training"])
    total_timesteps = int(
        training["total_timesteps"]
        if args.total_timesteps is None
        else args.total_timesteps
    )
    num_envs = int(training["num_envs"] if args.num_envs is None else args.num_envs)
    bc_epochs = int(bc["epochs"] if args.bc_epochs is None else args.bc_epochs)
    if args.smoke:
        total_timesteps = min(total_timesteps, 32)
        num_envs = 1
        bc_epochs = min(bc_epochs, 2)
        sac["learning_starts"] = 16
        sac["buffer_size"] = 256
        sac["batch_size"] = 16
        training["checkpoint_freq"] = 32
        training["status_freq"] = 16
    if min(total_timesteps, num_envs, bc_epochs) < 1:
        raise ValueError("training sizes must be positive")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_env = ManiSoftTeacherTrackingSACEnv(
        scenario,
        task_config_path=task_config,
        teacher_episode_path=teacher_episode,
        **environment,
    )
    teacher_observations = dataset_env.teacher_observation_batch()
    # The smooth teacher is the nominal feed-forward controller.  One SAC
    # actor learns bounded feedback residuals around it, so exact imitation is
    # the zero policy rather than a second approximate copy of the open-loop
    # actions.
    teacher_actions = np.zeros_like(dataset_env.teacher.actions, dtype=np.float32)
    dataset_env.close()

    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    class TeacherWarmupSAC(TeacherWarmupSACMixin, AnchoredSAC):
        pass

    factories = [
        lambda rank=rank: _make_env(
            str(scenario),
            str(task_config),
            str(teacher_episode),
            environment,
        )
        for rank in range(num_envs)
    ]
    vector = (
        DummyVecEnv(factories)
        if num_envs == 1
        else SubprocVecEnv(factories, start_method="forkserver")
    )
    vector.seed(seed)
    vector = VecNormalize(
        vector,
        norm_obs=bool(training.get("norm_obs", True)),
        norm_reward=bool(training.get("norm_reward", False)),
        clip_obs=float(training.get("clip_obs", 10.0)),
    )
    if vector.norm_obs:
        vector.obs_rms.update(teacher_observations)
        bc_observations = vector.normalize_obs(teacher_observations.copy())
    else:
        bc_observations = teacher_observations

    model = TeacherWarmupSAC(
        "MlpPolicy",
        vector,
        learning_rate=float(sac["learning_rate"]),
        buffer_size=int(sac["buffer_size"]),
        learning_starts=int(sac["learning_starts"]),
        batch_size=int(sac["batch_size"]),
        tau=float(sac["tau"]),
        gamma=float(sac["gamma"]),
        train_freq=int(sac["train_freq"]),
        gradient_steps=int(sac["gradient_steps"]),
        ent_coef=sac.get("ent_coef", "auto"),
        target_entropy=sac.get("target_entropy", "auto"),
        policy_kwargs={"net_arch": list(sac["net_arch"])},
        seed=seed,
        device=args.device,
        verbose=1,
    )
    bc_result = _behavior_clone_actor(
        model,
        bc_observations,
        teacher_actions,
        epochs=bc_epochs,
        batch_size=int(bc["batch_size"]),
        seed=seed,
        output=output,
    )
    if bool(bc.get("exact_zero_residual_mean", True)):
        # Zero residual is the certified smooth teacher controller.  Making
        # the deterministic mean mathematically exact avoids accumulating a
        # tiny supervised-regression bias over 1091 stiff dynamics steps.
        with torch.no_grad():
            model.actor.mu.weight.zero_()
            model.actor.mu.bias.zero_()
        bc_result["exact_zero_residual_mean"] = True
        (output / "behavior_cloning.json").write_text(
            json.dumps(bc_result, indent=2, sort_keys=True), encoding="utf-8"
        )
    model.enable_actor_anchor(float(training["actor_anchor_coef"]))
    model.delay_actor_updates_until(int(training.get("actor_learning_delay_steps", 0)))
    model.save(output / "bc_initial_model")
    vector.save(output / "bc_initial_vecnormalize.pkl")

    bc_evaluation = _evaluate_upright(
        model,
        vector,
        scenario=scenario,
        task_config=task_config,
        teacher_episode=teacher_episode,
        environment=environment,
        seed=seed + 100000,
    )
    (output / "bc_upright_evaluation.json").write_text(
        json.dumps(bc_evaluation, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"phase": "bc_upright_evaluation", **bc_evaluation}, sort_keys=True), flush=True)

    run_config = {
        "scenario": str(scenario),
        "task_config": str(task_config),
        "teacher_episode": str(teacher_episode),
        "training_config": str(config_path),
        "environment": environment,
        "sac": sac,
        "behavior_cloning": {**bc, "epochs": bc_epochs, "result": bc_result},
        "training": {
            **training,
            "total_timesteps": total_timesteps,
            "num_envs": num_envs,
        },
        "device": args.device,
        "provenance": {
            "git_head": _git_head(Path(__file__).resolve().parents[1]),
            "scenario_sha256": _sha256(scenario),
            "task_config_sha256": _sha256(task_config),
            "teacher_episode_sha256": _sha256(teacher_episode),
            "training_config_sha256": _sha256(config_path),
            "training_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "bc_upright_evaluation": bc_evaluation,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    checkpoint_freq = max(
        1, int(training["checkpoint_freq"]) // max(num_envs, 1)
    )
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(output / "checkpoints"),
                name_prefix="unified_teacher_sac",
                save_replay_buffer=True,
                save_vecnormalize=True,
            ),
            TeacherStatusCallback.create(
                output, int(training.get("status_freq", 500))
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
        raise
    else:
        model.save(output / "final_model")
        vector.save(output / "vecnormalize.pkl")
        final_evaluation = _evaluate_upright(
            model,
            vector,
            scenario=scenario,
            task_config=task_config,
            teacher_episode=teacher_episode,
            environment=environment,
            seed=seed + 200000,
        )
        (output / "final_upright_evaluation.json").write_text(
            json.dumps(final_evaluation, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps({"phase": "final_upright_evaluation", **final_evaluation}, sort_keys=True), flush=True)
    finally:
        try:
            vector.close()
        except (BrokenPipeError, EOFError):
            if not interrupted:
                raise


if __name__ == "__main__":
    main()
