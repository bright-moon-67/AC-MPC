"""Diagnose why the visual BC closed-loop success is 0%.

Runs three cheap, retrain-free checks on an existing checkpoint:
  1. pos_branch validation MSE (in meters) -- does v encode the active goal?
  2. v per-dim std over the validation set -- did the visual latent collapse?
  3. closed-loop ablation with v zeroed -- does the policy actually use v?

Run:  python -m experiments.state_only_feasibility.diagnose_visual_bc
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from experiments.state_only_feasibility.collect_pandareach import _first, _numpy
from experiments.state_only_feasibility.collect_pandareach_threewaypoint import (
    _state_from_observation,
)
from experiments.state_only_feasibility.maniskill_pandareach import (
    PandaArmOnlyActionWrapper,
)
from experiments.state_only_feasibility.train_pandareach_threewaypoint_bc import (
    _future_targets,
    load_koopman,
)
from experiments.state_only_feasibility.train_visual_pandareach_bc import (
    VisualBCConfig,
    _make_loader,
    _normalizers,
    _resolve_device,
    _visual_observation,
)
from experiments.state_only_feasibility.visual_encoder import VisualEncoder
from experiments.state_only_feasibility.visual_pandareach_env import (
    VisualPandaReachThreeWaypointEnv,
)
from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor


def _load_checkpoint(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = VisualBCConfig(**payload["config"])
    koopman, kp = load_koopman(
        Path(payload["koopman_checkpoint"]), device
    )
    encoder = VisualEncoder(
        config.v_dim,
        hidden_dims=(config.encoder_hidden_dim,),
        use_dct=config.use_dct,
        depth_scale=config.depth_scale,
        pos_dim=3,
    ).to(device)
    actor = KoopmanMPCActor(
        A=koopman.A,
        B=koopman.B,
        C=koopman.C,
        horizon=config.kmpc_horizon,
        context_dim=3 if config.use_goal_context else config.v_dim,
        hidden_dims=(config.costmap_hidden_dim,),
        action_low=-config.action_limit_rad,
        action_high=config.action_limit_rad,
        solver_iterations=config.kmpc_solver_iterations,
    ).to(device)
    encoder.load_state_dict(payload["encoder"])
    actor.load_state_dict(payload["actor"])
    encoder.eval()
    actor.eval()
    return encoder, actor, koopman, kp, config


def diagnose(
    dataset_path: Path,
    checkpoint_path: Path,
    device_name: str = "auto",
    ablation_episodes: int = 30,
) -> dict:
    device = _resolve_device(device_name)
    encoder, actor, koopman, kp, config = _load_checkpoint(
        checkpoint_path, device
    )
    with np.load(dataset_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    episode = data["episode_id"]
    masks = {
        split: np.isin(episode, data[f"{split}_episode_ids"])
        for split in ("train", "validation", "test")
    }
    center, scale, goal_center, goal_scale = _normalizers(
        data, kp, masks["train"]
    )
    state = ((data["state"] - center) / scale).astype(np.float32)
    goal = (
        (data["active_goal_position"] - goal_center) / goal_scale
    ).astype(np.float32)
    future, future_mask = _future_targets(data, config.kmpc_horizon)
    loader = _make_loader(
        state,
        data["rgb"],
        data["depth"],
        data["action"],
        goal,
        future,
        future_mask,
        masks["validation"],
        config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    # 1) pos_branch validation MSE (meters) and 2) v per-dim std.
    pos_errors: list[float] = []
    v_cells: list[np.ndarray] = []
    with torch.no_grad():
        for state_b, rgb, depth, action, goal_b, _, _ in loader:
            state_b, rgb, depth, goal_b = (
                state_b.to(device),
                rgb.to(device),
                depth.to(device),
                goal_b.to(device),
            )
            v, pos = encoder(rgb, depth)
            pos_err = (pos - goal_b) * torch.as_tensor(
                goal_scale, device=device
            )
            pos_errors.append(pos_err.square().mean(-1).cpu().numpy())
            v_cells.append(v.cpu().numpy())
    pos_rmse = float(np.sqrt(np.mean(np.concatenate(pos_errors))))
    v_all = np.concatenate(v_cells)
    v_std = v_all.std(axis=0)
    print(f"[1] pos_branch val RMSE = {pos_rmse:.4f} m  (goal span ~0.2-0.4 m)")
    print(f"[2] v per-dim std = {np.round(v_std, 4).tolist()}  min={v_std.min():.4f}")

    # 3) closed-loop ablation: real context vs zeroed context.
    def run_eval(zero_context: bool) -> dict:
        make_kwargs = dict(
            num_envs=1,
            obs_mode="rgb+depth",
            control_mode="pd_joint_delta_pos",
            reward_mode="sparse",
            render_mode=None,
            max_episode_steps=config.max_episode_steps,
            goal_threshold=config.goal_threshold,
        )
        if config.env_id == "ACMPC-VisualPandaReach1-v0":
            make_kwargs["goal_region_radius"] = config.goal_region_radius
            make_kwargs["goal_marker_scale"] = config.goal_marker_scale
        env = PandaArmOnlyActionWrapper(
            __import__("gymnasium", fromlist=["make"]).make(
                config.env_id, **make_kwargs
            )
        )
        successes = 0
        completed = []
        try:
            for episode_i in range(ablation_episodes):
                observation, _ = env.reset(
                    seed=config.evaluation_seed_start + episode_i
                )
                for step in range(config.max_episode_steps):
                    robot, rgb, depth = _visual_observation(observation)
                    state_t = torch.as_tensor(
                        (robot - center) / scale,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    rgb_t = torch.as_tensor(
                        rgb, dtype=torch.uint8, device=device
                    ).unsqueeze(0)
                    depth_t = torch.as_tensor(
                        depth, dtype=torch.uint16, device=device
                    ).unsqueeze(0)
                    with torch.no_grad():
                        lifted = koopman.lift(state_t)
                        v, pos = encoder(rgb_t, depth_t)
                        context = pos if config.use_goal_context else v
                        if zero_context:
                            context = torch.zeros_like(context)
                        action = actor(lifted, context).action
                    observation, _, terminated, truncated, info = env.step(
                        np.clip(
                            action.squeeze(0).cpu().numpy()
                            / config.action_limit_rad,
                            -1.0,
                            1.0,
                        ).astype(np.float32)
                    )
                    if bool(_first(terminated)) or bool(_first(truncated)):
                        break
                successes += int(bool(_first(info["success"])))
                completed.append(int(_first(info["waypoints_completed"])))
        finally:
            env.close()
        return {
            "episodes": ablation_episodes,
            "full_success_rate": successes / ablation_episodes,
            "mean_waypoints_completed": float(np.mean(completed)),
            "histogram": {
                str(i): int(np.count_nonzero(np.asarray(completed) == i))
                for i in range(4)
            },
        }

    real = run_eval(zero_context=False)
    zeroed = run_eval(zero_context=True)
    label = "pos" if config.use_goal_context else "v"
    print(f"[3] closed-loop real {label}  : {real}")
    print(f"[3] closed-loop zeroed {label}: {zeroed}")
    return {
        "pos_branch_rmse_m": pos_rmse,
        "v_std_min": float(v_std.min()),
        "v_std": v_std.tolist(),
        "real_v": real,
        "zeroed_v": zeroed,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose a visual BC checkpoint (pos_branch RMSE, "
        "v std, closed-loop context ablation)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "runs/visual_pandareach_single/data/"
            "visual_pandareach1_region_750ep.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/visual_pandareach_single/bc_region_750/best.pt"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ablation-episodes", type=int, default=30)
    args = parser.parse_args()
    diagnose(
        args.dataset,
        args.checkpoint,
        args.device,
        args.ablation_episodes,
    )


if __name__ == "__main__":
    sys.exit(main())
