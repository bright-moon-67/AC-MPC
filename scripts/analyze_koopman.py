"""Deep analysis of the newly trained global Koopman model (coverage 600K)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from antmaze_ac.koopman.model import DeepKoopman
from experiments.state_only_feasibility.train_pandareach_koopman import (
    build_windows,
    fit_normalizer,
    load_dataset,
    prediction_metrics,
    rollout_prediction_metrics,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 4096
K_STEP = 20
NEW = Path("runs/pandareach_threewaypoint/koopman_coverage/best.pt")
OLD = Path("runs/pandareach_threewaypoint/koopman/best.pt")
COVERAGE = Path("runs/pandareach_threewaypoint/data/pandareach_coverage_600k.npz")
DLS = Path("runs/pandareach_threewaypoint/data/pandareach_dls_500.npz")


def load_model(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch = ckpt["architecture"]
    model = DeepKoopman(
        state_dim=int(arch["state_dim"]),
        action_dim=int(arch["action_dim"]),
        lift_dim=int(arch["lift_dim"]),
        hidden_dims=tuple(arch["hidden_dims"]),
        activation=str(arch.get("activation", "silu")),
    )
    model.load_state_dict(ckpt["model_state"])
    model.freeze_dynamics()
    center = np.asarray(ckpt["normalizer"]["center"], dtype=np.float32)
    scale = np.asarray(ckpt["normalizer"]["scale"], dtype=np.float32)
    return model, ckpt, center, scale


def linear_analysis(model: DeepKoopman, tag: str) -> None:
    A = model.A.detach().cpu()
    B = model.B.detach().cpu()
    n, m = A.shape
    eig = torch.linalg.eigvals(A)
    mag = eig.abs()
    rho = mag.max().item()
    tol = 1e-6
    unstable_idx = torch.nonzero(mag >= 1.0 - 1e-4).flatten()
    print(f"\n=== {tag} linear analysis ===")
    print(f"  spectral radius rho(A) = {rho:.6f}")
    print(f"  |lambda|>=1 count = {len(unstable_idx)} / {n}")
    if len(unstable_idx):
        vals = eig[unstable_idx]
        print(f"  unstable eigenvalues: {[f'{v:.3f}' for v in vals.real.cpu().numpy().tolist()]}")
        print(f"  |lambda| max = {mag.max().item():.6f}")
    # PBH stabilizability margin for unstable modes
    margins = []
    for idx in unstable_idx:
        lam = eig[idx]
        pbh = torch.cat((lam * torch.eye(n) - A, B), dim=-1)
        margins.append(torch.linalg.svdvals(pbh)[-1].item())
    if margins:
        print(f"  PBH sigma_min over unstable modes: min={min(margins):.3e} max={max(margins):.3e}")
        print(f"    (sigma_min >> 0 => unstable modes are controllable)")
    # structure of A: norm of top-left 17x17 (physical->physical)
    A17 = A[:17, :17]
    print(f"  ||A[:17,:17] - I|| = {(A17 - torch.eye(17)).norm().item():.4f}  (identity-skip physical block)")
    print(f"  ||B||_F = {B.norm().item():.4f}")
    print(f"  ||A||_F = {A.norm().item():.4f}")


def eval_on(data_path: Path, tag: str) -> None:
    print(f"\n########## Dataset: {data_path.name} ({tag}) ##########")
    data, masks = load_dataset(data_path)
    state_kind = str(np.asarray(data["state_kind"]).item()) if "state_kind" in data else "q_qdot_error"
    for name, path in (("NEW(global)", NEW), ("OLD(expert)", OLD)):
        model, ckpt, center, scale = load_model(path)
        model = model.to(DEVICE)
        tail = "tcp" if state_kind == "q_qdot_tcp" else "e"
        # use the model's own normalizer (its training frame)
        ones = prediction_metrics(model, data, masks["test"] if "test" in masks else np.ones(len(data["state"]), bool),
                                  center, scale, DEVICE, BATCH, tail)
        # rollout on test episodes (or all for DLS which has no split)
        sel = data["test_episode_ids"] if "test_episode_ids" in data else data["episode_id"]
        windows = build_windows(data, sel, center, scale, k_step=K_STEP)
        roll = rollout_prediction_metrics(model, *windows, center, scale, DEVICE, BATCH, tail)
        print(f"\n  [{name}] best_epoch={ckpt.get('best_epoch')} state_kind={ckpt.get('state_kind')}")
        for group in ("q", "qdot", "tcp" if tail == "tcp" else "e", "all"):
            one = ones[group]
            r20 = roll["all_steps"][group]
            h1 = roll["horizons"]["1"][group]
            print(f"    {group:5s} one-step rmse={one['rmse']:.5f} (hold {one['hold_rmse']:.4f}) | "
                  f"rollout h1 rmse={h1['rmse']:.5f} h20 rmse={r20['rmse']:.5f} (hold20 {r20['hold_rmse']:.4f})")
        print(f"    rollout normalized_mse_all_steps={roll['normalized_mse_all_steps']:.6f}")


def main() -> None:
    report = json.load(open("runs/pandareach_threewaypoint/koopman_coverage/report.json"))
    print("### NEW Koopman report summary ###")
    print(f"best_epoch={report['best_epoch']} best_val_rollout_nMSE={report['best_validation_rollout_normalized_mse']:.6f}")
    print(f"spectral_radius_A={report['spectral_radius_A']:.6f} state_kind={report['state_kind']}")
    hist = report["history"]
    print("val_nMSE trajectory (every 50 epochs):")
    for rec in hist:
        if rec["epoch"] % 50 == 0 or rec["epoch"] == report["best_epoch"]:
            mark = " <-- best" if rec["epoch"] == report["best_epoch"] else ""
            print(f"  epoch={rec['epoch']:4d} train={rec['train_total']:.5f} val_nMSE={rec['validation_rollout_normalized_mse']:.6f}{mark}")

    new_model, new_ckpt, _, _ = load_model(NEW)
    old_model, old_ckpt, _, _ = load_model(OLD)
    linear_analysis(new_model, "NEW (coverage)")
    linear_analysis(old_model, "OLD (expert)")

    eval_on(COVERAGE, "coverage 600K test split")
    eval_on(DLS, "DLS expert 500")


if __name__ == "__main__":
    main()
