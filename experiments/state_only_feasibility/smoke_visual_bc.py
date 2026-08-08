"""Smoke test for the single-stage visual BC building blocks.

Validates (without any training data):
  1. the 2D DCT-II matrix matches scipy (orthonormal) when scipy is present,
  2. ``VisualEncoder`` forward + CAM map shapes,
  3. the full chain Koopman lift -> encoder -> ``KoopmanMPCActor`` -> action,
     using the frozen ``koopman_coverage`` checkpoint.

Run:  python -m experiments.state_only_feasibility.smoke_visual_bc
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from experiments.state_only_feasibility.train_pandareach_threewaypoint_bc import (
    load_koopman,
)
from experiments.state_only_feasibility.visual_encoder import (
    VisualEncoder,
    build_dct_matrix,
    dct_2d,
)
from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor


def _check_dct() -> None:
    transform = build_dct_matrix(8, torch.float64, torch.device("cpu"))
    eye_error = (transform @ transform.mT - torch.eye(8)).abs().max().item()
    assert eye_error < 1e-9, f"DCT matrix not orthonormal: {eye_error}"
    x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    y = dct_2d(x, transform)
    try:
        from scipy.fft import dctn

        reference = torch.as_tensor(
            dctn(x.numpy(), type=2, norm="ortho", axes=(-2, -1)),
            dtype=torch.float64,
        )
        scipy_error = (y - reference).abs().max().item()
        assert scipy_error < 1e-6, f"DCT mismatch vs scipy: {scipy_error}"
        print(f"  [ok] DCT-II matches scipy (max err {scipy_error:.2e})")
    except ImportError:
        print("  [ok] DCT-II orthonormal (scipy unavailable, skipped equivalence)")
    print(f"  [ok] DCT matrix orthonormal (max err {eye_error:.2e})")


def _check_encoder() -> None:
    encoder = VisualEncoder(v_dim=16, use_dct=True, depth_scale=2500.0)
    encoder.eval()
    rgb = torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8)
    depth = torch.randint(500, 1500, (2, 128, 128, 1), dtype=torch.uint16)
    with torch.no_grad():
        v, pos = encoder(rgb, depth)
        cam = encoder.activation_map(rgb, depth)
    assert tuple(v.shape) == (2, 16), v.shape
    assert tuple(pos.shape) == (2, 3), pos.shape
    assert torch.isfinite(v).all()
    assert cam.shape[:1] == (2,) and cam.ndim == 3, cam.shape
    print(f"  [ok] encoder v{tuple(v.shape)} pos{tuple(pos.shape)} cam{tuple(cam.shape)}")

    encoder_plain = VisualEncoder(v_dim=32, use_dct=False, depth_scale=2500.0)
    encoder_plain.eval()
    with torch.no_grad():
        v_plain, _ = encoder_plain(rgb, depth)
    assert tuple(v_plain.shape) == (2, 32)
    print(f"  [ok] encoder without DCT v{tuple(v_plain.shape)}")


def _check_full_chain(koopman_path: Path) -> None:
    device = torch.device("cpu")
    koopman, checkpoint = load_koopman(koopman_path, device)
    assert koopman.state_dim == 17 and koopman.lifted_dim == 49
    assert koopman.action_dim == 7
    encoder = VisualEncoder(v_dim=16, use_dct=True, depth_scale=2500.0)
    encoder.eval()
    actor = KoopmanMPCActor(
        A=koopman.A,
        B=koopman.B,
        C=koopman.C,
        horizon=10,
        context_dim=16,
        hidden_dims=(128,),
        action_low=-0.1,
        action_high=0.1,
        solver_iterations=20,
    )
    actor.eval()

    center = torch.as_tensor(checkpoint["normalizer"]["center"], dtype=torch.float32)
    scale = torch.as_tensor(checkpoint["normalizer"]["scale"], dtype=torch.float32)
    state = torch.tensor(
        [[-0.011, 0.398, 0.002, -1.957, -0.003, 2.356, 0.785, 0, 0, 0, 0, 0, 0, 0,
          -0.071, -0.005, 0.205]],
        dtype=torch.float32,
    )
    rgb = torch.randint(0, 255, (1, 128, 128, 3), dtype=torch.uint8)
    depth = torch.randint(500, 1500, (1, 128, 128, 1), dtype=torch.uint16)

    with torch.no_grad():
        lifted = koopman.lift((state - center) / scale)
        v, pos = encoder(rgb, depth)
        output = actor(lifted, v)
    assert tuple(lifted.shape) == (1, 49), lifted.shape
    assert tuple(v.shape) == (1, 16)
    assert tuple(output.action.shape) == (1, 7), output.action.shape
    assert tuple(output.action_sequence.shape) == (1, 10, 7)
    assert torch.isfinite(output.action).all()
    assert output.action.abs().max().item() <= 0.1 + 1e-5
    print(
        f"  [ok] chain lift{tuple(lifted.shape)} v{tuple(v.shape)} "
        f"u{tuple(output.action.shape)} seq{tuple(output.action_sequence.shape)} "
        f"max|u|={output.action.abs().max().item():.4f}"
    )


def main() -> None:
    koopman_path = Path(
        "runs/pandareach_threewaypoint/koopman_coverage/best.pt"
    )
    if not koopman_path.exists():
        raise SystemExit(f"Missing Koopman checkpoint: {koopman_path}")
    print("checking DCT...")
    _check_dct()
    print("checking encoder...")
    _check_encoder()
    print("checking full chain...")
    _check_full_chain(koopman_path)
    print("SMOKE OK")


if __name__ == "__main__":
    sys.exit(main())
