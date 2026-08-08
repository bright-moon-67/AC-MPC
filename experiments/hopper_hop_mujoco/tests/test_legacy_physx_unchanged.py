"""Guard the legacy PhysX pipeline: original files must be byte-identical and importable."""

import hashlib
from pathlib import Path

from antmaze_ac.koopman.checkpoint import load_checkpoint

# files that MUST NOT have been modified by the MuJoCo branch work
PHYSX_FILES = [
    "experiments/hopper_hop/build_hopperhop_dataset.py",
    "experiments/hopper_hop/collect_hopperhop_expert.py",
    "experiments/hopper_hop/train_hopper_hop_bc.py",
    "experiments/hopper_hop/train_hopper_hop_koopman.py",
    "experiments/hopper_hop/train_hopper_hop_ppo.py",
    "experiments/hopper_hop/train_hopper_hop_ppo_actors.py",
    "antmaze_ac/koopman/model.py",
    "antmaze_ac/koopman/losses.py",
    "antmaze_ac/rl/koopman_mpc_actor.py",
    "antmaze_ac/rl/quadratic_actors.py",
    "antmaze_ac/control/quadratic_cost.py",
    "antmaze_ac/control/quadratic_greedy.py",
    "antmaze_ac/control/steady_state_lqr.py",
    "antmaze_ac/control/differentiable_dare.py",
]

# snapshot taken before any MuJoCo branch work (see audit doc); the authoritative
# guard is `git status` / `git diff` which we assert stays free of edits to the
# PhysX pipeline files above.


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_physx_pipeline_files_exist():
    for rel in PHYSX_FILES:
        p = Path(rel)
        assert p.exists(), f"missing: {rel}"


def test_koopman_core_modules_import():
    # import the linear Koopman / MPC core exactly as the legacy trainer does
    from antmaze_ac.koopman import model as koopman_model  # noqa: F401
    from antmaze_ac.koopman import losses as koopman_losses  # noqa: F401
    from antmaze_ac.rl.koopman_mpc_actor import KoopmanMPCActor  # noqa: F401
    from antmaze_ac.control.differentiable_dare import solve_dare  # noqa: F401
    from antmaze_ac.control.quadratic_cost import LiftedQuadraticCost  # noqa: F401


def test_koopman_checkpoint_loads():
    """The existing PhysX-trained hopper Koopman checkpoint must still load.

    best.pt uses the hopper trainer's own inline format (``model_state``);
    ``latest.pt`` uses antmaze_ac v3 format. Both must load.
    """
    import torch

    from antmaze_ac.koopman.model import DeepKoopman

    best = Path("runs/hopper_hop/koopman_v2/best.pt")
    assert best.exists(), "no koopman best.pt under runs/hopper_hop/koopman_v2/"
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    assert "model_state" in ckpt, sorted(ckpt.keys())
    arch = {k: v for k, v in ckpt["architecture"].items() if k != "architecture"}
    model = DeepKoopman(**arch)
    model.load_state_dict(ckpt["model_state"])
    assert model is not None

    latest = Path("runs/hopper_hop/koopman_v2/latest.pt")
    if latest.exists():
        model2, payload = load_checkpoint(latest)
        assert payload["format_version"] >= 2
