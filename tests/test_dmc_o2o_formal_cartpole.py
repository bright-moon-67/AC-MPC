from pathlib import Path
from types import SimpleNamespace
import json
import zipfile
import numpy as np
import pytest

from experiments.dmc.o2o.formal_cartpole import FORMAL_METHODS, FORMAL_TRAINING_SEEDS, training_command
from experiments.dmc.o2o.formal_cartpole_dataset import selected_episode_indices
from experiments.dmc.o2o.formal_cartpole_koopman import training_command as koopman_command
from experiments.dmc.o2o.formal_cartpole_watcher import archive_checkpoints
from experiments.dmc.o2o.formal_cartpole_results import training_seed_statistics
from experiments.playground.train_koopman import _differentiable_spectral_radius


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_cartpole_formal_contract() -> None:
    assert len(FORMAL_METHODS) == 7
    assert FORMAL_TRAINING_SEEDS == tuple(range(20260851, 20260856))
    indices = selected_episode_indices()
    assert len(indices) == 100
    assert indices[:10] == tuple(range(0, 1000, 100))
    dataset = SimpleNamespace(path=Path("/tmp/cartpole100k.npz"))
    koopman = SimpleNamespace(path=Path("/tmp/koopman.npz"))
    for method in FORMAL_METHODS:
        command = training_command(method=method, training_seed=20260851, dataset=dataset,
                                   koopman=koopman, output_dir=Path("/tmp/run") / method,
                                   device="cuda")
        assert _option(command, "--offline-updates") == "50000"
        assert _option(command, "--online-steps") == "20000"
        assert _option(command, "--kmpc-horizon") == "20"


def test_cartpole_koopman_and_archive(tmp_path: Path) -> None:
    command = koopman_command(training_seed=20260851, prepared_data_dir=tmp_path / "prepared",
                              output_dir=tmp_path / "koopman", python_executable=Path("/x/python"))
    assert _option(command, "--lift-dim") == "10"
    assert _option(command, "--k-step") == "50"
    (tmp_path / "latest.pt").write_bytes(b"latest-checkpoint")
    (tmp_path / "online_020000.pt").write_bytes(b"final-checkpoint")
    archive = archive_checkpoints(tmp_path)
    assert not list(tmp_path.glob("*.pt"))
    with zipfile.ZipFile(archive) as handle:
        assert sorted(handle.namelist()) == ["latest.pt", "online_020000.pt"]
    manifest = json.loads((tmp_path / "checkpoints.archive.json").read_text())
    assert len(manifest["members"]) == 2


def test_gpu_safe_spectral_radius_estimator() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jp

    matrix = jp.asarray(np.diag([0.5, 0.8, 0.97]), dtype=jp.float32)
    estimate = float(_differentiable_spectral_radius(matrix, iterations=64))
    assert abs(estimate - 0.97) < 1e-4
    gradient = jax.grad(_differentiable_spectral_radius)(matrix)
    assert np.isfinite(np.asarray(gradient)).all()


def test_cartpole_formal_training_seed_statistics() -> None:
    summary = training_seed_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["mean"] == 3.0
    assert summary["sample_std"] == np.std([1, 2, 3, 4, 5], ddof=1)
    assert summary["ci95_student_t"][0] < 3.0 < summary["ci95_student_t"][1]
