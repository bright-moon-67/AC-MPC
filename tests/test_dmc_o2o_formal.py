from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from experiments.dmc.o2o.formal_walker import (
    FORMAL_METHODS,
    FORMAL_TRAINING_SEEDS,
    _resolve_prepared_data_path,
    final_evaluation_commands,
    training_command,
)
from experiments.dmc.o2o.formal_walker_koopman import (
    training_command as koopman_training_command,
)
from experiments.dmc.o2o.formal_walker_results import _training_seed_summary
from experiments.dmc.o2o.formal_walker_followup import (
    STRUCTURED_METHODS,
    _koopman_complete,
)


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_formal_method_set_and_single_method_command() -> None:
    assert FORMAL_METHODS == (
        "Cal-RLPD-KMPC",
        "Cal-RLPD",
        "Cal-RLPD-Lift",
        "Cal-QL",
        "RLPD",
        "AWAC",
        "IQL",
    )
    assert "SAC" not in FORMAL_METHODS
    assert not any("Raw" in method for method in FORMAL_METHODS)

    dataset = SimpleNamespace(path=Path("/tmp/formal_dataset.npz"))
    koopman = SimpleNamespace(path=Path("/tmp/formal_koopman.npz"))
    for method in FORMAL_METHODS:
        command = training_command(
            method=method,
            training_seed=20260851,
            dataset=dataset,
            koopman=koopman,
            output_dir=Path("/tmp/run") / method,
            device="cuda",
        )
        assert _option(command, "--offline-updates") == "50000"
        assert _option(command, "--offline-eval-interval-updates") == "5000"
        assert _option(command, "--online-steps") == "20000"
        assert _option(command, "--eval-interval-online-steps") == "2500"
        assert _option(command, "--kmpc-horizon") == "8"
        assert _option(command, "--mpve-total-horizon") == "4"
        if method in {"Cal-RLPD-KMPC", "Cal-RLPD-Lift"}:
            assert _option(command, "--koopman") == str(koopman.path)
        else:
            assert "--koopman" not in command


def test_formal_koopman_and_final_evaluation_commands() -> None:
    command = koopman_training_command(
        training_seed=20260851,
        prepared_data_dir=Path("/tmp/prepared"),
        output_dir=Path("/tmp/koopman/seed_123"),
        python_executable=Path("/tmp/jax-python"),
    )
    assert _option(command, "--k-step") == "20"
    assert _option(command, "--lift-dim") == "48"
    assert _option(command, "--seed") == "20260851"

    evaluations = final_evaluation_commands(
        method="IQL", run_dir=Path("/tmp/run/IQL")
    )
    assert [_option(value, "--checkpoint") for value in evaluations] == [
        "online_000000",
        "online_020000",
    ]
    assert all(_option(value, "--parallel-workers") == "10" for value in evaluations)


def test_formal_statistics_use_training_seed_means() -> None:
    values = [float(value) for value in range(len(FORMAL_TRAINING_SEEDS))]
    summary = _training_seed_summary(values)
    assert summary["training_seed_count"] == 5
    assert summary["mean"] == 2.0
    assert summary["sample_std"] > 0
    assert summary["standard_error"] == summary["sample_std"] / (5**0.5)
    assert summary["inference_unit"] == "independent_training_seed"


def test_formal_followup_requires_completed_koopman(tmp_path: Path) -> None:
    assert STRUCTURED_METHODS == ("Cal-RLPD-Lift", "Cal-RLPD-KMPC")
    assert not _koopman_complete(tmp_path)
    (tmp_path / "best.npz").write_bytes(b"checkpoint")
    (tmp_path / "run.json").write_text('{"completed": false}', encoding="utf-8")
    assert not _koopman_complete(tmp_path)
    (tmp_path / "run.json").write_text('{"completed": true}', encoding="utf-8")
    assert _koopman_complete(tmp_path)


def test_formal_prepared_data_relocation_is_resolved() -> None:
    path = "/root/autodl-tmp/AC-MPC/runs/o2o/data/koopman/WalkerRun"
    resolved = _resolve_prepared_data_path(path)
    assert str(resolved) == path or str(resolved).startswith(
        "/root/acmpc-o2o-nonformal-20260819/data/"
    )
