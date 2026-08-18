from __future__ import annotations

import fcntl
import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.dmc.o2o import run_matrix as matrix
from experiments.dmc.o2o import train as train_module
from experiments.dmc.o2o.checkpoint import (
    CHECKPOINT_KIND,
    atomic_torch_save,
    load_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _spec(tmp_path: Path, *, max_parallel: int = 1) -> matrix.MatrixSpec:
    dataset = tmp_path / "transitions.npz"
    koopman = tmp_path / "best.npz"
    dataset.write_bytes(b"synthetic dataset identity")
    koopman.write_bytes(b"synthetic koopman identity")
    return matrix.MatrixSpec(
        repo_root=REPO_ROOT,
        dataset=dataset,
        koopman=koopman,
        root=tmp_path / "matrix",
        seeds=(11,),
        device="cpu",
        eval_device="cpu",
        offline_updates=10,
        online_steps=5_000,
        max_parallel=max_parallel,
    ).resolved()


def _identity() -> dict[str, Any]:
    return {
        "git": {
            "commit": "a" * 40,
            "branch": "test",
            "dirty": False,
            "porcelain": [],
            "porcelain_sha256": "b" * 64,
        },
        "training_source": {"sha256": "c" * 64, "files": []},
        "result_source": {"sha256": "d" * 64, "files": []},
        "runner": {
            "path": "experiments/dmc/o2o/run_matrix.py",
            "sha256": "e" * 64,
        },
        "runtime": {"python_version": "test"},
    }


def _status(jobs: list[matrix.Job]) -> dict[str, Any]:
    return {
        "state": "running",
        "updated_utc": None,
        "jobs": {
            job.job_id: {
                "state": "pending",
                "attempts": [],
                "pid": None,
                "returncode": None,
            }
            for job in jobs
        },
    }


def _write_completed_run_metadata(
    spec: matrix.MatrixSpec,
    *,
    method: str,
    initialization: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    dataset = matrix._expected_dataset(spec)
    shared_koopman = matrix._expected_koopman(spec)
    koopman = matrix._method_koopman(method, shared_koopman)
    config = spec.config(method, 11)
    run_dir = spec.run_dir(11, method)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "latest.pt").write_bytes(b"synthetic latest identity")
    metadata = {
        "kind": matrix.RUN_KIND,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "dataset": dataset,
        "koopman": koopman,
        "initialization": initialization,
        "completed": True,
        "offline_updates_completed": (
            config.offline_updates if config.uses_calql else 0
        ),
        "online_steps_completed": config.online_steps,
    }
    (run_dir / "run.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )
    return run_dir, dataset, koopman


def _extension_lineage(
    spec: matrix.MatrixSpec, *, method: str, seed: int = 11
) -> dict[str, Any]:
    base = spec.config(method, seed)
    target = spec.target_config(method, seed)
    return {
        "kind": "acmpc_o2o_online_extension_v1",
        "previous_online_steps": base.online_steps,
        "extended_online_steps": target.online_steps,
        "previous_config_fingerprint": base.fingerprint,
        "extended_config_fingerprint": target.fingerprint,
        "requested_unix_seconds": 1234.5,
    }


def test_parse_seeds_and_plan_has_exact_mpve_fork(tmp_path: Path) -> None:
    assert "experiments/dmc/o2o/runner.py" in matrix._TRAINING_SOURCE_FILES
    assert matrix.parse_seeds("1, 2,3") == (1, 2, 3)
    with pytest.raises(Exception, match="duplicates"):
        matrix.parse_seeds("1,1")

    spec = _spec(tmp_path)
    dataset = matrix._expected_dataset(spec)
    koopman = matrix._expected_koopman(spec)
    jobs = matrix.build_jobs(spec, dataset=dataset, koopman=koopman)
    training = [job for job in jobs if job.stage == "train"]
    assert [job.method for job in training] == list(matrix.MATRIX_METHODS)
    for job in training:
        if job.method in matrix.RAW_METHODS:
            assert "--koopman" not in job.argv
        else:
            assert job.argv[job.argv.index("--koopman") + 1] == str(spec.koopman)
    mpve = training[-1]
    source = spec.offline_source(11)
    assert mpve.artifact_dependency == source
    index = mpve.argv.index("--initialize-from-offline")
    assert mpve.argv[index + 1] == str(source)
    evaluations = [job for job in jobs if job.stage == "evaluate"]
    assert len(evaluations) == len(matrix.MATRIX_METHODS)
    assert all(
        ("--koopman" not in job.argv)
        if job.method in matrix.RAW_METHODS
        else ("--koopman" in job.argv)
        for job in evaluations
    )
    assert all(job.depends_on for job in evaluations)
    assert jobs[-2].stage == "aggregate"
    aggregate = jobs[-2]
    manifest_index = aggregate.argv.index("--matrix-manifest")
    assert aggregate.argv[manifest_index + 1] == str(
        spec.root / "matrix_manifest.json"
    )
    assert jobs[-1].stage == "plot"


def test_dry_run_writes_atomic_audit_without_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, max_parallel=4)
    monkeypatch.setattr(matrix, "_identity_bundle", lambda _spec: _identity())

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("dry-run must not spawn a child")

    result = matrix.run_matrix(spec, dry_run=True, popen_factory=forbidden)
    assert result["state"] == "dry_run"
    manifest = json.loads((spec.root / "matrix_manifest.json").read_text())
    status = json.loads((spec.root / "matrix_status.json").read_text())
    assert manifest["kind"] == matrix.MANIFEST_KIND
    assert manifest["experiment"]["max_parallel"] == 4
    assert manifest["experiment"]["child_environment"] == matrix.CHILD_ENVIRONMENT
    contracts = manifest["experiment"]["method_observation_contract"]
    assert contracts["Cal-QL-Raw"]["koopman"] is None
    assert contracts["RLPD-Raw"]["actor_critic_input"] == (
        "raw_normalized_observation"
    )
    assert contracts["Cal-RLPD-AC-KMPC"]["koopman"]["sha256"] == (
        matrix._sha256_file(spec.koopman)
    )
    assert manifest["experiment"]["evaluation_grid_online_steps"] == [
        0,
        1_000,
        2_500,
        5_000,
    ]
    assert len(manifest["jobs"]) == 2 * len(matrix.MATRIX_METHODS) + 2
    assert status["state"] == "dry_run"
    assert {job["state"] for job in status["jobs"].values()} == {"planned"}


def test_run_json_only_initialization_window_is_safely_replayed(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    run_dir, dataset, koopman = _write_completed_run_metadata(
        spec, method="RLPD-Raw", initialization=None
    )
    (run_dir / "latest.pt").unlink()
    metadata_path = run_dir / "run.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in (
        "completed",
        "offline_updates_completed",
        "online_steps_completed",
    ):
        metadata.pop(key, None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert matrix._check_run_identity(
        spec,
        seed=11,
        method="RLPD-Raw",
        dataset=dataset,
        koopman=koopman,
    ) == "restart_initialization"

    # Once any post-initialization artifact exists, absence of latest.pt is
    # ambiguous and must remain fail-closed.
    (run_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no resumable latest.pt"):
        matrix._check_run_identity(
            spec,
            seed=11,
            method="RLPD-Raw",
            dataset=dataset,
            koopman=koopman,
        )


def test_matrix_extension_requires_every_completed_source_and_is_common(
    tmp_path: Path,
) -> None:
    source = _spec(tmp_path)
    dataset = matrix._expected_dataset(source)
    shared_koopman = matrix._expected_koopman(source)
    for method in matrix.MATRIX_METHODS:
        if method == matrix.MPVE_METHOD:
            offline = source.offline_source(11)
            offline.parent.mkdir(parents=True, exist_ok=True)
            offline.write_bytes(b"paired immutable offline checkpoint")
            initialization = {
                "kind": "acmpc_o2o_offline_fork_v1",
                "source_path": str(offline.resolve()),
                "source_sha256": matrix._sha256_file(offline),
                "source_method": matrix.MPVE_SOURCE_METHOD,
                "source_config_fingerprint": source.config(
                    matrix.MPVE_SOURCE_METHOD, 11
                ).fingerprint,
                "shared_state": "actor_critic_target_temperature_optimizers_rng",
            }
        else:
            initialization = None
        _write_completed_run_metadata(
            source, method=method, initialization=initialization
        )

    extension = dataclasses.replace(source, extend_online_steps=10_000)
    jobs = matrix.build_jobs(
        extension, dataset=dataset, koopman=shared_koopman
    )
    training = [job for job in jobs if job.stage == "train"]
    assert len(training) == len(matrix.MATRIX_METHODS)
    assert all(job.mode_at_plan == "extend" for job in training)
    assert all(
        job.argv[job.argv.index("--extend-online-steps") + 1] == "10000"
        for job in training
    )
    assert all("--initialize-from-offline" not in job.argv for job in training)

    (source.run_dir(11, "RLPD-Raw") / "run.json").unlink()
    with pytest.raises((RuntimeError, ValueError), match="artifacts exist|complete"):
        matrix.build_jobs(extension, dataset=dataset, koopman=shared_koopman)


def test_matrix_extension_retry_accepts_base_partial_and_target_complete_mix(
    tmp_path: Path,
) -> None:
    source = _spec(tmp_path)
    dataset = matrix._expected_dataset(source)
    shared_koopman = matrix._expected_koopman(source)
    initializations: dict[str, dict[str, Any] | None] = {}
    for method in matrix.MATRIX_METHODS:
        if method == matrix.MPVE_METHOD:
            offline = source.offline_source(11)
            offline.parent.mkdir(parents=True, exist_ok=True)
            offline.write_bytes(b"paired immutable offline checkpoint")
            initialization: dict[str, Any] | None = {
                "kind": "acmpc_o2o_offline_fork_v1",
                "source_path": str(offline.resolve()),
                "source_sha256": matrix._sha256_file(offline),
                "source_method": matrix.MPVE_SOURCE_METHOD,
                "source_config_fingerprint": source.config(
                    matrix.MPVE_SOURCE_METHOD, 11
                ).fingerprint,
                "shared_state": "actor_critic_target_temperature_optimizers_rng",
            }
        else:
            initialization = None
        initializations[method] = initialization
        _write_completed_run_metadata(
            source, method=method, initialization=initialization
        )

    extension = dataclasses.replace(source, extend_online_steps=10_000)
    partial_method = "RLPD-Raw"
    complete_method = "Cal-RLPD-Raw"
    for method, completed in ((partial_method, False), (complete_method, True)):
        path = source.run_dir(11, method) / "run.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        target = extension.target_config(method, 11)
        metadata.update(
            config=target.to_dict(),
            config_fingerprint=target.fingerprint,
            online_extension=_extension_lineage(extension, method=method),
            completed=completed,
        )
        if completed:
            metadata["online_steps_completed"] = target.online_steps
        else:
            metadata.pop("online_steps_completed", None)
            metadata.pop("offline_updates_completed", None)
        path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    jobs = matrix.build_jobs(
        extension, dataset=dataset, koopman=shared_koopman
    )
    modes = {
        job.method: job.mode_at_plan for job in jobs if job.stage == "train"
    }
    assert modes[partial_method] == "resume_extension"
    assert modes[complete_method] == "completed"
    assert all(
        modes[method] == "extend"
        for method in matrix.MATRIX_METHODS
        if method not in {partial_method, complete_method}
    )
    assert all(
        "--extend-online-steps" in job.argv
        for job in jobs
        if job.stage == "train"
    )

    partial_path = source.run_dir(11, partial_method) / "run.json"
    tampered = json.loads(partial_path.read_text(encoding="utf-8"))
    tampered["online_extension"]["previous_config_fingerprint"] = "0" * 64
    partial_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid lineage"):
        matrix.build_jobs(extension, dataset=dataset, koopman=shared_koopman)


def test_extension_manifest_archive_is_copy_first_idempotent_and_heals_status_gap(
    tmp_path: Path,
) -> None:
    source = _spec(tmp_path)
    extension = dataclasses.replace(source, extend_online_steps=10_000)
    dataset = matrix._expected_dataset(source)
    koopman = matrix._expected_koopman(source)
    identity = _identity()
    base_manifest = matrix._manifest(
        source, dataset=dataset, koopman=koopman, identity=identity, jobs=[]
    )
    extension_manifest = matrix._manifest(
        extension, dataset=dataset, koopman=koopman, identity=identity, jobs=[]
    )
    source.root.mkdir(parents=True, exist_ok=True)
    manifest_path = source.root / "matrix_manifest.json"
    status_path = source.root / "matrix_status.json"
    base_status = {
        "kind": matrix.STATUS_KIND,
        "matrix_fingerprint": base_manifest["matrix_fingerprint"],
        "state": "completed",
        "runner_pid": 999_999_999,
        "jobs": {},
        "invocations": [],
    }
    matrix._atomic_json(manifest_path, base_manifest)
    matrix._atomic_json(status_path, base_status)

    prepared = matrix._archive_completed_source_matrix_for_extension(
        extension,
        manifest_path=manifest_path,
        status_path=status_path,
        new_manifest=extension_manifest,
    )
    # Archiving is copy-first: both active base files remain valid until the
    # caller atomically replaces each active file.
    assert matrix._read_json(manifest_path) == base_manifest
    assert matrix._read_json(status_path) == base_status
    archived_manifest = Path(prepared["extension_lineage"]["archived_manifest"])
    archived_status = Path(prepared["extension_lineage"]["archived_status"])
    assert matrix._read_json(archived_manifest) == base_manifest
    assert matrix._read_json(archived_status) == base_status

    repeated = matrix._archive_completed_source_matrix_for_extension(
        extension,
        manifest_path=manifest_path,
        status_path=status_path,
        new_manifest=matrix._manifest(
            extension,
            dataset=dataset,
            koopman=koopman,
            identity=identity,
            jobs=[],
        ),
    )
    assert repeated["extension_lineage"] == prepared["extension_lineage"]
    replacement = matrix._compatible_existing_manifest(manifest_path, repeated)
    assert replacement["matrix_fingerprint"] == extension_manifest[
        "matrix_fingerprint"
    ]

    # Simulate a crash after replacing the active manifest but before replacing
    # its status.  A retry recovers lineage from the active manifest and treats
    # the archived completed base status as the expected predecessor.
    matrix._atomic_json(manifest_path, replacement)
    recovered = matrix._archive_completed_source_matrix_for_extension(
        extension,
        manifest_path=manifest_path,
        status_path=status_path,
        new_manifest=matrix._manifest(
            extension,
            dataset=dataset,
            koopman=koopman,
            identity=identity,
            jobs=[],
        ),
    )
    assert recovered["extension_lineage"] == prepared["extension_lineage"]
    status = matrix._initial_status(status_path, recovered, [])
    assert status["matrix_fingerprint"] == recovered["matrix_fingerprint"]
    assert status["state"] == "initializing"


@pytest.mark.parametrize("crash_point", ("latest", "best", "run"))
def test_train_extension_identity_migration_recovers_every_write_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
) -> None:
    source = _spec(tmp_path)
    base = source.config("RLPD-Raw", 11)
    output = tmp_path / crash_point
    output.mkdir()
    payload = {
        "kind": CHECKPOINT_KIND,
        "config": base.to_dict(),
        "config_fingerprint": base.fingerprint,
        "online_extension": None,
        "online_step": base.online_steps,
    }
    atomic_torch_save(output / "latest.pt", payload)
    best_payload = dict(payload)
    best_payload["online_step"] = 2_500
    atomic_torch_save(output / "best.pt", best_payload)
    run_metadata = {
        "kind": matrix.RUN_KIND,
        "config": base.to_dict(),
        "config_fingerprint": base.fingerprint,
        "online_extension": None,
        "completed": True,
        "offline_updates_completed": 0,
        "online_steps_completed": base.online_steps,
    }
    (output / "run.json").write_text(
        json.dumps(run_metadata, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(train_module, "_validate_resume", lambda *_a, **_k: None)
    real_save = train_module.atomic_torch_save
    real_json = train_module._atomic_json

    if crash_point in {"latest", "best"}:
        def crashing_save(path: Path, value: dict[str, Any]) -> None:
            real_save(path, value)
            if path.name == f"{crash_point}.pt":
                raise RuntimeError(f"crash after {crash_point}")

        monkeypatch.setattr(train_module, "atomic_torch_save", crashing_save)
    else:
        def crashing_json(path: Path, value: dict[str, Any]) -> None:
            real_json(path, value)
            raise RuntimeError("crash after run")

        monkeypatch.setattr(train_module, "_atomic_json", crashing_json)

    with pytest.raises(RuntimeError, match=f"crash after {crash_point}"):
        train_module._prepare_online_extension(
            base_config=base,
            extended_online_steps=10_000,
            output=output,
            dataset=object(),  # validation is isolated by the monkeypatch above
            koopman=None,
            observation_normalizer=None,
            environment_protocol={},
        )

    monkeypatch.setattr(train_module, "atomic_torch_save", real_save)
    monkeypatch.setattr(train_module, "_atomic_json", real_json)
    target, lineage = train_module._prepare_online_extension(
        base_config=base,
        extended_online_steps=10_000,
        output=output,
        dataset=object(),
        koopman=None,
        observation_normalizer=None,
        environment_protocol={},
    )
    assert target.online_steps == 10_000
    artifacts = [
        load_checkpoint(output / "latest.pt"),
        load_checkpoint(output / "best.pt"),
        json.loads((output / "run.json").read_text(encoding="utf-8")),
    ]
    assert all(value["config"] == target.to_dict() for value in artifacts)
    assert all(value["config_fingerprint"] == target.fingerprint for value in artifacts)
    assert all(value["online_extension"] == lineage for value in artifacts)
    assert artifacts[-1]["completed"] is False

    # A retry after the final write is a byte-preserving no-op and retains the
    # timestamp from the first persisted migration artifact.
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    repeated_target, repeated_lineage = train_module._prepare_online_extension(
        base_config=base,
        extended_online_steps=10_000,
        output=output,
        dataset=object(),
        koopman=None,
        observation_normalizer=None,
        environment_protocol={},
    )
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert repeated_target == target
    assert repeated_lineage == lineage
    assert after == before


def test_fresh_step_zero_checkpoint_precedes_evaluation_and_retry_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Dataset:
        path = tmp_path / "dataset.npz"
        sha256 = "1" * 64
        metadata: dict[str, Any] = {}
        arrays = {
            "observation": np.asarray(
                [[0.0] * 5, [1.0] * 5], dtype=np.float32
            )
        }

    class ProtocolEnvironment:
        def protocol_metadata(self) -> dict[str, Any]:
            return {"task": "cartpole_swingup", "step_limit": 1}

        def close(self) -> None:
            return None

    class StepZeroCrash(RuntimeError):
        pass

    class StopAfterEvaluation(RuntimeError):
        pass

    monkeypatch.setattr(train_module.OfflineDataset, "load", lambda _path: Dataset())
    monkeypatch.setattr(
        train_module,
        "make_dmc_adapter",
        lambda _task, seed: ProtocolEnvironment(),
    )
    config = train_module.O2OConfig(
        method="RLPD-Raw",
        seed=17,
        device="cpu",
        batch_size=2,
        hidden_dim=8,
        critic_hidden_layers=1,
        critic_ensemble_size=2,
        target_critic_subset=2,
        online_steps=1,
        online_utd=1,
        online_warmup_steps=0,
        replay_capacity=8,
        num_envs=1,
        env_workers=1,
        eval_interval_online_steps=1,
        eval_episodes=1,
        checkpoint_interval_updates=1,
        log_interval_updates=1,
    )
    output = tmp_path / "fresh"
    monkeypatch.setattr(
        train_module,
        "evaluate",
        lambda *_a, **_k: (_ for _ in ()).throw(StepZeroCrash("step-zero crash")),
    )
    with pytest.raises(StepZeroCrash, match="step-zero crash"):
        train_module.run(config, Path("unused.npz"), None, output)
    checkpoint = load_checkpoint(output / "latest.pt")
    assert checkpoint["online_step"] == 0
    assert checkpoint["offline_update"] == 0
    assert not (output / "metrics.jsonl").exists()

    evaluation_calls = 0

    def successful_evaluation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal evaluation_calls
        evaluation_calls += 1
        return {
            "return_mean": 1.0,
            "return_std_population": 0.0,
            "return_min": 1.0,
            "return_max": 1.0,
            "episode_length_mean": 1.0,
            "returns": [1.0],
        }

    monkeypatch.setattr(train_module, "evaluate", successful_evaluation)
    monkeypatch.setattr(
        train_module,
        "make_dmc_vector_env",
        lambda *_a, **_k: (_ for _ in ()).throw(
            StopAfterEvaluation("stop after evaluation")
        ),
    )
    with pytest.raises(StopAfterEvaluation, match="stop after evaluation"):
        train_module.run(config, Path("unused.npz"), None, output)
    assert evaluation_calls == 1
    rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [(row["phase"], row["online_step"]) for row in rows] == [("initial", 0)]

    monkeypatch.setattr(
        train_module,
        "evaluate",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("step-zero evaluation was duplicated")
        ),
    )
    with pytest.raises(StopAfterEvaluation, match="stop after evaluation"):
        train_module.run(config, Path("unused.npz"), None, output)


def test_second_runner_on_flocked_root_fails_closed_without_spawning(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    spec.root.mkdir(parents=True, exist_ok=True)
    spawned = False

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal spawned
        spawned = True
        raise AssertionError("a competing runner must not spawn a child")

    with (spec.root / ".matrix.lock").open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(RuntimeError, match="holds the lock"):
                matrix.run_matrix(spec, popen_factory=forbidden)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    assert not spawned
    assert not (spec.root / "matrix_manifest.json").exists()
    assert not (spec.root / "matrix_status.json").exists()


def test_existing_mpve_run_requires_exact_sibling_path_and_sha_and_non_mpve_has_none(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    source = spec.offline_source(11).resolve()
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"paired immutable offline checkpoint")
    initialization = {
        "kind": "acmpc_o2o_offline_fork_v1",
        "source_path": str(source),
        "source_sha256": matrix._sha256_file(source),
        "source_method": matrix.MPVE_SOURCE_METHOD,
        "source_config_fingerprint": spec.config(
            matrix.MPVE_SOURCE_METHOD, 11
        ).fingerprint,
        "shared_state": "actor_critic_target_temperature_optimizers_rng",
    }
    run_dir, dataset, koopman = _write_completed_run_metadata(
        spec,
        method=matrix.MPVE_METHOD,
        initialization=initialization,
    )
    assert (
        matrix._check_run_identity(
            spec,
            seed=11,
            method=matrix.MPVE_METHOD,
            dataset=dataset,
            koopman=koopman,
        )
        == "completed"
    )

    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    other_source = tmp_path / "other" / "offline.pt"
    other_source.parent.mkdir(parents=True)
    other_source.write_bytes(source.read_bytes())
    metadata["initialization"]["source_path"] = str(other_source.resolve())
    (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="paired offline source"):
        matrix._check_run_identity(
            spec,
            seed=11,
            method=matrix.MPVE_METHOD,
            dataset=dataset,
            koopman=koopman,
        )

    metadata["initialization"]["source_path"] = str(source)
    metadata["initialization"]["source_sha256"] = "0" * 64
    (run_dir / "run.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="paired offline source"):
        matrix._check_run_identity(
            spec,
            seed=11,
            method=matrix.MPVE_METHOD,
            dataset=dataset,
            koopman=koopman,
        )

    _non_mpve_dir, dataset, koopman = _write_completed_run_metadata(
        spec,
        method="RLPD-Raw",
        initialization=initialization,
    )
    with pytest.raises(ValueError, match="Non-MPVE"):
        matrix._check_run_identity(
            spec,
            seed=11,
                method="RLPD-Raw",
            dataset=dataset,
            koopman=koopman,
        )


def test_child_receives_current_matrix_lock_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path)
    job = matrix.Job(
        job_id="lock-inheritance",
        stage="train",
        argv=("fake", "train"),
        log_path=tmp_path / "lock-inheritance.log",
        output_paths=(),
    )
    status = _status([job])
    captured: dict[str, Any] = {}

    class FakeProcess:
        pid = 301

    def fake_popen(_argv: list[str], **kwargs: Any) -> FakeProcess:
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(matrix, "_assert_source_unchanged", lambda *_a, **_k: None)
    with matrix._exclusive_matrix_lock(spec.root):
        current_fd = matrix._ACTIVE_MATRIX_LOCK_FD
        assert isinstance(current_fd, int)
        active = matrix._start_job(
            job,
            spec=spec,
            identity=_identity(),
            status=status,
            status_path=tmp_path / "status.json",
            popen_factory=fake_popen,
        )
        try:
            assert captured["pass_fds"] == (current_fd,)
        finally:
            active.log_handle.close()


def test_inherited_lock_survives_parent_context_until_real_child_exits(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    child: subprocess.Popen[str] | None = None
    contender = None
    try:
        with matrix._exclusive_matrix_lock(spec.root):
            lock_fd = matrix._ACTIVE_MATRIX_LOCK_FD
            assert isinstance(lock_fd, int)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                pass_fds=(lock_fd,),
                text=True,
            )
        contender = (spec.root / ".matrix.lock").open("a+", encoding="utf-8")
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

        child.terminate()
        child.wait(timeout=5)
        fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        if contender is not None:
            fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
            contender.close()


def test_mpve_starts_as_soon_as_offline_artifact_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, max_parallel=2)
    source = spec.offline_source(11)
    events: list[str] = []

    class FakeProcess:
        def __init__(self, name: str, pid: int) -> None:
            self.name = name
            self.pid = pid
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            if self.name == "source" and self.polls == 1:
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(b"atomic offline snapshot")
                events.append("offline-ready")
                return None
            if self.name == "source":
                events.append("source-finished")
            else:
                events.append("mpve-finished")
            return 0

    next_pid = iter((101, 102))

    def fake_popen(argv: list[str], **_kwargs: Any) -> FakeProcess:
        name = "mpve" if "MPVE" in " ".join(argv) else "source"
        events.append(f"start-{name}")
        return FakeProcess(name, next(next_pid))

    source_job = matrix.Job(
        job_id="source",
        stage="train",
        argv=("fake", "source"),
        log_path=tmp_path / "source.log",
        output_paths=(),
    )
    mpve_job = matrix.Job(
        job_id="mpve",
        stage="train",
        argv=("fake", "MPVE"),
        log_path=tmp_path / "mpve.log",
        output_paths=(),
        artifact_dependency=source,
    )
    jobs = [source_job, mpve_job]
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(matrix, "_assert_source_unchanged", lambda *_a, **_k: None)
    monkeypatch.setattr(matrix, "_validate_job_output", lambda *_a, **_k: None)
    matrix._run_training_jobs(
        jobs,
        spec=spec,
        dataset={},
        koopman={},
        identity=_identity(),
        status=_status(jobs),
        status_path=status_path,
        popen_factory=fake_popen,
        sleep_fn=lambda _seconds: None,
    )
    assert events.index("offline-ready") < events.index("start-mpve")
    assert events.index("start-mpve") < events.index("source-finished")


def test_failure_stops_dispatching_new_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(tmp_path, max_parallel=1)
    started: list[str] = []

    class FailedProcess:
        pid = 201

        def poll(self) -> int:
            return 7

    def fake_popen(argv: list[str], **_kwargs: Any) -> FailedProcess:
        started.append(argv[-1])
        return FailedProcess()

    jobs = [
        matrix.Job(
            job_id=name,
            stage="train",
            argv=("fake", name),
            log_path=tmp_path / f"{name}.log",
            output_paths=(),
        )
        for name in ("first", "never-started")
    ]
    status = _status(jobs)
    monkeypatch.setattr(matrix, "_assert_source_unchanged", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="return code 7"):
        matrix._run_training_jobs(
            jobs,
            spec=spec,
            dataset={},
            koopman={},
            identity=_identity(),
            status=status,
            status_path=tmp_path / "status.json",
            popen_factory=fake_popen,
            sleep_fn=lambda _seconds: None,
        )
    assert started == ["first"]
    assert status["jobs"]["never-started"]["state"] == "blocked"
