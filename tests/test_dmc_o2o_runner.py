from __future__ import annotations

import fcntl
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from experiments.dmc.o2o import run_matrix as matrix


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
        online_steps=10,
        online_utd=2,
        num_envs=5,
        env_workers=5,
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
    koopman = matrix._expected_koopman(spec)
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


def test_parse_seeds_and_plan_has_exact_mpve_fork(tmp_path: Path) -> None:
    assert matrix.parse_seeds("1, 2,3") == (1, 2, 3)
    with pytest.raises(Exception, match="duplicates"):
        matrix.parse_seeds("1,1")

    spec = _spec(tmp_path)
    dataset = matrix._expected_dataset(spec)
    koopman = matrix._expected_koopman(spec)
    jobs = matrix.build_jobs(spec, dataset=dataset, koopman=koopman)
    training = [job for job in jobs if job.stage == "train"]
    assert [job.method for job in training] == list(matrix.MATRIX_METHODS)
    mpve = training[-1]
    source = spec.offline_source(11)
    assert mpve.artifact_dependency == source
    index = mpve.argv.index("--initialize-from-offline")
    assert mpve.argv[index + 1] == str(source)
    evaluations = [job for job in jobs if job.stage == "evaluate"]
    assert len(evaluations) == 5
    assert all(job.depends_on for job in evaluations)
    assert jobs[-2].stage == "aggregate"
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
    assert len(manifest["jobs"]) == 12
    assert status["state"] == "dry_run"
    assert {job["state"] for job in status["jobs"].values()} == {"planned"}


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
        method="RLPD-MLP",
        initialization=initialization,
    )
    with pytest.raises(ValueError, match="Non-MPVE"):
        matrix._check_run_identity(
            spec,
            seed=11,
            method="RLPD-MLP",
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
