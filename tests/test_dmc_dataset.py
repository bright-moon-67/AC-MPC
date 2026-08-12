from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.dmc.collect import build_dmc_datasets as builder
from experiments.dmc.collect import collect_dmc_data as collector
from experiments.dmc.tasks import adapter as adapter_module


TASK_NAME = "cartpole_swingup"
SEED_DIR = collector.SEED_DIRS[0]


class FakeAdapter:
    step_limit = 4

    def __init__(self, task_name: str, seed: int = 0, **_kwargs) -> None:
        self.task_name = task_name
        self.seed = seed
        self._step = 0
        self._state = np.zeros(5, dtype=np.float32)

    def protocol_metadata(self) -> dict:
        return {
            "protocol_name": "dmc_native_v1",
            "protocol_schema_version": 1,
            "task": TASK_NAME,
            "domain": "cartpole",
            "dmc_task": "cartpole:swingup",
            "dm_control_version": "test",
            "mujoco_version": "test",
            "obs_dim": 5,
            "action_dim": 1,
            "control_dt": 0.01,
            "physics_dt": 0.01,
            "n_substeps": 1,
            "time_limit": 0.04,
            "step_limit": 4,
            "action_low": [-1.0],
            "action_high": [1.0],
            "obs_layout": [["position", 3], ["velocity", 2]],
        }

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed = int(seed)
        self._step = 0
        self._state = np.asarray(
            [self.seed + 0.25, -self.seed - 0.5, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        return self._state.copy()

    def get_state(self) -> np.ndarray:
        return self._state.copy()

    def step(self, requested_action: np.ndarray):
        requested = np.asarray(requested_action, dtype=np.float32).reshape(1)
        applied = np.clip(requested, -1.0, 1.0).astype(np.float32)
        self._step += 1
        next_state = self._state.copy()
        next_state[0] += 0.5 * float(applied[0]) + 0.125
        next_state[1] -= 0.25 * float(applied[0])
        next_state[2] = self._step
        next_state[3] = applied[0]
        self._state = next_state
        done = self._step == self.step_limit
        info = {
            "requested_action": requested.copy(),
            "applied_action": applied.copy(),
            "discount": 1.0,
        }
        return next_state.copy(), float(1.0 - abs(applied[0])), done, info

    def close(self) -> None:
        pass


@pytest.fixture
def fake_dmc(monkeypatch):
    spec = SimpleNamespace(obs_dim=5, action_dim=1, k_step=2)
    monkeypatch.setattr(collector, "get_task_spec", lambda _name: spec)
    monkeypatch.setattr(builder, "get_task_spec", lambda _name: spec)
    monkeypatch.setattr(
        adapter_module,
        "make_dmc_adapter",
        lambda task_name, seed=0, **kwargs: FakeAdapter(
            task_name, seed=seed, **kwargs
        ),
    )
    return spec


def _collect(root: Path, transitions: int, **kwargs):
    return collector.collect(
        TASK_NAME,
        0,
        transitions_per_stage=transitions,
        output_root=root,
        chunk_flush_every=4,
        **kwargs,
    )


def _read_chunks(root: Path) -> list[dict[str, np.ndarray]]:
    paths = sorted((root / TASK_NAME / SEED_DIR).glob("coverage_*.npz"))
    result = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            result.append({name: archive[name] for name in archive.files})
    return result


def _rewrite_chunk(path: Path, transform) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    transform(payload)
    np.savez_compressed(path, **payload)


def _primary_resolved_contract() -> tuple[object, int, int]:
    experiment = builder.load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    execution = builder.resolve_execution_spec(experiment, "development")
    total_updates = int(execution["data"]["collection_total_updates"])
    max_transitions = int(execution["data"]["max_transitions_per_train_seed"])
    return experiment, max_transitions, total_updates


def _stage_for_update(update: int, total_updates: int) -> str:
    if update <= total_updates // 3:
        return "early"
    if update <= 2 * total_updates // 3:
        return "mid"
    return "late"


def _promote_chunks_to_primary(
    root: Path,
    *,
    strategy: str = "test_named_stage_selection_v99",
) -> tuple[object, int, int]:
    experiment, max_transitions, total_updates = _primary_resolved_contract()
    paths = sorted((root / TASK_NAME / SEED_DIR).glob("coverage_*.npz"))
    assert paths
    for index, path in enumerate(paths):
        completion_update = 1 + index * total_updates // len(paths)

        def add_primary_contract(payload: dict[str, np.ndarray]) -> None:
            count = len(payload["state"])
            stage = _stage_for_update(completion_update, total_updates)
            payload.update(
                collection_schema_version=np.asarray(4, dtype=np.int64),
                collection_stage=np.full(count, stage),
                collection_selection_strategy=np.asarray(strategy),
                collection_max_transitions=np.asarray(
                    max_transitions, dtype=np.int64
                ),
                collection_total_updates=np.asarray(
                    total_updates, dtype=np.int64
                ),
                update=np.full(count, completion_update, dtype=np.int64),
                training_seed=np.asarray(20260811, dtype=np.int64),
                actor_type=np.asarray("PPO"),
                training_approved=np.asarray(True, dtype=np.bool_),
                config_fingerprint=np.asarray(experiment.fingerprint),
                approval_profile=np.asarray("development"),
                approval_file_sha256=np.asarray("2" * 64),
                preflight_report_sha256=np.asarray("3" * 64),
                authorization_kind=np.asarray("dmc_training_approval_v1"),
                train_seed_index=np.asarray(0, dtype=np.int64),
            )

        _rewrite_chunk(path, add_primary_contract)
    return experiment, max_transitions, total_updates


def _formal_primary_build_config(
    root: Path, output: Path
) -> builder.BuildConfig:
    experiment, max_transitions, total_updates = _primary_resolved_contract()
    return builder.BuildConfig(
        task_name=TASK_NAME,
        collect_root=root / TASK_NAME,
        output=output,
        seed_dirs=(SEED_DIR,),
        source="ppo_training_stages",
        expected_config_fingerprint=experiment.fingerprint,
        expected_approval_profile="development",
        expected_training_seeds=(20260811,),
        expected_collection_max_transitions=max_transitions,
        expected_collection_total_updates=total_updates,
    )


@pytest.mark.parametrize(
    ("profile", "expected_seed_dirs"),
    [
        ("development", ("seed_20260812",)),
        (
            "benchmark",
            ("seed_20260812", "seed_20260811", "seed_20260813"),
        ),
    ],
)
def test_primary_builder_cli_derives_task_source_and_seeds_from_yaml(
    profile: str, expected_seed_dirs: tuple[str, ...]
) -> None:
    args = builder.parse_args(
        [
            "--config",
            "experiments/dmc/configs/cartpole_swingup.yaml",
            "--profile",
            profile,
        ]
    )
    config = builder.build_config_from_args(args)
    assert config.task_name == "cartpole_swingup"
    assert config.source == "ppo_training_stages"
    assert config.seed_dirs == expected_seed_dirs
    assert config.collect_root == Path(
        f"runs/dmc/data/cartpole_swingup/{profile}"
    )
    assert config.output == config.collect_root / "cartpole_swingup_koopman.npz"
    experiment = builder.load_experiment_config(
        "experiments/dmc/configs/cartpole_swingup.yaml"
    )
    assert config.expected_config_fingerprint == experiment.fingerprint
    assert config.expected_approval_profile == profile
    assert config.expected_training_seeds == tuple(
        int(value.removeprefix("seed_")) for value in expected_seed_dirs
    )
    execution = builder.resolve_execution_spec(experiment, profile)
    assert config.expected_collection_max_transitions == int(
        execution["data"]["max_transitions_per_train_seed"]
    )
    assert config.expected_collection_total_updates == int(
        execution["data"]["collection_total_updates"]
    )


def test_primary_builder_cli_rejects_identity_overrides() -> None:
    base = [
        "--config",
        "experiments/dmc/configs/cartpole_swingup.yaml",
        "--profile",
        "development",
    ]
    with pytest.raises(SystemExit):
        builder.parse_args([*base, "--seed-dirs", "seed_20240201"])
    with pytest.raises(SystemExit):
        builder.parse_args(
            [
                "--task",
                TASK_NAME,
                "--source",
                "ppo_training_stages",
                "--seed-dirs",
                SEED_DIR,
            ]
        )


def test_standalone_builder_cli_remains_explicit() -> None:
    args = builder.parse_args(
        [
            "--task",
            TASK_NAME,
            "--source",
            "standalone_ablation",
            "--seed-dirs",
            "seed_a,seed_b",
        ]
    )
    config = builder.build_config_from_args(args)
    assert config.source == "standalone_ablation"
    assert config.seed_dirs == ("seed_a", "seed_b")


def test_formal_primary_build_rejects_yaml_lineage_mismatch(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    _promote_chunks_to_primary(root)
    valid = _formal_primary_build_config(root, tmp_path / "valid.npz")
    assert builder.build(valid).is_file()
    with pytest.raises(ValueError, match="config fingerprint"):
        builder.build(
            replace(
                valid,
                output=tmp_path / "wrong_config.npz",
                expected_config_fingerprint="sha256:" + "9" * 64,
            )
        )
    with pytest.raises(ValueError, match="approval profile"):
        builder.build(
            replace(
                valid,
                output=tmp_path / "wrong_profile.npz",
                expected_approval_profile="benchmark",
            )
        )
    with pytest.raises(ValueError, match="training seeds"):
        builder.build(
            replace(
                valid,
                output=tmp_path / "wrong_seed.npz",
                expected_training_seeds=(20260812,),
            )
        )


def test_primary_builder_persists_stage_selection_contract_and_updates(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    _experiment, max_transitions, total_updates = _promote_chunks_to_primary(
        root
    )
    output = tmp_path / "primary_dataset.npz"
    builder.build(_formal_primary_build_config(root, output))

    with np.load(output, allow_pickle=False) as archive:
        assert archive["dataset_schema_version"].item() == 4
        assert archive["collection_schema_version"].item() == 4
        assert archive["collection_selection_strategy"].item() == (
            "test_named_stage_selection_v99"
        )
        assert archive["collection_max_transitions"].item() == max_transitions
        assert archive["collection_total_updates"].item() == total_updates
        assert archive["collection_stage_names"].tolist() == [
            "early",
            "mid",
            "late",
        ]
        assert archive["collection_stage_update_ranges"].tolist() == [
            [1, total_updates // 3],
            [total_updates // 3 + 1, 2 * total_updates // 3],
            [2 * total_updates // 3 + 1, total_updates],
        ]
        contract = json.loads(
            archive["collection_selection_contract_json"].item()
        )
        assert contract["episode_stage_basis"] == "completion_update"
        assert contract["max_transitions_per_train_seed"] == max_transitions
        assert archive["collection_stage"].shape == archive["update"].shape
        assert archive["episode_collection_stage"].shape == (
            len(archive["episode_table_ids"]),
        )
        assert archive["source_collection_transition_counts"].tolist() == [40]
        assert sum(
            archive["source_collection_stage_transition_counts"][0].tolist()
        ) == 40

        # The new stage provenance augments, rather than replaces, the
        # transition-level PPO update lineage used by Koopman window building.
        for episode_id in archive["episode_table_ids"]:
            mask = archive["episode_id"] == episode_id
            stage = archive["episode_collection_stage"][episode_id]
            assert set(archive["collection_stage"][mask].tolist()) == {stage}
            completion_update = int(archive["update"][mask][-1])
            assert archive["episode_completion_update"][episode_id] == (
                completion_update
            )
            assert stage == _stage_for_update(completion_update, total_updates)


@pytest.mark.parametrize(
    "missing_field",
    [
        "collection_stage",
        "collection_selection_strategy",
        "collection_max_transitions",
        "collection_total_updates",
    ],
)
def test_primary_builder_requires_complete_v4_collection_contract(
    tmp_path: Path, fake_dmc, missing_field: str
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    _promote_chunks_to_primary(root)
    path = root / TASK_NAME / SEED_DIR / "coverage_000000.npz"
    _rewrite_chunk(path, lambda payload: payload.pop(missing_field))

    with pytest.raises(KeyError, match="missing fields"):
        builder.build(_formal_primary_build_config(root, tmp_path / "bad.npz"))


def test_primary_builder_rejects_stage_not_matching_completion_update(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    _promote_chunks_to_primary(root)
    path = root / TASK_NAME / SEED_DIR / "coverage_000000.npz"
    _rewrite_chunk(
        path,
        lambda payload: payload.__setitem__(
            "collection_stage", np.full(len(payload["state"]), "late")
        ),
    )

    with pytest.raises(ValueError, match="does not match completion update"):
        builder.build(_formal_primary_build_config(root, tmp_path / "bad.npz"))


def test_primary_builder_rejects_episode_with_mixed_stage_labels(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    _promote_chunks_to_primary(root)
    path = root / TASK_NAME / SEED_DIR / "coverage_000000.npz"

    def mix_episode_stage(payload: dict[str, np.ndarray]) -> None:
        payload["collection_stage"][0] = "mid"

    _rewrite_chunk(path, mix_episode_stage)
    with pytest.raises(ValueError, match="mixes collection stages"):
        builder.build(_formal_primary_build_config(root, tmp_path / "bad.npz"))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("collection_selection_strategy", np.asarray("another_named_strategy")),
        ("collection_max_transitions", np.asarray(300001, dtype=np.int64)),
        ("collection_total_updates", np.asarray(489, dtype=np.int64)),
    ],
)
def test_primary_builder_rejects_mixed_contract_across_chunks(
    tmp_path: Path,
    fake_dmc,
    field: str,
    replacement: np.ndarray,
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    _promote_chunks_to_primary(root)
    path = root / TASK_NAME / SEED_DIR / "coverage_000001.npz"
    _rewrite_chunk(
        path,
        lambda payload: payload.__setitem__(field, replacement),
    )

    with pytest.raises(ValueError, match="mixes collection selection contracts"):
        builder.build(_formal_primary_build_config(root, tmp_path / "bad.npz"))


@pytest.mark.parametrize(
    ("field", "delta", "message"),
    [
        ("collection_max_transitions", 1, "transition cap"),
        ("collection_total_updates", 3, "total updates"),
    ],
)
def test_primary_builder_rejects_contract_budget_not_matching_yaml(
    tmp_path: Path,
    fake_dmc,
    field: str,
    delta: int,
    message: str,
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    _promote_chunks_to_primary(root)
    for path in (root / TASK_NAME / SEED_DIR).glob("coverage_*.npz"):
        _rewrite_chunk(
            path,
            lambda payload: payload.__setitem__(
                field,
                np.asarray(int(payload[field].item()) + delta, dtype=np.int64),
            ),
        )

    with pytest.raises(ValueError, match=message):
        builder.build(_formal_primary_build_config(root, tmp_path / "bad.npz"))


@pytest.mark.parametrize(
    ("field", "invalid_value", "error", "message"),
    [
        (
            "collection_selection_strategy",
            np.asarray(["named_strategy"]),
            ValueError,
            "must be scalar",
        ),
        (
            "collection_selection_strategy",
            np.asarray(""),
            ValueError,
            "invalid collection_selection_strategy",
        ),
        (
            "collection_max_transitions",
            np.asarray(300000.0),
            TypeError,
            "integer scalar",
        ),
        (
            "collection_max_transitions",
            np.asarray(0, dtype=np.int64),
            ValueError,
            "positive",
        ),
        (
            "collection_total_updates",
            np.asarray(True, dtype=np.bool_),
            TypeError,
            "integer scalar",
        ),
        (
            "collection_total_updates",
            np.asarray(0, dtype=np.int64),
            ValueError,
            "at least 3",
        ),
    ],
)
def test_primary_builder_validates_collection_contract_scalars(
    tmp_path: Path,
    fake_dmc,
    field: str,
    invalid_value: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    _promote_chunks_to_primary(root)
    path = root / TASK_NAME / SEED_DIR / "coverage_000000.npz"
    _rewrite_chunk(
        path,
        lambda payload: payload.__setitem__(field, invalid_value),
    )

    with pytest.raises(error, match=message):
        builder.build(_formal_primary_build_config(root, tmp_path / "bad.npz"))


def test_expert_arguments_are_validated_before_output_creation(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "never-created"
    with pytest.raises(ValueError, match="requires expert_checkpoint"):
        collector.collect(
            TASK_NAME,
            0,
            transitions_per_stage=4,
            output_root=output_root,
            random_policy=False,
            expert_policy=True,
            expert_checkpoint=None,
        )
    assert not output_root.exists()


def test_collector_resume_does_not_double_count_or_repeat_ids(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    first = _collect(root, 8)
    assert first["total_transitions"] == 8
    assert first["stage_transitions"]["random"] == 8
    first_files = sorted((root / TASK_NAME / SEED_DIR).glob("coverage_*.npz"))

    resumed = _collect(root, 8)
    assert resumed["total_transitions"] == 8
    assert sorted((root / TASK_NAME / SEED_DIR).glob("coverage_*.npz")) == first_files

    extended = _collect(root, 12)
    assert extended["total_transitions"] == 12
    chunks = _read_chunks(root)
    episode_ids = np.concatenate([chunk["episode_id"] for chunk in chunks])
    global_steps = np.concatenate([chunk["global_step"] for chunk in chunks])
    assert len(np.unique(episode_ids)) == 3
    assert len(np.unique(global_steps)) == len(global_steps) == 12
    status_path = root / TASK_NAME / SEED_DIR / "collection_status.json"
    status = json.loads(status_path.read_text())
    assert status["total_transitions"] == 12
    marker = json.loads(
        (root / TASK_NAME / SEED_DIR / "random_done.txt").read_text()
    )
    assert marker["durable_transitions"] == 12
    assert marker["target_transitions"] == 12


def test_builder_persists_protocol_and_episode_disjoint_splits(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    output = tmp_path / "dataset.npz"
    builder.build(
        builder.BuildConfig(
            task_name=TASK_NAME,
            collect_root=root / TASK_NAME,
            output=output,
            seed_dirs=(SEED_DIR,),
            source="standalone_ablation",
        )
    )
    with np.load(output, allow_pickle=False) as archive:
        assert archive["dataset_schema_version"].item() == 3
        assert "collection_stage" not in archive.files
        assert "episode_collection_stage" not in archive.files
        assert "collection_selection_contract_json" not in archive.files
        assert archive["protocol_name"].item() == "dmc_native_v1"
        assert archive["control_dt"].item() == pytest.approx(0.01)
        assert archive["step_limit"].item() == 4
        assert archive["dm_control_version"].item() == "test"
        assert archive["requested_action"].shape == archive["action"].shape
        assert archive["discount"].shape == archive["reward"].shape
        train = set(archive["train_episode_ids"].tolist())
        validation = set(archive["validation_episode_ids"].tolist())
        test = set(archive["test_episode_ids"].tolist())
        assert not train & validation
        assert not train & test
        assert not validation & test
        assert train | validation | test == set(
            archive["episode_table_ids"].tolist()
        )
        hashes = archive["episode_trajectory_sha256"].tolist()
        assert len(hashes) == len(set(hashes)) == 10


def test_builder_rejects_duplicate_episode_id(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 4)
    seed_root = root / TASK_NAME / SEED_DIR
    source = seed_root / "coverage_000000.npz"
    duplicate = seed_root / "coverage_000001.npz"
    duplicate.write_bytes(source.read_bytes())
    _rewrite_chunk(
        duplicate,
        lambda payload: payload.__setitem__(
            "global_step", payload["global_step"] + 100
        ),
    )
    with pytest.raises(ValueError, match="Duplicate episode_id"):
        builder.build(
            builder.BuildConfig(
                task_name=TASK_NAME,
                collect_root=root / TASK_NAME,
                output=tmp_path / "dataset.npz",
                seed_dirs=(SEED_DIR,),
                source="standalone_ablation",
            )
        )


def test_builder_rejects_duplicate_trajectory_under_new_id(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 4)
    seed_root = root / TASK_NAME / SEED_DIR
    source = seed_root / "coverage_000000.npz"
    duplicate = seed_root / "coverage_000001.npz"
    duplicate.write_bytes(source.read_bytes())

    def make_ids_look_new(payload: dict[str, np.ndarray]) -> None:
        payload["episode_id"] = payload["episode_id"] + 100
        payload["global_step"] = payload["global_step"] + 100
        payload["reset_seed"] = payload["reset_seed"] + 100

    _rewrite_chunk(duplicate, make_ids_look_new)
    with pytest.raises(ValueError, match="Duplicate trajectory"):
        builder.build(
            builder.BuildConfig(
                task_name=TASK_NAME,
                collect_root=root / TASK_NAME,
                output=tmp_path / "dataset.npz",
                seed_dirs=(SEED_DIR,),
                source="standalone_ablation",
            )
        )


def test_builder_requires_every_configured_seed_directory(
    tmp_path: Path, fake_dmc
) -> None:
    with pytest.raises(FileNotFoundError, match="Required seed directory"):
        builder.build(
            builder.BuildConfig(
                task_name=TASK_NAME,
                collect_root=tmp_path / "missing",
                output=tmp_path / "dataset.npz",
                seed_dirs=(SEED_DIR,),
                source="standalone_ablation",
            )
        )


def test_builder_accepts_vector_stride_and_episode_crossing_ppo_updates(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    for path in (root / TASK_NAME / SEED_DIR).glob("coverage_*.npz"):
        def vectorize(payload: dict[str, np.ndarray]) -> None:
            payload["global_step"] = payload["global_step"] * 8
            steps = payload["step_index"]
            payload["update"] = np.where(steps < 2, 1, 2).astype(np.int64)

        _rewrite_chunk(path, vectorize)
    output = tmp_path / "dataset.npz"
    builder.build(
        builder.BuildConfig(
            task_name=TASK_NAME,
            collect_root=root / TASK_NAME,
            output=output,
            seed_dirs=(SEED_DIR,),
            source="standalone_ablation",
        )
    )
    with np.load(output, allow_pickle=False) as archive:
        assert set(np.unique(archive["update"])) == {1, 2}
        assert np.all(archive["global_step"] % 8 == 0)


def test_builder_rejects_inconsistent_terminal_flags(
    tmp_path: Path, fake_dmc
) -> None:
    root = tmp_path / "data"
    _collect(root, 40)
    path = root / TASK_NAME / SEED_DIR / "coverage_000000.npz"

    def corrupt(payload: dict[str, np.ndarray]) -> None:
        payload["truncated"][-1] = False

    _rewrite_chunk(path, corrupt)
    with pytest.raises(ValueError, match="done/terminal"):
        builder.build(
            builder.BuildConfig(
                task_name=TASK_NAME,
                collect_root=root / TASK_NAME,
                output=tmp_path / "dataset.npz",
                seed_dirs=(SEED_DIR,),
                source="standalone_ablation",
            )
        )
