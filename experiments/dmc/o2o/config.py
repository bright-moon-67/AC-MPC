"""Resolved configuration and immutable identities for DMC O2O methods.

The representation boundary lives here rather than in string matching spread
through the trainer.  A ``*-Raw`` method is therefore structurally incapable
of requesting Koopman features; only AC-KMPC methods may do so.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from experiments.dmc.tasks.registry import get_task_spec


SUPPORTED_O2O_TASKS = frozenset({"cartpole_swingup", "walker_run"})


Representation = Literal["raw", "koopman_lifted"]
ActorQReduction = Literal["min", "mean"]
TemperatureObjective = Literal["calql_log_alpha", "rlpd"]
CriticHeadReduction = Literal["sum", "mean"]
OnlineCQLMode = Literal["all_valid_mc", "off"]
NetworkProfile = Literal["exorl_cql", "rlpd"]
MPVEScope = Literal["off", "offline_only", "online_only", "both"]
LearnerFamily = Literal["sac", "awac", "iql"]


@dataclass(frozen=True)
class MethodSpec:
    """Non-overridable algorithm/representation identity plus tuned defaults."""

    name: str
    representation: Representation
    actor: Literal["mlp", "ac_kmpc"]
    offline_pretraining: bool
    calql: bool
    offline_replay_online: bool
    mpve_scope: MPVEScope
    completed_online_returns: bool
    profile: str
    batch_size: int
    hidden_dim: int
    critic_hidden_layers: int
    controller_hidden_dim: int
    controller_hidden_layers: int
    critic_ensemble_size: int
    target_critic_subset: int
    target_tau: float
    actor_learning_rate: float
    critic_learning_rate: float
    temperature_learning_rate: float
    cql_actions: int
    network_profile: NetworkProfile
    backup_entropy: bool
    actor_q_reduction: ActorQReduction
    temperature_objective: TemperatureObjective
    target_entropy: float
    critic_head_reduction: CriticHeadReduction
    online_cql_mode: OnlineCQLMode
    calql_max_target_backup: bool
    online_utd: int
    online_warmup_steps: int
    num_envs: int
    env_workers: int
    learner_family: LearnerFamily = "sac"
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    advantage_weight_max: float = 100.0

    @property
    def requires_koopman(self) -> bool:
        return self.representation == "koopman_lifted"


_CALQL_RAW = dict(
    representation="raw",
    actor="mlp",
    offline_pretraining=True,
    calql=True,
    offline_replay_online=True,
    mpve_scope="off",
    completed_online_returns=True,
    profile="exorl_cql_backbone_calql_standard_single_tanh_v1",
    batch_size=1024,
    hidden_dim=1024,
    critic_hidden_layers=2,
    controller_hidden_dim=128,
    controller_hidden_layers=1,
    critic_ensemble_size=2,
    target_critic_subset=2,
    target_tau=0.01,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-4,
    temperature_learning_rate=1e-4,
    cql_actions=3,
    network_profile="exorl_cql",
    backup_entropy=False,
    actor_q_reduction="min",
    temperature_objective="calql_log_alpha",
    target_entropy=-1.0,
    critic_head_reduction="sum",
    online_cql_mode="all_valid_mc",
    # ExORL's task-matched DMC CQL backbone uses one next-policy action.
    # This deliberately differs from Cal-QL's repository default of max-over-K.
    calql_max_target_backup=False,
    online_utd=1,
    online_warmup_steps=0,
    num_envs=1,
    env_workers=1,
)
_RLPD_RAW = dict(
    representation="raw",
    actor="mlp",
    offline_pretraining=False,
    calql=False,
    offline_replay_online=True,
    mpve_scope="off",
    completed_online_returns=False,
    profile="rlpd_official_state_core_v1",
    batch_size=256,
    hidden_dim=256,
    critic_hidden_layers=2,
    controller_hidden_dim=128,
    controller_hidden_layers=1,
    critic_ensemble_size=10,
    target_critic_subset=2,
    target_tau=0.005,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temperature_learning_rate=3e-4,
    cql_actions=10,
    network_profile="rlpd",
    backup_entropy=True,
    actor_q_reduction="mean",
    temperature_objective="rlpd",
    target_entropy=-0.5,
    critic_head_reduction="mean",
    online_cql_mode="off",
    calql_max_target_backup=False,
    online_utd=20,
    online_warmup_steps=5_000,
    num_envs=5,
    env_workers=5,
)
_CALQL_AC_KMPC = {
    **_CALQL_RAW,
    "representation": "koopman_lifted",
    "actor": "ac_kmpc",
    "controller_hidden_dim": 1024,
    "controller_hidden_layers": 2,
    "profile": "exorl_cql_backbone_calql_ac_kmpc_lifted_v1",
}
_CAL_RLPD = dict(
    offline_pretraining=True,
    calql=True,
    offline_replay_online=True,
    completed_online_returns=False,
    profile="calql_regularized_rlpd_offline_then_rlpd_online_v1",
    batch_size=256,
    hidden_dim=256,
    critic_hidden_layers=2,
    controller_hidden_dim=128,
    controller_hidden_layers=1,
    critic_ensemble_size=10,
    target_critic_subset=2,
    target_tau=0.005,
    actor_learning_rate=3e-4,
    critic_learning_rate=3e-4,
    temperature_learning_rate=3e-4,
    cql_actions=10,
    network_profile="rlpd",
    backup_entropy=True,
    actor_q_reduction="mean",
    temperature_objective="rlpd",
    target_entropy=-0.5,
    critic_head_reduction="mean",
    online_cql_mode="off",
    # The hybrid retains RLPD's single-action REDQ target in both phases.
    calql_max_target_backup=False,
    online_utd=20,
    online_warmup_steps=0,
    num_envs=5,
    env_workers=5,
)

# Structured RLPD variants use the same controller capacity scale as the
# ordinary RLPD actor.  The controller remains structurally different (its
# output is the KMPC cost/plan parameterization), but its learned torso is
# two 256-wide layers rather than the wider Cal-QL 1024-wide profile.
_CAL_RLPD_KMPC_RLPD_WIDTH = {
    **{key: value for key, value in _CAL_RLPD.items() if key != "profile"},
    "controller_hidden_dim": 256,
    "controller_hidden_layers": 2,
}

METHOD_SPECS: dict[str, MethodSpec] = {
    "Cal-QL-Raw": MethodSpec(name="Cal-QL-Raw", **_CALQL_RAW),
    "Cal-QL-AC-KMPC": MethodSpec(
        name="Cal-QL-AC-KMPC", **_CALQL_AC_KMPC
    ),
    "RLPD-Raw": MethodSpec(name="RLPD-Raw", **_RLPD_RAW),
    "Cal-RLPD-Raw": MethodSpec(
        name="Cal-RLPD-Raw",
        representation="raw",
        actor="mlp",
        mpve_scope="off",
        **_CAL_RLPD,
    ),
    "Cal-RLPD-AC-KMPC": MethodSpec(
        name="Cal-RLPD-AC-KMPC",
        representation="koopman_lifted",
        actor="ac_kmpc",
        mpve_scope="off",
        **_CAL_RLPD,
    ),
    "Cal-RLPD-AC-KMPC-MPVE": MethodSpec(
        name="Cal-RLPD-AC-KMPC-MPVE",
        representation="koopman_lifted",
        actor="ac_kmpc",
        # MPVE is part of both phases: offline critic pretraining and online
        # value expansion use the same detached Koopman target construction.
        mpve_scope="both",
        **_CAL_RLPD,
    ),
}
# Standalone experiments are intentionally excluded from ``METHODS`` so the
# legacy matrix/aggregate contract remains unchanged.  They are launched and
# stopped independently and may have their own execution horizon.
STANDALONE_METHOD_SPECS: dict[str, MethodSpec] = {
    # Paper-facing names. The historical *-Raw identities above remain intact
    # so completed Cartpole/Walker matrices are still exactly reproducible.
    "Cal-QL": MethodSpec(name="Cal-QL", **_CALQL_RAW),
    "RLPD": MethodSpec(name="RLPD", **_RLPD_RAW),
    "Cal-RLPD": MethodSpec(
        name="Cal-RLPD",
        representation="raw",
        actor="mlp",
        mpve_scope="off",
        **_CAL_RLPD,
    ),
    "Cal-RLPD-Lift": MethodSpec(
        name="Cal-RLPD-Lift",
        representation="koopman_lifted",
        actor="mlp",
        mpve_scope="off",
        profile="calql_regularized_rlpd_lifted_mlp_v1",
        **{key: value for key, value in _CAL_RLPD.items() if key != "profile"},
    ),
    "AWAC": MethodSpec(
        name="AWAC",
        **{
            **_RLPD_RAW,
            "offline_pretraining": True,
            "profile": "awac_offline_to_online_v1",
            "critic_ensemble_size": 2,
            "target_critic_subset": 2,
            "backup_entropy": False,
            "actor_q_reduction": "min",
            "online_utd": 1,
            "online_warmup_steps": 0,
        },
        learner_family="awac",
        advantage_temperature=1.0,
    ),
    "IQL": MethodSpec(
        name="IQL",
        **{
            **_RLPD_RAW,
            "offline_pretraining": True,
            "profile": "iql_offline_to_online_v1",
            "critic_ensemble_size": 2,
            "target_critic_subset": 2,
            "backup_entropy": False,
            "actor_q_reduction": "min",
            "online_utd": 1,
            "online_warmup_steps": 0,
        },
        learner_family="iql",
        expectile=0.7,
        advantage_temperature=3.0,
    ),
    "Cal-RLPD-KMPC": MethodSpec(
        name="Cal-RLPD-KMPC",
        representation="koopman_lifted",
        actor="ac_kmpc",
        mpve_scope="off",
        profile="calql_regularized_rlpd_cal_kmpc_rlpd_width_v2",
        **_CAL_RLPD_KMPC_RLPD_WIDTH,
    ),
    "Cal-RLPD-MPVE": MethodSpec(
        name="Cal-RLPD-MPVE",
        representation="koopman_lifted",
        actor="ac_kmpc",
        mpve_scope="both",
        profile="calql_regularized_rlpd_cal_kmpc_mpve_rlpd_width_v2",
        **_CAL_RLPD_KMPC_RLPD_WIDTH,
    ),
    "Cal-QL-MPVE": MethodSpec(
        name="Cal-QL-MPVE",
        representation="koopman_lifted",
        actor="ac_kmpc",
        mpve_scope="both",
        profile="exorl_cql_backbone_calql_ac_kmpc_mpve_v1",
        **{
            key: value
            for key, value in _CALQL_AC_KMPC.items()
            if key not in ("profile", "representation", "actor", "mpve_scope")
        },
    ),
    "Cal-RLPD-AC-KMPC-Offline-MPVE": MethodSpec(
        name="Cal-RLPD-AC-KMPC-Offline-MPVE",
        representation="koopman_lifted",
        actor="ac_kmpc",
        # Standalone variant is also a complete offline-to-online MPVE
        # method; it is not a different actor, only a different MPVE scope.
        mpve_scope="both",
        profile="calql_regularized_rlpd_offline_mpve_auxiliary_v1",
        **{key: value for key, value in _CAL_RLPD.items() if key != "profile"},
    ),
}
METHODS = tuple(METHOD_SPECS)
TRAIN_METHOD_SPECS = {**METHOD_SPECS, **STANDALONE_METHOD_SPECS}
TRAIN_METHODS = tuple(TRAIN_METHOD_SPECS)


@dataclass(frozen=True)
class O2OConfig:
    task: str = "cartpole_swingup"
    method: str = "Cal-RLPD-Raw"
    seed: int = 20260821
    device: str = "cuda"

    # ``None`` means resolve from MethodSpec.  Concrete resolved values are
    # serialized, so a checkpoint records actual hyperparameters rather than
    # depending on future defaults.  Tests/smokes may explicitly use smaller
    # values without weakening the immutable representation identity.
    batch_size: int | None = None
    hidden_dim: int | None = None
    critic_hidden_layers: int | None = None
    controller_hidden_dim: int | None = None
    controller_hidden_layers: int | None = None
    critic_ensemble_size: int | None = None
    target_critic_subset: int | None = None
    discount: float = 0.99
    target_tau: float | None = None
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    temperature_learning_rate: float | None = None
    initial_temperature: float = 1.0
    target_entropy: float | None = None
    network_profile: NetworkProfile | None = None
    backup_entropy: bool | None = None
    actor_q_reduction: ActorQReduction | None = None
    temperature_objective: TemperatureObjective | None = None
    critic_head_reduction: CriticHeadReduction | None = None
    online_cql_mode: OnlineCQLMode | None = None
    calql_max_target_backup: bool | None = None
    gradient_clip_norm: float = 10.0

    # 500k matches the public ExORL optimization budget.  Cal-QL-Raw uses the
    # official-derived 2Q core; Cal-RLPD methods use calibrated pretraining
    # before their RLPD online phase.  RLPD-Raw ignores this field.
    offline_updates: int = 500_000
    cql_actions: int | None = None
    cql_temperature: float = 1.0
    cql_weight: float = 0.01

    # A shorter 50k CPU interaction budget is the initial protocol.  It can be
    # extended explicitly after inspecting fixed-seed evaluations.
    online_steps: int = 50_000
    online_utd: int | None = None
    offline_replay_ratio: float = 0.5
    online_warmup_steps: int | None = None
    replay_capacity: int = 200_000
    num_envs: int | None = None
    env_workers: int | None = None

    kmpc_horizon: int = 20
    kmpc_solver_iterations: int = 20
    mpve_total_horizon: int = 10
    mpve_loss_weight: float = 1.0

    eval_interval_online_steps: int = 2_500
    eval_episodes: int = 10
    checkpoint_interval_updates: int = 10_000
    log_interval_updates: int = 1_000
    offline_eval_interval_updates: int = 5_000

    def __post_init__(self) -> None:
        if self.method not in TRAIN_METHOD_SPECS:
            raise ValueError(f"Unknown method {self.method!r}; expected {TRAIN_METHODS}")
        spec = TRAIN_METHOD_SPECS[self.method]
        for name in (
            "batch_size",
            "hidden_dim",
            "critic_hidden_layers",
            "controller_hidden_dim",
            "controller_hidden_layers",
            "critic_ensemble_size",
            "target_critic_subset",
            "target_tau",
            "actor_learning_rate",
            "critic_learning_rate",
            "temperature_learning_rate",
            "cql_actions",
            "network_profile",
            "backup_entropy",
            "actor_q_reduction",
            "temperature_objective",
            "critic_head_reduction",
            "online_cql_mode",
            "calql_max_target_backup",
            "online_utd",
            "online_warmup_steps",
            "num_envs",
            "env_workers",
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, getattr(spec, name))
        # Target entropy follows the official task-dependent formula instead
        # of a frozen constant.  The ExORL DMC CQL backbone uses
        # ``-action_dim``; RLPD's SAC default when unspecified is
        # ``-action_dim / 2``.  Cartpole (action_dim=1) reproduces the frozen
        # values ``-1`` and ``-0.5`` exactly, so existing checkpoints are
        # unaffected while Walker (action_dim=6) resolves to ``-6`` / ``-3``.
        if self.target_entropy is None:
            if self.task not in SUPPORTED_O2O_TASKS:
                raise ValueError(
                    f"Unsupported O2O task {self.task!r}; expected "
                    f"{sorted(SUPPORTED_O2O_TASKS)}"
                )
            action_dim = get_task_spec(self.task).action_dim
            if spec.temperature_objective == "calql_log_alpha":
                resolved_entropy = -float(action_dim)
            elif spec.temperature_objective == "rlpd":
                resolved_entropy = -float(action_dim) / 2.0
            else:
                raise ValueError("Unknown temperature objective")
            object.__setattr__(self, "target_entropy", resolved_entropy)

    @property
    def method_spec(self) -> MethodSpec:
        return TRAIN_METHOD_SPECS[self.method]

    def validate(self) -> None:
        if self.task not in SUPPORTED_O2O_TASKS:
            raise ValueError(
                f"Unsupported O2O task {self.task!r}; expected "
                f"{sorted(SUPPORTED_O2O_TASKS)}"
            )
        if self.method not in TRAIN_METHODS:
            raise ValueError(f"Unknown method {self.method!r}; expected {TRAIN_METHODS}")
        integer_fields = (
            "batch_size", "hidden_dim", "critic_hidden_layers",
            "controller_hidden_dim", "controller_hidden_layers",
            "critic_ensemble_size", "target_critic_subset", "offline_updates",
            "cql_actions", "online_steps", "online_utd", "online_warmup_steps",
            "replay_capacity", "num_envs", "env_workers", "kmpc_horizon",
            "kmpc_solver_iterations",
            "mpve_total_horizon", "eval_interval_online_steps", "eval_episodes",
            "checkpoint_interval_updates", "log_interval_updates",
            "offline_eval_interval_updates",
        )
        for name in integer_fields:
            value = getattr(self, name)
            minimum = 0 if name == "online_warmup_steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        assert isinstance(self.target_critic_subset, int)
        assert isinstance(self.critic_ensemble_size, int)
        assert isinstance(self.env_workers, int)
        assert isinstance(self.num_envs, int)
        if self.target_critic_subset > self.critic_ensemble_size:
            raise ValueError("target_critic_subset cannot exceed critic ensemble size")
        if self.env_workers > self.num_envs:
            raise ValueError("env_workers cannot exceed num_envs")
        if self.requires_completed_online_returns and self.num_envs != 1:
            raise ValueError(
                "Methods with exact online MC returns require num_envs=1"
            )
        for name in ("online_steps", "online_warmup_steps", "eval_interval_online_steps"):
            if getattr(self, name) % self.num_envs:
                raise ValueError(f"{name} must be divisible by num_envs")
        if self.mpve_total_horizon > self.kmpc_horizon:
            raise ValueError("MPVE total horizon cannot exceed KMPC horizon")
        finite_positive = (
            "discount", "target_tau", "actor_learning_rate",
            "critic_learning_rate", "temperature_learning_rate",
            "initial_temperature", "gradient_clip_norm", "cql_temperature",
            "mpve_loss_weight",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must lie in (0, 1]")
        if not 0 < float(self.target_tau) <= 1:
            raise ValueError("target_tau must lie in (0, 1]")
        if not 0 <= self.offline_replay_ratio <= 1:
            raise ValueError("offline_replay_ratio must lie in [0, 1]")
        if not math.isfinite(self.cql_weight) or self.cql_weight < 0:
            raise ValueError("cql_weight must be finite and nonnegative")
        if not math.isfinite(self.target_entropy):
            raise ValueError("target_entropy must be finite")
        spec = self.method_spec
        if not 0.0 < spec.expectile < 1.0:
            raise ValueError("IQL expectile must lie in (0, 1)")
        if spec.advantage_temperature <= 0 or spec.advantage_weight_max <= 0:
            raise ValueError("Advantage weighting parameters must be positive")
        semantic_fields = (
            "network_profile",
            "backup_entropy",
            "actor_q_reduction",
            "temperature_objective",
            "critic_head_reduction",
            "online_cql_mode",
            "calql_max_target_backup",
        )
        for name in semantic_fields:
            if getattr(self, name) != getattr(spec, name):
                raise ValueError(f"{name} is fixed by the method identity")
        if self.actor_q_reduction not in ("min", "mean"):
            raise ValueError("actor_q_reduction must be min or mean")
        if self.temperature_objective not in ("calql_log_alpha", "rlpd"):
            raise ValueError("Unknown temperature objective")
        if self.critic_head_reduction not in ("sum", "mean"):
            raise ValueError("critic_head_reduction must be sum or mean")
        if self.online_cql_mode not in ("all_valid_mc", "off"):
            raise ValueError("Unknown online CQL mode")
        if self.network_profile not in ("exorl_cql", "rlpd"):
            raise ValueError("Unknown network profile")
        if self.online_cql_mode == "all_valid_mc" and (
            not spec.calql or not spec.completed_online_returns
        ):
            raise ValueError("Online Cal-QL requires completed Monte-Carlo returns")

    @property
    def representation(self) -> Representation:
        return self.method_spec.representation

    @property
    def requires_koopman(self) -> bool:
        return self.method_spec.requires_koopman

    @property
    def requires_completed_online_returns(self) -> bool:
        return self.method_spec.completed_online_returns

    @property
    def uses_offline_pretraining(self) -> bool:
        return self.method_spec.offline_pretraining

    @property
    def requires_own_offline_pretraining(self) -> bool:
        return self.uses_offline_pretraining and not self.requires_offline_fork

    @property
    def uses_offline_replay_online(self) -> bool:
        return self.method_spec.offline_replay_online

    @property
    def uses_calql(self) -> bool:
        return self.method_spec.calql

    @property
    def learner_family(self) -> LearnerFamily:
        return self.method_spec.learner_family

    def uses_calql_in_phase(self, phase: Literal["offline", "online"]) -> bool:
        if phase == "offline":
            return self.uses_calql
        if phase == "online":
            return self.uses_calql and self.online_cql_mode != "off"
        raise ValueError("phase must be exactly 'offline' or 'online'")

    def uses_calql_max_target_backup_in_phase(
        self, phase: Literal["offline", "online"]
    ) -> bool:
        return (
            self.uses_calql_in_phase(phase)
            and bool(self.calql_max_target_backup)
        )

    @property
    def uses_kmpc(self) -> bool:
        return self.method_spec.actor == "ac_kmpc"

    @property
    def uses_mpve(self) -> bool:
        return self.method_spec.mpve_scope != "off"

    @property
    def uses_offline_mpve(self) -> bool:
        return self.method_spec.mpve_scope in ("offline_only", "both")

    @property
    def uses_online_mpve(self) -> bool:
        return self.method_spec.mpve_scope in ("online_only", "both")

    @property
    def requires_offline_fork(self) -> bool:
        # A unified MPVE method must learn the offline critic with MPVE too;
        # it therefore owns its offline pretraining rather than forking a
        # non-MPVE AC-KMPC snapshot.  Legacy online-only fork checkpoints are
        # intentionally not accepted under the new method identity.
        return False

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return dataclasses.asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return sha256(payload).hexdigest()
