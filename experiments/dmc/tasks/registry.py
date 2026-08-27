"""DMC task registry for the AC-MPC benchmark ladder.

Verified specs (obs layout, action dim, control/physics timestep, episode
length) were measured on the installed dm_control 1.0.44 (2026-08-08) by
instantiating every task and reading the live environment.  The adapter
re-reads runtime values from the live env, so the framework never drifts from
the installed dm_control version (see ``measure_task_spec`` / adapter
validation).

Native protocol in dm_control 1.0.44:

  * cartpole / reacher have no explicit ``_CONTROL_TIMESTEP`` in the suite, so
    the env runs at the model physics timestep (cartpole 0.01 s = 100 Hz,
    reacher 0.02 s = 50 Hz).
  * hopper / walker / humanoid set explicit suite control timesteps (0.02 /
    0.025 / 0.025 s) and are unaffected by the version change.
  * All five tasks have a 1000-control-step episode at their native rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


DMC_NATIVE_PROTOCOL = "dmc_native_v1"
DMC_CUSTOM_PROTOCOL = "dmc_custom_v1"

# ---------------------------------------------------------------------------
# Reward probes: reproduce the official per-component reward decomposition for
# closed-loop diagnostics.  The authoritative step reward is read from the env
# itself (``TimeStep.reward``); these probes only split it into parts.
# ---------------------------------------------------------------------------


def _probe_cartpole(physics) -> dict[str, float]:
    from dm_control.utils import rewards as dcr

    upright = (physics.pole_angle_cosine() + 1) / 2
    centered = dcr.tolerance(physics.cart_position(), margin=2)
    centered = (1 + centered) / 2
    small_control = dcr.tolerance(
        physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
    )[0]
    small_control = (4 + small_control) / 5
    small_velocity = dcr.tolerance(physics.angular_vel(), margin=5).min()
    small_velocity = (1 + small_velocity) / 2
    reward = upright.mean() * small_control * small_velocity * centered
    return {
        "upright": float(upright.mean()),
        "centered": float(centered),
        "small_control": float(small_control),
        "small_velocity": float(small_velocity),
        "reward": float(reward),
    }


def _probe_reacher(physics) -> dict[str, float]:
    from dm_control.utils import rewards as dcr

    radii = float(
        physics.named.model.geom_size[["target", "finger"], 0].sum()
    )
    dist = float(physics.finger_to_target_dist())
    reward = float(dcr.tolerance(dist, (0, radii)))
    return {"dist": dist, "radii": radii, "reward": reward}


def _probe_hopper(physics) -> dict[str, float]:
    from dm_control.utils import rewards as dcr

    height = float(physics.height())
    speed = float(physics.speed())
    standing = float(dcr.tolerance(height, (0.6, 2.0)))
    hopping = float(
        dcr.tolerance(
            speed,
            bounds=(2.0, float("inf")),
            margin=1.0,
            value_at_margin=0.5,
            sigmoid="linear",
        )
    )
    return {
        "height": height,
        "speed": speed,
        "standing": standing,
        "hopping": hopping,
        "reward": standing * hopping,
    }


def _probe_walker(physics) -> dict[str, float]:
    from dm_control.utils import rewards as dcr

    height = float(physics.torso_height())
    upright = (1 + float(physics.torso_upright())) / 2
    standing = float(
        dcr.tolerance(height, bounds=(1.2, float("inf")), margin=0.6)
    )
    stand_reward = (3 * standing + upright) / 4
    speed = float(physics.horizontal_velocity())
    move = float(
        dcr.tolerance(
            speed,
            bounds=(8.0, float("inf")),
            margin=4.0,
            value_at_margin=0.5,
            sigmoid="linear",
        )
    )
    reward = stand_reward * (5 * move + 1) / 6
    return {
        "height": height,
        "upright": upright,
        "standing": standing,
        "speed": speed,
        "move": move,
        "reward": reward,
    }


def _probe_humanoid(physics) -> dict[str, float]:
    from dm_control.utils import rewards as dcr

    standing = float(
        dcr.tolerance(
            physics.head_height(),
            bounds=(1.4, float("inf")),
            margin=1.4 / 4,
        )
    )
    upright = float(
        dcr.tolerance(
            physics.torso_upright(),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
    )
    stand_reward = standing * upright
    small_control = float(
        dcr.tolerance(
            physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
        ).mean()
    )
    small_control = (4 + small_control) / 5
    com_velocity = float(
        np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
    )
    move = float(
        dcr.tolerance(
            com_velocity,
            bounds=(10.0, float("inf")),
            margin=10.0,
            value_at_margin=0,
            sigmoid="linear",
        )
    )
    reward = small_control * stand_reward * (5 * move + 1) / 6
    return {
        "head_height": float(physics.head_height()),
        "standing": standing,
        "upright": upright,
        "small_control": small_control,
        "com_velocity": com_velocity,
        "move": move,
        "reward": reward,
    }


# ---------------------------------------------------------------------------
# Env factories: build the dm_control env directly (suite.load cannot override
# control_timestep for tasks that pass it explicitly, so we construct the
# Environment ourselves with full control over timestep / time limit).
# ---------------------------------------------------------------------------


def _make_env(module_name, task_builder, native_control_dt, default_time_limit):
    def factory(random, control_timestep, time_limit):
        import importlib

        from dm_control.rl import control

        module = importlib.import_module(f"dm_control.suite.{module_name}")
        # Use the suite module's Physics subclass: the task's get_observation
        # calls helper methods (e.g. bounded_position, height, speed) that only
        # exist on the subclass, not on the base mujoco.Physics.
        physics = module.Physics.from_xml_string(*module.get_model_and_assets())
        task = task_builder(module, random)
        effective_dt = (
            float(native_control_dt)
            if control_timestep is None
            else float(control_timestep)
        )
        physics_dt = float(physics.timestep())
        if not np.isfinite(effective_dt) or effective_dt <= 0:
            raise ValueError("control_timestep must be finite and positive")
        substeps = effective_dt / physics_dt
        if not np.isclose(substeps, round(substeps), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Control timestep ({effective_dt}) must be an integer multiple "
                f"of physics timestep ({physics_dt}) for {module_name}. Use the "
                "native DMC protocol unless an explicit comparison protocol has "
                "been defined."
            )
        effective_time_limit = (
            float(default_time_limit) if time_limit is None else float(time_limit)
        )
        if not np.isfinite(effective_time_limit) or effective_time_limit <= 0:
            raise ValueError("time_limit must be finite and positive")
        return control.Environment(
            physics,
            task,
            time_limit=effective_time_limit,
            control_timestep=effective_dt,
        )

    return factory


@dataclass(frozen=True)
class TaskSpec:
    name: str
    domain: str
    task: str
    obs_layout: tuple[tuple[str, int], ...]  # (obs key, dim) in canonical order
    action_dim: int
    native_control_dt: float  # measured on dm_control 1.0.44
    native_physics_dt: float
    native_time_limit: float
    report_groups: tuple[tuple[str, tuple[int, int]], ...]  # (name, (lo, hi))
    reward_probe: Callable[[object], dict[str, float]]
    env_factory: Callable[[int, Optional[float], Optional[float]], object]
    # Koopman training defaults (per-task)
    lift_dim: int = 32
    k_step: int = 20
    hidden_dims: tuple[int, int] = (256, 256)
    description: str = ""

    @property
    def obs_dim(self) -> int:
        return sum(dim for _, dim in self.obs_layout)

    @property
    def native_step_limit(self) -> int:
        return round(self.native_time_limit / self.native_control_dt)


def _cartpole_task(module, random):
    return module.Balance(swing_up=True, sparse=False, random=random)


def _reacher_task(module, random):
    return module.Reacher(target_size=0.015, random=random)


def _hopper_task(module, random):
    return module.Hopper(hopping=True, random=random)


def _hopper_stand_task(module, random):
    return module.Hopper(hopping=False, random=random)


def _walker_task(module, random):
    return module.PlanarWalker(move_speed=8.0, random=random)


def _humanoid_task(module, random):
    return module.Humanoid(move_speed=10.0, pure_state=False, random=random)


def _humanoid_pure_state_task(module, random):
    return module.Humanoid(move_speed=10.0, pure_state=True, random=random)


TASK_SPECS: dict[str, TaskSpec] = {
    "cartpole_swingup": TaskSpec(
        name="cartpole_swingup",
        domain="cartpole",
        task="swingup",
        obs_layout=(("position", 3), ("velocity", 2)),
        action_dim=1,
        native_control_dt=0.01,
        native_physics_dt=0.01,
        native_time_limit=10.0,
        report_groups=(
            ("position", (0, 3)),
            ("velocity", (3, 5)),
            ("all", (0, 5)),
        ),
        reward_probe=_probe_cartpole,
        env_factory=_make_env("cartpole", _cartpole_task, 0.01, 10.0),
        lift_dim=10,
        k_step=50,  # 0.5 s @ 100 Hz (Cartpole development protocol)
        description=(
            "smooth dynamics, 1 act; swingup is energetically nonlinear; "
            "cheapest integration smoke test"
        ),
    ),
    "reacher_hard": TaskSpec(
        name="reacher_hard",
        domain="reacher",
        task="hard",
        obs_layout=(("position", 2), ("to_target", 2), ("velocity", 2)),
        action_dim=2,
        native_control_dt=0.02,
        native_physics_dt=0.02,
        native_time_limit=20.0,
        report_groups=(
            ("position", (0, 2)),
            ("to_target", (2, 4)),
            ("velocity", (4, 6)),
            ("all", (0, 6)),
        ),
        reward_probe=_probe_reacher,
        env_factory=_make_env("reacher", _reacher_task, 0.02, 20.0),
        lift_dim=32,
        k_step=20,  # 0.4 s @ 50 Hz
        description=(
            "no contact, nonlinear, near-sparse 0/1 reward (target radius "
            "0.015); closest to the validated PandaReach regime"
        ),
    ),
    "hopper_hop": TaskSpec(
        name="hopper_hop",
        domain="hopper",
        task="hop",
        obs_layout=(("position", 6), ("velocity", 7), ("touch", 2)),
        action_dim=4,
        native_control_dt=0.02,
        native_physics_dt=0.005,
        native_time_limit=20.0,
        report_groups=(
            ("qpos", (0, 6)),
            ("qvel", (6, 13)),
            ("touch", (13, 15)),
            ("all", (0, 15)),
        ),
        reward_probe=_probe_hopper,
        env_factory=_make_env("hopper", _hopper_task, 0.02, 20.0),
        lift_dim=48,
        k_step=40,  # 0.8 s @ 50 Hz (matches the validated 0.8 s @ 25 Hz/k20)
        description=(
            "single-leg contact; obs/reward 1:1 aligned with MS-HopperHop-v1; "
            "near drop-in from the validated MuJoCo branch"
        ),
    ),
    "hopper_stand": TaskSpec(
        name="hopper_stand",
        domain="hopper",
        task="stand",
        obs_layout=(("position", 6), ("velocity", 7), ("touch", 2)),
        action_dim=4,
        native_control_dt=0.02,
        native_physics_dt=0.005,
        native_time_limit=20.0,
        report_groups=(
            ("qpos", (0, 6)),
            ("qvel", (6, 13)),
            ("touch", (13, 15)),
            ("all", (0, 15)),
        ),
        reward_probe=_probe_hopper,
        env_factory=_make_env("hopper", _hopper_stand_task, 0.02, 20.0),
        lift_dim=48,
        k_step=20,  # 0.8 s at the TD-MPC2 outer rate (25 Hz)
        description=(
            "Hopper stand with the TD-MPC2 two-substep protocol: actions are "
            "held for two native 20 ms control steps and rewards are summed."
        ),
    ),
    "walker_run": TaskSpec(
        name="walker_run",
        domain="walker",
        task="run",
        obs_layout=(("orientations", 14), ("height", 1), ("velocity", 9)),
        action_dim=6,
        native_control_dt=0.025,
        native_physics_dt=0.0025,
        native_time_limit=25.0,
        report_groups=(
            ("orientations", (0, 14)),
            ("height", (14, 15)),
            ("velocity", (15, 24)),
            ("all", (0, 24)),
        ),
        reward_probe=_probe_walker,
        env_factory=_make_env("walker", _walker_task, 0.025, 25.0),
        lift_dim=64,
        k_step=30,  # 0.75 s @ 40 Hz
        description=(
            "bipedal strong contact + gait, target 8 m/s; the decisive "
            "locomotion+contact test (decision gate before humanoid)"
        ),
    ),
    "humanoid_run": TaskSpec(
        name="humanoid_run",
        domain="humanoid",
        task="run",
        obs_layout=(
            ("joint_angles", 21),
            ("head_height", 1),
            ("extremities", 12),
            ("torso_vertical", 3),
            ("com_velocity", 3),
            ("velocity", 27),
        ),
        action_dim=21,
        native_control_dt=0.025,
        native_physics_dt=0.005,
        native_time_limit=25.0,
        report_groups=(
            ("joint_angles", (0, 21)),
            ("features", (21, 40)),
            ("velocity", (40, 67)),
            ("all", (0, 67)),
        ),
        reward_probe=_probe_humanoid,
        env_factory=_make_env("humanoid", _humanoid_task, 0.025, 25.0),
        lift_dim=96,
        k_step=20,  # 0.5 s @ 40 Hz
        description=(
            "standard 67-obs humanoid_run (benchmark tag): high-dimensional "
            "stress test; best comparability with published baselines"
        ),
    ),
    "humanoid_run_pure_state": TaskSpec(
        name="humanoid_run_pure_state",
        domain="humanoid",
        task="run_pure_state",
        obs_layout=(("position", 28), ("velocity", 27)),
        action_dim=21,
        native_control_dt=0.025,
        native_physics_dt=0.005,
        native_time_limit=25.0,
        report_groups=(
            ("position", (0, 28)),
            ("velocity", (28, 55)),
            ("all", (0, 55)),
        ),
        reward_probe=_probe_humanoid,
        env_factory=_make_env(
            "humanoid", _humanoid_pure_state_task, 0.025, 25.0
        ),
        lift_dim=96,
        k_step=20,
        description=(
            "55-obs pure-state humanoid_run: no engineered features; extra "
            "evidence for state-only generalization (not the benchmark tag)"
        ),
    ),
}

# Canonical user-requested ladder (smooth -> nonlinear -> contact -> strong
# contact + locomotion -> pure-state high-dimensional stress test).  The 67D
# engineered-observation Humanoid task is kept as a separately approved public
# benchmark comparison; it is not silently added to the primary workload.
LADDER_ORDER = (
    "cartpole_swingup",
    "reacher_hard",
    "hopper_hop",
    "walker_run",
    "humanoid_run_pure_state",
)
OPTIONAL_COMPARISON_TASKS = ("humanoid_run",)
ALL_TASK_ORDER = LADDER_ORDER + OPTIONAL_COMPARISON_TASKS


def get_task_spec(name: str) -> TaskSpec:
    try:
        return TASK_SPECS[name]
    except KeyError:
        raise KeyError(
            f"Unknown DMC task {name!r}; available: {sorted(TASK_SPECS)}"
        ) from None


def measure_task_spec(name: str) -> dict:
    """Instantiate the live env and return measured runtime values.

    Used by the adapter to validate against the static spec (guards against
    dm_control version drift).
    """
    spec = get_task_spec(name)
    env = spec.env_factory(random=0, control_timestep=None, time_limit=None)
    try:
        obs = env.reset().observation
        act_spec = env.action_spec()
        return {
            "obs_keys": list(obs.keys()),
            "obs_dim": sum(int(np.asarray(v).size) for v in obs.values()),
            "action_dim": int(act_spec.shape[0]),
            "control_timestep": float(env.control_timestep()),
            "physics_timestep": float(env.physics.timestep()),
            "step_limit": int(getattr(env, "_step_limit", -1)),
        }
    finally:
        env.close()


def verify_task_spec(name: str) -> dict:
    """Measure the live env and compare against the static spec.

    Returns a dict of mismatches (empty = consistent).  Raises on any
    mismatch so CI/smoke tests catch version drift early.
    """
    spec = get_task_spec(name)
    measured = measure_task_spec(name)
    mismatches: dict[str, tuple[object, object]] = {}
    expected_keys = [key for key, _ in spec.obs_layout]
    if measured["obs_keys"] != expected_keys:
        mismatches["obs_keys"] = (expected_keys, measured["obs_keys"])
    for field_name, expected in (
        ("obs_dim", spec.obs_dim),
        ("action_dim", spec.action_dim),
        ("control_timestep", spec.native_control_dt),
        ("physics_timestep", spec.native_physics_dt),
        ("step_limit", spec.native_step_limit),
    ):
        actual = measured[field_name]
        if abs(float(actual) - float(expected)) > 1e-9:
            mismatches[field_name] = (expected, actual)
    if mismatches:
        raise RuntimeError(
            f"dm_control spec drift for {name!r}: {mismatches}. "
            f"Installed dm_control differs from the verified 1.0.44 layout; "
            f"update {__file__}."
        )
    return measured
