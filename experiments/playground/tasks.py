"""Canonical mapping for the five MuJoCo Playground comparison tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PLAYGROUND_COMMIT = "d5e6b47531328468da1eec33b8b304a00bb0873c"


@dataclass(frozen=True)
class PlaygroundTask:
    name: str
    observation_dim: int
    action_dim: int
    control_timestep: float
    simulation_timestep: float
    episode_steps: int = 1000
    koopman_lift_dim: int = 10
    koopman_horizon_steps: int = 50
    kmpc_horizon_steps: int = 20
    mpve_horizon_steps: int = 10
    mpve_reward_source: str = "learned"

    @property
    def substeps(self) -> int:
        ratio = self.control_timestep / self.simulation_timestep
        rounded = round(ratio)
        if abs(ratio - rounded) > 1e-9:
            raise ValueError(f"{self.name} has a non-integral substep ratio")
        return int(rounded)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["substeps"] = self.substeps
        return result


TASKS: dict[str, PlaygroundTask] = {
    "CartpoleSwingup": PlaygroundTask(
        "CartpoleSwingup",
        5,
        1,
        0.01,
        0.01,
        koopman_lift_dim=10,
        koopman_horizon_steps=50,
        kmpc_horizon_steps=20,
        mpve_horizon_steps=10,
        mpve_reward_source="exact_cartpole",
    ),
    "ReacherHard": PlaygroundTask(
        "ReacherHard",
        6,
        2,
        0.02,
        0.005,
        koopman_lift_dim=10,
        koopman_horizon_steps=25,
        kmpc_horizon_steps=10,
        mpve_horizon_steps=5,
        mpve_reward_source="exact_reacher_hard",
    ),
    "HopperHop": PlaygroundTask(
        "HopperHop",
        15,
        4,
        0.02,
        0.005,
        koopman_lift_dim=24,
        koopman_horizon_steps=25,
        kmpc_horizon_steps=10,
        mpve_horizon_steps=5,
    ),
    "HopperStand": PlaygroundTask(
        "HopperStand",
        15,
        4,
        0.04,
        0.005,
        episode_steps=500,
        koopman_lift_dim=48,
        koopman_horizon_steps=20,
        kmpc_horizon_steps=8,
        mpve_horizon_steps=4,
    ),
    "WalkerRun": PlaygroundTask(
        "WalkerRun",
        24,
        6,
        0.025,
        0.0025,
        koopman_lift_dim=32,
        koopman_horizon_steps=20,
        kmpc_horizon_steps=8,
        mpve_horizon_steps=4,
    ),
    "HumanoidRun": PlaygroundTask(
        "HumanoidRun",
        67,
        21,
        0.025,
        0.005,
        koopman_lift_dim=96,
        koopman_horizon_steps=20,
        kmpc_horizon_steps=8,
        mpve_horizon_steps=4,
        mpve_reward_source="exact_humanoid_run",
    ),
}


def load_task(name: str):
    """Load and validate one official Playground environment lazily."""

    try:
        expected = TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown Playground task {name!r}") from exc

    from mujoco_playground import registry

    config = registry.get_default_config(name)
    environment = registry.load(name, config=config)
    actual = {
        "observation_dim": int(environment.observation_size),
        "action_dim": int(environment.action_size),
        "control_timestep": float(config.ctrl_dt),
        "simulation_timestep": float(config.sim_dt),
        "episode_steps": int(config.episode_length),
    }
    wanted = {
        "observation_dim": expected.observation_dim,
        "action_dim": expected.action_dim,
        "control_timestep": expected.control_timestep,
        "simulation_timestep": expected.simulation_timestep,
        "episode_steps": expected.episode_steps,
    }
    if actual != wanted:
        raise RuntimeError(
            f"Playground task contract changed for {name}: "
            f"expected={wanted}, actual={actual}"
        )
    return environment
