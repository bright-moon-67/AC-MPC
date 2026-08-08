"""Stability / control-dt equivalence tests (the implicit-kd fix)."""

import numpy as np
import pytest

from experiments.hopper_hop_mujoco.envs.mujoco_hopper import (
    MuJoCoHopper,
    MuJoCoHopperConfig,
)
from experiments.hopper_hop_mujoco.envs.contact_config import PRESET_CONTACT_CONFIGS

MJCF = "experiments/hopper_hop_mujoco/assets/hopper_mujoco.xml"


@pytest.mark.parametrize("contact", list(PRESET_CONTACT_CONFIGS))
@pytest.mark.parametrize("dt", [0.005, 0.0025])
def test_random_actions_stable_all_configs(contact, dt):
    """Same stress protocol as the PhysX baseline: random(-1..1) x 200 x 3 seeds."""
    for seed in range(3):
        env = MuJoCoHopper(
            MuJoCoHopperConfig(mjcf_path=MJCF, contact=contact, seed=seed, physics_dt=dt)
        )
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(200):
            env.step(rng.uniform(-1, 1, 4))
            assert np.isfinite(env.data.qpos).all(), (contact, dt, seed)
            assert np.isfinite(env.data.qvel).all(), (contact, dt, seed)


def test_zero_action_free_fall_stable():
    """With implicit kd, even pure free fall (floor removed) must stay finite."""
    env = MuJoCoHopper(
        MuJoCoHopperConfig(mjcf_path=MJCF, contact="mujoco_compliant", seed=0, physics_dt=0.005)
    )
    env.reset(seed=0)
    floor = env.model.geom("floor").id
    env.model.geom_pos[floor, 2] = -100.0
    for _ in range(100):
        env.step(np.zeros(4))
        assert np.isfinite(env.data.qvel).all()


def test_implicit_kd_disabled_still_supported():
    """implicit_kd=False keeps the explicit PD and must still run (for debugging)."""
    env = MuJoCoHopper(
        MuJoCoHopperConfig(
            mjcf_path=MJCF, contact="mujoco_compliant", seed=0, physics_dt=0.001, implicit_kd=False
        )
    )
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.step(rng.uniform(-0.3, 0.3, 4))
        assert np.isfinite(env.data.qvel).all()


def test_control_dt_must_be_multiple_of_physics_dt():
    with pytest.raises(ValueError):
        MuJoCoHopper(
            MuJoCoHopperConfig(mjcf_path=MJCF, contact="mujoco_compliant", physics_dt=0.007)
        )
