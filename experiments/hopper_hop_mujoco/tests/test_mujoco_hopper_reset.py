"""MuJoCo Hopper branch tests (Phase 1-3). Does not touch PhysX."""

import numpy as np
import pytest

from experiments.hopper_hop_mujoco.envs.mujoco_hopper import (
    CONTROL_JOINTS,
    MuJoCoHopper,
    MuJoCoHopperConfig,
)
from experiments.hopper_hop_mujoco.envs.contact_config import PRESET_CONTACT_CONFIGS

MJCF = "experiments/hopper_hop_mujoco/assets/hopper_mujoco.xml"
CONFIGS = list(PRESET_CONTACT_CONFIGS)


@pytest.fixture(params=CONFIGS)
def env(request):
    cfg = MuJoCoHopperConfig(
        mjcf_path=MJCF, contact=request.param, seed=0, physics_dt=0.0025
    )
    return MuJoCoHopper(cfg)


def test_reset_state_finite_and_limits(env):
    for seed in range(5):
        obs = env.reset(seed=seed)
        qpos = env.data.qpos
        qvel = env.data.qvel
        assert np.isfinite(qpos).all()
        assert np.isfinite(qvel).all()
        # root planar joints follow ManiSkill convention
        assert qpos[env.joint_ids["rootx"]] == 0.0
        assert qpos[env.joint_ids["rootz"]] == 0.0
        assert -np.pi <= qpos[env.joint_ids["rooty"]] <= np.pi
        # sampled joints stay within (possibly slightly expanded) limits
        for jn in ["waist", "hip", "knee", "ankle"]:
            jid = env.joint_ids[jn]
            lo, hi = env.model.jnt_range[jid]
            adr = env.model.jnt_qposadr[jid]
            assert lo - 1e-9 <= qpos[adr] <= hi + 1e-9, (jn, qpos[adr], lo, hi)
        assert isinstance(obs, dict)


def test_reset_is_reproducible_with_seed(env):
    env.reset(seed=42)
    q1 = env.data.qpos.copy()
    env.reset(seed=42)
    q2 = env.data.qpos.copy()
    assert np.allclose(q1, q2)
    env.reset(seed=7)
    q3 = env.data.qpos.copy()
    assert not np.allclose(q1, q3)


def test_step_is_finite_under_random_actions(env):
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, rew, done, info = env.step(rng.uniform(-1, 1, 4))
        assert np.isfinite(env.data.qpos).all()
        assert np.isfinite(env.data.qvel).all()
        assert np.isfinite(obs["mechanical13"]).all()
        assert isinstance(rew, float) and np.isfinite(rew)


def test_step_advances_exactly_one_control_dt(env):
    env.reset(seed=0)
    t0 = env.data.time
    env.step(np.zeros(4))
    assert np.isclose(env.data.time - t0, env.control_dt, atol=1e-12)
    assert env.n_substeps * env.physics_dt == pytest.approx(env.control_dt)


def test_action_clipping(env):
    env.reset(seed=0)
    obs0, _, _, _ = env.step(np.full(4, 5.0))
    env.reset(seed=0)
    obs1, _, _, _ = env.step(np.full(4, 1.0))
    # clipped 5.0 == 1.0 must give identical trajectories
    assert np.allclose(obs0["mechanical13"], obs1["mechanical13"], atol=1e-6)
