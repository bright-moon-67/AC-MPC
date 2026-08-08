"""Canonical state mapping tests: mech13 / legacy15 vs PhysX schema."""

import numpy as np

from experiments.hopper_hop_mujoco.envs.mujoco_hopper import (
    MECH_QPOS_SLICE,
    MECH_QVEL_SLICE,
    MuJoCoHopper,
    MuJoCoHopperConfig,
)

MJCF = "experiments/hopper_hop_mujoco/assets/hopper_mujoco.xml"
JOINT_ORDER = ["rootx", "rootz", "rooty", "waist", "hip", "knee", "ankle"]


def _env():
    return MuJoCoHopper(MuJoCoHopperConfig(mjcf_path=MJCF, contact="mujoco_compliant", seed=0))


def test_joint_order_matches_physx():
    env = _env()
    names = [env.model.joint(i).name for i in range(env.model.njnt)]
    assert names == JOINT_ORDER, names


def test_mechanical13_shape_and_semantics():
    env = _env()
    env.reset(seed=0)
    s = env.get_mechanical_state()
    assert s.shape == (13,)
    assert s.dtype == np.float32
    qpos, qvel = s[:6], s[6:]
    # qpos(6) must be qpos[1:] (rootx dropped), qvel(7) full
    assert np.allclose(qpos, env.data.qpos[MECH_QPOS_SLICE], atol=1e-6)
    assert np.allclose(qvel, env.data.qvel[MECH_QVEL_SLICE], atol=1e-6)
    # spot check ordering: [rootz, rooty, waist, hip, knee, ankle]
    assert np.allclose(qpos[:2], env.data.qpos[[1, 2]], atol=1e-6)
    assert np.allclose(qpos[2:], env.data.qpos[3:], atol=1e-6)


def test_legacy15_shape_and_contact_dims():
    env = _env()
    env.reset(seed=0)
    s = env.get_legacy15_state()
    assert s.shape == (15,)
    assert np.allclose(s[:13], env.get_mechanical_state(), atol=1e-6)
    toe, heel = env.get_toe_heel_force()
    assert np.isclose(s[13], np.log1p(np.linalg.norm(toe)), atol=1e-5)
    assert np.isclose(s[14], np.log1p(np.linalg.norm(heel)), atol=1e-5)
    # touch is always >= 0 (log1p of a norm)
    assert s[13] >= 0.0 and s[14] >= 0.0


def test_contact_touch_changes_when_foot_hits_ground():
    env = _env()
    env.reset(seed=0)
    # let the hopper fall and land with zero action
    s0 = env.get_legacy15_state()
    landed = False
    for _ in range(40):  # up to 1.6 s
        env.step(np.zeros(4))
        s = env.get_legacy15_state()
        if s[13] > 1e-3 or s[14] > 1e-3:
            landed = True
            break
    # the hopper starts 0.53 m in the air (root body z=1.0, foot at 0.53) and
    # must eventually make contact with the ground under gravity
    assert landed, "hopper never touched the ground in 1.6 s"
    assert np.isfinite(s).all()
