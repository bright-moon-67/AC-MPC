"""Canonical action mapping tests: [hip,knee,waist,ankle], scale [2,2,2,0.8], kp=100, kd=10."""

import numpy as np

from experiments.hopper_hop_mujoco.envs.mujoco_hopper import (
    ACTION_SCALE,
    CONTROL_JOINTS,
    KD,
    KP,
    MuJoCoHopper,
    MuJoCoHopperConfig,
)

MJCF = "experiments/hopper_hop_mujoco/assets/hopper_mujoco.xml"
CONTROL_GEAR = {"waist": 30.0, "hip": 40.0, "knee": 30.0, "ankle": 10.0}


def test_control_joint_order_and_scale():
    assert CONTROL_JOINTS == ["hip", "knee", "waist", "ankle"]
    assert np.allclose(ACTION_SCALE, [2.0, 2.0, 2.0, 0.8])
    assert KP == 100.0 and KD == 10.0


def test_actuator_mapping_matches_physx_gear():
    env = MuJoCoHopper(MuJoCoHopperConfig(mjcf_path=MJCF, contact="mujoco_compliant", seed=0))
    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    assert names == ["waist", "hip", "knee", "ankle"]
    # canonical joints [hip,knee,waist,ankle] -> actuator ids
    expected_ids = [names.index(j) for j in CONTROL_JOINTS]
    assert env.control_actuator_ids.tolist() == expected_ids
    assert np.allclose(env.control_gear, [CONTROL_GEAR[j] for j in CONTROL_JOINTS])


def test_zero_action_holds_joint_positions():
    env = MuJoCoHopper(MuJoCoHopperConfig(mjcf_path=MJCF, contact="mujoco_compliant", seed=0))
    env.reset(seed=0)
    q0 = env.data.qpos[env.control_qpos_idx].copy()
    # one control step with zero action: PD target = q_start -> joints barely move
    # (allowed small drift under gravity, but must stay within 5 deg)
    env.step(np.zeros(4))
    q1 = env.data.qpos[env.control_qpos_idx]
    assert np.allclose(q1, q0, atol=np.deg2rad(5.0)), (q0, q1)


def test_torque_law_is_kp_spring_minus_implicit_kd():
    env = MuJoCoHopper(MuJoCoHopperConfig(mjcf_path=MJCF, contact="mujoco_compliant", seed=0))
    m, d = env.model, env.data
    # freeze gravity & zero velocity, then check spring torque = kp*(target-q)
    m.opt.gravity[:] = 0.0
    env.reset(seed=0)
    d.qvel[:] = 0.0
    mujoco = __import__("mujoco")
    q = d.qpos[env.control_qpos_idx].copy()
    target = q + np.array([0.3, -0.2, 0.1, -0.4])  # arbitrary targets
    # apply ctrl for that target and read the actuator torque
    d.ctrl[env.control_actuator_ids] = (100.0 * (target - q)) / env.control_gear
    mujoco.mj_forward(m, d)
    tau = d.qfrc_actuator[env.control_qvel_idx]
    assert np.allclose(tau, 100.0 * (target - q), atol=1e-6), (tau, 100.0 * (target - q))


def test_implicit_kd_is_applied_as_dof_damping():
    env = MuJoCoHopper(MuJoCoHopperConfig(mjcf_path=MJCF, contact="mujoco_compliant", seed=0))
    assert env.config.implicit_kd
    # with implicit_kd, dof_damping on control DOFs equals kd (=10)
    assert np.allclose(env.model.dof_damping[env.control_qvel_idx], KD)
    # passives on non-control DOFs stay zero (PhysX parity)
    root_dofs = [
        env.model.jnt_dofadr[env.joint_ids[n]] for n in ["rootx", "rootz", "rooty"]
    ]
    assert np.allclose(env.model.dof_damping[root_dofs], 0.0)
