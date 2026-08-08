"""Contact calibration tests: physically sensible drop metrics for all presets."""

import numpy as np
import pytest

from experiments.hopper_hop_mujoco.eval.contact_calibration import run_drop
from experiments.hopper_hop_mujoco.envs.contact_config import PRESET_CONTACT_CONFIGS


def test_impact_velocity_is_physical():
    # free fall from 0.5 m -> v = sqrt(2*g*h) = 3.13 m/s (measured ~2.9-3.0)
    for name in PRESET_CONTACT_CONFIGS:
        res = run_drop(PRESET_CONTACT_CONFIGS[name], physics_dt=0.0025)
        assert 2.5 <= res.impact_velocity <= 3.3, (name, res.impact_velocity)


def test_rest_force_equals_weight():
    # capsule mass ~1.223 kg -> m*g ~ 12.0 N at rest
    for name in PRESET_CONTACT_CONFIGS:
        res = run_drop(PRESET_CONTACT_CONFIGS[name], physics_dt=0.0025)
        assert abs(res.rest_force - 12.0) < 1.0, (name, res.rest_force)
        assert abs(res.foot_mass * 9.81 - res.rest_force) < 1.0, name


def test_impulse_equals_momentum_change():
    # ground impulse over the impact window = m * d(velocity) + m*g*dt_window
    # (the window also supports the capsule's weight while it decelerates)
    for name in ["mujoco_default", "mujoco_compliant", "mujoco_hard"]:
        res = run_drop(PRESET_CONTACT_CONFIGS[name], physics_dt=0.0025)
        m = res.foot_mass
        momentum = m * (res.impact_velocity + res.rebound_velocity)
        weight = m * 9.81 * res.impact_duration_s
        expected = momentum + weight
        assert 0.6 * expected <= res.impulse_Ns <= 1.4 * expected, (
            name,
            res.impulse_Ns,
            expected,
        )


def test_stiffness_ladder_default_compliant_hard():
    """Compliant must penetrate more and push less than default; hard the opposite."""
    c = run_drop(PRESET_CONTACT_CONFIGS["mujoco_compliant"], physics_dt=0.0025)
    d = run_drop(PRESET_CONTACT_CONFIGS["mujoco_default"], physics_dt=0.0025)
    h = run_drop(PRESET_CONTACT_CONFIGS["mujoco_hard"], physics_dt=0.0025)
    # compliant penetrates much deeper on impact
    assert c.max_penetration > d.max_penetration > h.max_penetration
    # compliant peak force much lower than default/hard
    assert c.peak_normal_force < d.peak_normal_force
    assert d.peak_normal_force < h.peak_normal_force
    # compliant impact duration is much longer (spread-out impulse)
    assert c.impact_duration_s > d.impact_duration_s
    assert d.impact_duration_s > h.impact_duration_s


def test_hard_is_near_rigid_at_dt_0_0025():
    h = run_drop(PRESET_CONTACT_CONFIGS["mujoco_hard"], physics_dt=0.0025)
    assert h.max_penetration < 1e-3  # < 1 mm
    assert abs(h.rest_gap) < 1e-3
    assert h.impact_duration_s <= 0.01
