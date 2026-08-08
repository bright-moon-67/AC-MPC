"""MuJoCo compliant-contact Hopper — parallel experiment branch.

This package implements a *separate* MuJoCo backend for the MS-HopperHop task so
that ``PhysX-hard`` and ``MuJoCo-compliant`` contacts can be compared strictly.

Design rules (see ``docs/hopper_mujoco_migration_plan.md``):
* Nothing here imports or modifies ``experiments/hopper_hop`` (the PhysX
  pipeline). The original ``MS-HopperHop-v1`` behavior is untouched.
* The MuJoCo backend exposes the same *canonical* action/state/timing as the
  PhysX ``pd_joint_delta_pos`` controller: action ``a in [-1,1]^4`` mapped to
  ``[hip, knee, waist, ankle]``, ``q_target = q + scale*a`` with
  ``scale=[2,2,2,0.8]``, PD ``tau = kp*(q_target-q) - kd*qdot`` with
  ``kp=100, kd=10``, control rate 25 Hz (control_dt 0.04 s).
* Contact compliance is controlled exclusively through native MuJoCo
  ``solref`` / ``solimp`` / ``margin`` / ``friction`` (set at runtime on the
  foot + floor geoms). No custom penalty-force contact model.
"""

from .envs.mujoco_hopper import MuJoCoHopper
from .envs.hopper_adapter import (
    HopperAdapter,
    HopperAdapterProtocol,
    make_hopper_adapter,
)
from .envs.contact_config import ContactConfig, PRESET_CONTACT_CONFIGS

__all__ = [
    "MuJoCoHopper",
    "HopperAdapter",
    "HopperAdapterProtocol",
    "make_hopper_adapter",
    "ContactConfig",
    "PRESET_CONTACT_CONFIGS",
]

__version__ = "0.1.0"
