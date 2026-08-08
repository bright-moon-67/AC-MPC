"""MuJoCo Hopper backend with a PhysX-canonical interface.

The original ``MS-HopperHop-v1`` (ManiSkill 3.0.1 / SAPIEN / PhysX 5.3.1) is
fully preserved elsewhere; this class is a *parallel* MuJoCo implementation of
the same task with the same canonical:

* action      : ``a in [-1,1]^4`` -> joints ``[hip, knee, waist, ankle]``
                ``q_target = q + scale * a``, ``scale = [2, 2, 2, 0.8]``
                ``tau = kp * (q_target - q) - kd * qdot``, ``kp=100, kd=10``
                (verified against ManiSkill PDJointPosController,
                 ``controllers/combined.py`` action order = body then ankle)
* timing      : control 25 Hz (control_dt = 0.04 s); MuJoCo physics dt
                configurable (default 0.005 s), n_substeps = control_dt / dt
* state (legacy15) : [qpos(6), qvel(7), toe_touch(1), heel_touch(1)]
                qpos(6) = [rootz, rooty, waist, hip, knee, ankle] (drop rootx)
                toe/heel_touch = log1p(|| net contact force ||) like PhysX
* state (mechanical13) : [qpos(6), qvel(7)]  (used by the new MuJoCo branch)
* reward       : HopperHop dense/normalized (height + subtreelinvelx tolerance)

Documented (unavoidable or configurable) backend differences vs PhysX:
* MuJoCo uses its native soft-constraint solver (Newton, iterations=100) with
  solref/solimp-controlled compliance; PhysX uses the TGS impulse solver.
* ``match_physx_passives=True`` (default) zeros joint damping/armature that the
  PhysX MJCF loader does NOT import (PhysX effective model has none).
* ``match_physx_actuation=True`` (default) disables MuJoCo motor ctrl clamping
  to match the PhysX drive force_limit ~ inf.
* Friction default is 1.0 in MuJoCo vs 0.3 in PhysX; the compliant/hard
  presets set sliding friction to 0.3 to match PhysX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import mujoco

from .contact_config import (
    PHYSX_FRICTION,
    ContactConfig,
    load_contact_config,
)

ASSET_DIR = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_MJCF = ASSET_DIR / "hopper_mujoco.xml"

# Joints in MJCF order (identical to the PhysX model).
JOINT_NAMES = ["rootx", "rootz", "rooty", "waist", "hip", "knee", "ankle"]

# Canonical action joints: order verified from the real env
# (body controller joints = [hip, knee, waist], then ankle).
CONTROL_JOINTS = ["hip", "knee", "waist", "ankle"]
ACTION_SCALE = np.array([2.0, 2.0, 2.0, 0.8], dtype=np.float64)
KP = 100.0
KD = 10.0

# mechanical state slices (PhysX get_proprioception convention)
MECH_QPOS_SLICE = slice(1, 7)  # drop rootx -> [rootz, rooty, waist, hip, knee, ankle]
MECH_QVEL_SLICE = slice(0, 7)

_CONTACT_GEOMS = ("foot_heel", "foot_toe", "floor")
_STAND_HEIGHT = 0.6
_HOP_SPEED = 2.0


def _tolerance_linear(
    x: np.ndarray,
    lower: float,
    upper: float,
    margin: float,
    value_at_margin: float,
) -> np.ndarray:
    """dm_control/ManiSkill 'linear' tolerance (1 inside bounds, else 1-d*scale)."""
    x = np.asarray(x, dtype=np.float64)
    in_bounds = (x >= lower) & (x <= upper)
    d = np.where(x < lower, lower - x, x - upper) / margin
    scale = 1.0 - value_at_margin
    out = np.where(in_bounds, 1.0, np.maximum(0.0, 1.0 - d * scale))
    return out


@dataclass
class MuJoCoHopperConfig:
    """Configuration for the MuJoCo Hopper backend (does not touch PhysX)."""

    mjcf_path: Union[str, Path] = DEFAULT_MJCF
    contact: Union[str, ContactConfig, Dict[str, Any], Path] = "mujoco_compliant"
    physics_dt: float = 0.005
    control_freq: int = 25
    # match the PhysX-imported model as closely as possible
    match_physx_passives: bool = True  # zero joint damping/armature (PhysX ignores them)
    match_physx_actuation: bool = True  # disable motor ctrl clamp (PhysX drive ~ inf limit)
    kp: float = KP
    kd: float = KD
    # Apply the PD damping kd as implicit `dof_damping` instead of an explicit
    # torque term. MuJoCo's implicitfast solver integrates dof damping
    # semi-implicitly, which is what keeps the stiff PD (kp=100) stable at
    # physics_dt>=0.0025 under violent actions (PhysX's TGS is lossy and
    # damps the same high-frequency modes; MuJoCo's accurate solver does not).
    # The net torque law is unchanged: tau = kp*(target-q) - kd*qdot.
    implicit_kd: bool = True
    action_scale: Tuple[float, ...] = tuple(ACTION_SCALE)
    seed: int = 0
    max_episode_steps: int = 600


class MuJoCoHopper:
    """Canonical MuJoCo Hopper environment (reset/step/state/reward/diagnostics)."""

    def __init__(self, config: Optional[MuJoCoHopperConfig] = None, **overrides) -> None:
        if config is None:
            config = MuJoCoHopperConfig(**overrides)
        else:
            for k, v in overrides.items():
                if not hasattr(config, k):
                    raise TypeError(f"Unknown override {k!r}")
                setattr(config, k, v)
        self.config = config

        self.model = mujoco.MjModel.from_xml_path(str(config.mjcf_path))
        self.data = mujoco.MjData(self.model)

        # ---- timing -----------------------------------------------------
        self.model.opt.timestep = float(config.physics_dt)
        # implicitfast is the MuJoCo-recommended integrator (stable with stiff
        # contacts); contact compliance still comes from solref/solimp only.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # elliptic friction cone: physically preferred (MuJoCo docs), and it
        # makes each contact's efc rows [normal, tan1, tan2] so contact forces
        # can be read exactly from efc_force (cfrc_ext needs an acc-stage
        # sensor and stays 0 otherwise in MuJoCo 3.x).
        self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.physics_dt = float(self.model.opt.timestep)
        self.control_freq = int(config.control_freq)
        self.control_dt = 1.0 / self.control_freq
        self.n_substeps = int(round(self.control_dt / self.physics_dt))
        if not np.isclose(self.n_substeps * self.physics_dt, self.control_dt):
            raise ValueError(
                "control_dt must be an integer multiple of physics_dt: "
                f"control_dt={self.control_dt}, physics_dt={self.physics_dt}"
            )

        # ---- ids --------------------------------------------------------
        self.joint_ids = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        }
        for n, i in self.joint_ids.items():
            if i < 0:
                raise ValueError(f"joint {n!r} not found in {config.mjcf_path}")
        self.body_ids = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("torso", "foot_heel", "foot_toe")
        }
        self.geom_ids = {
            n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in _CONTACT_GEOMS
        }
        if any(i < 0 for i in self.geom_ids.values()):
            raise ValueError("required contact geoms missing from model")

        # canonical control joints [hip, knee, waist, ankle]
        self.control_joint_ids = [self.joint_ids[n] for n in CONTROL_JOINTS]
        self.control_qpos_idx = np.array(
            [self.model.jnt_qposadr[j] for j in self.control_joint_ids], dtype=int
        )
        self.control_qvel_idx = np.array(
            [self.model.jnt_dofadr[j] for j in self.control_joint_ids], dtype=int
        )
        # NOTE: mujoco `data.ctrl` is indexed by ACTUATOR, not by qpos/joint.
        # Actuators are defined as [waist, hip, knee, ankle]; map canonical
        # action joints [hip, knee, waist, ankle] to actuator indices.
        self.control_actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in CONTROL_JOINTS
            ],
            dtype=int,
        )
        self.control_gear = np.array(
            [self.model.actuator_gear[i][0] for i in self.control_actuator_ids],
            dtype=np.float64,
        )
        self.action_scale = np.asarray(config.action_scale, dtype=np.float64)
        if self.action_scale.shape != (4,):
            raise ValueError("action_scale must have shape (4,)")
        self.kp = float(config.kp)
        self.kd = float(config.kd)

        # ---- joint limits (for reset) ----------------------------------
        self.qpos0 = self.model.qpos0.copy()
        self.qlimits = np.zeros((self.model.nq, 2))
        for j in range(self.model.njnt):
            adr = self.model.jnt_qposadr[j]
            if self.model.jnt_limited[j]:
                self.qlimits[adr] = self.model.jnt_range[j]
            else:
                self.qlimits[adr] = [-np.inf, np.inf]

        # ---- contact compliance (native MuJoCo params, runtime edit) ----
        self.contact_config = load_contact_config(config.contact)
        self.contact_config.validate()
        self._apply_contact_config()

        # ---- match PhysX-imported model ---------------------------------
        if config.match_physx_passives:
            self._zero_joint_passives()
        if config.match_physx_actuation:
            self.model.actuator_ctrllimited[:] = 0
        if config.implicit_kd:
            # implicit kd via dof damping (see config docstring); this must be
            # applied after _zero_joint_passives() so it is not wiped out
            self.model.dof_damping[self.control_qvel_idx] = self.kd

        # Never let MuJoCo silently auto-reset on divergence: an auto-reset
        # teleports the hopper back to qpos0 and corrupts episodes. Instead we
        # disable it and raise a clear error from step() when the state blows up.
        self.model.opt.disableflags = (
            self.model.opt.disableflags | int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        )

        self._step = 0
        self._rng = np.random.default_rng(config.seed)
        self._prev_xpos = None  # for finite-difference foot velocity
        self.reset(seed=config.seed)

    # ------------------------------------------------------------------ #
    # contact config application
    # ------------------------------------------------------------------ #
    def _apply_contact_config(self) -> None:
        c = self.contact_config
        for gname in _CONTACT_GEOMS:
            gid = self.geom_ids[gname]
            self.model.geom_solref[gid] = list(c.solref)
            self.model.geom_solimp[gid] = list(c.solimp)
            self.model.geom_margin[gid] = c.margin
            self.model.geom_friction[gid] = (c.sliding_friction, 0.005, 0.0001)

    def _zero_joint_passives(self) -> None:
        """Zero joint damping/armature so the MuJoCo model matches the PhysX
        MJCF import (ManiSkill's loader does not import joint damping/armature)."""
        for j in range(self.model.njnt):
            dof_adr = self.model.jnt_dofadr[j]
            n = self.model.jnt_dofadr[j + 1] - dof_adr if j + 1 < self.model.njnt else 1
            self.model.dof_damping[dof_adr : dof_adr + n] = 0.0
            self.model.dof_armature[dof_adr : dof_adr + n] = 0.0

    # ------------------------------------------------------------------ #
    # reset / step
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Randomize qpos like ManiSkill `_initialize_episode` and return obs."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        # sample only limited joints within their ranges (like ManiSkill, which
        # then overwrites the free root joints)
        for j in range(self.model.njnt):
            adr = self.model.jnt_qposadr[j]
            if self.model.jnt_limited[j]:
                lo, hi = self.model.jnt_range[j]
                qpos[adr] = self._rng.uniform(lo, hi)
        # planar free-root conventions (same as ManiSkill _initialize_episode)
        qpos[self.joint_ids["rootx"]] = 0.0
        qpos[self.joint_ids["rootz"]] = 0.0
        qpos[self.joint_ids["rooty"]] = np.pi * (2.0 * self._rng.random() - 1.0)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        self._prev_xpos = self.data.xpos.copy()
        return self.get_observation()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply one canonical action (25 Hz) and advance ``n_substeps`` physics steps."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a.shape != (4,):
            raise ValueError(f"canonical action must be 4D, got {a.shape}")
        # PhysX PD: target fixed for the whole control step, drive applied each substep
        q_start = self.data.qpos[self.control_qpos_idx].copy()
        target = q_start + self.action_scale * a
        for _ in range(self.n_substeps):
            q = self.data.qpos[self.control_qpos_idx]
            if self.config.implicit_kd:
                # kd is already applied implicitly via dof_damping
                tau = self.kp * (target - q)
            else:
                qdot = self.data.qvel[self.control_qvel_idx]
                tau = self.kp * (target - q) - self.kd * qdot
            self.data.ctrl[self.control_actuator_ids] = tau / self.control_gear
            mujoco.mj_step(self.model, self.data)
            if not (
                np.isfinite(self.data.qpos).all()
                and np.isfinite(self.data.qvel).all()
            ):
                raise FloatingPointError(
                    "MuJoCo hopper state diverged at time %.4f (qacc=%s)"
                    % (self.data.time, self.data.qacc)
                )
        self._step += 1
        self._prev_xpos = self.data.xpos.copy()
        return self.get_observation(), self.get_task_reward(), self.get_task_done(), self.get_info()

    # ------------------------------------------------------------------ #
    # canonical states
    # ------------------------------------------------------------------ #
    def get_mechanical_state(self) -> np.ndarray:
        """x_mech = [qpos(6), qvel(7)] = 13D, PhysX joint order/units."""
        return np.concatenate(
            (
                self.data.qpos[MECH_QPOS_SLICE].astype(np.float32),
                self.data.qvel[MECH_QVEL_SLICE].astype(np.float32),
            )
        )

    def get_legacy15_state(self) -> np.ndarray:
        """legacy15 = [qpos(6), qvel(7), toe_touch, heel_touch] (PhysX state)."""
        toe, heel = self.get_toe_heel_force()
        return np.concatenate(
            (
                self.get_mechanical_state(),
                np.asarray([np.log1p(np.linalg.norm(toe)), np.log1p(np.linalg.norm(heel))], dtype=np.float32),
            )
        )

    def get_observation(self) -> Dict[str, Any]:
        return {
            "mechanical13": self.get_mechanical_state(),
            "legacy15": self.get_legacy15_state(),
            "time": self.data.time,
        }

    def _net_contact_force(self, geom_name: str) -> np.ndarray:
        """3D net contact force on a foot geom's body (elliptic condim=3 rows).

        efc rows of a contact are [normal, tan1, tan2] in the contact frame;
        `contact.frame` maps them to world coordinates. Force direction is
        resolved so the returned vector acts ON the foot body.
        """
        gid = self.geom_ids[geom_name]
        F = np.zeros(3, dtype=np.float64)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.efc_address < 0:
                continue
            if c.geom1 != gid and c.geom2 != gid:
                continue
            dim = int(c.dim)
            rows = self.data.efc_force[c.efc_address : c.efc_address + dim]
            f_frame = np.zeros(3)
            f_frame[:dim] = rows
            frame = np.asarray(c.frame).reshape(3, 3)
            world = frame @ f_frame
            F += world if c.geom2 == gid else -world
        return F

    def get_toe_heel_force(self) -> Tuple[np.ndarray, np.ndarray]:
        """Net contact forces on the toe/heel bodies (like PhysX impulse/dt force)."""
        toe = self._net_contact_force("foot_toe")
        heel = self._net_contact_force("foot_heel")
        return toe, heel

    def get_contact_diagnostics(self) -> Dict[str, np.ndarray]:
        """Continuous contact diagnostics (for calibration and later state variants)."""
        toe_body, heel_body = self.body_ids["foot_toe"], self.body_ids["foot_heel"]
        toe_F, heel_F = self.get_toe_heel_force()
        toe_bottom = self.data.xpos[toe_body, 2] - 0.04  # capsule radius
        heel_bottom = self.data.xpos[heel_body, 2] - 0.04
        toe_vel = np.zeros(3, dtype=np.float32)
        heel_vel = np.zeros(3, dtype=np.float32)
        if self._prev_xpos is not None:
            toe_vel = (
                (self.data.xpos[toe_body] - self._prev_xpos[toe_body]) / self.control_dt
            ).astype(np.float32)
            heel_vel = (
                (self.data.xpos[heel_body] - self._prev_xpos[heel_body]) / self.control_dt
            ).astype(np.float32)

        # signed gaps / penetration from the narrow-phase contacts
        toe_pen = 0.0
        heel_pen = 0.0
        toe_contact = 0
        heel_contact = 0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            dist = con.dist
            if g1 == self.geom_ids["foot_toe"] or g2 == self.geom_ids["foot_toe"]:
                toe_contact = 1
                toe_pen = min(toe_pen, dist)  # negative = penetration
            if g1 == self.geom_ids["foot_heel"] or g2 == self.geom_ids["foot_heel"]:
                heel_contact = 1
                heel_pen = min(heel_pen, dist)
        return {
            "toe_force": toe_F.astype(np.float32),
            "heel_force": heel_F.astype(np.float32),
            "toe_touch": float(np.log1p(np.linalg.norm(toe_F))),
            "heel_touch": float(np.log1p(np.linalg.norm(heel_F))),
            "toe_gap": float(toe_bottom),
            "heel_gap": float(heel_bottom),
            "toe_penetration": float(min(0.0, toe_pen)),
            "heel_penetration": float(min(0.0, heel_pen)),
            "toe_velocity": toe_vel.astype(np.float32),
            "heel_velocity": heel_vel.astype(np.float32),
            "toe_contact": int(toe_contact),
            "heel_contact": int(heel_contact),
        }

    # ------------------------------------------------------------------ #
    # task quantities (replicate HopperEnv / HopperHopEnv)
    # ------------------------------------------------------------------ #
    @property
    def height(self) -> float:
        return float(self.data.xpos[self.body_ids["torso"], 2] - self.data.xpos[self.body_ids["foot_heel"], 2])

    @property
    def subtreelinvelx(self) -> float:
        """Mass-weighted x velocity of the subtree rooted at torso (links 1..nbody-1)."""
        masses = self.model.body_mass[1:]
        vels = self.data.cvel[1:, 0]
        total = float(masses.sum())
        return float(np.dot(masses, vels) / total) if total > 0 else 0.0

    def get_task_reward(self) -> float:
        standing = 1.0 if self._STAND_HEIGHT <= self.height <= 2.0 else 0.0
        hopping = _tolerance_linear(
            np.asarray([self.subtreelinvelx]), _HOP_SPEED, float("inf"),
            margin=_HOP_SPEED / 2, value_at_margin=0.5,
        )[0]
        return float(standing * hopping)

    def get_task_done(self) -> bool:
        return False  # MS-HopperHop has no early termination (600-step horizon)

    def get_info(self) -> Dict[str, Any]:
        return {
            "height": self.height,
            "subtreelinvelx": self.subtreelinvelx,
            "contact": self.get_contact_diagnostics(),
        }

    _STAND_HEIGHT = _STAND_HEIGHT

    def close(self) -> None:
        pass


def _main() -> None:
    """Smoke test: run a few mild random steps in each preset (like early PPO)."""
    for preset in ("mujoco_default", "mujoco_compliant", "mujoco_hard"):
        env = MuJoCoHopper(contact=preset, seed=0)
        rng = np.random.default_rng(0)
        obs = env.reset(seed=0)
        for _ in range(10):
            obs, rew, done, info = env.step(rng.uniform(-0.3, 0.3, 4))
        diag = info["contact"]
        print(
            f"[{preset}] mech13={obs['mechanical13'][:3]} "
            f"height={info['height']:.3f} vx={info['subtreelinvelx']:.3f} "
            f"toe_gap={diag['toe_gap']:.4f} toe_F={np.linalg.norm(diag['toe_force']):.1f}"
        )
        env.close()


if __name__ == "__main__":
    _main()
