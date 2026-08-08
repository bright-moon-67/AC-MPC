"""Backend-neutral Hopper adapter interface.

Both the original PhysX backend (`MS-HopperHop-v1`) and the new MuJoCo backend
are wrapped into the same canonical semantics so that a Koopman trainer /
evaluator never needs to know which simulator produced the data.

Canonical contract (verified against the real ManiSkill env, 2026-08-08):

  * action        : a in [-1, 1]^4 over joints [hip, knee, waist, ankle]
                    q_target = q + [2, 2, 2, 0.8] * a
                    tau = 100*(q_target - q) - 10*qdot
  * control rate  : 25 Hz, control_dt = 0.04 s
  * state         : mechanical13 = [qpos(6), qvel(7)]
                    qpos(6) = [rootz, rooty, waist, hip, knee, ankle]
  * contact       : toe_touch / heel_touch = log1p(|| net contact force ||)
  * task          : MS-HopperHop dense reward (height * hop-speed tolerance)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np


@runtime_checkable
class HopperAdapterProtocol(Protocol):
    """Minimal canonical interface shared by the PhysX and MuJoCo backends."""

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]: ...

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]: ...

    def get_mechanical_state(self) -> np.ndarray: ...

    def get_legacy15_state(self) -> np.ndarray: ...

    def get_contact_diagnostics(self) -> Dict[str, np.ndarray]: ...

    def get_task_reward(self) -> float: ...

    def get_task_done(self) -> bool: ...

    @property
    def control_dt(self) -> float: ...

    @property
    def physics_dt(self) -> float: ...

    @property
    def n_substeps(self) -> int: ...

    def close(self) -> None: ...


class HopperAdapter:
    """Thin wrapper exposing a common interface over a backend env object.

    The MuJoCo backend (`MuJoCoHopper`) already implements this interface
    directly; this class is provided so a future `PhysXHopperAdapter` (wrapping
    the untouched `MS-HopperHop-v1`) can be plugged in with identical call
    semantics, and so callers can depend on one small import.
    """

    def __init__(self, backend: Any, backend_name: str = "unknown") -> None:
        self._backend = backend
        self.backend_name = backend_name

    # ---- canonical interface (delegated) --------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return self._backend.reset(seed=seed)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return self._backend.step(action)

    def get_mechanical_state(self) -> np.ndarray:
        return self._backend.get_mechanical_state()

    def get_legacy15_state(self) -> np.ndarray:
        return self._backend.get_legacy15_state()

    def get_contact_diagnostics(self) -> Dict[str, np.ndarray]:
        return self._backend.get_contact_diagnostics()

    def get_task_reward(self) -> float:
        return self._backend.get_task_reward()

    def get_task_done(self) -> bool:
        return self._backend.get_task_done()

    @property
    def control_dt(self) -> float:
        return self._backend.control_dt

    @property
    def physics_dt(self) -> float:
        return self._backend.physics_dt

    @property
    def n_substeps(self) -> int:
        return self._backend.n_substeps

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def close(self) -> None:
        self._backend.close()

    def metadata(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "adapter": "HopperAdapter",
            "control_dt": self.control_dt,
            "physics_dt": self.physics_dt,
            "n_substeps": self.n_substeps,
        }


def make_hopper_adapter(
    backend: str = "mujoco_compliant",
    contact: Optional[Union[str, Any]] = None,
    physics_dt: float = 0.005,
    seed: int = 0,
    **kwargs: Any,
) -> HopperAdapter:
    """Factory: ``backend`` selects the simulator branch.

    Current implementations:
      * ``mujoco_*`` presets (mujoco_default / mujoco_compliant / mujoco_hard)
        -> the new MuJoCo backend.
    The PhysX branch is intentionally NOT instantiated here (it lives in the
    untouched ``experiments/hopper_hop`` pipeline); it can be added later
    without changing this interface.
    """
    from .mujoco_hopper import MuJoCoHopper, MuJoCoHopperConfig

    contact_name = contact if contact is not None else backend
    env = MuJoCoHopper(
        MuJoCoHopperConfig(contact=contact_name, physics_dt=physics_dt, seed=seed),
        **kwargs,
    )
    return HopperAdapter(env, backend_name=f"mujoco/{contact_name}")
