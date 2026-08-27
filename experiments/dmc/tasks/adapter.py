"""DMC environment adapter: canonical interface over dm_control suite tasks.

Mirrors the ``HopperAdapterProtocol`` style used by the hopper MuJoCo branch so
that downstream trainers / evaluators / collectors are backend-neutral: the
only thing they see is a canonical float32 observation vector, a [-1, 1]
continuous action, a scalar reward, and metadata.

Canonical contract (all tasks):

  * observation : float32 ``[obs_dim]``, the registry ``obs_layout`` flattened
                  in the order the dm_control task reports it (stable per task)
  * action      : float64 ``[action_dim]`` in ``[action_low, action_high]``
                  (always [-1, 1] for the DMC suite)
  * reward      : the task's official reward (read from ``TimeStep.reward``)
  * done        : end of episode (termination OR time limit); DMC has no
                  early termination, so episodes end at ``step_limit`` steps
  * diagnostics : the original ``TimeStep.discount``, requested/applied
                  actions, per-component reward decomposition, and contact
                  wrenches from dm_control's official ``contact_force`` API
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

import numpy as np

from .registry import DMC_CUSTOM_PROTOCOL, DMC_NATIVE_PROTOCOL, get_task_spec


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


class DMCAdapter:
    """Thin wrapper exposing the canonical interface over one dm_control env."""

    def __init__(
        self,
        task_name: str,
        seed: int = 0,
        control_timestep: Optional[float] = None,
        time_limit: Optional[float] = None,
        validate: bool = True,
    ) -> None:
        self._spec = get_task_spec(task_name)
        self._seed = int(seed)
        self._env = self._spec.env_factory(
            random=self._seed,
            control_timestep=control_timestep,
            time_limit=time_limit,
        )
        self._obs_keys = [key for key, _ in self._spec.obs_layout]
        if validate:
            self._validate_observation_layout()
        self._step_count = 0
        self._step_limit = int(getattr(self._env, "_step_limit", -1))
        self._effective_control_timestep = float(self._env.control_timestep())
        self._effective_time_limit = (
            self._step_limit * self._effective_control_timestep
        )
        self._last_obs: Optional[np.ndarray] = None
        self._last_reward = 0.0
        self._last_done = False
        self._last_discount: Optional[float] = None
        self._last_requested_action: Optional[np.ndarray] = None
        self._last_applied_action: Optional[np.ndarray] = None
        self._last_info: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _validate_observation_layout(self) -> None:
        live = list(self._env.reset().observation.keys())
        if live != self._obs_keys:
            raise RuntimeError(
                f"Observation layout drift for {self._spec.name!r}: expected "
                f"{self._obs_keys}, got {live}. dm_control version differs "
                f"from the verified 1.0.44 layout."
            )

    def _flatten(self, observation) -> np.ndarray:
        parts = []
        for key in self._obs_keys:
            value = observation[key]
            parts.append(np.asarray(value, dtype=np.float64).reshape(-1))
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------
    # canonical interface
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._seed = int(seed)
        self._env.close()
        self._env = self._spec.env_factory(
            random=self._seed,
            control_timestep=self._effective_control_timestep,
            time_limit=self._effective_time_limit,
        )
        self._step_count = 0
        time_step = self._env.reset()
        self._last_obs = self._flatten(time_step.observation)
        self._last_reward = 0.0
        self._last_done = False
        self._last_discount = (
            None if time_step.discount is None else float(time_step.discount)
        )
        self._last_requested_action = None
        self._last_applied_action = None
        self._last_info = {}
        return self._last_obs.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        requested_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if requested_action.shape != (self.action_dim,):
            raise ValueError(
                f"Expected action shape ({self.action_dim},), got "
                f"{requested_action.shape}"
            )
        if np.any(~np.isfinite(requested_action)):
            raise FloatingPointError("Action contains NaN or Inf")
        applied_action = np.clip(
            requested_action, self.action_low, self.action_high
        )
        time_step = self._env.step(applied_action)
        self._step_count += 1
        self._last_obs = self._flatten(time_step.observation)
        self._last_reward = float(time_step.reward)
        self._last_done = bool(time_step.last())
        self._last_discount = (
            None if time_step.discount is None else float(time_step.discount)
        )
        self._last_requested_action = requested_action.astype(np.float32, copy=True)
        self._last_applied_action = applied_action.astype(np.float32, copy=True)
        truncated = self._last_done and self._step_count >= self._step_limit
        step_type = getattr(time_step.step_type, "name", str(time_step.step_type))
        self._last_info = {
            "step_count": self._step_count,
            "step_type": step_type,
            "discount": self._last_discount,
            "terminated": bool(self._last_done and not truncated),
            "truncated": bool(truncated),
            "requested_action": self._last_requested_action.copy(),
            "applied_action": self._last_applied_action.copy(),
            "reward_components": self.get_reward_components(),
            "contact": self.get_contact_diagnostics(),
        }
        return (
            self._last_obs.copy(),
            self._last_reward,
            self._last_done,
            self.get_last_info(),
        )

    # ------------------------------------------------------------------
    # state / reward / diagnostics
    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        if self._last_obs is None:
            self.reset()
        return self._last_obs.copy()

    def get_last_info(self) -> dict[str, Any]:
        """Return diagnostics without exposing mutable adapter-owned arrays."""

        result = dict(self._last_info)
        for name in ("requested_action", "applied_action"):
            if name in result:
                result[name] = np.asarray(result[name]).copy()
        if "reward_components" in result:
            result["reward_components"] = dict(result["reward_components"])
        if "contact" in result:
            result["contact"] = dict(result["contact"])
        return result

    def get_task_reward(self) -> float:
        return self._last_reward

    def get_task_done(self) -> bool:
        return self._last_done

    def get_task_discount(self) -> Optional[float]:
        return self._last_discount

    def get_last_requested_action(self) -> Optional[np.ndarray]:
        if self._last_requested_action is None:
            return None
        return self._last_requested_action.copy()

    def get_last_applied_action(self) -> Optional[np.ndarray]:
        if self._last_applied_action is None:
            return None
        return self._last_applied_action.copy()

    def get_reward_components(self) -> dict[str, float]:
        return self._spec.reward_probe(self._env.physics)

    def get_contact_diagnostics(self) -> dict[str, Any]:
        """Contact force norms from dm_control's official contact API."""
        physics = self._env.physics
        data = physics.data
        forces: list[float] = []
        torques: list[float] = []
        n_contacts = int(data.ncon)
        for contact_id in range(n_contacts):
            wrench = np.asarray(data.contact_force(contact_id), dtype=np.float64)
            if wrench.shape != (2, 3):
                raise RuntimeError(
                    f"Unexpected contact wrench shape {wrench.shape}; expected (2, 3)"
                )
            forces.append(float(np.linalg.norm(wrench[0])))
            torques.append(float(np.linalg.norm(wrench[1])))
        forces = np.asarray(forces, dtype=np.float64)
        torques = np.asarray(torques, dtype=np.float64)
        return {
            "n_contacts": n_contacts,
            "n_force_components": int(3 * n_contacts),
            "total_force": float(forces.sum()) if len(forces) else 0.0,
            "max_force": float(forces.max()) if len(forces) else 0.0,
            "total_torque": float(torques.sum()) if len(torques) else 0.0,
            "max_torque": float(torques.max()) if len(torques) else 0.0,
        }

    # ------------------------------------------------------------------
    # metadata / properties
    # ------------------------------------------------------------------
    @property
    def spec(self):
        return self._spec

    @property
    def task_name(self) -> str:
        return self._spec.name

    @property
    def obs_dim(self) -> int:
        return self._spec.obs_dim

    @property
    def action_dim(self) -> int:
        return self._spec.action_dim

    @property
    def control_dt(self) -> float:
        return float(self._env.control_timestep())

    @property
    def physics_dt(self) -> float:
        return float(self._env.physics.timestep())

    @property
    def n_substeps(self) -> int:
        return int(round(self.control_dt / self.physics_dt))

    @property
    def step_limit(self) -> int:
        return self._step_limit

    @property
    def action_low(self) -> np.ndarray:
        return np.asarray(self._env.action_spec().minimum, dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.asarray(self._env.action_spec().maximum, dtype=np.float32)

    def protocol_metadata(self) -> dict[str, Any]:
        is_native = (
            np.isclose(self.control_dt, self._spec.native_control_dt)
            and np.isclose(self.physics_dt, self._spec.native_physics_dt)
            and self.step_limit == self._spec.native_step_limit
            and np.isclose(
                self.step_limit * self.control_dt, self._spec.native_time_limit
            )
        )
        return {
            "protocol_name": (
                DMC_NATIVE_PROTOCOL if is_native else DMC_CUSTOM_PROTOCOL
            ),
            "protocol_schema_version": 1,
            "task": self._spec.name,
            "domain": self._spec.domain,
            "dmc_task": f"{self._spec.domain}:{self._spec.task}",
            "dm_control_version": _package_version("dm-control"),
            "mujoco_version": _package_version("mujoco"),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "control_dt": self.control_dt,
            "physics_dt": self.physics_dt,
            "n_substeps": self.n_substeps,
            "time_limit": self.step_limit * self.control_dt,
            "step_limit": self.step_limit,
            "action_low": self.action_low.tolist(),
            "action_high": self.action_high.tolist(),
            "obs_layout": [list(item) for item in self._spec.obs_layout],
        }

    def metadata(self) -> dict[str, Any]:
        return {**self.protocol_metadata(), "seed": self._seed}

    def close(self) -> None:
        self._env.close()


class ActionRepeatDMCAdapter(DMCAdapter):
    """DMC adapter with a fixed outer action-repeat protocol.

    The Hopper TD-MPC2 archives contain 25 Hz transitions: one action is held
    for two native 20 ms DMC steps, the observation is taken after the second
    step, and the two rewards are accumulated.  Keeping this wrapper explicit
    prevents those data from being mixed with the native 50 Hz protocol.
    """

    def __init__(self, task_name: str, *, action_repeat: int = 2, **kwargs: Any):
        if action_repeat < 2:
            raise ValueError("Action-repeat adapter requires repeat >= 2")
        super().__init__(task_name, **kwargs)
        if self.step_limit % action_repeat:
            raise ValueError("Native step limit must divide evenly by action_repeat")
        self._action_repeat = int(action_repeat)
        self._native_step_limit = self._step_limit
        self._step_limit //= self._action_repeat
        self._effective_control_timestep *= self._action_repeat
        self._effective_time_limit = self._step_limit * self._effective_control_timestep

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        # Recreate the native environment at its native control timestep; the
        # outer timestep is represented by this wrapper, not dm_control.
        if seed is not None:
            self._seed = int(seed)
        self._env.close()
        self._env = self._spec.env_factory(
            random=self._seed,
            control_timestep=self._spec.native_control_dt,
            time_limit=self._spec.native_time_limit,
        )
        self._step_count = 0
        time_step = self._env.reset()
        self._last_obs = self._flatten(time_step.observation)
        self._last_reward = 0.0
        self._last_done = False
        self._last_discount = None if time_step.discount is None else float(time_step.discount)
        self._last_requested_action = None
        self._last_applied_action = None
        self._last_info = {}
        return self._last_obs.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        requested_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if requested_action.shape != (self.action_dim,):
            raise ValueError(f"Expected action shape ({self.action_dim},), got {requested_action.shape}")
        if np.any(~np.isfinite(requested_action)):
            raise FloatingPointError("Action contains NaN or Inf")
        applied_action = np.clip(requested_action, self.action_low, self.action_high)
        reward = 0.0
        time_step = None
        for _ in range(self._action_repeat):
            time_step = self._env.step(applied_action)
            reward += float(time_step.reward)
            if time_step.last():
                break
        assert time_step is not None
        self._step_count += 1
        self._last_obs = self._flatten(time_step.observation)
        self._last_reward = reward
        self._last_done = bool(time_step.last()) or self._step_count >= self._step_limit
        self._last_discount = None if time_step.discount is None else float(time_step.discount)
        self._last_requested_action = requested_action.astype(np.float32, copy=True)
        self._last_applied_action = applied_action.astype(np.float32, copy=True)
        truncated = self._last_done and self._step_count >= self._step_limit
        step_type = getattr(time_step.step_type, "name", str(time_step.step_type))
        self._last_info = {
            "step_count": self._step_count,
            "step_type": step_type,
            "discount": self._last_discount,
            "action_repeat": self._action_repeat,
            "native_steps": self._action_repeat,
            "terminated": bool(self._last_done and not truncated),
            "truncated": bool(truncated),
            "requested_action": self._last_requested_action.copy(),
            "applied_action": self._last_applied_action.copy(),
            "reward_components": self.get_reward_components(),
            "contact": self.get_contact_diagnostics(),
        }
        return self._last_obs.copy(), self._last_reward, self._last_done, self.get_last_info()

    @property
    def control_dt(self) -> float:
        return float(self._effective_control_timestep)

    @property
    def n_substeps(self) -> int:
        return int(self._action_repeat * round(self._env.control_timestep() / self.physics_dt))

    def protocol_metadata(self) -> dict[str, Any]:
        result = super().protocol_metadata()
        result.update(
            {
                "protocol_name": "tdmpc2_action_repeat2_v1",
                "action_repeat": self._action_repeat,
                "reward_semantics": "sum_of_native_substep_rewards",
                "observation_semantics": "final_native_substep_observation",
            }
        )
        return result


def make_dmc_adapter(
    task_name: str,
    seed: int = 0,
    control_timestep: Optional[float] = None,
    time_limit: Optional[float] = None,
    action_repeat: Optional[int] = None,
) -> DMCAdapter:
    """Build one adapter, optionally pinning the outer action-repeat protocol.

    ``None`` preserves the historical task default (AR2 for Hopper Stand and
    native control for every other task).  Formal Hopper Hop O2O callers pass
    ``action_repeat=2`` explicitly because their TD-MPC2 dataset is recorded
    at 25 Hz.  Keeping that choice explicit avoids silently changing native
    50 Hz Hopper Hop experiments elsewhere in the repository.
    """

    if action_repeat is None:
        resolved_action_repeat = 2 if task_name == "hopper_stand" else 1
    else:
        if isinstance(action_repeat, bool) or not isinstance(action_repeat, int):
            raise TypeError("action_repeat must be an integer or None")
        if action_repeat not in {1, 2}:
            raise ValueError("action_repeat must currently be 1 or 2")
        resolved_action_repeat = int(action_repeat)

    common = {
        "seed": seed,
        "control_timestep": control_timestep,
        "time_limit": time_limit,
    }
    if resolved_action_repeat == 1:
        return DMCAdapter(task_name, **common)
    return ActionRepeatDMCAdapter(
        task_name,
        action_repeat=resolved_action_repeat,
        **common,
    )
