from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manisoft.utils import KOOPMAN_PHYSICAL_STATE_DIM, KOOPMAN_TIP_POSITION_SLICE


# PyElastica's Position-Verlet step consumes these cached quantities before it
# recomputes all of them.  A moving-state reset must therefore restore them in
# addition to the four primary position/velocity/director/omega arrays.
ROD_INTERNAL_STATE_FIELDS = (
    "acceleration_collection",
    "alpha_collection",
    "radius",
    "mass_second_moment_of_inertia",
    "inv_mass_second_moment_of_inertia",
    "shear_matrix",
    "bend_matrix",
    "density",
    "volume",
    "mass",
    "internal_forces",
    "internal_torques",
    "external_forces",
    "external_torques",
    "lengths",
    "rest_lengths",
    "tangents",
    "dilatation",
    "dilatation_rate",
    "voronoi_dilatation",
    "rest_voronoi_lengths",
    "sigma",
    "kappa",
    "rest_sigma",
    "rest_kappa",
    "internal_stress",
    "internal_couple",
)


def pack_rod_internal_state(rod) -> np.ndarray:
    """Flatten every mutable Position-Verlet cache in a stable field order."""

    return np.concatenate(
        [np.asarray(getattr(rod, name)).reshape(-1) for name in ROD_INTERNAL_STATE_FIELDS]
    ).astype(np.float64)


def restore_rod_internal_state(rod, packed: np.ndarray) -> None:
    """Restore a cache vector produced by :func:`pack_rod_internal_state`."""

    values = np.asarray(packed, dtype=np.float64).reshape(-1)
    offset = 0
    for name in ROD_INTERNAL_STATE_FIELDS:
        destination = getattr(rod, name)
        size = int(destination.size)
        destination[...] = values[offset : offset + size].reshape(destination.shape)
        offset += size
    if offset != len(values):
        raise ValueError(
            f"rod internal-state size mismatch: restored {offset}, stored {len(values)}"
        )


@dataclass(frozen=True)
class TableEntryTrajectoryBank:
    """Simulator-validated open-loop motions from upright to the table area."""

    names: tuple[str, ...]
    physical_states: np.ndarray
    actions: np.ndarray
    node_positions: np.ndarray
    node_velocities: np.ndarray
    element_directors: np.ndarray
    element_omegas: np.ndarray
    rod_internal_states: np.ndarray
    control_dt: float
    scenario_sha256: str
    table_x_bounds: np.ndarray
    table_y_bounds: np.ndarray
    table_surface_z: float
    arm_radius: float
    safety_margin: float
    absolute_action_limit: float

    @property
    def trajectory_count(self) -> int:
        return int(self.actions.shape[0])

    @property
    def transition_count(self) -> int:
        return int(self.actions.shape[1])

    @property
    def tip_positions(self) -> np.ndarray:
        return self.physical_states[..., KOOPMAN_TIP_POSITION_SLICE]


def _scalar(archive: np.lib.npyio.NpzFile, key: str, dtype=float):
    if key not in archive:
        raise ValueError(f"table-entry bank is missing {key!r}")
    return dtype(np.asarray(archive[key]).reshape(()).item())


def load_table_entry_trajectory_bank(
    path: str | Path,
) -> TableEntryTrajectoryBank:
    """Load a versioned bank without pickle/object arrays."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing table-entry trajectory bank: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        schema_version = _scalar(archive, "schema_version", int)
        kind = _scalar(archive, "kind", str)
        if schema_version != 3 or kind != "manisoft_table_entry_trajectory_bank":
            raise ValueError(
                f"unsupported table-entry bank schema in {resolved}: "
                f"version={schema_version}, kind={kind!r}"
            )
        names = tuple(str(value) for value in np.asarray(archive["names"]).tolist())
        states = np.asarray(archive["physical_states"], dtype=np.float32)
        actions = np.asarray(archive["actions"], dtype=np.float32)
        # Moving Cosserat-rod snapshots are precision-sensitive. Converting
        # these arrays to float32 can destabilize the very next Verlet step.
        nodes = np.asarray(archive["node_positions"], dtype=np.float64)
        node_velocities = np.asarray(archive["node_velocities"], dtype=np.float64)
        directors = np.asarray(archive["element_directors"], dtype=np.float64)
        omegas = np.asarray(archive["element_omegas"], dtype=np.float64)
        internal_states = np.asarray(archive["rod_internal_states"], dtype=np.float64)
        control_dt = _scalar(archive, "control_dt", float)
        scenario_sha256 = _scalar(archive, "scenario_sha256", str)
        table_x_bounds = np.asarray(archive["table_x_bounds"], dtype=np.float64)
        table_y_bounds = np.asarray(archive["table_y_bounds"], dtype=np.float64)
        table_surface_z = _scalar(archive, "table_surface_z", float)
        arm_radius = _scalar(archive, "arm_radius", float)
        safety_margin = _scalar(archive, "safety_margin", float)
        absolute_action_limit = (
            _scalar(archive, "absolute_action_limit", float)
            if "absolute_action_limit" in archive
            else 0.30
        )

    if not names or len(set(names)) != len(names):
        raise ValueError("table-entry trajectory names must be nonempty and unique")
    if actions.ndim != 3 or actions.shape[0] != len(names) or actions.shape[2] != 18:
        raise ValueError("table-entry actions must have shape [trajectory, step, 18]")
    expected_states = (len(names), actions.shape[1] + 1, KOOPMAN_PHYSICAL_STATE_DIM)
    if states.shape != expected_states:
        raise ValueError(
            f"table-entry physical_states must have shape {expected_states}, "
            f"got {states.shape}"
        )
    if (
        nodes.ndim != 4
        or nodes.shape[:2] != states.shape[:2]
        or nodes.shape[-1] != 3
        or nodes.shape[2] < 2
    ):
        raise ValueError(
            "table-entry node_positions must have shape "
            "[trajectory, step+1, node, 3]"
        )
    if node_velocities.shape != nodes.shape:
        raise ValueError("node_velocities must have the same shape as node_positions")
    expected_directors = (
        len(names),
        actions.shape[1] + 1,
        nodes.shape[2] - 1,
        3,
        3,
    )
    if directors.shape != expected_directors:
        raise ValueError(
            f"element_directors must have shape {expected_directors}, got {directors.shape}"
        )
    expected_omegas = expected_directors[:3] + (3,)
    if omegas.shape != expected_omegas:
        raise ValueError(
            f"element_omegas must have shape {expected_omegas}, got {omegas.shape}"
        )
    if internal_states.ndim != 3 or internal_states.shape[:2] != states.shape[:2]:
        raise ValueError(
            "rod_internal_states must have shape [trajectory, step+1, cache]"
        )
    if not all(
        np.isfinite(value).all()
        for value in (
            states,
            actions,
            nodes,
            node_velocities,
            directors,
            omegas,
            internal_states,
        )
    ):
        raise ValueError("table-entry bank contains NaN or Inf")
    if absolute_action_limit <= 0:
        raise ValueError("stored activation limit must be positive")
    if np.max(np.abs(actions)) > absolute_action_limit + 1e-6:
        raise ValueError(
            "table-entry bank exceeds its stored activation limit"
        )
    if not np.allclose(
        states[..., KOOPMAN_TIP_POSITION_SLICE], nodes[..., -1, :], atol=2e-5
    ):
        raise ValueError("stored tip states and final rod nodes do not agree")
    if table_x_bounds.shape != (2,) or table_y_bounds.shape != (2,):
        raise ValueError("stored table bounds must each have shape [2]")
    if not (
        table_x_bounds[0] < table_x_bounds[1]
        and table_y_bounds[0] < table_y_bounds[1]
    ):
        raise ValueError("stored table bounds must be increasing")
    if control_dt <= 0 or arm_radius <= 0 or safety_margin < 0:
        raise ValueError("stored time step and clearance parameters are invalid")

    return TableEntryTrajectoryBank(
        names=names,
        physical_states=states,
        actions=actions,
        node_positions=nodes,
        node_velocities=node_velocities,
        element_directors=directors,
        element_omegas=omegas,
        rod_internal_states=internal_states,
        control_dt=control_dt,
        scenario_sha256=scenario_sha256,
        table_x_bounds=table_x_bounds,
        table_y_bounds=table_y_bounds,
        table_surface_z=table_surface_z,
        arm_radius=arm_radius,
        safety_margin=safety_margin,
        absolute_action_limit=absolute_action_limit,
    )
