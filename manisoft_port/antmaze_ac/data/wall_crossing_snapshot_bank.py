from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manisoft.utils import KOOPMAN_PHYSICAL_STATE_DIM, KOOPMAN_TIP_POSITION_SLICE


@dataclass(frozen=True)
class WallCrossingSnapshotBank:
    """Certified moving reset states for the distal-body wall-crossing task."""

    names: tuple[str, ...]
    source_episodes: np.ndarray
    source_frames: np.ndarray
    route_sides: np.ndarray
    crossed_fractions: np.ndarray
    physical_states: np.ndarray
    previous_actions: np.ndarray
    node_positions: np.ndarray
    node_velocities: np.ndarray
    element_directors: np.ndarray
    element_omegas: np.ndarray
    rod_internal_states: np.ndarray
    control_dt: float
    scenario_sha256: str
    collection_config_sha256: str
    absolute_action_limit: float
    muscle_torque_scale: float

    @property
    def snapshot_count(self) -> int:
        return len(self.names)


def _scalar(archive: np.lib.npyio.NpzFile, key: str, dtype=float):
    if key not in archive:
        raise ValueError(f"wall-crossing snapshot bank is missing {key!r}")
    return dtype(np.asarray(archive[key]).reshape(()).item())


def load_wall_crossing_snapshot_bank(
    path: str | Path,
) -> WallCrossingSnapshotBank:
    """Load and validate a versioned snapshot bank without object arrays."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing wall-crossing snapshot bank: {resolved}")
    with np.load(resolved, allow_pickle=False) as archive:
        schema_version = _scalar(archive, "schema_version", int)
        kind = _scalar(archive, "kind", str)
        if schema_version != 1 or kind != "manisoft_wall_crossing_snapshot_bank":
            raise ValueError(
                f"unsupported wall-crossing bank schema in {resolved}: "
                f"version={schema_version}, kind={kind!r}"
            )
        names = tuple(str(value) for value in np.asarray(archive["names"]).tolist())
        source_episodes = np.asarray(archive["source_episodes"], dtype=np.int64)
        source_frames = np.asarray(archive["source_frames"], dtype=np.int64)
        route_sides = np.asarray(archive["route_sides"], dtype=np.int8)
        crossed_fractions = np.asarray(
            archive["crossed_fractions"], dtype=np.float32
        )
        physical_states = np.asarray(archive["physical_states"], dtype=np.float32)
        previous_actions = np.asarray(archive["previous_actions"], dtype=np.float32)
        # Moving Position-Verlet snapshots are precision-sensitive.
        node_positions = np.asarray(archive["node_positions"], dtype=np.float64)
        node_velocities = np.asarray(archive["node_velocities"], dtype=np.float64)
        element_directors = np.asarray(
            archive["element_directors"], dtype=np.float64
        )
        element_omegas = np.asarray(archive["element_omegas"], dtype=np.float64)
        rod_internal_states = np.asarray(
            archive["rod_internal_states"], dtype=np.float64
        )
        control_dt = _scalar(archive, "control_dt", float)
        scenario_sha256 = _scalar(archive, "scenario_sha256", str)
        collection_config_sha256 = _scalar(
            archive, "collection_config_sha256", str
        )
        absolute_action_limit = _scalar(archive, "absolute_action_limit", float)
        # Banks created before torque scaling became configurable used the
        # ManiSoft default of 30.  Preserve compatibility while making new
        # material/actuation variants self-describing.
        muscle_torque_scale = (
            _scalar(archive, "muscle_torque_scale", float)
            if "muscle_torque_scale" in archive
            else 30.0
        )

    count = len(names)
    if count < 1 or len(set(names)) != count:
        raise ValueError("snapshot names must be nonempty and unique")
    vector_fields = (
        source_episodes,
        source_frames,
        route_sides,
        crossed_fractions,
    )
    if any(value.shape != (count,) for value in vector_fields):
        raise ValueError("snapshot metadata must have shape [snapshot]")
    if physical_states.shape != (count, KOOPMAN_PHYSICAL_STATE_DIM):
        raise ValueError("physical_states must have shape [snapshot,45]")
    if previous_actions.shape != (count, 18):
        raise ValueError("previous_actions must have shape [snapshot,18]")
    if (
        node_positions.ndim != 3
        or node_positions.shape[0] != count
        or node_positions.shape[2] != 3
        or node_positions.shape[1] < 2
    ):
        raise ValueError("node_positions must have shape [snapshot,node,3]")
    if node_velocities.shape != node_positions.shape:
        raise ValueError("node_velocities must match node_positions")
    element_count = node_positions.shape[1] - 1
    if element_directors.shape != (count, element_count, 3, 3):
        raise ValueError(
            "element_directors must have shape [snapshot,element,3,3]"
        )
    if element_omegas.shape != (count, element_count, 3):
        raise ValueError("element_omegas must have shape [snapshot,element,3]")
    if rod_internal_states.ndim != 2 or rod_internal_states.shape[0] != count:
        raise ValueError("rod_internal_states must have shape [snapshot,cache]")
    numeric = (
        crossed_fractions,
        physical_states,
        previous_actions,
        node_positions,
        node_velocities,
        element_directors,
        element_omegas,
        rod_internal_states,
    )
    if not all(np.isfinite(value).all() for value in numeric):
        raise ValueError("wall-crossing snapshot bank contains NaN or Inf")
    if not np.all(np.isin(route_sides, (-1, 1))):
        raise ValueError("route_sides must contain only -1 and +1")
    if np.any(source_frames < 1) or np.any(source_episodes < 0):
        raise ValueError("snapshot source indices are invalid")
    if np.any(crossed_fractions <= 0) or np.any(crossed_fractions >= 1):
        raise ValueError("snapshot crossed fractions must lie in (0,1)")
    if control_dt <= 0 or absolute_action_limit <= 0 or muscle_torque_scale <= 0:
        raise ValueError("stored time step, action limit and torque scale must be positive")
    if np.max(np.abs(previous_actions)) > absolute_action_limit + 1e-6:
        raise ValueError("stored previous action exceeds the action limit")
    if not np.allclose(
        physical_states[:, KOOPMAN_TIP_POSITION_SLICE],
        node_positions[:, -1],
        atol=2e-5,
    ):
        raise ValueError("stored physical tip and final rod node do not agree")

    return WallCrossingSnapshotBank(
        names=names,
        source_episodes=source_episodes,
        source_frames=source_frames,
        route_sides=route_sides,
        crossed_fractions=crossed_fractions,
        physical_states=physical_states,
        previous_actions=previous_actions,
        node_positions=node_positions,
        node_velocities=node_velocities,
        element_directors=element_directors,
        element_omegas=element_omegas,
        rod_internal_states=rod_internal_states,
        control_dt=control_dt,
        scenario_sha256=scenario_sha256,
        collection_config_sha256=collection_config_sha256,
        absolute_action_limit=absolute_action_limit,
        muscle_torque_scale=muscle_torque_scale,
    )
