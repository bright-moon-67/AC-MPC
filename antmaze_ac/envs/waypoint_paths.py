from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


PATH_FAMILIES = ("line", "polyline", "bezier", "s_curve", "reverse")
CURRICULUM_STAGES = (
    "point",
    "entry_tail",
    "entry_bridge",
    "entry_mid",
    "entry",
    "table_point",  # compatibility alias for the corrected entry stage
    "table_local_line",
    "table_waypoint_polyline",
    "table_long_waypoints",
    "table_local",
    "entry_local",
    "path",
    "recovery",
    "mixed",
)


@dataclass(frozen=True)
class WaypointWorkspace:
    """Geometric sampling limits for obstacle-free table-top references."""

    low: np.ndarray
    high: np.ndarray
    max_reach: float = 0.90
    xy_hull_equations: np.ndarray | None = None
    xy_triangulation: Any | None = None
    xy_valid_simplices: np.ndarray | None = None
    interpolation_dimensions: int = 2

    @classmethod
    def from_bounds(
        cls,
        low: Sequence[float] = (-0.30, 0.50, 0.445),
        high: Sequence[float] = (0.30, 0.80, 0.54),
        max_reach: float = 0.90,
        xy_hull_equations: np.ndarray | None = None,
        xy_triangulation: Any | None = None,
        xy_valid_simplices: np.ndarray | None = None,
        interpolation_dimensions: int = 2,
    ) -> "WaypointWorkspace":
        low_array = np.asarray(low, dtype=np.float64)
        high_array = np.asarray(high, dtype=np.float64)
        if low_array.shape != (3,) or high_array.shape != (3,):
            raise ValueError("workspace bounds must each contain three values")
        if not np.all(low_array < high_array):
            raise ValueError("workspace low bounds must be below high bounds")
        if max_reach <= 0:
            raise ValueError("max_reach must be positive")
        dimensions = int(interpolation_dimensions)
        if dimensions not in {2, 3}:
            raise ValueError("interpolation_dimensions must be two or three")
        equations = (
            None
            if xy_hull_equations is None
            else np.asarray(xy_hull_equations, dtype=np.float64)
        )
        if equations is not None and (
            equations.ndim != 2
            or equations.shape[1] != dimensions + 1
            or len(equations) < dimensions + 1
            or not np.isfinite(equations).all()
        ):
            raise ValueError(
                "hull equations must have shape "
                f"[N, {dimensions + 1}]"
            )
        valid_simplices = (
            None
            if xy_valid_simplices is None
            else np.asarray(xy_valid_simplices, dtype=bool)
        )
        if (xy_triangulation is None) != (valid_simplices is None):
            raise ValueError(
                "xy triangulation and valid-simplex mask must be provided together"
            )
        if valid_simplices is not None and (
            valid_simplices.ndim != 1
            or not hasattr(xy_triangulation, "simplices")
            or len(valid_simplices) != len(xy_triangulation.simplices)
            or int(getattr(xy_triangulation, "ndim", -1)) != dimensions
        ):
            raise ValueError("xy valid-simplex mask does not match triangulation")
        return cls(
            low=low_array,
            high=high_array,
            max_reach=float(max_reach),
            xy_hull_equations=equations,
            xy_triangulation=xy_triangulation,
            xy_valid_simplices=valid_simplices,
            interpolation_dimensions=dimensions,
        )

    def contains(self, points: np.ndarray, *, tolerance: float = 1e-7) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        bounded = np.all(
            (values >= self.low - tolerance) & (values <= self.high + tolerance),
            axis=-1,
        )
        reachable = np.linalg.norm(values, axis=-1) <= self.max_reach + tolerance
        if self.xy_hull_equations is None:
            inside_hull = True
        else:
            dimensions = self.interpolation_dimensions
            inside_hull = np.all(
                values[..., :dimensions]
                @ self.xy_hull_equations[:, :dimensions].T
                + self.xy_hull_equations[:, dimensions]
                <= tolerance,
                axis=-1,
            )
        if self.xy_triangulation is None:
            inside_valid_simplices = True
        else:
            dimensions = self.interpolation_dimensions
            original_shape = values.shape[:-1]
            flat_coordinates = values[..., :dimensions].reshape(-1, dimensions)
            simplex = self.xy_triangulation.find_simplex(
                flat_coordinates, tol=tolerance
            )
            inside_hull_flat = simplex >= 0
            inside = np.zeros(len(simplex), dtype=bool)
            inside[inside_hull_flat] = self.xy_valid_simplices[
                simplex[inside_hull_flat]
            ]
            # Qhull returns only one simplex on a shared edge. If that choice
            # is rejected, accept the point when an adjacent valid simplex
            # contains it within tolerance. This avoids false negatives caused
            # solely by which side of a certified boundary Qhull reports.
            unresolved = np.flatnonzero(inside_hull_flat & ~inside)
            if len(unresolved):
                valid_indices = np.flatnonzero(self.xy_valid_simplices)
                transforms = self.xy_triangulation.transform[valid_indices]
                for point_index in unresolved:
                    delta = (
                        flat_coordinates[point_index]
                        - transforms[:, dimensions]
                    )
                    first = np.einsum(
                        "nij,nj->ni", transforms[:, :dimensions], delta
                    )
                    weights = np.column_stack((first, 1.0 - np.sum(first, axis=1)))
                    inside[point_index] = bool(
                        np.any(np.all(weights >= -tolerance, axis=1))
                    )
            inside_valid_simplices = inside.reshape(original_shape)
        return bounded & reachable & inside_hull & inside_valid_simplices

    def clip(self, point: Sequence[float]) -> np.ndarray:
        clipped = np.clip(np.asarray(point, dtype=np.float64), self.low, self.high)
        if np.linalg.norm(clipped) > self.max_reach:
            # Project along a segment from a guaranteed feasible point. This
            # preserves both the box and spherical reach constraints.
            feasible = np.clip((self.low + self.high) / 2.0, self.low, self.high)
            if np.linalg.norm(feasible) > self.max_reach:
                raise ValueError("workspace box does not intersect the reach ball")
            lower, upper = 0.0, 1.0
            for _ in range(60):
                fraction = (lower + upper) / 2.0
                candidate = feasible + fraction * (clipped - feasible)
                if np.linalg.norm(candidate) <= self.max_reach:
                    lower = fraction
                else:
                    upper = fraction
            clipped = feasible + lower * (clipped - feasible)
        return clipped

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        for _ in range(10_000):
            point = rng.uniform(self.low, self.high)
            if bool(self.contains(point)):
                return point
        raise RuntimeError("could not sample a reachable point in the workspace")


@dataclass(frozen=True)
class ReferencePath:
    """A dense path with arc-length interpolation and human-readable anchors."""

    family: str
    anchors: np.ndarray
    points: np.ndarray
    cumulative_length: np.ndarray
    generation_mode: str = "direct"

    @classmethod
    def from_points(
        cls,
        family: str,
        anchors: np.ndarray,
        points: np.ndarray,
        *,
        generation_mode: str = "direct",
    ) -> "ReferencePath":
        anchors_array = np.asarray(anchors, dtype=np.float64)
        points_array = np.asarray(points, dtype=np.float64)
        if anchors_array.ndim != 2 or anchors_array.shape[1] != 3:
            raise ValueError("anchors must have shape [N, 3]")
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError("points must have shape [N, 3]")
        if len(points_array) < 2 or not np.isfinite(points_array).all():
            raise ValueError("a path needs at least two finite points")
        segment_lengths = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
        keep = np.concatenate(([True], segment_lengths > 1e-9))
        points_array = points_array[keep]
        if len(points_array) < 2:
            raise ValueError("path length must be nonzero")
        segment_lengths = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        return cls(
            family=str(family),
            anchors=anchors_array.astype(np.float32),
            points=points_array.astype(np.float32),
            cumulative_length=cumulative.astype(np.float64),
            generation_mode=str(generation_mode),
        )

    @property
    def length(self) -> float:
        return float(self.cumulative_length[-1])

    def sample(self, distance: float) -> np.ndarray:
        arc = float(np.clip(distance, 0.0, self.length))
        upper = int(np.searchsorted(self.cumulative_length, arc, side="right"))
        upper = min(max(upper, 1), len(self.points) - 1)
        lower = upper - 1
        start_arc = self.cumulative_length[lower]
        segment_length = self.cumulative_length[upper] - start_arc
        fraction = 0.0 if segment_length <= 0 else (arc - start_arc) / segment_length
        point = (1.0 - fraction) * self.points[lower] + fraction * self.points[upper]
        return np.asarray(point, dtype=np.float32)

    def project(
        self,
        point: Sequence[float],
        *,
        minimum_distance: float = 0.0,
        maximum_distance: float | None = None,
        coordinate_weights: Sequence[float] | None = None,
    ) -> tuple[float, float, np.ndarray]:
        """Project a point onto a bounded arc-length interval of the path.

        Bounding the interval prevents a self-near or returning path segment
        from falsely awarding a large amount of progress.  Optional coordinate
        weights change only the projection metric; returned progress remains
        the original three-dimensional arc length.  In particular, weights
        ``(1, 1, 0)`` let a table task advance geometrically in XY while the
        reference height is allowed to vary inside a separate tolerance band.
        """

        query = np.asarray(point, dtype=np.float64)
        if query.shape != (3,) or not np.isfinite(query).all():
            raise ValueError("projection point must be a finite 3-D value")
        weights = (
            np.ones(3, dtype=np.float64)
            if coordinate_weights is None
            else np.asarray(coordinate_weights, dtype=np.float64)
        )
        if (
            weights.shape != (3,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError(
                "coordinate_weights must be a finite non-negative 3-D value "
                "with at least one positive entry"
            )
        lower = float(np.clip(minimum_distance, 0.0, self.length))
        upper = self.length if maximum_distance is None else float(
            np.clip(maximum_distance, lower, self.length)
        )
        starts = np.asarray(self.points[:-1], dtype=np.float64)
        vectors = np.asarray(np.diff(self.points, axis=0), dtype=np.float64)
        lengths = np.linalg.norm(vectors, axis=1)
        segment_start = self.cumulative_length[:-1]
        segment_end = self.cumulative_length[1:]
        eligible = (segment_end >= lower - 1e-12) & (segment_start <= upper + 1e-12)
        if not np.any(eligible):
            projected = self.sample(lower)
            return lower, float(np.linalg.norm(query - projected)), projected
        indices = np.flatnonzero(eligible)
        candidate_starts = starts[indices]
        candidate_vectors = vectors[indices]
        candidate_lengths = lengths[indices]
        weighted_vectors = candidate_vectors * weights[None, :]
        weighted_offsets = (
            query[None, :] - candidate_starts
        ) * weights[None, :]
        weighted_squared_lengths = np.sum(
            np.square(weighted_vectors), axis=1
        )
        fractions = np.sum(
            weighted_offsets * weighted_vectors, axis=1
        ) / np.maximum(weighted_squared_lengths, 1e-18)
        arc_low = np.maximum(
            0.0, (lower - segment_start[indices]) / candidate_lengths
        )
        arc_high = np.minimum(
            1.0, (upper - segment_start[indices]) / candidate_lengths
        )
        fractions = np.clip(fractions, arc_low, arc_high)
        projected_points = candidate_starts + fractions[:, None] * candidate_vectors
        distances = np.linalg.norm(
            (projected_points - query[None, :]) * weights[None, :], axis=1
        )
        chosen = int(np.argmin(distances))
        distance_along_path = float(
            segment_start[indices[chosen]]
            + fractions[chosen] * candidate_lengths[chosen]
        )
        return (
            distance_along_path,
            float(distances[chosen]),
            np.asarray(projected_points[chosen], dtype=np.float32),
        )


def _densify_polyline(anchors: np.ndarray, spacing: float) -> np.ndarray:
    rows = [np.asarray(anchors[0], dtype=np.float64)]
    for start, end in zip(anchors[:-1], anchors[1:]):
        distance = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(distance / spacing)) + 1)
        rows.extend(np.linspace(start, end, count, endpoint=True)[1:])
    return np.asarray(rows, dtype=np.float64)


class WaypointPathGenerator:
    """Seedable path generator used by training, evaluation, and tests."""

    def __init__(
        self,
        workspace: WaypointWorkspace | None = None,
        *,
        dense_spacing: float = 0.005,
        waypoint_segment_count_range: Sequence[int] = (2, 4),
        waypoint_segment_count_probabilities: Sequence[float] | None = None,
        waypoint_segment_length_range: Sequence[float] = (0.015, 0.030),
        waypoint_first_segment_length_range: Sequence[float] | None = None,
        waypoint_maximum_extent: float = 0.045,
        waypoint_minimum_turn_degrees: float = 0.0,
        waypoint_maximum_turn_degrees: float = 135.0,
        waypoint_hard_turn_probability: float = 0.0,
        waypoint_hard_turn_range_degrees: Sequence[float] = (120.0, 175.0),
        waypoint_vertical_delta_range: Sequence[float] = (0.0, 0.0),
        waypoint_single_line_probability: float = 0.0,
    ) -> None:
        self.workspace = workspace or WaypointWorkspace.from_bounds()
        if dense_spacing <= 0:
            raise ValueError("dense_spacing must be positive")
        self.dense_spacing = float(dense_spacing)
        counts = np.asarray(waypoint_segment_count_range, dtype=np.int64)
        lengths = np.asarray(waypoint_segment_length_range, dtype=np.float64)
        first_lengths = (
            lengths.copy()
            if waypoint_first_segment_length_range is None
            else np.asarray(
                waypoint_first_segment_length_range, dtype=np.float64
            )
        )
        vertical = np.asarray(waypoint_vertical_delta_range, dtype=np.float64)
        hard_turns = np.asarray(
            waypoint_hard_turn_range_degrees, dtype=np.float64
        )
        if counts.shape != (2,) or counts[0] < 1 or counts[0] > counts[1]:
            raise ValueError(
                "waypoint_segment_count_range must be an increasing positive pair"
            )
        if lengths.shape != (2,) or lengths[0] <= 0 or lengths[0] > lengths[1]:
            raise ValueError(
                "waypoint_segment_length_range must be an increasing positive pair"
            )
        if (
            first_lengths.shape != (2,)
            or first_lengths[0] <= 0
            or first_lengths[0] > first_lengths[1]
        ):
            raise ValueError(
                "waypoint_first_segment_length_range must be an increasing "
                "positive pair"
            )
        if vertical.shape != (2,) or vertical[0] > vertical[1]:
            raise ValueError(
                "waypoint_vertical_delta_range must be an increasing pair"
            )
        if waypoint_maximum_extent < lengths[0]:
            raise ValueError(
                "waypoint_maximum_extent must cover the minimum segment length"
            )
        if not 0.0 <= waypoint_minimum_turn_degrees <= 180.0:
            raise ValueError(
                "waypoint_minimum_turn_degrees must lie in [0, 180]"
            )
        if not 0.0 < waypoint_maximum_turn_degrees <= 180.0:
            raise ValueError(
                "waypoint_maximum_turn_degrees must lie in (0, 180]"
            )
        if waypoint_minimum_turn_degrees > waypoint_maximum_turn_degrees:
            raise ValueError(
                "minimum waypoint turn cannot exceed the maximum"
            )
        if not 0.0 <= waypoint_hard_turn_probability <= 1.0:
            raise ValueError(
                "waypoint_hard_turn_probability must lie in [0, 1]"
            )
        if (
            hard_turns.shape != (2,)
            or hard_turns[0] < 0.0
            or hard_turns[0] > hard_turns[1]
            or hard_turns[1] > 180.0
        ):
            raise ValueError(
                "waypoint_hard_turn_range_degrees must be an increasing pair "
                "inside [0, 180]"
            )
        if not 0.0 <= waypoint_single_line_probability <= 1.0:
            raise ValueError(
                "waypoint_single_line_probability must lie in [0, 1]"
            )
        self.waypoint_segment_count_range = (int(counts[0]), int(counts[1]))
        count_values = np.arange(int(counts[0]), int(counts[1]) + 1)
        if waypoint_segment_count_probabilities is None:
            self.waypoint_segment_count_probabilities = None
        else:
            probabilities = np.asarray(
                waypoint_segment_count_probabilities, dtype=np.float64
            )
            if (
                probabilities.shape != count_values.shape
                or np.any(probabilities < 0)
                or not np.isfinite(probabilities).all()
                or float(np.sum(probabilities)) <= 0
            ):
                raise ValueError(
                    "waypoint_segment_count_probabilities must contain one "
                    "non-negative value per allowed segment count"
                )
            self.waypoint_segment_count_probabilities = (
                probabilities / np.sum(probabilities)
            )
        self.waypoint_segment_length_range = (
            float(lengths[0]),
            float(lengths[1]),
        )
        self.waypoint_first_segment_length_range = (
            float(first_lengths[0]),
            float(first_lengths[1]),
        )
        self.waypoint_maximum_extent = float(waypoint_maximum_extent)
        self.waypoint_minimum_turn_degrees = float(
            waypoint_minimum_turn_degrees
        )
        self.waypoint_maximum_turn_degrees = float(
            waypoint_maximum_turn_degrees
        )
        self.waypoint_hard_turn_probability = float(
            waypoint_hard_turn_probability
        )
        self.waypoint_hard_turn_range_degrees = (
            float(hard_turns[0]),
            float(hard_turns[1]),
        )
        self.waypoint_vertical_delta_range = (
            float(vertical[0]),
            float(vertical[1]),
        )
        self.waypoint_single_line_probability = float(
            waypoint_single_line_probability
        )
        self.last_generation_mode = "direct"

    def _segment_inside_workspace(
        self, start: np.ndarray, end: np.ndarray
    ) -> bool:
        """Check a straight segment in a potentially non-convex workspace."""

        distance = float(np.linalg.norm(end - start))
        # Pose-map triangles can form sub-millimetre slivers along a rejected
        # hole. A one-millimetre probe previously skipped a 0.45 mm invalid
        # interval and allowed a worker to fail only after training started.
        validation_spacing = min(self.dense_spacing, 0.0001)
        count = max(2, int(np.ceil(distance / validation_spacing)) + 1)
        points = np.linspace(start, end, count, endpoint=True)
        return bool(np.all(self.workspace.contains(points)))

    def _sample_separated(
        self,
        rng: np.random.Generator,
        origin: np.ndarray,
        minimum: float,
        maximum: float | None = None,
    ) -> np.ndarray:
        for _ in range(10_000):
            point = self.workspace.sample(rng)
            distance = float(np.linalg.norm(point - origin))
            if distance >= minimum and (maximum is None or distance <= maximum):
                return point
        raise RuntimeError("could not sample a sufficiently separated waypoint")

    def _sample_table_local(
        self,
        rng: np.random.Generator,
        origin: np.ndarray,
        minimum: float = 0.025,
        maximum: float = 0.10,
        vertical_low: float = -0.012,
        vertical_high: float = 0.008,
    ) -> np.ndarray:
        """Sample a short mostly-horizontal move around a table posture."""

        if vertical_low > vertical_high:
            raise ValueError("vertical bounds must be ordered")

        for _ in range(2_000):
            angle = rng.uniform(-np.pi, np.pi)
            radius = rng.uniform(minimum, maximum)
            # Descending a little is essential for reaching the side face of
            # a small object whose top is around z=0.44 m.  The workspace and
            # whole-arm table guard remain the hard physical safety limits.
            offset = np.asarray(
                [
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    rng.uniform(vertical_low, vertical_high),
                ]
            )
            candidate = origin + offset
            if vertical_low == 0.0 and vertical_high == 0.0:
                # Projection onto the spherical reach boundary can alter z.
                # Reject such candidates so the foundation curriculum remains
                # an actually planar task rather than merely a small-z task.
                if not bool(self.workspace.contains(candidate)):
                    continue
                target = candidate
            else:
                target = self.workspace.clip(candidate)
            distance = float(np.linalg.norm(target - origin))
            if (
                minimum <= distance <= maximum + 1e-8
                and self._segment_inside_workspace(origin, target)
            ):
                return target
        raise RuntimeError("could not sample a local table waypoint")

    def _local_point(
        self, rng: np.random.Generator, start: np.ndarray
    ) -> np.ndarray:
        # The first curriculum stage stays close to the natural settled pose,
        # including the high initial region. The next stage introduces the
        # full descent into the table workspace.
        local_low = np.array([-0.30, 0.0, 0.405], dtype=np.float64)
        local_high = np.array([0.30, 0.72, 0.98], dtype=np.float64)
        for _ in range(1_000):
            direction = rng.normal(size=3)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            target = np.clip(
                start + direction * rng.uniform(0.025, 0.08),
                local_low,
                local_high,
            )
            norm = float(np.linalg.norm(target))
            if norm > 0.98:
                target *= 0.98 / norm
            if np.linalg.norm(target - start) >= 0.02 and target[1] >= 0.0:
                return target
        return self.workspace.sample(rng)

    def _short_waypoint_polyline(
        self,
        rng: np.random.Generator,
        start: np.ndarray,
    ) -> np.ndarray:
        """Sample a feasible sequence of short, straight table-top segments.

        The full path remains close to its certified entry posture even though
        its cumulative arc length can be substantially longer. This matches a
        manipulation skill that visits several nearby points without asking
        the calibrated local action map for an unrealistic workspace sweep.
        """

        count_values = np.arange(
            self.waypoint_segment_count_range[0],
            self.waypoint_segment_count_range[1] + 1,
        )
        segment_count = int(
            rng.choice(
                count_values,
                p=self.waypoint_segment_count_probabilities,
            )
        )
        minimum_length, maximum_length = self.waypoint_segment_length_range
        first_minimum_length, first_maximum_length = (
            self.waypoint_first_segment_length_range
        )
        hard_turn_episode = bool(
            self.waypoint_hard_turn_probability > 0
            and rng.random() < self.waypoint_hard_turn_probability
        )
        if hard_turn_episode:
            minimum_turn, maximum_turn = np.deg2rad(
                self.waypoint_hard_turn_range_degrees
            )
        else:
            maximum_turn = np.deg2rad(self.waypoint_maximum_turn_degrees)
            minimum_turn = np.deg2rad(self.waypoint_minimum_turn_degrees)
        minimum_revisit = min(
            0.008, 0.45 * min(minimum_length, first_minimum_length)
        )

        # Random paths preserve distribution diversity, but a geometrically
        # tight request must reach the deterministic fallback quickly rather
        # than spending minutes in nested rejection loops during env.reset().
        for _ in range(16):
            rows = [np.asarray(start, dtype=np.float64)]
            heading: float | None = None
            for _segment in range(segment_count):
                accepted = False
                for _ in range(48):
                    if heading is None:
                        proposed_heading = rng.uniform(-np.pi, np.pi)
                    else:
                        proposed_heading = heading + rng.choice((-1.0, 1.0)) * rng.uniform(
                            minimum_turn, maximum_turn
                        )
                    segment_minimum = (
                        first_minimum_length if _segment == 0 else minimum_length
                    )
                    segment_maximum = (
                        first_maximum_length if _segment == 0 else maximum_length
                    )
                    distance = rng.uniform(segment_minimum, segment_maximum)
                    vertical_delta = rng.uniform(
                        self.waypoint_vertical_delta_range[0],
                        self.waypoint_vertical_delta_range[1],
                    )
                    candidate = rows[-1] + np.asarray(
                        [
                            distance * np.cos(proposed_heading),
                            distance * np.sin(proposed_heading),
                            vertical_delta,
                        ]
                    )
                    if not bool(self.workspace.contains(candidate)):
                        continue
                    if not self._segment_inside_workspace(rows[-1], candidate):
                        continue
                    if np.linalg.norm(candidate - start) > (
                        self.waypoint_maximum_extent + 1e-9
                    ):
                        continue
                    if len(rows) > 1 and np.min(
                        np.linalg.norm(np.asarray(rows[:-1]) - candidate, axis=1)
                    ) < minimum_revisit:
                        continue
                    rows.append(candidate)
                    heading = float(proposed_heading)
                    accepted = True
                    break
                if not accepted:
                    break
            if len(rows) == segment_count + 1:
                self.last_generation_mode = "random"
                return np.asarray(rows, dtype=np.float64)
        if minimum_length >= 0.08:
            workspace_fallback = self._workspace_chord_polyline_fallback(
                rng=rng,
                start=np.asarray(start, dtype=np.float64),
                segment_count=segment_count,
                minimum_turn=minimum_turn,
                maximum_turn=maximum_turn,
                minimum_length=minimum_length,
                maximum_length=maximum_length,
                first_minimum_length=first_minimum_length,
                first_maximum_length=first_maximum_length,
                minimum_revisit=minimum_revisit,
            )
            if workspace_fallback is not None:
                self.last_generation_mode = "workspace_chord_fallback"
                return workspace_fallback
        curved_fallback = self._randomized_curved_polyline_fallback(
            rng=rng,
            start=np.asarray(start, dtype=np.float64),
            segment_count=segment_count,
            minimum_turn=minimum_turn,
            maximum_turn=maximum_turn,
            minimum_length=minimum_length,
            maximum_length=maximum_length,
            first_minimum_length=first_minimum_length,
            first_maximum_length=first_maximum_length,
            minimum_revisit=minimum_revisit,
        )
        if curved_fallback is not None:
            self.last_generation_mode = "curved_fallback"
            return curved_fallback
        fallback = self._minimum_length_polyline_fallback(
            start=np.asarray(start, dtype=np.float64),
            segment_count=segment_count,
            minimum_turn=minimum_turn,
            maximum_turn=maximum_turn,
            minimum_length=minimum_length,
            first_minimum_length=first_minimum_length,
            minimum_revisit=minimum_revisit,
        )
        if fallback is not None:
            self.last_generation_mode = "deterministic_fallback"
            return fallback
        raise RuntimeError(
            "could not sample a feasible short waypoint polyline after random "
            "and deterministic fallback searches; check segment length, turn, "
            "extent, workspace, and reach constraints"
        )

    def _workspace_chord_polyline_fallback(
        self,
        *,
        rng: np.random.Generator,
        start: np.ndarray,
        segment_count: int,
        minimum_turn: float,
        maximum_turn: float,
        minimum_length: float,
        maximum_length: float,
        first_minimum_length: float,
        first_maximum_length: float,
        minimum_revisit: float,
    ) -> np.ndarray | None:
        """Sample long chords directly inside a bounded certified workspace.

        Direction-first rejection is efficient for centimetre-scale local
        steps, but not when a requested chord is comparable to the workspace
        diameter: most headings leave the convex hull. Sampling the endpoint
        first conditions on the certified hull and then enforces distance and
        turn limits. Multiple restarts provide lightweight stochastic
        backtracking for long out-and-back polylines.
        """

        vertical_low, vertical_high = self.waypoint_vertical_delta_range
        for _ in range(128):
            rows = [start.copy()]
            previous_heading: float | None = None
            for _segment in range(segment_count):
                accepted = False
                for _ in range(256):
                    candidate = self.workspace.sample(rng)
                    candidate[2] = rows[-1][2] + rng.uniform(
                        vertical_low, vertical_high
                    )
                    if not bool(self.workspace.contains(candidate)):
                        continue
                    if not self._segment_inside_workspace(rows[-1], candidate):
                        continue
                    displacement = candidate - rows[-1]
                    distance = float(np.linalg.norm(displacement))
                    segment_minimum = (
                        first_minimum_length if _segment == 0 else minimum_length
                    )
                    segment_maximum = (
                        first_maximum_length if _segment == 0 else maximum_length
                    )
                    if not (
                        segment_minimum - 1e-9
                        <= distance
                        <= segment_maximum + 1e-9
                    ):
                        continue
                    if np.linalg.norm(candidate - start) > (
                        self.waypoint_maximum_extent + 1e-9
                    ):
                        continue
                    heading = float(np.arctan2(displacement[1], displacement[0]))
                    if previous_heading is not None:
                        turn = abs(
                            float(
                                np.arctan2(
                                    np.sin(heading - previous_heading),
                                    np.cos(heading - previous_heading),
                                )
                            )
                        )
                        if not (
                            minimum_turn - 1e-9
                            <= turn
                            <= maximum_turn + 1e-9
                        ):
                            continue
                    if len(rows) > 1 and np.min(
                        np.linalg.norm(np.asarray(rows[:-1]) - candidate, axis=1)
                    ) < minimum_revisit:
                        continue
                    rows.append(candidate)
                    previous_heading = heading
                    accepted = True
                    break
                if not accepted:
                    break
            if len(rows) == segment_count + 1:
                return np.asarray(rows, dtype=np.float64)
        return None

    def _randomized_curved_polyline_fallback(
        self,
        *,
        rng: np.random.Generator,
        start: np.ndarray,
        segment_count: int,
        minimum_turn: float,
        maximum_turn: float,
        minimum_length: float,
        maximum_length: float,
        first_minimum_length: float,
        first_maximum_length: float,
        minimum_revisit: float,
    ) -> np.ndarray | None:
        """Sample diverse short arcs before using a deterministic fallback.

        A tight local radius can make independent uniform lengths and turns
        overwhelmingly infeasible.  For example, three 12 mm segments do not
        fit inside a 35 mm ball unless the path bends.  Biasing this fallback
        toward short, same-sense arcs preserves random headings and curvature
        instead of collapsing most low-turn curricula onto one template per
        entry posture.
        """

        workspace_center = (self.workspace.low + self.workspace.high) / 2.0
        inward = workspace_center[:2] - start[:2]
        inward_heading = float(np.arctan2(inward[1], inward[0]))
        vertical_low, vertical_high = self.waypoint_vertical_delta_range
        for _ in range(256):
            heading = float(
                rng.uniform(-np.pi, np.pi)
                if rng.random() < 0.35
                else inward_heading + rng.uniform(-np.pi / 2.0, np.pi / 2.0)
            )
            turn_sign = float(rng.choice((-1.0, 1.0)))
            turns = turn_sign * rng.uniform(
                max(minimum_turn, 0.55 * maximum_turn),
                maximum_turn,
                size=max(segment_count - 1, 0),
            )
            # Higher-turn curricula retain some S-bend variety.  At a tight
            # 30-degree limit an alternating bend generally cannot fit three
            # minimum-length segments in the configured local radius.
            if maximum_turn >= np.deg2rad(45.0) and rng.random() < 0.25:
                turns[1::2] *= -1.0
            lengths = minimum_length + (maximum_length - minimum_length) * rng.beta(
                1.0,
                5.0,
                size=segment_count,
            )
            lengths[0] = first_minimum_length + (
                first_maximum_length - first_minimum_length
            ) * rng.beta(1.0, 5.0)
            vertical_deltas = rng.uniform(
                vertical_low,
                vertical_high,
                size=segment_count,
            )
            rows = [start.copy()]
            feasible = True
            for segment_index in range(segment_count):
                if segment_index > 0:
                    heading += float(turns[segment_index - 1])
                candidate = rows[-1] + np.asarray(
                    [
                        lengths[segment_index] * np.cos(heading),
                        lengths[segment_index] * np.sin(heading),
                        vertical_deltas[segment_index],
                    ],
                    dtype=np.float64,
                )
                if not bool(self.workspace.contains(candidate)):
                    feasible = False
                    break
                if not self._segment_inside_workspace(rows[-1], candidate):
                    feasible = False
                    break
                if np.linalg.norm(candidate - start) > (
                    self.waypoint_maximum_extent + 1e-9
                ):
                    feasible = False
                    break
                if len(rows) > 1 and np.min(
                    np.linalg.norm(np.asarray(rows[:-1]) - candidate, axis=1)
                ) < minimum_revisit:
                    feasible = False
                    break
                rows.append(candidate)
            if feasible:
                return np.asarray(rows, dtype=np.float64)
        return None

    def _minimum_length_polyline_fallback(
        self,
        *,
        start: np.ndarray,
        segment_count: int,
        minimum_turn: float,
        maximum_turn: float,
        minimum_length: float,
        minimum_revisit: float,
        first_minimum_length: float | None = None,
    ) -> np.ndarray | None:
        """Construct a conservative path when rejection sampling is unlucky.

        Tight combinations such as three 12 mm segments, a 30 degree turn
        limit, and a 35 mm local radius have a very small feasible random
        volume. A single exhausted reset must not kill every worker in a long
        run, so try minimum-length, centre-biased arcs deterministically.
        """

        workspace_center = (self.workspace.low + self.workspace.high) / 2.0
        inward = workspace_center[:2] - start[:2]
        inward_heading = float(np.arctan2(inward[1], inward[0]))
        base_headings = inward_heading + np.linspace(
            0.0, 2.0 * np.pi, 96, endpoint=False
        )
        turn_magnitudes = tuple(
            value
            for value in (
                maximum_turn,
                0.75 * maximum_turn,
                0.5 * maximum_turn,
                0.25 * maximum_turn,
            )
            if value >= minimum_turn - 1e-9 and value > 1e-9
        )
        if not turn_magnitudes and minimum_turn <= maximum_turn:
            turn_magnitudes = (minimum_turn,)
        vertical_delta = float(
            np.clip(
                0.0,
                self.waypoint_vertical_delta_range[0],
                self.waypoint_vertical_delta_range[1],
            )
        )
        turn_count = max(segment_count - 1, 0)
        for turn_magnitude in turn_magnitudes:
            for turn_mask in range(1 << turn_count):
                for base_heading in base_headings:
                    rows = [start.copy()]
                    heading = float(base_heading)
                    feasible = True
                    for segment_index in range(segment_count):
                        if segment_index > 0:
                            sign = 1.0 if turn_mask & (1 << (segment_index - 1)) else -1.0
                            heading += sign * turn_magnitude
                        segment_length = (
                            (
                                minimum_length
                                if first_minimum_length is None
                                else first_minimum_length
                            )
                            if segment_index == 0
                            else minimum_length
                        )
                        candidate = rows[-1] + np.asarray(
                            [
                                segment_length * np.cos(heading),
                                segment_length * np.sin(heading),
                                vertical_delta,
                            ],
                            dtype=np.float64,
                        )
                        if not bool(self.workspace.contains(candidate)):
                            feasible = False
                            break
                        if not self._segment_inside_workspace(rows[-1], candidate):
                            feasible = False
                            break
                        if np.linalg.norm(candidate - start) > (
                            self.waypoint_maximum_extent + 1e-9
                        ):
                            feasible = False
                            break
                        if len(rows) > 1 and np.min(
                            np.linalg.norm(
                                np.asarray(rows[:-1]) - candidate,
                                axis=1,
                            )
                        ) < minimum_revisit:
                            feasible = False
                            break
                        rows.append(candidate)
                    if feasible:
                        return np.asarray(rows, dtype=np.float64)
        return None

    def _bezier(
        self, rng: np.random.Generator, start: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        end = self._sample_separated(rng, start, 0.12)
        chord = end - start
        horizontal = np.array([-chord[1], chord[0], 0.0])
        horizontal /= max(float(np.linalg.norm(horizontal)), 1e-12)
        bend = horizontal * rng.uniform(-0.10, 0.10)
        control_1 = self.workspace.clip(start + chord / 3.0 + bend)
        control_2 = self.workspace.clip(start + 2.0 * chord / 3.0 - bend)
        anchors = np.stack((start, control_1, control_2, end))
        count = max(40, int(np.ceil(np.linalg.norm(chord) / self.dense_spacing)) + 1)
        t = np.linspace(0.0, 1.0, count)[:, None]
        points = (
            (1.0 - t) ** 3 * start
            + 3.0 * (1.0 - t) ** 2 * t * control_1
            + 3.0 * (1.0 - t) * t ** 2 * control_2
            + t ** 3 * end
        )
        # Control points are in the convex workspace box and reach ball, so
        # the curve is bounded as well; clip only absorbs floating-point noise.
        points[1:] = np.asarray([self.workspace.clip(point) for point in points[1:]])
        return anchors, points

    def _s_curve(
        self, rng: np.random.Generator, start: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        end = self._sample_separated(rng, start, 0.14)
        chord = end - start
        normal = np.array([-chord[1], chord[0], 0.0])
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        amplitude = rng.uniform(0.025, 0.07) * rng.choice((-1.0, 1.0))
        count = max(50, int(np.ceil(np.linalg.norm(chord) / self.dense_spacing)) + 1)
        t = np.linspace(0.0, 1.0, count)
        points = start[None, :] + t[:, None] * chord[None, :]
        points += (amplitude * np.sin(2.0 * np.pi * t))[:, None] * normal[None, :]
        points[1:] = np.asarray([self.workspace.clip(point) for point in points[1:]])
        anchors = points[np.linspace(0, count - 1, 6).astype(int)]
        return anchors, points

    def generate(
        self,
        rng: np.random.Generator,
        start: Sequence[float],
        *,
        curriculum: str = "mixed",
        family: str | None = None,
        anchors: Sequence[Sequence[float]] | None = None,
    ) -> ReferencePath:
        if curriculum not in CURRICULUM_STAGES:
            raise ValueError(f"unknown curriculum stage: {curriculum}")
        start_array = np.asarray(start, dtype=np.float64)
        if start_array.shape != (3,) or not np.isfinite(start_array).all():
            raise ValueError("start must be a finite 3-D point")

        if anchors is not None:
            supplied = np.asarray(anchors, dtype=np.float64)
            if supplied.ndim != 2 or supplied.shape[1] != 3 or len(supplied) < 1:
                raise ValueError("anchors must have shape [N, 3]")
            if not np.all(self.workspace.contains(supplied)):
                raise ValueError("all supplied anchors must lie in the workspace")
            # Simulator-derived anchor banks can differ from the restored
            # float32 tip by sub-micrometre roundoff. Treat that as the same
            # start; prepending a near-duplicate would create a spurious gate.
            if np.linalg.norm(supplied[0] - start_array) > 1e-5:
                supplied = np.vstack((start_array, supplied))
            else:
                supplied = supplied.copy()
                supplied[0] = start_array
            points = _densify_polyline(supplied, self.dense_spacing)
            if (
                self.workspace.xy_triangulation is not None
                and not np.all(self.workspace.contains(points))
            ):
                raise ValueError(
                    "supplied anchor segments leave the certified workspace"
                )
            return ReferencePath.from_points(
                "custom", supplied, points, generation_mode="custom"
            )

        local_table = curriculum in {
            "table_local_line",
            "table_waypoint_polyline",
            "table_local",
            "entry_local",
            "path",
            "recovery",
        }
        if curriculum == "table_local_line":
            local_limit = 0.030
        elif curriculum == "table_waypoint_polyline":
            local_limit = self.waypoint_segment_length_range[1]
        elif curriculum == "table_local":
            local_limit = 0.055
        else:
            local_limit = 0.075
        if family is None:
            if curriculum == "point":
                family = "point"
            elif curriculum in {
                "entry_tail",
                "entry_bridge",
                "entry_mid",
                "entry",
                "table_point",
            }:
                family = "line"
            elif curriculum == "table_local_line":
                family = "line"
            elif curriculum == "table_waypoint_polyline":
                family = (
                    "line"
                    if rng.random() < self.waypoint_single_line_probability
                    else "waypoint_polyline"
                )
            elif curriculum in {"table_local", "entry_local", "path"}:
                family = str(rng.choice(("line", "polyline", "bezier", "s_curve")))
            elif curriculum == "recovery":
                family = str(rng.choice(("polyline", "reverse", "s_curve")))
            else:
                family = str(
                    rng.choice(
                        PATH_FAMILIES,
                        p=np.asarray((0.20, 0.25, 0.20, 0.20, 0.15)),
                    )
                )

        if family == "point":
            generation_mode = "point"
            target = self._local_point(rng, start_array)
            path_anchors = np.stack((start_array, target))
            points = _densify_polyline(path_anchors, self.dense_spacing)
        elif family == "line":
            generation_mode = "line"
            if curriculum in {"table_local_line", "table_waypoint_polyline"}:
                target = self._sample_table_local(
                    rng,
                    start_array,
                    (
                        0.018
                        if curriculum == "table_local_line"
                        else self.waypoint_segment_length_range[0]
                    ),
                    local_limit,
                    # The first local skill is deliberately planar.  The
                    # calibrated Cartesian action map holds height while SAC
                    # learns global table x/y tracking; small z excursions are
                    # introduced only after this foundation is reliable.
                    vertical_low=0.0,
                    vertical_high=0.0,
                )
            else:
                target = (
                    self._sample_table_local(rng, start_array, 0.020, local_limit)
                    if local_table
                    else self._sample_separated(rng, start_array, 0.08)
                )
            path_anchors = np.stack((start_array, target))
            points = _densify_polyline(path_anchors, self.dense_spacing)
        elif family in {"polyline", "waypoint_polyline"}:
            if curriculum == "table_waypoint_polyline":
                path_anchors = self._short_waypoint_polyline(rng, start_array)
                points = _densify_polyline(path_anchors, self.dense_spacing)
                family = "waypoint_polyline"
                generation_mode = self.last_generation_mode
            else:
                generation_mode = "polyline"
                count = int(rng.integers(3, 6) if local_table else rng.integers(4, 9))
                rows = [start_array]
                # Legacy paths may begin with a long descent. Physical curricula
                # are already at/after a certified entry and use short segments.
                rows.append(
                    self._sample_table_local(rng, rows[-1], 0.020, local_limit)
                    if local_table
                    else self._sample_separated(rng, rows[-1], 0.08)
                )
                for _ in range(count - 2):
                    rows.append(
                        self._sample_table_local(
                            rng, rows[-1], 0.020, local_limit
                        )
                        if local_table
                        else self._sample_separated(rng, rows[-1], 0.05, 0.22)
                    )
                path_anchors = np.asarray(rows)
                points = _densify_polyline(path_anchors, self.dense_spacing)
        elif family == "bezier":
            generation_mode = "bezier"
            if local_table:
                end = self._sample_table_local(
                    rng, start_array, 0.035, local_limit
                )
                chord = end - start_array
                normal = np.asarray([-chord[1], chord[0], 0.0])
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                bend = normal * rng.uniform(-0.025, 0.025)
                control_1 = self.workspace.clip(start_array + chord / 3.0 + bend)
                control_2 = self.workspace.clip(start_array + 2.0 * chord / 3.0 + bend)
                path_anchors = np.stack((start_array, control_1, control_2, end))
                t = np.linspace(0.0, 1.0, 40)[:, None]
                points = (
                    (1.0 - t) ** 3 * start_array
                    + 3.0 * (1.0 - t) ** 2 * t * control_1
                    + 3.0 * (1.0 - t) * t**2 * control_2
                    + t**3 * end
                )
            else:
                path_anchors, points = self._bezier(rng, start_array)
        elif family == "s_curve":
            generation_mode = "s_curve"
            if local_table:
                end = self._sample_table_local(
                    rng, start_array, 0.040, min(0.070, local_limit)
                )
                chord = end - start_array
                normal = np.asarray([-chord[1], chord[0], 0.0])
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                t = np.linspace(0.0, 1.0, 50)
                points = start_array[None, :] + t[:, None] * chord[None, :]
                points += (
                    rng.uniform(0.012, 0.025)
                    * rng.choice((-1.0, 1.0))
                    * np.sin(2.0 * np.pi * t)
                )[:, None] * normal[None, :]
                points = np.asarray([self.workspace.clip(row) for row in points])
                path_anchors = points[np.linspace(0, len(points) - 1, 6).astype(int)]
            else:
                path_anchors, points = self._s_curve(rng, start_array)
        elif family == "reverse":
            generation_mode = "reverse"
            outward = (
                self._sample_table_local(
                    rng, start_array, 0.040, local_limit
                )
                if local_table
                else self._sample_separated(rng, start_array, 0.12)
            )
            return_point = (
                self._sample_table_local(rng, outward, 0.025, local_limit)
                if local_table
                else self._sample_separated(rng, outward, 0.08, 0.22)
            )
            path_anchors = np.stack((start_array, outward, return_point, outward))
            points = _densify_polyline(path_anchors, self.dense_spacing)
        else:
            raise ValueError(f"unknown path family: {family}")

        return ReferencePath.from_points(
            family,
            path_anchors,
            points,
            generation_mode=generation_mode,
        )
