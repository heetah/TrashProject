"""Robust spatial motion field and lightweight traffic-flow clustering.

The algorithms operate on Phase-1 local motion metadata only.  They neither
estimate a homography nor alter event attribution.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class MotionFieldConfig:
    grid_rows: int = 12
    grid_cols: int = 20
    min_cell_samples: int = 3
    direction_trim_fraction: float = 0.15
    confidence_sample_target: int = 12

    def __post_init__(self) -> None:
        if self.grid_rows < 1 or self.grid_cols < 1:
            raise ValueError("motion-field grid dimensions must be positive")
        if self.min_cell_samples < 1 or self.confidence_sample_target < 1:
            raise ValueError("motion-field sample counts must be positive")
        if not 0.0 <= self.direction_trim_fraction < 0.5:
            raise ValueError("direction trim fraction must be within [0, 0.5)")

    @classmethod
    def from_env(cls) -> "MotionFieldConfig":
        defaults = cls()
        return cls(
            grid_rows=_env_int("DYNAMIC_HOMOGRAPHY_GRID_ROWS", defaults.grid_rows),
            grid_cols=_env_int("DYNAMIC_HOMOGRAPHY_GRID_COLS", defaults.grid_cols),
            min_cell_samples=_env_int(
                "DYNAMIC_HOMOGRAPHY_MIN_CELL_SAMPLES", defaults.min_cell_samples
            ),
            direction_trim_fraction=_env_float(
                "DYNAMIC_HOMOGRAPHY_DIRECTION_TRIM_FRAC",
                defaults.direction_trim_fraction,
            ),
            confidence_sample_target=_env_int(
                "DYNAMIC_HOMOGRAPHY_CELL_SAMPLE_TARGET",
                defaults.confidence_sample_target,
            ),
        )


@dataclass(frozen=True)
class FlowClusteringConfig:
    position_radius_fraction: float = 0.10
    max_direction_angle_deg: float = 30.0
    continuity_radius_fraction: float = 0.25
    max_continuity_angle_deg: float = 55.0
    min_samples: int = 4
    min_tracks: int = 2
    confidence_sample_target: int = 20

    def __post_init__(self) -> None:
        numeric = (
            self.position_radius_fraction,
            self.max_direction_angle_deg,
            self.continuity_radius_fraction,
            self.max_continuity_angle_deg,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("flow-clustering values must be finite")
        if not 0.0 < self.position_radius_fraction <= 1.0:
            raise ValueError("position radius must be within (0, 1]")
        if not 0.0 < self.continuity_radius_fraction <= 2.0:
            raise ValueError("continuity radius must be within (0, 2]")
        if not 0.0 < self.max_direction_angle_deg <= 180.0:
            raise ValueError("direction angle must be within (0, 180]")
        if not 0.0 < self.max_continuity_angle_deg <= 180.0:
            raise ValueError("continuity angle must be within (0, 180]")
        if self.min_samples < 2 or self.min_tracks < 1:
            raise ValueError("flow cluster support is too small")
        if self.confidence_sample_target < self.min_samples:
            raise ValueError("cluster sample target must cover min_samples")

    @classmethod
    def from_env(cls) -> "FlowClusteringConfig":
        defaults = cls()
        return cls(
            position_radius_fraction=_env_float(
                "DYNAMIC_HOMOGRAPHY_FLOW_POSITION_RADIUS",
                defaults.position_radius_fraction,
            ),
            max_direction_angle_deg=_env_float(
                "DYNAMIC_HOMOGRAPHY_FLOW_ANGLE_DEG",
                defaults.max_direction_angle_deg,
            ),
            continuity_radius_fraction=_env_float(
                "DYNAMIC_HOMOGRAPHY_FLOW_CONTINUITY_RADIUS",
                defaults.continuity_radius_fraction,
            ),
            max_continuity_angle_deg=_env_float(
                "DYNAMIC_HOMOGRAPHY_FLOW_CONTINUITY_ANGLE_DEG",
                defaults.max_continuity_angle_deg,
            ),
            min_samples=_env_int(
                "DYNAMIC_HOMOGRAPHY_FLOW_MIN_SAMPLES", defaults.min_samples
            ),
            min_tracks=_env_int(
                "DYNAMIC_HOMOGRAPHY_FLOW_MIN_TRACKS", defaults.min_tracks
            ),
            confidence_sample_target=_env_int(
                "DYNAMIC_HOMOGRAPHY_FLOW_SAMPLE_TARGET",
                defaults.confidence_sample_target,
            ),
        )


@dataclass(frozen=True)
class MotionFieldCell:
    row: int
    col: int
    sample_count: int
    dominant_direction: np.ndarray
    direction_variance: float
    median_speed: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dominant_direction",
            np.asarray(self.dominant_direction, dtype=float).reshape(2),
        )


@dataclass(frozen=True)
class TrafficFlowCluster:
    cluster_id: int
    observation_indices: Tuple[int, ...]
    track_keys: Tuple[Tuple[str, int], ...]
    mean_direction: np.ndarray
    spatial_region: Tuple[float, float, float, float]
    direction_variance: float
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "mean_direction", np.asarray(self.mean_direction, dtype=float).reshape(2)
        )


def _unit_direction(value: object) -> Optional[np.ndarray]:
    try:
        direction = np.asarray(value, dtype=float).reshape(2)
    except (TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return direction / norm


def _angular_distance(first: np.ndarray, second: np.ndarray) -> float:
    return float(math.acos(np.clip(float(first @ second), -1.0, 1.0)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = float(np.sum(sorted_weights)) * 0.5
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def robust_direction_summary(
    directions: Sequence[object],
    qualities: Optional[Sequence[float]] = None,
    trim_fraction: float = 0.15,
) -> Tuple[np.ndarray, float]:
    """Return a circular-residual-trimmed direction and circular variance."""

    valid = []
    valid_weights = []
    raw_weights = qualities if qualities is not None else [1.0] * len(directions)
    for direction, weight in zip(directions, raw_weights):
        unit = _unit_direction(direction)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            continue
        if unit is not None and math.isfinite(weight) and weight > 0.0:
            valid.append(unit)
            valid_weights.append(weight)
    if not valid:
        return np.zeros(2, dtype=float), 1.0

    vectors = np.asarray(valid, dtype=float)
    weights = np.asarray(valid_weights, dtype=float)
    # Circular medoid is robust to wraparound and opposite-direction outliers.
    pairwise = np.arccos(np.clip(vectors @ vectors.T, -1.0, 1.0))
    medoid = vectors[int(np.argmin(pairwise @ weights))]
    residuals = np.arccos(np.clip(vectors @ medoid, -1.0, 1.0))
    keep_count = max(1, int(math.ceil(len(vectors) * (1.0 - trim_fraction))))
    keep = np.argsort(residuals, kind="stable")[:keep_count]
    resultant = np.sum(vectors[keep] * weights[keep, None], axis=0)
    resultant_norm = float(np.linalg.norm(resultant))
    if resultant_norm <= 1e-9:
        direction = medoid
        concentration = 0.0
    else:
        direction = resultant / resultant_norm
        concentration = resultant_norm / max(float(np.sum(weights[keep])), 1e-9)
    return direction, float(np.clip(1.0 - concentration, 0.0, 1.0))


class TrafficMotionField:
    def __init__(self, config: Optional[MotionFieldConfig] = None):
        self.config = config or MotionFieldConfig()
        self.cells: Dict[Tuple[int, int], MotionFieldCell] = {}
        self.image_shape: Optional[Tuple[int, int]] = None

    def build(self, observations: Iterable[object], image_shape: Sequence[int]):
        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_shape must be positive")
        self.image_shape = (height, width)
        grouped = defaultdict(list)
        for observation in observations:
            try:
                position = np.asarray(observation.position, dtype=float).reshape(2)
                direction = _unit_direction(observation.direction)
                speed = float(observation.speed)
                quality = float(observation.quality)
            except (AttributeError, TypeError, ValueError):
                continue
            if (
                direction is None or not np.isfinite(position).all()
                or not math.isfinite(speed) or speed < 0.0
                or not math.isfinite(quality) or quality <= 0.0
                or position[0] < 0.0 or position[0] >= width
                or position[1] < 0.0 or position[1] >= height
            ):
                continue
            row = min(int(position[1] / height * self.config.grid_rows), self.config.grid_rows - 1)
            col = min(int(position[0] / width * self.config.grid_cols), self.config.grid_cols - 1)
            grouped[(row, col)].append((direction, speed, quality))

        cells = {}
        for (row, col), values in grouped.items():
            if len(values) < self.config.min_cell_samples:
                continue
            directions = [value[0] for value in values]
            speeds = np.asarray([value[1] for value in values], dtype=float)
            qualities = np.asarray([value[2] for value in values], dtype=float)
            dominant, variance = robust_direction_summary(
                directions, qualities, self.config.direction_trim_fraction
            )
            median_speed = _weighted_median(speeds, qualities)
            sample_score = min(
                len(values) / float(self.config.confidence_sample_target), 1.0
            )
            confidence = float(np.clip(
                sample_score * float(np.median(qualities)) * (1.0 - variance),
                0.0,
                1.0,
            ))
            cells[(row, col)] = MotionFieldCell(
                row=row,
                col=col,
                sample_count=len(values),
                dominant_direction=dominant,
                direction_variance=variance,
                median_speed=median_speed,
                confidence=confidence,
            )
        self.cells = cells
        return self

    @property
    def spatial_coverage(self) -> float:
        total = self.config.grid_rows * self.config.grid_cols
        return len(self.cells) / float(total)

    def summary(self) -> dict:
        return {
            "grid_rows": int(self.config.grid_rows),
            "grid_cols": int(self.config.grid_cols),
            "active_cells": int(len(self.cells)),
            "spatial_coverage": float(self.spatial_coverage),
            "median_cell_confidence": float(np.median([
                cell.confidence for cell in self.cells.values()
            ])) if self.cells else 0.0,
        }


def cluster_traffic_flows(
    observations: Sequence[object],
    image_shape: Sequence[int],
    config: Optional[FlowClusteringConfig] = None,
) -> Tuple[List[TrafficFlowCluster], Tuple[int, ...]]:
    """Cluster local motions by position, direction and track continuity.

    This is a deterministic, grid-indexed DBSCAN variant. It avoids a new ML
    dependency and avoids the quadratic all-pairs scan for typical traffic.
    """

    cfg = config or FlowClusteringConfig()
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")

    normalized_positions = []
    directions = []
    qualities = []
    track_keys = []
    frame_ids = []
    source_indices = []
    for source_index, observation in enumerate(observations):
        try:
            position = np.asarray(observation.position, dtype=float).reshape(2)
            direction = _unit_direction(observation.direction)
            quality = float(observation.quality)
            track_key = (str(observation.class_name), int(observation.track_id))
            frame_id = int(observation.frame_id)
        except (AttributeError, TypeError, ValueError):
            continue
        if (
            direction is None or not np.isfinite(position).all()
            or not math.isfinite(quality) or quality <= 0.0
        ):
            continue
        normalized_positions.append(np.asarray([position[0] / width, position[1] / height]))
        directions.append(direction)
        qualities.append(quality)
        track_keys.append(track_key)
        frame_ids.append(frame_id)
        source_indices.append(source_index)
    count = len(source_indices)
    if count == 0:
        return [], ()

    positions = np.asarray(normalized_positions, dtype=float)
    directions_array = np.asarray(directions, dtype=float)
    bin_size = cfg.position_radius_fraction
    spatial_bins = defaultdict(list)
    for index, position in enumerate(positions):
        spatial_bins[(int(math.floor(position[0] / bin_size)), int(math.floor(position[1] / bin_size)))].append(index)

    continuity_neighbors = defaultdict(set)
    by_track = defaultdict(list)
    for index, key in enumerate(track_keys):
        by_track[key].append(index)
    continuity_angle = math.radians(cfg.max_continuity_angle_deg)
    for indices in by_track.values():
        ordered = sorted(indices, key=lambda index: frame_ids[index])
        for left, right in zip(ordered, ordered[1:]):
            if (
                np.linalg.norm(positions[right] - positions[left])
                <= cfg.continuity_radius_fraction
                and _angular_distance(directions_array[left], directions_array[right])
                <= continuity_angle
            ):
                continuity_neighbors[left].add(right)
                continuity_neighbors[right].add(left)

    max_angle = math.radians(cfg.max_direction_angle_deg)

    def region_query(index: int) -> List[int]:
        position = positions[index]
        cell = (int(math.floor(position[0] / bin_size)), int(math.floor(position[1] / bin_size)))
        candidates = set(continuity_neighbors[index])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.update(spatial_bins.get((cell[0] + dx, cell[1] + dy), ()))
        return sorted(candidate for candidate in candidates if (
            candidate == index
            or candidate in continuity_neighbors[index]
            or (
                np.linalg.norm(positions[candidate] - position)
                <= cfg.position_radius_fraction
                and _angular_distance(directions_array[index], directions_array[candidate])
                <= max_angle
            )
        ))

    labels = np.full(count, -2, dtype=int)  # -2 unvisited, -1 noise
    cluster_id = 0
    for index in range(count):
        if labels[index] != -2:
            continue
        neighbors = region_query(index)
        if len(neighbors) < cfg.min_samples:
            labels[index] = -1
            continue
        labels[index] = cluster_id
        queue_items = deque(neighbors)
        queued = set(neighbors)
        while queue_items:
            neighbor = queue_items.popleft()
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            if labels[neighbor] != -2:
                continue
            labels[neighbor] = cluster_id
            expanded = region_query(neighbor)
            if len(expanded) >= cfg.min_samples:
                for candidate in expanded:
                    if candidate not in queued:
                        queued.add(candidate)
                        queue_items.append(candidate)
        cluster_id += 1

    clusters = []
    retained_indices = set()
    for raw_cluster_id in range(cluster_id):
        members = np.flatnonzero(labels == raw_cluster_id).tolist()
        member_tracks = sorted({track_keys[index] for index in members})
        if len(members) < cfg.min_samples or len(member_tracks) < cfg.min_tracks:
            continue
        member_directions = [directions_array[index] for index in members]
        member_qualities = [qualities[index] for index in members]
        mean_direction, variance = robust_direction_summary(
            member_directions, member_qualities
        )
        member_positions = positions[members]
        min_xy = np.min(member_positions, axis=0)
        max_xy = np.max(member_positions, axis=0)
        sample_score = min(len(members) / float(cfg.confidence_sample_target), 1.0)
        track_score = min(len(member_tracks) / max(float(cfg.min_tracks * 2), 1.0), 1.0)
        confidence = float(np.clip(
            sample_score * track_score * np.median(member_qualities) * (1.0 - variance),
            0.0,
            1.0,
        ))
        output_id = len(clusters)
        source_members = tuple(sorted(source_indices[index] for index in members))
        retained_indices.update(source_members)
        clusters.append(TrafficFlowCluster(
            cluster_id=output_id,
            observation_indices=source_members,
            track_keys=tuple(member_tracks),
            mean_direction=mean_direction,
            spatial_region=(
                float(min_xy[0]), float(min_xy[1]),
                float(max_xy[0]), float(max_xy[1]),
            ),
            direction_variance=variance,
            confidence=confidence,
        ))
    noise = tuple(sorted(set(source_indices) - retained_indices))
    return clusters, noise


def render_motion_debug(
    frame: np.ndarray,
    observations: Sequence[object],
    motion_field: Optional[TrafficMotionField] = None,
    clusters: Sequence[TrafficFlowCluster] = (),
) -> np.ndarray:
    """Render local segments, flow clusters and cell arrows on a copy."""

    canvas = np.asarray(frame).copy()
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("debug frame must be an HxWx3 image")
    height, width = canvas.shape[:2]
    palette = (
        (255, 120, 40), (40, 180, 255), (80, 210, 90), (210, 80, 210),
        (60, 220, 220), (220, 150, 60), (120, 80, 240), (180, 220, 80),
    )
    index_to_cluster = {}
    for cluster in clusters:
        for index in cluster.observation_indices:
            index_to_cluster[int(index)] = int(cluster.cluster_id)
    for index, observation in enumerate(observations):
        try:
            position = np.asarray(observation.position, dtype=float).reshape(2)
            direction = _unit_direction(observation.direction)
            speed = float(observation.speed)
        except (AttributeError, TypeError, ValueError):
            continue
        if direction is None or not np.isfinite(position).all():
            continue
        color = (
            palette[index_to_cluster[index] % len(palette)]
            if index in index_to_cluster else (128, 128, 128)
        )
        start = tuple(np.rint(position).astype(int))
        length = float(np.clip(8.0 + math.log1p(max(speed, 0.0)) * 3.0, 8.0, 28.0))
        end = tuple(np.rint(position + direction * length).astype(int))
        cv2.circle(canvas, start, 2, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, start, end, color, 1, cv2.LINE_AA, tipLength=0.3)

    if motion_field is not None and motion_field.cells:
        rows = motion_field.config.grid_rows
        cols = motion_field.config.grid_cols
        arrow_length = max(8.0, min(width / cols, height / rows) * 0.35)
        for cell in motion_field.cells.values():
            center = np.asarray([
                (cell.col + 0.5) * width / cols,
                (cell.row + 0.5) * height / rows,
            ])
            start = tuple(np.rint(center).astype(int))
            end = tuple(np.rint(
                center + cell.dominant_direction * arrow_length
            ).astype(int))
            strength = int(round(80 + 175 * cell.confidence))
            cv2.arrowedLine(
                canvas, start, end, (0, strength, 255), 2, cv2.LINE_AA,
                tipLength=0.3,
            )

    label = "calibration phase2: motions={} cells={} flows={}".format(
        len(observations),
        len(motion_field.cells) if motion_field is not None else 0,
        len(clusters),
    )
    cv2.putText(
        canvas, label, (10, max(20, int(height * 0.04))),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, label, (10, max(20, int(height * 0.04))),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA,
    )
    return canvas
