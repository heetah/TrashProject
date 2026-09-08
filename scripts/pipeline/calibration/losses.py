"""Robust Phase-4/5 losses for evaluating pseudo-homography candidates.

These losses describe temporal and local traffic-motion consistency after an
image-to-ground projection.  They are research diagnostics only: they do not
update the active transform or participate in litter attribution.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import os
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .motion_field import TrafficFlowCluster, robust_direction_summary
from .transform import (
    HomographyValidationConfig,
    project_image_points,
    validate_homography,
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class CalibrationLossConfig:
    """Dimensionless robust-loss settings; all component weights are explicit."""

    motion_weight: float = 1.0
    speed_weight: float = 1.0
    curvature_weight: float = 1.0
    direction_weight: float = 1.0
    lane_weight: float = 1.0
    motion_huber_delta: float = 0.50
    speed_huber_delta: float = 0.35
    curvature_huber_delta: float = math.radians(15.0)
    direction_huber_delta: float = math.radians(15.0)
    lane_huber_delta: float = 0.03
    direction_probe_fraction: float = 0.01
    lane_neighborhood_fraction: float = 0.10
    lane_min_peer_observations: int = 2
    lane_min_peer_tracks: int = 1
    min_track_points: int = 3
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        weights = (
            self.motion_weight,
            self.speed_weight,
            self.curvature_weight,
            self.direction_weight,
            self.lane_weight,
        )
        deltas = (
            self.motion_huber_delta,
            self.speed_huber_delta,
            self.curvature_huber_delta,
            self.direction_huber_delta,
            self.lane_huber_delta,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in weights):
            raise ValueError("calibration-loss weights must be finite and non-negative")
        if not any(value > 0.0 for value in weights):
            raise ValueError("at least one calibration-loss weight must be positive")
        if not all(math.isfinite(value) and value > 0.0 for value in deltas):
            raise ValueError("Huber deltas must be finite and positive")
        if not math.isfinite(self.direction_probe_fraction) or not (
            0.0 < self.direction_probe_fraction <= 0.25
        ):
            raise ValueError("direction_probe_fraction must be within (0, 0.25]")
        if not math.isfinite(self.lane_neighborhood_fraction) or not (
            0.0 < self.lane_neighborhood_fraction <= 1.0
        ):
            raise ValueError("lane_neighborhood_fraction must be within (0, 1]")
        if self.lane_min_peer_observations < 1 or self.lane_min_peer_tracks < 1:
            raise ValueError("lane peer support must be positive")
        if self.min_track_points < 3:
            raise ValueError("min_track_points must be at least three")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")

    @classmethod
    def from_env(cls) -> "CalibrationLossConfig":
        defaults = cls()
        return cls(
            motion_weight=_env_float(
                "DYNAMIC_HOMOGRAPHY_MOTION_WEIGHT", defaults.motion_weight
            ),
            speed_weight=_env_float(
                "DYNAMIC_HOMOGRAPHY_SPEED_WEIGHT", defaults.speed_weight
            ),
            curvature_weight=_env_float(
                "DYNAMIC_HOMOGRAPHY_CURVATURE_WEIGHT", defaults.curvature_weight
            ),
            direction_weight=_env_float(
                "DYNAMIC_HOMOGRAPHY_DIRECTION_WEIGHT", defaults.direction_weight
            ),
            lane_weight=_env_float(
                "DYNAMIC_HOMOGRAPHY_LANE_WEIGHT", defaults.lane_weight
            ),
            motion_huber_delta=_env_float(
                "DYNAMIC_HOMOGRAPHY_MOTION_HUBER_DELTA",
                defaults.motion_huber_delta,
            ),
            speed_huber_delta=_env_float(
                "DYNAMIC_HOMOGRAPHY_SPEED_HUBER_DELTA", defaults.speed_huber_delta
            ),
            curvature_huber_delta=_env_float(
                "DYNAMIC_HOMOGRAPHY_CURVATURE_HUBER_DELTA",
                defaults.curvature_huber_delta,
            ),
            direction_huber_delta=_env_float(
                "DYNAMIC_HOMOGRAPHY_DIRECTION_HUBER_DELTA",
                defaults.direction_huber_delta,
            ),
            lane_huber_delta=_env_float(
                "DYNAMIC_HOMOGRAPHY_LANE_HUBER_DELTA",
                defaults.lane_huber_delta,
            ),
            direction_probe_fraction=_env_float(
                "DYNAMIC_HOMOGRAPHY_DIRECTION_PROBE_FRAC",
                defaults.direction_probe_fraction,
            ),
            lane_neighborhood_fraction=_env_float(
                "DYNAMIC_HOMOGRAPHY_LANE_NEIGHBORHOOD_FRAC",
                defaults.lane_neighborhood_fraction,
            ),
            lane_min_peer_observations=_env_int(
                "DYNAMIC_HOMOGRAPHY_LANE_MIN_PEER_OBS",
                defaults.lane_min_peer_observations,
            ),
            lane_min_peer_tracks=_env_int(
                "DYNAMIC_HOMOGRAPHY_LANE_MIN_PEER_TRACKS",
                defaults.lane_min_peer_tracks,
            ),
            min_track_points=defaults.min_track_points,
            epsilon=defaults.epsilon,
        )


@dataclass(frozen=True)
class CalibrationLossReport:
    homography_valid: bool
    evaluable: bool
    reasons: Tuple[str, ...]
    total_loss: Optional[float]
    component_losses: Mapping[str, Optional[float]]
    weighted_losses: Mapping[str, Optional[float]]
    sample_counts: Mapping[str, int]
    active_weight_sum: float

    def as_dict(self) -> dict:
        return {
            "homography_valid": bool(self.homography_valid),
            "evaluable": bool(self.evaluable),
            "reasons": list(self.reasons),
            "total_loss": self.total_loss,
            "component_losses": dict(self.component_losses),
            "weighted_losses": dict(self.weighted_losses),
            "sample_counts": {
                key: int(value) for key, value in self.sample_counts.items()
            },
            "active_weight_sum": float(self.active_weight_sum),
            "scale_invariant": True,
            "robust_penalty": "huber",
            "affects_attribution": False,
        }


def _huber(residual: float, delta: float) -> float:
    value = abs(float(residual))
    if value <= delta:
        return 0.5 * value * value
    return delta * (value - 0.5 * delta)


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(value_array)
        & np.isfinite(weight_array)
        & (weight_array > 0.0)
    )
    if not np.any(valid):
        return None
    return float(np.average(value_array[valid], weights=weight_array[valid]))


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    order = np.argsort(value_array, kind="stable")
    ordered_values = value_array[order]
    ordered_weights = weight_array[order]
    threshold = float(np.sum(ordered_weights)) * 0.5
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _track_losses(
    points: Sequence[object],
    homography: np.ndarray,
    cfg: CalibrationLossConfig,
    validation_cfg: HomographyValidationConfig,
) -> Tuple[dict, dict]:
    if len(points) < cfg.min_track_points:
        return {name: [] for name in ("motion", "speed", "curvature")}, {
            name: [] for name in ("motion", "speed", "curvature")
        }
    image_points = np.asarray([point.image_point for point in points], dtype=float)
    projection = project_image_points(
        image_points,
        homography,
        min_abs_denominator=validation_cfg.min_abs_denominator,
        max_abs_coordinate=validation_cfg.max_abs_coordinate,
    )
    timestamps = np.asarray([float(point.timestamp) for point in points], dtype=float)
    qualities = np.asarray([float(point.quality_score) for point in points], dtype=float)
    dt = np.diff(timestamps)
    deltas = np.diff(projection.points, axis=0)
    segment_valid = (
        projection.valid_mask[:-1]
        & projection.valid_mask[1:]
        & np.isfinite(dt)
        & (dt > cfg.epsilon)
        & np.isfinite(deltas).all(axis=1)
    )
    velocities = np.full_like(deltas, np.nan, dtype=float)
    velocities[segment_valid] = deltas[segment_valid] / dt[segment_valid, None]
    speeds = np.linalg.norm(velocities, axis=1)
    positive_speeds = speeds[segment_valid & (speeds > cfg.epsilon)]
    if positive_speeds.size == 0:
        return {name: [] for name in ("motion", "speed", "curvature")}, {
            name: [] for name in ("motion", "speed", "curvature")
        }
    speed_scale = max(float(np.median(positive_speeds)), cfg.epsilon)

    losses = {name: [] for name in ("motion", "speed", "curvature")}
    weights = {name: [] for name in losses}
    for index in range(1, len(velocities)):
        if not (segment_valid[index - 1] and segment_valid[index]):
            continue
        quality = float(np.min(qualities[index - 1:index + 2]))
        motion_residual = float(
            np.linalg.norm(velocities[index] - velocities[index - 1]) / speed_scale
        )
        speed_residual = abs(math.log(
            (speeds[index] + cfg.epsilon) / (speeds[index - 1] + cfg.epsilon)
        ))
        losses["motion"].append(_huber(motion_residual, cfg.motion_huber_delta))
        weights["motion"].append(quality)
        losses["speed"].append(_huber(speed_residual, cfg.speed_huber_delta))
        weights["speed"].append(quality)

    turn_angles = []
    turn_weights = []
    for index in range(1, len(deltas)):
        if not (segment_valid[index - 1] and segment_valid[index]):
            turn_angles.append(None)
            turn_weights.append(0.0)
            continue
        left, right = deltas[index - 1], deltas[index]
        cross = float(left[0] * right[1] - left[1] * right[0])
        dot = float(left @ right)
        turn_angles.append(math.atan2(cross, dot))
        turn_weights.append(float(np.min(qualities[index - 1:index + 2])))
    for left, right, left_weight, right_weight in zip(
        turn_angles, turn_angles[1:], turn_weights, turn_weights[1:]
    ):
        if left is None or right is None:
            continue
        residual = abs(_wrap_angle(right - left))
        losses["curvature"].append(
            _huber(residual, cfg.curvature_huber_delta)
        )
        weights["curvature"].append(min(left_weight, right_weight))
    return losses, weights


def _project_local_observations(
    homography: np.ndarray,
    observations: Sequence[object],
    image_shape: Sequence[int],
    cfg: CalibrationLossConfig,
    validation_cfg: HomographyValidationConfig,
) -> Tuple[dict, dict]:
    height, width = int(image_shape[0]), int(image_shape[1])
    probe = math.hypot(width, height) * cfg.direction_probe_fraction
    transformed_points = {}
    transformed_directions = {}
    for index, observation in enumerate(observations):
        position = np.asarray(observation.position, dtype=float).reshape(2)
        direction = np.asarray(observation.direction, dtype=float).reshape(2)
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= cfg.epsilon:
            continue
        sample = np.vstack((position, position + direction / norm * probe))
        projected = project_image_points(
            sample,
            homography,
            min_abs_denominator=validation_cfg.min_abs_denominator,
            max_abs_coordinate=validation_cfg.max_abs_coordinate,
        )
        if not np.all(projected.valid_mask):
            continue
        delta = projected.points[1] - projected.points[0]
        delta_norm = float(np.linalg.norm(delta))
        if not math.isfinite(delta_norm) or delta_norm <= cfg.epsilon:
            continue
        transformed_points[index] = projected.points[0]
        transformed_directions[index] = delta / delta_norm
    return transformed_points, transformed_directions


def _direction_loss(
    observations: Sequence[object],
    clusters: Sequence[TrafficFlowCluster],
    transformed_directions: Mapping[int, np.ndarray],
    cfg: CalibrationLossConfig,
) -> Tuple[list, list]:

    losses = []
    weights = []
    for cluster in clusters:
        indices = [
            index for index in cluster.observation_indices
            if index in transformed_directions
        ]
        if len(indices) < 2:
            continue
        directions = [transformed_directions[index] for index in indices]
        qualities = [float(observations[index].quality) for index in indices]
        center, _ = robust_direction_summary(directions, qualities)
        center_norm = float(np.linalg.norm(center))
        if center_norm <= cfg.epsilon:
            continue
        for index, direction, quality in zip(indices, directions, qualities):
            residual = math.acos(np.clip(float(direction @ center), -1.0, 1.0))
            losses.append(_huber(residual, cfg.direction_huber_delta))
            weights.append(max(quality, 0.0) * max(float(cluster.confidence), cfg.epsilon))
    return losses, weights


def _lane_loss(
    observations: Sequence[object],
    clusters: Sequence[TrafficFlowCluster],
    transformed_points: Mapping[int, np.ndarray],
    transformed_directions: Mapping[int, np.ndarray],
    image_shape: Sequence[int],
    ground_scale: float,
    cfg: CalibrationLossConfig,
) -> Tuple[list, list]:
    """Leave-one-track-out moving-median centerline consistency.

    Neighborhood membership is fixed in normalized image coordinates so a
    candidate H cannot select easier peers.  Only the perpendicular residual
    to the robust local tangent is charged, preserving progress along a curved
    path and avoiding a global straight-line assumption.
    """

    height, width = int(image_shape[0]), int(image_shape[1])
    image_diagonal = max(math.hypot(width, height), cfg.epsilon)
    radius_px = max(image_diagonal * cfg.lane_neighborhood_fraction, cfg.epsilon)
    losses = []
    weights = []
    for cluster in clusters:
        members = [
            index for index in cluster.observation_indices
            if index in transformed_points and index in transformed_directions
        ]
        spatial_bins = defaultdict(list)
        image_positions = {}
        for index in members:
            position = np.asarray(observations[index].position, dtype=float).reshape(2)
            image_positions[index] = position
            spatial_bins[
                (int(math.floor(position[0] / radius_px)),
                 int(math.floor(position[1] / radius_px)))
            ].append(index)
        for index in members:
            observation = observations[index]
            current_key = (str(observation.class_name), int(observation.track_id))
            current_image = image_positions[index]
            cell = (
                int(math.floor(current_image[0] / radius_px)),
                int(math.floor(current_image[1] / radius_px)),
            )
            peers = []
            candidates = []
            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    candidates.extend(spatial_bins.get(
                        (cell[0] + offset_x, cell[1] + offset_y), ()
                    ))
            for candidate in candidates:
                if candidate == index:
                    continue
                peer = observations[candidate]
                peer_key = (str(peer.class_name), int(peer.track_id))
                if peer_key == current_key:
                    continue
                peer_image = image_positions[candidate]
                if (
                    float(np.linalg.norm(peer_image - current_image)) <= radius_px
                ):
                    peers.append(candidate)
            peer_tracks = {
                (str(observations[peer].class_name), int(observations[peer].track_id))
                for peer in peers
            }
            if (
                len(peers) < cfg.lane_min_peer_observations
                or len(peer_tracks) < cfg.lane_min_peer_tracks
            ):
                continue
            peer_qualities = [max(float(observations[peer].quality), 0.0) for peer in peers]
            if not any(value > 0.0 for value in peer_qualities):
                continue
            center = np.asarray([
                _weighted_median(
                    [transformed_points[peer][axis] for peer in peers],
                    peer_qualities,
                )
                for axis in (0, 1)
            ])
            tangent, _ = robust_direction_summary(
                [transformed_directions[peer] for peer in peers], peer_qualities
            )
            tangent_norm = float(np.linalg.norm(tangent))
            if tangent_norm <= cfg.epsilon:
                continue
            tangent = tangent / tangent_norm
            offset = transformed_points[index] - center
            perpendicular = abs(float(offset[0] * tangent[1] - offset[1] * tangent[0]))
            residual = perpendicular / max(float(ground_scale), cfg.epsilon)
            losses.append(_huber(residual, cfg.lane_huber_delta))
            support_quality = float(np.median(peer_qualities))
            weights.append(
                max(float(observation.quality), 0.0)
                * support_quality
                * max(float(cluster.confidence), cfg.epsilon)
            )
    return losses, weights


def evaluate_calibration_losses(
    homography: object,
    tracks: Mapping[object, Sequence[object]],
    image_shape: Sequence[int],
    *,
    motion_observations: Sequence[object] = (),
    flow_clusters: Sequence[TrafficFlowCluster] = (),
    config: Optional[CalibrationLossConfig] = None,
    validation_config: Optional[HomographyValidationConfig] = None,
) -> CalibrationLossReport:
    """Evaluate a candidate H using scale-resistant, quality-weighted losses."""

    cfg = config or CalibrationLossConfig()
    validation_cfg = validation_config or HomographyValidationConfig()
    validation = validate_homography(homography, image_shape, validation_cfg)
    names = ("motion", "speed", "curvature", "direction", "lane")
    empty_losses = {name: None for name in names}
    empty_counts = {name: 0 for name in names}
    if not validation.valid:
        return CalibrationLossReport(
            homography_valid=False,
            evaluable=False,
            reasons=tuple(validation.reasons),
            total_loss=None,
            component_losses=empty_losses,
            weighted_losses=empty_losses.copy(),
            sample_counts=empty_counts,
            active_weight_sum=0.0,
        )

    values = {name: [] for name in names}
    qualities = {name: [] for name in names}
    for points in tracks.values():
        track_values, track_weights = _track_losses(
            points, validation.normalized_homography, cfg, validation_cfg
        )
        for name in ("motion", "speed", "curvature"):
            values[name].extend(track_values[name])
            qualities[name].extend(track_weights[name])
    transformed_points, transformed_directions = _project_local_observations(
        validation.normalized_homography,
        motion_observations,
        image_shape,
        cfg,
        validation_cfg,
    )
    direction_values, direction_weights = _direction_loss(
        motion_observations,
        flow_clusters,
        transformed_directions,
        cfg,
    )
    values["direction"].extend(direction_values)
    qualities["direction"].extend(direction_weights)
    ground_scale = math.sqrt(max(
        abs(float(validation.projected_signed_area or 0.0)), cfg.epsilon
    ))
    lane_values, lane_weights = _lane_loss(
        motion_observations,
        flow_clusters,
        transformed_points,
        transformed_directions,
        image_shape,
        ground_scale,
        cfg,
    )
    values["lane"].extend(lane_values)
    qualities["lane"].extend(lane_weights)

    component_losses = {
        name: _weighted_mean(values[name], qualities[name]) for name in names
    }
    configured_weights = {
        "motion": cfg.motion_weight,
        "speed": cfg.speed_weight,
        "curvature": cfg.curvature_weight,
        "direction": cfg.direction_weight,
        "lane": cfg.lane_weight,
    }
    weighted_losses = {
        name: (
            None
            if component_losses[name] is None
            else float(configured_weights[name] * component_losses[name])
        )
        for name in names
    }
    active = [
        name for name in names
        if component_losses[name] is not None and configured_weights[name] > 0.0
    ]
    active_weight_sum = float(sum(configured_weights[name] for name in active))
    total_loss = (
        float(sum(weighted_losses[name] for name in active) / active_weight_sum)
        if active_weight_sum > 0.0
        else None
    )
    reasons = () if total_loss is not None else ("insufficient_motion_evidence",)
    return CalibrationLossReport(
        homography_valid=True,
        evaluable=total_loss is not None,
        reasons=reasons,
        total_loss=total_loss,
        component_losses=component_losses,
        weighted_losses=weighted_losses,
        sample_counts={name: len(values[name]) for name in names},
        active_weight_sum=active_weight_sum,
    )
