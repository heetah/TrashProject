"""Phase-1 observations for online pseudo-homography self-calibration.

Dynamic vehicles are observations of long-term traffic geometry, never static
point correspondences.  This module deliberately stops before estimating or
updating a homography: it extracts robust mask contact points, rejects weak
measurements, stores lightweight track metadata and emits local motion
segments.  Later calibration phases can consume these immutable observations
without retaining video frames.
"""

from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
import math
import os
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
class CalibrationPhase1Config:
    """Configurable evidence thresholds for the collection-only first phase."""

    enabled: bool = False
    window_seconds: float = 100.0
    bottom_percentile: float = 98.0
    min_mask_area: float = 64.0
    border_margin_fraction: float = 0.01
    min_confidence: float = 0.45
    quality_threshold: float = 0.60
    min_track_points: int = 3
    min_track_path_length_px: float = 6.0
    min_local_displacement_px: float = 1.0
    max_speed_diagonals_per_second: float = 2.0
    max_acceleration_diagonals_per_second2: float = 4.0
    max_local_turn_degrees: float = 135.0
    max_track_gap_seconds: float = 2.0
    min_mask_bbox_fill_ratio: float = 0.10
    max_points_per_track: int = 600
    max_tracks: int = 2048

    def __post_init__(self) -> None:
        numeric = (
            self.window_seconds,
            self.bottom_percentile,
            self.min_mask_area,
            self.border_margin_fraction,
            self.min_confidence,
            self.quality_threshold,
            self.min_track_path_length_px,
            self.min_local_displacement_px,
            self.max_speed_diagonals_per_second,
            self.max_acceleration_diagonals_per_second2,
            self.max_local_turn_degrees,
            self.max_track_gap_seconds,
            self.min_mask_bbox_fill_ratio,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("calibration config values must be finite")
        if self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        if not 0.0 <= self.bottom_percentile <= 100.0:
            raise ValueError("bottom_percentile must be within 0..100")
        if self.min_mask_area <= 0.0:
            raise ValueError("min_mask_area must be positive")
        if not 0.0 <= self.border_margin_fraction < 0.5:
            raise ValueError("border_margin_fraction must be within [0, 0.5)")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be within 0..1")
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be within 0..1")
        if self.min_track_points < 2:
            raise ValueError("min_track_points must be at least two")
        if self.min_track_path_length_px < 0.0 or self.min_local_displacement_px < 0.0:
            raise ValueError("motion distances must be non-negative")
        if self.max_speed_diagonals_per_second <= 0.0:
            raise ValueError("max speed must be positive")
        if self.max_acceleration_diagonals_per_second2 <= 0.0:
            raise ValueError("max acceleration must be positive")
        if not 0.0 < self.max_local_turn_degrees <= 180.0:
            raise ValueError("max local turn must be within (0, 180]")
        if self.max_track_gap_seconds <= 0.0:
            raise ValueError("max track gap must be positive")
        if not 0.0 <= self.min_mask_bbox_fill_ratio <= 1.0:
            raise ValueError("mask bbox fill ratio must be within 0..1")
        if self.max_points_per_track < 2 or self.max_tracks < 1:
            raise ValueError("buffer limits are too small")

    @classmethod
    def from_env(cls) -> "CalibrationPhase1Config":
        defaults = cls()
        return cls(
            enabled=_env_bool("DYNAMIC_HOMOGRAPHY", defaults.enabled),
            window_seconds=_env_float(
                "DYNAMIC_HOMOGRAPHY_WINDOW_SEC", defaults.window_seconds
            ),
            bottom_percentile=_env_float(
                "DYNAMIC_HOMOGRAPHY_BOTTOM_PERCENTILE", defaults.bottom_percentile
            ),
            min_mask_area=_env_float(
                "DYNAMIC_HOMOGRAPHY_MIN_MASK_AREA", defaults.min_mask_area
            ),
            border_margin_fraction=_env_float(
                "DYNAMIC_HOMOGRAPHY_BORDER_MARGIN_FRAC",
                defaults.border_margin_fraction,
            ),
            min_confidence=_env_float(
                "DYNAMIC_HOMOGRAPHY_MIN_CONF", defaults.min_confidence
            ),
            quality_threshold=_env_float(
                "DYNAMIC_HOMOGRAPHY_QUALITY_THRESHOLD",
                defaults.quality_threshold,
            ),
            min_track_points=_env_int(
                "DYNAMIC_HOMOGRAPHY_MIN_TRACK_POINTS", defaults.min_track_points
            ),
            min_track_path_length_px=_env_float(
                "DYNAMIC_HOMOGRAPHY_MIN_TRACK_PATH_PX",
                defaults.min_track_path_length_px,
            ),
            min_local_displacement_px=_env_float(
                "DYNAMIC_HOMOGRAPHY_MIN_LOCAL_MOTION_PX",
                defaults.min_local_displacement_px,
            ),
            max_speed_diagonals_per_second=_env_float(
                "DYNAMIC_HOMOGRAPHY_MAX_SPEED_DIAG_SEC",
                defaults.max_speed_diagonals_per_second,
            ),
            max_acceleration_diagonals_per_second2=_env_float(
                "DYNAMIC_HOMOGRAPHY_MAX_ACCEL_DIAG_SEC2",
                defaults.max_acceleration_diagonals_per_second2,
            ),
            max_local_turn_degrees=_env_float(
                "DYNAMIC_HOMOGRAPHY_MAX_LOCAL_TURN_DEG",
                defaults.max_local_turn_degrees,
            ),
            max_track_gap_seconds=_env_float(
                "DYNAMIC_HOMOGRAPHY_MAX_TRACK_GAP_SEC",
                defaults.max_track_gap_seconds,
            ),
            min_mask_bbox_fill_ratio=_env_float(
                "DYNAMIC_HOMOGRAPHY_MIN_MASK_BBOX_FILL",
                defaults.min_mask_bbox_fill_ratio,
            ),
            max_points_per_track=_env_int(
                "DYNAMIC_HOMOGRAPHY_MAX_POINTS_PER_TRACK",
                defaults.max_points_per_track,
            ),
            max_tracks=_env_int(
                "DYNAMIC_HOMOGRAPHY_MAX_TRACKS", defaults.max_tracks
            ),
        )


def _mask_pixels(mask: object) -> Tuple[np.ndarray, np.ndarray]:
    """Return global x/y pixels for either a binary mask or polygon vertices."""

    try:
        values = np.asarray(mask)
    except (TypeError, ValueError):
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    if values.ndim == 2 and values.shape[1] == 2:
        polygon = np.asarray(values, dtype=np.float32)
        if polygon.shape[0] < 3 or not np.isfinite(polygon).all():
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        minimum = np.floor(np.min(polygon, axis=0)).astype(int)
        maximum = np.ceil(np.max(polygon, axis=0)).astype(int)
        width, height = (maximum - minimum + 3).tolist()
        if width <= 1 or height <= 1 or width * height > 20_000_000:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        local = np.zeros((height, width), dtype=np.uint8)
        shifted = np.rint(polygon - minimum + 1).astype(np.int32)
        cv2.fillPoly(local, [shifted], 1)
        ys, xs = np.where(local > 0)
        return xs.astype(float) + minimum[0] - 1, ys.astype(float) + minimum[1] - 1
    if values.ndim == 2:
        ys, xs = np.where(values > 0)
        return xs.astype(float), ys.astype(float)
    return np.empty(0, dtype=float), np.empty(0, dtype=float)


def extract_vehicle_ground_point(
    mask: object,
    bottom_percentile: float = 98.0,
) -> Optional[np.ndarray]:
    """Estimate a robust vehicle contact point from the bottom mask band.

    The median of the lowest percentile band is intentionally used instead of
    the single lowest pixel, which is highly sensitive to segmentation noise.
    Ultralytics polygon masks and dense binary masks are both accepted.
    """

    xs, ys = _mask_pixels(mask)
    return _ground_point_from_pixels(xs, ys, bottom_percentile)


def _ground_point_from_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    bottom_percentile: float,
) -> Optional[np.ndarray]:
    if xs.size == 0:
        return None
    threshold = float(np.percentile(ys, np.clip(bottom_percentile, 0.0, 100.0)))
    selected = ys >= threshold
    if not np.any(selected):
        return None
    point = np.asarray(
        [np.median(xs[selected]), np.median(ys[selected])], dtype=np.float32
    )
    return point if np.isfinite(point).all() else None


@dataclass(frozen=True)
class VehicleTrackPoint:
    track_id: int
    timestamp: float
    frame_id: int
    image_point: np.ndarray
    velocity: np.ndarray
    quality_score: float
    mask_area: float
    confidence: float
    class_name: str = "vehicle"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "image_point", np.asarray(self.image_point, dtype=float).reshape(2)
        )
        object.__setattr__(
            self, "velocity", np.asarray(self.velocity, dtype=float).reshape(2)
        )


@dataclass(frozen=True)
class LocalMotionObservation:
    position: np.ndarray
    direction: np.ndarray
    speed: float
    track_id: int
    quality: float
    frame_id: int
    class_name: str = "vehicle"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "position", np.asarray(self.position, dtype=float).reshape(2)
        )
        object.__setattr__(
            self, "direction", np.asarray(self.direction, dtype=float).reshape(2)
        )


class CalibrationBuffer:
    """Rolling lightweight per-track metadata; no images or masks are retained."""

    def __init__(self, window_seconds=100.0, max_points_per_track=600, max_tracks=2048):
        self.window_seconds = float(window_seconds)
        self.max_points_per_track = max(int(max_points_per_track), 2)
        self.max_tracks = max(int(max_tracks), 1)
        self._tracks: "OrderedDict[Tuple[str, int], Deque[VehicleTrackPoint]]" = OrderedDict()
        self._last_purge_timestamp = float("-inf")

    def add(self, point: VehicleTrackPoint) -> None:
        key = (str(point.class_name), int(point.track_id))
        track = self._tracks.get(key)
        if track is None:
            if len(self._tracks) >= self.max_tracks:
                self._tracks.popitem(last=False)
            track = deque(maxlen=self.max_points_per_track)
            self._tracks[key] = track
        else:
            self._tracks.move_to_end(key)
        if track and point.timestamp <= track[-1].timestamp:
            return
        track.append(point)
        if point.timestamp - self._last_purge_timestamp >= 1.0:
            self.purge(point.timestamp)

    def purge(self, timestamp: float) -> None:
        self._last_purge_timestamp = float(timestamp)
        cutoff = float(timestamp) - self.window_seconds
        for key in list(self._tracks):
            track = self._tracks[key]
            while track and track[0].timestamp < cutoff:
                track.popleft()
            if not track:
                del self._tracks[key]

    def tracks(self) -> Dict[Tuple[str, int], Tuple[VehicleTrackPoint, ...]]:
        return {key: tuple(points) for key, points in self._tracks.items()}

    def get_track(self, key: Tuple[str, int]) -> Tuple[VehicleTrackPoint, ...]:
        return tuple(self._tracks.get(key, ()))

    def reset_track(self, key: Tuple[str, int]) -> None:
        self._tracks.pop(key, None)

    @property
    def point_count(self) -> int:
        return sum(len(points) for points in self._tracks.values())

    @property
    def track_count(self) -> int:
        return len(self._tracks)


@dataclass
class VehicleTrajectoryCollector:
    """Convert observed YOLO-Seg vehicles into filtered calibration evidence."""

    config: CalibrationPhase1Config = field(default_factory=CalibrationPhase1Config)
    buffer: CalibrationBuffer = field(init=False)
    received: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    _image_shape: Optional[Tuple[int, int]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.buffer = CalibrationBuffer(
            window_seconds=self.config.window_seconds,
            max_points_per_track=self.config.max_points_per_track,
            max_tracks=self.config.max_tracks,
        )

    @staticmethod
    def _track_path_length(points: Sequence[VehicleTrackPoint]) -> float:
        return float(sum(
            np.linalg.norm(right.image_point - left.image_point)
            for left, right in zip(points, points[1:])
        ))

    def _reject(self, reason: str) -> None:
        self.rejected[str(reason)] += 1

    def add_vehicle_observation(
        self,
        track_id: int,
        timestamp: float,
        frame_id: int,
        mask: object,
        confidence: float,
        bbox: Optional[Sequence[float]] = None,
        image_shape: Optional[Sequence[int]] = None,
        *,
        class_name: str = "vehicle",
        observed: bool = True,
        occluded: bool = False,
        tracking_stable: bool = True,
    ) -> Optional[VehicleTrackPoint]:
        self.received += 1
        if not observed:
            self._reject("cached_observation")
            return None
        if bool(occluded):
            self._reject("occluded_observation")
            return None
        if not bool(tracking_stable):
            self._reject("unstable_track_id")
            return None
        try:
            confidence = float(confidence)
            timestamp = float(timestamp)
            frame_id = int(frame_id)
            track_id = int(track_id)
        except (TypeError, ValueError):
            self._reject("invalid_metadata")
            return None
        if not all(math.isfinite(value) for value in (confidence, timestamp)):
            self._reject("invalid_metadata")
            return None
        if confidence < self.config.min_confidence:
            self._reject("low_confidence")
            return None

        resolved_image_shape = None
        if image_shape is not None:
            try:
                resolved_image_shape = (int(image_shape[0]), int(image_shape[1]))
            except (TypeError, ValueError, IndexError):
                self._reject("invalid_image_shape")
                return None
            if resolved_image_shape[0] <= 0 or resolved_image_shape[1] <= 0:
                self._reject("invalid_image_shape")
                return None
            if self._image_shape is None:
                self._image_shape = resolved_image_shape
            elif resolved_image_shape != self._image_shape:
                self._reject("image_shape_changed")
                return None

        xs, ys = _mask_pixels(mask)
        mask_area = float(xs.size)
        if mask_area < self.config.min_mask_area:
            self._reject("mask_too_small")
            return None
        ground_point = _ground_point_from_pixels(
            xs, ys, self.config.bottom_percentile
        )
        if ground_point is None:
            self._reject("invalid_mask")
            return None

        if bbox is not None and resolved_image_shape is not None:
            try:
                x1, y1, x2, y2 = map(float, bbox[:4])
                height, width = resolved_image_shape
                margin_x = width * self.config.border_margin_fraction
                margin_y = height * self.config.border_margin_fraction
                if (
                    x1 <= margin_x or y1 <= margin_y
                    or x2 >= width - margin_x or y2 >= height - margin_y
                ):
                    self._reject("image_border")
                    return None
                bbox_area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
                if (
                    bbox_area <= 0.0
                    or mask_area / bbox_area < self.config.min_mask_bbox_fill_ratio
                ):
                    self._reject("low_mask_bbox_fill")
                    return None
            except (TypeError, ValueError, IndexError):
                self._reject("invalid_bbox")
                return None

        key = (str(class_name), track_id)
        previous_track = self.buffer.get_track(key)
        if (
            previous_track
            and timestamp - previous_track[-1].timestamp
            > self.config.max_track_gap_seconds
        ):
            self.buffer.reset_track(key)
            previous_track = ()
        velocity = np.zeros(2, dtype=float)
        continuity_score = 1.0
        acceleration_score = 1.0
        if previous_track:
            previous = previous_track[-1]
            dt = timestamp - previous.timestamp
            if dt <= 0.0 or frame_id <= previous.frame_id:
                self._reject("non_monotonic_track")
                return None
            velocity = (ground_point.astype(float) - previous.image_point) / dt
            if resolved_image_shape is not None:
                diagonal = math.hypot(
                    float(resolved_image_shape[1]), float(resolved_image_shape[0])
                )
                normalized_speed = float(np.linalg.norm(velocity)) / max(diagonal, 1.0)
                if normalized_speed > self.config.max_speed_diagonals_per_second:
                    self._reject("teleport_motion")
                    return None
                if len(previous_track) >= 2:
                    acceleration = float(
                        np.linalg.norm(velocity - previous.velocity)
                    ) / max(dt * diagonal, 1e-9)
                    if acceleration > self.config.max_acceleration_diagonals_per_second2:
                        self._reject("unstable_ground_point")
                        return None
                    acceleration_score = max(
                        0.0,
                        1.0 - acceleration
                        / self.config.max_acceleration_diagonals_per_second2,
                    )
                    previous_delta = (
                        previous.image_point - previous_track[-2].image_point
                    )
                    previous_norm = float(np.linalg.norm(previous_delta))
                    current_delta = ground_point.astype(float) - previous.image_point
                    current_norm = float(np.linalg.norm(current_delta))
                    if previous_norm > 1e-9 and current_norm > 1e-9:
                        turn = math.degrees(math.acos(np.clip(
                            float(previous_delta @ current_delta)
                            / (previous_norm * current_norm),
                            -1.0,
                            1.0,
                        )))
                        if turn > self.config.max_local_turn_degrees:
                            self._reject("extreme_local_turn")
                            return None
                continuity_score = max(
                    0.0,
                    1.0 - normalized_speed / self.config.max_speed_diagonals_per_second,
                )

        confidence_score = np.clip(
            (confidence - self.config.min_confidence)
            / max(1.0 - self.config.min_confidence, 1e-9),
            0.0,
            1.0,
        )
        area_score = min(mask_area / max(4.0 * self.config.min_mask_area, 1.0), 1.0)
        quality_score = float(np.mean((
            confidence_score,
            area_score,
            continuity_score,
            acceleration_score,
        )))
        if quality_score < self.config.quality_threshold:
            self._reject("quality_below_threshold")
            return None

        point = VehicleTrackPoint(
            track_id=track_id,
            timestamp=timestamp,
            frame_id=frame_id,
            image_point=ground_point,
            velocity=velocity,
            quality_score=quality_score,
            mask_area=mask_area,
            confidence=confidence,
            class_name=str(class_name),
        )
        self.buffer.add(point)
        self.accepted += 1
        return point

    def local_motion_observations(self) -> List[LocalMotionObservation]:
        result: List[LocalMotionObservation] = []
        for (class_name, track_id), points_tuple in self.buffer.tracks().items():
            points = list(points_tuple)
            if len(points) < self.config.min_track_points:
                continue
            if self._track_path_length(points) < self.config.min_track_path_length_px:
                continue
            for left, right in zip(points, points[1:]):
                delta = right.image_point - left.image_point
                displacement = float(np.linalg.norm(delta))
                dt = float(right.timestamp - left.timestamp)
                if displacement < self.config.min_local_displacement_px or dt <= 0.0:
                    continue
                result.append(LocalMotionObservation(
                    position=(left.image_point + right.image_point) * 0.5,
                    direction=delta / displacement,
                    speed=displacement / dt,
                    track_id=track_id,
                    quality=min(left.quality_score, right.quality_score),
                    frame_id=right.frame_id,
                    class_name=class_name,
                ))
        return result

    def traffic_motion_field(self, observations=None):
        if self._image_shape is None:
            return None
        from .motion_field import MotionFieldConfig, TrafficMotionField

        return TrafficMotionField(MotionFieldConfig.from_env()).build(
            observations if observations is not None else self.local_motion_observations(),
            self._image_shape,
        )

    def traffic_flow_clusters(self, observations=None):
        if self._image_shape is None:
            return [], ()
        from .motion_field import FlowClusteringConfig, cluster_traffic_flows

        return cluster_traffic_flows(
            observations if observations is not None else self.local_motion_observations(),
            self._image_shape,
            FlowClusteringConfig.from_env(),
        )

    def summary(self, *, run_optimizer: bool = True) -> dict:
        tracks = self.buffer.tracks()
        valid_tracks = sum(
            1 for points in tracks.values()
            if len(points) >= self.config.min_track_points
            and self._track_path_length(points) >= self.config.min_track_path_length_px
        )
        motion_observations = self.local_motion_observations()
        motion_field = self.traffic_motion_field(motion_observations)
        flow_clusters, flow_noise = self.traffic_flow_clusters(motion_observations)
        if self._image_shape is None:
            initial_transform = None
            calibration_losses = None
            optimization = None
        else:
            from .losses import CalibrationLossConfig, evaluate_calibration_losses
            from .optimizer import (
                HomographyOptimizerConfig,
                estimate_candidate_homography,
            )
            from .transform import (
                HomographyValidationConfig,
                initial_homography_snapshot,
                initial_normalized_image_homography,
            )

            initial_transform = initial_homography_snapshot(self._image_shape)
            initial_homography = initial_normalized_image_homography(self._image_shape)
            loss_config = CalibrationLossConfig.from_env()
            validation_config = HomographyValidationConfig.from_env()
            calibration_losses = evaluate_calibration_losses(
                initial_homography,
                tracks,
                self._image_shape,
                motion_observations=motion_observations,
                flow_clusters=flow_clusters,
                config=loss_config,
                validation_config=validation_config,
            ).as_dict()
            optimizer_config = HomographyOptimizerConfig.from_env()
            if optimizer_config.enabled and run_optimizer:
                optimization = estimate_candidate_homography(
                    tracks,
                    motion_observations,
                    flow_clusters,
                    motion_field,
                    self._image_shape,
                    initial_homography,
                    loss_config=loss_config,
                    optimizer_config=optimizer_config,
                    validation_config=validation_config,
                ).as_dict()
            else:
                optimization = {
                    "enabled": False,
                    "update_recommended": False,
                    "reasons": [
                        "optimizer_disabled"
                        if not optimizer_config.enabled
                        else "delegated_to_state_machine"
                    ],
                    "applied_to_production": False,
                    "affects_attribution": False,
                }
        return {
            "schema": "dynamic-homography-observations/v2",
            "enabled": bool(self.config.enabled),
            "phase": "motion_field_and_flow_clustering",
            "implemented_through_phase": 6,
            "status": "COLLECTING" if self.config.enabled else "DISABLED",
            "affects_attribution": False,
            "window_seconds": float(self.config.window_seconds),
            "received_observations": int(self.received),
            "accepted_observations": int(self.accepted),
            "rejected_observations": {
                key: int(value) for key, value in sorted(self.rejected.items())
            },
            "buffered_points": int(self.buffer.point_count),
            "buffered_tracks": int(len(tracks)),
            "valid_motion_tracks": int(valid_tracks),
            "local_motion_observations": int(len(motion_observations)),
            "motion_field": (
                motion_field.summary() if motion_field is not None else None
            ),
            "flow_cluster_count": int(len(flow_clusters)),
            "flow_noise_observations": int(len(flow_noise)),
            "flow_clusters": [
                {
                    "cluster_id": int(cluster.cluster_id),
                    "sample_count": int(len(cluster.observation_indices)),
                    "track_count": int(len(cluster.track_keys)),
                    "mean_direction": [
                        float(value) for value in cluster.mean_direction
                    ],
                    "spatial_region": [
                        float(value) for value in cluster.spatial_region
                    ],
                    "direction_variance": float(cluster.direction_variance),
                    "confidence": float(cluster.confidence),
                }
                for cluster in flow_clusters
            ],
            "initial_transform": initial_transform,
            "calibration_losses": calibration_losses,
            "optimization": optimization,
        }
