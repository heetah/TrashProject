"""Persistent Phase-7 dynamic pseudo-homography calibration state machine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np

from .homography_phase1 import CalibrationPhase1Config, VehicleTrajectoryCollector
from .losses import CalibrationLossConfig, evaluate_calibration_losses
from .optimizer import (
    HomographyOptimizationResult,
    HomographyOptimizerConfig,
    estimate_candidate_homography,
)
from .stabilization import (
    BackgroundStabilizerConfig,
    StabilizationResult,
    StaticBackgroundStabilizer,
    compose_runtime_homography,
)
from .transform import (
    HomographyValidationConfig,
    image_to_ground as transform_image_to_ground,
    initial_normalized_image_homography,
)


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


class CalibrationStatus(str, Enum):
    UNCALIBRATED = "UNCALIBRATED"
    COLLECTING = "COLLECTING"
    ESTIMATING = "ESTIMATING"
    WARMING_UP = "WARMING_UP"
    LOCKED = "LOCKED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    RECALIBRATING = "RECALIBRATING"


@dataclass(frozen=True)
class CalibrationStateConfig:
    evaluation_interval_seconds: float = 30.0
    min_successful_updates_to_lock: int = 2
    stable_windows_to_lock: int = 2
    drift_persistence_windows: int = 3
    drift_residual_ratio: float = 2.0
    drift_absolute_increase: float = 0.05
    drift_centroid_fraction: float = 0.08
    drift_spread_fraction: float = 0.08
    drift_direction_tensor: float = 0.25
    drift_background_motion: float = 0.03
    residual_baseline_alpha: float = 0.10
    confidence_alpha: float = 0.25
    drift_confidence_decay: float = 0.80
    rollback_history_size: int = 8
    metrics_history_size: int = 100

    def __post_init__(self) -> None:
        positive = (
            self.evaluation_interval_seconds,
            self.drift_residual_ratio,
            self.drift_absolute_increase,
            self.drift_centroid_fraction,
            self.drift_spread_fraction,
            self.drift_direction_tensor,
            self.drift_background_motion,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("state-machine positive thresholds must be finite")
        counts = (
            self.min_successful_updates_to_lock,
            self.stable_windows_to_lock,
            self.drift_persistence_windows,
            self.rollback_history_size,
            self.metrics_history_size,
        )
        if not all(value >= 1 for value in counts):
            raise ValueError("state-machine counts must be positive")
        if not 0.0 < self.residual_baseline_alpha <= 1.0:
            raise ValueError("residual_baseline_alpha must be within (0, 1]")
        if not 0.0 < self.confidence_alpha <= 1.0:
            raise ValueError("confidence_alpha must be within (0, 1]")
        if not 0.0 < self.drift_confidence_decay <= 1.0:
            raise ValueError("drift_confidence_decay must be within (0, 1]")

    @classmethod
    def from_env(cls) -> "CalibrationStateConfig":
        defaults = cls()
        return cls(
            evaluation_interval_seconds=_env_float(
                "DYNAMIC_HOMOGRAPHY_EVAL_INTERVAL_SEC",
                defaults.evaluation_interval_seconds,
            ),
            min_successful_updates_to_lock=_env_int(
                "DYNAMIC_HOMOGRAPHY_LOCK_MIN_UPDATES",
                defaults.min_successful_updates_to_lock,
            ),
            stable_windows_to_lock=_env_int(
                "DYNAMIC_HOMOGRAPHY_LOCK_STABLE_WINDOWS",
                defaults.stable_windows_to_lock,
            ),
            drift_persistence_windows=_env_int(
                "DYNAMIC_HOMOGRAPHY_DRIFT_WINDOWS",
                defaults.drift_persistence_windows,
            ),
            drift_residual_ratio=_env_float(
                "DYNAMIC_HOMOGRAPHY_DRIFT_RESIDUAL_RATIO",
                defaults.drift_residual_ratio,
            ),
            drift_absolute_increase=_env_float(
                "DYNAMIC_HOMOGRAPHY_DRIFT_ABS_INCREASE",
                defaults.drift_absolute_increase,
            ),
            drift_centroid_fraction=_env_float(
                "DYNAMIC_HOMOGRAPHY_DRIFT_CENTROID_FRAC",
                defaults.drift_centroid_fraction,
            ),
            drift_spread_fraction=_env_float(
                "DYNAMIC_HOMOGRAPHY_DRIFT_SPREAD_FRAC",
                defaults.drift_spread_fraction,
            ),
            drift_direction_tensor=_env_float(
                "DYNAMIC_HOMOGRAPHY_DRIFT_DIRECTION_TENSOR",
                defaults.drift_direction_tensor,
            ),
            drift_background_motion=_env_float(
                "DYNAMIC_HOMOGRAPHY_DRIFT_BACKGROUND_MOTION",
                defaults.drift_background_motion,
            ),
            residual_baseline_alpha=_env_float(
                "DYNAMIC_HOMOGRAPHY_RESIDUAL_BASELINE_ALPHA",
                defaults.residual_baseline_alpha,
            ),
            confidence_alpha=_env_float(
                "DYNAMIC_HOMOGRAPHY_CONFIDENCE_ALPHA",
                defaults.confidence_alpha,
            ),
            drift_confidence_decay=_env_float(
                "DYNAMIC_HOMOGRAPHY_DRIFT_CONFIDENCE_DECAY",
                defaults.drift_confidence_decay,
            ),
            rollback_history_size=_env_int(
                "DYNAMIC_HOMOGRAPHY_ROLLBACK_HISTORY",
                defaults.rollback_history_size,
            ),
            metrics_history_size=_env_int(
                "DYNAMIC_HOMOGRAPHY_METRICS_HISTORY",
                defaults.metrics_history_size,
            ),
        )


@dataclass(frozen=True)
class MotionGeometrySignature:
    centroid: np.ndarray
    spread: np.ndarray
    direction_tensor: np.ndarray
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "centroid", np.asarray(self.centroid, dtype=float).reshape(2))
        object.__setattr__(self, "spread", np.asarray(self.spread, dtype=float).reshape(2))
        object.__setattr__(self, "direction_tensor", np.asarray(self.direction_tensor, dtype=float).reshape(2, 2))


@dataclass(frozen=True)
class EventHomographySnapshot:
    timestamp: float
    homography_version: int
    confidence: float
    status: str
    image_shape: Tuple[int, int]
    calibration_homography: np.ndarray
    current_to_reference: np.ndarray
    runtime_homography: np.ndarray
    relative_scale_only: bool = True
    meters_per_unit: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_shape",
            (int(self.image_shape[0]), int(self.image_shape[1])),
        )
        for name in ("calibration_homography", "current_to_reference", "runtime_homography"):
            matrix = np.asarray(getattr(self, name), dtype=float).reshape(3, 3).copy()
            matrix.setflags(write=False)
            object.__setattr__(self, name, matrix)

    def as_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "homography_version": int(self.homography_version),
            "confidence": float(self.confidence),
            "status": str(self.status),
            "image_shape": list(self.image_shape),
            "calibration_homography": self.calibration_homography.tolist(),
            "current_to_reference": self.current_to_reference.tolist(),
            "runtime_homography": self.runtime_homography.tolist(),
            "relative_scale_only": bool(self.relative_scale_only),
            "meters_per_unit": self.meters_per_unit,
            "immutable_event_snapshot": True,
        }


def _motion_signature(observations: Sequence[object], image_shape: Sequence[int]) -> Optional[MotionGeometrySignature]:
    height, width = int(image_shape[0]), int(image_shape[1])
    positions = []
    directions = []
    weights = []
    for observation in observations:
        try:
            position = np.asarray(observation.position, dtype=float).reshape(2)
            direction = np.asarray(observation.direction, dtype=float).reshape(2)
            quality = float(observation.quality)
        except (AttributeError, TypeError, ValueError):
            continue
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(position).all() or norm <= 1e-9 or quality <= 0.0:
            continue
        positions.append((position / np.asarray([width, height], dtype=float)).tolist())
        directions.append(direction / norm)
        weights.append(quality)
    if not positions:
        return None
    position_array = np.asarray(positions, dtype=float)
    direction_array = np.asarray(directions, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    centroid = np.average(position_array, axis=0, weights=weight_array)
    spread = np.sqrt(np.average((position_array - centroid) ** 2, axis=0, weights=weight_array))
    tensor = np.average(
        direction_array[:, :, None] * direction_array[:, None, :],
        axis=0,
        weights=weight_array,
    )
    return MotionGeometrySignature(centroid, spread, tensor, len(positions))


def detect_calibration_drift(
    baseline_residual: Optional[float],
    current_residual: Optional[float],
    reference_signature: Optional[MotionGeometrySignature],
    current_signature: Optional[MotionGeometrySignature],
    config: CalibrationStateConfig,
    *,
    background_motion_score: Optional[float] = None,
) -> dict:
    """Return individual drift signals; persistence is handled by the state."""

    residual_increase = False
    if baseline_residual is not None and current_residual is not None:
        threshold = max(
            baseline_residual * config.drift_residual_ratio,
            baseline_residual + config.drift_absolute_increase,
        )
        residual_increase = current_residual > threshold
    centroid_shift = spread_shift = direction_shift = 0.0
    if reference_signature is not None and current_signature is not None:
        centroid_shift = float(np.linalg.norm(
            current_signature.centroid - reference_signature.centroid
        ))
        spread_shift = float(np.linalg.norm(
            current_signature.spread - reference_signature.spread
        ))
        direction_shift = float(np.linalg.norm(
            current_signature.direction_tensor - reference_signature.direction_tensor,
            ord="fro",
        ))
    background_signal = (
        background_motion_score is not None
        and math.isfinite(float(background_motion_score))
        and float(background_motion_score) > config.drift_background_motion
    )
    signals = {
        "residual_increase": bool(residual_increase),
        "centroid_shift": centroid_shift,
        "spread_shift": spread_shift,
        "direction_tensor_shift": direction_shift,
        "background_motion_score": background_motion_score,
        "geometry_shifted": bool(
            centroid_shift > config.drift_centroid_fraction
            or spread_shift > config.drift_spread_fraction
            or direction_shift > config.drift_direction_tensor
        ),
        "background_shifted": bool(background_signal),
    }
    signals["drift_signal"] = bool(
        signals["residual_increase"]
        or signals["geometry_shifted"]
        or signals["background_shifted"]
    )
    return signals


class DynamicHomographyCalibrator:
    """Persistent calibration state isolated from production attribution."""

    def __init__(
        self,
        phase1_config: Optional[CalibrationPhase1Config] = None,
        loss_config: Optional[CalibrationLossConfig] = None,
        optimizer_config: Optional[HomographyOptimizerConfig] = None,
        validation_config: Optional[HomographyValidationConfig] = None,
        state_config: Optional[CalibrationStateConfig] = None,
        stabilizer_config: Optional[BackgroundStabilizerConfig] = None,
    ) -> None:
        self.phase1_config = phase1_config or CalibrationPhase1Config.from_env()
        self.loss_config = loss_config or CalibrationLossConfig.from_env()
        self.optimizer_config = optimizer_config or HomographyOptimizerConfig.from_env()
        self.validation_config = validation_config or HomographyValidationConfig.from_env()
        self.state_config = state_config or CalibrationStateConfig.from_env()
        self.stabilizer_config = stabilizer_config or BackgroundStabilizerConfig.from_env()
        self.stabilizer = StaticBackgroundStabilizer(self.stabilizer_config)
        self.collector = VehicleTrajectoryCollector(self.phase1_config)
        self._image_shape = None
        self._homography = None
        self._status = CalibrationStatus.UNCALIBRATED
        self._confidence = 0.0
        self._version = 0
        self._last_evaluation_timestamp = None
        self._last_h_update_timestamp = None
        self._successful_updates = 0
        self._stable_windows = 0
        self._drift_windows = 0
        self._residual_baseline = None
        self._reference_signature = None
        self._last_drift_signals = None
        self._last_optimization: Optional[HomographyOptimizationResult] = None
        self._shape_mismatch = False
        self._last_stabilization: Optional[StabilizationResult] = None
        self._history = deque(maxlen=self.state_config.rollback_history_size)
        self._metrics = deque(maxlen=self.state_config.metrics_history_size)

    def _initialize(self, image_shape: Sequence[int]) -> None:
        shape = (int(image_shape[0]), int(image_shape[1]))
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("image_shape must be positive")
        if self._image_shape is not None and shape != self._image_shape:
            self._shape_mismatch = True
            self._status = CalibrationStatus.DRIFT_DETECTED
            self._last_drift_signals = {
                "drift_signal": True,
                "image_shape_changed": True,
                "previous_image_shape": list(self._image_shape),
                "observed_image_shape": list(shape),
            }
            return
        if self._image_shape is None:
            self._image_shape = shape
            self._homography = initial_normalized_image_homography(shape)

    def add_vehicle_observation(self, *args, image_shape=None, **kwargs):
        if image_shape is not None:
            self._initialize(image_shape)
        point = self.collector.add_vehicle_observation(
            *args, image_shape=image_shape, **kwargs
        )
        if point is not None and self._status == CalibrationStatus.UNCALIBRATED:
            self._status = CalibrationStatus.COLLECTING
        return point

    def _evidence(self):
        tracks = self.collector.buffer.tracks()
        observations = self.collector.local_motion_observations()
        field = self.collector.traffic_motion_field(observations)
        clusters, noise = self.collector.traffic_flow_clusters(observations)
        return tracks, observations, field, clusters, noise

    def _metric(self, timestamp, tracks, observations, field, clusters, result=None, losses=None):
        component_losses = losses.component_losses if losses is not None else {}
        metric = {
            "timestamp": float(timestamp),
            "status": self._status.value,
            "confidence": float(self._confidence),
            "num_tracks": int(len(tracks)),
            "num_valid_tracks": int(sum(len(points) >= 3 for points in tracks.values())),
            "num_motion_samples": int(len(observations)),
            "spatial_coverage": float(getattr(field, "spatial_coverage", 0.0) or 0.0),
            "flow_cluster_count": int(len(clusters)),
            "motion_loss": component_losses.get("motion"),
            "curvature_loss": component_losses.get("curvature"),
            "lane_loss": component_losses.get("lane"),
            "direction_loss": component_losses.get("direction"),
            "speed_loss": component_losses.get("speed"),
            "perspective_loss": result.perspective_loss if result is not None else None,
            "temporal_loss": result.temporal_loss if result is not None else None,
            "total_loss": losses.total_loss if losses is not None else None,
            "candidate_improvement": result.relative_improvement if result is not None else 0.0,
            "update_alpha": result.update_alpha if result is not None else 0.0,
            "update_magnitude": (
                float(np.linalg.norm(
                    result.proposed_homography - result.previous_homography
                ))
                if result is not None else 0.0
            ),
            "homography_version": int(self._version),
        }
        self._metrics.append(metric)
        return metric

    def update(self, timestamp: float, *, force: bool = False, background_motion_score=None) -> dict:
        timestamp = float(timestamp)
        if not self.phase1_config.enabled or self._image_shape is None:
            return self.get_state()
        if self._shape_mismatch:
            return self.get_state()
        if (
            not force
            and self._last_evaluation_timestamp is not None
            and timestamp - self._last_evaluation_timestamp
            < self.state_config.evaluation_interval_seconds
        ):
            return self.get_state()
        self._last_evaluation_timestamp = timestamp
        tracks, observations, field, clusters, _ = self._evidence()
        losses = evaluate_calibration_losses(
            self._homography,
            tracks,
            self._image_shape,
            motion_observations=observations,
            flow_clusters=clusters,
            config=self.loss_config,
            validation_config=self.validation_config,
        )
        signature = _motion_signature(observations, self._image_shape)
        if (
            background_motion_score is None
            and self._last_stabilization is not None
            and self._last_stabilization.valid
        ):
            background_motion_score = self._last_stabilization.motion_score

        if self._status == CalibrationStatus.LOCKED:
            self._last_drift_signals = detect_calibration_drift(
                self._residual_baseline,
                losses.total_loss,
                self._reference_signature,
                signature,
                self.state_config,
                background_motion_score=background_motion_score,
            )
            if self._last_drift_signals["drift_signal"]:
                self._drift_windows += 1
                self._confidence *= self.state_config.drift_confidence_decay
            else:
                self._drift_windows = 0
                if losses.total_loss is not None:
                    if self._residual_baseline is None:
                        self._residual_baseline = losses.total_loss
                    else:
                        alpha = self.state_config.residual_baseline_alpha
                        self._residual_baseline = (
                            (1.0 - alpha) * self._residual_baseline
                            + alpha * losses.total_loss
                        )
            if self._drift_windows >= self.state_config.drift_persistence_windows:
                self._status = CalibrationStatus.DRIFT_DETECTED
            self._metric(timestamp, tracks, observations, field, clusters, losses=losses)
            return self.get_state()

        if self._status == CalibrationStatus.DRIFT_DETECTED:
            self._status = CalibrationStatus.RECALIBRATING

        status_before_estimation = self._status
        if self._status in {
            CalibrationStatus.COLLECTING,
            CalibrationStatus.LOW_CONFIDENCE,
        }:
            self._status = CalibrationStatus.ESTIMATING

        result = estimate_candidate_homography(
            tracks,
            observations,
            clusters,
            field,
            self._image_shape,
            self._homography,
            loss_config=self.loss_config,
            optimizer_config=self.optimizer_config,
            validation_config=self.validation_config,
        )
        self._last_optimization = result
        previous_confidence = self._confidence
        target_confidence = result.confidence
        non_improvement_only = not (
            set(result.reasons) - {
                "insufficient_candidate_improvement",
                "confidence_below_freeze_threshold",
            }
        )
        if non_improvement_only and result.confidence_components:
            target_confidence = float(np.mean([
                value for name, value in result.confidence_components.items()
                if name != "improvement"
            ]))
        confidence_alpha = self.state_config.confidence_alpha
        self._confidence = float(np.clip(
            (1.0 - confidence_alpha) * previous_confidence
            + confidence_alpha * target_confidence,
            0.0,
            1.0,
        ))
        if result.update_recommended:
            self._history.append({
                "homography": self._homography.copy(),
                "version": self._version,
                "confidence": previous_confidence,
                "status": self._status,
                "last_h_update_timestamp": self._last_h_update_timestamp,
            })
            self._homography = result.proposed_homography.copy()
            self._last_h_update_timestamp = timestamp
            self._version += 1
            self._successful_updates += 1
            self._stable_windows = 0
            self._status = CalibrationStatus.WARMING_UP
        elif self._status == CalibrationStatus.WARMING_UP:
            hard_failures = set(result.reasons) - {
                "insufficient_candidate_improvement",
                "confidence_below_freeze_threshold",
            }
            confidence_without_improvement = np.mean([
                value for name, value in result.confidence_components.items()
                if name != "improvement"
            ]) if result.confidence_components else 0.0
            if not hard_failures and confidence_without_improvement >= self.optimizer_config.freeze_confidence:
                self._stable_windows += 1
            else:
                self._stable_windows = 0
            if (
                self._successful_updates >= self.state_config.min_successful_updates_to_lock
                and self._stable_windows >= self.state_config.stable_windows_to_lock
            ):
                self._status = CalibrationStatus.LOCKED
                self._residual_baseline = losses.total_loss
                self._reference_signature = signature
                self._drift_windows = 0
        elif self._status == CalibrationStatus.ESTIMATING:
            hard_failures = set(result.reasons) - {
                "insufficient_candidate_improvement",
                "confidence_below_freeze_threshold",
            }
            if hard_failures or self._confidence < self.optimizer_config.freeze_confidence:
                self._status = CalibrationStatus.LOW_CONFIDENCE
            else:
                self._status = status_before_estimation

        metric_losses = (
            evaluate_calibration_losses(
                self._homography,
                tracks,
                self._image_shape,
                motion_observations=observations,
                flow_clusters=clusters,
                config=self.loss_config,
                validation_config=self.validation_config,
            )
            if result.update_recommended else result.candidate_data_report
        )
        self._metric(
            timestamp, tracks, observations, field, clusters,
            result=result, losses=metric_losses,
        )
        return self.get_state()

    def image_to_ground(self, points: object) -> np.ndarray:
        if self._homography is None:
            raise RuntimeError("calibrator has no image shape or initial homography")
        return transform_image_to_ground(
            points,
            self.get_runtime_homography(),
            min_abs_denominator=self.validation_config.min_abs_denominator,
            max_abs_coordinate=self.validation_config.max_abs_coordinate,
        )

    def get_homography(self) -> Optional[np.ndarray]:
        return None if self._homography is None else self._homography.copy()

    def get_runtime_homography(self) -> Optional[np.ndarray]:
        if self._homography is None:
            return None
        return compose_runtime_homography(
            self._homography,
            self.stabilizer.get_transform(),
        )

    def stabilize_frame(self, frame: np.ndarray, *, exclusion_mask: object) -> StabilizationResult:
        result = self.stabilizer.update(frame, exclusion_mask=exclusion_mask)
        self._last_stabilization = result
        return result

    def capture_event_snapshot(self, timestamp: float) -> EventHomographySnapshot:
        if self._homography is None:
            raise RuntimeError("calibrator has no initialized homography")
        stabilization = self.stabilizer.get_transform()
        return EventHomographySnapshot(
            timestamp=float(timestamp),
            homography_version=self._version,
            confidence=self._confidence,
            status=self._status.value,
            image_shape=self._image_shape,
            calibration_homography=self._homography,
            current_to_reference=stabilization,
            runtime_homography=compose_runtime_homography(self._homography, stabilization),
        )

    def get_confidence(self) -> float:
        return float(self._confidence)

    def get_state(self) -> dict:
        last_metric = self._metrics[-1] if self._metrics else None
        return {
            "status": self._status.value,
            "confidence": float(self._confidence),
            "homography_version": int(self._version),
            "homography": self._homography.tolist() if self._homography is not None else None,
            "H": self._homography.tolist() if self._homography is not None else None,
            "image_shape": list(self._image_shape) if self._image_shape is not None else None,
            "num_tracks": int(self.collector.buffer.track_count),
            "num_motion_observations": int(
                last_metric["num_motion_samples"] if last_metric is not None else 0
            ),
            "last_update_timestamp": self._last_h_update_timestamp,
            "residual": last_metric["total_loss"] if last_metric is not None else None,
            "relative_scale_only": True,
            "successful_updates": int(self._successful_updates),
            "stable_windows": int(self._stable_windows),
            "drift_windows": int(self._drift_windows),
            "residual_baseline": self._residual_baseline,
            "last_drift_signals": self._last_drift_signals,
            "shape_mismatch_requires_reset": bool(self._shape_mismatch),
            "rollback_depth": int(len(self._history)),
            "last_metric": last_metric,
            "affects_attribution": False,
        }

    def summary(self) -> dict:
        summary = self.collector.summary(run_optimizer=False)
        summary["implemented_through_phase"] = 8
        summary["state"] = self.get_state()
        summary["optimization"] = (
            self._last_optimization.as_dict()
            if self._last_optimization is not None
            else {
                "enabled": bool(self.optimizer_config.enabled),
                "update_recommended": False,
                "reasons": ["not_evaluated"],
                "applied_to_production": False,
                "affects_attribution": False,
            }
        )
        summary["metrics_history"] = list(self._metrics)
        summary["stabilization"] = (
            self._last_stabilization.as_dict()
            if self._last_stabilization is not None
            else {
                "enabled": bool(self.stabilizer_config.enabled),
                "status": "not_evaluated",
                "affects_attribution": False,
            }
        )
        summary["affects_attribution"] = False
        return summary

    def rollback(self) -> bool:
        if not self._history:
            return False
        snapshot = self._history.pop()
        self._homography = snapshot["homography"].copy()
        self._version = int(snapshot["version"])
        self._confidence = float(snapshot["confidence"])
        self._status = snapshot["status"]
        self._last_h_update_timestamp = snapshot["last_h_update_timestamp"]
        self._stable_windows = 0
        self._drift_windows = 0
        return True

    def reset(self) -> None:
        self.collector = VehicleTrajectoryCollector(self.phase1_config)
        self._image_shape = None
        self._homography = None
        self._status = CalibrationStatus.UNCALIBRATED
        self._confidence = 0.0
        self._version = 0
        self._last_evaluation_timestamp = None
        self._last_h_update_timestamp = None
        self._successful_updates = 0
        self._stable_windows = 0
        self._drift_windows = 0
        self._residual_baseline = None
        self._reference_signature = None
        self._last_drift_signals = None
        self._last_optimization = None
        self._shape_mismatch = False
        self._last_stabilization = None
        self.stabilizer.reset()
        self._history.clear()
        self._metrics.clear()
