# -*- coding: utf-8 -*-
"""GPU-free actor box Kalman filter and offline RTS smoothing.

State layout is intentionally fixed::

    [foot_u, foot_v, log_w, log_h, v_u, v_v, v_logw, v_logh]

``foot_u`` is the horizontal box centre and ``foot_v`` is the box bottom.
Tracking log-width/log-height keeps predicted boxes positive.  The filter only
estimates actor motion; violation attribution remains the responsibility of
the downstream cost graph / min-cost-flow resolver.
"""
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


STATE_DIM = 8
MEASUREMENT_DIM = 4


@dataclass
class KalmanConfig:
    """Noise policy for :class:`BoxKalmanFilter`.

    Process-noise fractions are relative to the current box scale, so the same
    policy works for near/large and far/small actors. Public queries use frame
    indices, while transition/process dynamics use seconds.
    """

    min_box_size: float = 1.0
    max_log_size: float = 20.0
    min_confidence: float = 0.05
    confidence_variance_power: float = 2.0
    frames_per_second: float = 10.0
    noise_reference_fps: float = 10.0

    measurement_position_std_fraction: float = 0.04
    measurement_position_std_floor: float = 1.0
    measurement_log_size_std: float = 0.08

    initial_position_covariance_scale: float = 1.0
    initial_velocity_std_fraction: float = 0.25
    initial_log_size_velocity_std: float = 0.12

    process_position_accel_fractions: Mapping[str, float] = field(
        default_factory=lambda: {
            "person": 0.080,
            "scooter": 0.050,
            "vehicle": 0.030,
            "litter": 0.120,
            "default": 0.060,
        }
    )
    process_log_size_accel_stds: Mapping[str, float] = field(
        default_factory=lambda: {
            "person": 0.035,
            "scooter": 0.025,
            "vehicle": 0.015,
            "litter": 0.060,
            "default": 0.030,
        }
    )

    max_extrapolation_frames: int = 8
    covariance_jitter: float = 1e-9

    def __post_init__(self) -> None:
        if self.min_box_size <= 0.0:
            raise ValueError("min_box_size must be positive")
        if not 0.0 < self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in (0, 1]")
        if self.confidence_variance_power < 0.0:
            raise ValueError("confidence_variance_power must be non-negative")
        if self.max_extrapolation_frames < 0:
            raise ValueError("max_extrapolation_frames must be non-negative")
        if not math.isfinite(self.frames_per_second) or self.frames_per_second <= 0.0:
            raise ValueError("frames_per_second must be positive")
        if not math.isfinite(self.noise_reference_fps) or self.noise_reference_fps <= 0.0:
            raise ValueError("noise_reference_fps must be positive")


@dataclass(frozen=True)
class TrackMeasurement:
    """One detector observation for an actor track."""

    frame_index: int
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        frame_index = int(self.frame_index)
        if frame_index != self.frame_index:
            raise ValueError("frame_index must be an integer")
        bbox = tuple(float(value) for value in self.bbox_xyxy)
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            raise ValueError("bbox_xyxy must contain four finite values")
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox_xyxy must have positive width and height")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or confidence < 0.0:
            raise ValueError("confidence must be finite and non-negative")
        object.__setattr__(self, "frame_index", frame_index)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "confidence", confidence)


@dataclass
class StateEstimate:
    """State and uncertainty at one frame."""

    frame_index: int
    mean: np.ndarray
    covariance: np.ndarray
    observed: bool

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        return state_to_bbox(self.mean)


def bbox_to_measurement(
    bbox_xyxy: Sequence[float], min_box_size: float = 1.0
) -> np.ndarray:
    """Convert ``xyxy`` box to ``[foot_u, foot_v, log_w, log_h]``."""

    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four values")
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("bbox_xyxy must contain finite values")
    width = max(x2 - x1, float(min_box_size))
    height = max(y2 - y1, float(min_box_size))
    return np.asarray(
        [(x1 + x2) * 0.5, y2, math.log(width), math.log(height)],
        dtype=np.float64,
    )


def state_to_bbox(state: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert state (or four-position prefix) to a finite ``xyxy`` box."""

    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.size < MEASUREMENT_DIM:
        raise ValueError("state must contain at least four values")
    foot_u, foot_v, log_width, log_height = values[:MEASUREMENT_DIM]
    # Clipping is a numerical guard only; normal detector scales are far inside.
    width = math.exp(float(np.clip(log_width, -20.0, 20.0)))
    height = math.exp(float(np.clip(log_height, -20.0, 20.0)))
    return (
        float(foot_u - 0.5 * width),
        float(foot_v - height),
        float(foot_u + 0.5 * width),
        float(foot_v),
    )


def constant_velocity_transition(dt: float) -> np.ndarray:
    """Return 8-D constant-velocity transition matrix for arbitrary ``dt``."""

    dt = float(dt)
    if not math.isfinite(dt):
        raise ValueError("dt must be finite")
    transition = np.eye(STATE_DIM, dtype=np.float64)
    transition[:MEASUREMENT_DIM, MEASUREMENT_DIM:] = (
        np.eye(MEASUREMENT_DIM, dtype=np.float64) * dt
    )
    return transition


def _class_key(class_name: str) -> str:
    value = str(class_name).strip().lower()
    if value in {"car", "truck", "bus", "van"}:
        return "vehicle"
    if value in {"motorcycle", "motorbike", "bicycle", "bike"}:
        return "scooter"
    return value


def _mapping_value(
    values: Mapping[str, float], class_name: str, fallback: float
) -> float:
    key = _class_key(class_name)
    if key in values:
        return float(values[key])
    if "default" in values:
        return float(values["default"])
    return float(fallback)


def _box_scale_from_mean(mean: np.ndarray, config: KalmanConfig) -> Tuple[float, float]:
    log_min = math.log(config.min_box_size)
    log_width = float(np.clip(mean[2], log_min, config.max_log_size))
    log_height = float(np.clip(mean[3], log_min, config.max_log_size))
    return math.exp(log_width), math.exp(log_height)


def process_covariance(
    mean: Sequence[float],
    dt: float,
    class_name: str = "person",
    config: Optional[KalmanConfig] = None,
) -> np.ndarray:
    """Scale/class-adaptive continuous white-acceleration covariance.

    ``dt`` is seconds. Negative ``dt`` is supported for bounded reverse
    extrapolation; its position/velocity cross-covariance changes sign.
    """

    cfg = config or KalmanConfig()
    values = np.asarray(mean, dtype=np.float64).reshape(STATE_DIM)
    dt = float(dt)
    if not math.isfinite(dt):
        raise ValueError("dt must be finite")
    width, height = _box_scale_from_mean(values, cfg)
    # Existing fractions were calibrated per frame at REF=10 FPS. Convert
    # them to continuous-time spectral-density scale once, so changing video
    # FPS does not change uncertainty for the same elapsed seconds.
    spectral_scale = cfg.noise_reference_fps ** 1.5
    position_fraction = _mapping_value(
        cfg.process_position_accel_fractions, class_name, 0.060
    ) * spectral_scale
    log_accel_std = _mapping_value(
        cfg.process_log_size_accel_stds, class_name, 0.030
    ) * spectral_scale
    accel_stds = np.asarray(
        [
            max(cfg.measurement_position_std_floor * 0.25, position_fraction * width),
            max(cfg.measurement_position_std_floor * 0.25, position_fraction * height),
            log_accel_std,
            log_accel_std,
        ],
        dtype=np.float64,
    )

    duration = abs(dt)
    duration2 = duration * duration
    duration3 = duration2 * duration
    signed_cross = math.copysign(0.5 * duration2, dt) if dt != 0.0 else 0.0
    covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
    for index, accel_std in enumerate(accel_stds):
        variance = accel_std * accel_std
        covariance[index, index] = (duration3 / 3.0) * variance
        covariance[index, index + MEASUREMENT_DIM] = signed_cross * variance
        covariance[index + MEASUREMENT_DIM, index] = signed_cross * variance
        covariance[index + MEASUREMENT_DIM, index + MEASUREMENT_DIM] = (
            duration * variance
        )
    return covariance


def measurement_covariance(
    measurement: Sequence[float],
    confidence: float,
    config: Optional[KalmanConfig] = None,
) -> np.ndarray:
    """Confidence-adaptive detector covariance in measurement coordinates."""

    cfg = config or KalmanConfig()
    values = np.asarray(measurement, dtype=np.float64).reshape(MEASUREMENT_DIM)
    width = math.exp(float(np.clip(values[2], -20.0, cfg.max_log_size)))
    height = math.exp(float(np.clip(values[3], -20.0, cfg.max_log_size)))
    confidence = min(max(float(confidence), cfg.min_confidence), 1.0)
    variance_multiplier = confidence ** (-cfg.confidence_variance_power)
    base_stds = np.asarray(
        [
            max(
                cfg.measurement_position_std_floor,
                cfg.measurement_position_std_fraction * width,
            ),
            max(
                cfg.measurement_position_std_floor,
                cfg.measurement_position_std_fraction * height,
            ),
            cfg.measurement_log_size_std,
            cfg.measurement_log_size_std,
        ],
        dtype=np.float64,
    )
    return np.diag(base_stds * base_stds * variance_multiplier)


def _stabilize_covariance(
    covariance: np.ndarray, jitter: float
) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    try:
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    except np.linalg.LinAlgError:
        minimum_eigenvalue = -1.0
    if minimum_eigenvalue < jitter:
        covariance = covariance + np.eye(covariance.shape[0]) * (
            jitter - minimum_eigenvalue
        )
    return covariance


class BoxKalmanFilter:
    """Variable-frame-gap 8-D actor box Kalman filter."""

    def __init__(
        self,
        initial: TrackMeasurement,
        class_name: str = "person",
        config: Optional[KalmanConfig] = None,
    ) -> None:
        self.config = config or KalmanConfig()
        self.class_name = str(class_name)
        self.frame_index = int(initial.frame_index)
        position = bbox_to_measurement(
            initial.bbox_xyxy, self.config.min_box_size
        )
        self._mean = np.zeros(STATE_DIM, dtype=np.float64)
        self._mean[:MEASUREMENT_DIM] = position

        detector_covariance = measurement_covariance(
            position, initial.confidence, self.config
        )
        self._covariance = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
        self._covariance[:MEASUREMENT_DIM, :MEASUREMENT_DIM] = (
            detector_covariance * self.config.initial_position_covariance_scale
        )
        width, height = _box_scale_from_mean(self._mean, self.config)
        velocity_stds = np.asarray(
            [
                self.config.initial_velocity_std_fraction
                * self.config.noise_reference_fps * width,
                self.config.initial_velocity_std_fraction
                * self.config.noise_reference_fps * height,
                self.config.initial_log_size_velocity_std
                * self.config.noise_reference_fps,
                self.config.initial_log_size_velocity_std
                * self.config.noise_reference_fps,
            ],
            dtype=np.float64,
        )
        self._covariance[
            MEASUREMENT_DIM:, MEASUREMENT_DIM:
        ] = np.diag(velocity_stds * velocity_stds)
        self._covariance = _stabilize_covariance(
            self._covariance, self.config.covariance_jitter
        )

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    def estimate(self, observed: bool = False) -> StateEstimate:
        return StateEstimate(
            frame_index=self.frame_index,
            mean=self.mean,
            covariance=self.covariance,
            observed=bool(observed),
        )

    def _predict_delta(self, dt: float) -> None:
        transition = constant_velocity_transition(dt)
        process_noise = process_covariance(
            self._mean, dt, self.class_name, self.config
        )
        self._mean = transition.dot(self._mean)
        self._mean[2:4] = np.clip(
            self._mean[2:4],
            math.log(self.config.min_box_size),
            self.config.max_log_size,
        )
        self._covariance = (
            transition.dot(self._covariance).dot(transition.T) + process_noise
        )
        self._covariance = _stabilize_covariance(
            self._covariance, self.config.covariance_jitter
        )

    def predict(self, frame_index: int) -> StateEstimate:
        """Predict through every missing integer frame up to ``frame_index``."""

        frame_index = int(frame_index)
        if frame_index < self.frame_index:
            raise ValueError("cannot predict backwards")
        while self.frame_index < frame_index:
            self._predict_delta(1.0 / self.config.frames_per_second)
            self.frame_index += 1
        return self.estimate(observed=False)

    def update(self, measurement: TrackMeasurement) -> StateEstimate:
        """Predict if needed, then apply confidence-weighted Joseph update."""

        if measurement.frame_index < self.frame_index:
            raise ValueError("cannot update with an older measurement")
        if measurement.frame_index > self.frame_index:
            self.predict(measurement.frame_index)

        observed = bbox_to_measurement(
            measurement.bbox_xyxy, self.config.min_box_size
        )
        observation_matrix = np.zeros(
            (MEASUREMENT_DIM, STATE_DIM), dtype=np.float64
        )
        observation_matrix[:, :MEASUREMENT_DIM] = np.eye(
            MEASUREMENT_DIM, dtype=np.float64
        )
        detector_covariance = measurement_covariance(
            observed, measurement.confidence, self.config
        )
        innovation = observed - observation_matrix.dot(self._mean)
        innovation_covariance = (
            observation_matrix.dot(self._covariance).dot(observation_matrix.T)
            + detector_covariance
        )
        cross_covariance = self._covariance.dot(observation_matrix.T)
        try:
            gain = np.linalg.solve(
                innovation_covariance.T, cross_covariance.T
            ).T
        except np.linalg.LinAlgError:
            gain = cross_covariance.dot(np.linalg.pinv(innovation_covariance))

        self._mean = self._mean + gain.dot(innovation)
        self._mean[2:4] = np.clip(
            self._mean[2:4],
            math.log(self.config.min_box_size),
            self.config.max_log_size,
        )

        # Joseph form preserves symmetry/positive semidefiniteness better than
        # (I-KH)P under long missing sequences and low-confidence detections.
        identity = np.eye(STATE_DIM, dtype=np.float64)
        residual_transform = identity - gain.dot(observation_matrix)
        self._covariance = (
            residual_transform.dot(self._covariance).dot(residual_transform.T)
            + gain.dot(detector_covariance).dot(gain.T)
        )
        self._covariance = _stabilize_covariance(
            self._covariance, self.config.covariance_jitter
        )
        return self.estimate(observed=True)

    def step(
        self,
        frame_index: int,
        measurement: Optional[TrackMeasurement] = None,
    ) -> StateEstimate:
        """Run one observed or missing-data step."""

        frame_index = int(frame_index)
        if measurement is None:
            return self.predict(frame_index)
        if measurement.frame_index != frame_index:
            raise ValueError("measurement.frame_index does not match frame_index")
        return self.update(measurement)


@dataclass
class SmoothedTracklet:
    """Dense per-frame RTS-smoothed actor states.

    ``frames`` includes missing detector frames.  ``observed`` says which rows
    had a real detector update.  Queries just outside the tracklet use bounded
    constant-velocity extrapolation; distant queries are rejected.
    """

    track_id: Any
    class_name: str
    frames: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    observed: np.ndarray
    config: KalmanConfig
    _frame_lookup: Dict[int, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=np.int64)
        self.means = np.asarray(self.means, dtype=np.float64)
        self.covariances = np.asarray(self.covariances, dtype=np.float64)
        self.observed = np.asarray(self.observed, dtype=bool)
        count = len(self.frames)
        if count == 0:
            raise ValueError("SmoothedTracklet cannot be empty")
        if self.means.shape != (count, STATE_DIM):
            raise ValueError("means must have shape (N, 8)")
        if self.covariances.shape != (count, STATE_DIM, STATE_DIM):
            raise ValueError("covariances must have shape (N, 8, 8)")
        if self.observed.shape != (count,):
            raise ValueError("observed must have shape (N,)")
        if np.any(np.diff(self.frames) <= 0):
            raise ValueError("frames must be strictly increasing")
        self._frame_lookup = {
            int(frame): index for index, frame in enumerate(self.frames)
        }

    def state_at(self, frame_index: int) -> StateEstimate:
        """Return smoothed state or bounded forward/backward extrapolation."""

        frame_index = int(frame_index)
        exact_index = self._frame_lookup.get(frame_index)
        if exact_index is not None:
            return StateEstimate(
                frame_index=frame_index,
                mean=self.means[exact_index].copy(),
                covariance=self.covariances[exact_index].copy(),
                observed=bool(self.observed[exact_index]),
            )

        first_frame = int(self.frames[0])
        last_frame = int(self.frames[-1])
        if frame_index < first_frame:
            endpoint_index = 0
            frame_delta = frame_index - first_frame
        else:
            endpoint_index = len(self.frames) - 1
            frame_delta = frame_index - last_frame
        if abs(frame_delta) > self.config.max_extrapolation_frames:
            raise ValueError(
                "frame is outside bounded extrapolation window "
                "(max_extrapolation_frames={})".format(
                    self.config.max_extrapolation_frames
                )
            )

        mean = self.means[endpoint_index].copy()
        covariance = self.covariances[endpoint_index].copy()
        step_sign = 1.0 if frame_delta > 0 else -1.0
        dt_seconds = step_sign / self.config.frames_per_second
        for _ in range(abs(frame_delta)):
            transition = constant_velocity_transition(dt_seconds)
            covariance = (
                transition.dot(covariance).dot(transition.T)
                + process_covariance(
                    mean,
                    dt_seconds,
                    self.class_name,
                    self.config,
                )
            )
            mean = transition.dot(mean)
            mean[2:4] = np.clip(
                mean[2:4],
                math.log(self.config.min_box_size),
                self.config.max_log_size,
            )
            covariance = _stabilize_covariance(
                covariance, self.config.covariance_jitter
            )
        return StateEstimate(
            frame_index=frame_index,
            mean=mean,
            covariance=covariance,
            observed=False,
        )

    def predicted_bbox(
        self, frame_index: int
    ) -> Tuple[float, float, float, float]:
        """Return smoothed/extrapolated ``xyxy`` box at requested frame."""

        return self.state_at(frame_index).bbox_xyxy


def smooth_tracklet(
    measurements: Sequence[TrackMeasurement],
    class_name: str = "person",
    track_id: Any = None,
    config: Optional[KalmanConfig] = None,
) -> SmoothedTracklet:
    """Filter dense frames then run Rauch-Tung-Striebel backward smoothing."""

    if not measurements:
        raise ValueError("measurements cannot be empty")
    cfg = config or KalmanConfig()

    # A detector should emit at most one observation per track/frame.  If a
    # caller supplies duplicates, retain the highest-confidence observation.
    by_frame: Dict[int, TrackMeasurement] = {}
    for measurement in measurements:
        previous = by_frame.get(measurement.frame_index)
        if previous is None or measurement.confidence > previous.confidence:
            by_frame[measurement.frame_index] = measurement
    ordered = [by_frame[frame] for frame in sorted(by_frame)]
    first_frame = ordered[0].frame_index
    last_frame = ordered[-1].frame_index
    frames = np.arange(first_frame, last_frame + 1, dtype=np.int64)
    frame_count = len(frames)

    filtered_means = np.zeros((frame_count, STATE_DIM), dtype=np.float64)
    filtered_covariances = np.zeros(
        (frame_count, STATE_DIM, STATE_DIM), dtype=np.float64
    )
    predicted_means = np.zeros_like(filtered_means)
    predicted_covariances = np.zeros_like(filtered_covariances)
    observed_flags = np.zeros(frame_count, dtype=bool)
    transitions = np.zeros(
        (max(frame_count - 1, 0), STATE_DIM, STATE_DIM), dtype=np.float64
    )

    kalman = BoxKalmanFilter(ordered[0], class_name=class_name, config=cfg)
    filtered_means[0] = kalman.mean
    filtered_covariances[0] = kalman.covariance
    predicted_means[0] = kalman.mean
    predicted_covariances[0] = kalman.covariance
    observed_flags[0] = True

    for index in range(1, frame_count):
        frame_index = int(frames[index])
        transitions[index - 1] = constant_velocity_transition(
            1.0 / cfg.frames_per_second
        )
        kalman.predict(frame_index)
        predicted_means[index] = kalman.mean
        predicted_covariances[index] = kalman.covariance
        measurement = by_frame.get(frame_index)
        if measurement is not None:
            kalman.update(measurement)
            observed_flags[index] = True
        filtered_means[index] = kalman.mean
        filtered_covariances[index] = kalman.covariance

    smoothed_means = filtered_means.copy()
    smoothed_covariances = filtered_covariances.copy()
    for index in range(frame_count - 2, -1, -1):
        transition = transitions[index]
        cross_covariance = filtered_covariances[index].dot(transition.T)
        next_prediction_covariance = predicted_covariances[index + 1]
        try:
            smoother_gain = np.linalg.solve(
                next_prediction_covariance.T, cross_covariance.T
            ).T
        except np.linalg.LinAlgError:
            smoother_gain = cross_covariance.dot(
                np.linalg.pinv(next_prediction_covariance)
            )
        smoothed_means[index] = filtered_means[index] + smoother_gain.dot(
            smoothed_means[index + 1] - predicted_means[index + 1]
        )
        smoothed_means[index, 2:4] = np.clip(
            smoothed_means[index, 2:4],
            math.log(cfg.min_box_size),
            cfg.max_log_size,
        )
        smoothed_covariances[index] = (
            filtered_covariances[index]
            + smoother_gain.dot(
                smoothed_covariances[index + 1]
                - next_prediction_covariance
            ).dot(smoother_gain.T)
        )
        smoothed_covariances[index] = _stabilize_covariance(
            smoothed_covariances[index], cfg.covariance_jitter
        )

    return SmoothedTracklet(
        track_id=track_id,
        class_name=str(class_name),
        frames=frames,
        means=smoothed_means,
        covariances=smoothed_covariances,
        observed=observed_flags,
        config=cfg,
    )


__all__ = [
    "BoxKalmanFilter",
    "KalmanConfig",
    "MEASUREMENT_DIM",
    "STATE_DIM",
    "SmoothedTracklet",
    "StateEstimate",
    "TrackMeasurement",
    "bbox_to_measurement",
    "constant_velocity_transition",
    "measurement_covariance",
    "process_covariance",
    "smooth_tracklet",
    "state_to_bbox",
]
