"""Bounded Phase-6 candidate search, confidence and safe-update proposals.

The optimizer is deliberately pure: it never mutates a production transform.
It searches small parameterized perturbations around a validated previous H,
scores evidence and returns either a validated proposal or an explicit freeze.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .losses import (
    CalibrationLossConfig,
    CalibrationLossReport,
    evaluate_calibration_losses,
)
from .transform import (
    HomographyValidationConfig,
    normalize_homography,
    project_image_points,
    validate_homography,
)


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
class HomographyOptimizerConfig:
    enabled: bool = False
    iterations: int = 3
    anisotropy_step: float = 0.08
    shear_step: float = 0.04
    perspective_step: float = 0.08
    step_decay: float = 0.5
    max_abs_parameter: float = 0.35
    data_weight: float = 1.0
    perspective_weight: float = 0.25
    temporal_weight: float = 0.25
    perspective_huber_delta: float = 0.25
    temporal_huber_delta: float = 0.05
    min_valid_tracks: int = 4
    min_flow_clusters: int = 1
    min_spatial_coverage: float = 0.01
    min_duration_seconds: float = 10.0
    min_relative_improvement: float = 0.01
    freeze_confidence: float = 0.55
    max_update_alpha: float = 0.20
    target_tracks: int = 12
    target_flows: int = 2
    target_spatial_coverage: float = 0.10
    target_duration_seconds: float = 60.0
    target_quality: float = 0.80
    residual_scale: float = 0.10
    stability_scale: float = 0.05
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("optimizer iterations must be positive")
        positive = (
            self.anisotropy_step,
            self.shear_step,
            self.perspective_step,
            self.max_abs_parameter,
            self.perspective_huber_delta,
            self.temporal_huber_delta,
            self.target_spatial_coverage,
            self.target_duration_seconds,
            self.target_quality,
            self.residual_scale,
            self.stability_scale,
            self.epsilon,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("optimizer positive bounds must be finite and positive")
        if not 0.0 < self.step_decay < 1.0:
            raise ValueError("step_decay must be within (0, 1)")
        weights = (self.data_weight, self.perspective_weight, self.temporal_weight)
        if not all(math.isfinite(value) and value >= 0.0 for value in weights):
            raise ValueError("optimizer objective weights must be non-negative")
        if self.data_weight <= 0.0:
            raise ValueError("data_weight must be positive")
        if self.min_valid_tracks < 1 or self.min_flow_clusters < 1:
            raise ValueError("minimum evidence counts must be positive")
        if self.target_tracks < self.min_valid_tracks or self.target_flows < self.min_flow_clusters:
            raise ValueError("confidence targets must cover minimum evidence")
        fractions = (
            self.min_spatial_coverage,
            self.min_relative_improvement,
            self.freeze_confidence,
            self.max_update_alpha,
        )
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("optimizer fractions must be within 0..1")
        if self.min_duration_seconds < 0.0:
            raise ValueError("min_duration_seconds must be non-negative")
        if not 0.0 < self.min_relative_improvement <= 1.0:
            raise ValueError("min_relative_improvement must be within (0, 1]")
        if not 0.0 < self.target_spatial_coverage <= 1.0:
            raise ValueError("target_spatial_coverage must be within (0, 1]")
        if not 0.0 < self.target_quality <= 1.0:
            raise ValueError("target_quality must be within (0, 1]")

    @classmethod
    def from_env(cls) -> "HomographyOptimizerConfig":
        defaults = cls()
        values = {}
        float_fields = {
            "anisotropy_step": "ANISOTROPY_STEP",
            "shear_step": "SHEAR_STEP",
            "perspective_step": "PERSPECTIVE_STEP",
            "step_decay": "STEP_DECAY",
            "max_abs_parameter": "MAX_ABS_PARAMETER",
            "data_weight": "DATA_WEIGHT",
            "perspective_weight": "PERSPECTIVE_WEIGHT",
            "temporal_weight": "TEMPORAL_WEIGHT",
            "perspective_huber_delta": "PERSPECTIVE_HUBER_DELTA",
            "temporal_huber_delta": "TEMPORAL_HUBER_DELTA",
            "min_spatial_coverage": "MIN_SPATIAL_COVERAGE",
            "min_duration_seconds": "MIN_DURATION_SEC",
            "min_relative_improvement": "MIN_REL_IMPROVEMENT",
            "freeze_confidence": "FREEZE_CONFIDENCE",
            "max_update_alpha": "MAX_UPDATE_ALPHA",
            "target_spatial_coverage": "TARGET_SPATIAL_COVERAGE",
            "target_duration_seconds": "TARGET_DURATION_SEC",
            "target_quality": "TARGET_QUALITY",
            "residual_scale": "RESIDUAL_SCALE",
            "stability_scale": "STABILITY_SCALE",
        }
        int_fields = {
            "iterations": "ITERATIONS",
            "min_valid_tracks": "MIN_VALID_TRACKS",
            "min_flow_clusters": "MIN_FLOW_CLUSTERS",
            "target_tracks": "TARGET_TRACKS",
            "target_flows": "TARGET_FLOWS",
        }
        for field_name, suffix in float_fields.items():
            values[field_name] = _env_float(
                f"DYNAMIC_HOMOGRAPHY_OPT_{suffix}", getattr(defaults, field_name)
            )
        for field_name, suffix in int_fields.items():
            values[field_name] = _env_int(
                f"DYNAMIC_HOMOGRAPHY_OPT_{suffix}", getattr(defaults, field_name)
            )
        values["enabled"] = _env_bool("DYNAMIC_HOMOGRAPHY_OPTIMIZE", defaults.enabled)
        return cls(**values)


@dataclass(frozen=True)
class CandidateScore:
    homography: np.ndarray
    parameters: np.ndarray
    objective: Optional[float]
    data_report: CalibrationLossReport
    perspective_loss: Optional[float]
    temporal_loss: Optional[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "homography", np.asarray(self.homography, dtype=float).reshape(3, 3))
        object.__setattr__(self, "parameters", np.asarray(self.parameters, dtype=float).reshape(5))


@dataclass(frozen=True)
class HomographyOptimizationResult:
    enabled: bool
    update_recommended: bool
    reasons: Tuple[str, ...]
    previous_homography: np.ndarray
    candidate_homography: np.ndarray
    proposed_homography: np.ndarray
    selected_parameters: np.ndarray
    baseline_objective: Optional[float]
    candidate_objective: Optional[float]
    absolute_improvement: float
    relative_improvement: float
    confidence: float
    confidence_components: Mapping[str, float]
    update_alpha: float
    candidates_evaluated: int
    valid_candidates: int
    candidate_data_report: CalibrationLossReport
    perspective_loss: Optional[float]
    temporal_loss: Optional[float]

    def __post_init__(self) -> None:
        for name in ("previous_homography", "candidate_homography", "proposed_homography"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float).reshape(3, 3))
        object.__setattr__(self, "selected_parameters", np.asarray(self.selected_parameters, dtype=float).reshape(5))

    def as_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "update_recommended": bool(self.update_recommended),
            "reasons": list(self.reasons),
            "previous_homography": self.previous_homography.tolist(),
            "candidate_homography": self.candidate_homography.tolist(),
            "proposed_homography": self.proposed_homography.tolist(),
            "selected_parameters": self.selected_parameters.tolist(),
            "parameter_order": ["log_anisotropy", "shear_x", "shear_y", "perspective_x", "perspective_y"],
            "baseline_objective": self.baseline_objective,
            "candidate_objective": self.candidate_objective,
            "absolute_improvement": float(self.absolute_improvement),
            "relative_improvement": float(self.relative_improvement),
            "confidence": float(self.confidence),
            "confidence_components": dict(self.confidence_components),
            "update_alpha": float(self.update_alpha),
            "candidates_evaluated": int(self.candidates_evaluated),
            "valid_candidates": int(self.valid_candidates),
            "candidate_losses": self.candidate_data_report.as_dict(),
            "perspective_loss": self.perspective_loss,
            "temporal_loss": self.temporal_loss,
            "parameterization": "left_composed_normalized_ground_perturbation",
            "applied_to_production": False,
            "affects_attribution": False,
        }


def _huber(value: float, delta: float) -> float:
    magnitude = abs(float(value))
    if magnitude <= delta:
        return 0.5 * magnitude * magnitude
    return delta * (magnitude - 0.5 * delta)


def _perturbation_matrix(parameters: Sequence[float]) -> np.ndarray:
    anisotropy, shear_x, shear_y, perspective_x, perspective_y = map(float, parameters)
    return np.asarray([
        [math.exp(anisotropy), shear_x, 0.0],
        [shear_y, math.exp(-anisotropy), 0.0],
        [perspective_x, perspective_y, 1.0],
    ], dtype=float)


def compose_candidate_homography(previous_homography: object, parameters: Sequence[float]) -> np.ndarray:
    """Apply a bounded dimensionless perturbation in pseudo-ground coordinates."""

    previous = normalize_homography(previous_homography)
    return normalize_homography(_perturbation_matrix(parameters) @ previous)


def _perspective_regularization(
    homography: np.ndarray,
    image_shape: Sequence[int],
    cfg: HomographyOptimizerConfig,
    validation_cfg: HomographyValidationConfig,
) -> Optional[float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    fraction = 0.01
    values = []
    area_values = []
    for y_fraction in (0.2, 0.5, 0.8):
        for x_fraction in (0.2, 0.5, 0.8):
            point = np.asarray([x_fraction * width, y_fraction * height])
            samples = np.vstack((
                point,
                point + np.asarray([fraction * width, 0.0]),
                point + np.asarray([0.0, fraction * height]),
            ))
            projected = project_image_points(
                samples,
                homography,
                min_abs_denominator=validation_cfg.min_abs_denominator,
                max_abs_coordinate=validation_cfg.max_abs_coordinate,
            )
            if not np.all(projected.valid_mask):
                return None
            jacobian = np.column_stack((
                (projected.points[1] - projected.points[0]) / fraction,
                (projected.points[2] - projected.points[0]) / fraction,
            ))
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            if singular_values[-1] <= cfg.epsilon:
                return None
            values.append(abs(math.log(float(singular_values[0] / singular_values[-1]))))
            area_values.append(abs(float(np.linalg.det(jacobian))))
    median_area = float(np.median(area_values))
    if median_area <= cfg.epsilon:
        return None
    area_residuals = [abs(math.log(max(area, cfg.epsilon) / median_area)) for area in area_values]
    penalties = [
        _huber(value, cfg.perspective_huber_delta)
        for value in values + area_residuals
    ]
    return float(np.mean(penalties))


def _temporal_regularization(
    previous: np.ndarray,
    candidate: np.ndarray,
    image_shape: Sequence[int],
    cfg: HomographyOptimizerConfig,
    validation_cfg: HomographyValidationConfig,
) -> Optional[float]:
    height, width = int(image_shape[0]), int(image_shape[1])
    grid = np.asarray([
        (x * width, y * height)
        for y in np.linspace(0.0, 1.0, 5)
        for x in np.linspace(0.0, 1.0, 7)
    ])
    old_projection = project_image_points(
        grid, previous,
        min_abs_denominator=validation_cfg.min_abs_denominator,
        max_abs_coordinate=validation_cfg.max_abs_coordinate,
    )
    new_projection = project_image_points(
        grid, candidate,
        min_abs_denominator=validation_cfg.min_abs_denominator,
        max_abs_coordinate=validation_cfg.max_abs_coordinate,
    )
    valid = old_projection.valid_mask & new_projection.valid_mask
    if not np.any(valid):
        return None
    old_bounds = np.ptp(old_projection.points[old_projection.valid_mask], axis=0)
    scale = math.sqrt(max(float(old_bounds[0] * old_bounds[1]), cfg.epsilon))
    displacement = np.linalg.norm(
        new_projection.points[valid] - old_projection.points[valid], axis=1
    ) / scale
    return float(np.mean([
        _huber(value, cfg.temporal_huber_delta) for value in displacement
    ]))


def _score_candidate(
    homography: np.ndarray,
    parameters: np.ndarray,
    previous: np.ndarray,
    tracks: Mapping[object, Sequence[object]],
    motion_observations: Sequence[object],
    flow_clusters: Sequence[object],
    image_shape: Sequence[int],
    loss_cfg: CalibrationLossConfig,
    optimizer_cfg: HomographyOptimizerConfig,
    validation_cfg: HomographyValidationConfig,
) -> CandidateScore:
    data = evaluate_calibration_losses(
        homography,
        tracks,
        image_shape,
        motion_observations=motion_observations,
        flow_clusters=flow_clusters,
        config=loss_cfg,
        validation_config=validation_cfg,
    )
    perspective = None
    temporal = None
    objective = None
    if data.homography_valid and data.evaluable and data.total_loss is not None:
        perspective = _perspective_regularization(
            homography, image_shape, optimizer_cfg, validation_cfg
        )
        temporal = _temporal_regularization(
            previous, homography, image_shape, optimizer_cfg, validation_cfg
        )
        if perspective is not None and temporal is not None:
            numerator = (
                optimizer_cfg.data_weight * data.total_loss
                + optimizer_cfg.perspective_weight * perspective
                + optimizer_cfg.temporal_weight * temporal
            )
            denominator = (
                optimizer_cfg.data_weight
                + optimizer_cfg.perspective_weight
                + optimizer_cfg.temporal_weight
            )
            objective = float(numerator / denominator)
    return CandidateScore(homography, parameters, objective, data, perspective, temporal)


def _evidence_statistics(
    tracks: Mapping[object, Sequence[object]],
    motion_field: object,
) -> Tuple[int, float, float, float, float]:
    valid_tracks = []
    qualities = []
    durations = []
    for points in tracks.values():
        if len(points) < 3:
            continue
        path_length = float(sum(
            np.linalg.norm(right.image_point - left.image_point)
            for left, right in zip(points, points[1:])
        ))
        if not math.isfinite(path_length) or path_length <= 1e-9:
            continue
        valid_tracks.append(points)
        qualities.extend(float(point.quality_score) for point in points)
        track_times = [float(point.timestamp) for point in points]
        durations.append(max(track_times) - min(track_times))
    duration = float(np.median(durations)) if durations else 0.0
    quality = float(np.median(qualities)) if qualities else 0.0
    coverage = float(getattr(motion_field, "spatial_coverage", 0.0) or 0.0)
    cells = getattr(motion_field, "cells", {}) or {}
    field_stability = float(np.median([
        max(0.0, 1.0 - float(cell.direction_variance)) for cell in cells.values()
    ])) if cells else 0.0
    return len(valid_tracks), duration, quality, coverage, field_stability


def estimate_candidate_homography(
    tracks: Mapping[object, Sequence[object]],
    motion_observations: Sequence[object],
    flow_clusters: Sequence[object],
    motion_field: object,
    image_shape: Sequence[int],
    previous_homography: object,
    *,
    loss_config: Optional[CalibrationLossConfig] = None,
    optimizer_config: Optional[HomographyOptimizerConfig] = None,
    validation_config: Optional[HomographyValidationConfig] = None,
) -> HomographyOptimizationResult:
    """Search bounded perturbations and return a non-mutating safe proposal."""

    loss_cfg = loss_config or CalibrationLossConfig()
    cfg = optimizer_config or HomographyOptimizerConfig()
    validation_cfg = validation_config or HomographyValidationConfig()
    previous_validation = validate_homography(previous_homography, image_shape, validation_cfg)
    previous = previous_validation.normalized_homography
    zero = np.zeros(5, dtype=float)
    baseline = _score_candidate(
        previous, zero, previous, tracks, motion_observations, flow_clusters,
        image_shape, loss_cfg, cfg, validation_cfg,
    )
    if not cfg.enabled or not previous_validation.valid or baseline.objective is None:
        reasons = []
        if not cfg.enabled:
            reasons.append("optimizer_disabled")
        if not previous_validation.valid:
            reasons.append("invalid_previous_homography")
        if baseline.objective is None:
            reasons.append("insufficient_loss_evidence")
        return HomographyOptimizationResult(
            cfg.enabled, False, tuple(reasons), previous, previous, previous, zero,
            baseline.objective, baseline.objective, 0.0, 0.0, 0.0, {}, 0.0,
            1, int(baseline.objective is not None), baseline.data_report,
            baseline.perspective_loss, baseline.temporal_loss,
        )

    best = baseline
    evaluated = 1
    valid_candidates = 1
    base_steps = np.asarray([
        cfg.anisotropy_step,
        cfg.shear_step,
        cfg.shear_step,
        cfg.perspective_step,
        cfg.perspective_step,
    ])
    for iteration in range(cfg.iterations):
        steps = base_steps * (cfg.step_decay ** iteration)
        center = best.parameters.copy()
        iteration_best = best
        for dimension in range(5):
            for sign in (-1.0, 1.0):
                parameters = center.copy()
                parameters[dimension] = np.clip(
                    parameters[dimension] + sign * steps[dimension],
                    -cfg.max_abs_parameter,
                    cfg.max_abs_parameter,
                )
                candidate_h = compose_candidate_homography(previous, parameters)
                score = _score_candidate(
                    candidate_h, parameters, previous, tracks, motion_observations,
                    flow_clusters, image_shape, loss_cfg, cfg, validation_cfg,
                )
                evaluated += 1
                if score.objective is None:
                    continue
                valid_candidates += 1
                if score.objective < iteration_best.objective:
                    iteration_best = score
        best = iteration_best

    absolute_improvement = max(float(baseline.objective - best.objective), 0.0)
    relative_improvement = absolute_improvement / max(abs(float(baseline.objective)), cfg.epsilon)
    valid_tracks, duration, quality, coverage, field_stability = _evidence_statistics(
        tracks, motion_field
    )
    flow_count = len(flow_clusters)
    improvement_score = float(np.clip(
        relative_improvement / max(cfg.min_relative_improvement, cfg.epsilon), 0.0, 1.0
    ))
    temporal_distance = float(best.temporal_loss or 0.0)
    components = {
        "tracks": min(valid_tracks / float(cfg.target_tracks), 1.0),
        "flows": min(flow_count / float(cfg.target_flows), 1.0),
        "coverage": min(coverage / cfg.target_spatial_coverage, 1.0),
        "duration": min(duration / cfg.target_duration_seconds, 1.0),
        "quality": min(quality / cfg.target_quality, 1.0),
        "field_stability": float(np.clip(field_stability, 0.0, 1.0)),
        "residual": 1.0 / (1.0 + float(best.data_report.total_loss or 0.0) / cfg.residual_scale),
        "lane": (
            1.0 / (1.0 + float(best.data_report.component_losses["lane"]) / cfg.residual_scale)
            if best.data_report.component_losses.get("lane") is not None else 0.0
        ),
        "improvement": improvement_score,
        "historical_stability": 1.0 / (1.0 + temporal_distance / cfg.stability_scale),
    }
    confidence = float(np.mean(list(components.values())))
    reasons = []
    if valid_tracks < cfg.min_valid_tracks:
        reasons.append("too_few_valid_tracks")
    if flow_count < cfg.min_flow_clusters:
        reasons.append("too_few_independent_flows")
    if coverage < cfg.min_spatial_coverage:
        reasons.append("insufficient_spatial_coverage")
    if duration < cfg.min_duration_seconds:
        reasons.append("insufficient_trajectory_duration")
    if relative_improvement < cfg.min_relative_improvement:
        reasons.append("insufficient_candidate_improvement")
    if confidence < cfg.freeze_confidence:
        reasons.append("confidence_below_freeze_threshold")
    update_recommended = not reasons
    alpha = (
        min(cfg.max_update_alpha * confidence * improvement_score, cfg.max_update_alpha)
        if update_recommended else 0.0
    )
    proposed = previous
    if update_recommended and alpha > 0.0:
        proposed_parameters = best.parameters * alpha
        proposed_candidate = compose_candidate_homography(previous, proposed_parameters)
        proposed_validation = validate_homography(
            proposed_candidate, image_shape, validation_cfg
        )
        proposed_score = _score_candidate(
            proposed_candidate, proposed_parameters, previous, tracks,
            motion_observations, flow_clusters, image_shape, loss_cfg, cfg,
            validation_cfg,
        )
        if (
            proposed_validation.valid
            and proposed_score.objective is not None
            and proposed_score.objective < baseline.objective
        ):
            proposed = proposed_validation.normalized_homography
        else:
            update_recommended = False
            alpha = 0.0
            reasons.append(
                "safe_update_invalid"
                if not proposed_validation.valid
                else "safe_update_no_improvement"
            )
    return HomographyOptimizationResult(
        True, update_recommended, tuple(reasons), previous, best.homography, proposed,
        best.parameters, baseline.objective, best.objective, absolute_improvement,
        relative_improvement, confidence, components, alpha, evaluated,
        valid_candidates, best.data_report, best.perspective_loss, best.temporal_loss,
    )
