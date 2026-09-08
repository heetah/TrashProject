"""Explicit cost cells for litter-person-vehicle backtracking."""

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .trajectory import ReleaseHypothesis
from .spatial import EventSpatialTransform


ActorKey = Tuple[str, int]


@dataclass(frozen=True)
class BacktrackCostConfig:
    """Immutable, explicit cost weights used by a resolver trial.

    Defaults exactly reproduce the production component weights.  Research
    callers select a named ablation instead of altering process-wide env vars.
    Spatial/uncertainty hard gates intentionally stay outside these weights.
    Actor observation time uses a configurable hard legacy mode or the
    production soft penalty described by ``observation_time_cost_mode``.
    """

    ba_weights: Mapping[str, float] = field(default_factory=lambda: {
        "release_distance": 1.25, "uncertainty": .5, "time": .4,
        "quality": .45, "direction": .35, "release_prior": 1.0,
    })
    bc_weights: Mapping[str, float] = field(default_factory=lambda: {
        "direct_distance": 1.0, "spatial_mahalanobis": .2,
        "uncertainty": .2, "time": .35, "quality": .4,
        "release_prior": 1.0,
        # Research diagnostics. Zero keeps production ranking unchanged until
        # reviewed ablation demonstrates a stable direction across cases.
        "reverse_direction": 0.0, "exit_deficit": 0.0,
        "relative_motion_deficit": 0.0, "boundary_depth": 0.0,
    })
    ac_weights: Mapping[str, float] = field(default_factory=lambda: {
        # Wrong-depth vehicle boxes often cover a nearby person. Footpoint
        # distance is primary; overlap remains recorded but has zero weight.
        "endpoint_proximity": 1.0, "overlap": 0.0, "time": .2,
        "quality": .3, "uncertainty": .15, "continuity": .25,
    })
    # Production/full keeps the historical feature values. Research D+T
    # stages express distance and time as fractions of their hard gates so
    # weights compare dimensionless quantities with the same 0..1 meaning.
    normalize_distance_time_by_gate: bool = False
    normalize_bc_distance_time_by_gate: bool = False
    # ``soft`` keeps every actor state available inside the bounded tracklet
    # horizon.  The old thresholds become dimensionless bend points:
    # z=max(dt/tau_seconds, df/tau_frames),
    # rho(z)=z+kappa*max(0,z-1)^2. ``hard`` reproduces the legacy AND gate.
    observation_time_cost_mode: str = "soft"
    observation_time_soft_kappa: float = 4.0

    def __post_init__(self):
        if self.observation_time_cost_mode not in {"hard", "soft"}:
            raise ValueError("observation_time_cost_mode must be hard or soft")
        if (
            not math.isfinite(float(self.observation_time_soft_kappa))
            or float(self.observation_time_soft_kappa) < 0.0
        ):
            raise ValueError("observation_time_soft_kappa must be non-negative")

    @classmethod
    def for_stage(
        cls,
        stage: str,
        *,
        distance_weight: float = 1.0,
        time_weight: float = 1.0,
    ) -> "BacktrackCostConfig":
        stage = str(stage)
        distance_weight = float(distance_weight)
        time_weight = float(time_weight)
        if distance_weight < 0.0 or time_weight < 0.0:
            raise ValueError("distance/time weights must be non-negative")
        if distance_weight + time_weight <= 0.0:
            raise ValueError("at least one distance/time weight must be positive")
        if stage in {"distance_time", "kalman_rts"}:
            return cls(
                ba_weights={
                    "release_distance": distance_weight,
                    "time": time_weight,
                },
                bc_weights={
                    "direct_distance": distance_weight,
                    "time": time_weight,
                },
                ac_weights={
                    "endpoint_proximity": distance_weight,
                    "time": time_weight,
                },
                normalize_distance_time_by_gate=True,
            )
        if stage == "confidence":
            base = cls.for_stage(
                "distance_time",
                distance_weight=distance_weight,
                time_weight=time_weight,
            )
            return cls(
                ba_weights={**base.ba_weights, "quality": .45},
                bc_weights={**base.bc_weights, "quality": .4},
                ac_weights={**base.ac_weights, "quality": .3},
                normalize_distance_time_by_gate=True,
            )
        if stage == "uncertainty":
            base = cls.for_stage(
                "confidence",
                distance_weight=distance_weight,
                time_weight=time_weight,
            )
            return cls(
                ba_weights={**base.ba_weights, "uncertainty": .5},
                bc_weights={**base.bc_weights, "uncertainty": .2, "spatial_mahalanobis": .2},
                ac_weights={**base.ac_weights, "uncertainty": .15},
                normalize_distance_time_by_gate=True,
            )
        if stage in {"reverse", "full"}:
            return cls()
        raise ValueError("unknown backtrack cost stage: {}".format(stage))


@dataclass(frozen=True)
class ActorObservation:
    """One observed or Kalman-predicted actor state."""

    cls_name: str
    track_id: int
    frame_index: int
    bbox: Tuple[float, float, float, float]
    confidence: float = 1.0
    observed: bool = True
    covariance_uv: np.ndarray = field(
        default_factory=lambda: np.eye(2, dtype=float) * 4.0
    )
    source: str = "detector"
    # Frame of nearest real detector measurement supporting this state.
    # A Kalman state may live exactly at release time, but its D+T time cost
    # must still expose how stale the real evidence is.
    evidence_frame_index: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "cls_name", str(self.cls_name).lower())
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(
            self,
            "evidence_frame_index",
            (
                int(self.frame_index)
                if self.evidence_frame_index is None
                else int(self.evidence_frame_index)
            ),
        )
        object.__setattr__(
            self, "bbox", tuple(float(v) for v in self.bbox[:4])
        )
        object.__setattr__(
            self,
            "covariance_uv",
            np.asarray(self.covariance_uv, dtype=float).reshape(2, 2),
        )

    @property
    def actor_key(self) -> ActorKey:
        return self.cls_name, self.track_id

    @property
    def width(self) -> float:
        return max(self.bbox[2] - self.bbox[0], 1.0)

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1.0)

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)

    @property
    def footpoint(self) -> np.ndarray:
        x1, _, x2, y2 = self.bbox
        return np.asarray([(x1 + x2) * 0.5, y2], dtype=float)

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        return self.bbox

    @classmethod
    def from_snapshot(cls, snapshot: Mapping, frame_index: Optional[int] = None):
        confidence = snapshot.get(
            "confidence", snapshot.get("pose_conf", snapshot.get("conf", 1.0))
        )
        return cls(
            cls_name=snapshot.get("cls", snapshot.get("cls_name", "")),
            track_id=snapshot["track_id"],
            frame_index=(
                snapshot.get("frame_index", 0)
                if frame_index is None
                else frame_index
            ),
            bbox=snapshot.get("box", snapshot.get("bbox")),
            confidence=float(confidence),
            observed=bool(snapshot.get("observed", True)),
            covariance_uv=snapshot.get(
                "covariance_uv", np.eye(2, dtype=float) * 4.0
            ),
            source=str(snapshot.get("source", "detector")),
            evidence_frame_index=snapshot.get("evidence_frame_index"),
        )

    @classmethod
    def from_tracklet(
        cls,
        tracklet,
        frame_index: int,
        confidence: float = 1.0,
        source: str = "kalman_rts",
    ):
        """Adapt ``kalman.SmoothedTracklet`` without coupling module imports."""

        state = tracklet.state_at(int(frame_index))
        observed_frames = [
            int(frame)
            for frame, observed in zip(tracklet.frames, tracklet.observed)
            if bool(observed)
        ]
        return cls(
            cls_name=tracklet.class_name,
            track_id=tracklet.track_id,
            frame_index=state.frame_index,
            bbox=state.bbox_xyxy,
            confidence=float(confidence),
            observed=bool(state.observed),
            covariance_uv=state.covariance[:2, :2],
            source=source,
            evidence_frame_index=(
                min(
                    observed_frames,
                    key=lambda observed_frame: (
                        abs(observed_frame - int(frame_index)),
                        observed_frame,
                    ),
                )
                if observed_frames else int(frame_index)
            ),
        )


@dataclass(frozen=True)
class CostCell:
    valid: bool
    total: float
    components: Mapping[str, float]
    best_release_frame: Optional[int] = None
    reject_reason: Optional[str] = None
    # ``components`` remains the runtime/backward-compatible weighted
    # contribution table.  Sidecar calibration additionally needs the raw
    # feature and its weight so that a new weight can be evaluated without
    # rerunning detector/tracker inference.
    raw_features: Mapping[str, float] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def rejected(cls, reason: str, **components):
        return cls(
            valid=False,
            total=float("inf"),
            components={k: float(v) for k, v in components.items()},
            best_release_frame=None,
            reject_reason=str(reason),
        )


def _weighted_components(raw_features, weights):
    return {
        str(name): float(raw_features[name]) * float(weights[name])
        for name in raw_features if name in weights
    }


def _gate_fraction(value, gate, enabled):
    value = max(float(value), 0.0)
    if not enabled:
        return value
    return min(value / max(float(gate), 1e-9), 1.0)


def _hybrid_time_soft_penalty(
    gap_frames: int,
    fps: float,
    seconds_scale: float,
    frame_scale: Optional[int],
    kappa: float,
) -> float:
    """Return a continuous seconds/frames observation-freshness penalty.

    Seconds and frames measure the same gap, so ``max`` represents the former
    AND constraint without double-counting it.  The old thresholds are bend
    points rather than cutoffs.  Cost is linear through the supported region
    and gains a C1-continuous squared excess penalty outside it.
    """

    gap_frames = max(int(gap_frames), 0)
    safe_fps = max(float(fps), 1e-9)
    z_seconds = (gap_frames / safe_fps) / max(float(seconds_scale), 1e-9)
    z_frames = (
        gap_frames / max(float(frame_scale), 1e-9)
        if frame_scale is not None else 0.0
    )
    z_time = max(z_seconds, z_frames)
    excess = max(z_time - 1.0, 0.0)
    return float(z_time + max(float(kappa), 0.0) * excess * excess)


def _nearest_observation(
    observations: Sequence[ActorObservation],
    frame_index: int,
    max_gap_frames: Optional[int],
) -> Optional[ActorObservation]:
    if not observations:
        return None
    observation = min(
        observations,
        key=lambda item: abs(int(item.frame_index) - int(frame_index)),
    )
    if (
        max_gap_frames is not None
        and abs(int(observation.frame_index) - int(frame_index))
        > int(max_gap_frames)
    ):
        return None
    return observation


def _physical_gap_limit_frames(
    seconds: float,
    fps: float,
    frame_cap: Optional[int],
) -> int:
    """Convert the seconds gate to frames, then apply an optional AND cap.

    ``floor`` is intentional: rounding upward could admit an observation whose
    real elapsed time is greater than the configured seconds limit.
    """

    seconds_limit = max(
        int(math.floor(max(float(seconds), 0.0) * max(float(fps), 0.0) + 1e-9)),
        0,
    )
    if frame_cap is None:
        return seconds_limit
    return min(seconds_limit, max(int(frame_cap), 0))


def _point_to_rect_vector(point: np.ndarray, rect) -> np.ndarray:
    x1, y1, x2, y2 = map(float, rect)
    nearest = np.asarray(
        [
            np.clip(float(point[0]), min(x1, x2), max(x1, x2)),
            np.clip(float(point[1]), min(y1, y2), max(y1, y2)),
        ],
        dtype=float,
    )
    return np.asarray(point, dtype=float) - nearest


def _safe_inverse(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return np.linalg.pinv(matrix + np.eye(matrix.shape[0]) * 1e-6, rcond=1e-10)


def _quality_cost(observation: ActorObservation) -> float:
    confidence_cost = max(0.0, 1.0 - float(observation.confidence))
    prediction_cost = 0.35 if not observation.observed else 0.0
    return confidence_cost + prediction_cost


def _uncertainty_ratio(covariance: np.ndarray, reference_height: float) -> float:
    covariance = np.asarray(covariance, dtype=float)
    if not np.isfinite(covariance).all():
        return float("inf")
    try:
        eig = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        return float("inf")
    return np.sqrt(max(float(np.max(eig)), 0.0)) / max(
        float(reference_height), 1.0
    )


def _valid_covariance(covariance: np.ndarray) -> bool:
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        return False
    if not np.allclose(covariance, covariance.T, rtol=1e-5, atol=1e-7):
        return False
    try:
        return float(np.min(np.linalg.eigvalsh(covariance))) >= -1e-6
    except np.linalg.LinAlgError:
        return False


def _finite_observation(observation: ActorObservation) -> bool:
    return (
        np.isfinite(np.asarray(observation.bbox, dtype=float)).all()
        and _valid_covariance(observation.covariance_uv)
        and np.isfinite(float(observation.confidence))
    )


def _finite_release(release: ReleaseHypothesis) -> bool:
    return (
        np.isfinite(release.mean_uv).all()
        and _valid_covariance(release.covariance_uv)
        and np.isfinite(release.velocity_uv).all()
        and np.isfinite(float(release.prior_cost))
    )


def compute_c_ba(
    releases: Sequence[ReleaseHypothesis],
    person_observations: Sequence[ActorObservation],
    fps: float,
    max_observation_gap_seconds: float = 0.25,
    max_observation_gap_frames: Optional[int] = 3,
    max_uncertainty_height_ratio: float = 1.5,
    normalized_distance_gate: float = 0.85,
    cost_config: Optional[BacktrackCostConfig] = None,
    spatial_transform: Optional[EventSpatialTransform] = None,
) -> CostCell:
    """Litter-person cost using a person's upper-body release zone.

    Footpoints are appropriate for ground-distance association but physically
    wrong for hand release.  Here the valid zone spans the person's upper 72%.
    """

    people = [item for item in person_observations if item.cls_name == "person"]
    if not people:
        return CostCell.rejected("no_person_observation")
    if not releases:
        return CostCell.rejected("no_release_hypothesis")
    resolved_cost_config = cost_config or BacktrackCostConfig()
    soft_time = resolved_cost_config.observation_time_cost_mode == "soft"
    max_gap = (
        None if soft_time else _physical_gap_limit_frames(
            max_observation_gap_seconds, fps, max_observation_gap_frames
        )
    )

    candidates = []
    rejected_uncertainty = False
    minimum_distance = float("inf")
    minimum_uncertainty_ratio = float("inf")
    for release in releases:
        if not _finite_release(release):
            continue
        person = _nearest_observation(people, release.frame_index, max_gap)
        if person is None or not _finite_observation(person):
            continue
        attribution_covariance = (
            release.covariance_uv + person.covariance_uv
        )
        uncertainty_ratio = _uncertainty_ratio(
            attribution_covariance, person.height
        )
        minimum_uncertainty_ratio = min(
            minimum_uncertainty_ratio, float(uncertainty_ratio)
        )
        if uncertainty_ratio > float(max_uncertainty_height_ratio):
            rejected_uncertainty = True
            continue

        x1, y1, x2, y2 = person.bbox
        release_zone = (x1, y1, x2, y1 + 0.72 * person.height)
        residual = _point_to_rect_vector(release.mean_uv, release_zone)
        if spatial_transform is not None:
            normalized_distance = spatial_transform.normalized_point_to_rect(
                release.mean_uv,
                release_zone,
                scale="height",
            )
            if normalized_distance is None:
                continue
        else:
            normalized_distance = float(np.linalg.norm(residual)) / max(
                person.height, 1.0
            )
        minimum_distance = min(minimum_distance, normalized_distance)
        # Uncertainty must never turn a physically distant release into a
        # plausible hand/torso release.
        if normalized_distance > float(normalized_distance_gate):
            continue
        person_upper_center = np.asarray(
            [(x1 + x2) * 0.5, y1 + 0.36 * person.height], dtype=float
        )
        displacement = release.mean_uv - person_upper_center
        release_velocity = release.velocity_uv
        direction_scale = person.height
        if spatial_transform is not None:
            projected_pair = spatial_transform.project(
                (release.mean_uv, person_upper_center)
            )
            projected_velocity = spatial_transform.vector_at(
                release.mean_uv, release.velocity_uv
            )
            if projected_pair is None or projected_velocity is None:
                continue
            displacement = projected_pair[0] - projected_pair[1]
            release_velocity = projected_velocity
            projected_height = spatial_transform.rect_height_scale(person.bbox)
            if projected_height is None:
                continue
            direction_scale = projected_height
        displacement_norm = float(np.linalg.norm(displacement))
        velocity_norm = float(np.linalg.norm(release_velocity))
        direction_cost = 0.0
        if displacement_norm > 0.12 * direction_scale and velocity_norm > 1e-6:
            cosine = float(
                displacement @ release_velocity
                / (displacement_norm * velocity_norm)
            )
            direction_cost = 0.35 * max(0.0, -cosine)

        evidence_gap_frames = abs(
            person.evidence_frame_index - release.frame_index
        )
        if not soft_time and evidence_gap_frames > int(max_gap):
            continue
        time_gap = evidence_gap_frames / max(
            float(fps), 1e-6
        )
        quality = _quality_cost(person)
        normalized_covariance = attribution_covariance / max(
            person.height * person.height, 1.0
        )
        _, logdet = np.linalg.slogdet(np.eye(2) + normalized_covariance)
        uncertainty_cost = 0.5 * max(float(logdet), 0.0)
        normalize_dt = bool(
            resolved_cost_config.normalize_distance_time_by_gate
        )
        time_feature = (
            _hybrid_time_soft_penalty(
                evidence_gap_frames,
                fps,
                max_observation_gap_seconds,
                max_observation_gap_frames,
                resolved_cost_config.observation_time_soft_kappa,
            )
            if soft_time else _gate_fraction(
                time_gap, max_observation_gap_seconds, normalize_dt
            )
        )
        raw_features = {
            "release_distance": _gate_fraction(
                normalized_distance, normalized_distance_gate, normalize_dt
            ),
            "uncertainty": max(float(logdet), 0.0),
            "time": time_feature,
            "quality": quality,
            "direction": (
                direction_cost / 0.35 if direction_cost > 0.0 else 0.0
            ),
            "release_prior": float(release.prior_cost),
        }
        weights = dict(resolved_cost_config.ba_weights) or {
            # Physical distance is deliberately independent of covariance.
            # Combined with monotone uncertainty_cost, increasing covariance
            # cannot improve the same geometric B-A hypothesis.
            "release_distance": 1.25,
            "uncertainty": 0.5,
            "time": 0.4,
            "quality": 0.45,
            "direction": 0.35,
            "release_prior": 1.0,
        }
        components = _weighted_components(raw_features, weights)
        total = float(sum(components.values()))
        if np.isfinite(total):
            candidates.append(
                (
                    total,
                    components,
                    int(release.frame_index),
                    raw_features,
                    weights,
                )
            )

    if not candidates:
        reason = (
            "release_uncertainty_too_large"
            if rejected_uncertainty
            else "person_release_gate_failed"
        )
        diagnostics = {
            "normalized_distance_gate": float(normalized_distance_gate),
            "uncertainty_height_ratio_gate": float(
                max_uncertainty_height_ratio
            ),
        }
        if np.isfinite(minimum_distance):
            diagnostics["minimum_normalized_distance"] = minimum_distance
        if np.isfinite(minimum_uncertainty_ratio):
            diagnostics["minimum_uncertainty_height_ratio"] = (
                minimum_uncertainty_ratio
            )
        return CostCell.rejected(reason, **diagnostics)
    total, components, frame_index, raw_features, weights = min(
        candidates, key=lambda item: item[0]
    )
    return CostCell(
        True,
        total,
        components,
        frame_index,
        None,
        raw_features,
        weights,
    )


def compute_c_bc(
    releases: Sequence[ReleaseHypothesis],
    vehicle_observations: Sequence[ActorObservation],
    fps: float,
    max_observation_gap_seconds: float = 0.25,
    max_observation_gap_frames: Optional[int] = 3,
    max_uncertainty_height_ratio: float = 1.5,
    normalized_distance_gate: float = 0.4,
    cost_config: Optional[BacktrackCostConfig] = None,
    litter_last_point: Optional[Sequence[float]] = None,
    litter_last_frame: Optional[int] = None,
    vehicle_bbox_expand_x_ratio: float = 0.0,
    vehicle_bbox_expand_y_ratio: float = 0.0,
    spatial_transform: Optional[EventSpatialTransform] = None,
) -> CostCell:
    """Direct litter-vehicle route with a hard physical release gate."""

    vehicles = [
        item
        for item in vehicle_observations
        if item.cls_name in ("vehicle", "scooter")
    ]
    if not vehicles:
        return CostCell.rejected("no_vehicle_observation")
    if not releases:
        return CostCell.rejected("no_release_hypothesis")
    resolved_cost_config = cost_config or BacktrackCostConfig()
    soft_time = resolved_cost_config.observation_time_cost_mode == "soft"
    max_gap = (
        None if soft_time else _physical_gap_limit_frames(
            max_observation_gap_seconds, fps, max_observation_gap_frames
        )
    )

    candidates = []
    rejected_uncertainty = False
    minimum_distance = float("inf")
    minimum_uncertainty_ratio = float("inf")
    for release in releases:
        if not _finite_release(release):
            continue
        vehicle = _nearest_observation(vehicles, release.frame_index, max_gap)
        if vehicle is None or not _finite_observation(vehicle):
            continue
        attribution_covariance = (
            release.covariance_uv + vehicle.covariance_uv
        )
        uncertainty_ratio = _uncertainty_ratio(
            attribution_covariance, vehicle.height
        )
        minimum_uncertainty_ratio = min(
            minimum_uncertainty_ratio, float(uncertainty_ratio)
        )
        if uncertainty_ratio > float(max_uncertainty_height_ratio):
            rejected_uncertainty = True
            continue

        # Production measures distance from the observed vehicle bbox itself.
        # Expansion remains an explicit research override for legacy replay.
        x1, y1, x2, y2 = vehicle.bbox
        margin_x = max(float(vehicle_bbox_expand_x_ratio), 0.0) * vehicle.width
        margin_y = max(float(vehicle_bbox_expand_y_ratio), 0.0) * vehicle.height
        release_u, release_v = map(float, release.mean_uv)
        if x1 <= release_u <= x2 and y1 <= release_v <= y2:
            interior_depth = min(
                release_u - x1,
                x2 - release_u,
                release_v - y1,
                y2 - release_v,
            )
            # Boundary depth is a soft diagnostic inside the original bbox,
            # independent of whether the hard-gate rectangle is expanded.
            boundary_scale = max(
                0.18 * vehicle.width, 0.15 * vehicle.height, 1.0
            )
            boundary_depth = min(
                max(float(interior_depth), 0.0)
                / boundary_scale,
                1.0,
            )
        else:
            boundary_depth = 0.0
        residual = _point_to_rect_vector(
            release.mean_uv,
            (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y),
        )
        distance_rect = (
            x1 - margin_x,
            y1 - margin_y,
            x2 + margin_x,
            y2 + margin_y,
        )
        if spatial_transform is not None:
            normalized_distance = spatial_transform.normalized_point_to_rect(
                release.mean_uv,
                distance_rect,
                scale="diagonal",
            )
            if normalized_distance is None:
                continue
        else:
            normalized_distance = float(np.linalg.norm(residual)) / max(
                np.hypot(vehicle.width, vehicle.height), 1.0
            )
        minimum_distance = min(minimum_distance, normalized_distance)
        if normalized_distance > float(normalized_distance_gate):
            continue

        combined_cov = release.covariance_uv + vehicle.covariance_uv
        mahalanobis = np.sqrt(
            max(float(residual @ _safe_inverse(combined_cov) @ residual), 0.0)
        )
        quality = _quality_cost(vehicle)
        evidence_gap_frames = abs(
            vehicle.evidence_frame_index - release.frame_index
        )
        if not soft_time and evidence_gap_frames > int(max_gap):
            continue
        time_gap = evidence_gap_frames / max(
            float(fps), 1e-6
        )
        normalize_dt = bool(
            resolved_cost_config.normalize_distance_time_by_gate
            or resolved_cost_config.normalize_bc_distance_time_by_gate
        )
        time_feature = (
            _hybrid_time_soft_penalty(
                evidence_gap_frames,
                fps,
                max_observation_gap_seconds,
                max_observation_gap_frames,
                resolved_cost_config.observation_time_soft_kappa,
            )
            if soft_time else _gate_fraction(
                time_gap, max_observation_gap_seconds, normalize_dt
            )
        )
        raw_features = {
            "direct_distance": _gate_fraction(
                normalized_distance, normalized_distance_gate, normalize_dt
            ),
            "spatial_mahalanobis": mahalanobis,
            "uncertainty": np.log1p(
                (
                    float(np.trace(release.covariance_uv))
                    + float(np.trace(vehicle.covariance_uv))
                )
                / max(vehicle.height * vehicle.height, 1.0)
            ),
            "time": time_feature,
            "quality": quality,
            "release_prior": float(release.prior_cost),
            "boundary_depth": boundary_depth,
        }
        if litter_last_point is not None and litter_last_frame is not None:
            try:
                last_point = np.asarray(litter_last_point, dtype=float).reshape(2)
                later_vehicle = min(
                    vehicles,
                    key=lambda item: (
                        abs(item.frame_index - int(litter_last_frame)),
                        item.frame_index,
                    ),
                )
                outward = release.mean_uv - vehicle.center
                release_velocity = release.velocity_uv
                if spatial_transform is not None:
                    projected_pair = spatial_transform.project(
                        (release.mean_uv, vehicle.center)
                    )
                    projected_velocity = spatial_transform.vector_at(
                        release.mean_uv, release.velocity_uv
                    )
                    if projected_pair is None or projected_velocity is None:
                        raise ValueError("unsafe pseudo-ground direction projection")
                    outward = projected_pair[0] - projected_pair[1]
                    release_velocity = projected_velocity
                velocity_norm = float(np.linalg.norm(release_velocity))
                outward_norm = float(np.linalg.norm(outward))
                outward_cosine = float(
                    release_velocity @ outward
                    / max(velocity_norm * outward_norm, 1e-6)
                )
                if spatial_transform is not None:
                    scale = spatial_transform.rect_diagonal_scale(vehicle.bbox)
                    release_box_distance = spatial_transform.point_to_rect_distance(
                        release.mean_uv, vehicle.bbox
                    )
                    later_box_distance = spatial_transform.point_to_rect_distance(
                        last_point, later_vehicle.bbox
                    )
                    release_center_pair = spatial_transform.project(
                        (release.mean_uv, vehicle.center)
                    )
                    later_center_pair = spatial_transform.project(
                        (last_point, later_vehicle.center)
                    )
                    if (
                        scale is None
                        or release_box_distance is None
                        or later_box_distance is None
                        or release_center_pair is None
                        or later_center_pair is None
                    ):
                        raise ValueError("unsafe pseudo-ground BC diagnostic projection")
                    release_center_distance = float(np.linalg.norm(
                        release_center_pair[0] - release_center_pair[1]
                    ))
                    later_center_distance = float(np.linalg.norm(
                        later_center_pair[0] - later_center_pair[1]
                    ))
                else:
                    scale = max(np.hypot(vehicle.width, vehicle.height), 1.0)
                    release_box_distance = float(np.linalg.norm(
                        _point_to_rect_vector(release.mean_uv, vehicle.bbox)
                    ))
                    later_box_distance = float(np.linalg.norm(
                        _point_to_rect_vector(last_point, later_vehicle.bbox)
                    ))
                    release_center_distance = float(
                        np.linalg.norm(release.mean_uv - vehicle.center)
                    )
                    later_center_distance = float(
                        np.linalg.norm(last_point - later_vehicle.center)
                    )
                exit_gain = (
                    later_box_distance - release_box_distance
                ) / max(float(scale), 1e-12)
                relative_gain = (
                    later_center_distance - release_center_distance
                ) / max(float(scale), 1e-12)
                raw_features.update({
                    "reverse_direction": (1.0 - np.clip(
                        outward_cosine, -1.0, 1.0
                    )) * 0.5,
                    "exit_deficit": max(0.25 - exit_gain, 0.0),
                    "relative_motion_deficit": max(
                        0.25 - relative_gain, 0.0
                    ),
                })
            except (TypeError, ValueError):
                pass
        weights = dict(resolved_cost_config.bc_weights) or {
            "direct_distance": 1.0,
            "spatial_mahalanobis": 0.2,
            "uncertainty": 0.2,
            "time": 0.35,
            "quality": 0.4,
            "release_prior": 1.0,
        }
        components = _weighted_components(raw_features, weights)
        total = float(sum(components.values()))
        if np.isfinite(total):
            candidates.append(
                (
                    total,
                    components,
                    int(release.frame_index),
                    raw_features,
                    weights,
                )
            )

    if not candidates:
        reason = (
            "release_uncertainty_too_large"
            if rejected_uncertainty
            else "direct_vehicle_gate_failed"
        )
        diagnostics = {
            "normalized_distance_gate": float(normalized_distance_gate),
            "uncertainty_height_ratio_gate": float(
                max_uncertainty_height_ratio
            ),
        }
        if np.isfinite(minimum_distance):
            diagnostics["minimum_normalized_distance"] = minimum_distance
        if np.isfinite(minimum_uncertainty_ratio):
            diagnostics["minimum_uncertainty_height_ratio"] = (
                minimum_uncertainty_ratio
            )
        return CostCell.rejected(reason, **diagnostics)
    total, components, frame_index, raw_features, weights = min(
        candidates, key=lambda item: item[0]
    )
    return CostCell(
        True,
        total,
        components,
        frame_index,
        None,
        raw_features,
        weights,
    )


def _intersection_over_minimum(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    return intersection / max(min(area_a, area_b), 1e-6)


def compute_c_ac(
    person_observations: Sequence[ActorObservation],
    vehicle_observations: Sequence[ActorObservation],
    fps: float,
    max_pair_gap_seconds: float = 0.25,
    max_observation_gap_frames: Optional[int] = 3,
    proximity_gate: float = 1.2,
    max_uncertainty_height_ratio: float = 1.5,
    min_dwell_seconds: float = 0.15,
    max_support_gap_seconds: float = 0.50,
    cost_config: Optional[BacktrackCostConfig] = None,
    spatial_transform: Optional[EventSpatialTransform] = None,
) -> CostCell:
    """Person-vehicle cost from their own endpoints and overlap only.

    No litter location enters this function.  Therefore C_AC cannot double
    count the B-A anchor and a person may remain linked after walking away.
    """

    people = sorted(
        (
            item
            for item in person_observations
            if item.cls_name == "person" and _finite_observation(item)
        ),
        key=lambda item: int(item.frame_index),
    )
    vehicles = sorted(
        (
            item
            for item in vehicle_observations
            if (
                item.cls_name in ("vehicle", "scooter")
                and _finite_observation(item)
            )
        ),
        key=lambda item: int(item.frame_index),
    )
    if not people or not vehicles:
        return CostCell.rejected("missing_person_or_vehicle")
    resolved_cost_config = cost_config or BacktrackCostConfig()
    soft_time = resolved_cost_config.observation_time_cost_mode == "soft"
    max_gap = (
        None if soft_time else _physical_gap_limit_frames(
            max_pair_gap_seconds, fps, max_observation_gap_frames
        )
    )

    pairs = []
    vehicle_index = 0
    for person in people:
        # Both lists are time-sorted. A monotonic nearest-neighbour cursor makes
        # each A-C track-pair O(len(A)+len(C)), not O(len(A)*len(C)).
        while (
            vehicle_index + 1 < len(vehicles)
            and abs(
                int(vehicles[vehicle_index + 1].frame_index)
                - int(person.frame_index)
            )
            < abs(
                int(vehicles[vehicle_index].frame_index)
                - int(person.frame_index)
            )
        ):
            vehicle_index += 1
        vehicle = vehicles[vehicle_index]
        if (
            max_gap is not None
            and
            abs(int(vehicle.frame_index) - int(person.frame_index))
            > max_gap
        ):
            continue
        if spatial_transform is not None:
            proximity = spatial_transform.normalized_point_distance(
                person.footpoint,
                vehicle.footpoint,
                vehicle.bbox,
            )
            if proximity is None:
                continue
        else:
            foot_distance = float(np.linalg.norm(
                person.footpoint - vehicle.footpoint
            ))
            vehicle_scale = max(np.hypot(vehicle.width, vehicle.height), 1.0)
            proximity = foot_distance / vehicle_scale
        pair_covariance = person.covariance_uv + vehicle.covariance_uv
        if _uncertainty_ratio(
            pair_covariance, max(person.height, vehicle.height)
        ) > float(max_uncertainty_height_ratio):
            continue
        iom = _intersection_over_minimum(person.bbox, vehicle.bbox)
        evidence_gap_frames = abs(
            person.evidence_frame_index - vehicle.evidence_frame_index
        )
        if max_gap is not None and evidence_gap_frames > max_gap:
            continue
        pair_gap = evidence_gap_frames / max(
            float(fps), 1e-6
        )
        quality = 0.5 * (
            _quality_cost(person) + _quality_cost(vehicle)
        )
        uncertainty = 0.15 * np.log1p(
            float(np.trace(pair_covariance))
            / max(max(person.height, vehicle.height) ** 2, 1.0)
        )
        pairs.append(
            (
                person, vehicle, proximity, iom, evidence_gap_frames,
                pair_gap, quality, uncertainty,
            )
        )

    if not pairs:
        return CostCell.rejected("no_temporal_overlap")

    pairs.sort(key=lambda item: item[0].frame_index)
    supportive = [
        item
        for item in pairs
        if item[2] <= float(proximity_gate) or item[3] >= 0.05
    ]
    if not supportive:
        return CostCell.rejected("person_vehicle_gate_failed")

    observed_support = [
        item for item in supportive if item[0].observed or item[1].observed
    ]
    max_support_gap_frames = max(
        int(round(float(max_support_gap_seconds) * float(fps))), 1
    )
    support_runs = []
    for item in observed_support:
        frame_index = int(item[0].frame_index)
        if (
            not support_runs
            or frame_index - int(support_runs[-1][-1][0].frame_index)
            > max_support_gap_frames
        ):
            support_runs.append([item])
        else:
            support_runs[-1].append(item)

    def _run_metrics(run):
        frames = sorted({int(item[0].frame_index) for item in run})
        span = (
            (frames[-1] - frames[0]) / max(float(fps), 1e-6)
            if len(frames) >= 2
            else 0.0
        )
        person_observed = any(item[0].observed for item in run)
        vehicle_observed = any(item[1].observed for item in run)
        evidence_weight = sum(
            1.0 if item[0].observed and item[1].observed else 0.35
            for item in run
        )
        return frames, span, person_observed, vehicle_observed, evidence_weight

    valid_runs = []
    longest_support_span = 0.0
    longest_support_frames = 0
    strongest_evidence_weight = 0.0
    for run in support_runs:
        frames, span, person_seen, vehicle_seen, evidence_weight = _run_metrics(run)
        longest_support_span = max(longest_support_span, span)
        longest_support_frames = max(longest_support_frames, len(frames))
        strongest_evidence_weight = max(
            strongest_evidence_weight, float(evidence_weight)
        )
        if (
            len(frames) >= 2
            and span >= float(min_dwell_seconds)
            and person_seen
            and vehicle_seen
            and evidence_weight >= 1.5
        ):
            valid_runs.append((run, frames, span, evidence_weight))
    selected_run = (
        max(valid_runs, key=lambda item: (item[2], item[3], len(item[1])))
        if valid_runs
        else None
    )
    sustained = selected_run is not None

    # Entering/exiting a vehicle can be brief. Permit one strong endpoint only
    # when the track itself has at least two temporal samples; a one-frame
    # passer-by cannot become a durable A-C relation.
    endpoint_evidence = [pairs[0], pairs[-1]]
    strong_endpoint = [
        item
        for item in endpoint_evidence
        if item[3] >= 0.20 and item[0].observed and item[1].observed
    ]
    endpoint_transition = len(pairs) >= 2 and bool(strong_endpoint)
    if not sustained and not endpoint_transition:
        return CostCell.rejected(
            "person_vehicle_dwell_failed",
            support_frames=longest_support_frames,
            support_seconds=longest_support_span,
            strongest_evidence_weight=strongest_evidence_weight,
            minimum_proximity=min(item[2] for item in supportive),
            maximum_iom=max(item[3] for item in supportive),
        )

    evidence = selected_run[0] if sustained else strong_endpoint
    support_frames = selected_run[1] if sustained else [
        int(item[0].frame_index) for item in strong_endpoint
    ]
    normalize_dt = bool(
        resolved_cost_config.normalize_distance_time_by_gate
    )
    weights = dict(resolved_cost_config.ac_weights) or {
        "endpoint_proximity": 1.0,
        "overlap": 0.8,
        "time": 0.2,
        "quality": 0.3,
        "uncertainty": 0.15,
        "continuity": 0.25,
    }

    def _pair_raw_features(item):
        (
            _, _, pair_proximity, pair_iom, pair_gap_frames, pair_gap,
            pair_quality, pair_uncertainty,
        ) = item
        time_feature = (
            _hybrid_time_soft_penalty(
                pair_gap_frames,
                fps,
                max_pair_gap_seconds,
                max_observation_gap_frames,
                resolved_cost_config.observation_time_soft_kappa,
            )
            if soft_time else _gate_fraction(
                pair_gap, max_pair_gap_seconds, normalize_dt
            )
        )
        return {
            "endpoint_proximity": _gate_fraction(
                min(pair_proximity, float(proximity_gate)),
                proximity_gate,
                normalize_dt,
            ),
            "overlap": 1.0 - pair_iom,
            "time": time_feature,
            "quality": pair_quality,
            "uncertainty": (
                float(pair_uncertainty) / 0.15
                if float(pair_uncertainty) > 0.0 else 0.0
            ),
            "continuity": (
                1.0 - min(len(support_frames) / max(len(pairs), 1), 1.0)
            ),
        }

    # Candidate-frame selection must use the same enabled features and trial
    # weights as the returned cell. Otherwise a D+T ablation would still pick
    # its observation using hidden overlap/quality/uncertainty components.
    best = min(
        evidence,
        key=lambda item: sum(
            _weighted_components(_pair_raw_features(item), weights).values()
        ),
    )
    (
        person, vehicle, proximity, iom, _pair_gap_frames, pair_gap,
        quality, uncertainty,
    ) = best
    if proximity > float(proximity_gate) and iom < 0.05:
        return CostCell.rejected(
            "person_vehicle_gate_failed", proximity=proximity, overlap=iom
        )

    raw_features = _pair_raw_features(best)
    components = _weighted_components(raw_features, weights)
    return CostCell(
        valid=True,
        total=float(sum(components.values())),
        components=components,
        best_release_frame=None,
        reject_reason=None,
        raw_features=raw_features,
        weights=weights,
    )


def build_ba_costs(
    releases: Sequence[ReleaseHypothesis],
    person_tracks: Mapping[ActorKey, Sequence[ActorObservation]],
    fps: float,
    **kwargs,
) -> Dict[ActorKey, CostCell]:
    return {
        key: compute_c_ba(releases, observations, fps, **kwargs)
        for key, observations in person_tracks.items()
    }


def build_bc_costs(
    releases: Sequence[ReleaseHypothesis],
    vehicle_tracks: Mapping[ActorKey, Sequence[ActorObservation]],
    fps: float,
    **kwargs,
) -> Dict[ActorKey, CostCell]:
    return {
        key: compute_c_bc(releases, observations, fps, **kwargs)
        for key, observations in vehicle_tracks.items()
    }


def build_ac_costs(
    person_tracks: Mapping[ActorKey, Sequence[ActorObservation]],
    vehicle_tracks: Mapping[ActorKey, Sequence[ActorObservation]],
    fps: float,
    **kwargs,
) -> Dict[Tuple[ActorKey, ActorKey], CostCell]:
    return {
        (person_key, vehicle_key): compute_c_ac(
            person_observations, vehicle_observations, fps, **kwargs
        )
        for person_key, person_observations in person_tracks.items()
        for vehicle_key, vehicle_observations in vehicle_tracks.items()
    }


def topk_valid(
    costs: Mapping[object, CostCell], k: int = 5
) -> List[Tuple[object, CostCell]]:
    return sorted(
        (
            (key, cell)
            for key, cell in costs.items()
            if cell.valid and np.isfinite(float(cell.total))
        ),
        key=lambda item: (float(item[1].total), repr(item[0])),
    )[: max(int(k), 0)]


# Mathematical aliases used in design notes.
C_BA = compute_c_ba
C_BC = compute_c_bc
C_AC = compute_c_ac
