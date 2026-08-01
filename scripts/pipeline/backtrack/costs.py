"""Explicit cost cells for litter-person-vehicle backtracking."""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .trajectory import ReleaseHypothesis


ActorKey = Tuple[str, int]


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

    def __post_init__(self):
        object.__setattr__(self, "cls_name", str(self.cls_name).lower())
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "frame_index", int(self.frame_index))
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
        return cls(
            cls_name=tracklet.class_name,
            track_id=tracklet.track_id,
            frame_index=state.frame_index,
            bbox=state.bbox_xyxy,
            confidence=float(confidence),
            observed=bool(state.observed),
            covariance_uv=state.covariance[:2, :2],
            source=source,
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
        str(name): float(raw_features[name]) * float(weights.get(name, 1.0))
        for name in raw_features
    }


def _nearest_observation(
    observations: Sequence[ActorObservation],
    frame_index: int,
    max_gap_frames: int,
) -> Optional[ActorObservation]:
    if not observations:
        return None
    observation = min(
        observations,
        key=lambda item: abs(int(item.frame_index) - int(frame_index)),
    )
    if abs(int(observation.frame_index) - int(frame_index)) > int(max_gap_frames):
        return None
    return observation


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
    max_uncertainty_height_ratio: float = 1.5,
    normalized_distance_gate: float = 0.85,
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
    max_gap = max(int(round(float(max_observation_gap_seconds) * float(fps))), 0)

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
        displacement_norm = float(np.linalg.norm(displacement))
        velocity_norm = float(np.linalg.norm(release.velocity_uv))
        direction_cost = 0.0
        if displacement_norm > 0.12 * person.height and velocity_norm > 1e-6:
            cosine = float(
                displacement @ release.velocity_uv
                / (displacement_norm * velocity_norm)
            )
            direction_cost = 0.35 * max(0.0, -cosine)

        time_gap = abs(person.frame_index - release.frame_index) / max(
            float(fps), 1e-6
        )
        quality = _quality_cost(person)
        normalized_covariance = attribution_covariance / max(
            person.height * person.height, 1.0
        )
        _, logdet = np.linalg.slogdet(np.eye(2) + normalized_covariance)
        uncertainty_cost = 0.5 * max(float(logdet), 0.0)
        raw_features = {
            "release_distance": normalized_distance,
            "uncertainty": max(float(logdet), 0.0),
            "time": time_gap,
            "quality": quality,
            "direction": (
                direction_cost / 0.35 if direction_cost > 0.0 else 0.0
            ),
            "release_prior": float(release.prior_cost),
        }
        weights = {
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
    max_uncertainty_height_ratio: float = 1.5,
    normalized_distance_gate: float = 0.8,
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
    max_gap = max(int(round(float(max_observation_gap_seconds) * float(fps))), 0)

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

        # A modest expansion permits a window/door release, but a far vehicle
        # can never become valid merely through a low additive cost.
        x1, y1, x2, y2 = vehicle.bbox
        margin_x = 0.18 * vehicle.width
        margin_y = 0.15 * vehicle.height
        residual = _point_to_rect_vector(
            release.mean_uv,
            (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y),
        )
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
        time_gap = abs(vehicle.frame_index - release.frame_index) / max(
            float(fps), 1e-6
        )
        raw_features = {
            "direct_distance": normalized_distance,
            "spatial_mahalanobis": mahalanobis,
            "uncertainty": np.log1p(
                (
                    float(np.trace(release.covariance_uv))
                    + float(np.trace(vehicle.covariance_uv))
                )
                / max(vehicle.height * vehicle.height, 1.0)
            ),
            "time": time_gap,
            "quality": quality,
            "release_prior": float(release.prior_cost),
        }
        weights = {
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
    proximity_gate: float = 1.2,
    max_uncertainty_height_ratio: float = 1.5,
    min_dwell_seconds: float = 0.15,
    max_support_gap_seconds: float = 0.50,
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
    max_gap = max(int(round(float(max_pair_gap_seconds) * float(fps))), 0)

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
            abs(int(vehicle.frame_index) - int(person.frame_index))
            > max_gap
        ):
            continue
        foot_distance = float(np.linalg.norm(person.footpoint - vehicle.footpoint))
        vehicle_scale = max(np.hypot(vehicle.width, vehicle.height), 1.0)
        pair_covariance = person.covariance_uv + vehicle.covariance_uv
        if _uncertainty_ratio(
            pair_covariance, max(person.height, vehicle.height)
        ) > float(max_uncertainty_height_ratio):
            continue
        proximity = foot_distance / vehicle_scale
        iom = _intersection_over_minimum(person.bbox, vehicle.bbox)
        pair_gap = abs(person.frame_index - vehicle.frame_index) / max(
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
            (person, vehicle, proximity, iom, pair_gap, quality, uncertainty)
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
    best = min(
        evidence,
        key=lambda item: (
            min(item[2], float(proximity_gate))
            + 0.8 * (1.0 - item[3])
            + 0.2 * item[4]
            + 0.3 * item[5]
            + item[6]
        ),
    )
    person, vehicle, proximity, iom, pair_gap, quality, uncertainty = best
    if proximity > float(proximity_gate) and iom < 0.05:
        return CostCell.rejected(
            "person_vehicle_gate_failed", proximity=proximity, overlap=iom
        )

    raw_features = {
        "endpoint_proximity": min(proximity, float(proximity_gate)),
        "overlap": 1.0 - iom,
        "time": pair_gap,
        "quality": quality,
        "uncertainty": (
            float(uncertainty) / 0.15 if float(uncertainty) > 0.0 else 0.0
        ),
        "continuity": (
            1.0 - min(len(support_frames) / max(len(pairs), 1), 1.0)
        ),
    }
    weights = {
        "endpoint_proximity": 1.0,
        "overlap": 0.8,
        "time": 0.2,
        "quality": 0.3,
        "uncertainty": 0.15,
        "continuity": 0.25,
    }
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
