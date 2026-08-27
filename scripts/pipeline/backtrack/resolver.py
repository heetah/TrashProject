# -*- coding: utf-8 -*-
"""Kalman/RTS + reverse trajectory + explicit costs + min-cost-flow resolver."""
from dataclasses import dataclass, field, replace
import math
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .costs import (
    ActorObservation,
    BacktrackCostConfig,
    CostCell,
    build_ac_costs,
    build_ba_costs,
    build_bc_costs,
    compute_c_ba,
    compute_c_bc,
    topk_valid,
)
from .flow import COST_SCALE, Assignment, RouteCandidate, solve_event_routes
from .kalman import KalmanConfig, TrackMeasurement, smooth_tracklet
from .trajectory import (
    ReleaseHypothesis,
    airborne_prefix,
    build_release_hypotheses,
)


ActorKey = Tuple[str, int]


def _float_env(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _int_env(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


@dataclass(frozen=True)
class SmartBacktrackConfig:
    max_back_frames: int = 24
    top_k_people: int = 5
    top_k_vehicles: int = 5
    dustbin_cost: float = 7.0
    null_vehicle_penalty: float = 1.4
    direct_vehicle_penalty: float = 0.9
    ac_weight: float = 0.75
    bc_support_bonus: float = 0.25
    sigma_floor_px: float = 2.0
    two_point_max_back_seconds: float = 0.4
    two_point_prior_cost: float = 1.0
    max_forward_release_seconds: float = 0.5
    release_window_prior_weight: float = 0.35
    # Physical association gates. Defaults preserve production behavior;
    # research replays may override them explicitly.
    max_observation_gap_seconds: float = 0.25
    max_observation_gap_frames: Optional[int] = None
    normalized_distance_gate_person: float = 0.85
    normalized_distance_gate_vehicle: float = 0.8
    cost_config: BacktrackCostConfig = field(default_factory=BacktrackCostConfig)
    # Research switches default to the current production behavior.  They are
    # intentionally constructor-only; production never reads them from env.
    use_kalman_rts: bool = True
    confidence_aware_kalman: bool = True
    use_uncertainty_gates: bool = True
    kalman_process_noise_scale: float = 1.0
    kalman_measurement_noise_scale: float = 1.0
    kalman_max_extrapolation_seconds: Optional[float] = None
    use_reverse_trajectory: bool = True
    confidence_weighted_trajectory: bool = True

    @classmethod
    def from_env(cls, fps=10.0):
        # 24 frames was the original 10 FPS research search span (=2.4 s).
        # Scale the computational guard with FPS; it is not a physical gate.
        default_max_back = max(1, int(round(float(fps or 10.0) * 2.4)))
        study_stage = str(
            os.environ.get("SMART_BACKTRACK_STUDY_STAGE", "full")
        ).strip().lower()
        if study_stage not in {
            "distance_time", "kalman_rts", "confidence", "uncertainty",
            "reverse", "full"
        }:
            study_stage = "full"
        distance_weight = max(
            0.0, _float_env("SMART_BACKTRACK_DT_DISTANCE_WEIGHT", 1.0)
        )
        time_weight = max(
            0.0, _float_env("SMART_BACKTRACK_DT_TIME_WEIGHT", 1.0)
        )
        if distance_weight + time_weight <= 0.0:
            distance_weight = time_weight = 1.0
        cost_config = BacktrackCostConfig.for_stage(
            study_stage,
            distance_weight=distance_weight,
            time_weight=time_weight,
        )
        boundary_depth_weight = max(
            0.0,
            _float_env("SMART_BACKTRACK_BC_BOUNDARY_DEPTH_WEIGHT", 0.0),
        )
        if study_stage in {"reverse", "full"} and boundary_depth_weight > 0.0:
            cost_config = replace(
                cost_config,
                bc_weights={
                    **cost_config.bc_weights,
                    "boundary_depth": boundary_depth_weight,
                },
            )
        return cls(
            max_back_frames=max(
                1,
                _int_env("SMART_BACKTRACK_MAX_BACK_FRAMES", default_max_back),
            ),
            top_k_people=max(1, _int_env("SMART_BACKTRACK_TOPK_PERSON", 5)),
            top_k_vehicles=max(1, _int_env("SMART_BACKTRACK_TOPK_VEHICLE", 5)),
            dustbin_cost=max(0.1, _float_env("SMART_BACKTRACK_DUSTBIN_COST", 7.0)),
            null_vehicle_penalty=max(
                0.0, _float_env("SMART_BACKTRACK_NULL_VEHICLE_COST", 1.4)
            ),
            direct_vehicle_penalty=max(
                0.0, _float_env("SMART_BACKTRACK_DIRECT_VEHICLE_COST", 0.9)
            ),
            ac_weight=max(0.0, _float_env("SMART_BACKTRACK_AC_WEIGHT", 0.75)),
            bc_support_bonus=max(
                0.0, _float_env("SMART_BACKTRACK_BC_SUPPORT_BONUS", 0.25)
            ),
            sigma_floor_px=max(
                0.25, _float_env("SMART_BACKTRACK_SIGMA_FLOOR", 2.0)
            ),
            two_point_max_back_seconds=max(
                0.0,
                _float_env("SMART_BACKTRACK_TWO_POINT_MAX_BACK_SEC", 0.4),
            ),
            two_point_prior_cost=max(
                0.0,
                _float_env("SMART_BACKTRACK_TWO_POINT_PRIOR_COST", 1.0),
            ),
            max_forward_release_seconds=max(
                0.0,
                _float_env("SMART_BACKTRACK_MAX_FORWARD_RELEASE_SEC", 0.5),
            ),
            release_window_prior_weight=max(
                0.0,
                _float_env("SMART_BACKTRACK_RELEASE_WINDOW_WEIGHT", 0.35),
            ),
            cost_config=cost_config,
            use_kalman_rts=study_stage in {
                "kalman_rts", "uncertainty", "reverse", "full"
            },
            confidence_aware_kalman=study_stage != "kalman_rts",
            use_uncertainty_gates=study_stage in {
                "uncertainty", "reverse", "full"
            },
            use_reverse_trajectory=study_stage in {"reverse", "full"},
            confidence_weighted_trajectory=study_stage == "full",
        )


@dataclass(frozen=True)
class SmartResolution:
    litter_id: int
    person_key: Optional[ActorKey]
    vehicle_key: Optional[ActorKey]
    direct_vehicle: bool
    total_cost: float
    margin_to_second: Optional[float]
    actor_margins: Mapping[str, object]
    route_type: str
    route_id: str
    release_frame: Optional[int]
    release_point: Optional[Tuple[float, float]]
    release_covariance: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    components: Mapping[str, object]
    routes: Sequence[RouteCandidate]

    @property
    def actor_key(self):
        return self.person_key if self.person_key is not None else self.vehicle_key


def _actor_specific_margins(routes: Sequence[RouteCandidate]):
    """Separate route ambiguity from person/vehicle identity ambiguity.

    Multiple releases or route types can describe the same actor.  Those are
    first collapsed to the actor's cheapest complete route; only then is the
    runner-up *distinct actor* compared.  This prevents same-actor route ties
    from being mistaken for an identity tie.
    """

    def _margin_for(field_name):
        best_by_actor = {}
        for route in routes:
            if route.is_null or not math.isfinite(float(route.cost)):
                continue
            actor_key = getattr(route, field_name)
            if actor_key is None:
                continue
            old = best_by_actor.get(actor_key)
            candidate = (float(route.cost), str(route.route_id))
            if old is None or candidate < old:
                best_by_actor[actor_key] = candidate
        ranked = sorted(
            (
                (cost_and_route[0], repr(actor_key), actor_key, cost_and_route[1])
                for actor_key, cost_and_route in best_by_actor.items()
            ),
            key=lambda item: (item[0], item[1], item[3]),
        )
        if not ranked:
            return {
                "best_key": None,
                "best_cost": None,
                "second_key": None,
                "second_cost": None,
                "margin": None,
                "tie_count": 0,
            }
        best_cost, _, best_key, _ = ranked[0]
        second = ranked[1] if len(ranked) >= 2 else None
        # Flow costs are quantized to 0.001.  Count identities within the same
        # quantized objective cell as tied, while preserving the raw margin.
        tie_count = sum(
            1 for cost, *_rest in ranked
            if int(math.floor(cost * COST_SCALE + 0.5))
            == int(math.floor(best_cost * COST_SCALE + 0.5))
        )
        return {
            "best_key": best_key,
            "best_cost": best_cost,
            "second_key": second[2] if second is not None else None,
            "second_cost": second[0] if second is not None else None,
            "margin": (
                max(0.0, float(second[0] - best_cost))
                if second is not None else None
            ),
            "tie_count": int(tie_count),
        }

    finite_null = [
        float(route.cost) for route in routes
        if route.is_null and math.isfinite(float(route.cost))
    ]
    finite_non_null = [
        float(route.cost) for route in routes
        if not route.is_null and math.isfinite(float(route.cost))
    ]
    null_cost = min(finite_null) if finite_null else None
    best_non_null = min(finite_non_null) if finite_non_null else None
    return {
        "person": _margin_for("person_key"),
        "vehicle": _margin_for("vehicle_key"),
        "null": {
            "null_cost": null_cost,
            "best_non_null_cost": best_non_null,
            # Positive: non-NULL is cheaper. Negative: NULL is safer.
            "margin": (
                float(null_cost - best_non_null)
                if null_cost is not None and best_non_null is not None
                else None
            ),
        },
    }


def _cell_payload(cell: Optional[CostCell]):
    if cell is None:
        return None
    raw_features = {
        str(key): float(value)
        for key, value in cell.raw_features.items()
        if math.isfinite(float(value))
    }
    weights = {
        str(key): float(value)
        for key, value in cell.weights.items()
        if math.isfinite(float(value))
    }
    components = {
        str(key): float(value) for key, value in cell.components.items()
        if math.isfinite(float(value))
    }
    return {
        "valid": bool(cell.valid),
        "total": float(cell.total) if math.isfinite(cell.total) else None,
        "component_semantics": (
            "weighted_contribution" if cell.valid else "reject_diagnostics"
        ),
        "components": components,
        "raw_features": raw_features,
        "weights": weights,
        "feature_details": {
            name: {
                "raw": raw_features.get(name),
                "weight": weights.get(name, 1.0),
                "contribution": components.get(name),
            }
            for name in sorted(set(raw_features) | set(weights) | set(components))
        },
        "best_release_frame": cell.best_release_frame,
        "reject_reason": cell.reject_reason,
    }


def _release_payload(releases, frame_index):
    if frame_index is None or not releases:
        return {}
    release = min(
        releases,
        key=lambda item: abs(int(item.frame_index) - int(frame_index)),
    )
    return {
        "release_frame": int(release.frame_index),
        "release_point": tuple(float(v) for v in release.mean_uv),
        "release_covariance": tuple(
            tuple(float(v) for v in row) for row in release.covariance_uv
        ),
        "release_velocity": tuple(float(v) for v in release.velocity_uv),
        "release_model": str(release.model),
    }


def _release_hypothesis_payload(release):
    return {
        "frame_index": int(release.frame_index),
        "mean_uv": [float(value) for value in release.mean_uv],
        "covariance_uv": [
            [float(value) for value in row]
            for row in release.covariance_uv
        ],
        "velocity_uv": [float(value) for value in release.velocity_uv],
        "model": str(release.model),
        "prior_cost": float(release.prior_cost),
        "observation_gap_frames": release.observation_gap_frames,
        "zero_cost_window_start_frame": release.zero_cost_window_start_frame,
        "zero_cost_window_end_frame": release.zero_cost_window_end_frame,
        "window_prior_cost": float(release.window_prior_cost),
        "direction_consistency": release.direction_consistency,
        "source_direction_uv": (
            list(release.source_direction_uv)
            if release.source_direction_uv is not None else None
        ),
        "search_truncated": bool(release.search_truncated),
        "truncation_reason": release.truncation_reason,
    }


def _route_payload(route):
    return {
        "route_id": str(route.route_id),
        "route_type": str(route.route_type),
        "person_key": (
            list(route.person_key) if route.person_key is not None else None
        ),
        "vehicle_key": (
            list(route.vehicle_key) if route.vehicle_key is not None else None
        ),
        "cost": float(route.cost) if math.isfinite(float(route.cost)) else None,
        "metadata": dict(route.metadata or {}),
    }


def _actor_candidate_payload(actor_key, observations):
    """JSON-safe summary; tolerant of lightweight monkeypatched test tracks."""

    frames = [
        int(observation.frame_index)
        for observation in observations
        if hasattr(observation, "frame_index")
    ]
    return {
        "actor_key": list(actor_key),
        "observation_count": len(observations),
        "observed_count": sum(
            bool(getattr(observation, "observed", False))
            for observation in observations
        ),
        "first_frame": min(frames) if frames else None,
        "last_frame": max(frames) if frames else None,
    }


class SmartBacktrackResolver:
    """Build full B-A-C route hypotheses and solve their min-cost flow."""

    def __init__(self, fps=30.0, config=None):
        self.fps = float(fps) if fps and float(fps) > 0.0 else 30.0
        self.config = config or SmartBacktrackConfig.from_env(fps=self.fps)

    def _kalman_max_extrapolation_frames(self, fps: float) -> int:
        if self.config.kalman_max_extrapolation_seconds is None:
            return int(self.config.max_back_frames + 4)
        return max(
            0,
            int(round(
                float(fps) * self.config.kalman_max_extrapolation_seconds
            )),
        )

    def _release_hypotheses(self, task) -> List[ReleaseHypothesis]:
        points = list(task.get("history") or [])
        frames = list(task.get("history_frames") or [])
        confidence_values = list(task.get("history_confidences") or [])
        count = min(len(points), len(frames))
        if count:
            points, frames = points[-count:], frames[-count:]
            if confidence_values:
                confidence_values = confidence_values[-count:]
                if len(confidence_values) < count:
                    confidence_values = (
                        [1.0] * (count - len(confidence_values))
                        + confidence_values
                    )
            else:
                confidence_values = [1.0] * count
        birth_frame = int(task.get("birth_frame", task.get("confirm_frame", 0)))
        if not self.config.use_reverse_trajectory:
            point = task.get("birth_centroid")
            if point is None and points:
                point = points[0]
            if point is None:
                point = (0.0, 0.0)
            return [ReleaseHypothesis(
                frame_index=birth_frame,
                mean_uv=np.asarray(point, dtype=float),
                covariance_uv=np.eye(2, dtype=float) * self.config.sigma_floor_px ** 2,
                velocity_uv=np.zeros(2, dtype=float),
                model="birth_anchor",
                prior_cost=0.0,
            )]
        points, frames = airborne_prefix(
            points,
            frames,
            fps=float(task.get("fps", self.fps) or self.fps),
        )
        confidence_values = confidence_values[:len(points)]
        return build_release_hypotheses(
            points,
            frames,
            birth_frame=birth_frame,
            max_back_frames=self.config.max_back_frames,
            fps=float(task.get("fps", self.fps) or self.fps),
            sigma_floor_px=self.config.sigma_floor_px,
            confidences=(
                confidence_values if self.config.confidence_weighted_trajectory
                else [1.0] * len(points)
            ),
            fallback_prior_cost=self.config.dustbin_cost,
            two_point_max_back_seconds=self.config.two_point_max_back_seconds,
            two_point_prior_cost=self.config.two_point_prior_cost,
            max_forward_release_seconds=self.config.max_forward_release_seconds,
            window_prior_weight=self.config.release_window_prior_weight,
        )

    def _build_actor_tracks(self, task):
        fps = float(task.get("fps", self.fps) or self.fps)
        snapshots_by_uid = {}
        all_frames = []
        for frame_snapshot in task.get("actor_frames", []) or []:
            frame_index = int(frame_snapshot.get("frame_index", 0))
            all_frames.append(frame_index)
            for snapshot in frame_snapshot.get("actors", []) or []:
                try:
                    actor_key = (
                        str(snapshot.get("cls", "")).lower(),
                        int(snapshot["track_id"]),
                    )
                    box = tuple(map(float, snapshot["box"][:4]))
                except (KeyError, TypeError, ValueError):
                    continue
                if actor_key[0] not in ("person", "vehicle", "scooter"):
                    continue
                tracklet_uid = str(
                    snapshot.get(
                        "tracklet_uid",
                        "{}:{}".format(actor_key[0], actor_key[1]),
                    )
                )
                group_key = (actor_key, tracklet_uid)
                copied = dict(snapshot)
                copied["box"] = box
                snapshots_by_uid.setdefault(group_key, []).append(
                    (frame_index, copied)
                )

        actor_tracks = {}
        actor_track_scores = {}
        birth_frame = int(task.get("birth_frame", task.get("confirm_frame", 0)))
        if not self.config.use_kalman_rts:
            # Baseline study mode: retain detector observations and IDs, but do
            # not inject RTS positions/covariance into a distance/time result.
            for (key, _tracklet_uid), items in snapshots_by_uid.items():
                observations = []
                for frame_index, snapshot in items:
                    if not bool(snapshot.get("observed", True)):
                        continue
                    try:
                        observations.append(ActorObservation.from_snapshot(
                            snapshot, frame_index=frame_index
                        ))
                    except (KeyError, TypeError, ValueError):
                        continue
                if not observations:
                    continue
                score = (
                    min(abs(item.frame_index - birth_frame) for item in observations),
                    -len(observations), int(observations[0].frame_index),
                )
                if key not in actor_tracks or score < actor_track_scores[key]:
                    actor_tracks[key] = observations
                    actor_track_scores[key] = score
            return actor_tracks
        for (key, _tracklet_uid), items in snapshots_by_uid.items():
            # Cached reuse is a prediction target, never a detector update.
            real_items = [
                (frame_index, snapshot)
                for frame_index, snapshot in items
                if bool(snapshot.get("observed", True))
            ]
            if not real_items:
                continue
            measurements = []
            valid_real_items = []
            for frame_index, snapshot in real_items:
                try:
                    measurement = TrackMeasurement(
                        frame_index=frame_index,
                        bbox_xyxy=tuple(snapshot["box"]),
                        confidence=(
                            float(
                                snapshot.get(
                                    "confidence",
                                    snapshot.get(
                                        "pose_conf", snapshot.get("conf", 1.0)
                                    ),
                                )
                            )
                            if self.config.confidence_aware_kalman else 1.0
                        ),
                    )
                except (TypeError, ValueError):
                    continue
                measurements.append(measurement)
                valid_real_items.append((frame_index, snapshot))
            real_items = valid_real_items
            if not measurements:
                continue
            max_extrapolation_frames = (
                self._kalman_max_extrapolation_frames(fps)
            )
            kalman_config = KalmanConfig(
                frames_per_second=fps,
                max_extrapolation_frames=max_extrapolation_frames,
            )
            kalman_config.process_position_accel_fractions = {
                name: float(value) * self.config.kalman_process_noise_scale
                for name, value in
                kalman_config.process_position_accel_fractions.items()
            }
            kalman_config.process_log_size_accel_stds = {
                name: float(value) * self.config.kalman_process_noise_scale
                for name, value in
                kalman_config.process_log_size_accel_stds.items()
            }
            kalman_config.measurement_position_std_fraction *= (
                self.config.kalman_measurement_noise_scale
            )
            kalman_config.measurement_position_std_floor *= (
                self.config.kalman_measurement_noise_scale
            )
            kalman_config.measurement_log_size_std *= (
                self.config.kalman_measurement_noise_scale
            )
            try:
                smoothed = smooth_tracklet(
                    measurements,
                    class_name=key[0],
                    track_id=key,
                    config=kalman_config,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue

            first_observed = int(smoothed.frames[0])
            last_observed = int(smoothed.frames[-1])
            first_frame = max(
                birth_frame - self.config.max_back_frames,
                first_observed - max_extrapolation_frames,
            )
            last_frame = min(
                int(task.get("confirm_frame", last_observed))
                + self.config.max_back_frames,
                last_observed + max_extrapolation_frames,
            )
            observed_by_frame = {
                frame_index: snapshot for frame_index, snapshot in real_items
            }
            real_frames = sorted(observed_by_frame)
            observations = []
            for frame_index in range(first_frame, last_frame + 1):
                try:
                    state = smoothed.state_at(frame_index)
                except ValueError:
                    continue
                bbox = state.bbox_xyxy
                source_snapshot = observed_by_frame.get(frame_index)
                confidence = (
                    float(
                        source_snapshot.get(
                            "confidence",
                            source_snapshot.get(
                                "pose_conf", source_snapshot.get("conf", 1.0)
                            ),
                        )
                    )
                    if source_snapshot is not None
                    else 0.5
                )
                observations.append(
                    ActorObservation(
                        cls_name=key[0],
                        track_id=key[1],
                        frame_index=frame_index,
                        bbox=bbox,
                        confidence=confidence,
                        observed=source_snapshot is not None,
                        covariance_uv=state.covariance[:2, :2],
                        source=(
                            str(source_snapshot.get("source", "detector"))
                            if source_snapshot is not None
                            else "kalman_rts"
                        ),
                        evidence_frame_index=min(
                            real_frames,
                            key=lambda observed_frame: (
                                abs(observed_frame - frame_index),
                                observed_frame,
                            ),
                        ),
                    )
                )
            if observations:
                # External actor_key remains compatible with drawing/OCR. If an
                # upstream tracker recycled the same numeric ID, keep only the
                # physical tracklet closest to this litter's birth.
                score = (
                    min(abs(frame_index - birth_frame) for frame_index, _ in real_items),
                    -len(real_items),
                    int(real_items[0][0]),
                )
                if key not in actor_tracks or score < actor_track_scores[key]:
                    actor_tracks[key] = observations
                    actor_track_scores[key] = score
        return actor_tracks

    def build_routes(self, task) -> List[RouteCandidate]:
        fps = float(task.get("fps", self.fps) or self.fps)
        max_extrapolation_frames = (
            self._kalman_max_extrapolation_frames(fps)
        )
        releases = self._release_hypotheses(task)
        actor_tracks = self._build_actor_tracks(task)
        person_tracks = {
            key: values for key, values in actor_tracks.items()
            if key[0] == "person"
        }
        vehicle_tracks = {
            key: values for key, values in actor_tracks.items()
            if key[0] in ("vehicle", "scooter")
        }
        litter_points = list(task.get("history") or [])
        litter_frames = list(task.get("history_frames") or [])
        litter_count = min(len(litter_points), len(litter_frames))
        bc_context = {}
        if litter_count:
            bc_context = {
                "litter_last_point": litter_points[litter_count - 1],
                "litter_last_frame": litter_frames[litter_count - 1],
            }
        max_uncertainty_ratio = (
            1.5 if self.config.use_uncertainty_gates else float("inf")
        )
        ba_costs = build_ba_costs(
            releases, person_tracks, fps,
            cost_config=self.config.cost_config,
            max_uncertainty_height_ratio=max_uncertainty_ratio,
            max_observation_gap_seconds=self.config.max_observation_gap_seconds,
            max_observation_gap_frames=self.config.max_observation_gap_frames,
            normalized_distance_gate=self.config.normalized_distance_gate_person,
        )
        bc_costs = build_bc_costs(
            releases, vehicle_tracks, fps,
            cost_config=self.config.cost_config,
            max_uncertainty_height_ratio=max_uncertainty_ratio,
            max_observation_gap_seconds=self.config.max_observation_gap_seconds,
            max_observation_gap_frames=self.config.max_observation_gap_frames,
            normalized_distance_gate=self.config.normalized_distance_gate_vehicle,
            **bc_context,
        )
        ac_costs = build_ac_costs(
            person_tracks, vehicle_tracks, fps,
            cost_config=self.config.cost_config,
            max_uncertainty_height_ratio=max_uncertainty_ratio,
            max_pair_gap_seconds=self.config.max_observation_gap_seconds,
            max_observation_gap_frames=self.config.max_observation_gap_frames,
        )
        # Keep pre-collapse per-release cells for the research sidecar.  A
        # collapsed pair cost alone cannot tell whether the GT release was
        # missing, rejected by a gate, or merely lost during route ranking.
        ba_by_person_release = {
            (person_key, int(release.frame_index)): compute_c_ba(
                [release], observations, fps,
                cost_config=self.config.cost_config,
                max_uncertainty_height_ratio=max_uncertainty_ratio,
                max_observation_gap_seconds=self.config.max_observation_gap_seconds,
                max_observation_gap_frames=self.config.max_observation_gap_frames,
                normalized_distance_gate=self.config.normalized_distance_gate_person,
            )
            for person_key, observations in person_tracks.items()
            for release in releases
        }
        # C_BC depends on vehicle and release time, never on person. Cache this
        # event table once instead of repeating it for every B-A candidate.
        bc_by_vehicle_release = {
            (vehicle_key, int(release.frame_index)): compute_c_bc(
                [release], observations, fps,
                cost_config=self.config.cost_config,
                max_uncertainty_height_ratio=max_uncertainty_ratio,
                max_observation_gap_seconds=self.config.max_observation_gap_seconds,
                max_observation_gap_frames=self.config.max_observation_gap_frames,
                normalized_distance_gate=self.config.normalized_distance_gate_vehicle,
                **bc_context,
            )
            for vehicle_key, observations in vehicle_tracks.items()
            for release in releases
        }
        # Rank person identities only after complete A/C routes exist. BA-only
        # pruning can delete the true route when BA is slightly worse but AC is
        # much stronger.
        people = topk_valid(ba_costs, len(ba_costs))
        all_direct_vehicles = topk_valid(bc_costs, len(bc_costs))
        direct_vehicles = all_direct_vehicles[: self.config.top_k_vehicles]

        routes: List[RouteCandidate] = []
        person_routes: List[RouteCandidate] = []
        pre_prune_routes: List[RouteCandidate] = []
        for person_key, _collapsed_ba in people:
            ba_by_release = []
            for release in releases:
                ba_at_release = ba_by_person_release[
                    (person_key, int(release.frame_index))
                ]
                if ba_at_release.valid:
                    ba_by_release.append((release, ba_at_release))
            if not ba_by_release:
                continue
            release_for_null, ba_for_null = min(
                ba_by_release,
                key=lambda item: (
                    float(item[1].total),
                    int(item[0].frame_index),
                ),
            )
            metadata = {
                "costs": {
                    "BA": _cell_payload(ba_for_null),
                    "AC": None,
                    "BC": None,
                },
                **_release_payload(releases, release_for_null.frame_index),
            }
            person_null_route = RouteCandidate(
                route_id="person:{}:null".format(person_key[1]),
                cost=float(
                    ba_for_null.total + self.config.null_vehicle_penalty
                ),
                person_key=person_key,
                vehicle_key=None,
                route_type="person",
                metadata=metadata,
            )
            person_routes.append(person_null_route)
            pre_prune_routes.append(person_null_route)
            # A person route does not hard-require B near C: somebody may leave
            # a vehicle, walk away, then throw. B-C is only bounded support.
            # Enumerate release time before minimizing the complete route:
            # min_t [C_BA(B,A,t) + w*C_AC(A,C) - support(C_BC(B,C,t))].
            linked_vehicles = []
            for (candidate_person_key, vehicle_key), ac in ac_costs.items():
                if candidate_person_key != person_key or not ac.valid:
                    continue
                time_options = []
                for release, ba_at_release in ba_by_release:
                    bc_at_release = bc_by_vehicle_release[
                        (vehicle_key, int(release.frame_index))
                    ]
                    support_bonus = (
                        self.config.bc_support_bonus
                        * math.exp(-float(bc_at_release.total))
                        if bc_at_release.valid
                        else 0.0
                    )
                    total = max(
                        0.0,
                        float(ba_at_release.total)
                        + self.config.ac_weight * float(ac.total)
                        - support_bonus,
                    )
                    time_options.append(
                        (
                            float(total),
                            int(release.frame_index),
                            release,
                            ba_at_release,
                            bc_at_release,
                        )
                    )
                if not time_options:
                    continue
                total, _, release, ba, bc = min(
                    time_options,
                    key=lambda item: (item[0], item[1]),
                )
                linked_vehicles.append(
                    (float(total), vehicle_key, ac, ba, bc, release)
                )
            linked_vehicles.sort(key=lambda item: (item[0], repr(item[1])))
            for vehicle_rank, (
                total,
                vehicle_key,
                ac,
                ba,
                bc,
                release,
            ) in enumerate(linked_vehicles, start=1):
                metadata = {
                    "costs": {
                        "BA": _cell_payload(ba),
                        "AC": _cell_payload(ac),
                        "BC": _cell_payload(bc),
                    },
                    "bc_role": "support_only",
                    "vehicle_rank_for_person": int(vehicle_rank),
                    **_release_payload(releases, release.frame_index),
                }
                candidate_route = RouteCandidate(
                    route_id="person:{}:vehicle:{}:{}".format(
                        person_key[1], vehicle_key[0], vehicle_key[1]
                    ),
                    cost=total,
                    person_key=person_key,
                    vehicle_key=vehicle_key,
                    route_type="person_vehicle",
                    metadata=metadata,
                )
                pre_prune_routes.append(candidate_route)
                if vehicle_rank <= self.config.top_k_vehicles:
                    person_routes.append(candidate_route)

        best_person_route_cost = {}
        for route in person_routes:
            best_person_route_cost[route.person_key] = min(
                float(route.cost),
                best_person_route_cost.get(route.person_key, float("inf")),
            )
        selected_people = {
            person_key
            for person_key, _ in sorted(
                best_person_route_cost.items(),
                key=lambda item: (float(item[1]), repr(item[0])),
            )[: self.config.top_k_people]
        }
        routes.extend(
            route for route in person_routes if route.person_key in selected_people
        )

        # NULL_A -> C is a direct vehicle release and therefore C_BC is a hard
        # gate, unlike the support-only B-C term on person routes.
        kept_direct_keys = {key for key, _cell in direct_vehicles}
        for direct_rank, (vehicle_key, bc) in enumerate(
            all_direct_vehicles, start=1
        ):
            direct_route = RouteCandidate(
                route_id="direct:{}:{}".format(vehicle_key[0], vehicle_key[1]),
                cost=float(bc.total + self.config.direct_vehicle_penalty),
                person_key=None,
                vehicle_key=vehicle_key,
                route_type="direct_vehicle",
                metadata={
                    "costs": {"BA": None, "AC": None, "BC": _cell_payload(bc)},
                    "direct_vehicle_rank": int(direct_rank),
                    **_release_payload(releases, bc.best_release_frame),
                },
            )
            pre_prune_routes.append(direct_route)
            if vehicle_key in kept_direct_keys:
                routes.append(direct_route)

        null_route = RouteCandidate(
            route_id="null",
            cost=float(self.config.dustbin_cost),
            person_key=None,
            vehicle_key=None,
            route_type="null",
            metadata={"costs": {"BA": None, "AC": None, "BC": None}},
        )
        pre_prune_routes.append(null_route)

        kept_route_ids = {str(route.route_id) for route in routes}
        kept_route_ids.add("null")
        pre_prune_sorted = sorted(
            pre_prune_routes,
            key=lambda route: (float(route.cost), str(route.route_id)),
        )
        selected_people_json = [
            list(key) for key in sorted(selected_people, key=repr)
        ]
        diagnostics = {
            "schema_version": 1,
            "cost_model": "smart_backtrack_components_v1",
            "component_semantics": "weighted_contribution_with_raw_feature",
            "resolver_config": {
                "max_back_frames": int(self.config.max_back_frames),
                "top_k_people": int(self.config.top_k_people),
                "top_k_vehicles": int(self.config.top_k_vehicles),
                "dustbin_cost": float(self.config.dustbin_cost),
                "null_vehicle_penalty": float(
                    self.config.null_vehicle_penalty
                ),
                "direct_vehicle_penalty": float(
                    self.config.direct_vehicle_penalty
                ),
                "ac_weight": float(self.config.ac_weight),
                "bc_support_bonus": float(self.config.bc_support_bonus),
                "sigma_floor_px": float(self.config.sigma_floor_px),
                "release_window_prior_weight": float(
                    self.config.release_window_prior_weight
                ),
                "release_window_semantics": "B0_B1_observation_gap_soft_prior",
                "max_back_semantics": "computational_guard_not_physical_gate",
                "use_kalman_rts": bool(self.config.use_kalman_rts),
                "confidence_aware_kalman": bool(
                    self.config.confidence_aware_kalman
                ),
                "use_uncertainty_gates": bool(
                    self.config.use_uncertainty_gates
                ),
                "kalman_process_noise_scale": float(
                    self.config.kalman_process_noise_scale
                ),
                "kalman_measurement_noise_scale": float(
                    self.config.kalman_measurement_noise_scale
                ),
                "kalman_max_extrapolation_seconds": (
                    float(self.config.kalman_max_extrapolation_seconds)
                    if self.config.kalman_max_extrapolation_seconds is not None
                    else None
                ),
                "kalman_effective_max_extrapolation_frames": int(
                    max_extrapolation_frames
                ),
                "kalman_time_semantics": (
                    "nearest_real_detection_frame"
                    if self.config.use_kalman_rts
                    else "detector_frame"
                ),
                "use_reverse_trajectory": bool(
                    self.config.use_reverse_trajectory
                ),
            },
            "release_hypotheses": [
                _release_hypothesis_payload(release) for release in releases
            ],
            "actor_candidates": [
                _actor_candidate_payload(actor_key, observations)
                for actor_key, observations in sorted(
                    actor_tracks.items(), key=lambda item: repr(item[0])
                )
            ],
            "pair_costs": {
                "BA": [
                    {
                        "person_key": list(person_key),
                        "cell": _cell_payload(cell),
                    }
                    for person_key, cell in sorted(
                        ba_costs.items(), key=lambda item: repr(item[0])
                    )
                ],
                "BA_by_release": [
                    {
                        "person_key": list(person_key),
                        "release_frame": int(release_frame),
                        "cell": _cell_payload(cell),
                    }
                    for (person_key, release_frame), cell in sorted(
                        ba_by_person_release.items(),
                        key=lambda item: (repr(item[0][0]), item[0][1]),
                    )
                ],
                "AC": [
                    {
                        "person_key": list(person_key),
                        "vehicle_key": list(vehicle_key),
                        "cell": _cell_payload(cell),
                    }
                    for (person_key, vehicle_key), cell in sorted(
                        ac_costs.items(),
                        key=lambda item: (
                            repr(item[0][0]),
                            repr(item[0][1]),
                        ),
                    )
                ],
                "BC": [
                    {
                        "vehicle_key": list(vehicle_key),
                        "cell": _cell_payload(cell),
                    }
                    for vehicle_key, cell in sorted(
                        bc_costs.items(), key=lambda item: repr(item[0])
                    )
                ],
                "BC_by_release": [
                    {
                        "vehicle_key": list(vehicle_key),
                        "release_frame": int(release_frame),
                        "cell": _cell_payload(cell),
                    }
                    for (vehicle_key, release_frame), cell in sorted(
                        bc_by_vehicle_release.items(),
                        key=lambda item: (repr(item[0][0]), item[0][1]),
                    )
                ],
            },
            "pruning": {
                "selected_people": selected_people_json,
                "pre_prune_route_count": len(pre_prune_routes),
                "post_prune_route_count": len(routes) + 1,
            },
            "pre_prune_routes": [
                {
                    **_route_payload(route),
                    "rank_before_pruning": int(rank),
                    "kept_after_pruning": str(route.route_id)
                    in kept_route_ids,
                    "pruned_by": (
                        None
                        if str(route.route_id) in kept_route_ids
                        else (
                            "top_k_people"
                            if (
                                route.person_key is not None
                                and route.person_key not in selected_people
                            )
                            else "top_k_vehicles"
                        )
                    ),
                }
                for rank, route in enumerate(pre_prune_sorted, start=1)
            ],
        }
        null_metadata = dict(null_route.metadata)
        null_metadata["candidate_diagnostics"] = diagnostics
        routes.append(
            RouteCandidate(
                route_id=null_route.route_id,
                cost=null_route.cost,
                person_key=null_route.person_key,
                vehicle_key=null_route.vehicle_key,
                route_type=null_route.route_type,
                metadata=null_metadata,
            )
        )
        return routes

    @staticmethod
    def _resolution(litter_id, assignment, routes):
        route = assignment.route
        metadata = dict(route.metadata or {})
        # Candidate diagnostics belong in the research sidecar, not in the
        # compact selected-event payload.  They remain attached to the NULL
        # route inside ``routes`` so final global re-solving cannot lose them.
        metadata.pop("candidate_diagnostics", None)
        release_point = metadata.get("release_point")
        release_covariance = metadata.get("release_covariance")
        return SmartResolution(
            litter_id=int(litter_id),
            person_key=route.person_key,
            vehicle_key=route.vehicle_key,
            direct_vehicle=bool(route.is_direct_vehicle),
            total_cost=float(assignment.total_cost),
            margin_to_second=assignment.margin_to_second,
            actor_margins=_actor_specific_margins(routes),
            route_type=str(route.route_type),
            route_id=str(route.route_id),
            release_frame=metadata.get("release_frame"),
            release_point=(
                tuple(map(float, release_point))
                if release_point is not None else None
            ),
            release_covariance=(
                tuple(tuple(map(float, row)) for row in release_covariance)
                if release_covariance is not None else None
            ),
            components=metadata,
            routes=list(routes),
        )

    def resolve_task(self, task) -> SmartResolution:
        litter_id = int(task.get("litter_id", -1))
        routes = self.build_routes(task)
        assignment = solve_event_routes(
            {litter_id: routes}, null_cost=self.config.dustbin_cost
        )[litter_id]
        return self._resolution(litter_id, assignment, routes)

    def solve_routes(self, routes_by_litter):
        assignments = solve_event_routes(
            routes_by_litter, null_cost=self.config.dustbin_cost
        )
        return {
            int(litter_id): self._resolution(
                litter_id, assignment, routes_by_litter[litter_id]
            )
            for litter_id, assignment in assignments.items()
        }


__all__ = [
    "SmartBacktrackConfig",
    "SmartBacktrackResolver",
    "SmartResolution",
]
