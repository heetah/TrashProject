"""Auditable guarded route re-selection for production Smart Backtrack.

The policy promotes the frozen 2026-09-15 research composition.  It never
creates a route, relaxes a hard gate, confirms an event, or removes NULL.  A
route may only replace the min-cost selection when the relevant evidence is
present and every documented guard passes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import statistics
from typing import Mapping, Sequence

import cv2
import numpy as np

from .flow import COST_SCALE, Assignment, RouteCandidate


@dataclass(frozen=True)
class GuardedReselectionConfig:
    enabled: bool = True
    mask_temporal_enabled: bool = True
    mask_pre_seconds: float = 0.20
    mask_post_seconds: float = 0.10
    mask_pre_min_observations: int = 1
    mask_release_min_observations: int = 1
    mask_pre_max_observations: int = 2
    mask_release_max_observations: int = 2
    boundary_selected_min: float = 0.99
    boundary_alternative_max: float = 0.20
    boundary_cost_delta_max: float = 0.50
    ac_endpoint_improvement_min: float = 0.02
    ac_cost_delta_max: float = 0.15
    bc_quality_selected_min: float = 0.80
    bc_quality_alternative_max: float = 0.05
    bc_quality_cost_delta_max: float = 0.50
    direct_person_endpoint_max: float = 0.25
    direct_person_overlap_max: float = 0.10
    direct_person_cost_delta_max: float = 1.00
    motion_selected_reverse_min: float = 0.10
    motion_selected_exit_min: float = 0.20
    motion_alternative_reverse_max: float = 0.02
    motion_alternative_exit_max: float = 0.05
    motion_alternative_relative_max: float = 0.02
    motion_cost_delta_max: float = 2.00

    def __post_init__(self):
        if not (
            math.isfinite(float(self.mask_pre_seconds))
            and math.isfinite(float(self.mask_post_seconds))
            and self.mask_pre_seconds >= 0.0
            and self.mask_post_seconds >= 0.0
            and 0
            < self.mask_pre_min_observations
            <= self.mask_pre_max_observations
            and 0
            < self.mask_release_min_observations
            <= self.mask_release_max_observations
        ):
            raise ValueError("invalid mask time/observation window")


def _actor_key(value):
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        return None
    try:
        return str(value[0]).lower(), int(value[1])
    except (TypeError, ValueError):
        return None


def _raw(route: RouteCandidate, edge: str, name: str):
    try:
        cell = (route.metadata.get("costs") or {}).get(edge) or {}
        value = (cell.get("raw_features") or {}).get(name)
        result = float(value)
    except (AttributeError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _release_model(route: RouteCandidate) -> str:
    return str((route.metadata or {}).get("release_model") or "")


def _valid_routes(routes: Sequence[RouteCandidate]):
    return [route for route in routes if math.isfinite(float(route.cost))]


def _finite_polygon(value):
    try:
        polygon = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if (
        polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2
        or not np.isfinite(polygon).all()
        or abs(float(cv2.contourArea(polygon))) <= 0.0
    ):
        return None
    return polygon


def compact_mask_polygon(value, max_vertices: int = 512):
    """Return a bounded deterministic contour for resolver-only geometry."""
    polygon = _finite_polygon(value)
    if polygon is None:
        return None
    limit = max(int(max_vertices), 3)
    if polygon.shape[0] > limit:
        indices = np.linspace(0, polygon.shape[0] - 1, limit, dtype=int)
        polygon = polygon[indices]
    if _finite_polygon(polygon) is None:
        return None
    return tuple((float(point[0]), float(point[1])) for point in polygon)


def _signed_mask_value(point, snapshot):
    polygon = _finite_polygon(snapshot.get("mask_contour_xy"))
    box = snapshot.get("box")
    if polygon is None or not isinstance(box, (tuple, list)) or len(box) < 4:
        return None
    try:
        width = max(float(box[2]) - float(box[0]), 0.0)
        height = max(float(box[3]) - float(box[1]), 0.0)
        diagonal = math.hypot(width, height)
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return None
    if diagonal <= 0.0 or not all(map(math.isfinite, (x, y, diagonal))):
        return None
    distance = float(
        cv2.pointPolygonTest(polygon.reshape((-1, 1, 2)), (x, y), True)
    )
    return math.tanh(distance / diagonal)


def _observed_mask_index(task: Mapping):
    result = {}
    for row in task.get("actor_frames", []) or []:
        try:
            frame = int(row.get("frame_index"))
        except (AttributeError, TypeError, ValueError):
            continue
        for actor in row.get("actors", []) or []:
            key = _actor_key(actor.get("actor_key"))
            if key is None:
                try:
                    key = (str(actor.get("cls", "")).lower(), int(actor["track_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
            if (
                key[0] not in {"vehicle", "scooter"}
                or not bool(actor.get("observed", True))
                or str(actor.get("source", "")) not in {"seg_track", "seg_predict"}
                or _finite_polygon(actor.get("mask_contour_xy")) is None
            ):
                continue
            result[(frame, key)] = actor
    return result


def _point_for_frame(route: RouteCandidate, releases, frame: int, fps: float):
    model = _release_model(route)
    for release in releases:
        if release.model == model and int(release.frame_index) == int(frame):
            return tuple(map(float, release.mean_uv))
    metadata = route.metadata or {}
    origin = metadata.get("release_point")
    velocity = metadata.get("release_velocity")
    origin_frame = metadata.get("release_frame")
    if not (
        isinstance(origin, (tuple, list)) and len(origin) == 2
        and isinstance(velocity, (tuple, list)) and len(velocity) == 2
        and origin_frame is not None and fps > 0.0
    ):
        return None
    dt = (int(frame) - int(origin_frame)) / fps
    return (
        float(origin[0]) + float(velocity[0]) * dt,
        float(origin[1]) + float(velocity[1]) * dt,
    )


def _mask_window_samples(
    index,
    key,
    route,
    releases,
    release_frame,
    fps,
    *,
    pre,
    seconds,
    maximum,
):
    """Return nearest valid fresh masks in a seconds-defined CFR window."""
    frame_span = max(0, int(math.floor(float(seconds) * float(fps))))
    samples = []
    for (frame, actor_key), snapshot in index.items():
        if actor_key != key:
            continue
        offset = int(frame) - int(release_frame)
        in_window = (
            -frame_span <= offset < 0
            if pre else 0 <= offset <= frame_span
        )
        if not in_window:
            continue
        point = _point_for_frame(route, releases, frame, fps)
        value = _signed_mask_value(point, snapshot) if point is not None else None
        if value is None or not math.isfinite(value):
            continue
        samples.append({
            "frame_offset": int(offset),
            "time_offset_seconds": float(offset / fps),
            "signed_distance": float(value),
        })
    samples.sort(key=lambda row: (abs(row["frame_offset"]), row["frame_offset"]))
    return samples[:maximum], len(samples), frame_span


def attach_mask_temporal_evidence(
    routes, task: Mapping, releases, config: GuardedReselectionConfig | None = None
):
    """Attach seconds-based scalar mask evidence; missing data stays missing."""
    cfg = config or GuardedReselectionConfig()
    index = _observed_mask_index(task)
    fps = float(task.get("fps", 0.0) or 0.0)
    history = list(task.get("history") or [])
    history_frames = list(task.get("history_frames") or [])
    output = []
    for route in routes:
        if route.route_type != "direct_vehicle" or route.vehicle_key is None:
            output.append(route)
            continue
        key = _actor_key(route.vehicle_key)
        release_frame = (route.metadata or {}).get("release_frame")
        if (
            key is None
            or release_frame is None
            or not math.isfinite(fps)
            or fps <= 0.0
        ):
            output.append(route)
            continue
        pre_samples, pre_available, pre_frame_span = _mask_window_samples(
            index, key, route, releases, int(release_frame), fps,
            pre=True,
            seconds=cfg.mask_pre_seconds,
            maximum=cfg.mask_pre_max_observations,
        )
        (
            release_samples,
            release_available,
            release_frame_span,
        ) = _mask_window_samples(
            index, key, route, releases, int(release_frame), fps,
            pre=False,
            seconds=cfg.mask_post_seconds,
            maximum=cfg.mask_release_max_observations,
        )
        offsets = {
            str(sample["frame_offset"]): sample["signed_distance"]
            for sample in (*pre_samples, *release_samples)
        }
        history_values = []
        for point, frame in zip(history, history_frames):
            snapshot = index.get((int(frame), key))
            value = _signed_mask_value(point, snapshot) if snapshot else None
            if value is not None and math.isfinite(value):
                history_values.append(float(value))
        metadata = dict(route.metadata or {})
        metadata["mask_temporal_evidence"] = {
            "schema": "mask-temporal-route-evidence/v2",
            "time_basis": "frame_offset_over_constant_fps",
            "pre_seconds": float(cfg.mask_pre_seconds),
            "post_seconds": float(cfg.mask_post_seconds),
            "pre_frame_span": int(pre_frame_span),
            "release_frame_span": int(release_frame_span),
            "pre_samples": pre_samples,
            "release_samples": release_samples,
            "pre_signed": [sample["signed_distance"] for sample in pre_samples],
            "release_signed": [
                sample["signed_distance"] for sample in release_samples
            ],
            "pre_observed_count": len(pre_samples),
            "release_observed_count": len(release_samples),
            "pre_available_count": int(pre_available),
            "release_available_count": int(release_available),
            "pre_min_observations": int(cfg.mask_pre_min_observations),
            "release_min_observations": int(cfg.mask_release_min_observations),
            "release_signed_by_offset": offsets,
            "history_signed": history_values,
            "history_observed_count": len(history_values),
            "history_expected_count": min(len(history), len(history_frames)),
        }
        output.append(replace(route, metadata=metadata))
    return output


def _mask(route):
    value = (route.metadata or {}).get("mask_temporal_evidence")
    return value if isinstance(value, Mapping) else {}


def _offset(route, offset):
    try:
        value = float((_mask(route).get("release_signed_by_offset") or {})[str(offset)])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _finite_values(route, name):
    values = _mask(route).get(name)
    if not isinstance(values, (tuple, list)):
        return []
    output = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(value):
            return []
        output.append(value)
    return output


def _required_count(route, name, fallback=1):
    try:
        value = int(_mask(route).get(name, fallback))
    except (TypeError, ValueError):
        return fallback
    return max(value, 1)


def _aggregate_release(route):
    values = _finite_values(route, "release_signed")
    if values:
        if len(values) < _required_count(route, "release_min_observations"):
            return None
    else:
        values = [_offset(route, offset) for offset in (0, 1)]
        values = [value for value in values if value is not None]
    return float(statistics.median(values)) if values else None


def _aggregate_pre(route):
    values = _finite_values(route, "pre_signed")
    if values:
        if len(values) < _required_count(route, "pre_min_observations"):
            return None
        return float(statistics.median(values))
    return _offset(route, -1)


def _pre_values(route):
    values = _finite_values(route, "pre_signed")
    if values:
        return (
            values
            if len(values) >= _required_count(route, "pre_min_observations")
            else []
        )
    return [
        value for value in (_offset(route, -2), _offset(route, -1))
        if value is not None
    ]


def _q25(values):
    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            finite.append(value)
    values = sorted(finite)
    return values[int(round(0.25 * (len(values) - 1)))] if values else None


def _history(route):
    values = _mask(route).get("history_signed") or []
    if not isinstance(values, (tuple, list)):
        return []
    output = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(value):
            return []
        output.append(value)
    return output


def _mask_rule(selected, routes):
    if selected.route_type != "direct_vehicle":
        return None
    selected_value = _aggregate_release(selected)
    selected_history = _history(selected)
    if selected_value is None:
        return None
    model = _release_model(selected)
    candidates = []
    for route in routes:
        if (
            route.route_type != "direct_vehicle" or route.route_id == selected.route_id
            or _release_model(route) != model
        ):
            continue
        value = _aggregate_release(route)
        if value is None:
            continue
        if "ballistic" in model:
            history = _history(route)
            release = _aggregate_release(route)
            before = _aggregate_pre(route)
            pre_values = _pre_values(route)
            route_distance = _raw(route, "BC", "direct_distance")
            route_q25 = _q25(history)
            selected_q25 = _q25(selected_history)
            if (
                not selected_history or not history or release is None or before is None
                or value <= selected_value or release <= 0.0
                or release - before <= 0.0
                or route_q25 is None or selected_q25 is None
                or route_q25 >= selected_q25
                or not pre_values
                or not (
                    all(item < 0.0 for item in history)
                    or all(item < 0.0 for item in pre_values)
                )
                or route_distance is None or route_distance > 0.0
            ):
                continue
            candidates.append(route)
        elif "constant_velocity" in model:
            selected_release = _aggregate_release(selected)
            release = _aggregate_release(route)
            selected_mahal = _raw(selected, "BC", "spatial_mahalanobis")
            route_mahal = _raw(route, "BC", "spatial_mahalanobis")
            selected_uncertainty = _raw(selected, "BC", "uncertainty")
            route_uncertainty = _raw(route, "BC", "uncertainty")
            if (
                selected_release is None or release is None
                or selected_release <= 0.0 or release >= 0.0
                or value >= selected_value
                or None in (selected_mahal, route_mahal, selected_uncertainty, route_uncertainty)
                or route_mahal <= selected_mahal
                or route_uncertainty > selected_uncertainty
            ):
                continue
            candidates.append(route)
    if not candidates:
        return None
    if "ballistic" in model:
        return min(candidates, key=lambda route: (
            float(route.cost), -float(_aggregate_release(route)), str(route.route_id)
        ))
    return min(candidates, key=lambda route: (
        float(_aggregate_release(route)), float(route.cost), str(route.route_id)
    ))


def _boundary_rule(selected, routes, cfg):
    depth = _raw(selected, "BC", "boundary_depth")
    if (
        selected.route_type != "direct_vehicle" or _release_model(selected) != "ballistic"
        or depth is None or depth < cfg.boundary_selected_min
    ):
        return None
    return min((route for route in routes if
        route.route_type == "direct_vehicle" and route.route_id != selected.route_id
        and _release_model(route) == "ballistic"
        and (_raw(route, "BC", "boundary_depth") is not None)
        and _raw(route, "BC", "boundary_depth") <= cfg.boundary_alternative_max
        and float(route.cost) - float(selected.cost) <= cfg.boundary_cost_delta_max),
        key=lambda route: (float(route.cost) - float(selected.cost), str(route.route_id)),
        default=None)


def _same_vehicle_person_candidates(selected, routes):
    return [route for route in routes if
        route.route_type == "person_vehicle"
        and _actor_key(route.vehicle_key) == _actor_key(selected.vehicle_key)
        and _actor_key(route.person_key) != _actor_key(selected.person_key)]


def _ac_endpoint_rule(selected, routes, cfg):
    value = _raw(selected, "AC", "endpoint_proximity")
    if selected.route_type != "person_vehicle" or value is None:
        return None
    return min((route for route in _same_vehicle_person_candidates(selected, routes)
        if _raw(route, "AC", "endpoint_proximity") is not None
        and value - _raw(route, "AC", "endpoint_proximity") >= cfg.ac_endpoint_improvement_min
        and float(route.cost) - float(selected.cost) <= cfg.ac_cost_delta_max),
        key=lambda route: (float(route.cost), str(route.route_id)), default=None)


def _bc_quality_rule(selected, routes, cfg):
    value = _raw(selected, "BC", "quality")
    if selected.route_type != "person_vehicle" or value is None or value < cfg.bc_quality_selected_min:
        return None
    return min((route for route in _same_vehicle_person_candidates(selected, routes)
        if _raw(route, "BC", "quality") is not None
        and _raw(route, "BC", "quality") <= cfg.bc_quality_alternative_max
        and float(route.cost) - float(selected.cost) <= cfg.bc_quality_cost_delta_max),
        key=lambda route: (float(route.cost), str(route.route_id)), default=None)


def _direct_person_rule(selected, routes, cfg):
    endpoint = _raw(selected, "AC", "endpoint_proximity")
    overlap = _raw(selected, "AC", "overlap")
    if (
        selected.route_type != "person_vehicle" or endpoint is None or overlap is None
        or endpoint > cfg.direct_person_endpoint_max or overlap > cfg.direct_person_overlap_max
    ):
        return None
    return min((route for route in routes if route.route_type == "direct_vehicle"
        and _actor_key(route.vehicle_key) == _actor_key(selected.vehicle_key)
        and float(route.cost) - float(selected.cost) <= cfg.direct_person_cost_delta_max),
        key=lambda route: (float(route.cost), str(route.route_id)), default=None)


def _motion_rule(selected, routes, cfg):
    reverse = _raw(selected, "BC", "reverse_direction")
    exit_deficit = _raw(selected, "BC", "exit_deficit")
    relative_deficit = _raw(selected, "BC", "relative_motion_deficit")
    if (
        selected.route_type != "direct_vehicle"
        or reverse is None or exit_deficit is None or relative_deficit is None
        or reverse < cfg.motion_selected_reverse_min
        or exit_deficit < cfg.motion_selected_exit_min
    ):
        return None
    candidates = []
    for route in routes:
        if (
            route.route_type != "direct_vehicle"
            or route.route_id == selected.route_id
            or _release_model(route) != _release_model(selected)
        ):
            continue
        values = (
            _raw(route, "BC", "reverse_direction"),
            _raw(route, "BC", "exit_deficit"),
            _raw(route, "BC", "relative_motion_deficit"),
        )
        if None in values:
            continue
        if (
            values[0] <= cfg.motion_alternative_reverse_max
            and values[1] <= cfg.motion_alternative_exit_max
            and values[2] <= cfg.motion_alternative_relative_max
            and float(route.cost) - float(selected.cost) <= cfg.motion_cost_delta_max
        ):
            candidates.append(route)
    return min(candidates, key=lambda route: (float(route.cost), str(route.route_id)), default=None)


def apply_guarded_reselection(litter_id, assignment, routes, config=None):
    cfg = config or GuardedReselectionConfig()
    routes = list(routes)
    if not cfg.enabled or assignment.route.is_null:
        return assignment, routes
    selected = assignment.route
    steps = []
    rules = []
    if cfg.mask_temporal_enabled:
        rules.append(("mask_temporal", lambda route: _mask_rule(route, _valid_routes(routes))))
    rules.extend((
        ("ballistic_boundary", lambda route: _boundary_rule(route, _valid_routes(routes), cfg)),
        ("ac_endpoint", lambda route: _ac_endpoint_rule(route, _valid_routes(routes), cfg)),
        ("bc_quality", lambda route: _bc_quality_rule(route, _valid_routes(routes), cfg)),
        ("direct_person", lambda route: _direct_person_rule(route, _valid_routes(routes), cfg)),
        ("motion_consistency", lambda route: _motion_rule(route, _valid_routes(routes), cfg)),
    ))
    for name, rule in rules:
        alternative = rule(selected)
        if alternative is not None and alternative.route_id != selected.route_id:
            steps.append({"rule": name, "from": selected.route_id, "to": alternative.route_id})
            selected = alternative
    if not steps:
        return assignment, routes
    metadata = dict(selected.metadata or {})
    metadata["guarded_reselection"] = {
        "schema": "guarded-route-reselection/v1",
        "initial_route_id": assignment.route.route_id,
        "final_route_id": selected.route_id,
        "steps": steps,
    }
    selected = replace(selected, metadata=metadata)
    routes = [selected if route.route_id == selected.route_id else route for route in routes]
    selected_scaled = int(math.floor(float(selected.cost) * COST_SCALE + 0.5))
    other = [
        int(math.floor(float(route.cost) * COST_SCALE + 0.5))
        for route in routes if route.route_id != selected.route_id and math.isfinite(float(route.cost))
    ]
    margin = ((min(other) - selected_scaled) / COST_SCALE) if other else None
    return Assignment(
        litter_id=litter_id,
        route=selected,
        total_cost=float(selected.cost),
        scaled_cost=selected_scaled,
        margin_to_second=margin,
    ), routes


__all__ = [
    "GuardedReselectionConfig",
    "apply_guarded_reselection",
    "attach_mask_temporal_evidence",
    "compact_mask_polygon",
]
