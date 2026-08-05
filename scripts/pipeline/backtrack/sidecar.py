# -*- coding: utf-8 -*-
"""JSONL sidecar records for smart-backtrack candidates and components.

The sidecar is intentionally independent from the model/runtime code.  It
captures the complete candidate table needed for later annotation, rejection
analysis and cost calibration without requiring RouteCandidate/dataclass or
NumPy-aware JSON encoders at the call site.
"""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional


SCHEMA_NAME = "smart-backtrack-candidates/v1"
COST_SCALE = 1000
_ACTOR_CLASSES = frozenset(("person", "vehicle", "scooter"))


def _json_safe(value: Any) -> Any:
    """Recursively convert runtime values to strict, portable JSON values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(float(value)) else None

    # NumPy scalars expose item(), while ndarray exposes tolist().  Keeping
    # these duck-typed avoids making NumPy a dependency of the serializer.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _json_safe(scalar)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError):
            pass

    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            str(field.name): _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            _json_key(key): _json_safe(item_value)
            for key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item_value) for item_value in value]
    if isinstance(value, (set, frozenset)):
        return [
            _json_safe(item_value)
            for item_value in sorted(value, key=lambda item_value: repr(item_value))
        ]
    return str(value)


def _json_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return "|".join(str(item) for item in value)
    return str(value)


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _scaled_cost(value: Any) -> Optional[int]:
    """Scale cost with the same half-away-from-zero rule as min-cost flow."""

    cost = _finite_float(value)
    if cost is None:
        return None
    scaled = cost * COST_SCALE
    if scaled >= 0.0:
        return int(math.floor(scaled + 0.5))
    return int(math.ceil(scaled - 0.5))


def _path_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _mapping(value: Any) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _route_value(route: Any, key: str, default: Any = None) -> Any:
    if isinstance(route, Mapping):
        return route.get(key, default)
    return getattr(route, key, default)


def _is_null_route(route: Any) -> bool:
    route_type = str(_route_value(route, "route_type", ""))
    person_key = _route_value(route, "person_key")
    vehicle_key = _route_value(route, "vehicle_key")
    return (
        route_type == "null"
        or (person_key is None and vehicle_key is None)
    )


def _merge_diagnostics(target: dict, incoming: Any) -> None:
    """Merge diagnostic payloads, preferring the first authoritative copy."""

    if not isinstance(incoming, Mapping):
        return
    for key, value in incoming.items():
        if key not in target:
            target[key] = value
            continue
        if isinstance(target[key], Mapping) and isinstance(value, Mapping):
            merged = dict(target[key])
            for nested_key, nested_value in value.items():
                merged.setdefault(nested_key, nested_value)
            target[key] = merged


def _serialize_routes(routes: Iterable[Any], selected_route_id: Any):
    prepared = []
    extracted_diagnostics = {}
    selected_text = (
        str(selected_route_id) if selected_route_id is not None else None
    )

    for input_order, route in enumerate(routes or []):
        route_id = str(_route_value(route, "route_id", ""))
        route_type = str(_route_value(route, "route_type", "person_vehicle"))
        raw_cost = _route_value(route, "cost")
        cost = _finite_float(raw_cost)
        metadata = dict(_mapping(_route_value(route, "metadata", {})))

        # The resolver places the event-wide diagnostic table on the explicit
        # NULL route.  Extract it once and do not repeat the (potentially large)
        # table inside the route's metadata.
        diagnostics = metadata.pop("candidate_diagnostics", None)
        if diagnostics is not None and _is_null_route(route):
            _merge_diagnostics(extracted_diagnostics, diagnostics)

        metadata_valid = metadata.get("valid")
        valid = (
            bool(metadata_valid)
            if metadata_valid is not None
            else cost is not None
        )
        reject_reason = metadata.get("reject_reason")
        if not valid and reject_reason is None and cost is None:
            reject_reason = "non_finite_route_cost"

        prepared.append({
            "_input_order": int(input_order),
            "route_id": route_id,
            "route_type": route_type,
            "person_key": _json_safe(_route_value(route, "person_key")),
            "vehicle_key": _json_safe(_route_value(route, "vehicle_key")),
            "valid": bool(valid),
            "reject_reason": (
                str(reject_reason) if reject_reason is not None else None
            ),
            "cost": cost,
            "scaled_cost": _scaled_cost(raw_cost),
            "selected": selected_text is not None and route_id == selected_text,
            "metadata": _json_safe(metadata),
        })

    # Emit routes in the order a reviewer expects to inspect them: finite costs
    # first, then rejected/non-finite routes, with deterministic tie breaking.
    prepared.sort(
        key=lambda item: (
            item["cost"] is None,
            item["cost"] if item["cost"] is not None else float("inf"),
            str(item["route_id"]),
            item["_input_order"],
        )
    )
    finite_rank = 0
    for route in prepared:
        if route["cost"] is not None:
            finite_rank += 1
            route["rank"] = finite_rank
        else:
            route["rank"] = None
        route.pop("_input_order", None)
    return prepared, extracted_diagnostics


def _strip_event_diagnostics(event: Any):
    """Return an event copy without a duplicated selected-NULL diagnostic blob."""

    copied = _json_safe(event)
    if not isinstance(copied, dict):
        return copied, {}
    extracted = {}
    backtrack = copied.get("backtrack")
    if isinstance(backtrack, dict):
        components = backtrack.get("components")
        if isinstance(components, dict):
            diagnostics = components.pop("candidate_diagnostics", None)
            _merge_diagnostics(extracted, diagnostics)
    return copied, extracted


def _litter_history(task: Mapping) -> List[dict]:
    points = list(task.get("history") or [])
    frames = list(task.get("history_frames") or [])
    boxes = list(task.get("history_boxes") or [])
    confidences = list(task.get("history_confidences") or [])
    sample_count = max(
        len(points), len(frames), len(boxes), len(confidences), 0
    )
    samples = []
    for index in range(sample_count):
        samples.append({
            "frame_index": (
                _json_safe(frames[index]) if index < len(frames) else None
            ),
            "point_uv": (
                _json_safe(points[index]) if index < len(points) else None
            ),
            "bbox_xyxy": (
                _json_safe(boxes[index]) if index < len(boxes) else None
            ),
            "confidence": (
                _json_safe(confidences[index])
                if index < len(confidences)
                else None
            ),
        })
    return samples


def _candidate_actor_tracklets(task: Mapping) -> List[dict]:
    grouped = {}
    for frame_snapshot in task.get("actor_frames", []) or []:
        try:
            frame_index = int(frame_snapshot.get("frame_index", 0))
        except (AttributeError, TypeError, ValueError):
            continue
        for actor in _mapping(frame_snapshot).get("actors", []) or []:
            if not isinstance(actor, Mapping):
                continue
            class_name = str(
                actor.get("cls", actor.get("cls_name", ""))
            ).lower()
            if class_name not in _ACTOR_CLASSES:
                continue
            try:
                track_id = int(actor["track_id"])
            except (KeyError, TypeError, ValueError):
                continue
            tracklet_uid = str(
                actor.get(
                    "tracklet_uid",
                    "{}:{}".format(class_name, track_id),
                )
            )
            group_key = (class_name, track_id, tracklet_uid)
            observation = dict(actor)
            observation["frame_index"] = frame_index
            grouped.setdefault(group_key, []).append(_json_safe(observation))

    tracklets = []
    for (class_name, track_id, tracklet_uid), observations in grouped.items():
        observations.sort(
            key=lambda item: (
                item.get("frame_index") is None,
                item.get("frame_index")
                if item.get("frame_index") is not None else 0,
            )
        )
        frame_values = [
            int(item["frame_index"])
            for item in observations
            if item.get("frame_index") is not None
        ]
        tracklets.append({
            "actor_key": [class_name, track_id],
            "class_name": class_name,
            "track_id": track_id,
            "tracklet_uid": tracklet_uid,
            "frame_range": (
                [min(frame_values), max(frame_values)]
                if frame_values else [None, None]
            ),
            "observations": observations,
        })
    tracklets.sort(
        key=lambda item: (
            item["class_name"],
            item["track_id"],
            item["tracklet_uid"],
        )
    )
    return tracklets


def _build_assignment(event: Mapping, serialized_routes: List[dict]) -> dict:
    backtrack = _mapping(event.get("backtrack"))
    route_id = backtrack.get("route_id")
    selected = next(
        (route for route in serialized_routes if route["selected"]),
        None,
    )
    cost = (
        selected.get("cost")
        if selected is not None
        else _finite_float(backtrack.get("score"))
    )
    scaled_cost = (
        selected.get("scaled_cost")
        if selected is not None
        else _scaled_cost(backtrack.get("score"))
    )
    return {
        "status": _json_safe(
            event.get("backtrack_status", backtrack.get("status"))
        ),
        "route_id": (
            str(route_id) if route_id is not None else None
        ),
        "route_type": _json_safe(backtrack.get("route_type")),
        "person_key": _json_safe(
            backtrack.get("person_key", event.get("thrower_key"))
        ),
        "vehicle_key": _json_safe(
            backtrack.get("vehicle_key", event.get("vehicle_key"))
        ),
        "cost": cost,
        "scaled_cost": scaled_cost,
        "margin_to_second": _json_safe(
            backtrack.get("margin_to_second")
        ),
        "release_frame": _json_safe(backtrack.get("release_frame")),
        "release_point": _json_safe(backtrack.get("release_point")),
    }


def build_run_record(
    input_video,
    output_video,
    fps,
    frame_count,
    smart_summary=None,
    extra=None,
):
    """Build one run-level header record for a candidate sidecar."""

    try:
        normalized_frame_count = (
            int(frame_count) if frame_count is not None else None
        )
    except (TypeError, ValueError):
        normalized_frame_count = None
    return {
        "schema": SCHEMA_NAME,
        "record_type": "run",
        "video": {
            "input_video": _path_value(input_video),
            "output_video": _path_value(output_video),
            "fps": _finite_float(fps),
            "frame_count": normalized_frame_count,
        },
        "smart_summary": _json_safe(smart_summary or {}),
        "extra": _json_safe(extra or {}),
    }


def build_candidate_record(
    task,
    event,
    routes,
    input_video,
    output_video=None,
):
    """Build one JSON-safe record for a confirmed litter event."""

    task = _mapping(task)
    event_mapping = _mapping(event)
    backtrack = _mapping(event_mapping.get("backtrack"))
    selected_route_id = backtrack.get("route_id")
    serialized_routes, route_diagnostics = _serialize_routes(
        routes or [], selected_route_id
    )
    serialized_event, event_diagnostics = _strip_event_diagnostics(
        event_mapping
    )
    _merge_diagnostics(route_diagnostics, event_diagnostics)

    release_hypotheses = route_diagnostics.pop(
        "release_hypotheses",
        route_diagnostics.pop("releases", []),
    )
    pair_costs = route_diagnostics.pop("pair_costs", {})

    return {
        "schema": SCHEMA_NAME,
        "record_type": "candidate",
        "video": {
            "input_video": _path_value(input_video),
            "output_video": _path_value(output_video),
            "fps": _finite_float(task.get("fps")),
        },
        "event": serialized_event,
        "assignment": _build_assignment(event_mapping, serialized_routes),
        "litter_history": _litter_history(task),
        "release_hypotheses": _json_safe(release_hypotheses or []),
        "pair_costs": _json_safe(pair_costs or {}),
        "candidate_actors": _candidate_actor_tracklets(task),
        "routes": serialized_routes,
        "candidate_diagnostics": _json_safe(route_diagnostics),
        # A research trial must be able to rebuild the *same* resolver input
        # without loading a model or relying on a mutable tracker cache.  This
        # is deliberately separate from the human-facing annotation schema;
        # annotation tools never expose it as a model decision.
        "resolver_input": _json_safe(task),
    }


def write_jsonl(records, path):
    """Write strict UTF-8 JSONL and return the number of records written."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(records, Mapping):
        records = [records]
    count = 0
    with output_path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(
                json.dumps(
                    _json_safe(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
            file_handle.write("\n")
            count += 1
    return count


def read_jsonl(path):
    """Read a JSONL sidecar, ignoring blank lines."""

    records = []
    with Path(path).open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid JSONL at {}:{}: {}".format(
                        path, line_number, exc.msg
                    )
                ) from exc
    return records


__all__ = [
    "SCHEMA_NAME",
    "build_candidate_record",
    "build_run_record",
    "read_jsonl",
    "write_jsonl",
]
