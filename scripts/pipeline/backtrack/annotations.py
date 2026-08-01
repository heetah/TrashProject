# -*- coding: utf-8 -*-
"""Annotation templates, validation, and metrics for smart backtracking.

The runtime sidecar and the human annotation file deliberately use different
schemas.  In particular, :func:`init_annotation_records` never copies a
selected route, actor identity, or release frame from a runtime prediction into
ground truth.

Candidate sidecar schema
========================

``smart-backtrack-candidates/v1`` contains ``run`` and ``candidate`` JSONL
records.  A ``candidate`` row is the event-level ranked candidate table.
The canonical runtime event shape is::

    {
      "schema": "smart-backtrack-candidates/v1",
      "record_type": "candidate",
      "run_id": "...",
      "video": {"clip_id": "...", "input_path": "...", "fps": 10},
      "event": {"event_id": "...", "litter_id": 3},
      "routes": [{
        "rank": 1,
        "route_id": "person:2:vehicle:vehicle:8",
        "route_type": "person_vehicle",
        "person_key": ["person", 2],
        "vehicle_key": ["vehicle", 8],
        "cost": 0.72,
        "metadata": {"release_frame": 41, "costs": {...}}
      }]
    }

Annotation schema
=================

``smart-backtrack-annotations/v1`` is independent ground truth.  An event is
evaluated only after the relevant fields are human-reviewed.  Multiple
``admissible_routes`` express ambiguity without forcing one arbitrary answer::

    {
      "schema": "smart-backtrack-annotations/v1",
      "record_type": "event",
      "run_id": "...",
      "clip_id": "...",
      "event_id": "...",
      "ignore": false,
      "review": {
        "event": "reviewed", "person": "reviewed",
        "vehicle": "reviewed", "release": "reviewed"
      },
      "event_label": "litter",
      "admissible_routes": [{
        "route_type": "person_vehicle",
        "person_id": "person:2",
        "vehicle_id": "vehicle:8"
      }],
      "release_interval": {"start_frame": 38, "end_frame": 43}
    }

The same vehicle may legally appear in routes for any number of events and
people.  Validation intentionally imposes no one-to-one person/vehicle rule.
"""

from __future__ import division

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CANDIDATE_SCHEMA = "smart-backtrack-candidates/v1"
ANNOTATION_SCHEMA = "smart-backtrack-annotations/v1"
RECORD_TYPES = {"run", "event"}
ROUTE_TYPES = {"person_vehicle", "person", "direct_vehicle", "null"}
REVIEW_STATES = {"unreviewed", "reviewed", "not_applicable"}
EVENT_LABELS = {"litter", "not_litter"}
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg"
}


class AnnotationError(ValueError):
    """Raised when annotation/candidate data cannot be parsed or evaluated."""


def _record_type(record: Mapping[str, Any]) -> Optional[str]:
    value = record.get("record_type", record.get("type"))
    normalized = str(value).lower() if value is not None else None
    # Runtime calls each event row a "candidate" record.  In the annotation
    # domain it is still an event containing a ranked candidate table.
    return "event" if normalized == "candidate" else normalized


def _nested(record: Mapping[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _stable_path_id(path: str) -> str:
    absolute = os.path.abspath(os.path.expanduser(str(path)))
    digest = hashlib.sha1(absolute.encode("utf-8")).hexdigest()[:12]
    return "run-{}".format(digest)


def _video_path(record: Mapping[str, Any]) -> Optional[str]:
    video = record.get("video")
    if isinstance(video, Mapping):
        value = _first_present(
            video.get("input_path"),
            video.get("input_video"),
            video.get("video_path"),
            video.get("path"),
        )
    else:
        value = None
    value = _first_present(
        value,
        record.get("video_path"),
        record.get("input_path"),
    )
    return str(value) if value is not None else None


def _clip_id(record: Mapping[str, Any]) -> Optional[str]:
    video_path = _video_path(record)
    value = _first_present(
        record.get("clip_id"),
        _nested(record, "video", "clip_id"),
        _nested(record, "event", "clip_id"),
    )
    if value is not None:
        return str(value)
    if video_path:
        return Path(video_path).stem
    return None


def _run_id(record: Mapping[str, Any]) -> Optional[str]:
    value = _first_present(
        record.get("run_id"),
        _nested(record, "run", "run_id"),
        _nested(record, "run", "id"),
        _nested(record, "event", "run_id"),
        _nested(record, "video", "run_id"),
    )
    if value is not None:
        return str(value)
    video_path = _video_path(record)
    return _stable_path_id(video_path) if video_path else None


def _event_id(record: Mapping[str, Any]) -> Optional[str]:
    value = _first_present(
        record.get("event_id"),
        _nested(record, "event", "event_id"),
        _nested(record, "event", "id"),
        record.get("litter_id"),
        _nested(record, "event", "litter_id"),
        _nested(record, "assignment", "litter_id"),
    )
    return str(value) if value is not None else None


def _fps(record: Mapping[str, Any]) -> Optional[float]:
    value = _first_present(record.get("fps"), _nested(record, "video", "fps"))
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if math.isfinite(fps) and fps > 0.0 else None


def infer_clip_weak_label(video_path: Optional[str]) -> str:
    """Infer only a clip-level weak label from explicit folder names.

    A directory component named exactly ``litter`` is accepted as weak
    supervision.  Actor identities, event validity, routes, and release time
    are never inferred here.
    """

    if not video_path:
        return "unreviewed"
    parts = {part.lower() for part in Path(str(video_path)).parts}
    return "litter" if "litter" in parts else "unreviewed"


def _explicit_clip_weak_label(
    record: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Read an explicit *clip-level weak* label without promoting it to GT."""

    payload = _nested(record, "extra", "clip_label")
    if not isinstance(payload, Mapping):
        return None, None
    value = payload.get("value")
    if value not in {"litter", "non_litter"}:
        return None, None
    metadata = {
        "value": value,
        "strength": str(payload.get("strength", "weak")),
        "source": str(payload.get("source", "explicit")),
    }
    return str(value), metadata


def discover_record_files(path: os.PathLike) -> List[Path]:
    """Return deterministic JSON/JSONL files from a file or directory."""

    target = Path(path)
    if target.is_file():
        if target.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
            raise AnnotationError("Not a JSON/JSONL input: {}".format(target))
        return [target]
    if not target.is_dir():
        raise AnnotationError("Input does not exist: {}".format(target))
    files = [
        item for item in target.rglob("*")
        if item.is_file() and item.suffix.lower() in {".json", ".jsonl", ".ndjson"}
    ]
    return sorted(files, key=lambda item: str(item))


def load_records(path: os.PathLike) -> List[Dict[str, Any]]:
    """Load JSONL records (or JSON lists) from a file or directory."""

    records: List[Dict[str, Any]] = []
    for source in discover_record_files(path):
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise AnnotationError("Cannot read {}: {}".format(source, exc))
        if not text.strip():
            continue
        if source.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AnnotationError("{}: {}".format(source, exc))
            values = payload if isinstance(payload, list) else [payload]
            for index, value in enumerate(values, 1):
                if not isinstance(value, Mapping):
                    raise AnnotationError(
                        "{} item {} is not an object".format(source, index)
                    )
                copied = dict(value)
                copied.setdefault("_source_file", str(source))
                records.append(copied)
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnnotationError(
                    "{}:{}: {}".format(source, line_number, exc)
                )
            if not isinstance(value, Mapping):
                raise AnnotationError(
                    "{}:{} is not an object".format(source, line_number)
                )
            copied = dict(value)
            copied.setdefault("_source_file", str(source))
            copied.setdefault("_source_line", line_number)
            records.append(copied)
    return records


def write_records(
    records: Iterable[Mapping[str, Any]],
    output_path: os.PathLike,
    overwrite: bool = False,
) -> Path:
    """Write records atomically as UTF-8 JSONL."""

    target = Path(output_path)
    if target.exists() and not overwrite:
        raise AnnotationError(
            "Output already exists (use --overwrite): {}".format(target)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            clean = {
                key: value for key, value in record.items()
                if not str(key).startswith("_source_")
            }
            handle.write(
                json.dumps(clean, ensure_ascii=False, sort_keys=True) + "\n"
            )
    os.replace(str(temporary), str(target))
    return target


def discover_videos(path: os.PathLike) -> List[Path]:
    target = Path(path)
    if target.is_file():
        return [target] if target.suffix.lower() in VIDEO_EXTENSIONS else []
    if not target.is_dir():
        return []
    return sorted(
        (
            item for item in target.rglob("*")
            if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda item: str(item),
    )


def _annotation_run(
    run_id: str,
    clip_id: str,
    video_path: Optional[str],
    fps: Optional[float] = None,
    frame_count: Optional[int] = None,
    weak_label: Optional[str] = None,
    weak_label_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    video: Dict[str, Any] = {
        "clip_id": str(clip_id),
        "input_path": str(video_path) if video_path else None,
    }
    if fps is not None:
        video["fps"] = float(fps)
    if frame_count is not None:
        video["frame_count"] = int(frame_count)
    result = {
        "schema": ANNOTATION_SCHEMA,
        "record_type": "run",
        "run_id": str(run_id),
        "clip_id": str(clip_id),
        "video": video,
        "clip_weak_label": (
            weak_label
            if weak_label in {"litter", "non_litter"}
            else infer_clip_weak_label(video_path)
        ),
        "weak_label_only": True,
        "provenance": {
            "template_initialization_only": True,
            "runtime_prediction_used_as_ground_truth": False,
        },
    }
    if weak_label_metadata:
        result["clip_weak_label_metadata"] = dict(weak_label_metadata)
    return result


def _annotation_event(
    run_id: str,
    clip_id: str,
    event_id: str,
) -> Dict[str, Any]:
    return {
        "schema": ANNOTATION_SCHEMA,
        "record_type": "event",
        "run_id": str(run_id),
        "clip_id": str(clip_id),
        "event_id": str(event_id),
        "ignore": False,
        "review": {
            "event": "unreviewed",
            "person": "unreviewed",
            "vehicle": "unreviewed",
            "release": "unreviewed",
        },
        "event_label": None,
        "admissible_routes": [],
        "release_interval": None,
        "notes": "",
    }


def init_annotation_records(
    candidate_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Create an independent, entirely unreviewed GT template.

    Candidate records are used only to create stable run/event slots.  Runtime
    assignment, actor identities, candidate ranks, costs, and release estimates
    are intentionally not copied.
    """

    candidates = [
        record for record in candidate_records
        if record.get("schema") == CANDIDATE_SCHEMA
    ]
    if not candidates and candidate_records:
        raise AnnotationError(
            "No {} records found".format(CANDIDATE_SCHEMA)
        )

    run_sources: Dict[str, Mapping[str, Any]] = {}
    event_sources: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for record in candidates:
        run_id = _run_id(record)
        if not run_id:
            raise AnnotationError("Candidate record is missing run_id/video path")
        record_type = _record_type(record)
        if record_type == "run":
            run_sources[run_id] = record
        elif record_type == "event":
            run_sources.setdefault(run_id, record)
            event_id = _event_id(record)
            if event_id is None:
                raise AnnotationError("Candidate event is missing event_id/litter_id")
            key = (run_id, event_id)
            if key in event_sources:
                raise AnnotationError(
                    "Duplicate candidate event {} / {}".format(run_id, event_id)
                )
            event_sources[key] = record

    output: List[Dict[str, Any]] = []
    for run_id in sorted(run_sources):
        source = run_sources[run_id]
        video_path = _video_path(source)
        clip_id = _clip_id(source) or run_id
        frame_count = _first_present(
            source.get("frame_count"),
            _nested(source, "video", "frame_count"),
        )
        try:
            frame_count = int(frame_count) if frame_count is not None else None
        except (TypeError, ValueError):
            frame_count = None
        explicit_weak_label, weak_label_metadata = _explicit_clip_weak_label(source)
        output.append(
            _annotation_run(
                run_id,
                clip_id,
                video_path,
                fps=_fps(source),
                frame_count=frame_count,
                weak_label=explicit_weak_label,
                weak_label_metadata=weak_label_metadata,
            )
        )
        for event_run_id, event_id in sorted(event_sources):
            if event_run_id == run_id:
                output.append(_annotation_event(run_id, clip_id, event_id))
    return output


def init_annotation_records_from_videos(path: os.PathLike) -> List[Dict[str, Any]]:
    """Create run-level templates from videos when no sidecar exists yet."""

    output = []
    for video_path in discover_videos(path):
        absolute = str(video_path.resolve())
        output.append(
            _annotation_run(
                _stable_path_id(absolute),
                video_path.stem,
                absolute,
            )
        )
    if not output:
        raise AnnotationError("No videos found under {}".format(path))
    return output


def init_annotations_from_path(path: os.PathLike) -> List[Dict[str, Any]]:
    """Initialize from candidate JSONL, or from a video file/directory."""

    target = Path(path)
    if target.is_file() and target.suffix.lower() in VIDEO_EXTENSIONS:
        return init_annotation_records_from_videos(target)
    record_files = []
    if target.is_file() and target.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
        record_files = [target]
    elif target.is_dir():
        record_files = [
            item for item in target.rglob("*")
            if item.is_file()
            and item.suffix.lower() in {".json", ".jsonl", ".ndjson"}
        ]
    if record_files:
        records = load_records(target)
        candidate_records = [
            item for item in records if item.get("schema") == CANDIDATE_SCHEMA
        ]
        if candidate_records:
            return init_annotation_records(candidate_records)
    return init_annotation_records_from_videos(target)


def _location(record: Mapping[str, Any], index: int) -> str:
    source = record.get("_source_file")
    line = record.get("_source_line")
    if source and line:
        return "{}:{}".format(source, line)
    if source:
        return str(source)
    return "record[{}]".format(index)


def _canonical_route_type(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "person_only": "person",
        "person_null_vehicle": "person",
        "vehicle": "direct_vehicle",
        "vehicle_only": "direct_vehicle",
        "null_person_vehicle": "null",
        "dustbin": "null",
        "none": "null",
    }
    return aliases.get(normalized, normalized)


def _actor_id(value: Any, expected_kind: str) -> Optional[str]:
    if value is None or value == "":
        return None
    if isinstance(value, Mapping):
        kind = _first_present(
            value.get("cls"), value.get("class"), value.get("kind"), expected_kind
        )
        identifier = _first_present(
            value.get("track_id"), value.get("id"), value.get("actor_id")
        )
        if identifier is None:
            return str(value)
        return "{}:{}".format(str(kind).lower(), identifier)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return "{}:{}".format(str(value[0]).lower(), value[1])
    text = str(value)
    if ":" in text:
        return text.lower()
    return "{}:{}".format(expected_kind, text)


def normalize_route(route: Any, fallback_rank: int = 1) -> Dict[str, Any]:
    """Normalize runtime or GT route aliases to a comparable representation."""

    if isinstance(route, str):
        return {
            "route_id": route,
            "route_type": None,
            "person_id": None,
            "vehicle_id": None,
            "rank": int(fallback_rank),
            "cost": None,
            "release_frame": None,
            "valid": True,
            "selected": False,
        }
    if not isinstance(route, Mapping):
        raise AnnotationError("Route must be an object or route_id string")

    person = _actor_id(
        _first_present(
            route.get("person_id"),
            route.get("person_key"),
            _nested(route, "actors", "person"),
        ),
        "person",
    )
    vehicle = _actor_id(
        _first_present(
            route.get("vehicle_id"),
            route.get("vehicle_key"),
            _nested(route, "actors", "vehicle"),
        ),
        "vehicle",
    )
    route_type = _canonical_route_type(
        _first_present(route.get("route_type"), route.get("type"))
    )
    if route_type is None:
        if person is not None and vehicle is not None:
            route_type = "person_vehicle"
        elif person is not None:
            route_type = "person"
        elif vehicle is not None:
            route_type = "direct_vehicle"
        else:
            route_type = "null"

    release_frame = _first_present(
        route.get("release_frame"),
        _nested(route, "metadata", "release_frame"),
        _nested(route, "release", "frame"),
    )
    try:
        release_frame = (
            int(round(float(release_frame))) if release_frame is not None else None
        )
    except (TypeError, ValueError):
        release_frame = None

    valid = bool(route.get("valid", True))
    rank_value = route.get("rank", fallback_rank)
    if rank_value is None and not valid:
        rank = None
    else:
        try:
            rank = int(rank_value)
        except (TypeError, ValueError):
            rank = int(fallback_rank)
    cost = _first_present(route.get("cost"), route.get("total_cost"))
    try:
        cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost = None
    return {
        "route_id": (
            str(route.get("route_id")) if route.get("route_id") is not None else None
        ),
        "route_type": route_type,
        "person_id": person,
        "vehicle_id": vehicle,
        "rank": rank,
        "cost": cost,
        "release_frame": release_frame,
        "valid": valid,
        "selected": bool(route.get("selected", False)),
    }


def candidate_routes(
    record: Mapping[str, Any],
    include_invalid: bool = False,
) -> List[Dict[str, Any]]:
    values = record.get("routes")
    if not isinstance(values, list):
        values = record.get("candidates")
    if not isinstance(values, list):
        values = []
    routes = [normalize_route(value, index) for index, value in enumerate(values, 1)]
    if not include_invalid:
        routes = [route for route in routes if route["valid"]]
    routes.sort(
        key=lambda route: (
            int(route["rank"]) if route["rank"] is not None else 10 ** 12,
            float(route["cost"]) if route["cost"] is not None else float("inf"),
            str(route["route_id"]),
        )
    )
    return routes


def _route_errors(route: Dict[str, Any], prefix: str) -> List[str]:
    errors = []
    route_type = route["route_type"]
    if route_type not in ROUTE_TYPES:
        return ["{} route_type must be one of {}".format(
            prefix, sorted(ROUTE_TYPES)
        )]
    person = route["person_id"]
    vehicle = route["vehicle_id"]
    if route_type == "person_vehicle" and (person is None or vehicle is None):
        errors.append("{} person_vehicle requires person_id and vehicle_id".format(prefix))
    elif route_type == "person" and (person is None or vehicle is not None):
        errors.append("{} person requires person_id and NULL vehicle".format(prefix))
    elif route_type == "direct_vehicle" and (
        person is not None or vehicle is None
    ):
        errors.append(
            "{} direct_vehicle requires NULL person and vehicle_id".format(prefix)
        )
    elif route_type == "null" and (person is not None or vehicle is not None):
        errors.append("{} null requires NULL person and vehicle".format(prefix))
    return errors


def parse_release_interval(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        start = _first_present(
            value.get("start_frame"), value.get("start"), value.get("min_frame")
        )
        end = _first_present(
            value.get("end_frame"), value.get("end"), value.get("max_frame")
        )
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    else:
        raise AnnotationError(
            "release_interval must be [start,end], an object, or null"
        )
    try:
        start_int = int(round(float(start)))
        end_int = int(round(float(end)))
    except (TypeError, ValueError):
        raise AnnotationError("release_interval endpoints must be numeric")
    if start_int > end_int:
        raise AnnotationError("release_interval start must be <= end")
    return start_int, end_int


def validate_records(
    records: Sequence[Mapping[str, Any]],
    require_reviewed: bool = False,
) -> Dict[str, Any]:
    """Validate candidate/annotation records and return structured diagnostics."""

    errors: List[str] = []
    warnings: List[str] = []
    counts: Counter = Counter()
    run_keys = set()
    event_keys = set()
    schemas = set()

    for index, record in enumerate(records):
        location = _location(record, index)
        schema = record.get("schema")
        schemas.add(schema)
        if schema not in {CANDIDATE_SCHEMA, ANNOTATION_SCHEMA}:
            errors.append("{} unknown schema {!r}".format(location, schema))
            continue
        record_type = _record_type(record)
        if record_type not in RECORD_TYPES:
            errors.append("{} record_type must be run or event".format(location))
            continue
        counts["{}_{}".format(
            "candidate" if schema == CANDIDATE_SCHEMA else "annotation",
            record_type,
        )] += 1
        run_id = _run_id(record)
        if not run_id:
            errors.append("{} missing run_id".format(location))
            continue
        if record_type == "run":
            key = (schema, run_id)
            if key in run_keys:
                errors.append("{} duplicate run_id {}".format(location, run_id))
            run_keys.add(key)
            if schema == ANNOTATION_SCHEMA:
                weak = record.get("clip_weak_label", "unreviewed")
                if weak not in {"litter", "non_litter", "unreviewed"}:
                    errors.append(
                        "{} invalid clip_weak_label {!r}".format(location, weak)
                    )
            continue

        event_id = _event_id(record)
        if event_id is None:
            errors.append("{} missing event_id/litter_id".format(location))
            continue
        key = (schema, run_id, event_id)
        if key in event_keys:
            errors.append(
                "{} duplicate event {} / {}".format(location, run_id, event_id)
            )
        event_keys.add(key)

        if schema == CANDIDATE_SCHEMA:
            routes = candidate_routes(record, include_invalid=True)
            if not routes:
                errors.append("{} event has no routes".format(location))
                continue
            seen_ranks = set()
            seen_route_ids = set()
            selected_count = 0
            raw_routes = record.get("routes", record.get("candidates", []))
            for route_index, route in enumerate(routes, 1):
                errors.extend(
                    _route_errors(route, "{} route[{}]".format(location, route_index))
                )
                if route["valid"]:
                    if route["rank"] is None or route["rank"] < 1:
                        errors.append(
                            "{} valid route rank must be >= 1".format(location)
                        )
                    elif route["rank"] in seen_ranks:
                        errors.append(
                            "{} duplicate valid route rank {}".format(
                                location, route["rank"]
                            )
                        )
                    else:
                        seen_ranks.add(route["rank"])
                if route["route_id"]:
                    if route["route_id"] in seen_route_ids:
                        errors.append(
                            "{} duplicate route_id {}".format(
                                location, route["route_id"]
                            )
                        )
                    seen_route_ids.add(route["route_id"])
            if isinstance(raw_routes, list):
                selected_count = sum(
                    bool(item.get("selected"))
                    for item in raw_routes if isinstance(item, Mapping)
                )
            if selected_count != 1:
                errors.append(
                    "{} must have exactly one selected route (found {})".format(
                        location, selected_count
                    )
                )
            selected_routes = [route for route in routes if route["selected"]]
            if selected_routes and not selected_routes[0]["valid"]:
                errors.append("{} selected route must be valid".format(location))
            full_null_routes = [
                route for route in routes
                if route["valid"]
                and route["route_type"] == "null"
                and route["person_id"] is None
                and route["vehicle_id"] is None
            ]
            if len(full_null_routes) != 1:
                errors.append(
                    "{} must have exactly one valid full NULL route (found {})".format(
                        location, len(full_null_routes)
                    )
                )
            continue

        ignore = record.get("ignore", False)
        if not isinstance(ignore, bool):
            errors.append("{} ignore must be boolean".format(location))
        review = record.get("review")
        if not isinstance(review, Mapping):
            errors.append("{} annotation event requires review object".format(location))
            review = {}
        for field in ("event", "person", "vehicle", "release"):
            state = review.get(field, "unreviewed")
            if state not in REVIEW_STATES:
                errors.append(
                    "{} review.{} must be one of {}".format(
                        location, field, sorted(REVIEW_STATES)
                    )
                )
            if require_reviewed and not ignore and state == "unreviewed":
                errors.append(
                    "{} review.{} is still unreviewed".format(location, field)
                )
        event_label = record.get("event_label")
        if event_label is not None and event_label not in EVENT_LABELS:
            errors.append(
                "{} event_label must be litter, not_litter, or null".format(location)
            )
        if review.get("event") == "reviewed" and event_label is None and not ignore:
            errors.append("{} reviewed event requires event_label".format(location))

        raw_routes = record.get("admissible_routes")
        if not isinstance(raw_routes, list):
            errors.append("{} admissible_routes must be a list".format(location))
            raw_routes = []
        normalized_gt = []
        for route_index, raw_route in enumerate(raw_routes, 1):
            try:
                route = normalize_route(raw_route, route_index)
            except AnnotationError as exc:
                errors.append("{} route[{}]: {}".format(location, route_index, exc))
                continue
            normalized_gt.append(route)
            errors.extend(
                _route_errors(route, "{} admissible_routes[{}]".format(
                    location, route_index
                ))
            )
        route_reviewed = (
            review.get("person") == "reviewed"
            and review.get("vehicle") == "reviewed"
        )
        if (
            route_reviewed
            and event_label == "litter"
            and not ignore
            and not normalized_gt
        ):
            errors.append(
                "{} reviewed litter attribution requires admissible_routes".format(
                    location
                )
            )
        if event_label == "not_litter" and normalized_gt:
            errors.append(
                "{} not_litter event must not define attribution routes".format(
                    location
                )
            )
        try:
            interval = parse_release_interval(record.get("release_interval"))
        except AnnotationError as exc:
            errors.append("{}: {}".format(location, exc))
            interval = None
        if (
            review.get("release") == "reviewed"
            and event_label == "litter"
            and interval is None
            and not ignore
        ):
            errors.append(
                "{} reviewed litter release requires release_interval".format(
                    location
                )
            )

    # Candidate and annotation records may be validated together.  Events only
    # need a same-schema run if at least one run record of that schema was
    # provided; standalone event shards remain valid.
    for schema in (CANDIDATE_SCHEMA, ANNOTATION_SCHEMA):
        schema_runs = {run_id for item_schema, run_id in run_keys if item_schema == schema}
        if schema_runs:
            for item_schema, run_id, event_id in event_keys:
                if item_schema == schema and run_id not in schema_runs:
                    errors.append(
                        "{} event {} references missing run record".format(
                            run_id, event_id
                        )
                    )

    return {
        "valid": not errors,
        "schemas": sorted(str(value) for value in schemas if value is not None),
        "counts": dict(sorted(counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def _event_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    run_id = _run_id(record)
    event_id = _event_id(record)
    if run_id is None or event_id is None:
        raise AnnotationError("Event record is missing run_id/event_id")
    return run_id, event_id


def _route_matches(candidate: Mapping[str, Any], truth: Mapping[str, Any]) -> bool:
    truth_route_id = truth.get("route_id")
    # A route-id-only annotation is supported for legacy/manual convenience.
    if truth_route_id and truth.get("route_type") is None:
        return candidate.get("route_id") == truth_route_id
    return (
        candidate.get("route_type") == truth.get("route_type")
        and candidate.get("person_id") == truth.get("person_id")
        and candidate.get("vehicle_id") == truth.get("vehicle_id")
    )


def _any_route_match(
    candidates: Sequence[Mapping[str, Any]],
    truths: Sequence[Mapping[str, Any]],
) -> bool:
    return any(
        _route_matches(candidate, truth)
        for candidate in candidates
        for truth in truths
    )


def _safe_ratio(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator) / denominator if denominator else None


def _binary_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Any]:
    total = tp + tn + fp + fn
    return {
        "support": total,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": _safe_ratio(tp + tn, total),
        "precision": _safe_ratio(tp, tp + fp),
        "recall": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "positive_class": "null",
    }


def evaluate_candidates(
    candidate_records: Sequence[Mapping[str, Any]],
    annotation_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate ranked routes against reviewed, possibly ambiguous GT.

    Metrics:

    * exact route Top-1 and Recall@1/3/5,
    * person, vehicle, and full-route candidate coverage,
    * top-1 release interval hit and distance-to-interval MAE,
    * binary NULL-vs-non-NULL classification.
    """

    candidate_validation = validate_records(candidate_records)
    annotation_validation = validate_records(annotation_records)
    if not candidate_validation["valid"]:
        raise AnnotationError(
            "Invalid candidates:\n{}".format(
                "\n".join(candidate_validation["errors"])
            )
        )
    if not annotation_validation["valid"]:
        raise AnnotationError(
            "Invalid annotations:\n{}".format(
                "\n".join(annotation_validation["errors"])
            )
        )

    candidate_events = {
        _event_key(record): record
        for record in candidate_records
        if record.get("schema") == CANDIDATE_SCHEMA
        and _record_type(record) == "event"
    }
    annotation_events = [
        record for record in annotation_records
        if record.get("schema") == ANNOTATION_SCHEMA
        and _record_type(record) == "event"
    ]
    candidate_runs = {
        _run_id(record): record
        for record in candidate_records
        if record.get("schema") == CANDIDATE_SCHEMA
        and _record_type(record) == "run"
    }

    ignored = 0
    unreviewed = 0
    not_litter = 0
    route_eligible = 0
    missing_candidate_event = 0
    exact_top1_hits = 0
    recall_hits = {1: 0, 3: 0, 5: 0}
    person_denominator = 0
    person_hits = 0
    vehicle_denominator = 0
    vehicle_hits = 0
    route_coverage_hits = 0
    release_denominator = 0
    release_hits = 0
    release_errors_frames: List[float] = []
    release_errors_seconds: List[float] = []
    release_missing_prediction = 0
    null_tp = null_tn = null_fp = null_fn = 0
    null_ambiguous_skipped = 0
    null_missing_prediction = 0

    for truth_record in annotation_events:
        if truth_record.get("ignore", False):
            ignored += 1
            continue
        review = truth_record.get("review", {})
        if (
            review.get("event") != "reviewed"
            or truth_record.get("event_label") is None
        ):
            unreviewed += 1
            continue
        if truth_record.get("event_label") == "not_litter":
            not_litter += 1
            continue
        if (
            review.get("person") != "reviewed"
            or review.get("vehicle") != "reviewed"
        ):
            unreviewed += 1
            continue

        truths = [
            normalize_route(route, index)
            for index, route in enumerate(
                truth_record.get("admissible_routes", []), 1
            )
        ]
        if not truths:
            # This is normally prevented by validation; keep the evaluator safe
            # if validation behavior changes.
            unreviewed += 1
            continue

        route_eligible += 1
        candidate_record = candidate_events.get(_event_key(truth_record))
        routes = candidate_routes(candidate_record) if candidate_record else []
        if candidate_record is None:
            missing_candidate_event += 1
        top1 = routes[:1]
        if _any_route_match(top1, truths):
            exact_top1_hits += 1
        for k in recall_hits:
            if _any_route_match(routes[:k], truths):
                recall_hits[k] += 1

        if _any_route_match(routes, truths):
            route_coverage_hits += 1
        gt_persons = {
            route["person_id"] for route in truths
            if route["person_id"] is not None
        }
        if gt_persons:
            person_denominator += 1
            candidate_persons = {
                route["person_id"] for route in routes
                if route["person_id"] is not None
            }
            if candidate_persons.intersection(gt_persons):
                person_hits += 1
        gt_vehicles = {
            route["vehicle_id"] for route in truths
            if route["vehicle_id"] is not None
        }
        if gt_vehicles:
            vehicle_denominator += 1
            candidate_vehicles = {
                route["vehicle_id"] for route in routes
                if route["vehicle_id"] is not None
            }
            if candidate_vehicles.intersection(gt_vehicles):
                vehicle_hits += 1

        gt_null_values = {
            route["route_type"] == "null" for route in truths
        }
        if len(gt_null_values) != 1:
            null_ambiguous_skipped += 1
        elif not top1:
            null_missing_prediction += 1
        else:
            gt_null = next(iter(gt_null_values))
            pred_null = top1[0]["route_type"] == "null"
            if gt_null and pred_null:
                null_tp += 1
            elif gt_null and not pred_null:
                null_fn += 1
            elif not gt_null and pred_null:
                null_fp += 1
            else:
                null_tn += 1

        if review.get("release") == "reviewed":
            interval = parse_release_interval(truth_record.get("release_interval"))
            if interval is not None:
                release_denominator += 1
                prediction = top1[0]["release_frame"] if top1 else None
                if prediction is None:
                    release_missing_prediction += 1
                else:
                    start, end = interval
                    if start <= prediction <= end:
                        distance = 0.0
                        release_hits += 1
                    else:
                        distance = float(
                            start - prediction
                            if prediction < start
                            else prediction - end
                        )
                    release_errors_frames.append(distance)
                    run_id, _ = _event_key(truth_record)
                    fps = _fps(candidate_record or {})
                    if fps is None:
                        fps = _fps(candidate_runs.get(run_id, {}))
                    if fps:
                        release_errors_seconds.append(distance / fps)

    all_annotation_keys = {
        _event_key(record) for record in annotation_events
    }
    extra_candidate_events = sum(
        key not in all_annotation_keys for key in candidate_events
    )
    evaluated_release_predictions = len(release_errors_frames)
    report = {
        "schema": "smart-backtrack-evaluation/v1",
        "counts": {
            "candidate_events": len(candidate_events),
            "annotation_events": len(annotation_events),
            "route_evaluable_events": route_eligible,
            "ignored_events": ignored,
            "unreviewed_events": unreviewed,
            "not_litter_events_excluded": not_litter,
            "missing_candidate_events": missing_candidate_event,
            "extra_candidate_events": extra_candidate_events,
        },
        "exact_route_top1": {
            "hits": exact_top1_hits,
            "support": route_eligible,
            "value": _safe_ratio(exact_top1_hits, route_eligible),
        },
        "recall_at_k": {
            str(k): {
                "hits": recall_hits[k],
                "support": route_eligible,
                "value": _safe_ratio(recall_hits[k], route_eligible),
            }
            for k in (1, 3, 5)
        },
        "candidate_coverage": {
            "person": {
                "hits": person_hits,
                "support": person_denominator,
                "value": _safe_ratio(person_hits, person_denominator),
            },
            "vehicle": {
                "hits": vehicle_hits,
                "support": vehicle_denominator,
                "value": _safe_ratio(vehicle_hits, vehicle_denominator),
            },
            "route": {
                "hits": route_coverage_hits,
                "support": route_eligible,
                "value": _safe_ratio(route_coverage_hits, route_eligible),
            },
        },
        "release_interval": {
            "support": release_denominator,
            "predictions_with_frame": evaluated_release_predictions,
            "missing_prediction": release_missing_prediction,
            "hits": release_hits,
            "hit_rate": _safe_ratio(release_hits, release_denominator),
            "mae_frames": (
                sum(release_errors_frames) / evaluated_release_predictions
                if evaluated_release_predictions else None
            ),
            "mae_seconds": (
                sum(release_errors_seconds) / len(release_errors_seconds)
                if release_errors_seconds else None
            ),
        },
        "null_classification": {
            **_binary_metrics(null_tp, null_tn, null_fp, null_fn),
            "ambiguous_null_and_non_null_skipped": null_ambiguous_skipped,
            "missing_prediction_skipped": null_missing_prediction,
        },
        "metric_notes": {
            "ambiguity": (
                "A prediction is correct if it matches any admissible route."
            ),
            "release_mae": (
                "Distance to the nearest interval boundary; zero inside interval."
            ),
            "null_positive_class": "NULL is the positive class.",
            "multi_person_same_vehicle": (
                "Legal: no one-to-one capacity is imposed by this evaluator."
            ),
        },
    }
    return report


# Friendly aliases for external scripts/tests.
initialize_annotations = init_annotation_records
evaluate = evaluate_candidates
validate = validate_records


__all__ = [
    "ANNOTATION_SCHEMA",
    "AnnotationError",
    "CANDIDATE_SCHEMA",
    "candidate_routes",
    "discover_record_files",
    "discover_videos",
    "evaluate",
    "evaluate_candidates",
    "infer_clip_weak_label",
    "init_annotation_records",
    "init_annotation_records_from_videos",
    "init_annotations_from_path",
    "initialize_annotations",
    "load_records",
    "normalize_route",
    "parse_release_interval",
    "validate",
    "validate_records",
    "write_records",
]
