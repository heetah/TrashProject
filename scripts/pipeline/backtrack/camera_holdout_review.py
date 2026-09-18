"""Single-reviewer camera/site holdout review schema.

This module is a research/data-audit utility with no production caller.  It
creates a model-blind, one-reviewer worksheet from a source manifest and
validates the completed provenance fields before a camera-disjoint evaluation
is attempted.  It never infers camera IDs, promotes model output to truth, or
computes accuracy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "camera-holdout-review/v1"
SOURCE_INDEX_SCHEMA = "target41-camera-safe-review-index/v1"
CLIP_STATUSES = frozenset({"positive", "negative", "ambiguous", "unusable"})
REVIEW_TABLE_COLUMNS = (
    "case_id",
    "video_filename",
    "source_video",
    "source_sha256",
    "review_state",
    "reviewer_id",
    "reviewed_at",
    "clip_status",
    "video_usable",
    "camera_id",
    "site_id",
    "session_id",
    "source_recording_id",
    "source_sha256_confirmed",
    "events_json",
    "notes",
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class CameraHoldoutValidationError(ValueError):
    """Raised when a camera-holdout queue violates its data contract."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraHoldoutValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, label: str) -> str:
    result = _text(value, label).lower()
    if _SHA256.fullmatch(result) is None:
        raise CameraHoldoutValidationError(f"{label} must be a 64-character SHA-256")
    return result


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _blank_review() -> dict[str, Any]:
    return {
        "review_state": "unreviewed",
        "reviewer_id": None,
        "reviewed_at": None,
        "clip_status": None,
        "video_usable": None,
        "camera_id": None,
        "site_id": None,
        "session_id": None,
        "source_recording_id": None,
        "source_sha256_confirmed": None,
        "events": [],
        "notes": None,
    }


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _parse_csv_optional_bool(value: Any, label: str) -> bool | None:
    text = "" if value is None else str(value).strip().casefold()
    if not text:
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    raise CameraHoldoutValidationError(f"{label} must be true, false, or blank")


def _parse_csv_optional_text(value: Any, label: str) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _optional_text(value, label)


def export_review_table(queue: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return a model-blind, one-row-per-case CSV representation."""

    cases = _queue_cases(queue)
    rows: list[dict[str, str]] = []
    for case in sorted(cases, key=lambda item: str(item["case_id"])):
        review = case["review"]
        events = review.get("events", [])
        if not isinstance(events, list):
            raise CameraHoldoutValidationError(
                f"{case['case_id']}.review.events must be a list"
            )
        row = {
            "case_id": str(case["case_id"]),
            "video_filename": str(case["video_filename"]),
            "source_video": str(case["source_video"]),
            "source_sha256": str(case["source_sha256"]).lower(),
            "review_state": _csv_value(review.get("review_state")),
            "reviewer_id": _csv_value(review.get("reviewer_id")),
            "reviewed_at": _csv_value(review.get("reviewed_at")),
            "clip_status": _csv_value(review.get("clip_status")),
            "video_usable": _csv_value(review.get("video_usable")),
            "camera_id": _csv_value(review.get("camera_id")),
            "site_id": _csv_value(review.get("site_id")),
            "session_id": _csv_value(review.get("session_id")),
            "source_recording_id": _csv_value(review.get("source_recording_id")),
            "source_sha256_confirmed": _csv_value(
                review.get("source_sha256_confirmed")
            ),
            "events_json": json.dumps(
                events, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            "notes": _csv_value(review.get("notes")),
        }
        rows.append(row)
    return rows


def _table_review(row: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    expected = set(REVIEW_TABLE_COLUMNS)
    actual = set(row)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise CameraHoldoutValidationError(
            f"{case_id}: review table missing columns {sorted(missing)}"
        )
    if extra:
        raise CameraHoldoutValidationError(
            "{}: review table contains non-provenance columns {}".format(
                case_id, sorted((str(value) for value in extra))
            )
        )
    state = _text(row.get("review_state"), f"{case_id}.review_state").casefold()
    if state not in {"unreviewed", "reviewed"}:
        raise CameraHoldoutValidationError(
            f"{case_id}.review_state must be unreviewed or reviewed"
        )
    events_text = "" if row.get("events_json") is None else str(row["events_json"]).strip()
    if not events_text:
        events: list[Any] = []
    else:
        try:
            events = json.loads(events_text)
        except json.JSONDecodeError as exc:
            raise CameraHoldoutValidationError(
                f"{case_id}.events_json is not valid JSON"
            ) from exc
        if not isinstance(events, list):
            raise CameraHoldoutValidationError(f"{case_id}.events_json must encode a list")
    return {
        "review_state": state,
        "reviewer_id": _parse_csv_optional_text(
            row.get("reviewer_id"), f"{case_id}.reviewer_id"
        ),
        "reviewed_at": _parse_csv_optional_text(
            row.get("reviewed_at"), f"{case_id}.reviewed_at"
        ),
        "clip_status": _parse_csv_optional_text(
            row.get("clip_status"), f"{case_id}.clip_status"
        ),
        "video_usable": _parse_csv_optional_bool(
            row.get("video_usable"), f"{case_id}.video_usable"
        ),
        "camera_id": _parse_csv_optional_text(
            row.get("camera_id"), f"{case_id}.camera_id"
        ),
        "site_id": _parse_csv_optional_text(row.get("site_id"), f"{case_id}.site_id"),
        "session_id": _parse_csv_optional_text(
            row.get("session_id"), f"{case_id}.session_id"
        ),
        "source_recording_id": _parse_csv_optional_text(
            row.get("source_recording_id"), f"{case_id}.source_recording_id"
        ),
        "source_sha256_confirmed": _parse_csv_optional_bool(
            row.get("source_sha256_confirmed"),
            f"{case_id}.source_sha256_confirmed",
        ),
        "events": events,
        "notes": _parse_csv_optional_text(row.get("notes"), f"{case_id}.notes"),
    }


def import_review_table(
    queue: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Merge a reviewer-edited CSV table into an immutable queue template.

    Source identity fields are checked against the template and can never be
    changed by the table.  Model-output columns, duplicate IDs and membership
    changes fail closed.  The returned queue may still be partially reviewed;
    ``validate_completed_review`` remains the completion gate.
    """

    cases = _queue_cases(queue)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise CameraHoldoutValidationError("review table rows must be non-empty")
    by_id = {str(case["case_id"]): case for case in cases}
    parsed: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 2):
        if not isinstance(row, Mapping):
            raise CameraHoldoutValidationError(f"review table row {index} must be an object")
        case_id = _text(row.get("case_id"), f"review table row {index}.case_id")
        if case_id in parsed:
            raise CameraHoldoutValidationError(f"review table duplicate case_id {case_id}")
        if case_id not in by_id:
            raise CameraHoldoutValidationError(f"review table has unknown case_id {case_id}")
        template = by_id[case_id]
        for field in ("video_filename", "source_video"):
            if _text(row.get(field), f"{case_id}.{field}") != str(template[field]):
                raise CameraHoldoutValidationError(
                    f"{case_id}.{field} does not match queue template"
                )
        if _sha256(row.get("source_sha256"), f"{case_id}.source_sha256") != str(
            template["source_sha256"]
        ).lower():
            raise CameraHoldoutValidationError(
                f"{case_id}.source_sha256 does not match queue template"
            )
        parsed[case_id] = _table_review(row, case_id)
    if set(parsed) != set(by_id):
        missing = sorted(set(by_id) - set(parsed))
        raise CameraHoldoutValidationError(
            f"review table case membership mismatch; missing {missing}"
        )
    result = dict(queue)
    result["cases"] = []
    for case in cases:
        copied = dict(case)
        copied["review"] = parsed[str(case["case_id"])]
        result["cases"].append(copied)
    return result


def _normalise_case(case: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(case, Mapping):
        raise CameraHoldoutValidationError(f"case {index} must be an object")
    return {
        "case_id": _text(case.get("case_id"), f"case {index}.case_id"),
        "video_filename": _text(
            case.get("video_filename"), f"case {index}.video_filename"
        ),
        "source_video": _text(
            case.get("source_video"), f"case {index}.source_video"
        ),
        "source_sha256": _sha256(
            case.get("source_sha256"), f"case {index}.source_sha256"
        ),
        "review": _blank_review(),
    }


def build_single_reviewer_queue(
    cases: Sequence[Mapping[str, Any]],
    *,
    selection_manifest: str | None = None,
    run_manifest: str | None = None,
    selection_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Create a deterministic model-blind worksheet for one reviewer.

    Only immutable source identity is copied from ``cases``.  Model summaries,
    route scores, assignments, release hypotheses and annotated output paths
    are intentionally omitted.  ``selection_manifest_sha256`` is optional
    provenance supplied by the caller and is never computed or guessed here.
    """

    if isinstance(cases, (str, bytes)) or not isinstance(cases, Sequence):
        raise CameraHoldoutValidationError("cases must be a non-empty sequence")
    if not cases:
        raise CameraHoldoutValidationError("cases must be a non-empty sequence")
    if selection_manifest_sha256 is not None:
        selection_manifest_sha256 = _sha256(
            selection_manifest_sha256, "selection_manifest_sha256"
        )
    normalized = [_normalise_case(case, index) for index, case in enumerate(cases, 1)]
    case_ids = [case["case_id"] for case in normalized]
    if len(set(case_ids)) != len(case_ids):
        raise CameraHoldoutValidationError("case_id values must be unique")
    normalized.sort(key=lambda case: case["case_id"])
    return {
        "schema": SCHEMA,
        "purpose": (
            "Single-reviewer camera/site/source truth collection; model output "
            "is not included or treated as a label."
        ),
        "reviewer_count": 1,
        "model_outputs_included": False,
        "selection": {
            "manifest": _optional_text(selection_manifest, "selection_manifest"),
            "manifest_sha256": selection_manifest_sha256,
            "run_manifest": _optional_text(run_manifest, "run_manifest"),
        },
        "cases": normalized,
    }


def build_from_review_index(index: Mapping[str, Any]) -> dict[str, Any]:
    """Strip model predictions from a camera-safe review index.

    The source index may contain model summaries and annotated output paths;
    none of those fields cross the blinding boundary.  The input schema and
    ``case_count`` are checked before building the new queue.
    """

    if not isinstance(index, Mapping) or index.get("schema") != SOURCE_INDEX_SCHEMA:
        raise CameraHoldoutValidationError(
            f"index must use {SOURCE_INDEX_SCHEMA}"
        )
    cases = index.get("cases")
    if not isinstance(cases, list):
        raise CameraHoldoutValidationError("index.cases must be a list")
    if index.get("case_count") != len(cases):
        raise CameraHoldoutValidationError("index.case_count does not match cases")
    return build_single_reviewer_queue(
        cases,
        selection_manifest=index.get("selection_manifest"),
        run_manifest=index.get("run_manifest"),
    )


def _queue_cases(queue: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(queue, Mapping) or queue.get("schema") != SCHEMA:
        raise CameraHoldoutValidationError(f"queue must use {SCHEMA}")
    if queue.get("reviewer_count") != 1:
        raise CameraHoldoutValidationError("reviewer_count must be exactly one")
    if queue.get("model_outputs_included") is not False:
        raise CameraHoldoutValidationError("queue must declare model_outputs_included=false")
    cases = queue.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CameraHoldoutValidationError("queue.cases must be a non-empty list")
    seen: set[str] = set()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, Mapping):
            raise CameraHoldoutValidationError(f"case {index} must be an object")
        allowed = {"case_id", "video_filename", "source_video", "source_sha256", "review"}
        extra = set(case) - allowed
        if extra:
            raise CameraHoldoutValidationError(
                f"case {index} contains non-provenance fields: {sorted(extra)}"
            )
        case_id = _text(case.get("case_id"), f"case {index}.case_id")
        if case_id in seen:
            raise CameraHoldoutValidationError("case_id values must be unique")
        seen.add(case_id)
        _text(case.get("video_filename"), f"case {index}.video_filename")
        _text(case.get("source_video"), f"case {index}.source_video")
        _sha256(case.get("source_sha256"), f"case {index}.source_sha256")
        if not isinstance(case.get("review"), Mapping):
            raise CameraHoldoutValidationError(f"case {index}.review must be an object")
    return cases


def validate_unreviewed_queue(queue: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that a newly built queue is blank and model-blind."""

    cases = _queue_cases(queue)
    expected = _blank_review()
    for index, case in enumerate(cases, 1):
        if dict(case["review"]) != expected:
            raise CameraHoldoutValidationError(f"case {index}.review is not blank")
    return {
        "valid": True,
        "schema": SCHEMA,
        "case_count": len(cases),
        "reviewer_count": 1,
        "model_outputs_included": False,
        "review_state": "unreviewed",
    }


def verify_source_files(queue: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every source SHA-256 and fail closed on missing/mismatched files."""

    cases = _queue_cases(queue)
    for case in cases:
        source = Path(str(case["source_video"]))
        if not source.is_file():
            raise CameraHoldoutValidationError(
                f"source file is missing: {source}"
            )
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        expected = _sha256(case["source_sha256"], f"{case['case_id']}.source_sha256")
        if digest.hexdigest() != expected:
            raise CameraHoldoutValidationError(
                f"source SHA-256 mismatch for {case['case_id']}"
            )
    return {"valid": True, "verified_source_files": len(cases)}


def _normalise_id_set(values: Iterable[str] | None, label: str) -> set[str] | None:
    if values is None:
        return None
    result = set()
    for value in values:
        result.add(_text(value, label))
    return result


def validate_completed_review(
    queue: Mapping[str, Any],
    *,
    reviewed_camera_ids: Iterable[str] | None = None,
    reviewed_source_recording_ids: Iterable[str] | None = None,
    reviewed_source_hashes: Iterable[str] | None = None,
    expected_case_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate one reviewer's completed provenance and report the holdout gate.

    Missing or malformed review fields raise.  A well-formed but incomplete
    review returns ``ready_for_camera_disjoint_eval=False`` with explicit
    reasons.  Camera independence is reported as verified only when the caller
    supplies non-empty reviewed camera/source-recording sets and no overlap is
    found; this function never invents those sets.
    """

    cases = _queue_cases(queue)
    expected_ids = _normalise_id_set(expected_case_ids, "expected_case_id")
    actual_ids = {str(case["case_id"]) for case in cases}
    if expected_ids is not None and actual_ids != expected_ids:
        raise CameraHoldoutValidationError("queue case IDs do not match expected_case_ids")
    known_cameras = _normalise_id_set(reviewed_camera_ids, "reviewed_camera_id")
    known_sources = _normalise_id_set(
        reviewed_source_recording_ids, "reviewed_source_recording_id"
    )
    known_hashes = (
        {_sha256(value, "reviewed_source_hash") for value in reviewed_source_hashes}
        if reviewed_source_hashes is not None
        else None
    )

    source_hashes: list[str] = []
    camera_ids: set[str] = set()
    source_recording_ids: set[str] = set()
    counts = {status: 0 for status in sorted(CLIP_STATUSES)}
    not_ready: list[str] = []
    for index, case in enumerate(cases, 1):
        review = case["review"]
        case_id = str(case["case_id"])
        if review.get("review_state") != "reviewed":
            raise CameraHoldoutValidationError(f"{case_id} is not marked reviewed")
        _text(review.get("reviewer_id"), f"{case_id}.reviewer_id")
        _text(review.get("reviewed_at"), f"{case_id}.reviewed_at")
        status = review.get("clip_status")
        if status not in CLIP_STATUSES:
            raise CameraHoldoutValidationError(
                f"{case_id}.clip_status must be one of {sorted(CLIP_STATUSES)}"
            )
        counts[status] += 1
        if type(review.get("video_usable")) is not bool:
            raise CameraHoldoutValidationError(f"{case_id}.video_usable must be boolean")
        if review.get("source_sha256_confirmed") is not True:
            raise CameraHoldoutValidationError(
                f"{case_id}.source_sha256_confirmed must be true"
            )
        camera = _text(review.get("camera_id"), f"{case_id}.camera_id")
        source_recording = _text(
            review.get("source_recording_id"), f"{case_id}.source_recording_id"
        )
        _text(review.get("site_id"), f"{case_id}.site_id")
        _text(review.get("session_id"), f"{case_id}.session_id")
        source_hash = _sha256(case.get("source_sha256"), f"{case_id}.source_sha256")
        source_hashes.append(source_hash)
        camera_ids.add(camera)
        source_recording_ids.add(source_recording)
        events = review.get("events")
        if not isinstance(events, list):
            raise CameraHoldoutValidationError(f"{case_id}.events must be a list")
        if status == "positive" and not events:
            not_ready.append(f"{case_id}: positive clip has no reviewed event rows")
        if status == "negative" and events:
            not_ready.append(f"{case_id}: negative clip contains event rows")
        if status in {"ambiguous", "unusable"}:
            not_ready.append(f"{case_id}: clip status is {status}")
        if review.get("video_usable") is not True:
            not_ready.append(f"{case_id}: video_usable is not true")

    if len(source_hashes) != len(set(source_hashes)):
        raise CameraHoldoutValidationError("holdout source_sha256 values must be unique")
    overlaps: dict[str, list[str]] = {}
    if known_cameras is not None:
        overlaps["camera_ids"] = sorted(camera_ids & known_cameras)
    if known_sources is not None:
        overlaps["source_recording_ids"] = sorted(source_recording_ids & known_sources)
    if known_hashes is not None:
        overlaps["source_hashes"] = sorted(set(source_hashes) & known_hashes)
    overlap_values = {key: value for key, value in overlaps.items() if value}
    if overlap_values:
        not_ready.append(f"reviewed/holdout provenance overlap: {overlap_values}")

    independence_verified = bool(
        known_cameras
        and known_sources
        and not overlap_values
    )
    if not independence_verified:
        not_ready.append("reviewed camera/source-recording groups were not independently supplied")
    ready = not not_ready and independence_verified and counts["positive"] > 0 and counts["negative"] > 0
    if counts["negative"] == 0:
        not_ready.append("no reviewed negative clips")
    return {
        "valid": True,
        "schema": SCHEMA,
        "case_count": len(cases),
        "clip_status_counts": counts,
        "camera_group_count": len(camera_ids),
        "source_recording_group_count": len(source_recording_ids),
        "independence_verified": independence_verified,
        "provenance_overlaps": overlap_values,
        "ready_for_camera_disjoint_eval": ready,
        "not_ready_reasons": not_ready,
    }


__all__ = [
    "CLIP_STATUSES",
    "CameraHoldoutValidationError",
    "REVIEW_TABLE_COLUMNS",
    "SCHEMA",
    "SOURCE_INDEX_SCHEMA",
    "build_from_review_index",
    "build_single_reviewer_queue",
    "export_review_table",
    "import_review_table",
    "validate_completed_review",
    "validate_unreviewed_queue",
    "verify_source_files",
]
