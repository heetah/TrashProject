"""Build model-blind review inputs for release validation.

This module never promotes a machine suggestion, folder name, or legacy
annotation to reviewed ground truth.  It creates immutable manifests and blank
review forms so two reviewers can work independently before adjudication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpeg", ".mpg"}
REVIEW_SCHEMA = "release-validation-review/v1"
NEGATIVE_SCHEMA = "release-validation-negative-candidate/v1"
CAMERA_SCHEMA = "release-validation-camera-suggestion/v1"


class ReleaseValidationError(ValueError):
    """Raised when a release-validation package cannot be built safely."""


@dataclass(frozen=True)
class VideoProbe:
    fps: Optional[float]
    frame_count: Optional[int]
    width: Optional[int]
    height: Optional[int]
    duration_sec: Optional[float]
    fingerprint: Optional[str]
    decodable: bool


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReleaseValidationError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _case_key(filename: str) -> str:
    match = re.search(r"litter_case_(\d+)", filename, flags=re.IGNORECASE)
    if match:
        return f"litter_case_{int(match.group(1))}"
    return Path(filename).stem.removesuffix("_annotated")


def discover_videos(roots: Iterable[Path]) -> List[Path]:
    videos = set()
    for root in roots:
        root = root.expanduser().resolve()
        if root.is_file() and root.suffix.lower() in VIDEO_SUFFIXES:
            videos.add(root)
        elif root.is_dir():
            videos.update(
                item.resolve()
                for item in root.rglob("*")
                if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
            )
    return sorted(videos, key=str)


def map_source_videos(source_root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = {}
    for path in discover_videos([source_root]):
        key = _case_key(path.name)
        if key in mapping:
            duplicates.setdefault(key, [str(mapping[key])]).append(str(path))
        else:
            mapping[key] = path
    if duplicates:
        detail = "; ".join(f"{key}: {values}" for key, values in sorted(duplicates.items()))
        raise ReleaseValidationError(f"ambiguous source videos: {detail}")
    return mapping


def _blank_review() -> Dict[str, Any]:
    return {
        "review_state": "unreviewed",
        "reviewer_id": None,
        "reviewed_at": None,
        "event_label": None,
        "ignore": False,
        "ignore_reason": None,
        "release_interval": {"start_frame": None, "end_frame": None},
        "release_point": {"frame": None, "x": None, "y": None},
        "admissible_routes": [],
        "notes": None,
    }


def _blank_clip_review() -> Dict[str, Any]:
    return {
        "review_state": "unreviewed",
        "reviewer_id": None,
        "reviewed_at": None,
        "reviewed_full_duration": False,
        "video_usable": None,
        "contains_litter": None,
        "unjudgeable_reason": None,
        "notes": None,
    }


def _probe_video(path: Path, with_fingerprint: bool) -> VideoProbe:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return VideoProbe(None, None, None, None, None, None, False)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return VideoProbe(None, None, None, None, None, None, False)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    if with_fingerprint and count > 0:
        for index in sorted({0, count // 2, max(0, count - 1)}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(gray, (17, 8), interpolation=cv2.INTER_AREA))
    capture.release()
    fingerprint = None
    if frames:
        median = np.median(np.stack(frames), axis=0)
        bits = median[:, 1:] > median[:, :-1]
        fingerprint = "".join(f"{byte:02x}" for byte in np.packbits(bits).tolist())
    valid_fps = fps if fps > 0 else None
    duration = count / fps if count > 0 and fps > 0 else None
    return VideoProbe(valid_fps, count or None, width or None, height or None, duration, fingerprint, True)


def hamming_hex(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ReleaseValidationError("fingerprints must have equal length")
    return (int(left, 16) ^ int(right, 16)).bit_count()


def camera_suggestions(
    positive_rows: Sequence[Mapping[str, Any]], threshold: int = 18
) -> List[Dict[str, Any]]:
    """Return review-only connected components from 128-bit visual hashes."""

    ids = [str(row["event_id"]) for row in positive_rows if row.get("fingerprint")]
    fingerprints = {
        str(row["event_id"]): str(row["fingerprint"])
        for row in positive_rows
        if row.get("fingerprint")
    }
    parent = {item: item for item in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    edges: List[Tuple[int, str, str]] = []
    for index, left in enumerate(ids):
        for right in ids[index + 1 :]:
            distance = hamming_hex(fingerprints[left], fingerprints[right])
            if distance <= threshold:
                union(left, right)
                edges.append((distance, left, right))
    groups: Dict[str, List[str]] = {}
    for item in ids:
        groups.setdefault(find(item), []).append(item)
    group_ids = {
        root: f"suggested-camera-{index:03d}"
        for index, root in enumerate(sorted(groups), 1)
    }
    return [
        {
            "schema": CAMERA_SCHEMA,
            "suggested_group_id": group_ids[find(item)],
            "event_id": item,
            "fingerprint": fingerprints[item],
            "status": "unreviewed_machine_suggestion",
            "same_camera": None,
            "reviewer_id": None,
            "method": "three-frame-median-dhash-128",
            "retrieval_threshold_hamming": threshold,
            "linked_neighbors": [
                {"event_id": right if left == item else left, "hamming": distance}
                for distance, left, right in sorted(edges)
                if left == item or right == item
            ],
        }
        for item in ids
    ]


def build_positive_queue(
    event_rows: Sequence[Mapping[str, Any]], source_root: Path
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sources = map_source_videos(source_root)
    queue: List[Dict[str, Any]] = []
    legacy: List[Dict[str, Any]] = []
    for event in sorted(event_rows, key=lambda row: str(row.get("gt_event_id", ""))):
        event_id = str(event.get("gt_event_id") or "")
        filename = str(event.get("video_filename") or "")
        if not event_id or not filename:
            raise ReleaseValidationError("event row lacks gt_event_id or video_filename")
        source = sources.get(_case_key(filename))
        if source is None:
            raise ReleaseValidationError(f"source video not found for {filename}")
        probe = _probe_video(source, with_fingerprint=True)
        digest = sha256_file(source)
        queue.append(
            {
                "schema": REVIEW_SCHEMA,
                "record_type": "positive_event_review",
                "event_id": event_id,
                "video_id": event.get("video_id"),
                "source_video": {
                    "path": str(source),
                    "sha256": digest,
                    "fps": probe.fps,
                    "frame_count": probe.frame_count,
                    "width": probe.width,
                    "height": probe.height,
                    "duration_sec": probe.duration_sec,
                    "decodable": probe.decodable,
                    "visual_fingerprint": probe.fingerprint,
                },
                "blinding": {
                    "model_outputs_included": False,
                    "legacy_seed_included": False,
                    "reviewers_independent": True,
                },
                "reviewer_a": _blank_review(),
                "reviewer_b": _blank_review(),
                "adjudication": {
                    "review_state": "not_started",
                    "adjudicator_id": None,
                    "adjudicated_at": None,
                    "final_label": None,
                    "reason": None,
                },
            }
        )
        legacy.append(
            {
                "event_id": event_id,
                "source_annotation_sha256": hashlib.sha256(
                    json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "provenance": "legacy_unreviewed_seed_for_adjudication_only",
                "annotation": dict(event),
            }
        )
    return queue, legacy


def build_negative_queue(roots: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, path in enumerate(discover_videos(roots), 1):
        probe = _probe_video(path, with_fingerprint=False)
        rows.append(
            {
                "schema": NEGATIVE_SCHEMA,
                "record_type": "negative_video_candidate",
                "candidate_id": f"negative-candidate-{index:05d}",
                "source_video": {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "fps": probe.fps,
                    "frame_count": probe.frame_count,
                    "duration_sec": probe.duration_sec,
                    "decodable": probe.decodable,
                },
                "candidate_status": "unreviewed_negative_candidate",
                "source_folder_is_not_ground_truth": True,
                "reviewer_a": _blank_clip_review(),
                "reviewer_b": _blank_clip_review(),
                "adjudication": {
                    "review_state": "not_started",
                    "adjudicator_id": None,
                    "adjudicated_at": None,
                    "final_contains_litter": None,
                    "reason": None,
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def validate_built_rows(
    positive_rows: Sequence[Mapping[str, Any]],
    negative_rows: Sequence[Mapping[str, Any]],
    camera_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Enforce the package's fail-closed, model-blind invariants."""

    errors: List[str] = []
    event_ids = [str(row.get("event_id")) for row in positive_rows]
    if len(event_ids) != len(set(event_ids)):
        errors.append("positive event_id values are not unique")
    for index, row in enumerate(positive_rows, 1):
        if row.get("schema") != REVIEW_SCHEMA:
            errors.append(f"positive row {index} has the wrong schema")
        if row.get("blinding", {}).get("model_outputs_included") is not False:
            errors.append(f"positive row {index} is not model-blind")
        for reviewer in ("reviewer_a", "reviewer_b"):
            review = row.get(reviewer, {})
            if (
                review.get("review_state") != "unreviewed"
                or review.get("event_label") is not None
                or review.get("admissible_routes") != []
            ):
                errors.append(f"positive row {index} {reviewer} is not blank")
    for index, row in enumerate(negative_rows, 1):
        if row.get("candidate_status") != "unreviewed_negative_candidate":
            errors.append(f"negative row {index} was promoted without review")
        for reviewer in ("reviewer_a", "reviewer_b"):
            if row.get(reviewer, {}).get("contains_litter") is not None:
                errors.append(f"negative row {index} {reviewer} contains a truth label")
        if row.get("adjudication", {}).get("final_contains_litter") is not None:
            errors.append(f"negative row {index} contains an adjudicated truth label")
    for index, row in enumerate(camera_rows, 1):
        if (
            row.get("status") != "unreviewed_machine_suggestion"
            or row.get("same_camera") is not None
        ):
            errors.append(f"camera row {index} was promoted without review")
    if errors:
        raise ReleaseValidationError("; ".join(errors))
    return {
        "valid": True,
        "positive_events": len(positive_rows),
        "negative_candidates": len(negative_rows),
        "camera_suggestions": len(camera_rows),
        "positive_decodable": sum(
            row.get("source_video", {}).get("decodable") is True for row in positive_rows
        ),
        "negative_decodable": sum(
            row.get("source_video", {}).get("decodable") is True for row in negative_rows
        ),
    }
