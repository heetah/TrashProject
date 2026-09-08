#!/usr/bin/env python3
"""Legacy join of user-supplied run-local IDs with attribution sidecars.

This is a research/reporting utility.  It deliberately reports both clip-level
and event-level rows: multiple confirmed tracks in one clip are not independent
observations and must not be silently counted as extra samples.

Numeric tracker IDs are not stable across inference runs.  Do not use this
utility for a different run than the one from which ``VEHICLE_GT`` was
transcribed.  Use ``evaluate_reviewed_readiness.py`` for same-run box-to-track
mapping and product-readiness decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable


# IDs transcribed from the user's reviewed table on 2026-08-26.  ``None`` is
# an omitted value, while ``?`` is explicitly unknown.  Case 41 accepts either
# of the two observed vehicle identities.
VEHICLE_GT: dict[int, int | str | None] = {
    7: 2, 9: 3, 12: 4, 13: 1, 14: 4, 15: 2, 16: 2, 17: 2,
    18: 1, 20: 4, 21: 1, 23: 2, 25: 2, 26: 7, 29: 1, 30: 2,
    34: 1, 35: 2, 36: 2, 41: "2|1", 42: 1, 49: 1, 50: 13,
    51: 3, 60: 1, 66: 1, 67: 2, 68: 1, 69: 1, 74: "?", 75: 1,
    76: 1, 77: 1, 78: 1, 79: 1, 87: 1, 89: 1, 92: None, 99: 1,
    102: 2, 106: 1, 111: 1, 116: 2, 135: 20, 138: 2, 141: 2,
    143: 11, 145: 7, 149: 2, 152: 1, 156: 1, 159: 4, 161: 4,
    164: 3, 165: 1, 167: 3, 168: 1, 173: 2, 174: "?", 178: 1,
    193: 9, 194: 1,
}

PERSON_GT: dict[int, int] = {92: 2, 141: 3, 156: 1, 161: 1}

# The reviewed table leaves the vehicle ID unknown for these two usable clips,
# but the user has explicitly adjudicated both routes as correct by eye.  Keep
# this separate from VEHICLE_GT: an unknown ID must never be converted into a
# fabricated numeric identity.  The override is used only for the explicitly
# reported, adjudicated route metric.
HUMAN_VERIFIED_CORRECT: frozenset[int] = frozenset({74, 174})


def _case_id(filename: str | Path) -> int:
    match = re.search(r"litter_case_(\d+)", Path(filename).name)
    if not match:
        raise ValueError(f"Cannot extract litter_case ID from {filename}")
    return int(match.group(1))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _actor_id(value: Any) -> int | None:
    if isinstance(value, list) and len(value) >= 2:
        try:
            return int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _allowed_ids(value: int | str | None) -> set[int] | None:
    if isinstance(value, int):
        return {value}
    if isinstance(value, str) and value not in ("", "?"):
        try:
            return {int(part) for part in value.split("|")}
        except ValueError:
            return None
    return None


def _json_number(value: Any) -> float | int | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _distance(a: Any, b: Any) -> float | None:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) < 2 or len(b) < 2:
        return None
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError):
        return None


def _component_fields(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    backtrack = record.get("event", {}).get("backtrack", {})
    costs = ((backtrack.get("components") or {}).get("costs") or {}).get("BC") or {}
    return costs.get("raw_features") or {}, costs.get("components") or {}


def _sidecar_events(sidecar_dir: Path) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    # Production batch runs isolate each clip in ``case_<id>/`` while older
    # research replays wrote every sidecar directly under one directory.
    # Support both layouts so reporting cannot silently turn a completed batch
    # into zero detected events merely because its storage topology changed.
    for path in sorted(sidecar_dir.rglob("litter_case_*_annotated_backtrack_candidates.jsonl")):
        cid = _case_id(path)
        records = _read_jsonl(path)
        result.setdefault(cid, []).extend(
            record for record in records if record.get("record_type") == "candidate"
        )
    return result


def _confirmed_count(sidecar_dir: Path, cid: int) -> int:
    paths = sorted(sidecar_dir.rglob(f"litter_case_{cid}_annotated_analysis.json"))
    if not paths:
        return 0
    return sum(
        int(
            (
                json.loads(path.read_text(encoding="utf-8")).get("litter_detection")
                or {}
            ).get("confirmed_event_count")
            or 0
        )
        for path in paths
    )


def _event_row(cid: int, index: int, record: dict[str, Any], gt_event: dict[str, Any], unused: bool) -> dict[str, Any]:
    assignment = record.get("assignment") or {}
    event = record.get("event") or {}
    resolver = record.get("resolver_input") or {}
    route = event.get("backtrack") or {}
    pred_vehicle = _actor_id(assignment.get("vehicle_key"))
    pred_person = _actor_id(assignment.get("person_key"))
    gt_vehicle = VEHICLE_GT.get(cid)
    gt_person = PERSON_GT.get(cid)
    allowed_vehicle = _allowed_ids(gt_vehicle)
    vehicle_match = None if unused or allowed_vehicle is None else pred_vehicle in allowed_vehicle
    person_match = None if unused or gt_person is None else pred_person == gt_person
    human_verified_correct = (cid in HUMAN_VERIFIED_CORRECT) and not unused
    if human_verified_correct:
        # This is a manual adjudication, not an inferred vehicle-ID match.
        # Preserve the unknown ID and expose the override explicitly.
        route_match = True
    elif unused or (allowed_vehicle is None and gt_person is None):
        route_match = None
    elif gt_person is not None and allowed_vehicle is not None:
        route_match = pred_person == gt_person and pred_vehicle in allowed_vehicle
    elif allowed_vehicle is not None:
        route_match = pred_person is None and pred_vehicle in allowed_vehicle
    else:
        route_match = person_match

    release_point = assignment.get("release_point")
    birth_point = resolver.get("birth_centroid")
    release_frame = assignment.get("release_frame")
    birth_frame = resolver.get("birth_frame")
    try:
        delta_frames = int(birth_frame) - int(release_frame)
    except (TypeError, ValueError):
        delta_frames = None
    fps = float(resolver.get("fps") or record.get("video", {}).get("fps") or 10.0)
    distance_px = _distance(release_point, birth_point)
    actor_diagonal = gt_event.get("vehicle_diagonal") if allowed_vehicle is not None else gt_event.get("person_diagonal")
    try:
        distance_norm = distance_px / float(actor_diagonal) if distance_px is not None and actor_diagonal else None
    except (TypeError, ValueError, ZeroDivisionError):
        distance_norm = None
    raw, weighted = _component_fields(record)
    actor_margins = assignment.get("actor_margins") or {}
    selected_class = "person" if pred_person is not None and pred_vehicle is None else "vehicle"
    selected_margin = (actor_margins.get(selected_class) or {}).get("margin")
    null_margin = (actor_margins.get("null") or {}).get("margin")
    return {
        "litter_case": cid,
        "event_index": index,
        "unused": unused,
        "route_type": assignment.get("route_type"),
        "route_id": assignment.get("route_id"),
        "predicted_vehicle_id": pred_vehicle,
        "predicted_person_id": pred_person,
        "gt_vehicle_id": gt_vehicle,
        "gt_person_id": gt_person,
        "human_verified_correct": human_verified_correct,
        "vehicle_match": vehicle_match,
        "person_match": person_match,
        "route_match": route_match,
        "cost": _json_number(assignment.get("cost")),
        "margin_to_second": _json_number(assignment.get("margin_to_second")),
        "same_class_margin": _json_number(selected_margin),
        "null_margin": _json_number(null_margin),
        "release_frame": release_frame,
        "birth_frame": birth_frame,
        "release_to_birth_delta_frames": delta_frames,
        "release_to_birth_delta_sec": (delta_frames / fps if delta_frames is not None else None),
        "release_point_x": release_point[0] if isinstance(release_point, (list, tuple)) and len(release_point) >= 2 else None,
        "release_point_y": release_point[1] if isinstance(release_point, (list, tuple)) and len(release_point) >= 2 else None,
        "birth_point_x": birth_point[0] if isinstance(birth_point, (list, tuple)) and len(birth_point) >= 2 else None,
        "birth_point_y": birth_point[1] if isinstance(birth_point, (list, tuple)) and len(birth_point) >= 2 else None,
        "release_birth_distance_px": distance_px,
        "release_birth_distance_over_actor_diagonal": distance_norm,
        "raw_distance_feature": _json_number(raw.get("direct_distance")),
        "raw_time_feature": _json_number(raw.get("time")),
        "raw_direction_feature": _json_number(raw.get("reverse_direction")),
        "weighted_distance_feature": _json_number(weighted.get("direct_distance")),
        "weighted_time_feature": _json_number(weighted.get("time")),
        "weighted_direction_feature": _json_number(weighted.get("reverse_direction")),
        "confirmed_litter_id": event.get("litter_id"),
        "confirm_frame": resolver.get("confirm_frame", event.get("confirm_frame")),
    }


def _summary_stats(rows: Iterable[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    if not values:
        return {"n": 0, "median": None, "mean": None, "min": None, "max": None}
    return {
        "n": len(values),
        "median": median(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def build_report(sidecar_dir: Path, ground_truth_path: Path, clip_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gt_events = {_case_id(row["video_filename"]): row for row in _read_jsonl(ground_truth_path)}
    clips = {_case_id(row["video_filename"]): row for row in _read_jsonl(clip_path)}
    unused = {cid for cid, row in clips.items() if not row.get("video_usable")}
    sidecars = _sidecar_events(sidecar_dir)
    event_rows: list[dict[str, Any]] = []
    clip_rows: list[dict[str, Any]] = []
    # Include every clip in the grounding-truth set as well as every row in
    # the user table.  Case 24 is intentionally absent from that table but is
    # still emitted and marked UNUSED/MISSING rather than silently dropped.
    all_cases = sorted(set(VEHICLE_GT) | set(clips) | set(gt_events))
    for cid in all_cases:
        gt_event = gt_events.get(cid, {})
        records = sidecars.get(cid, [])
        rows = [_event_row(cid, index, record, gt_event, cid in unused) for index, record in enumerate(records)]
        event_rows.extend(rows)
        gt_vehicle = VEHICLE_GT.get(cid)
        gt_person = PERSON_GT.get(cid)
        allowed_vehicle = _allowed_ids(gt_vehicle)
        known_label = allowed_vehicle is not None or gt_person is not None
        usable = cid not in unused
        route_matches = [row["route_match"] for row in rows if row["route_match"] is not None]
        vehicle_matches = [row["vehicle_match"] for row in rows if row["vehicle_match"] is not None]
        person_matches = [row["person_match"] for row in rows if row["person_match"] is not None]
        human_verified_correct = (cid in HUMAN_VERIFIED_CORRECT) and usable
        clip_rows.append({
            "litter_case": cid,
            "video_usable": usable,
            "label_status": "UNUSED" if not usable else ("KNOWN" if known_label else ("UNKNOWN" if gt_vehicle == "?" else "MISSING")),
            "gt_vehicle_id": gt_vehicle,
            "gt_person_id": gt_person,
            "human_verified_correct": human_verified_correct,
            "confirmed_event_count": _confirmed_count(sidecar_dir, cid),
            "predicted_vehicle_ids": sorted({row["predicted_vehicle_id"] for row in rows if row["predicted_vehicle_id"] is not None}),
            "predicted_person_ids": sorted({row["predicted_person_id"] for row in rows if row["predicted_person_id"] is not None}),
            "route_prediction_count": len(rows),
            "vehicle_match_any": any(vehicle_matches) if vehicle_matches else (False if known_label and allowed_vehicle is not None and usable else None),
            "person_match_any": any(person_matches) if person_matches else (False if gt_person is not None and usable else None),
            "route_match_any": (True if human_verified_correct else (any(route_matches) if route_matches else (False if known_label and usable else None))),
            "margin_median": median([row["margin_to_second"] for row in rows if isinstance(row.get("margin_to_second"), (int, float))]) if rows else None,
            "release_birth_distance_median_px": median([row["release_birth_distance_px"] for row in rows if isinstance(row.get("release_birth_distance_px"), (int, float))]) if rows else None,
            "release_birth_delta_median_frames": median([row["release_to_birth_delta_frames"] for row in rows if isinstance(row.get("release_to_birth_delta_frames"), (int, float))]) if rows else None,
        })

    usable_rows = [row for row in clip_rows if row["video_usable"]]
    known_rows = [row for row in usable_rows if row["label_status"] == "KNOWN"]
    adjudicated_rows = [
        row for row in usable_rows
        if row["label_status"] == "KNOWN" or row["human_verified_correct"]
    ]
    vehicle_rows = [row for row in known_rows if _allowed_ids(row["gt_vehicle_id"]) is not None]
    person_rows = [row for row in known_rows if row["gt_person_id"] is not None]
    confirmed_usable = sum(row["confirmed_event_count"] > 0 for row in usable_rows)
    known_event_rows = [row for row in event_rows if row["route_match"] is not None]
    summary = {
        "schema": "actor-ground-truth-metrics/v1",
        "validity": {
            "status": "LEGACY_RUN_LOCAL_ID_COMPARISON",
            "product_readiness_eligible": False,
            "warning": (
                "Numeric tracker IDs are run-local. This report is invalid for "
                "cross-run accuracy unless the ID table came from this exact run."
            ),
            "replacement": "scripts/evaluate_reviewed_readiness.py",
        },
        "sidecar_directory": str(sidecar_dir),
        "unused_cases": sorted(unused),
        "clip_count": len(clip_rows),
        "usable_clip_count": len(usable_rows),
        "confirmed_usable_clip_count": confirmed_usable,
        "confirmed_usable_clip_rate": confirmed_usable / len(usable_rows) if usable_rows else None,
        "known_actor_clip_count": len(known_rows),
        "vehicle_accuracy": {
            "correct_clip_count": sum(bool(row["vehicle_match_any"]) for row in vehicle_rows),
            "denominator": len(vehicle_rows),
        },
        "person_accuracy": {
            "correct_clip_count": sum(bool(row["person_match_any"]) for row in person_rows),
            "denominator": len(person_rows),
        },
        "strict_route_accuracy": {
            "correct_clip_count": sum(bool(row["route_match_any"]) for row in known_rows),
            "denominator": len(known_rows),
        },
        "adjudicated_route_accuracy": {
            "correct_clip_count": sum(bool(row["route_match_any"]) for row in adjudicated_rows),
            "denominator": len(adjudicated_rows),
            "human_verified_correct_cases": sorted(HUMAN_VERIFIED_CORRECT),
            "note": "Adds explicit visual adjudications for unknown vehicle IDs; this does not establish a numeric vehicle-ID match or detector confirmation.",
        },
        "event_level_route_accuracy": {
            "correct_event_count": sum(bool(row["route_match"]) for row in known_event_rows),
            "denominator": len(known_event_rows),
        },
        "event_row_count": len(event_rows),
        "route_match_event_count": sum(row["route_match"] is True for row in event_rows),
        "margin": {
            "all": _summary_stats(event_rows, "margin_to_second"),
            "route_match": _summary_stats([row for row in event_rows if row["route_match"] is True], "margin_to_second"),
            "route_miss": _summary_stats([row for row in event_rows if row["route_match"] is False], "margin_to_second"),
        },
        "release_birth_distance_px": {
            "all": _summary_stats(event_rows, "release_birth_distance_px"),
            "route_match": _summary_stats([row for row in event_rows if row["route_match"] is True], "release_birth_distance_px"),
            "route_miss": _summary_stats([row for row in event_rows if row["route_match"] is False], "release_birth_distance_px"),
        },
        "release_birth_delta_frames": {
            "all": _summary_stats(event_rows, "release_to_birth_delta_frames"),
            "route_match": _summary_stats([row for row in event_rows if row["route_match"] is True], "release_to_birth_delta_frames"),
            "route_miss": _summary_stats([row for row in event_rows if row["route_match"] is False], "release_to_birth_delta_frames"),
        },
        "feature_summary": {
            name: _summary_stats(event_rows, name)
            for name in (
                "raw_distance_feature", "raw_time_feature", "raw_direction_feature",
                "weighted_distance_feature", "weighted_time_feature", "weighted_direction_feature",
            )
        },
        "interpretation": [
            "Clip rows are the independent unit; event rows are dependent repeated tracks.",
            "UNUSED clips are excluded from accuracy denominators.",
            "A route_match is based on the user-supplied actor IDs, not on detector confidence.",
            "Unknown '?' actor IDs are reported but excluded from accuracy denominators.",
            "Cases 74 and 174 are included only in adjudicated_route_accuracy because the user marked them visually correct while leaving vehicle ID unknown.",
        ],
    }
    return event_rows, clip_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, default=Path("runs/grounding_truth/event_annotations.jsonl"))
    parser.add_argument("--clip-annotations", type=Path, default=Path("runs/grounding_truth/clip_annotations.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--acknowledge-run-local-ids",
        action="store_true",
        help=(
            "Required acknowledgement that VEHICLE_GT IDs came from the exact "
            "same tracker run. This legacy output is never release-eligible."
        ),
    )
    args = parser.parse_args()
    if not args.acknowledge_run_local_ids:
        parser.error(
            "numeric tracker IDs are run-local; use evaluate_reviewed_readiness.py "
            "or pass --acknowledge-run-local-ids for an exact historical replay"
        )
    event_rows, clip_rows, summary = build_report(args.sidecar_dir, args.ground_truth, args.clip_annotations)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "actor_event_metrics.csv", event_rows)
    _write_csv(args.output / "actor_clip_metrics.csv", clip_rows)
    (args.output / "actor_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
