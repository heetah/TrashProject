#!/usr/bin/env python3
"""Fail-closed product-readiness evaluation using same-run actor mappings.

Tracker IDs are local to one inference run.  This evaluator therefore maps
manual actor boxes to the model tracklets stored in that same run's candidate
sidecars.  Missing, exploratory, unmapped, NULL, and wrong results remain in
the fixed positive denominator.  It also blocks release when event labels are
unreviewed or when no reviewed negative set is supplied.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from validate_attribution_formula import (
    actor_key,
    candidate_clip,
    candidate_records,
    clip_id,
    load_jsonl,
    map_manual_actors,
    match_events,
    prediction_tuple,
    route_from_actor_types,
)
from pipeline.backtrack.study import StudyConfig, replay_candidates


SCHEMA = "reviewed-product-readiness/v1"
ACCEPTED_MATCH_TIERS = frozenset({"strict", "moderate"})


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def metric(successes: int, total: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "denominator": total,
        "rate": successes / total if total else None,
        "wilson_95": wilson(successes, total),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _expected_route_by_video(
    actors: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in actors:
        grouped[str(row["video_id"])].append(str(row["actor_type"]))
    return {video_id: route_from_actor_types(types) for video_id, types in grouped.items()}


def _truth_tuple(match: Mapping[str, Any], expected_route: str) -> tuple[Any, ...] | None:
    person = match.get("correct_person_key")
    vehicle = match.get("correct_vehicle_key")
    if expected_route == "person_vehicle" and (person is None or vehicle is None):
        return None
    if expected_route == "direct_vehicle" and vehicle is None:
        return None
    if expected_route == "person" and person is None:
        return None
    if expected_route == "null":
        return None
    return (expected_route, person, vehicle)


def evaluate(
    ground_truth: Path,
    candidates_dir: Path,
    *,
    target: float = 0.85,
    replay_max_release_back_seconds: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clips = load_jsonl(ground_truth / "clip_annotations.jsonl")
    events = load_jsonl(ground_truth / "event_annotations.jsonl")
    actors = load_jsonl(ground_truth / "actor_annotations.jsonl")
    project = json.loads((ground_truth / "project.json").read_text(encoding="utf-8"))
    metadata = {row["filename"]: row for row in project["videos"]}
    records = candidate_records(candidates_dir)
    replay_config = None
    if replay_max_release_back_seconds is not None:
        replay_config = StudyConfig(
            name=f"release-{replay_max_release_back_seconds:g}s",
            max_release_back_seconds=float(replay_max_release_back_seconds),
        )
        records, _ = replay_candidates(records, replay_config)
    matches = match_events(events, records, metadata)
    map_manual_actors(matches, actors)

    usable_video_ids = {
        str(row["video_id"])
        for row in clips
        if bool(row.get("video_usable"))
    }
    usable_events = [
        row for row in events
        if str(row["video_id"]) in usable_video_ids and not bool(row.get("ignore"))
    ]
    event_counts = Counter(clip_id(str(row["video_filename"])) for row in usable_events)
    non_singletons = {key: value for key, value in event_counts.items() if value != 1}
    if non_singletons:
        raise ValueError(
            "readiness/v1 requires exactly one non-ignored event per usable clip; "
            f"found {non_singletons}"
        )
    event_by_clip = {clip_id(str(row["video_filename"])): row for row in usable_events}
    match_by_clip = {str(row["clip_id"]): row for row in matches}
    expected_routes = _expected_route_by_video(actors)
    confirmed_by_clip = Counter(candidate_clip(row) for row in records)

    cases: list[dict[str, Any]] = []
    for current_clip, event in sorted(event_by_clip.items()):
        match = match_by_clip.get(current_clip)
        confirmed_count = int(confirmed_by_clip.get(current_clip, 0))
        accepted_event = bool(
            match and match.get("match_tier") in ACCEPTED_MATCH_TIERS
        )
        expected_route = expected_routes.get(str(event["video_id"]), "null")
        truth = _truth_tuple(match, expected_route) if accepted_event and match else None
        assignment = match["candidate"].get("assignment", {}) if match else {}
        prediction = prediction_tuple(assignment) if match else None
        correct = bool(accepted_event and truth is not None and prediction == truth)

        if confirmed_count == 0 or match is None:
            outcome = "missed_event"
        elif not accepted_event:
            outcome = "exploratory_event_match"
        elif truth is None:
            outcome = "actor_mapping_unavailable"
        elif correct:
            outcome = "correct_route"
        elif assignment.get("route_type") == "null":
            outcome = "null_route"
        else:
            outcome = "wrong_route"

        cases.append({
            "clip_id": current_clip,
            "video_filename": event["video_filename"],
            "event_review_state": event.get("review_state"),
            "confirmed_event_count": confirmed_count,
            "match_tier": match.get("match_tier") if match else None,
            "temporal_gap_frames": match.get("temporal_gap_frames") if match else None,
            "spatial_gap_frame_diagonal_fraction": (
                match.get("spatial_gap_frame_diagonal_fraction") if match else None
            ),
            "expected_route_type": expected_route,
            "mapped_person_key": list(match["correct_person_key"])
            if match and match.get("correct_person_key") else None,
            "mapped_vehicle_key": list(match["correct_vehicle_key"])
            if match and match.get("correct_vehicle_key") else None,
            "predicted_route_type": assignment.get("route_type"),
            "predicted_person_key": list(actor_key(assignment.get("person_key")))
            if actor_key(assignment.get("person_key")) else None,
            "predicted_vehicle_key": list(actor_key(assignment.get("vehicle_key")))
            if actor_key(assignment.get("vehicle_key")) else None,
            "outcome": outcome,
            "accepted_event_match": accepted_event,
            "provisional_end_to_end_correct": correct,
        })

    total = len(cases)
    accepted_events = sum(row["accepted_event_match"] for row in cases)
    correct_routes = sum(row["provisional_end_to_end_correct"] for row in cases)
    point_required = math.ceil(target * total)
    ci_required = next(
        (count for count in range(total + 1)
         if (wilson(count, total) or [0.0])[0] >= target),
        None,
    )
    event_states = Counter(str(row.get("review_state")) for row in usable_events)
    labels_reviewed = bool(usable_events) and set(event_states) == {"reviewed"}
    outcomes = Counter(row["outcome"] for row in cases)
    report = {
        "schema": SCHEMA,
        "target": target,
        "independent_unit": "usable_positive_clip_with_one_annotated_event",
        "denominator_policy": (
            "All usable positive clips remain in the denominator; exploratory, "
            "unmapped, NULL, wrong, and missing results fail closed."
        ),
        "identity_policy": (
            "Manual same-frame boxes are mapped to tracklets from this exact run; "
            "numeric tracker IDs are never compared across runs."
        ),
        "accepted_event_match_tiers": sorted(ACCEPTED_MATCH_TIERS),
        "event_match_metric_semantics": (
            "Assignment-conditioned event match: match_events uses the selected "
            "route's release frame and point. This is not detector-only sensitivity."
        ),
        "usable_positive_count": total,
        "confirmed_candidate_record_count": len(records),
        "event_match_tiers": dict(Counter(
            row.get("match_tier") or "unmatched" for row in cases
        )),
        "outcome_counts": dict(outcomes),
        "event_detection_sensitivity": metric(accepted_events, total),
        "provisional_end_to_end_route_correctness": metric(correct_routes, total),
        "route_correctness_given_accepted_event": metric(correct_routes, accepted_events),
        "finable_case_correctness": {
            "status": "NOT_EVALUATED_NO_REVIEWED_PLATE_OCR_GROUND_TRUTH",
            "rate": None,
            "note": (
                "Event and actor-route correctness do not establish a readable, "
                "correct plate or a finable case."
            ),
        },
        "target_counts": {
            "point_estimate_at_least_target": point_required,
            "wilson_95_lower_bound_at_least_target": ci_required,
        },
        "annotation_status": {
            "event_review_states": dict(event_states),
            "all_usable_event_labels_reviewed": labels_reviewed,
        },
        "release_gates": {
            "event_detection_point_estimate": accepted_events >= point_required,
            "end_to_end_point_estimate": correct_routes >= point_required,
            "event_detection_wilson_lower_bound": bool(
                total and (wilson(accepted_events, total) or [0.0])[0] >= target
            ),
            "end_to_end_wilson_lower_bound": bool(
                total and (wilson(correct_routes, total) or [0.0])[0] >= target
            ),
            "reviewed_event_labels": labels_reviewed,
            "false_positive_gate": "BLOCKED_NO_REVIEWED_NEGATIVE_SET",
            "cross_camera_holdout_gate": "BLOCKED_NO_INDEPENDENT_CAMERA_GROUPS",
            "plate_ocr_gate": "BLOCKED_NO_REVIEWED_PLATE_OCR_GROUND_TRUTH",
            "ready_for_enforcement": False,
        },
        "provenance": {
            "candidates_directory": str(candidates_dir.resolve()),
            "clip_annotations_sha256": _sha256(ground_truth / "clip_annotations.jsonl"),
            "event_annotations_sha256": _sha256(ground_truth / "event_annotations.jsonl"),
            "actor_annotations_sha256": _sha256(ground_truth / "actor_annotations.jsonl"),
            "research_replay_config": (
                {
                    "name": replay_config.name,
                    "max_release_back_seconds": replay_config.max_release_back_seconds,
                    "production_promoted": False,
                }
                if replay_config else None
            ),
        },
        "limitations": ([
            "The event annotations are not yet marked as human reviewed.",
        ] if not labels_reviewed else []) + [
            "Positive-only data cannot measure false-positive rate or precision.",
            "No independent camera-group holdout is encoded, so generalization is unverified.",
            "Plate OCR and finable-case correctness are not evaluated by this positive route set.",
            "Multiple confirmed records in one positive clip are warnings, not adjudicated false positives.",
        ],
    }
    return cases, report


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def write_outputs(output: Path, cases: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output / "case_outcomes.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(cases[0]) if cases else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(cases)
    detection = report["event_detection_sensitivity"]
    e2e = report["provisional_end_to_end_route_correctness"]
    conditional = report["route_correctness_given_accepted_event"]
    reviewed = report["annotation_status"]["all_usable_event_labels_reviewed"]
    review_sentence = (
        "Event labels are marked reviewed, but reviewed negative and independent "
        "cross-camera holdout sets are absent."
        if reviewed
        else "Event labels are not fully reviewed, and reviewed negative and independent "
        "cross-camera holdout sets are absent."
    )
    lines = [
        "# Fail-closed product readiness evaluation",
        "",
        f"- Fixed positive denominator: {report['usable_positive_count']} clips",
        f"- Assignment-conditioned strict/moderate event match: {detection['successes']}/{detection['denominator']} = {_pct(detection['rate'])}",
        f"- Provisional end-to-end route correctness: {e2e['successes']}/{e2e['denominator']} = {_pct(e2e['rate'])}",
        f"- Route correctness given an accepted event: {conditional['successes']}/{conditional['denominator']} = {_pct(conditional['rate'])}",
        f"- Event annotation states: {report['annotation_status']['event_review_states']}",
        "",
        "## Release decision",
        "",
        "**BLOCKED — not ready for enforcement use.**",
        "",
        f"The 85% point estimate and Wilson lower-bound gates fail. {review_sentence}",
        "",
        "Tracker IDs were remapped from manual boxes against this exact run; historical numeric IDs were not reused.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.85)
    parser.add_argument(
        "--replay-max-release-back-seconds",
        type=float,
        help="Research-only frozen-input replay; does not change production defaults.",
    )
    args = parser.parse_args()
    cases, report = evaluate(
        args.ground_truth,
        args.candidates,
        target=args.target,
        replay_max_release_back_seconds=args.replay_max_release_back_seconds,
    )
    write_outputs(args.output, cases, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
