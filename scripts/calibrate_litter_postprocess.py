#!/usr/bin/env python3
"""Match reviewed litter boxes to RT-DETR gate traces and tracker IDs.

The report is intentionally separate from ``event_annotations.jsonl``. It
produces ID proposals and gate diagnostics without silently changing labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _bbox_xyxy(value) -> list[float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return [float(value[key]) for key in ("x1", "y1", "x2", "y2")]
    except (KeyError, TypeError, ValueError):
        return None


def bbox_iou(first, second) -> float:
    ax1, ay1, ax2, ay2 = map(float, first)
    bx1, by1, bx2, by2 = map(float, second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = (
        max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _center(box) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)


def _shape_compatible(first, second, tolerance=0.60) -> bool:
    aw = max(float(first[2]) - float(first[0]), 1.0)
    ah = max(float(first[3]) - float(first[1]), 1.0)
    bw = max(float(second[2]) - float(second[0]), 1.0)
    bh = max(float(second[3]) - float(second[1]), 1.0)
    return abs(bw - aw) / aw <= tolerance and abs(bh - ah) / ah <= tolerance


def match_event(event: dict, records: list[dict], *, min_iou=0.25,
                continuation_frames=10, continuation_distance=250.0) -> dict:
    gt_box = _bbox_xyxy(event.get("litter_bbox"))
    frame_value = event.get("litter_bbox_frame")
    if gt_box is None or frame_value is None:
        return {
            "match_status": "detector_miss_sentinel",
            "matched_iou": None,
            "first_gate_reason": "no_rtdetr_detection",
            "correct_litter_id": None,
            "id_match_method": None,
            "tracker_confirmed": False,
            "confirmed_track_ids": [],
            "unverified_confirmed_track_ids": [],
        }

    frame_index = int(frame_value)
    candidates = [
        record for record in records
        if record.get("record_type") == "litter_candidate"
        and int(record.get("frame_index", -1)) == frame_index
    ]
    confirmed_track_ids = sorted({
        int(record["tracker_litter_id"])
        for record in records
        if record.get("tracker_litter_id") is not None
        and record.get("tracker_state") == "confirmed"
    })
    if not candidates:
        return {
            "match_status": "no_candidate_at_gt_frame",
            "matched_iou": None,
            "first_gate_reason": "no_rtdetr_detection",
            "correct_litter_id": None,
            "id_match_method": None,
            "tracker_confirmed": False,
            "confirmed_track_ids": confirmed_track_ids,
            "unverified_confirmed_track_ids": confirmed_track_ids,
        }

    ranked = sorted(
        ((bbox_iou(gt_box, record["bbox"]), record) for record in candidates),
        key=lambda item: item[0], reverse=True,
    )
    best_iou, best = ranked[0]
    if best_iou < min_iou:
        return {
            "match_status": "bbox_iou_below_threshold",
            "matched_iou": round(best_iou, 6),
            "first_gate_reason": None,
            "correct_litter_id": None,
            "id_match_method": None,
            "tracker_confirmed": False,
            "confirmed_track_ids": confirmed_track_ids,
            "unverified_confirmed_track_ids": confirmed_track_ids,
        }

    litter_id = best.get("tracker_litter_id")
    method = "exact_frame_tracker_assignment" if litter_id is not None else None
    ambiguous = False
    if litter_id is None:
        current_box = list(best["bbox"])
        for next_frame in range(frame_index + 1, frame_index + continuation_frames + 1):
            next_records = [
                record for record in records
                if record.get("record_type") == "litter_candidate"
                and int(record.get("frame_index", -1)) == next_frame
                and _shape_compatible(current_box, record["bbox"])
            ]
            if not next_records:
                continue
            cx, cy = _center(current_box)
            distances = sorted(
                [
                    (math.hypot(_center(record["bbox"])[0] - cx,
                                _center(record["bbox"])[1] - cy), record)
                    for record in next_records
                ],
                key=lambda item: item[0],
            )
            distance_value, continuation = distances[0]
            if distance_value > continuation_distance:
                continue
            if len(distances) > 1 and distances[1][0] <= max(
                distance_value * 1.25, distance_value + 10.0
            ):
                ambiguous = True
                break
            current_box = list(continuation["bbox"])
            if continuation.get("tracker_litter_id") is not None:
                litter_id = int(continuation["tracker_litter_id"])
                method = "forward_continuation"
                break

    tracker_confirmed = bool(
        litter_id is not None and litter_id in confirmed_track_ids and not ambiguous
    )
    return {
        "match_status": "ambiguous_continuation" if ambiguous else "matched",
        "matched_iou": round(best_iou, 6),
        "matched_candidate_bbox": list(best["bbox"]),
        "matched_candidate_confidence": best.get("confidence"),
        "first_gate_outcome": best.get("filter_outcome"),
        "first_gate_reason": best.get("filter_reason"),
        "correct_litter_id": None if ambiguous or litter_id is None else int(litter_id),
        "id_match_method": None if ambiguous else method,
        "tracker_confirmed": tracker_confirmed,
        "confirmed_track_ids": confirmed_track_ids,
        "unverified_confirmed_track_ids": [
            track_id for track_id in confirmed_track_ids if track_id != litter_id
        ],
    }


def _wilson_interval(successes: int, trials: int, z=1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    proportion = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (proportion + z2 / (2 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / trials + z2 / (4 * trials * trials)
    ) / denominator
    return [
        round(max(0.0, center - radius), 6),
        round(min(1.0, center + radius), 6),
    ]


def build_report(events: list[dict], sidecar_dir: Path, **match_options) -> dict:
    results = []
    for event in events:
        video_name = str(event.get("video_filename", ""))
        sidecar = sidecar_dir / f"{Path(video_name).stem}_litter_candidates.jsonl"
        records = _read_jsonl(sidecar) if sidecar.exists() else []
        matched = match_event(event, records, **match_options)
        results.append({
            "gt_event_id": event.get("gt_event_id"),
            "video_filename": video_name,
            "litter_bbox_frame": event.get("litter_bbox_frame"),
            "review_state": event.get("review_state"),
            **matched,
        })

    gate_counts = Counter(
        result.get("first_gate_reason") or "not_applicable" for result in results
    )
    status_counts = Counter(result["match_status"] for result in results)
    review_state_counts = Counter(
        str(result.get("review_state") or "missing") for result in results
    )
    confirmed_correct = sum(result["tracker_confirmed"] for result in results)
    unverified_count = sum(
        len(result["unverified_confirmed_track_ids"]) for result in results
    )
    return {
        "schema": "litter-postprocess-calibration/v1",
        "independent_event_count": len(results),
        "matched_tracker_id_count": sum(
            result["correct_litter_id"] is not None for result in results
        ),
        "confirmed_correct_track_count": confirmed_correct,
        "confirmed_correct_rate": (
            confirmed_correct / len(results) if results else None
        ),
        "confirmed_correct_rate_wilson_95": _wilson_interval(
            confirmed_correct, len(results)
        ),
        "unverified_confirmed_track_count": unverified_count,
        "match_status_counts": dict(sorted(status_counts.items())),
        "first_gate_reason_counts": dict(sorted(gate_counts.items())),
        "review_state_counts": dict(sorted(review_state_counts.items())),
        "non_reviewed_event_count": sum(
            count for state, count in review_state_counts.items()
            if state != "reviewed"
        ),
        "events": results,
    }


def summarize_clip_outputs(clip_annotations: list[dict], sidecar_dir: Path) -> dict:
    """Summarize confirmed tracks at clip level without treating candidates as samples."""
    rows = []
    for clip in clip_annotations:
        video_name = str(clip.get("video_filename", ""))
        sidecar = sidecar_dir / f"{Path(video_name).stem}_litter_candidates.jsonl"
        records = _read_jsonl(sidecar) if sidecar.exists() else []
        confirmed_ids = sorted({
            int(record["tracker_litter_id"])
            for record in records
            if record.get("record_type") == "litter_candidate"
            and record.get("tracker_litter_id") is not None
            and record.get("tracker_state") == "confirmed"
        })
        rows.append({
            "video_filename": video_name,
            "video_usable": bool(clip.get("video_usable")),
            "confirmed": bool(confirmed_ids),
            "confirmed_track_ids": confirmed_ids,
        })
    usable = [row for row in rows if row["video_usable"]]
    unusable = [row for row in rows if not row["video_usable"]]
    return {
        "clip_count": len(rows),
        "usable_clip_count": len(usable),
        "unusable_clip_count": len(unusable),
        "confirmed_clip_count": sum(row["confirmed"] for row in rows),
        "confirmed_usable_clip_count": sum(row["confirmed"] for row in usable),
        "confirmed_unusable_clip_count": sum(row["confirmed"] for row in unusable),
        "confirmed_clip_rate": (
            sum(row["confirmed"] for row in rows) / len(rows) if rows else None
        ),
        "confirmed_clip_rate_wilson_95": _wilson_interval(
            sum(row["confirmed"] for row in rows), len(rows)
        ),
    }


def compare_reports(baseline: dict, candidate: dict, *, bootstrap=10000, seed=20260826) -> dict:
    baseline_by_id = {event["gt_event_id"]: event for event in baseline["events"]}
    candidate_by_id = {event["gt_event_id"]: event for event in candidate["events"]}
    differences = []
    for event_id in sorted(set(baseline_by_id) & set(candidate_by_id)):
        differences.append(
            int(bool(candidate_by_id[event_id]["tracker_confirmed"]))
            - int(bool(baseline_by_id[event_id]["tracker_confirmed"]))
        )
    gains = sum(value == 1 for value in differences)
    losses = sum(value == -1 for value in differences)
    rng = random.Random(seed)
    bootstrap_values = []
    for _ in range(max(int(bootstrap), 1)):
        if differences:
            bootstrap_values.append(
                sum(rng.choice(differences) for _ in differences) / len(differences)
            )
    paired_ci = None
    if bootstrap_values:
        bootstrap_values.sort()
        lower = bootstrap_values[int(0.025 * (len(bootstrap_values) - 1))]
        upper = bootstrap_values[int(0.975 * (len(bootstrap_values) - 1))]
        paired_ci = [round(lower, 6), round(upper, 6)]
    discordant = gains + losses
    if discordant:
        tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
        mcnemar_p = min(1.0, 2.0 * tail / (2 ** discordant))
    else:
        mcnemar_p = 1.0
    baseline_unverified = int(baseline["unverified_confirmed_track_count"])
    candidate_unverified = int(candidate["unverified_confirmed_track_count"])
    comparison = {
        "paired_event_count": len(differences),
        "confirmed_correct_gain_count": gains,
        "confirmed_correct_loss_count": losses,
        "confirmed_correct_rate_difference": (
            round(sum(differences) / len(differences), 6) if differences else None
        ),
        "paired_bootstrap_95": paired_ci,
        "mcnemar_exact_two_sided_p": round(mcnemar_p, 6),
        "baseline_unverified_confirmed_track_count": baseline_unverified,
        "candidate_unverified_confirmed_track_count": candidate_unverified,
        "unverified_confirmed_track_delta": candidate_unverified - baseline_unverified,
        "unverified_confirmed_tracks_nonincreasing": candidate_unverified <= baseline_unverified,
        "false_positive_caveat": (
            "All usable clips contain true litter. Unverified extra confirmed tracks are a "
            "safety proxy, not a measured false-positive rate; negative clips are required."
        ),
    }
    if "clip_summary" in baseline and "clip_summary" in candidate:
        comparison["baseline_clip_summary"] = baseline["clip_summary"]
        comparison["candidate_clip_summary"] = candidate["clip_summary"]
        comparison["confirmed_clip_delta"] = (
            candidate["clip_summary"]["confirmed_clip_count"]
            - baseline["clip_summary"]["confirmed_clip_count"]
        )
    return comparison


def _write_outputs(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = sorted({key for event in report["events"] for key in event})
    with (output_dir / "calibration_events.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["events"])
    with (output_dir / "ground_truth_litter_id_proposals.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for event in report["events"]:
            proposal = {
                key: event.get(key) for key in (
                    "gt_event_id", "video_filename", "correct_litter_id",
                    "id_match_method", "match_status", "matched_iou",
                )
            }
            handle.write(json.dumps(proposal, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path,
                        default=Path("runs/grounding_truth/event_annotations.jsonl"))
    parser.add_argument("--clip-annotations", type=Path,
                        default=Path("runs/grounding_truth/clip_annotations.jsonl"))
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--baseline-sidecar-dir", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/litter_postprocess_calibration"))
    parser.add_argument("--min-iou", type=float, default=0.25)
    args = parser.parse_args()
    events = _read_jsonl(args.ground_truth)
    report = build_report(events, args.sidecar_dir, min_iou=args.min_iou)
    if args.clip_annotations.exists():
        report["clip_summary"] = summarize_clip_outputs(
            _read_jsonl(args.clip_annotations), args.sidecar_dir
        )
    if args.baseline_sidecar_dir is not None:
        baseline_report = build_report(
            events, args.baseline_sidecar_dir, min_iou=args.min_iou
        )
        if args.clip_annotations.exists():
            baseline_report["clip_summary"] = summarize_clip_outputs(
                _read_jsonl(args.clip_annotations), args.baseline_sidecar_dir
            )
        report["comparison_to_baseline"] = compare_reports(baseline_report, report)
    _write_outputs(report, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "events"},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
