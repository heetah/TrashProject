#!/usr/bin/env python3
"""Evaluate full-denominator litter/vehicle readiness without hiding failures.

The reviewed usable clip is the independent unit.  Missing events, NULL routes
and failed processes remain in the denominator.  Positive-only clips cannot
estimate false-positive rate, so this report marks that release gate blocked
instead of manufacturing a precision number.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _ids(value: str) -> list[int]:
    try:
        parsed = ast.literal_eval(value or "[]")
    except (ValueError, SyntaxError):
        return []
    result = []
    for item in parsed if isinstance(parsed, list) else []:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            pass
    return result


def _metric(successes: int, total: int) -> dict:
    low, high = wilson(successes, total)
    return {
        "successes": successes,
        "denominator": total,
        "rate": successes / total if total else None,
        "wilson_95": [low, high] if total else None,
    }


def evaluate(metrics_csv: Path, target: float = 0.85) -> tuple[list[dict], dict]:
    source = list(csv.DictReader(metrics_csv.open(encoding="utf-8")))
    usable = [row for row in source if _bool(row.get("video_usable"))]
    cases = []
    for row in usable:
        case_id = int(row["litter_case"])
        confirmed = int(row.get("confirmed_event_count") or 0)
        predicted = _ids(row.get("predicted_vehicle_ids", "[]"))
        known_match = _bool(row.get("vehicle_match_any"))
        unknown_human_match = (
            row.get("label_status") == "UNKNOWN"
            and _bool(row.get("human_verified_correct"))
            and confirmed > 0
            and bool(predicted)
        )
        # A label-side match flag is never enough for end-to-end correctness:
        # this exact run must have emitted both a confirmed event and a vehicle
        # route.  Keeping those prerequisites here makes malformed/stale joined
        # CSV rows fail closed instead of inflating readiness.
        correct = bool(
            confirmed > 0
            and predicted
            and (known_match or unknown_human_match)
        )
        if confirmed <= 0:
            outcome = "missed_event"
        elif correct:
            outcome = "correct_vehicle"
        elif predicted:
            outcome = "wrong_vehicle"
        else:
            outcome = "unresolved_or_person_only"
        cases.append({
            "litter_case": case_id,
            "confirmed_event_count": confirmed,
            "predicted_vehicle_ids": predicted,
            "gt_vehicle_id": row.get("gt_vehicle_id"),
            "outcome": outcome,
            "end_to_end_correct": correct,
            "excess_confirmed_event_count": max(confirmed - 1, 0),
        })

    total = len(cases)
    detected = sum(row["confirmed_event_count"] > 0 for row in cases)
    correct = sum(row["end_to_end_correct"] for row in cases)
    selected = sum(bool(row["predicted_vehicle_ids"]) for row in cases)
    wrong = sum(row["outcome"] == "wrong_vehicle" for row in cases)
    unresolved = sum(row["outcome"] == "unresolved_or_person_only" for row in cases)
    required_point = math.ceil(target * total)
    required_ci = next(
        (
            count for count in range(total + 1)
            if wilson(count, total)[0] >= target
        ),
        None,
    )
    report = {
        "schema": "product-readiness/v1",
        "independent_unit": "reviewed_usable_clip",
        "target": target,
        "denominator_policy": (
            "all usable clips; missed events, NULL routes, person-only routes and "
            "failed outputs are not removed"
        ),
        "usable_clip_count": total,
        "event_detection_sensitivity": _metric(detected, total),
        "end_to_end_vehicle_correctness": _metric(correct, total),
        "vehicle_correctness_given_confirmed_event": _metric(correct, detected),
        "vehicle_selection_rate": _metric(selected, total),
        "failure_counts": {
            "missed_event": total - detected,
            "wrong_vehicle": wrong,
            "unresolved_or_person_only": unresolved,
            "excess_confirmed_events_unreviewed": sum(
                row["excess_confirmed_event_count"] for row in cases
            ),
        },
        "target_counts": {
            "point_estimate_at_least_target": required_point,
            "two_sided_wilson_95_lower_bound_at_least_target": required_ci,
        },
        "release_gates": {
            "event_detection_point_estimate": detected >= required_point,
            "end_to_end_vehicle_point_estimate": correct >= required_point,
            "event_detection_wilson_lower_bound": wilson(detected, total)[0] >= target,
            "end_to_end_vehicle_wilson_lower_bound": wilson(correct, total)[0] >= target,
            "false_positive_rate": None,
            "false_positive_gate_status": "BLOCKED_NO_REVIEWED_NEGATIVE_CLIPS",
        },
        "interpretation": [
            "Confirmed coverage is sensitivity, not attribution accuracy.",
            "Human verification for an unknown ID counts only when this run emitted a vehicle route.",
            "Excess event records are a duplicate/false-positive warning, not a verified FP count.",
            "A separate reviewed negative-video set is mandatory before production release.",
        ],
    }
    return cases, report


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def write_report(output: Path, cases: list[dict], report: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "case_outcomes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]) if cases else [])
        if cases:
            writer.writeheader()
            writer.writerows(cases)
    event = report["event_detection_sensitivity"]
    e2e = report["end_to_end_vehicle_correctness"]
    conditional = report["vehicle_correctness_given_confirmed_event"]
    failures = report["failure_counts"]
    target_counts = report["target_counts"]
    lines = [
        "# Product readiness evaluation",
        "",
        f"- Independent denominator: all {report['usable_clip_count']} reviewed usable clips",
        f"- Event detection sensitivity: {event['successes']}/{event['denominator']} = {_pct(event['rate'])}",
        f"- End-to-end correct vehicle: {e2e['successes']}/{e2e['denominator']} = {_pct(e2e['rate'])}",
        f"- Correct vehicle given a confirmed event: {conditional['successes']}/{conditional['denominator']} = {_pct(conditional['rate'])}",
        f"- Failure split: {failures['missed_event']} missed events, {failures['wrong_vehicle']} wrong vehicles, {failures['unresolved_or_person_only']} unresolved/person-only",
        "",
        "## 85% gate",
        "",
        f"- Point estimate requires at least {target_counts['point_estimate_at_least_target']}/{report['usable_clip_count']} clips.",
        f"- Wilson 95% lower bound >=85% requires at least {target_counts['two_sided_wilson_95_lower_bound_at_least_target']}/{report['usable_clip_count']} clips.",
        f"- Current point-estimate gate: {'PASS' if report['release_gates']['end_to_end_vehicle_point_estimate'] else 'FAIL'}",
        f"- False-positive gate: {report['release_gates']['false_positive_gate_status']}",
        "",
        "Confirmed coverage is not accuracy. Positive-only clips cannot estimate false-positive rate.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=0.85)
    args = parser.parse_args()
    cases, report = evaluate(args.clip_metrics, args.target)
    write_report(args.output, cases, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
