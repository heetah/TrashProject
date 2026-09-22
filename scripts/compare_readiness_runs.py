#!/usr/bin/env python3
"""Paired comparison of two fail-closed readiness case tables."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from typing import Any


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _load(path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    keyed = {row["clip_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate clip_id in {path}")
    return keyed


def exact_two_sided_sign_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(gains, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def compare(baseline_csv: Path, candidate_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = _load(baseline_csv)
    candidate = _load(candidate_csv)
    if set(baseline) != set(candidate):
        raise ValueError("paired comparison requires identical clip_id sets")
    changes = []
    for clip in sorted(baseline):
        first, second = baseline[clip], candidate[clip]
        first_correct = _bool(first["provisional_end_to_end_correct"])
        second_correct = _bool(second["provisional_end_to_end_correct"])
        first_event = _bool(first["accepted_event_match"])
        second_event = _bool(second["accepted_event_match"])
        if (
            first_correct != second_correct
            or first_event != second_event
            or first.get("outcome") != second.get("outcome")
            or first.get("predicted_route_type") != second.get("predicted_route_type")
            or first.get("predicted_person_key") != second.get("predicted_person_key")
            or first.get("predicted_vehicle_key") != second.get("predicted_vehicle_key")
        ):
            changes.append({
                "clip_id": clip,
                "baseline_outcome": first.get("outcome"),
                "candidate_outcome": second.get("outcome"),
                "baseline_match_tier": first.get("match_tier"),
                "candidate_match_tier": second.get("match_tier"),
                "baseline_route": first.get("predicted_route_type"),
                "candidate_route": second.get("predicted_route_type"),
                "baseline_person_key": first.get("predicted_person_key"),
                "candidate_person_key": second.get("predicted_person_key"),
                "baseline_vehicle_key": first.get("predicted_vehicle_key"),
                "candidate_vehicle_key": second.get("predicted_vehicle_key"),
                "event_match_gain": not first_event and second_event,
                "event_match_loss": first_event and not second_event,
                "correctness_gain": not first_correct and second_correct,
                "correctness_loss": first_correct and not second_correct,
            })
    gains = sum(row["correctness_gain"] for row in changes)
    losses = sum(row["correctness_loss"] for row in changes)
    event_gains = sum(row["event_match_gain"] for row in changes)
    event_losses = sum(row["event_match_loss"] for row in changes)
    unaccepted_reductions = [
        row["clip_id"] for row in changes
        if row["baseline_outcome"] == "exploratory_event_match"
        and row["candidate_outcome"] == "missed_event"
    ]
    unaccepted_increases = [
        row["clip_id"] for row in changes
        if row["baseline_outcome"] == "missed_event"
        and row["candidate_outcome"] == "exploratory_event_match"
    ]
    report = {
        "schema": "readiness-paired-comparison/v1",
        "denominator": len(baseline),
        "baseline_correct": sum(_bool(row["provisional_end_to_end_correct"]) for row in baseline.values()),
        "candidate_correct": sum(_bool(row["provisional_end_to_end_correct"]) for row in candidate.values()),
        "correctness_gains": gains,
        "correctness_losses": losses,
        "correctness_gain_cases": [row["clip_id"] for row in changes if row["correctness_gain"]],
        "correctness_loss_cases": [row["clip_id"] for row in changes if row["correctness_loss"]],
        "paired_exact_two_sided_p": exact_two_sided_sign_p(gains, losses),
        "event_match_gains": event_gains,
        "event_match_losses": event_losses,
        "event_match_gain_cases": [row["clip_id"] for row in changes if row["event_match_gain"]],
        "event_match_loss_cases": [row["clip_id"] for row in changes if row["event_match_loss"]],
        "unaccepted_confirmation_reduction_cases": unaccepted_reductions,
        "unaccepted_confirmation_increase_cases": unaccepted_increases,
        "unaccepted_confirmation_note": (
            "Exploratory-to-missed is a safety proxy, not an adjudicated false-positive reduction."
        ),
        "outcome_transition_counts": dict(Counter(
            f"{row['baseline_outcome']} -> {row['candidate_outcome']}"
            for row in changes
        )),
        "promotion_gate": {
            "no_correctness_regression": losses == 0,
            "no_event_match_regression": event_losses == 0,
            "statistically_significant_improvement": exact_two_sided_sign_p(gains, losses) < 0.05,
            "passes": losses == 0 and event_losses == 0 and gains > 0
            and exact_two_sided_sign_p(gains, losses) < 0.05,
        },
        "safety_change_gate": {
            "no_correctness_regression": losses == 0,
            "no_accepted_event_regression": event_losses == 0,
            "no_unaccepted_confirmation_increase": not unaccepted_increases,
            "has_unaccepted_confirmation_reduction": bool(unaccepted_reductions),
            "passes": (
                losses == 0 and event_losses == 0
                and not unaccepted_increases and bool(unaccepted_reductions)
            ),
            "scope": "development safety proxy; not a product release or false-positive gate",
        },
    }
    return changes, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    changes, report = compare(args.baseline, args.candidate)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output / "paired_changes.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(changes[0]) if changes else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(changes)
    (args.output / "REPORT.md").write_text(
        "\n".join([
            "# Paired readiness comparison",
            "",
            f"- Baseline: {report['baseline_correct']}/{report['denominator']}",
            f"- Candidate: {report['candidate_correct']}/{report['denominator']}",
            f"- Correctness gains/losses: {report['correctness_gains']}/{report['correctness_losses']}",
            f"- Event-match gains/losses: {report['event_match_gains']}/{report['event_match_losses']}",
            f"- Exact paired p-value: {report['paired_exact_two_sided_p']:.6f}",
            f"- Promotion gate: {'PASS' if report['promotion_gate']['passes'] else 'FAIL'}",
            f"- Safety-change gate: {'PASS' if report['safety_change_gate']['passes'] else 'FAIL'}",
            "",
            "This development-set comparison is not an independent holdout result.",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
