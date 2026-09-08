import csv
from pathlib import Path

from scripts.compare_readiness_runs import compare, exact_two_sided_sign_p


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_paired_comparison_counts_gains_and_losses(tmp_path: Path):
    base = [
        {"clip_id": "a", "outcome": "wrong_route", "match_tier": "moderate", "predicted_route_type": "direct_vehicle", "predicted_person_key": "", "predicted_vehicle_key": "['vehicle', 1]", "accepted_event_match": "True", "provisional_end_to_end_correct": "False"},
        {"clip_id": "b", "outcome": "correct_route", "match_tier": "strict", "predicted_route_type": "direct_vehicle", "predicted_person_key": "", "predicted_vehicle_key": "['vehicle', 2]", "accepted_event_match": "True", "provisional_end_to_end_correct": "True"},
    ]
    candidate = [
        {**base[0], "outcome": "correct_route", "provisional_end_to_end_correct": "True"},
        {**base[1], "outcome": "exploratory_event_match", "match_tier": "exploratory", "accepted_event_match": "False", "provisional_end_to_end_correct": "False"},
    ]
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write(first, base)
    _write(second, candidate)

    changes, report = compare(first, second)

    assert len(changes) == 2
    assert report["correctness_gains"] == 1
    assert report["correctness_losses"] == 1
    assert report["event_match_losses"] == 1
    assert report["promotion_gate"]["passes"] is False
    assert report["safety_change_gate"]["passes"] is False
    assert exact_two_sided_sign_p(1, 1) == 1.0


def test_safety_change_gate_accepts_only_no_regression_reduction(tmp_path: Path):
    baseline = [{
        "clip_id": "a", "outcome": "exploratory_event_match",
        "match_tier": "exploratory", "predicted_route_type": "direct_vehicle",
        "predicted_person_key": "", "predicted_vehicle_key": "['vehicle', 1]",
        "accepted_event_match": "False", "provisional_end_to_end_correct": "False",
    }]
    candidate = [{
        **baseline[0], "outcome": "missed_event", "match_tier": "",
        "predicted_route_type": "", "predicted_vehicle_key": "",
    }]
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    _write(first, baseline)
    _write(second, candidate)

    _, report = compare(first, second)

    assert report["promotion_gate"]["passes"] is False
    assert report["safety_change_gate"]["passes"] is True
    assert report["unaccepted_confirmation_reduction_cases"] == ["a"]
