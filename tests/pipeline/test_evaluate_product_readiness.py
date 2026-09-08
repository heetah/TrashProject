import csv
from pathlib import Path

from scripts.evaluate_product_readiness import evaluate, wilson


def test_wilson_target_count_for_58_clips():
    assert wilson(55, 58)[0] >= 0.85
    assert wilson(54, 58)[0] < 0.85


def test_full_denominator_keeps_misses_and_requires_emitted_unknown_route(tmp_path: Path):
    rows = [
        {
            "litter_case": "1", "video_usable": "True", "label_status": "KNOWN",
            "confirmed_event_count": "1", "predicted_vehicle_ids": "[2]",
            "gt_vehicle_id": "2", "vehicle_match_any": "True",
            "human_verified_correct": "False",
        },
        {
            "litter_case": "2", "video_usable": "True", "label_status": "UNKNOWN",
            "confirmed_event_count": "0", "predicted_vehicle_ids": "[]",
            "gt_vehicle_id": "?", "vehicle_match_any": "",
            "human_verified_correct": "True",
        },
        {
            "litter_case": "3", "video_usable": "False", "label_status": "UNUSED",
            "confirmed_event_count": "0", "predicted_vehicle_ids": "[]",
            "gt_vehicle_id": "3", "vehicle_match_any": "False",
            "human_verified_correct": "False",
        },
    ]
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cases, report = evaluate(path)

    assert len(cases) == 2
    assert report["event_detection_sensitivity"]["successes"] == 1
    assert report["end_to_end_vehicle_correctness"] == {
        "successes": 1,
        "denominator": 2,
        "rate": 0.5,
        "wilson_95": list(wilson(1, 2)),
    }
    assert cases[1]["outcome"] == "missed_event"


def test_stale_match_flag_cannot_make_missing_event_end_to_end_correct(tmp_path: Path):
    rows = [{
        "litter_case": "1",
        "video_usable": "True",
        "label_status": "KNOWN",
        "confirmed_event_count": "0",
        "predicted_vehicle_ids": "[2]",
        "gt_vehicle_id": "2",
        "vehicle_match_any": "True",
        "human_verified_correct": "False",
    }]
    path = tmp_path / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cases, report = evaluate(path)

    assert cases[0]["outcome"] == "missed_event"
    assert cases[0]["end_to_end_correct"] is False
    assert report["end_to_end_vehicle_correctness"]["successes"] == 0
