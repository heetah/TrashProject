import json
from pathlib import Path

from scripts.evaluate_reviewed_readiness import evaluate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_same_run_mapping_and_fail_closed_denominator(tmp_path: Path):
    ground_truth = tmp_path / "ground_truth"
    candidates = tmp_path / "candidates"
    ground_truth.mkdir()
    (candidates / "case_1").mkdir(parents=True)
    (candidates / "case_2").mkdir(parents=True)

    clips = [
        {"video_id": "v1", "video_filename": "litter_case_1_annotated.mp4", "video_usable": True},
        {"video_id": "v2", "video_filename": "litter_case_2_annotated.mp4", "video_usable": True},
        {"video_id": "v3", "video_filename": "litter_case_3_annotated.mp4", "video_usable": True},
    ]
    events = [
        {
            "video_id": f"v{case}",
            "video_filename": f"litter_case_{case}_annotated.mp4",
            "gt_event_id": f"gt{case}",
            "ignore": False,
            "release_start_frame": 10,
            "release_end_frame": 10,
            "release_x": 5.0,
            "release_y": 5.0,
            "review_state": "unreviewed",
        }
        for case in (1, 2, 3)
    ]
    actors = [
        {
            "video_id": f"v{case}",
            "actor_id": "vehicle_1",
            "actor_type": "vehicle",
            "bbox_frame": 10,
            "bbox": {"x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        }
        for case in (1, 2, 3)
    ]
    _write_jsonl(ground_truth / "clip_annotations.jsonl", clips)
    _write_jsonl(ground_truth / "event_annotations.jsonl", events)
    _write_jsonl(ground_truth / "actor_annotations.jsonl", actors)
    (ground_truth / "project.json").write_text(
        json.dumps({"videos": [
            {"filename": f"litter_case_{case}_annotated.mp4", "fps": 10.0, "width": 100, "height": 100}
            for case in (1, 2, 3)
        ]}),
        encoding="utf-8",
    )

    def record(case: int, release_point: list[float]) -> dict:
        return {
            "record_type": "candidate",
            "video": {"input_video": f"litter_case_{case}.mp4"},
            "event": {"litter_id": case},
            "assignment": {
                "route_type": "direct_vehicle",
                "person_key": None,
                "vehicle_key": ["vehicle", 91 + case],
                "release_frame": 10,
                "release_point": release_point,
            },
            "resolver_input": {
                "actor_frames": [{
                    "frame_index": 10,
                    "actors": [{
                        "cls": "vehicle",
                        "actor_key": ["vehicle", 91 + case],
                        "box": [0.0, 0.0, 10.0, 10.0],
                    }],
                }],
            },
        }

    _write_jsonl(
        candidates / "case_1" / "litter_case_1_annotated_backtrack_candidates.jsonl",
        [record(1, [5.0, 5.0])],
    )
    _write_jsonl(
        candidates / "case_2" / "litter_case_2_annotated_backtrack_candidates.jsonl",
        [record(2, [90.0, 90.0])],
    )

    cases, report = evaluate(ground_truth, candidates)

    by_clip = {row["clip_id"]: row for row in cases}
    assert by_clip["litter_case_1"]["outcome"] == "correct_route"
    assert by_clip["litter_case_1"]["mapped_vehicle_key"] == ["vehicle", 92]
    assert by_clip["litter_case_2"]["outcome"] == "exploratory_event_match"
    assert by_clip["litter_case_3"]["outcome"] == "missed_event"
    assert report["event_detection_sensitivity"]["successes"] == 1
    assert report["provisional_end_to_end_route_correctness"]["successes"] == 1
    assert report["provisional_end_to_end_route_correctness"]["denominator"] == 3
    assert report["annotation_status"]["all_usable_event_labels_reviewed"] is False
    assert report["finable_case_correctness"]["rate"] is None
    assert report["release_gates"]["plate_ocr_gate"].startswith("BLOCKED_")
    assert report["release_gates"]["ready_for_enforcement"] is False
