import json
from pathlib import Path

from pipeline.backtrack.release_validation import (
    REVIEW_SCHEMA,
    build_negative_queue,
    build_positive_queue,
    camera_suggestions,
    hamming_hex,
    validate_built_rows,
)


def test_hamming_hex_counts_bits():
    assert hamming_hex("00", "03") == 2


def test_camera_suggestions_remain_unreviewed():
    rows = camera_suggestions(
        [
            {"event_id": "a", "fingerprint": "00"},
            {"event_id": "b", "fingerprint": "01"},
            {"event_id": "c", "fingerprint": "ff"},
        ],
        threshold=1,
    )
    by_id = {row["event_id"]: row for row in rows}
    assert by_id["a"]["suggested_group_id"] == by_id["b"]["suggested_group_id"]
    assert by_id["a"]["suggested_group_id"] != by_id["c"]["suggested_group_id"]
    assert all(row["same_camera"] is None for row in rows)
    assert all(row["status"] == "unreviewed_machine_suggestion" for row in rows)


def test_positive_queue_is_blank_and_legacy_is_separate(tmp_path, monkeypatch):
    source = tmp_path / "litter_case_7.mp4"
    source.write_bytes(b"video")
    monkeypatch.setattr(
        "pipeline.backtrack.release_validation._probe_video",
        lambda path, with_fingerprint: type(
            "Probe", (), {"fps": 10.0, "frame_count": 20, "width": 30,
                           "height": 40, "duration_sec": 2.0,
                           "fingerprint": "00", "decodable": True}
        )(),
    )
    event = {
        "gt_event_id": "case-7-event-1",
        "video_id": "vid7",
        "video_filename": "litter_case_7_annotated.mp4",
        "event_label": "litter",
        "vehicle_id": "vehicle_1",
        "review_state": "unreviewed",
    }
    queue, legacy = build_positive_queue([event], tmp_path)
    assert queue[0]["schema"] == REVIEW_SCHEMA
    assert queue[0]["reviewer_a"]["event_label"] is None
    assert queue[0]["reviewer_b"]["admissible_routes"] == []
    assert "vehicle_id" not in json.dumps(queue[0])
    assert legacy[0]["annotation"]["vehicle_id"] == "vehicle_1"


def test_negative_folder_name_is_never_promoted(tmp_path, monkeypatch):
    normal = tmp_path / "normal"
    normal.mkdir()
    video = normal / "normal_case1.mp4"
    video.write_bytes(b"video")
    monkeypatch.setattr(
        "pipeline.backtrack.release_validation._probe_video",
        lambda path, with_fingerprint: type(
            "Probe", (), {"fps": None, "frame_count": None, "width": None,
                           "height": None, "duration_sec": None,
                           "fingerprint": None, "decodable": False}
        )(),
    )
    rows = build_negative_queue([normal])
    assert rows[0]["candidate_status"] == "unreviewed_negative_candidate"
    assert rows[0]["reviewer_a"]["contains_litter"] is None
    assert rows[0]["adjudication"]["final_contains_litter"] is None


def test_package_validation_rejects_truth_leak():
    positive = [{
        "schema": REVIEW_SCHEMA,
        "event_id": "event-1",
        "source_video": {"decodable": True},
        "blinding": {"model_outputs_included": False},
        "reviewer_a": {"review_state": "unreviewed", "event_label": "litter", "admissible_routes": []},
        "reviewer_b": {"review_state": "unreviewed", "event_label": None, "admissible_routes": []},
    }]
    try:
        validate_built_rows(positive, [], [])
    except ValueError as exc:
        assert "is not blank" in str(exc)
    else:
        raise AssertionError("truth leak must fail validation")
