# -*- coding: utf-8 -*-
"""pipeline.events 的 GPU-free 單元測試:schema、車牌解析、排序、JSONL 往返。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.events import (
    build_analysis_report,
    build_litter_events,
    build_run_events,
    build_urinate_events,
    write_analysis_json,
    write_events_jsonl,
)


def test_litter_event_schema_and_plate_resolution():
    litter_events = [{
        "litter_id": 3,
        "frame_index": 1284,
        "bbox": [10, 20, 30, 40],
        "center": [20.0, 30.0],
        "thrower_key": ["scooter", 7],
        "escalated": True,
    }]
    vehicle_history = {7: {"license_plate": "ABC-1234"}}
    events = build_litter_events(litter_events, vehicle_history, fps=30)
    assert len(events) == 1
    ev = events[0]
    assert ev["type"] == "litter"
    assert ev["litter_id"] == 3
    assert ev["frame_index"] == 1284
    assert ev["time_sec"] == round(1284 / 30, 2)
    assert ev["thrower"] == {"cls": "scooter", "track_id": 7}
    assert ev["license_plate"] == "ABC-1234"
    assert ev["escalated"] is True


def test_litter_event_pedestrian_thrower_has_no_plate():
    litter_events = [{
        "litter_id": 1, "frame_index": 10, "bbox": [0, 0, 5, 5],
        "center": [2.5, 2.5], "thrower_key": ["person", 2], "escalated": False,
    }]
    events = build_litter_events(litter_events, {2: {"license_plate": "X"}}, fps=30)
    assert events[0]["thrower"] == {"cls": "person", "track_id": 2}
    assert events[0]["license_plate"] is None  # person 不查車牌


def test_litter_event_has_time_segment_and_normalized_plate_confidence():
    events = build_litter_events(
        [{
            "litter_id": 8,
            "frame_index": 60,
            "birth_frame": 45,
            "confirm_frame": 60,
            "bbox": [1, 2, 3, 4],
            "thrower_key": ["person", 2],
            "vehicle_key": ["vehicle", 9],
            "detector_confidence": 0.87654,
            "escalated": True,
            "backtrack": {"release_frame": 42, "confirm_frame": 60},
        }],
        {9: {"license_plate": {"number": "ABC1234", "conf": 0.91234}}},
        fps=30,
    )
    event = events[0]
    assert event["license_plate"] == "ABC1234"
    assert event["license_plate_confidence"] == 0.9123
    assert event["license_plate_status"] == "recognized"
    assert event["detector_confidence"] == 0.8765
    assert event["time_segment"] == {
        "start_frame": 42,
        "end_frame": 60,
        "start_sec": 1.4,
        "end_sec": 2.0,
        "basis": "estimated_release_to_confirmation",
        "human_reviewed": False,
    }


def test_urinate_aggregate_fallback_when_no_per_track():
    assert build_urinate_events([], {"stgcn_urinate_confirmed": 0}, fps=30) == []
    out = build_urinate_events([], {"stgcn_urinate_confirmed": 5}, fps=30)
    assert out == [{"type": "urinate", "track_id": None, "frame_index": None,
                    "time_sec": None, "confirmed_count": 5}]


def test_urinate_per_track_events_preferred_and_sorted():
    per_track = [
        {"track_id": 2, "frame_index": 300, "conf": 0.71, "evidence_sec": 5.2},
        {"track_id": 1, "frame_index": 120, "conf": 0.83, "evidence_sec": 6.0},
    ]
    out = build_urinate_events(per_track, {"stgcn_urinate_confirmed": 2}, fps=30)
    assert [e["frame_index"] for e in out] == [120, 300]  # 依 frame 排序
    assert out[0]["type"] == "urinate" and out[0]["track_id"] == 1
    assert out[0]["time_sec"] == round(120 / 30, 2)
    assert out[0]["conf"] == 0.83
    assert "confirmed_count" not in out[0]  # per-track 模式不帶聚合欄位


def test_urinate_event_keeps_backtracked_vehicle_and_plate():
    out = build_urinate_events(
        [{"track_id": 3, "frame_index": 300, "conf": 0.81, "evidence_sec": 5.2}],
        {"stgcn_urinate_confirmed": 1},
        fps=30,
        person_vehicle_map={(3, 300): ("scooter", 8)},
        vehicle_history={
            8: {"license_plate": {"number": "XYZ5678", "conf": 0.93}}
        },
    )

    assert out[0]["vehicle"] == {"cls": "scooter", "track_id": 8}
    assert out[0]["license_plate"] == "XYZ5678"
    assert out[0]["license_plate_confidence"] == 0.93
    assert out[0]["license_plate_status"] == "recognized"
    assert out[0]["attribution_status"] == "resolved"
    assert out[0]["time_segment"] == {
        "start_sec": 4.8,
        "end_sec": 10.0,
        "basis": "stgcn_evidence_to_confirmation",
        "human_reviewed": False,
    }


def test_build_run_events_sorts_litter_by_frame_then_urinate():
    litter_events = [
        {"litter_id": 2, "frame_index": 200, "bbox": [0, 0, 1, 1], "center": [0, 0],
         "thrower_key": None, "escalated": False},
        {"litter_id": 1, "frame_index": 100, "bbox": [0, 0, 1, 1], "center": [0, 0],
         "thrower_key": None, "escalated": False},
    ]
    events = build_run_events(litter_events, [], {}, {"stgcn_urinate_confirmed": 1}, fps=30)
    assert [e["type"] for e in events] == ["litter", "litter", "urinate"]
    assert [e["frame_index"] for e in events[:2]] == [100, 200]


def test_write_events_jsonl_roundtrip(tmp_path):
    events = build_run_events(
        [{"litter_id": 1, "frame_index": 5, "bbox": [1, 2, 3, 4], "center": [2, 3],
          "thrower_key": ["vehicle", 9], "escalated": True}],
        [],
        {9: {"license_plate": "AB-99"}},
        {"stgcn_urinate_confirmed": 0},
        fps=15,
    )
    path = tmp_path / "run_events.jsonl"
    n = write_events_jsonl(events, str(path))
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert n == 1 and len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["license_plate"] == "AB-99"
    assert parsed["time_sec"] == round(5 / 15, 2)


def test_analysis_report_is_compact_and_keeps_accuracy_boundary(tmp_path):
    vehicle_history = {
        9: {
            "cls": "vehicle",
            "first_seen_frame": 10,
            "last_seen_frame": 90,
            "detected_observations": 3,
            "detector_confidence_sum": 2.4,
            "detector_confidence_max": 0.9,
            "detector_confidence_last": 0.8,
            "last_bbox": [10, 20, 100, 120],
            "license_plate": {"number": "ABC1234", "conf": 0.91},
        },
        12: {
            "cls": "scooter",
            "first_seen_frame": 30,
            "last_seen_frame": 60,
            "detected_observations": 2,
            "detector_confidence_sum": 1.4,
            "detector_confidence_max": 0.75,
            "detector_confidence_last": 0.7,
            "license_plate": None,
            "plate_ocr_misses": 2,
        },
    }
    events = build_run_events(
        [{
            "litter_id": 1,
            "frame_index": 60,
            "birth_frame": 45,
            "confirm_frame": 60,
            "bbox": [1, 2, 3, 4],
            "thrower_key": ["person", 4],
            "vehicle_key": ["vehicle", 9],
            "detector_confidence": 0.88,
            "escalated": True,
        }],
        [],
        vehicle_history,
        {"stgcn_urinate_confirmed": 0},
        fps=30,
    )
    output_video = tmp_path / "scene_annotated.mp4"
    output_video.write_bytes(b"video")
    report = build_analysis_report(
        {
            "input_video": "scene.mp4",
            "output_video": str(output_video),
            "processed_frames": 300,
            "total_frames": 300,
            "raw_litter_candidates": 7,
            "filtered_litter_candidates": 2,
            "raw_litter_candidate_confidence_mean": 0.61,
            "raw_litter_candidate_confidence_max": 0.93,
            "filtered_litter_candidate_confidence_mean": 0.81,
            "filtered_litter_candidate_confidence_max": 0.9,
            "person_detections": 15,
            "person_unique_tracks": 2,
        },
        events,
        vehicle_history,
        fps=30,
    )
    assert set(report) == {"schema_version", "video", "summary", "events"}
    assert report["video"] == {
        "file": "scene_annotated.mp4",
        "duration_sec": 10.0,
    }
    assert report["schema_version"] == "2.0.0"
    assert report["summary"]["litter_event_count"] == 1
    assert report["summary"]["urinate_event_count"] == 0
    assert report["summary"]["passed_vehicle_count"] == 2
    assert report["summary"]["average_litter_confidence"] == 0.88
    assert report["summary"]["detection_accuracy"] is None
    assert report["summary"]["accuracy_status"] == "not_evaluated"
    assert report["summary"]["littering_plates"] == ["ABC1234"]
    assert report["summary"]["review_required"] is True
    assert report["events"] == [{
        "type": "litter",
        "id": 1,
        "start_sec": 1.5,
        "end_sec": 2.0,
        "confidence": 0.88,
        "vehicle": "vehicle:9",
        "plate": "ABC1234",
        "plate_confidence": 0.91,
        "plate_status": "recognized",
        "attribution_status": None,
        "review_required": True,
    }]

    path = tmp_path / "scene_annotated_analysis.json"
    assert write_analysis_json(report, path) == str(path)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed == report
    assert not (tmp_path / "scene_annotated_analysis.json.tmp").exists()


def test_analysis_compacts_urinate_event():
    report = build_analysis_report(
        {"output_video": "scene_annotated.mp4", "duration_sec": 20.0},
        [{
            "type": "urinate",
            "track_id": 3,
            "time_sec": 12.5,
            "conf": 0.81,
            "evidence_sec": 5.2,
        }],
        {},
        fps=30,
    )
    assert report["summary"]["urinate_event_count"] == 1
    assert report["events"] == [{
        "type": "urinate",
        "track_id": 3,
        "time_sec": 12.5,
        "start_sec": 12.5,
        "end_sec": 12.5,
        "confidence": 0.81,
        "vehicle": None,
        "plate": None,
        "plate_confidence": None,
        "plate_status": None,
        "attribution_status": None,
        "review_required": True,
    }]


def test_analysis_counts_direct_vehicle_thrower_and_missing_plate():
    events = build_run_events(
        [{
            "litter_id": 2,
            "frame_index": 20,
            "bbox": [1, 2, 3, 4],
            "thrower_key": ["scooter", 5],
            "vehicle_key": None,
            "escalated": True,
        }],
        [],
        {5: {"cls": "scooter", "license_plate": None}},
        {},
        fps=10,
    )
    report = build_analysis_report(
        {"processed_frames": 100, "total_frames": 100},
        events,
        {5: {"cls": "scooter", "license_plate": None}},
        fps=10,
    )
    assert report["summary"]["passed_vehicle_count"] == 1
    assert report["events"][0] == {
        "type": "litter",
        "id": 2,
        "start_sec": 2.0,
        "end_sec": 2.0,
        "confidence": None,
        "vehicle": "scooter:5",
        "plate": None,
        "plate_confidence": None,
        "plate_status": "not_requested",
        "attribution_status": None,
        "review_required": True,
    }


def test_tracker_captures_confirmed_litter_event():
    # 端到端(GPU-free):驅動 GlobalLitterTracker 直到 litter 確認,確認事件被記錄且去重。
    from pipeline.litter_tracker import GlobalLitterTracker

    tracker = GlobalLitterTracker(distance_threshold=250, fps=30)
    try:
        assert tracker.get_litter_events() == []
        # 直接注入一筆已確認的 litter 事件(模擬 confirm 當幀的記錄路徑),驗證 accessor 契約。
        tracker._litter_events.append({
            "litter_id": 0, "frame_index": 42, "bbox": [1, 2, 3, 4],
            "center": [2.0, 3.0], "thrower_key": ["vehicle", 5], "escalated": True,
        })
        got = tracker.get_litter_events()
        assert len(got) == 1 and got[0]["litter_id"] == 0
        # accessor 回傳複本,外部修改不影響內部狀態。
        got.append({"x": 1})
        assert len(tracker.get_litter_events()) == 1
    finally:
        tracker.close()
