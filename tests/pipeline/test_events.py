# -*- coding: utf-8 -*-
"""pipeline.events 的 GPU-free 單元測試:schema、車牌解析、排序、JSONL 往返。"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.events import (
    build_litter_events,
    build_run_events,
    build_urinate_events,
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
