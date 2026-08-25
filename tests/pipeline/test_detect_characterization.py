# -*- coding: utf-8 -*-
"""detect() 的 GPU-free 特徵化(golden-master)測試。

detect() 在給定 precomputed actors + stub 偵測輸出、model 傳 None/stub 時可完全在 CPU
執行(不碰 YOLO/RTDETR/STGCN)。本測試對固定場景把「標註影格 sha + 可觀察狀態」釘成
golden,作為 detect() 拆解成 stage 函式時的行為不變閘門——輸出必須逐位元相同。

    conda run -n rtdetr python -m pytest tests/pipeline/test_detect_characterization.py -q
"""
import hashlib
import os
import sys
from collections import defaultdict, deque

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.detect import _stage_render_actors, _stage_render_rtdetr_debug, detect
from pipeline.litter_tracker import GlobalLitterTracker

COLORS = {
    "litter": (128, 0, 128),
    "person": (255, 200, 128),
    "vehicle": (0, 255, 0),
    "scooter": (0, 255, 255),
}

GOLDEN = {
    "A": {
        "annotated_sha": "2c52bd94e768a828123a66d9f505eb45ae06ce4eaf36d480f042a654ecf662d4",
        "annotated_shape": [120, 160, 3],
        "stats": {
            "filtered_litter_candidates": 0,
            "person_detections": 1,
            "person_frame_hits": 1,
            "raw_litter_candidates": 0,
            "rtdetr_litter_candidates": 0,
        },
        "veh10_centroids": [[110.0, 70.0]],
        "violator_keys": [],
        "active_litter_ids": [],
    },
    "B": {
        "annotated_sha": "2c52bd94e768a828123a66d9f505eb45ae06ce4eaf36d480f042a654ecf662d4",
        "annotated_shape": [120, 160, 3],
        "stats": {
            "filtered_litter_candidate_frames": [
                {"candidate_count": 1, "frame_index": 0}
            ],
            "filtered_litter_candidates": 1,
            "geometry_litter_candidate_frames": [
                {"candidate_count": 1, "frame_index": 0}
            ],
            "person_detections": 1,
            "person_frame_hits": 1,
            "raw_litter_candidates": 1,
            "rtdetr_evaluated_frames": 1,
            "rtdetr_litter_candidate_frames": [
                {"candidate_count": 1, "frame_index": 0}
            ],
            "rtdetr_litter_candidates": 1,
        },
        "veh10_centroids": [[110.0, 70.0]],
        "violator_keys": [],
        "active_litter_ids": [0],
    },
}


def _fresh_vehicle_history():
    return defaultdict(lambda: {
        "centroids": deque(maxlen=30),
        "license_plate": None,
        "plate_search_until_found": False,
        "plate_blocked_since_litter": False,
    })


class _StubBox:
    def __init__(self, cls, conf, xyxy):
        self.cls = [int(cls)]
        self.conf = [float(conf)]
        self.xyxy = [np.array(xyxy, dtype=float)]


class _StubResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _StubTrash:
    names = {0: "litter"}


def _base_inputs():
    rng = np.random.RandomState(1234)
    frame = rng.randint(0, 256, size=(120, 160, 3), dtype=np.uint8)
    prev_frame = frame.copy()  # 相同 → shake_mag ~0,不觸發晃動冷卻
    persons = [{"box": np.array([20.0, 20.0, 60.0, 100.0]), "track_id": 1,
                "cls": "person", "mask_poly": None}]
    vehicles = [{"box": np.array([80.0, 30.0, 140.0, 110.0]), "track_id": 10,
                 "cls": "vehicle", "mask_poly": None}]
    return frame, prev_frame, persons, vehicles


def _fingerprint(annotated, stats, vehicle_history, violator_cache, tracker):
    return {
        "annotated_sha": hashlib.sha256(annotated.tobytes()).hexdigest(),
        "annotated_shape": list(annotated.shape),
        "stats": {k: (sorted(v) if isinstance(v, set) else v)
                  for k, v in sorted(stats.items())},
        "veh10_centroids": [list(c) for c in vehicle_history[10]["centroids"]],
        "violator_keys": sorted(str(k) for k in violator_cache),
        "active_litter_ids": sorted(tracker.active_litters.keys()),
    }


def _run(model_trash=None, trash_results=None, with_motion=False):
    frame, prev_frame, persons, vehicles = _base_inputs()
    fg_mask = np.zeros((120, 160), dtype=np.uint8)
    if with_motion:
        fg_mask[40:60, 40:60] = 255
    tracker = GlobalLitterTracker(distance_threshold=250, fps=30)
    vehicle_history = _fresh_vehicle_history()
    stats, violator_cache = {}, {}
    try:
        annotated = detect(
            frame, None, model_trash, COLORS, fg_mask, tracker, vehicle_history,
            fps=30, violator_display_cache=violator_cache, action_module=None,
            frame_index=0, yolo_seg_cache={}, precomputed_persons=persons,
            precomputed_vehicles=vehicles, precomputed_trash_results=trash_results,
            stats=stats, actor_mode="predict", prev_frame=prev_frame,
        )
        return _fingerprint(annotated, stats, vehicle_history, violator_cache, tracker)
    finally:
        tracker.close()


def test_detect_scenario_a_actors_only():
    assert _run() == GOLDEN["A"]


def test_detect_scenario_b_with_litter_candidate():
    trash = [_StubResult([_StubBox(0, 0.9, [40, 40, 60, 60])])]
    assert _run(model_trash=_StubTrash(), trash_results=trash, with_motion=True) == GOLDEN["B"]


def test_rtdetr_frame_is_recorded_before_geometry_rejection():
    # 2px 寬的 bbox 是 RT-DETR litter output，但會被基本 geometry gate 淘汰。
    trash = [_StubResult([_StubBox(0, 0.9, [40, 40, 42, 60])])]
    result = _run(model_trash=_StubTrash(), trash_results=trash, with_motion=True)

    assert result["stats"]["rtdetr_litter_candidates"] == 1
    assert result["stats"]["rtdetr_litter_candidate_frames"] == [
        {"frame_index": 0, "candidate_count": 1}
    ]
    assert result["stats"]["raw_litter_candidates"] == 0
    assert result["stats"]["filtered_litter_candidates"] == 0


def test_debug_render_includes_actor_track_ids(monkeypatch):
    labels = []
    monkeypatch.setenv("LITTER_DEBUG", "1")
    monkeypatch.setattr(
        "pipeline.detect.cv2.putText",
        lambda _frame, text, *_args, **_kwargs: labels.append(text),
    )
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    _stage_render_actors(
        frame,
        [
            {"box": [10, 10, 40, 80], "cls": "person", "track_id": 7},
            {"box": [60, 20, 140, 100], "cls": "vehicle", "track_id": 12},
        ],
        {},
        80.0,
        COLORS,
        {},
        _fresh_vehicle_history(),
        1,
        0.5,
        1,
        None,
    )
    assert labels == ["person ID:7", "vehicle ID:12"]


def test_debug_render_includes_rtdetr_box_before_geometry_rejection(monkeypatch):
    rectangles, labels = [], []
    monkeypatch.setenv("LITTER_DEBUG", "1")
    monkeypatch.setattr(
        "pipeline.detect.cv2.rectangle",
        lambda _frame, p1, p2, color, thickness: rectangles.append(
            (p1, p2, color, thickness)
        ),
    )
    monkeypatch.setattr(
        "pipeline.detect.cv2.putText",
        lambda _frame, text, *_args, **_kwargs: labels.append(text),
    )

    _stage_render_rtdetr_debug(
        np.zeros((120, 160, 3), dtype=np.uint8),
        [[40, 40, 42, 60, 0.9]],  # 2px wide: later geometry gate rejects it.
        None,
    )

    assert rectangles == [((40, 40), (42, 60), (255, 0, 255), 2)]
    assert labels == ["RTDETR raw 0.90"]


def test_rtdetr_debug_overlay_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LITTER_DEBUG", raising=False)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    _stage_render_rtdetr_debug(frame, [[40, 40, 60, 60, 0.9]], None)

    assert not frame.any()
