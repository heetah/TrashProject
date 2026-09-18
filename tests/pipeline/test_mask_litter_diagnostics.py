import json

import numpy as np

from pipeline.backtrack.mask_diagnostics import build_mask_litter_diagnostics
from pipeline.backtrack.resolver import SmartBacktrackResolver
from pipeline.litter_tracker import GlobalLitterTracker


def _vehicle(source="seg_predict", observed=True, mask=True, track_id=2):
    actor = {
        "cls": "vehicle",
        "track_id": track_id,
        "tracklet_uid": f"vehicle:{track_id}:0",
        "box": [10.0, 10.0, 50.0, 50.0],
        "confidence": 0.9,
        "observed": observed,
        "source": source,
    }
    if mask:
        actor["mask_poly"] = np.asarray(
            [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
            dtype=np.float32,
        )
    return actor


def test_mask_litter_summary_is_bounded_deterministic_and_json_safe():
    litters = [
        [20.0, 20.0, 30.0, 30.0, 0.8],
        [60.0, 20.0, 70.0, 30.0, 0.7],
    ]
    result = build_mask_litter_diagnostics(
        [_vehicle(track_id=3), _vehicle(track_id=2)], litters, frame_index=7
    )

    assert [
        (row["raw_litter_index"], row["actor_key"])
        for row in result["records"]
    ] == [
        (0, ["vehicle", 2]),
        (0, ["vehicle", 3]),
        (1, ["vehicle", 2]),
        (1, ["vehicle", 3]),
    ]
    inside = result["records"][0]
    outside = result["records"][2]
    assert inside["mask_overlap_ratio"] == 1.0
    assert inside["bbox_containment_ratio"] == 1.0
    assert 0.0 < inside["signed_distance_actor_scale"] < 1.0
    assert outside["mask_overlap_ratio"] == 0.0
    assert outside["bbox_containment_ratio"] == 0.0
    assert -1.0 < outside["signed_distance_actor_scale"] < 0.0
    assert "mask_poly" not in json.dumps(result, allow_nan=False)


def test_mask_litter_summary_rejects_cached_missing_and_invalid_masks():
    actors = [
        _vehicle(source="cache", observed=False, track_id=1),
        _vehicle(source="kalman_rts", observed=True, track_id=2),
        _vehicle(mask=False, track_id=3),
        {
            **_vehicle(track_id=4),
            "mask_poly": [[0.0, 0.0], [1.0, float("nan")], [2.0, 0.0]],
        },
    ]
    result = build_mask_litter_diagnostics(
        actors, [[20.0, 20.0, 30.0, 30.0, 0.8]], frame_index=9
    )

    assert result["records"] == []
    assert result["skipped"] == {
        "invalid_actor_mask": 2,
        "not_observed": 1,
        "unsupported_source": 1,
    }


def test_tracker_flag_off_adds_no_diagnostic_key(monkeypatch):
    monkeypatch.delenv("SMART_BACKTRACK_MASK_DIAGNOSTICS", raising=False)
    tracker = GlobalLitterTracker(fps=10)
    try:
        tracker._record_actor_frame(
            [_vehicle()],
            frame_index=3,
            dynamic_litters=[[20.0, 20.0, 30.0, 30.0, 0.8]],
        )
        row = tracker._smart_actor_history[-1]
        assert list(row) == ["frame_index", "actors"]
        assert "mask_poly" not in json.dumps(row, allow_nan=False)
    finally:
        tracker.close()


def test_tracker_flag_on_records_scalars_but_never_polygon(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK_MASK_DIAGNOSTICS", "1")
    tracker = GlobalLitterTracker(fps=10)
    try:
        tracker._record_actor_frame(
            [_vehicle()],
            frame_index=3,
            dynamic_litters=[[20.0, 20.0, 30.0, 30.0, 0.8]],
        )
        row = tracker._smart_actor_history[-1]
        diagnostic = row["mask_litter_diagnostics"]
        assert len(diagnostic["records"]) == 1
        encoded = json.dumps(row, allow_nan=False)
        assert "mask_poly" not in encoded
        assert len(encoded) < 3_000
    finally:
        tracker.close()


def test_resolver_routes_are_identical_with_frame_level_diagnostics():
    actor_frames = []
    for frame in range(8, 13):
        actor_frames.append({
            "frame_index": frame,
            "actors": [{
                "cls": "vehicle",
                "track_id": 2,
                "tracklet_uid": "vehicle:2:0",
                "box": [50.0, 120.0, 180.0, 230.0],
                "confidence": 0.9,
                "observed": True,
                "source": "seg_predict",
            }],
        })
    task = {
        "litter_id": 9,
        "fps": 10.0,
        "birth_frame": 10,
        "confirm_frame": 12,
        "history": [(108.0, 95.0), (110.0, 100.0), (112.0, 108.0)],
        "history_frames": [10, 11, 12],
        "history_boxes": [
            [104.0, 91.0, 112.0, 99.0],
            [106.0, 96.0, 114.0, 104.0],
            [108.0, 104.0, 116.0, 112.0],
        ],
        "history_confidences": [0.9, 0.9, 0.9],
        "actor_frames": actor_frames,
    }
    with_diagnostics = {
        **task,
        "actor_frames": [
            {
                **row,
                "mask_litter_diagnostics": {
                    "schema": "mask-litter-diagnostics/v1",
                    "frame_index": row["frame_index"],
                    "records": [],
                    "skipped": {},
                },
            }
            for row in actor_frames
        ],
    }

    resolver = SmartBacktrackResolver(fps=10)
    baseline = resolver.resolve_task(task)
    candidate = resolver.resolve_task(with_diagnostics)

    assert candidate.route_id == baseline.route_id
    assert candidate.total_cost.hex() == baseline.total_cost.hex()
    assert [
        (route.route_id, route.cost.hex())
        for route in candidate.routes
    ] == [
        (route.route_id, route.cost.hex())
        for route in baseline.routes
    ]
