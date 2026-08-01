# -*- coding: utf-8 -*-
"""GPU-free integration checks for the production smart backtrack path."""
import os
import sys

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.backtrack.tracking import KalmanHungarianTracker
from pipeline.backtrack.costs import CostCell
from pipeline.backtrack.resolver import SmartBacktrackConfig, SmartBacktrackResolver
from pipeline.backtrack.trajectory import ReleaseHypothesis
import pipeline.backtrack.resolver as resolver_module
from pipeline.detect import _assign_fast_actor_track_ids, _clone_cached_actors
from pipeline.litter_tracker import GlobalLitterTracker


def _actor(cls_name, track_id, box, confidence=0.9, observed=True):
    return {
        "cls": cls_name,
        "track_id": track_id,
        "box": np.asarray(box, dtype=np.float32),
        "confidence": confidence,
        "observed": observed,
        "source": "test",
    }


def test_kalman_hungarian_only_keeps_cross_frame_identity():
    tracker = KalmanHungarianTracker(iou_threshold=0.1, max_missed_frames=5)
    first = tracker.update(
        [
            {"cls": "vehicle", "box": (0, 0, 40, 20), "confidence": 0.9},
            {"cls": "vehicle", "box": (100, 0, 140, 20), "confidence": 0.9},
        ],
        frame_index=0,
    )
    left_id, right_id = first[0]["track_id"], first[1]["track_id"]

    # Skip a frame; Kalman predicts to frame 2 before the full Hungarian solve.
    second = tracker.update(
        [
            {"cls": "vehicle", "box": (18, 0, 58, 20), "confidence": 0.9},
            {"cls": "vehicle", "box": (82, 0, 122, 20), "confidence": 0.9},
        ],
        frame_index=2,
    )
    assert second[0]["track_id"] == left_id
    assert second[1]["track_id"] == right_id
    assert all(item["observed"] is True for item in second)


def test_actor_cache_reuse_is_not_a_new_kalman_measurement():
    source = [{
        "cls": "vehicle",
        "track_id": 3,
        "box": np.asarray((0, 0, 20, 20), dtype=np.float32),
        "confidence": 0.9,
        "observed": True,
    }]

    cached = _clone_cached_actors(source)

    assert cached[0]["observed"] is False
    assert cached[0]["source"] == "cache"
    assert source[0]["observed"] is True


def test_fast_actor_tracker_receives_real_video_fps():
    cache = {}
    _assign_fast_actor_track_ids(
        [{"cls": "vehicle", "box": (0, 0, 40, 20), "confidence": 0.9}],
        cache,
        frame_index=0,
        fps=25.0,
    )

    assert cache["fast_actor_tracker"].fps == 25.0
    assert (
        cache["fast_actor_tracker"]._tracks[1].kalman.config.frames_per_second
        == 25.0
    )


def test_hungarian_does_not_resurrect_expired_track():
    tracker = KalmanHungarianTracker(iou_threshold=0.1, max_missed_frames=2)
    first = tracker.update(
        [{"cls": "vehicle", "box": (0, 0, 40, 20), "confidence": 0.9}],
        frame_index=0,
    )
    old_id = first[0]["track_id"]

    much_later = tracker.update(
        [{"cls": "vehicle", "box": (0, 0, 40, 20), "confidence": 0.9}],
        frame_index=100,
    )

    assert much_later[0]["track_id"] != old_id


def test_invalid_pair_is_gated_before_hungarian(monkeypatch):
    tracker = KalmanHungarianTracker(iou_threshold=0.3, max_missed_frames=5)
    first = tracker.update(
        [
            {"cls": "vehicle", "box": (0, 0, 40, 20), "confidence": 0.9},
            {"cls": "vehicle", "box": (100, 0, 140, 20), "confidence": 0.9},
        ],
        frame_index=0,
    )
    track_1, track_2 = first[0]["track_id"], first[1]["track_id"]

    def fake_cost(track, actor, _frame_index):
        slot = actor["slot"]
        if track.track_id == track_1 and slot == "a":
            return 0.01, 0.0, 100.0  # cheapest numerically, but invalid
        if track.track_id == track_1 and slot == "b":
            return 0.20, 0.6, 1.0
        if track.track_id == track_2 and slot == "a":
            return 0.20, 0.6, 1.0
        return 0.90, 0.6, 1.0

    monkeypatch.setattr(tracker, "_association_cost", fake_cost)
    assigned = tracker.update(
        [
            {
                "cls": "vehicle", "slot": "a",
                "box": (100, 0, 140, 20), "confidence": 0.9,
            },
            {
                "cls": "vehicle", "slot": "b",
                "box": (0, 0, 40, 20), "confidence": 0.9,
            },
        ],
        frame_index=1,
    )

    assert assigned[0]["track_id"] == track_2
    assert assigned[1]["track_id"] == track_1


def test_bad_detection_does_not_abort_other_actor_tracks():
    tracker = KalmanHungarianTracker(iou_threshold=0.1, max_missed_frames=5)

    assigned = tracker.update(
        [
            {"cls": "person", "box": (0, 0, 20, 80), "confidence": 0.9},
            {"cls": "person", "box": (np.nan, 0, 20, 80), "confidence": 0.9},
        ],
        frame_index=0,
    )

    assert len(assigned) == 1
    assert assigned[0]["track_id"] == 1


def test_person_topk_ranks_complete_route_not_ba_alone(monkeypatch):
    resolver = SmartBacktrackResolver(
        fps=10,
        config=SmartBacktrackConfig(
            top_k_people=1,
            top_k_vehicles=1,
            dustbin_cost=7.0,
        ),
    )
    release = ReleaseHypothesis(
        frame_index=10,
        mean_uv=np.asarray((100.0, 100.0)),
        covariance_uv=np.eye(2),
        velocity_uv=np.zeros(2),
        model="test",
        prior_cost=0.0,
    )
    person_1 = ("person", 1)
    person_2 = ("person", 2)
    vehicle = ("vehicle", 9)
    monkeypatch.setattr(resolver, "_release_hypotheses", lambda _task: [release])
    monkeypatch.setattr(
        resolver,
        "_build_actor_tracks",
        lambda _task: {
            person_1: ["person_1"],
            person_2: ["person_2"],
            vehicle: [],
        },
    )
    monkeypatch.setattr(
        resolver_module,
        "build_ba_costs",
        lambda *_args, **_kwargs: {
            person_1: CostCell(True, 1.0, {}, 10, None),
            person_2: CostCell(True, 1.1, {}, 10, None),
        },
    )
    monkeypatch.setattr(resolver_module, "build_bc_costs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        resolver_module,
        "build_ac_costs",
        lambda *_args, **_kwargs: {
            (person_2, vehicle): CostCell(True, 0.0, {}, None, None)
        },
    )
    monkeypatch.setattr(
        resolver_module,
        "compute_c_ba",
        lambda releases, observations, *_args, **_kwargs: CostCell(
            True,
            1.0 if observations[0] == "person_1" else 1.1,
            {},
            releases[0].frame_index,
            None,
        ),
    )
    seen_release_frames = []

    def fake_bc(releases, *_args, **_kwargs):
        seen_release_frames.extend(item.frame_index for item in releases)
        return CostCell.rejected("no_direct_support")

    monkeypatch.setattr(resolver_module, "compute_c_bc", fake_bc)

    routes = resolver.build_routes({"fps": 10})
    person_routes = [route for route in routes if route.person_key is not None]

    assert {route.person_key for route in person_routes} == {person_2}
    assert any(route.vehicle_key == vehicle for route in person_routes)
    assert seen_release_frames == [10]


def test_person_vehicle_route_minimizes_over_shared_release_time(monkeypatch):
    resolver = SmartBacktrackResolver(
        fps=10,
        config=SmartBacktrackConfig(
            top_k_people=1,
            top_k_vehicles=1,
            dustbin_cost=7.0,
            ac_weight=0.0,
            bc_support_bonus=1.0,
        ),
    )
    releases = [
        ReleaseHypothesis(
            frame_index=frame,
            mean_uv=np.asarray((100.0, 100.0)),
            covariance_uv=np.eye(2),
            velocity_uv=np.zeros(2),
            model="test",
            prior_cost=0.0,
        )
        for frame in (1, 2)
    ]
    person = ("person", 1)
    vehicle = ("vehicle", 9)
    monkeypatch.setattr(resolver, "_release_hypotheses", lambda _task: releases)
    monkeypatch.setattr(
        resolver,
        "_build_actor_tracks",
        lambda _task: {person: ["person"], vehicle: ["vehicle"]},
    )
    monkeypatch.setattr(
        resolver_module,
        "build_ba_costs",
        lambda *_args, **_kwargs: {
            person: CostCell(True, 1.0, {}, 1, None)
        },
    )
    monkeypatch.setattr(resolver_module, "build_bc_costs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        resolver_module,
        "build_ac_costs",
        lambda *_args, **_kwargs: {
            (person, vehicle): CostCell(True, 0.0, {}, None, None)
        },
    )
    monkeypatch.setattr(
        resolver_module,
        "compute_c_ba",
        lambda release_items, *_args, **_kwargs: CostCell(
            True,
            1.0 if release_items[0].frame_index == 1 else 1.1,
            {},
            release_items[0].frame_index,
            None,
        ),
    )
    monkeypatch.setattr(
        resolver_module,
        "compute_c_bc",
        lambda release_items, *_args, **_kwargs: (
            CostCell.rejected("no_support")
            if release_items[0].frame_index == 1
            else CostCell(True, 0.0, {}, 2, None)
        ),
    )

    routes = resolver.build_routes({"fps": 10})
    vehicle_route = next(
        route for route in routes if route.route_type == "person_vehicle"
    )

    assert vehicle_route.metadata["release_frame"] == 2
    assert np.isclose(vehicle_route.cost, 0.1)


def test_finalize_flushes_last_task_and_rewrites_authoritative_event(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK", "1")
    tracker = GlobalLitterTracker(distance_threshold=250, fps=10)
    try:
        # Person was near the release and was previously associated with the
        # same vehicle. Both actor tracks have real observations for RTS.
        for frame_index in range(4, 14):
            person_x = 100 + frame_index * 2
            tracker._record_actor_frame(
                [
                    _actor(
                        "person", 1,
                        (person_x - 20, 80, person_x + 20, 200),
                    ),
                    _actor("vehicle", 7, (60, 150, 180, 240)),
                ],
                frame_index,
            )

        event = {
            "litter_id": 12,
            "frame_index": 13,
            "bbox": [155, 190, 165, 200],
            "center": [160.0, 195.0],
            "thrower_key": ["vehicle", 999],  # deliberately wrong provisional
            "vehicle_key": None,
            "escalated": True,
            "backtrack_status": "pending",
        }
        tracker._litter_events.append(event)
        tracker._litter_events_by_id[12] = event
        submitted = tracker._submit_backward_resolution(
            litter_id=12,
            litter_data={
                "birth_frame": 10,
                "birth_centroid": (130, 150),
                "birth_bbox": (125, 145, 135, 155, 0.9),
                "bbox": (155, 190, 165, 200, 0.9),
                "history": [(130, 150), (140, 160), (150, 175), (160, 195)],
                "history_frames": [10, 11, 12, 13],
            },
            current_bbox=(155, 190, 165, 200, 0.9),
            current_centroid=(160, 195),
            confirm_frame=13,
            prev_thrower_key=("vehicle", 999),
        )
        assert submitted is True

        summary = tracker.finalize_backtracking(timeout=4.0)
        rewritten = tracker.get_litter_events()[0]
        assert summary["worker_flushed"] is True
        assert summary["candidate_tables"] == 1
        assert rewritten["backtrack_status"] == "resolved"
        assert rewritten["thrower_key"] == ["person", 1]
        assert rewritten["vehicle_key"] == ["vehicle", 7]
        assert rewritten["thrower_key"] != ["vehicle", 999]
        assert rewritten["backtrack"]["route_type"] == "person_vehicle"
        sidecar_records = tracker.get_backtrack_candidate_records(
            input_video="synthetic.mp4",
            output_video=None,
        )
        assert len(sidecar_records) == 1
        sidecar = sidecar_records[0]
        assert sidecar["record_type"] == "candidate"
        assert sidecar["assignment"]["route_id"] == rewritten["backtrack"]["route_id"]
        assert sum(route["selected"] for route in sidecar["routes"]) == 1
        assert len(sidecar["release_hypotheses"]) >= 1
        assert set(sidecar["pair_costs"]) == {
            "BA", "BA_by_release", "AC", "BC", "BC_by_release"
        }
        assert sidecar["candidate_diagnostics"]["pre_prune_routes"]
    finally:
        tracker.close()


def test_person_can_walk_away_from_vehicle_before_throw():
    actor_frames = []
    for frame_index in range(19):
        # Person begins nested in vehicle, then walks far to the right.
        person_x = 80 if frame_index < 5 else 100 + 20 * frame_index
        actor_frames.append({
            "frame_index": frame_index,
            "actors": [
                _actor(
                    "person", 1,
                    (person_x - 20, 80, person_x + 20, 200),
                ),
                _actor("vehicle", 7, (20, 140, 150, 240)),
            ],
        })
    task = {
        "litter_id": 2,
        "fps": 10.0,
        "birth_frame": 15,
        "confirm_frame": 18,
        "history": [(400, 150), (410, 160), (420, 175), (430, 195)],
        "history_frames": [15, 16, 17, 18],
        "actor_frames": actor_frames,
    }

    resolution = SmartBacktrackResolver(fps=10).resolve_task(task)

    assert resolution.person_key == ("person", 1)
    assert resolution.vehicle_key == ("vehicle", 7)
    # B-C is invalid because the vehicle is far from the release, but it is
    # support-only on a person route; old direct-vehicle gating must not kill it.
    assert resolution.components["costs"]["BC"]["valid"] is False
    assert resolution.route_type == "person_vehicle"


def test_stale_worker_revision_cannot_overwrite_new_assignment(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK", "1")
    tracker = GlobalLitterTracker(fps=10)
    try:
        event = {
            "litter_id": 4,
            "frame_index": 1,
            "bbox": [0, 0, 1, 1],
            "thrower_key": None,
            "vehicle_key": None,
            "escalated": False,
        }
        tracker._litter_events.append(event)
        tracker._litter_events_by_id[4] = event
        tracker._apply_backward_result({
            "litter_id": 4,
            "revision": 2,
            "status": "resolved",
            "actor_key": ("person", 20),
            "person_key": ("person", 20),
            "vehicle_key": None,
            "mark_items": [],
        })
        tracker._apply_backward_result({
            "litter_id": 4,
            "revision": 1,
            "status": "resolved",
            "actor_key": ("person", 10),
            "person_key": ("person", 10),
            "vehicle_key": None,
            "mark_items": [],
        })
        assert event["thrower_key"] == ["person", 20]
        assert event["backtrack"]["revision"] == 2
    finally:
        tracker.close()


def test_final_assignment_retracts_superseded_litter_mark(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK", "1")
    tracker = GlobalLitterTracker(fps=10)
    try:
        tracker._apply_backward_result({
            "litter_id": 8,
            "revision": 1,
            "status": "resolved",
            "actor_key": ("person", 10),
            "person_key": ("person", 10),
            "vehicle_key": None,
            "mark_items": [{
                "actor_key": ("person", 10),
                "center": (50.0, 50.0),
            }],
        })
        assert ("person", 10) in tracker.violators

        tracker._apply_backward_result({
            "litter_id": 8,
            "revision": 2,
            "status": "dustbin",
            "actor_key": None,
            "person_key": None,
            "vehicle_key": None,
            "mark_items": [],
        })
        assert ("person", 10) not in tracker.violators
    finally:
        tracker.close()


def test_legacy_exception_fallback_preserves_linked_vehicle(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK", "1")
    tracker = GlobalLitterTracker(fps=10)

    class _BrokenResolver:
        @staticmethod
        def resolve_task(_task):
            raise RuntimeError("synthetic smart failure")

    try:
        tracker._smart_resolver = _BrokenResolver()
        tracker._resolve_backward_task_legacy = lambda _task: {
            "litter_id": 3,
            "actor_key": ("person", 4),
            "plate_key": ("vehicle", 9),
            "mark_items": [
                {"actor_key": ("person", 4), "center": (10.0, 10.0)},
                {"actor_key": ("vehicle", 9), "center": (20.0, 20.0)},
            ],
            "score": 1.0,
        }

        result = tracker._resolve_backward_task({
            "litter_id": 3,
            "revision": 1,
            "actor_frames": [],
        })

        assert result["status"] == "legacy"
        assert result["person_key"] == ("person", 4)
        assert result["vehicle_key"] == ("vehicle", 9)
        assert result["route_type"] == "person_vehicle"
    finally:
        tracker.close()
