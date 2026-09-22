# -*- coding: utf-8 -*-
"""GPU-free integration checks for the production smart backtrack path."""
import os
import sys
import copy

import numpy as np


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

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


def test_confirmed_only_raw_prefix_recovery_keeps_confirmation_state_separate(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK", "0")
    tracker = GlobalLitterTracker(distance_threshold=250, fps=10)
    try:
        tracker._record_raw_litter_frame([[68, 38, 72, 42, 0.8]], 4)
        tracker._record_raw_litter_frame([[58, 48, 62, 52, 0.8]], 5)
        tracker._record_raw_litter_frame(
            [[48, 58, 52, 62, 0.9], [180, 180, 190, 190, 0.99]], 6
        )

        history, frames, boxes, confidences, recovered = (
            tracker._recover_raw_litter_prefix(
                [(40.0, 70.0), (30.0, 80.0)],
                [7, 8],
                [(38, 68, 42, 72), (28, 78, 32, 82)],
                [0.9, 0.9],
            )
        )

        assert frames == [4, 5, 6, 7, 8]
        assert recovered == [4, 5, 6]
        assert history[:3] == [(70.0, 40.0), (60.0, 50.0), (50.0, 60.0)]
        assert boxes[0] == (68.0, 38.0, 72.0, 42.0)
        assert confidences[0] == 0.8
        # Raw observations alone never enter active_litters/confirmation.
        tracker.update(
            [], [], frame_index=9,
            raw_detected_litters=[[20, 20, 30, 30, 0.99]],
        )
        assert tracker.active_litters == {}
    finally:
        tracker.close()


def test_raw_prefix_has_explicit_runtime_rollback(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK", "0")
    monkeypatch.setenv("SMART_BACKTRACK_RAW_PREFIX", "0")
    tracker = GlobalLitterTracker(distance_threshold=250, fps=10)
    try:
        assert tracker._raw_prefix_enabled is False
    finally:
        tracker.close()


def _submitted_lineage_task(monkeypatch, litter_data, current_source="detector"):
    monkeypatch.setenv("SMART_BACKTRACK", "1")
    tracker = GlobalLitterTracker(distance_threshold=250, fps=10)
    tracker._record_actor_frame(
        [_actor("person", 1, (0, 0, 20, 40))], frame_index=8
    )
    submitted = tracker._submit_backward_resolution(
        litter_id=4,
        litter_data=litter_data,
        current_bbox=(28, 78, 32, 82, 0.9),
        current_centroid=(30.0, 80.0),
        confirm_frame=8,
        current_source=current_source,
    )
    assert submitted is True
    return tracker, tracker._smart_tasks[4]


def test_detector_history_keeps_compatible_and_exact_sources(monkeypatch):
    tracker, task = _submitted_lineage_task(monkeypatch, {
        "birth_frame": 7,
        "history": [(40.0, 70.0), (30.0, 80.0)],
        "history_frames": [7, 8],
        "history_boxes": [(38, 68, 42, 72), (28, 78, 32, 82)],
        "history_confidences": [0.8, 0.9],
        "history_sources": ["detector", "detector"],
    })
    try:
        assert task["history_sources"] == ["accepted_tracker", "accepted_tracker"]
        assert task["history_provenance"] == ["detector", "detector"]
        assert all(
            row["independent_measurement"] for row in task["history_lineage"]
        )
    finally:
        tracker.close()


def test_visual_bridge_and_raw_prefix_keep_aligned_lineage(monkeypatch):
    tracker, task = _submitted_lineage_task(monkeypatch, {
        "birth_frame": 7,
        "history": [(40.0, 70.0), (30.0, 80.0)],
        "history_frames": [7, 8],
        "history_boxes": [(38, 68, 42, 72), (28, 78, 32, 82)],
        "history_confidences": [0.8, 0.9],
        # Confirmation happens before the current source is copied back into
        # active_litters, so submission must append it in lockstep.
        "history_sources": ["detector"],
    }, current_source="visual_bridge")
    try:
        # Insert a raw detector prefix and resubmit as a new immutable snapshot.
        tracker._record_raw_litter_frame([[48, 58, 52, 62, 0.7]], 6)
        assert tracker._submit_backward_resolution(
            litter_id=5,
            litter_data={
                "birth_frame": 7,
                "history": [(40.0, 70.0), (30.0, 80.0)],
                "history_frames": [7, 8],
                "history_boxes": [(38, 68, 42, 72), (28, 78, 32, 82)],
                "history_confidences": [0.8, 0.9],
                "history_sources": ["detector"],
            },
            current_bbox=(28, 78, 32, 82, 0.9),
            current_centroid=(30.0, 80.0),
            confirm_frame=8,
            current_source="visual_bridge",
        ) is True
        task = tracker._smart_tasks[5]
        lengths = {
            len(task[name]) for name in (
                "history", "history_frames", "history_boxes",
                "history_confidences", "history_sources",
                "history_provenance", "history_lineage",
            )
        }
        assert lengths == {3}
        assert task["history_frames"] == [6, 7, 8]
        assert task["history_sources"] == [
            "raw_rtdetr_recovered", "accepted_tracker", "accepted_tracker"
        ]
        assert task["history_provenance"] == [
            "raw_rtdetr_recovered", "detector", "visual_bridge"
        ]
        raw, detector, bridge = task["history_lineage"]
        assert raw["independent_measurement"] is False
        assert detector["independent_measurement"] is True
        assert bridge["independent_measurement"] is False
        assert bridge["parent_observation_id"] == detector["observation_id"]
        assert bridge["independence_group_id"] == detector["independence_group_id"]
    finally:
        tracker.close()


def test_legacy_missing_sources_are_not_claimed_as_detector_evidence(monkeypatch):
    tracker, task = _submitted_lineage_task(monkeypatch, {
        "birth_frame": 7,
        "history": [(40.0, 70.0), (30.0, 80.0)],
        "history_frames": [7, 8],
        "history_boxes": [(38, 68, 42, 72), (28, 78, 32, 82)],
        "history_confidences": [0.8, 0.9],
    })
    try:
        assert task["history_sources"] == ["accepted_tracker", "accepted_tracker"]
        assert task["history_provenance"] == ["legacy_unknown", "legacy_unknown"]
        assert not any(
            row["independent_measurement"] for row in task["history_lineage"]
        )
    finally:
        tracker.close()


def test_duplicate_detector_identity_is_not_counted_twice():
    lineage = GlobalLitterTracker._litter_observation_lineage(
        9, [4, 4], ["detector", "detector"]
    )
    assert lineage[0]["independent_measurement"] is True
    assert lineage[1]["independent_measurement"] is False
    assert lineage[1]["independence_group_id"] == lineage[0]["independence_group_id"]


def test_partial_legacy_source_array_fails_closed(monkeypatch):
    monkeypatch.setenv("SMART_BACKTRACK", "1")
    tracker = GlobalLitterTracker(distance_threshold=250, fps=10)
    try:
        tracker._record_actor_frame(
            [_actor("person", 1, (0, 0, 20, 40))], frame_index=8
        )
        assert tracker._submit_backward_resolution(
            litter_id=4,
            litter_data={
                "birth_frame": 6,
                "history": [(50, 60), (40, 70), (30, 80)],
                "history_frames": [6, 7, 8],
                "history_boxes": [
                    (48, 58, 52, 62), (38, 68, 42, 72), (28, 78, 32, 82)
                ],
                "history_confidences": [0.7, 0.8, 0.9],
                "history_sources": ["detector"],
            },
            current_bbox=(28, 78, 32, 82, 0.9),
            current_centroid=(30, 80),
            confirm_frame=8,
        ) is False
        assert 4 not in tracker._smart_tasks
    finally:
        tracker.close()


def test_resolver_ignores_phase0_provenance_and_lineage_fields():
    task = {
        "litter_id": 2,
        "fps": 10.0,
        "birth_frame": 15,
        "confirm_frame": 18,
        "history": [(400, 150), (410, 160), (420, 175), (430, 195)],
        "history_frames": [15, 16, 17, 18],
        "history_confidences": [0.9, 0.8, 0.7, 0.6],
        "actor_frames": [
            {
                "frame_index": frame,
                "actors": [
                    _actor("person", 1, (390, 80, 450, 220)),
                    _actor("vehicle", 7, (350, 150, 500, 260)),
                ],
            }
            for frame in range(10, 19)
        ],
    }
    enriched = copy.deepcopy(task)
    enriched["history_sources"] = ["accepted_tracker"] * 4
    enriched["history_provenance"] = [
        "detector", "visual_bridge", "visual_bridge", "detector"
    ]
    enriched["history_lineage"] = GlobalLitterTracker._litter_observation_lineage(
        2, task["history_frames"], enriched["history_provenance"]
    )

    baseline = SmartBacktrackResolver(fps=10).resolve_task(task)
    candidate = SmartBacktrackResolver(fps=10).resolve_task(enriched)

    assert candidate.route_id == baseline.route_id
    assert candidate.person_key == baseline.person_key
    assert candidate.vehicle_key == baseline.vehicle_key
    assert [
        (route.route_id, route.cost) for route in candidate.routes
    ] == [
        (route.route_id, route.cost) for route in baseline.routes
    ]


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


def test_confirm_frame_does_not_change_release_window_or_direction():
    config = SmartBacktrackConfig(
        max_back_frames=12,
        release_window_prior_weight=0.5,
    )
    resolver = SmartBacktrackResolver(fps=10, config=config)
    base = {
        "birth_frame": 20,
        "history": [(100.0, 100.0), (110.0, 100.0), (120.0, 100.0)],
        "history_frames": [20, 22, 24],
        "history_confidences": [0.9, 0.9, 0.9],
        "fps": 10.0,
    }

    early = resolver._release_hypotheses({**base, "confirm_frame": 24})
    late = resolver._release_hypotheses({**base, "confirm_frame": 40})

    assert [item.frame_index for item in early] == [
        item.frame_index for item in late
    ]
    assert [item.prior_cost for item in early] == [
        item.prior_cost for item in late
    ]
    assert early[0].zero_cost_window_start_frame == 18
    assert early[0].source_direction_uv == (-1.0, 0.0)


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
        release_diagnostic = sidecar["release_hypotheses"][0]
        assert release_diagnostic["observation_gap_frames"] == 1
        assert release_diagnostic["zero_cost_window_start_frame"] == 8
        assert release_diagnostic["zero_cost_window_end_frame"] == 10
        assert "window_prior_cost" in release_diagnostic
        assert "direction_consistency" in release_diagnostic
        assert release_diagnostic["search_truncated"] is False
        assert release_diagnostic["truncation_reason"] is None
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
