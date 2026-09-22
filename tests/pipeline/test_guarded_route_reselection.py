from dataclasses import replace

import pytest

from pipeline.backtrack.flow import COST_SCALE, Assignment, RouteCandidate
from pipeline.backtrack.route_reselection import (
    GuardedReselectionConfig,
    apply_guarded_reselection,
    attach_mask_temporal_evidence,
    compact_mask_polygon,
)
from pipeline.backtrack.resolver import SmartBacktrackConfig, SmartBacktrackResolver


def _route(route_id, cost, route_type="direct_vehicle", person=None,
           vehicle=("vehicle", 1), model="ballistic", ac=None, bc=None,
           mask=None):
    metadata = {
        "release_model": model,
        "release_frame": 10,
        "release_point": [0.0, 0.0],
        "release_velocity": [0.0, 0.0],
        "costs": {
            "BA": None,
            "AC": {"raw_features": ac or {}},
            "BC": {"raw_features": bc or {}},
        },
    }
    if mask is not None:
        metadata["mask_temporal_evidence"] = mask
    return RouteCandidate(
        route_id=route_id,
        cost=cost,
        route_type=route_type,
        person_key=person,
        vehicle_key=vehicle,
        metadata=metadata,
    )


def _assignment(route):
    return Assignment(
        litter_id=7,
        route=route,
        total_cost=route.cost,
        scaled_cost=round(route.cost * COST_SCALE),
        margin_to_second=None,
    )


def _select(selected, routes, **config):
    assignment, output_routes = apply_guarded_reselection(
        7,
        _assignment(selected),
        routes,
        GuardedReselectionConfig(**config),
    )
    return assignment, output_routes


def _mask(offsets, history):
    return {
        "release_signed_by_offset": {str(key): value for key, value in offsets.items()},
        "history_signed": history,
    }


def _seconds_mask(pre, release, history, *, pre_min=1, release_min=1):
    return {
        "schema": "mask-temporal-route-evidence/v2",
        "pre_signed": pre,
        "release_signed": release,
        "pre_min_observations": pre_min,
        "release_min_observations": release_min,
        "history_signed": history,
    }


def test_disabled_and_null_routes_are_never_reselected():
    selected = _route("selected", 1.0, bc={"boundary_depth": 1.0})
    alternative = _route("alternative", 1.1, vehicle=("vehicle", 2),
                         bc={"boundary_depth": 0.0})
    assignment, _ = _select(selected, [selected, alternative], enabled=False)
    assert assignment.route.route_id == "selected"

    null = RouteCandidate("null", 7.0, route_type="null", vehicle_key=None)
    assignment, _ = _select(null, [null, alternative])
    assert assignment.route.route_id == "null"


def test_production_defaults_and_rollback_switches(monkeypatch):
    monkeypatch.delenv("SMART_BACKTRACK_DIRECT_VEHICLE_COST", raising=False)
    monkeypatch.delenv("SMART_BACKTRACK_GUARDED_RESELECT", raising=False)
    monkeypatch.delenv("SMART_BACKTRACK_MASK_RESELECT", raising=False)
    monkeypatch.delenv("SMART_BACKTRACK_MAX_RELEASE_BACK_SEC", raising=False)
    monkeypatch.delenv("SMART_BACKTRACK_RELEASE_SOFT_SEC", raising=False)
    monkeypatch.delenv("SMART_BACKTRACK_MAX_BACK_FRAMES", raising=False)
    config = SmartBacktrackConfig.from_env(fps=10)
    assert config.direct_vehicle_penalty == 1.1
    assert config.use_guarded_route_reselection is True
    assert config.use_mask_temporal_reselection is True
    assert config.max_release_back_seconds == 1.0
    assert config.release_soft_seconds == 0.25
    assert config.max_back_frames == 10

    assert SmartBacktrackConfig.from_env(fps=30).max_back_frames == 30

    monkeypatch.setenv("SMART_BACKTRACK_GUARDED_RESELECT", "0")
    monkeypatch.setenv("SMART_BACKTRACK_MASK_RESELECT", "0")
    config = SmartBacktrackConfig.from_env(fps=10)
    assert config.use_guarded_route_reselection is False
    assert config.use_mask_temporal_reselection is False


def test_global_solver_applies_guarded_reselection():
    selected = _route("selected", 1.0, bc={"boundary_depth": 1.0})
    alternative = _route("alternative", 1.1, vehicle=("vehicle", 2),
                         bc={"boundary_depth": 0.0})
    resolver = SmartBacktrackResolver(
        fps=10,
        config=SmartBacktrackConfig(use_mask_temporal_reselection=False),
    )
    resolution = resolver.solve_routes({7: [selected, alternative]})[7]
    assert resolution.route_id == "alternative"
    assert resolution.components["guarded_reselection"]["steps"] == [{
        "rule": "ballistic_boundary", "from": "selected", "to": "alternative"
    }]


@pytest.mark.parametrize(
    ("selected", "alternative", "expected_rule"),
    [
        (
            _route("selected", 1.0, bc={"boundary_depth": 1.0}),
            _route("alternative", 1.1, vehicle=("vehicle", 2),
                   bc={"boundary_depth": 0.0}),
            "ballistic_boundary",
        ),
        (
            _route("selected", 1.0, route_type="person_vehicle",
                   person=("person", 1), ac={"endpoint_proximity": 0.30}),
            _route("alternative", 1.1, route_type="person_vehicle",
                   person=("person", 2), ac={"endpoint_proximity": 0.10}),
            "ac_endpoint",
        ),
        (
            _route("selected", 1.0, route_type="person_vehicle",
                   person=("person", 1), bc={"quality": 0.90}),
            _route("alternative", 1.2, route_type="person_vehicle",
                   person=("person", 2), bc={"quality": 0.01}),
            "bc_quality",
        ),
        (
            _route("selected", 1.0, route_type="person_vehicle",
                   person=("person", 1),
                   ac={"endpoint_proximity": 0.20, "overlap": 0.05}),
            _route("alternative", 1.5),
            "direct_person",
        ),
        (
            _route("selected", 1.0,
                   bc={"reverse_direction": 0.20, "exit_deficit": 0.30,
                       "relative_motion_deficit": 0.20}),
            _route("alternative", 2.0, vehicle=("vehicle", 2),
                   bc={"reverse_direction": 0.01, "exit_deficit": 0.01,
                       "relative_motion_deficit": 0.01}),
            "motion_consistency",
        ),
    ],
)
def test_each_guard_can_reselect_an_existing_valid_route(
    selected, alternative, expected_rule
):
    assignment, routes = _select(selected, [selected, alternative])
    assert assignment.route.route_id == "alternative"
    assert assignment.route.metadata["guarded_reselection"]["steps"][-1][
        "rule"
    ] == expected_rule
    assert {route.route_id for route in routes} == {"selected", "alternative"}


def test_missing_guard_evidence_fails_closed():
    selected = _route("selected", 1.0, bc={"boundary_depth": 1.0})
    alternative = _route("alternative", 1.1, vehicle=("vehicle", 2), bc={})
    assignment, _ = _select(selected, [selected, alternative])
    assert assignment.route.route_id == "selected"


def test_ballistic_mask_crossing_rule_matches_frozen_policy():
    selected = _route(
        "selected", 1.0,
        bc={"direct_distance": 0.0},
        mask=_mask({-1: -0.2, 0: -0.1, 1: -0.1}, [-0.2, -0.1]),
    )
    alternative = _route(
        "alternative", 1.2, vehicle=("vehicle", 2),
        bc={"direct_distance": 0.0},
        mask=_mask({-2: -0.4, -1: -0.2, 0: 0.2, 1: 0.4}, [-0.8, -0.7]),
    )
    assignment, _ = _select(selected, [selected, alternative])
    assert assignment.route.route_id == "alternative"
    assert assignment.route.metadata["guarded_reselection"]["steps"][0][
        "rule"
    ] == "mask_temporal"


def test_seconds_mask_crossing_requires_fresh_pre_and_release_observations():
    selected = _route(
        "selected", 1.0,
        bc={"direct_distance": 0.0},
        mask=_seconds_mask([-0.1], [-0.2], [-0.2, -0.1]),
    )
    alternative = _route(
        "alternative", 1.2, vehicle=("vehicle", 2),
        bc={"direct_distance": 0.0},
        mask=_seconds_mask([-0.3], [0.3], [-0.8, -0.7]),
    )
    assignment, _ = _select(selected, [selected, alternative])
    assert assignment.route.route_id == "alternative"

    missing_pre = replace(
        alternative,
        metadata={
            **alternative.metadata,
            "mask_temporal_evidence": _seconds_mask([], [0.3], [-0.8, -0.7]),
        },
    )
    assignment, _ = _select(selected, [selected, missing_pre])
    assert assignment.route.route_id == "selected"


def test_mask_evidence_is_computed_only_from_observed_segmentation_contours():
    routes = [
        _route("inside", 1.0, vehicle=("vehicle", 1)),
        _route("outside", 1.1, vehicle=("vehicle", 2)),
    ]
    routes = [replace(
        route,
        metadata={**route.metadata, "release_point": [5.0, 5.0]},
    ) for route in routes]
    actors = [
        {
            "actor_key": ["vehicle", 1], "box": [0, 0, 10, 10],
            "observed": True, "source": "seg_track",
            "mask_contour_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
        },
        {
            "actor_key": ["vehicle", 2], "box": [20, 20, 30, 30],
            "observed": False, "source": "cache",
            "mask_contour_xy": [[20, 20], [30, 20], [30, 30], [20, 30]],
        },
    ]
    task = {
        "fps": 10.0,
        "history": [[5.0, 5.0]],
        "history_frames": [10],
        "actor_frames": [{"frame_index": 10, "actors": actors}],
    }
    output = attach_mask_temporal_evidence(routes, task, releases=[])
    inside = output[0].metadata["mask_temporal_evidence"]
    outside = output[1].metadata["mask_temporal_evidence"]
    assert inside["schema"] == "mask-temporal-route-evidence/v2"
    assert inside["release_signed_by_offset"]["0"] > 0.0
    assert inside["release_observed_count"] == 1
    assert inside["pre_observed_count"] == 0
    assert inside["history_observed_count"] == 1
    assert outside["release_signed_by_offset"] == {}
    assert outside["history_observed_count"] == 0


@pytest.mark.parametrize(
    ("fps", "expected_pre_span", "expected_release_span"),
    [(10.0, 2, 1), (12.0, 2, 1), (30.0, 6, 3)],
)
def test_mask_windows_use_seconds_and_bound_fresh_observation_counts(
    fps, expected_pre_span, expected_release_span
):
    route = replace(
        _route("vehicle", 1.0),
        metadata={
            **_route("vehicle", 1.0).metadata,
            "release_point": [5.0, 5.0],
        },
    )
    actor_frames = []
    for offset in range(-expected_pre_span, expected_release_span + 1):
        actor_frames.append({
            "frame_index": 10 + offset,
            "actors": [{
                "actor_key": ["vehicle", 1],
                "box": [0, 0, 10, 10],
                "observed": True,
                "source": "seg_track",
                "mask_contour_xy": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }],
        })
    evidence = attach_mask_temporal_evidence(
        [route],
        {
            "fps": fps,
            "history": [],
            "history_frames": [],
            "actor_frames": actor_frames,
        },
        releases=[],
    )[0].metadata["mask_temporal_evidence"]
    assert evidence["pre_frame_span"] == expected_pre_span
    assert evidence["release_frame_span"] == expected_release_span
    assert evidence["pre_observed_count"] == 2
    assert evidence["release_observed_count"] == 2
    assert [row["frame_offset"] for row in evidence["pre_samples"]] == [-1, -2]
    assert [row["frame_offset"] for row in evidence["release_samples"]] == [0, 1]


def test_invalid_mask_window_fails_closed_at_configuration_boundary():
    with pytest.raises(ValueError):
        GuardedReselectionConfig(
            mask_pre_min_observations=2,
            mask_pre_max_observations=1,
        )


def test_compact_polygon_is_bounded_and_rejects_degenerate_input():
    polygon = [(float(index), float(index % 17)) for index in range(2_000)]
    compact = compact_mask_polygon(polygon)
    assert compact is not None
    assert len(compact) == 512
    assert compact_mask_polygon([[0, 0], [1, 1], [2, 2]]) is None
