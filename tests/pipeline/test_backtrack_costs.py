import numpy as np
import pytest

from pipeline.backtrack.costs import (
    ActorObservation,
    BacktrackCostConfig,
    compute_c_ac,
    compute_c_ba,
    compute_c_bc,
)
from pipeline.backtrack.trajectory import (
    ReleaseHypothesis,
    build_release_hypotheses,
    fit_ballistic_trajectory,
)
from pipeline.backtrack.kalman import TrackMeasurement, smooth_tracklet


def _physical_trajectory(fps, duration_seconds=1.0):
    frames = np.arange(0, int(round(duration_seconds * fps)) + 1)
    t = frames / float(fps)
    points = np.column_stack(
        [
            50.0 + 30.0 * t,
            40.0 - 20.0 * t + 10.0 * t * t,
        ]
    )
    return frames, points


def _release(frame, mean, covariance=None, velocity=(20.0, 5.0)):
    return ReleaseHypothesis(
        frame_index=frame,
        mean_uv=np.asarray(mean, dtype=float),
        covariance_uv=(
            np.eye(2, dtype=float) * 4.0
            if covariance is None
            else np.asarray(covariance, dtype=float)
        ),
        velocity_uv=np.asarray(velocity, dtype=float),
        model="ballistic",
        prior_cost=0.0,
    )


def test_ballistic_fit_is_fps_invariant_in_seconds():
    frames_10, points_10 = _physical_trajectory(10)
    frames_20, points_20 = _physical_trajectory(20)
    model_10 = fit_ballistic_trajectory(points_10, frames_10, fps=10)
    model_20 = fit_ballistic_trajectory(points_20, frames_20, fps=20)

    mean_10, _, velocity_10 = model_10.predict(-5)   # t = -0.5 s
    mean_20, _, velocity_20 = model_20.predict(-10)  # t = -0.5 s

    np.testing.assert_allclose(mean_10, mean_20, atol=1e-8)
    np.testing.assert_allclose(velocity_10, velocity_20, atol=1e-8)


def test_reverse_extrapolation_covariance_grows_away_from_observations():
    frames, points = _physical_trajectory(10)
    model = fit_ballistic_trajectory(
        points + np.column_stack([np.sin(frames), np.cos(frames)]) * 0.4,
        frames,
        fps=10,
        sigma_floor_px=0.5,
    )

    _, covariance_middle, _ = model.predict(5)
    _, covariance_far, _ = model.predict(-20)

    assert np.trace(covariance_far) > np.trace(covariance_middle) * 2.0


def test_low_litter_detection_confidence_increases_release_uncertainty():
    frames, points = _physical_trajectory(10)
    high = fit_ballistic_trajectory(
        points, frames, fps=10, confidences=np.ones(len(frames)) * 0.95
    )
    low = fit_ballistic_trajectory(
        points, frames, fps=10, confidences=np.ones(len(frames)) * 0.2
    )

    _, high_covariance, _ = high.predict(-10)
    _, low_covariance, _ = low.predict(-10)

    assert np.trace(low_covariance) > np.trace(high_covariance)


def test_release_hypotheses_fall_back_to_birth_when_fit_is_impossible():
    hypotheses = build_release_hypotheses(
        points_uv=[(12.0, 34.0)],
        frame_indices=[20],
        birth_frame=20,
        max_back_frames=10,
        fps=10,
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].frame_index == 20
    assert hypotheses[0].model == "birth_fallback"
    assert hypotheses[0].prior_cost >= 6.0
    np.testing.assert_allclose(hypotheses[0].mean_uv, [12.0, 34.0])


def test_two_point_release_uses_gap_window_and_computational_guard():
    hypotheses = build_release_hypotheses(
        points_uv=[(100.0, 200.0), (110.0, 220.0)],
        frame_indices=[20, 22],
        birth_frame=20,
        max_back_frames=10,
        fps=10,
        two_point_max_back_seconds=0.3,
        two_point_prior_cost=1.0,
    )

    # The legacy 0.3-second value is no longer a physical cutoff. The 10-frame
    # max_back guard keeps all hypotheses available for later cost comparison.
    assert [item.frame_index for item in hypotheses] == list(range(10, 21))
    assert all(item.model == "constant_velocity_2point" for item in hypotheses)
    np.testing.assert_allclose(hypotheses[-1].mean_uv, [100.0, 200.0])
    np.testing.assert_allclose(hypotheses[0].mean_uv, [50.0, 100.0])
    np.testing.assert_allclose(hypotheses[-1].velocity_uv, [50.0, 100.0])
    assert np.trace(hypotheses[0].covariance_uv) > np.trace(
        hypotheses[-1].covariance_uv
    )
    by_frame = {item.frame_index: item for item in hypotheses}
    assert by_frame[18].prior_cost == by_frame[20].prior_cost == 1.0
    assert by_frame[17].prior_cost > 1.0
    assert by_frame[17].window_prior_cost > 0.0
    assert by_frame[20].observation_gap_frames == 2
    assert by_frame[20].zero_cost_window_start_frame == 18
    assert by_frame[20].zero_cost_window_end_frame == 20
    assert by_frame[20].source_direction_uv == pytest.approx(
        (-1.0 / np.sqrt(5.0), -2.0 / np.sqrt(5.0))
    )
    assert by_frame[20].search_truncated is True
    assert by_frame[20].truncation_reason == "max_back_frames_computational_guard"


def test_two_point_release_is_fps_invariant_in_seconds():
    low = build_release_hypotheses(
        [(100.0, 200.0), (110.0, 220.0)], [20, 22],
        birth_frame=20, max_back_frames=30, fps=10,
    )
    high = build_release_hypotheses(
        [(100.0, 200.0), (110.0, 220.0)], [60, 66],
        birth_frame=60, max_back_frames=90, fps=30,
    )

    np.testing.assert_allclose(low[0].mean_uv, high[0].mean_uv)
    np.testing.assert_allclose(low[0].velocity_uv, high[0].velocity_uv)
    np.testing.assert_allclose(low[0].covariance_uv, high[0].covariance_uv)


def test_ballistic_release_window_may_extend_after_detector_birth():
    hypotheses = build_release_hypotheses(
        points_uv=[(100.0, 200.0), (105.0, 180.0), (110.0, 190.0), (115.0, 220.0)],
        frame_indices=[20, 21, 22, 23],
        birth_frame=20,
        max_back_frames=2,
        fps=10,
        max_forward_release_seconds=0.5,
    )

    assert [item.frame_index for item in hypotheses] == [18, 19, 20, 21, 22, 23]
    assert hypotheses[2].prior_cost == 0.0
    # Forward frames are observed-airborne candidates, so they keep the
    # seconds-based prior rather than the reverse observation-gap denominator.
    assert hypotheses[-1].prior_cost == pytest.approx(0.35 * 0.3)


def test_three_point_direction_uses_b0_b1_b2_and_reports_consistency():
    hypotheses = build_release_hypotheses(
        points_uv=[(10.0, 10.0), (12.0, 10.0), (14.0, 10.0)],
        frame_indices=[20, 22, 24],
        birth_frame=20,
        max_back_frames=4,
        fps=10,
    )

    assert hypotheses
    assert hypotheses[0].direction_consistency == pytest.approx(1.0)
    assert hypotheses[0].source_direction_uv == pytest.approx((-1.0, 0.0))
    assert hypotheses[0].zero_cost_window_start_frame == 18


def test_release_window_uses_earliest_recovered_observation_not_tracker_birth():
    hypotheses = build_release_hypotheses(
        points_uv=[(10.0, 10.0), (12.0, 10.0), (14.0, 11.0)],
        frame_indices=[18, 20, 22],
        birth_frame=20,
        max_back_frames=4,
        fps=10,
    )

    assert hypotheses
    assert hypotheses[0].observation_gap_frames == 2
    assert hypotheses[0].zero_cost_window_start_frame == 16
    assert hypotheses[0].zero_cost_window_end_frame == 18


def test_gap_window_keeps_older_release_with_soft_penalty():
    hypotheses = build_release_hypotheses(
        points_uv=[(100.0, 100.0), (110.0, 100.0)],
        frame_indices=[20, 21],
        birth_frame=20,
        max_back_frames=12,
        fps=10,
        window_prior_weight=0.5,
    )
    by_frame = {item.frame_index: item for item in hypotheses}

    assert 10 in by_frame  # B0 - 1 second is retained at 10 FPS.
    assert by_frame[19].window_prior_cost == 0.0
    assert by_frame[20].window_prior_cost == 0.0
    assert by_frame[18].window_prior_cost == pytest.approx(0.5)
    assert by_frame[10].window_prior_cost == pytest.approx(4.5)


def test_c_bc_emits_zero_weight_causality_diagnostics_without_changing_total():
    vehicle = [ActorObservation(
        cls_name="vehicle",
        track_id=2,
        frame_index=10,
        bbox=(80.0, 80.0, 140.0, 140.0),
    )]
    release = _release(10, (110.0, 110.0))

    without = compute_c_bc([release], vehicle, fps=10)
    with_context = compute_c_bc(
        [release], vehicle, fps=10,
        litter_last_point=(180.0, 180.0),
        litter_last_frame=11,
    )

    assert with_context.total == pytest.approx(without.total)
    assert with_context.weights["reverse_direction"] == 0.0
    assert with_context.weights["exit_deficit"] == 0.0
    assert with_context.weights["relative_motion_deficit"] == 0.0
    assert set((
        "reverse_direction", "exit_deficit", "relative_motion_deficit"
    )).issubset(with_context.raw_features)


def test_c_bc_boundary_depth_penalizes_deep_bbox_containment_only_when_enabled():
    vehicle = [ActorObservation(
        cls_name="vehicle",
        track_id=2,
        frame_index=10,
        bbox=(0.0, 0.0, 200.0, 100.0),
    )]
    deep_release = _release(10, (100.0, 50.0))
    edge_release = _release(10, (-5.0, 50.0))
    baseline_deep = compute_c_bc([deep_release], vehicle, fps=10)
    baseline_edge = compute_c_bc([edge_release], vehicle, fps=10)
    weighted_config = BacktrackCostConfig(
        bc_weights={
            **BacktrackCostConfig().bc_weights,
            "boundary_depth": 0.5,
        }
    )
    weighted_deep = compute_c_bc(
        [deep_release], vehicle, fps=10, cost_config=weighted_config
    )
    weighted_edge = compute_c_bc(
        [edge_release], vehicle, fps=10, cost_config=weighted_config
    )

    assert baseline_deep.raw_features["boundary_depth"] == pytest.approx(1.0)
    assert baseline_edge.raw_features["boundary_depth"] == pytest.approx(0.0)
    assert baseline_deep.weights["boundary_depth"] == pytest.approx(0.0)
    assert weighted_deep.total == pytest.approx(baseline_deep.total + 0.5)
    assert weighted_edge.total == pytest.approx(baseline_edge.total)


def test_c_bc_production_gate_uses_unexpanded_bbox_and_diagonal_scale():
    vehicle = [ActorObservation(
        cls_name="vehicle",
        track_id=2,
        frame_index=10,
        bbox=(0.0, 0.0, 80.0, 60.0),  # diagonal = 100 px
    )]

    at_039 = compute_c_bc([_release(10, (119.0, 30.0))], vehicle, fps=10)
    at_041 = compute_c_bc([_release(10, (121.0, 30.0))], vehicle, fps=10)

    assert at_039.valid
    assert at_039.raw_features["direct_distance"] == pytest.approx(0.39)
    assert not at_041.valid
    assert at_041.components["minimum_normalized_distance"] == pytest.approx(0.41)


def test_c_bc_legacy_bbox_expansion_is_an_explicit_research_override():
    vehicle = [ActorObservation(
        cls_name="vehicle",
        track_id=2,
        frame_index=10,
        bbox=(0.0, 0.0, 80.0, 60.0),
    )]
    release = _release(10, (95.0, 30.0))

    production = compute_c_bc(
        [release], vehicle, fps=10, normalized_distance_gate=0.10
    )
    legacy = compute_c_bc(
        [release],
        vehicle,
        fps=10,
        normalized_distance_gate=0.10,
        vehicle_bbox_expand_x_ratio=0.18,
        vehicle_bbox_expand_y_ratio=0.15,
    )

    assert not production.valid
    assert legacy.valid
    assert legacy.raw_features["direct_distance"] == pytest.approx(0.006)


def test_observation_time_soft_penalty_uses_stricter_seconds_or_frames_scale():
    vehicle = [ActorObservation(
        cls_name="vehicle",
        track_id=2,
        frame_index=14,
        bbox=(0.0, 0.0, 80.0, 60.0),
    )]
    release = _release(10, (40.0, 30.0))

    hybrid = compute_c_bc([release], vehicle, fps=30)
    seconds_only = compute_c_bc(
        [release], vehicle, fps=30, max_observation_gap_frames=None
    )

    assert hybrid.valid
    assert seconds_only.valid
    # At 30 FPS, 4 frames is 0.133 s: seconds fraction=.533, frame
    # fraction=1.333. max() chooses frames, then kappa=4 penalizes excess.
    assert hybrid.raw_features["time"] == pytest.approx(
        4 / 3 + 4 * (1 / 3) ** 2
    )
    assert hybrid.components["time"] == pytest.approx(
        0.35 * hybrid.raw_features["time"]
    )
    assert hybrid.components["release_prior"] == pytest.approx(0.0)
    assert seconds_only.raw_features["time"] == pytest.approx((4 / 30) / .25)

    low_fps_vehicle = [ActorObservation(
        cls_name="vehicle",
        track_id=2,
        frame_index=13,
        bbox=(0.0, 0.0, 80.0, 60.0),
    )]
    low_fps = compute_c_bc([release], low_fps_vehicle, fps=10)
    assert low_fps.valid
    # At 10 FPS, seconds is stricter: z=max(.30/.25, 3/3)=1.2.
    assert low_fps.raw_features["time"] == pytest.approx(1.2 + 4 * .2 ** 2)


def test_legacy_observation_time_hard_gate_remains_replayable():
    vehicle = [ActorObservation(
        cls_name="vehicle",
        track_id=2,
        frame_index=14,
        bbox=(0.0, 0.0, 80.0, 60.0),
    )]
    release = _release(10, (40.0, 30.0))
    legacy = BacktrackCostConfig(observation_time_cost_mode="hard")

    hybrid = compute_c_bc([release], vehicle, fps=30, cost_config=legacy)

    assert not hybrid.valid
    assert hybrid.reject_reason == "direct_vehicle_gate_failed"


def test_c_ba_uses_upper_body_release_zone_not_person_footpoint():
    # Release is near the hand/torso and 110 px away from the footpoint.
    person = ActorObservation(
        cls_name="person",
        track_id=7,
        frame_index=50,
        bbox=(90.0, 50.0, 130.0, 200.0),
        confidence=0.95,
    )
    release = _release(50, (110.0, 90.0))

    cost = compute_c_ba([release], [person], fps=10)

    assert np.linalg.norm(release.mean_uv - person.footpoint) > 100.0
    assert cost.valid
    assert cost.best_release_frame == 50
    assert cost.components["release_distance"] == 0.0


def test_c_ba_rejects_release_with_unbounded_uncertainty():
    person = ActorObservation(
        cls_name="person",
        track_id=7,
        frame_index=50,
        bbox=(90.0, 50.0, 130.0, 200.0),
    )
    release = _release(
        50,
        (110.0, 90.0),
        covariance=np.eye(2, dtype=float) * 300.0 ** 2,
    )

    cost = compute_c_ba([release], [person], fps=10)

    assert not cost.valid
    assert cost.reject_reason == "release_uncertainty_too_large"
    assert np.isinf(cost.total)


def test_c_ba_rejects_actor_whose_kalman_state_is_unbounded():
    person = ActorObservation(
        cls_name="person",
        track_id=7,
        frame_index=50,
        bbox=(90.0, 50.0, 130.0, 200.0),
        covariance_uv=np.eye(2, dtype=float) * 300.0 ** 2,
        observed=False,
    )

    cost = compute_c_ba([_release(50, (110.0, 90.0))], [person], fps=10)

    assert not cost.valid
    assert cost.reject_reason == "release_uncertainty_too_large"


def test_c_ba_uncertainty_cannot_make_a_far_person_valid():
    person = ActorObservation(
        cls_name="person",
        track_id=7,
        frame_index=50,
        bbox=(90.0, 50.0, 130.0, 200.0),
    )
    low_uncertainty = _release(
        50, (330.0, 90.0), covariance=np.eye(2, dtype=float) * 2.0 ** 2
    )
    high_uncertainty = _release(
        50, (330.0, 90.0), covariance=np.eye(2, dtype=float) * 100.0 ** 2
    )

    low_cost = compute_c_ba([low_uncertainty], [person], fps=10)
    high_cost = compute_c_ba([high_uncertainty], [person], fps=10)

    assert not low_cost.valid
    assert not high_cost.valid
    assert low_cost.reject_reason == "person_release_gate_failed"
    assert high_cost.reject_reason == "person_release_gate_failed"


def test_c_ba_more_uncertainty_increases_cost_for_same_release():
    person = ActorObservation(
        cls_name="person",
        track_id=7,
        frame_index=50,
        bbox=(90.0, 50.0, 130.0, 200.0),
    )
    low = compute_c_ba(
        [_release(50, (110.0, 90.0), covariance=np.eye(2) * 2.0 ** 2)],
        [person],
        fps=10,
    )
    high = compute_c_ba(
        [_release(50, (110.0, 90.0), covariance=np.eye(2) * 100.0 ** 2)],
        [person],
        fps=10,
    )

    assert low.valid and high.valid
    assert high.total > low.total


def test_c_ba_uncertainty_never_improves_in_gate_boundary_match():
    person = ActorObservation(
        cls_name="person",
        track_id=7,
        frame_index=50,
        bbox=(90.0, 50.0, 130.0, 200.0),
    )
    point = (220.0, 90.0)  # 0.6 person-heights outside the release zone.
    costs = [
        compute_c_ba(
            [_release(50, point, covariance=np.eye(2) * std ** 2)],
            [person],
            fps=10,
        )
        for std in (2.0, 50.0, 100.0)
    ]

    assert all(cost.valid for cost in costs)
    assert costs[0].total < costs[1].total < costs[2].total


def test_non_finite_release_never_becomes_a_valid_cost():
    person = ActorObservation(
        cls_name="person",
        track_id=7,
        frame_index=50,
        bbox=(90.0, 50.0, 130.0, 200.0),
    )
    release = _release(50, (np.nan, 90.0))

    cost = compute_c_ba([release], [person], fps=10)

    assert not cost.valid
    assert np.isinf(cost.total)


def test_c_bc_direct_vehicle_route_has_a_hard_spatial_gate():
    vehicle = ActorObservation(
        cls_name="vehicle",
        track_id=3,
        frame_index=20,
        bbox=(0.0, 100.0, 160.0, 200.0),
    )
    near = compute_c_bc([_release(20, (150.0, 140.0))], [vehicle], fps=10)
    far = compute_c_bc([_release(20, (800.0, 140.0))], [vehicle], fps=10)

    assert near.valid
    assert not far.valid
    assert far.reject_reason == "direct_vehicle_gate_failed"


def test_valid_cost_cells_preserve_raw_weight_and_contribution():
    person = ActorObservation(
        cls_name="person",
        track_id=7,
        frame_index=50,
        bbox=(90.0, 50.0, 130.0, 200.0),
        confidence=0.8,
    )
    cost = compute_c_ba([_release(50, (150.0, 90.0))], [person], fps=10)

    assert cost.valid
    assert cost.raw_features
    assert cost.weights
    assert set(cost.components) == set(cost.raw_features) == set(cost.weights)
    for name, contribution in cost.components.items():
        assert np.isclose(
            contribution,
            cost.raw_features[name] * cost.weights[name],
        )
    assert np.isclose(cost.total, sum(cost.components.values()))


def test_c_ac_can_link_at_an_endpoint_after_person_walks_away():
    people = [
        ActorObservation("person", 1, 0, (40.0, 30.0, 80.0, 150.0)),
        ActorObservation("person", 1, 10, (400.0, 30.0, 440.0, 150.0)),
    ]
    vehicles = [
        ActorObservation("vehicle", 2, 0, (0.0, 60.0, 180.0, 170.0)),
        ActorObservation("vehicle", 2, 10, (0.0, 60.0, 180.0, 170.0)),
    ]

    cost = compute_c_ac(people, vehicles, fps=10)

    assert cost.valid
    assert cost.components["overlap"] < 0.8


def test_c_ac_rejects_one_frame_pass_by_between_far_endpoints():
    people = [
        ActorObservation("person", 1, 0, (400.0, 30.0, 440.0, 150.0)),
        ActorObservation("person", 1, 5, (40.0, 30.0, 80.0, 150.0)),
        ActorObservation("person", 1, 10, (400.0, 30.0, 440.0, 150.0)),
    ]
    vehicles = [
        ActorObservation("vehicle", 2, frame, (0.0, 60.0, 180.0, 170.0))
        for frame in (0, 5, 10)
    ]

    cost = compute_c_ac(people, vehicles, fps=10)

    assert not cost.valid
    assert cost.reject_reason == "person_vehicle_dwell_failed"


def test_c_ac_accepts_sustained_near_vehicle_evidence():
    people = [
        ActorObservation("person", 1, frame, (40.0, 30.0, 80.0, 150.0))
        for frame in (0, 2, 4)
    ]
    vehicles = [
        ActorObservation("vehicle", 2, frame, (0.0, 60.0, 180.0, 170.0))
        for frame in (0, 2, 4)
    ]

    cost = compute_c_ac(people, vehicles, fps=10)

    assert cost.valid


def test_c_ac_records_sync_gap_without_charging_a_time_cost():
    people = [
        ActorObservation(
            "person", 1, frame, (40.0, 30.0, 80.0, 150.0),
            evidence_frame_index=frame,
        )
        for frame in (0, 2)
    ]
    vehicles = [
        ActorObservation(
            "vehicle", 2, frame, (0.0, 60.0, 180.0, 170.0),
            evidence_frame_index=frame + 1,
        )
        for frame in (0, 2)
    ]

    cost = compute_c_ac(people, vehicles, fps=10)

    assert cost.valid
    assert cost.raw_features["time"] > 0.0
    assert cost.weights["time"] == pytest.approx(0.0)
    assert cost.components["time"] == pytest.approx(0.0)


def test_c_ac_dwell_must_be_contiguous_not_two_distant_points():
    people = [
        ActorObservation("person", 1, frame, (180.0, 30.0, 220.0, 150.0))
        for frame in (0, 100)
    ]
    vehicles = [
        ActorObservation("vehicle", 2, frame, (0.0, 60.0, 180.0, 170.0))
        for frame in (0, 100)
    ]

    cost = compute_c_ac(people, vehicles, fps=10)

    assert not cost.valid
    assert cost.reject_reason == "person_vehicle_dwell_failed"


def test_c_ac_predicted_only_person_cannot_accumulate_dwell():
    people = [
        ActorObservation(
            "person",
            1,
            frame,
            (40.0, 30.0, 80.0, 150.0),
            observed=False,
        )
        for frame in (0, 2, 4)
    ]
    vehicles = [
        ActorObservation("vehicle", 2, frame, (0.0, 60.0, 180.0, 170.0))
        for frame in (0, 2, 4)
    ]

    cost = compute_c_ac(people, vehicles, fps=10)

    assert not cost.valid
    assert cost.reject_reason == "person_vehicle_dwell_failed"


def test_actor_observation_adapts_smoothed_missing_frame():
    tracklet = smooth_tracklet(
        [
            TrackMeasurement(0, (0.0, 0.0, 20.0, 100.0), 0.9),
            TrackMeasurement(2, (4.0, 0.0, 24.0, 100.0), 0.9),
        ],
        class_name="person",
        track_id=9,
    )

    observation = ActorObservation.from_tracklet(tracklet, 1)

    assert observation.actor_key == ("person", 9)
    assert not observation.observed
    assert observation.source == "kalman_rts"
    assert observation.covariance_uv.shape == (2, 2)
