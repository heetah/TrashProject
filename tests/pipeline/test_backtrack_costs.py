import numpy as np

from pipeline.backtrack.costs import (
    ActorObservation,
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
