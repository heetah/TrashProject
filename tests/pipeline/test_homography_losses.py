import math

import numpy as np
import pytest

from scripts.pipeline.calibration import (
    CalibrationLossConfig,
    LocalMotionObservation,
    TrafficFlowCluster,
    VehicleTrackPoint,
    evaluate_calibration_losses,
    initial_normalized_image_homography,
)


IMAGE_SHAPE = (100, 200)
H0 = initial_normalized_image_homography(IMAGE_SHAPE)


def _point(index, xy, timestamp=None, quality=1.0):
    return VehicleTrackPoint(
        track_id=1,
        timestamp=float(index if timestamp is None else timestamp),
        frame_id=index,
        image_point=np.asarray(xy, dtype=float),
        velocity=np.zeros(2),
        quality_score=quality,
        mask_area=100.0,
        confidence=0.9,
    )


def _evaluate(points, homography=H0, **kwargs):
    return evaluate_calibration_losses(
        homography,
        {("vehicle", 1): tuple(points)},
        IMAGE_SHAPE,
        **kwargs,
    )


def test_constant_velocity_has_zero_motion_and_speed_loss_with_irregular_times():
    times = [0.0, 0.5, 1.5, 3.0]
    points = [_point(i, (10.0 + 8.0 * time, 50.0), time) for i, time in enumerate(times)]

    report = _evaluate(points)

    assert report.homography_valid
    assert report.evaluable
    assert report.component_losses["motion"] == pytest.approx(0.0, abs=1e-12)
    assert report.component_losses["speed"] == pytest.approx(0.0, abs=1e-12)
    assert report.component_losses["curvature"] == pytest.approx(0.0, abs=1e-12)


def test_curvature_penalizes_turn_discontinuity_not_a_smooth_curve():
    smooth = [
        _point(0, (30, 30)),
        _point(1, (40, 30)),
        _point(2, (49.66, 32.59)),
        _point(3, (58.32, 37.59)),
        _point(4, (65.39, 44.66)),
    ]
    kink = [
        _point(0, (30, 30)),
        _point(1, (40, 30)),
        _point(2, (50, 30)),
        _point(3, (50, 40)),
        _point(4, (60, 40)),
    ]

    smooth_loss = _evaluate(smooth).component_losses["curvature"]
    kink_loss = _evaluate(kink).component_losses["curvature"]

    assert smooth_loss is not None and kink_loss is not None
    assert smooth_loss < kink_loss


def test_abrupt_speed_jump_costs_more_than_constant_speed():
    constant = [_point(i, (20 + 5 * i, 40)) for i in range(5)]
    jump = [
        _point(0, (20, 40)),
        _point(1, (25, 40)),
        _point(2, (30, 40)),
        _point(3, (55, 40)),
        _point(4, (60, 40)),
    ]

    constant_loss = _evaluate(constant).component_losses["speed"]
    jump_loss = _evaluate(jump).component_losses["speed"]

    assert constant_loss == pytest.approx(0.0, abs=1e-12)
    assert jump_loss is not None and jump_loss > constant_loss


def test_huber_keeps_extreme_outlier_finite_and_projection_scale_invariant():
    points = [
        _point(0, (10, 50)),
        _point(1, (20, 50)),
        _point(2, (190, 50)),
        _point(3, (191, 50)),
    ]

    first = _evaluate(points)
    second = _evaluate(points, homography=H0 * 7.0)

    assert first.total_loss is not None and math.isfinite(first.total_loss)
    assert second.total_loss == pytest.approx(first.total_loss)


def test_invalid_homography_is_rejected_before_loss_evaluation():
    report = _evaluate(
        [_point(i, (10 + i, 50)) for i in range(4)],
        homography=np.zeros((3, 3)),
    )

    assert not report.homography_valid
    assert not report.evaluable
    assert report.total_loss is None
    assert "invalid_matrix" in report.reasons


def test_local_direction_consistency_is_computed_with_cluster_support():
    observations = [
        LocalMotionObservation(
            position=np.asarray((30 + index * 10, 50), dtype=float),
            direction=np.asarray((1.0, 0.01 * (-1) ** index)),
            speed=5.0,
            track_id=index % 2,
            quality=0.9,
            frame_id=index,
        )
        for index in range(4)
    ]
    cluster = TrafficFlowCluster(
        cluster_id=0,
        observation_indices=(0, 1, 2, 3),
        track_keys=(("vehicle", 0), ("vehicle", 1)),
        mean_direction=np.asarray((1.0, 0.0)),
        spatial_region=(0.1, 0.4, 0.4, 0.6),
        direction_variance=0.01,
        confidence=0.8,
    )

    report = _evaluate(
        [_point(i, (20 + i * 5, 50)) for i in range(4)],
        motion_observations=observations,
        flow_clusters=(cluster,),
    )

    assert report.sample_counts["direction"] == 4
    assert report.component_losses["direction"] is not None
    assert report.component_losses["direction"] < 0.001


def test_weighted_total_uses_only_components_with_evidence():
    config = CalibrationLossConfig(
        motion_weight=2.0,
        speed_weight=1.0,
        curvature_weight=3.0,
        direction_weight=4.0,
    )
    points = [
        _point(0, (20, 40)),
        _point(1, (25, 40)),
        _point(2, (35, 40)),
    ]

    report = _evaluate(points, config=config)

    expected = (
        2.0 * report.component_losses["motion"]
        + report.component_losses["speed"]
    ) / 3.0
    assert report.active_weight_sum == pytest.approx(3.0)
    assert report.total_loss == pytest.approx(expected)
    assert report.component_losses["curvature"] is None
    assert report.component_losses["direction"] is None
    assert report.component_losses["lane"] is None


def _flow_observation(index, track_id, position, direction=(1.0, 0.0)):
    return LocalMotionObservation(
        position=np.asarray(position, dtype=float),
        direction=np.asarray(direction, dtype=float),
        speed=5.0,
        track_id=track_id,
        quality=0.9,
        frame_id=index,
    )


def _lane_report(second_track_y):
    observations = []
    for track_id, y in ((1, 50.0), (2, second_track_y)):
        for x in (40.0, 60.0, 80.0):
            observations.append(
                _flow_observation(len(observations), track_id, (x, y))
            )
    cluster = TrafficFlowCluster(
        cluster_id=0,
        observation_indices=tuple(range(6)),
        track_keys=(("vehicle", 1), ("vehicle", 2)),
        mean_direction=np.asarray((1.0, 0.0)),
        spatial_region=(0.2, 0.4, 0.4, 0.8),
        direction_variance=0.0,
        confidence=0.9,
    )
    config = CalibrationLossConfig(lane_neighborhood_fraction=0.20)
    return _evaluate(
        [_point(i, (20 + i * 5, 30)) for i in range(4)],
        motion_observations=observations,
        flow_clusters=(cluster,),
        config=config,
    )


def test_cross_vehicle_lane_loss_prefers_nearby_curve_family():
    nearby = _lane_report(51.0)
    dispersed = _lane_report(70.0)

    assert nearby.sample_counts["lane"] == 6
    assert dispersed.sample_counts["lane"] == 6
    assert nearby.component_losses["lane"] < dispersed.component_losses["lane"]


def test_lane_loss_is_uniform_projection_scale_invariant():
    first = _lane_report(55.0)
    observations = []
    for track_id, y in ((1, 50.0), (2, 55.0)):
        for x in (40.0, 60.0, 80.0):
            observations.append(
                _flow_observation(len(observations), track_id, (x, y))
            )
    cluster = TrafficFlowCluster(
        cluster_id=0,
        observation_indices=tuple(range(6)),
        track_keys=(("vehicle", 1), ("vehicle", 2)),
        mean_direction=np.asarray((1.0, 0.0)),
        spatial_region=(0.2, 0.4, 0.4, 0.8),
        direction_variance=0.0,
        confidence=0.9,
    )
    scaled = _evaluate(
        [_point(i, (20 + i * 5, 30)) for i in range(4)],
        homography=H0 * 9.0,
        motion_observations=observations,
        flow_clusters=(cluster,),
        config=CalibrationLossConfig(lane_neighborhood_fraction=0.20),
    )

    assert scaled.component_losses["lane"] == pytest.approx(
        first.component_losses["lane"]
    )


def test_lane_loss_requires_other_track_evidence():
    observations = [
        _flow_observation(index, 1, (40 + index * 10, 50))
        for index in range(4)
    ]
    cluster = TrafficFlowCluster(
        cluster_id=0,
        observation_indices=(0, 1, 2, 3),
        track_keys=(("vehicle", 1),),
        mean_direction=np.asarray((1.0, 0.0)),
        spatial_region=(0.2, 0.4, 0.4, 0.6),
        direction_variance=0.0,
        confidence=0.9,
    )

    report = _evaluate(
        [_point(i, (20 + i * 5, 30)) for i in range(4)],
        motion_observations=observations,
        flow_clusters=(cluster,),
    )

    assert report.sample_counts["lane"] == 0
    assert report.component_losses["lane"] is None


def _curved_lane_change_report(change_quality):
    observations = []
    xs = (40.0, 60.0, 80.0, 100.0)
    for track_id, offset in ((1, -1.0), (2, 0.0), (3, 1.0)):
        for x in xs:
            curve_y = 45.0 + 0.002 * (x - 40.0) ** 2 + offset
            observations.append(LocalMotionObservation(
                position=np.asarray((x, curve_y)),
                direction=np.asarray((1.0, 0.004 * (x - 40.0))),
                speed=5.0,
                track_id=track_id,
                quality=0.9,
                frame_id=len(observations),
            ))
    for index, x in enumerate(xs):
        curve_y = 45.0 + 0.002 * (x - 40.0) ** 2
        observations.append(LocalMotionObservation(
            position=np.asarray((x, curve_y + index * 7.0)),
            direction=np.asarray((1.0, 0.35 * index)),
            speed=5.0,
            track_id=4,
            quality=change_quality,
            frame_id=len(observations),
        ))
    cluster = TrafficFlowCluster(
        cluster_id=0,
        observation_indices=tuple(range(len(observations))),
        track_keys=tuple(("vehicle", track_id) for track_id in range(1, 5)),
        mean_direction=np.asarray((1.0, 0.0)),
        spatial_region=(0.2, 0.3, 0.5, 0.8),
        direction_variance=0.1,
        confidence=0.9,
    )
    return _evaluate(
        [_point(i, (20 + i * 5, 30)) for i in range(4)],
        motion_observations=observations,
        flow_clusters=(cluster,),
        config=CalibrationLossConfig(lane_neighborhood_fraction=0.35),
    )


def test_curved_lane_change_has_bounded_lower_influence_when_quality_is_low():
    full_weight = _curved_lane_change_report(0.9)
    lower_weight = _curved_lane_change_report(0.1)

    assert full_weight.component_losses["lane"] is not None
    assert lower_weight.component_losses["lane"] is not None
    assert lower_weight.component_losses["lane"] < full_weight.component_losses["lane"]
