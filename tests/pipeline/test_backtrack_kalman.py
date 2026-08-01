# -*- coding: utf-8 -*-
"""GPU-free tests for actor box Kalman filter and RTS smoother."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.backtrack.kalman import (
    BoxKalmanFilter,
    KalmanConfig,
    TrackMeasurement,
    bbox_to_measurement,
    constant_velocity_transition,
    smooth_tracklet,
    state_to_bbox,
)


def _box(foot_u, foot_v=100.0, width=20.0, height=40.0):
    return (
        foot_u - width * 0.5,
        foot_v - height,
        foot_u + width * 0.5,
        foot_v,
    )


def test_missing_predictions_increase_position_uncertainty():
    kalman = BoxKalmanFilter(
        TrackMeasurement(0, _box(10.0), confidence=0.95),
        class_name="person",
    )
    initial_variance = np.trace(kalman.covariance[:2, :2])
    first_missing = kalman.predict(1)
    second_missing = kalman.predict(2)

    assert np.trace(first_missing.covariance[:2, :2]) > initial_variance
    assert (
        np.trace(second_missing.covariance[:2, :2])
        > np.trace(first_missing.covariance[:2, :2])
    )


def test_low_confidence_detection_updates_less_than_high_confidence():
    initial = TrackMeasurement(0, _box(0.0), confidence=1.0)
    high_confidence = BoxKalmanFilter(initial)
    low_confidence = BoxKalmanFilter(initial)

    high_confidence.predict(1)
    low_confidence.predict(1)
    prior_u = high_confidence.mean[0]
    high = high_confidence.update(
        TrackMeasurement(1, _box(20.0), confidence=1.0)
    )
    low = low_confidence.update(
        TrackMeasurement(1, _box(20.0), confidence=0.20)
    )

    assert abs(high.mean[0] - prior_u) > abs(low.mean[0] - prior_u)
    assert high.mean[0] > low.mean[0]


def test_rts_uses_future_detection_to_correct_missing_past_frame():
    first = TrackMeasurement(0, _box(0.0), confidence=1.0)
    forward_only = BoxKalmanFilter(first)
    filtered_missing_u = forward_only.predict(1).mean[0]

    tracklet = smooth_tracklet(
        [
            first,
            TrackMeasurement(2, _box(20.0), confidence=1.0),
        ],
        class_name="person",
        track_id=17,
    )
    smoothed_missing = tracklet.state_at(1)

    assert tracklet.frames.tolist() == [0, 1, 2]
    assert tracklet.observed.tolist() == [True, False, True]
    assert smoothed_missing.mean[0] > filtered_missing_u + 1.0
    assert smoothed_missing.mean[0] < 20.0
    assert not smoothed_missing.observed


def test_predicted_bbox_round_trip_and_bounded_extrapolation():
    original = _box(10.0, foot_v=90.0, width=24.0, height=48.0)
    measurement = bbox_to_measurement(original)
    state = np.zeros(8, dtype=np.float64)
    state[:4] = measurement
    assert np.allclose(state_to_bbox(state), original)

    tracklet = smooth_tracklet(
        [
            TrackMeasurement(0, original, confidence=1.0),
            TrackMeasurement(
                1,
                _box(14.0, foot_v=90.0, width=24.0, height=48.0),
                confidence=1.0,
            ),
        ],
        config=KalmanConfig(max_extrapolation_frames=3),
    )
    predicted = tracklet.predicted_bbox(3)
    predicted_foot_u = 0.5 * (predicted[0] + predicted[2])

    assert predicted[2] > predicted[0]
    assert predicted[3] > predicted[1]
    assert predicted_foot_u > 14.0
    assert predicted[3] == pytest.approx(90.0)
    with pytest.raises(ValueError, match="outside bounded extrapolation"):
        tracklet.predicted_bbox(5)


def test_transition_matrix_uses_requested_variable_dt():
    transition = constant_velocity_transition(2.5)
    state = np.asarray(
        [10.0, 20.0, 3.0, 4.0, 2.0, -4.0, 0.2, -0.1],
        dtype=np.float64,
    )
    predicted = transition.dot(state)

    assert predicted[0] == pytest.approx(15.0)
    assert predicted[1] == pytest.approx(10.0)
    assert predicted[2] == pytest.approx(3.5)
    assert predicted[3] == pytest.approx(3.75)


def test_prediction_uses_seconds_not_raw_frame_count():
    initial = TrackMeasurement(0, _box(10.0), confidence=0.95)
    at_10_fps = BoxKalmanFilter(
        initial,
        class_name="person",
        config=KalmanConfig(frames_per_second=10.0),
    )
    at_20_fps = BoxKalmanFilter(
        initial,
        class_name="person",
        config=KalmanConfig(frames_per_second=20.0),
    )
    at_10_fps.mean[4] = 30.0
    at_20_fps.mean[4] = 30.0

    state_10 = at_10_fps.predict(5)
    state_20 = at_20_fps.predict(10)

    np.testing.assert_allclose(state_10.mean, state_20.mean, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        state_10.covariance, state_20.covariance, rtol=1e-9, atol=1e-9
    )
