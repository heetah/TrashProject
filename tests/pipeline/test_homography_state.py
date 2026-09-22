from types import SimpleNamespace

import numpy as np
import pytest

import scripts.pipeline.calibration.state as state_module
from scripts.pipeline.calibration import (
    CalibrationLossReport,
    CalibrationPhase1Config,
    CalibrationStateConfig,
    CalibrationStatus,
    DynamicHomographyCalibrator,
    HomographyOptimizationResult,
    HomographyOptimizerConfig,
    MotionGeometrySignature,
    compose_candidate_homography,
    detect_calibration_drift,
)


IMAGE_SHAPE = (120, 160)


def _calibrator(**state_overrides):
    phase1 = CalibrationPhase1Config(
        enabled=True,
        min_track_points=3,
        min_track_path_length_px=2.0,
        min_local_displacement_px=0.5,
    )
    optimizer = HomographyOptimizerConfig(
        enabled=True,
        min_valid_tracks=1,
        min_flow_clusters=1,
        min_duration_seconds=0.0,
        freeze_confidence=0.5,
        target_tracks=4,
        target_flows=1,
    )
    settings = dict(
        evaluation_interval_seconds=1.0,
        min_successful_updates_to_lock=1,
        stable_windows_to_lock=1,
        drift_persistence_windows=2,
    )
    settings.update(state_overrides)
    return DynamicHomographyCalibrator(
        phase1_config=phase1,
        optimizer_config=optimizer,
        state_config=CalibrationStateConfig(**settings),
    )


def _add_tracks(calibrator, y_offset=0):
    for frame in range(5):
        for track_id, base_y in ((1, 35), (2, 65), (3, 85), (4, 100)):
            mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
            x1 = 25 + frame * 4
            y1 = base_y + y_offset - 15
            mask[y1:base_y + y_offset, x1:x1 + 20] = 1
            calibrator.add_vehicle_observation(
                track_id=track_id,
                timestamp=float(frame),
                frame_id=frame,
                mask=mask,
                confidence=0.95,
                bbox=(x1, y1, x1 + 20, base_y + y_offset),
                image_shape=IMAGE_SHAPE,
            )


def _loss_report():
    components = {
        "motion": 0.01,
        "speed": 0.01,
        "curvature": 0.01,
        "direction": 0.01,
        "lane": 0.01,
    }
    return CalibrationLossReport(
        homography_valid=True,
        evaluable=True,
        reasons=(),
        total_loss=0.01,
        component_losses=components,
        weighted_losses=components,
        sample_counts={name: 8 for name in components},
        active_weight_sum=5.0,
    )


def _optimization(previous, recommended):
    parameters = np.asarray((-0.08, 0.0, 0.0, 0.0, 0.0))
    candidate = compose_candidate_homography(previous, parameters)
    proposed = compose_candidate_homography(previous, parameters * 0.1) if recommended else previous
    components = {
        "tracks": 0.9, "flows": 0.9, "coverage": 0.9,
        "duration": 0.9, "quality": 0.9, "field_stability": 0.9,
        "residual": 0.9, "lane": 0.9,
        "improvement": 0.9 if recommended else 0.0,
        "historical_stability": 0.9,
    }
    return HomographyOptimizationResult(
        enabled=True,
        update_recommended=recommended,
        reasons=() if recommended else ("insufficient_candidate_improvement",),
        previous_homography=previous,
        candidate_homography=candidate,
        proposed_homography=proposed,
        selected_parameters=parameters,
        baseline_objective=0.10,
        candidate_objective=0.08 if recommended else 0.10,
        absolute_improvement=0.02 if recommended else 0.0,
        relative_improvement=0.20 if recommended else 0.0,
        confidence=float(np.mean(list(components.values()))),
        confidence_components=components,
        update_alpha=0.10 if recommended else 0.0,
        candidates_evaluated=31,
        valid_candidates=31,
        candidate_data_report=_loss_report(),
        perspective_loss=0.01,
        temporal_loss=0.01,
    )


def test_runtime_api_initializes_and_collects_without_affecting_attribution():
    calibrator = _calibrator()
    _add_tracks(calibrator)

    state = calibrator.get_state()

    assert state["status"] == CalibrationStatus.COLLECTING.value
    assert state["homography_version"] == 0
    assert state["H"] == state["homography"]
    assert state["num_tracks"] == 4
    assert state["num_motion_observations"] == 0
    assert state["last_update_timestamp"] is None
    assert state["residual"] is None
    assert state["relative_scale_only"] is True
    assert state["affects_attribution"] is False
    assert calibrator.get_homography().shape == (3, 3)
    assert np.isfinite(calibrator.image_to_ground([[40, 50]])).all()


def test_successful_update_then_stable_window_locks_and_can_rollback(monkeypatch):
    calibrator = _calibrator()
    _add_tracks(calibrator)
    calls = iter((True, False))

    def fake_estimate(*args, **kwargs):
        return _optimization(args[5], next(calls))

    monkeypatch.setattr(state_module, "estimate_candidate_homography", fake_estimate)
    original = calibrator.get_homography()

    first = calibrator.update(5.0, force=True)
    second = calibrator.update(6.0, force=True)

    assert first["status"] == CalibrationStatus.WARMING_UP.value
    assert first["homography_version"] == 1
    assert first["last_update_timestamp"] == pytest.approx(5.0)
    assert first["num_motion_observations"] > 0
    assert first["residual"] is not None
    assert second["status"] == CalibrationStatus.LOCKED.value
    assert second["residual_baseline"] is not None
    assert calibrator.rollback()
    np.testing.assert_allclose(calibrator.get_homography(), original)
    assert calibrator.get_state()["homography_version"] == 0
    assert calibrator.get_state()["last_update_timestamp"] is None


def test_persistent_background_drift_not_single_window_triggers_recalibration(monkeypatch):
    calibrator = _calibrator()
    _add_tracks(calibrator)
    calls = iter((True, False, False))
    monkeypatch.setattr(
        state_module,
        "estimate_candidate_homography",
        lambda *args, **kwargs: _optimization(args[5], next(calls)),
    )
    calibrator.update(5.0, force=True)
    assert calibrator.update(6.0, force=True)["status"] == "LOCKED"

    first = calibrator.update(7.0, force=True, background_motion_score=0.2)
    second = calibrator.update(8.0, force=True, background_motion_score=0.2)
    third = calibrator.update(9.0, force=True)

    assert first["status"] == "LOCKED"
    assert first["drift_windows"] == 1
    assert second["status"] == "DRIFT_DETECTED"
    assert third["status"] == "RECALIBRATING"


def test_low_evidence_freezes_h_and_keeps_version_zero():
    calibrator = _calibrator()
    mask = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    mask[30:60, 30:60] = 1
    calibrator.add_vehicle_observation(
        track_id=1, timestamp=0.0, frame_id=0, mask=mask,
        confidence=0.95, bbox=(30, 30, 60, 60), image_shape=IMAGE_SHAPE,
    )
    previous = calibrator.get_homography()

    state = calibrator.update(1.0, force=True)

    assert state["status"] == "LOW_CONFIDENCE"
    assert state["homography_version"] == 0
    np.testing.assert_allclose(calibrator.get_homography(), previous)


def test_state_enum_exposes_all_contract_statuses():
    assert {status.value for status in CalibrationStatus} == {
        "UNCALIBRATED",
        "COLLECTING",
        "ESTIMATING",
        "WARMING_UP",
        "LOCKED",
        "LOW_CONFIDENCE",
        "DRIFT_DETECTED",
        "RECALIBRATING",
    }


def test_drift_signal_reports_geometry_components_and_requires_external_persistence():
    reference = MotionGeometrySignature(
        centroid=np.asarray((0.2, 0.3)),
        spread=np.asarray((0.1, 0.1)),
        direction_tensor=np.eye(2) * 0.5,
        sample_count=20,
    )
    shifted = MotionGeometrySignature(
        centroid=np.asarray((0.5, 0.3)),
        spread=np.asarray((0.1, 0.1)),
        direction_tensor=np.eye(2) * 0.5,
        sample_count=20,
    )

    report = detect_calibration_drift(
        0.02, 0.021, reference, shifted, CalibrationStateConfig()
    )

    assert report["geometry_shifted"]
    assert report["drift_signal"]
    assert not report["residual_increase"]


def test_resolution_change_is_flagged_and_old_h_is_frozen():
    calibrator = _calibrator()
    _add_tracks(calibrator)
    previous = calibrator.get_homography()
    mask = np.ones((100, 100), dtype=np.uint8)

    calibrator.add_vehicle_observation(
        track_id=9, timestamp=6.0, frame_id=6, mask=mask,
        confidence=0.95, bbox=(10, 10, 80, 80), image_shape=(100, 100),
    )

    state = calibrator.get_state()
    assert state["status"] == "DRIFT_DETECTED"
    assert state["last_drift_signals"]["image_shape_changed"]
    np.testing.assert_allclose(calibrator.get_homography(), previous)


def test_metrics_have_required_calibration_window_fields(monkeypatch):
    calibrator = _calibrator()
    _add_tracks(calibrator)
    monkeypatch.setattr(
        state_module,
        "estimate_candidate_homography",
        lambda *args, **kwargs: _optimization(args[5], True),
    )

    calibrator.update(5.0, force=True)
    metric = calibrator.get_state()["last_metric"]

    required = {
        "timestamp", "status", "confidence", "num_tracks", "num_valid_tracks",
        "num_motion_samples", "spatial_coverage", "flow_cluster_count",
        "motion_loss", "curvature_loss", "lane_loss", "direction_loss",
        "speed_loss", "perspective_loss", "temporal_loss", "total_loss",
        "candidate_improvement", "update_alpha", "homography_version",
        "update_magnitude",
    }
    assert required <= set(metric)


def test_persistent_confidence_rises_gradually_across_good_windows(monkeypatch):
    calibrator = _calibrator(
        min_successful_updates_to_lock=10,
        stable_windows_to_lock=2,
        confidence_alpha=0.25,
    )
    _add_tracks(calibrator)
    monkeypatch.setattr(
        state_module,
        "estimate_candidate_homography",
        lambda *args, **kwargs: _optimization(args[5], True),
    )

    confidences = [
        calibrator.update(timestamp, force=True)["confidence"]
        for timestamp in (5.0, 6.0, 7.0)
    ]

    assert 0.0 < confidences[0] < confidences[1] < confidences[2] < 1.0
    assert calibrator.get_state()["homography_version"] == 3
