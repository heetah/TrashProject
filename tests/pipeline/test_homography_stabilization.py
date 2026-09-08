import cv2
import numpy as np
import pytest

from scripts.pipeline.calibration import (
    BackgroundStabilizerConfig,
    CalibrationPhase1Config,
    DynamicHomographyCalibrator,
    StaticBackgroundStabilizer,
    build_dynamic_exclusion_mask,
    compose_runtime_homography,
    initial_normalized_image_homography,
    render_calibration_debug,
)


def _textured_frame(seed=4):
    rng = np.random.default_rng(seed)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(180):
        center = tuple(rng.integers((5, 5), (315, 235)).tolist())
        color = tuple(int(value) for value in rng.integers(80, 256, size=3))
        cv2.circle(frame, center, int(rng.integers(1, 4)), color, -1)
    return frame


def _config(**overrides):
    values = dict(
        enabled=True,
        max_features=1000,
        min_matches=12,
        target_matches=40,
        min_inlier_ratio=0.4,
        max_translation_fraction=0.15,
    )
    values.update(overrides)
    return BackgroundStabilizerConfig(**values)


def test_exclusion_mask_combines_dense_polygon_and_box_regions():
    dense = np.zeros((100, 120), dtype=np.uint8)
    dense[10:20, 10:20] = 1
    polygon = np.asarray(((40, 40), (60, 40), (50, 60)))

    mask = build_dynamic_exclusion_mask(
        (100, 120),
        (dense, {"mask_poly": polygon}, {"box": (80, 70, 100, 90)}),
    )

    assert mask[15, 15] == 255
    assert mask[48, 50] == 255
    assert mask[80, 90] == 255
    assert mask[0, 0] == 0


def test_stabilizer_requires_explicit_dynamic_exclusion_mask():
    stabilizer = StaticBackgroundStabilizer(_config())

    result = stabilizer.update(_textured_frame(), exclusion_mask=None)

    assert not result.valid
    assert result.status == "missing_or_invalid_exclusion_mask"
    assert result.as_dict()["uses_dynamic_object_correspondences"] is False


def test_orb_static_background_estimates_current_to_reference_translation():
    reference = _textured_frame()
    current = cv2.warpAffine(
        reference,
        np.asarray(((1.0, 0.0, 6.0), (0.0, 1.0, 4.0))),
        (reference.shape[1], reference.shape[0]),
    )
    exclusion = np.zeros(reference.shape[:2], dtype=np.uint8)
    exclusion[90:150, 130:190] = 255
    stabilizer = StaticBackgroundStabilizer(_config())

    initialized = stabilizer.update(reference, exclusion_mask=exclusion)
    result = stabilizer.update(current, exclusion_mask=exclusion)

    assert initialized.status == "reference_initialized"
    assert result.valid
    assert result.inlier_count >= 12
    assert result.transform_current_to_reference[0, 2] == pytest.approx(-6.0, abs=1.5)
    assert result.transform_current_to_reference[1, 2] == pytest.approx(-4.0, abs=1.5)
    assert result.confidence > 0.0


def test_invalid_new_estimate_keeps_last_valid_stabilization_transform():
    reference = _textured_frame()
    current = cv2.warpAffine(
        reference,
        np.asarray(((1.0, 0.0, 5.0), (0.0, 1.0, 0.0))),
        (reference.shape[1], reference.shape[0]),
    )
    exclusion = np.zeros(reference.shape[:2], dtype=np.uint8)
    stabilizer = StaticBackgroundStabilizer(_config())
    stabilizer.update(reference, exclusion_mask=exclusion)
    assert stabilizer.update(current, exclusion_mask=exclusion).valid
    accepted = stabilizer.get_transform()

    failed = stabilizer.update(np.zeros_like(reference), exclusion_mask=exclusion)

    assert not failed.valid
    np.testing.assert_allclose(stabilizer.get_transform(), accepted)


def test_runtime_composition_matches_reference_geometry():
    calibration = initial_normalized_image_homography((100, 200))
    current_to_reference = np.asarray(((1, 0, -10), (0, 1, -5), (0, 0, 1)), dtype=float)

    runtime = compose_runtime_homography(calibration, current_to_reference)
    current_point = np.asarray((110.0, 55.0, 1.0))
    projected = runtime @ current_point
    projected = projected[:2] / projected[2]

    np.testing.assert_allclose(projected, (0.5, 0.5))


def test_event_snapshot_is_immutable_and_freezes_runtime_h():
    calibrator = DynamicHomographyCalibrator(
        phase1_config=CalibrationPhase1Config(enabled=True),
        stabilizer_config=_config(),
    )
    frame = _textured_frame()
    exclusion = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[100:150, 100:150] = 1
    calibrator.add_vehicle_observation(
        track_id=1, timestamp=0.0, frame_id=0, mask=mask,
        confidence=0.95, bbox=(100, 100, 150, 150), image_shape=frame.shape[:2],
    )
    calibrator.stabilize_frame(frame, exclusion_mask=exclusion)

    snapshot = calibrator.capture_event_snapshot(1.5)
    original = snapshot.runtime_homography.copy()
    calibrator.reset()

    np.testing.assert_allclose(snapshot.runtime_homography, original)
    assert not snapshot.runtime_homography.flags.writeable
    assert snapshot.as_dict()["immutable_event_snapshot"] is True


def test_debug_renderer_returns_copy_with_tracks_inset_and_metrics():
    from scripts.pipeline.calibration import VehicleTrackPoint

    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    points = tuple(
        VehicleTrackPoint(
            track_id=1,
            timestamp=float(index),
            frame_id=index,
            image_point=np.asarray((30 + index * 20, 90)),
            velocity=np.asarray((20, 0)),
            quality_score=0.9,
            mask_area=100,
            confidence=0.9,
        )
        for index in range(4)
    )
    state = {
        "status": "LOCKED",
        "homography_version": 2,
        "confidence": 0.8,
        "last_metric": {
            "num_valid_tracks": 1,
            "spatial_coverage": 0.2,
            "total_loss": 0.03,
            "update_alpha": 0.0,
        },
    }

    rendered = render_calibration_debug(
        frame,
        {("vehicle", 1): points},
        (),
        None,
        (),
        initial_normalized_image_homography(frame.shape[:2]),
        state,
    )

    assert not np.shares_memory(rendered, frame)
    assert np.count_nonzero(frame) == 0
    assert np.count_nonzero(rendered) > 0
