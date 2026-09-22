import numpy as np
import pytest

from pipeline.calibration import (
    CalibrationBuffer,
    CalibrationPhase1Config,
    VehicleTrackPoint,
    VehicleTrajectoryCollector,
    extract_vehicle_ground_point,
)
from pipeline.litter_tracker import GlobalLitterTracker


def _rectangle_mask(x1, y1, x2, y2, shape=(120, 160)):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return mask


def test_ground_point_uses_bottom_band_median_not_single_lowest_noise_pixel():
    mask = _rectangle_mask(30, 20, 70, 80)
    mask[100, 150] = 1

    point = extract_vehicle_ground_point(mask, bottom_percentile=98.0)

    assert point is not None
    assert point[0] == pytest.approx(49.5, abs=0.6)
    assert point[1] == pytest.approx(79.0)


def test_repeated_segmentation_jitter_keeps_bottom_band_ground_points_bounded():
    rng = np.random.default_rng(7)
    points = []
    for _ in range(40):
        dx, dy = rng.integers(-2, 3, size=2)
        mask = _rectangle_mask(30 + dx, 20 + dy, 70 + dx, 80 + dy)
        mask[int(rng.integers(90, 115)), int(rng.integers(0, 160))] = 1
        points.append(extract_vehicle_ground_point(mask))

    points = np.asarray(points)
    assert np.std(points[:, 0]) < 2.0
    assert np.std(points[:, 1]) < 2.0
    assert np.max(points[:, 1]) < 83.0


def test_ground_point_accepts_ultralytics_polygon_coordinates():
    polygon = np.asarray([[20, 20], [60, 20], [60, 70], [20, 70]], dtype=float)

    point = extract_vehicle_ground_point(polygon)

    assert point is not None
    assert point[0] == pytest.approx(40.0, abs=0.6)
    assert point[1] == pytest.approx(70.0, abs=0.6)


def test_calibration_buffer_expires_old_points_without_storing_frames():
    buffer = CalibrationBuffer(window_seconds=2.0)
    for frame_id, timestamp in enumerate((0.0, 1.0, 3.1)):
        buffer.add(VehicleTrackPoint(
            track_id=1,
            timestamp=timestamp,
            frame_id=frame_id,
            image_point=[frame_id, 0],
            velocity=[1, 0],
            quality_score=1.0,
            mask_area=100.0,
            confidence=0.9,
        ))

    track = buffer.get_track(("vehicle", 1))
    assert [point.timestamp for point in track] == [3.1]
    assert not hasattr(track[0], "frame")


def test_collector_preserves_curved_local_motion_instead_of_fitting_one_line():
    config = CalibrationPhase1Config(
        enabled=True,
        min_track_points=3,
        min_track_path_length_px=5.0,
        min_local_displacement_px=1.0,
    )
    collector = VehicleTrajectoryCollector(config)
    masks = (
        _rectangle_mask(20, 30, 45, 65),
        _rectangle_mask(30, 30, 55, 65),
        _rectangle_mask(30, 40, 55, 75),
    )
    boxes = ((20, 30, 45, 65), (30, 30, 55, 65), (30, 40, 55, 75))
    for index, (mask, bbox) in enumerate(zip(masks, boxes)):
        assert collector.add_vehicle_observation(
            track_id=7,
            timestamp=float(index),
            frame_id=index,
            mask=mask,
            confidence=0.9,
            bbox=bbox,
            image_shape=mask.shape,
        ) is not None

    motions = collector.local_motion_observations()
    assert len(motions) == 2
    assert motions[0].direction == pytest.approx([1.0, 0.0], abs=1e-6)
    assert motions[1].direction == pytest.approx([0.0, 1.0], abs=1e-6)
    assert collector.summary()["valid_motion_tracks"] == 1


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"confidence": 0.1}, "low_confidence"),
        ({"observed": False}, "cached_observation"),
        ({"bbox": (0, 10, 30, 50)}, "image_border"),
        ({"mask": _rectangle_mask(20, 20, 22, 22)}, "mask_too_small"),
        ({"occluded": True}, "occluded_observation"),
        ({"tracking_stable": False}, "unstable_track_id"),
    ],
)
def test_collector_rejects_weak_calibration_observations(overrides, reason):
    collector = VehicleTrajectoryCollector(CalibrationPhase1Config(enabled=True))
    arguments = {
        "track_id": 1,
        "timestamp": 0.0,
        "frame_id": 0,
        "mask": _rectangle_mask(20, 20, 50, 60),
        "confidence": 0.9,
        "bbox": (20, 20, 50, 60),
        "image_shape": (120, 160),
        "observed": True,
    }
    arguments.update(overrides)

    assert collector.add_vehicle_observation(**arguments) is None
    assert collector.summary()["rejected_observations"][reason] == 1


def test_sudden_id_switch_like_teleport_is_rejected():
    collector = VehicleTrajectoryCollector(CalibrationPhase1Config(enabled=True))
    first = _rectangle_mask(20, 30, 50, 70, shape=(120, 200))
    switched = _rectangle_mask(140, 30, 170, 70, shape=(120, 200))
    assert collector.add_vehicle_observation(
        track_id=8, timestamp=0.0, frame_id=0, mask=first,
        confidence=0.95, bbox=(20, 30, 50, 70), image_shape=first.shape,
    ) is not None

    assert collector.add_vehicle_observation(
        track_id=8, timestamp=0.01, frame_id=1, mask=switched,
        confidence=0.95, bbox=(140, 30, 170, 70), image_shape=switched.shape,
    ) is None
    assert collector.summary()["rejected_observations"]["teleport_motion"] == 1


def test_long_track_gap_starts_new_tracklet_instead_of_connecting_reused_id():
    collector = VehicleTrajectoryCollector(CalibrationPhase1Config(
        enabled=True, max_track_gap_seconds=1.0
    ))
    first = _rectangle_mask(20, 30, 50, 70)
    reused = _rectangle_mask(100, 30, 130, 70)
    assert collector.add_vehicle_observation(
        track_id=3, timestamp=0.0, frame_id=0, mask=first,
        confidence=0.95, bbox=(20, 30, 50, 70), image_shape=first.shape,
    ) is not None
    assert collector.add_vehicle_observation(
        track_id=3, timestamp=5.0, frame_id=50, mask=reused,
        confidence=0.95, bbox=(100, 30, 130, 70), image_shape=reused.shape,
    ) is not None

    track = collector.buffer.get_track(("vehicle", 3))
    assert len(track) == 1
    assert track[0].frame_id == 50


def test_extreme_local_u_turn_is_rejected_without_forcing_normal_curve_straight():
    collector = VehicleTrajectoryCollector(CalibrationPhase1Config(
        enabled=True,
        max_local_turn_degrees=120.0,
        max_acceleration_diagonals_per_second2=100.0,
    ))
    observations = (
        (0, _rectangle_mask(20, 30, 50, 70), (20, 30, 50, 70)),
        (1, _rectangle_mask(30, 30, 60, 70), (30, 30, 60, 70)),
        (2, _rectangle_mask(20, 30, 50, 70), (20, 30, 50, 70)),
    )
    for frame, mask, box in observations[:2]:
        assert collector.add_vehicle_observation(
            track_id=5, timestamp=float(frame), frame_id=frame, mask=mask,
            confidence=0.95, bbox=box, image_shape=mask.shape,
        ) is not None

    frame, mask, box = observations[2]
    assert collector.add_vehicle_observation(
        track_id=5, timestamp=float(frame), frame_id=frame, mask=mask,
        confidence=0.95, bbox=box, image_shape=mask.shape,
    ) is None
    assert collector.summary()["rejected_observations"]["extreme_local_turn"] == 1


def test_global_tracker_collects_phase1_vehicle_masks_without_affecting_attribution(
    monkeypatch,
):
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY", "1")
    tracker = GlobalLitterTracker(fps=10.0)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    try:
        for frame_id, shift in enumerate((0, 5, 10)):
            polygon = np.asarray(
                [[20 + shift, 20], [60 + shift, 20],
                 [60 + shift, 70], [20 + shift, 70]],
                dtype=float,
            )
            tracker._record_actor_frame([{
                "cls": "vehicle",
                "track_id": 4,
                "box": (20 + shift, 20, 60 + shift, 70),
                "confidence": 0.9,
                "observed": True,
                "source": "seg_predict",
                "mask_poly": polygon,
            }], frame_id, frame=frame)

        summary = tracker.get_homography_calibration_summary()
        assert summary["accepted_observations"] == 3
        assert summary["valid_motion_tracks"] == 1
        assert summary["affects_attribution"] is False
    finally:
        tracker.close()


def test_optional_stabilizer_is_called_with_actor_and_litter_exclusion(monkeypatch):
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY", "1")
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY_STABILIZE", "1")
    tracker = GlobalLitterTracker(fps=10.0)
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    try:
        tracker._record_actor_frame(
            [{
                "cls": "person",
                "track_id": 2,
                "box": (20, 20, 50, 80),
                "confidence": 0.9,
                "observed": True,
            }],
            0,
            frame=frame,
            dynamic_litters=[(70, 70, 80, 80, 0.9)],
        )

        summary = tracker.get_homography_calibration_summary()
        assert summary["implemented_through_phase"] == 8
        assert summary["stabilization"]["status"] == "insufficient_static_features"
        assert summary["stabilization"]["affects_attribution"] is False
    finally:
        tracker.close()
