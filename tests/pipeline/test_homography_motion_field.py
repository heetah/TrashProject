import numpy as np
import pytest

from pipeline.calibration import (
    CalibrationPhase1Config,
    FlowClusteringConfig,
    LocalMotionObservation,
    MotionFieldConfig,
    TrafficMotionField,
    VehicleTrajectoryCollector,
    cluster_traffic_flows,
    render_motion_debug,
    robust_direction_summary,
)


def _motion(position, direction, track_id, frame_id, speed=10.0, quality=0.9):
    return LocalMotionObservation(
        position=position,
        direction=direction,
        speed=speed,
        track_id=track_id,
        quality=quality,
        frame_id=frame_id,
    )


def test_robust_circular_direction_trims_opposite_outlier():
    direction, variance = robust_direction_summary(
        [[1, 0]] * 9 + [[-1, 0]],
        [1.0] * 10,
        trim_fraction=0.1,
    )

    assert direction == pytest.approx([1.0, 0.0], abs=1e-8)
    assert variance == pytest.approx(0.0, abs=1e-8)


def test_motion_field_uses_robust_direction_and_weighted_median_speed():
    observations = [
        _motion((20, 20), (1, 0), index, index, speed=10 + index % 2)
        for index in range(9)
    ]
    observations.append(_motion((21, 20), (-1, 0), 99, 9, speed=1000.0))
    field = TrafficMotionField(MotionFieldConfig(
        grid_rows=2,
        grid_cols=2,
        min_cell_samples=3,
        direction_trim_fraction=0.1,
        confidence_sample_target=5,
    )).build(observations, (100, 100))

    cell = field.cells[(0, 0)]
    assert cell.dominant_direction == pytest.approx([1.0, 0.0], abs=1e-8)
    assert cell.median_speed in (10.0, 11.0)
    assert cell.median_speed != 1000.0
    assert field.spatial_coverage == pytest.approx(0.25)


def test_flow_clustering_separates_opposite_directions_in_same_region():
    observations = []
    for track_id in (1, 2):
        observations.extend([
            _motion((20 + track_id, 20), (1, 0), track_id, 1),
            _motion((25 + track_id, 20), (1, 0), track_id, 2),
        ])
    for track_id in (3, 4):
        observations.extend([
            _motion((21 + track_id, 22), (-1, 0), track_id, 1),
            _motion((26 + track_id, 22), (-1, 0), track_id, 2),
        ])

    clusters, noise = cluster_traffic_flows(
        observations,
        (100, 100),
        FlowClusteringConfig(
            position_radius_fraction=0.15,
            max_direction_angle_deg=20,
            continuity_radius_fraction=0.3,
            max_continuity_angle_deg=30,
            min_samples=4,
            min_tracks=2,
            confidence_sample_target=4,
        ),
    )

    assert len(clusters) == 2
    assert noise == ()
    x_directions = sorted(round(float(cluster.mean_direction[0])) for cluster in clusters)
    assert x_directions == [-1, 1]


def test_flow_clustering_does_not_merge_far_regions_by_direction_alone():
    observations = []
    for base, track_ids in (((10, 10), (1, 2)), ((80, 80), (3, 4))):
        for track_id in track_ids:
            observations.extend([
                _motion(base, (1, 0), track_id, 1),
                _motion((base[0] + 3, base[1]), (1, 0), track_id, 2),
            ])

    clusters, noise = cluster_traffic_flows(
        observations,
        (100, 100),
        FlowClusteringConfig(
            position_radius_fraction=0.10,
            max_direction_angle_deg=20,
            continuity_radius_fraction=0.15,
            max_continuity_angle_deg=30,
            min_samples=4,
            min_tracks=2,
            confidence_sample_target=4,
        ),
    )

    assert len(clusters) == 2
    assert noise == ()
    assert clusters[0].spatial_region != clusters[1].spatial_region


def test_sparse_motion_is_noise_not_a_confident_flow():
    observations = [
        _motion((20, 20), (1, 0), 1, 1),
        _motion((25, 20), (1, 0), 1, 2),
    ]

    clusters, noise = cluster_traffic_flows(
        observations,
        (100, 100),
        FlowClusteringConfig(min_samples=3, min_tracks=2),
    )

    assert clusters == []
    assert noise == (0, 1)


def test_debug_renderer_returns_annotated_copy_without_mutating_input():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    observations = [
        _motion((20, 20), (1, 0), track_id, frame_id)
        for track_id in (1, 2)
        for frame_id in (1, 2)
    ]
    field = TrafficMotionField(MotionFieldConfig(
        grid_rows=2, grid_cols=2, min_cell_samples=3
    )).build(observations, frame.shape[:2])
    clusters, _ = cluster_traffic_flows(
        observations,
        frame.shape[:2],
        FlowClusteringConfig(min_samples=4, min_tracks=2),
    )

    rendered = render_motion_debug(frame, observations, field, clusters)

    assert rendered.shape == frame.shape
    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(frame) == 0


def test_collector_summary_exposes_phase2_field_and_flow_metrics(monkeypatch):
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY_MIN_CELL_SAMPLES", "2")
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY_GRID_ROWS", "2")
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY_GRID_COLS", "2")
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY_FLOW_MIN_SAMPLES", "4")
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY_FLOW_MIN_TRACKS", "2")
    monkeypatch.setenv("DYNAMIC_HOMOGRAPHY_FLOW_POSITION_RADIUS", "0.4")
    collector = VehicleTrajectoryCollector(CalibrationPhase1Config(
        enabled=True,
        min_track_points=3,
        min_track_path_length_px=4.0,
    ))
    for track_id, base_y in ((1, 30), (2, 55)):
        for frame_id, shift in enumerate((0, 5, 10)):
            mask = np.zeros((120, 160), dtype=np.uint8)
            mask[base_y:base_y + 25, 30 + shift:55 + shift] = 1
            assert collector.add_vehicle_observation(
                track_id=track_id,
                timestamp=float(frame_id),
                frame_id=frame_id,
                mask=mask,
                confidence=0.9,
                bbox=(30 + shift, base_y, 55 + shift, base_y + 25),
                image_shape=mask.shape,
            ) is not None

    summary = collector.summary()
    assert summary["schema"] == "dynamic-homography-observations/v2"
    assert summary["implemented_through_phase"] == 6
    assert summary["motion_field"]["active_cells"] >= 1
    assert summary["flow_cluster_count"] == 1
    assert summary["flow_clusters"][0]["track_count"] == 2
    assert summary["initial_transform"]["validation"]["valid"] is True
    assert summary["initial_transform"]["affects_attribution"] is False
    assert summary["calibration_losses"]["evaluable"] is True
    assert summary["calibration_losses"]["affects_attribution"] is False
    assert summary["optimization"]["enabled"] is False
    assert summary["optimization"]["affects_attribution"] is False
