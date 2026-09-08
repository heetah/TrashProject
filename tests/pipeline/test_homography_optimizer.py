from types import SimpleNamespace

import numpy as np
import pytest

from scripts.pipeline.calibration import (
    CalibrationLossConfig,
    HomographyOptimizerConfig,
    LocalMotionObservation,
    TrafficFlowCluster,
    VehicleTrackPoint,
    compose_candidate_homography,
    estimate_candidate_homography,
    initial_normalized_image_homography,
    validate_homography,
)


IMAGE_SHAPE = (100, 200)
H0 = initial_normalized_image_homography(IMAGE_SHAPE)


def _point(track_id, frame, x, y):
    return VehicleTrackPoint(
        track_id=track_id,
        timestamp=float(frame),
        frame_id=frame,
        image_point=np.asarray((x, y), dtype=float),
        velocity=np.asarray((5.0, 0.0)),
        quality_score=0.9,
        mask_area=200.0,
        confidence=0.9,
    )


def _evidence(coverage=0.2):
    tracks = {}
    observations = []
    for track_id, y in enumerate((35.0, 45.0, 55.0, 65.0), start=1):
        points = tuple(
            _point(track_id, frame, 35.0 + frame * 12.0, y)
            for frame in range(5)
        )
        tracks[("vehicle", track_id)] = points
        for frame in range(4):
            observations.append(LocalMotionObservation(
                position=np.asarray((41.0 + frame * 12.0, y)),
                direction=np.asarray((1.0, 0.0)),
                speed=12.0,
                track_id=track_id,
                quality=0.9,
                frame_id=frame + 1,
            ))
    cluster = TrafficFlowCluster(
        cluster_id=0,
        observation_indices=tuple(range(len(observations))),
        track_keys=tuple(("vehicle", track_id) for track_id in range(1, 5)),
        mean_direction=np.asarray((1.0, 0.0)),
        spatial_region=(0.1, 0.2, 0.5, 0.8),
        direction_variance=0.0,
        confidence=0.9,
    )
    cell = SimpleNamespace(direction_variance=0.02)
    field = SimpleNamespace(spatial_coverage=coverage, cells={(0, 0): cell})
    return tracks, observations, (cluster,), field


def _run(previous, *, enabled=True, coverage=0.2, **overrides):
    tracks, observations, clusters, field = _evidence(coverage)
    settings = dict(
        enabled=enabled,
        iterations=3,
        min_valid_tracks=4,
        min_flow_clusters=1,
        min_spatial_coverage=0.01,
        min_duration_seconds=1.0,
        min_relative_improvement=0.001,
        freeze_confidence=0.0,
        target_tracks=4,
        target_flows=1,
        target_duration_seconds=4.0,
    )
    settings.update(overrides)
    return estimate_candidate_homography(
        tracks,
        observations,
        clusters,
        field,
        IMAGE_SHAPE,
        previous,
        loss_config=CalibrationLossConfig(lane_neighborhood_fraction=0.25),
        optimizer_config=HomographyOptimizerConfig(**settings),
    )


def test_parameterized_composition_preserves_projective_scale_and_validity():
    candidate = compose_candidate_homography(H0 * 9.0, (0.05, 0.01, -0.01, 0.02, 0.0))

    assert candidate[2, 2] == pytest.approx(1.0)
    assert validate_homography(candidate, IMAGE_SHAPE).valid


def test_disabled_optimizer_returns_exact_previous_homography():
    previous = compose_candidate_homography(H0, (0.1, 0.0, 0.0, 0.0, 0.0))

    result = _run(previous, enabled=False)

    assert not result.update_recommended
    assert "optimizer_disabled" in result.reasons
    np.testing.assert_allclose(result.proposed_homography, previous)


def test_bounded_search_improves_deliberately_anisotropic_previous_h():
    previous = compose_candidate_homography(H0, (0.24, 0.0, 0.0, 0.0, 0.0))

    result = _run(previous)

    assert result.candidates_evaluated == 31
    assert result.candidate_objective < result.baseline_objective
    assert result.relative_improvement > 0.0
    assert abs(result.selected_parameters[0]) > 0.0
    assert result.update_recommended
    assert 0.0 < result.update_alpha <= 0.20
    assert validate_homography(result.proposed_homography, IMAGE_SHAPE).valid
    assert not np.allclose(result.proposed_homography, previous)


def test_update_magnitude_decreases_as_previous_h_approaches_optimum():
    far = compose_candidate_homography(H0, (0.24, 0.0, 0.0, 0.0, 0.0))
    near = compose_candidate_homography(H0, (0.08, 0.0, 0.0, 0.0, 0.0))

    far_result = _run(far)
    near_result = _run(near)
    far_step = float(np.linalg.norm(far_result.proposed_homography - far))
    near_step = float(np.linalg.norm(near_result.proposed_homography - near))

    assert far_result.update_recommended
    assert near_result.update_recommended
    assert near_step < far_step


def test_low_spatial_coverage_forces_freeze_even_when_candidate_improves():
    previous = compose_candidate_homography(H0, (0.24, 0.0, 0.0, 0.0, 0.0))

    result = _run(previous, coverage=0.001)

    assert result.relative_improvement > 0.0
    assert not result.update_recommended
    assert result.update_alpha == 0.0
    assert "insufficient_spatial_coverage" in result.reasons
    np.testing.assert_allclose(result.proposed_homography, previous)


def test_empty_evidence_freezes_without_fake_zero_objective():
    result = estimate_candidate_homography(
        {}, (), (), SimpleNamespace(spatial_coverage=0.0, cells={}),
        IMAGE_SHAPE, H0,
        optimizer_config=HomographyOptimizerConfig(enabled=True),
    )

    assert not result.update_recommended
    assert result.baseline_objective is None
    assert "insufficient_loss_evidence" in result.reasons
    assert result.confidence == 0.0


def test_confidence_components_are_bounded_and_auditable():
    previous = compose_candidate_homography(H0, (0.24, 0.0, 0.0, 0.0, 0.0))

    result = _run(previous)

    assert set(result.confidence_components) == {
        "tracks", "flows", "coverage", "duration", "quality",
        "field_stability", "residual", "lane", "improvement",
        "historical_stability",
    }
    assert all(0.0 <= value <= 1.0 for value in result.confidence_components.values())
    assert result.confidence == pytest.approx(
        np.mean(list(result.confidence_components.values()))
    )
