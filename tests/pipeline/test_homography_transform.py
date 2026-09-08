import numpy as np
import pytest

from pipeline.calibration import (
    HomographyValidationConfig,
    image_to_ground,
    initial_homography_snapshot,
    initial_normalized_image_homography,
    normalize_homography,
    project_image_points,
    validate_homography,
)


def test_initial_homography_maps_image_to_unit_relative_coordinates():
    homography = initial_normalized_image_homography((100, 200))
    projected = image_to_ground(
        [[0, 0], [100, 50], [200, 100]], homography
    )

    np.testing.assert_allclose(
        projected,
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
    )
    report = validate_homography(homography, (100, 200))
    assert report.valid
    assert report.reasons == ()
    assert report.projected_signed_area == pytest.approx(1.0)


def test_normalization_removes_projective_scale_without_mutating_input():
    original = initial_normalized_image_homography((100, 200)) * 7.0
    before = original.copy()

    normalized = normalize_homography(original)

    np.testing.assert_allclose(
        normalized,
        initial_normalized_image_homography((100, 200)),
    )
    np.testing.assert_allclose(original, before)


def test_projection_marks_nan_small_denominator_and_explosion_invalid():
    crossing = np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.02, -1.0],
    ])
    result = project_image_points(
        [[10, 10], [10, 50], [np.nan, 1]],
        crossing,
        min_abs_denominator=1e-3,
        max_abs_coordinate=20.0,
    )

    assert result.valid_mask.tolist() == [True, False, False]
    assert np.isnan(result.points[1:]).all()


@pytest.mark.parametrize(
    ("homography", "expected_reason"),
    [
        (np.zeros((3, 3)), "invalid_matrix"),
        (np.diag([-0.01, 0.01, 1.0]), "orientation_reversed"),
        (
            np.asarray([[1, 0, 0], [0, 1, 0], [0, 0.02, -1]], dtype=float),
            "unsafe_projection",
        ),
    ],
)
def test_validation_rejects_singular_mirrored_and_infinite_candidates(
    homography, expected_reason
):
    report = validate_homography(homography, (100, 100))

    assert not report.valid
    assert expected_reason in report.reasons


def test_validation_rejects_ill_conditioned_candidate():
    homography = np.diag([1e-10, 1.0, 1.0])
    report = validate_homography(
        homography,
        (100, 100),
        HomographyValidationConfig(max_condition_number=1e5),
    )

    assert not report.valid
    assert "ill_conditioned" in report.reasons


def test_initial_snapshot_is_explicitly_relative_and_non_authoritative():
    snapshot = initial_homography_snapshot((1080, 1920))

    assert snapshot["validation"]["valid"] is True
    assert snapshot["relative_scale_only"] is True
    assert snapshot["confidence"] == 0.0
    assert snapshot["version"] == 0
    assert snapshot["affects_attribution"] is False


def test_projection_rejects_malformed_points_shape():
    with pytest.raises(ValueError, match="shape"):
        image_to_ground([1, 2, 3], np.eye(3))
