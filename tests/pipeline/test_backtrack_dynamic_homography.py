import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.backtrack.costs import ActorObservation, compute_c_bc
from pipeline.backtrack.resolver import SmartBacktrackConfig, SmartBacktrackResolver
from pipeline.backtrack.spatial import EventSpatialTransform
from pipeline.backtrack.trajectory import ReleaseHypothesis


def _payload(*, status="LOCKED", confidence=0.9, eligible=True):
    return {
        "attribution_eligible": eligible,
        "fallback_reason": None if eligible else "calibration_not_locked",
        "status": status,
        "confidence": confidence,
        "homography_version": 4,
        "image_shape": [100, 100],
        "runtime_homography": [
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }


def _release(point=(60.0, 50.0)):
    return ReleaseHypothesis(
        frame_index=10,
        mean_uv=np.asarray(point, dtype=float),
        covariance_uv=np.eye(2),
        velocity_uv=np.asarray((1.0, 0.0)),
        model="test",
        prior_cost=0.0,
    )


def _vehicle():
    return ActorObservation(
        frame_index=10,
        cls_name="vehicle",
        track_id=2,
        bbox=np.asarray((20.0, 40.0, 50.0, 60.0)),
        confidence=0.9,
        covariance_uv=np.eye(2),
        evidence_frame_index=10,
    )


def test_event_snapshot_requires_locked_confident_valid_transform():
    transform, reason = EventSpatialTransform.from_payload(
        _payload(), minimum_confidence=0.65
    )
    assert reason == "ok"
    assert transform is not None
    assert transform.version == 4

    transform, reason = EventSpatialTransform.from_payload(
        _payload(confidence=0.4), minimum_confidence=0.65
    )
    assert transform is None
    assert reason == "snapshot_confidence_below_threshold"

    transform, reason = EventSpatialTransform.from_payload(
        _payload(status="COLLECTING", eligible=False), minimum_confidence=0.65
    )
    assert transform is None
    assert reason == "calibration_not_locked"


def test_uniform_projective_scale_preserves_normalized_vehicle_distance():
    transform, _ = EventSpatialTransform.from_payload(
        _payload(), minimum_confidence=0.65
    )
    image_cell = compute_c_bc([_release()], [_vehicle()], fps=10.0)
    ground_cell = compute_c_bc(
        [_release()], [_vehicle()], fps=10.0, spatial_transform=transform
    )

    assert image_cell.valid and ground_cell.valid
    assert ground_cell.raw_features["direct_distance"] == pytest.approx(
        image_cell.raw_features["direct_distance"]
    )


def test_resolver_records_applied_transform_and_safe_fallback():
    applied_config = SmartBacktrackConfig(use_event_homography=True)
    applied_routes = SmartBacktrackResolver(
        fps=10.0, config=applied_config
    ).build_routes({"fps": 10.0, "homography_snapshot": _payload()})
    applied = applied_routes[-1].metadata["candidate_diagnostics"][
        "spatial_calibration"
    ]
    assert applied["dynamic_homography_applied"] is True
    assert applied["coordinate_space"] == "pseudo_ground_relative"

    fallback_routes = SmartBacktrackResolver(
        fps=10.0, config=applied_config
    ).build_routes({"fps": 10.0})
    fallback = fallback_routes[-1].metadata["candidate_diagnostics"][
        "spatial_calibration"
    ]
    assert fallback == {
        "coordinate_space": "image",
        "dynamic_homography_applied": False,
        "dynamic_homography_reason": "missing_event_snapshot",
    }
