"""Event-frozen pseudo-ground geometry for Smart Backtrack.

Kalman/RTS and detector evidence remain in image coordinates.  This module
only projects spatial comparisons after an event has captured one immutable,
validated calibration snapshot, so a changing online H cannot mix coordinate
systems inside one min-cost-flow task.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from ..calibration.transform import (
    HomographyValidationConfig,
    project_image_points,
    validate_homography,
)


def _rect_corners(rect: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, rect[:4])
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return np.asarray(
        ((left, top), (right, top), (right, bottom), (left, bottom)),
        dtype=float,
    )


def _point_to_segment_vector(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    segment = b - a
    denominator = float(segment @ segment)
    if denominator <= 1e-18:
        return point - a
    fraction = float(np.clip((point - a) @ segment / denominator, 0.0, 1.0))
    return point - (a + fraction * segment)


@dataclass(frozen=True)
class EventSpatialTransform:
    """One validated image-to-relative-ground matrix frozen for one event."""

    homography: np.ndarray
    image_shape: Tuple[int, int]
    version: int
    confidence: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.homography, dtype=float).reshape(3, 3).copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "homography", matrix)
        object.__setattr__(
            self,
            "image_shape",
            (int(self.image_shape[0]), int(self.image_shape[1])),
        )

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        minimum_confidence: float,
    ) -> Tuple[Optional["EventSpatialTransform"], str]:
        if not isinstance(payload, Mapping):
            return None, "missing_event_snapshot"
        if not bool(payload.get("attribution_eligible", False)):
            return None, str(payload.get("fallback_reason") or "snapshot_not_eligible")
        if str(payload.get("status", "")) != "LOCKED":
            return None, "calibration_not_locked"
        try:
            confidence = float(payload.get("confidence", 0.0))
            version = int(payload.get("homography_version", 0))
            image_shape = tuple(int(value) for value in payload["image_shape"][:2])
            homography = np.asarray(payload["runtime_homography"], dtype=float).reshape(3, 3)
        except (KeyError, TypeError, ValueError, IndexError):
            return None, "invalid_event_snapshot"
        if not math.isfinite(confidence) or confidence < float(minimum_confidence):
            return None, "snapshot_confidence_below_threshold"
        validation = validate_homography(
            homography,
            image_shape,
            HomographyValidationConfig(),
        )
        if not validation.valid:
            return None, "invalid_runtime_homography"
        return cls(
            validation.normalized_homography,
            image_shape,
            version,
            confidence,
        ), "ok"

    def project(self, points: object) -> Optional[np.ndarray]:
        result = project_image_points(points, self.homography)
        if not np.all(result.valid_mask):
            return None
        return result.points

    def projected_rect(self, rect: Sequence[float]) -> Optional[np.ndarray]:
        return self.project(_rect_corners(rect))

    def point_to_rect_distance(
        self,
        point: Sequence[float],
        rect: Sequence[float],
    ) -> Optional[float]:
        image_point = np.asarray(point, dtype=float).reshape(2)
        x1, y1, x2, y2 = map(float, rect[:4])
        if (
            min(x1, x2) <= image_point[0] <= max(x1, x2)
            and min(y1, y2) <= image_point[1] <= max(y1, y2)
        ):
            return 0.0
        projected_point = self.project(image_point.reshape(1, 2))
        polygon = self.projected_rect(rect)
        if projected_point is None or polygon is None:
            return None
        point_ground = projected_point[0]
        distances = [
            np.linalg.norm(_point_to_segment_vector(
                point_ground,
                polygon[index],
                polygon[(index + 1) % len(polygon)],
            ))
            for index in range(len(polygon))
        ]
        return float(min(distances)) if distances else None

    def rect_diagonal_scale(self, rect: Sequence[float]) -> Optional[float]:
        polygon = self.projected_rect(rect)
        if polygon is None:
            return None
        distances = [
            float(np.linalg.norm(polygon[right] - polygon[left]))
            for left in range(len(polygon))
            for right in range(left + 1, len(polygon))
        ]
        return max(distances) if distances else None

    def rect_height_scale(self, rect: Sequence[float]) -> Optional[float]:
        x1, y1, x2, y2 = map(float, rect[:4])
        middle_x = 0.5 * (x1 + x2)
        projected = self.project(((middle_x, y1), (middle_x, y2)))
        if projected is None:
            return None
        return float(np.linalg.norm(projected[1] - projected[0]))

    def normalized_point_to_rect(
        self,
        point: Sequence[float],
        rect: Sequence[float],
        *,
        scale: str,
    ) -> Optional[float]:
        distance = self.point_to_rect_distance(point, rect)
        denominator = (
            self.rect_height_scale(rect)
            if scale == "height"
            else self.rect_diagonal_scale(rect)
        )
        if distance is None or denominator is None or denominator <= 1e-12:
            return None
        return float(distance / denominator)

    def normalized_point_distance(
        self,
        left: Sequence[float],
        right: Sequence[float],
        scale_rect: Sequence[float],
    ) -> Optional[float]:
        projected = self.project((left, right))
        denominator = self.rect_diagonal_scale(scale_rect)
        if projected is None or denominator is None or denominator <= 1e-12:
            return None
        return float(np.linalg.norm(projected[0] - projected[1]) / denominator)

    def vector_at(
        self,
        point: Sequence[float],
        vector: Sequence[float],
    ) -> Optional[np.ndarray]:
        origin = np.asarray(point, dtype=float).reshape(2)
        delta = np.asarray(vector, dtype=float).reshape(2)
        if float(np.linalg.norm(delta)) <= 1e-12:
            return np.zeros(2, dtype=float)
        projected = self.project((origin, origin + delta))
        if projected is None:
            return None
        return projected[1] - projected[0]


def image_space_metadata(reason: str = "feature_disabled") -> dict:
    return {
        "coordinate_space": "image",
        "dynamic_homography_applied": False,
        "dynamic_homography_reason": str(reason),
    }


def pseudo_ground_metadata(transform: EventSpatialTransform) -> dict:
    return {
        "coordinate_space": "pseudo_ground_relative",
        "dynamic_homography_applied": True,
        "dynamic_homography_version": int(transform.version),
        "dynamic_homography_confidence": float(transform.confidence),
    }
