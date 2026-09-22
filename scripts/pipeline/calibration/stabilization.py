"""Optional static-background frame stabilization for Phase 8.

The estimator requires an explicit dynamic-object exclusion mask.  It stores
one reference feature set, not a frame history, and never uses vehicle motion
as a camera-motion correspondence source.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .transform import normalize_homography


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class BackgroundStabilizerConfig:
    enabled: bool = False
    max_features: int = 800
    ratio_test: float = 0.75
    min_matches: int = 20
    target_matches: int = 80
    ransac_threshold_px: float = 3.0
    min_inlier_ratio: float = 0.50
    max_translation_fraction: float = 0.08
    max_rotation_degrees: float = 5.0
    max_scale_change: float = 0.08
    exclusion_dilation_px: int = 7

    def __post_init__(self) -> None:
        if self.max_features < 50 or self.min_matches < 3:
            raise ValueError("background feature support is too small")
        if self.target_matches < self.min_matches:
            raise ValueError("target_matches must cover min_matches")
        if not 0.0 < self.ratio_test < 1.0:
            raise ValueError("ratio_test must be within (0, 1)")
        fractions = (
            self.min_inlier_ratio,
            self.max_translation_fraction,
            self.max_scale_change,
        )
        if not all(math.isfinite(value) and 0.0 < value < 1.0 for value in fractions):
            raise ValueError("stabilizer fractions must be within (0, 1)")
        if not math.isfinite(self.ransac_threshold_px) or self.ransac_threshold_px <= 0.0:
            raise ValueError("ransac_threshold_px must be positive")
        if not math.isfinite(self.max_rotation_degrees) or not 0.0 < self.max_rotation_degrees < 45.0:
            raise ValueError("max_rotation_degrees must be within (0, 45)")
        if self.exclusion_dilation_px < 0:
            raise ValueError("exclusion_dilation_px must be non-negative")

    @classmethod
    def from_env(cls) -> "BackgroundStabilizerConfig":
        defaults = cls()
        return cls(
            enabled=_env_bool("DYNAMIC_HOMOGRAPHY_STABILIZE", defaults.enabled),
            max_features=_env_int("DYNAMIC_HOMOGRAPHY_STAB_MAX_FEATURES", defaults.max_features),
            ratio_test=_env_float("DYNAMIC_HOMOGRAPHY_STAB_RATIO_TEST", defaults.ratio_test),
            min_matches=_env_int("DYNAMIC_HOMOGRAPHY_STAB_MIN_MATCHES", defaults.min_matches),
            target_matches=_env_int("DYNAMIC_HOMOGRAPHY_STAB_TARGET_MATCHES", defaults.target_matches),
            ransac_threshold_px=_env_float("DYNAMIC_HOMOGRAPHY_STAB_RANSAC_PX", defaults.ransac_threshold_px),
            min_inlier_ratio=_env_float("DYNAMIC_HOMOGRAPHY_STAB_MIN_INLIER_RATIO", defaults.min_inlier_ratio),
            max_translation_fraction=_env_float("DYNAMIC_HOMOGRAPHY_STAB_MAX_TRANSLATION_FRAC", defaults.max_translation_fraction),
            max_rotation_degrees=_env_float("DYNAMIC_HOMOGRAPHY_STAB_MAX_ROTATION_DEG", defaults.max_rotation_degrees),
            max_scale_change=_env_float("DYNAMIC_HOMOGRAPHY_STAB_MAX_SCALE_CHANGE", defaults.max_scale_change),
            exclusion_dilation_px=_env_int("DYNAMIC_HOMOGRAPHY_STAB_EXCLUSION_DILATE_PX", defaults.exclusion_dilation_px),
        )


@dataclass(frozen=True)
class StabilizationResult:
    valid: bool
    status: str
    transform_current_to_reference: np.ndarray
    confidence: float
    motion_score: float
    match_count: int
    inlier_count: int
    inlier_ratio: float
    translation_fraction: float
    rotation_degrees: float
    scale_change: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "transform_current_to_reference",
            np.asarray(self.transform_current_to_reference, dtype=float).reshape(3, 3),
        )

    def as_dict(self) -> dict:
        return {
            "valid": bool(self.valid),
            "status": self.status,
            "transform_current_to_reference": self.transform_current_to_reference.tolist(),
            "confidence": float(self.confidence),
            "motion_score": float(self.motion_score),
            "match_count": int(self.match_count),
            "inlier_count": int(self.inlier_count),
            "inlier_ratio": float(self.inlier_ratio),
            "translation_fraction": float(self.translation_fraction),
            "rotation_degrees": float(self.rotation_degrees),
            "scale_change": float(self.scale_change),
            "uses_dynamic_object_correspondences": False,
            "affects_attribution": False,
        }


def compose_runtime_homography(calibration_homography: object, current_to_reference: object) -> np.ndarray:
    """Compose reference-to-ground H with current-to-reference stabilization."""

    return normalize_homography(
        normalize_homography(calibration_homography)
        @ normalize_homography(current_to_reference)
    )


def build_dynamic_exclusion_mask(
    image_shape: Sequence[int],
    regions: Iterable[object],
) -> np.ndarray:
    """Rasterize known vehicle/person/litter masks, polygons or boxes."""

    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must be positive")
    result = np.zeros((height, width), dtype=np.uint8)
    for region in regions:
        value = region
        if isinstance(region, dict):
            value = region.get("mask", region.get("mask_poly", region.get("box")))
        if value is None:
            continue
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.shape == (height, width):
            result[array > 0] = 255
        elif array.ndim == 2 and array.shape[1] == 2 and len(array) >= 3:
            polygon = np.rint(array).astype(np.int32)
            cv2.fillPoly(result, [polygon], 255)
        elif array.size >= 4:
            x1, y1, x2, y2 = np.rint(array.reshape(-1)[:4]).astype(int)
            x1, x2 = sorted((int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))))
            y1, y2 = sorted((int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))))
            if x2 > x1 and y2 > y1:
                result[y1:y2, x1:x2] = 255
    return result


class StaticBackgroundStabilizer:
    def __init__(self, config: Optional[BackgroundStabilizerConfig] = None) -> None:
        self.config = config or BackgroundStabilizerConfig()
        self._orb = cv2.ORB_create(nfeatures=self.config.max_features)
        self._reference_keypoints = None
        self._reference_descriptors = None
        self._image_shape = None
        self._last_valid_transform = np.eye(3, dtype=float)
        self._last_result = self._result(False, "not_initialized")

    @staticmethod
    def _result(valid, status, transform=None, **kwargs):
        return StabilizationResult(
            valid=valid,
            status=status,
            transform_current_to_reference=(
                np.eye(3, dtype=float) if transform is None else transform
            ),
            confidence=kwargs.get("confidence", 0.0),
            motion_score=kwargs.get("motion_score", 0.0),
            match_count=kwargs.get("match_count", 0),
            inlier_count=kwargs.get("inlier_count", 0),
            inlier_ratio=kwargs.get("inlier_ratio", 0.0),
            translation_fraction=kwargs.get("translation_fraction", 0.0),
            rotation_degrees=kwargs.get("rotation_degrees", 0.0),
            scale_change=kwargs.get("scale_change", 0.0),
        )

    def _feature_mask(self, exclusion_mask: object, image_shape: Sequence[int]) -> Optional[np.ndarray]:
        if exclusion_mask is None:
            return None
        values = np.asarray(exclusion_mask)
        if values.shape != tuple(image_shape[:2]):
            return None
        excluded = (values > 0).astype(np.uint8) * 255
        if self.config.exclusion_dilation_px > 0:
            size = self.config.exclusion_dilation_px * 2 + 1
            excluded = cv2.dilate(excluded, np.ones((size, size), dtype=np.uint8))
        return cv2.bitwise_not(excluded)

    def update(self, frame: np.ndarray, *, exclusion_mask: object) -> StabilizationResult:
        if not self.config.enabled:
            self._last_result = self._result(False, "disabled")
            return self._last_result
        image = np.asarray(frame)
        if image.ndim not in (2, 3):
            self._last_result = self._result(False, "invalid_frame")
            return self._last_result
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature_mask = self._feature_mask(exclusion_mask, gray.shape)
        if feature_mask is None:
            self._last_result = self._result(False, "missing_or_invalid_exclusion_mask")
            return self._last_result
        keypoints, descriptors = self._orb.detectAndCompute(gray, feature_mask)
        if descriptors is None or len(keypoints) < self.config.min_matches:
            self._last_result = self._result(False, "insufficient_static_features")
            return self._last_result
        if self._reference_descriptors is None:
            self._reference_keypoints = tuple(keypoints)
            self._reference_descriptors = descriptors.copy()
            self._image_shape = tuple(gray.shape)
            self._last_result = self._result(False, "reference_initialized")
            return self._last_result
        if tuple(gray.shape) != self._image_shape:
            self._last_result = self._result(False, "image_shape_changed")
            return self._last_result

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        pairs = matcher.knnMatch(descriptors, self._reference_descriptors, k=2)
        matches = [
            pair[0] for pair in pairs
            if len(pair) == 2 and pair[0].distance < self.config.ratio_test * pair[1].distance
        ]
        if len(matches) < self.config.min_matches:
            self._last_result = self._result(
                False, "insufficient_static_matches", match_count=len(matches)
            )
            return self._last_result
        current_points = np.float32([keypoints[match.queryIdx].pt for match in matches])
        reference_points = np.float32([
            self._reference_keypoints[match.trainIdx].pt for match in matches
        ])
        affine, inliers = cv2.estimateAffinePartial2D(
            current_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold_px,
        )
        if affine is None or inliers is None:
            self._last_result = self._result(False, "ransac_failed", match_count=len(matches))
            return self._last_result
        transform = np.eye(3, dtype=float)
        transform[:2] = affine
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = inlier_count / float(len(matches))
        height, width = gray.shape
        translation_fraction = float(np.linalg.norm(affine[:, 2]) / max(math.hypot(width, height), 1.0))
        scale = math.sqrt(max(float(affine[0, 0] ** 2 + affine[1, 0] ** 2), 0.0))
        scale_change = abs(scale - 1.0)
        rotation_degrees = abs(math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0]))))
        motion_score = max(
            translation_fraction,
            math.radians(rotation_degrees) / math.pi,
            scale_change,
        )
        support = min(len(matches) / float(self.config.target_matches), 1.0)
        confidence = float(np.clip(support * inlier_ratio, 0.0, 1.0))
        valid = (
            inlier_ratio >= self.config.min_inlier_ratio
            and translation_fraction <= self.config.max_translation_fraction
            and rotation_degrees <= self.config.max_rotation_degrees
            and scale_change <= self.config.max_scale_change
            and float(np.linalg.det(transform)) > 0.0
        )
        status = "ok" if valid else "motion_out_of_bounds"
        self._last_result = self._result(
            valid,
            status,
            transform,
            confidence=confidence,
            motion_score=motion_score,
            match_count=len(matches),
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            translation_fraction=translation_fraction,
            rotation_degrees=rotation_degrees,
            scale_change=scale_change,
        )
        if valid:
            self._last_valid_transform = transform.copy()
        return self._last_result

    def get_transform(self) -> np.ndarray:
        return self._last_valid_transform.copy()

    def get_last_result(self) -> StabilizationResult:
        return self._last_result

    def reset(self) -> None:
        self._reference_keypoints = None
        self._reference_descriptors = None
        self._image_shape = None
        self._last_valid_transform = np.eye(3, dtype=float)
        self._last_result = self._result(False, "not_initialized")
