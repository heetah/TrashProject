"""Numerically safe initial image-to-pseudo-ground transformation.

This Phase-3 module provides a deterministic relative-coordinate baseline and
validation primitives.  It does not claim metric calibration and does not
update production attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class HomographyValidationConfig:
    min_abs_determinant: float = 1e-12
    max_condition_number: float = 1e6
    min_abs_denominator: float = 1e-6
    max_denominator_ratio: float = 100.0
    max_abs_coordinate: float = 20.0
    min_projected_area: float = 1e-6
    require_orientation_preserved: bool = True
    grid_rows: int = 5
    grid_cols: int = 7

    def __post_init__(self) -> None:
        values = (
            self.min_abs_determinant,
            self.max_condition_number,
            self.min_abs_denominator,
            self.max_denominator_ratio,
            self.max_abs_coordinate,
            self.min_projected_area,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("homography validation bounds must be finite and positive")
        if self.grid_rows < 2 or self.grid_cols < 2:
            raise ValueError("homography validation grid must be at least 2x2")

    @classmethod
    def from_env(cls) -> "HomographyValidationConfig":
        defaults = cls()
        return cls(
            min_abs_determinant=_env_float(
                "DYNAMIC_HOMOGRAPHY_MIN_ABS_DET", defaults.min_abs_determinant
            ),
            max_condition_number=_env_float(
                "DYNAMIC_HOMOGRAPHY_MAX_CONDITION", defaults.max_condition_number
            ),
            min_abs_denominator=_env_float(
                "DYNAMIC_HOMOGRAPHY_MIN_DENOMINATOR", defaults.min_abs_denominator
            ),
            max_denominator_ratio=_env_float(
                "DYNAMIC_HOMOGRAPHY_MAX_DENOM_RATIO", defaults.max_denominator_ratio
            ),
            max_abs_coordinate=_env_float(
                "DYNAMIC_HOMOGRAPHY_MAX_ABS_COORD", defaults.max_abs_coordinate
            ),
            min_projected_area=_env_float(
                "DYNAMIC_HOMOGRAPHY_MIN_PROJECTED_AREA", defaults.min_projected_area
            ),
            require_orientation_preserved=_env_bool(
                "DYNAMIC_HOMOGRAPHY_REQUIRE_ORIENTATION",
                defaults.require_orientation_preserved,
            ),
            grid_rows=defaults.grid_rows,
            grid_cols=defaults.grid_cols,
        )


@dataclass(frozen=True)
class ProjectionResult:
    points: np.ndarray
    valid_mask: np.ndarray
    denominators: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", np.asarray(self.points, dtype=float))
        object.__setattr__(self, "valid_mask", np.asarray(self.valid_mask, dtype=bool))
        object.__setattr__(self, "denominators", np.asarray(self.denominators, dtype=float))


@dataclass(frozen=True)
class HomographyValidationReport:
    valid: bool
    reasons: Tuple[str, ...]
    normalized_homography: np.ndarray
    determinant: Optional[float]
    condition_number: Optional[float]
    min_abs_denominator: Optional[float]
    denominator_ratio: Optional[float]
    projected_signed_area: Optional[float]
    projected_bounds: Optional[Tuple[float, float, float, float]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "normalized_homography",
            np.asarray(self.normalized_homography, dtype=float).reshape(3, 3),
        )

    def as_dict(self) -> dict:
        return {
            "valid": bool(self.valid),
            "reasons": list(self.reasons),
            "determinant": self.determinant,
            "condition_number": self.condition_number,
            "min_abs_denominator": self.min_abs_denominator,
            "denominator_ratio": self.denominator_ratio,
            "projected_signed_area": self.projected_signed_area,
            "projected_bounds": (
                list(self.projected_bounds) if self.projected_bounds is not None else None
            ),
        }


def normalize_homography(homography: object, epsilon: float = 1e-12) -> np.ndarray:
    """Resolve projective scale without mutating the caller's matrix."""

    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")
    if not np.isfinite(matrix).all():
        raise ValueError("homography must contain only finite values")
    scale = float(matrix[2, 2])
    if abs(scale) <= epsilon:
        scale = float(np.max(np.abs(matrix)))
    if not math.isfinite(scale) or abs(scale) <= epsilon:
        raise ValueError("homography has no stable normalization scale")
    normalized = matrix / scale
    if normalized[2, 2] < 0.0:
        normalized = -normalized
    return normalized


def initial_normalized_image_homography(image_shape: Sequence[int]) -> np.ndarray:
    """Map the image rectangle to a stable relative [0,1]x[0,1] plane."""

    try:
        height, width = int(image_shape[0]), int(image_shape[1])
    except (TypeError, ValueError, IndexError) as error:
        raise ValueError("image_shape must provide positive height and width") from error
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must provide positive height and width")
    return np.asarray([
        [1.0 / float(width), 0.0, 0.0],
        [0.0, 1.0 / float(height), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def project_image_points(
    points: object,
    homography: object,
    *,
    min_abs_denominator: float = 1e-6,
    max_abs_coordinate: float = 20.0,
) -> ProjectionResult:
    """Project N image points and mark unsafe rows as NaN instead of exploding."""

    matrix = normalize_homography(homography)
    values = np.asarray(points, dtype=float)
    if values.ndim == 1 and values.size == 2:
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if not math.isfinite(float(min_abs_denominator)) or min_abs_denominator <= 0.0:
        raise ValueError("min_abs_denominator must be finite and positive")
    if not math.isfinite(float(max_abs_coordinate)) or max_abs_coordinate <= 0.0:
        raise ValueError("max_abs_coordinate must be finite and positive")

    homogeneous = np.column_stack((values, np.ones(len(values), dtype=float)))
    transformed = (matrix @ homogeneous.T).T
    denominators = transformed[:, 2]
    finite_input = np.isfinite(values).all(axis=1)
    safe_denominator = np.isfinite(denominators) & (
        np.abs(denominators) >= float(min_abs_denominator)
    )
    projected = np.full((len(values), 2), np.nan, dtype=float)
    initial_valid = finite_input & safe_denominator
    projected[initial_valid] = (
        transformed[initial_valid, :2] / denominators[initial_valid, None]
    )
    bounded = np.isfinite(projected).all(axis=1) & (
        np.max(np.abs(projected), axis=1, initial=-np.inf)
        <= float(max_abs_coordinate)
    )
    valid = initial_valid & bounded
    projected[~valid] = np.nan
    return ProjectionResult(projected, valid, denominators)


def image_to_ground(
    points: object,
    homography: object,
    *,
    min_abs_denominator: float = 1e-6,
    max_abs_coordinate: float = 20.0,
) -> np.ndarray:
    """Return projected coordinates; unsafe inputs are represented by NaN rows."""

    return project_image_points(
        points,
        homography,
        min_abs_denominator=min_abs_denominator,
        max_abs_coordinate=max_abs_coordinate,
    ).points


def _signed_polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def validate_homography(
    homography: object,
    image_shape: Sequence[int],
    config: Optional[HomographyValidationConfig] = None,
) -> HomographyValidationReport:
    """Validate matrix algebra and its behavior over the complete image ROI."""

    cfg = config or HomographyValidationConfig()
    reasons = []
    try:
        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0:
            raise ValueError
    except (TypeError, ValueError, IndexError):
        height, width = 0, 0
        reasons.append("invalid_image_shape")
    try:
        matrix = normalize_homography(homography)
    except (TypeError, ValueError):
        matrix = np.eye(3, dtype=float)
        reasons.append("invalid_matrix")
        return HomographyValidationReport(
            False, tuple(reasons), matrix, None, None, None, None, None, None
        )

    determinant = float(np.linalg.det(matrix))
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:
        condition = float("inf")
    if abs(determinant) < cfg.min_abs_determinant:
        reasons.append("near_singular")
    if not math.isfinite(condition) or condition > cfg.max_condition_number:
        reasons.append("ill_conditioned")
    if height <= 0 or width <= 0:
        return HomographyValidationReport(
            False, tuple(reasons), matrix, determinant, condition,
            None, None, None, None,
        )

    xs = np.linspace(0.0, float(width), cfg.grid_cols)
    ys = np.linspace(0.0, float(height), cfg.grid_rows)
    grid = np.asarray([(x, y) for y in ys for x in xs], dtype=float)
    projection = project_image_points(
        grid,
        matrix,
        min_abs_denominator=cfg.min_abs_denominator,
        max_abs_coordinate=cfg.max_abs_coordinate,
    )
    abs_denominators = np.abs(projection.denominators)
    finite_denominators = abs_denominators[np.isfinite(abs_denominators)]
    minimum_denominator = (
        float(np.min(finite_denominators)) if finite_denominators.size else None
    )
    denominator_ratio = None
    if minimum_denominator is not None and minimum_denominator > 0.0:
        denominator_ratio = float(np.max(finite_denominators) / minimum_denominator)
    if not np.all(projection.valid_mask):
        reasons.append("unsafe_projection")
    if denominator_ratio is None or denominator_ratio > cfg.max_denominator_ratio:
        reasons.append("extreme_denominator_variation")
    finite_raw_denominators = projection.denominators[np.isfinite(projection.denominators)]
    if (
        finite_raw_denominators.size
        and np.min(finite_raw_denominators) < 0.0 < np.max(finite_raw_denominators)
    ):
        reasons.append("denominator_sign_change")

    corners = np.asarray([
        [0.0, 0.0], [float(width), 0.0],
        [float(width), float(height)], [0.0, float(height)],
    ])
    corner_projection = project_image_points(
        corners,
        matrix,
        min_abs_denominator=cfg.min_abs_denominator,
        max_abs_coordinate=cfg.max_abs_coordinate,
    )
    signed_area = None
    bounds = None
    if np.all(corner_projection.valid_mask):
        projected_corners = corner_projection.points
        signed_area = _signed_polygon_area(projected_corners)
        bounds = (
            float(np.min(projected_corners[:, 0])),
            float(np.min(projected_corners[:, 1])),
            float(np.max(projected_corners[:, 0])),
            float(np.max(projected_corners[:, 1])),
        )
        if abs(signed_area) < cfg.min_projected_area:
            reasons.append("collapsed_ground_region")
        if cfg.require_orientation_preserved and signed_area <= 0.0:
            reasons.append("orientation_reversed")
    else:
        reasons.append("unsafe_image_corners")

    reasons = list(dict.fromkeys(reasons))
    return HomographyValidationReport(
        valid=not reasons,
        reasons=tuple(reasons),
        normalized_homography=matrix,
        determinant=determinant,
        condition_number=condition,
        min_abs_denominator=minimum_denominator,
        denominator_ratio=denominator_ratio,
        projected_signed_area=signed_area,
        projected_bounds=bounds,
    )


def initial_homography_snapshot(image_shape: Sequence[int]) -> dict:
    """Create the auditable Phase-3 relative-scale baseline snapshot."""

    matrix = initial_normalized_image_homography(image_shape)
    report = validate_homography(
        matrix, image_shape, HomographyValidationConfig.from_env()
    )
    return {
        "status": "COLLECTING",
        "version": 0,
        "confidence": 0.0,
        "relative_scale_only": True,
        "source": "normalized_image_coordinates",
        "homography": matrix.tolist(),
        "validation": report.as_dict(),
        "affects_attribution": False,
    }

