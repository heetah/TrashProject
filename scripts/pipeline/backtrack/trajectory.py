"""Ballistic reverse extrapolation for litter release hypotheses.

The model deliberately works in image coordinates.  A single Kalman filter must
not mix points produced by a changing pseudo-homography.  Callers that have a
fixed, calibrated homography may transform all input points before using this
module.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ReleaseHypothesis:
    """A possible litter release state at one video frame."""

    frame_index: int
    mean_uv: np.ndarray
    covariance_uv: np.ndarray
    velocity_uv: np.ndarray
    model: str
    prior_cost: float

    def __post_init__(self):
        object.__setattr__(self, "mean_uv", np.asarray(self.mean_uv, dtype=float).reshape(2))
        object.__setattr__(
            self,
            "covariance_uv",
            np.asarray(self.covariance_uv, dtype=float).reshape(2, 2),
        )
        object.__setattr__(
            self,
            "velocity_uv",
            np.asarray(self.velocity_uv, dtype=float).reshape(2),
        )


@dataclass(frozen=True)
class BallisticTrajectory:
    """x(t) is linear and y(t) is quadratic, with t measured in seconds."""

    reference_frame: int
    fps: float
    x_coefficients: np.ndarray
    y_coefficients: np.ndarray
    x_coefficient_covariance: np.ndarray
    y_coefficient_covariance: np.ndarray
    residual_sigma_uv: np.ndarray
    observed_frame_min: int
    observed_frame_max: int

    def __post_init__(self):
        object.__setattr__(
            self, "x_coefficients", np.asarray(self.x_coefficients, dtype=float).reshape(2)
        )
        object.__setattr__(
            self, "y_coefficients", np.asarray(self.y_coefficients, dtype=float).reshape(3)
        )
        object.__setattr__(
            self,
            "x_coefficient_covariance",
            np.asarray(self.x_coefficient_covariance, dtype=float).reshape(2, 2),
        )
        object.__setattr__(
            self,
            "y_coefficient_covariance",
            np.asarray(self.y_coefficient_covariance, dtype=float).reshape(3, 3),
        )
        object.__setattr__(
            self,
            "residual_sigma_uv",
            np.asarray(self.residual_sigma_uv, dtype=float).reshape(2),
        )

    def time_at(self, frame_index: int) -> float:
        return (float(frame_index) - float(self.reference_frame)) / self.fps

    def predict(self, frame_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mean position, position covariance, and pixels/second velocity."""

        t = self.time_at(frame_index)
        phi_x = np.asarray([t, 1.0], dtype=float)
        phi_y = np.asarray([t * t, t, 1.0], dtype=float)

        mean = np.asarray(
            [
                float(phi_x @ self.x_coefficients),
                float(phi_y @ self.y_coefficients),
            ],
            dtype=float,
        )
        variance_x = float(phi_x @ self.x_coefficient_covariance @ phi_x)
        variance_y = float(phi_y @ self.y_coefficient_covariance @ phi_y)

        # Residual uncertainty is observation noise.  Parameter uncertainty
        # grows naturally outside the fitted time interval.
        variance_x += float(self.residual_sigma_uv[0] ** 2)
        variance_y += float(self.residual_sigma_uv[1] ** 2)
        covariance = np.diag(
            [max(variance_x, 1e-6), max(variance_y, 1e-6)]
        )
        velocity = np.asarray(
            [
                self.x_coefficients[0],
                2.0 * self.y_coefficients[0] * t + self.y_coefficients[1],
            ],
            dtype=float,
        )
        return mean, covariance, velocity


def _fit_axis(
    design: np.ndarray,
    values: np.ndarray,
    sigma_floor: float,
    weights: Optional[np.ndarray] = None,
):
    if weights is None:
        weights = np.ones(values.size, dtype=float)
    weights = np.clip(np.asarray(weights, dtype=float), 0.01, 1.0)
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_values = values * sqrt_weights
    coefficients, _, _, _ = np.linalg.lstsq(
        weighted_design, weighted_values, rcond=None
    )
    residuals = values - design @ coefficients
    dof = max(float(np.sum(weights)) - float(design.shape[1]), 1.0)
    residual_variance = max(
        float(np.sum(weights * residuals * residuals)) / dof,
        float(sigma_floor) ** 2,
    )
    gram_inverse = np.linalg.pinv(
        design.T @ (weights[:, None] * design), rcond=1e-10
    )
    coefficient_covariance = residual_variance * gram_inverse
    return coefficients, coefficient_covariance, np.sqrt(residual_variance)


def fit_ballistic_trajectory(
    points_uv: Sequence[Sequence[float]],
    frame_indices: Sequence[int],
    fps: float,
    sigma_floor_px: float = 2.0,
    min_points: int = 3,
    confidences: Optional[Sequence[float]] = None,
) -> Optional[BallisticTrajectory]:
    """Fit x(t)=a*t+b and y(t)=c*t^2+d*t+e.

    Using seconds rather than frame numbers keeps velocity, acceleration, and
    temporal priors invariant when the same motion is sampled at another FPS.
    """

    points = np.asarray(points_uv, dtype=float)
    frames = np.asarray(frame_indices, dtype=int)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or points.shape[0] != frames.size
        or points.shape[0] < int(min_points)
        or not np.isfinite(points).all()
        or not np.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        return None

    # Duplicate frames make X'X artificially confident. Average them first.
    if confidences is None:
        confidence_values = np.ones(frames.size, dtype=float)
    else:
        confidence_values = np.asarray(confidences, dtype=float)
        if confidence_values.size != frames.size:
            return None
        confidence_values = np.clip(
            np.nan_to_num(confidence_values, nan=0.05, posinf=1.0, neginf=0.05),
            0.05,
            1.0,
        )

    unique_frames = np.unique(frames)
    if unique_frames.size < int(min_points):
        return None
    unique_points = []
    unique_confidences = []
    for frame in unique_frames:
        mask = frames == frame
        frame_weights = confidence_values[mask]
        unique_points.append(
            np.average(points[mask], axis=0, weights=frame_weights)
        )
        unique_confidences.append(float(np.max(frame_weights)))
    unique_points = np.asarray(unique_points, dtype=float)
    unique_confidences = np.asarray(unique_confidences, dtype=float)
    fit_weights = unique_confidences * unique_confidences

    reference_frame = int(unique_frames[0])
    times = (unique_frames.astype(float) - reference_frame) / float(fps)
    design_x = np.column_stack([times, np.ones_like(times)])
    design_y = np.column_stack([times * times, times, np.ones_like(times)])
    if np.linalg.matrix_rank(design_x) < 2 or np.linalg.matrix_rank(design_y) < 3:
        return None

    x_coef, x_cov, sigma_x = _fit_axis(
        design_x, unique_points[:, 0], sigma_floor_px, fit_weights
    )
    y_coef, y_cov, sigma_y = _fit_axis(
        design_y, unique_points[:, 1], sigma_floor_px, fit_weights
    )
    # RANSAC-lite: one detector jump can badly rotate a backward extrapolation.
    # With >=5 samples, remove the largest joint residual once and refit both
    # axes using the same inlier set. Clean data keeps every sample.
    if unique_frames.size >= 5:
        residual_x = unique_points[:, 0] - design_x @ x_coef
        residual_y = unique_points[:, 1] - design_y @ y_coef
        joint_residual_sq = residual_x * residual_x + residual_y * residual_y
        outlier_index = int(np.argmax(joint_residual_sq))
        robust_scale_sq = max(
            float(np.median(joint_residual_sq)),
            2.0 * float(sigma_floor_px) ** 2,
        )
        if float(joint_residual_sq[outlier_index]) > 9.0 * robust_scale_sq:
            keep = np.ones(unique_frames.size, dtype=bool)
            keep[outlier_index] = False
            x_coef, x_cov, sigma_x = _fit_axis(
                design_x[keep],
                unique_points[keep, 0],
                sigma_floor_px,
                fit_weights[keep],
            )
            y_coef, y_cov, sigma_y = _fit_axis(
                design_y[keep],
                unique_points[keep, 1],
                sigma_floor_px,
                fit_weights[keep],
            )
    return BallisticTrajectory(
        reference_frame=reference_frame,
        fps=float(fps),
        x_coefficients=x_coef,
        y_coefficients=y_coef,
        x_coefficient_covariance=x_cov,
        y_coefficient_covariance=y_cov,
        residual_sigma_uv=np.asarray([sigma_x, sigma_y], dtype=float),
        observed_frame_min=int(unique_frames[0]),
        observed_frame_max=int(unique_frames[-1]),
    )


def airborne_prefix(
    points_uv: Sequence[Sequence[float]],
    frame_indices: Sequence[int],
    fps: float,
    bounce_velocity_px_s: float = 15.0,
    rest_velocity_px_s: float = 6.0,
):
    """Remove post-impact bounce/rest samples before ballistic fitting."""

    count = min(len(points_uv), len(frame_indices))
    if count <= 0:
        return [], []
    if count < 2 or not np.isfinite(float(fps)) or float(fps) <= 0.0:
        return list(points_uv)[-count:], list(frame_indices)[-count:]
    points = np.asarray(points_uv[-count:], dtype=float)
    frames = np.asarray(frame_indices[-count:], dtype=int)
    cut = count
    previous_vy = None
    rest_run = 0
    for index in range(1, count):
        dt_seconds = max(
            (int(frames[index]) - int(frames[index - 1])) / float(fps),
            1.0 / float(fps),
        )
        velocity = (points[index] - points[index - 1]) / dt_seconds
        if float(np.linalg.norm(velocity)) < float(rest_velocity_px_s):
            rest_run += 1
            if rest_run >= 2:
                cut = max(index - 1, 2)
                break
        else:
            rest_run = 0
        if (
            previous_vy is not None
            and previous_vy > float(bounce_velocity_px_s)
            and float(velocity[1]) < -float(bounce_velocity_px_s)
        ):
            cut = index
            break
        previous_vy = float(velocity[1])
    return points[:cut].tolist(), frames[:cut].tolist()


def _birth_fallback(
    points_uv: Sequence[Sequence[float]],
    frame_indices: Sequence[int],
    birth_frame: int,
    sigma_floor_px: float,
    prior_cost: float,
) -> ReleaseHypothesis:
    points = np.asarray(points_uv, dtype=float)
    frames = np.asarray(frame_indices, dtype=int)
    if points.ndim == 2 and points.shape[0] and points.shape[1] == 2:
        idx = int(np.argmin(np.abs(frames - int(birth_frame)))) if frames.size else 0
        mean = points[min(idx, points.shape[0] - 1)]
        if points.shape[0] >= 2:
            spread = np.maximum(np.std(points, axis=0), float(sigma_floor_px))
        else:
            spread = np.full(2, max(float(sigma_floor_px), 4.0))
    else:
        mean = np.zeros(2, dtype=float)
        spread = np.full(2, max(float(sigma_floor_px), 8.0))
    return ReleaseHypothesis(
        frame_index=int(birth_frame),
        mean_uv=mean,
        covariance_uv=np.diag(spread * spread),
        velocity_uv=np.zeros(2, dtype=float),
        model="birth_fallback",
        prior_cost=float(prior_cost),
    )


def build_release_hypotheses(
    points_uv: Sequence[Sequence[float]],
    frame_indices: Sequence[int],
    birth_frame: int,
    max_back_frames: int,
    fps: float,
    sigma_floor_px: float = 2.0,
    min_points: int = 3,
    extrapolation_cost_per_second: float = 0.35,
    confidences: Optional[Sequence[float]] = None,
    fallback_prior_cost: float = 6.0,
) -> List[ReleaseHypothesis]:
    """Fit a trajectory and enumerate release frames from birth backwards.

    A birth-only fallback is always returned when fitting is impossible, so the
    caller can route to a dustbin instead of crashing or inventing certainty.
    """

    model = fit_ballistic_trajectory(
        points_uv,
        frame_indices,
        fps=fps,
        sigma_floor_px=sigma_floor_px,
        min_points=min_points,
        confidences=confidences,
    )
    if model is None:
        return [
            _birth_fallback(
                points_uv,
                frame_indices,
                birth_frame,
                sigma_floor_px,
                fallback_prior_cost,
            )
        ]

    hypotheses = []
    lower = int(birth_frame) - max(int(max_back_frames), 0)
    for frame_index in range(lower, int(birth_frame) + 1):
        mean, covariance, velocity = model.predict(frame_index)
        seconds_before_birth = (int(birth_frame) - frame_index) / float(fps)
        hypotheses.append(
            ReleaseHypothesis(
                frame_index=frame_index,
                mean_uv=mean,
                covariance_uv=covariance,
                velocity_uv=velocity,
                model="ballistic",
                prior_cost=max(seconds_before_birth, 0.0)
                * float(extrapolation_cost_per_second),
            )
        )
    return hypotheses


# Short aliases used by integration code and research notebooks.
fit_ballistic = fit_ballistic_trajectory
release_hypotheses = build_release_hypotheses
