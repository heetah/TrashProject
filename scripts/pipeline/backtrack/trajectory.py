"""Ballistic reverse extrapolation for litter release hypotheses.

The model deliberately works in image coordinates.  A single Kalman filter must
not mix points produced by a changing pseudo-homography.  Callers that have a
fixed, calibrated homography may transform all input points before using this
module.
"""

from dataclasses import dataclass, replace
import math
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
    observation_gap_frames: Optional[int] = None
    zero_cost_window_start_frame: Optional[int] = None
    zero_cost_window_end_frame: Optional[int] = None
    window_prior_cost: float = 0.0
    direction_consistency: Optional[float] = None
    source_direction_uv: Optional[Tuple[float, float]] = None
    search_truncated: bool = False
    truncation_reason: Optional[str] = None
    time_prior_policy: str = "observation_gap"
    release_back_seconds: Optional[float] = None
    max_release_back_seconds: Optional[float] = None

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
        if self.source_direction_uv is not None:
            object.__setattr__(
                self,
                "source_direction_uv",
                tuple(float(value) for value in self.source_direction_uv),
            )


def _ordered_unique_observations(points_uv, frame_indices):
    """Return one mean point per frame in chronological order."""

    points = np.asarray(points_uv, dtype=float)
    frames = np.asarray(frame_indices, dtype=int)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or points.shape[0] != frames.size
        or not np.isfinite(points).all()
    ):
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=int)
    unique_frames = np.unique(frames)
    unique_points = np.asarray(
        [np.mean(points[frames == frame], axis=0) for frame in unique_frames],
        dtype=float,
    )
    return unique_points, unique_frames


def _early_motion_diagnostics(points_uv, frame_indices, fps):
    """Measure early motion from B0/B1/B2 without using confirm_frame."""

    points, frames = _ordered_unique_observations(points_uv, frame_indices)
    if points.shape[0] < 2 or float(fps) <= 0.0:
        return None, None
    dt01 = (int(frames[1]) - int(frames[0])) / float(fps)
    if dt01 <= 0.0:
        return None, None
    velocity01 = (points[1] - points[0]) / dt01
    speed01 = float(np.linalg.norm(velocity01))
    source_direction = (
        tuple((-velocity01 / speed01).tolist()) if speed01 > 1e-9 else None
    )
    if points.shape[0] < 3:
        return source_direction, None
    dt12 = (int(frames[2]) - int(frames[1])) / float(fps)
    if dt12 <= 0.0:
        return source_direction, None
    velocity12 = (points[2] - points[1]) / dt12
    speed12 = float(np.linalg.norm(velocity12))
    if speed01 <= 1e-9 or speed12 <= 1e-9:
        return source_direction, None
    consistency = float(
        np.clip(
            velocity01 @ velocity12 / (speed01 * speed12),
            -1.0,
            1.0,
        )
    )
    return source_direction, consistency


def _observation_gap_policy(points_uv, frame_indices, birth_frame):
    """Build I0=[B0-(B1-B0), B0] from accepted detector observations."""

    _, frames = _ordered_unique_observations(points_uv, frame_indices)
    if frames.size < 2:
        return None, int(birth_frame), int(birth_frame)
    gap_frames = max(int(frames[1]) - int(frames[0]), 1)
    first_observation = int(frames[0])
    return gap_frames, first_observation - gap_frames, first_observation


def _window_prior(
    frame_index,
    window_start,
    window_end,
    gap_frames,
    weight,
    *,
    fps,
    forward_cost_per_second,
):
    if gap_frames is None or int(gap_frames) <= 0:
        return 0.0
    if int(frame_index) < int(window_start):
        outside_frames = int(window_start) - int(frame_index)
    elif int(frame_index) > int(window_end):
        # Post-birth hypotheses lie on the observed airborne trajectory. They
        # are not an unknown reverse gap, so retain the original seconds-based
        # prior instead of dividing them by H0.
        return (
            float(forward_cost_per_second)
            * float(int(frame_index) - int(window_end))
            / max(float(fps), 1e-9)
        )
    else:
        outside_frames = 0
    return float(weight) * float(outside_frames) / float(gap_frames)


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


def _two_point_constant_velocity_hypotheses(
    points_uv: Sequence[Sequence[float]],
    frame_indices: Sequence[int],
    birth_frame: int,
    max_back_frames: int,
    fps: float,
    sigma_floor_px: float,
    prior_cost: float,
    window_prior_weight: float,
    forward_cost_per_second: float,
) -> List[ReleaseHypothesis]:
    """Build conservative reverse hypotheses from exactly two observations."""

    points = np.asarray(points_uv, dtype=float)
    frames = np.asarray(frame_indices, dtype=int)
    order = np.argsort(frames, kind="stable")
    points = points[order]
    frames = frames[order]
    unique_frames = np.unique(frames)
    if (
        points.shape != (2, 2)
        or unique_frames.size != 2
        or not np.isfinite(points).all()
        or not np.isfinite(float(fps))
        or float(fps) <= 0.0
    ):
        return []
    frame_delta = int(frames[1]) - int(frames[0])
    if frame_delta <= 0:
        return []
    velocity = (points[1] - points[0]) / (frame_delta / float(fps))
    speed = float(np.linalg.norm(velocity))
    if speed > 1e-9:
        direction = velocity / speed
    else:
        direction = np.asarray([1.0, 0.0], dtype=float)
    normal = np.asarray([-direction[1], direction[0]], dtype=float)

    gap_frames, window_start, window_end = _observation_gap_policy(
        points, frames, birth_frame
    )
    source_direction, direction_consistency = _early_motion_diagnostics(
        points, frames, fps
    )
    # max_back_frames is a computational guard, not a physical rejection rule.
    horizon_frames = max(int(max_back_frames), 0)
    lower = int(birth_frame) - horizon_frames
    hypotheses = []
    for frame_index in range(lower, int(birth_frame) + 1):
        seconds_from_first = (
            int(frame_index) - int(frames[0])
        ) / float(fps)
        seconds_before_birth = max(
            (int(birth_frame) - int(frame_index)) / float(fps), 0.0
        )
        mean = points[0] + velocity * seconds_from_first
        # Two samples determine velocity but cannot estimate acceleration.
        # Uncertainty grows anisotropically backward: most along the observed
        # motion direction, less across it. Physical distance gates remain
        # independent in the cost layer.
        sigma_along = float(sigma_floor_px) + 0.35 * speed * seconds_before_birth
        sigma_cross = float(sigma_floor_px) + 0.15 * speed * seconds_before_birth
        covariance = (
            sigma_along ** 2 * np.outer(direction, direction)
            + sigma_cross ** 2 * np.outer(normal, normal)
        )
        window_cost = _window_prior(
            frame_index,
            window_start,
            window_end,
            gap_frames,
            window_prior_weight,
            fps=fps,
            forward_cost_per_second=forward_cost_per_second,
        )
        hypotheses.append(ReleaseHypothesis(
            frame_index=int(frame_index),
            mean_uv=mean,
            covariance_uv=covariance,
            velocity_uv=velocity,
            model="constant_velocity_2point",
            prior_cost=float(prior_cost) + window_cost,
            observation_gap_frames=gap_frames,
            zero_cost_window_start_frame=window_start,
            zero_cost_window_end_frame=window_end,
            window_prior_cost=window_cost,
            direction_consistency=direction_consistency,
            source_direction_uv=source_direction,
            search_truncated=bool(horizon_frames > 0),
            truncation_reason=(
                "max_back_frames_computational_guard"
                if horizon_frames > 0 else None
            ),
        ))
    return hypotheses


def _build_release_hypotheses(
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
    two_point_max_back_seconds: float = 0.3,
    two_point_prior_cost: float = 1.0,
    max_forward_release_seconds: float = 0.5,
    window_prior_weight: float = 0.35,
) -> List[ReleaseHypothesis]:
    """Fit a trajectory and enumerate release frames from birth backwards.

    B0/B1 define a zero-cost release-time window. Frames outside that interval
    remain candidates but receive a soft prior. ``max_back_frames`` only bounds
    computation; it is reported as truncation rather than physical evidence.

    ``two_point_max_back_seconds`` and ``extrapolation_cost_per_second`` remain
    accepted for replay compatibility, but no longer impose a second physical
    cutoff or penalize frames inside the B0/B1 window.
    """

    point_count = min(len(points_uv), len(frame_indices))
    if point_count == 2:
        hypotheses = _two_point_constant_velocity_hypotheses(
            list(points_uv)[-2:],
            list(frame_indices)[-2:],
            birth_frame=birth_frame,
            max_back_frames=max_back_frames,
            fps=fps,
            sigma_floor_px=sigma_floor_px,
            prior_cost=two_point_prior_cost,
            window_prior_weight=window_prior_weight,
            forward_cost_per_second=extrapolation_cost_per_second,
        )
        if hypotheses:
            return hypotheses

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

    gap_frames, window_start, window_end = _observation_gap_policy(
        points_uv, frame_indices, birth_frame
    )
    source_direction, direction_consistency = _early_motion_diagnostics(
        points_uv, frame_indices, fps
    )
    hypotheses = []
    lower = int(birth_frame) - max(int(max_back_frames), 0)
    upper = min(
        int(model.observed_frame_max),
        int(birth_frame) + max(
            int(round(float(max_forward_release_seconds) * float(fps))), 0
        ),
    )
    for frame_index in range(lower, upper + 1):
        mean, covariance, velocity = model.predict(frame_index)
        window_cost = _window_prior(
            frame_index,
            window_start,
            window_end,
            gap_frames,
            window_prior_weight,
            fps=fps,
            forward_cost_per_second=extrapolation_cost_per_second,
        )
        hypotheses.append(
            ReleaseHypothesis(
                frame_index=frame_index,
                mean_uv=mean,
                covariance_uv=covariance,
                velocity_uv=velocity,
                model="ballistic",
                prior_cost=window_cost,
                observation_gap_frames=gap_frames,
                zero_cost_window_start_frame=window_start,
                zero_cost_window_end_frame=window_end,
                window_prior_cost=window_cost,
                direction_consistency=direction_consistency,
                source_direction_uv=source_direction,
                search_truncated=bool(max(int(max_back_frames), 0) > 0),
                truncation_reason=(
                    "max_back_frames_computational_guard"
                    if max(int(max_back_frames), 0) > 0 else None
                ),
            )
        )
    return hypotheses


def validate_release_time_policy(max_seconds, soft_seconds, weight):
    if max_seconds is None:
        return
    if not all(math.isfinite(float(v)) for v in (max_seconds, soft_seconds, weight)):
        raise ValueError("release time policy values must be finite")
    if not 0 <= soft_seconds < max_seconds or weight < 0:
        raise ValueError("require 0 <= release soft seconds < max seconds and weight >= 0")


def build_release_hypotheses(
    points_uv, frame_indices, birth_frame, max_back_frames, fps,
    sigma_floor_px=2.0, min_points=3, extrapolation_cost_per_second=0.35,
    confidences=None, fallback_prior_cost=6.0,
    two_point_max_back_seconds=0.3, two_point_prior_cost=1.0,
    max_forward_release_seconds=0.5, window_prior_weight=0.35,
    max_release_back_seconds=None, release_soft_seconds=0.25,
    release_time_weight=1.0,
) -> List[ReleaseHypothesis]:
    """Enumerate legacy releases or an explicit seconds-based quadratic prior.

    The new policy replaces only the backward window cost. Model fallback and
    two-point priors, and the existing post-birth airborne penalty, survive.
    It is opt-in until reviewed replay supports production promotion.
    """
    validate_release_time_policy(
        max_release_back_seconds, release_soft_seconds, release_time_weight
    )
    horizon = max_back_frames
    if max_release_back_seconds is not None:
        if not math.isfinite(float(fps)) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        physical_frames = max(0, int(math.floor(max_release_back_seconds * fps)))
        available_frames = min(physical_frames, max(0, int(birth_frame)))
        horizon = min(max(0, int(max_back_frames)), available_frames)
    hypotheses = _build_release_hypotheses(
        points_uv, frame_indices, birth_frame, horizon, fps,
        sigma_floor_px, min_points, extrapolation_cost_per_second,
        confidences, fallback_prior_cost, two_point_max_back_seconds,
        two_point_prior_cost, max_forward_release_seconds, window_prior_weight,
    )
    if max_release_back_seconds is None:
        return hypotheses
    result = []
    for item in hypotheses:
        if item.frame_index < 0:
            continue
        backward_seconds = max(0.0, (birth_frame - item.frame_index) / fps)
        if backward_seconds > max_release_back_seconds:
            continue
        window_cost = item.window_prior_cost
        if release_time_weight == 0.0:
            # Production keeps release time as a bounded search dimension but
            # charges no backward or forward release-time cost. Model-quality
            # priors (two-point/fallback) remain separate in ``prior_cost``.
            window_cost = 0.0
        elif item.frame_index <= birth_frame:
            fraction = max(0.0, backward_seconds - release_soft_seconds) / (
                max_release_back_seconds - release_soft_seconds
            )
            window_cost = release_time_weight * fraction ** 2
        truncated = horizon < available_frames
        result.append(replace(
            item,
            prior_cost=item.prior_cost - item.window_prior_cost + window_cost,
            window_prior_cost=window_cost,
            zero_cost_window_start_frame=max(
                0, birth_frame - int(math.floor(release_soft_seconds * fps))
            ),
            zero_cost_window_end_frame=birth_frame,
            time_prior_policy=(
                "seconds_bound_only"
                if release_time_weight == 0.0 else "seconds_quadratic"
            ),
            release_back_seconds=backward_seconds,
            max_release_back_seconds=max_release_back_seconds,
            search_truncated=truncated,
            truncation_reason=("max_back_frames_computational_guard" if truncated else None),
        ))
    return result


# Short aliases used by integration code and research notebooks.
fit_ballistic = fit_ballistic_trajectory
release_hypotheses = build_release_hypotheses
