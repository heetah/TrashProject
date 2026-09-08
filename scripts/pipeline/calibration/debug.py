"""Opt-in calibration visualizations; all renderers operate on frame copies."""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np

from .motion_field import render_motion_debug
from .transform import project_image_points


def render_calibration_debug(
    frame: np.ndarray,
    tracks: Mapping[object, Sequence[object]],
    observations: Sequence[object],
    motion_field: object,
    flow_clusters: Sequence[object],
    homography: object,
    state: Mapping[str, object],
) -> np.ndarray:
    """Render image tracks, flow field, pseudo-ground inset and state metrics."""

    canvas = render_motion_debug(frame, observations, motion_field, flow_clusters)
    height, width = canvas.shape[:2]
    palette = ((255, 180, 40), (40, 210, 255), (90, 220, 100), (220, 90, 210))
    projected_tracks = []
    for track_index, points in enumerate(tracks.values()):
        image_points = np.asarray([point.image_point for point in points], dtype=float)
        if len(image_points) < 1:
            continue
        color = palette[track_index % len(palette)]
        polyline = np.rint(image_points).astype(np.int32).reshape(-1, 1, 2)
        if len(polyline) >= 2:
            cv2.polylines(canvas, [polyline], False, color, 1, cv2.LINE_AA)
        for point in polyline[:, 0]:
            cv2.circle(canvas, tuple(point), 2, color, -1, cv2.LINE_AA)
        projection = project_image_points(image_points, homography)
        valid = projection.points[projection.valid_mask]
        if len(valid):
            projected_tracks.append((valid, color))

    inset_width = max(180, int(width * 0.34))
    inset_height = max(130, int(height * 0.34))
    inset_width = min(inset_width, width)
    inset_height = min(inset_height, height)
    x0, y0 = width - inset_width, height - inset_height
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (width - 1, height - 1), (18, 18, 18), -1)
    canvas = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0.0)
    cv2.putText(canvas, "pseudo-ground", (x0 + 8, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    if projected_tracks:
        all_points = np.vstack([points for points, _ in projected_tracks])
        minimum = np.min(all_points, axis=0)
        maximum = np.max(all_points, axis=0)
        span = np.maximum(maximum - minimum, 1e-9)
        for points, color in projected_tracks:
            normalized = (points - minimum) / span
            inset_points = np.column_stack((
                x0 + 10 + normalized[:, 0] * max(inset_width - 20, 1),
                y0 + 26 + (1.0 - normalized[:, 1]) * max(inset_height - 36, 1),
            ))
            polyline = np.rint(inset_points).astype(np.int32).reshape(-1, 1, 2)
            if len(polyline) >= 2:
                cv2.polylines(canvas, [polyline], False, color, 1, cv2.LINE_AA)

    metric = state.get("last_metric") or {}
    lines = (
        f"H {state.get('status', 'UNKNOWN')} v{state.get('homography_version', 0)}",
        f"conf={float(state.get('confidence', 0.0)):.3f} tracks={metric.get('num_valid_tracks', 0)}",
        f"coverage={float(metric.get('spatial_coverage', 0.0) or 0.0):.3f} loss={metric.get('total_loss')}",
        f"alpha={float(metric.get('update_alpha', 0.0) or 0.0):.4f} "
        f"dH={float(metric.get('update_magnitude', 0.0) or 0.0):.5f}",
    )
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (8, 18 + 17 * index), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas
