# -*- coding: utf-8 -*-
"""Diagnostic-only visible-mask summaries for Smart Backtrack research.

These measurements characterize the relation between a raw litter candidate
and an observed vehicle silhouette.  They are deliberately not route costs and
must not be interpreted as ordinal or metric depth.  The signed value follows
OpenCV's `pointPolygonTest` convention: positive means the litter center is
inside the visible polygon and negative means outside.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np


SCHEMA_NAME = "mask-litter-diagnostics/v1"
_VEHICLE_CLASSES = frozenset(("vehicle", "scooter"))
_OBSERVED_SOURCES = frozenset(("seg_track", "seg_predict"))


def _finite_box(value, minimum_items=4):
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < minimum_items or not np.isfinite(array[:minimum_items]).all():
        return None
    return tuple(float(item) for item in array[:minimum_items])


def _finite_polygon(value):
    try:
        polygon = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if (
        polygon.ndim != 2
        or polygon.shape[0] < 3
        or polygon.shape[1] != 2
        or not np.isfinite(polygon).all()
    ):
        return None
    if abs(float(cv2.contourArea(polygon))) <= 0.0:
        return None
    return polygon


def _box_intersection_fraction(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    inner_w = max(0.0, ix2 - ix1)
    inner_h = max(0.0, iy2 - iy1)
    inner_area = inner_w * inner_h
    if inner_area <= 0.0:
        return None
    overlap_w = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    overlap_h = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    return min(max((overlap_w * overlap_h) / inner_area, 0.0), 1.0)


def _mask_overlap_fraction(litter_box, polygon):
    lx1, ly1, lx2, ly2 = litter_box
    rx1 = int(math.floor(min(lx1, lx2)))
    ry1 = int(math.floor(min(ly1, ly2)))
    rx2 = int(math.ceil(max(lx1, lx2)))
    ry2 = int(math.ceil(max(ly1, ly2)))
    width = rx2 - rx1
    height = ry2 - ry1
    if width <= 0 or height <= 0:
        return None
    local = np.zeros((height, width), dtype=np.uint8)
    shifted = polygon.copy()
    shifted[:, 0] -= rx1
    shifted[:, 1] -= ry1
    cv2.fillPoly(local, [np.rint(shifted).astype(np.int32)], 1)
    return min(max(float(np.count_nonzero(local)) / float(width * height), 0.0), 1.0)


def build_mask_litter_diagnostics(
    actors: Iterable[Mapping],
    litter_boxes: Iterable[Sequence[float]],
    frame_index: int,
):
    """Return compact deterministic summaries; never return mask vertices."""

    skipped = Counter()
    prepared_actors = []
    for actor in actors or []:
        class_name = str(actor.get("cls", "")).lower()
        if class_name not in _VEHICLE_CLASSES:
            skipped["unsupported_actor_class"] += 1
            continue
        if not bool(actor.get("observed", True)):
            skipped["not_observed"] += 1
            continue
        if str(actor.get("source", "")) not in _OBSERVED_SOURCES:
            skipped["unsupported_source"] += 1
            continue
        box = _finite_box(actor.get("box"))
        polygon = _finite_polygon(actor.get("mask_poly"))
        if box is None:
            skipped["invalid_actor_box"] += 1
            continue
        if polygon is None:
            skipped["invalid_actor_mask"] += 1
            continue
        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        area = width * height
        diagonal = math.hypot(width, height)
        if area <= 0.0 or diagonal <= 0.0:
            skipped["degenerate_actor_box"] += 1
            continue
        try:
            actor_key = [class_name, int(actor["track_id"])]
        except (KeyError, TypeError, ValueError):
            skipped["invalid_actor_key"] += 1
            continue
        prepared_actors.append({
            "actor_key": actor_key,
            "tracklet_uid": str(
                actor.get("tracklet_uid", "{}:{}".format(*actor_key))
            ),
            "box": box,
            "polygon": polygon,
            "bbox_area": area,
            "bbox_diagonal": diagonal,
        })

    prepared_litters = []
    for litter_index, litter in enumerate(litter_boxes or []):
        box = _finite_box(litter)
        if box is None or box[2] <= box[0] or box[3] <= box[1]:
            skipped["invalid_litter_box"] += 1
            continue
        confidence = None
        try:
            if len(litter) >= 5 and math.isfinite(float(litter[4])):
                confidence = float(litter[4])
        except (TypeError, ValueError):
            confidence = None
        prepared_litters.append((int(litter_index), box, confidence))

    records = []
    for actor in sorted(
        prepared_actors,
        key=lambda item: (item["actor_key"][0], item["actor_key"][1], item["tracklet_uid"]),
    ):
        polygon = actor["polygon"]
        fill_ratio = min(
            max(abs(float(cv2.contourArea(polygon))) / actor["bbox_area"], 0.0),
            1.0,
        )
        contour = polygon.reshape((-1, 1, 2))
        for litter_index, litter_box, confidence in prepared_litters:
            center = (
                0.5 * (litter_box[0] + litter_box[2]),
                0.5 * (litter_box[1] + litter_box[3]),
            )
            signed_distance = float(
                cv2.pointPolygonTest(contour, center, True)
            )
            record = {
                "schema": SCHEMA_NAME,
                "frame_index": int(frame_index),
                "actor_key": actor["actor_key"],
                "tracklet_uid": actor["tracklet_uid"],
                "raw_litter_index": litter_index,
                "litter_bbox_xyxy": list(litter_box),
                "litter_confidence": confidence,
                "mask_overlap_ratio": _mask_overlap_fraction(litter_box, polygon),
                "bbox_containment_ratio": _box_intersection_fraction(
                    litter_box, actor["box"]
                ),
                "mask_bbox_fill_ratio": fill_ratio,
                "signed_distance_actor_scale": math.tanh(
                    signed_distance / actor["bbox_diagonal"]
                ),
            }
            if all(
                value is None or not isinstance(value, float) or math.isfinite(value)
                for value in record.values()
            ):
                records.append(record)
            else:
                skipped["non_finite_result"] += 1

    records.sort(
        key=lambda item: (
            item["raw_litter_index"],
            item["actor_key"][0],
            item["actor_key"][1],
            item["tracklet_uid"],
        )
    )
    return {
        "schema": SCHEMA_NAME,
        "frame_index": int(frame_index),
        "records": records,
        "skipped": dict(sorted(skipped.items())),
    }


__all__ = ["SCHEMA_NAME", "build_mask_litter_diagnostics"]
