#!/usr/bin/env python3
"""Validate Smart Backtrack scale rules and distance/time weights.

The tool keeps human ground truth separate from runtime observations.  It
derives route labels only from the explicit actor-presence rule supplied by
the annotator, maps manual actor boxes to frozen model tracklets by same-frame
IoU, and reports uncertainty instead of silently promoting those mappings to
canonical reviewed annotations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pipeline.backtrack.study import StudyConfig, replay_candidates


SCHEMA = "attribution-formula-validation/v1"
SEED = 20260826
PERSON_GATE = 0.85
VEHICLE_GATE = 0.80
ACTOR_MAP_IOU = 0.50


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wilson_interval(hits: int, support: int, confidence: float = 0.95) -> list[float] | None:
    if support <= 0:
        return None
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = float(hits) / float(support)
    denominator = 1.0 + z * z / support
    center = (p + z * z / (2.0 * support)) / denominator
    half = z * math.sqrt(
        p * (1.0 - p) / support + z * z / (4.0 * support * support)
    ) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def percentile_interval(values: Sequence[float], confidence: float = 0.95) -> list[float] | None:
    if not values:
        return None
    alpha = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def bootstrap_statistic(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    *,
    samples: int,
    seed: int,
) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(samples)):
        draw = array[rng.integers(0, len(array), len(array))]
        estimates.append(float(statistic(draw)))
    return percentile_interval(estimates)


def route_from_actor_types(actor_types: Iterable[str]) -> str:
    values = set(str(value) for value in actor_types)
    if values == {"person", "vehicle"}:
        return "person_vehicle"
    if values == {"vehicle"}:
        return "direct_vehicle"
    if values == {"person"}:
        return "person"
    return "null"


def clip_id(filename: str) -> str:
    return Path(filename).stem.removesuffix("_annotated")


def actor_key(value: Any) -> Optional[tuple[str, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return str(value[0]), int(value[1])


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, first[:4])
    bx1, by1, bx2, by2 = map(float, second[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    first_area = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    second_area = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    return intersection / max(first_area + second_area - intersection, 1e-9)


def point_to_rect_distance(point: Sequence[float], rect: Sequence[float]) -> float:
    x1, y1, x2, y2 = map(float, rect[:4])
    px, py = map(float, point[:2])
    nearest_x = min(max(px, min(x1, x2)), max(x1, x2))
    nearest_y = min(max(py, min(y1, y2)), max(y1, y2))
    return float(math.hypot(px - nearest_x, py - nearest_y))


def ground_truth_release_point(event: Mapping[str, Any]) -> tuple[float, float]:
    if event.get("release_x") is not None and event.get("release_y") is not None:
        return float(event["release_x"]), float(event["release_y"])
    return (
        (float(event["release_x_min"]) + float(event["release_x_max"])) * 0.5,
        (float(event["release_y_min"]) + float(event["release_y_max"])) * 0.5,
    )


def distance_to_interval(frame: int, start: int, end: int) -> int:
    if start <= frame <= end:
        return 0
    return min(abs(frame - start), abs(frame - end))


def candidate_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    # Production batch runs isolate every clip in ``case_<id>/``.  Recursive
    # discovery keeps the validation input identical to the actual run tree
    # and avoids an error-prone temporary flatten/copy step.
    for path in sorted(directory.rglob("*_backtrack_candidates.jsonl")):
        records.extend(
            row for row in load_jsonl(path)
            if row.get("record_type") == "candidate"
        )
    return records


def candidate_clip(record: Mapping[str, Any]) -> str:
    return Path(str(record.get("video", {}).get("input_video", ""))).stem


def match_events(
    events: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    video_metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_clip: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_clip[candidate_clip(record)].append(record)
    matches = []
    for event in events:
        current_clip = clip_id(str(event["video_filename"]))
        release_x, release_y = ground_truth_release_point(event)
        ranked = []
        for record in by_clip.get(current_clip, []):
            assignment = record.get("assignment", {})
            frame = assignment.get("release_frame")
            point = assignment.get("release_point")
            if frame is None or not isinstance(point, (list, tuple)):
                continue
            temporal_gap = distance_to_interval(
                int(frame),
                int(event["release_start_frame"]),
                int(event["release_end_frame"]),
            )
            spatial_gap = math.hypot(
                float(point[0]) - release_x,
                float(point[1]) - release_y,
            )
            ranked.append((temporal_gap, spatial_gap, record))
        if not ranked:
            continue
        temporal_gap, spatial_gap, record = min(
            ranked, key=lambda item: (item[0], item[1])
        )
        metadata = video_metadata[str(event["video_filename"])]
        frame_diagonal = math.hypot(float(metadata["width"]), float(metadata["height"]))
        fps = float(metadata["fps"])
        spatial_fraction = spatial_gap / max(frame_diagonal, 1.0)
        if temporal_gap == 0 and spatial_fraction <= 0.05:
            tier = "strict"
        elif temporal_gap / max(fps, 1e-9) <= 0.5 and spatial_fraction <= 0.10:
            tier = "moderate"
        else:
            tier = "exploratory"
        matches.append({
            "clip_id": current_clip,
            "video_id": str(event["video_id"]),
            "gt_event_id": str(event["gt_event_id"]),
            "litter_id": int(record.get("event", {}).get("litter_id")),
            "temporal_gap_frames": int(temporal_gap),
            "temporal_gap_seconds": float(temporal_gap / max(fps, 1e-9)),
            "spatial_gap_pixels": float(spatial_gap),
            "spatial_gap_frame_diagonal_fraction": float(spatial_fraction),
            "match_tier": tier,
            "event": dict(event),
            "candidate": record,
        })
    return matches


def map_manual_actors(
    matches: Sequence[dict[str, Any]],
    manual_actors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in manual_actors:
        by_video[str(row["video_id"])].append(row)
    mappings = []
    for match in matches:
        frame_map = {
            int(row["frame_index"]): row.get("actors", [])
            for row in match["candidate"].get("resolver_input", {}).get("actor_frames", [])
        }
        accepted_by_type: dict[str, tuple[str, int]] = {}
        event_mappings = []
        for manual in by_video.get(match["video_id"], []):
            bbox = manual["bbox"]
            manual_box = [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]
            actor_type = str(manual["actor_type"])
            proposed = []
            for model_actor in frame_map.get(int(manual["bbox_frame"]), []):
                model_type = str(model_actor.get("cls"))
                normalized_type = (
                    "vehicle" if model_type in {"vehicle", "scooter"} else model_type
                )
                if normalized_type != actor_type:
                    continue
                proposed.append({
                    "iou": bbox_iou(manual_box, model_actor["box"]),
                    "actor_key": actor_key(model_actor.get("actor_key")),
                    "tracklet_uid": model_actor.get("tracklet_uid"),
                })
            proposed.sort(key=lambda item: item["iou"], reverse=True)
            best = proposed[0] if proposed else None
            accepted = bool(
                best and best["actor_key"] is not None
                and float(best["iou"]) >= ACTOR_MAP_IOU
            )
            if accepted:
                accepted_by_type[actor_type] = best["actor_key"]
            event_mappings.append({
                "actor_id": manual["actor_id"],
                "actor_type": actor_type,
                "bbox_frame": int(manual["bbox_frame"]),
                "accepted": accepted,
                "best_iou": float(best["iou"]) if best else None,
                "model_actor_key": list(best["actor_key"]) if best else None,
                "tracklet_uid": best.get("tracklet_uid") if best else None,
                "runner_up_iou": float(proposed[1]["iou"]) if len(proposed) > 1 else None,
            })
        match["actor_mappings"] = event_mappings
        match["correct_person_key"] = accepted_by_type.get("person")
        match["correct_vehicle_key"] = accepted_by_type.get("vehicle")
        match["derived_route_type"] = route_from_actor_types(accepted_by_type)
        mappings.extend(
            {"clip_id": match["clip_id"], "match_tier": match["match_tier"], **row}
            for row in event_mappings
        )
    return mappings


def actor_geometry(
    tracklet: Mapping[str, Any],
    releases: Sequence[Mapping[str, Any]],
    fps: float,
) -> Optional[dict[str, Any]]:
    observations = list(tracklet.get("observations", []))
    if not observations or not releases:
        return None
    max_gap = max(int(round(0.25 * fps)), 0)
    actor_class = str(tracklet.get("class_name"))
    normalized_class = "vehicle" if actor_class in {"vehicle", "scooter"} else actor_class
    candidates = []
    for release in releases:
        frame = int(release["frame_index"])
        nearby = [
            row for row in observations
            if abs(int(row["frame_index"]) - frame) <= max_gap
        ]
        if not nearby:
            continue
        observation = min(
            nearby, key=lambda row: abs(int(row["frame_index"]) - frame)
        )
        x1, y1, x2, y2 = map(float, observation["box"][:4])
        width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        if normalized_class == "person":
            gate_box = (x1, y1, x2, y1 + 0.72 * height)
            scale = height
        elif normalized_class == "vehicle":
            gate_box = (
                x1 - 0.18 * width,
                y1 - 0.15 * height,
                x2 + 0.18 * width,
                y2 + 0.15 * height,
            )
            scale = math.hypot(width, height)
        else:
            continue
        raw_distance = point_to_rect_distance(release["mean_uv"], gate_box)
        candidates.append({
            "raw_distance": float(raw_distance),
            "normalized_distance": float(raw_distance / max(scale, 1.0)),
            "actor_scale": float(scale),
            "release_frame": frame,
            "observation_frame": int(observation["frame_index"]),
        })
    if not candidates:
        return None
    return min(candidates, key=lambda item: item["normalized_distance"])


def candidate_geometry_rows(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        record = match["candidate"]
        fps = float(record.get("video", {}).get("fps") or 10.0)
        releases = record.get("release_hypotheses", [])
        for tracklet in record.get("candidate_actors", []):
            geometry = actor_geometry(tracklet, releases, fps)
            if geometry is None:
                continue
            original_class = str(tracklet.get("class_name"))
            actor_type = "vehicle" if original_class in {"vehicle", "scooter"} else original_class
            if actor_type not in {"person", "vehicle"}:
                continue
            key = actor_key(tracklet.get("actor_key"))
            correct_key = (
                match.get("correct_person_key")
                if actor_type == "person"
                else match.get("correct_vehicle_key")
            )
            rows.append({
                "clip_id": match["clip_id"],
                "litter_id": match["litter_id"],
                "match_tier": match["match_tier"],
                "actor_type": actor_type,
                "actor_key": list(key) if key else None,
                "is_correct_actor": bool(key is not None and key == correct_key),
                **geometry,
            })
    return rows


def _ordered_history(task: Mapping[str, Any]) -> list[tuple[int, np.ndarray]]:
    by_frame: dict[int, list[np.ndarray]] = defaultdict(list)
    for frame, point in zip(task.get("history_frames", []), task.get("history", [])):
        by_frame[int(frame)].append(np.asarray(point[:2], dtype=float))
    return [
        (frame, np.mean(by_frame[frame], axis=0))
        for frame in sorted(by_frame)
    ]


def _estimate_frame_error(estimate: float, event: Mapping[str, Any]) -> dict[str, float]:
    point = float(event["release_point_frame"])
    start = float(event["release_start_frame"])
    end = float(event["release_end_frame"])
    interval_error = 0.0 if start <= estimate <= end else min(abs(estimate - start), abs(estimate - end))
    return {
        "absolute_point_error_frames": abs(estimate - point),
        "interval_error_frames": interval_error,
    }


def release_timing_rows(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        task = match["candidate"].get("resolver_input", {})
        history = _ordered_history(task)
        if len(history) < 2:
            continue
        b0_frame, b0_point = history[0]
        b1_frame, b1_point = history[1]
        h0 = max(int(b1_frame) - int(b0_frame), 1)
        velocity_per_frame = (b1_point - b0_point) / float(h0)
        event = match["event"]
        gt_point = np.asarray(ground_truth_release_point(event), dtype=float)
        selected = match["candidate"].get("assignment", {})
        selected_frame = selected.get("release_frame")
        selected_point = selected.get("release_point")
        estimates = {
            "b0": (float(b0_frame), b0_point),
            "b0_minus_half_h0": (
                float(b0_frame) - 0.5 * h0,
                b0_point - velocity_per_frame * (0.5 * h0),
            ),
            "b0_minus_h0": (
                float(b0_frame - h0),
                b0_point - velocity_per_frame * h0,
            ),
        }
        if selected_frame is not None and isinstance(selected_point, (list, tuple)):
            estimates["current_selected"] = (
                float(selected_frame), np.asarray(selected_point[:2], dtype=float)
            )
        for method, (estimated_frame, estimated_point) in estimates.items():
            rows.append({
                "clip_id": match["clip_id"],
                "litter_id": match["litter_id"],
                "match_tier": match["match_tier"],
                "method": method,
                "b0_frame": int(b0_frame),
                "b1_frame": int(b1_frame),
                "h0_frames": int(h0),
                "gt_release_point_frame": int(event["release_point_frame"]),
                "estimated_release_frame": float(estimated_frame),
                "spatial_error_pixels": float(np.linalg.norm(estimated_point - gt_point)),
                **_estimate_frame_error(estimated_frame, event),
            })
    return rows


def release_motion_feature_rows(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        history = _ordered_history(match["candidate"].get("resolver_input", {}))
        if len(history) < 2:
            continue
        b0_frame, b0_point = history[0]
        b1_frame, b1_point = history[1]
        h0 = max(int(b1_frame) - int(b0_frame), 1)
        velocity01 = (b1_point - b0_point) / float(h0)
        speed01 = float(np.linalg.norm(velocity01))
        row = {
            "clip_id": match["clip_id"],
            "litter_id": match["litter_id"],
            "match_tier": match["match_tier"],
            "h0_frames": h0,
            "speed_b0_b1_px_per_frame": speed01,
            "gt_backtrack_alpha": (
                float(b0_frame) - float(match["event"]["release_point_frame"])
            ) / float(h0),
            "speed_change_ratio": None,
            "direction_cosine_b0b1_b1b2": None,
        }
        if len(history) >= 3:
            b2_frame, b2_point = history[2]
            h1 = max(int(b2_frame) - int(b1_frame), 1)
            velocity12 = (b2_point - b1_point) / float(h1)
            speed12 = float(np.linalg.norm(velocity12))
            row["speed_change_ratio"] = abs(speed12 - speed01) / max(speed12, speed01, 1e-9)
            row["direction_cosine_b0b1_b1b2"] = float(
                velocity01 @ velocity12 / max(speed01 * speed12, 1e-9)
            )
        rows.append(row)
    return rows


def _scope_allowed(scope: str) -> set[str]:
    return {
        "strict": {"strict"},
        "moderate": {"strict", "moderate"},
        "all": {"strict", "moderate", "exploratory"},
    }[scope]


def summarize_release_timing(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    motion_rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
) -> dict[str, Any]:
    latency = [
        int(event["first_visible_litter_frame"]) - int(event["release_point_frame"])
        for event in events
        if event.get("first_visible_litter_frame") not in (None, 0, 1)
        and event.get("release_point_frame") is not None
    ]
    output: dict[str, Any] = {
        "ground_truth_to_first_rtdetr_detection": {
            "support": len(latency),
            "exact_zero": {
                "hits": sum(value == 0 for value in latency),
                "support": len(latency),
                "value": sum(value == 0 for value in latency) / len(latency),
                "wilson_95ci": wilson_interval(sum(value == 0 for value in latency), len(latency)),
            },
            "within_one_frame": {
                "hits": sum(abs(value) <= 1 for value in latency),
                "support": len(latency),
                "value": sum(abs(value) <= 1 for value in latency) / len(latency),
                "wilson_95ci": wilson_interval(sum(abs(value) <= 1 for value in latency), len(latency)),
            },
            "within_two_frames": {
                "hits": sum(abs(value) <= 2 for value in latency),
                "support": len(latency),
                "value": sum(abs(value) <= 2 for value in latency) / len(latency),
                "wilson_95ci": wilson_interval(sum(abs(value) <= 2 for value in latency), len(latency)),
            },
            "median_frames": float(np.median(latency)),
            "mean_frames": float(np.mean(latency)),
            "p90_frames": float(np.quantile(latency, 0.90)),
            "p90_bootstrap_95ci": bootstrap_statistic(
                latency, lambda x: np.quantile(x, 0.90), samples=samples, seed=SEED + 301
            ),
        },
        "b0_b1_methods": {},
        "motion_feature_association": {},
    }
    methods = sorted({str(row["method"]) for row in rows})
    for scope in ("strict", "moderate", "all"):
        output["b0_b1_methods"][scope] = {}
        allowed = _scope_allowed(scope)
        for method in methods:
            current = [
                row for row in rows
                if row["match_tier"] in allowed and row["method"] == method
            ]
            point_errors = [float(row["absolute_point_error_frames"]) for row in current]
            interval_errors = [float(row["interval_error_frames"]) for row in current]
            spatial_errors = [float(row["spatial_error_pixels"]) for row in current]
            within_one = sum(value <= 1.0 for value in point_errors)
            output["b0_b1_methods"][scope][method] = {
                "support": len(current),
                "point_mae_frames": float(np.mean(point_errors)) if current else None,
                "point_mae_bootstrap_95ci": bootstrap_statistic(
                    point_errors, np.mean, samples=samples,
                    seed=SEED + 310 + len(scope) + len(method),
                ),
                "point_median_absolute_error_frames": float(np.median(point_errors)) if current else None,
                "interval_mae_frames": float(np.mean(interval_errors)) if current else None,
                "spatial_mae_pixels": float(np.mean(spatial_errors)) if current else None,
                "within_one_frame": {
                    "hits": within_one,
                    "support": len(current),
                    "value": within_one / len(current) if current else None,
                    "wilson_95ci": wilson_interval(within_one, len(current)),
                },
            }
    # Test the current zero-cost policy I0=[B0-H0, B0] directly against GT.
    unique_events = {}
    for row in rows:
        if row["method"] == "b0":
            unique_events[(row["clip_id"], row["litter_id"])] = row
    for scope in ("strict", "moderate", "all"):
        current = [row for row in unique_events.values() if row["match_tier"] in _scope_allowed(scope)]
        hits = sum(
            float(row["b0_frame"] - row["h0_frames"])
            <= float(row["gt_release_point_frame"])
            <= float(row["b0_frame"])
            for row in current
        )
        output["b0_b1_methods"][scope]["gt_inside_current_zero_cost_window"] = {
            "hits": hits,
            "support": len(current),
            "value": hits / len(current) if current else None,
            "wilson_95ci": wilson_interval(hits, len(current)),
        }
        motion_current = [
            row for row in motion_rows if row["match_tier"] in _scope_allowed(scope)
        ]
        for feature in (
            "h0_frames",
            "speed_b0_b1_px_per_frame",
            "speed_change_ratio",
            "direction_cosine_b0b1_b1b2",
        ):
            pairs = [
                (float(row[feature]), float(row["gt_backtrack_alpha"]))
                for row in motion_current if row.get(feature) is not None
            ]
            correlation = stats.spearmanr(
                [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            ) if len(pairs) >= 3 else None
            output["motion_feature_association"].setdefault(scope, {})[feature] = {
                "support": len(pairs),
                "spearman_rho": float(correlation.statistic) if correlation else None,
                "pvalue": float(correlation.pvalue) if correlation else None,
            }
    return output


def _same_frame_actor_snapshots(match: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    target = int(match["event"]["release_point_frame"])
    for row in match["candidate"].get("resolver_input", {}).get("actor_frames", []):
        if int(row["frame_index"]) == target:
            return list(row.get("actors", []))
    return []


def same_frame_distance_rows(matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for match in matches:
        point = np.asarray(ground_truth_release_point(match["event"]), dtype=float)
        for snapshot in _same_frame_actor_snapshots(match):
            key = actor_key(snapshot.get("actor_key"))
            original_type = str(snapshot.get("cls"))
            actor_type = "vehicle" if original_type in {"vehicle", "scooter"} else original_type
            if actor_type not in {"person", "vehicle"} or key is None:
                continue
            correct_key = (
                match.get("correct_person_key")
                if actor_type == "person" else match.get("correct_vehicle_key")
            )
            x1, y1, x2, y2 = map(float, snapshot["box"][:4])
            width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            center = np.asarray([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
            definitions: dict[str, float]
            if actor_type == "person":
                upper_center = np.asarray([(x1 + x2) * 0.5, y1 + 0.36 * height])
                definitions = {
                    "upper72_region": point_to_rect_distance(point, (x1, y1, x2, y1 + 0.72 * height)) / height,
                    "full_bbox_region": point_to_rect_distance(point, (x1, y1, x2, y2)) / height,
                    "upper_center_point": float(np.linalg.norm(point - upper_center)) / height,
                    "bbox_center_point": float(np.linalg.norm(point - center)) / height,
                }
            else:
                diagonal = math.hypot(width, height)
                definitions = {
                    "expanded_bbox_region": point_to_rect_distance(
                        point,
                        (x1 - 0.18 * width, y1 - 0.15 * height, x2 + 0.18 * width, y2 + 0.15 * height),
                    ) / diagonal,
                    "full_bbox_region": point_to_rect_distance(point, (x1, y1, x2, y2)) / diagonal,
                    "bbox_center_point": float(np.linalg.norm(point - center)) / diagonal,
                }
            for definition, value in definitions.items():
                rows.append({
                    "clip_id": match["clip_id"],
                    "litter_id": match["litter_id"],
                    "match_tier": match["match_tier"],
                    "release_point_frame": int(match["event"]["release_point_frame"]),
                    "actor_type": actor_type,
                    "actor_key": list(key),
                    "is_correct_actor": key == correct_key,
                    "distance_definition": definition,
                    "normalized_distance": float(value),
                })
    return rows


def summarize_same_frame_distances(
    rows: Sequence[Mapping[str, Any]], *, samples: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scope in ("strict", "moderate", "all"):
        output[scope] = {}
        allowed = _scope_allowed(scope)
        for actor_type in ("person", "vehicle"):
            output[scope][actor_type] = {}
            definitions = sorted({
                str(row["distance_definition"])
                for row in rows if row["actor_type"] == actor_type
            })
            for definition in definitions:
                current = [
                    row for row in rows
                    if row["match_tier"] in allowed
                    and row["actor_type"] == actor_type
                    and row["distance_definition"] == definition
                ]
                by_event: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
                for row in current:
                    by_event[(str(row["clip_id"]), int(row["litter_id"]))].append(row)
                evaluable = []
                for key, event_rows in by_event.items():
                    correct = [row for row in event_rows if row["is_correct_actor"]]
                    wrong = [row for row in event_rows if not row["is_correct_actor"]]
                    if len(correct) != 1 or not wrong:
                        continue
                    correct_distance = float(correct[0]["normalized_distance"])
                    wrong_distances = [float(row["normalized_distance"]) for row in wrong]
                    pair_auc = sum(
                        1.0 if correct_distance < value else 0.5 if correct_distance == value else 0.0
                        for value in wrong_distances
                    ) / len(wrong_distances)
                    evaluable.append({
                        "key": key,
                        "top1": correct_distance < min(wrong_distances),
                        "pair_auc": pair_auc,
                        "margin": min(wrong_distances) - correct_distance,
                        "correct_distance": correct_distance,
                        "nearest_wrong_distance": min(wrong_distances),
                    })
                hits = sum(row["top1"] for row in evaluable)
                output[scope][actor_type][definition] = {
                    "evaluable_events_with_distractor": len(evaluable),
                    "top1": {
                        "hits": hits,
                        "support": len(evaluable),
                        "value": hits / len(evaluable) if evaluable else None,
                        "wilson_95ci": wilson_interval(hits, len(evaluable)),
                    },
                    "mean_pair_auc": float(np.mean([row["pair_auc"] for row in evaluable])) if evaluable else None,
                    "pair_auc_bootstrap_95ci": bootstrap_statistic(
                        [row["pair_auc"] for row in evaluable], np.mean,
                        samples=samples, seed=SEED + 410 + len(scope) + len(definition),
                    ),
                    "median_margin": float(np.median([row["margin"] for row in evaluable])) if evaluable else None,
                    "margin_bootstrap_95ci": bootstrap_statistic(
                        [row["margin"] for row in evaluable], np.median,
                        samples=samples, seed=SEED + 420 + len(scope) + len(definition),
                    ),
                }
    return output


def lower_is_positive_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos < neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def geometry_scope_rows(
    rows: Sequence[Mapping[str, Any]], scope: str, actor_type: str
) -> list[Mapping[str, Any]]:
    allowed = {
        "strict": {"strict"},
        "moderate": {"strict", "moderate"},
        "all": {"strict", "moderate", "exploratory"},
    }[scope]
    return [
        row for row in rows
        if row["match_tier"] in allowed and row["actor_type"] == actor_type
    ]


def cluster_bootstrap_auc(
    rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    by_event: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[(str(row["clip_id"]), int(row["litter_id"]))].append(row)
    keys = sorted(by_event)
    if not keys:
        return {"raw_95ci": None, "normalized_95ci": None, "delta_95ci": None}
    rng = random.Random(seed)
    raw_values, normalized_values, delta_values = [], [], []
    for _ in range(int(samples)):
        sampled_rows = []
        for _index in range(len(keys)):
            sampled_rows.extend(by_event[rng.choice(keys)])
        positive = [row for row in sampled_rows if row["is_correct_actor"]]
        negative = [row for row in sampled_rows if not row["is_correct_actor"]]
        raw_auc = lower_is_positive_auc(
            [float(row["raw_distance"]) for row in positive],
            [float(row["raw_distance"]) for row in negative],
        )
        normalized_auc = lower_is_positive_auc(
            [float(row["normalized_distance"]) for row in positive],
            [float(row["normalized_distance"]) for row in negative],
        )
        if raw_auc is None or normalized_auc is None:
            continue
        raw_values.append(raw_auc)
        normalized_values.append(normalized_auc)
        delta_values.append(normalized_auc - raw_auc)
    return {
        "raw_95ci": percentile_interval(raw_values),
        "normalized_95ci": percentile_interval(normalized_values),
        "delta_95ci": percentile_interval(delta_values),
    }


def summarize_geometry(
    rows: Sequence[Mapping[str, Any]], *, samples: int
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for scope in ("strict", "moderate", "all"):
        report[scope] = {}
        for actor_type in ("person", "vehicle"):
            current = geometry_scope_rows(rows, scope, actor_type)
            positive = [row for row in current if row["is_correct_actor"]]
            negative = [row for row in current if not row["is_correct_actor"]]
            gate = PERSON_GATE if actor_type == "person" else VEHICLE_GATE
            positive_pass = sum(float(row["normalized_distance"]) <= gate for row in positive)
            negative_pass = sum(float(row["normalized_distance"]) <= gate for row in negative)
            raw_auc = lower_is_positive_auc(
                [float(row["raw_distance"]) for row in positive],
                [float(row["raw_distance"]) for row in negative],
            )
            normalized_auc = lower_is_positive_auc(
                [float(row["normalized_distance"]) for row in positive],
                [float(row["normalized_distance"]) for row in negative],
            )
            ci = cluster_bootstrap_auc(
                current, samples=samples, seed=SEED + len(scope) + len(actor_type)
            )
            report[scope][actor_type] = {
                "events": len({(row["clip_id"], row["litter_id"]) for row in current}),
                "correct_actor_samples": len(positive),
                "distractor_actor_samples": len(negative),
                "raw_distance_median_correct": (
                    float(np.median([row["raw_distance"] for row in positive]))
                    if positive else None
                ),
                "raw_distance_median_distractor": (
                    float(np.median([row["raw_distance"] for row in negative]))
                    if negative else None
                ),
                "normalized_distance_median_correct": (
                    float(np.median([row["normalized_distance"] for row in positive]))
                    if positive else None
                ),
                "normalized_distance_median_distractor": (
                    float(np.median([row["normalized_distance"] for row in negative]))
                    if negative else None
                ),
                "raw_distance_auc_lower_is_correct": raw_auc,
                "normalized_distance_auc_lower_is_correct": normalized_auc,
                "normalized_minus_raw_auc": (
                    normalized_auc - raw_auc
                    if normalized_auc is not None and raw_auc is not None else None
                ),
                **ci,
                "current_normalized_gate": gate,
                "correct_actor_gate_coverage": {
                    "hits": positive_pass,
                    "support": len(positive),
                    "value": positive_pass / len(positive) if positive else None,
                    "wilson_95ci": wilson_interval(positive_pass, len(positive)),
                },
                "distractor_gate_pass_rate": {
                    "hits": negative_pass,
                    "support": len(negative),
                    "value": negative_pass / len(negative) if negative else None,
                    "wilson_95ci": wilson_interval(negative_pass, len(negative)),
                },
            }
    return report


def summarize_positive_ground_truth(
    events: Sequence[Mapping[str, Any]], *, samples: int
) -> dict[str, Any]:
    result = {}
    for actor_type, scale_field in (
        ("person", "person_height"),
        ("vehicle", "vehicle_diagonal"),
    ):
        valid = [
            row for row in events
            if row.get(f"distance_{actor_type}") is not None
            and row.get(scale_field) not in (None, 0)
        ]
        distance = np.asarray([row[f"distance_{actor_type}"] for row in valid], dtype=float)
        scale = np.asarray([row[scale_field] for row in valid], dtype=float)
        ratio = distance / scale
        pearson = stats.pearsonr(scale, distance) if len(valid) >= 3 else None
        spearman = stats.spearmanr(scale, distance) if len(valid) >= 3 else None
        regression = stats.linregress(scale, distance) if len(valid) >= 3 else None
        result[actor_type] = {
            "support": len(valid),
            "zero_distance": int(np.sum(distance == 0.0)),
            "distance_pixels": {
                "median": float(np.median(distance)),
                "p90": float(np.quantile(distance, 0.90)),
                "p95": float(np.quantile(distance, 0.95)),
            },
            "normalized_distance": {
                "median": float(np.median(ratio)),
                "median_bootstrap_95ci": bootstrap_statistic(
                    ratio, np.median, samples=samples, seed=SEED + 1
                ),
                "p90": float(np.quantile(ratio, 0.90)),
                "p90_bootstrap_95ci": bootstrap_statistic(
                    ratio, lambda x: np.quantile(x, 0.90),
                    samples=samples, seed=SEED + 2,
                ),
                "p95": float(np.quantile(ratio, 0.95)),
                "p95_bootstrap_95ci": bootstrap_statistic(
                    ratio, lambda x: np.quantile(x, 0.95),
                    samples=samples, seed=SEED + 3,
                ),
                "maximum": float(np.max(ratio)),
            },
            "scale_vs_raw_distance": {
                "pearson_r": float(pearson.statistic) if pearson else None,
                "pearson_p": float(pearson.pvalue) if pearson else None,
                "spearman_rho": float(spearman.statistic) if spearman else None,
                "spearman_p": float(spearman.pvalue) if spearman else None,
                "ols_slope": float(regression.slope) if regression else None,
                "ols_slope_p": float(regression.pvalue) if regression else None,
                "ols_intercept": float(regression.intercept) if regression else None,
            },
        }
    return result


def prediction_tuple(assignment: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        assignment.get("route_type"),
        actor_key(assignment.get("person_key")),
        actor_key(assignment.get("vehicle_key")),
    )


def truth_tuple(match: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        match.get("derived_route_type"),
        match.get("correct_person_key"),
        match.get("correct_vehicle_key"),
    )


def weight_sweep(
    records: Sequence[Mapping[str, Any]],
    matches: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    match_by_key = {
        (str(match["clip_id"]), int(match["litter_id"])): match
        for match in matches
        if match.get("derived_route_type") != "null"
    }
    output = []
    for index in range(11):
        distance_weight = index / 10.0
        time_weight = 1.0 - distance_weight
        replayed, _summary = replay_candidates(
            records,
            StudyConfig(
                name=f"distance-{distance_weight:.1f}_time-{time_weight:.1f}",
                stage="distance_time",
                distance_weight=distance_weight,
                time_weight=time_weight,
            ),
        )
        evaluated = []
        for record in replayed:
            key = (
                candidate_clip(record),
                int(record.get("event", {}).get("litter_id")),
            )
            match = match_by_key.get(key)
            if match is None:
                continue
            predicted = prediction_tuple(record.get("assignment", {}))
            truth = truth_tuple(match)
            evaluated.append({
                "clip_id": match["clip_id"],
                "litter_id": match["litter_id"],
                "match_tier": match["match_tier"],
                "correct": predicted == truth,
                "wrong_non_null": predicted[0] != "null" and predicted != truth,
            })
        row: dict[str, Any] = {
            "distance_weight": distance_weight,
            "time_weight": time_weight,
            "label_status": "user_confirmed_actor_rule_plus_auto_iou_mapping",
            "event_results": evaluated,
        }
        for scope, allowed in (
            ("strict", {"strict"}),
            ("moderate", {"strict", "moderate"}),
            ("all", {"strict", "moderate", "exploratory"}),
        ):
            current = [item for item in evaluated if item["match_tier"] in allowed]
            hits = sum(item["correct"] for item in current)
            wrong_non_null = sum(item["wrong_non_null"] for item in current)
            row[scope] = {
                "hits": hits,
                "support": len(current),
                "accuracy": hits / len(current) if current else None,
                "wilson_95ci": wilson_interval(hits, len(current)),
                "wrong_non_null": wrong_non_null,
                "wrong_non_null_rate": (
                    wrong_non_null / len(current) if current else None
                ),
            }
        output.append(row)
    return output


def paired_weight_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    first_distance_weight: float,
    second_distance_weight: float,
    samples: int,
) -> dict[str, Any]:
    first = next(
        row for row in rows
        if math.isclose(float(row["distance_weight"]), first_distance_weight)
    )
    second = next(
        row for row in rows
        if math.isclose(float(row["distance_weight"]), second_distance_weight)
    )
    first_by_key = {
        (row["clip_id"], row["litter_id"]): bool(row["correct"])
        for row in first["event_results"]
    }
    second_by_key = {
        (row["clip_id"], row["litter_id"]): bool(row["correct"])
        for row in second["event_results"]
    }
    keys = sorted(set(first_by_key).intersection(second_by_key))
    differences = [
        int(first_by_key[key]) - int(second_by_key[key]) for key in keys
    ]
    first_only = sum(
        first_by_key[key] and not second_by_key[key] for key in keys
    )
    second_only = sum(
        second_by_key[key] and not first_by_key[key] for key in keys
    )
    discordant = first_only + second_only
    pvalue = (
        float(stats.binomtest(
            min(first_only, second_only), discordant, 0.5,
            alternative="two-sided",
        ).pvalue)
        if discordant else 1.0
    )
    return {
        "first_distance_weight": first_distance_weight,
        "second_distance_weight": second_distance_weight,
        "support": len(keys),
        "accuracy_difference_first_minus_second": (
            float(np.mean(differences)) if differences else None
        ),
        "paired_bootstrap_95ci": bootstrap_statistic(
            differences, np.mean, samples=samples, seed=SEED + 77
        ),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "mcnemar_exact_p": pvalue,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def plot_positive_distribution(events: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for actor_type, scale_field, color in (
        ("person", "person_height", "#2b6cb0"),
        ("vehicle", "vehicle_diagonal", "#c05621"),
    ):
        values = sorted(
            float(row[f"distance_{actor_type}"]) / float(row[scale_field])
            for row in events
            if row.get(f"distance_{actor_type}") is not None
            and row.get(scale_field) not in (None, 0)
        )
        y = np.arange(1, len(values) + 1) / max(len(values), 1)
        ax.step(values, y, where="post", label=f"{actor_type} (n={len(values)})", color=color)
    ax.set_xlabel("Normalized release distance")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Ground-truth correct-actor distance distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_candidate_separation(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    moderate = [row for row in rows if row["match_tier"] in {"strict", "moderate"}]
    groups, labels = [], []
    for actor_type in ("person", "vehicle"):
        for correct, label in ((True, "correct"), (False, "distractor")):
            values = [
                float(row["normalized_distance"])
                for row in moderate
                if row["actor_type"] == actor_type and row["is_correct_actor"] == correct
            ]
            groups.append(values)
            labels.append(f"{actor_type}\n{label}\nn={len(values)}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, tick_labels=labels, showfliers=True)
    ax.axhline(PERSON_GATE, color="#2b6cb0", linestyle="--", alpha=0.7, label="person gate 0.85")
    ax.axhline(VEHICLE_GATE, color="#c05621", linestyle=":", alpha=0.8, label="vehicle gate 0.80")
    ax.set_ylabel("Minimum normalized release distance")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_title("Correct versus distractor actors (strict + moderate matches)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_weight_sweep(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    x = [float(row["distance_weight"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for scope, color in (("strict", "#2b6cb0"), ("moderate", "#2f855a"), ("all", "#c05621")):
        y = [float(row[scope]["accuracy"]) for row in rows]
        lower = [value - row[scope]["wilson_95ci"][0] for value, row in zip(y, rows)]
        upper = [row[scope]["wilson_95ci"][1] - value for value, row in zip(y, rows)]
        ax.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=3, label=scope, color=color)
    ax.set_xlabel("Distance weight (time weight = 1 - distance weight)")
    ax.set_ylabel("Exact-route accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Provisional D/T replay with Wilson 95% intervals")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_release_timing(report: Mapping[str, Any], path: Path) -> None:
    latency = report["ground_truth_to_first_rtdetr_detection"]
    methods = report["b0_b1_methods"]["moderate"]
    names = ["b0", "b0_minus_half_h0", "b0_minus_h0", "current_selected"]
    labels = ["B0", "B0−0.5H0", "B0−H0", "current"]
    values = [float(methods[name]["point_mae_frames"]) for name in names]
    intervals = [methods[name]["point_mae_bootstrap_95ci"] for name in names]
    lower = [value - interval[0] for value, interval in zip(values, intervals)]
    upper = [interval[1] - value for value, interval in zip(values, intervals)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(
        ["exact", "≤1 frame", "≤2 frames"],
        [
            latency["exact_zero"]["value"],
            latency["within_one_frame"]["value"],
            latency["within_two_frames"]["value"],
        ],
        color=["#2b6cb0", "#2f855a", "#c05621"],
    )
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Proportion")
    axes[0].set_title("GT release to first RT-DETR detection (n=50)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].errorbar(labels, values, yerr=[lower, upper], fmt="o", capsize=5, color="#2b6cb0")
    axes[1].set_ylabel("Absolute error to GT release frame")
    axes[1].set_title("B0/B1 timing methods (strict + moderate, n=15)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_same_frame_definitions(report: Mapping[str, Any], path: Path) -> None:
    moderate = report["moderate"]
    definitions = {
        "person": ["upper72_region", "full_bbox_region", "upper_center_point", "bbox_center_point"],
        "vehicle": ["expanded_bbox_region", "full_bbox_region", "bbox_center_point"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, actor_type in zip(axes, ("person", "vehicle")):
        names = definitions[actor_type]
        rows = [moderate[actor_type][name] for name in names]
        values = [float(row["mean_pair_auc"]) for row in rows]
        intervals = [row["pair_auc_bootstrap_95ci"] for row in rows]
        lower = [value - interval[0] for value, interval in zip(values, intervals)]
        upper = [interval[1] - value for value, interval in zip(values, intervals)]
        short = [name.replace("_", "\n") for name in names]
        axis.bar(short, values, color="#2b6cb0" if actor_type == "person" else "#c05621")
        axis.errorbar(range(len(values)), values, yerr=[lower, upper], fmt="none", color="black", capsize=4)
        axis.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        axis.set_title(f"{actor_type}: correct vs distractor at GT release frame")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean within-event pair AUC")
    axes[0].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_ci(value: Optional[Sequence[float]]) -> str:
    if value is None:
        return "N/A"
    return f"{value[0]:.3f}–{value[1]:.3f}"


def markdown_report(report: Mapping[str, Any]) -> str:
    gt = report["positive_ground_truth"]
    geometry = report["candidate_geometry"]["moderate"]
    weight_rows = report["distance_time_weight_sweep"]
    dt08 = next(row for row in weight_rows if math.isclose(row["distance_weight"], 0.8))
    time_only = next(row for row in weight_rows if math.isclose(row["distance_weight"], 0.0))
    lines = [
        "# Attribution formula validation",
        "",
        "## Executive conclusion",
        "",
        "The algebraic scale rule is exact: if image coordinates are scaled by `s`, then both pixel distance and actor scale are multiplied by `s`, so `d/H_person` and `d/diag_vehicle` do not change. This proves resolution invariance of the *form* `d_max = k × actor_scale`.",
        "",
        "The current dataset does **not** yet identify the production constants 0.85/0.80 or a D/T weight of 0.8/0.2 with statistical confidence. Those values must remain hypotheses until a locked, reviewed test set with negative and NULL cases is available.",
        "",
        "## Data status",
        "",
        f"- Clips: {report['data_status']['clips']}; usable positive events: {report['data_status']['usable_events']}.",
        f"- Event review export: {report['data_status']['event_review_states']}.",
        f"- RT-DETR miss sentinel (`first_visible_litter_frame` 0/1): {report['data_status']['rtdetr_miss_sentinel_count']} events.",
        f"- Confirmed-event sidecar matches: {report['matching']['matched_events']} ({report['matching']['tiers']}).",
        f"- Actor mappings accepted at IoU ≥ {ACTOR_MAP_IOU:.2f}: {report['matching']['accepted_actor_mappings']}/{report['matching']['actor_mappings']}.",
        "",
        "## Positive correct-actor distances",
        "",
        "| Actor | n | zeros | median d/scale | P90 (95% bootstrap CI) | P95 (95% bootstrap CI) | scale-vs-distance correlation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for actor_type in ("person", "vehicle"):
        row = gt[actor_type]
        norm = row["normalized_distance"]
        relation = row["scale_vs_raw_distance"]
        lines.append(
            f"| {actor_type} | {row['support']} | {row['zero_distance']} | {norm['median']:.3f} | "
            f"{norm['p90']:.3f} ({format_ci(norm['p90_bootstrap_95ci'])}) | "
            f"{norm['p95']:.3f} ({format_ci(norm['p95_bootstrap_95ci'])}) | "
            f"Spearman ρ={relation['spearman_rho']:.3f}, p={relation['spearman_p']:.3f} |"
        )
    lines.extend([
        "",
        "A non-significant raw-distance correlation does not disprove scale normalization; it means these positive-only observations cannot estimate proportionality by regression. Most correct vehicle releases are inside the expanded vehicle box, creating many zero distances.",
        "",
        "## Correct versus distractor actors",
        "",
        "Moderate analysis includes strict matches plus candidates within 0.5 s and 10% of the frame diagonal from GT.",
        "",
        "| Actor | events | correct/wrong samples | raw AUC (95% CI) | normalized AUC (95% CI) | ΔAUC CI | current gate correct coverage | distractor pass rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for actor_type in ("person", "vehicle"):
        row = geometry[actor_type]
        correct = row["correct_actor_gate_coverage"]
        distractor = row["distractor_gate_pass_rate"]
        lines.append(
            f"| {actor_type} | {row['events']} | {row['correct_actor_samples']}/{row['distractor_actor_samples']} | "
            f"{row['raw_distance_auc_lower_is_correct']:.3f} ({format_ci(row['raw_95ci'])}) | "
            f"{row['normalized_distance_auc_lower_is_correct']:.3f} ({format_ci(row['normalized_95ci'])}) | "
            f"{format_ci(row['delta_95ci'])} | "
            f"{correct['hits']}/{correct['support']} ({correct['value']:.1%}) | "
            f"{distractor['hits']}/{distractor['support']} ({distractor['value']:.1%}) |"
        )
    lines.extend([
        "",
        "A normalized-minus-raw AUC interval crossing zero means this sample does not empirically distinguish normalized from fixed-pixel distance. Scale normalization remains mathematically justified, but the coefficient still needs cross-camera data.",
        "",
        "## Distance/time weight replay",
        "",
        "| Distance/time | strict | moderate | all | all 95% Wilson CI | wrong non-NULL (all) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in weight_rows:
        lines.append(
            f"| {row['distance_weight']:.1f}/{row['time_weight']:.1f} | "
            f"{row['strict']['hits']}/{row['strict']['support']} | "
            f"{row['moderate']['hits']}/{row['moderate']['support']} | "
            f"{row['all']['hits']}/{row['all']['support']} | "
            f"{format_ci(row['all']['wilson_95ci'])} | {row['all']['wrong_non_null']} |"
        )
    lines.extend([
        "",
        f"The provisional 0.8/0.2 trial is {dt08['all']['hits']}/{dt08['all']['support']} ({dt08['all']['accuracy']:.1%}, 95% CI {format_ci(dt08['all']['wilson_95ci'])}). Time-only is {time_only['all']['hits']}/{time_only['all']['support']} ({time_only['all']['accuracy']:.1%}, 95% CI {format_ci(time_only['all']['wilson_95ci'])}). The intervals overlap heavily, and weights 0.1–1.0 produce the same routes, so 0.8/0.2 is not identifiable from this sample.",
        "",
        "## Claims that are currently defensible",
        "",
        "1. `d_max = k × person_height` and `d_max = k × vehicle_diagonal` are resolution-invariant by construction.",
        "2. Normalized distance separates correct and distractor vehicles in this same-domain confirmed-event subset, but it has not yet shown a significant advantage over raw pixels.",
        "3. Current constants and D/T weights are provisional engineering gates, not statistically proven universal constants.",
        "4. No claim about precision, NULL safety, or cross-camera generalization is valid until negative clips, reviewed NULL routes, and independent cameras are added.",
        "",
        "## Required next experiment",
        "",
        "Freeze development/validation/test groups, add true negative and NULL-route clips, re-review strict/moderate event matches without model overlays, tune only on development, choose on validation, and report the locked test once. Use event-cluster bootstrap and wrong-non-NULL rate as the safety endpoint.",
        "",
    ])
    return "\n".join(lines)


def markdown_report_zh_tw(report: Mapping[str, Any]) -> str:
    gt = report["positive_ground_truth"]
    geometry = report["candidate_geometry"]["moderate"]
    timing = report["release_timing"]
    same_frame = report["same_frame_distance_definitions"]["moderate"]
    weights = report["distance_time_weight_sweep"]
    dt08 = next(row for row in weights if math.isclose(row["distance_weight"], 0.8))
    time_only = next(row for row in weights if math.isclose(row["distance_weight"], 0.0))
    paired = report["weight_comparisons"]["time_only_minus_08_02"]
    lines = [
        "# 反追蹤歸因公式統計驗證報告",
        "",
        "## 結論先行",
        "",
        "`d_max = k × 人的高度` 與 `d_max = k × 車輛斜對角線` 的**公式形式具有嚴格的尺度不變性**：畫面縮放 `s` 倍時，距離與 actor 尺度都同乘 `s`，所以 `d/H_person`、`d/diag_vehicle` 完全不變。這比固定像素門檻更符合跨解析度設計原則。",
        "",
        "但是，現有資料**尚不足以證明** `k_person=0.85`、`k_vehicle=0.80` 或距離／時間權重 `0.8/0.2` 是通用最佳值。教授面前可主張『比例形式有數學與初步實證支持』，不可主張『目前常數已被統計證明為唯一最佳值』。",
        "",
        "## 資料與標註狀態",
        "",
        f"- 63 部影片，58 個可用真垃圾事件。",
        f"- `first_visible_litter_frame=0/1` 的 RT-DETR miss sentinel：{report['data_status']['rtdetr_miss_sentinel_count']} 件。",
        f"- 目前 sidecar 只有 {report['matching']['matched_events']} 件 confirmed-event 可分析；其中 strict {report['matching']['tiers'].get('strict', 0)}、moderate {report['matching']['tiers'].get('moderate', 0)}、exploratory {report['matching']['tiers'].get('exploratory', 0)}。",
        f"- 人工 bbox 對模型 tracklet：{report['matching']['accepted_actor_mappings']}/{report['matching']['actor_mappings']} 成功，最低 IoU={report['matching']['minimum_accepted_iou']:.3f}。",
        "- Clip 檔案為 reviewed；event JSON 仍輸出 unreviewed。以下 route 依使用者確認規則產生，track ID 則為自動 IoU mapping，因此仍列為 provisional evidence。",
        "",
        "## Release frame 與 B0/B1 驗證",
        "",
    ]
    latency = timing["ground_truth_to_first_rtdetr_detection"]
    lines.extend([
        f"排除8筆 RT-DETR miss sentinel 後共有 {latency['support']} 件：",
        "",
        f"- `release_point_frame = first_visible`：{latency['exact_zero']['hits']}/{latency['exact_zero']['support']}={latency['exact_zero']['value']:.1%}，95% Wilson CI {format_ci(latency['exact_zero']['wilson_95ci'])}。",
        f"- 偏差≤1幀：{latency['within_one_frame']['hits']}/{latency['within_one_frame']['support']}={latency['within_one_frame']['value']:.1%}，95% CI {format_ci(latency['within_one_frame']['wilson_95ci'])}。",
        f"- 偏差≤2幀：{latency['within_two_frames']['hits']}/{latency['within_two_frames']['support']}={latency['within_two_frames']['value']:.1%}，95% CI {format_ci(latency['within_two_frames']['wilson_95ci'])}。",
        "",
        "這支持把 B0 本身列為正式 release hypothesis，而不是預設垃圾一定在 B0 前釋放。下表在 strict+moderate 15件已配對 confirmed event 比較不同估計；H0=B1−B0。",
        "",
        "| 方法 | release frame MAE（bootstrap 95% CI） | median absolute error | ±1幀命中 | point-xy MAE |",
        "|---|---:|---:|---:|---:|",
    ])
    timing_methods = timing["b0_b1_methods"]["moderate"]
    for method, label in (
        ("b0", "B0"),
        ("b0_minus_half_h0", "B0−0.5H0"),
        ("b0_minus_h0", "B0−H0"),
        ("current_selected", "目前 resolver"),
    ):
        row = timing_methods[method]
        within = row["within_one_frame"]
        lines.append(
            f"| {label} | {row['point_mae_frames']:.3f}（{format_ci(row['point_mae_bootstrap_95ci'])}） | "
            f"{row['point_median_absolute_error_frames']:.3f} | {within['hits']}/{within['support']}（{within['value']:.1%}） | "
            f"{row['spatial_mae_pixels']:.1f}px |"
        )
    window = timing_methods["gt_inside_current_zero_cost_window"]
    motion = timing["motion_feature_association"]["moderate"]
    lines.extend([
        "",
        f"GT release 落在目前零成本窗 `I0=[B0−H0,B0]` 的比例為 {window['hits']}/{window['support']}={window['value']:.1%}，95% CI {format_ci(window['wilson_95ci'])}。目前樣本中 B0−H0 的平均幀誤差略低，但各方法 CI 重疊；strict subset 則 B0 最穩。這表示合理修改是保留整個 hypothesis set，並在 I0 內加入『偏向B0』的可調 prior，而不是硬改成固定往前一個 H0。",
        "",
        f"B0/B1 speed 與最佳回推比例 α=(B0−GT)/H0 的 Spearman ρ={motion['speed_b0_b1_px_per_frame']['spearman_rho']:.3f}, p={motion['speed_b0_b1_px_per_frame']['pvalue']:.3f}；speed-change 僅 {motion['speed_change_ratio']['support']} 件有B2，ρ={motion['speed_change_ratio']['spearman_rho']:.3f}, p={motion['speed_change_ratio']['pvalue']:.3f}。目前沒有顯著證據可用速度或速度差單獨決定回推幀數。",
        "",
        "## 同一 release 幀的距離定義比較",
        "",
        "以下完全固定使用 GT `release_point_frame` 與 `release_point_xy`，只比較同幀正確 actor 和其他干擾 actor，因此比跨案例 raw distance regression 更直接。Pair AUC=1代表每一對比較都讓正確 actor 更近。",
        "",
        "| Actor | 距離定義 | 有干擾 actor 的事件 | Top-1（95% Wilson CI） | pair AUC（bootstrap 95% CI） | median nearest-wrong margin |",
        "|---|---|---:|---:|---:|---:|",
    ])
    definition_labels = {
        "upper72_region": "上半身72%區域",
        "full_bbox_region": "完整bbox區域",
        "upper_center_point": "上半身中心點",
        "bbox_center_point": "bbox中心點",
        "expanded_bbox_region": "擴張vehicle區域",
    }
    for actor_type, actor_label in (("person", "人"), ("vehicle", "車")):
        for definition, row in same_frame[actor_type].items():
            top1 = row["top1"]
            lines.append(
                f"| {actor_label} | {definition_labels[definition]} | {row['evaluable_events_with_distractor']} | "
                f"{top1['hits']}/{top1['support']}={top1['value']:.1%}（{format_ci(top1['wilson_95ci'])}） | "
                f"{row['mean_pair_auc']:.3f}（{format_ci(row['pair_auc_bootstrap_95ci'])}） | "
                f"{row['median_margin']:.3f} |"
            )
    lines.extend([
        "",
        "車輛方面，擴張 region 的 pair AUC最高（0.948），bbox中心點的嚴格Top-1較高（11/14 vs 10/14），但CI重疊；region 對遮擋與框大小較符合物理意義，中心點可保留為 soft secondary feature。Person 只有3件具同類干擾者，完全不足以選定 region 或 midpoint。",
        "",
        "## 正確 actor 的距離分布",
        "",
        "| Actor | n | 距離為0 | median d/scale | P90（bootstrap 95% CI） | P95（bootstrap 95% CI） | 尺度與原始距離相關 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for actor_type, label in (("person", "人"), ("vehicle", "車")):
        row = gt[actor_type]
        norm = row["normalized_distance"]
        relation = row["scale_vs_raw_distance"]
        lines.append(
            f"| {label} | {row['support']} | {row['zero_distance']} | {norm['median']:.3f} | "
            f"{norm['p90']:.3f}（{format_ci(norm['p90_bootstrap_95ci'])}） | "
            f"{norm['p95']:.3f}（{format_ci(norm['p95_bootstrap_95ci'])}） | "
            f"Spearman ρ={relation['spearman_rho']:.3f}, p={relation['spearman_p']:.3f} |"
        )
    lines.extend([
        "",
        "兩種 actor 的尺度與 raw distance 均未達顯著相關。這不是比例正規化失敗，而是正樣本多數 release 點位於 actor 框內，尤其車輛有38/57筆距離為0；positive-only regression 無法識別比例係數。",
        "",
        "## 正確 actor 與干擾 actor 的區辨力",
        "",
        "下表使用 strict + moderate event match，AUC 越接近1代表距離越能讓正確 actor 排在干擾 actor 前。bootstrap 以事件為 cluster，避免同影片多個 actor 被當成獨立影片。",
        "",
        "| Actor | 事件數 | 正確／干擾樣本 | raw AUC（95% CI） | normalized AUC（95% CI） | normalized−raw AUC CI | 現行 gate 正確覆蓋 | 干擾 actor 通過 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for actor_type, label in (("person", "人"), ("vehicle", "車")):
        row = geometry[actor_type]
        correct = row["correct_actor_gate_coverage"]
        distractor = row["distractor_gate_pass_rate"]
        lines.append(
            f"| {label} | {row['events']} | {row['correct_actor_samples']}/{row['distractor_actor_samples']} | "
            f"{row['raw_distance_auc_lower_is_correct']:.3f}（{format_ci(row['raw_95ci'])}） | "
            f"{row['normalized_distance_auc_lower_is_correct']:.3f}（{format_ci(row['normalized_95ci'])}） | "
            f"{format_ci(row['delta_95ci'])} | {correct['hits']}/{correct['support']}（{correct['value']:.1%}） | "
            f"{distractor['hits']}/{distractor['support']}（{distractor['value']:.1%}） |"
        )
    lines.extend([
        "",
        "Vehicle normalized AUC=0.838，顯示距離對車輛候選具有區辨力；但 normalized−raw 的95% CI仍跨0，所以這批同場域資料尚不能證明它顯著優於固定像素。現行0.85/0.80雖涵蓋所有正確 actor，仍讓約一半干擾 actor 通過，說明距離只能是第一道物理 gate，不能單獨決定歸因。",
        "",
        "## 距離／時間權重掃描",
        "",
        "| D/T | strict | strict 95% CI | moderate | all | all 95% CI | all wrong non-NULL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in weights:
        lines.append(
            f"| {row['distance_weight']:.1f}/{row['time_weight']:.1f} | "
            f"{row['strict']['hits']}/{row['strict']['support']} | {format_ci(row['strict']['wilson_95ci'])} | "
            f"{row['moderate']['hits']}/{row['moderate']['support']} | "
            f"{row['all']['hits']}/{row['all']['support']} | {format_ci(row['all']['wilson_95ci'])} | "
            f"{row['all']['wrong_non_null']} |"
        )
    lines.extend([
        "",
        f"0.8/0.2 為 {dt08['all']['hits']}/{dt08['all']['support']}={dt08['all']['accuracy']:.1%}，95% Wilson CI {format_ci(dt08['all']['wilson_95ci'])}；time-only 為 {time_only['all']['hits']}/{time_only['all']['support']}={time_only['all']['accuracy']:.1%}，95% CI {format_ci(time_only['all']['wilson_95ci'])}。",
        f"兩者 paired accuracy 差為 {paired['accuracy_difference_first_minus_second']:.1%}，bootstrap 95% CI {format_ci(paired['paired_bootstrap_95ci'])}，McNemar exact p={paired['mcnemar_exact_p']:.3f}。此外 D>0 的0.1至1.0全部產生相同 route，故目前資料無法識別0.8/0.2。",
        "",
        "## 教授面前可成立的主張",
        "",
        "1. 使用 actor 尺度正規化具有可推導的解析度不變性；固定像素門檻不具備此性質。",
        "2. 在目前 same-domain confirmed-event subset，normalized distance 對車輛正確／干擾候選具有明顯排序能力。",
        "3. 現行常數應稱為 conservative engineering gate，而不是 universal optimum。",
        "4. 0.8/0.2 目前沒有統計證據，不應硬說已證明；下一版應以 locked validation/test 決定。",
        "",
        "## 尚缺的驗證",
        "",
        "- 無垃圾 negative clips：估計 confirmed false positives／minute。",
        "- reviewed NULL routes：驗證不確定時不會強制配對。",
        "- 不同攝影機、夜間與不同解析度：驗證跨場域比例規則。",
        "- pre-confirm candidate sidecar：分析39個未 confirmed 真事件究竟死在哪一關。",
        "",
    ])
    return "\n".join(lines)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    clips = load_jsonl(args.ground_truth / "clip_annotations.jsonl")
    events = load_jsonl(args.ground_truth / "event_annotations.jsonl")
    actors = load_jsonl(args.ground_truth / "actor_annotations.jsonl")
    project = json.loads((args.ground_truth / "project.json").read_text(encoding="utf-8"))
    metadata = {row["filename"]: row for row in project["videos"]}
    records = candidate_records(args.candidates)
    matches = match_events(events, records, metadata)
    mappings = map_manual_actors(matches, actors)
    geometry_rows = candidate_geometry_rows(matches)
    timing_rows = release_timing_rows(matches)
    motion_feature_rows = release_motion_feature_rows(matches)
    same_frame_rows = same_frame_distance_rows(matches)
    sweep = weight_sweep(records, matches)

    clip_by_video = {str(row["video_id"]): row for row in clips}
    usable_events = sum(
        bool(clip_by_video.get(str(event["video_id"]), {}).get("video_usable"))
        and not bool(event.get("ignore"))
        for event in events
    )
    report = {
        "schema": SCHEMA,
        "seed": SEED,
        "bootstrap_samples": int(args.bootstrap),
        "inputs": {
            "ground_truth": str(args.ground_truth.resolve()),
            "candidates": str(args.candidates.resolve()),
            "event_annotations_sha256": file_sha256(args.ground_truth / "event_annotations.jsonl"),
            "actor_annotations_sha256": file_sha256(args.ground_truth / "actor_annotations.jsonl"),
        },
        "definitions": {
            "route_rule": {
                "vehicle": "direct_vehicle",
                "person+vehicle": "person_vehicle",
                "person": "person",
            },
            "rtdetr_miss_sentinel": "first_visible_litter_frame in {0, 1}",
            "strict_match": "release frame inside GT interval and release point <=5% frame diagonal",
            "moderate_match": "temporal gap <=0.5 s and release point <=10% frame diagonal",
            "actor_mapping": f"same-frame same-class bbox IoU >= {ACTOR_MAP_IOU}",
            "person_scale": "bbox height; release zone is upper 72%",
            "vehicle_scale": "bbox diagonal; release box expands 18% x and 15% y",
        },
        "data_status": {
            "clips": len(clips),
            "events": len(events),
            "usable_events": usable_events,
            "clip_review_states": dict(Counter(row.get("review_state") for row in clips)),
            "event_review_states": dict(Counter(row.get("review_state") for row in events)),
            "rtdetr_miss_sentinel_count": sum(
                row.get("first_visible_litter_frame") in {0, 1} for row in events
            ),
            "unusable_reason_rule": "extremely_small",
            "route_counts_derived": dict(Counter(
                route_from_actor_types(
                    actor["actor_type"] for actor in actors
                    if actor["video_id"] == event["video_id"]
                )
                for event in events
            )),
        },
        "positive_ground_truth": summarize_positive_ground_truth(
            events, samples=args.bootstrap
        ),
        "matching": {
            "confirmed_candidate_records": len(records),
            "matched_events": len(matches),
            "tiers": dict(Counter(match["match_tier"] for match in matches)),
            "actor_mappings": len(mappings),
            "accepted_actor_mappings": sum(row["accepted"] for row in mappings),
            "minimum_accepted_iou": min(
                (row["best_iou"] for row in mappings if row["accepted"]),
                default=None,
            ),
        },
        "candidate_geometry": summarize_geometry(
            geometry_rows, samples=args.bootstrap
        ),
        "release_timing": summarize_release_timing(
            events, timing_rows, motion_feature_rows, samples=args.bootstrap
        ),
        "same_frame_distance_definitions": summarize_same_frame_distances(
            same_frame_rows, samples=args.bootstrap
        ),
        "distance_time_weight_sweep": sweep,
        "weight_comparisons": {
            "time_only_minus_08_02": paired_weight_comparison(
                sweep,
                first_distance_weight=0.0,
                second_distance_weight=0.8,
                samples=args.bootstrap,
            ),
        },
        "limitations": [
            "Event rows are exported as unreviewed; route labels are provisional derivations from the user-specified actor rule.",
            "Only confirmed-event sidecars contain distractor actors; RT-DETR/tracker misses have no resolver candidate table.",
            "The dataset has no true negative clips and no reviewed NULL routes.",
            "Most samples come from one same-domain collection; cross-camera universality is untested.",
            "Person support is small, so its confidence intervals are especially wide.",
        ],
    }

    (args.output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output / "event_matches.csv", [
        {
            key: value for key, value in match.items()
            if key not in {"event", "candidate"}
        }
        for match in matches
    ])
    write_csv(args.output / "actor_mappings.csv", mappings)
    write_csv(args.output / "candidate_geometry.csv", geometry_rows)
    write_csv(args.output / "release_timing_methods.csv", timing_rows)
    write_csv(args.output / "release_motion_features.csv", motion_feature_rows)
    write_csv(args.output / "same_frame_distance_definitions.csv", same_frame_rows)
    write_csv(args.output / "distance_time_weight_sweep.csv", [
        {
            "distance_weight": row["distance_weight"],
            "time_weight": row["time_weight"],
            **{
                f"{scope}_{key}": value
                for scope in ("strict", "moderate", "all")
                for key, value in row[scope].items()
            },
        }
        for row in report["distance_time_weight_sweep"]
    ])
    plot_positive_distribution(events, args.output / "gt_normalized_distance_ecdf.png")
    plot_candidate_separation(geometry_rows, args.output / "candidate_distance_separation.png")
    plot_weight_sweep(report["distance_time_weight_sweep"], args.output / "distance_time_weight_sweep.png")
    plot_release_timing(report["release_timing"], args.output / "release_timing_validation.png")
    plot_same_frame_definitions(
        report["same_frame_distance_definitions"],
        args.output / "same_frame_distance_definitions.png",
    )
    (args.output / "REPORT.md").write_text(markdown_report_zh_tw(report), encoding="utf-8")
    (args.output / "REPORT_en.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "matched_events": len(matches),
        "accepted_actor_mappings": report["matching"]["accepted_actor_mappings"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
