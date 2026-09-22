# -*- coding: utf-8 -*-
"""Kalman + Hungarian local actor identity association.

This module has exactly one responsibility: attach the same actor ID to
detections in different frames.  It does *not* associate people with vehicles
or litter; those relations are many-to-many hypotheses solved downstream.
"""
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import (
    BoxKalmanFilter,
    KalmanConfig,
    TrackMeasurement,
    bbox_to_measurement,
    measurement_covariance,
)


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class _ActorTrack:
    track_id: int
    class_name: str
    kalman: BoxKalmanFilter
    last_observed_frame: int


class KalmanHungarianTracker:
    """Prediction-gated one-to-one association for detector boxes."""

    def __init__(
        self,
        iou_threshold=0.3,
        max_missed_frames=30,
        next_track_id=1,
        nis_gate=13.28,
        fps=10.0,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = max(int(max_missed_frames), 1)
        self.next_track_id = max(int(next_track_id), 1)
        self.nis_gate = max(float(nis_gate), 0.0)
        self.fps = float(fps) if fps and float(fps) > 0.0 else 10.0
        self._tracks: Dict[int, _ActorTrack] = {}
        self._last_frame_index: Optional[int] = None

    @staticmethod
    def _measurement(actor, frame_index):
        confidence = actor.get("confidence", actor.get("pose_conf", 1.0))
        return TrackMeasurement(
            frame_index=int(frame_index),
            bbox_xyxy=tuple(map(float, actor["box"][:4])),
            confidence=float(confidence),
        )

    @staticmethod
    def _association_cost(track, actor, frame_index):
        measurement = KalmanHungarianTracker._measurement(actor, frame_index)
        z = bbox_to_measurement(
            measurement.bbox_xyxy, track.kalman.config.min_box_size
        )
        predicted = track.kalman.mean
        innovation = z - predicted[:4]
        covariance = (
            track.kalman.covariance[:4, :4]
            + measurement_covariance(z, measurement.confidence, track.kalman.config)
        )
        try:
            nis = float(innovation.T.dot(np.linalg.solve(covariance, innovation)))
        except np.linalg.LinAlgError:
            nis = float(innovation.T.dot(np.linalg.pinv(covariance)).dot(innovation))
        iou = _bbox_iou(track.kalman.estimate().bbox_xyxy, measurement.bbox_xyxy)
        # Either geometric overlap or statistically plausible KF innovation may
        # open the gate.  The latter recovers a fast actor after short occlusion.
        if iou < 0.0 or not math.isfinite(nis):
            return math.inf
        return (1.0 - iou) + 0.04 * min(max(nis, 0.0), 50.0), iou, nis

    def update(self, actors: Iterable[dict], frame_index=None) -> List[dict]:
        actors = [dict(actor) for actor in actors]
        if frame_index is None:
            frame_index = 0 if self._last_frame_index is None else self._last_frame_index + 1
        frame_index = int(frame_index)
        if self._last_frame_index is not None and frame_index < self._last_frame_index:
            raise ValueError("actor frames must be monotonic")
        self._last_frame_index = frame_index

        # One malformed detector result must not abort every actor track.
        valid_actors = []
        for actor in actors:
            try:
                self._measurement(actor, frame_index)
            except (KeyError, TypeError, ValueError):
                continue
            valid_actors.append(actor)
        actors = valid_actors

        # A long-expired track must not be allowed into the assignment matrix:
        # otherwise a recycled actor at the same location can resurrect its ID.
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_observed_frame <= self.max_missed_frames
        }

        # Prediction is performed once before building the complete assignment
        # matrix; Hungarian then binds locations only within the same class.
        for track in self._tracks.values():
            track.kalman.predict(frame_index)

        assigned_detection_ids = set()
        assigned_track_ids = set()
        classes = sorted({
            str(actor.get("cls", "")).lower() for actor in actors
        } | {
            track.class_name for track in self._tracks.values()
        })
        for class_name in classes:
            detection_indices = [
                index for index, actor in enumerate(actors)
                if str(actor.get("cls", "")).lower() == class_name
            ]
            tracks = sorted(
                (
                    track for track in self._tracks.values()
                    if track.class_name == class_name
                ),
                key=lambda item: item.track_id,
            )
            if not detection_indices or not tracks:
                continue

            costs = np.full((len(tracks), len(detection_indices)), 1e9, dtype=np.float64)
            gates = {}
            for row, track in enumerate(tracks):
                for col, detection_index in enumerate(detection_indices):
                    value = self._association_cost(
                        track, actors[detection_index], frame_index
                    )
                    if not isinstance(value, tuple):
                        continue
                    cost, iou, nis = value
                    # Gate before Hungarian. Letting an invalid pair occupy a
                    # row/column and rejecting it afterwards can suppress a
                    # valid off-diagonal assignment.
                    if iou < self.iou_threshold and nis > self.nis_gate:
                        continue
                    gates[(row, col)] = (iou, nis)
                    costs[row, col] = cost

            row_indices, col_indices = linear_sum_assignment(costs)
            for row, col in zip(row_indices.tolist(), col_indices.tolist()):
                if costs[row, col] >= 1e8:
                    continue
                iou, nis = gates[(row, col)]
                track = tracks[row]
                detection_index = detection_indices[col]
                measurement = self._measurement(actors[detection_index], frame_index)
                track.kalman.update(measurement)
                track.last_observed_frame = frame_index
                actors[detection_index]["track_id"] = track.track_id
                assigned_detection_ids.add(detection_index)
                assigned_track_ids.add(track.track_id)

        for detection_index, actor in enumerate(actors):
            if detection_index in assigned_detection_ids:
                continue
            measurement = self._measurement(actor, frame_index)
            track_id = self.next_track_id
            self.next_track_id += 1
            self._tracks[track_id] = _ActorTrack(
                track_id=track_id,
                class_name=str(actor.get("cls", "")).lower(),
                kalman=BoxKalmanFilter(
                    measurement,
                    class_name=str(actor.get("cls", "")).lower(),
                    config=KalmanConfig(frames_per_second=self.fps),
                ),
                last_observed_frame=frame_index,
            )
            actor["track_id"] = track_id
            assigned_track_ids.add(track_id)

        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if frame_index - track.last_observed_frame <= self.max_missed_frames
        }
        for actor in actors:
            actor["observed"] = True
            actor.setdefault("source", "seg_predict")
        return actors


__all__ = ["KalmanHungarianTracker"]
