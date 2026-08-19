# -*- coding: utf-8 -*-
"""Reproducible, label-safe utilities for Smart Backtrack research.

This module never runs detection or changes litter confirmation.  It replays
the immutable resolver snapshot written to a candidate sidecar, so parameter
experiments compare attribution only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .annotations import (
    CANDIDATE_SCHEMA,
    AnnotationError,
    evaluate_candidates,
    candidate_routes,
    load_records,
    normalize_route,
    _event_id,
    _run_id,
)
from .costs import BacktrackCostConfig
from .resolver import SmartBacktrackConfig, SmartBacktrackResolver
from .sidecar import build_candidate_record, build_run_record, write_jsonl


STUDY_SCHEMA = "smart-backtrack-study/v1"
SPLITS = ("development", "validation", "test")


@dataclass(frozen=True)
class StudyConfig:
    """Serializable resolver parameters for one, deterministic trial.

    The current production cost functions already expose the global route
    weights below.  More granular cost-feature ablations must be introduced
    as explicit resolver options rather than environment variables, so every
    trial remains auditable.
    """

    name: str = "full-default"
    stage: str = "full"
    max_back_frames: Optional[int] = None
    top_k_people: int = 5
    top_k_vehicles: int = 5
    dustbin_cost: float = 7.0
    null_vehicle_penalty: float = 1.4
    direct_vehicle_penalty: float = 0.9
    ac_weight: float = 0.75
    bc_support_bonus: float = 0.25
    sigma_floor_px: float = 2.0
    two_point_max_back_seconds: float = 0.4
    two_point_prior_cost: float = 1.0
    max_forward_release_seconds: float = 0.5
    distance_weight: float = 1.0
    time_weight: float = 1.0
    kalman_process_noise_scale: float = 1.0
    kalman_measurement_noise_scale: float = 1.0
    kalman_max_extrapolation_seconds: Optional[float] = None
    confidence_weighted_trajectory: Optional[bool] = None
    ac_overlap_weight: Optional[float] = None

    def __post_init__(self):
        if self.stage not in {
            "distance_time", "kalman_rts", "confidence", "uncertainty",
            "reverse", "full"
        }:
            raise ValueError("unknown study stage: {}".format(self.stage))
        if self.top_k_people < 1 or self.top_k_vehicles < 1:
            raise ValueError("top-k must be positive")
        if self.dustbin_cost <= 0.0 or self.sigma_floor_px <= 0.0:
            raise ValueError("dustbin_cost and sigma_floor_px must be positive")
        if self.two_point_max_back_seconds < 0.0:
            raise ValueError("two_point_max_back_seconds must be non-negative")
        if self.two_point_prior_cost < 0.0:
            raise ValueError("two_point_prior_cost must be non-negative")
        if self.max_forward_release_seconds < 0.0:
            raise ValueError("max_forward_release_seconds must be non-negative")
        if self.distance_weight < 0.0 or self.time_weight < 0.0:
            raise ValueError("distance/time weights must be non-negative")
        if self.distance_weight + self.time_weight <= 0.0:
            raise ValueError("at least one distance/time weight must be positive")
        if self.ac_overlap_weight is not None and self.ac_overlap_weight < 0.0:
            raise ValueError("ac_overlap_weight must be non-negative")
        if self.kalman_process_noise_scale <= 0.0:
            raise ValueError("kalman_process_noise_scale must be positive")
        if self.kalman_measurement_noise_scale <= 0.0:
            raise ValueError("kalman_measurement_noise_scale must be positive")
        if (
            self.kalman_max_extrapolation_seconds is not None
            and self.kalman_max_extrapolation_seconds < 0.0
        ):
            raise ValueError(
                "kalman_max_extrapolation_seconds must be non-negative"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StudyConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError("unknown study config keys: {}".format(sorted(unknown)))
        return cls(**dict(value))

    def resolver_config(self, fps: float) -> SmartBacktrackConfig:
        defaults = SmartBacktrackConfig.from_env(fps=fps)
        cost_config = BacktrackCostConfig.for_stage(
            self.stage,
            distance_weight=float(self.distance_weight),
            time_weight=float(self.time_weight),
        )
        if self.ac_overlap_weight is not None:
            cost_config = replace(
                cost_config,
                ac_weights={
                    **cost_config.ac_weights,
                    "overlap": float(self.ac_overlap_weight),
                },
            )
        use_kalman_rts = self.stage in {
            "kalman_rts", "uncertainty", "reverse", "full"
        }
        use_reverse_trajectory = self.stage in {"reverse", "full"}
        return SmartBacktrackConfig(
            max_back_frames=(
                int(self.max_back_frames)
                if self.max_back_frames is not None else defaults.max_back_frames
            ),
            top_k_people=int(self.top_k_people),
            top_k_vehicles=int(self.top_k_vehicles),
            dustbin_cost=float(self.dustbin_cost),
            null_vehicle_penalty=float(self.null_vehicle_penalty),
            direct_vehicle_penalty=float(self.direct_vehicle_penalty),
            ac_weight=float(self.ac_weight),
            bc_support_bonus=float(self.bc_support_bonus),
            sigma_floor_px=float(self.sigma_floor_px),
            two_point_max_back_seconds=float(self.two_point_max_back_seconds),
            two_point_prior_cost=float(self.two_point_prior_cost),
            max_forward_release_seconds=float(
                self.max_forward_release_seconds
            ),
            cost_config=cost_config,
            use_kalman_rts=use_kalman_rts,
            confidence_aware_kalman=self.stage != "kalman_rts",
            use_uncertainty_gates=self.stage in {
                "uncertainty", "reverse", "full"
            },
            kalman_process_noise_scale=float(
                self.kalman_process_noise_scale
            ),
            kalman_measurement_noise_scale=float(
                self.kalman_measurement_noise_scale
            ),
            kalman_max_extrapolation_seconds=(
                float(self.kalman_max_extrapolation_seconds)
                if self.kalman_max_extrapolation_seconds is not None else None
            ),
            use_reverse_trajectory=use_reverse_trajectory,
            confidence_weighted_trajectory=(
                bool(self.confidence_weighted_trajectory)
                if self.confidence_weighted_trajectory is not None
                else self.stage == "full"
            ),
        )


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _candidate_events(records: Iterable[Mapping[str, Any]]):
    return [
        item for item in records
        if item.get("schema") == CANDIDATE_SCHEMA
        and str(item.get("record_type", "")).lower() == "candidate"
    ]


def build_manifest(
    candidate_records: Sequence[Mapping[str, Any]], *, seed: int = 20260802
) -> Dict[str, Any]:
    """Create a deterministic 60/20/20 group split without leakage.

    Parent directory is the conservative default group until camera/date/session
    metadata is supplied upstream.  A caller may edit only this manifest before
    it is locked; its digest is stored in every later trial output.
    """
    groups: Dict[str, list] = {}
    for record in _candidate_events(candidate_records):
        video = record.get("video", {}) if isinstance(record.get("video"), Mapping) else {}
        path = video.get("input_video")
        group = str(record.get("study_group") or (Path(path).parent if path else "unknown"))
        groups.setdefault(group, []).append(record)
    if not groups:
        raise AnnotationError("candidate sidecar contains no confirmed-event records")
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    totals = {split: 0 for split in SPLITS}
    entries = []
    for group in ordered:
        # Keep whole groups together; choose currently smallest target bucket.
        split = min(SPLITS, key=lambda item: (totals[item] / {"development": .6, "validation": .2, "test": .2}[item], item))
        for record in groups[group]:
            video = record.get("video", {})
            event = record.get("event", {})
            entries.append({
                "run_id": str(_run_id(record) or ""),
                "event_id": str(_event_id(record) or ""),
                "input_video": video.get("input_video"),
                "video_sha256": _file_sha256(video.get("input_video")),
                "fps": video.get("fps"),
                "group": group,
                "split": split,
            })
            totals[split] += 1
    manifest = {
        "schema": STUDY_SCHEMA,
        "seed": int(seed),
        "split_rule": "grouped_60_20_20_parent_directory_default",
        "git_commit": _git_commit(),
        "entries": sorted(entries, key=lambda item: (item["group"], item["event_id"])),
    }
    manifest["digest"] = _stable_digest({key: value for key, value in manifest.items() if key != "digest"})
    return manifest


def _manifest_keys(manifest: Mapping[str, Any], split: str):
    if split not in SPLITS:
        raise ValueError("split must be one of {}".format(SPLITS))
    return {
        (str(item.get("run_id") or ""), str(item.get("event_id") or ""))
        for item in manifest.get("entries", [])
        if item.get("split") == split
    }


def replay_candidates(
    candidate_records: Sequence[Mapping[str, Any]], study_config: StudyConfig,
    *, manifest: Optional[Mapping[str, Any]] = None, split: Optional[str] = None,
) -> list:
    """Replay only frozen resolver inputs; v1 rows without them are rejected."""
    selected_keys = _manifest_keys(manifest, split) if manifest and split else None
    output = []
    started = time.monotonic()
    for record in _candidate_events(candidate_records):
        event = record.get("event", {})
        key = (str(_run_id(record) or ""), str(_event_id(record) or ""))
        if selected_keys is not None and key not in selected_keys:
            continue
        task = record.get("resolver_input")
        if not isinstance(task, Mapping) or not task.get("actor_frames"):
            raise AnnotationError("candidate {} has no replayable resolver_input; regenerate sidecar".format(key))
        fps = float(task.get("fps", record.get("video", {}).get("fps", 10.0)) or 10.0)
        resolver = SmartBacktrackResolver(fps=fps, config=study_config.resolver_config(fps))
        resolution = resolver.resolve_task(dict(task))
        rewritten_event = dict(event)
        rewritten_event["backtrack"] = {
            "route_id": resolution.route_id,
            "route_type": resolution.route_type,
            "person_key": resolution.person_key,
            "vehicle_key": resolution.vehicle_key,
            "score": resolution.total_cost,
            "release_frame": resolution.release_frame,
            "release_point": resolution.release_point,
            "margin_to_second": resolution.margin_to_second,
            "actor_margins": dict(resolution.actor_margins),
        }
        video = record.get("video", {})
        trial_record = build_candidate_record(
            task, rewritten_event, resolution.routes,
            video.get("input_video"), video.get("output_video"),
        )
        trial_record["study"] = {
            "schema": STUDY_SCHEMA,
            "config": asdict(study_config),
            "config_digest": _stable_digest(asdict(study_config)),
            "manifest_digest": manifest.get("digest") if manifest else None,
            "split": split,
        }
        output.append(trial_record)
    elapsed = time.monotonic() - started
    return output, {"events": len(output), "elapsed_seconds": elapsed}


def evaluate_trial(candidates, annotations, *, manifest=None, split=None):
    """Evaluate reviewed labels only; never turns coverage into accuracy."""
    if manifest and split:
        keys = _manifest_keys(manifest, split)
        def keep(record):
            event = record.get("event", {})
            return (str(_run_id(record) or ""), str(_event_id(record) or "")) in keys
        candidates = [item for item in candidates if item.get("record_type") == "candidate" and keep(item)]
        annotations = [item for item in annotations if item.get("record_type") == "event" and (str(_run_id(item) or ""), str(_event_id(item) or "")) in keys]
    report = evaluate_candidates(candidates, annotations)
    candidate_by_key = {
        (str(_run_id(item) or ""), str(_event_id(item) or "")): item
        for item in candidates
    }
    non_null_errors = []
    for truth in annotations:
        if truth.get("ignore") or truth.get("event_label") != "litter":
            continue
        review = truth.get("review", {})
        if review.get("person") != "reviewed" or review.get("vehicle") != "reviewed":
            continue
        candidate = candidate_by_key.get((str(_run_id(truth) or ""), str(_event_id(truth) or "")))
        if not candidate:
            continue
        ranked = candidate_routes(candidate)
        if not ranked or ranked[0]["route_type"] == "null":
            continue
        truths = [normalize_route(item, index) for index, item in enumerate(truth.get("admissible_routes", []), 1)]
        correct = any(
            ranked[0]["route_type"] == item["route_type"]
            and ranked[0]["person_id"] == item["person_id"]
            and ranked[0]["vehicle_id"] == item["vehicle_id"]
            for item in truths
        )
        non_null_errors.append(0 if correct else 1)
    ci = None
    if non_null_errors:
        rng = random.Random(20260802)
        samples = []
        for _ in range(2000):
            samples.append(sum(rng.choice(non_null_errors) for _ in non_null_errors) / len(non_null_errors))
        samples.sort()
        ci = [samples[49], samples[1949]]
    report["safety"] = {
        "selected_non_null": len(non_null_errors),
        "wrong_non_null": sum(non_null_errors),
        "wrong_non_null_rate": (sum(non_null_errors) / len(non_null_errors) if non_null_errors else None),
        "wrong_non_null_rate_bootstrap_95ci": ci,
        "selection_policy": "report only; compare validation trials before locked test",
    }
    report["study"] = {"schema": STUDY_SCHEMA, "manifest_digest": manifest.get("digest") if manifest else None, "split": split}
    return report


def load_config(path: str) -> StudyConfig:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("study config must be JSON (YAML needs an explicit converter): {}".format(exc))
    if not isinstance(value, Mapping):
        raise ValueError("study config must be an object")
    return StudyConfig.from_mapping(value)


__all__ = [
    "STUDY_SCHEMA", "StudyConfig", "build_manifest", "evaluate_trial",
    "load_config", "replay_candidates",
]
