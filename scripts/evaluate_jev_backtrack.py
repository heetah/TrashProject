#!/usr/bin/env python3
"""Research-only JEV reranking evaluation for frozen Smart Backtrack sidecars.

The script sends compact, structured route evidence to the TypeSafe/JEV API.
It never sends video frames, masks, raw actor trajectories, API credentials, or
the full sidecar.  JEV may select only one route ID already produced by the
local solver; an invalid response is guarded by falling back to the local
assignment.  The script does not change production behavior.

The reported route metrics are exploratory because the reviewed set is a
positive-only, single-camera set.  The fixed 58-case denominator is retained
so upstream missed events cannot disappear from the comparison.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.error
import urllib.request


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from validate_attribution_formula import (  # noqa: E402
    candidate_records,
    load_jsonl,
    map_manual_actors,
    match_events,
    prediction_tuple,
)


API_URL = "https://api.typesafe.ai/v1/systemone"
ACCEPTED_MATCH_TIERS = frozenset({"strict", "moderate"})
SCHEMA = "jev-backtrack-evaluation/v1"
FEATURE_KEYS = (
    "direct_distance",
    "endpoint_proximity",
    "release_distance",
    "spatial_mahalanobis",
    "quality",
    "time",
    "uncertainty",
    "continuity",
    "overlap",
    "direction",
    "relative_motion_deficit",
    "reverse_direction",
    "release_prior",
    "exit_deficit",
    "boundary_depth",
)


@dataclass
class ApiResult:
    clip_id: str
    baseline_route_id: str | None
    candidate_ids: list[str]
    choice: str | None
    guarded_choice: str | None
    confidence: float | None
    probabilities: Mapping[str, Any] | None
    latency_sec: float
    status: int | None
    error: str | None
    input_tokens: int | None
    output_tokens: int | None
    payload_bytes: int


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rounded(value: Any, digits: int = 4) -> float | None:
    number = _finite(value)
    return round(number, digits) if number is not None else None


def _round_vector(value: Any, digits: int = 2) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    values = [_rounded(item, digits) for item in value]
    return values if all(item is not None for item in values) else None


def _component_summary(component: Any) -> dict[str, Any] | None:
    if not isinstance(component, Mapping):
        return None
    output: dict[str, Any] = {"total": _rounded(component.get("total"), 5)}
    raw = component.get("raw_features")
    if isinstance(raw, Mapping):
        output["features"] = {
            key: _rounded(raw.get(key), 4)
            for key in FEATURE_KEYS
            if _finite(raw.get(key)) is not None
        }
    return output


def _mask_summary(metadata: Mapping[str, Any]) -> dict[str, Any] | None:
    mask = metadata.get("mask_temporal_evidence")
    if not isinstance(mask, Mapping):
        return None
    signed = [
        number for number in (_finite(value) for value in mask.get("history_signed", []))
        if number is not None
    ]
    return {
        "history_expected": mask.get("history_expected_count"),
        "history_observed": mask.get("history_observed_count"),
        "mean_signed": _rounded(sum(signed) / len(signed), 4) if signed else None,
        "mean_abs_signed": _rounded(sum(abs(value) for value in signed) / len(signed), 4)
        if signed
        else None,
        "last_signed": _rounded(signed[-1], 4) if signed else None,
    }


def route_summary(route: Mapping[str, Any]) -> dict[str, Any]:
    metadata = route.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    costs = metadata.get("costs")
    costs = costs if isinstance(costs, Mapping) else {}
    return {
        "route_id": route.get("route_id"),
        "route_type": route.get("route_type"),
        "person_key": route.get("person_key"),
        "vehicle_key": route.get("vehicle_key"),
        "local_cost": _rounded(route.get("cost"), 5),
        "local_rank": route.get("rank"),
        "solver_selected": bool(route.get("selected")),
        "valid": bool(route.get("valid", True)),
        "release_frame": metadata.get("release_frame"),
        "release_point": _round_vector(metadata.get("release_point"), 1),
        "release_velocity": _round_vector(metadata.get("release_velocity"), 1),
        "components": {
            name: _component_summary(costs.get(name))
            for name in ("BA", "AC", "BC")
            if costs.get(name) is not None
        },
        "mask": _mask_summary(metadata),
    }


def candidate_summaries(record: Mapping[str, Any], top_k: int) -> list[dict[str, Any]]:
    routes = [
        route
        for route in record.get("routes", [])
        if isinstance(route, Mapping) and route.get("route_id")
    ]
    routes.sort(key=lambda item: (int(item.get("rank", 10**9)), str(item["route_id"])))
    chosen = routes[: max(1, int(top_k))]
    if not any(route.get("route_id") == "null" for route in chosen):
        chosen.extend(route for route in routes if route.get("route_id") == "null")
    unique: dict[str, dict[str, Any]] = {}
    for route in chosen:
        summary = route_summary(route)
        unique[str(summary["route_id"])] = summary
    return list(unique.values())


def build_payload(record: Mapping[str, Any], top_k: int) -> tuple[dict[str, Any], list[str]]:
    candidates = candidate_summaries(record, top_k)
    candidate_ids = [str(item["route_id"]) for item in candidates]
    event = record.get("event") if isinstance(record.get("event"), Mapping) else {}
    history = record.get("litter_history")
    history = history if isinstance(history, list) else []
    assignment = record.get("assignment")
    assignment = assignment if isinstance(assignment, Mapping) else {}
    instructions = (
        "You are a conservative Smart Backtrack route reranker. Choose exactly "
        "one listed route ID for this already-confirmed litter event. The local "
        "solver generated the candidate set; you may not invent an ID, change an "
        "actor key, or infer a new route. Lower local_cost is generally better, "
        "but it is only one signal. BA is person-to-litter evidence, AC is "
        "person-to-vehicle continuity, and BC is vehicle-to-litter evidence. "
        "A direct_vehicle route needs BC support; a person_vehicle route needs "
        "compatible BA, AC, and BC support; a person route needs BA support. "
        "Check release-frame proximity, spatial distance, continuity, direction, "
        "quality, uncertainty, and mask temporal evidence. Do not treat the "
        "solver's selected flag or local rank as ground truth. Choose null when "
        "all non-null routes are contradictory or insufficient."
    )
    payload = {
        "state": {
            "schema": "smart-backtrack-jev-rerank-input/v1",
            "instructions": instructions,
            "event": {
                "clip_id": str(record.get("video", {}).get("input_video", "")).rsplit("/", 1)[-1].removesuffix(".mp4"),
                "litter_id": event.get("litter_id"),
                "history_frames": [item.get("frame_index") for item in history[:12]],
                "history_count": len(history),
                "local_assignment": {
                    "route_id": assignment.get("route_id"),
                    "route_type": assignment.get("route_type"),
                    "release_frame": assignment.get("release_frame"),
                },
            },
            "candidates": {item["route_id"]: item for item in candidates},
        },
        "model": "jev-latest",
        "questions": {
            "route_choice": {
                "type": "choice",
                "criteria": {
                    item["route_id"]: (
                        f"Candidate {item['route_id']}; inspect its structured "
                        "evidence in state.candidates and return this ID only if "
                        "it is the best supported attribution."
                    )
                    for item in candidates
                },
            }
        },
    }
    return payload, candidate_ids


def _choice_from_response(body: Mapping[str, Any]) -> tuple[str | None, float | None, Mapping[str, Any] | None, int | None, int | None]:
    answers = body.get("answers")
    answer = answers.get("route_choice") if isinstance(answers, Mapping) else None
    if not isinstance(answer, Mapping):
        return None, None, None, None, None
    choice = answer.get("choice")
    if choice is not None:
        choice = str(choice).strip()
        if choice.lower() == "null":
            choice = "null"
    return (
        choice,
        _finite(answer.get("confidence")),
        answer.get("probabilities") if isinstance(answer.get("probabilities"), Mapping) else None,
        int(body.get("usage", {}).get("input_tokens")) if body.get("usage", {}).get("input_tokens") is not None else None,
        int(body.get("usage", {}).get("output_tokens")) if body.get("usage", {}).get("output_tokens") is not None else None,
    )


def call_jev(
    payload: Mapping[str, Any],
    *,
    api_key: str,
    clip_id: str,
    baseline_route_id: str | None,
    candidate_ids: Sequence[str],
    timeout_sec: float,
    max_retries: int = 2,
) -> ApiResult:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    last_error: str | None = None
    status: int | None = None
    response_body: Mapping[str, Any] | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                status = int(response.status)
                decoded = json.loads(response.read().decode("utf-8"))
                response_body = decoded if isinstance(decoded, Mapping) else None
                break
        except urllib.error.HTTPError as error:
            status = int(error.code)
            raw = error.read().decode("utf-8", errors="replace")
            last_error = f"HTTP {status}: {raw[:500]}"
            if status not in {408, 425, 429} and status < 500:
                break
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = f"{type(error).__name__}: {str(error)[:300]}"
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
    latency = time.perf_counter() - started
    if response_body is None:
        return ApiResult(
            clip_id=clip_id,
            baseline_route_id=baseline_route_id,
            candidate_ids=list(candidate_ids),
            choice=None,
            guarded_choice=baseline_route_id,
            confidence=None,
            probabilities=None,
            latency_sec=latency,
            status=status,
            error=last_error or "empty response",
            input_tokens=None,
            output_tokens=None,
            payload_bytes=len(body),
        )
    choice, confidence, probabilities, input_tokens, output_tokens = _choice_from_response(response_body)
    guarded = choice if choice in set(candidate_ids) else baseline_route_id
    error = None if choice in set(candidate_ids) else "invalid_choice_guarded_to_baseline"
    return ApiResult(
        clip_id=clip_id,
        baseline_route_id=baseline_route_id,
        candidate_ids=list(candidate_ids),
        choice=choice,
        guarded_choice=guarded,
        confidence=confidence,
        probabilities=probabilities,
        latency_sec=latency,
        status=status,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        payload_bytes=len(body),
    )


def _truth_tuple(match: Mapping[str, Any], expected_route: str) -> tuple[Any, ...] | None:
    # The expected route comes from the manual actor annotations.  The route
    # derived from ``match`` describes which manually annotated actors were
    # successfully mapped to model tracklets; using it as truth would turn an
    # actor-mapping failure into an apparent success.
    route = expected_route
    person = match.get("correct_person_key")
    vehicle = match.get("correct_vehicle_key")
    if route == "person_vehicle" and (person is None or vehicle is None):
        return None
    if route == "direct_vehicle" and vehicle is None:
        return None
    if route == "person" and person is None:
        return None
    if route == "null":
        return None
    return (route, person, vehicle)


def _assignment_for_choice(record: Mapping[str, Any], choice: str | None) -> Mapping[str, Any] | None:
    for route in record.get("routes", []):
        if isinstance(route, Mapping) and route.get("route_id") == choice:
            return {
                "route_type": route.get("route_type"),
                "person_key": route.get("person_key"),
                "vehicle_key": route.get("vehicle_key"),
            }
    return None


def _metric(successes: int, total: int) -> dict[str, Any]:
    rate = successes / total if total else None
    return {"successes": successes, "denominator": total, "rate": rate}


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2.0 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "semantics": "exploratory route attribution proxy; positive-only reviewed set",
    }


def _baseline_fps(log_root: Path) -> dict[str, Any]:
    values: list[dict[str, float]] = []
    pattern = re.compile(
        r"Frames:\s+(?P<frames>\d+)\s+\| Wall:\s+(?P<wall>[0-9.]+)s.*?"
        r"Video loop:\s+(?P<loop>[0-9.]+)s\s+\| Throughput:\s+(?P<fps>[0-9.]+) frame/s",
        re.S,
    )
    for path in log_root.rglob("pipeline.log"):
        match = pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
        if not match:
            continue
        values.append({key: float(value) for key, value in match.groupdict().items()})
    if not values:
        return {"log_count": 0}
    frames = sum(item["frames"] for item in values)
    loop = sum(item["loop"] for item in values)
    wall = sum(item["wall"] for item in values)
    return {
        "log_count": len(values),
        "frames": int(frames),
        "aggregate_loop_sec": loop,
        "aggregate_wall_sec": wall,
        "aggregate_loop_fps": frames / loop if loop else None,
        "aggregate_wall_fps": frames / wall if wall else None,
        "median_case_fps": statistics.median(item["fps"] for item in values),
    }


def evaluate(
    *,
    ground_truth: Path,
    candidates_root: Path,
    case_outcomes_path: Path,
    production_output_root: Path,
    api_key: str,
    output: Path,
    top_k: int,
    timeout_sec: float,
    max_cases: int | None,
) -> dict[str, Any]:
    gt_root = ground_truth
    clips = load_jsonl(gt_root / "clip_annotations.jsonl")
    events = load_jsonl(gt_root / "event_annotations.jsonl")
    actors = load_jsonl(gt_root / "actor_annotations.jsonl")
    project = json.loads((gt_root / "project.json").read_text(encoding="utf-8"))
    metadata = {row["filename"]: row for row in project["videos"]}
    records = candidate_records(candidates_root)
    matches = match_events(events, records, metadata)
    map_manual_actors(matches, actors)
    match_by_clip = {str(row["clip_id"]): row for row in matches}
    record_by_clip = {str(row["clip_id"]): row["candidate"] for row in matches}
    target_rows = list(csv.DictReader(case_outcomes_path.open(encoding="utf-8", newline="")))
    target_by_clip = {str(row["clip_id"]): row for row in target_rows}
    if max_cases is not None:
        target_ids = set(sorted(match_by_clip)[: max(0, int(max_cases))])
    else:
        target_ids = set(match_by_clip)

    api_results: list[ApiResult] = []
    request_meta: list[dict[str, Any]] = []
    for index, clip in enumerate(sorted(target_ids), start=1):
        match = match_by_clip[clip]
        record = record_by_clip[clip]
        payload, candidate_ids = build_payload(record, top_k)
        assignment = record.get("assignment") if isinstance(record.get("assignment"), Mapping) else {}
        result = call_jev(
            payload,
            api_key=api_key,
            clip_id=clip,
            baseline_route_id=str(assignment.get("route_id")) if assignment.get("route_id") else None,
            candidate_ids=candidate_ids,
            timeout_sec=timeout_sec,
        )
        api_results.append(result)
        request_meta.append({
            "clip_id": clip,
            "candidate_count": len(candidate_ids),
            "payload_bytes": result.payload_bytes,
            "latency_sec": result.latency_sec,
            "status": result.status,
            "error": result.error,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "baseline_route_id": result.baseline_route_id,
            "jev_choice": result.choice,
            "guarded_choice": result.guarded_choice,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
        })
        print(f"JEV {index}/{len(target_ids)} {clip}: {result.choice or 'ERROR'} ({result.latency_sec:.3f}s)")

    result_by_clip = {row.clip_id: row for row in api_results}
    exact_rows: list[dict[str, Any]] = []
    baseline_correct = 0
    jev_correct = 0
    guarded_correct = 0
    accepted_total = 0
    mapped_total = 0
    for clip, target in target_by_clip.items():
        match = match_by_clip.get(clip)
        accepted = str(target.get("accepted_event_match", "")).lower() == "true"
        if accepted:
            accepted_total += 1
        expected_route = str(target.get("expected_route_type") or "null")
        truth = _truth_tuple(match, expected_route) if match and accepted else None
        if truth is not None:
            mapped_total += 1
        record = record_by_clip.get(clip)
        baseline_assignment = record.get("assignment", {}) if isinstance(record, Mapping) else {}
        baseline_prediction = prediction_tuple(baseline_assignment) if record else None
        base_ok = bool(accepted and truth is not None and baseline_prediction == truth)
        if base_ok:
            baseline_correct += 1
        result = result_by_clip.get(clip)
        raw_assignment = _assignment_for_choice(record, result.choice) if result and record else None
        guarded_assignment = _assignment_for_choice(record, result.guarded_choice) if result and record else None
        raw_prediction = prediction_tuple(raw_assignment) if raw_assignment else None
        guarded_prediction = prediction_tuple(guarded_assignment) if guarded_assignment else None
        raw_ok = bool(accepted and truth is not None and raw_prediction == truth)
        guarded_ok = bool(accepted and truth is not None and guarded_prediction == truth)
        if raw_ok:
            jev_correct += 1
        if guarded_ok:
            guarded_correct += 1
        exact_rows.append({
            "clip_id": clip,
            "match_tier": match.get("match_tier") if match else None,
            "accepted_event_match": accepted,
            "truth": list(truth) if truth else None,
            "baseline_route_id": baseline_assignment.get("route_id") if record else None,
            "jev_route_id": result.choice if result else None,
            "guarded_route_id": result.guarded_choice if result else None,
            "baseline_correct": base_ok,
            "jev_correct": raw_ok,
            "guarded_correct": guarded_ok,
            "target_outcome": target.get("outcome"),
            "latency_sec": result.latency_sec if result else None,
            "confidence": result.confidence if result else None,
            "error": result.error if result else "no_api_result",
        })

    def _proxy(prediction_field: str, correct_field: str) -> dict[str, Any]:
        tp = 0
        fp = 0
        for row in exact_rows:
            if not row["accepted_event_match"] or row["truth"] is None:
                continue
            pred = row[prediction_field]
            if row[correct_field]:
                tp += 1
            elif pred not in (None, "null"):
                fp += 1
        return _prf(tp, fp, mapped_total - tp)

    baseline_proxy = _proxy("baseline_route_id", "baseline_correct")
    jev_proxy = _proxy("jev_route_id", "jev_correct")
    guarded_proxy = _proxy("guarded_route_id", "guarded_correct")
    baseline_fps = _baseline_fps(production_output_root)
    latency = [result.latency_sec for result in api_results if result.status == 200]
    successful = sum(result.status == 200 for result in api_results)
    candidate_record_count = len(records)
    mean_latency = statistics.mean(latency) if latency else None
    synchronous_extra = (mean_latency or 0.0) * candidate_record_count
    sync_loop_fps = None
    sync_wall_fps = None
    if baseline_fps.get("frames") and baseline_fps.get("aggregate_loop_sec") is not None:
        sync_loop_fps = baseline_fps["frames"] / (baseline_fps["aggregate_loop_sec"] + synchronous_extra)
        sync_wall_fps = baseline_fps["frames"] / (baseline_fps["aggregate_wall_sec"] + synchronous_extra)
    changed = [row for row in exact_rows if row["baseline_route_id"] != row["jev_route_id"] and row["jev_route_id"] is not None]
    improved = [row for row in exact_rows if not row["baseline_correct"] and row["jev_correct"]]
    regressed = [row for row in exact_rows if row["baseline_correct"] and not row["jev_correct"]]
    report = {
        "schema": SCHEMA,
        "api": {
            "endpoint": API_URL,
            "model_requested": "jev-latest",
            "api_key_configured": bool(api_key),
            "api_key_saved": False,
            "successful_calls": successful,
            "attempted_calls": len(api_results),
            "mean_latency_sec": mean_latency,
            "median_latency_sec": statistics.median(latency) if latency else None,
            "p95_latency_sec": sorted(latency)[max(0, math.ceil(0.95 * len(latency)) - 1)] if latency else None,
            "total_input_tokens": sum(result.input_tokens or 0 for result in api_results),
            "total_output_tokens": sum(result.output_tokens or 0 for result in api_results),
            "mean_payload_bytes": statistics.mean(result.payload_bytes for result in api_results) if api_results else None,
        },
        "candidate_policy": {
            "top_k_by_local_rank": top_k,
            "null_always_included": True,
            "matched_cases_with_candidate_records": len(match_by_clip),
            "fixed_reviewed_case_denominator": len(target_rows),
            "candidate_record_count_in_production_run": candidate_record_count,
            "frames_uploaded": False,
            "raw_masks_uploaded": False,
        },
        "route_metrics": {
            "baseline_end_to_end_route_correctness": _metric(baseline_correct, len(target_rows)),
            "jev_end_to_end_route_correctness": _metric(jev_correct, len(target_rows)),
            "guarded_jev_end_to_end_route_correctness": _metric(guarded_correct, len(target_rows)),
            "baseline_route_correctness_given_accepted_event": _metric(baseline_correct, accepted_total),
            "jev_route_correctness_given_accepted_event": _metric(jev_correct, accepted_total),
            "guarded_jev_route_correctness_given_accepted_event": _metric(guarded_correct, accepted_total),
            "mapped_route_subset_denominator": mapped_total,
            "baseline_precision_recall_f1_proxy": baseline_proxy,
            "jev_precision_recall_f1_proxy": jev_proxy,
            "guarded_jev_precision_recall_f1_proxy": guarded_proxy,
        },
        "decision_changes": {
            "changed_count": len(changed),
            "improved_count": len(improved),
            "regressed_count": len(regressed),
            "null_choices": sum(result.choice == "null" for result in api_results),
            "invalid_or_failed_guarded_to_baseline": sum(result.guarded_choice == result.baseline_route_id and result.choice != result.baseline_route_id for result in api_results),
            "changed_cases": changed,
        },
        "fps_analysis": {
            "baseline": baseline_fps,
            "mean_api_latency_extrapolated_to_all_candidate_records_sec": synchronous_extra,
            "synchronous_extrapolated_loop_fps": sync_loop_fps,
            "synchronous_extrapolated_wall_fps": sync_wall_fps,
            "async_steady_state_loop_fps": baseline_fps.get("aggregate_loop_fps"),
            "interpretation": (
                "A synchronous remote call adds one network round trip per confirmed "
                "candidate record. An asynchronous queue can preserve steady-state "
                "frame-loop FPS, but adds result-drain/tail latency and does not make "
                "the local GPU kernels faster. The extrapolation assumes the measured "
                "mean latency and is not a production throughput certification."
            ),
        },
        "limitations": [
            "JEV receives structured route evidence, not frames; it cannot recover an event that the local tracker never confirmed.",
            "The 58-case route set is positive-only and single-camera; precision/recall/F1 are explicitly exploratory proxies.",
            "No reviewed negative or cross-camera holdout is present, so enforcement accuracy and generalization are not certified.",
            "JEV Choice confidence/probabilities are model outputs and are not accuracy estimates.",
            "API latency and selection can vary across calls; this run is a one-pass external-service measurement.",
        ],
        "provenance": {
            "ground_truth": str(ground_truth.resolve()),
            "candidates_root": str(candidates_root.resolve()),
            "case_outcomes": str(case_outcomes_path.resolve()),
            "top_k": top_k,
            "selected_case_count": len(api_results),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "api_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in request_meta),
        encoding="utf-8",
    )
    with (output / "case_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(exact_rows[0]) if exact_rows else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(exact_rows)
    (output / "REPORT.md").write_text(render_report(report), encoding="utf-8")
    return report


def _pct(value: Any) -> str:
    return "N/A" if value is None else f"{float(value) * 100.0:.2f}%"


def render_report(report: Mapping[str, Any]) -> str:
    route = report["route_metrics"]
    fps = report["fps_analysis"]
    api = report["api"]
    changes = report["decision_changes"]
    baseline = route["baseline_end_to_end_route_correctness"]
    jev = route["jev_end_to_end_route_correctness"]
    guarded = route["guarded_jev_end_to_end_route_correctness"]
    lines = [
        "# JEV Smart Backtrack reranking evaluation",
        "",
        "This is a frozen-input, research-only comparison. JEV received compact structured route evidence; no video frames or raw masks were uploaded.",
        "",
        "## Route result",
        "",
        f"- Fixed 58-case end-to-end baseline: {baseline['successes']}/{baseline['denominator']} ({_pct(baseline['rate'])})",
        f"- Fixed 58-case raw JEV result: {jev['successes']}/{jev['denominator']} ({_pct(jev['rate'])})",
        f"- Fixed 58-case guarded JEV result: {guarded['successes']}/{guarded['denominator']} ({_pct(guarded['rate'])})",
        f"- JEV calls: {api['successful_calls']}/{api['attempted_calls']}; median latency {api['median_latency_sec']:.3f}s" if api["median_latency_sec"] is not None else "- JEV calls: no successful calls",
        f"- Changed choices: {changes['changed_count']}; improved: {changes['improved_count']}; regressed: {changes['regressed_count']}; NULL: {changes['null_choices']}",
        "",
        "The route precision/recall/F1 rows in `report.json` are labelled exploratory proxies. The current dataset has no reviewed negatives or independent camera holdout, so they are not production accuracy certification.",
        "",
        "## FPS impact",
        "",
        f"- Frozen production aggregate loop baseline: {fps.get('baseline', {}).get('aggregate_loop_fps', float('nan')):.2f} FPS",
        f"- Synchronous extrapolation with one JEV call per {report['candidate_policy']['candidate_record_count_in_production_run']} candidate records: {fps.get('synchronous_extrapolated_loop_fps', float('nan')):.2f} FPS",
        f"- Asynchronous steady-state loop estimate: {fps.get('async_steady_state_loop_fps', float('nan')):.2f} FPS, with remote result-drain latency added at the tail",
        "",
        "A remote JEV call cannot speed up YOLO, RT-DETR, or the local tracker. Queueing it after local candidate generation can preserve the frame loop, while synchronous use lowers throughput by the network round trip.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--case-outcomes", type=Path, required=True)
    parser.add_argument("--production-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="TYPESAFE_API_KEY")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing API key in environment variable {args.api_key_env}")
    report = evaluate(
        ground_truth=args.ground_truth,
        candidates_root=args.candidates_root,
        case_outcomes_path=args.case_outcomes,
        production_output_root=args.production_output_root,
        api_key=api_key,
        output=args.output,
        top_k=args.top_k,
        timeout_sec=args.timeout_sec,
        max_cases=args.max_cases,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
