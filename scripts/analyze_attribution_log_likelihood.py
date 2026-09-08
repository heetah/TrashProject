#!/usr/bin/env python3
"""Fit a minimal TrackFlow-inspired attribution likelihood on reviewed data.

This is a research analysis, not a production cost replacement.  It evaluates
only the first release-to-actor edge: direct-vehicle routes target the reviewed
vehicle, while person-vehicle routes target the reviewed person.  Candidate
features come from frozen sidecars; labels come only from reviewed manual data.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import optimize


SEED = 20260826
EPSILON = 1e-9
PRIMARY_L2 = 1.0
L2_SENSITIVITY = (0.01, 0.1, 1.0, 10.0)


def _load_validator(path: Path):
    spec = importlib.util.spec_from_file_location("attribution_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _target_actor(match: Mapping[str, Any]) -> tuple[str, tuple[str, int]] | None:
    route = str(match.get("derived_route_type"))
    if route == "direct_vehicle" and match.get("correct_vehicle_key") is not None:
        return "vehicle", tuple(match["correct_vehicle_key"])
    if route in {"person", "person_vehicle"} and match.get("correct_person_key") is not None:
        return "person", tuple(match["correct_person_key"])
    return None


def _history_timing(validator: Any, match: Mapping[str, Any]) -> tuple[int, int]:
    history = validator._ordered_history(match["candidate"].get("resolver_input", {}))
    if len(history) < 2:
        raise ValueError("fewer than two litter observations")
    b0 = int(history[0][0])
    h0 = max(int(history[1][0]) - b0, 1)
    return b0, h0


def _actor_type(tracklet: Mapping[str, Any]) -> str:
    value = str(tracklet.get("class_name"))
    return "vehicle" if value in {"vehicle", "scooter"} else value


def _geometry_at_release(
    validator: Any,
    tracklet: Mapping[str, Any],
    release: Mapping[str, Any],
    fps: float,
) -> dict[str, float] | None:
    observations = list(tracklet.get("observations", []))
    if not observations:
        return None
    release_frame = int(release["frame_index"])
    max_gap = max(int(round(0.25 * fps)), 0)
    nearby = [
        row for row in observations
        if abs(int(row["frame_index"]) - release_frame) <= max_gap
    ]
    if not nearby:
        return None
    observation = min(
        nearby,
        key=lambda row: (abs(int(row["frame_index"]) - release_frame), int(row["frame_index"])),
    )
    x1, y1, x2, y2 = map(float, observation["box"][:4])
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    actor_type = _actor_type(tracklet)
    if actor_type == "person":
        region = (x1, y1, x2, y1 + 0.72 * height)
        scale = height
        gate = 0.85
        direction_anchor = np.asarray(
            [(x1 + x2) * 0.5, y1 + 0.36 * height], dtype=float
        )
        direction_minimum_displacement = 0.12 * height
    elif actor_type == "vehicle":
        region = (
            x1 - 0.18 * width,
            y1 - 0.15 * height,
            x2 + 0.18 * width,
            y2 + 0.15 * height,
        )
        scale = math.hypot(width, height)
        gate = 0.80
        direction_anchor = np.asarray(
            [(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float
        )
        direction_minimum_displacement = 1e-6
    else:
        return None
    raw_distance = validator.point_to_rect_distance(release["mean_uv"], region)
    normalized_distance = raw_distance / max(scale, 1.0)
    release_point = np.asarray(release["mean_uv"][:2], dtype=float)
    velocity = np.asarray(release.get("velocity_uv", [0.0, 0.0])[:2], dtype=float)
    outward = release_point - direction_anchor
    outward_norm = float(np.linalg.norm(outward))
    velocity_norm = float(np.linalg.norm(velocity))
    direction_defined = bool(
        outward_norm > direction_minimum_displacement and velocity_norm > 1e-6
    )
    if direction_defined:
        direction_cosine = float(np.clip(
            outward @ velocity / (outward_norm * velocity_norm), -1.0, 1.0
        ))
        # Under a von Mises model centered on outward motion, negative
        # log-likelihood is affine in (1-cos(theta)).  Dividing by two keeps
        # the feature bounded in [0, 1] without changing its ordering.
        direction_penalty = 0.5 * (1.0 - direction_cosine)
    else:
        # cos(theta) has expectation zero under an uninformative uniform
        # direction, hence a neutral penalty of (1-0)/2 = 0.5.
        direction_cosine = 0.0
        direction_penalty = 0.5
    return {
        "normalized_distance": float(normalized_distance),
        "raw_distance": float(raw_distance),
        "actor_scale": float(scale),
        "gate": float(gate),
        "observation_frame": int(observation["frame_index"]),
        "direction_cosine": direction_cosine,
        "direction_penalty": direction_penalty,
        "direction_defined": direction_defined,
    }


def build_pair_rows(validator: Any, matches: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in matches:
        target = _target_actor(match)
        if target is None:
            continue
        target_type, correct_key = target
        try:
            b0, h0 = _history_timing(validator, match)
        except ValueError:
            continue
        event = match["event"]
        start = int(event["release_start_frame"])
        end = int(event["release_end_frame"])
        point_frame = int(event["release_point_frame"])
        fps = float(match["candidate"].get("video", {}).get("fps") or 10.0)
        event_key = f"{match['clip_id']}:{match['litter_id']}"
        for tracklet in match["candidate"].get("candidate_actors", []):
            if _actor_type(tracklet) != target_type:
                continue
            key = validator.actor_key(tracklet.get("actor_key"))
            if key is None:
                continue
            for release in match["candidate"].get("release_hypotheses", []):
                geometry = _geometry_at_release(validator, tracklet, release, fps)
                if geometry is None or geometry["normalized_distance"] > geometry["gate"]:
                    continue
                release_frame = int(release["frame_index"])
                is_correct_actor = tuple(key) == correct_key
                is_release_hit = start <= release_frame <= end
                rows.append({
                    "event_key": event_key,
                    "clip_id": str(match["clip_id"]),
                    "litter_id": int(match["litter_id"]),
                    "match_tier": str(match["match_tier"]),
                    "route_type": str(match["derived_route_type"]),
                    "actor_type": target_type,
                    "actor_key": list(key),
                    "correct_actor_key": list(correct_key),
                    "release_frame": release_frame,
                    "release_start_frame": start,
                    "release_end_frame": end,
                    "b0_frame": b0,
                    "h0_frames": h0,
                    "alpha": float((b0 - release_frame) / h0),
                    "alpha_gt": float((b0 - point_frame) / h0),
                    "is_correct_actor": bool(is_correct_actor),
                    "is_release_hit": bool(is_release_hit),
                    "label": bool(is_correct_actor and is_release_hit),
                    **geometry,
                })
    return rows


def _events(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(row["event_key"]) for row in rows})


def _time_center(rows: Sequence[Mapping[str, Any]]) -> float:
    by_event: dict[str, float] = {}
    for row in rows:
        by_event[str(row["event_key"])] = float(row["alpha_gt"])
    return float(np.median(list(by_event.values())))


def _matrix(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    alpha_center: float,
) -> np.ndarray:
    columns = []
    for feature in features:
        if feature == "distance":
            columns.append([float(row["normalized_distance"]) for row in rows])
        elif feature == "time":
            columns.append([abs(float(row["alpha"]) - alpha_center) for row in rows])
        elif feature == "direction":
            columns.append([float(row["direction_penalty"]) for row in rows])
        else:
            raise ValueError(f"unknown feature: {feature}")
    return np.asarray(columns, dtype=float).T


def _event_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["event_key"])
        counts[key] = counts.get(key, 0) + 1
    return np.asarray([1.0 / counts[str(row["event_key"])] for row in rows], dtype=float)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -value))


def fit_logistic(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    *,
    alpha_center: float,
    l2: float,
    monotone_penalties: bool = False,
) -> dict[str, Any]:
    x = _matrix(rows, features, alpha_center)
    y = np.asarray([float(bool(row["label"])) for row in rows], dtype=float)
    weights = _event_weights(rows)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 1e-12))
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(rows)), standardized])

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ beta
        loss = np.sum(weights * (np.logaddexp(0.0, logits) - y * logits))
        loss += 0.5 * float(l2) * float(beta[1:] @ beta[1:])
        probability = _sigmoid(logits)
        gradient = design.T @ (weights * (probability - y))
        gradient[1:] += float(l2) * beta[1:]
        return float(loss), gradient

    result = optimize.minimize(
        lambda beta: objective(beta),
        np.zeros(design.shape[1], dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=(
            [(None, None)] + [(None, 0.0)] * (design.shape[1] - 1)
            if monotone_penalties else None
        ),
    )
    if not result.success:
        raise RuntimeError(f"logistic fit failed: {result.message}")
    return {
        "features": list(features),
        "alpha_center": float(alpha_center),
        "l2": float(l2),
        "monotone_penalties": bool(monotone_penalties),
        "mean": mean,
        "scale": scale,
        "beta": np.asarray(result.x, dtype=float),
        "objective": float(result.fun),
    }


def predict(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    x = _matrix(rows, model["features"], float(model["alpha_center"]))
    standardized = (x - model["mean"]) / model["scale"]
    design = np.column_stack([np.ones(len(rows)), standardized])
    return _sigmoid(design @ model["beta"])


def _within_event_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return None
    values = []
    for pos in positive:
        values.extend(1.0 if pos > neg else 0.5 if pos == neg else 0.0 for neg in negative)
    return float(np.mean(values))


def evaluate_event(rows: Sequence[Mapping[str, Any]], probability: np.ndarray) -> dict[str, Any]:
    labels = np.asarray([int(bool(row["label"])) for row in rows])
    if labels.sum() == 0:
        raise ValueError("event has no feasible positive candidate")
    order = np.argsort(-probability, kind="stable")
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    normalized = probability / max(float(probability.sum()), EPSILON)
    correct_mass = float(normalized[labels == 1].sum())
    binary_nll = float(np.mean(
        -(labels * np.log(np.maximum(probability, EPSILON))
          + (1 - labels) * np.log(np.maximum(1.0 - probability, EPSILON)))
    ))
    ranked_labels = labels[order]
    first_positive_rank = int(np.flatnonzero(ranked_labels == 1)[0]) + 1
    selected = rows[best]
    return {
        "event_key": str(rows[0]["event_key"]),
        "clip_id": str(rows[0]["clip_id"]),
        "route_type": str(rows[0]["route_type"]),
        "actor_type": str(rows[0]["actor_type"]),
        "candidate_rows": len(rows),
        "positive_rows": int(labels.sum()),
        "exact_top1": bool(labels[best]),
        "actor_top1": bool(selected["is_correct_actor"]),
        "release_top1": bool(selected["is_release_hit"]),
        "selected_release_frame": int(selected["release_frame"]),
        "selected_actor_key": selected["actor_key"],
        "top1_probability": float(probability[best]),
        "probability_margin": float(probability[best] - probability[second]),
        "choice_nll": float(-math.log(max(correct_mass, EPSILON))),
        "binary_nll": binary_nll,
        "brier": float(np.mean((probability - labels) ** 2)),
        "within_event_auc": _within_event_auc(labels, probability),
        "reciprocal_rank": float(1.0 / first_positive_rank),
    }


def leave_one_event_out(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    *,
    l2: float,
    monotone_penalties: bool = False,
) -> list[dict[str, Any]]:
    output = []
    for held_out in _events(rows):
        train = [row for row in rows if row["event_key"] != held_out]
        test = [row for row in rows if row["event_key"] == held_out]
        if not train or not test or not any(bool(row["label"]) for row in test):
            continue
        center = _time_center(train)
        model = fit_logistic(
            train,
            features,
            alpha_center=center,
            l2=l2,
            monotone_penalties=monotone_penalties,
        )
        output.append(evaluate_event(test, predict(model, test)))
    return output


def _percentile(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def bootstrap_mean_ci(values: Sequence[float], *, samples: int, seed: int) -> list[float] | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or samples <= 0:
        return None
    rng = np.random.default_rng(seed)
    estimates = [
        float(np.mean(array[rng.integers(0, len(array), len(array))]))
        for _ in range(samples)
    ]
    return _percentile(estimates)


def summarize_predictions(predictions: Sequence[Mapping[str, Any]], *, samples: int, seed: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {"events": len(predictions)}
    for name in (
        "exact_top1", "actor_top1", "release_top1", "choice_nll",
        "binary_nll", "brier", "within_event_auc", "reciprocal_rank",
    ):
        values = [float(row[name]) for row in predictions if row.get(name) is not None]
        metrics[name] = {
            "mean": float(np.mean(values)) if values else None,
            "bootstrap_95ci": bootstrap_mean_ci(values, samples=samples, seed=seed + len(name)),
        }
    return metrics


def summarize_by_actor_type(
    predictions: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    return {
        actor_type: summarize_predictions(
            [row for row in predictions if row["actor_type"] == actor_type],
            samples=samples,
            seed=seed + index * 100,
        )
        for index, actor_type in enumerate(("vehicle", "person"))
    }


def uniform_choice_nll(rows: Sequence[Mapping[str, Any]]) -> float:
    values = []
    for event in _events(rows):
        current = [row for row in rows if row["event_key"] == event]
        positives = sum(bool(row["label"]) for row in current)
        if positives:
            values.append(-math.log(positives / len(current)))
    return float(np.mean(values))


def coefficient_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
    *,
    l2: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    keys = _events(rows)
    by_event = {key: [row for row in rows if row["event_key"] == key] for key in keys}
    rng = np.random.default_rng(seed)
    coefficients: list[np.ndarray] = []
    centers = []
    for sample_index in range(samples):
        sampled: list[dict[str, Any]] = []
        for draw_index, source_index in enumerate(rng.integers(0, len(keys), len(keys))):
            source_key = keys[int(source_index)]
            for row in by_event[source_key]:
                copy = dict(row)
                copy["event_key"] = f"bootstrap-{sample_index}-{draw_index}"
                sampled.append(copy)
        try:
            center = _time_center(sampled)
            model = fit_logistic(sampled, features, alpha_center=center, l2=l2)
        except (RuntimeError, ValueError):
            continue
        coefficients.append(np.asarray(model["beta"][1:], dtype=float))
        centers.append(center)
    array = np.asarray(coefficients, dtype=float)
    output = {
        "successful_samples": len(coefficients),
        "alpha_center_median": float(np.median(centers)) if centers else None,
        "alpha_center_95ci": _percentile(centers),
        "standardized_coefficients": {},
    }
    for index, feature in enumerate(features):
        values = array[:, index].tolist() if len(array) else []
        output["standardized_coefficients"][feature] = {
            "median": float(np.median(values)) if values else None,
            "bootstrap_95ci": _percentile(values),
            "fraction_negative": float(np.mean(np.asarray(values) < 0.0)) if values else None,
        }
    return output


def risk_coverage(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(predictions, key=lambda row: float(row["probability_margin"]), reverse=True)
    output = []
    for fraction in (0.25, 0.50, 0.75, 1.00):
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        accepted = ordered[:count]
        output.append({
            "requested_coverage": fraction,
            "accepted": count,
            "support": len(ordered),
            "actual_coverage": count / len(ordered),
            "exact_accuracy": float(np.mean([row["exact_top1"] for row in accepted])),
            "actor_accuracy": float(np.mean([row["actor_top1"] for row in accepted])),
            "minimum_margin": float(accepted[-1]["probability_margin"]),
        })
    return output


def paired_bootstrap_difference(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    first_map = {str(row["event_key"]): float(row[metric]) for row in first}
    second_map = {str(row["event_key"]): float(row[metric]) for row in second}
    keys = sorted(set(first_map).intersection(second_map))
    differences = np.asarray([first_map[key] - second_map[key] for key in keys], dtype=float)
    return {
        "support": len(keys),
        "mean_first_minus_second": float(np.mean(differences)) if len(differences) else None,
        "bootstrap_95ci": bootstrap_mean_ci(differences, samples=samples, seed=seed),
    }


def paired_sign_flip_test(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, Any]:
    """Exact paired randomization test for H1: first has a lower mean metric."""
    first_map = {str(row["event_key"]): float(row[metric]) for row in first}
    second_map = {str(row["event_key"]): float(row[metric]) for row in second}
    keys = sorted(set(first_map).intersection(second_map))
    differences = np.asarray([first_map[key] - second_map[key] for key in keys], dtype=float)
    observed = float(np.mean(differences)) if len(differences) else None
    if not len(differences):
        return {"support": 0, "observed_mean": None, "one_sided_p": None}
    if len(differences) <= 20:
        estimates = []
        for mask in range(1 << len(differences)):
            signs = np.asarray([
                -1.0 if mask & (1 << index) else 1.0
                for index in range(len(differences))
            ])
            estimates.append(float(np.mean(differences * signs)))
    else:
        rng = np.random.default_rng(SEED + 900)
        estimates = [
            float(np.mean(differences * rng.choice((-1.0, 1.0), len(differences))))
            for _ in range(100000)
        ]
    return {
        "support": len(differences),
        "observed_mean": observed,
        "one_sided_p": float(np.mean(np.asarray(estimates) <= observed + 1e-15)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def _ci(value: Sequence[float] | None) -> str:
    return "N/A" if value is None else f"{value[0]:.3f}–{value[1]:.3f}"


def markdown_report(report: Mapping[str, Any]) -> str:
    primary = report["primary"]
    models = primary["models"]
    dtd = models["distance_time_direction"]
    dtd_monotone = models["distance_time_direction_monotone"]
    coefficients = primary["coefficient_bootstrap"]
    lines = [
        "# TrackFlow-inspired -log attribution analysis",
        "",
        "## 結論",
        "",
        f"主分析只有 {primary['events']} 件可用事件（direct vehicle {primary['route_counts'].get('direct_vehicle', 0)}、person→vehicle {primary['route_counts'].get('person_vehicle', 0)}）。這是63片中同時具有 reviewed GT、confirmed sidecar、B0/B1、release hypothesis 與 actor track mapping 的 strict+moderate 子集。",
        "",
        "`-log P(correct association)` 可以計算，但目前樣本只能視為可行性研究，不能用來宣稱通用公式已成立。尤其沒有 reviewed NULL 與真正 negative clips，無法驗證自動拒判的安全性。",
        "",
        "## 模型定義",
        "",
        "- `D`: release point 到 actor release region 的距離；人除以 person height，車除以 vehicle diagonal。",
        "- `alpha=(B0-release_frame)/(B1-B0)`。",
        "- `T=|alpha-alpha0|`，其中 `alpha0` 只由每個 training fold 的 GT 中位數估計。",
        "- `A=(1-cos(theta))/2`，theta 是 actor anchor→release point 與 release forward velocity 的夾角；方向不足時用均勻方向的中性值0.5。",
        "- `P=sigmoid(beta0+betaD*D+betaT*T+betaA*A)`；route cost 為 `-log(P)`。",
        "- `A` 來自 von Mises 圓形分布：其負對數概似與 `1-cos(theta)` 成正比，不直接對角度做不連續線性回歸。",
        "- direct-vehicle route 評估 litter→vehicle；person→vehicle route 評估 litter→person，不把下游 person→vehicle 邊偷算成 direct litter→vehicle。",
        "",
        "## Leave-one-event-out 結果",
        "",
        "| Model | Exact top-1 | Actor top-1 | Release hit | within-event AUC | choice NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("distance", "D only"),
        ("time", "T only"),
        ("direction", "A only"),
        ("distance_time", "D+T"),
        ("distance_time_direction", "D+T+A"),
        ("distance_time_direction_monotone", "D+T+A monotone"),
    ):
        row = models[key]["summary"]
        lines.append(
            f"| {label} | {row['exact_top1']['mean']:.1%} ({_ci(row['exact_top1']['bootstrap_95ci'])}) | "
            f"{row['actor_top1']['mean']:.1%} | {row['release_top1']['mean']:.1%} | "
            f"{row['within_event_auc']['mean']:.3f} | {row['choice_nll']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "`Exact top-1` 要求 actor 與 release interval 同時正確；`Actor top-1` 與 `Release hit` 分開顯示，避免一項成功掩蓋另一項失敗。所有預測均為 held-out event，沒有用該事件估計係數或 alpha0。",
        f"均勻分配候選機率的 choice NLL 基準為 {primary['uniform_choice_nll']:.3f}；D+T+A為 {dtd['summary']['choice_nll']['mean']:.3f}，物理單調約束版為 {dtd_monotone['summary']['choice_nll']['mean']:.3f}。",
        "",
        "### D+T+A 依 release actor 類型拆分",
        "",
        "| Actor | n | Exact top-1 | Actor top-1 | Release hit | within-event AUC |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for actor_type, label in (("vehicle", "direct vehicle"), ("person", "person→vehicle 的 person")):
        row = dtd["summary_by_actor_type"][actor_type]
        lines.append(
            f"| {label} | {row['events']} | {row['exact_top1']['mean']:.1%} | "
            f"{row['actor_top1']['mean']:.1%} | {row['release_top1']['mean']:.1%} | "
            f"{row['within_event_auc']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "Actor top-1 明顯高於 exact top-1，表示主要瓶頸是精確 release frame，而非所有 actor 排序都失效；但 person 子集只有6件，仍不足以估 person 專用係數。",
        "",
        "## D+T+A 係數穩定性",
        "",
        f"完整資料的 alpha0={dtd['full_fit']['alpha_center']:.3f}；event bootstrap alpha0 95% CI {_ci(coefficients['alpha_center_95ci'])}。",
        "",
        "| Feature | standardized beta median | bootstrap 95% CI | beta<0 比例 |",
        "|---|---:|---:|---:|",
    ])
    for feature in ("distance", "time", "direction"):
        row = coefficients["standardized_coefficients"][feature]
        lines.append(
            f"| {feature} | {row['median']:.3f} | {_ci(row['bootstrap_95ci'])} | {row['fraction_negative']:.1%} |"
        )
    lines.extend([
        "",
        "若 beta 的95% CI跨0，代表現有資料無法證明該參數具有穩定且獨立的貢獻；不能因完整資料係數方向看起來合理就宣稱證明成功。",
        "",
        "## 第三參數方向的增益檢驗",
        "",
        f"- Exact accuracy：D+T+A − D+T = {primary['comparisons']['dtd_minus_dt_exact']['mean_first_minus_second']:.3f}，95% CI {_ci(primary['comparisons']['dtd_minus_dt_exact']['bootstrap_95ci'])}。",
        f"- Choice NLL：D+T+A − D+T = {primary['comparisons']['dtd_minus_dt_nll']['mean_first_minus_second']:.3f}，95% CI {_ci(primary['comparisons']['dtd_minus_dt_nll']['bootstrap_95ci'])}；負值代表加入方向較好。",
        f"- 對上述逐事件 NLL 差做exact paired sign-flip test：單尾 p={primary['comparisons']['dtd_minus_dt_nll_sign_flip']['one_sided_p']:.4f}。",
        f"- 單調約束版 D+T+A − D+T NLL = {primary['comparisons']['monotone_dtd_minus_dt_nll']['mean_first_minus_second']:.3f}，95% CI {_ci(primary['comparisons']['monotone_dtd_minus_dt_nll']['bootstrap_95ci'])}，單尾sign-flip p={primary['comparisons']['monotone_dtd_minus_dt_nll_sign_flip']['one_sided_p']:.4f}。",
        "",
        "## Selective coverage（不能視為 NULL 驗證）",
        "",
        "以下只把 margin 較低的正事件暫時拒判，沒有真正 NULL case，因此只能顯示 accuracy/coverage trade-off，不能估計 false attribution on NULL。",
        "",
        "| Coverage | accepted | exact accuracy | actor accuracy | minimum probability margin |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in primary["risk_coverage"]:
        lines.append(
            f"| {row['actual_coverage']:.1%} | {row['accepted']}/{row['support']} | "
            f"{row['exact_accuracy']:.1%} | {row['actor_accuracy']:.1%} | {row['minimum_margin']:.6f} |"
        )
    lines.extend([
        "",
        "## L2 sensitivity",
        "",
        "| L2 | Exact top-1 | choice NLL | betaD | betaT | betaA |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in primary["l2_sensitivity"]:
        lines.append(
            f"| {row['l2']:.2f} | {row['exact_top1']:.1%} | {row['choice_nll']:.3f} | "
            f"{row['beta_distance']:.3f} | {row['beta_time']:.3f} | {row['beta_direction']:.3f} |"
        )
    lines.extend([
        "",
        "## 教授面前的判定規則",
        "",
        "1. 只有 D+T+A 的 held-out NLL 穩定優於 D+T，且方向係數CI不跨0，才接受方向提供獨立證據。",
        "2. 若 betaD、betaT 或 betaA 的 bootstrap CI跨0，該參數保留為待驗證假說，不轉成 production常數。",
        "3. 現有 `dmax` 仍只作 conservative physical gate；`-log P` 只在通過 gate 的候選間排序。",
        "4. 在補上 reviewed NULL、未confirmed真事件與跨攝影機資料前，不設定正式自動開罰 threshold。",
        "",
        f"敏感度分析另外納入 exploratory match 共 {report['sensitivity_all']['events']} 件；它不是主結論，完整數字見 `analysis.json`。",
        "",
    ])
    return "\n".join(lines)


def analyze_scope(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    bootstrap_coefficients: int,
) -> dict[str, Any]:
    usable_events = []
    for event in _events(rows):
        current = [row for row in rows if row["event_key"] == event]
        if any(bool(row["label"]) for row in current):
            usable_events.append(event)
    current_rows = [row for row in rows if row["event_key"] in set(usable_events)]
    route_counts: dict[str, int] = {}
    for event in usable_events:
        route = str(next(row["route_type"] for row in current_rows if row["event_key"] == event))
        route_counts[route] = route_counts.get(route, 0) + 1
    specifications = {
        "distance": (("distance",), False),
        "time": (("time",), False),
        "direction": (("direction",), False),
        "distance_time": (("distance", "time"), False),
        "distance_time_direction": (("distance", "time", "direction"), False),
        "distance_time_monotone": (("distance", "time"), True),
        "distance_time_direction_monotone": (
            ("distance", "time", "direction"), True
        ),
    }
    models: dict[str, Any] = {}
    predictions_by_model: dict[str, list[dict[str, Any]]] = {}
    center = _time_center(current_rows)
    for index, (name, (features, monotone)) in enumerate(specifications.items()):
        predictions = leave_one_event_out(
            current_rows,
            features,
            l2=PRIMARY_L2,
            monotone_penalties=monotone,
        )
        predictions_by_model[name] = predictions
        full = fit_logistic(
            current_rows,
            features,
            alpha_center=center,
            l2=PRIMARY_L2,
            monotone_penalties=monotone,
        )
        models[name] = {
            "features": list(features),
            "monotone_penalties": monotone,
            "summary": summarize_predictions(predictions, samples=samples, seed=SEED + 100 * index),
            "summary_by_actor_type": summarize_by_actor_type(
                predictions, samples=samples, seed=SEED + 1000 + 100 * index
            ),
            "predictions": predictions,
            "full_fit": {
                "alpha_center": center,
                "standardized_beta": full["beta"].tolist(),
                "feature_mean": full["mean"].tolist(),
                "feature_scale": full["scale"].tolist(),
                "objective": full["objective"],
            },
        }
    dt_predictions = predictions_by_model["distance_time"]
    dtd_predictions = predictions_by_model["distance_time_direction"]
    dt_monotone_predictions = predictions_by_model["distance_time_monotone"]
    dtd_monotone_predictions = predictions_by_model[
        "distance_time_direction_monotone"
    ]
    l2_rows = []
    for l2 in L2_SENSITIVITY:
        predictions = leave_one_event_out(
            current_rows,
            ("distance", "time", "direction"),
            l2=l2,
            monotone_penalties=True,
        )
        full = fit_logistic(
            current_rows,
            ("distance", "time", "direction"),
            alpha_center=center,
            l2=l2,
            monotone_penalties=True,
        )
        summary = summarize_predictions(predictions, samples=0, seed=SEED)
        l2_rows.append({
            "l2": l2,
            "exact_top1": summary["exact_top1"]["mean"],
            "choice_nll": summary["choice_nll"]["mean"],
            "beta_distance": float(full["beta"][1]),
            "beta_time": float(full["beta"][2]),
            "beta_direction": float(full["beta"][3]),
        })
    return {
        "events": len(usable_events),
        "candidate_pairs": len(current_rows),
        "positive_pairs": sum(bool(row["label"]) for row in current_rows),
        "route_counts": route_counts,
        "uniform_choice_nll": uniform_choice_nll(current_rows),
        "models": models,
        "coefficient_bootstrap": coefficient_bootstrap(
            current_rows,
            ("distance", "time", "direction"),
            l2=PRIMARY_L2,
            samples=bootstrap_coefficients,
            seed=SEED + 500,
        ),
        "comparisons": {
            "dt_minus_d_exact": paired_bootstrap_difference(
                dt_predictions, predictions_by_model["distance"], "exact_top1",
                samples=samples, seed=SEED + 601,
            ),
            "dt_minus_t_exact": paired_bootstrap_difference(
                dt_predictions, predictions_by_model["time"], "exact_top1",
                samples=samples, seed=SEED + 602,
            ),
            "dt_minus_d_nll": paired_bootstrap_difference(
                dt_predictions, predictions_by_model["distance"], "choice_nll",
                samples=samples, seed=SEED + 603,
            ),
            "dt_minus_t_nll": paired_bootstrap_difference(
                dt_predictions, predictions_by_model["time"], "choice_nll",
                samples=samples, seed=SEED + 604,
            ),
            "dtd_minus_dt_exact": paired_bootstrap_difference(
                dtd_predictions, dt_predictions, "exact_top1",
                samples=samples, seed=SEED + 605,
            ),
            "dtd_minus_dt_nll": paired_bootstrap_difference(
                dtd_predictions, dt_predictions, "choice_nll",
                samples=samples, seed=SEED + 606,
            ),
            "dtd_minus_dt_nll_sign_flip": paired_sign_flip_test(
                dtd_predictions, dt_predictions, "choice_nll"
            ),
            "monotone_dtd_minus_dt_exact": paired_bootstrap_difference(
                dtd_monotone_predictions, dt_monotone_predictions, "exact_top1",
                samples=samples, seed=SEED + 607,
            ),
            "monotone_dtd_minus_dt_nll": paired_bootstrap_difference(
                dtd_monotone_predictions, dt_monotone_predictions, "choice_nll",
                samples=samples, seed=SEED + 608,
            ),
            "monotone_dtd_minus_dt_nll_sign_flip": paired_sign_flip_test(
                dtd_monotone_predictions, dt_monotone_predictions, "choice_nll"
            ),
        },
        "risk_coverage": risk_coverage(dtd_monotone_predictions),
        "l2_sensitivity": l2_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--coefficient-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    validator = _load_validator(Path(__file__).with_name("validate_attribution_formula.py"))
    events = validator.load_jsonl(args.ground_truth / "event_annotations.jsonl")
    actors = validator.load_jsonl(args.ground_truth / "actor_annotations.jsonl")
    project = json.loads((args.ground_truth / "project.json").read_text(encoding="utf-8"))
    metadata = {row["filename"]: row for row in project["videos"]}
    records = validator.candidate_records(args.candidates)
    matches = validator.match_events(events, records, metadata)
    validator.map_manual_actors(matches, actors)
    pair_rows = build_pair_rows(validator, matches)

    primary_rows = [row for row in pair_rows if row["match_tier"] in {"strict", "moderate"}]
    report = {
        "schema": "trackflow-inspired-attribution-log-likelihood/v1",
        "seed": SEED,
        "data": {
            "ground_truth_events": len(events),
            "candidate_records": len(records),
            "matched_events": len(matches),
            "pair_rows_before_scope": len(pair_rows),
        },
        "primary": analyze_scope(
            primary_rows,
            samples=args.bootstrap,
            bootstrap_coefficients=args.coefficient_bootstrap,
        ),
        "sensitivity_all": analyze_scope(
            pair_rows,
            samples=args.bootstrap,
            bootstrap_coefficients=max(200, args.coefficient_bootstrap // 2),
        ),
    }
    safe = _json_safe(report)
    (args.output / "analysis.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output / "candidate_pairs.csv", pair_rows)
    prediction_rows = []
    for model_name, model in report["primary"]["models"].items():
        prediction_rows.extend({"model": model_name, **row} for row in model["predictions"])
    _write_csv(args.output / "loeo_predictions.csv", prediction_rows)
    (args.output / "REPORT.md").write_text(markdown_report(safe), encoding="utf-8")


if __name__ == "__main__":
    main()
