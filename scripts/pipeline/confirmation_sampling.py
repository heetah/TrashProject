"""Deterministic sampled diagnostics for litter confirmation runs.

This is a read-only analysis helper.  It never infers a confirmation reason
from aggregate counts when the run artifact predates reason instrumentation.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


LEGACY_REASON = "unavailable_legacy_artifact"


def load_case_results_csv(path) -> list[dict]:
    """Load case rows while preserving raw fields for provenance."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def deterministic_sample(
    rows: Sequence[Mapping],
    sample_size: int,
    *,
    key: str = "clip_id",
) -> list[dict]:
    """Return evenly spaced, key-sorted rows; no random seed required."""
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: str(row.get(key, "")),
    )
    size = max(int(sample_size), 0)
    if size >= len(ordered):
        return ordered
    if size == 0 or not ordered:
        return []
    if size == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(i * (len(ordered) - 1) / (size - 1)) for i in range(size)]
    return [ordered[index] for index in indices]


def _json_object(value):
    if not value:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def confirmation_reasons(row: Mapping) -> list[str]:
    """Read explicit reason fields only; return legacy marker otherwise."""
    direct = str(row.get("confirmation_reason", "")).strip()
    if direct:
        return [direct]
    payload = _json_object(row.get("confirmation_evidence"))
    if isinstance(payload, Mapping):
        # ``supported`` is a generic success label.  Prefer the actual
        # confirmation rule so sampled groups explain which gate fired.
        rule = str(payload.get("rule", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        if rule and reason in ("", "supported"):
            return [rule]
        return [reason] if reason else [LEGACY_REASON]
    if isinstance(payload, list):
        reasons = [
            str(item.get("reason", "")).strip()
            for item in payload
            if isinstance(item, Mapping) and str(item.get("reason", "")).strip()
        ]
        return reasons or [LEGACY_REASON]
    return [LEGACY_REASON]


def summarize_sample(rows: Iterable[Mapping], *, source: str | None = None) -> dict:
    """Build bounded summary; counts are diagnostic observations, not accuracy."""
    materialized = [dict(row) for row in rows]
    reasons = Counter()
    stages = Counter()
    for row in materialized:
        reasons.update(confirmation_reasons(row))
        stage = str(row.get("error_stage", "")).strip() or "not_recorded"
        stages[stage] += 1
    return {
        "schema": "confirmation-sample/v1",
        "source": source,
        "population_rows": len(materialized),
        "reason_source_counts": dict(sorted(reasons.items())),
        "error_stage_counts": dict(sorted(stages.items())),
        "sampled_cases": [
            {
                "clip_id": row.get("clip_id"),
                "error_stage": row.get("error_stage") or None,
                "confirmation_reasons": confirmation_reasons(row),
                "geometry_candidate_count": row.get("geometry_candidate_count"),
                "motion_holding_candidate_count": row.get(
                    "motion_holding_candidate_count"
                ),
                "confirmed_event_count": row.get("confirmed_event_count"),
            }
            for row in materialized
        ],
    }


def sample_case_results(path, sample_size: int = 12) -> dict:
    """Load and evenly sample case-results CSV for review."""
    rows = load_case_results_csv(path)
    sampled = deterministic_sample(rows, sample_size)
    report = summarize_sample(sampled, source=str(Path(path)))
    report["population_rows"] = len(rows)
    report["sample_size"] = len(sampled)
    return report


__all__ = [
    "LEGACY_REASON",
    "confirmation_reasons",
    "deterministic_sample",
    "load_case_results_csv",
    "sample_case_results",
    "summarize_sample",
]
