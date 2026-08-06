"""Read production analysis JSON without promoting model scores to accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SUPPORTED_SCHEMA_VERSIONS = {"2.0.0"}


def load_analysis(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("analysis JSON 頂層必須是物件")
    schema_version = str(payload.get("schema_version", ""))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"不支援的 analysis schema: {schema_version or 'missing'}")
    if not isinstance(payload.get("video"), dict):
        raise ValueError("analysis JSON 缺少 video")
    if not isinstance(payload.get("summary"), dict):
        raise ValueError("analysis JSON 缺少 summary")
    if not isinstance(payload.get("events"), list):
        raise ValueError("analysis JSON 缺少 events")
    return payload


def event_key(event: dict[str, Any], index: int) -> str:
    event_type = str(event.get("type") or "event")
    if event_type == "litter" and event.get("id") is not None:
        return f"litter:{event['id']}"
    if event_type == "urinate":
        track_id = event.get("track_id", "unknown")
        time_sec = event.get("time_sec", "unknown")
        return f"urinate:{track_id}:{time_sec}"
    return f"{event_type}:{index}"


def review_units(
    analysis: dict[str, Any],
    reviews: dict[str, dict[str, Any]],
    plate_corrections: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    corrections = plate_corrections or {}
    events = analysis.get("events") or []
    if not events:
        key = "video:no-ai-event"
        return [
            {
                "event_key": key,
                "kind": "no_ai_event",
                "event": None,
                "review": reviews.get(key),
                "plate_correction": None,
            }
        ]

    units = []
    used: set[str] = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        key = event_key(event, index)
        if key in used:
            key = f"{key}:{index}"
        used.add(key)
        units.append(
            {
                "event_key": key,
                "kind": str(event.get("type") or "event"),
                "event": event,
                "review": reviews.get(key),
                "plate_correction": corrections.get(key),
            }
        )
    if units:
        return units
    key = "video:no-ai-event"
    return [
        {
            "event_key": key,
            "kind": "no_ai_event",
            "event": None,
            "review": reviews.get(key),
            "plate_correction": None,
        }
    ]
