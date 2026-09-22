#!/usr/bin/env python3
"""Summarize event-level dynamic-H use and fallback reasons from sidecars."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def _jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def summarize(output_root: Path, manifest: Path) -> dict:
    manifest_rows = list(_jsonl(manifest)) if manifest.exists() else []
    records = []
    for path in output_root.rglob("*_backtrack_candidates.jsonl"):
        records.extend(
            row for row in _jsonl(path) if row.get("record_type") == "candidate"
        )
    reasons, statuses = Counter(), Counter()
    applied = 0
    missing_snapshot = 0
    immutable = 0
    for record in records:
        snapshot = (record.get("resolver_input") or {}).get("homography_snapshot")
        diagnostics = record.get("candidate_diagnostics") or {}
        spatial = diagnostics.get("spatial_calibration") or {}
        if not isinstance(snapshot, dict):
            missing_snapshot += 1
            reasons["missing_event_snapshot"] += 1
            continue
        statuses[str(snapshot.get("status") or "not_available")] += 1
        if snapshot.get("immutable_event_snapshot") is True:
            immutable += 1
        if spatial.get("dynamic_homography_applied") is True:
            applied += 1
        else:
            reasons[str(
                spatial.get("dynamic_homography_reason")
                or snapshot.get("fallback_reason")
                or "unknown"
            )] += 1
    dominant = reasons.most_common(1)[0][0] if reasons else None
    return {
        "schema": "dynamic-homography-validation/v1",
        "completed_clips": sum(row.get("status") == "completed" for row in manifest_rows),
        "failed_clips": sum(row.get("status") != "completed" for row in manifest_rows),
        "sidecar_files": len(list(output_root.rglob("*_backtrack_candidates.jsonl"))),
        "event_snapshots": len(records),
        "applied_events": applied,
        "fallback_events": len(records) - applied,
        "missing_snapshot_events": missing_snapshot,
        "immutable_snapshot_events": immutable,
        "status_counts": dict(statuses),
        "fallback_reason_counts": dict(reasons),
        "dominant_fallback_reason": dominant,
        "interpretation": (
            "Application count proves runtime branch use only. Accuracy impact "
            "requires a paired reviewed comparison on continuous camera footage."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.output_root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
