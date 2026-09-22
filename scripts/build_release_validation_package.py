#!/usr/bin/env python3
"""Create a model-blind, double-review release-validation package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from pipeline.backtrack.release_validation import (
    build_negative_queue,
    build_positive_queue,
    camera_suggestions,
    read_jsonl,
    sha256_file,
    validate_built_rows,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--positive-source-root", required=True, type=Path)
    parser.add_argument("--negative-candidate-root", action="append", default=[], type=Path)
    parser.add_argument("--camera-hamming-threshold", type=int, default=18)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    ground_truth = args.ground_truth.resolve()
    output = args.output.resolve()
    event_path = ground_truth / "event_annotations.jsonl"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(event_path)
    positive_queue, legacy = build_positive_queue(events, args.positive_source_root)
    negative_queue = build_negative_queue(args.negative_candidate_root)
    suggestion_inputs = [
        {
            "event_id": row["event_id"],
            "fingerprint": row["source_video"].get("visual_fingerprint"),
        }
        for row in positive_queue
    ]
    camera_rows = camera_suggestions(suggestion_inputs, args.camera_hamming_threshold)
    validation = validate_built_rows(positive_queue, negative_queue, camera_rows)

    write_jsonl(output / "positive_double_review_queue.jsonl", positive_queue)
    write_jsonl(output / "negative_candidate_double_review_queue.jsonl", negative_queue)
    write_jsonl(output / "camera_group_suggestions.jsonl", camera_rows)
    write_jsonl(output / "adjudication_only" / "legacy_unreviewed_seed.jsonl", legacy)

    source_files = sorted(
        path for path in ground_truth.iterdir()
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".xlsx"}
    )
    manifest = {
        "schema": "release-validation-package/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth_root": str(ground_truth),
        "positive_source_root": str(args.positive_source_root.resolve()),
        "negative_candidate_roots": [str(path.resolve()) for path in args.negative_candidate_root],
        "counts": {
            "positive_events": len(positive_queue),
            "negative_candidates": len(negative_queue),
            "camera_suggestions": len(camera_rows),
        },
        "source_annotation_hashes": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_files
        ],
        "evidence_status": {
            "positive_event_reviews_complete": False,
            "reviewed_negatives_available": False,
            "camera_groups_human_verified": False,
            "independent_camera_holdout_available": False,
            "ready_for_enforcement_evaluation": False,
        },
        "package_validation": validation,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
