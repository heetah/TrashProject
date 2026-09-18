#!/usr/bin/env python3
"""Gate a completed camera/site holdout worksheet before evaluation.

The command is intentionally fail-closed: a blank worksheet is reported as
not ready, and a partially completed or malformed worksheet exits non-zero.
It validates provenance only; it never fills labels, infers camera identity,
or runs an accuracy evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pipeline.backtrack.camera_holdout_review import (
    validate_completed_review,
    validate_unreviewed_queue,
    verify_source_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review",
        type=Path,
        required=True,
        help="JSON worksheet produced by build_camera_holdout_review.py.",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Recompute every source SHA-256 before applying the gate.",
    )
    parser.add_argument(
        "--reviewed-camera-id",
        action="append",
        dest="reviewed_camera_ids",
        help="Camera ID from the independently reviewed reference set (repeatable).",
    )
    parser.add_argument(
        "--reviewed-source-recording-id",
        action="append",
        dest="reviewed_source_recording_ids",
        help="Source-recording ID from the independently reviewed reference set (repeatable).",
    )
    parser.add_argument(
        "--reviewed-source-hash",
        action="append",
        dest="reviewed_source_hashes",
        help="SHA-256 from the independently reviewed reference set (repeatable).",
    )
    parser.add_argument(
        "--expected-case-id",
        action="append",
        dest="expected_case_ids",
        help="Expected case ID (repeatable); rejects a changed queue membership.",
    )
    return parser


def _check_expected_case_ids(queue: dict, expected_case_ids: Sequence[str] | None) -> None:
    if expected_case_ids is None:
        return
    expected = {value.strip() for value in expected_case_ids}
    if any(not value for value in expected_case_ids) or len(expected) != len(expected_case_ids):
        raise ValueError("expected_case_id values must be unique and non-empty")
    actual = {
        str(case.get("case_id"))
        for case in queue.get("cases", [])
        if isinstance(case, dict)
    }
    if actual != expected:
        raise ValueError("queue case IDs do not match expected_case_ids")


def _blank_report(
    queue: dict,
    source_report: dict | None,
    expected_case_ids: Sequence[str] | None,
) -> dict:
    _check_expected_case_ids(queue, expected_case_ids)
    report = validate_unreviewed_queue(queue)
    report.update({
        "ready_for_camera_disjoint_eval": False,
        "independence_verified": False,
        "not_ready_reasons": [
            "reviewer has not completed the worksheet; no evaluation may run"
        ],
        "source_report": source_report,
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        queue = json.loads(args.review.read_text(encoding="utf-8"))
        if not isinstance(queue, dict):
            raise ValueError("worksheet root must be a JSON object")
        source_report = verify_source_files(queue) if args.verify_sources else None

        review_states = [
            case.get("review", {}).get("review_state")
            for case in queue.get("cases", [])
            if isinstance(case, dict)
        ]
        if review_states and all(state == "unreviewed" for state in review_states):
            report = _blank_report(queue, source_report, args.expected_case_ids)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 2

        report = validate_completed_review(
            queue,
            reviewed_camera_ids=args.reviewed_camera_ids,
            reviewed_source_recording_ids=args.reviewed_source_recording_ids,
            reviewed_source_hashes=args.reviewed_source_hashes,
            expected_case_ids=args.expected_case_ids,
        )
        report["source_report"] = source_report
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["ready_for_camera_disjoint_eval"] else 2
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"camera holdout review failed closed: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
