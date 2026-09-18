#!/usr/bin/env python3
"""Build a model-blind single-reviewer camera/site holdout worksheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.backtrack.camera_holdout_review import (
    build_from_review_index,
    validate_unreviewed_queue,
    verify_source_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Recompute every source SHA-256 before writing the worksheet.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    try:
        index = json.loads(args.index.read_text(encoding="utf-8"))
        queue = build_from_review_index(index)
        validate_unreviewed_queue(queue)
        source_report = None
        if args.verify_sources:
            source_report = verify_source_files(queue)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(queue, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"camera holdout worksheet failed closed: {exc}") from exc
    print(json.dumps({
        "output": str(args.output),
        "schema": queue["schema"],
        "case_count": len(queue["cases"]),
        "model_outputs_included": queue["model_outputs_included"],
        "source_report": source_report,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
