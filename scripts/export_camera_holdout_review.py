#!/usr/bin/env python3
"""Export a model-blind camera/site worksheet to a deterministic CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from pipeline.backtrack.camera_holdout_review import (
    REVIEW_TABLE_COLUMNS,
    SCHEMA,
    export_review_table,
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
        "--output",
        type=Path,
        required=True,
        help="New CSV table for the single reviewer; existing files are never overwritten.",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Recompute every source SHA-256 before exporting the table.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    try:
        queue = json.loads(args.review.read_text(encoding="utf-8"))
        if not isinstance(queue, dict):
            raise ValueError("worksheet root must be a JSON object")
        source_report = verify_source_files(queue) if args.verify_sources else None
        rows = export_review_table(queue)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REVIEW_TABLE_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, csv.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"camera holdout CSV export failed closed: {exc}") from exc
    print(json.dumps({
        "output": str(args.output),
        "schema": SCHEMA,
        "case_count": len(rows),
        "columns": list(REVIEW_TABLE_COLUMNS),
        "model_outputs_included": False,
        "source_report": source_report,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
