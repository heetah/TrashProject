#!/usr/bin/env python3
"""Import a reviewer-edited CSV table into an immutable JSON worksheet template."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from pipeline.backtrack.camera_holdout_review import (
    REVIEW_TABLE_COLUMNS,
    SCHEMA,
    import_review_table,
    verify_source_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Original model-blind JSON worksheet used as the immutable template.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        required=True,
        help="CSV table exported by export_camera_holdout_review.py and edited by one reviewer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New merged JSON worksheet; existing files are never overwritten.",
    )
    parser.add_argument(
        "--verify-sources",
        action="store_true",
        help="Recompute every source SHA-256 after importing the table.",
    )
    return parser


def _read_rows(path: Path) -> list[dict[str, str]]:
    # utf-8-sig accepts a harmless spreadsheet BOM while retaining strict
    # header/order and provenance checks below.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(REVIEW_TABLE_COLUMNS):
            raise ValueError(
                "review table header must exactly match "
                f"{list(REVIEW_TABLE_COLUMNS)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("review table must contain at least one case row")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    try:
        queue = json.loads(args.template.read_text(encoding="utf-8"))
        if not isinstance(queue, dict):
            raise ValueError("worksheet root must be a JSON object")
        rows = _read_rows(args.table)
        merged = import_review_table(queue, rows)
        source_report = verify_source_files(merged) if args.verify_sources else None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(merged, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, csv.Error, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise SystemExit(f"camera holdout CSV import failed closed: {exc}") from exc
    state_counts: dict[str, int] = {}
    for case in merged["cases"]:
        state = str(case["review"]["review_state"])
        state_counts[state] = state_counts.get(state, 0) + 1
    print(json.dumps({
        "output": str(args.output),
        "schema": SCHEMA,
        "case_count": len(merged["cases"]),
        "review_state_counts": state_counts,
        "model_outputs_included": merged["model_outputs_included"],
        "source_report": source_report,
        "ready_for_camera_disjoint_eval": False,
        "note": "Run validate_camera_holdout_review.py after the reviewer completes all required fields.",
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
