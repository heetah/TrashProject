#!/usr/bin/env python3
"""Blind-ground-truth queue and evaluation CLI for Smart Backtrack."""
import argparse
import json
from pathlib import Path

from pipeline.backtrack.annotations import (
    ANNOTATION_SCHEMA, CANDIDATE_SCHEMA, AnnotationError,
    evaluate_candidates, init_annotations_from_path, load_records,
    validate_records, write_records,
)


def _records_with_schema(path, schema):
    records = [
        record for record in load_records(path)
        if record.get("schema") == schema
    ]
    if not records:
        raise AnnotationError(
            "{} contains no {} records".format(path, schema)
        )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="make unreviewed, prediction-free label queue")
    init.add_argument("input")
    init.add_argument("--output", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("annotations")
    validate.add_argument("--require-reviewed", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--candidates", required=True)
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "init":
        write_records(init_annotations_from_path(args.input), args.output)
        return
    if args.command == "validate":
        result = validate_records(load_records(args.annotations), args.require_reviewed)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["valid"] else 1)
    report = evaluate_candidates(
        _records_with_schema(args.candidates, CANDIDATE_SCHEMA),
        _records_with_schema(args.annotations, ANNOTATION_SCHEMA),
    )
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
