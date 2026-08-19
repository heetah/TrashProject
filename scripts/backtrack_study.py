#!/usr/bin/env python3
"""CLI for immutable Smart Backtrack research manifests and replays."""
import argparse
import json
from pathlib import Path

from pipeline.backtrack.annotations import load_records
from pipeline.backtrack.sidecar import SCHEMA_NAME, write_jsonl
from pipeline.backtrack.study import (
    build_manifest, evaluate_trial, load_config, replay_candidates,
)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_records(path):
    return [
        record for record in load_records(path)
        if record.get("schema") == SCHEMA_NAME
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest", help="create deterministic grouped split")
    manifest.add_argument("--candidates", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--seed", type=int, default=20260802)
    replay = sub.add_parser("replay", help="replay frozen resolver snapshots")
    replay.add_argument("--candidates", required=True)
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--config", required=True)
    replay.add_argument("--split", choices=("development", "validation", "test"), required=True)
    replay.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate", help="evaluate reviewed labels only")
    evaluate.add_argument("--candidates", required=True)
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", choices=("development", "validation", "test"), required=True)
    evaluate.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.command == "manifest":
        result = build_manifest(_candidate_records(args.candidates), seed=args.seed)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    if args.command == "replay":
        records, summary = replay_candidates(
            _candidate_records(args.candidates), load_config(args.config),
            manifest=_json(args.manifest), split=args.split,
        )
        write_jsonl(records, args.output)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return
    report = evaluate_trial(
        _candidate_records(args.candidates), load_records(args.annotations),
        manifest=_json(args.manifest), split=args.split,
    )
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
