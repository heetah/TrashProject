#!/usr/bin/env python3
"""Re-run every case in a JSONL manifest with isolated, resumable outputs.

This is orchestration only: inference remains ``scripts/main.py``.  A case is
considered complete only when its subprocess exits zero and an analysis JSON
exists.  The manifest keeps the raw per-case status; downstream reports must
still use reviewed labels for accuracy claims.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_SOURCE_ROOT = Path(
    "/home/under115a/under115a/under115a/backtrack_testcase"
)
DEFAULT_SOURCE_MANIFEST = Path(
    "artifacts/backtrack_testcase_production_20260919/batch_manifest.jsonl"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _analysis(case_output: Path) -> tuple[dict, Path | None]:
    candidates = sorted(case_output.glob("*_annotated_analysis.json"))
    if not candidates:
        return {}, None
    try:
        return json.loads(candidates[0].read_text(encoding="utf-8")), candidates[0]
    except (OSError, json.JSONDecodeError):
        return {}, candidates[0]


def _record_summary(analysis: dict, analysis_path: Path | None) -> dict:
    detection = dict(analysis.get("litter_detection") or {})
    summary = dict(analysis.get("summary") or {})
    summary.update(
        {
            "confirmed_event_count": detection.get("confirmed_event_count"),
            "litter_detection": detection,
            "analysis_file": str(analysis_path) if analysis_path else None,
        }
    )
    return summary


def _latest_records(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    latest = {}
    for row in _read_jsonl(path):
        latest[int(row["case_index"])] = row
    return latest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser


def run(args: argparse.Namespace) -> int:
    source_root = args.source_root.expanduser().resolve()
    source_manifest = args.source_manifest
    if not source_manifest.is_absolute():
        source_manifest = Path.cwd() / source_manifest
    source_manifest = source_manifest.resolve()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    output_root = output_root.resolve()

    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    if not source_manifest.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {source_manifest}")

    rows = sorted(_read_jsonl(source_manifest), key=lambda row: int(row["case_index"]))
    if args.max_cases is not None:
        rows = rows[: max(int(args.max_cases), 0)]
    if not rows:
        raise ValueError("source manifest contains no cases")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "batch_manifest.jsonl"
    if manifest_path.exists() and not args.resume:
        raise FileExistsError(
            f"{manifest_path} exists; pass --resume to continue or choose a new output root"
        )
    previous = _latest_records(manifest_path)

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for position, row in enumerate(rows, 1):
            case_index = int(row["case_index"])
            relative = Path(row["source_relative"])
            source = source_root / relative
            case_output = output_root / relative.parent / relative.stem
            case_output.mkdir(parents=True, exist_ok=True)
            log_path = case_output / "pipeline.log"
            analysis_path = case_output / f"{source.stem}_annotated_analysis.json"

            old = previous.get(case_index)
            if (
                args.resume
                and old is not None
                and old.get("status") == "completed"
                and analysis_path.is_file()
                and analysis_path.stat().st_size > 0
            ):
                print(
                    f"[{position}/{len(rows)}] case={case_index} resume-skip",
                    flush=True,
                )
                continue

            started = time.monotonic()
            status = "completed"
            returncode = 0
            if not source.is_file():
                status = "missing_input"
                returncode = -2
            else:
                env = os.environ.copy()
                env.update(
                    {
                        "OUTPUT_ROOT": str(case_output),
                        "SMART_BACKTRACK": "1",
                        "SMART_BACKTRACK_SIDECAR": "1",
                        "PYTHONUNBUFFERED": "1",
                    }
                )
                try:
                    with log_path.open("w", encoding="utf-8") as log:
                        completed = subprocess.run(
                            [sys.executable, "scripts/main.py", str(source)],
                            cwd=Path(__file__).resolve().parents[1],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=float(args.timeout_sec),
                            check=False,
                        )
                    returncode = int(completed.returncode)
                    if returncode != 0:
                        status = "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    returncode = -9

            analysis, actual_analysis_path = _analysis(case_output)
            summary = _record_summary(analysis, actual_analysis_path)
            record = {
                "record_type": "case",
                "case_index": case_index,
                "category": row.get("category"),
                "source": str(source),
                "source_relative": str(relative),
                "source_sha256_expected": row.get("source_sha256"),
                "output_dir": str(case_output),
                "log": str(log_path),
                "status": status,
                "returncode": returncode,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "summary": summary,
            }
            manifest.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            manifest.flush()
            previous[case_index] = record
            print(
                f"[{position}/{len(rows)}] case={case_index} "
                f"category={row.get('category')} status={status} "
                f"events={summary.get('confirmed_event_count')} "
                f"elapsed={record['elapsed_sec']:.1f}s",
                flush=True,
            )

    statuses = Counter(row.get("status") for row in previous.values())
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "requested_cases": len(rows),
                "latest_records": len(previous),
                "statuses": dict(sorted(statuses.items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    expected = {int(row["case_index"]) for row in rows}
    if expected != set(previous) or any(
        previous[index].get("status") != "completed" for index in expected
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run(_parser().parse_args()))
