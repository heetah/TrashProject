#!/usr/bin/env python3
"""Run the production pipeline once for each usable reviewed clip.

This is an orchestration utility, not a second inference implementation.  Each
clip gets an isolated output directory and a log, so a failed clip does not
erase completed results and the batch can be resumed with ``--resume``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _case_id(filename: str) -> int:
    stem = Path(filename).stem
    return int(stem.split("_")[2])


def _load_usable(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(
        [row for row in rows if row.get("video_usable")],
        key=lambda row: _case_id(row["video_filename"]),
    )


def _read_summary(output_dir: Path) -> dict:
    analysis = next(output_dir.glob("*_analysis.json"), None)
    if analysis is None:
        return {}
    try:
        data = json.loads(analysis.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    summary = dict(data.get("summary") or {})
    detection = dict(data.get("litter_detection") or {})
    summary["confirmed_event_count"] = detection.get("confirmed_event_count")
    summary["analysis_file"] = str(analysis)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=Path("runs/grounding_truth/clip_annotations.jsonl"))
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout-sec", type=float, default=3600.0)
    parser.add_argument(
        "--cases",
        default=None,
        help="Optional comma-separated litter case IDs for a targeted rerun.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or (args.output_root / "batch_manifest.jsonl")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if args.resume and manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[int(row["case"])] = row

    usable = _load_usable(args.clips)
    if args.cases:
        selected = {
            int(value.strip())
            for value in str(args.cases).split(",")
            if value.strip()
        }
        usable = [
            row for row in usable
            if _case_id(row["video_filename"]) in selected
        ]
    total = len(usable)
    with manifest.open("a", encoding="utf-8") as manifest_out:
        for position, clip in enumerate(usable, 1):
            case = _case_id(clip["video_filename"])
            if args.resume and existing.get(case, {}).get("status") == "completed":
                print(f"[{position}/{total}] case {case}: already completed", flush=True)
                continue
            source = args.video_dir / clip["video_filename"].replace("_annotated.mp4", ".mp4")
            if not source.exists():
                row = {"case": case, "status": "missing_input", "source": str(source)}
                manifest_out.write(json.dumps(row) + "\n")
                manifest_out.flush()
                print(f"[{position}/{total}] case {case}: missing {source}", flush=True)
                continue

            output_dir = args.output_root / f"case_{case}"
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "pipeline.log"
            env = os.environ.copy()
            env["OUTPUT_ROOT"] = str(output_dir)
            env["SMART_BACKTRACK_SIDECAR"] = "1"
            started = time.monotonic()
            status = "completed"
            returncode = 0
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
            row = {
                "case": case,
                "source": str(source),
                "output_dir": str(output_dir),
                "log": str(log_path),
                "status": status,
                "returncode": returncode,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "summary": _read_summary(output_dir),
            }
            manifest_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest_out.flush()
            print(
                f"[{position}/{total}] case {case}: {status} "
                f"events={row['summary'].get('confirmed_event_count')} "
                f"elapsed={row['elapsed_sec']:.1f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
