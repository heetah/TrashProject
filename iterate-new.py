import argparse
import csv
import json
import os
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT_DIR = Path("/mnt/8tb_hdd/under115a/resources/urinate_long_test")
DEFAULT_OUTPUT_DIR = Path("/mnt/8tb_hdd/under115a/output/urinate_long_test_iterate0625")
DEFAULT_MAIN_PY = Path("/home/se_copilot/trashProject/scripts-old-test/main.py")
DEFAULT_CLASSES = ("litter",)
DEFAULT_EXTENSIONS = (".mp4", ".avi")


CSV_FIELDS = (
    "case_class",
    "file_path",
    "video_name",
    "file_name",
    "suffix",
    "returncode",
    "ok",
    "rtdetr_enabled",
    "stgcn_pose_enabled",
    "raw_litter_detected",
    "confirm_litter",
    "backtracked_thrower",
    "has_person",
    "has_keypoints",
    "has_urinate",
    "has_littering",
    "raw_litter_candidates",
    "filtered_litter_candidates",
    "confirmed_litter_ids",
    "confirmed_litter_frame_hits",
    "confirmed_litter_thrower_ids",
    "confirmed_litter_thrower_frame_hits",
    "backtracked_thrower_ids",
    "backtracked_thrower_frame_hits",
    "urinate_count",
    "littering_count",
    "keypoint_matches",
    "person_frame_hits",
    "person_detections",
    "stgcn_predict_calls",
    "stgcn_alerts",
    "stgcn_alert_frame_hits",
    "output_video",
    "summary_json",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--main-py", default=str(DEFAULT_MAIN_PY))
    parser.add_argument("--python-bin", default="python")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="optional filter: only process clips whose inferred class (sub-folder name) is in this list; default processes all",
    )
    parser.add_argument(
        "--case-class",
        default=None,
        help="force this ground-truth class for every clip; default infers it from the clip's folder (sub-dir name, or root folder name for clips directly under --root-dir)",
    )
    parser.add_argument("--extensions", nargs="*", default=list(DEFAULT_EXTENSIONS))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument(
        "--rtdetr",
        choices=("on", "off"),
        default="on",
        help="off: disable RTDETR litter detection and plate OCR (sets RTDETR_ENABLED=0); scripts-old-stgcn uses --enable-rtdetr flag instead",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="max videos to process; useful for smoke tests",
    )
    parser.add_argument(
        "--table-limit",
        type=int,
        default=120,
        help="max case rows printed in terminal table; 0 prints all rows",
    )
    parser.add_argument(
        "--main-arg",
        action="append",
        default=[],
        help="extra argument passed to scripts-old-test/main.py; repeat for multiple args",
    )
    parser.add_argument(
        "--per-clip-timeout",
        type=float,
        default=None,
        help="kill main.py and mark the clip failed (returncode 124) if it runs longer than N seconds; guards against a single deadlock stalling the whole batch",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip clips whose summary json already exists and parses (crash recovery / incremental); does NOT re-run them",
    )
    parser.add_argument(
        "--capture-logs",
        action="store_true",
        help="redirect each clip's main.py stdout/stderr to summaries/<name>.log instead of the terminal",
    )
    return parser.parse_args()


def is_generated_clip(path):
    return "_clips" in path.stem


def infer_case_class(root_dir, file_path, override=None):
    # ��冽�� ground-truth class嚗�--case-class 憿臬��閬�撖怠�芸��嚗���血����� clip �����典��鞈����憭曉��
    # 嚗�撌Ｙ�������� root 銝�蝚砌��撅文����桅��嚗�clip ��湔�交�曉�� root ������ root 鞈����憭曉��嚗����
    if override:
        return override
    try:
        rel = file_path.relative_to(root_dir)
    except ValueError:
        return root_dir.name or "unknown"
    if len(rel.parts) > 1:
        return rel.parts[0]
    return root_dir.name or "unknown"


def iter_video_files(root_dir, classes, extensions, case_class):
    # ���餈湔�����������敶梁��嚗�class ��梯�����憭暹�冽�瘀��--classes ��交�����靘����雿���粹��瞈曄�賢����柴��
    extension_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    class_filter = {str(c) for c in classes} if classes else None
    for file_path in sorted(root_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in extension_set:
            continue
        if is_generated_clip(file_path):
            continue
        cls = infer_case_class(root_dir, file_path, override=case_class)
        if class_filter is not None and cls not in class_filter:
            continue
        yield cls, file_path


def format_duration(seconds):
    seconds = int(round(max(0.0, float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def write_manifest(output_csv, args, root_dir, main_py, total):
    # 蝝�������甈� iterate �����舫����曇��閮�嚗�argv���敶梢�輻����������啣��霈���詻��git SHA��������菜��璅����
    relevant_env = {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("ACTION_", "LITTER_", "PLATE_", "YOLO_")) or key.endswith("_DEVICE")
    }
    git_sha = ""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(main_py).parent),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0:
            git_sha = proc.stdout.strip()
    except Exception:
        pass

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "root_dir": str(root_dir),
        "main_py": str(main_py),
        "output_csv": str(output_csv),
        "total_clips": total,
        "skip_existing": bool(args.skip_existing),
        "capture_logs": bool(args.capture_logs),
        "per_clip_timeout": args.per_clip_timeout,
        "main_args": list(args.main_arg),
        "relevant_env": relevant_env,
        "git_sha": git_sha,
    }
    manifest_path = output_csv.with_name(output_csv.stem + "_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest_path


def summary_path_for(output_dir, root_dir, cls, file_path):
    # main.py ��芸��������嚗�{output_dir}/{stem}_annotated_summary.json
    return output_dir / f"{file_path.stem}_annotated_summary.json"


def build_command(args, root_dir, main_py, cls, file_path):
    # main.py ��芣�亙��敶梁��頝臬��嚗�頛詨�箇�桅����� subprocess cwd= ��批�塚��summary JSON ��芸�����������
    cmd = [
        args.python_bin,
        str(main_py),
        str(file_path),
    ]
    if args.rtdetr == "on" and "scripts-old-stgcn" in str(main_py):
        cmd.append("--enable-rtdetr")
    cmd.extend(args.main_arg)
    return cmd


def build_subprocess_env(args):
    # 撠� --rtdetr off 頧���� RTDETR_ENABLED=0 env var嚗�scripts-old-test/main.py 霈����甇文�潦��
    if args.rtdetr == "off":
        return {**os.environ, "RTDETR_ENABLED": "0"}
    return None  # None ��� subprocess 蝜潭�輻�園�脩�� env嚗����閮剖����剁��


def bool_text(value):
    return "1" if bool(value) else "0"


def yes_no(value):
    return "YES" if str(value) == "1" or value is True else "NO"


def int_from_summary(summary, *keys):
    for key in keys:
        value = summary.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0


def output_path_from_summary(output_dir, summary):
    output_video = str(summary.get("output_video", "") or "")
    if not output_video:
        return ""
    output_path = Path(output_video)
    if output_path.is_absolute():
        return str(output_path)
    return str(output_dir / output_path)


def build_row(cls, file_path, returncode, summary_json, summary, output_dir):
    summary = dict(summary or {})
    raw_litter_candidates = int_from_summary(summary, "raw_litter_candidates")
    filtered_litter_candidates = int_from_summary(summary, "filtered_litter_candidates")
    confirmed_litter_ids = int_from_summary(summary, "confirmed_litter_ids")
    confirmed_litter_frame_hits = int_from_summary(summary, "confirmed_litter_frame_hits")
    confirmed_litter_thrower_ids = int_from_summary(summary, "confirmed_litter_thrower_ids")
    confirmed_litter_thrower_frame_hits = int_from_summary(summary, "confirmed_litter_thrower_frame_hits")
    backtracked_thrower_ids = int_from_summary(summary, "backtracked_thrower_ids")
    backtracked_thrower_frame_hits = int_from_summary(summary, "backtracked_thrower_frame_hits")
    confirm_litter = confirmed_litter_ids > 0 or confirmed_litter_frame_hits > 0
    backtracked_thrower = (
        bool(summary.get("has_backtracked_thrower", False)) or
        backtracked_thrower_ids > 0 or
        backtracked_thrower_frame_hits > 0
    )
    return {
        "case_class": cls,
        "file_path": str(file_path),
        "video_name": file_path.name,
        "file_name": file_path.name,
        "suffix": file_path.suffix.lower(),
        "returncode": int(returncode),
        "ok": bool_text(returncode == 0),
        "rtdetr_enabled": bool_text(summary.get("rtdetr_enabled", False)),
        "stgcn_pose_enabled": bool_text(summary.get("stgcn_pose_enabled", False)),
        "raw_litter_detected": bool_text(raw_litter_candidates > 0),
        "confirm_litter": bool_text(confirm_litter),
        "backtracked_thrower": bool_text(backtracked_thrower),
        "has_person": bool_text(summary.get("has_person", False)),
        "has_keypoints": bool_text(summary.get("has_keypoints", False)),
        "has_urinate": bool_text(summary.get("has_urinate", False)),
        "has_littering": bool_text(summary.get("has_littering", False)),
        "raw_litter_candidates": raw_litter_candidates,
        "filtered_litter_candidates": filtered_litter_candidates,
        "confirmed_litter_ids": confirmed_litter_ids,
        "confirmed_litter_frame_hits": confirmed_litter_frame_hits,
        "confirmed_litter_thrower_ids": confirmed_litter_thrower_ids,
        "confirmed_litter_thrower_frame_hits": confirmed_litter_thrower_frame_hits,
        "backtracked_thrower_ids": backtracked_thrower_ids,
        "backtracked_thrower_frame_hits": backtracked_thrower_frame_hits,
        "urinate_count": int_from_summary(summary, "stgcn_urination", "stgcn_pred_urinate"),
        "littering_count": int_from_summary(summary, "stgcn_littering"),
        "keypoint_matches": int_from_summary(summary, "stgcn_pose_matches"),
        "person_frame_hits": int_from_summary(summary, "person_frame_hits"),
        "person_detections": int_from_summary(summary, "person_detections"),
        "stgcn_predict_calls": int_from_summary(summary, "stgcn_predict_calls"),
        "stgcn_alerts": int_from_summary(summary, "stgcn_alerts"),
        "stgcn_alert_frame_hits": int_from_summary(summary, "stgcn_alert_frame_hits"),
        "output_video": output_path_from_summary(output_dir, summary),
        "summary_json": str(summary_json),
    }


def write_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def display_width(value):
    width = 0
    for char in str(value):
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def clip_display(value, max_width):
    text = str(value)
    if max_width is None or display_width(text) <= max_width:
        return text
    if max_width <= 3:
        return "." * max_width
    width = 0
    output = []
    for char in text:
        char_width = 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
        if width + char_width > max_width - 3:
            break
        output.append(char)
        width += char_width
    return "".join(output) + "..."


def pad_display(value, width):
    text = str(value)
    return text + " " * max(width - display_width(text), 0)


def print_table(title, headers, rows, max_widths=None):
    print("")
    print(title)
    if not rows:
        print("(no rows)")
        return

    max_widths = dict(max_widths or {})
    clipped_rows = []
    for row in rows:
        clipped_rows.append([
            clip_display(cell, max_widths.get(headers[idx]))
            for idx, cell in enumerate(row)
        ])

    widths = []
    for idx, header in enumerate(headers):
        column_values = [header] + [row[idx] for row in clipped_rows]
        widths.append(max(display_width(value) for value in column_values))

    header_line = " | ".join(pad_display(header, widths[idx]) for idx, header in enumerate(headers))
    separator = "-+-".join("-" * width for width in widths)
    print(header_line)
    print(separator)
    for row in clipped_rows:
        print(" | ".join(pad_display(cell, widths[idx]) for idx, cell in enumerate(row)))


def count_yes(rows, key):
    return sum(1 for row in rows if str(row.get(key, "0")) == "1")


def build_class_rows(rows, class_order):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_class"]].append(row)

    output = []
    for cls in class_order:
        class_rows = grouped.get(cls, [])
        if not class_rows:
            continue
        output.append([
            cls,
            len(class_rows),
            count_yes(class_rows, "ok"),
            count_yes(class_rows, "raw_litter_detected"),
            count_yes(class_rows, "confirm_litter"),
            count_yes(class_rows, "backtracked_thrower"),
            count_yes(class_rows, "has_person"),
            count_yes(class_rows, "has_keypoints"),
            count_yes(class_rows, "has_urinate"),
        ])
    return output


def print_final_tables(rows, output_csv, class_order, table_limit):
    total = len(rows)
    ok_count = count_yes(rows, "ok")
    overview_rows = [
        ["Cases", total],
        ["OK", ok_count],
        ["Failed", total - ok_count],
        ["Raw litter detected", count_yes(rows, "raw_litter_detected")],
        ["Confirm litter", count_yes(rows, "confirm_litter")],
        ["Backtracked thrower", count_yes(rows, "backtracked_thrower")],
        ["Person detected", count_yes(rows, "has_person")],
        ["Keypoints detected", count_yes(rows, "has_keypoints")],
        ["Urinate detected", count_yes(rows, "has_urinate")],
        ["CSV", output_csv],
    ]
    print_table("Overview", ["Metric", "Value"], overview_rows, {"Value": 90})

    print_table(
        "By class",
        ["Class", "Cases", "OK", "Raw", "Confirm", "Thrower", "Person", "Keypoints", "Urinate"],
        build_class_rows(rows, class_order),
    )

    failures = [row for row in rows if str(row.get("ok", "0")) != "1"]
    if failures:
        print_table(
            "Failures",
            ["Class", "File", "returncode"],
            [[row["case_class"], row["file_name"], row["returncode"]] for row in failures],
            {"File": 56},
        )

    visible_rows = rows if table_limit == 0 else rows[:max(int(table_limit), 0)]
    case_rows = [
        [
            row["case_class"],
            row["file_name"],
            yes_no(row["ok"]),
            yes_no(row["raw_litter_detected"]),
            yes_no(row["confirm_litter"]),
            yes_no(row["backtracked_thrower"]),
            yes_no(row["has_person"]),
            yes_no(row["has_keypoints"]),
            yes_no(row["has_urinate"]),
            row["output_video"],
        ]
        for row in visible_rows
    ]
    print_table(
        "Case results",
        ["Class", "File", "OK", "Raw", "Confirm", "Thrower", "Person", "Keypoints", "Urinate", "Output"],
        case_rows,
        {"File": 56, "Output": 72},
    )
    if table_limit != 0 and total > len(visible_rows):
        print(f"\nCase table truncated: showing {len(visible_rows)}/{total}. Full detail in CSV.")


def main():
    args = parse_args()
    # ��券�刻圾������蝯�撠�頝臬��嚗�subprocess 隞� cwd=output_dir ��瑁��嚗���詨����� main_py / 敶梁��頝臬�����閫������航炊���
    root_dir = Path(args.root_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    main_py = Path(args.main_py).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else output_dir / f"iterate_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    files = list(iter_video_files(root_dir, args.classes, args.extensions, args.case_class))
    if args.limit is not None:
        files = files[: max(int(args.limit), 0)]
    total = len(files)

    manifest_path = write_manifest(output_csv, args, root_dir, main_py, total)
    print(f"manifest: {manifest_path}")
    print(f"queued {total} clip(s) from {root_dir}")

    rows = []
    run_start = time.monotonic()
    durations = []
    for idx, (cls, file_path) in enumerate(files, start=1):
        # main.py ��芸�������� summary JSON ��� cwd嚗�= output_dir嚗�銝�
        summary_json = summary_path_for(output_dir, root_dir, cls, file_path)

        # ��琿��蝥�頝�嚗�summary 撌脣����其����航圾��� ��� ��湔�交窒��剁��銝����頝����
        if args.skip_existing and summary_json.exists():
            try:
                summary = json.loads(summary_json.read_text(encoding="utf-8"))
                print(f"[{idx}/{total}] skip existing: [{cls}] {file_path.name}")
                row = build_row(cls, file_path, 0, summary_json, summary, output_dir)
                rows.append(row)
                write_csv(output_csv, rows)
                continue
            except json.JSONDecodeError:
                pass  # 憯������� summary ��� ���頝�

        summary_json.unlink(missing_ok=True)
        cmd = build_command(args, root_dir, main_py, cls, file_path)

        eta = ""
        if durations:
            avg = sum(durations) / len(durations)
            eta = f" (eta ~{format_duration(avg * (total - idx + 1))})"
        print(f"[{idx}/{total}] run: [{cls}] {file_path.name}{eta}")

        clip_start = time.monotonic()
        returncode = 0
        log_handle = None
        log_path = output_dir / f"{file_path.stem}_annotated_summary.log"
        subprocess_env = build_subprocess_env(args)
        try:
            if args.capture_logs:
                log_handle = log_path.open("w", encoding="utf-8")
                result = subprocess.run(
                    cmd, check=False, cwd=str(output_dir),
                    stdout=log_handle, stderr=subprocess.STDOUT,
                    timeout=args.per_clip_timeout,
                    env=subprocess_env,
                )
            else:
                result = subprocess.run(
                    cmd, check=False, cwd=str(output_dir),
                    timeout=args.per_clip_timeout,
                    env=subprocess_env,
                )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            print(f"[{idx}/{total}] TIMEOUT after {args.per_clip_timeout}s: {file_path.name}")
        finally:
            if log_handle is not None:
                log_handle.close()
        clip_elapsed = time.monotonic() - clip_start
        durations.append(clip_elapsed)

        summary = {}
        if summary_json.exists():
            try:
                summary = json.loads(summary_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"summary json parse failed: {summary_json}: {exc}")
        elif returncode == 0:
            print(f"summary json missing: {summary_json}")

        row = build_row(cls, file_path, returncode, summary_json, summary, output_dir)
        rows.append(row)
        write_csv(output_csv, rows)
        print(
            "case result: "
            f"ok={row['ok']} raw={row['raw_litter_detected']} confirm={row['confirm_litter']} "
            f"thrower={row['backtracked_thrower']} person={row['has_person']} "
            f"keypoints={row['has_keypoints']} urinate={row['has_urinate']} "
            f"| {format_duration(clip_elapsed)} | elapsed {format_duration(time.monotonic() - run_start)}"
        )

    write_csv(output_csv, rows)
    class_order = sorted({row["case_class"] for row in rows})
    print_final_tables(rows, output_csv, class_order, args.table_limit)
    print(f"\nTotal wall: {format_duration(time.monotonic() - run_start)} for {total} clip(s).")


if __name__ == "__main__":
    main()
