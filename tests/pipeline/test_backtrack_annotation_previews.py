import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "render_backtrack_annotation_previews.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "render_backtrack_annotation_previews", _SCRIPT_PATH
)
preview = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(preview)


def _write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            handle.write("\n")


def _write_test_video(path, frame_count=18, size=(160, 120)):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        size,
    )
    assert writer.isOpened()
    for frame_index in range(frame_count):
        frame = np.full(
            (size[1], size[0], 3),
            (20 + frame_index, 35, 50),
            dtype=np.uint8,
        )
        cv2.putText(
            frame,
            str(frame_index),
            (65, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def _candidate_record(video_path):
    return {
        "schema": preview.SCHEMA_NAME,
        "record_type": "candidate",
        "video": {"input_video": str(video_path), "fps": 10.0},
        "event": {
            "litter_id": 4,
            "birth_frame": 2,
            "confirm_frame": 12,
            "backtrack": {"release_frame": 7},
        },
        "assignment": {
            "route_id": "person:1:null",
            "route_type": "person",
            "release_frame": 7,
            "cost": 0.42,
            "margin_to_second": 0.91,
        },
        "litter_history": [
            {
                "frame_index": frame_index,
                "point_uv": [70 + frame_index, 45 + frame_index],
                "bbox_xyxy": [
                    67 + frame_index,
                    42 + frame_index,
                    73 + frame_index,
                    48 + frame_index,
                ],
            }
            for frame_index in range(2, 13)
        ],
        "release_hypotheses": [
            {
                "uid": "R{}".format(frame_index),
                "frame": frame_index,
                "point_uv": [70 + frame_index, 45 + frame_index],
                "covariance_uv": [[9.0, 0.0], [0.0, 4.0]],
            }
            for frame_index in (6, 7, 8)
        ],
        "candidate_actors": [
            {
                "class_name": "person",
                "track_id": 1,
                "tracklet_uid": "person:1@0",
                "observations": [
                    {
                        "frame_index": frame_index,
                        "box": [15 + frame_index, 15, 55 + frame_index, 105],
                    }
                    for frame_index in range(18)
                ],
            },
            {
                "class_name": "vehicle",
                "track_id": 9,
                "tracklet_uid": "vehicle:9@0",
                "observations": [
                    {
                        "frame_index": frame_index,
                        "bbox_xyxy": [85, 50, 155, 112],
                    }
                    for frame_index in range(18)
                ],
            },
        ],
        "routes": [
            {
                "route_id": "person:1:null",
                "rank": 1,
                "cost": 0.42,
                "selected": True,
            },
            {
                "route_id": "null",
                "rank": 2,
                "cost": 7.0,
                "selected": False,
            },
        ],
    }


def test_renderer_writes_model_blind_event_sheet_and_manifest(tmp_path):
    video_path = tmp_path / "clip.avi"
    _write_test_video(video_path)
    sidecar_path = tmp_path / "clip.backtrack.candidates.jsonl"
    candidate = _candidate_record(video_path)
    _write_jsonl(
        sidecar_path,
        [
            {
                "schema": preview.SCHEMA_NAME,
                "record_type": "run",
                "video": {
                    "input_video": str(video_path),
                    "fps": 10.0,
                    "frame_count": 18,
                },
            },
            candidate,
        ],
    )

    manifest_path, records = preview.render_previews(
        sidecar_path,
        tmp_path / "previews",
        strict=True,
        tile_width=240,
        tile_height=150,
    )

    assert manifest_path.is_file()
    assert len(records) == 1
    manifest = records[0]
    assert manifest["record_type"] == "event"
    assert manifest["status"] == "ok"
    assert manifest["input_video"] == str(video_path.resolve())
    assert manifest["litter_id"] == 4
    assert 8 <= len(manifest["frame_indices"]) <= 12
    assert {2, 7, 12}.issubset(manifest["frame_indices"])
    sheet_path = manifest_path.parent / manifest["sheet_path"]
    sheet = cv2.imread(str(sheet_path))
    assert sheet is not None and sheet.size > 0

    loaded_manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert loaded_manifest == records
    assert "cost" not in manifest and "selected" not in manifest
    assert preview._model_decision_lines(candidate, show=False) == []
    assert any(
        "cost=" in line
        for line in preview._model_decision_lines(candidate, show=True)
    )


def test_overlay_draws_history_release_and_actor_tracklets():
    candidate = _candidate_record("unused.avi")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    overlay = preview._draw_event_overlay(
        frame, candidate, 7, show_release_hypotheses=True
    )

    assert np.any(overlay != frame)
    # Person bbox, vehicle bbox, litter point and release diamond all change
    # their corresponding neighborhoods.
    assert overlay[15, 22].sum() > 0
    assert overlay[50, 85].sum() > 0
    assert overlay[52, 77].sum() > 0


def test_zero_event_run_emits_clip_manifest_without_sheet(tmp_path):
    sidecar_path = tmp_path / "empty.backtrack.candidates.jsonl"
    _write_jsonl(
        sidecar_path,
        [{
            "schema": preview.SCHEMA_NAME,
            "record_type": "run",
            "video": {
                "input_video": "not-opened-for-zero-events.mp4",
                "fps": 10.0,
                "frame_count": 20,
            },
        }],
    )

    manifest_path, records = preview.render_previews(
        sidecar_path, tmp_path / "previews", strict=True
    )

    assert manifest_path.is_file()
    assert records == [{
        "record_type": "clip",
        "status": "no_events",
        "sidecar_path": str(sidecar_path),
        "input_video": "not-opened-for-zero-events.mp4",
        "litter_id": None,
        "sheet_path": None,
        "frame_indices": [],
        "warnings": [],
    }]
    assert not list((tmp_path / "previews" / "sheets").glob("*.jpg"))


def test_strict_mode_rejects_missing_video(tmp_path):
    sidecar_path = tmp_path / "bad.backtrack.candidates.jsonl"
    _write_jsonl(
        sidecar_path,
        [_candidate_record(tmp_path / "missing.avi")],
    )

    with pytest.raises(preview.PreviewError, match="input video not found"):
        preview.render_previews(
            sidecar_path, tmp_path / "previews", strict=True
        )


def test_cli_help_is_available():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "sidecar_input" in result.stdout
    assert "--strict" in result.stdout
    assert "--show-release-hypotheses" in result.stdout
    assert "--show-model-decisions" in result.stdout


def test_zero_litter_id_is_not_rendered_as_unknown():
    assert preview._safe_slug(0, "unknown") == "0"


def test_release_hypothesis_overlay_is_hidden_by_default():
    candidate = _candidate_record("unused.avi")
    candidate["litter_history"] = []
    candidate["candidate_actors"] = []
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    blind = preview._draw_event_overlay(frame, candidate, 7)
    visible = preview._draw_event_overlay(
        frame, candidate, 7, show_release_hypotheses=True
    )

    assert np.array_equal(blind, frame)
    assert np.any(visible != frame)
