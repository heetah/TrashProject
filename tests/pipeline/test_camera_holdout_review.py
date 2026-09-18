from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_camera_holdout_review import main as export_review_table_main
from scripts.import_camera_holdout_review import main as import_review_table_main
from scripts.validate_camera_holdout_review import main as validate_review_main
from pipeline.backtrack.camera_holdout_review import (
    REVIEW_TABLE_COLUMNS,
    SCHEMA,
    CameraHoldoutValidationError,
    build_from_review_index,
    build_single_reviewer_queue,
    export_review_table,
    import_review_table,
    validate_completed_review,
    validate_unreviewed_queue,
    verify_source_files,
)


def _sha(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()


def _cases(count: int = 3):
    return [
        {
            "case_id": f"litter_case_{index}",
            "video_filename": f"litter_case_{index}.mp4",
            "source_video": f"/data/litter_case_{index}.mp4",
            "source_sha256": _sha(str(index)),
            "model_summary": {"accuracy_status": "not_evaluated"},
            "output_video": f"/tmp/case_{index}.mp4",
        }
        for index in range(1, count + 1)
    ]


def test_builder_is_deterministic_and_strips_model_outputs() -> None:
    first = build_single_reviewer_queue(
        list(reversed(_cases())),
        selection_manifest="/tmp/selection.jsonl",
        run_manifest="/tmp/run.json",
    )
    second = build_single_reviewer_queue(
        _cases(),
        selection_manifest="/tmp/selection.jsonl",
        run_manifest="/tmp/run.json",
    )
    assert first == second
    assert first["schema"] == SCHEMA
    assert first["reviewer_count"] == 1
    assert first["model_outputs_included"] is False
    assert all("model_summary" not in case for case in first["cases"])
    assert all("output_video" not in case for case in first["cases"])
    assert validate_unreviewed_queue(first)["valid"] is True


def test_index_builder_requires_expected_source_index_and_blinds_predictions() -> None:
    index = {
        "schema": "target41-camera-safe-review-index/v1",
        "case_count": 1,
        "selection_manifest": "/tmp/selection.jsonl",
        "run_manifest": "/tmp/run.json",
        "cases": _cases(1),
    }
    queue = build_from_review_index(index)
    assert queue["selection"]["manifest"] == "/tmp/selection.jsonl"
    assert "model_summary" not in queue["cases"][0]


def test_source_files_are_rehashed_before_evaluation(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"immutable source")
    queue = build_single_reviewer_queue([{
        "case_id": "case-1",
        "video_filename": source.name,
        "source_video": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }])
    assert verify_source_files(queue)["verified_source_files"] == 1
    source.write_bytes(b"changed source")
    with pytest.raises(CameraHoldoutValidationError, match="SHA-256 mismatch"):
        verify_source_files(queue)


def test_unreviewed_validator_rejects_prediction_or_nonblank_leak() -> None:
    queue = build_single_reviewer_queue(_cases())
    leaked = copy.deepcopy(queue)
    leaked["cases"][0]["model_score"] = 0.9
    with pytest.raises(CameraHoldoutValidationError, match="non-provenance"):
        validate_unreviewed_queue(leaked)

    nonblank = copy.deepcopy(queue)
    nonblank["cases"][0]["review"]["clip_status"] = "positive"
    with pytest.raises(CameraHoldoutValidationError, match="not blank"):
        validate_unreviewed_queue(nonblank)


def _completed_queue():
    queue = build_single_reviewer_queue(_cases())
    for index, case in enumerate(queue["cases"]):
        review = case["review"]
        review.update({
            "review_state": "reviewed",
            "reviewer_id": "reviewer-1",
            "reviewed_at": "2026-09-15T00:00:00Z",
            "clip_status": "positive" if index == 0 else "negative",
            "video_usable": True,
            "camera_id": f"camera-{index}",
            "site_id": f"site-{index}",
            "session_id": f"session-{index}",
            "source_recording_id": f"recording-{index}",
            "source_sha256_confirmed": True,
            "events": [{"event_id": "human-event"}] if index == 0 else [],
        })
    return queue


def test_completed_review_reports_ready_only_with_known_disjoint_groups() -> None:
    queue = _completed_queue()
    report = validate_completed_review(
        queue,
        reviewed_camera_ids={"reviewed-camera"},
        reviewed_source_recording_ids={"reviewed-recording"},
        reviewed_source_hashes={_sha("reviewed")},
    )
    assert report["valid"] is True
    assert report["independence_verified"] is True
    assert report["ready_for_camera_disjoint_eval"] is True
    assert report["clip_status_counts"] == {
        "ambiguous": 0,
        "negative": 2,
        "positive": 1,
        "unusable": 0,
    }


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("review_state", "unreviewed", "not marked reviewed"),
        ("source_sha256_confirmed", False, "source_sha256_confirmed"),
        ("camera_id", None, "camera_id"),
        ("source_recording_id", None, "source_recording_id"),
    ],
)
def test_completed_review_fails_closed_on_missing_required_provenance(field, value, match) -> None:
    queue = _completed_queue()
    queue["cases"][0]["review"][field] = value
    with pytest.raises(CameraHoldoutValidationError, match=match):
        validate_completed_review(queue)


def test_overlap_or_ambiguous_status_blocks_readiness_without_fabricating_labels() -> None:
    queue = _completed_queue()
    queue["cases"][1]["review"]["clip_status"] = "ambiguous"
    report = validate_completed_review(
        queue,
        reviewed_camera_ids={"camera-1"},
        reviewed_source_recording_ids={"reviewed-recording"},
    )
    assert report["independence_verified"] is False
    assert report["ready_for_camera_disjoint_eval"] is False
    assert report["provenance_overlaps"]["camera_ids"] == ["camera-1"]
    assert any("ambiguous" in reason for reason in report["not_ready_reasons"])


def test_duplicate_case_and_source_hash_fail_closed() -> None:
    with pytest.raises(CameraHoldoutValidationError, match="case_id"):
        build_single_reviewer_queue(_cases() + [_cases(1)[0]])
    duplicate_hash = _cases()
    duplicate_hash[1]["source_sha256"] = duplicate_hash[0]["source_sha256"]
    queue = build_single_reviewer_queue(duplicate_hash)
    for index, case in enumerate(queue["cases"]):
        case["review"].update({
            "review_state": "reviewed", "reviewer_id": "r", "reviewed_at": "t",
            "clip_status": "negative", "video_usable": True,
            "camera_id": f"c{index}", "site_id": f"s{index}",
            "session_id": f"ss{index}", "source_recording_id": f"sr{index}",
            "source_sha256_confirmed": True, "events": [],
        })
    with pytest.raises(CameraHoldoutValidationError, match="source_sha256"):
        validate_completed_review(queue)


def test_validation_cli_fails_closed_for_blank_queue_and_rehashes_sources(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"immutable source")
    queue = build_single_reviewer_queue([{
        "case_id": "case-1",
        "video_filename": source.name,
        "source_video": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }])
    review = tmp_path / "review.json"
    review.write_text(json.dumps(queue), encoding="utf-8")

    assert validate_review_main([
        "--review", str(review), "--verify-sources"
    ]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["review_state"] == "unreviewed"
    assert report["ready_for_camera_disjoint_eval"] is False
    assert report["source_report"] == {"valid": True, "verified_source_files": 1}


def test_validation_cli_accepts_completed_queue_only_with_independent_groups(
    tmp_path: Path, capsys
) -> None:
    review = tmp_path / "completed.json"
    review.write_text(json.dumps(_completed_queue()), encoding="utf-8")
    assert validate_review_main([
        "--review", str(review),
        "--reviewed-camera-id", "reviewed-camera",
        "--reviewed-source-recording-id", "reviewed-recording",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready_for_camera_disjoint_eval"] is True
    assert report["independence_verified"] is True


def test_validation_cli_rejects_changed_blank_queue_membership(tmp_path: Path) -> None:
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(build_single_reviewer_queue(_cases(1))), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="failed closed"):
        validate_review_main(["--review", str(review), "--expected-case-id", "other"])


def test_validation_cli_rejects_non_object_worksheet(tmp_path: Path) -> None:
    review = tmp_path / "review.json"
    review.write_text("[]", encoding="utf-8")
    with pytest.raises(SystemExit, match="failed closed"):
        validate_review_main(["--review", str(review)])


def test_review_table_round_trip_preserves_blank_model_blind_queue() -> None:
    queue = build_single_reviewer_queue(_cases())
    rows = export_review_table(queue)
    assert list(rows[0]) == list(REVIEW_TABLE_COLUMNS)
    assert import_review_table(queue, rows) == queue


def test_review_table_round_trip_preserves_completed_review_values() -> None:
    queue = _completed_queue()
    restored = import_review_table(queue, export_review_table(queue))
    assert restored == queue
    assert restored["cases"][0]["review"]["video_usable"] is True
    assert restored["cases"][0]["review"]["source_sha256_confirmed"] is True
    assert restored["cases"][0]["review"]["events"] == [{"event_id": "human-event"}]


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[0].update({"model_score": "0.9"}), "non-provenance"),
    (lambda rows: rows[0].update({"source_video": "/other/clip.mp4"}), "source_video"),
    (lambda rows: rows[0].update({"source_sha256": _sha("changed")}), "source_sha256"),
])
def test_review_table_rejects_model_columns_and_changed_source_identity(mutation, match) -> None:
    queue = build_single_reviewer_queue(_cases())
    rows = export_review_table(queue)
    mutation(rows)
    with pytest.raises(CameraHoldoutValidationError, match=match):
        import_review_table(queue, rows)


@pytest.mark.parametrize("make_rows,match", [
    (lambda rows: rows[:1] + [dict(rows[0])] + rows[2:], "duplicate case_id"),
    (lambda rows: rows[:-1], "membership mismatch"),
])
def test_review_table_rejects_duplicate_or_missing_case_membership(make_rows, match) -> None:
    queue = build_single_reviewer_queue(_cases())
    rows = make_rows(export_review_table(queue))
    with pytest.raises(CameraHoldoutValidationError, match=match):
        import_review_table(queue, rows)


@pytest.mark.parametrize("field,value,match", [
    ("video_usable", "maybe", "video_usable must be true, false, or blank"),
    ("source_sha256_confirmed", "1", "source_sha256_confirmed must be true, false, or blank"),
    ("events_json", "{}", "events_json must encode a list"),
])
def test_review_table_rejects_malformed_reviewer_values(field, value, match) -> None:
    queue = build_single_reviewer_queue(_cases())
    rows = export_review_table(queue)
    rows[0][field] = value
    with pytest.raises(CameraHoldoutValidationError, match=match):
        import_review_table(queue, rows)


def test_review_table_cli_exports_and_imports_without_overwriting(tmp_path: Path) -> None:
    queue = build_single_reviewer_queue(_cases())
    template = tmp_path / "template.json"
    table = tmp_path / "review.csv"
    restored_path = tmp_path / "restored.json"
    template.write_text(json.dumps(queue), encoding="utf-8")

    assert export_review_table_main([
        "--review", str(template), "--output", str(table)
    ]) == 0
    assert import_review_table_main([
        "--template", str(template), "--table", str(table),
        "--output", str(restored_path),
    ]) == 0
    assert json.loads(restored_path.read_text(encoding="utf-8")) == queue
    with pytest.raises(SystemExit, match="overwrite"):
        export_review_table_main(["--review", str(template), "--output", str(table)])
    with pytest.raises(SystemExit, match="overwrite"):
        import_review_table_main([
            "--template", str(template), "--table", str(table),
            "--output", str(restored_path),
        ])


def test_review_table_cli_rejects_header_with_model_output_column(tmp_path: Path) -> None:
    queue = build_single_reviewer_queue(_cases(1))
    template = tmp_path / "template.json"
    table = tmp_path / "review.csv"
    output = tmp_path / "restored.json"
    template.write_text(json.dumps(queue), encoding="utf-8")
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(REVIEW_TABLE_COLUMNS) + ["model_score"])
        writer.writerow(["case-1"] + [""] * (len(REVIEW_TABLE_COLUMNS) - 1) + ["0.9"])
    with pytest.raises(SystemExit, match="failed closed"):
        import_review_table_main([
            "--template", str(template), "--table", str(table),
            "--output", str(output),
        ])
