import json
from pathlib import Path

import pytest

pytest.importorskip("flask")

from UI.backend.app import create_app
from UI.backend.config import build_config


def make_app(tmp_path: Path):
    allowed = tmp_path / "allowed"
    outputs = tmp_path / "outputs"
    allowed.mkdir()
    outputs.mkdir()
    config = build_config(
        env_path=tmp_path / "missing.env",
        environ={
            "UI_DATABASE_PATH": str(tmp_path / "ui.sqlite3"),
            "UI_UPLOAD_ROOT": str(tmp_path / "uploads"),
            "UI_OUTPUT_ROOT": str(outputs),
            "UI_DISCOVERY_ROOTS": str(outputs),
            "UI_ALLOWED_INPUT_ROOTS": str(allowed),
            "UI_FRONTEND_DIST": str(tmp_path / "dist"),
            "UI_LOG_ROOT": str(tmp_path / "logs"),
            "UI_EXPORT_ROOT": str(tmp_path / "exports"),
            "UI_IMPORT_EXISTING": "0",
            "UI_WORKER_ENABLED": "0",
        },
    )
    return create_app(config, start_worker=False), outputs


def test_health_and_folder_validation(tmp_path):
    app, _outputs = make_app(tmp_path)
    client = app.test_client()

    assert client.get("/api/health").get_json()["status"] == "ok"
    response = client.post("/api/jobs/folder", json={"folder_path": str(tmp_path)})
    assert response.status_code == 400
    assert "允許範圍" in response.get_json()["error"]


def test_review_api_moves_completed_case_to_reviewed(tmp_path):
    app, outputs = make_app(tmp_path)
    output_video = outputs / "clip_annotated.mp4"
    analysis_path = outputs / "clip_annotated_analysis.json"
    output_video.write_bytes(b"video")
    analysis_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "video": {"file": output_video.name, "duration_sec": 4.0},
                "summary": {
                    "litter_event_count": 0,
                    "urinate_event_count": 0,
                    "passed_vehicle_count": 0,
                    "average_litter_confidence": None,
                    "detection_accuracy": None,
                    "accuracy_status": "not_evaluated",
                    "littering_plates": [],
                    "review_required": False,
                },
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    service = app.extensions["ui_service"]
    assert service.discover_existing() == 1
    job_id = service.list_cases()["items"][0]["id"]
    client = app.test_client()

    response = client.put(
        f"/api/jobs/{job_id}/reviews/video:no-ai-event",
        json={"verdict": "accepted", "note": "完整看過影片"},
    )

    assert response.status_code == 200
    payload = client.get("/api/jobs?review_status=reviewed").get_json()
    assert payload["counts"]["reviewed"] == 1
    assert payload["items"][0]["id"] == job_id

    response = client.put(
        f"/api/jobs/{job_id}/reviews/video:no-ai-event",
        json={"verdict": "uncertain"},
    )
    assert response.status_code == 400
    assert "accepted 或 rejected" in response.get_json()["error"]


def test_plate_correction_api_preserves_ai_plate(tmp_path):
    app, outputs = make_app(tmp_path)
    output_video = outputs / "plate_clip_annotated.mp4"
    analysis_path = outputs / "plate_clip_annotated_analysis.json"
    output_video.write_bytes(b"video")
    analysis_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "video": {"file": output_video.name, "duration_sec": 4.0},
                "summary": {
                    "litter_event_count": 1,
                    "urinate_event_count": 0,
                    "passed_vehicle_count": 1,
                    "average_litter_confidence": 0.8,
                    "detection_accuracy": None,
                    "accuracy_status": "not_evaluated",
                    "littering_plates": ["AI-0000"],
                    "review_required": True,
                },
                "events": [
                    {
                        "type": "litter",
                        "id": 7,
                        "confidence": 0.8,
                        "plate": "AI-0000",
                        "plate_status": "recognized",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    service = app.extensions["ui_service"]
    assert service.discover_existing() == 1
    job_id = service.list_cases()["items"][0]["id"]
    client = app.test_client()

    response = client.put(
        f"/api/jobs/{job_id}/events/litter:7/plate",
        json={"corrected_plate": "abc-1234"},
    )

    assert response.status_code == 200
    unit = response.get_json()["case"]["review_units"][0]
    assert unit["event"]["plate"] == "AI-0000"
    assert unit["plate_correction"]["corrected_plate"] == "ABC-1234"
    assert unit["review"] is None

    response = client.delete(f"/api/jobs/{job_id}/events/litter:7/plate")
    assert response.status_code == 200
    assert response.get_json()["case"]["review_units"][0]["plate_correction"] is None


def test_reviewed_export_api_returns_zip_download(tmp_path, monkeypatch):
    app, outputs = make_app(tmp_path)
    archive = outputs / "reviewed_violations.zip"
    archive.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    service = app.extensions["ui_service"]
    monkeypatch.setattr(service, "export_reviewed_violations", lambda: archive)

    response = app.test_client().post("/api/exports/reviewed")

    assert response.status_code == 200
    assert response.mimetype == "application/zip"
    assert "attachment" in response.headers["Content-Disposition"]
    assert "reviewed_violations.zip" in response.headers["Content-Disposition"]
