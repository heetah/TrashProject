"""Flask API for the React review console."""

from __future__ import annotations

import atexit
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from .config import UIConfig, build_config
from .database import Database
from .runner import PipelineWorker
from .service import ReviewService


def create_app(config: UIConfig | None = None, *, start_worker: bool = False) -> Flask:
    settings = config or build_config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length

    database = Database(settings.database_path)
    database.initialize()
    service = ReviewService(settings, database)
    if settings.import_existing:
        service.discover_existing()

    worker = PipelineWorker(settings, database)
    if start_worker and settings.worker_enabled:
        worker.start()
        atexit.register(worker.stop)

    app.extensions["ui_settings"] = settings
    app.extensions["ui_database"] = database
    app.extensions["ui_service"] = service
    app.extensions["ui_worker"] = worker

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "worker_enabled": settings.worker_enabled})

    @app.get("/api/config")
    def client_config():
        return jsonify(
            {
                "poll_seconds": settings.poll_seconds,
                "max_upload_mb": settings.max_upload_mb,
                "allowed_input_roots": [str(path) for path in settings.allowed_input_roots],
                "folder_recursive": settings.folder_recursive,
                "confidence_notice": "模型 confidence 不是人工驗證後的 accuracy",
            }
        )

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify(service.list_cases(request.args.get("review_status")))

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        return jsonify(service.get_case(job_id))

    @app.post("/api/jobs/upload")
    def upload_jobs():
        files = request.files.getlist("videos")
        if not files:
            raise ValueError("請選擇至少一支影片")
        return jsonify({"jobs": [service.enqueue_upload(item) for item in files]}), 201

    @app.post("/api/jobs/folder")
    def folder_jobs():
        payload = request.get_json(silent=True) or {}
        folder_path = str(payload.get("folder_path") or "").strip()
        if not folder_path:
            raise ValueError("folder_path 不可為空")
        recursive = payload.get("recursive")
        return jsonify({"jobs": service.enqueue_folder(folder_path, recursive=recursive)}), 201

    @app.post("/api/jobs/discover")
    def discover_jobs():
        return jsonify({"imported": service.discover_existing()})

    @app.post("/api/jobs/<job_id>/retry")
    def retry_job(job_id: str):
        return jsonify({"job": service.retry(job_id)})

    @app.post("/api/exports/reviewed")
    def export_reviewed_violations():
        archive = service.export_reviewed_violations()
        return send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name=archive.name,
            conditional=True,
        )

    @app.put("/api/jobs/<job_id>/reviews/<path:event_key>")
    def save_review(job_id: str, event_key: str):
        payload = request.get_json(silent=True) or {}
        review = service.save_review(
            job_id,
            event_key,
            verdict=str(payload.get("verdict") or ""),
            note=str(payload.get("note") or ""),
            reviewer=str(payload.get("reviewer") or ""),
        )
        return jsonify({"review": review, "case": service.get_case(job_id)})

    @app.put("/api/jobs/<job_id>/events/<path:event_key>/plate")
    def save_plate_correction(job_id: str, event_key: str):
        payload = request.get_json(silent=True) or {}
        correction = service.save_plate_correction(
            job_id,
            event_key,
            corrected_plate=str(payload.get("corrected_plate") or ""),
        )
        return jsonify(
            {"plate_correction": correction, "case": service.get_case(job_id)}
        )

    @app.delete("/api/jobs/<job_id>/events/<path:event_key>/plate")
    def delete_plate_correction(job_id: str, event_key: str):
        deleted = service.delete_plate_correction(job_id, event_key)
        return jsonify({"deleted": deleted, "case": service.get_case(job_id)})

    @app.get("/api/jobs/<job_id>/video")
    def job_video(job_id: str):
        job = database.get_job(job_id)
        if job is None:
            raise KeyError("找不到工作")
        output_video = job.get("output_video")
        if not output_video or not Path(output_video).is_file():
            raise KeyError("找不到輸出影片")
        return send_file(output_video, mimetype="video/mp4", conditional=True)

    @app.errorhandler(ValueError)
    def bad_request(exc: ValueError):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(KeyError)
    def not_found(exc: KeyError):
        return jsonify({"error": str(exc.args[0] if exc.args else exc)}), 404

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc: RequestEntityTooLarge):
        return jsonify({"error": f"上傳超過 {settings.max_upload_mb} MB 限制"}), 413

    @app.get("/")
    def frontend_index():
        index = settings.frontend_dist / "index.html"
        if index.is_file():
            return send_file(index)
        return jsonify(
            {
                "message": "React 尚未建置；開發時請啟動 Vite，部署前執行 npm run build。",
                "api": "/api/health",
            }
        )

    @app.get("/<path:asset_path>")
    def frontend_asset(asset_path: str):
        candidate = settings.frontend_dist / asset_path
        if candidate.is_file():
            return send_from_directory(settings.frontend_dist, asset_path)
        index = settings.frontend_dist / "index.html"
        if index.is_file():
            return send_file(index)
        return jsonify({"error": "React build 不存在"}), 404

    return app


def main() -> None:
    settings = build_config()
    app = create_app(settings, start_worker=True)
    app.run(host=settings.host, port=settings.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
