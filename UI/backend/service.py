"""Application service for upload/folder intake, result discovery and reviews."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

from .analysis import load_analysis, review_units
from .config import UIConfig
from .database import Database, utc_now


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
REVIEW_VERDICTS = {"accepted", "rejected", "uncertain"}
MAX_CORRECTED_PLATE_LENGTH = 32


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = re.sub(r"[^\w.()\- ]+", "_", name, flags=re.UNICODE)
    return name[:180] or "video.mp4"


class ReviewService:
    def __init__(self, config: UIConfig, database: Database):
        self.config = config
        self.database = database
        self.config.upload_root.mkdir(parents=True, exist_ok=True)
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self.config.log_root.mkdir(parents=True, exist_ok=True)

    def _validate_video(self, path: Path, *, allowed_roots: Iterable[Path]) -> Path:
        resolved = path.expanduser().resolve()
        roots = tuple(root.expanduser().resolve() for root in allowed_roots)
        if not any(_inside(resolved, root) for root in roots):
            raise ValueError("路徑不在 UI_ALLOWED_INPUT_ROOTS 允許範圍內")
        if not resolved.is_file():
            raise ValueError("找不到影片檔案")
        if resolved.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("不支援的影片格式")
        return resolved

    def _new_job_values(
        self,
        job_id: str,
        source_kind: str,
        source_path: Path,
        original_name: str,
    ) -> dict[str, Any]:
        output_dir = self.config.output_root / job_id
        return {
            "id": job_id,
            "source_kind": source_kind,
            "source_path": str(source_path),
            "original_name": original_name,
            "status": "queued",
            "status_message": "等待 AI pipeline",
            "output_dir": str(output_dir),
            "output_video": None,
            "analysis_path": None,
            "log_path": str(self.config.log_root / f"{job_id}.log"),
            "error_message": None,
            "exit_code": None,
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
        }

    def enqueue_path(self, path: str | Path, *, source_kind: str = "folder") -> dict[str, Any]:
        source = self._validate_video(Path(path), allowed_roots=self.config.allowed_input_roots)
        job_id = uuid.uuid4().hex
        values = self._new_job_values(job_id, source_kind, source, source.name)
        self.database.insert_job(values)
        return self.get_case(job_id)["job"]

    def enqueue_folder(self, folder_path: str | Path, *, recursive: bool | None = None) -> list[dict[str, Any]]:
        folder = Path(folder_path).expanduser().resolve()
        if not any(_inside(folder, root.resolve()) for root in self.config.allowed_input_roots):
            raise ValueError("資料夾不在 UI_ALLOWED_INPUT_ROOTS 允許範圍內")
        if not folder.is_dir():
            raise ValueError("找不到指定資料夾")
        use_recursive = self.config.folder_recursive if recursive is None else bool(recursive)
        iterator = folder.rglob("*") if use_recursive else folder.iterdir()
        videos = sorted(
            (
                path
                for path in iterator
                if path.is_file()
                and path.suffix.lower() in VIDEO_EXTENSIONS
                and not path.stem.lower().endswith("_annotated")
            ),
            key=lambda item: str(item).lower(),
        )
        if not videos:
            raise ValueError("資料夾內沒有支援的影片")
        return [self.enqueue_path(video, source_kind="folder") for video in videos]

    def enqueue_upload(self, uploaded_file: Any) -> dict[str, Any]:
        original_name = Path(str(uploaded_file.filename or "")).name
        if not original_name:
            raise ValueError("上傳檔案缺少檔名")
        if Path(original_name).suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"不支援的影片格式: {original_name}")
        job_id = uuid.uuid4().hex
        upload_dir = self.config.upload_root / job_id
        upload_dir.mkdir(parents=True, exist_ok=False)
        destination = upload_dir / _safe_filename(original_name)
        temporary = destination.with_suffix(destination.suffix + ".uploading")
        try:
            uploaded_file.save(temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        values = self._new_job_values(job_id, "upload", destination, original_name)
        self.database.insert_job(values)
        return self.get_case(job_id)["job"]

    def discover_existing(self) -> int:
        imported = 0
        for root in self.config.discovery_roots:
            if not root.is_dir():
                continue
            for analysis_path in root.rglob("*_annotated_analysis.json"):
                try:
                    analysis = load_analysis(analysis_path)
                    video_name = analysis.get("video", {}).get("file")
                    output_video = (analysis_path.parent / str(video_name)).resolve() if video_name else None
                    if output_video is not None and not _inside(output_video, root.resolve()):
                        output_video = None
                    identifier = hashlib.sha256(str(analysis_path.resolve()).encode("utf-8")).hexdigest()[:24]
                    stem = analysis_path.name.removesuffix("_annotated_analysis.json")
                    values = {
                        "id": f"existing_{identifier}",
                        "source_kind": "existing",
                        "source_path": None,
                        "original_name": f"{stem}.mp4",
                        "status": "completed",
                        "status_message": "已載入既有分析結果",
                        "output_dir": str(analysis_path.parent.resolve()),
                        "output_video": str(output_video) if output_video and output_video.is_file() else None,
                        "analysis_path": str(analysis_path.resolve()),
                        "log_path": None,
                        "error_message": None,
                        "exit_code": 0,
                        "created_at": utc_now(),
                        "started_at": None,
                        "finished_at": utc_now(),
                    }
                    imported += int(self.database.insert_job(values, ignore_existing=True))
                except (OSError, ValueError):
                    continue
        return imported

    def _hydrate(self, job: dict[str, Any]) -> dict[str, Any]:
        public = dict(job)
        analysis = None
        units: list[dict[str, Any]] = []
        analysis_error = None
        if job["status"] == "completed" and job.get("analysis_path"):
            try:
                analysis = load_analysis(job["analysis_path"])
                units = review_units(
                    analysis,
                    self.database.get_reviews(job["id"]),
                    self.database.get_plate_corrections(job["id"]),
                )
            except (OSError, ValueError) as exc:
                analysis_error = str(exc)

        total_units = len(units)
        reviewed_units = sum(1 for unit in units if unit.get("review"))
        review_status = (
            "reviewed"
            if job["status"] == "completed" and total_units > 0 and reviewed_units == total_units
            else "unreviewed"
        )
        public.update(
            {
                "analysis_summary": analysis.get("summary") if analysis else None,
                "review_status": review_status,
                "reviewed_units": reviewed_units,
                "total_units": total_units,
                "video_available": bool(job.get("output_video") and Path(job["output_video"]).is_file()),
                "video_url": f"/api/jobs/{job['id']}/video" if job.get("output_video") else None,
                "analysis_error": analysis_error,
            }
        )
        return public

    def list_cases(self, review_status: str | None = None) -> dict[str, Any]:
        all_items = [self._hydrate(job) for job in self.database.list_jobs()]
        counts = {
            "all": len(all_items),
            "unreviewed": sum(item["review_status"] == "unreviewed" for item in all_items),
            "reviewed": sum(item["review_status"] == "reviewed" for item in all_items),
            "queued": sum(item["status"] == "queued" for item in all_items),
            "running": sum(item["status"] == "running" for item in all_items),
            "failed": sum(item["status"] == "failed" for item in all_items),
        }
        items = all_items
        if review_status in {"reviewed", "unreviewed"}:
            items = [item for item in all_items if item["review_status"] == review_status]
        return {"items": items, "counts": counts}

    def get_case(self, job_id: str) -> dict[str, Any]:
        job = self.database.get_job(job_id)
        if job is None:
            raise KeyError("找不到工作")
        public = self._hydrate(job)
        analysis = None
        units: list[dict[str, Any]] = []
        if job["status"] == "completed" and job.get("analysis_path"):
            analysis = load_analysis(job["analysis_path"])
            units = review_units(
                analysis,
                self.database.get_reviews(job_id),
                self.database.get_plate_corrections(job_id),
            )
        return {"job": public, "analysis": analysis, "review_units": units}

    def _require_litter_event(self, job_id: str, event_key: str) -> None:
        case = self.get_case(job_id)
        unit = next(
            (item for item in case["review_units"] if item["event_key"] == event_key),
            None,
        )
        if unit is None:
            raise ValueError("event_key 不屬於這支影片")
        if unit["kind"] != "litter":
            raise ValueError("只有垃圾事件可以人工修正車牌")

    def save_plate_correction(
        self,
        job_id: str,
        event_key: str,
        *,
        corrected_plate: str,
    ) -> dict[str, Any]:
        plate = corrected_plate.strip().upper()
        if not plate:
            raise ValueError("車牌不可為空")
        if len(plate) > MAX_CORRECTED_PLATE_LENGTH:
            raise ValueError(f"車牌不可超過 {MAX_CORRECTED_PLATE_LENGTH} 字")
        if any(character in plate for character in "\r\n\t"):
            raise ValueError("車牌不可包含換行或定位字元")
        self._require_litter_event(job_id, event_key)
        return self.database.save_plate_correction(job_id, event_key, plate)

    def delete_plate_correction(self, job_id: str, event_key: str) -> bool:
        self._require_litter_event(job_id, event_key)
        return self.database.delete_plate_correction(job_id, event_key)

    def save_review(
        self,
        job_id: str,
        event_key: str,
        *,
        verdict: str,
        note: str = "",
        reviewer: str = "",
    ) -> dict[str, Any]:
        if verdict not in REVIEW_VERDICTS:
            raise ValueError("verdict 必須是 accepted、rejected 或 uncertain")
        if len(note) > 2000:
            raise ValueError("備註不可超過 2000 字")
        if len(reviewer) > 80:
            raise ValueError("審核者名稱不可超過 80 字")
        case = self.get_case(job_id)
        valid_keys = {unit["event_key"] for unit in case["review_units"]}
        if event_key not in valid_keys:
            raise ValueError("event_key 不屬於這支影片")
        return self.database.save_review(
            job_id, event_key, verdict, note.strip(), reviewer.strip()
        )

    def retry(self, job_id: str) -> dict[str, Any]:
        if self.database.get_job(job_id) is None:
            raise KeyError("找不到工作")
        if not self.database.retry_job(job_id):
            raise ValueError("只有失敗的工作可以重試")
        return self.get_case(job_id)["job"]
