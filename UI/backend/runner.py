"""One-at-a-time background runner for the GPU production entrypoint."""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from .config import UIConfig
from .database import Database, utc_now


class PipelineWorker:
    def __init__(self, config: UIConfig, database: Database):
        self.config = config
        self.database = database
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pipeline-worker", daemon=True)

    def start(self) -> None:
        self.database.recover_interrupted_jobs()
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.database.claim_next_job()
            if job is None:
                self._stop.wait(1.0)
                continue
            self._run_job(job)

    def _command(self, source_path: Path) -> list[str]:
        return [
            self.config.conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            self.config.conda_environment,
            "python",
            str(self.config.pipeline_entrypoint),
            str(source_path),
        ]

    def _run_job(self, job: dict[str, object]) -> None:
        job_id = str(job["id"])
        source_path = Path(str(job["source_path"]))
        output_dir = Path(str(job["output_dir"]))
        log_path = Path(str(job["log_path"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        expected_video = output_dir / f"{source_path.stem}_annotated.mp4"
        expected_analysis = expected_video.with_name(expected_video.stem + "_analysis.json")
        environment = os.environ.copy()
        environment["OUTPUT_ROOT"] = str(output_dir)
        environment["PIPELINE_BATCH"] = str(self.config.pipeline_batch)
        environment.setdefault("SMART_BACKTRACK_SIDECAR", "0")

        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    self._command(source_path),
                    cwd=self.config.repository_root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if result.returncode != 0:
                self.database.update_job(
                    job_id,
                    status="failed",
                    status_message="AI pipeline 執行失敗",
                    error_message=f"pipeline 結束碼 {result.returncode}；請查看工作紀錄",
                    exit_code=result.returncode,
                    finished_at=utc_now(),
                )
                return
            if not expected_video.is_file() or not expected_analysis.is_file():
                self.database.update_job(
                    job_id,
                    status="failed",
                    status_message="pipeline 未產生完整輸出",
                    error_message="缺少 annotated MP4 或 analysis JSON",
                    exit_code=result.returncode,
                    finished_at=utc_now(),
                )
                return
            self.database.update_job(
                job_id,
                status="completed",
                status_message="等待人工審核",
                output_video=str(expected_video.resolve()),
                analysis_path=str(expected_analysis.resolve()),
                exit_code=result.returncode,
                finished_at=utc_now(),
            )
        except OSError as exc:
            self.database.update_job(
                job_id,
                status="failed",
                status_message="無法啟動 AI pipeline",
                error_message=str(exc),
                finished_at=utc_now(),
            )

