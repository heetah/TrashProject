"""UI configuration loaded from ``UI/.env`` without changing pipeline defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


UI_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = UI_ROOT.parent


def load_env_file(path: Path, environ: dict[str, str] | None = None) -> None:
    """Load a small dotenv subset while preserving explicitly exported values."""

    target = os.environ if environ is None else environ
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        target.setdefault(key, value)


def _bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    value = environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _int(environ: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    value = environ.get(name)
    try:
        parsed = int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _float(
    environ: Mapping[str, str], name: str, default: float, minimum: float = 0.0
) -> float:
    value = environ.get(name)
    try:
        parsed = float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, parsed)


def _path(value: str | None, default: Path) -> Path:
    candidate = Path(value).expanduser() if value else default
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _paths(value: str | None, default: tuple[Path, ...]) -> tuple[Path, ...]:
    if not value:
        return tuple(path.resolve() for path in default)
    return tuple(
        _path(item.strip(), REPOSITORY_ROOT)
        for item in value.split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class UIConfig:
    repository_root: Path
    pipeline_entrypoint: Path
    database_path: Path
    upload_root: Path
    output_root: Path
    discovery_roots: tuple[Path, ...]
    allowed_input_roots: tuple[Path, ...]
    frontend_dist: Path
    log_root: Path
    export_root: Path
    ffmpeg_executable: str
    export_pre_roll_sec: float
    export_post_roll_sec: float
    host: str
    port: int
    conda_executable: str
    conda_environment: str
    pipeline_batch: int
    max_upload_mb: int
    poll_seconds: int
    worker_enabled: bool
    import_existing: bool
    folder_recursive: bool

    @property
    def max_content_length(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def build_config(
    *,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> UIConfig:
    if environ is None:
        load_env_file(env_path or UI_ROOT / ".env")
        values: Mapping[str, str] = os.environ
    else:
        mutable_values = dict(environ)
        load_env_file(env_path or UI_ROOT / ".env", mutable_values)
        values = mutable_values

    data_root = UI_ROOT / "data"
    resources_root = REPOSITORY_ROOT / "resources"
    output_root = _path(values.get("UI_OUTPUT_ROOT"), REPOSITORY_ROOT / "output/ui_runs")
    return UIConfig(
        repository_root=REPOSITORY_ROOT,
        pipeline_entrypoint=_path(
            values.get("UI_PIPELINE_ENTRYPOINT"), REPOSITORY_ROOT / "scripts/main.py"
        ),
        database_path=_path(values.get("UI_DATABASE_PATH"), data_root / "ui.sqlite3"),
        upload_root=_path(values.get("UI_UPLOAD_ROOT"), data_root / "uploads"),
        output_root=output_root,
        discovery_roots=_paths(
            values.get("UI_DISCOVERY_ROOTS"), (REPOSITORY_ROOT / "output",)
        ),
        allowed_input_roots=_paths(
            values.get("UI_ALLOWED_INPUT_ROOTS"), (resources_root,)
        ),
        frontend_dist=_path(
            values.get("UI_FRONTEND_DIST"), UI_ROOT / "frontend/dist"
        ),
        log_root=_path(values.get("UI_LOG_ROOT"), data_root / "logs"),
        export_root=_path(
            values.get("UI_EXPORT_ROOT"), REPOSITORY_ROOT / "output/ui_exports"
        ),
        ffmpeg_executable=values.get("UI_FFMPEG_EXECUTABLE", "ffmpeg"),
        export_pre_roll_sec=_float(
            values, "UI_EXPORT_PRE_ROLL_SEC", 3.0, minimum=0.0
        ),
        export_post_roll_sec=_float(
            values, "UI_EXPORT_POST_ROLL_SEC", 3.0, minimum=0.0
        ),
        host=values.get("UI_HOST", "127.0.0.1"),
        port=_int(values, "UI_PORT", 5000, minimum=1),
        conda_executable=values.get("UI_CONDA_EXECUTABLE", "conda"),
        conda_environment=values.get("UI_CONDA_ENV", "rtdetr"),
        pipeline_batch=_int(values, "UI_PIPELINE_BATCH", 8, minimum=1),
        max_upload_mb=_int(values, "UI_MAX_UPLOAD_MB", 2048, minimum=1),
        poll_seconds=_int(values, "UI_POLL_SECONDS", 3, minimum=1),
        worker_enabled=_bool(values, "UI_WORKER_ENABLED", True),
        import_existing=_bool(values, "UI_IMPORT_EXISTING", True),
        folder_recursive=_bool(values, "UI_FOLDER_RECURSIVE", True),
    )
