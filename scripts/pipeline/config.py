# -*- coding: utf-8 -*-
"""Production pipeline 的集中設定與 root ``.env`` 載入器。

``scripts/main.py`` 會在 import Torch/Ultralytics 與其他 pipeline 模組前先呼叫
``load_project_env()``。Shell 或 UI worker 已明確 export 的值優先，``.env`` 只補上
尚未存在的 key，避免 per-job 設定被本機檔案覆蓋。

常調整的模型路徑、confidence、I/O 與 motion 參數集中在 ``PipelineConfig``；
物理 gate、route cost 等低頻演算法常數仍留在各自模組，既有進階 env 則列於
root ``.env.example``。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import MutableMapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(
    path: Path,
    environ: MutableMapping[str, str] | None = None,
) -> Optional[Path]:
    """Load a small dotenv subset without overriding explicit environment values."""

    target = os.environ if environ is None else environ
    env_path = Path(path).expanduser()
    if not env_path.is_file():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _ENV_KEY_PATTERN.fullmatch(key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        target.setdefault(key, value)
    return env_path.resolve()


def load_project_env(
    path: Path | str | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> Optional[Path]:
    """Load root ``.env`` or ``PIPELINE_ENV_FILE`` and return the loaded path."""

    target = os.environ if environ is None else environ
    selected = path or target.get("PIPELINE_ENV_FILE") or DEFAULT_ENV_PATH
    selected_path = Path(selected).expanduser()
    if not selected_path.is_absolute():
        selected_path = PROJECT_ROOT / selected_path
    return load_env_file(selected_path, target)


def _env_int(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _env_optional_int(name, default=None):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _env_str(name, default):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _env_optional_str(name, default=None):
    v = os.environ.get(name)
    return str(v).strip() if v not in (None, "") else default


def _env_bool(name, default):
    v = os.environ.get(name)
    if v in (None, ""):
        return bool(default)
    normalized = str(v).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _env_path(name, default):
    value = _env_str(name, default)
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


@dataclass
class PipelineConfig:
    # --- 模型 / runtime ---
    pose_model_path: str = str(PROJECT_ROOT / "modules_weight/yolo26x-pose.pt")
    stgcn_weight_path: str = str(PROJECT_ROOT / "modules_weight/best_stgcn_0623.pth")
    stgcn_config_path: str = str(
        PROJECT_ROOT / "mmaction2/configs/skeleton/stgcnpp/custom_trash_stgcnpp.py"
    )
    bbox_model_path: str = str(PROJECT_ROOT / "modules_weight/best-yolo-seg_v3.pt")
    trash_model_path: str = str(
        PROJECT_ROOT / "modules_weight/best-rtdetr-4c-background.pt"
    )
    bbox_model_path_batch: str = str(
        PROJECT_ROOT / "modules_weight/batch/best-yolo-seg_v3.pt"
    )
    trash_model_path_batch: str = str(
        PROJECT_ROOT / "modules_weight/best-rtdetr-4c-background.pt"
    )
    prefer_tensorrt: bool = True
    rtdetr_enabled: bool = True
    action_device: Optional[str] = None

    # --- 批次 / 偵測路徑 ---
    batch_size: int = 8
    pipeline_queue_size: int = 8
    prepare_4c_in_reader: bool = True
    yolo_seg_frame_skip: int = 2
    actor_mode: str = "predict"
    rtdetr_zero_repair: str = "off"
    bbox_conf: float = 0.45
    trash_conf: float = 0.4
    actor_track_iou: float = 0.3
    moving_threshold: float = 0.25
    core_moving_threshold: float = 0.3

    # --- Motion mask / component evidence ---
    motion_mask_mode: str = "temporal"
    fg_mask_scale: float = 0.5
    motion_diff_threshold: int = 10
    motion_dilate_iterations: int = 2
    motion_blur_kernel: int = 0
    motion_open_kernel: int = 3
    motion_open_iterations: int = 0
    motion_close_kernel: int = 0
    motion_close_iterations: int = 0
    motion_min_component_area: int = 4
    motion_min_largest_component_ratio: float = 0.25
    motion_mog2_detect_shadows: bool = True

    # --- 違規顯示 ---
    violator_display_ttl: int = 60
    violator_display_max_jump: float = 80.0

    # --- STGCN 動作 ---
    action_threshold: float = 0.5
    action_window: int = 100
    urination_window_sec: float = 8.0
    urination_min_sec: float = 5.0

    # --- 垃圾追蹤 ---
    litter_distance_threshold: int = 250

    # --- 影片輸入 / 輸出 ---
    output_root: str = "."
    video_hw_accel: str = "any"
    video_hw_device: Optional[int] = None
    video_capture_buffer_size: int = 8
    video_read_threads: int = 0
    writer_queue_size: int = 16
    writer_preset: str = "fast"
    writer_crf: int = 23
    writer_encoder: str = "auto"
    ffmpeg_bin: Optional[str] = None
    profile_enabled: bool = True

    def model_paths_for_batch(self, batch_size: int) -> tuple[str, str]:
        """Return actor/litter model defaults for the requested runtime batch."""

        if int(batch_size) > 1:
            return self.bbox_model_path_batch, self.trash_model_path_batch
        return self.bbox_model_path, self.trash_model_path

    @classmethod
    def from_env(cls):
        # ``load_project_env`` 由 entrypoint 先執行；此處只做型別化與 relative path 解析。
        d = cls()
        return cls(
            pose_model_path=_env_path("POSE_MODEL_PATH", d.pose_model_path),
            stgcn_weight_path=_env_path("STGCN_WEIGHT_PATH", d.stgcn_weight_path),
            stgcn_config_path=_env_path("STGCN_CONFIG_PATH", d.stgcn_config_path),
            bbox_model_path=_env_path("MODEL_BBOX_PATH", d.bbox_model_path),
            trash_model_path=_env_path("MODEL_TRASH_PATH", d.trash_model_path),
            bbox_model_path_batch=_env_path(
                "MODEL_BBOX_PATH_BATCH", d.bbox_model_path_batch
            ),
            trash_model_path_batch=_env_path(
                "MODEL_TRASH_PATH_BATCH", d.trash_model_path_batch
            ),
            prefer_tensorrt=_env_bool("PREFER_TENSORRT", d.prefer_tensorrt),
            rtdetr_enabled=_env_bool("RTDETR_ENABLED", d.rtdetr_enabled),
            action_device=_env_optional_str("ACTION_DEVICE", d.action_device),
            batch_size=_env_int("PIPELINE_BATCH", d.batch_size),
            pipeline_queue_size=_env_int("PIPELINE_QUEUE_SIZE", d.pipeline_queue_size),
            prepare_4c_in_reader=_env_bool("PIPELINE_PREPARE_4C", d.prepare_4c_in_reader),
            yolo_seg_frame_skip=_env_int("YOLO_SEG_FRAME_SKIP", d.yolo_seg_frame_skip),
            actor_mode=_env_str("ACTOR_MODE", d.actor_mode),
            rtdetr_zero_repair=_env_str("RTDETR_ZERO_REPAIR", d.rtdetr_zero_repair),
            bbox_conf=_env_float("BBOX_CONF", d.bbox_conf),
            trash_conf=_env_float("TRASH_CONF", d.trash_conf),
            actor_track_iou=_env_float("ACTOR_TRACK_IOU", d.actor_track_iou),
            moving_threshold=_env_float("MOVING_THRESHOLD", d.moving_threshold),
            core_moving_threshold=_env_float("CORE_MOVING_THRESHOLD", d.core_moving_threshold),
            motion_mask_mode=_env_str("MOTION_MASK_MODE", d.motion_mask_mode),
            fg_mask_scale=_env_float("FG_MASK_SCALE", d.fg_mask_scale),
            motion_diff_threshold=_env_int(
                "MOTION_DIFF_THRESHOLD", d.motion_diff_threshold
            ),
            motion_dilate_iterations=_env_int(
                "MOTION_DILATE_ITERATIONS", d.motion_dilate_iterations
            ),
            motion_blur_kernel=_env_int("MOTION_BLUR_KERNEL", d.motion_blur_kernel),
            motion_open_kernel=_env_int("MOTION_OPEN_KERNEL", d.motion_open_kernel),
            motion_open_iterations=_env_int(
                "MOTION_OPEN_ITERATIONS", d.motion_open_iterations
            ),
            motion_close_kernel=_env_int("MOTION_CLOSE_KERNEL", d.motion_close_kernel),
            motion_close_iterations=_env_int(
                "MOTION_CLOSE_ITERATIONS", d.motion_close_iterations
            ),
            motion_min_component_area=_env_int(
                "MOTION_MIN_COMPONENT_AREA", d.motion_min_component_area
            ),
            motion_min_largest_component_ratio=_env_float(
                "MOTION_MIN_LARGEST_COMPONENT_RATIO",
                d.motion_min_largest_component_ratio,
            ),
            motion_mog2_detect_shadows=_env_bool(
                "MOTION_MOG2_DETECT_SHADOWS", d.motion_mog2_detect_shadows
            ),
            violator_display_ttl=_env_int("VIOLATOR_DISPLAY_TTL", d.violator_display_ttl),
            violator_display_max_jump=_env_float(
                "VIOLATOR_DISPLAY_MAX_JUMP", d.violator_display_max_jump
            ),
            action_threshold=_env_float("ACTION_THRESHOLD", d.action_threshold),
            action_window=_env_int("ACTION_WINDOW", d.action_window),
            urination_window_sec=_env_float("URINATION_WINDOW_SEC", d.urination_window_sec),
            urination_min_sec=_env_float("URINATION_MIN_SEC", d.urination_min_sec),
            litter_distance_threshold=_env_int(
                "LITTER_DISTANCE_THRESHOLD", d.litter_distance_threshold
            ),
            output_root=_env_str("OUTPUT_ROOT", d.output_root),
            video_hw_accel=_env_str("VIDEO_HW_ACCEL", d.video_hw_accel),
            video_hw_device=_env_optional_int("VIDEO_HW_DEVICE", d.video_hw_device),
            video_capture_buffer_size=_env_int(
                "VIDEO_CAPTURE_BUFFER_SIZE", d.video_capture_buffer_size
            ),
            video_read_threads=_env_int("VIDEO_READ_THREADS", d.video_read_threads),
            writer_queue_size=_env_int("WRITER_QUEUE_SIZE", d.writer_queue_size),
            writer_preset=_env_str("WRITER_PRESET", d.writer_preset),
            writer_crf=_env_int("WRITER_CRF", d.writer_crf),
            writer_encoder=_env_str("WRITER_ENCODER", d.writer_encoder),
            ffmpeg_bin=_env_optional_str("FFMPEG_BIN", d.ffmpeg_bin),
            profile_enabled=_env_bool("PROFILE_ENABLED", d.profile_enabled),
        )
