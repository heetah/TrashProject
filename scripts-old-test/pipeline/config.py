# -*- coding: utf-8 -*-
"""管線執行參數的單一集中設定。

原本這些值以魔術數字散落在 main.py 主流程;集中成一個 dataclass 後,調參/擴充只需改
一處,並可透過環境變數覆寫(from_env)。預設值 = 原 main.py 的固定值,行為不變。

注意:模組內部各自在「使用點」讀取的 env(VEHICLE_GATE*、ACTION_URINATE_*、BBOX_DEVICE …)
維持原樣,不搬進此處;本設定只收斂原本 main.py 硬編碼、無覆寫管道的執行參數。
"""
import os
from dataclasses import dataclass


def _env_int(name, default):
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


@dataclass
class PipelineConfig:
    # --- 批次 / 偵測路徑 ---
    batch_size: int = 8
    yolo_seg_frame_skip: int = 2
    actor_mode: str = "predict"
    rtdetr_zero_repair: str = "off"
    bbox_conf: float = 0.45
    trash_conf: float = 0.4
    actor_track_iou: float = 0.3
    moving_threshold: float = 0.25
    core_moving_threshold: float = 0.3
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
    # --- 影片輸出 ---
    writer_queue_size: int = 16
    writer_preset: str = "fast"
    writer_crf: int = 23
    writer_encoder: str = "auto"

    @classmethod
    def from_env(cls):
        # 環境變數覆寫;未設定者沿用預設(= 原固定值),故不設任何 env 時行為與過去一致。
        d = cls()
        return cls(
            batch_size=_env_int("PIPELINE_BATCH", d.batch_size),
            yolo_seg_frame_skip=_env_int("YOLO_SEG_FRAME_SKIP", d.yolo_seg_frame_skip),
            actor_mode=_env_str("ACTOR_MODE", d.actor_mode),
            rtdetr_zero_repair=_env_str("RTDETR_ZERO_REPAIR", d.rtdetr_zero_repair),
            bbox_conf=_env_float("BBOX_CONF", d.bbox_conf),
            trash_conf=_env_float("TRASH_CONF", d.trash_conf),
            actor_track_iou=_env_float("ACTOR_TRACK_IOU", d.actor_track_iou),
            moving_threshold=_env_float("MOVING_THRESHOLD", d.moving_threshold),
            core_moving_threshold=_env_float("CORE_MOVING_THRESHOLD", d.core_moving_threshold),
            violator_display_ttl=_env_int("VIOLATOR_DISPLAY_TTL", d.violator_display_ttl),
            violator_display_max_jump=_env_float("VIOLATOR_DISPLAY_MAX_JUMP", d.violator_display_max_jump),
            action_threshold=_env_float("ACTION_THRESHOLD", d.action_threshold),
            action_window=_env_int("ACTION_WINDOW", d.action_window),
            urination_window_sec=_env_float("URINATION_WINDOW_SEC", d.urination_window_sec),
            urination_min_sec=_env_float("URINATION_MIN_SEC", d.urination_min_sec),
            litter_distance_threshold=_env_int("LITTER_DISTANCE_THRESHOLD", d.litter_distance_threshold),
            writer_queue_size=_env_int("WRITER_QUEUE_SIZE", d.writer_queue_size),
            writer_preset=_env_str("WRITER_PRESET", d.writer_preset),
            writer_crf=_env_int("WRITER_CRF", d.writer_crf),
            writer_encoder=_env_str("WRITER_ENCODER", d.writer_encoder),
        )
