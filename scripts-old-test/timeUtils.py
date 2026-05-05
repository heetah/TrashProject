# -*- coding: utf-8 -*-
# Pipeline 耗時計算工具：提供可巢狀使用的 profiler、CJK 對齊摘要與函式計時 decorator。
import time
import threading
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Optional


@dataclass
class TimingStat:
    # 單一階段統計：次數、總耗時、最小/最大耗時與平均耗時。
    count: int = 0
    total: float = 0.0
    minimum: Optional[float] = None
    maximum: float = 0.0

    def add(self, duration):
        # 累加一次量測結果。
        duration = float(duration)
        self.count += 1
        self.total += duration
        self.minimum = duration if self.minimum is None else min(self.minimum, duration)
        self.maximum = max(self.maximum, duration)

    @property
    def average(self):
        # 平均秒數；沒有樣本時回傳 0。
        return self.total / self.count if self.count else 0.0


class PipelineProfiler:
    # 執行緒安全 profiler：主流程與車牌背景 thread 可共同寫入。
    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._stats = {}
        self._lock = threading.Lock()

    def record(self, name, duration):
        # 依 stage name 累積耗時，enabled=False 時零成本略過。
        if not self.enabled:
            return
        with self._lock:
            stat = self._stats.setdefault(str(name), TimingStat())
            stat.add(duration)

    @contextmanager
    def time_block(self, name):
        # context manager 形式的計時區塊，搭配 with 使用。
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - start_time)

    def snapshot(self):
        # 複製目前統計快照，避免列印時持有 lock 過久。
        with self._lock:
            return {
                name: TimingStat(
                    count=stat.count,
                    total=stat.total,
                    minimum=stat.minimum,
                    maximum=stat.maximum,
                )
                for name, stat in self._stats.items()
            }

    def print_summary(self, title="Pipeline Timing Summary", wall_label="pipeline.total_wall"):
        # 詳細版耗時表：列出所有 stage，依總耗時排序。
        stats = self.snapshot()
        if not stats:
            print("\n=== Pipeline Timing Summary ===")
            print("No timing data collected.")
            return

        wall_total = stats.get(wall_label).total if wall_label in stats else 0.0
        if wall_total <= 0.0:
            wall_total = max((stat.total for stat in stats.values()), default=0.0)

        print(f"\n=== {title} ===")
        if wall_total > 0.0:
            print(f"Total wall time: {wall_total:.3f}s")
        print(f"{'stage':48s} {'count':>8s} {'total(s)':>10s} {'avg(ms)':>10s} {'min(ms)':>10s} {'max(ms)':>10s} {'wall%':>8s}")
        print("-" * 112)

        for name, stat in sorted(stats.items(), key=lambda item: item[1].total, reverse=True):
            minimum = stat.minimum if stat.minimum is not None else 0.0
            wall_pct = (stat.total / wall_total * 100.0) if wall_total > 0.0 else 0.0
            print(
                f"{name:48s} "
                f"{stat.count:8d} "
                f"{stat.total:10.3f} "
                f"{stat.average * 1000.0:10.2f} "
                f"{minimum * 1000.0:10.2f} "
                f"{stat.maximum * 1000.0:10.2f} "
                f"{wall_pct:7.1f}%"
            )

    def print_compact_summary(self, frame_count=0, title="Pipeline Timing Summary / 推理與處理耗時統計"):
        # 精簡版耗時表：依使用者關心的階段分組，並輸出瓶頸 Top 3。
        stats = self.snapshot()
        if not stats:
            print("\n=== Pipeline Timing Summary ===")
            print("No timing data collected.")
            return

        wall_total = stats.get("pipeline.total_wall").total if "pipeline.total_wall" in stats else 0.0
        if wall_total <= 0.0:
            wall_total = max((stat.total for stat in stats.values()), default=0.0)

        sections = [
            # 分組定義：同一列可合併多個低階 profiler key。
            (
                "1. 模型前置載入",
                [
                    (
                        "模型前置載入時間",
                        [
                            "model_load.action_module_total",
                            "model_warmup.pose_yolo",
                            "model_warmup.stgcn_predict",
                            "model_load.bbox_yolo",
                            "model_warmup.bbox_yolo",
                            "model_load.trash_rtdetr",
                            "model_warmup.trash_rtdetr",
                            "model_load.plate_models_total",
                            "model_warmup.plate_yolo",
                            "model_warmup.plate_ocr",
                        ],
                    ),
                ],
            ),
            (
                "2. 影像處理",
                [
                    (
                        "影片處理總長度",
                        ["process.video_loop_total"],
                    ),
                    (
                        "YOLO",
                        [
                            "detect.yolo_actor_track",
                            "detect.yolo_actor_track_batch",
                            "detect.yolo_actor_track_fallback",
                            "detect.yolo_actor_parse",
                            "detect.yolo_actor_parse_batch",
                            "detect.yolo_actor_cache_reuse",
                        ],
                    ),
                    (
                        "RTDETR",
                        [
                            "detect.rtdetr_litter_predict",
                            "detect.rtdetr_litter_predict_batch",
                            "detect.rtdetr_parse",
                        ],
                    ),
                    (
                        "STGCN",
                        [
                            "action.pose_predict",
                            "action.pose_parse",
                            "action.track_match_and_state",
                            "action.stgcn_predict",
                            "detect.stgcn_violator_register",
                        ],
                    ),
                    (
                        "PaddleOCR",
                        [
                            "license_plate.prepare_rois",
                            "license_plate.yolo_plate_predict",
                            "license_plate.parse_and_ocr",
                            "cleanup.wait_license_plate_jobs",
                        ],
                    ),
                ],
            ),
            (
                "3. 影像寫入",
                [
                    (
                        "影像寫入/壓縮/清理",
                        [
                            "frame.write_temp_avi",
                            "encode.ffmpeg_compress",
                            "cleanup.remove_temp_avi",
                            "cleanup.release_video_io",
                        ],
                    ),
                ],
            ),
            (
                "其他重要",
                [
                    (
                        "影片讀取/解碼",
                        [
                            "video.open_capture",
                            "frame.read",
                        ],
                    ),
                    (
                        "前景遮罩",
                        ["frame.foreground_mask"],
                    ),
                    (
                        "追蹤/過濾/渲染",
                        [
                            "detect.person_vehicle_map",
                            "detect.vehicle_history_filter",
                            "detect.motion_holding_filter",
                            "detect.litter_tracker_update",
                            "detect.violator_cache_decay",
                            "detect.violator_cache_refresh",
                            "detect.render_actors",
                            "detect.render_litters",
                        ],
                    ),
                    (
                        "基礎初始化",
                        [
                            "setup.output_dir",
                            "setup.background_subtractor",
                            "video.open_writer",
                            "setup.litter_tracker",
                        ],
                    ),
                ],
            ),
        ]

        rows_by_section = []
        flat_rows = []
        # 將低階 stage 聚合成顯示列，方便閱讀整體耗時。
        for section_title, groups in sections:
            section_rows = []
            for label, names in groups:
                total = sum(stats[name].total for name in names if name in stats)
                avg_ms = (total / frame_count * 1000.0) if frame_count else 0.0
                wall_pct = (total / wall_total * 100.0) if wall_total > 0.0 else 0.0
                row = (label, total, avg_ms, wall_pct)
                section_rows.append(row)
                flat_rows.append(row)
            rows_by_section.append((section_title, section_rows))

        total_avg_ms = (wall_total / frame_count * 1000.0) if frame_count else 0.0
        slowest_rows = sorted(flat_rows, key=lambda row: row[1], reverse=True)[:3]

        # CJK 對齊：中文寬字元需要 display width，不可只用 len()。
        name_width = 24
        line_width = 72
        print(f"\n{'=' * line_width}")
        print(title)
        print("-" * line_width)
        print(f"Frames: {int(frame_count):>8d} | Total: {wall_total:>8.3f}s | Avg: {total_avg_ms:>8.2f} ms/frame")
        print("=" * line_width)

        header = (
            f"{_pad_display('項目', name_width)}  "
            f"{'總秒數':>10s}  "
            f"{'每幀ms':>10s}  "
            f"{'佔比':>8s}"
        )
        separator = "-" * line_width
        for section_title, section_rows in rows_by_section:
            print(f"\n[{section_title}]")
            print(header)
            print(separator)
            for label, total, avg_ms, wall_pct in section_rows:
                print(
                    f"{_pad_display(label, name_width)}  "
                    f"{total:10.3f}  "
                    f"{avg_ms:10.2f}  "
                    f"{wall_pct:7.1f}%"
                )

        if slowest_rows:
            print("\n[瓶頸排序 Top 3]")
            print(header)
            print(separator)
            for label, total, avg_ms, wall_pct in slowest_rows:
                print(
                    f"{_pad_display(label, name_width)}  "
                    f"{total:10.3f}  "
                    f"{avg_ms:10.2f}  "
                    f"{wall_pct:7.1f}%"
                )


def _display_width(text):
    # 計算終端顯示寬度，中文全形/寬字元算 2。
    width = 0
    for char in str(text):
        width += 2 if unicodedata.east_asian_width(char) in ("F", "W") else 1
    return width


def _pad_display(text, width):
    # 以 display width 補空白，讓中英文混排表格對齊。
    text = str(text)
    return text + " " * max(width - _display_width(text), 0)


@contextmanager
def profile_block(profiler, name):
    # 安全包裝：profiler=None 時仍可用同一個 with 寫法。
    if profiler is None:
        yield
    else:
        with profiler.time_block(name):
            yield

def time_it(func):
    # 簡易 decorator：用於臨時測量單一函式耗時。
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 記錄開始時間。
        start_time = time.perf_counter()
        
        # 執行原始函式。
        result = func(*args, **kwargs)
        
        # 記錄結束時間。
        end_time = time.perf_counter()
        
        # 計算並輸出耗時。
        duration = end_time - start_time
        
        print(f"Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result
    
    return wrapper
