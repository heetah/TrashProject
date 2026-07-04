# -*- coding: utf-8 -*-
"""基礎設施與驗證工具：engine/模型/ffmpeg/video 能力探測與非同步 reader/writer/mask。

從 main.py 抽離，集中所有「驗證、探測、fallback」邏輯，讓 main.py 專注於主流程編排。
"""
import cv2
import numpy as np
import os
import queue
import subprocess
import threading
from pathlib import Path

from pipeline.devices import BBOX_DEVICE, BBOX_HALF, TRASH_DEVICE, TRASH_HALF

# === 共用常數（基礎設施用） ===
SUPPORTED_BATCH_SIZES = (1, 2, 4, 8)
DEFAULT_FG_MASK_SCALE = 0.5
DEFAULT_MOTION_DIFF_THRESHOLD = 10
DEFAULT_MOTION_DILATE_ITERATIONS = 2
DEFAULT_MOTION_BLUR_KERNEL = 0
DEFAULT_MOTION_OPEN_KERNEL = 3
DEFAULT_MOTION_OPEN_ITERATIONS = 0
DEFAULT_MOTION_CLOSE_KERNEL = 0
DEFAULT_MOTION_CLOSE_ITERATIONS = 0
DEFAULT_MOTION_MIN_COMPONENT_AREA = 4
DEFAULT_MOTION_MIN_LARGEST_COMPONENT_RATIO = 0.25
DEFAULT_READER_QUEUE_SIZE = 32
DEFAULT_CAPTURE_BUFFER_SIZE = 8
LEGACY_RESOURCE_ROOT = Path("/mnt/8tb_hdd/under115a")


# torch 是選用依賴：若不可用，TensorRT engine 檢查會自動略過。
try:
    import torch
except Exception:
    torch = None


def _resolve_video_path(file_arg):
    # 支援 AGENTS.md 的 resources/foo.mp4、單純檔名與舊 /mnt/8tb_hdd/under115a/resources 來源。
    raw = Path(str(file_arg)).expanduser()
    if raw.is_absolute():
        return str(raw)

    candidates = [raw]
    if len(raw.parts) == 1:
        candidates.insert(0, Path("resources") / raw)
        candidates.insert(1, LEGACY_RESOURCE_ROOT / "resources" / raw)
    elif raw.parts[0] == "resources":
        candidates.append(LEGACY_RESOURCE_ROOT / raw)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _odd_kernel_size(value):
    # OpenCV morphology / blur kernel 需要正奇數；小於 3 視為關閉。
    value = int(value or 0)
    if value < 3:
        return 0
    return value if value % 2 == 1 else value + 1


def _video_accel_value(name):
    # OpenCV VideoCapture 硬解參數；平台/編譯不支援時回傳 None 走 fallback。
    accel_name = str(name or "any").strip().lower()
    if accel_name in ("none", "off", "false", "0"):
        return getattr(cv2, "VIDEO_ACCELERATION_NONE", 0)
    if accel_name in ("any", "auto", "on", "true", "1"):
        return getattr(cv2, "VIDEO_ACCELERATION_ANY", None)
    if accel_name == "vaapi":
        return getattr(cv2, "VIDEO_ACCELERATION_VAAPI", None)
    if accel_name in ("mfx", "qsv"):
        return getattr(cv2, "VIDEO_ACCELERATION_MFX", None)
    if accel_name == "d3d11":
        return getattr(cv2, "VIDEO_ACCELERATION_D3D11", None)
    raise ValueError(f"Unsupported video hardware acceleration mode: {name}")


def _append_capture_param(params, prop_name, value):
    prop = getattr(cv2, prop_name, None)
    if prop is None or value is None:
        return
    params.extend([int(prop), int(value)])


def _open_video_capture(video_path, hw_accel="any", hw_device=None,
                        buffer_size=DEFAULT_CAPTURE_BUFFER_SIZE,
                        read_threads=0, profiler=None):
    # OpenCV FFmpeg backend 可用硬體解碼時先試；失敗保留軟解，frame 順序與數量不變。
    api_preference = getattr(cv2, "CAP_FFMPEG", 0)
    params = []
    accel_value = _video_accel_value(hw_accel)
    if accel_value is not None:
        _append_capture_param(params, "CAP_PROP_HW_ACCELERATION", accel_value)
        if hw_device is not None and int(hw_device) >= 0:
            _append_capture_param(params, "CAP_PROP_HW_DEVICE", int(hw_device))
    if read_threads and int(read_threads) > 0:
        _append_capture_param(params, "CAP_PROP_N_THREADS", int(read_threads))

    cap = None
    open_error = None
    if params:
        try:
            with profiler.time_block("video.open_capture_hw"):
                cap = cv2.VideoCapture(str(video_path), api_preference, params)
        except Exception as exc:
            open_error = exc
            cap = None

    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        if open_error is not None:
            print(f"Warning: hardware VideoCapture open failed; fallback to software decode: {open_error}")
        with profiler.time_block("video.open_capture_fallback"):
            cap = cv2.VideoCapture(str(video_path), api_preference)
        if not cap.isOpened():
            cap.release()
            with profiler.time_block("video.open_capture_fallback"):
                cap = cv2.VideoCapture(str(video_path))

    if buffer_size and int(buffer_size) > 0:
        buffer_prop = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffer_prop is not None:
            cap.set(buffer_prop, int(buffer_size))

    try:
        backend_name = cap.getBackendName()
    except Exception:
        backend_name = "unknown"
    return cap, backend_name


def _can_try_tensorrt_engine(engine_path):
    # TensorRT engine 需要 CUDA + tensorrt 套件；缺任一項就回退 .pt 權重。
    if torch is None or not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        print(f"Warning: CUDA is unavailable; skip TensorRT engine {engine_path}.")
        return False

    try:
        import tensorrt  # noqa: F401
    except Exception as exc:
        print(f"Warning: TensorRT import failed; skip engine {engine_path}: {exc}")
        return False

    return True


def _engine_path_for_batch(model_path, batch_size):
    # batch 1 使用同名 .engine；batch N 使用 *_bN.engine，避免不同 batch engine 互相覆蓋。
    path = Path(model_path)
    if int(batch_size) <= 1:
        return path.with_suffix(".engine")
    return path.with_name(f"{path.stem}_b{int(batch_size)}.engine")


def _engine_batch_size_from_path(model_path, fallback=1):
    # 從 *_bN.engine 還原 TensorRT fixed batch；.pt 則用呼叫端期望 batch。
    path = Path(model_path)
    if path.suffix != ".engine":
        return max(int(fallback or 1), 1)
    stem = path.stem
    if "_b" in stem:
        maybe_batch = stem.rsplit("_b", 1)[1]
        if maybe_batch.isdigit():
            return int(maybe_batch)
    return 1


def _round_supported_batch_size(target):
    target = max(int(target or 1), 1)
    for batch_size in SUPPORTED_BATCH_SIZES:
        if batch_size >= target:
            return batch_size
    return SUPPORTED_BATCH_SIZES[-1]


def _estimate_actor_batch_size(pipeline_batch_size, yolo_seg_frame_skip):
    # actor 可跳幀；batch 8 + skip 2 實際只需 4 張 actor 推理，不應強迫塞 b8。
    pipeline_batch_size = max(int(pipeline_batch_size or 1), 1)
    skip = max(int(yolo_seg_frame_skip or 1), 1)
    target = (pipeline_batch_size + skip - 1) // skip
    return _round_supported_batch_size(target)


def _model_path_candidates(model_path, prefer_engine=True, batch_size=1):
    return _model_path_candidates_for_batches(model_path, prefer_engine, [batch_size])


def _model_path_candidates_for_batches(model_path, prefer_engine=True, batch_sizes=None):
    # 建立模型候選順序：優先 engine，失敗或不存在時回退原始權重。
    path = Path(model_path)
    candidates = []
    batch_sizes = list(batch_sizes or [1])

    if prefer_engine and path.suffix == ".pt":
        seen_batches = set()
        for batch_size in batch_sizes:
            batch_size = max(int(batch_size or 1), 1)
            if batch_size in seen_batches:
                continue
            seen_batches.add(batch_size)
            engine_path = _engine_path_for_batch(path, batch_size)
            if engine_path.exists():
                if _can_try_tensorrt_engine(engine_path):
                    candidates.append(str(engine_path))
            else:
                print(f"Warning: TensorRT engine not found for {path} at batch={batch_size}.")

    candidates.append(str(path))

    if prefer_engine and path.suffix == ".engine":
        if _can_try_tensorrt_engine(path):
            candidates.insert(0, str(path))
        else:
            candidates = []
        pt_path = path.with_suffix(".pt")
        if pt_path.exists():
            candidates.append(str(pt_path))

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _load_model_with_warmup(label, candidates, model_factory, warmup_func, profiler):
    # 逐一嘗試候選模型；成功載入後立即 warmup，讓正式影片處理不吃第一次推理成本。
    last_error = None
    for idx, model_path in enumerate(candidates):
        try:
            if idx > 0:
                print(f"{label}: retrying with fallback model {model_path}")
            with profiler.time_block(f"model_load.{label}"):
                model = model_factory(model_path)
            with profiler.time_block(f"model_warmup.{label}"):
                warmup_func(model, model_path)
            print(f"{label}: loaded and warmed up {model_path}")
            return model, model_path
        except Exception as exc:
            last_error = exc
            print(f"{label}: failed to initialize {model_path}: {exc}")

    raise RuntimeError(f"{label}: all model candidates failed: {candidates}") from last_error


_FFMPEG_ENCODER_CACHE = {}
_FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def _set_ffmpeg_bin(ffmpeg_bin):
    # conda env 內的 ffmpeg 可能沒有 NVENC；允許 main.py 改用指定 binary。
    global _FFMPEG_BIN
    if ffmpeg_bin:
        _FFMPEG_BIN = str(ffmpeg_bin)


def _ffmpeg_available_encoders(ffmpeg_bin=None):
    # 只探測一次 ffmpeg 支援的 encoder，避免每次開 writer 都重跑清單。
    ffmpeg_bin = str(ffmpeg_bin or _FFMPEG_BIN)
    if ffmpeg_bin in _FFMPEG_ENCODER_CACHE:
        return _FFMPEG_ENCODER_CACHE[ffmpeg_bin]

    try:
        result = subprocess.run(
            [ffmpeg_bin, '-hide_banner', '-encoders'],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except Exception:
        _FFMPEG_ENCODER_CACHE[ffmpeg_bin] = set()
        return _FFMPEG_ENCODER_CACHE[ffmpeg_bin]

    encoders = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in 'VASD':
            encoders.add(parts[1])

    _FFMPEG_ENCODER_CACHE[ffmpeg_bin] = encoders
    return _FFMPEG_ENCODER_CACHE[ffmpeg_bin]


def _ffmpeg_encoder_available(encoder_name, ffmpeg_bin=None):
    # 讓 writer 可以先選可用 encoder，再決定要不要退回 x264。
    return str(encoder_name or '').strip() in _ffmpeg_available_encoders(ffmpeg_bin)


def _select_ffmpeg_bin(preferred=None):
    # rtdetr conda ffmpeg 常缺 NVENC；若系統 ffmpeg 有 h264_nvenc，auto 改用系統版。
    preferred = str(preferred).strip() if preferred else None
    if preferred:
        return preferred

    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin:
        return env_bin

    default_bin = "ffmpeg"
    system_bin = "/usr/bin/ffmpeg"
    if _ffmpeg_encoder_available("h264_nvenc", default_bin):
        return default_bin
    if Path(system_bin).exists() and _ffmpeg_encoder_available("h264_nvenc", system_bin):
        print(f"FFmpeg auto-selected {system_bin} for NVENC support.")
        return system_bin
    return default_bin


def _resolve_video_encoder(preferred="auto", frame_width=None, frame_height=None, ffmpeg_bin=None):
    # auto 優先用 NVENC，若 ffmpeg 沒有或使用者指定其他值，就回退到 libx264。
    preferred_name = str(preferred or 'auto').strip().lower()

    width = int(frame_width or 0)
    height = int(frame_height or 0)
    nvenc_supported = min(width, height) >= 128 if width > 0 and height > 0 else True

    if preferred_name in ('h264_nvenc', 'hevc_nvenc', 'libx264'):
        if preferred_name in ('h264_nvenc', 'hevc_nvenc') and not nvenc_supported:
            print(
                f"Warning: Frame size {width}x{height} is too small for NVENC; fallback to libx264."
            )
            return 'libx264'
        if _ffmpeg_encoder_available(preferred_name, ffmpeg_bin):
            return preferred_name
        print(f"Warning: FFmpeg encoder {preferred_name} unavailable; fallback to libx264.")
        return 'libx264'

    if preferred_name not in ('', 'auto'):
        print(f"Warning: Unsupported video encoder {preferred_name}; fallback to auto selection.")

    for candidate in ('h264_nvenc', 'hevc_nvenc', 'libx264'):
        if candidate in ('h264_nvenc', 'hevc_nvenc') and not nvenc_supported:
            continue
        if _ffmpeg_encoder_available(candidate, ffmpeg_bin):
            return candidate

    print("Warning: No preferred FFmpeg encoder found; fallback to libx264.")
    return 'libx264'


def _nvenc_preset_from_generic(preset_name):
    # 將原本 x264-style preset 名稱映射成 NVENC 可接受的 p1~p7。
    preset = str(preset_name or '').strip().lower()
    if preset in ('p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7'):
        return preset
    preset_map = {
        'ultrafast': 'p1',
        'superfast': 'p1',
        'veryfast': 'p1',
        'faster': 'p2',
        'fast': 'p2',
        'medium': 'p4',
        'slow': 'p5',
        'slower': 'p6',
        'veryslow': 'p7',
    }
    return preset_map.get(preset, 'p1')


def _build_ffmpeg_video_encoder_args(encoder_name, preset="fast", crf=23):
    # 依 encoder 類型組出參數；只改輸出編碼，不動前面偵測與渲染流程。
    quality = max(0, min(int(crf), 51))
    if encoder_name in ('h264_nvenc', 'hevc_nvenc'):
        return [
            '-vcodec', encoder_name,
            '-preset', _nvenc_preset_from_generic(preset),
            '-rc', 'vbr',
            '-cq', str(quality),
            '-b:v', '0',
        ]

    return [
        '-vcodec', 'libx264',
        '-preset', str(preset),
        '-crf', str(quality),
    ]


class AsyncFFmpegVideoWriter:
    # 背景 FFmpeg writer：主執行緒只排隊 frame，編碼與 muxing 由背景 thread 處理。
    def __init__(self, output_path, width, height, fps, profiler=None,
                 queue_size=16, preset="fast", crf=23, encoder="auto", ffmpeg_bin=None):
        self.output_path = str(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.profiler = profiler
        self.ffmpeg_bin = str(ffmpeg_bin or _FFMPEG_BIN)
        self._queue = queue.Queue(maxsize=max(int(queue_size or 1), 1))
        self._stop_token = object()
        self._error = None
        self._closed = False
        self._process = None
        self.encoder_name = _resolve_video_encoder(
            encoder,
            frame_width=self.width,
            frame_height=self.height,
            ffmpeg_bin=self.ffmpeg_bin,
        )
        encoder_args = _build_ffmpeg_video_encoder_args(self.encoder_name, preset=preset, crf=crf)

        print(f"FFmpeg binary: {self.ffmpeg_bin}")
        print(f"FFmpeg video encoder selected: {self.encoder_name}")

        ffmpeg_cmd = [
            self.ffmpeg_bin, '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-an',
            *encoder_args,
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            self.output_path,
        ]

        with profiler.time_block("video.open_ffmpeg_writer"):
            self._process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        self._thread = threading.Thread(target=self._worker, name="ffmpeg-writer", daemon=True)
        self._thread.start()

    def write(self, frame):
        if self._closed:
            raise RuntimeError("FFmpeg writer is already closed.")
        if self._error is not None:
            raise RuntimeError("FFmpeg writer failed.") from self._error
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(f"Frame size mismatch: got {frame.shape[1]}x{frame.shape[0]}, expected {self.width}x{self.height}")

        # 確保傳給 FFmpeg 的資料是連續 BGR buffer，避免 stdin.write 時產生格式問題。
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        with self.profiler.time_block("frame.queue_output_frame"):
            self._queue.put(frame)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._stop_token)
        with self.profiler.time_block("encode.ffmpeg_pipe_close"):
            self._thread.join()
            if self._process.stdin is not None:
                self._process.stdin.close()
            return_code = self._process.wait()
        if self._error is not None:
            raise RuntimeError("FFmpeg writer failed.") from self._error
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with code {return_code}")

    def _worker(self):
        try:
            while True:
                frame = self._queue.get()
                try:
                    if frame is self._stop_token:
                        break
                    with self.profiler.time_block("frame.ffmpeg_stdin_write"):
                        self._process.stdin.write(memoryview(frame))
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._error = exc
        finally:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.flush()
            except Exception as exc:
                if self._error is None:
                    self._error = exc


class MotionMaskBuilder:
    # 前景 mask 建立器：預設用 temporal diff 加速；必要時可切回原 MOG2。
    def __init__(self, mode="temporal", scale_factor=DEFAULT_FG_MASK_SCALE,
                 diff_threshold=DEFAULT_MOTION_DIFF_THRESHOLD,
                 dilate_iterations=DEFAULT_MOTION_DILATE_ITERATIONS,
                 blur_kernel_size=DEFAULT_MOTION_BLUR_KERNEL,
                 open_kernel_size=DEFAULT_MOTION_OPEN_KERNEL,
                 open_iterations=DEFAULT_MOTION_OPEN_ITERATIONS,
                 close_kernel_size=DEFAULT_MOTION_CLOSE_KERNEL,
                 close_iterations=DEFAULT_MOTION_CLOSE_ITERATIONS,
                 mog2_history=300, mog2_var_threshold=25, mog2_detect_shadows=True):
        self.mode = str(mode or "temporal").lower()
        self.scale_factor = float(scale_factor or 1.0)
        self.diff_threshold = int(diff_threshold)
        self.dilate_iterations = max(int(dilate_iterations or 0), 0)
        self.blur_kernel_size = _odd_kernel_size(blur_kernel_size)
        self.open_iterations = max(int(open_iterations or 0), 0)
        self.close_iterations = max(int(close_iterations or 0), 0)
        self.prev_gray = None
        self.temporal_kernel = np.ones((3, 3), dtype=np.uint8) if self.dilate_iterations > 0 else None
        open_kernel_size = _odd_kernel_size(open_kernel_size)
        close_kernel_size = _odd_kernel_size(close_kernel_size)
        self.open_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
            if self.open_iterations > 0 and open_kernel_size > 0
            else None
        )
        self.close_kernel = (
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
            if self.close_iterations > 0 and close_kernel_size > 0
            else None
        )
        self.back_sub = None

        if self.mode == "mog2":
            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=int(mog2_history),
                varThreshold=float(mog2_var_threshold),
                detectShadows=bool(mog2_detect_shadows),
            )
        elif self.mode != "temporal":
            raise ValueError(f"Unsupported motion mask mode: {mode}")

    def build(self, frame, profiler):
        if self.mode == "mog2":
            return self._build_mog2(frame, profiler)
        return self._build_temporal(frame, profiler)

    def _scaled_frame(self, frame, profiler):
        if self.scale_factor == 1.0:
            return frame
        with profiler.time_block("frame.motion_resize"):
            return cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

    def _build_mog2(self, frame, profiler):
        # 原始 MOG2 路徑：保留給需要逐像素背景模型時回退使用。
        mask_frame = self._scaled_frame(frame, profiler)
        with profiler.time_block("frame.foreground_mog2_apply"):
            fg_mask = self.back_sub.apply(mask_frame, learningRate=0.005)
        with profiler.time_block("frame.foreground_threshold"):
            _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        fg_mask = self._cleanup_mask(fg_mask, profiler)
        return fg_mask

    def _build_temporal(self, frame, profiler):
        # 快速路徑：只比較相鄰幀灰階差異，符合目前「litter 是否正在移動」用途。
        mask_frame = self._scaled_frame(frame, profiler)
        with profiler.time_block("frame.motion_gray"):
            gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        if self.blur_kernel_size > 0:
            with profiler.time_block("frame.motion_blur"):
                gray = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return np.full(gray.shape, 255, dtype=np.uint8)

        with profiler.time_block("frame.motion_absdiff"):
            diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        with profiler.time_block("frame.motion_threshold"):
            _, motion_mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        motion_mask = self._cleanup_mask(motion_mask, profiler)
        if self.temporal_kernel is not None:
            with profiler.time_block("frame.motion_dilate"):
                motion_mask = cv2.dilate(motion_mask, self.temporal_kernel, iterations=self.dilate_iterations)
        return motion_mask

    def _cleanup_mask(self, motion_mask, profiler):
        # 先 opening 去掉孤立亮點，再視需要 closing 補小洞；典型監視器噪聲濾波。
        if self.open_kernel is not None:
            with profiler.time_block("frame.motion_open"):
                motion_mask = cv2.morphologyEx(
                    motion_mask,
                    cv2.MORPH_OPEN,
                    self.open_kernel,
                    iterations=self.open_iterations,
                )
        if self.close_kernel is not None:
            with profiler.time_block("frame.motion_close"):
                motion_mask = cv2.morphologyEx(
                    motion_mask,
                    cv2.MORPH_CLOSE,
                    self.close_kernel,
                    iterations=self.close_iterations,
                )
        return motion_mask


class AsyncVideoFrameReader:
    # 背景讀取/解碼/前景 mask thread：主流程跑推理時，下一批 frame 已先準備好。
    def __init__(self, cap, motion_masker, profiler, queue_size=DEFAULT_READER_QUEUE_SIZE):
        self.cap = cap
        self.motion_masker = motion_masker
        self.profiler = profiler
        self._queue = queue.Queue(maxsize=max(int(queue_size or 1), 1))
        self._stop_event = threading.Event()
        self._sentinel = object()
        self._error = None
        self._done = False
        self._thread = threading.Thread(target=self._worker, name="video-reader", daemon=True)
        self._thread.start()

    def read_batch(self, batch_size):
        if self._done:
            if self._error is not None:
                raise RuntimeError("Async video reader failed.") from self._error
            return [], []

        frames = []
        fg_masks = []
        target = max(int(batch_size or 1), 1)
        while len(frames) < target:
            with self.profiler.time_block("frame.reader_dequeue"):
                item = self._queue.get()
            try:
                if item is self._sentinel:
                    self._done = True
                    break
                frame, fg_mask = item
                frames.append(frame)
                fg_masks.append(fg_mask)
            finally:
                self._queue.task_done()

        if self._done and self._error is not None and not frames:
            raise RuntimeError("Async video reader failed.") from self._error
        return frames, fg_masks

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def _put(self, item):
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _worker(self):
        try:
            while not self._stop_event.is_set():
                with self.profiler.time_block("frame.read"):
                    ret, frame = self.cap.read()
                if not ret:
                    break

                with self.profiler.time_block("frame.foreground_mask"):
                    fg_mask = self.motion_masker.build(frame, self.profiler)
                if not self._put((frame, fg_mask)):
                    break
        except Exception as exc:
            self._error = exc
        finally:
            try:
                self.cap.release()
            finally:
                self._put(self._sentinel)


def _batched_dummy_frame(batch_size, imgsz=640, channels=3):
    # warmup 使用假 frame；batch 模式需傳入 list，才能讓 backend 建立正確 batch shape。
    dummy_frame = np.zeros((int(imgsz), int(imgsz), int(channels)), dtype=np.uint8)
    if int(batch_size) <= 1:
        return dummy_frame
    return [dummy_frame.copy() for _ in range(int(batch_size))]


def _read_engine_input_shape(engine_path):
    # 讀取 engine input binding shape，回傳 (batch, channels, h, w) 或 None。
    # 優先從 ultralytics metadata prefix JSON 讀取（快速，不需要起 TRT Runtime）；
    # 舊格式 engine 沒有 prefix 時才退回 TRT Runtime 讀取 binding shape。
    #
    # ultralytics metadata prefix 格式：
    #   [4-byte little-endian signed length][UTF-8 JSON][raw TRT serialized bytes]
    path = Path(engine_path)
    if path.suffix != ".engine":
        return None
    try:
        import json as _json
        with open(str(path), "rb") as f:
            raw = f.read()

        # ── 嘗試讀取 metadata prefix ──────────────────────────────────────────
        meta_len = int.from_bytes(raw[:4], byteorder="little", signed=True)
        if 0 < meta_len < 65536:
            try:
                meta = _json.loads(raw[4:4 + meta_len].decode("utf-8"))
                imgsz_raw = meta.get("imgsz")
                channels = int(meta.get("channels", 3))
                batch = int(meta.get("batch", 1))
                if imgsz_raw is not None:
                    if isinstance(imgsz_raw, (list, tuple)) and len(imgsz_raw) >= 2:
                        h, w = int(imgsz_raw[0]), int(imgsz_raw[1])
                    else:
                        h = w = int(imgsz_raw)
                    shape = (batch, channels, h, w)
                    print(f"Engine input shape from metadata: {shape}")
                    return shape
            except Exception:
                pass
            # metadata prefix 存在但 imgsz 缺失 → 跳過 prefix，用 TRT 讀 binding
            trt_bytes = raw[4 + meta_len:]
        else:
            # 沒有 metadata prefix（舊格式 engine）→ 直接 TRT 解析
            trt_bytes = raw

        # ── TRT Runtime fallback ───────────────────────────────────────────────
        import tensorrt as trt  # noqa: F401 – optional import
        trt_logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(trt_logger)
        engine = runtime.deserialize_cuda_engine(trt_bytes)
        if engine is None:
            return None
        if hasattr(engine, "num_io_tensors"):  # TRT 10+
            name = engine.get_tensor_name(0)
            shape = tuple(engine.get_tensor_shape(name))
        else:
            shape = tuple(engine.get_binding_shape(0))
        print(f"Engine input shape from TRT: {shape}")
        return shape  # (batch, ch, h, w)
    except Exception as exc:
        print(f"Warning: could not read engine input shape from {path}: {exc}")
        return None


def _apply_engine_overrides(model, model_path):
    # engine 載入後，用 TRT binding shape 補回 imgsz override，讓後續 predict() 前處理正確。
    shape = _read_engine_input_shape(model_path)
    if shape is None or len(shape) != 4:
        return
    _, ch, h, w = shape
    imgsz = max(h, w)
    model.overrides["imgsz"] = imgsz
    print(f"Applied engine overrides: imgsz={imgsz}, input_channels={ch}")


def _get_model_input_channels(model) -> int:
    """讀取模型的 input channel 數，優先從 .pt checkpoint yaml 取，再試 AutoBackend backend。
    4c 模型的 .pt 內 yaml["channels"] = 4；找不到時回傳 3（標準 3-channel 模型預設）。
    """
    # .pt 路徑：model.model 是 PyTorch YOLO/RTDETR 模型，yaml 存有 channels key
    try:
        ch = model.model.yaml.get("channels", None)
        if ch is not None:
            return int(ch)
    except Exception:
        pass
    # engine 路徑：AutoBackend backend 的 channels 屬性（由 apply_metadata 設定）
    try:
        return int(model.predictor.model.backend.channels)
    except Exception:
        pass
    return 3


def _get_model_warmup_imgsz(model) -> int:
    """讀取模型應使用的 warmup imgsz。
    優先從 model.overrides（.pt checkpoint 有；engine 需 _apply_engine_overrides 設定過）讀取，
    再試 model.backend.imgsz（engine metadata），最後回退到 640。
    """
    try:
        raw = model.overrides.get("imgsz", None)
        if raw is not None:
            if isinstance(raw, (list, tuple)):
                return max(int(v) for v in raw)
            return int(raw)
    except Exception:
        pass
    try:
        raw = model.predictor.model.backend.imgsz
        if isinstance(raw, (list, tuple)):
            return max(int(v) for v in raw)
        return int(raw)
    except Exception:
        pass
    return 640


def _warmup_bbox_model(model, batch_size=1):
    # YOLO actor model warmup：預先觸發 CUDA/TensorRT kernel 初始化。
    dummy_source = _batched_dummy_frame(batch_size)
    model.predict(
        dummy_source,
        conf=0.01,
        device=BBOX_DEVICE,
        half=BBOX_HALF,
        verbose=False,
    )


def _warmup_trash_model(model, batch_size=1, imgsz=640, channels=3):
    # RTDETR litter model warmup：確保垃圾模型在正式迴圈前已初始化。
    # engine 傳入 imgsz/channels 才能產生符合 binding shape 的 dummy frame。
    dummy_source = _batched_dummy_frame(batch_size, imgsz=imgsz, channels=channels)
    model.predict(
        dummy_source,
        conf=0.01,
        device=TRASH_DEVICE,
        half=TRASH_HALF,
        verbose=False,
        imgsz=imgsz,
    )
