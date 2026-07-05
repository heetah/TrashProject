# -*- coding: utf-8 -*-
"""影片 I/O:來源解析、硬解 VideoCapture、FFmpeg encoder 選擇、非同步 reader/writer。"""
import os
import queue
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    DEFAULT_CAPTURE_BUFFER_SIZE,
    DEFAULT_READER_QUEUE_SIZE,
    LEGACY_RESOURCE_ROOT,
)


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
