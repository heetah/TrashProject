# -*- coding: utf-8 -*-
"""前景 motion mask 建立器(temporal diff 快速路徑 / MOG2 回退)。"""
import cv2
import numpy as np

from .constants import (
    DEFAULT_FG_MASK_SCALE,
    DEFAULT_MOTION_DIFF_THRESHOLD,
    DEFAULT_MOTION_DILATE_ITERATIONS,
    DEFAULT_MOTION_BLUR_KERNEL,
    DEFAULT_MOTION_OPEN_KERNEL,
    DEFAULT_MOTION_OPEN_ITERATIONS,
    DEFAULT_MOTION_CLOSE_KERNEL,
    DEFAULT_MOTION_CLOSE_ITERATIONS,
)


def _odd_kernel_size(value):
    # OpenCV morphology / blur kernel 需要正奇數；小於 3 視為關閉。
    value = int(value or 0)
    if value < 3:
        return 0
    return value if value % 2 == 1 else value + 1


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
