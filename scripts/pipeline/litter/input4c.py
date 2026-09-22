# -*- coding: utf-8 -*-
"""RT-DETR litter 模型的 4-channel inference input 建立器。

此處刻意對齊 `/mnt/8tb_hdd/under115a/4c-yolo/run_4ch.py`：前三個
channel 是 RGB，第四個 channel 是逐幀 min-max normalization 後乘 1.5 的
temporal pixel-change map。Ultralytics 8.4.41 不會替 4-channel NumPy input
執行 BGR->RGB，因此必須在送入 ``model.predict`` 前明確轉換。
"""

import cv2
import numpy as np


PIXEL_CHANGE_GAIN = 1.5


def compute_pixel_change_map(prev_frame, curr_frame):
    """依 reference inference 建立 ``uint8`` temporal change map。"""
    if prev_frame is None:
        return np.zeros(curr_frame.shape[:2], dtype=np.uint8)

    diff = cv2.absdiff(prev_frame, curr_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(diff_gray * PIXEL_CHANGE_GAIN, 0, 255).astype(np.uint8)


def build_litter_model_input(prev_frame, curr_frame):
    """回傳 reference-compatible ``H x W x 4`` RGB+change ``uint8`` image。"""
    rgb_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
    change_map = compute_pixel_change_map(prev_frame, curr_frame)
    return np.dstack((rgb_frame, change_map))
