# -*- coding: utf-8 -*-
"""litter tracker 的純輔助函式:反追蹤環境變數讀取 + 彈道軌跡擬合
(airborne prefix / 最小二乘拋物線 / 外插)。從 litter_tracker.py 抽出,
類別本體與所有 tracker 常數不動。"""
import math
import os

import numpy as np


def _backtrack_int_env(name, default):
    # 反追蹤調參用：讀整數環境變數，格式錯誤時回退預設。
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _backtrack_float_env(name, default):
    # 反追蹤調參用：讀浮點環境變數，格式錯誤時回退預設。
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _trajfit_airborne_prefix(points, frames, bounce_eps=1.5, rest_eps=0.6):
    # 軌跡反演第一步:取 litter 質心軌跡的「空中段」前綴。
    # 落地反彈(vy 由明顯向下轉明顯向上)或靜止(連續兩步近零速)之後的點會污染拋物線
    # 方向,必須切掉。history 與 history_frames 上限不同(fps-scaled vs 常數),從尾端對齊。
    n = min(len(points), len(frames))
    if n < 2:
        return [], []
    pts = [(float(p[0]), float(p[1])) for p in points[-n:]]
    fs = [int(f) for f in frames[-n:]]
    cut = n
    prev_vy = None
    rest_run = 0
    for i in range(1, n):
        dt = max(fs[i] - fs[i - 1], 1)
        vx = (pts[i][0] - pts[i - 1][0]) / dt
        vy = (pts[i][1] - pts[i - 1][1]) / dt
        if math.hypot(vx, vy) < rest_eps:
            rest_run += 1
            if rest_run >= 2:
                cut = max(i - 1, 2)
                break
        else:
            rest_run = 0
        if prev_vy is not None and prev_vy > bounce_eps and vy < -bounce_eps:
            cut = i          # pts[i-1] 是落地點(保留);pts[i] 已反彈
            break
        prev_vy = vy
    return pts[:cut], fs[:cut]


def _trajfit_fit_ballistic(pts, fs, sigma_floor=2.0, min_pts=3,
                           min_speed=2.0, g_min=-0.05, g_max=60.0):
    # 軌跡反演第二步:空中段最小二乘拋物線 x(t)=x0+vx·t, y(t)=y0+vy·t+0.5g·t²(t=幀偏移)。
    # 固定機位 → 影像重力方向恆定(y 向下為正),曲率 g 應非負;負曲率=非拋射 → None(fallback)。
    # ≥5 點時剔除一個最大殘差點重擬合(RANSAC-lite,抗單點偵測離群)。
    # 回傳含擬合殘差 sigma:下游門檻由 sigma 自動縮放,取代手調像素常數。
    n = min(len(pts), len(fs))
    if n < max(int(min_pts), 3):
        return None
    t = np.asarray(fs[:n], dtype=np.float64)
    t = t - t[0]
    xs = np.asarray([p[0] for p in pts[:n]], dtype=np.float64)
    ys = np.asarray([p[1] for p in pts[:n]], dtype=np.float64)
    span = max(float(t[-1]), 1.0)
    if math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0])) / span < min_speed:
        return None             # 近靜止 = 放置非拋擲,交給 holding 邏輯
    try:
        cx = np.polyfit(t, xs, 1)
        cy = np.polyfit(t, ys, 2)
        if n >= 5:
            res2 = (xs - np.polyval(cx, t)) ** 2 + (ys - np.polyval(cy, t)) ** 2
            keep = np.ones(n, dtype=bool)
            keep[int(np.argmax(res2))] = False
            cx = np.polyfit(t[keep], xs[keep], 1)
            cy = np.polyfit(t[keep], ys[keep], 2)
            t_u, xs_u, ys_u = t[keep], xs[keep], ys[keep]
        else:
            t_u, xs_u, ys_u = t, xs, ys
    except (np.linalg.LinAlgError, ValueError):
        return None
    g = 2.0 * float(cy[0])
    if not (g_min <= g <= g_max):
        return None
    res = np.hypot(xs_u - np.polyval(cx, t_u), ys_u - np.polyval(cy, t_u))
    sigma = max(float(np.sqrt(np.mean(res * res))), float(sigma_floor))
    return {'cx': cx, 'cy': cy, 't0_frame': int(fs[0]), 'sigma': sigma, 'g': g}


def _trajfit_point_at(fit, frame_index):
    # 軌跡反演第三步:沿擬合拋物線外插到任意幀(向後外插 = 釋放事件搜索)。
    t = float(int(frame_index) - fit['t0_frame'])
    return (float(np.polyval(fit['cx'], t)), float(np.polyval(fit['cy'], t)))
