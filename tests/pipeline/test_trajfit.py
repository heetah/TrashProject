# -*- coding: utf-8 -*-
"""pipeline.litter.trajfit 的 GPU-free call-level 測試 + 從 litter_tracker 抽出後的 re-export。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from pipeline.litter import trajfit


def test_backtrack_env_helpers(monkeypatch):
    monkeypatch.delenv("BT_X", raising=False)
    assert trajfit._backtrack_int_env("BT_X", 7) == 7
    assert trajfit._backtrack_float_env("BT_X", 1.5) == 1.5
    monkeypatch.setenv("BT_X", "12")
    assert trajfit._backtrack_int_env("BT_X", 7) == 12
    monkeypatch.setenv("BT_X", "bad")
    assert trajfit._backtrack_int_env("BT_X", 7) == 7  # 格式錯 → 回退


def test_ballistic_fit_on_parabola():
    # y = 2 t^2, x = 4 t → 向下曲率 g = 2*cy[0] = 4 (>0),應成功擬合。
    frames = list(range(6))
    pts = [(4.0 * t, 2.0 * t * t) for t in frames]
    fit = trajfit._trajfit_fit_ballistic(pts, frames)
    assert fit is not None
    assert fit["g"] > 0
    # 外插回 t0 應約等於起點。
    x0, y0 = trajfit._trajfit_point_at(fit, 0)
    assert abs(x0 - 0.0) < 1.0 and abs(y0 - 0.0) < 1.0


def test_airborne_prefix_cuts_after_rest():
    # 前段等速上升,後段近靜止 → prefix 應切在靜止之前。
    pts = [(0, 0), (2, -4), (4, -8), (6, -12), (6, -12), (6, -12)]
    frames = [0, 1, 2, 3, 4, 5]
    ppts, pfs = trajfit._trajfit_airborne_prefix(pts, frames)
    assert len(ppts) == len(pfs)
    assert len(ppts) < len(pts)  # 有切掉尾端靜止段


def test_reexport_through_litter_tracker_shim():
    # litter_tracker 把 helper import 回自身命名空間;舊路徑 shim 仍解析到同一物件。
    import litterTracker
    from pipeline import litter_tracker
    assert litter_tracker._trajfit_fit_ballistic is trajfit._trajfit_fit_ballistic
    assert litterTracker._trajfit_point_at is trajfit._trajfit_point_at
    # 類別與常數仍在原處。
    assert hasattr(litter_tracker, "GlobalLitterTracker")
    assert litter_tracker.REF_FPS == 10.0
