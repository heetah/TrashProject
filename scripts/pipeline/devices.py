# -*- coding: utf-8 -*-
"""actor / litter 偵測器的裝置(GPU/CPU)與 half precision 選擇。

原本住在 detect.py,但 examine.py(infra 層:模型載入/warmup)反過來 import
detect.BBOX_DEVICE 造成「基礎設施相依偵測器」的耦合。抽到此中性模組後,detect 與
examine 都改從這裡取用,依相依方向恢復正常(infra 不再指向偵測器)。

注意:action.py 與 plate.py 各有自己語意不同的 device helper(ACTION_DEVICE /
車牌),簽章不同,本次不合併,僅集中 actor/trash 這組。
"""
import os

try:
    import torch
except Exception:
    torch = None


def _select_device(env_name):
    # 依環境變數選 GPU/CPU;若使用者要求 CUDA 但不可用,自動回退 CPU。
    requested = os.environ.get(env_name) or os.environ.get("YOLO_DEVICE")
    if requested:
        requested_str = str(requested).strip().lower()
        wants_cuda = requested_str.isdigit() or requested_str.startswith("cuda")
        if wants_cuda:
            if torch is not None and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return int(requested_str) if requested_str.isdigit() else requested
            print(f"{env_name}={requested} requested but CUDA is unavailable; fallback to CPU")
            return "cpu"
        return requested

    if torch is not None and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return 0
    return "cpu"


def _can_use_half(device):
    # half precision 只在 CUDA 裝置上啟用,避免 CPU/MPS 不支援。
    if torch is None or not torch.cuda.is_available():
        return False
    return str(device).lower() not in ("cpu", "mps")


BBOX_DEVICE = _select_device("BBOX_DEVICE")
TRASH_DEVICE = _select_device("TRASH_DEVICE")
BBOX_HALF = _can_use_half(BBOX_DEVICE)
TRASH_HALF = _can_use_half(TRASH_DEVICE)
