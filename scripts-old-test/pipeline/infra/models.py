# -*- coding: utf-8 -*-
"""模型/TensorRT engine 探測、候選順序、載入與 warmup。"""
from pathlib import Path

import numpy as np

from .constants import SUPPORTED_BATCH_SIZES
from pipeline.devices import BBOX_DEVICE, BBOX_HALF, TRASH_DEVICE, TRASH_HALF

# torch 是選用依賴:若不可用,TensorRT engine 檢查會自動略過。
try:
    import torch
except Exception:
    torch = None


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
