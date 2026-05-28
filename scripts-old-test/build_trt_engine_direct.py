#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 直接用 TRT Python API 從 ONNX 建 engine，繞過 ultralytics virtualMemoryBuffer 問題。
# 在 ONNX parse 之前設定 CUDA_VISIBLE_DEVICES，讓 TRT 在乾淨的 CUDA context 上建置。
#
# 輸出格式與 ultralytics 相容：
#   [4-byte metadata JSON 長度 (little-endian signed)] [metadata JSON] [TRT serialized engine]
# 這樣 ultralytics TensorRTBackend.load_model() 才能正確讀出 names / channels / imgsz。
import argparse
import json
import os
import sys
from pathlib import Path

# ── 在 import tensorrt 前設定，讓 CUDA context 只綁定目標 GPU ──────────────────
def _setup_cuda_device(device_index: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX   = str(PROJECT_ROOT / "modules_weight" / "best-rtdetr-4c.onnx")
DEFAULT_ENGINE = str(PROJECT_ROOT / "modules_weight" / "best-rtdetr-4c_b8.engine")

# ultralytics engine metadata 格式：
# [4-byte little-endian signed length] [UTF-8 JSON] [raw TRT serialized bytes]
# 載入時 TensorRTBackend.load_model() 讀出 JSON 並呼叫 apply_metadata()，
# 讓 model.names / model.channels / model.imgsz 全部正確。


def _read_onnx_metadata(onnx_path: str) -> dict:
    """從 ONNX 檔案的 metadata_props 讀取 names / imgsz / task / stride / nc。
    ultralytics 在 export 時把這些寫入 ONNX；直接用 TRT API 建 engine 時需自己補回。
    """
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        props = {p.key: p.value for p in m.metadata_props}
    except Exception as e:
        print(f"[meta] Warning: could not read ONNX metadata from {onnx_path}: {e}")
        props = {}

    # ── names ──────────────────────────────────────────────────────────────────
    # ONNX 把 names 存成 Python literal string，例如 "{0: 'litter'}"。
    # JSON 裡要存 dict（JSON 不允許 int key，存成 str key 也可以，apply_metadata 會轉回 int）。
    names_raw = props.get("names", "{0: 'litter'}")
    try:
        import ast
        names_dict = ast.literal_eval(names_raw)        # {0: 'litter'}
        # 轉成 JSON 相容 str-key dict
        names_json = {str(k): str(v) for k, v in names_dict.items()}
    except Exception:
        names_json = {"0": "litter"}

    # ── imgsz ──────────────────────────────────────────────────────────────────
    imgsz_raw = props.get("imgsz", "[1280, 1280]")
    try:
        imgsz = ast.literal_eval(imgsz_raw)
    except Exception:
        imgsz = [1280, 1280]

    # ── nc ─────────────────────────────────────────────────────────────────────
    nc = len(names_json)

    return {
        "names": names_json,
        "nc": nc,
        "imgsz": imgsz,
        "task": props.get("task", "detect"),
        "stride": int(props.get("stride", 32)),
    }


def _write_engine_with_metadata(serialized_bytes: bytes, engine_path: str,
                                 metadata: dict) -> None:
    """以 ultralytics 相容格式寫出 engine 檔案。
    格式：[4-byte metadata length (little-endian signed)] [metadata JSON] [TRT bytes]
    """
    meta = json.dumps(metadata)
    meta_len_bytes = len(meta).to_bytes(4, byteorder="little", signed=True)
    with open(engine_path, "wb") as f:
        f.write(meta_len_bytes)
        f.write(meta.encode("utf-8"))
        f.write(serialized_bytes)
    size_mb = Path(engine_path).stat().st_size / (1024 * 1024)
    print(f"[TRT] Engine saved (with ultralytics metadata): {engine_path}  ({size_mb:.1f} MB)")


def build_engine(onnx_path: str, engine_path: str,
                 workspace_gib: float = 4.0, fp16: bool = True, verbose: bool = False,
                 channels: int = 4, batch: int = 8):
    import tensorrt as trt  # import 在設定 env 之後

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, "")

    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, trt_logger)
    config  = builder.create_builder_config()

    # ── workspace 限制（減少 tactic 評估時的暫存記憶體壓力）──────────────────────
    workspace_bytes = int(workspace_gib * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    print(f"[TRT] workspace limit: {workspace_gib} GiB")

    # ── FP16 ─────────────────────────────────────────────────────────────────────
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] FP16 enabled")
        else:
            print("[TRT] Warning: FP16 not supported, using FP32")

    # ── 減少 tactic 來源，避免 virtualMemoryBuffer 爆炸性分配 ───────────────────
    # 只保留 CUBLAS + CUBLAS_LT；移除 CUDNN / EDGE_MASK_CONVOLUTIONS 等高記憶體路徑
    try:
        tactic_mask = (
            1 << int(trt.TacticSource.CUBLAS) |
            1 << int(trt.TacticSource.CUBLAS_LT)
        )
        config.set_tactic_sources(tactic_mask)
        print("[TRT] Tactic sources: CUBLAS + CUBLAS_LT only")
    except Exception as e:
        print(f"[TRT] set_tactic_sources skipped: {e}")

    # ── parse ONNX ────────────────────────────────────────────────────────────────
    onnx_path = str(onnx_path)
    print(f"[TRT] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"  Parse error {i}: {parser.get_error(i)}")
        raise RuntimeError("ONNX parsing failed")

    print(f"[TRT] Network: {network.num_layers} layers")
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input [{i}]: {inp.name}  shape={inp.shape}  dtype={inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output[{i}]: {out.name}  shape={out.shape}  dtype={out.dtype}")

    # ── build ─────────────────────────────────────────────────────────────────────
    print("[TRT] Building serialized network (may take 5-15 min)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT build_serialized_network returned None")

    # ── 讀 ONNX metadata，補回 names / imgsz / task，並加上 channels ──────────────
    onnx_meta = _read_onnx_metadata(onnx_path)
    engine_metadata = {
        **onnx_meta,
        "channels": channels,   # 4c 模型必須正確寫入，ultralytics 才會用 4ch 前處理
        "batch": batch,
    }
    print(f"[TRT] Embedding metadata: {json.dumps(engine_metadata, ensure_ascii=False)}")

    engine_path = str(engine_path)
    _write_engine_with_metadata(bytes(serialized), engine_path, engine_metadata)
    return engine_path


def main():
    ap = argparse.ArgumentParser(
        description="Build TRT engine directly from ONNX (bypass ultralytics virtualMemoryBuffer)"
    )
    ap.add_argument("--onnx",      default=DEFAULT_ONNX,   help="Input ONNX path")
    ap.add_argument("--engine",    default=DEFAULT_ENGINE,  help="Output .engine path")
    ap.add_argument("--workspace", type=float, default=4.0, help="TRT workspace GiB (default 4)")
    ap.add_argument("--device",    type=int,   default=1,   help="CUDA device index (default 1 = more free VRAM)")
    ap.add_argument("--no-half",   dest="fp16", action="store_false", help="FP32 instead of FP16")
    ap.add_argument("--force",     action="store_true", help="Overwrite existing engine")
    ap.add_argument("--verbose",   action="store_true", help="TRT verbose log")
    ap.add_argument("--channels",  type=int,   default=4,   help="Number of input channels embedded in metadata (default 4 for 4c model)")
    ap.add_argument("--batch",     type=int,   default=8,   help="Batch size embedded in metadata (default 8)")
    ap.set_defaults(fp16=True)
    args = ap.parse_args()

    engine_path = Path(args.engine)
    if engine_path.exists() and not args.force:
        print(f"Engine already exists: {engine_path}")
        print("Use --force to rebuild.")
        sys.exit(0)

    if not Path(args.onnx).exists():
        print(f"ERROR: ONNX not found: {args.onnx}")
        sys.exit(1)

    # CUDA device 設定必須在 import tensorrt 前完成
    _setup_cuda_device(args.device)
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING','(unset)')}")

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        workspace_gib=args.workspace,
        fp16=args.fp16,
        verbose=args.verbose,
        channels=args.channels,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()
