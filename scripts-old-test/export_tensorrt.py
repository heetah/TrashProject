# -*- coding: utf-8 -*-
# TensorRT 匯出工具：將 YOLO actor 與 RTDETR litter 權重匯出成對應 batch 的 .engine。
import argparse
import os
from pathlib import Path

import cv2
import torch
from ultralytics import RTDETR, YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 匯出預設需和 main.py 對齊：batch 1 用 root 權重，batch N 用 modules_weight/batch。
SUPPORTED_BATCH_SIZES = (1, 2, 4, 8, 16)
ROOT_BBOX_MODEL = PROJECT_ROOT / "modules_weight" / "best-yolo-seg_v3.pt"
ROOT_TRASH_MODEL = PROJECT_ROOT / "modules_weight" / "best-rtdetr-seg.pt"
BATCH_BBOX_MODEL = PROJECT_ROOT / "modules_weight" / "batch" / "best-yolo-seg_v3.pt"
BATCH_TRASH_MODEL = PROJECT_ROOT / "modules_weight" / "batch" / "best-rtdetr-seg.pt"
DEFAULT_SMOKE_VIDEO = PROJECT_ROOT / "resources" / "resize.mp4"


def _device_index(device):
    # 將 cuda:0 / 0 這類格式統一轉成整數 device index。
    value = str(device).strip().lower()
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    return int(value)


def _check_cuda(device):
    # TensorRT engine 必須在目標 GPU 上建置；RTX 4090 以外只警告不阻擋。
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. TensorRT engine export must run on the target RTX 4090 host.")
    index = _device_index(device)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {index} is unavailable; visible device count is {torch.cuda.device_count()}.")
    name = torch.cuda.get_device_name(index)
    if "4090" not in name:
        print(f"Warning: exporting on {name}, not RTX 4090. TensorRT engines should be built on the target GPU.")
    print(f"Using CUDA device {index}: {name}")


def _default_imgsz(model, fallback):
    # 優先讀取 checkpoint 內的 imgsz 設定，沒有才使用 fallback。
    overrides = getattr(model, "overrides", {}) or {}
    imgsz = overrides.get("imgsz", fallback)
    if isinstance(imgsz, (list, tuple)):
        return imgsz[0] if len(imgsz) == 1 else list(imgsz)
    return int(imgsz)


def _engine_path_for_batch(model_path, batch_size):
    # batch 1 產生同名 .engine；batch N 產生 *_bN.engine。
    model_path = Path(model_path)
    if int(batch_size) <= 1:
        return model_path.with_suffix(".engine")
    return model_path.with_name(f"{model_path.stem}_b{int(batch_size)}.engine")


def _temporary_backup_path(path):
    # 匯出期間暫存舊 engine，避免 Ultralytics 固定輸出名稱覆蓋錯檔。
    return path.with_name(f"{path.name}.pre_batch_export_{os.getpid()}")


def _resolve_default_model_paths(args):
    # CLI 未指定模型時，依 batch 選擇 main.py 實際會載入的權重。
    if args.bbox_model is None:
        args.bbox_model = BATCH_BBOX_MODEL if int(args.batch) > 1 else ROOT_BBOX_MODEL
    if args.trash_model is None:
        args.trash_model = BATCH_TRASH_MODEL if int(args.batch) > 1 else ROOT_TRASH_MODEL


def _parse_frame_indices(value):
    # 支援 "70,75" 或重複空白；用於 export 後 engine smoke。
    indices = []
    for item in str(value or "").replace(",", " ").split():
        indices.append(int(item))
    return indices or [70, 75]


def _read_smoke_frames(video_path, frame_indices):
    video_path = Path(video_path).expanduser()
    if not video_path.exists():
        print(f"Warning: smoke video not found: {video_path}; skip engine smoke test.")
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: unable to open smoke video: {video_path}; skip engine smoke test.")
        return []

    frames = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        print(f"Warning: no smoke frames could be read from {video_path}; skip engine smoke test.")
    return frames


def _box_counts(results, keep_count):
    counts = []
    for result in list(results)[:keep_count]:
        boxes = getattr(result, "boxes", None)
        counts.append(0 if boxes is None else len(boxes))
    return counts


def _smoke_test_engine(label, engine_path, loader, task, args, conf):
    # 使用專案 regression frame 驗證 engine 不是「可載入但輸出全 0」的壞檔。
    if not args.smoke_test:
        return

    engine_path = Path(engine_path)
    if engine_path.suffix != ".engine":
        return

    frames = _read_smoke_frames(args.smoke_video, _parse_frame_indices(args.smoke_frames))
    if not frames:
        return

    model = loader(str(engine_path), task=task) if task else loader(str(engine_path))
    batch = max(int(args.batch or 1), 1)
    counts = []
    if batch <= 1:
        for frame in frames:
            results = model.predict(
                frame,
                conf=conf,
                device=args.device,
                half=args.half,
                verbose=False,
            )
            counts.extend(_box_counts(results, 1))
    else:
        for start in range(0, len(frames), batch):
            chunk = list(frames[start:start + batch])
            keep_count = len(chunk)
            if len(chunk) < batch:
                chunk.extend([chunk[-1]] * (batch - len(chunk)))
            results = model.predict(
                chunk,
                conf=conf,
                device=args.device,
                half=args.half,
                verbose=False,
            )
            counts.extend(_box_counts(results, keep_count))

    print(f"{label}: smoke counts on {Path(args.smoke_video).name} frames {args.smoke_frames}: {counts}")
    if not any(count > 0 for count in counts):
        raise RuntimeError(
            f"{label}: exported engine produced zero boxes on smoke frames; "
            "do not use this engine for inference."
        )


def _export_model(label, model_path, loader, task, args):
    # 單一模型匯出流程：載入權重、決定 imgsz、執行 export、搬到目標 engine path。
    model_path = Path(model_path).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    default_engine_path = model_path.with_suffix(".engine")
    engine_path = _engine_path_for_batch(model_path, args.batch)
    if engine_path.exists() and not args.force:
        print(f"{label}: {engine_path} exists; use --force to rebuild.")
        return engine_path

    model = loader(str(model_path), task=task) if task else loader(str(model_path))
    imgsz = args.imgsz or _default_imgsz(model, args.fallback_imgsz)
    print(
        f"{label}: exporting {model_path} -> {engine_path} "
        f"(imgsz={imgsz}, batch={args.batch}, half={args.half}, dynamic={args.dynamic})"
    )

    backup_path = None
    try:
        # batch engine 需要改名時，先備份同名 engine，避免被 export 中途覆蓋。
        if engine_path != default_engine_path and default_engine_path.exists():
            backup_path = _temporary_backup_path(default_engine_path)
            default_engine_path.rename(backup_path)

        exported = model.export(
            format="engine",
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            half=args.half,
            dynamic=args.dynamic,
            simplify=False,
            workspace=args.workspace,
            opset=args.opset,
            verbose=args.verbose,
        )
        exported_path = Path(exported)
        if not exported_path.exists():
            raise RuntimeError(f"{label}: export reported {exported_path}, but the file does not exist.")

        if exported_path != engine_path:
            if engine_path.exists():
                if not args.force:
                    raise FileExistsError(f"{engine_path} exists; use --force to replace it.")
                engine_path.unlink()
            exported_path.rename(engine_path)

        print(f"{label}: exported {engine_path}")
        return engine_path
    except Exception:
        if backup_path is not None and default_engine_path.exists():
            default_engine_path.unlink()
        raise
    finally:
        if backup_path is not None and backup_path.exists() and not default_engine_path.exists():
            backup_path.rename(default_engine_path)


def parse_args():
    # CLI 參數：控制 device、batch、imgsz、FP16、dynamic shape 與是否重建既有 engine。
    parser = argparse.ArgumentParser(description="Export project YOLO/RTDETR weights to TensorRT engines.")
    parser.add_argument("--device", default="0", help="CUDA device index, e.g. 0 or cuda:0")
    parser.add_argument("--batch", type=int, default=8, choices=SUPPORTED_BATCH_SIZES,
                        help="TensorRT optimization batch size")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace size in GiB")
    parser.add_argument("--opset", type=int, default=19, help="ONNX opset for TensorRT export")
    parser.add_argument("--imgsz", type=int, default=None, help="Override export image size for both models")
    parser.add_argument("--bbox-imgsz", type=int, default=None, help="Override export image size for bbox model")
    parser.add_argument("--trash-imgsz", type=int, default=None, help="Override export image size for trash model")
    parser.add_argument("--bbox-model", type=Path, default=None, help="YOLO-seg actor model path")
    parser.add_argument("--trash-model", type=Path, default=None, help="RTDETR litter model path")
    parser.add_argument("--skip-bbox", action="store_true", help="Do not export the bbox/actor model")
    parser.add_argument("--skip-trash", action="store_true", help="Do not export the trash/litter model")
    parser.add_argument("--fallback-imgsz", type=int, default=640, help="Fallback image size if a checkpoint has no imgsz")
    parser.add_argument("--dynamic", dest="dynamic", action="store_true", default=False,
                        help="Build dynamic-shape TensorRT engines")
    parser.add_argument("--static", dest="dynamic", action="store_false",
                        help="Build fixed-shape TensorRT engines")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Build FP32 engines instead of FP16")
    parser.add_argument("--smoke-test", dest="smoke_test", action="store_true", default=True,
                        help="Smoke-test exported engines on project regression frames")
    parser.add_argument("--no-smoke-test", dest="smoke_test", action="store_false",
                        help="Skip post-export engine smoke test")
    parser.add_argument("--smoke-video", type=Path, default=DEFAULT_SMOKE_VIDEO,
                        help="Video used for post-export engine smoke test")
    parser.add_argument("--smoke-frames", default="70,75",
                        help="Comma/space separated frame indices for post-export smoke test")
    parser.add_argument("--smoke-bbox-conf", type=float, default=0.45,
                        help="BBOX confidence threshold for smoke test")
    parser.add_argument("--smoke-trash-conf", type=float, default=0.4,
                        help="RTDETR confidence threshold for smoke test")
    parser.add_argument("--force", action="store_true", help="Rebuild existing .engine files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose TensorRT logs")
    parser.set_defaults(half=True)
    return parser.parse_args()


def main():
    # 主流程：檢查 CUDA 後依參數匯出 bbox/trash engine。
    args = parse_args()
    _resolve_default_model_paths(args)
    _check_cuda(args.device)
    outputs = []
    if not args.skip_bbox:
        bbox_args = argparse.Namespace(**vars(args))
        bbox_args.imgsz = args.bbox_imgsz or args.imgsz
        bbox_engine = _export_model("bbox", args.bbox_model, YOLO, "segment", bbox_args)
        _smoke_test_engine("bbox", bbox_engine, YOLO, "segment", bbox_args, args.smoke_bbox_conf)
        outputs.append(bbox_engine)
    if not args.skip_trash:
        trash_args = argparse.Namespace(**vars(args))
        trash_args.imgsz = args.trash_imgsz or args.imgsz
        trash_engine = _export_model("trash", args.trash_model, RTDETR, None, trash_args)
        _smoke_test_engine("trash", trash_engine, RTDETR, None, trash_args, args.smoke_trash_conf)
        outputs.append(trash_engine)
    if not outputs:
        raise RuntimeError("Nothing to export: both --skip-bbox and --skip-trash were set.")
    print("Export complete:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
