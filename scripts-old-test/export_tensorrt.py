# -*- coding: utf-8 -*-
# TensorRT 匯出工具：將 YOLO actor 與 RTDETR litter 權重匯出成對應 batch 的 .engine。
import argparse
import os
from pathlib import Path

import torch
from ultralytics import RTDETR, YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 預設匯出 batch 權重資料夾；main.py batch 8 會優先尋找 *_b8.engine。
BBOX_MODEL = PROJECT_ROOT / "modules_weight/batch" / "best-yolo-seg_v3.pt"
TRASH_MODEL = PROJECT_ROOT / "modules_weight/batch" / "best-rtdetr-seg.pt"


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
    parser.add_argument("--batch", type=int, default=8, choices=(1, 8), help="TensorRT optimization batch size")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace size in GiB")
    parser.add_argument("--opset", type=int, default=19, help="ONNX opset for TensorRT export")
    parser.add_argument("--imgsz", type=int, default=None, help="Override export image size for both models")
    parser.add_argument("--bbox-imgsz", type=int, default=None, help="Override export image size for bbox model")
    parser.add_argument("--trash-imgsz", type=int, default=None, help="Override export image size for trash model")
    parser.add_argument("--bbox-model", type=Path, default=BBOX_MODEL, help="YOLO-seg actor model path")
    parser.add_argument("--trash-model", type=Path, default=TRASH_MODEL, help="RTDETR litter model path")
    parser.add_argument("--skip-bbox", action="store_true", help="Do not export the bbox/actor model")
    parser.add_argument("--skip-trash", action="store_true", help="Do not export the trash/litter model")
    parser.add_argument("--fallback-imgsz", type=int, default=640, help="Fallback image size if a checkpoint has no imgsz")
    parser.add_argument("--dynamic", dest="dynamic", action="store_true", default=None,
                        help="Build dynamic-shape TensorRT engines")
    parser.add_argument("--static", dest="dynamic", action="store_false",
                        help="Build fixed-shape TensorRT engines")
    parser.add_argument("--no-half", dest="half", action="store_false", help="Build FP32 engines instead of FP16")
    parser.add_argument("--force", action="store_true", help="Rebuild existing .engine files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose TensorRT logs")
    parser.set_defaults(half=True)
    return parser.parse_args()


def main():
    # 主流程：檢查 CUDA 後依參數匯出 bbox/trash engine。
    args = parse_args()
    if args.dynamic is None:
        args.dynamic = args.batch > 1
    _check_cuda(args.device)
    outputs = []
    if not args.skip_bbox:
        bbox_args = argparse.Namespace(**vars(args))
        bbox_args.imgsz = args.bbox_imgsz or args.imgsz
        outputs.append(_export_model("bbox", args.bbox_model, YOLO, "segment", bbox_args))
    if not args.skip_trash:
        trash_args = argparse.Namespace(**vars(args))
        trash_args.imgsz = args.trash_imgsz or args.imgsz
        outputs.append(_export_model("trash", args.trash_model, RTDETR, None, trash_args))
    if not outputs:
        raise RuntimeError("Nothing to export: both --skip-bbox and --skip-trash were set.")
    print("Export complete:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
