# -*- coding: utf-8 -*-
# 主流程入口：負責模型載入、影片讀寫、逐幀/批次推理、輸出壓縮與耗時統計。
import cv2
import numpy as np
import os
import subprocess
from tqdm import tqdm
from pathlib import Path
import argparse
from collections import defaultdict, deque

from ultralytics import YOLO
from ultralytics import RTDETR

from detect import BBOX_DEVICE, BBOX_HALF, TRASH_DEVICE, TRASH_HALF, detect, detect_batch
from litterTracker import GlobalLitterTracker
from action import STGCNActionModule
from licensePlate import disable_license_plate_models, preload_license_plate_models, wait_for_plate_jobs
from timeUtils import PipelineProfiler

# 畫面標註顏色設定：各類別在輸出影片中的 bbox 顏色。
COLORS = {
    'litter': (128, 0, 128), # 紫色
    'person': (255, 200, 128), # 淺藍
    'vehicle': (0, 255, 0), # 綠色
    'scooter': (0, 255, 255) # 黃色
}

# 預設模型路徑：batch 1 使用一般權重；batch 8 使用 batch 匯出/訓練資料夾中的權重。
POSE_MODEL_PATH = 'modules_weight/yolo26x-pose.pt'
STGCN_WEIGHT_PATH = 'modules_weight/best_acc_top1_epoch_13.pth'
STGCN_CONFIG_PATH = 'mmaction2/configs/skeleton/stgcnpp/custom_trash_stgcnpp.py'
# MODEL_BBOX_PATH = 'modules_weight/yolo26x-seg.pt'
MODEL_BBOX_PATH = 'modules_weight/best-yolo-seg_v3.pt'
MODEL_TRASH_PATH = 'modules_weight/best-rtdetr-seg.pt'
MODEL_BBOX_PATH_BATCH_2 = 'modules_weight/batch/best-yolo-seg_v3.pt'
MODEL_TRASH_PATH_BATCH_2 = 'modules_weight/batch/best-rtdetr-seg.pt'

# torch 是選用依賴：若不可用，TensorRT engine 檢查會自動略過。
try:
    import torch
except Exception:
    torch = None


def _set_env_if_present(name, value):
    # CLI 有傳值才寫入環境變數，避免覆蓋使用者原本的設定。
    if value is not None:
        os.environ[name] = str(value)


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


def _model_path_candidates(model_path, prefer_engine=True, batch_size=1):
    # 建立模型候選順序：優先 engine，失敗或不存在時回退原始權重。
    path = Path(model_path)
    candidates = []

    if prefer_engine and path.suffix == ".pt":
        engine_path = _engine_path_for_batch(path, batch_size)
        if engine_path.exists():
            if _can_try_tensorrt_engine(engine_path):
                candidates.append(str(engine_path))
        else:
            print(f"Warning: TensorRT engine not found for {path} at batch={batch_size}; using PyTorch weights.")

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
                warmup_func(model)
            print(f"{label}: loaded and warmed up {model_path}")
            return model, model_path
        except Exception as exc:
            last_error = exc
            print(f"{label}: failed to initialize {model_path}: {exc}")

    raise RuntimeError(f"{label}: all model candidates failed: {candidates}") from last_error


def _batched_dummy_frame(batch_size):
    # warmup 使用假 frame；batch 模式需傳入 list，才能讓 backend 建立正確 batch shape。
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    if int(batch_size) <= 1:
        return dummy_frame
    return [dummy_frame.copy() for _ in range(int(batch_size))]


def _default_model_paths_for_batch(batch_size):
    # 使用者未手動指定模型時，依 --batch 自動切換成對應權重。
    if int(batch_size) == 2:
        return MODEL_BBOX_PATH_BATCH_2, MODEL_TRASH_PATH_BATCH_2
    return MODEL_BBOX_PATH, MODEL_TRASH_PATH


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


def _warmup_trash_model(model, batch_size=1):
    # RTDETR litter model warmup：確保垃圾模型在正式迴圈前已初始化。
    dummy_source = _batched_dummy_frame(batch_size)
    model.predict(
        dummy_source,
        conf=0.01,
        device=TRASH_DEVICE,
        half=TRASH_HALF,
        verbose=False,
    )


def _foreground_mask(back_sub, frame):
    # 背景減除前先縮圖，降低每幀前景遮罩成本；輸出再放回原解析度。
    scale_factor = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    small_mask = back_sub.apply(small_frame, learningRate=0.005)
    _, small_mask = cv2.threshold(small_mask, 254, 255, cv2.THRESH_BINARY)
    return cv2.resize(small_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


def _read_frame_batch(cap, back_sub, batch_size, profiler):
    # 一次讀取 batch_size 幀，同步產生每幀的前景遮罩，供後續 motion filter 使用。
    frames = []
    fg_masks = []
    for _ in range(int(batch_size)):
        with profiler.time_block("frame.read"):
            ret, frame = cap.read()
        if not ret:
            break

        with profiler.time_block("frame.foreground_mask"):
            fg_mask = _foreground_mask(back_sub, frame)

        frames.append(frame)
        fg_masks.append(fg_mask)

    return frames, fg_masks


if __name__ == "__main__":
    # CLI 參數：控制模型、batch 模式、STGCN、車牌辨識與偵測信心門檻。
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="file name in resources/", default="TThrow.mp4")
    parser.add_argument(
        "--disable-speed-filter",
        action="store_true",
        help="Disable vehicle speed filtering based on calculate_speed"
    )
    parser.add_argument("--pose-model", default=POSE_MODEL_PATH, help="YOLO pose model path")
    parser.add_argument("--stgcn-weight", default=STGCN_WEIGHT_PATH, help="STGCN++ checkpoint path")
    parser.add_argument("--stgcn-config", default=STGCN_CONFIG_PATH, help="STGCN++ config path")
    parser.add_argument("--action-threshold", type=float, default=0.5, help="littering action threshold")
    parser.add_argument("--action-window", type=int, default=30, help="STGCN sequence window size")
    parser.add_argument("--action-device", default=os.environ.get("ACTION_DEVICE"), help="ACTION_DEVICE override, e.g. cuda:0 or cpu")
    parser.add_argument("--action-predict-interval", type=int, default=None, help="run STGCN every N frames after the sequence window is full")
    parser.add_argument("--action-pose-imgsz", type=int, default=None, help="optional YOLO pose imgsz override")
    parser.add_argument("--disable-action", action="store_true", help="Disable STGCN action recognition; enabled by default")
    parser.add_argument("--yolo-seg-frame-skip", type=int, default=2, help="run YOLO-seg actor tracking every N frames")
    parser.add_argument("--bbox-model", default=None, help="YOLO-seg actor model path")
    parser.add_argument("--trash-model", default=None, help="RTDETR litter model path")
    parser.add_argument("--no-engine", action="store_true", help="Use .pt weights even when a sibling .engine exists")
    parser.add_argument("--batch", "--batch-size", dest="batch_size", type=int, default=1, choices=(1, 2),
                        help="inference mode: 1 uses normal single-frame weights, 2 uses batch-2 weights")
    parser.add_argument("--bbox-conf", type=float, default=0.45, help="YOLO-seg actor confidence threshold")
    parser.add_argument("--trash-conf", type=float, default=0.4, help="RTDETR litter confidence threshold")
    parser.add_argument("--disable-plate", action="store_true", help="Disable license plate detection/OCR for faster litter-only processing")
    parser.add_argument("--skip-plate-preload", action="store_true", help="Do not preload license plate detector/OCR models")
    args = parser.parse_args()

    # 資源句柄與計數器集中管理，finally 可安全釋放攝影機與輸出檔。
    profiler = PipelineProfiler(enabled=True)
    cap = None
    out = None
    temp_output = None
    final_output = None
    processed_frames = 0

    try:
        with profiler.time_block("pipeline.total_wall"):
            # 將 action 相關 CLI 覆寫同步到環境變數，供 action.py 內部讀取。
            _set_env_if_present("ACTION_DEVICE", args.action_device)
            _set_env_if_present("ACTION_PREDICT_INTERVAL", args.action_predict_interval)
            _set_env_if_present("ACTION_POSE_IMGSZ", args.action_pose_imgsz)

            prefer_engine = not args.no_engine
            # 若使用者未指定 --bbox-model/--trash-model，依 batch 1/8 自動選擇預設權重。
            default_bbox_model_path, default_trash_model_path = _default_model_paths_for_batch(args.batch_size)
            bbox_model_path_arg = args.bbox_model or default_bbox_model_path
            trash_model_path_arg = args.trash_model or default_trash_model_path
            bbox_model_candidates = _model_path_candidates(bbox_model_path_arg, prefer_engine, args.batch_size)
            trash_model_candidates = _model_path_candidates(trash_model_path_arg, prefer_engine, args.batch_size)

            print("Preloading all configured models before video processing...")
            print(f"Pipeline batch size: {args.batch_size}")
            print(f"Default BBOX model for batch {args.batch_size}: {default_bbox_model_path}")
            print(f"Default Trash model for batch {args.batch_size}: {default_trash_model_path}")
            print(f"BBOX candidates: {bbox_model_candidates}")
            print(f"Trash candidates: {trash_model_candidates}")
            action_module = None
            if not args.disable_action:
                # STGCN 先載入 pose model 與 skeleton classifier，後續只在偵測到 person 時更新。
                print(f"Pose model: {args.pose_model}")
                print(f"STGCN weight: {args.stgcn_weight}")
                with profiler.time_block("model_load.action_module_total"):
                    action_module = STGCNActionModule(
                        pose_model_path=args.pose_model,
                        stgcn_weight_path=args.stgcn_weight,
                        stgcn_config_path=args.stgcn_config,
                        action_threshold=args.action_threshold,
                        window_size=args.action_window,
                        device=args.action_device,
                        profiler=profiler,
                    )
                action_module.warmup(profiler=profiler)
            else:
                print("STGCN action module skipped by --disable-action.")

            # 主要兩個偵測模型：actor 使用 YOLO-seg，垃圾使用 RTDETR。
            model_bbox, bbox_model_path = _load_model_with_warmup(
                "bbox_yolo",
                bbox_model_candidates,
                lambda model_path: YOLO(model_path, task='segment'),
                lambda model: _warmup_bbox_model(model, args.batch_size),
                profiler,
            )
            model_trash, trash_model_path = _load_model_with_warmup(
                "trash_rtdetr",
                trash_model_candidates,
                RTDETR,
                lambda model: _warmup_trash_model(model, args.batch_size),
                profiler,
            )
            print(f"BBOX model selected: {bbox_model_path}")
            print(f"Trash model selected: {trash_model_path}")
            if args.disable_plate:
                # 車牌 OCR 可停用，減少只測 litter pipeline 時的背景執行成本。
                disable_license_plate_models()
                print("License plate detector/OCR disabled by --disable-plate.")
            elif not args.skip_plate_preload:
                preload_license_plate_models(profiler=profiler)
            else:
                print("License plate detector/OCR preload skipped by --skip-plate-preload.")

            with profiler.time_block("setup.background_subtractor"):
                # 前景遮罩用於判定 litter bbox 內是否有動態像素。
                back_sub = cv2.createBackgroundSubtractorMOG2(
                    history = 300, varThreshold = 25, detectShadows = True
                )

            # === 影片處理參數設定 ===
            video_path = f"resources/{args.file}"
            output_dir = "output"
            with profiler.time_block("setup.output_dir"):
                os.makedirs(output_dir, exist_ok = True)
            temp_output = os.path.join(output_dir, f"temp_{Path(video_path).stem}.avi")
            final_output = os.path.join(output_dir, f"{Path(video_path).stem}_annotated.mp4")

            with profiler.time_block("video.open_capture"):
                # 讀取影片屬性；fps 無效時用 30 避免 writer 初始化失敗。
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                raw_fps = cap.get(cv2.CAP_PROP_FPS)
                fps = round(raw_fps) if raw_fps > 0 else 30
            if not cap.isOpened():
                raise FileNotFoundError(f"Unable to open video: {video_path}")
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Invalid video size for {video_path}: {width}x{height}")

            with profiler.time_block("video.open_writer"):
                # 先寫暫存 AVI，最後再用 ffmpeg 壓成 mp4。
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            if not out.isOpened():
                raise RuntimeError(f"Unable to open temp video writer: {temp_output}")

            # 垃圾反追蹤物件初始化
            with profiler.time_block("setup.litter_tracker"):
                litter_tracker = GlobalLitterTracker(distance_threshold=250)

            # 紀錄車輛歷史軌跡
            vehicle_history = defaultdict(lambda: {'centroids': deque(maxlen=30), 'license_plate': None})
            # 違規顯示快取：僅用於畫面標註持續時間
            violator_display_cache = {}
            yolo_seg_cache = {}
            frame_index = 0

            # 影片主迴圈：batch 1 走 detect；batch 8 走 detect_batch。
            with profiler.time_block("process.video_loop_total"):
                with tqdm(total = total_frames, desc = "Processing Video... ", unit="frame") as pbar:
                    while cap.isOpened():
                        frames, fg_masks = _read_frame_batch(cap, back_sub, args.batch_size, profiler)
                        if not frames:
                            break

                        with profiler.time_block("detect.total"):
                            if args.batch_size > 1:
                                annotated_frames = detect_batch(
                                    frames, model_bbox, model_trash, COLORS,
                                    fg_masks, litter_tracker, vehicle_history,
                                    fps=fps,
                                    vehicle_relative_speed_threshold_pct_per_s=20.0,
                                    enable_vehicle_speed_filter=not args.disable_speed_filter,
                                    violator_display_cache=violator_display_cache,
                                    violator_display_ttl=60,
                                    violator_display_max_jump=80.0,
                                    action_module=action_module,
                                    frame_start_index=frame_index,
                                    yolo_seg_frame_skip=args.yolo_seg_frame_skip,
                                    yolo_seg_cache=yolo_seg_cache,
                                    bbox_conf=args.bbox_conf,
                                    trash_conf=args.trash_conf,
                                    profiler=profiler,
                                    moving_threshold=0.25,
                                    core_moving_threshold=0.3,
                                    batch_size=args.batch_size,
                                )
                            else:
                                annotated_frames = [
                                    detect(
                                        frames[0], model_bbox, model_trash, COLORS,
                                        fg_masks[0], litter_tracker, vehicle_history,
                                        fps=fps,
                                        vehicle_relative_speed_threshold_pct_per_s=20.0,
                                        enable_vehicle_speed_filter=not args.disable_speed_filter,
                                        violator_display_cache=violator_display_cache,
                                        violator_display_ttl=60,
                                        violator_display_max_jump=80.0,
                                        action_module=action_module,
                                        frame_index=frame_index,
                                        yolo_seg_frame_skip=args.yolo_seg_frame_skip,
                                        yolo_seg_cache=yolo_seg_cache,
                                        bbox_conf=args.bbox_conf,
                                        trash_conf=args.trash_conf,
                                        profiler=profiler,
                                        moving_threshold=0.25,
                                        core_moving_threshold=0.3,
                                    )
                                ]

                        with profiler.time_block("frame.write_temp_avi"):
                            # detect_batch 可能回傳多幀；保持輸出順序與讀取順序一致。
                            for annotated_frame in annotated_frames:
                                out.write(annotated_frame)
                        pbar.update(len(annotated_frames))
                        frame_index += len(annotated_frames)
                        processed_frames += len(annotated_frames)

            with profiler.time_block("cleanup.release_video_io"):
                cap.release()
                cap = None
                out.release()
                out = None

            # ffmpeg 壓縮影片
            fmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', temp_output,
                '-vcodec', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
                '-movflags', '+faststart',
                final_output
            ]

            try:
                with profiler.time_block("encode.ffmpeg_compress"):
                    # ffmpeg 壓縮成功後才刪除暫存 AVI，避免失敗時丟失中間成果。
                    subprocess.run(fmpeg_cmd, check = True, stdout = subprocess.DEVNULL,
                                   stderr = subprocess.STDOUT)
                print(f"Video saved to {final_output}")

                if os.path.exists(temp_output):
                    with profiler.time_block("cleanup.remove_temp_avi"):
                        os.remove(temp_output)
                    print(f"Temporary file {temp_output} removed.")
            except subprocess.CalledProcessError as e:
                print(f"Error during video compression: {e}")

            wait_for_plate_jobs(profiler=profiler)
    finally:
        # 任一階段拋錯時仍釋放 OpenCV 句柄。
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

    # 最後統一印出模型載入、影片處理、寫檔與瓶頸排行。
    profiler.print_compact_summary(frame_count=processed_frames)
