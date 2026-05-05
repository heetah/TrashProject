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

from detect import BBOX_DEVICE, BBOX_HALF, TRASH_DEVICE, TRASH_HALF, detect
from litterTracker import GlobalLitterTracker
from action import STGCNActionModule
from licensePlate import disable_license_plate_models, preload_license_plate_models, wait_for_plate_jobs
from timeUtils import PipelineProfiler

COLORS = {
    'litter': (128, 0, 128), # 紫色
    'person': (255, 200, 128), # 淺藍
    'vehicle': (0, 255, 0), # 綠色
    'scooter': (0, 255, 255) # 黃色
}

POSE_MODEL_PATH = 'modules_weight/yolo26x-pose.pt'
STGCN_WEIGHT_PATH = 'modules_weight/best_acc_top1_epoch_13.pth'
STGCN_CONFIG_PATH = 'mmaction2/configs/skeleton/stgcnpp/custom_trash_stgcnpp.py'
# MODEL_BBOX_PATH = 'modules_weight/yolo26x-seg.pt'
MODEL_BBOX_PATH = 'modules_weight/best-yolo-seg_v3.pt'
MODEL_TRASH_PATH = 'modules_weight/best-rtdetr-seg.pt'

try:
    import torch
except Exception:
    torch = None


def _set_env_if_present(name, value):
    if value is not None:
        os.environ[name] = str(value)


def _can_try_tensorrt_engine(engine_path):
    if torch is None or not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        print(f"Warning: CUDA is unavailable; skip TensorRT engine {engine_path}.")
        return False

    try:
        import tensorrt  # noqa: F401
    except Exception as exc:
        print(f"Warning: TensorRT import failed; skip engine {engine_path}: {exc}")
        return False

    return True


def _model_path_candidates(model_path, prefer_engine=True):
    path = Path(model_path)
    candidates = []

    if prefer_engine and path.suffix == ".pt":
        engine_path = path.with_suffix(".engine")
        if engine_path.exists():
            if _can_try_tensorrt_engine(engine_path):
                candidates.append(str(engine_path))
        else:
            print(f"Warning: TensorRT engine not found for {path}; using PyTorch weights.")

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


def _warmup_bbox_model(model):
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(
        dummy_frame,
        conf=0.01,
        device=BBOX_DEVICE,
        half=BBOX_HALF,
        verbose=False,
    )


def _warmup_trash_model(model):
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(
        dummy_frame,
        conf=0.01,
        device=TRASH_DEVICE,
        half=TRASH_HALF,
        verbose=False,
    )


if __name__ == "__main__":
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
    parser.add_argument("--disable-action", action="store_true", help="Disable STGCN action recognition; enabled by default for faster litter-only processing")
    parser.add_argument("--yolo-seg-frame-skip", type=int, default=2, help="run YOLO-seg actor tracking every N frames")
    parser.add_argument("--bbox-model", default=None, help="YOLO-seg actor model path")
    parser.add_argument("--trash-model", default=None, help="RTDETR litter model path")
    parser.add_argument("--no-engine", action="store_true", help="Use .pt weights even when a sibling .engine exists")
    parser.add_argument("--bbox-conf", type=float, default=0.45, help="YOLO-seg actor confidence threshold")
    parser.add_argument("--trash-conf", type=float, default=0.4, help="RTDETR litter confidence threshold")
    parser.add_argument("--disable-plate", action="store_true", help="Disable license plate detection/OCR for faster litter-only processing")
    parser.add_argument("--skip-plate-preload", action="store_true", help="Do not preload license plate detector/OCR models")
    args = parser.parse_args()

    profiler = PipelineProfiler(enabled=True)
    cap = None
    out = None
    temp_output = None
    final_output = None
    processed_frames = 0

    try:
        with profiler.time_block("pipeline.total_wall"):
            _set_env_if_present("ACTION_DEVICE", args.action_device)
            _set_env_if_present("ACTION_PREDICT_INTERVAL", args.action_predict_interval)
            _set_env_if_present("ACTION_POSE_IMGSZ", args.action_pose_imgsz)

            prefer_engine = not args.no_engine
            bbox_model_candidates = _model_path_candidates(args.bbox_model or MODEL_BBOX_PATH, prefer_engine)
            trash_model_candidates = _model_path_candidates(args.trash_model or MODEL_TRASH_PATH, prefer_engine)

            print("Preloading all configured models before video processing...")
            print(f"BBOX candidates: {bbox_model_candidates}")
            print(f"Trash candidates: {trash_model_candidates}")
            action_module = None
            if not args.disable_action:
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
                print("STGCN action module skipped; use --disable-action when action recognition is required.")

            model_bbox, bbox_model_path = _load_model_with_warmup(
                "bbox_yolo",
                bbox_model_candidates,
                lambda model_path: YOLO(model_path, task='segment'),
                _warmup_bbox_model,
                profiler,
            )
            model_trash, trash_model_path = _load_model_with_warmup(
                "trash_rtdetr",
                trash_model_candidates,
                RTDETR,
                _warmup_trash_model,
                profiler,
            )
            print(f"BBOX model selected: {bbox_model_path}")
            print(f"Trash model selected: {trash_model_path}")
            if args.disable_plate:
                disable_license_plate_models()
                print("License plate detector/OCR disabled by --disable-plate.")
            elif not args.skip_plate_preload:
                preload_license_plate_models(profiler=profiler)
            else:
                print("License plate detector/OCR preload skipped by --skip-plate-preload.")

            with profiler.time_block("setup.background_subtractor"):
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

            with profiler.time_block("process.video_loop_total"):
                with tqdm(total = total_frames, desc = "Processing Video... ", unit="frame") as pbar:
                    while cap.isOpened():
                        with profiler.time_block("frame.read"):
                            ret, frame = cap.read()
                        if not ret:
                            break

                        with profiler.time_block("frame.foreground_mask"):
                            scale_factor = 0.5
                            small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
                            small_mask = back_sub.apply(small_frame, learningRate=0.005)
                            _, small_mask = cv2.threshold(small_mask, 254, 255, cv2.THRESH_BINARY)
                            fg_mask = cv2.resize(small_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                        with profiler.time_block("detect.total"):
                            annotated_frame = detect(
                                frame, model_bbox, model_trash, COLORS,
                                fg_mask, litter_tracker, vehicle_history,
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

                        with profiler.time_block("frame.write_temp_avi"):
                            out.write(annotated_frame)
                        pbar.update(1)
                        frame_index += 1
                        processed_frames += 1

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
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

    profiler.print_compact_summary(frame_count=processed_frames)
