# -*- coding: utf-8 -*-
# 主流程入口：負責模型載入、影片讀寫、逐幀/批次推理、輸出壓縮與耗時統計。
import cv2
import numpy as np
import os
import queue
import subprocess
import threading
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
MODEL_BBOX_PATH_BATCH_2 = 'modules_weight/yolo26x-seg.pt'
MODEL_TRASH_PATH_BATCH_2 = 'modules_weight/batch/best-rtdetr-seg.pt'
DEFAULT_FG_MASK_SCALE = 0.5
DEFAULT_MOTION_DIFF_THRESHOLD = 10
DEFAULT_MOTION_DILATE_ITERATIONS = 2

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


class AsyncFFmpegVideoWriter:
    # 背景 FFmpeg writer：主執行緒只排隊 frame，編碼與 muxing 由背景 thread 處理。
    def __init__(self, output_path, width, height, fps, profiler=None,
                 queue_size=16, preset="fast", crf=23):
        self.output_path = str(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.profiler = profiler
        self._queue = queue.Queue(maxsize=max(int(queue_size or 1), 1))
        self._stop_token = object()
        self._error = None
        self._closed = False
        self._process = None

        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',
            '-an',
            '-vcodec', 'libx264',
            '-preset', str(preset),
            '-crf', str(int(crf)),
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            self.output_path,
        ]

        with profiler.time_block("video.open_ffmpeg_writer"):
            self._process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        self._thread = threading.Thread(target=self._worker, name="ffmpeg-writer", daemon=True)
        self._thread.start()

    def write(self, frame):
        if self._closed:
            raise RuntimeError("FFmpeg writer is already closed.")
        if self._error is not None:
            raise RuntimeError("FFmpeg writer failed.") from self._error
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(f"Frame size mismatch: got {frame.shape[1]}x{frame.shape[0]}, expected {self.width}x{self.height}")

        # 確保傳給 FFmpeg 的資料是連續 BGR buffer，避免 stdin.write 時產生格式問題。
        frame = np.ascontiguousarray(frame)
        with self.profiler.time_block("frame.queue_output_frame"):
            self._queue.put(frame)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._stop_token)
        with self.profiler.time_block("encode.ffmpeg_pipe_close"):
            self._thread.join()
            if self._process.stdin is not None:
                self._process.stdin.close()
            return_code = self._process.wait()
        if self._error is not None:
            raise RuntimeError("FFmpeg writer failed.") from self._error
        if return_code != 0:
            raise RuntimeError(f"FFmpeg exited with code {return_code}")

    def _worker(self):
        try:
            while True:
                frame = self._queue.get()
                try:
                    if frame is self._stop_token:
                        break
                    with self.profiler.time_block("frame.ffmpeg_stdin_write"):
                        self._process.stdin.write(frame.tobytes())
                finally:
                    self._queue.task_done()
        except Exception as exc:
            self._error = exc
        finally:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.flush()
            except Exception as exc:
                if self._error is None:
                    self._error = exc


class MotionMaskBuilder:
    # 前景 mask 建立器：預設用 temporal diff 加速；必要時可切回原 MOG2。
    def __init__(self, mode="temporal", scale_factor=DEFAULT_FG_MASK_SCALE,
                 diff_threshold=DEFAULT_MOTION_DIFF_THRESHOLD,
                 dilate_iterations=DEFAULT_MOTION_DILATE_ITERATIONS,
                 mog2_history=300, mog2_var_threshold=25, mog2_detect_shadows=True):
        self.mode = str(mode or "temporal").lower()
        self.scale_factor = float(scale_factor or 1.0)
        self.diff_threshold = int(diff_threshold)
        self.dilate_iterations = max(int(dilate_iterations or 0), 0)
        self.prev_gray = None
        self.temporal_kernel = np.ones((3, 3), dtype=np.uint8) if self.dilate_iterations > 0 else None
        self.back_sub = None

        if self.mode == "mog2":
            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=int(mog2_history),
                varThreshold=float(mog2_var_threshold),
                detectShadows=bool(mog2_detect_shadows),
            )
        elif self.mode != "temporal":
            raise ValueError(f"Unsupported motion mask mode: {mode}")

    def build(self, frame, profiler):
        if self.mode == "mog2":
            return self._build_mog2(frame, profiler)
        return self._build_temporal(frame, profiler)

    def _scaled_frame(self, frame, profiler):
        if self.scale_factor == 1.0:
            return frame
        with profiler.time_block("frame.motion_resize"):
            return cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

    def _build_mog2(self, frame, profiler):
        # 原始 MOG2 路徑：保留給需要逐像素背景模型時回退使用。
        mask_frame = self._scaled_frame(frame, profiler)
        with profiler.time_block("frame.foreground_mog2_apply"):
            fg_mask = self.back_sub.apply(mask_frame, learningRate=0.005)
        with profiler.time_block("frame.foreground_threshold"):
            _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        return fg_mask

    def _build_temporal(self, frame, profiler):
        # 快速路徑：只比較相鄰幀灰階差異，符合目前「litter 是否正在移動」用途。
        mask_frame = self._scaled_frame(frame, profiler)
        with profiler.time_block("frame.motion_gray"):
            gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return np.full(gray.shape, 255, dtype=np.uint8)

        with profiler.time_block("frame.motion_absdiff"):
            diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        with profiler.time_block("frame.motion_threshold"):
            _, motion_mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        if self.temporal_kernel is not None:
            with profiler.time_block("frame.motion_dilate"):
                motion_mask = cv2.dilate(motion_mask, self.temporal_kernel, iterations=self.dilate_iterations)
        return motion_mask


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


def _read_frame_batch(cap, motion_masker, batch_size, profiler):
    # 一次讀取 batch_size 幀，同步產生每幀的前景遮罩，供後續 motion filter 使用。
    frames = []
    fg_masks = []
    for _ in range(int(batch_size)):
        with profiler.time_block("frame.read"):
            ret, frame = cap.read()
        if not ret:
            break

        with profiler.time_block("frame.foreground_mask"):
            fg_mask = motion_masker.build(frame, profiler)

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
    parser.add_argument("--fg-mask-scale", type=float, default=DEFAULT_FG_MASK_SCALE,
                        help="motion mask scale; 0.5 keeps previous low-res mask behavior")
    parser.add_argument("--motion-mask-mode", choices=("temporal", "mog2"), default="temporal",
                        help="temporal uses fast frame differencing; mog2 uses the previous background subtractor")
    parser.add_argument("--motion-diff-threshold", type=int, default=DEFAULT_MOTION_DIFF_THRESHOLD,
                        help="pixel difference threshold for temporal motion mask")
    parser.add_argument("--motion-dilate-iterations", type=int, default=DEFAULT_MOTION_DILATE_ITERATIONS,
                        help="dilation iterations for temporal motion mask")
    parser.add_argument("--mog2-no-shadows", action="store_true",
                        help="disable MOG2 shadow detection when --motion-mask-mode mog2")
    parser.add_argument("--writer-queue-size", type=int, default=16,
                        help="number of annotated frames buffered before FFmpeg writer blocks")
    args = parser.parse_args()

    # 資源句柄與計數器集中管理，finally 可安全釋放攝影機與輸出檔。
    profiler = PipelineProfiler(enabled=True)
    cap = None
    out = None
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

            with profiler.time_block("setup.motion_masker"):
                # motion mask 只用於判定 litter bbox 是否有動態像素；confirmed 規則仍由 tracker 控制。
                motion_masker = MotionMaskBuilder(
                    mode=args.motion_mask_mode,
                    scale_factor=args.fg_mask_scale,
                    diff_threshold=args.motion_diff_threshold,
                    dilate_iterations=args.motion_dilate_iterations,
                    mog2_detect_shadows=not args.mog2_no_shadows,
                )

            # === 影片處理參數設定 ===
            video_path = f"resources/{args.file}"
            output_dir = "output"
            with profiler.time_block("setup.output_dir"):
                os.makedirs(output_dir, exist_ok = True)
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
                # 直接將 BGR raw frame 串流到 FFmpeg，避免先寫 AVI 再二次壓縮。
                out = AsyncFFmpegVideoWriter(
                    final_output,
                    width,
                    height,
                    fps,
                    profiler=profiler,
                    queue_size=args.writer_queue_size,
                    preset="fast",
                    crf=23,
                )

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
                        frames, fg_masks = _read_frame_batch(
                            cap,
                            motion_masker,
                            args.batch_size,
                            profiler,
                        )
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
                                    fg_mask_scale=args.fg_mask_scale,
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
                                        fg_mask_scale=args.fg_mask_scale,
                                    )
                                ]

                        with profiler.time_block("frame.write_output"):
                            # detect_batch 可能回傳多幀；保持輸出順序與讀取順序一致。
                            for annotated_frame in annotated_frames:
                                out.write(annotated_frame)
                        pbar.update(len(annotated_frames))
                        frame_index += len(annotated_frames)
                        processed_frames += len(annotated_frames)

            with profiler.time_block("cleanup.release_video_io"):
                cap.release()
                cap = None
                out.close()
                out = None
            print(f"Video saved to {final_output}")

            wait_for_plate_jobs(profiler=profiler)
    finally:
        # 任一階段拋錯時仍釋放 OpenCV 句柄。
        if cap is not None:
            cap.release()
        if out is not None:
            out.close()

    # 最後統一印出模型載入、影片處理、寫檔與瓶頸排行。
    profiler.print_compact_summary(frame_count=processed_frames)
