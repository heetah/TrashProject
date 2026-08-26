import os
import argparse
from pathlib import Path
from collections import defaultdict, deque

from pipeline.config import PipelineConfig, load_project_env

# 必須早於 Torch/Ultralytics 與 pipeline.devices import，device/model 設定才會完整生效。
LOADED_ENV_PATH = load_project_env()

import cv2
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics import RTDETR

from pipeline.detect import detect_batch
from pipeline.litter_tracker import GlobalLitterTracker
from pipeline.action import STGCNActionModule
from pipeline.plate import (
    disable_license_plate_models,
    dispatch_license_plate_rois,
    preload_license_plate_models,
    wait_for_plate_jobs,
)
from pipeline.profiling import PipelineProfiler
from pipeline.events import (
    build_analysis_report,
    build_run_events,
    write_analysis_json,
)
from pipeline.backtrack.sidecar import (
    build_run_record as build_backtrack_run_record,
    write_jsonl as write_backtrack_jsonl,
)
from pipeline.litter.input4c import build_litter_model_input

from pipeline.infra import (
    SUPPORTED_BATCH_SIZES,
    MotionMaskBuilder,
    AsyncFFmpegVideoWriter,
    AsyncVideoFrameReader,
    _set_ffmpeg_bin,
    _select_ffmpeg_bin,
    _estimate_actor_batch_size,
    _model_path_candidates,
    _model_path_candidates_for_batches,
    _load_model_with_warmup,
    _warmup_bbox_model,
    _warmup_trash_model,
    _engine_batch_size_from_path,
    _read_engine_input_shape,
    _apply_engine_overrides,
    _get_model_input_channels,
    _get_model_warmup_imgsz,
    _resolve_video_path,
    _open_video_capture,
)

# 畫面標註顏色設定：各類別在輸出影片中的 bbox 顏色。
COLORS = {
    'litter': (128, 0, 128), # 紫色
    'person': (255, 200, 128), # 淺藍
    'vehicle': (0, 255, 0), # 綠色
    'scooter': (0, 255, 255) # 黃色
}


def _safe_float_env(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="video path or file name in resources/", default="TThrow.mp4")
    args = parser.parse_args()
    cfg = PipelineConfig.from_env()
    _set_ffmpeg_bin(_select_ffmpeg_bin(cfg.ffmpeg_bin))

    # 資源句柄與計數器集中管理，finally 可安全釋放攝影機與輸出檔。
    profiler = PipelineProfiler(enabled=cfg.profile_enabled)
    cap = None
    out = None
    frame_reader = None
    final_output = None
    processed_frames = 0
    litter_tracker = None

    try:
        with profiler.time_block("pipeline.total_wall"):
            # 執行參數集中於 root .env + PipelineConfig；未設定時保留原 production 預設。
            # Vehicle gate、actor fast path 與 zero-repair 都可由 .env 明確覆寫。
            _BATCH_SIZE = cfg.batch_size
            _YOLO_SEG_FRAME_SKIP = cfg.yolo_seg_frame_skip
            _ACTOR_MODE = cfg.actor_mode
            _RTDETR_ZERO_REPAIR = cfg.rtdetr_zero_repair
            _RTDETR_ENABLED = cfg.rtdetr_enabled

            prefer_engine = cfg.prefer_tensorrt
            default_bbox_model_path, default_trash_model_path = cfg.model_paths_for_batch(
                _BATCH_SIZE
            )
            desired_actor_batch_size = _estimate_actor_batch_size(_BATCH_SIZE, _YOLO_SEG_FRAME_SKIP)
            bbox_candidate_batches = [desired_actor_batch_size]
            for candidate_batch in sorted(SUPPORTED_BATCH_SIZES, reverse=True):
                if (
                    candidate_batch < desired_actor_batch_size and
                    desired_actor_batch_size % candidate_batch == 0
                ):
                    bbox_candidate_batches.append(candidate_batch)
            if _BATCH_SIZE not in bbox_candidate_batches:
                bbox_candidate_batches.append(_BATCH_SIZE)
            bbox_model_candidates = _model_path_candidates_for_batches(
                default_bbox_model_path,
                prefer_engine,
                bbox_candidate_batches,
            )
            trash_model_candidates = _model_path_candidates(default_trash_model_path, prefer_engine, _BATCH_SIZE)
            pose_model_candidates = _model_path_candidates(
                cfg.pose_model_path, prefer_engine, 1
            )

            print(f"Configuration file: {LOADED_ENV_PATH or 'built-in defaults'}")
            print("Preloading all configured models before video processing...")
            print(f"Pipeline batch size: {_BATCH_SIZE}")
            print(f"Actor mode: {_ACTOR_MODE}")
            print(f"Actor target batch size: {desired_actor_batch_size}")
            print(f"Default BBOX model for batch {_BATCH_SIZE}: {default_bbox_model_path}")
            print(f"Default Trash model for batch {_BATCH_SIZE}: {default_trash_model_path}")
            print(f"BBOX candidates: {bbox_model_candidates}")
            print(f"Trash candidates: {trash_model_candidates}")
            print(f"Pose candidates: {pose_model_candidates}")
            print(f"RTDETR batch zero repair: {_RTDETR_ZERO_REPAIR}")
            print("Extreme speed: detector fast path enabled; STGCN/OCR keep their normal enable flags.")
            # STGCN 先載入 pose model 與 skeleton classifier，後續只在偵測到 person 時更新。
            print(f"Pose model candidates: {pose_model_candidates}")
            print(f"STGCN weight: {cfg.stgcn_weight_path}")
            with profiler.time_block("model_load.action_module_total"):
                action_module = STGCNActionModule(
                    pose_model_path=pose_model_candidates,
                    stgcn_weight_path=cfg.stgcn_weight_path,
                    stgcn_config_path=cfg.stgcn_config_path,
                    action_threshold=cfg.action_threshold,
                    urinate_conf_high=None,
                    urinate_conf_low=None,
                    window_size=cfg.action_window,
                    urination_window_sec=cfg.urination_window_sec,
                    urination_min_sec=cfg.urination_min_sec,
                    device=cfg.action_device,
                    profiler=profiler,
                )
            action_module.warmup(profiler=profiler)

            # 主要兩個偵測模型：actor 使用 YOLO-seg，垃圾使用 RTDETR。
            model_bbox, bbox_model_path = _load_model_with_warmup(
                "bbox_yolo",
                bbox_model_candidates,
                lambda model_path: YOLO(model_path, task='segment'),
                lambda model, model_path: _warmup_bbox_model(
                    model,
                    _engine_batch_size_from_path(model_path, desired_actor_batch_size),
                ),
                profiler,
            )
            bbox_runtime_batch_size = _engine_batch_size_from_path(
                bbox_model_path,
                desired_actor_batch_size,
            )
            if _RTDETR_ENABLED:
                # engine 載入前先讀 input shape，取得正確的 imgsz 和 channels 供 warmup 使用。
                # 用 TRT Runtime 讀取，不需要呼叫 predict()。
                _trash_engine_shape = _read_engine_input_shape(trash_model_candidates[0]) if trash_model_candidates else None
                _trash_warmup_imgsz = int(_trash_engine_shape[2]) if _trash_engine_shape else 640
                _trash_warmup_channels = int(_trash_engine_shape[1]) if _trash_engine_shape else 3

                def _trash_warmup(model, model_path):
                    engine_shape = _read_engine_input_shape(model_path)
                    if engine_shape is not None:
                        _apply_engine_overrides(model, model_path)
                        imgsz = int(engine_shape[2])
                        channels = int(engine_shape[1])
                    else:
                        channels = _get_model_input_channels(model)
                        imgsz = _get_model_warmup_imgsz(model)
                        if imgsz != 640:
                            model.overrides["imgsz"] = imgsz
                        print(f"[trash_warmup] fallback: imgsz={imgsz}, channels={channels}")
                    _warmup_trash_model(
                        model,
                        _engine_batch_size_from_path(model_path, _BATCH_SIZE),
                        imgsz=imgsz,
                        channels=channels,
                    )

                model_trash, trash_model_path = _load_model_with_warmup(
                    "trash_rtdetr",
                    trash_model_candidates,
                    RTDETR,
                    _trash_warmup,
                    profiler,
                )
                trash_runtime_batch_size = _engine_batch_size_from_path(
                    trash_model_path,
                    _BATCH_SIZE,
                )
                print(f"Trash model selected: {trash_model_path}")
                print(f"Trash runtime batch size: {trash_runtime_batch_size}")
                preload_license_plate_models(profiler=profiler)
            else:
                model_trash = None
                trash_runtime_batch_size = _BATCH_SIZE
                print("RTDETR disabled (RTDETR_ENABLED=0): skipping trash model and plate OCR.")
            print(f"BBOX model selected: {bbox_model_path}")
            print(f"BBOX runtime batch size: {bbox_runtime_batch_size}")

            with profiler.time_block("setup.motion_masker"):
                # motion mask 只用於判定 litter bbox 是否有動態像素；confirmed 規則仍由 tracker 控制。
                motion_masker = MotionMaskBuilder(
                    mode=cfg.motion_mask_mode,
                    scale_factor=cfg.fg_mask_scale,
                    diff_threshold=cfg.motion_diff_threshold,
                    dilate_iterations=cfg.motion_dilate_iterations,
                    blur_kernel_size=cfg.motion_blur_kernel,
                    open_kernel_size=cfg.motion_open_kernel,
                    open_iterations=cfg.motion_open_iterations,
                    close_kernel_size=cfg.motion_close_kernel,
                    close_iterations=cfg.motion_close_iterations,
                    mog2_detect_shadows=cfg.motion_mog2_detect_shadows,
                )

            # === 影片處理參數設定 ===
            video_path = _resolve_video_path(args.file)
            # 預設輸出到 CWD；iterate-new.py 透過 subprocess cwd= 控制落點。
            output_dir = Path(cfg.output_root).expanduser()
            with profiler.time_block("setup.output_dir"):
                output_dir.mkdir(parents=True, exist_ok=True)
            final_output = str(output_dir / f"{Path(video_path).stem}_annotated.mp4")

            with profiler.time_block("video.open_capture"):
                # 讀取影片屬性；fps 無效時用 30 避免 writer 初始化失敗。
                cap, capture_backend = _open_video_capture(
                    video_path,
                    hw_accel=cfg.video_hw_accel,
                    hw_device=cfg.video_hw_device,
                    buffer_size=cfg.video_capture_buffer_size,
                    read_threads=cfg.video_read_threads,
                    profiler=profiler,
                )
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                raw_fps = cap.get(cv2.CAP_PROP_FPS)
                fps = round(raw_fps) if raw_fps > 0 else 30
            if not cap.isOpened():
                raise FileNotFoundError(f"Unable to open video: {video_path}")
            if width <= 0 or height <= 0:
                raise RuntimeError(f"Invalid video size for {video_path}: {width}x{height}")
            print(
                f"VideoCapture backend: {capture_backend}; "
                f"hw_accel={cfg.video_hw_accel}; async_reader=True; "
                f"reader_queue={cfg.pipeline_queue_size}; "
                f"prepare_4c_in_reader={cfg.prepare_4c_in_reader and _RTDETR_ENABLED}"
            )

            with profiler.time_block("video.open_writer"):
                # 直接將 BGR raw frame 串流到 FFmpeg，避免先寫 AVI 再二次壓縮。
                out = AsyncFFmpegVideoWriter(
                    final_output,
                    width,
                    height,
                    fps,
                    profiler=profiler,
                    queue_size=cfg.writer_queue_size,
                    preset=cfg.writer_preset,
                    crf=cfg.writer_crf,
                    encoder=cfg.writer_encoder,
                )

            # 垃圾反追蹤物件初始化
            with profiler.time_block("setup.litter_tracker"):
                litter_tracker = GlobalLitterTracker(distance_threshold=cfg.litter_distance_threshold, fps=fps)

            # 紀錄車輛歷史軌跡
            vehicle_history = defaultdict(lambda: {
                'centroids': deque(maxlen=30),
                'license_plate': None,
                'plate_search_until_found': False,
                'plate_blocked_since_litter': False,
            })
            # 違規顯示快取：僅用於畫面標註持續時間
            violator_display_cache = {}
            detection_stats = {
                'rtdetr_evaluated_frames': 0,
                'rtdetr_litter_candidates': 0,
                'rtdetr_litter_candidate_frames': [],
                'raw_litter_candidates': 0,
                'geometry_litter_candidate_frames': [],
                'filtered_litter_candidates': 0,
                'filtered_litter_candidate_frames': [],
                'confirmed_litter_ids': set(),
                'confirmed_litter_frame_hits': 0,
                'confirmed_litter_thrower_ids': set(),
                'confirmed_litter_thrower_frame_hits': 0,
                'backtracked_thrower_ids': set(),
                'backtracked_thrower_frame_hits': 0,
                'first_confirmed_litter_frame': None,
                'rtdetr_batch_zero_frames': 0,
                'rtdetr_batch_zero_repaired_frames': 0,
                'person_frame_hits': 0,
                'person_detections': 0,
                'yolo_actor_infer_frames': 0,
                'yolo_actor_padded_frames': 0,
                'stgcn_person_frames': 0,
                'stgcn_pose_boxes': 0,
                'stgcn_pose_matches': 0,
                'stgcn_pose_unmatched': 0,
                'stgcn_window_ready': 0,
                'stgcn_predict_calls': 0,
                'stgcn_alerts': 0,
                'stgcn_registered_violators': 0,
            }
            if os.environ.get("LITTER_CANDIDATE_SIDECAR", "0") not in ("0", ""):
                detection_stats['litter_candidate_records'] = []
            yolo_seg_cache = {}
            rtdetr_batch_context = {}
            frame_index = 0
            last_frame = None  # Track previous frame for 4-channel litter detection
            with profiler.time_block("video.start_async_reader"):
                frame_reader = AsyncVideoFrameReader(
                    cap,
                    motion_masker,
                    profiler,
                    queue_size=cfg.pipeline_queue_size,
                    litter_input_builder=(
                        build_litter_model_input
                        if cfg.prepare_4c_in_reader and _RTDETR_ENABLED
                        else None
                    ),
                )
            cap = None

            # 影片主迴圈：batch=8 走 detect_batch。
            with profiler.time_block("process.video_loop_total"):
                with tqdm(total=total_frames, desc="Processing Video... ", unit="frame") as pbar:
                    while True:
                        prepared_frames = frame_reader.read_prepared_batch(_BATCH_SIZE)
                        if not prepared_frames:
                            break

                        expected_indices = list(
                            range(frame_index, frame_index + len(prepared_frames))
                        )
                        packet_indices = [packet.index for packet in prepared_frames]
                        if packet_indices != expected_indices:
                            raise RuntimeError(
                                "Prepared frame order mismatch: "
                                f"expected {expected_indices}, got {packet_indices}"
                            )

                        frames = [packet.source_bgr for packet in prepared_frames]
                        fg_masks = [packet.foreground_mask for packet in prepared_frames]
                        prepared_litter_inputs = (
                            [packet.litter_model_input for packet in prepared_frames]
                            if prepared_frames[0].litter_model_input is not None
                            else None
                        )

                        with profiler.time_block("detect.total"):
                            # Create prev_frames list: first frame's prev is last_frame from previous batch
                            prev_frames = [last_frame] + frames[:-1]
                            annotated_frames = detect_batch(
                                frames, model_bbox, model_trash, COLORS,
                                fg_masks, litter_tracker, vehicle_history,
                                fps=fps,
                                violator_display_cache=violator_display_cache,
                                violator_display_ttl=cfg.violator_display_ttl,
                                violator_display_max_jump=cfg.violator_display_max_jump,
                                action_module=action_module,
                                frame_start_index=frame_index,
                                yolo_seg_frame_skip=_YOLO_SEG_FRAME_SKIP,
                                yolo_seg_cache=yolo_seg_cache,
                                bbox_conf=cfg.bbox_conf,
                                trash_conf=cfg.trash_conf,
                                profiler=profiler,
                                moving_threshold=cfg.moving_threshold,
                                core_moving_threshold=cfg.core_moving_threshold,
                                motion_min_component_area=cfg.motion_min_component_area,
                                motion_min_largest_component_ratio=cfg.motion_min_largest_component_ratio,
                                batch_size=_BATCH_SIZE,
                                bbox_batch_size=bbox_runtime_batch_size,
                                trash_batch_size=trash_runtime_batch_size,
                                fg_mask_scale=cfg.fg_mask_scale,
                                stats=detection_stats,
                                rtdetr_zero_repair=_RTDETR_ZERO_REPAIR,
                                rtdetr_batch_context=rtdetr_batch_context,
                                actor_mode=_ACTOR_MODE,
                                actor_track_iou=cfg.actor_track_iou,
                                prev_frames=prev_frames,
                                prepared_litter_inputs=prepared_litter_inputs,
                            )
                            # Update last_frame for next batch
                            if frames:
                                last_frame = frames[-1]

                        with profiler.time_block("frame.write_output"):
                            # detect_batch 可能回傳多幀；保持輸出順序與讀取順序一致。
                            for annotated_frame in annotated_frames:
                                out.write(annotated_frame)
                        pbar.update(len(annotated_frames))
                        frame_index += len(annotated_frames)
                        processed_frames += len(annotated_frames)

            with profiler.time_block("cleanup.release_video_io"):
                if frame_reader is not None:
                    frame_reader.close()
                    frame_reader = None
                if cap is not None:
                    cap.release()
                    cap = None
                out.close()
                out = None
            print(f"Video saved to {final_output}")
            # Attribution 必須先於 summary/events 收尾：flush 尾端 worker，
            # 再以 event-expanded Min-Cost Flow 套用 authoritative assignment。
            smart_backtrack_summary = None
            if litter_tracker is not None and hasattr(litter_tracker, "finalize_backtracking"):
                smart_backtrack_summary = litter_tracker.finalize_backtracking(
                    vehicle_history=vehicle_history,
                    timeout=8.0,
                )
                if not smart_backtrack_summary.get("worker_flushed", False):
                    print("Smart backtrack worker still draining; retrying EOF flush once.")
                    smart_backtrack_summary = litter_tracker.finalize_backtracking(
                        vehicle_history=vehicle_history,
                        timeout=8.0,
                    )
                print(
                    "Smart backtrack: "
                    f"solver={smart_backtrack_summary['solver']}, "
                    f"resolved={smart_backtrack_summary['resolved']}, "
                    f"dustbin={smart_backtrack_summary['dustbin']}, "
                    f"legacy={smart_backtrack_summary['legacy']}, "
                    f"pending={smart_backtrack_summary['pending']}, "
                    f"worker_flushed={smart_backtrack_summary['worker_flushed']}"
                )
                late_plate_items = litter_tracker.consume_backward_plate_roi_items()
                if late_plate_items:
                    dispatched = dispatch_license_plate_rois(
                        late_plate_items, vehicle_history, profiler=profiler
                    )
                    if not dispatched:
                        wait_for_plate_jobs(profiler=profiler)
                        dispatched = dispatch_license_plate_rois(
                            late_plate_items, vehicle_history, profiler=profiler
                        )
                    if not dispatched:
                        litter_tracker.restore_backward_plate_roi_items(late_plate_items)
                wait_for_plate_jobs(profiler=profiler)

            final_litter_events = (
                litter_tracker.get_litter_events() if litter_tracker is not None else []
            )
            confirmed_litter_ids = detection_stats.get('confirmed_litter_ids', set())
            confirmed_litter_thrower_ids = detection_stats.get(
                'confirmed_litter_thrower_ids', set()
            )
            backtracked_thrower_ids = detection_stats.get(
                'backtracked_thrower_ids', set()
            )
            # 尾端 worker 的結果可能來不及進逐幀 stats；以 finalized event 補齊。
            for event in final_litter_events:
                litter_id = int(event.get('litter_id', -1))
                if event.get('thrower_key') is not None:
                    confirmed_litter_thrower_ids.add(litter_id)
                if event.get('backtrack_status') in ('resolved', 'legacy'):
                    backtracked_thrower_ids.add(litter_id)
            first_confirmed = detection_stats.get('first_confirmed_litter_frame')
            first_confirmed_text = "None" if first_confirmed is None else str(first_confirmed)
            stgcn_urinate = int(detection_stats.get('stgcn_pred_urinate', 0))
            stgcn_urinate_confirmed = int(detection_stats.get('stgcn_urinate_confirmed', 0))
            print(
                "Litter detection summary: "
                f"rtdetr_candidates={detection_stats.get('rtdetr_litter_candidates', 0)}, "
                f"geometry_passed_candidates={detection_stats.get('raw_litter_candidates', 0)}, "
                f"motion_filtered_candidates={detection_stats.get('filtered_litter_candidates', 0)}, "
                f"confirmed_ids={len(confirmed_litter_ids)}, "
                f"confirmed_frame_hits={detection_stats.get('confirmed_litter_frame_hits', 0)}, "
                f"confirmed_thrower_ids={len(confirmed_litter_thrower_ids)}, "
                f"backtracked_thrower_ids={len(backtracked_thrower_ids)}, "
                f"first_confirmed_frame={first_confirmed_text}, "
                f"rtdetr_zero_frames={detection_stats.get('rtdetr_batch_zero_frames', 0)}, "
                f"rtdetr_zero_repaired={detection_stats.get('rtdetr_batch_zero_repaired_frames', 0)}, "
                f"person_frame_hits={detection_stats.get('person_frame_hits', 0)}, "
                f"person_detections={detection_stats.get('person_detections', 0)}, "
                f"yolo_actor_infer_frames={detection_stats.get('yolo_actor_infer_frames', 0)}, "
                f"yolo_actor_padded_frames={detection_stats.get('yolo_actor_padded_frames', 0)}, "
                f"stgcn_person_frames={detection_stats.get('stgcn_person_frames', 0)}, "
                f"stgcn_pose_matches={detection_stats.get('stgcn_pose_matches', 0)}, "
                f"stgcn_window_ready={detection_stats.get('stgcn_window_ready', 0)}, "
                f"stgcn_predicts={detection_stats.get('stgcn_predict_calls', 0)}, "
                f"stgcn_urinate={stgcn_urinate}, "
                f"stgcn_urinate_blocked_on_vehicle={detection_stats.get('stgcn_urinate_blocked_on_vehicle', 0)}, "
                f"stgcn_urinate_confirmed={stgcn_urinate_confirmed}, "
                f"stgcn_alerts={detection_stats.get('stgcn_alerts', 0)}, "
                f"stgcn_registered_violators={detection_stats.get('stgcn_registered_violators', 0)}, "
                f"vehicle_gate_skipped_frames={detection_stats.get('vehicle_gate_skipped_frames', 0)}"
            )
            run_summary = {
                "input_video": str(video_path),
                "output_video": str(final_output),
                "processed_frames": int(processed_frames),
                "total_frames": int(total_frames),
                "rtdetr_enabled": _RTDETR_ENABLED,
                "stgcn_pose_enabled": True,
                "plate_enabled": _RTDETR_ENABLED,
                "duration_sec": round(int(processed_frames) / float(fps), 3),
                "rtdetr_confidence_threshold": float(cfg.trash_conf),
                "rtdetr_evaluated_frames": int(
                    detection_stats.get('rtdetr_evaluated_frames', 0)
                ),
                "rtdetr_litter_candidates": int(
                    detection_stats.get('rtdetr_litter_candidates', 0)
                ),
                "rtdetr_litter_candidate_frames": list(
                    detection_stats.get('rtdetr_litter_candidate_frames', [])
                ),
                "raw_litter_candidates": int(detection_stats.get('raw_litter_candidates', 0)),
                "geometry_litter_candidate_frames": list(
                    detection_stats.get('geometry_litter_candidate_frames', [])
                ),
                "filtered_litter_candidates": int(detection_stats.get('filtered_litter_candidates', 0)),
                "filtered_litter_candidate_frames": list(
                    detection_stats.get('filtered_litter_candidate_frames', [])
                ),
                "confirmed_litter_ids": len(confirmed_litter_ids),
                "confirmed_litter_frame_hits": int(detection_stats.get('confirmed_litter_frame_hits', 0)),
                "confirmed_litter_thrower_ids": len(confirmed_litter_thrower_ids),
                "confirmed_litter_thrower_frame_hits": int(
                    detection_stats.get('confirmed_litter_thrower_frame_hits', 0)
                ),
                "backtracked_thrower_ids": len(backtracked_thrower_ids),
                "backtracked_thrower_frame_hits": int(
                    detection_stats.get('backtracked_thrower_frame_hits', 0)
                ),
                "person_frame_hits": int(detection_stats.get('person_frame_hits', 0)),
                "person_detections": int(detection_stats.get('person_detections', 0)),
                "stgcn_person_frames": int(detection_stats.get('stgcn_person_frames', 0)),
                "stgcn_pose_matches": int(detection_stats.get('stgcn_pose_matches', 0)),
                "stgcn_pose_unmatched": int(detection_stats.get('stgcn_pose_unmatched', 0)),
                "stgcn_pose_boxes": int(detection_stats.get('stgcn_pose_boxes', 0)),
                "stgcn_window_ready": int(detection_stats.get('stgcn_window_ready', 0)),
                "stgcn_predict_calls": int(detection_stats.get('stgcn_predict_calls', 0)),
                "stgcn_urination": stgcn_urinate,
                "stgcn_pred_urinate": stgcn_urinate,
                "stgcn_urinate_confirmed": stgcn_urinate_confirmed,
                "stgcn_alerts": int(detection_stats.get('stgcn_alerts', 0)),
                "stgcn_registered_violators": int(detection_stats.get('stgcn_registered_violators', 0)),
                "vehicle_gate_skipped_frames": int(detection_stats.get('vehicle_gate_skipped_frames', 0)),
                "has_urinate": stgcn_urinate > 0 or stgcn_urinate_confirmed > 0,
                "has_littering": any(
                    bool(event.get('escalated', False))
                    for event in final_litter_events
                ),
                "has_confirm_litter": len(confirmed_litter_ids) > 0,
                "has_confirmed_litter_thrower": len(confirmed_litter_thrower_ids) > 0,
                "has_backtracked_thrower": len(backtracked_thrower_ids) > 0,
                "has_keypoints": int(detection_stats.get('stgcn_pose_matches', 0)) > 0,
                "has_person": int(detection_stats.get('person_detections', 0)) > 0,
            }
            if smart_backtrack_summary is not None:
                run_summary["smart_backtrack"] = smart_backtrack_summary
            # Legacy-only Offline P↔V Hungarian。Smart mode 會回 None，避免
            # 1-to-1 結果覆蓋 many-to-many Min-Cost Flow attribution。
            if litter_tracker is not None and hasattr(litter_tracker, "finalize_associations"):
                pv_assoc = litter_tracker.finalize_associations()
                if pv_assoc is not None:
                    run_summary["person_vehicle_assoc"] = pv_assoc
                    print(
                        "Person-Vehicle association: "
                        f"persons={pv_assoc['persons']}, "
                        f"confirmed_vehicles={pv_assoc['confirmed_vehicles']}, "
                        f"bound={pv_assoc['bound_persons']}, "
                        f"unbound={pv_assoc['unbound_persons']}, "
                        f"litter_events={pv_assoc['litter_events']}"
                    )

            # Research/calibration sidecar: immutable model candidates and every
            # weighted/raw component. Ground truth is stored separately by the
            # annotation tool so rerunning inference never overwrites labels.
            if (
                litter_tracker is not None
                and hasattr(litter_tracker, "get_backtrack_candidate_records")
                and os.environ.get("SMART_BACKTRACK_SIDECAR", "0")
                not in ("0", "")
            ):
                sidecar_path = Path(final_output).with_name(
                    Path(final_output).stem
                    + "_backtrack_candidates.jsonl"
                )
                weak_clip_label = (
                    {
                        "value": "litter",
                        "strength": "weak_gt",
                        "source": "parent_directory",
                    }
                    if Path(video_path).parent.name.lower() == "litter"
                    else None
                )
                candidate_records = (
                    litter_tracker.get_backtrack_candidate_records(
                        input_video=str(video_path),
                        output_video=str(final_output),
                    )
                )
                run_record = build_backtrack_run_record(
                    input_video=str(video_path),
                    output_video=str(final_output),
                    fps=float(fps),
                    frame_count=int(processed_frames),
                    smart_summary=smart_backtrack_summary,
                    extra={"clip_label": weak_clip_label},
                )
                write_backtrack_jsonl(
                    [run_record, *candidate_records],
                    str(sidecar_path),
                )
                run_summary["backtrack_candidate_sidecar"] = str(sidecar_path)
                run_summary["backtrack_candidate_records"] = len(
                    candidate_records
                )
                print(
                    "Backtrack candidate sidecar: "
                    f"events={len(candidate_records)} -> {sidecar_path}"
                )

            # Opt-in calibration trace: one record per RT-DETR litter candidate,
            # including the exact gate rejection reason and tracker ID when
            # accepted. This is research evidence, not ground truth.
            litter_candidate_records = detection_stats.get('litter_candidate_records')
            if isinstance(litter_candidate_records, list):
                containment_threshold = min(
                    max(_safe_float_env("LITTER_FP_CONTAINMENT_THR", 0.999), 0.0),
                    1.0,
                )
                dedup_iou_threshold = min(
                    max(_safe_float_env("LITTER_CANDIDATE_DEDUP_IOU", 0.5), 0.0),
                    1.0,
                )
                candidate_trace_path = Path(final_output).with_name(
                    Path(final_output).stem + "_litter_candidates.jsonl"
                )
                candidate_trace_run = {
                    "record_type": "run",
                    "schema": "litter-candidate-gates/v1",
                    "input_video": str(video_path),
                    "output_video": str(final_output),
                    "fps": float(fps),
                    "frame_count": int(processed_frames),
                    "confidence_threshold": float(cfg.trash_conf),
                    "containment_threshold": containment_threshold,
                    "dedup_enabled": os.environ.get(
                        "LITTER_CANDIDATE_DEDUP", "0"
                    ) not in ("0", ""),
                    "dedup_iou_threshold": dedup_iou_threshold,
                    # Tracker confirmation knobs are recorded alongside the
                    # candidate trace so an A/B replay is reproducible from
                    # its output directory rather than from shell history.
                    "confirm_require_birth_actor": os.environ.get(
                        "LITTER_CONFIRM_REQUIRE_BIRTH_ACTOR", "1"
                    ) not in ("0", ""),
                    "min_confirm_age_vehicle": _safe_float_env(
                        "LITTER_MIN_CONFIRM_AGE_VEHICLE", 3
                    ),
                    "min_confirm_downward_vehicle": _safe_float_env(
                        "LITTER_MIN_CONFIRM_DOWNWARD_VEHICLE", 12
                    ),
                    "min_confirm_horizontal_displacement": _safe_float_env(
                        "LITTER_MIN_CONFIRM_HORIZONTAL_DISPLACEMENT", 5
                    ),
                    "max_horiz_to_down_ratio_vehicle": _safe_float_env(
                        "LITTER_MAX_HORIZ_TO_DOWN_RATIO_VEHICLE", 3.5
                    ),
                    "min_vehicle_relative_separation": _safe_float_env(
                        "LITTER_MIN_VEHICLE_RELATIVE_SEPARATION", 60
                    ),
                    "fp_streak_ratio": _safe_float_env(
                        "LITTER_FP_STREAK_RATIO", 5
                    ),
                    "allow_shake_candidates": os.environ.get(
                        "LITTER_ALLOW_SHAKE_CANDIDATES", "0"
                    ) not in ("0", ""),
                }
                write_backtrack_jsonl(
                    [candidate_trace_run, *litter_candidate_records],
                    str(candidate_trace_path),
                )
                run_summary['litter_candidate_sidecar'] = str(candidate_trace_path)
                run_summary['litter_candidate_records'] = len(litter_candidate_records)
                print(
                    "Litter candidate sidecar: "
                    f"candidates={len(litter_candidate_records)} -> {candidate_trace_path}"
                )

            # 每支影片只輸出一份前端 JSON；研究 sidecar 預設關閉，需明確啟用。
            analysis_path = Path(final_output).with_name(
                Path(final_output).stem + "_analysis.json"
            )
            run_summary["analysis_json"] = str(analysis_path)

            run_events = build_run_events(
                final_litter_events,
                action_module.get_urinate_events() if action_module is not None else [],
                vehicle_history,
                run_summary,
                fps,
                action_vehicle_associations=(
                    litter_tracker.get_action_vehicle_associations()
                    if litter_tracker is not None
                    and hasattr(litter_tracker, "get_action_vehicle_associations")
                    else {}
                ),
            )
            analysis_report = build_analysis_report(
                run_summary,
                run_events,
                vehicle_history,
                fps=fps,
            )
            write_analysis_json(analysis_report, analysis_path)
            print(f"Analysis written: {analysis_path}")

            if litter_tracker is not None:
                litter_tracker.close()
                litter_tracker = None
            wait_for_plate_jobs(profiler=profiler)
    finally:
        # 任一階段拋錯時仍釋放 OpenCV 句柄。
        if frame_reader is not None:
            frame_reader.close()
        if cap is not None:
            cap.release()
        if out is not None:
            out.close()
        if litter_tracker is not None:
            litter_tracker.close()

    # 最後統一印出模型載入、影片處理、寫檔與瓶頸排行。
    profiler.print_compact_summary(frame_count=processed_frames)
