import argparse
import importlib
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from ultralytics import RTDETR, YOLO


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BBOX_MODEL = "modules_weight/best-yolo-seg_v3.pt"
DEFAULT_TRASH_MODEL = "modules_weight/best-rtdetr-seg.pt"
COLORS = {
    "litter": (128, 0, 128),
    "person": (255, 200, 128),
    "vehicle": (0, 255, 0),
    "scooter": (0, 255, 255),
}


def _mask_metrics(litter_box, actor):
    mask_poly = actor.get("mask_poly")
    if mask_poly is None:
        return None, None

    poly = np.asarray(mask_poly, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return None, None

    lx1, ly1, lx2, ly2 = map(float, litter_box[:4])
    anchor = ((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0)
    signed_dist = cv2.pointPolygonTest(
        poly.reshape((-1, 1, 2)),
        (float(anchor[0]), float(anchor[1])),
        True,
    )

    rx1 = int(np.floor(min(lx1, lx2)))
    ry1 = int(np.floor(min(ly1, ly2)))
    rx2 = int(np.ceil(max(lx1, lx2)))
    ry2 = int(np.ceil(max(ly1, ly2)))
    roi_w = max(rx2 - rx1, 1)
    roi_h = max(ry2 - ry1, 1)
    local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    local_poly = poly.copy()
    local_poly[:, 0] -= rx1
    local_poly[:, 1] -= ry1
    cv2.fillPoly(local_mask, [np.round(local_poly).astype(np.int32)], 255)
    overlap = float(np.count_nonzero(local_mask)) / float(roi_w * roi_h)
    return float(signed_dist), overlap


def _load_pipeline(script_dir):
    pipeline_path = str((REPO_ROOT / script_dir).resolve())
    sys.path.insert(0, pipeline_path)
    for module_name in (
        "detect",
        "litterTracker",
        "smallFunction",
        "licensePlate",
        "timeUtils",
    ):
        sys.modules.pop(module_name, None)

    detect_mod = importlib.import_module("detect")
    tracker_mod = importlib.import_module("litterTracker")

    def _skip_plate_detection(*args, **kwargs):
        return None

    detect_mod.detect_license_plates = _skip_plate_detection
    if hasattr(detect_mod, "dispatch_license_plate_rois"):
        detect_mod.dispatch_license_plate_rois = _skip_plate_detection
    return detect_mod, tracker_mod


def _foreground_mask(back_sub, frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    small_mask = back_sub.apply(small_frame, learningRate=0.005)
    _, small_mask = cv2.threshold(small_mask, 254, 255, cv2.THRESH_BINARY)
    return cv2.resize(
        small_mask,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def _run_video(args, detect_mod, tracker_mod, model_bbox, model_trash, video_name):
    video_path = REPO_ROOT / "resources" / video_name
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = round(raw_fps) if raw_fps > 0 else 30
    max_frames = args.max_frames or total_frames

    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=300,
        varThreshold=25,
        detectShadows=True,
    )
    litter_tracker = tracker_mod.GlobalLitterTracker(distance_threshold=250)
    vehicle_history = defaultdict(lambda: {"centroids": deque(maxlen=30), "license_plate": None})
    violator_display_cache = {}
    yolo_seg_cache = {}

    confirmed_ids = set()
    confirmed_frame_hits = 0
    first_confirmed_frame = None
    printed_debug_ids = set()
    max_active_confirmed = 0
    processed_frames = 0
    frame_index = 0

    while processed_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = _foreground_mask(back_sub, frame)
        detect_mod.detect(
            frame,
            model_bbox,
            model_trash,
            COLORS,
            fg_mask,
            litter_tracker,
            vehicle_history,
            fps=fps,
            vehicle_relative_speed_threshold_pct_per_s=20.0,
            enable_vehicle_speed_filter=True,
            violator_display_cache=violator_display_cache,
            violator_display_ttl=60,
            violator_display_max_jump=80.0,
            action_module=None,
            frame_index=frame_index,
            yolo_seg_frame_skip=args.yolo_seg_frame_skip,
            yolo_seg_cache=yolo_seg_cache,
            bbox_conf=args.bbox_conf,
            trash_conf=args.trash_conf,
            profiler=None,
            moving_threshold=args.moving_threshold,
            core_moving_threshold=args.core_moving_threshold,
        )

        active_confirmed = [
            litter_id
            for litter_id, litter_data in litter_tracker.active_litters.items()
            if litter_data.get("state") == "confirmed"
        ]
        if active_confirmed:
            confirmed_frame_hits += 1
            confirmed_ids.update(active_confirmed)
            max_active_confirmed = max(max_active_confirmed, len(active_confirmed))
            if first_confirmed_frame is None:
                first_confirmed_frame = frame_index
            if args.debug_confirmed:
                for litter_id in active_confirmed:
                    if litter_id in printed_debug_ids:
                        continue
                    litter_data = litter_tracker.active_litters[litter_id]
                    print(
                        f"DEBUG {video_name} frame={frame_index} litter_id={litter_id} "
                        f"bbox={litter_data.get('bbox')} "
                        f"history={litter_data.get('history')} "
                        f"thrower={litter_data.get('thrower_key')}"
                    )
                    history = litter_data.get("history") or []
                    if len(history) >= 2 and "smallFunction" in sys.modules:
                        debug_actors = yolo_seg_cache.get("persons", []) + yolo_seg_cache.get("vehicles", [])
                        holding_result = sys.modules["smallFunction"].litter_holding(
                            litter_data.get("bbox"),
                            debug_actors,
                            prev_litter_center=history[-2],
                            vehicle_history=vehicle_history,
                        )
                        actor_keys = [
                            (actor.get("cls"), actor.get("track_id"))
                            for actor in debug_actors
                        ]
                        sf_mod = sys.modules["smallFunction"]
                        print(
                            f"DEBUG holding_recheck={holding_result} actor_keys={actor_keys} "
                            f"smallFunction={getattr(sf_mod, '__file__', None)} "
                            f"classes={getattr(sf_mod, 'ACTOR_CLASSES', None)}"
                        )
                    thrower_key = litter_data.get("thrower_key")
                    vehicles = yolo_seg_cache.get("vehicles", [])
                    for vehicle in vehicles:
                        vehicle_key = (vehicle.get("cls"), int(vehicle.get("track_id", -1)))
                        if thrower_key is None or vehicle_key == thrower_key:
                            vehicle_track_id = int(vehicle.get("track_id", -1))
                            hist = vehicle_history[vehicle_track_id]["centroids"]
                            print(
                                f"DEBUG vehicle key={vehicle_key} box={vehicle.get('box')} "
                                f"centroids={list(hist)[-5:]} "
                                f"mask_metrics={_mask_metrics(litter_data.get('bbox'), vehicle)}"
                            )
                    printed_debug_ids.add(litter_id)

        processed_frames += 1
        frame_index += 1

    cap.release()
    if hasattr(litter_tracker, "close"):
        litter_tracker.close()
    return {
        "video": video_name,
        "frames": processed_frames,
        "confirmed_ids": len(confirmed_ids),
        "confirmed_frame_hits": confirmed_frame_hits,
        "first_confirmed_frame": first_confirmed_frame,
        "max_active_confirmed": max_active_confirmed,
    }


def _print_result(result):
    first_frame = result["first_confirmed_frame"]
    first_text = "None" if first_frame is None else str(first_frame)
    print(
        f"{result['video']}: frames={result['frames']} "
        f"confirmed_ids={result['confirmed_ids']} "
        f"confirmed_frame_hits={result['confirmed_frame_hits']} "
        f"first_confirmed_frame={first_text} "
        f"max_active_confirmed={result['max_active_confirmed']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", help="video names under resources/")
    parser.add_argument("--pipeline-dir", default="scripts-old-test")
    parser.add_argument("--bbox-model", default=DEFAULT_BBOX_MODEL)
    parser.add_argument("--trash-model", default=DEFAULT_TRASH_MODEL)
    parser.add_argument("--bbox-conf", type=float, default=0.45)
    parser.add_argument("--trash-conf", type=float, default=0.4)
    parser.add_argument("--moving-threshold", type=float, default=0.25)
    parser.add_argument("--core-moving-threshold", type=float, default=0.3)
    parser.add_argument("--yolo-seg-frame-skip", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--expect-positive", nargs="*", default=[])
    parser.add_argument("--expect-zero", nargs="*", default=[])
    parser.add_argument("--debug-confirmed", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("ACTION_DEVICE", "cpu")
    detect_mod, tracker_mod = _load_pipeline(args.pipeline_dir)
    model_bbox = YOLO(args.bbox_model, task="segment")
    model_trash = RTDETR(args.trash_model)

    failed = False
    for video_name in args.videos:
        result = _run_video(args, detect_mod, tracker_mod, model_bbox, model_trash, video_name)
        _print_result(result)
        if video_name in args.expect_positive and result["confirmed_ids"] <= 0:
            print(f"FAIL: {video_name} expected confirmed litter")
            failed = True
        if video_name in args.expect_zero and result["confirmed_ids"] != 0:
            print(f"FAIL: {video_name} expected zero confirmed litter")
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
