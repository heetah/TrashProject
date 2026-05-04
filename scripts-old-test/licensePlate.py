import os
import threading
import numpy as np
from timeUtils import profile_block

_plate_model = None
_ocr_model = None
_plate_lock = threading.Lock()          # prevents overlapping background jobs
_plate_thread = None
_plate_disabled = False

try:
    import torch
except Exception:
    torch = None


def _get_plate_device():
    env_device = os.environ.get("PLATE_DETECT_DEVICE")
    if env_device:
        return env_device
    if torch is not None and torch.cuda.is_available():
        return 0
    return "cpu"


def _can_use_half(device):
    if torch is None or not torch.cuda.is_available():
        return False
    return str(device).lower() != "cpu"


def _get_plate_models(profiler=None):
    global _plate_model, _ocr_model

    if _plate_model is None:
        from ultralytics import YOLO
        with profile_block(profiler, "model_load.plate_yolo"):
            _plate_model = YOLO("test_ocr/license_plate_model/runs/detect/license_plate/yolo26n_v1/weights/best.pt")

    if _ocr_model is None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddlex import create_model
        ocr_device = os.environ.get("PLATE_OCR_DEVICE", "cpu")
        with profile_block(profiler, "model_load.plate_ocr"):
            _ocr_model = create_model(model_name="en_PP-OCRv5_mobile_rec", device=ocr_device)

    return _plate_model, _ocr_model


def preload_license_plate_models(profiler=None):
    global _plate_disabled
    try:
        with profile_block(profiler, "model_load.plate_models_total"):
            plate_model, ocr_model = _get_plate_models(profiler=profiler)
        plate_device = _get_plate_device()
        dummy_vehicle_roi = np.zeros((160, 320, 3), dtype=np.uint8)
        dummy_plate_roi = np.zeros((48, 160, 3), dtype=np.uint8)
        with profile_block(profiler, "model_warmup.plate_yolo"):
            plate_model.predict(
                dummy_vehicle_roi,
                save=False,
                max_det=1,
                conf=0.01,
                verbose=False,
                device=plate_device,
                half=_can_use_half(plate_device),
            )
        with profile_block(profiler, "model_warmup.plate_ocr"):
            list(ocr_model.predict(dummy_plate_roi))
        _plate_disabled = False
        print("License plate detector and OCR models preloaded.")
        return True
    except Exception as exc:
        _plate_disabled = True
        print(f"License plate detector/OCR preload failed; plate OCR disabled: {exc}")
        return False


def _plate_worker(roi_items, vehicle_history, profiler=None):
    """Heavy lifting: runs YOLO plate detection + OCR in a background thread."""
    try:
        if _plate_disabled:
            return
        with profile_block(profiler, "license_plate.worker_total"):
            print('running.......')
            plate_model, ocr_model = _get_plate_models(profiler=profiler)

            plate_device = _get_plate_device()
            with profile_block(profiler, "license_plate.yolo_plate_predict"):
                plate_results = plate_model.predict(
                    [roi for _, roi in roi_items],
                    save=False,
                    max_det=1,
                    conf=0.6,
                    verbose=False,
                    device=plate_device,
                    half=_can_use_half(plate_device),
                )

            if not isinstance(plate_results, list):
                plate_results = [plate_results]

            with profile_block(profiler, "license_plate.parse_and_ocr"):
                for (vehicle, vehicle_roi), plate_box_result in zip(roi_items, plate_results):
                    if not plate_box_result or plate_box_result.boxes is None:
                        continue

                    boxes = plate_box_result.boxes.xyxy.cpu().numpy()
                    confs = plate_box_result.boxes.conf.cpu().numpy()

                    for box, box_conf in zip(boxes, confs):
                        px1, py1, px2, py2 = map(int, box)
                        px1 = max(0, min(px1, vehicle_roi.shape[1]))
                        px2 = max(0, min(px2, vehicle_roi.shape[1]))
                        py1 = max(0, min(py1, vehicle_roi.shape[0]))
                        py2 = max(0, min(py2, vehicle_roi.shape[0]))
                        if px2 <= px1 or py2 <= py1:
                            continue

                        plate_img = vehicle_roi[py1:py2, px1:px2]
                        if plate_img.size == 0:
                            continue

                        with profile_block(profiler, "license_plate.ocr_predict"):
                            plate_ocr_results = list(ocr_model.predict(plate_img))[0]
                        print("OCR results:", plate_ocr_results['rec_text'], " rec_score:", plate_ocr_results['rec_score'], " box_conf:", box_conf)
                        if (
                            plate_ocr_results['rec_score'] >= 0.85 and
                            float(box_conf) >= 0.8
                        ):
                            print(
                                "[Plate]",
                                vehicle['track_id'],
                                legal_license_plate(plate_ocr_results['rec_text']),
                                plate_ocr_results['rec_score']
                            )
                            vehicle_history[vehicle['track_id']]['license_plate'] = {
                                'number': legal_license_plate(plate_ocr_results['rec_text']),
                                'conf': plate_ocr_results['rec_score'],
                            }
                            break
    except Exception as exc:
        print(f"License plate detector/OCR worker failed: {exc}")
    finally:
        _plate_lock.release()


def wait_for_plate_jobs(profiler=None, timeout=None):
    thread = _plate_thread
    if thread is not None and thread.is_alive():
        with profile_block(profiler, "cleanup.wait_license_plate_jobs"):
            thread.join(timeout=timeout)


def detect_license_plates(frame, vehicles, vehicle_history, skip=10, profiler=None):
    """Returns almost immediately. Plate detection + OCR runs in a background thread."""
    global _plate_thread
    if _plate_disabled:
        return

    vehicles = [
        vehicle for vehicle in vehicles
        if vehicle_history[vehicle['track_id']].get('license_plate') is None
    ]
    if not vehicles:
        return

    if not hasattr(detect_license_plates, "counter"):
        detect_license_plates.counter = 0

    if detect_license_plates.counter < skip:
        detect_license_plates.counter += 1
        return

    detect_license_plates.counter = 0

    # If a previous job is still running, skip this invocation
    if not _plate_lock.acquire(blocking=False):
        return

    # --- lightweight prep (still on the caller's thread) ---
    with profile_block(profiler, "license_plate.prepare_rois"):
        frame_h, frame_w = frame.shape[:2]
        roi_items = []
        for vehicle in vehicles:
            x1, y1, x2, y2 = map(int, vehicle['box'])
            x1 = max(0, min(x1, frame_w))
            x2 = max(0, min(x2, frame_w))
            y1 = max(0, min(y1, frame_h))
            y2 = max(0, min(y2, frame_h))
            if x2 <= x1 or y2 <= y1:
                continue

            # .copy() so the caller can safely reuse / mutate the frame
            vehicle_roi = frame[y1:y2, x1:x2].copy()
            if vehicle_roi.size == 0:
                continue

            roi_items.append((vehicle, vehicle_roi))

    if not roi_items:
        _plate_lock.release()
        return

    # --- fire-and-forget background thread ---
    print("creating background thread for license plate detection + OCR...")
    t = threading.Thread(
        target=_plate_worker,
        args=(roi_items, vehicle_history, profiler),
        daemon=True,
    )
    _plate_thread = t
    t.start()


def legal_license_plate(plate_number):
    new_plate_number = plate_number.replace("O", "0").replace("I", "1")
    new_plate_number = new_plate_number.replace(".", "").replace("-", "").replace(":", "").replace(" ", "")
    return new_plate_number


def get_plate_number(vehicle_history, id):
    if vehicle_history.get(id, {}).get('license_plate') is not None:
        return vehicle_history[id]['license_plate']['number']
    return "Unknown"
