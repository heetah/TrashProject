# -*- coding: utf-8 -*-
# 車牌辨識模組：只對已鎖定違規的 vehicle/scooter ROI 派工，背景執行 YOLO 車牌偵測與 PaddleOCR。
import os
import threading
import numpy as np
from timeUtils import profile_block

# 全域模型與背景執行狀態：避免每幀重複載入 OCR/plate detector。
_plate_model = None
_ocr_model = None
_plate_lock = threading.Lock()          # 防止背景車牌任務重疊執行。
_plate_thread = None
_plate_disabled = False

try:
    import torch
except Exception:
    torch = None


def _get_plate_device():
    # 車牌 YOLO 使用 PLATE_DETECT_DEVICE；未指定時優先 CUDA，否則 CPU。
    env_device = os.environ.get("PLATE_DETECT_DEVICE")
    if env_device:
        return env_device
    if torch is not None and torch.cuda.is_available():
        return 0
    return "cpu"


def _can_use_half(device):
    # plate detector 只有 CUDA 裝置才使用 half precision。
    if torch is None or not torch.cuda.is_available():
        return False
    return str(device).lower() != "cpu"


def _get_plate_models(profiler=None):
    # lazy load 車牌偵測與 OCR 模型；第一次使用或 preload 時才載入。
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
    # 主流程開場預載並 warmup，避免第一次違規時才卡住推理。
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


def disable_license_plate_models():
    # CLI 可停用車牌流程，適合只驗證 litter/thrower pipeline。
    global _plate_disabled
    _plate_disabled = True


def _plate_worker(roi_items, vehicle_history, profiler=None):
    """背景重工作業：執行 YOLO 車牌偵測與 OCR。"""
    try:
        if _plate_disabled:
            return
        with profile_block(profiler, "license_plate.worker_total"):
            print('running.......')
            plate_model, ocr_model = _get_plate_models(profiler=profiler)

            plate_device = _get_plate_device()
            # 一次對多個 vehicle ROI 做 plate detection，降低模型呼叫次數。
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
                # 對每台車最多取一張高信心車牌，OCR 成功後寫回 vehicle_history。
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
                            history_entry = vehicle_history[vehicle['track_id']]
                            history_entry['license_plate'] = {
                                'number': legal_license_plate(plate_ocr_results['rec_text']),
                                'conf': plate_ocr_results['rec_score'],
                            }
                            history_entry['plate_search_until_found'] = False
                            history_entry['plate_blocked_since_litter'] = False
                            break
                    else:
                        vehicle_history[vehicle['track_id']]['plate_ocr_misses'] = (
                            vehicle_history[vehicle['track_id']].get('plate_ocr_misses', 0) + 1
                        )
    except Exception as exc:
        print(f"License plate detector/OCR worker failed: {exc}")
    finally:
        _plate_lock.release()


def wait_for_plate_jobs(profiler=None, timeout=None):
    # 影片結尾等待背景 OCR 完成，避免輸出摘要前仍有 thread 在跑。
    thread = _plate_thread
    if thread is not None and thread.is_alive():
        with profile_block(profiler, "cleanup.wait_license_plate_jobs"):
            thread.join(timeout=timeout)


def detect_license_plates(frame, vehicles, vehicle_history, skip=10, profiler=None):
    """呼叫端幾乎立即返回；車牌偵測與 OCR 會在背景 thread 執行。"""
    global _plate_thread
    if _plate_disabled:
        return

    vehicles = [
        vehicle for vehicle in vehicles
        if vehicle_history[vehicle['track_id']].get('license_plate') is None
    ]
    if not vehicles:
        return

    force_search = any(
        vehicle_history[vehicle['track_id']].get('plate_search_until_found', False)
        for vehicle in vehicles
    )

    if not hasattr(detect_license_plates, "counter"):
        detect_license_plates.counter = 0

    if detect_license_plates.counter < skip and not force_search:
        detect_license_plates.counter += 1
        return

    detect_license_plates.counter = 0

    # 若上一個背景任務還在跑，本幀略過，避免 OCR 堆積拖慢主流程。
    if not _plate_lock.acquire(blocking=False):
        return

    # 輕量前處理：在主 thread 裁切 vehicle ROI，再丟給背景任務。
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

            # copy 後呼叫端可安全覆寫 frame，不影響背景 OCR。
            vehicle_roi = frame[y1:y2, x1:x2].copy()
            if vehicle_roi.size == 0:
                continue

            roi_items.append((vehicle, vehicle_roi))

    if not roi_items:
        _plate_lock.release()
        return

    # 背景 thread：主流程不用等待車牌偵測/OCR。
    print("creating background thread for license plate detection + OCR...")
    t = threading.Thread(
        target=_plate_worker,
        args=(roi_items, vehicle_history, profiler),
        daemon=True,
    )
    _plate_thread = t
    t.start()


def dispatch_license_plate_rois(roi_items, vehicle_history, profiler=None):
    """從 backward resolver 的歷史 vehicle ROI 直接派工 OCR，不依賴車輛仍在當前畫面。"""
    global _plate_thread
    if _plate_disabled:
        return True

    prepared_items = []
    for vehicle, vehicle_roi in roi_items or []:
        if vehicle_roi is None or getattr(vehicle_roi, "size", 0) == 0:
            continue
        try:
            track_id = int(vehicle['track_id'])
        except (KeyError, TypeError, ValueError):
            continue
        if vehicle_history[track_id].get('license_plate') is not None:
            continue
        vehicle_history[track_id]['plate_search_until_found'] = True
        vehicle_history[track_id]['plate_search_source'] = 'backward'
        prepared_vehicle = dict(vehicle)
        prepared_vehicle['track_id'] = track_id
        prepared_items.append((prepared_vehicle, vehicle_roi.copy()))

    if not prepared_items:
        return True

    # 與一般 plate worker 共用鎖，避免兩條 OCR/YOLO plate 任務互搶 GPU/CPU。
    if not _plate_lock.acquire(blocking=False):
        return False

    print("creating background thread for backward license plate detection + OCR...")
    t = threading.Thread(
        target=_plate_worker,
        args=(prepared_items, vehicle_history, profiler),
        daemon=True,
    )
    _plate_thread = t
    t.start()
    return True


def legal_license_plate(plate_number):
    # 車牌字元正規化：常見 OCR 混淆字轉換並移除標點空白。
    new_plate_number = plate_number.replace("O", "0").replace("I", "1")
    new_plate_number = new_plate_number.replace(".", "").replace("-", "").replace(":", "").replace(" ", "")
    return new_plate_number


def get_plate_number(vehicle_history, id):
    # 渲染階段讀取已辨識車牌；尚未辨識則顯示 Unknown。
    if vehicle_history.get(id, {}).get('license_plate') is not None:
        return vehicle_history[id]['license_plate']['number']
    if vehicle_history.get(id, {}).get('plate_search_until_found', False):
        return "PlateSearching"
    return "Unknown"
