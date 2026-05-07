# -*- coding: utf-8 -*-
# 單幀與批次偵測核心：整合 actor tracking、litter detection、holding/motion filter、tracker 與畫面渲染。
import cv2
import numpy as np
import math
import os
from licensePlate import detect_license_plates, dispatch_license_plate_rois, get_plate_number
from timeUtils import profile_block
from smallFunction import (
    calculate_iou_matrix,
    litter_holding,
    motion_evidence,
)

# 類別顏色與名稱正規化：避免不同模型 label-space 造成 actor 解析錯誤。
BLACK = (0, 0, 0)
WARN = (0, 0, 255)
ACTOR_CLASSES = ('person', 'scooter', 'vehicle')
VEHICLE_LIKE_CLASSES = ('scooter', 'vehicle')
ACTION_WARNING_LABELS = {
    'littering': 'LITTERING',
    'urination': 'URINATING',
}
CLASS_ALIASES = {
    'motorcycle': 'scooter',
    'motorbike': 'scooter',
    'bike': 'scooter',
    'car': 'vehicle',
    'bus': 'vehicle',
    'truck': 'vehicle',
}
COCO_VEHICLE_IDS = {2, 3, 5, 7}
_WARNED_MESSAGES = set()

try:
    import torch
except Exception:
    torch = None


def _select_device(env_name):
    # 依環境變數選 GPU/CPU；若使用者要求 CUDA 但不可用，自動回退 CPU。
    requested = os.environ.get(env_name) or os.environ.get("YOLO_DEVICE")
    if requested:
        requested_str = str(requested).strip().lower()
        wants_cuda = requested_str.isdigit() or requested_str.startswith("cuda")
        if wants_cuda:
            if torch is not None and torch.cuda.is_available() and torch.cuda.device_count() > 0:
                return int(requested_str) if requested_str.isdigit() else requested
            print(f"{env_name}={requested} requested but CUDA is unavailable; fallback to CPU")
            return "cpu"
        return requested

    if torch is not None and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return 0
    return "cpu"


def _can_use_half(device):
    # half precision 只在 CUDA 裝置上啟用，避免 CPU/MPS 不支援。
    if torch is None or not torch.cuda.is_available():
        return False
    return str(device).lower() not in ("cpu", "mps")


BBOX_DEVICE = _select_device("BBOX_DEVICE")
TRASH_DEVICE = _select_device("TRASH_DEVICE")
BBOX_HALF = _can_use_half(BBOX_DEVICE)
TRASH_HALF = _can_use_half(TRASH_DEVICE)


def _model_class_name(model, cls_id):
    # 將模型 class id 轉成專案需要的 actor 類別，兼容自訂模型與 COCO fallback。
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        raw_name = names.get(int(cls_id), str(cls_id))
    elif isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
        raw_name = names[int(cls_id)]
    else:
        raw_name = str(cls_id)

    class_name = str(raw_name).strip().lower()
    class_name = CLASS_ALIASES.get(class_name, class_name)
    if class_name in ACTOR_CLASSES:
        return class_name

    # 自訂專案模型常用 id 0 表示 litter；若有明確名稱，不能套 COCO actor fallback。
    if not str(raw_name).strip().isdigit():
        return None

    # 舊 COCO 類模型相容路徑。
    if int(cls_id) == 0:
        return 'person'
    if int(cls_id) in COCO_VEHICLE_IDS:
        return 'vehicle'
    return None


def _clone_actors(actors):
    # 複製 actor dict，避免快取資料在後續流程被原地修改。
    return [dict(actor) for actor in actors]


def _as_result_list(results):
    # Ultralytics 有時回傳單物件、有時回傳 list；統一成 list 方便批次處理。
    if results is None:
        return []
    if isinstance(results, (list, tuple)):
        return list(results)
    return [results]


def _warn_once(key, message):
    # batch fallback 若每批都印會淹沒進度列；同類問題只提示一次。
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    print(message)


def _result_box_count(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return 0
    try:
        return len(boxes)
    except TypeError:
        return 0


def _add_stat(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def _warning_label_for_action(action):
    action_key = str(action or '').strip().lower()
    return ACTION_WARNING_LABELS.get(action_key, 'LITTERING')


def _actor_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _split_actors(actors):
    persons = []
    vehicles = []
    for actor in actors:
        if actor['cls'] == 'person':
            persons.append(actor)
        elif actor['cls'] in VEHICLE_LIKE_CLASSES:
            vehicles.append(actor)
    return persons, vehicles


def _extract_actor_tracks(results, model_bbox):
    # 從 YOLO track 結果抽出 person/vehicle/scooter，保留 bbox、track id 與 segmentation polygon。
    persons = []
    vehicles = []

    for result in results:
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        if result.masks is not None and getattr(result.masks, 'xy', None) is not None:
            mask_xy = result.masks.xy
        else:
            mask_xy = [None] * len(boxes)

        for det_idx, (box, cls_id, track_id) in enumerate(zip(boxes, classes, track_ids)):
            class_name = _model_class_name(model_bbox, cls_id)
            if class_name is None:
                continue

            actor_mask_poly = mask_xy[det_idx] if det_idx < len(mask_xy) else None
            # 將 actor 資訊統一放在一個 dict 裡，方便後續處理與擴展
            actor_info = {
                'box': box,
                'track_id': int(track_id),
                'cls': class_name,
                # 保留 polygon，讓 holding 可用 segmentation，而不是只看 bbox overlap。
                'mask_poly': actor_mask_poly,
            }

            if class_name == 'person':
                persons.append(actor_info)
            elif class_name in VEHICLE_LIKE_CLASSES:
                vehicles.append(actor_info)

    return persons, vehicles


def _extract_actor_detections(results, model_bbox):
    # YOLO predict 沒有 tracker id；先抽出 actor detection，後面用輕量 IoU id 補上。
    actors = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        if result.masks is not None and getattr(result.masks, 'xy', None) is not None:
            mask_xy = result.masks.xy
        else:
            mask_xy = [None] * len(boxes)

        for det_idx, (box, cls_id) in enumerate(zip(boxes, classes)):
            class_name = _model_class_name(model_bbox, cls_id)
            if class_name is None:
                continue

            actors.append({
                'box': box,
                'cls': class_name,
                'mask_poly': mask_xy[det_idx] if det_idx < len(mask_xy) else None,
            })

    return actors


def _assign_fast_actor_track_ids(actors, yolo_seg_cache, iou_threshold=0.3):
    # 極速模式用 predict 取代 BoT-SORT；用前一次 bbox IoU 給穩定 id，保留後續 holding/backtrack 基本需求。
    tracks = yolo_seg_cache.setdefault('fast_actor_tracks', [])
    next_id = int(yolo_seg_cache.get('fast_next_track_id', 1))
    used_tracks = set()
    assigned = []

    for actor in actors:
        best_idx = None
        best_iou = 0.0
        for idx, track in enumerate(tracks):
            if idx in used_tracks or track.get('cls') != actor['cls']:
                continue
            iou = _actor_iou(actor['box'], track['box'])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        actor = dict(actor)
        if best_idx is not None and best_iou >= iou_threshold:
            actor['track_id'] = int(tracks[best_idx]['track_id'])
            used_tracks.add(best_idx)
        else:
            actor['track_id'] = next_id
            next_id += 1
        assigned.append(actor)

    yolo_seg_cache['fast_next_track_id'] = next_id
    yolo_seg_cache['fast_actor_tracks'] = [
        {
            'track_id': int(actor['track_id']),
            'box': np.asarray(actor['box']).copy(),
            'cls': actor['cls'],
        }
        for actor in assigned
    ]
    return _split_actors(assigned)


def detect(frame, model_bbox, model_trash,
           color_dict, fg_mask, litter_tracker, vehicle_history,
           fps=30.0, vehicle_relative_speed_threshold_pct_per_s=20.0,
           enable_vehicle_speed_filter=True,
           violator_display_cache=None, violator_display_ttl=60,
           violator_display_max_jump=80.0,
           action_module=None,
           frame_index=None,
           yolo_seg_frame_skip=2,
           yolo_seg_cache=None,
           bbox_conf=0.3,
           trash_conf=0.5,
           profiler=None,
           moving_threshold=0.25,
           core_moving_threshold=0.3,
           motion_min_component_area=4,
           motion_min_largest_component_ratio=0.25,
           precomputed_persons=None,
           precomputed_vehicles=None,
           precomputed_trash_results=None,
           fg_mask_scale=1.0,
           stats=None,
           actor_mode="track",
           actor_track_iou=0.3):
    # 單幀偵測入口：負責一幀內完整 actor、litter、違規者、渲染流程。
    if violator_display_cache is None:
        violator_display_cache = {}
    if yolo_seg_cache is None:
        yolo_seg_cache = {}
    if frame_index is None:
        frame_index = int(yolo_seg_cache.get('next_frame_index', 0))
        yolo_seg_cache['next_frame_index'] = frame_index + 1
    yolo_seg_frame_skip = max(int(yolo_seg_frame_skip or 1), 1)

    # UI 顯示層快取：每幀倒數，讓違規框可持續顯示
    with profile_block(profiler, "detect.violator_cache_decay"):
        for actor_key in list(violator_display_cache.keys()):
            cache_entry = violator_display_cache[actor_key]
            if (
                cache_entry.get('until_plate_found', False) and
                actor_key[0] in VEHICLE_LIKE_CLASSES and
                vehicle_history.get(actor_key[1], {}).get('license_plate') is None
            ):
                cache_entry['ttl'] = max(int(cache_entry.get('ttl', 0)), int(violator_display_ttl))
                continue
            cache_entry['until_plate_found'] = False
            violator_display_cache[actor_key]['ttl'] -= 1
            if violator_display_cache[actor_key]['ttl'] <= 0:
                del violator_display_cache[actor_key]

    # 第一段：YOLO actor tracking。若 batch 外層已預先計算，直接使用 precomputed 結果。
    if precomputed_persons is not None and precomputed_vehicles is not None:
        persons = _clone_actors(precomputed_persons)
        vehicles = _clone_actors(precomputed_vehicles)
    else:
        should_run_yolo_seg = (
            frame_index % yolo_seg_frame_skip == 0 or
            'persons' not in yolo_seg_cache or
            'vehicles' not in yolo_seg_cache
        )
        if should_run_yolo_seg:
            # 每 N 幀才跑 YOLO tracking，其餘幀沿用快取以降低耗時。
            if actor_mode == "predict":
                with profile_block(profiler, "detect.yolo_actor_predict"):
                    results = model_bbox.predict(
                        frame,
                        conf=bbox_conf,
                        device=BBOX_DEVICE,
                        half=BBOX_HALF,
                        verbose=False,
                    )
                with profile_block(profiler, "detect.yolo_actor_parse"):
                    actor_detections = _extract_actor_detections(results, model_bbox)
                    persons, vehicles = _assign_fast_actor_track_ids(
                        actor_detections,
                        yolo_seg_cache,
                        iou_threshold=actor_track_iou,
                    )
            else:
                with profile_block(profiler, "detect.yolo_actor_track"):
                    results = model_bbox.track(
                        frame,
                        persist = True,
                        conf = bbox_conf,
                        device = BBOX_DEVICE,
                        half = BBOX_HALF,
                        verbose = False,
                        tracker = "botsort.yaml"
                    )
                with profile_block(profiler, "detect.yolo_actor_parse"):
                    persons, vehicles = _extract_actor_tracks(results, model_bbox)
            yolo_seg_cache['persons'] = _clone_actors(persons)
            yolo_seg_cache['vehicles'] = _clone_actors(vehicles)
            yolo_seg_cache['frame_index'] = frame_index
        else:
            # 快取重用：保留上次 actor 狀態，讓跳幀不會讓畫面完全沒有 actor。
            with profile_block(profiler, "detect.yolo_actor_cache_reuse"):
                persons = _clone_actors(yolo_seg_cache.get('persons', []))
                vehicles = _clone_actors(yolo_seg_cache.get('vehicles', []))

    annotated_frame = frame.copy()
    box_thickness = 4
    font_scale = 1.0
    text_thickness = 2

    # 計算重疊比例，建立 person 到最重疊車輛的對應
    person_vehicle_map = {}

    with profile_block(profiler, "detect.person_vehicle_map"):
        if persons and vehicles:
            person_boxes = [p['box'] for p in persons]
            person_ids = [p['track_id'] for p in persons]
            vehicle_boxes = [v['box'] for v in vehicles]

            # 將 vehicle 的 (cls, track_id) 組合成一個 list，方便後續建立對應關係
            detected_vehicle_keys = [(v['cls'], int(v['track_id'])) for v in vehicles]

            # 計算所有 person 與 vehicle 之間的 IoU，得到一個 shape 為 (num_persons, num_vehicles) 的矩陣
            iou_matrix = calculate_iou_matrix(person_boxes, vehicle_boxes)

            overlap_mask = np.any(iou_matrix > 0.2, axis=1)

            for i, has_overlap in enumerate(overlap_mask):
                if has_overlap:
                    max_veh_idx = np.argmax(iou_matrix[i])
                    person_vehicle_map[person_ids[i]] = detected_vehicle_keys[max_veh_idx]

    all_objects = persons + vehicles

    # 如果有動作模組，先取得每個人的動作資訊，供後續違規判斷使用
    person_action_map = {}
    if action_module is not None:
        with profile_block(profiler, "detect.action_update"):
            person_action_map = action_module.update(frame, persons, profiler=profiler, stats=stats)

    filtererd_objects = []

    tracking_objects = all_objects

    # 第二段：更新車輛中心點歷史，供 holding 與後續相對運動判斷使用。
    with profile_block(profiler, "detect.vehicle_history_filter"):
        for obj in all_objects:
            # 針對 vehicle 進行相對速度過濾
            if obj['cls'] in VEHICLE_LIKE_CLASSES:
                track_id = obj['track_id']
                x1, y1, x2, y2 = map(int, obj['box'])
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                vehicle_history[track_id]['centroids'].append(centroid)

            filtererd_objects.append(obj)

    # 第三段：RTDETR 全圖偵測垃圾。
    current_frame_litters = []

    # 直接使用全圖進行 RTDETR 追蹤/偵測
    if precomputed_trash_results is None:
        with profile_block(profiler, "detect.rtdetr_litter_predict"):
            chunk_results = model_trash.predict(
                frame,          # 直接傳入整張圖
                conf=trash_conf,
                device=TRASH_DEVICE,
                half=TRASH_HALF,
                verbose=False
            )
    else:
        chunk_results = _as_result_list(precomputed_trash_results)

    # 解析 RTDETR 結果：全圖推理不需要 ROI 座標偏移。
    with profile_block(profiler, "detect.rtdetr_parse"):
        for r_res in chunk_results:
            if r_res.boxes is None:
                continue
                
            for r_box in r_res.boxes:
                r_cls_id = int(r_box.cls[0])
                r_class_name = str(model_trash.names[r_cls_id]).strip().lower()
                r_conf = float(r_box.conf[0])

                if r_class_name == 'litter':
                    lx1, ly1, lx2, ly2 = map(int, r_box.xyxy[0])
                    bbox_width = lx2 - lx1
                    bbox_height = ly2 - ly1

                    # 基礎 bbox 尺寸與長寬比過濾：去掉極端扁長或過小雜訊。
                    aspect_ratio = bbox_width / max(bbox_height, 1e-6)
                    if aspect_ratio > 6.0 or aspect_ratio < 0.15 or bbox_width < 3 or bbox_height < 3:
                        continue
                    
                    # 座標已經是全域座標，直接加入本幀候選 litter。
                    current_frame_litters.append([lx1, ly1, lx2, ly2, r_conf])

    if stats is not None:
        stats['raw_litter_candidates'] = stats.get('raw_litter_candidates', 0) + len(current_frame_litters)
                
    # 第四段：motion + holding 前處理。只有真的在動、且不像仍被人車持有的 litter 才進 tracker。
    filtered_frame_litters = []
    with profile_block(profiler, "detect.motion_holding_filter"):
        for litter_box in current_frame_litters:
            lx1, ly1, lx2, ly2, _ = litter_box
            litter_w = max(int(lx2 - lx1), 1)
            litter_h = max(int(ly2 - ly1), 1)

            # 檢查整個 litter bbox 的前景像素比例，先排除靜止舊垃圾。
            is_moving = motion_evidence(
                fg_mask,
                (int(lx1), int(ly1), int(lx2), int(ly2)),
                threshold=moving_threshold,
                mask_scale=fg_mask_scale,
                min_component_area=motion_min_component_area,
                min_largest_component_ratio=motion_min_largest_component_ratio,
            )
            if not is_moving:
                continue

            # 在中心區域再做一次 motion 驗證，抑制「旁邊人車移動」造成的舊垃圾誤觸發
            if litter_w >= 8 and litter_h >= 8:
                core_x1 = int(lx1 + 0.2 * litter_w)
                core_y1 = int(ly1 + 0.2 * litter_h)
                core_x2 = int(lx2 - 0.2 * litter_w)
                core_y2 = int(ly2 - 0.2 * litter_h)

                is_core_moving = motion_evidence(
                    fg_mask,
                    (core_x1, core_y1, core_x2, core_y2),
                    threshold=core_moving_threshold,
                    mask_scale=fg_mask_scale,
                    min_component_area=motion_min_component_area,
                    min_largest_component_ratio=motion_min_largest_component_ratio,
                )
                if not is_core_moving:
                    continue

            # 從 tracker 取最近上一幀中心，供 holding 判斷相對位移與釋放方向。
            prev_litter_center = None
            prev_litter_missed = None
            prev_litter_history = None
            min_prev_dist = float('inf')
            curr_center = ((lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0)
            for l_data in litter_tracker.active_litters.values():
                prev_box = l_data['bbox']
                prev_center = (
                    (float(prev_box[0]) + float(prev_box[2])) / 2.0,
                    (float(prev_box[1]) + float(prev_box[3])) / 2.0,
                )
                dist = math.hypot(curr_center[0] - prev_center[0], curr_center[1] - prev_center[1])

                # 若距離小於追蹤器的距離閾值或小於目前找到的最近距離，則更新 prev_litter_center
                if dist < litter_tracker.distance_threshold and dist < min_prev_dist:
                    min_prev_dist = dist
                    prev_litter_center = prev_center
                    prev_litter_missed = int(l_data.get('missed', 0))
                    prev_litter_history = list(l_data.get('history', []))

            # 新出現目標先進 tracker 建立一個 history anchor；第二幀起才能判斷它
            # 是否相對車輛真的往下分離，避免把 resize.mp4 這類剛丟出的垃圾第一點擋掉。
            if prev_litter_center is None:
                filtered_frame_litters.append(litter_box)
                continue

            is_holding_like, _ = litter_holding(
                litter_box,
                tracking_objects,
                prev_litter_center=prev_litter_center,
                prev_litter_missed=prev_litter_missed,
                prev_litter_history=prev_litter_history,
                vehicle_history=vehicle_history,
            )
            if is_holding_like:
                continue

            filtered_frame_litters.append(litter_box)

    if stats is not None:
        stats['filtered_litter_candidates'] = stats.get('filtered_litter_candidates', 0) + len(filtered_frame_litters)

    # 第五段：更新 GlobalLitterTracker，將 pending litter 依軌跡轉成 confirmed。
    with profile_block(profiler, "detect.litter_tracker_update"):
        tracked_litters, active_violators = litter_tracker.update(
            filtered_frame_litters,
            tracking_objects,
            person_vehicle_map=person_vehicle_map,
            frame_index=frame_index,
            frame=frame,
            vehicle_history=vehicle_history,
        )
    if stats is not None:
        confirmed_ids = [
            litter_id for litter_id, litter_data in tracked_litters.items()
            if litter_data.get('state') == 'confirmed'
        ]
        if confirmed_ids:
            stats.setdefault('confirmed_litter_ids', set()).update(confirmed_ids)
            stats['confirmed_litter_frame_hits'] = stats.get('confirmed_litter_frame_hits', 0) + 1
            if stats.get('first_confirmed_litter_frame') is None:
                stats['first_confirmed_litter_frame'] = frame_index
    if person_action_map:
        # STGCN 若判定 person 正在違規動作，也可直接把人/車註冊成違規者。
        with profile_block(profiler, "detect.stgcn_violator_register"):
            stgcn_violators = litter_tracker.register_action_violators(
                person_action_map,
                tracking_objects,
                person_vehicle_map=person_vehicle_map,
                ttl=violator_display_ttl,
                frame_index=frame_index,
                vehicle_history=vehicle_history,
            )
        _add_stat(stats, "stgcn_registered_violators", len(stgcn_violators))
        active_violators = set(active_violators) | stgcn_violators

    # 車牌辨識只對已鎖定違規者派工；避免每 10 幀掃描所有車輛造成不必要延遲。
    with profile_block(profiler, "detect.plate_dispatch"):
        backward_plate_roi_items = []
        if hasattr(litter_tracker, "consume_backward_plate_roi_items"):
            backward_plate_roi_items = litter_tracker.consume_backward_plate_roi_items()
        if backward_plate_roi_items:
            dispatched = dispatch_license_plate_rois(backward_plate_roi_items, vehicle_history, profiler=profiler)
            if not dispatched and hasattr(litter_tracker, "restore_backward_plate_roi_items"):
                litter_tracker.restore_backward_plate_roi_items(backward_plate_roi_items)

        plate_target_keys = set(active_violators)
        plate_target_keys.update(
            key for key in violator_display_cache.keys()
            if key[0] in VEHICLE_LIKE_CLASSES
        )
        plate_target_vehicles = [
            obj for obj in vehicles
            if (obj['cls'], obj['track_id']) in plate_target_keys
        ]
        detect_license_plates(frame, plate_target_vehicles, vehicle_history, profiler=profiler)

    with profile_block(profiler, "detect.violator_cache_refresh"):
        for obj in tracking_objects:
            actor_key = (obj['cls'], obj['track_id'])
            if actor_key in active_violators:
                x1, y1, x2, y2 = map(float, obj['box'])
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                tracker_info = (
                    litter_tracker.get_violator_info(actor_key)
                    if hasattr(litter_tracker, "get_violator_info")
                    else {}
                )
                action_name = tracker_info.get('action')
                if not action_name and actor_key[0] == 'person':
                    action_name = person_action_map.get(actor_key[1], {}).get('action')
                until_plate_found = bool(tracker_info.get('until_plate_found', False))
                if actor_key[0] in VEHICLE_LIKE_CLASSES:
                    until_plate_found = until_plate_found or bool(
                        vehicle_history[obj['track_id']].get('plate_search_until_found', False) and
                        vehicle_history[obj['track_id']].get('license_plate') is None
                    )
                violator_display_cache[actor_key] = {
                    'ttl': int(violator_display_ttl),
                    'center': center,
                    'until_plate_found': until_plate_found,
                    'action': action_name,
                }

        # 已鎖定違規者即使被速度過濾暫時排除，也要保留渲染框。
        filtered_keys = {(obj['cls'], obj['track_id']) for obj in filtererd_objects}
        render_objects = list(filtererd_objects)
        for obj in tracking_objects:
            obj_key = (obj['cls'], obj['track_id'])
            if obj_key in violator_display_cache and obj_key not in filtered_keys:
                render_objects.append(obj)

    # 第六段：統一渲染 actor。違規者紅框，正常人車用各類別顏色。
    with profile_block(profiler, "detect.render_actors"):
        for obj in render_objects:
            x1, y1, x2, y2 = map(int, obj['box'])
            cls_name = obj['cls']
            track_id = obj['track_id']
            actor_key = (cls_name, track_id)

            # 檢查是否為「被反追蹤鎖定的丟擲者」(存在於違規快取中)
            cache_entry = violator_display_cache.get(actor_key)
            is_violator = cache_entry is not None

            if is_violator:
                current_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                saved_center = cache_entry.get('center', current_center)
                jump_dist = math.hypot(current_center[0] - saved_center[0], current_center[1] - saved_center[1])
                
                if jump_dist > violator_display_max_jump:
                    violator_display_cache.pop(actor_key, None)
                    is_violator = False 
                else:
                    cache_entry['center'] = current_center 

            # === 根據身分狀態決定 BBox 顏色 ===
            if is_violator:
                # 丟擲者確認：畫上紅色 BBox 
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), WARN, box_thickness + 2)
                plate_str = get_plate_number(vehicle_history, track_id) if cls_name in VEHICLE_LIKE_CLASSES else ""
                warning_label = _warning_label_for_action(cache_entry.get('action'))
                label_text = f"{cls_name} -{warning_label}- {plate_str}".strip()
                
                cv2.putText(annotated_frame, label_text, (x1, max(10, y1 - 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, WARN, text_thickness)
                            
            else:
                # 正常路人/車輛
                color = color_dict.get(cls_name, BLACK)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)
                label_text = cls_name
                if cls_name == 'person' and track_id in person_action_map:
                    action_info = person_action_map[track_id]
                    stgcn_conf = action_info.get('stgcn_conf', action_info.get('conf', 0.0))
                    if action_info.get('alert', False):
                        action_name = _warning_label_for_action(action_info.get('action'))
                        color = WARN
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness + 2)
                        label_text = f"person {action_name} STGCN {stgcn_conf:.2f}"
                    else:
                        label_text = f"person {action_info.get('action', 'normal')} STGCN {stgcn_conf:.2f}"

                cv2.putText(annotated_frame, label_text, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)

    # 第七段：只畫 confirmed litter，避免 pending 候選框造成誤解。
    with profile_block(profiler, "detect.render_litters"):
        for l_id, l_data in tracked_litters.items():
            lx1, ly1, lx2, ly2, conf = l_data['bbox']
            lx1, ly1, lx2, ly2 = map(int, [lx1, ly1, lx2, ly2])

            if l_data['state'] in ['confirmed']:
                l_color = color_dict['litter']
                cv2.rectangle(annotated_frame, (lx1, ly1), (lx2, ly2), l_color, box_thickness + 2)
                cv2.putText(annotated_frame, f"Litter {l_id} ({l_data['state']})", (lx1, max(20, ly1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, l_color, 2)

    return annotated_frame


def _run_batched_actor_track(model_bbox, frames, bbox_conf, export_batch_size=1, profiler=None, stats=None):
    # 批次 YOLO actor tracking；尾批/跳幀不足 batch 時 padding，輸出再裁回原長度。
    export_batch_size = max(int(export_batch_size or 1), 1)
    _add_stat(stats, "yolo_actor_infer_frames", len(frames))
    padded_frame_count = 0
    try:
        result_list = []
        chunk_size = export_batch_size if export_batch_size > 1 else 1
        for start in range(0, len(frames), chunk_size):
            chunk = list(frames[start:start + chunk_size])
            keep_count = len(chunk)
            if export_batch_size > len(chunk):
                padded_frame_count += export_batch_size - len(chunk)
                chunk.extend([chunk[-1]] * (export_batch_size - len(chunk)))
            source = chunk if len(chunk) > 1 else chunk[0]
            with profile_block(profiler, "detect.yolo_actor_track_batch"):
                results = model_bbox.track(
                    source,
                    persist=True,
                    conf=bbox_conf,
                    device=BBOX_DEVICE,
                    half=BBOX_HALF,
                    verbose=False,
                    tracker="botsort.yaml",
                )
            chunk_results = _as_result_list(results)
            if len(chunk_results) < keep_count:
                raise RuntimeError(f"expected at least {keep_count} YOLO results, got {len(chunk_results)}")
            result_list.extend(chunk_results[:keep_count])
        _add_stat(stats, "yolo_actor_padded_frames", padded_frame_count)
        return result_list[:len(frames)]
    except Exception as exc:
        if export_batch_size > len(frames):
            padded_frames = list(frames) + [frames[-1]] * (export_batch_size - len(frames))
            padded_source = padded_frames if len(padded_frames) > 1 else padded_frames[0]
            with profile_block(profiler, "detect.yolo_actor_track_fallback"):
                repair_results = model_bbox.track(
                    padded_source,
                    persist=True,
                    conf=bbox_conf,
                    device=BBOX_DEVICE,
                    half=BBOX_HALF,
                    verbose=False,
                    tracker="botsort.yaml",
                )
            repair_list = _as_result_list(repair_results)
            if len(repair_list) >= len(frames):
                _warn_once(
                    "yolo_batch_padding_repair",
                    f"Warning: YOLO actor batch needed padded source repair: {exc}",
                )
                return repair_list[:len(frames)]

        if len(frames) <= 1:
            raise

        _warn_once(
            "yolo_batch_per_frame_repair",
            f"Warning: batched YOLO actor tracking failed; falling back to per-frame tracking: {exc}",
        )
        result_list = []
        for frame in frames:
            repair_source = [frame] * export_batch_size if export_batch_size > 1 else frame
            with profile_block(profiler, "detect.yolo_actor_track_fallback"):
                single_results = model_bbox.track(
                    repair_source,
                    persist=True,
                    conf=bbox_conf,
                    device=BBOX_DEVICE,
                    half=BBOX_HALF,
                    verbose=False,
                    tracker="botsort.yaml",
                )
            single_list = _as_result_list(single_results)
            if not single_list:
                raise RuntimeError("YOLO actor fallback returned no results")
            result_list.append(single_list[0])
        return result_list


def _run_batched_actor_predict(model_bbox, frames, bbox_conf, export_batch_size=1, profiler=None, stats=None):
    # 極速模式：YOLO predict 支援真正 batch，避開 BoT-SORT track 的 per-frame 成本。
    export_batch_size = max(int(export_batch_size or 1), 1)
    _add_stat(stats, "yolo_actor_infer_frames", len(frames))
    padded_frame_count = 0
    result_list = []
    chunk_size = export_batch_size if export_batch_size > 1 else 1
    for start in range(0, len(frames), chunk_size):
        chunk = list(frames[start:start + chunk_size])
        keep_count = len(chunk)
        if export_batch_size > len(chunk):
            padded_frame_count += export_batch_size - len(chunk)
            chunk.extend([chunk[-1]] * (export_batch_size - len(chunk)))
        source = chunk if len(chunk) > 1 else chunk[0]
        with profile_block(profiler, "detect.yolo_actor_predict_batch"):
            results = model_bbox.predict(
                source,
                conf=bbox_conf,
                device=BBOX_DEVICE,
                half=BBOX_HALF,
                verbose=False,
            )
        chunk_results = _as_result_list(results)
        if len(chunk_results) < keep_count:
            raise RuntimeError(f"expected at least {keep_count} YOLO predict results, got {len(chunk_results)}")
        result_list.extend(chunk_results[:keep_count])
    _add_stat(stats, "yolo_actor_padded_frames", padded_frame_count)
    if len(result_list) < len(frames):
        raise RuntimeError(f"expected at least {len(frames)} YOLO predict results, got {len(result_list)}")
    return result_list[:len(frames)]


def _select_zero_repair_positions(box_counts, mode, context=None):
    if mode == "off":
        return []
    zero_positions = [idx for idx, count in enumerate(box_counts) if count == 0]
    if mode == "all":
        return zero_positions
    if mode != "adjacent":
        mode = "adjacent"

    selected = []
    last_count = 0
    if context is not None:
        last_count = int(context.get("last_trash_box_count", 0) or 0)
    for idx in zero_positions:
        prev_count = box_counts[idx - 1] if idx > 0 else last_count
        next_count = box_counts[idx + 1] if idx + 1 < len(box_counts) else 0
        if prev_count > 0 or next_count > 0:
            selected.append(idx)
    return selected


def _run_batched_trash_predict(model_trash, frames, trash_conf, export_batch_size,
                               profiler=None, zero_repair="adjacent",
                               zero_repair_context=None, stats=None):
    # 批次 RTDETR litter predict；尾端不足 batch 時用最後一幀 padding，輸出再裁回原長度。
    infer_frames = list(frames)
    if export_batch_size > len(infer_frames):
        infer_frames.extend([infer_frames[-1]] * (export_batch_size - len(infer_frames)))

    source = infer_frames if len(infer_frames) > 1 else infer_frames[0]
    try:
        with profile_block(profiler, "detect.rtdetr_litter_predict_batch"):
            results = model_trash.predict(
                source,
                conf=trash_conf,
                device=TRASH_DEVICE,
                half=TRASH_HALF,
                verbose=False,
            )
    except Exception as exc:
        if len(frames) <= 1:
            raise
        _warn_once(
            "rtdetr_batch_exception",
            f"Warning: batched RTDETR failed; repairing frame-by-frame with padded batch source: {exc}",
        )
        result_list = []
        for frame in frames:
            repair_source = [frame] * export_batch_size if export_batch_size > 1 else frame
            with profile_block(profiler, "detect.rtdetr_litter_predict_batch_repair"):
                repair_results = model_trash.predict(
                    repair_source,
                    conf=trash_conf,
                    device=TRASH_DEVICE,
                    half=TRASH_HALF,
                    verbose=False,
                )
            repair_list = _as_result_list(repair_results)
            if not repair_list:
                raise RuntimeError("RTDETR repair returned no results") from exc
            result_list.append(repair_list[0])
        return result_list

    result_list = _as_result_list(results)
    if len(result_list) < len(frames):
        raise RuntimeError(f"expected at least {len(frames)} RTDETR results, got {len(result_list)}")

    result_list = result_list[:len(frames)]
    box_counts = [_result_box_count(result) for result in result_list]
    zero_positions = [idx for idx, count in enumerate(box_counts) if count == 0]
    _add_stat(stats, "rtdetr_batch_zero_frames", len(zero_positions))

    # TensorRT batch RTDETR 曾出現 mixed batch 中某些 frame 掉成 0 box。
    # 預設只補前後有 detection 的可疑 0 frame，避免長影片空 frame 被整批雙跑。
    if zero_positions and any(count > 0 for count in box_counts):
        repair_positions = _select_zero_repair_positions(
            box_counts,
            zero_repair,
            context=zero_repair_context,
        )
        _warn_once(
            "rtdetr_batch_zero_repair",
            f"Warning: RTDETR batch returned mixed zero-box frames; zero repair mode={zero_repair}, "
            f"repairing {len(repair_positions)}/{len(zero_positions)} zero frames.",
        )
        _add_stat(stats, "rtdetr_batch_zero_repaired_frames", len(repair_positions))
        for pos in repair_positions:
            frame = frames[pos]
            repair_source = [frame] * export_batch_size if export_batch_size > 1 else frame
            with profile_block(profiler, "detect.rtdetr_litter_predict_batch_repair"):
                repair_results = model_trash.predict(
                    repair_source,
                    conf=trash_conf,
                    device=TRASH_DEVICE,
                    half=TRASH_HALF,
                    verbose=False,
                )
            repair_list = _as_result_list(repair_results)
            if repair_list:
                result_list[pos] = repair_list[0]

    if zero_repair_context is not None and result_list:
        zero_repair_context["last_trash_box_count"] = _result_box_count(result_list[-1])

    return result_list


def _prepare_actor_batch(frames, frame_indices, model_bbox, bbox_conf,
                         yolo_seg_frame_skip, yolo_seg_cache,
                         batch_size=1, bbox_batch_size=None,
                         actor_mode="track", actor_track_iou=0.3,
                         profiler=None, stats=None):
    # 為 detect_batch 準備每幀 actor 結果：該跑 YOLO 的幀跑推理，其餘幀沿用快取。
    actor_pairs = [None] * len(frames)
    run_positions = []
    run_frames = []
    cache_available = 'persons' in yolo_seg_cache and 'vehicles' in yolo_seg_cache

    for pos, (frame, frame_index) in enumerate(zip(frames, frame_indices)):
        should_run_yolo_seg = (
            frame_index % yolo_seg_frame_skip == 0 or
            not cache_available
        )
        if should_run_yolo_seg:
            run_positions.append(pos)
            run_frames.append(frame)
            cache_available = True
        else:
            actor_pairs[pos] = None

    run_result_map = {}
    if run_frames:
        actor_export_batch_size = int(bbox_batch_size or batch_size)
        if actor_mode == "predict":
            run_results = _run_batched_actor_predict(
                model_bbox,
                run_frames,
                bbox_conf,
                export_batch_size=actor_export_batch_size,
                profiler=profiler,
                stats=stats,
            )
        else:
            run_results = _run_batched_actor_track(
                model_bbox,
                run_frames,
                bbox_conf,
                export_batch_size=actor_export_batch_size,
                profiler=profiler,
                stats=stats,
            )
        for pos, result in zip(run_positions, run_results):
            run_result_map[pos] = result

    for pos, frame_index in enumerate(frame_indices):
        if pos in run_result_map:
            with profile_block(profiler, "detect.yolo_actor_parse_batch"):
                if actor_mode == "predict":
                    actor_detections = _extract_actor_detections([run_result_map[pos]], model_bbox)
                    persons, vehicles = _assign_fast_actor_track_ids(
                        actor_detections,
                        yolo_seg_cache,
                        iou_threshold=actor_track_iou,
                    )
                else:
                    persons, vehicles = _extract_actor_tracks([run_result_map[pos]], model_bbox)
            yolo_seg_cache['persons'] = _clone_actors(persons)
            yolo_seg_cache['vehicles'] = _clone_actors(vehicles)
            yolo_seg_cache['frame_index'] = frame_index
            actor_pairs[pos] = (persons, vehicles)

        if actor_pairs[pos] is None:
            actor_pairs[pos] = (
                _clone_actors(yolo_seg_cache.get('persons', [])),
                _clone_actors(yolo_seg_cache.get('vehicles', [])),
            )

    return actor_pairs


def detect_batch(frames, model_bbox, model_trash,
                 color_dict, fg_masks, litter_tracker, vehicle_history,
                 fps=30.0, vehicle_relative_speed_threshold_pct_per_s=20.0,
                 enable_vehicle_speed_filter=True,
                 violator_display_cache=None, violator_display_ttl=60,
                 violator_display_max_jump=80.0,
                 action_module=None,
                 frame_start_index=0,
                 yolo_seg_frame_skip=2,
                 yolo_seg_cache=None,
                 bbox_conf=0.3,
                 trash_conf=0.5,
                 profiler=None,
                 moving_threshold=0.25,
                 core_moving_threshold=0.3,
                 motion_min_component_area=4,
                 motion_min_largest_component_ratio=0.25,
                 batch_size=1,
                 bbox_batch_size=None,
                 trash_batch_size=None,
                 fg_mask_scale=1.0,
                 stats=None,
                 rtdetr_zero_repair="adjacent",
                 rtdetr_batch_context=None,
                 actor_mode="track",
                 actor_track_iou=0.3):
    # 批次偵測入口：RTDETR 批次推理、actor 可跳幀快取，最後逐幀套用單幀後處理。
    if not frames:
        return []
    if len(frames) != len(fg_masks):
        raise ValueError(f"frames and fg_masks length mismatch: {len(frames)} != {len(fg_masks)}")
    if violator_display_cache is None:
        violator_display_cache = {}
    if yolo_seg_cache is None:
        yolo_seg_cache = {}

    batch_size = max(int(batch_size or 1), 1)
    trash_batch_size = max(int(trash_batch_size or batch_size), 1)
    frame_indices = [int(frame_start_index) + i for i in range(len(frames))]

    if batch_size <= 1:
        # batch 被關閉時走單幀流程；batch N 的尾批仍需走 batch path，固定 TensorRT engine 才能 padding 到 N。
        return [
            detect(
                frame, model_bbox, model_trash, color_dict,
                fg_mask, litter_tracker, vehicle_history,
                fps=fps,
                vehicle_relative_speed_threshold_pct_per_s=vehicle_relative_speed_threshold_pct_per_s,
                enable_vehicle_speed_filter=enable_vehicle_speed_filter,
                violator_display_cache=violator_display_cache,
                violator_display_ttl=violator_display_ttl,
                violator_display_max_jump=violator_display_max_jump,
                action_module=action_module,
                frame_index=frame_index,
                yolo_seg_frame_skip=yolo_seg_frame_skip,
                yolo_seg_cache=yolo_seg_cache,
                bbox_conf=bbox_conf,
                trash_conf=trash_conf,
                profiler=profiler,
                moving_threshold=moving_threshold,
                core_moving_threshold=core_moving_threshold,
                motion_min_component_area=motion_min_component_area,
                motion_min_largest_component_ratio=motion_min_largest_component_ratio,
                fg_mask_scale=fg_mask_scale,
                stats=stats,
                actor_mode=actor_mode,
                actor_track_iou=actor_track_iou,
            )
            for frame, fg_mask, frame_index in zip(frames, fg_masks, frame_indices)
        ]

    # 批次前處理：actor 結果可跳幀快取，litter 結果用 RTDETR 批次推理。
    actor_pairs = _prepare_actor_batch(
        frames,
        frame_indices,
        model_bbox,
        bbox_conf,
        yolo_seg_frame_skip,
        yolo_seg_cache,
        batch_size=batch_size,
        bbox_batch_size=bbox_batch_size,
        actor_mode=actor_mode,
        actor_track_iou=actor_track_iou,
        profiler=profiler,
        stats=stats,
    )
    trash_results = _run_batched_trash_predict(
        model_trash,
        frames,
        trash_conf,
        trash_batch_size,
        profiler=profiler,
        zero_repair=rtdetr_zero_repair,
        zero_repair_context=rtdetr_batch_context,
        stats=stats,
    )

    annotated_frames = []
    # 批次推理完成後，逐幀套用相同的 motion/holding/tracker/render 後處理。
    for frame, fg_mask, frame_index, actor_pair, trash_result in zip(
        frames, fg_masks, frame_indices, actor_pairs, trash_results
    ):
        persons, vehicles = actor_pair
        annotated_frames.append(
            detect(
                frame, model_bbox, model_trash, color_dict,
                fg_mask, litter_tracker, vehicle_history,
                fps=fps,
                vehicle_relative_speed_threshold_pct_per_s=vehicle_relative_speed_threshold_pct_per_s,
                enable_vehicle_speed_filter=enable_vehicle_speed_filter,
                violator_display_cache=violator_display_cache,
                violator_display_ttl=violator_display_ttl,
                violator_display_max_jump=violator_display_max_jump,
                action_module=action_module,
                frame_index=frame_index,
                yolo_seg_frame_skip=yolo_seg_frame_skip,
                yolo_seg_cache=yolo_seg_cache,
                bbox_conf=bbox_conf,
                trash_conf=trash_conf,
                profiler=profiler,
                moving_threshold=moving_threshold,
                core_moving_threshold=core_moving_threshold,
                motion_min_component_area=motion_min_component_area,
                motion_min_largest_component_ratio=motion_min_largest_component_ratio,
                precomputed_persons=persons,
                precomputed_vehicles=vehicles,
                precomputed_trash_results=[trash_result],
                fg_mask_scale=fg_mask_scale,
                stats=stats,
                actor_mode=actor_mode,
                actor_track_iou=actor_track_iou,
            )
        )

    return annotated_frames
