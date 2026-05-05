import cv2
import numpy as np
import math
import os
from licensePlate import detect_license_plates, get_plate_number
from timeUtils import profile_block
from smallFunction import (
    calculate_iou_matrix,
    check_motion,
    litter_holding,
)

BLACK = (0, 0, 0)
WARN = (0, 0, 255)
ACTOR_CLASSES = ('person', 'scooter', 'vehicle')
VEHICLE_LIKE_CLASSES = ('scooter', 'vehicle')
CLASS_ALIASES = {
    'motorcycle': 'scooter',
    'motorbike': 'scooter',
    'bike': 'scooter',
    'car': 'vehicle',
    'bus': 'vehicle',
    'truck': 'vehicle',
}
COCO_VEHICLE_IDS = {2, 3, 5, 7}

try:
    import torch
except Exception:
    torch = None


def _select_device(env_name):
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
    if torch is None or not torch.cuda.is_available():
        return False
    return str(device).lower() not in ("cpu", "mps")


BBOX_DEVICE = _select_device("BBOX_DEVICE")
TRASH_DEVICE = _select_device("TRASH_DEVICE")
BBOX_HALF = _can_use_half(BBOX_DEVICE)
TRASH_HALF = _can_use_half(TRASH_DEVICE)


def _model_class_name(model, cls_id):
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

    # Custom project models use id 0 for litter. Do not apply COCO id fallback
    # when the model provides an explicit non-actor class name.
    if not str(raw_name).strip().isdigit():
        return None

    # Backward-compatible fallback for older COCO-style person/vehicle models.
    if int(cls_id) == 0:
        return 'person'
    if int(cls_id) in COCO_VEHICLE_IDS:
        return 'vehicle'
    return None


def _clone_actors(actors):
    return [dict(actor) for actor in actors]


def _extract_actor_tracks(results, model_bbox):
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
                # Keep polygon so holding can use segmentation instead of bbox-only overlap.
                'mask_poly': actor_mask_poly,
            }

            if class_name == 'person':
                persons.append(actor_info)
            elif class_name in VEHICLE_LIKE_CLASSES:
                vehicles.append(actor_info)

    return persons, vehicles


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
           core_moving_threshold=0.3,):
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
            violator_display_cache[actor_key]['ttl'] -= 1
            if violator_display_cache[actor_key]['ttl'] <= 0:
                del violator_display_cache[actor_key]

    # 1. 第一次偵測
    # === YOLO 人與車輛偵測 ===
    should_run_yolo_seg = (
        frame_index % yolo_seg_frame_skip == 0 or
        'persons' not in yolo_seg_cache or
        'vehicles' not in yolo_seg_cache
    )
    if should_run_yolo_seg:
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
            person_action_map = action_module.update(frame, persons, profiler=profiler)

    filtererd_objects = []

    tracking_objects = all_objects

    # 第二次遍歷：依 bbox 相對速度過濾車輛
    with profile_block(profiler, "detect.vehicle_history_filter"):
        for obj in all_objects:
            # 針對 vehicle 進行相對速度過濾
            if obj['cls'] in VEHICLE_LIKE_CLASSES:
                track_id = obj['track_id']
                x1, y1, x2, y2 = map(int, obj['box'])
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                vehicle_history[track_id]['centroids'].append(centroid)

            filtererd_objects.append(obj)

    # 2. 第二次偵測: RTDETR 全圖偵測垃圾
    current_frame_litters = []

    # 直接使用全圖進行 RTDETR 追蹤/偵測
    with profile_block(profiler, "detect.rtdetr_litter_predict"):
        chunk_results = model_trash.predict(
            frame,          # 直接傳入整張圖 
            conf=trash_conf,
            device=TRASH_DEVICE,
            half=TRASH_HALF,
            verbose=False
        )

    # 解析 RTDETR 結果 (因為是全圖，不須再加 ROI 偏移量)
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

                    # 垃圾長寬比例過濾，防止動態模糊被誤殺
                    aspect_ratio = bbox_width / max(bbox_height, 1e-6)
                    if aspect_ratio > 6.0 or aspect_ratio < 0.15 or bbox_width < 3 or bbox_height < 3:
                        continue
                    
                    # 座標已經是全域的，直接加入
                    current_frame_litters.append([lx1, ly1, lx2, ly2, r_conf])
                
    # detect 端先做 motion + holding 過濾，tracker 僅做反追蹤關聯
    filtered_frame_litters = []
    with profile_block(profiler, "detect.motion_holding_filter"):
        for litter_box in current_frame_litters:
            lx1, ly1, lx2, ly2, _ = litter_box
            litter_w = max(int(lx2 - lx1), 1)
            litter_h = max(int(ly2 - ly1), 1)

            # 傳入整張 mask 與litter 中心座標，讓 check_motion 自行決定要檢查整個 bbox 還是中心點，並根據目標大小調整閾值
            is_moving = check_motion(
                fg_mask,
                (int(lx1), int(ly1), int(lx2), int(ly2)),
                threshold=moving_threshold,
            )
            if not is_moving:
                continue

            # 在中心區域再做一次 motion 驗證，抑制「旁邊人車移動」造成的舊垃圾誤觸發
            if litter_w >= 8 and litter_h >= 8:
                core_x1 = int(lx1 + 0.2 * litter_w)
                core_y1 = int(ly1 + 0.2 * litter_h)
                core_x2 = int(lx2 - 0.2 * litter_w)
                core_y2 = int(ly2 - 0.2 * litter_h)

                is_core_moving = check_motion(
                    fg_mask,
                    (core_x1, core_y1, core_x2, core_y2),
                    threshold=core_moving_threshold,
                )
                if not is_core_moving:
                    continue

            # 嘗試從現有 tracker 軌跡取最近上一幀中心，供 holding 判斷
            prev_litter_center = None
            prev_litter_missed = None
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
                vehicle_history=vehicle_history,
            )
            if is_holding_like:
                continue

            filtered_frame_litters.append(litter_box)

    # 更新垃圾追蹤器，取得目前的追蹤狀態
    with profile_block(profiler, "detect.litter_tracker_update"):
        tracked_litters, active_violators = litter_tracker.update(
            filtered_frame_litters,
            tracking_objects,
            person_vehicle_map=person_vehicle_map
        )
    if person_action_map:
        with profile_block(profiler, "detect.stgcn_violator_register"):
            stgcn_violators = litter_tracker.register_action_violators(
                person_action_map,
                tracking_objects,
                person_vehicle_map=person_vehicle_map,
                ttl=violator_display_ttl,
            )
        active_violators = set(active_violators) | stgcn_violators

    # 車牌辨識只對已鎖定違規者派工；避免每 10 幀掃描所有車輛造成不必要延遲。
    with profile_block(profiler, "detect.plate_dispatch"):
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
                violator_display_cache[actor_key] = {
                    'ttl': int(violator_display_ttl),
                    'center': center,
                }

        # Ensure locked violators remain drawable even when temporarily filtered by speed.
        filtered_keys = {(obj['cls'], obj['track_id']) for obj in filtererd_objects}
        render_objects = list(filtererd_objects)
        for obj in tracking_objects:
            obj_key = (obj['cls'], obj['track_id'])
            if obj_key in violator_display_cache and obj_key not in filtered_keys:
                render_objects.append(obj)

    # 2. 統一渲染：畫出人、車與違規紅框
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
                label_text = f"{cls_name} -LITTERING- {plate_str}".strip()
                
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
                        color = WARN
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness + 2)
                        label_text = f"person LITTERING STGCN {stgcn_conf:.2f}"
                    else:
                        label_text = f"person {action_info.get('action', 'normal')} STGCN {stgcn_conf:.2f}"

                cv2.putText(annotated_frame, label_text, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)

    # 3. 統一渲染：畫出垃圾追蹤框
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
