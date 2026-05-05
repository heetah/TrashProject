# -*- coding: utf-8 -*-
# 共用幾何與判斷工具：前景 motion、IoU、mask overlap、holding 判斷、物理軌跡驗證。
import math
import cv2
import numpy as np

# actor 類別定義：holding 只處理人、車、機車，避免其他類別干擾。
ACTOR_CLASSES = ('person', 'vehicle', 'scooter')
VEHICLE_LIKE_CLASSES = ('vehicle', 'scooter')

def check_motion(fg_mask, coords, threshold, mask_scale=1.0):
    # 檢查指定 bbox 內前景像素比例；用於排除靜止舊垃圾或雜訊框。
    x1, y1, x2, y2 = coords
    scale = float(mask_scale or 1.0)
    if scale != 1.0:
        x1 = int(math.floor(float(x1) * scale))
        y1 = int(math.floor(float(y1) * scale))
        x2 = int(math.ceil(float(x2) * scale))
        y2 = int(math.ceil(float(y2) * scale))
    h, w = fg_mask.shape[:2]
    
    # 嚴格限制邊界，避免負數索引與越界。
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    # 裁切 check_motion 區域
    mask_crop = fg_mask[y1:y2, x1:x2]
    
    if mask_crop.size == 0:
        return False
    
    white_pixels = np.sum(mask_crop == 255)
    total_pixels = mask_crop.size

    ratio = white_pixels / total_pixels

    return ratio >= threshold

def calculate_iou_matrix(boxes1, boxes2):
    # 計算兩組 bbox 的 IoU 矩陣，用於 person 與 vehicle 關聯。
    b1 = np.array(boxes1)
    b2 = np.array(boxes2)

    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)))
    
    # 擴張維度
    b1 = np.expand_dims(b1, axis=1)
    b2 = np.expand_dims(b2, axis=0)

    # 計算重疊的左上 & 右下座標
    xx1 = np.maximum(b1[..., 0], b2[..., 0])
    yy1 = np.maximum(b1[..., 1], b2[..., 1])
    xx2 = np.minimum(b1[..., 2], b2[..., 2])
    yy2 = np.minimum(b1[..., 3], b2[..., 3])

    # 計算重疊區域面積
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter_area = w * h

    # 計算各自面積
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    # 計算聯集面積
    union_area = area1 + area2 - inter_area
    union_area = np.maximum(union_area, 1e-6) # 避免除以 0

    return inter_area / union_area


def calculate_mask_overlap_ratio(litter_box, actor_mask_poly):
    # 計算 litter bbox 被 actor segmentation mask 覆蓋的比例，支援 polygon-aware holding。
    if actor_mask_poly is None:
        return None

    poly = np.asarray(actor_mask_poly, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return None

    lx1, ly1, lx2, ly2 = map(float, litter_box[:4])
    rx1 = int(math.floor(min(lx1, lx2)))
    ry1 = int(math.floor(min(ly1, ly2)))
    rx2 = int(math.ceil(max(lx1, lx2)))
    ry2 = int(math.ceil(max(ly1, ly2)))

    roi_w = rx2 - rx1
    roi_h = ry2 - ry1
    if roi_w <= 0 or roi_h <= 0:
        return 0.0

    local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    local_poly = poly.copy()
    local_poly[:, 0] -= rx1
    local_poly[:, 1] -= ry1
    local_poly = np.round(local_poly).astype(np.int32)

    cv2.fillPoly(local_mask, [local_poly], 255)

    overlap_pixels = float(np.count_nonzero(local_mask))
    litter_area = float(roi_w * roi_h)
    return min(max(overlap_pixels / max(litter_area, 1.0), 0.0), 1.0)


def _resolve_litter_anchor(litter_ref):
    # litter anchor 使用 bbox 中心；若傳入點座標也可直接相容。
    if litter_ref is None:
        return None

    arr = np.asarray(litter_ref, dtype=np.float32).reshape(-1)
    if arr.size >= 4:
        x1, y1, x2, y2 = map(float, arr[:4])
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if arr.size >= 2:
        return (float(arr[0]), float(arr[1]))
    return None

def _signed_distance_to_polygon(point, polygon):
    # OpenCV pointPolygonTest：正值代表點在 polygon 內，負值代表在外側。
    if point is None or polygon is None:
        return None

    poly = np.asarray(polygon, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return None

    contour = poly.reshape((-1, 1, 2))
    return float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))


def _point_in_box(point, box, margin_px=0.0):
    # bbox 內點判斷，可加 margin 容忍偵測框抖動。
    px, py = map(float, point)
    x1, y1, x2, y2 = map(float, box[:4])
    return (
        (x1 - margin_px) <= px <= (x2 + margin_px) and
        (y1 - margin_px) <= py <= (y2 + margin_px)
    )


def _box_overlap_ratio(inner_box, outer_box):
    # inner_box 被 outer_box 覆蓋的比例；保留給 bbox-based holding fallback。
    ix1, iy1, ix2, iy2 = map(float, inner_box[:4])
    ox1, oy1, ox2, oy2 = map(float, outer_box[:4])

    x1 = max(ix1, ox1)
    y1 = max(iy1, oy1)
    x2 = min(ix2, ox2)
    y2 = min(iy2, oy2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inner_area = max(1.0, max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1))
    return (inter_w * inter_h) / inner_area


def _expand_box(box, margin_px):
    # 擴張 bbox，用於容忍 segmentation 或 bbox 邊緣小幅抖動。
    x1, y1, x2, y2 = map(float, box[:4])
    margin = float(margin_px)
    return (x1 - margin, y1 - margin, x2 + margin, y2 + margin)


def _normalized_point_in_box(point, box):
    # 將點轉成 bbox 內的 0~1 相對座標，方便判斷靠近側邊/底部。
    px, py = map(float, point)
    x1, y1, x2, y2 = map(float, box[:4])
    return (
        (px - x1) / max(x2 - x1, 1e-6),
        (py - y1) / max(y2 - y1, 1e-6),
    )


def _polygon_bounds(polygon):
    # 取得 polygon 外接框，作為 mask 內相對位置的計算基準。
    if polygon is None:
        return None

    poly = np.asarray(polygon, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return None

    return (
        float(np.min(poly[:, 0])),
        float(np.min(poly[:, 1])),
        float(np.max(poly[:, 0])),
        float(np.max(poly[:, 1])),
    )


def _normalized_point_in_polygon_bounds(point, polygon):
    # 將點轉成 polygon 外接框內的 0~1 相對座標。
    bounds = _polygon_bounds(polygon)
    if point is None or bounds is None:
        return None

    px, py = map(float, point)
    x1, y1, x2, y2 = bounds
    return (
        (px - x1) / max(x2 - x1, 1e-6),
        (py - y1) / max(y2 - y1, 1e-6),
    )


def _latest_distinct_velocity(points, min_motion_px=0.75):
    # 從歷史中心點找最近一次有效位移，避免連續幀小抖動被當成速度。
    if points is None or len(points) < 2:
        return 0.0, 0.0

    current = points[-1]
    for prev in reversed(list(points)[:-1]):
        dx = float(current[0]) - float(prev[0])
        dy = float(current[1]) - float(prev[1])
        if math.hypot(dx, dy) >= float(min_motion_px):
            return dx, dy
    return 0.0, 0.0


def _same_actor(prev_actor_id, cls_name, track_id):
    # 判斷上一個關聯 actor 是否與目前 actor 相同，兼容 tuple 與舊版純 id。
    if isinstance(prev_actor_id, tuple) and len(prev_actor_id) == 2:
        try:
            return str(prev_actor_id[0]).lower() == cls_name and int(prev_actor_id[1]) == int(track_id)
        except (TypeError, ValueError):
            return False
    try:
        return int(prev_actor_id) == int(track_id)
    except (TypeError, ValueError):
        return False


def litter_holding(litter_box, actors, prev_actor_id=None,
                   person_dist_threshold=55.0,
                   vehicle_dist_threshold=90.0,
                   overlap_ratio_threshold=0.55,
                   mask_overlap_ratio_threshold=0.35,
                   min_mask_overlap_for_vehicle_distance=0.08,
                   prev_litter_center=None,
                   prev_litter_missed=None,
                   vehicle_history=None,
                   person_mask_dilation_px=10.0,
                   vehicle_mask_dilation_px=16.0,
                   vehicle_mask_gap_threshold=100.0,
                   vehicle_relative_motion_threshold=12.0,
                   bbox_margin_px=8.0,
                   allow_distance_holding=True,
                   vehicle_release_downward_threshold=15.0,
                   vehicle_release_horizontal_threshold=12.0,
                   vehicle_release_relative_motion_threshold=18.0,
                   vehicle_release_abs_downward_threshold=10.0,
                   vehicle_release_abs_horizontal_threshold=10.0,
                   vehicle_release_abs_motion_threshold=18.0,
                   vehicle_release_min_mask_gap_px=8.0,
                   vehicle_release_max_mask_overlap=0.08,
                   vehicle_release_lower_edge_ratio=0.90,
                   vehicle_release_side_min_y_ratio=0.55,
                   vehicle_release_side_min_mask_gap_px=16.0,
                   vehicle_release_side_max_mask_overlap=0.01,
                   vehicle_release_max_anchor_missed=3,
                   vehicle_bbox_gap_threshold=48.0,
                   vehicle_side_edge_ratio=0.12,
                   vehicle_lower_edge_ratio=0.75,
                   vehicle_bottom_gap_ratio=0.95):
    # holding 主判斷：若 litter 仍在 actor mask/bbox 內或近旁，視為尚未被丟出。
    if not actors:
        return False, None

    anchor_point = _resolve_litter_anchor(litter_box)
    if anchor_point is None:
        return False, None

    lc_x, lc_y = anchor_point

    # 第一段：取得垃圾移動速度與方向；沒有歷史中心時，預設尚無位移證據。
    litter_vx, litter_vy = 0.0, 0.0
    is_litter_static = False
    
    if prev_litter_center is not None:
        litter_vx = lc_x - float(prev_litter_center[0])
        litter_vy = lc_y - float(prev_litter_center[1])
        # 垃圾靜止特徵：X 與 Y 軸位移極小
        if abs(litter_vx) < 1.0 and abs(litter_vy) < 1.0:
            is_litter_static = True
    best_actor_key = None
    best_distance = float('inf')

    # 第二段：逐一檢查所有 actor，依 person/vehicle 採用不同 holding 與 release 條件。
    for actor in actors:
        cls_name = str(actor.get('cls', '')).lower()
        if cls_name not in ACTOR_CLASSES:
            continue

        ax1, ay1, ax2, ay2 = map(float, actor['box'])
        aw = max(ax2 - ax1, 1.0)
        ah = max(ay2 - ay1, 1.0)
        try:
            track_id = int(actor['track_id'])
        except (TypeError, ValueError):
            continue

        if cls_name in VEHICLE_LIKE_CLASSES:
            # 車輛類 actor 要同時考慮車輛自身速度與 litter 相對車輛的分離方向。
            veh_vx, veh_vy = 0.0, 0.0
            is_vehicle_moving = False
            
            if vehicle_history and track_id in vehicle_history:
                history = vehicle_history[track_id]['centroids']
                if len(history) >= 2:
                    veh_vx, veh_vy = _latest_distinct_velocity(history)
                    if abs(veh_vx) > 1.5 or abs(veh_vy) > 1.5:
                        is_vehicle_moving = True

            relative_vx = litter_vx - veh_vx
            relative_vy = litter_vy - veh_vy
            relative_motion = math.hypot(relative_vx, relative_vy)
            abs_litter_motion = math.hypot(litter_vx, litter_vy)
            is_absolute_release_motion = (
                prev_litter_center is not None and
                litter_vy >= float(vehicle_release_abs_downward_threshold) and
                abs(litter_vx) >= float(vehicle_release_abs_horizontal_threshold) and
                abs_litter_motion >= float(vehicle_release_abs_motion_threshold)
            )
            is_relative_release_motion = (
                prev_litter_center is not None and
                relative_vy >= float(vehicle_release_downward_threshold) and
                abs(relative_vx) >= float(vehicle_release_horizontal_threshold) and
                relative_motion >= float(vehicle_release_relative_motion_threshold)
            )
            is_vehicle_release_motion = is_relative_release_motion or is_absolute_release_motion

            # 解綁條件 A：垃圾靜止在地上，但車子還在開，表示不是持有狀態。
            if is_litter_static and is_vehicle_moving:
                continue 

            actor_anchor = ((ax1 + ax2) / 2.0, ay2)
            dilation_px = vehicle_mask_dilation_px
            adaptive_thr = vehicle_dist_threshold * (1.5 if _same_actor(prev_actor_id, cls_name, track_id) else 1.0)

        else:
            actor_anchor = ((ax1 + ax2) / 2.0, ay2 - (ah * 0.2))
            dilation_px = person_mask_dilation_px
            adaptive_thr = person_dist_threshold * (1.5 if _same_actor(prev_actor_id, cls_name, track_id) else 1.0)

        dist = math.hypot(lc_x - actor_anchor[0], lc_y - actor_anchor[1])
        if dist < best_distance:
            best_distance = dist
            best_actor_key = (cls_name, track_id)

        mask_poly = actor.get('mask_poly')
        if mask_poly is not None:
            # 優先使用 segmentation mask：比 bbox 更能判斷 litter 是否仍貼在車體/人體上。
            mask_signed_dist = _signed_distance_to_polygon(anchor_point, mask_poly)
            mask_overlap_ratio = calculate_mask_overlap_ratio(litter_box, mask_poly)
            is_inside_actual_region = (
                mask_signed_dist is not None and
                mask_signed_dist >= 0.0
            )
            is_inside = (
                mask_signed_dist is not None and
                mask_signed_dist >= -float(dilation_px)
            )

            if cls_name in VEHICLE_LIKE_CLASSES:
                # 車輛 release gate：必須有向下、水平、相對或絕對位移，才允許從 holding 解綁。
                norm_xy = _normalized_point_in_polygon_bounds(anchor_point, mask_poly)
                norm_x, norm_y = norm_xy if norm_xy is not None else (None, None)
                is_near_side_edge = (
                    norm_x is not None and (
                        norm_x <= float(vehicle_side_edge_ratio) or
                        norm_x >= (1.0 - float(vehicle_side_edge_ratio))
                    )
                )
                # 側邊拋出不一定會落到車體最下緣；但側邊最容易受車體移動誤導，
                # 因此必須有相對車體的分離，或車體本身沒有明顯移動時才接受絕對位移。
                has_release_side_gap = (
                    not is_inside_actual_region and
                    is_near_side_edge and
                    norm_y is not None and
                    norm_y >= float(vehicle_release_side_min_y_ratio) and
                    (
                        mask_signed_dist is None or
                        mask_signed_dist <= -float(vehicle_release_side_min_mask_gap_px)
                    ) and
                    (
                        mask_overlap_ratio is None or
                        mask_overlap_ratio <= float(vehicle_release_side_max_mask_overlap)
                    )
                )
                has_release_lower_gap = (
                    not is_inside_actual_region and
                    norm_y is not None and
                    norm_y >= float(vehicle_release_lower_edge_ratio) and
                    (
                        mask_signed_dist is None or
                        mask_signed_dist <= -float(vehicle_release_min_mask_gap_px)
                    ) and
                    (
                        mask_overlap_ratio is None or
                        mask_overlap_ratio <= float(vehicle_release_max_mask_overlap)
                    )
                )
                has_fresh_litter_anchor = (
                    prev_litter_missed is None or
                    int(prev_litter_missed) <= int(vehicle_release_max_anchor_missed)
                )
                side_release_motion = (
                    has_fresh_litter_anchor and (
                        is_relative_release_motion or
                        (is_absolute_release_motion and not is_vehicle_moving)
                    )
                )
                mask_aware_release_motion = (
                    (
                        (is_relative_release_motion or is_absolute_release_motion) and
                        has_release_lower_gap
                    ) or (
                        side_release_motion and
                        has_release_side_gap
                    )
                )
                if is_inside_actual_region:
                    if norm_x is not None and (
                        norm_x <= float(vehicle_side_edge_ratio) or
                        norm_x >= (1.0 - float(vehicle_side_edge_ratio))
                    ):
                        return True, (cls_name, track_id)

                is_lower_side_gap = (
                    not is_inside_actual_region and
                    mask_signed_dist is not None and
                    mask_signed_dist >= -float(vehicle_bbox_gap_threshold) and
                    norm_y is not None and
                    norm_y >= float(vehicle_lower_edge_ratio) and
                    norm_x is not None and (
                        norm_x <= float(vehicle_side_edge_ratio) or
                        norm_x >= (1.0 - float(vehicle_side_edge_ratio))
                    )
                )
                if is_lower_side_gap and not mask_aware_release_motion:
                    return True, (cls_name, track_id)

                is_bottom_gap = (
                    not is_inside_actual_region and
                    mask_signed_dist is not None and
                    mask_signed_dist >= -float(vehicle_bbox_gap_threshold) and
                    norm_y is not None and
                    norm_y >= float(vehicle_bottom_gap_ratio)
                )
                if is_bottom_gap and not mask_aware_release_motion:
                    return True, (cls_name, track_id)

                is_mask_attached = (
                    mask_overlap_ratio is not None and
                    mask_overlap_ratio >= float(min_mask_overlap_for_vehicle_distance)
                )
                is_vehicle_attached = is_inside or is_mask_attached
                if is_vehicle_attached:
                    if mask_aware_release_motion:
                        continue
                    return True, (cls_name, track_id)

                is_near_vehicle_mask = (
                    mask_signed_dist is not None and
                    mask_signed_dist >= -float(vehicle_mask_gap_threshold)
                )
                if allow_distance_holding and is_near_vehicle_mask and not mask_aware_release_motion:
                    return True, (cls_name, track_id)

                continue

            if is_inside:
                return True, (cls_name, track_id)

            if (
                mask_overlap_ratio is not None and
                mask_overlap_ratio >= float(mask_overlap_ratio_threshold)
            ):
                return True, (cls_name, track_id)

            continue

        if cls_name in VEHICLE_LIKE_CLASSES:
            # 沒有 mask 時退回 bbox 距離與相對速度判斷。
            is_distance_holding = (
                allow_distance_holding and
                dist <= adaptive_thr and
                relative_motion <= float(vehicle_relative_motion_threshold)
            )
            if is_distance_holding:
                if is_vehicle_release_motion:
                    continue
                return True, (cls_name, track_id)
            continue

        is_distance_holding = allow_distance_holding and dist <= adaptive_thr
        if is_distance_holding:
            return True, (cls_name, track_id)

    return False, best_actor_key

def validate_trajectory(centroid_history):
    """
    驗證軌跡是否符合物理拋落特性
    """
    # confirmed 前的物理軌跡驗證：排除原地閃爍、亂跳雜訊與非向下移動。
    if len(centroid_history) < 2: # 至少需要 2 幀來確認軌跡
        return False, 0.0

    pts = np.array(centroid_history)
    
    # 1. 檢查總位移 (過濾在原地閃爍的雜訊)
    start_pt = pts[0]
    end_pt = pts[-1]
    total_displacement = np.linalg.norm(end_pt - start_pt)
    
    if total_displacement < 15.0: # 總位移不到 15 pixel 視為雜訊
        return False, 0.0

    # 2. 軌跡平滑度：起終點直線距離 / 實際軌跡總長度。
    # 平滑度 = 起終點直線距離 / 實際走過的軌跡總長度
    # 物理掉落物的平滑度極高 (接近 1.0)；而雜訊亂跳的軌跡平滑度會很低 (< 0.5)
    path_length = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    straightness = total_displacement / max(path_length, 1e-6)

    # 3. Y 軸向下趨勢 (重力原則)
    # 畫面中 Y 軸越往下數值越大
    is_falling = end_pt[1] > start_pt[1]

    # 綜合判斷：軌跡夠平滑且往下掉
    is_valid = straightness > 0.85 and is_falling
    
    return is_valid, straightness
