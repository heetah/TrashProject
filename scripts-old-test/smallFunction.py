import math
import cv2
import numpy as np

ACTOR_CLASSES = ('person', 'vehicle', 'scooter')
VEHICLE_LIKE_CLASSES = ('vehicle', 'scooter')

def check_motion(fg_mask, coords, threshold):
    x1, y1, x2, y2 = coords
    h, w = fg_mask.shape[:2]
    
    # 嚴格限制邊界，避免負數索引與 Out of Bounds
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

# 計算重疊區域比例
def calculate_iou_matrix(boxes1, boxes2):
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
    if litter_ref is None:
        return None

    arr = np.asarray(litter_ref, dtype=np.float32).reshape(-1)
    if arr.size >= 4:
        x1, y1, x2, y2 = map(float, arr[:4])
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    if arr.size >= 2:
        return (float(arr[0]), float(arr[1]))
    return None


def _point_in_dilated_polygon(point, polygon, dilation_px):
    if point is None or polygon is None:
        return False

    poly = np.asarray(polygon, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return False

    contour = poly.reshape((-1, 1, 2))
    dist = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)
    return dist >= -float(dilation_px)


def _signed_distance_to_polygon(point, polygon):
    if point is None or polygon is None:
        return None

    poly = np.asarray(polygon, dtype=np.float32)
    if poly.ndim != 2 or poly.shape[0] < 3:
        return None

    contour = poly.reshape((-1, 1, 2))
    return float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))


def _point_in_box(point, box, margin_px=0.0):
    px, py = map(float, point)
    x1, y1, x2, y2 = map(float, box[:4])
    return (
        (x1 - margin_px) <= px <= (x2 + margin_px) and
        (y1 - margin_px) <= py <= (y2 + margin_px)
    )


def _same_actor(prev_actor_id, cls_name, track_id):
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
                   vehicle_history=None,
                   person_mask_dilation_px=10.0,
                   vehicle_mask_dilation_px=16.0,
                   vehicle_mask_gap_threshold=100.0,
                   vehicle_relative_motion_threshold=12.0,
                   bbox_margin_px=8.0):
    if not actors:
        return False, None

    anchor_point = _resolve_litter_anchor(litter_box)
    if anchor_point is None:
        return False, None

    lc_x, lc_y = anchor_point

    # 1. 取得垃圾的移動速度與方向 (若沒有歷史中心，預設無移動)
    litter_vx, litter_vy = 0.0, 0.0
    is_litter_falling = False
    is_litter_static = False
    
    if prev_litter_center is not None:
        litter_vx = lc_x - float(prev_litter_center[0])
        litter_vy = lc_y - float(prev_litter_center[1])
        # 垃圾掉落特徵：Y 軸正向位移明顯
        if litter_vy > 2.0: 
            is_litter_falling = True
        # 垃圾靜止特徵：X 與 Y 軸位移極小
        if abs(litter_vx) < 1.0 and abs(litter_vy) < 1.0:
            is_litter_static = True
    has_significant_litter_motion = math.hypot(litter_vx, litter_vy) >= 4.0 or litter_vy >= 2.0

    best_actor_key = None
    best_distance = float('inf')

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
            veh_vx, veh_vy = 0.0, 0.0
            is_vehicle_moving = False
            
            if vehicle_history and track_id in vehicle_history:
                history = vehicle_history[track_id]['centroids']
                if len(history) >= 2:
                    veh_vx = history[-1][0] - history[-2][0]
                    veh_vy = history[-1][1] - history[-2][1]
                    if abs(veh_vx) > 1.5 or abs(veh_vy) > 1.5:
                        is_vehicle_moving = True

            # 解綁條件 A：垃圾靜止在地上，但車子還在開 (絕對不是 Holding)
            if is_litter_static and is_vehicle_moving:
                continue 

            # 解綁條件 B：垃圾正在往下掉，但車子在橫向移動 (分離瞬間)
            # 這裡簡單判斷：若垃圾 Y 軸位移大，但車輛主要是 X 軸位移
            if is_litter_falling and abs(veh_vx) > abs(veh_vy):
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
            mask_signed_dist = _signed_distance_to_polygon(anchor_point, mask_poly)
            is_inside_actual_region = (
                mask_signed_dist is not None and
                mask_signed_dist >= 0.0
            )
            is_inside = (
                mask_signed_dist is not None and
                mask_signed_dist >= -float(dilation_px)
            )

            # Vehicle-like objects often have large bboxes. Once the litter has moved
            # past the actual segmentation boundary, do not keep it attached by dilation alone.
            if (
                is_inside and
                not is_inside_actual_region and
                cls_name in VEHICLE_LIKE_CLASSES and
                has_significant_litter_motion
            ):
                continue

            if is_inside:
                return True, (cls_name, track_id)

            if cls_name in VEHICLE_LIKE_CLASSES and _point_in_box(anchor_point, actor['box'], bbox_margin_px):
                relative_motion = math.hypot(litter_vx - veh_vx, litter_vy - veh_vy)
                if (
                    (mask_signed_dist is not None and mask_signed_dist >= -float(vehicle_mask_gap_threshold)) or
                    relative_motion <= float(vehicle_relative_motion_threshold)
                ):
                    return True, (cls_name, track_id)

            continue

        is_inside_actual_region = _point_in_box(anchor_point, actor['box'], 0.0)
        is_inside = is_inside_actual_region or _point_in_box(anchor_point, actor['box'], bbox_margin_px)
        is_distance_holding = dist <= adaptive_thr
        if is_inside or is_distance_holding:
            return True, (cls_name, track_id)

    return False, best_actor_key

def validate_trajectory(centroid_history):
    
    """
    驗證軌跡是否符合物理拋落特性
    """
    if len(centroid_history) < 2: # 至少需要 2 幀來確認軌跡
        return False, 0.0

    pts = np.array(centroid_history)
    
    # 1. 檢查總位移 (過濾在原地閃爍的雜訊)
    start_pt = pts[0]
    end_pt = pts[-1]
    total_displacement = np.linalg.norm(end_pt - start_pt)
    
    if total_displacement < 15.0: # 總位移不到 15 pixel 視為雜訊
        return False, 0.0

    # 2. 軌跡平滑度 (Straightness Index)
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
