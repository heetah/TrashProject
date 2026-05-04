import math
from scipy.spatial import distance
from smallFunction import validate_trajectory

class GlobalLitterTracker:
    def __init__(self, distance_threshold=250):
        self.active_litters = {}
        # violators: {(cls, track_id): {'ttl': int, 'center': (x, y)}}
        self.violators = {}
        self.next_id = 0
        self.distance_threshold = distance_threshold
        self.max_missed_frames = 10
        # 允許違規者在連續幀中的中心點最大跳動，避免 ID 切換造成誤綁
        self.max_violator_jump = 80.0
        # 違規者若暫時消失可容忍的幀數，超過立即失效避免 ID 重用誤綁
        self.violator_max_missed = 5
        # 允許同類別 actor 以空間連續性重新綁定，吸收追蹤器短暫 ID 切換
        self.violator_rebind_distance = 60.0
        # confirmed 後違規標記持續幀數
        self.confirmed_violator_ttl = 60
        # 軌跡長度限制
        self.trajectory_history_len = 15
        # 尺寸一致性門檻：pending 較嚴，confirmed 較寬鬆
        self.pending_shape_change_ratio = 0.60
        self.confirmed_shape_change_ratio = 1.20
        # pending 轉 confirmed 的最低觀測幀數與位移條件
        self.min_confirm_age = 2
        self.min_confirm_abs_displacement = 18.0
        self.min_confirm_downward_displacement = 10.0

        self.person_to_vehicle_history = {}

    def update(self, detected_litters, actors, person_vehicle_map=None):
        if person_vehicle_map:
            for p_id, vehicle_key_or_id in person_vehicle_map.items():
                self.person_to_vehicle_history[int(p_id)] = self._normalize_vehicle_like_key(vehicle_key_or_id)

        for actor_key in list(self.violators.keys()):
            self.violators[actor_key]['ttl'] -= 1
            if self.violators[actor_key]['ttl'] <= 0:
                del self.violators[actor_key]
        
        new_active_litters = {}

        # 聚焦 litter bbox 的中心點
        for litter_box in detected_litters:
            lx1, ly1, lx2, ly2, _ = litter_box
            centroid = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)

            curr_w = max(lx2 - lx1, 1e-6)
            curr_h = max(ly2 - ly1, 1e-6)

            best_id = None
            min_dist = float("inf")

            # 找到 active litters 的相關資料
            for l_id, l_data in self.active_litters.items():
                prev_box = l_data['bbox']
                prev_centroid = (
                    (prev_box[0] + prev_box[2]) / 2, 
                    (prev_box[1] + prev_box[3]) / 2
                )

                dist = distance.euclidean(centroid, prev_centroid)
                ref_w, ref_h = l_data.get('ref_shape', l_data.get('init_shape', (curr_w, curr_h)))
                w_diff_ratio = abs(curr_w - ref_w) / max(ref_w, 1e-6)
                h_diff_ratio = abs(curr_h - ref_h) / max(ref_h, 1e-6)

                prev_state = l_data.get('state', 'pending')
                shape_thr = (
                    self.confirmed_shape_change_ratio
                    if prev_state == 'confirmed'
                    else self.pending_shape_change_ratio
                )
                is_shape_consistent = (w_diff_ratio <= shape_thr) and (h_diff_ratio <= shape_thr)

                # confirmed 軌跡優先保持連續，避免尺寸波動導致 ID 斷裂
                allow_confirmed_dist_only = (
                    prev_state == 'confirmed' and dist < (self.distance_threshold * 0.7)
                )

                if dist < self.distance_threshold and (is_shape_consistent or allow_confirmed_dist_only) and dist < min_dist:
                    min_dist = dist
                    best_id = l_id

            if best_id is not None:
                l_data = self.active_litters[best_id]
                l_data['history'].append(centroid)
                if len(l_data['history']) > self.trajectory_history_len:
                    l_data['history'].pop(0)

                age = l_data.get('age', 1) + 1
                state = l_data.get('state', 'pending')
                
                # 繼承剛出生時記錄的肇事者
                thrower_key = l_data.get('thrower_key')
                thrower_center = l_data.get('thrower_center')

                # ===== 時空軌跡 =====
                if state == 'pending':
                    # 取得初始長寬
                    init_w, init_h = l_data.get('init_shape', (curr_w, curr_h))
                    
                    # 1. 判定移動距離與向下位移是否達標
                    start_centroid = l_data['history'][0]
                    moved_dist = distance.euclidean(start_centroid, centroid)
                    is_moved_enough = moved_dist > max(
                        2.5 * max(init_w, init_h),
                        self.min_confirm_abs_displacement,
                    )
                    downward_disp = float(centroid[1] - start_centroid[1])
                    is_downward_enough = downward_disp >= self.min_confirm_downward_displacement
                    
                    # 2. 使用軌跡物理特徵檢查（至少要有足夠歷史幀）
                    is_physics_valid = False
                    if age >= self.min_confirm_age:
                        is_physics_valid, _ = validate_trajectory(l_data['history'])

                    # 3. 確認條件：
                    # - 軌跡符合物理特性且有向下位移
                    # - 或 有明顯位移 + 向下位移 + 具 thrower 關聯（避免舊垃圾堆誤判）
                    can_confirm_by_trajectory = (
                        age >= self.min_confirm_age and
                        is_physics_valid and
                        is_downward_enough
                    )
                    can_confirm_by_motion = (
                        age >= (self.min_confirm_age + 1) and
                        is_moved_enough and
                        is_downward_enough and
                        thrower_key is not None
                    )

                    if can_confirm_by_trajectory or can_confirm_by_motion:
                        state = 'confirmed' # 確認為垃圾！
                        
                        if thrower_key is not None:
                            current_actor_center = thrower_center # 預設為舊位置
                            
                            # 去目前的 actors 裡面找他現在開到哪了
                            for actor in actors:
                                if self._actor_key(actor) == thrower_key:
                                    current_actor_center = self._actor_center(actor)
                                    break
                                    
                            self._mark_violator(thrower_key, current_actor_center, ttl=self.confirmed_violator_ttl)

                            cls_name, track_id = thrower_key
                            if cls_name == 'person' and track_id in self.person_to_vehicle_history:
                                veh_key = self.person_to_vehicle_history[track_id]
                                
                                # 去當前畫面找車輛中心點
                                veh_center = current_actor_center
                                for actor in actors:
                                    if self._actor_key(actor) == veh_key:
                                        veh_center = self._actor_center(actor)
                                        break
                                        
                                # 同時將該車輛標記為違規！(讓畫面畫紅框並抓取車牌)
                                self._mark_violator(veh_key, veh_center, ttl=self.confirmed_violator_ttl)
                
                new_active_litters[best_id] = {
                    'bbox': litter_box, 
                    'history': l_data['history'],
                    'missed': 0,
                    'thrower_key': thrower_key,
                    'thrower_center': thrower_center,
                    'age': age,
                    'state': state,
                    'init_shape': l_data['init_shape'],
                    'ref_shape': (
                        0.7 * float(l_data.get('ref_shape', l_data['init_shape'])[0]) + 0.3 * float(curr_w),
                        0.7 * float(l_data.get('ref_shape', l_data['init_shape'])[1]) + 0.3 * float(curr_h),
                    ),
                }
                del self.active_litters[best_id]
            else:
                # 在垃圾剛出現的「第 0 幀」先用幾何位置估計最可能的丟棄者。
                thrower_key, thrower_center = self._find_thrower_at_birth(centroid, actors)

                litter_id = self.next_id
                new_active_litters[litter_id] = {
                    'bbox': litter_box,
                    'history': [centroid], 
                    'missed': 0,
                    'thrower_key': thrower_key,        # 紀錄嫌疑犯
                    'thrower_center': thrower_center,
                    'age': 1, 
                    'state': 'pending',
                    'init_shape': (curr_w, curr_h),
                    'ref_shape': (curr_w, curr_h),
                }

                self.next_id += 1
        
        for l_id, l_data in self.active_litters.items():
            l_data['missed'] += 1
            if l_data['missed'] < self.max_missed_frames:
                new_active_litters[l_id] = l_data
        
        self.active_litters = new_active_litters

        # 僅回傳本幀中「位置連續」的違規者，並允許短暫 miss 與同類別近距離 rebind
        active_violator_keys = set()
        actor_center_map = {}
        for actor in actors:
            actor_key = self._actor_key(actor)
            actor_center_map[actor_key] = self._actor_center(actor)

        occupied_actor_keys = set()
        for violator_key in list(self.violators.keys()):
            if violator_key not in self.violators:
                continue

            v_data = self.violators[violator_key]
            saved_center = v_data.get('center')

            # 1) 先嘗試同 key 直接延續
            if violator_key in actor_center_map:
                actor_center = actor_center_map[violator_key]
                jump_dist = distance.euclidean(actor_center, saved_center)

                if jump_dist <= self.max_violator_jump:
                    active_violator_keys.add(violator_key)
                    occupied_actor_keys.add(violator_key)
                    v_data['center'] = actor_center
                    v_data['missed'] = 0
                    continue

                # 同 key 但位置跳太遠，視為可能 ID 重用，改走 rebind

            # 2) 嘗試同類別近距離 rebind，吸收追蹤器 ID 變更
            rebound_key, rebound_center = self._find_rebind_actor(
                violator_key,
                saved_center,
                actor_center_map,
                occupied_actor_keys,
            )
            if rebound_key is not None:
                rebound_data = {
                    'ttl': v_data.get('ttl', self.confirmed_violator_ttl),
                    'center': rebound_center,
                    'missed': 0,
                }
                self.violators[rebound_key] = rebound_data
                if rebound_key != violator_key:
                    del self.violators[violator_key]

                active_violator_keys.add(rebound_key)
                occupied_actor_keys.add(rebound_key)
                continue

            # 3) 本幀找不到可延續對象：累積 miss，超過門檻才釋放
            v_data['missed'] = v_data.get('missed', 0) + 1
            if v_data['missed'] > self.violator_max_missed:
                del self.violators[violator_key]

        return self.active_litters, active_violator_keys

    def register_action_violators(self, person_action_map, actors, person_vehicle_map=None, ttl=None):
        if not person_action_map:
            return set()

        if person_vehicle_map:
            for p_id, vehicle_key_or_id in person_vehicle_map.items():
                self.person_to_vehicle_history[int(p_id)] = self._normalize_vehicle_like_key(vehicle_key_or_id)

        ttl = int(ttl or self.confirmed_violator_ttl)
        actor_center_map = {}
        for actor in actors:
            actor_center_map[self._actor_key(actor)] = self._actor_center(actor)

        marked_violators = set()
        for raw_person_id, action_info in person_action_map.items():
            if not isinstance(action_info, dict) or not action_info.get('alert', False):
                continue

            try:
                person_id = int(raw_person_id)
            except (TypeError, ValueError):
                continue

            person_key = ('person', person_id)
            person_center = actor_center_map.get(person_key)
            if person_center is None:
                continue

            self._mark_violator(person_key, person_center, ttl=ttl)
            marked_violators.add(person_key)

            vehicle_key = self.person_to_vehicle_history.get(person_id)
            if vehicle_key is None:
                vehicle_key = self._find_vehicle_for_person(person_id, actors)
                if vehicle_key is not None:
                    self.person_to_vehicle_history[person_id] = vehicle_key

            if vehicle_key is not None:
                vehicle_center = actor_center_map.get(vehicle_key, person_center)
                self._mark_violator(vehicle_key, vehicle_center, ttl=ttl)
                marked_violators.add(vehicle_key)

        return marked_violators

    def _actor_key(self, actor):
        return (actor['cls'], int(actor['track_id']))

    def _normalize_vehicle_like_key(self, actor_key_or_id):
        if isinstance(actor_key_or_id, tuple) and len(actor_key_or_id) == 2:
            cls_name = str(actor_key_or_id[0]).lower()
            if cls_name not in ('vehicle', 'scooter'):
                cls_name = 'vehicle'
            return (cls_name, int(actor_key_or_id[1]))

        # Backward-compatible path for old callers that passed only a vehicle id.
        return ('vehicle', int(actor_key_or_id))

    def _actor_center(self, actor):
        ax1, ay1, ax2, ay2 = actor['box']
        return ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)

    def _mark_violator(self, actor_key, center, ttl):
        if actor_key is None:
            return

        prev = self.violators.get(actor_key)
        if prev is None:
            self.violators[actor_key] = {'ttl': ttl, 'center': center, 'missed': 0}
            return

        prev['ttl'] = max(prev['ttl'], ttl)
        if center is not None:
            prev['center'] = center
        prev['missed'] = 0

    def _find_vehicle_for_person(self, person_id, actors):
        person_box = None
        for actor in actors:
            actor_key = self._actor_key(actor)
            if actor_key == ('person', int(person_id)):
                person_box = actor['box']
                break

        if person_box is None:
            return None

        best_key = None
        best_iou = 0.0
        for actor in actors:
            actor_key = self._actor_key(actor)
            if actor_key[0] not in ('vehicle', 'scooter'):
                continue

            iou = self._box_iou(person_box, actor['box'])
            if iou > best_iou:
                best_iou = iou
                best_key = actor_key

        return best_key if best_iou >= 0.05 else None

    @staticmethod
    def _box_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = map(float, box_a[:4])
        bx1, by1, bx2, by2 = map(float, box_b[:4])

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0

        return inter / union

    def _find_rebind_actor(self, violator_key, saved_center, actor_center_map, occupied_actor_keys):
        if saved_center is None:
            return None, None

        target_cls = violator_key[0]
        best_key = None
        best_center = None
        best_dist = float('inf')

        for actor_key, actor_center in actor_center_map.items():
            if actor_key in occupied_actor_keys:
                continue
            if actor_key[0] != target_cls:
                continue

            d = distance.euclidean(saved_center, actor_center)
            if d <= self.violator_rebind_distance and d < best_dist:
                best_dist = d
                best_key = actor_key
                best_center = actor_center

        return best_key, best_center

    def _find_thrower_at_birth(self, start_centroid, actors):
        """
        在垃圾剛出現的瞬間 (Frame 0)，尋找最可能的丟棄者。
        """
        best_actor_key = None
        best_center = None
        min_dist = float('inf')

        # === 新增：備案機制變數，用來記錄全場絕對距離最近的物件 ===
        fallback_actor_key = None
        fallback_center = None
        fallback_min_dist = float('inf')

        for actor in actors:
            ax1, ay1, ax2, ay2 = map(float, actor['box'])
            track_id = int(actor['track_id'])
            cls_name = actor['cls']
            
            actor_center = ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)
            
            # --- 新增：記錄每個物件的絕對距離，供備用機制使用 ---
            abs_dist = math.hypot(start_centroid[0] - actor_center[0], start_centroid[1] - actor_center[1])
            if abs_dist < fallback_min_dist:
                fallback_min_dist = abs_dist
                fallback_actor_key = (cls_name, track_id)
                fallback_center = actor_center
            # --------------------------------------------------

            dist = float('inf')
            threshold = 0.0

            if cls_name == 'person':
                # 無姿態模型時，直接以行人中心點做近距離關聯。
                dist = math.hypot(start_centroid[0] - actor_center[0], start_centroid[1] - actor_center[1])
                threshold = 90.0

            elif cls_name in ('vehicle', 'scooter'):
                # 車輛：通常從車窗丟出，使用 BBox 下半部中心來計算
                vehicle_bottom = ((ax1 + ax2) / 2.0, ay1 + (ay2 - ay1) * 0.7)
                dist = math.hypot(start_centroid[0] - vehicle_bottom[0], start_centroid[1] - vehicle_bottom[1])
                
                # 使用動態門檻，防止高畫質大車造成的誤判
                vehicle_width = ax2 - ax1
                threshold = max(180.0, vehicle_width * 0.6) # 最少 180 像素，或車寬的 60%

            # 綜合評判 (原本的嚴格門檻邏輯)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                best_actor_key = (cls_name, track_id)
                best_center = actor_center

        # === 新增：如果原本的嚴格門檻抓不到人，就退回使用「最靠近的物件」 ===
        if (
            best_actor_key is None and
            fallback_actor_key is not None
        ):
            best_actor_key = fallback_actor_key
            best_center = fallback_center

        return best_actor_key, best_center
