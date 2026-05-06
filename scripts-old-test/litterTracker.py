# -*- coding: utf-8 -*-
# 全域垃圾追蹤器：把 RTDETR 候選 litter 串成軌跡，判斷 pending/confirmed，並反推丟擲者。
import math
import queue
import threading
import numpy as np
from collections import deque
from scipy.spatial import distance
from smallFunction import validate_trajectory

class GlobalLitterTracker:
    def __init__(self, distance_threshold=250):
        # active_litters 保存仍在追蹤中的垃圾；violators 保存已確認違規者與顯示 TTL。
        self.active_litters = {}
        # violators 結構：{(類別, track_id): {'ttl': int, 'center': (x, y)}}。
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
        self.min_confirm_horizontal_displacement = 8.0

        self.person_to_vehicle_history = {}
        # 反追蹤 thrower 時，用車輛/機車 bbox 底邊點估計地面 homography，
        # 把 2D 影像點轉成 pseudo-ground 座標後再計算距離。
        self.homography_min_depth = 25.0
        self.thrower_previous_bonus = 0.85
        self.thrower_fallback_score_limit = 1.35
        # 長距離丟出後，垃圾當前點會離 thrower 很遠；只在軌跡起點貼近 actor 時啟用較寬的反追蹤 fallback。
        self.thrower_release_origin_score_limit = 4.0
        # backward resolver：confirmed 後回推到 litter 出生幀，用 actor ring buffer 找 thrower。
        self.backward_actor_history_len = 120
        self.backward_pre_birth_frames = 24
        self.backward_post_birth_frames = 18
        self.backward_score_limit = 2.30
        self.backward_release_score_limit = 3.60
        self.backward_plate_roi_per_result = 3
        self.actor_frame_history = deque(maxlen=self.backward_actor_history_len)
        self.backward_plate_roi_items = []
        self._actor_history_lock = threading.Lock()
        self._backward_plate_lock = threading.Lock()
        self._backward_tasks = queue.Queue(maxsize=64)
        self._backward_results = queue.Queue()
        self._backward_stop = object()
        self._backward_thread = threading.Thread(
            target=self._backward_worker,
            name="litter-backward-resolver",
            daemon=True,
        )
        self._backward_thread.start()
        self._fallback_frame_index = 0

    def update(self, detected_litters, actors, person_vehicle_map=None, frame_index=None,
               frame=None, vehicle_history=None):
        # 主更新流程：接收本幀通過前處理的 litter，更新軌跡與違規者集合。
        if frame_index is None:
            frame_index = self._fallback_frame_index
            self._fallback_frame_index += 1
        frame_index = int(frame_index)

        self._record_actor_frame(actors, frame_index, frame=frame)
        self._drain_backward_results(vehicle_history=vehicle_history)

        if person_vehicle_map:
            for p_id, vehicle_key_or_id in person_vehicle_map.items():
                self.person_to_vehicle_history[int(p_id)] = self._normalize_vehicle_like_key(vehicle_key_or_id)

        for actor_key in list(self.violators.keys()):
            v_data = self.violators[actor_key]
            if (
                v_data.get('until_plate_found', False) and
                actor_key[0] in ('vehicle', 'scooter') and
                vehicle_history is not None
            ):
                plate_entry = vehicle_history.get(actor_key[1], {})
                if plate_entry.get('license_plate') is None:
                    v_data['ttl'] = max(v_data.get('ttl', 0), self.confirmed_violator_ttl)
                    continue
                v_data['until_plate_found'] = False

            self.violators[actor_key]['ttl'] -= 1
            if self.violators[actor_key]['ttl'] <= 0:
                del self.violators[actor_key]
        
        new_active_litters = {}

        # 第一段：把每個 detected litter 與既有 active litter 做距離/尺寸配對。
        for litter_box in detected_litters:
            lx1, ly1, lx2, ly2, _ = litter_box
            centroid = ((lx1 + lx2) / 2, (ly1 + ly2) / 2)

            curr_w = max(lx2 - lx1, 1e-6)
            curr_h = max(ly2 - ly1, 1e-6)

            best_id = None
            min_dist = float("inf")

            # 找到目前仍在追蹤的 litter 相關資料。
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

                # confirmed 軌跡優先保持連續，避免尺寸波動導致 ID 斷裂。
                allow_confirmed_dist_only = (
                    prev_state == 'confirmed' and dist < (self.distance_threshold * 0.7)
                )

                if dist < self.distance_threshold and (is_shape_consistent or allow_confirmed_dist_only) and dist < min_dist:
                    min_dist = dist
                    best_id = l_id

            if best_id is not None:
                # 第二段：延續既有 litter，更新 history、age、shape reference。
                l_data = self.active_litters[best_id]
                l_data['history'].append(centroid)
                if len(l_data['history']) > self.trajectory_history_len:
                    l_data['history'].pop(0)

                age = l_data.get('age', 1) + 1
                state = l_data.get('state', 'pending')
                
                # 繼承剛出生時記錄的肇事者，並在 pending 階段依 homography 座標重新評分。
                thrower_key = l_data.get('thrower_key')
                thrower_center = l_data.get('thrower_center')

                # ===== 時空軌跡 =====
                if state == 'pending':
                    recalculated_key, recalculated_center = self._find_thrower_for_litter(
                        litter_box,
                        actors,
                        history=l_data['history'],
                        prev_thrower_key=thrower_key,
                    )
                    if recalculated_key is not None:
                        thrower_key = recalculated_key
                        thrower_center = recalculated_center

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
                    horizontal_disp = abs(float(centroid[0] - start_centroid[0]))
                    is_horizontal_enough = horizontal_disp >= self.min_confirm_horizontal_displacement
                    
                    # 2. 使用軌跡物理特徵檢查（至少要有足夠歷史幀）
                    is_physics_valid = False
                    if age >= self.min_confirm_age:
                        is_physics_valid, _ = validate_trajectory(l_data['history'])

                    # 3. confirmed 條件：
                    # - 軌跡符合物理特性且有向下位移
                    # - 或 有明顯位移 + 向下位移 + 具 thrower 關聯（避免舊垃圾堆誤判）
                    can_confirm_by_trajectory = (
                        age >= self.min_confirm_age and
                        is_physics_valid and
                        is_downward_enough and
                        is_horizontal_enough and
                        (age >= 3 or thrower_key is not None)
                    )
                    can_confirm_by_motion = (
                        age >= self.min_confirm_age and
                        is_moved_enough and
                        is_downward_enough and
                        is_horizontal_enough and
                        thrower_key is not None
                    )

                    if can_confirm_by_trajectory or can_confirm_by_motion:
                        state = 'confirmed' # 確認為垃圾！
                        if not l_data.get('backward_submitted', False):
                            self._submit_backward_resolution(
                                litter_id=best_id,
                                litter_data=l_data,
                                current_bbox=litter_box,
                                current_centroid=centroid,
                                confirm_frame=frame_index,
                                prev_thrower_key=thrower_key,
                            )
                        
                        if thrower_key is not None:
                            # confirmed 後標記 thrower；若該人綁定車輛，也同步標記車輛。
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
                    'birth_frame': l_data.get('birth_frame', frame_index),
                    'birth_centroid': l_data.get('birth_centroid', l_data['history'][0]),
                    'birth_bbox': l_data.get('birth_bbox', l_data.get('bbox')),
                    'history_frames': (l_data.get('history_frames', []) + [frame_index])[-self.trajectory_history_len:],
                    'backward_submitted': l_data.get('backward_submitted', False) or state == 'confirmed',
                    'backward_result': l_data.get('backward_result'),
                    'ref_shape': (
                        0.7 * float(l_data.get('ref_shape', l_data['init_shape'])[0]) + 0.3 * float(curr_w),
                        0.7 * float(l_data.get('ref_shape', l_data['init_shape'])[1]) + 0.3 * float(curr_h),
                    ),
                }
                del self.active_litters[best_id]
            else:
                # 第三段：新 litter 建立 pending 狀態，出生幀先估計最可能丟棄者。
                thrower_key, thrower_center = self._find_thrower_for_litter(
                    litter_box,
                    actors,
                    history=[centroid],
                )

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
                    'birth_frame': frame_index,
                    'birth_centroid': centroid,
                    'birth_bbox': litter_box,
                    'history_frames': [frame_index],
                    'backward_submitted': False,
                    'backward_result': None,
                }

                self.next_id += 1
        # 第四段：處理本幀沒被配對到的舊 litter；短暫消失可保留，超過門檻移除。
        for l_id, l_data in self.active_litters.items():
            l_data['missed'] += 1
            if l_data['missed'] < self.max_missed_frames:
                new_active_litters[l_id] = l_data
        
        self.active_litters = new_active_litters
        self._drain_backward_results()

        # 第五段：僅回傳本幀中位置連續的違規者；允許短暫 miss 與同類別近距離 rebind。
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
                    'until_plate_found': bool(v_data.get('until_plate_found', False)),
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

    def close(self, timeout=2.0):
        # 結束 backward worker；daemon 可兜底，但正常釋放可避免測試殘留 thread。
        try:
            self._backward_tasks.put(self._backward_stop, timeout=timeout)
        except queue.Full:
            pass
        if self._backward_thread.is_alive():
            self._backward_thread.join(timeout=timeout)
        self._drain_backward_results()

    def consume_backward_plate_roi_items(self):
        # detect.py 取走 backward worker 找到的歷史車輛 ROI，交給車牌 OCR 背景任務。
        with self._backward_plate_lock:
            items = list(self.backward_plate_roi_items)
            self.backward_plate_roi_items.clear()
        return items

    def restore_backward_plate_roi_items(self, items):
        # OCR worker 忙碌時不可丟掉 backward 歷史 ROI；留到下一幀再派工。
        if not items:
            return
        with self._backward_plate_lock:
            self.backward_plate_roi_items = list(items) + list(self.backward_plate_roi_items)

    def get_violator_info(self, actor_key):
        # detect.py 讀取 backward resolver 附加狀態，例如車牌遮擋時的無限警示。
        return dict(self.violators.get(actor_key, {}))

    def _record_actor_frame(self, actors, frame_index, frame=None):
        # 每幀保留 actor 快照。confirmed 延遲出現時，backward worker 可回看出生幀附近。
        actor_snapshots = []
        frame_h = frame.shape[0] if frame is not None else 0
        frame_w = frame.shape[1] if frame is not None else 0

        for actor in actors or []:
            try:
                cls_name = str(actor.get('cls', '')).lower()
                if cls_name not in ('person', 'vehicle', 'scooter'):
                    continue
                track_id = int(actor['track_id'])
                box = tuple(map(float, actor['box'][:4]))
            except (KeyError, TypeError, ValueError):
                continue

            snapshot = {
                'cls': cls_name,
                'track_id': track_id,
                'box': box,
                'center': self._box_center(box),
            }

            if frame is not None and cls_name in ('vehicle', 'scooter'):
                roi = self._crop_actor_roi(frame, box, frame_w, frame_h)
                if roi is not None:
                    snapshot['plate_roi'] = roi

            actor_snapshots.append(snapshot)

        with self._actor_history_lock:
            self.actor_frame_history.append({
                'frame_index': int(frame_index),
                'actors': actor_snapshots,
            })

    def _submit_backward_resolution(self, litter_id, litter_data, current_bbox,
                                    current_centroid, confirm_frame, prev_thrower_key=None):
        # task 用 immutable snapshot，worker 不碰 main thread 追蹤狀態。
        birth_frame = int(litter_data.get('birth_frame', confirm_frame))
        birth_centroid = tuple(litter_data.get('birth_centroid', litter_data['history'][0]))
        birth_bbox = litter_data.get('birth_bbox', litter_data.get('bbox', current_bbox))
        history = [tuple(p) for p in litter_data.get('history', [])]
        if not history or history[-1] != tuple(current_centroid):
            history.append(tuple(current_centroid))

        with self._actor_history_lock:
            actor_frames = [
                {
                    'frame_index': item['frame_index'],
                    'actors': [dict(actor) for actor in item.get('actors', [])],
                }
                for item in self.actor_frame_history
                if (
                    birth_frame - self.backward_pre_birth_frames
                    <= int(item.get('frame_index', -1))
                    <= int(confirm_frame) + self.backward_post_birth_frames
                )
            ]

        if not actor_frames:
            return False

        task = {
            'litter_id': int(litter_id),
            'birth_frame': birth_frame,
            'confirm_frame': int(confirm_frame),
            'birth_centroid': birth_centroid,
            'birth_bbox': birth_bbox,
            'current_bbox': current_bbox,
            'current_centroid': tuple(current_centroid),
            'history': history,
            'prev_thrower_key': prev_thrower_key,
            'actor_frames': actor_frames,
        }

        try:
            self._backward_tasks.put_nowait(task)
            return True
        except queue.Full:
            return False

    def _backward_worker(self):
        # 第三條 worker thread：只做 CPU 幾何評分，不阻塞主推論 thread。
        while True:
            task = self._backward_tasks.get()
            try:
                if task is self._backward_stop:
                    return
                result = self._resolve_backward_task(task)
                if result is not None:
                    self._backward_results.put(result)
            except Exception as exc:
                self._backward_results.put({
                    'litter_id': task.get('litter_id') if isinstance(task, dict) else None,
                    'error': str(exc),
                })
            finally:
                try:
                    self._backward_tasks.task_done()
                except ValueError:
                    pass

    def _resolve_backward_task(self, task):
        birth_ref = task.get('birth_bbox')
        if birth_ref is None:
            birth_ref = task.get('birth_centroid')
        birth_anchor = self._litter_ground_anchor(birth_ref)
        if birth_anchor is None:
            return None

        history = task.get('history') or []
        prev_thrower_key = task.get('prev_thrower_key')
        birth_frame = int(task.get('birth_frame', 0))
        actor_candidates = {}

        for frame_snapshot in task.get('actor_frames', []):
            frame_index = int(frame_snapshot.get('frame_index', birth_frame))
            actors = []
            for actor_snapshot in frame_snapshot.get('actors', []):
                actor = self._snapshot_to_actor(actor_snapshot)
                if actor is not None:
                    actors.append(actor)
            if not actors:
                continue

            homography = self._estimate_ground_homography(actors, birth_anchor)
            birth_world = self._project_point(birth_anchor, homography)
            start_world = self._project_point(history[0], homography) if history else None
            end_world = self._project_point(history[-1], homography) if history else None

            for actor_snapshot in frame_snapshot.get('actors', []):
                actor = self._snapshot_to_actor(actor_snapshot)
                if actor is None:
                    continue

                score, release_like = self._score_backward_actor(
                    actor=actor,
                    birth_anchor=birth_anchor,
                    birth_world=birth_world,
                    start_world=start_world,
                    end_world=end_world,
                    homography=homography,
                    history=history,
                    frame_index=frame_index,
                    birth_frame=birth_frame,
                    prev_thrower_key=prev_thrower_key,
                )
                if score is None:
                    continue
                if score > self.backward_score_limit and not (
                    release_like and score <= self.backward_release_score_limit
                ):
                    continue

                actor_key = self._actor_key(actor)
                candidate = actor_candidates.setdefault(actor_key, {
                    'best_score': float('inf'),
                    'evidence_count': 0,
                    'best_center': None,
                    'best_frame': None,
                    'best_frame_actors': None,
                })
                candidate['evidence_count'] += 1
                if score < candidate['best_score']:
                    candidate['best_score'] = score
                    candidate['best_center'] = self._actor_center(actor)
                    candidate['best_frame'] = frame_index
                    candidate['best_frame_actors'] = frame_snapshot.get('actors', [])

        if not actor_candidates:
            return None

        best_key = None
        best_data = None
        best_final_score = float('inf')
        for actor_key, data in actor_candidates.items():
            continuity_bonus = 1.0 - min(int(data.get('evidence_count', 1)), 6) * 0.035
            class_bonus = 0.92 if actor_key[0] in ('vehicle', 'scooter') else 1.0
            final_score = float(data['best_score']) * continuity_bonus * class_bonus
            if final_score < best_final_score:
                best_key = actor_key
                best_data = data
                best_final_score = final_score

        if best_key is None or best_data is None:
            return None

        mark_items = [{'actor_key': best_key, 'center': best_data.get('best_center')}]
        linked_vehicle_key = self._linked_vehicle_key(best_key, best_data.get('best_frame_actors') or [])
        if linked_vehicle_key is not None and linked_vehicle_key != best_key:
            linked_center = self._snapshot_center_for_key(linked_vehicle_key, best_data.get('best_frame_actors') or [])
            mark_items.append({'actor_key': linked_vehicle_key, 'center': linked_center or best_data.get('best_center')})

        plate_key = linked_vehicle_key if linked_vehicle_key is not None else (
            best_key if best_key[0] in ('vehicle', 'scooter') else None
        )

        plate_roi_items = self._plate_roi_items_for_key(
            plate_key,
            task.get('actor_frames', []),
            birth_frame,
        ) if plate_key is not None else []

        return {
            'litter_id': int(task.get('litter_id')),
            'actor_key': best_key,
            'score': best_final_score,
            'birth_frame': birth_frame,
            'confirm_frame': int(task.get('confirm_frame', birth_frame)),
            'mark_items': mark_items,
            'plate_key': plate_key,
            'plate_roi_items': plate_roi_items,
            'plate_blocked_since_litter': plate_key is not None and not plate_roi_items,
        }

    def _score_backward_actor(self, actor, birth_anchor, birth_world,
                              start_world, end_world, homography, history,
                              frame_index, birth_frame, prev_thrower_key=None):
        cls_name = str(actor.get('cls', '')).lower()
        if cls_name not in ('person', 'vehicle', 'scooter') or birth_world is None:
            return None, False

        try:
            actor_key = self._actor_key(actor)
            ax1, ay1, ax2, ay2 = map(float, actor['box'])
        except (KeyError, TypeError, ValueError):
            return None, False

        actor_anchor = self._actor_ground_anchor(actor)
        actor_world = self._project_point(actor_anchor, homography)
        if actor_world is None:
            return None, False

        world_dist = math.hypot(
            birth_world[0] - actor_world[0],
            birth_world[1] - actor_world[1],
        )
        world_width = self._projected_actor_width((ax1, ay1, ax2, ay2), homography)
        if cls_name in ('vehicle', 'scooter'):
            threshold = max(160.0, world_width * 0.85)
            origin_margin = max(130.0, (ax2 - ax1) * 1.05, (ay2 - ay1) * 0.45)
        else:
            threshold = max(85.0, world_width * 1.9)
            origin_margin = max(75.0, (ax2 - ax1) * 1.25, (ay2 - ay1) * 0.45)

        score = world_dist / max(threshold, 1e-6)
        if actor_key == prev_thrower_key:
            score *= self.thrower_previous_bonus

        if start_world is not None and end_world is not None:
            score *= self._trajectory_direction_factor(actor_world, start_world, end_world)

        frame_gap = abs(int(frame_index) - int(birth_frame))
        score *= (1.0 + min(frame_gap, 45) * 0.025)

        release_like = self._release_origin_near_actor(history, actor)
        if release_like:
            score *= 0.82

        if self._point_to_box_distance(birth_anchor, actor['box']) <= origin_margin:
            score *= 0.88

        return score, release_like

    def _drain_backward_results(self, vehicle_history=None):
        # worker 結果只能在 main/update thread 套用，避免 shared dict 競爭。
        while True:
            try:
                result = self._backward_results.get_nowait()
            except queue.Empty:
                break

            if result.get('error'):
                continue

            litter_id = result.get('litter_id')
            actor_key = result.get('actor_key')
            if actor_key is None:
                continue

            if litter_id in self.active_litters:
                self.active_litters[litter_id]['thrower_key'] = actor_key
                self.active_litters[litter_id]['thrower_center'] = result.get('mark_items', [{}])[0].get('center')
                self.active_litters[litter_id]['backward_result'] = {
                    'actor_key': actor_key,
                    'score': result.get('score'),
                    'birth_frame': result.get('birth_frame'),
                    'confirm_frame': result.get('confirm_frame'),
                }

            plate_key = result.get('plate_key')
            plate_blocked_since_litter = bool(result.get('plate_blocked_since_litter', False))

            if (
                plate_blocked_since_litter and
                plate_key is not None and
                plate_key[0] in ('vehicle', 'scooter') and
                vehicle_history is not None
            ):
                plate_entry = vehicle_history[plate_key[1]]
                if plate_entry.get('license_plate') is None:
                    plate_entry['plate_search_until_found'] = True
                    plate_entry['plate_blocked_since_litter'] = True
                    plate_entry['plate_search_birth_frame'] = result.get('birth_frame')
                    plate_entry['plate_search_litter_id'] = litter_id

            for mark in result.get('mark_items', []):
                mark_key = mark.get('actor_key')
                self._mark_violator(
                    mark_key,
                    mark.get('center'),
                    ttl=self.confirmed_violator_ttl,
                    until_plate_found=(
                        plate_blocked_since_litter and
                        mark_key == plate_key
                    ),
                )

            plate_items = result.get('plate_roi_items') or []
            if plate_items:
                with self._backward_plate_lock:
                    self.backward_plate_roi_items.extend(plate_items)

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = map(float, box[:4])
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _crop_actor_roi(frame, box, frame_w, frame_h):
        x1, y1, x2, y2 = map(float, box[:4])
        pad = max(4.0, 0.04 * max(x2 - x1, y2 - y1, 1.0))
        ix1 = max(0, min(int(math.floor(x1 - pad)), int(frame_w)))
        iy1 = max(0, min(int(math.floor(y1 - pad)), int(frame_h)))
        ix2 = max(0, min(int(math.ceil(x2 + pad)), int(frame_w)))
        iy2 = max(0, min(int(math.ceil(y2 + pad)), int(frame_h)))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        roi = frame[iy1:iy2, ix1:ix2].copy()
        return roi if roi.size > 0 else None

    @staticmethod
    def _snapshot_to_actor(snapshot):
        try:
            return {
                'cls': str(snapshot.get('cls', '')).lower(),
                'track_id': int(snapshot['track_id']),
                'box': np.asarray(snapshot['box'], dtype=np.float32),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _linked_vehicle_key(self, actor_key, frame_actors):
        if actor_key is None:
            return None
        if actor_key[0] in ('vehicle', 'scooter'):
            return actor_key

        actors = []
        for snapshot in frame_actors:
            actor = self._snapshot_to_actor(snapshot)
            if actor is not None:
                actors.append(actor)
        return self._find_vehicle_for_person(actor_key[1], actors)

    @staticmethod
    def _snapshot_center_for_key(actor_key, frame_actors):
        for snapshot in frame_actors:
            try:
                key = (str(snapshot.get('cls', '')).lower(), int(snapshot['track_id']))
            except (KeyError, TypeError, ValueError):
                continue
            if key == actor_key:
                return snapshot.get('center')
        return None

    def _plate_roi_items_for_key(self, actor_key, actor_frames, birth_frame):
        if actor_key is None or actor_key[0] not in ('vehicle', 'scooter'):
            return []

        items = []
        sorted_frames = sorted(
            actor_frames,
            key=lambda frame: abs(int(frame.get('frame_index', birth_frame)) - int(birth_frame)),
        )
        for frame_snapshot in sorted_frames:
            for snapshot in frame_snapshot.get('actors', []):
                try:
                    key = (str(snapshot.get('cls', '')).lower(), int(snapshot['track_id']))
                except (KeyError, TypeError, ValueError):
                    continue
                if key != actor_key or snapshot.get('plate_roi') is None:
                    continue
                vehicle = {
                    'cls': key[0],
                    'track_id': key[1],
                    'box': np.asarray(snapshot.get('box'), dtype=np.float32),
                }
                items.append((vehicle, snapshot['plate_roi']))
                if len(items) >= self.backward_plate_roi_per_result:
                    return items
        return items

    def register_action_violators(self, person_action_map, actors, person_vehicle_map=None, ttl=None):
        # STGCN 旁路：動作模型已判定 littering 時，直接註冊 person 與其對應車輛。
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
        # 將 actor 統一成 (class, track_id) key，避免 person/vehicle id 空間互相衝突。
        return (actor['cls'], int(actor['track_id']))

    def _normalize_vehicle_like_key(self, actor_key_or_id):
        # person_vehicle_map 可能傳 tuple 或舊版純 id；統一成車輛類 key。
        if isinstance(actor_key_or_id, tuple) and len(actor_key_or_id) == 2:
            cls_name = str(actor_key_or_id[0]).lower()
            if cls_name not in ('vehicle', 'scooter'):
                cls_name = 'vehicle'
            return (cls_name, int(actor_key_or_id[1]))

        # 舊呼叫端只傳 vehicle id 時的相容路徑。
        return ('vehicle', int(actor_key_or_id))

    def _actor_center(self, actor):
        # actor bbox 中心點，用於違規顯示連續性與 rebind。
        ax1, ay1, ax2, ay2 = actor['box']
        return ((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0)

    def _mark_violator(self, actor_key, center, ttl, until_plate_found=False):
        # 寫入或延長違規者 TTL；center 用來避免 ID 重用造成誤標。
        if actor_key is None:
            return

        prev = self.violators.get(actor_key)
        if prev is None:
            self.violators[actor_key] = {
                'ttl': ttl,
                'center': center,
                'missed': 0,
                'until_plate_found': bool(until_plate_found),
            }
            return

        prev['ttl'] = max(prev['ttl'], ttl)
        prev['until_plate_found'] = (
            bool(prev.get('until_plate_found', False)) or bool(until_plate_found)
        )
        if center is not None:
            prev['center'] = center
        prev['missed'] = 0

    def _find_vehicle_for_person(self, person_id, actors):
        # 找出與 person bbox 重疊最多的 vehicle/scooter，供 STGCN 違規同步標車。
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
        # 基礎 bbox IoU，給 person-to-vehicle 關聯使用。
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
        # 追蹤器短暫換 ID 時，允許同類別、近距離 actor 繼承違規狀態。
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
        # 舊 API 相容：出生幀 thrower 也走新版 pseudo-ground 評分。
        return self._find_thrower_for_litter(start_centroid, actors)

    def _find_thrower_for_litter(self, litter_ref, actors, history=None, prev_thrower_key=None):
        """
        依車輛 bbox 底邊點估計 homography，將 litter 與 actor 投影到類 3D
        pseudo-ground 座標後，重新計算最可能的垃圾丟棄者。
        """
        # 使用 pseudo-ground 座標計算 litter 與 actor 距離，降低透視造成的誤選。
        best_actor_key = None
        best_center = None
        best_score = float('inf')

        fallback_actor_key = None
        fallback_center = None
        fallback_score = float('inf')
        release_actor_key = None
        release_center = None
        release_score = float('inf')

        litter_anchor = self._litter_ground_anchor(litter_ref)
        if litter_anchor is None:
            return None, None

        homography = self._estimate_ground_homography(actors, litter_anchor)
        litter_world = self._project_point(litter_anchor, homography)
        start_world = None
        end_world = None
        if history and len(history) >= 2:
            start_world = self._project_point(history[0], homography)
            end_world = self._project_point(history[-1], homography)

        for actor in actors:
            cls_name = str(actor.get('cls', '')).lower()
            if cls_name not in ('person', 'vehicle', 'scooter'):
                continue

            try:
                track_id = int(actor['track_id'])
                ax1, ay1, ax2, ay2 = map(float, actor['box'])
            except (KeyError, TypeError, ValueError):
                continue

            actor_key = (cls_name, track_id)
            actor_center = self._actor_center(actor)
            actor_anchor = self._actor_ground_anchor(actor)
            actor_world = self._project_point(actor_anchor, homography)
            if actor_world is None or litter_world is None:
                continue

            world_dist = math.hypot(
                litter_world[0] - actor_world[0],
                litter_world[1] - actor_world[1],
            )
            world_width = self._projected_actor_width((ax1, ay1, ax2, ay2), homography)
            if cls_name in ('vehicle', 'scooter'):
                threshold = max(180.0, world_width * 0.75)
            else:
                threshold = max(90.0, world_width * 1.8)

            score = world_dist / max(threshold, 1e-6)
            if actor_key == prev_thrower_key:
                score *= self.thrower_previous_bonus

            if start_world is not None and end_world is not None:
                score *= self._trajectory_direction_factor(actor_world, start_world, end_world)

            if score < fallback_score:
                fallback_score = score
                fallback_actor_key = actor_key
                fallback_center = actor_center

            if (
                score <= self.thrower_release_origin_score_limit and
                self._release_origin_near_actor(history, actor) and
                score < release_score
            ):
                release_score = score
                release_actor_key = actor_key
                release_center = actor_center

            if score <= 1.0 and score < best_score:
                best_score = score
                best_actor_key = actor_key
                best_center = actor_center

        if (
            best_actor_key is None and
            fallback_actor_key is not None and
            fallback_score <= self.thrower_fallback_score_limit
        ):
            best_actor_key = fallback_actor_key
            best_center = fallback_center
        elif best_actor_key is None and release_actor_key is not None:
            best_actor_key = release_actor_key
            best_center = release_center

        return best_actor_key, best_center

    def _release_origin_near_actor(self, history, actor):
        # 反追蹤專用：軌跡起點要貼近 actor，終點要已離開 actor，避免把持有中物件或舊垃圾誤綁。
        if not history or len(history) < 2:
            return False

        try:
            box = actor['box']
            cls_name = str(actor.get('cls', '')).lower()
            ax1, ay1, ax2, ay2 = map(float, box[:4])
        except (KeyError, TypeError, ValueError):
            return False

        if cls_name not in ('person', 'vehicle', 'scooter'):
            return False

        start_point = history[0]
        end_point = history[-1]
        dx = float(end_point[0]) - float(start_point[0])
        dy = float(end_point[1]) - float(start_point[1])
        if abs(dx) < self.min_confirm_horizontal_displacement or dy < self.min_confirm_downward_displacement:
            return False

        width = max(ax2 - ax1, 1.0)
        height = max(ay2 - ay1, 1.0)
        if cls_name in ('vehicle', 'scooter'):
            start_margin = max(120.0, width * 0.9, height * 0.35)
        else:
            start_margin = max(70.0, width * 1.2, height * 0.35)

        if self._point_to_box_distance(start_point, box) > start_margin:
            return False

        release_gap = max(12.0, min(45.0, start_margin * 0.18))
        return self._point_to_box_distance(end_point, box) >= release_gap

    @staticmethod
    def _point_to_box_distance(point, box):
        # 點到 bbox 的最短距離；點在框內時距離為 0。
        px, py = map(float, point)
        x1, y1, x2, y2 = map(float, box[:4])
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return math.hypot(dx, dy)

    def _litter_ground_anchor(self, litter_ref):
        # litter anchor 使用 bbox 底部中心，較接近地面接觸點。
        try:
            values = np.asarray(litter_ref, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None

        if values.size >= 4:
            x1, y1, x2, y2 = map(float, values[:4])
            return ((x1 + x2) / 2.0, y2)
        if values.size >= 2:
            return (float(values[0]), float(values[1]))
        return None

    def _actor_ground_anchor(self, actor):
        # actor anchor 使用 bbox 底部中心，對齊 ground homography 估計。
        ax1, ay1, ax2, ay2 = map(float, actor['box'])
        # vehicle/scooter 以 bbox 最底部中心當作接地點，對齊 homography 的估計來源。
        return ((ax1 + ax2) / 2.0, ay2)

    def _estimate_ground_homography(self, actors, litter_anchor):
        # 用畫面中的車輛底部點估計簡化 homography；無車輛時退回穩定的等比例投影。
        vehicle_bottoms = []
        vehicle_heights = []

        for actor in actors:
            cls_name = str(actor.get('cls', '')).lower()
            if cls_name not in ('vehicle', 'scooter'):
                continue
            try:
                ax1, ay1, ax2, ay2 = map(float, actor['box'])
            except (KeyError, TypeError, ValueError):
                continue

            vehicle_bottoms.append(((ax1 + ax2) / 2.0, ay2))
            vehicle_heights.append(max(ay2 - ay1, 1.0))

        if not vehicle_bottoms:
            return np.asarray([
                [100.0, 0.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 0.0, 100.0],
            ], dtype=np.float32)

        bottom_x = np.asarray([p[0] for p in vehicle_bottoms], dtype=np.float32)
        bottom_y = np.asarray([p[1] for p in vehicle_bottoms], dtype=np.float32)
        median_h = float(np.median(vehicle_heights)) if vehicle_heights else 80.0
        y_spread = float(np.max(bottom_y) - np.min(bottom_y)) if bottom_y.size > 1 else 0.0

        center_x = float(np.median(bottom_x))
        ground_ref_y = max(float(np.max(bottom_y)), float(litter_anchor[1]))
        horizon_offset = max(80.0, median_h * 1.2, y_spread * 1.5)
        horizon_y = float(np.min(bottom_y)) - horizon_offset
        ground_scale = max(ground_ref_y - horizon_y, 80.0)

        return np.asarray([
            [ground_scale, 0.0, -ground_scale * center_x],
            [0.0, -ground_scale, ground_scale * ground_ref_y],
            [0.0, 1.0, -horizon_y],
        ], dtype=np.float32)

    def _project_point(self, point, homography):
        # 將影像點投影到 pseudo-ground，並避免深度接近 0 造成數值爆炸。
        if point is None:
            return None

        x, y = map(float, point)
        projected = homography @ np.asarray([x, y, 1.0], dtype=np.float32)
        depth = float(projected[2])
        if abs(depth) < self.homography_min_depth:
            depth = self.homography_min_depth if depth >= 0.0 else -self.homography_min_depth

        return (float(projected[0]) / depth, float(projected[1]) / depth)

    def _projected_actor_width(self, box, homography):
        # 投影後 actor 寬度作為距離容忍門檻，讓遠近車輛尺度更一致。
        ax1, ay1, ax2, ay2 = map(float, box[:4])
        left = self._project_point((ax1, ay2), homography)
        right = self._project_point((ax2, ay2), homography)
        if left is None or right is None:
            return max(ax2 - ax1, 1.0)

        return max(math.hypot(right[0] - left[0], right[1] - left[1]), 1.0)

    @staticmethod
    def _trajectory_direction_factor(actor_world, start_world, end_world):
        # 軌跡方向若像是從 actor 往外離開，稍微降低該 actor 的 score。
        move_vec = (end_world[0] - start_world[0], end_world[1] - start_world[1])
        from_actor_vec = (start_world[0] - actor_world[0], start_world[1] - actor_world[1])
        move_len = math.hypot(move_vec[0], move_vec[1])
        actor_len = math.hypot(from_actor_vec[0], from_actor_vec[1])
        if move_len < 1e-6 or actor_len < 1e-6:
            return 1.0

        cosine = (
            move_vec[0] * from_actor_vec[0] +
            move_vec[1] * from_actor_vec[1]
        ) / (move_len * actor_len)
        if cosine >= 0.25:
            return 0.9
        if cosine <= -0.25:
            return 1.15
        return 1.0
