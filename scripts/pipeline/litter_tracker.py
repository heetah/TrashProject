# -*- coding: utf-8 -*-
# 全域垃圾追蹤器：把 RTDETR 候選 litter 串成軌跡，判斷 pending/confirmed，並反推丟擲者。
import math
import os
import queue
import threading
import time
import numpy as np
from collections import deque
from scipy.spatial import distance
from pipeline.geometry import validate_trajectory, calculate_mask_overlap_ratio


from pipeline.litter.trajfit import (
    _backtrack_int_env,
    _backtrack_float_env,
    _trajfit_airborne_prefix,
    _trajfit_fit_ballistic,
    _trajfit_point_at,
)

# === fps 正規化基準 ===
# 所有「像素」閾值（位移、span、相對分離…）是與取樣率無關的物理事實，維持不變。
# 只有「幀窗 / age / 每幀速度上限」這些與 fps 相關的參數需要隨 fps 縮放：
#   - 高 fps（如 30）每幀位移較小，但同一物理運動會持續更多幀；
#   - 用更長的幀窗去累積到同樣的像素位移，即可在不調降門檻的前提下救回 30fps 漏判。
# 以 10fps 為基準：fps=10 時 frame_scale=1.0，行為與舊版完全相同（零退化）。
REF_FPS = 10.0
FPS_CLAMP_MIN = 5.0
FPS_CLAMP_MAX = 60.0

# === 軌跡與配對 ===
MAX_MISSED_FRAMES = 10
TRAJECTORY_HISTORY_LEN = 15
PENDING_SHAPE_CHANGE_RATIO = 0.60       # pending 嚴格，避免雜訊延續
CONFIRMED_SHAPE_CHANGE_RATIO = 1.20     # confirmed 寬鬆，吸收形變

# === Confirm 門檻：一般 thrower ===
MIN_CONFIRM_AGE = 2
MIN_CONFIRM_ABS_DISPLACEMENT = 14.0
MIN_CONFIRM_DOWNWARD_DISPLACEMENT = 7.0
MIN_CONFIRM_HORIZONTAL_DISPLACEMENT = 5.0

# === Confirm 門檻：vehicle thrower 加嚴（FP 多為車輛部件）===
MIN_CONFIRM_AGE_VEHICLE = 3
MIN_CONFIRM_DOWNWARD_DISPLACEMENT_VEHICLE = 12.0
MAX_HORIZ_TO_DOWN_RATIO_VEHICLE = 3.5    # 純水平滑動非丟擲
MAX_VEHICLE_THROWER_STEP_PX = 200.0      # 真實單步 < 150px

# === 車身/貨物誤判 FP 抑制（相對載體車輛分離判別）===
# FP 大宗：載貨貨車車斗貨物 / 車身 / 車牌 / 鄰車（後照鏡、車燈）被 litter detector 誤判。
# 靜態重疊無法區分「被丟出但仍與車重疊的真實垃圾」與「車身部件」。
# 改以「相對載體車輛的淨位移」判別：部件隨車移動 (rel ≈ 0) → 擋；被丟出者脫離車輛 (rel 大) → 放行。
CARRIER_OVERLAP_MIN = 0.15              # litter 與某車輛重疊達此值才視為「在車上」，啟用分離檢查
MIN_VEHICLE_RELATIVE_SEPARATION = 60.0  # litter 相對載體車輛的最小淨位移（小於此視為隨車移動的部件）

# 註：隨車部件 / 純水平條紋 / 車輛共動 等 litter 候選 FP 篩選已集中到 detect 前處理
# （smallFunction.litter_candidate_is_vehicle_fp）；tracker 只負責追蹤與軌跡確認，不再做這些判斷。

# === 快速落下特例 (10fps 場景，age=2 vehicle thrower) ===
FAST_DROP_MIN_DOWNWARD = 35.0
FAST_DROP_MIN_HORIZ_RATIO = 0.15         # 真丟擲水平/向下 > 0.15；< 0.15 多為 detector jitter
FAST_DROP_ACTORLESS_MAX_HORIZ_RATIO = 1.20  # 無 birth actor 時，須為向下主導，避免遠方車輛晚到認領
FAST_DROP_MAX_FRAME_GAP = 2

# === Fall-then-stable confirm（driver throw → 落地不動）===
# 軌跡曾有 cy 下降 + 後段穩定 → 推測為落地後 litter；vehicle thrower 必填、防 FP。
FALL_STABLE_MIN_AGE = 8
FALL_STABLE_MIN_FALL_DISP = 25.0         # peak 後 cy 增加至少 25px（落地有下落）
FALL_STABLE_MAX_TAIL_JITTER = 15.0       # 最後 4 幀 cy 範圍 < 15px (穩定落地)
FALL_STABLE_TAIL_WINDOW = 4

# === 靜止舊垃圾抑制 ===
STATIC_CANDIDATE_MIN_AGE = 3
MIN_HISTORY_SPAN_FOR_CONFIRM = 10.0
STATIONARY_LOCK_AGE = 10
STATIONARY_LOCK_SPAN = 8.0
# 靜止鎖定軌跡不吸收新偵測，會在同一靜止點每幀生出未上鎖的「雙胞胎」軌跡；
# 該雙胞胎一旦遇 detector 抖動跳動就可能誤 confirm。
# 修正：新軌跡若與既有 stationary_locked 軌跡共位（半徑內）則繼承鎖定，斷開連鎖。
# 半徑為像素鄰近度（fps 無關），需大於 detector 抖動、小於真實丟擲位移。
LOCK_INHERIT_RADIUS = 30.0

# === 違規者顯示 ===
CONFIRMED_VIOLATOR_TTL = 60
MAX_VIOLATOR_JUMP = 80.0
VIOLATOR_MAX_MISSED = 5
VIOLATOR_REBIND_DISTANCE = 60.0

# === Thrower scoring (前向 + backward) ===
HOMOGRAPHY_MIN_DEPTH = 25.0
THROWER_PREVIOUS_BONUS = 0.85
THROWER_FALLBACK_SCORE_LIMIT = 1.25
THROWER_BIRTH_BOX_DIST_LIMIT = 270.0
THROWER_RELEASE_ORIGIN_SCORE_LIMIT = 4.0
THROWER_EDGE_RELEASE_MIN_SEPARATION = 40.0

# === Backward resolver ===
BACKWARD_ACTOR_HISTORY_LEN = 120
BACKWARD_PRE_BIRTH_FRAMES = 24
BACKWARD_POST_BIRTH_FRAMES = 18
BACKWARD_SCORE_LIMIT = 2.10
BACKWARD_RELEASE_SCORE_LIMIT = 3.40
BACKWARD_PLATE_ROI_PER_RESULT = 3

# === STGCN action backtrack ===
ACTION_VEHICLE_BACKTRACK_FRAMES = 90


class GlobalLitterTracker:
    # 對外 attribute alias（測試 / detect.py 直接讀）
    stationary_lock_age = STATIONARY_LOCK_AGE

    def __init__(self, distance_threshold=250, fps=REF_FPS):
        self.distance_threshold = distance_threshold

        # === fps 正規化 ===
        self.fps = float(fps) if (fps and float(fps) > 0) else REF_FPS
        _fps_clamped = min(max(self.fps, FPS_CLAMP_MIN), FPS_CLAMP_MAX)
        # frame_scale：高 fps > 1（幀窗放長），10fps == 1.0（不變）。
        self._frame_scale = _fps_clamped / REF_FPS

        def _scale_up(frames):
            # 幀數/age：與 fps 同向縮放（高 fps 需要更多幀涵蓋同一段真實時間）。
            return max(int(round(frames * self._frame_scale)), int(frames))

        def _scale_down_px_per_frame(px):
            # 每幀像素速度上限：與 fps 反向縮放（高 fps 每幀位移較小，維持 px/sec 不變）。
            return px / self._frame_scale

        # --- 像素閾值：fps 無關，維持原值（同時對外暴露，供測試/調校讀取）---
        self.min_history_span_for_confirm = MIN_HISTORY_SPAN_FOR_CONFIRM
        self.stationary_lock_span = STATIONARY_LOCK_SPAN
        self.thrower_fallback_score_limit = THROWER_FALLBACK_SCORE_LIMIT
        self.min_confirm_downward_displacement = MIN_CONFIRM_DOWNWARD_DISPLACEMENT
        self.min_confirm_downward_displacement_vehicle = MIN_CONFIRM_DOWNWARD_DISPLACEMENT_VEHICLE

        # --- 持續性 / 幀窗參數：隨 fps 放長 ---
        # 高 fps（且小物件偵測稀疏，detection gap 大）時，唯有放長軌跡記憶與容錯幀數，
        # 才能在偵測斷續間維持同一條軌跡，避免每隔數幀就重生新 id、age 永遠長不大。
        self.trajectory_history_len = _scale_up(TRAJECTORY_HISTORY_LEN)
        self.max_missed_frames = _scale_up(MAX_MISSED_FRAMES)
        self.fast_drop_max_frame_gap = _scale_up(FAST_DROP_MAX_FRAME_GAP)

        # --- age / 成熟度門檻：以「偵測次數」計，與 fps 無關 ---
        # age 數的是「被配對到的偵測次數」而非經過幀數。小快物件偵測稀疏，
        # 把 age 門檻隨 fps 放大會讓 confirm 變得不可達（實測 case 13 即因此漏判）。
        # 真正的證據強度由像素位移 + 軌跡物理 + 每幀速度檢查把關，age 維持基準值即可。
        self.min_confirm_age = MIN_CONFIRM_AGE
        self.min_confirm_age_vehicle = MIN_CONFIRM_AGE_VEHICLE
        self.static_candidate_min_age = STATIC_CANDIDATE_MIN_AGE
        self.stationary_lock_age = STATIONARY_LOCK_AGE
        self.fall_stable_min_age = FALL_STABLE_MIN_AGE
        self.fall_stable_tail_window = FALL_STABLE_TAIL_WINDOW

        # --- 每幀像素速度上限：隨 fps 反向縮放 ---
        self.max_vehicle_thrower_step_px = _scale_down_px_per_frame(MAX_VEHICLE_THROWER_STEP_PX)

        # 違規升級門檻：LITTER_REQUIRE_VEHICLE=1 時，confirmed litter 只有在 thrower 關聯到
        # vehicle/scooter 時才升級為違規（畫框 + 車牌 OCR）。針對「行人/小便場景被誤判丟擲」的
        # FP：純行人 thrower（無車輛關聯）不再升級成違規，但 litter 物件追蹤本身不受影響。
        # 預設 0（行為不變，可 env 開啟並 A/B）。
        try:
            self.require_vehicle_for_violation = int(os.environ.get("LITTER_REQUIRE_VEHICLE", "0")) != 0
        except ValueError:
            self.require_vehicle_for_violation = False

        # 反追蹤方向因子：LITTER_BACKTRACK_REVVEL=1 時，把舊的「整段 start→end + 3 級 cosine 分桶」
        # 換成 OCM(Observation-Centric Momentum)式「早期速度反向外插」連續因子。litter 被拋出後，
        # 釋放後前段速度最能保留拋擲方向(bounce/roll 之前)；反向速度向量指回真正來源，能在 litter
        # 橫跨多物件時排除「飛過但非來源」的路過 actor。預設 0(行為不變，可 env 開啟並 A/B)。
        # 影像空間版(v1)：與舊因子同樣留在 2D 影像座標(避免空中點投影爆走)；BEV 度量空間為後續精修。
        try:
            self._revvel_enabled = int(os.environ.get("LITTER_BACKTRACK_REVVEL", "0")) != 0
        except ValueError:
            self._revvel_enabled = False
        self._revvel_early_pts = max(2, _backtrack_int_env("LITTER_BACKTRACK_REVVEL_PTS", 5))
        self._revvel_min_speed = max(0.0, _backtrack_float_env("LITTER_BACKTRACK_REVVEL_MIN_SPEED", 4.0))
        self._revvel_gain = max(0.0, _backtrack_float_env("LITTER_BACKTRACK_REVVEL_GAIN", 0.25))
        self._revvel_max_bonus = _backtrack_float_env("LITTER_BACKTRACK_REVVEL_MAX_BONUS", 0.75)
        self._revvel_max_penalty = _backtrack_float_env("LITTER_BACKTRACK_REVVEL_MAX_PENALTY", 1.25)

        # BEV 基板：LITTER_BEV_STABLE=1 時，把每幀從「當前車輛底邊」重估的 ground homography，
        # 改成「跨幀累積車輛底邊 → 擬合單一穩定地面平面 → 全程重用」(固定機位的一次性 BEV 標定)。
        # 穩定度量空間讓 world_dist 距離評分不再逐幀抖動，反追蹤關聯更可靠。預設 0(行為不變，可 A/B)。
        # 觀測數不足時自動 fallback 回每幀估計。
        try:
            self._bev_stable_enabled = int(os.environ.get("LITTER_BEV_STABLE", "0")) != 0
        except ValueError:
            self._bev_stable_enabled = False
        self._bev_min_obs = max(3, _backtrack_int_env("LITTER_BEV_MIN_OBS", 12))
        self._bev_recompute_every = max(1, _backtrack_int_env("LITTER_BEV_RECOMPUTE_EVERY", 8))
        self._bev_buffer_cap = max(self._bev_min_obs, _backtrack_int_env("LITTER_BEV_BUFFER_CAP", 600))

        # 情境二 dismount 持久邊：LITTER_DISMOUNT_EDGE=1 時，person↔vehicle 綁定改成 TTL-aware
        # 持久邊(記 first/last_bound_frame + bound_count)。person 下車後走遠(IoM 掉到 0)，邊仍在
        # TTL 內有效 → 便溺/丟垃圾歸因可回溯到原車(scenario 2)；超過 TTL 自動失效 → 避免 track_id
        # 回收造成的舊綁定誤歸因;bound_count >= min_bind 過濾「路過車輛 1 幀」假綁定。預設 0(行為不變)。
        # 對照舊 person_to_vehicle_history(無 TTL、永久記憶、1 幀即綁)。
        try:
            self._dismount_edge_enabled = int(os.environ.get("LITTER_DISMOUNT_EDGE", "0")) != 0
        except ValueError:
            self._dismount_edge_enabled = False
        self._dismount_ttl_sec = max(0.0, _backtrack_float_env("LITTER_DISMOUNT_TTL_SEC", 8.0))
        self._dismount_min_bind = max(1, _backtrack_int_env("LITTER_DISMOUNT_MIN_BIND", 2))

        # 軌跡反演歸因(LITTER_BACKTRACK_TRAJFIT=1):backward worker 的評分對象從「落點鄰近度
        # + 修正因子」改成「釋放事件時空相交」——litter 空中段擬合拋物線,向後外插出釋放位置,
        # 只有「在釋放時刻出現在釋放點附近」的 actor 得分。時間一致性內建 → 消滅「落地後才
        # 路過落點」的 FP;門檻由擬合殘差 sigma 縮放,取代常數 px 門檻。空中段點數不足、近靜止
        # (放置)、曲率非物理時回 None → 自動 fallback 既有落點評分鏈。預設 0(行為不變)。
        try:
            self._trajfit_enabled = int(os.environ.get("LITTER_BACKTRACK_TRAJFIT", "0")) != 0
        except ValueError:
            self._trajfit_enabled = False
        self._trajfit_min_air_pts = max(3, _backtrack_int_env("LITTER_TRAJFIT_MIN_AIR_PTS", 3))
        self._trajfit_max_back = max(1, _backtrack_int_env(
            "LITTER_TRAJFIT_MAX_BACK_FRAMES", BACKWARD_PRE_BIRTH_FRAMES))
        self._trajfit_gate = max(0.1, _backtrack_float_env("LITTER_TRAJFIT_GATE", 1.0))
        self._trajfit_min_speed = max(0.0, _backtrack_float_env("LITTER_TRAJFIT_MIN_SPEED", 2.0))
        self._trajfit_sigma_floor = max(0.5, _backtrack_float_env("LITTER_TRAJFIT_SIGMA_FLOOR", 2.0))

        # Offline Person↔Vehicle 關聯(PV_ASSOC=1):全片累積輕量 actor 歷史 + confirmed litter 事件,
        # 收尾(finalize_associations) 跑事件錨定 1對1 匈牙利+dustbin(person_vehicle_assoc.py)。
        # 不同於 online dismount-edge,這是全片 offline 全域最優配對(車先到/人下車走遠便溺再上車的跨時間
        # 關聯)。預設 0:不累積、不 finalize、summary 不變 → regression byte-identical。
        try:
            self._pv_assoc_enabled = int(os.environ.get("PV_ASSOC", "0")) != 0
        except ValueError:
            self._pv_assoc_enabled = False
        self._pv_full_history = []       # [{frame_index, actors:[{cls,track_id,box,center}]}](無 plate_roi)
        self._pv_litter_events = []      # [{frame_index, center}] confirmed litter 錨定
        self._pv_litter_seen_ids = set() # litter id 去重,confirm 跨幀只記一次

        # confirmed litter 事件記錄(events.jsonl 來源,與 PV_ASSOC 無關、一律開啟)。
        # 每個 litter id 於「首次確認」記一次,含 frame、bbox、thrower、是否升級為違規。
        self._litter_events = []
        self._litter_event_seen_ids = set()
        self._litter_events_by_id = {}

        # === Mutable state ===
        self.active_litters = {}            # {litter_id: {bbox, history, age, state, thrower_key, ...}}
        self.violators = {}                 # {(cls, track_id): {ttl, center, action, ...}}
        self.next_id = 0
        self.person_to_vehicle_history = {}
        # (person_id, action_frame)→vehicle，避免同一 person 多次 episode 被片尾最新車輛覆蓋。
        self._action_vehicle_associations = {}
        self._latest_action_event_frames = {}
        # 情境二 dismount 持久邊：person_id -> {vehicle_key, first_frame, last_bound_frame, bound_count}。
        self._dismount_edges = {}
        self._current_frame_index = 0
        self.actor_frame_history = deque(maxlen=BACKWARD_ACTOR_HISTORY_LEN)
        self._smart_context_frames = max(
            BACKWARD_PRE_BIRTH_FRAMES,
            int(round(
                self.fps
                * max(
                    1.0,
                    _backtrack_float_env("SMART_BACKTRACK_CONTEXT_SEC", 10.0),
                )
            )),
        )
        # Longer history is lightweight (no image ROI); short ring above keeps
        # only OCR crops and bounds image memory.
        self._smart_actor_history = deque(
            maxlen=self._smart_context_frames + BACKWARD_POST_BIRTH_FRAMES + 2
        )
        # RT-DETR class/confidence outputs before geometry/motion/holding
        # filters. They never enter confirmation. After an event is confirmed,
        # a short, motion-consistent prefix may be recovered solely for release
        # trajectory fitting.
        self._raw_litter_history = deque(
            maxlen=self._smart_context_frames + BACKWARD_POST_BIRTH_FRAMES + 2
        )
        self._raw_prefix_enabled = (
            os.environ.get("SMART_BACKTRACK_RAW_PREFIX", "1") not in ("0", "")
        )
        self.backward_plate_roi_items = []
        self._actor_tracklet_epochs = {}

        # BEV 基板：跨幀累積車輛/機車底邊中心 + 高度，擬合穩定地面平面並快取。
        self._bev_bottoms = deque(maxlen=self._bev_buffer_cap)
        self._bev_heights = deque(maxlen=self._bev_buffer_cap)
        self._bev_cached_homography = None
        self._bev_cache_count = 0
        self._bev_lock = threading.Lock()

        # === Backward worker thread ===
        self._actor_history_lock = threading.Lock()
        self._backward_plate_lock = threading.Lock()
        self._backward_tasks = queue.Queue(maxsize=64)
        self._backward_results = queue.Queue()
        self._backward_stop = object()
        self._backward_accepting = True
        self._backward_stop_sent = False
        self._closed = False
        self._backtrack_revisions = {}
        self._applied_backtrack_revisions = {}
        self._applied_backtrack_signatures = {}
        self._backtrack_marked_keys = {}
        self._smart_candidate_tables = {}
        self._smart_tasks = {}
        self._smart_backtrack_enabled = (
            os.environ.get("SMART_BACKTRACK", "1") not in ("0", "")
        )
        self._smart_resolver = None
        if self._smart_backtrack_enabled:
            try:
                from pipeline.backtrack.resolver import SmartBacktrackResolver
                self._smart_resolver = SmartBacktrackResolver(fps=self.fps)
                # Do not accumulate/run the legacy one-person↔one-vehicle matrix
                # only after smart initialization actually succeeds.
                self._pv_assoc_enabled = False
            except Exception as exc:  # noqa: BLE001 - legacy resolver remains safe fallback
                self._smart_backtrack_enabled = False
                print(f"[SMART_BACKTRACK] initialization failed; legacy fallback: {exc}")
        self._backward_thread = threading.Thread(
            target=self._backward_worker,
            name="litter-backward-resolver",
            daemon=True,
        )
        self._backward_thread.start()

        self._fallback_frame_index = 0
        self._debug = os.environ.get("LITTER_DEBUG", "0") not in ("0", "")   # LITTER_DEBUG=1 開啟 per-frame 印出

    def _record_raw_litter_frame(self, raw_detected_litters, frame_index):
        """Keep detector outputs for confirmed-only trajectory recovery.

        This buffer is observational evidence only. It is never passed into
        the pending/confirmed state machine below.
        """
        boxes = []
        for litter_box in raw_detected_litters or []:
            try:
                values = tuple(float(value) for value in litter_box[:5])
            except (TypeError, ValueError):
                continue
            if len(values) != 5 or not np.isfinite(values).all():
                continue
            boxes.append(values)
        self._raw_litter_history.append({
            'frame_index': int(frame_index),
            'boxes': boxes,
        })

    def _recover_raw_litter_prefix(
        self,
        history,
        history_frames,
        history_boxes,
        history_confidences,
    ):
        """Prepend a motion-consistent raw prefix to a confirmed trajectory.

        At least two accepted observations anchor a constant-velocity backward
        prediction. Each earlier raw point must fall within two observed-box
        diagonals of that prediction. The anchor is updated after every match,
        allowing acceleration while preventing unrelated detector boxes from
        entering merely because they are nearby. Recovery stops at the first
        missing/rejected frame and never changes confirmation or birth_frame.
        """
        count = min(
            len(history), len(history_frames), len(history_boxes),
            len(history_confidences),
        )
        if count < 2:
            return (
                list(history), list(history_frames), list(history_boxes),
                list(history_confidences), [],
            )
        rows = sorted(
            zip(
                history_frames[-count:], history[-count:],
                history_boxes[-count:], history_confidences[-count:],
            ),
            key=lambda item: int(item[0]),
        )
        # One accepted observation per frame; preserve the latest copy if an
        # upstream path duplicated a frame during confirmation.
        by_frame = {int(row[0]): row for row in rows}
        rows = [by_frame[frame] for frame in sorted(by_frame)]
        if len(rows) < 2 or int(rows[1][0]) <= int(rows[0][0]):
            return (
                [row[1] for row in rows], [int(row[0]) for row in rows],
                [row[2] for row in rows], [float(row[3]) for row in rows], [],
            )

        raw_by_frame = {
            int(item['frame_index']): list(item.get('boxes', []))
            for item in self._raw_litter_history
        }
        recovered_frames = []
        while len(rows) < self.trajectory_history_len:
            first_frame = int(rows[0][0])
            target_frame = first_frame - 1
            candidates = raw_by_frame.get(target_frame)
            if not candidates:
                break

            second_frame = int(rows[1][0])
            first_point = np.asarray(rows[0][1], dtype=float)
            second_point = np.asarray(rows[1][1], dtype=float)
            velocity_per_frame = (
                (second_point - first_point)
                / max(float(second_frame - first_frame), 1.0)
            )
            predicted = first_point - velocity_per_frame
            first_box = np.asarray(rows[0][2], dtype=float)
            first_diagonal = float(np.hypot(
                first_box[2] - first_box[0],
                first_box[3] - first_box[1],
            ))
            ranked = []
            for candidate in candidates:
                candidate_box = np.asarray(candidate[:4], dtype=float)
                candidate_point = np.asarray([
                    (candidate_box[0] + candidate_box[2]) * 0.5,
                    (candidate_box[1] + candidate_box[3]) * 0.5,
                ])
                candidate_diagonal = float(np.hypot(
                    candidate_box[2] - candidate_box[0],
                    candidate_box[3] - candidate_box[1],
                ))
                scale = max(first_diagonal, candidate_diagonal, 1.0)
                residual = float(np.linalg.norm(candidate_point - predicted))
                if residual <= 2.0 * scale:
                    ranked.append((residual / scale, candidate, candidate_point))
            if not ranked:
                break
            _, candidate, candidate_point = min(
                ranked, key=lambda item: (item[0], -float(item[1][4]))
            )
            rows.insert(0, (
                target_frame,
                tuple(float(value) for value in candidate_point),
                tuple(float(value) for value in candidate[:4]),
                float(candidate[4]),
            ))
            recovered_frames.append(target_frame)

        recovered_frames.sort()
        return (
            [row[1] for row in rows],
            [int(row[0]) for row in rows],
            [row[2] for row in rows],
            [float(row[3]) for row in rows],
            recovered_frames,
        )

    def update(self, detected_litters, actors, person_vehicle_map=None, frame_index=None,
               frame=None, vehicle_history=None, raw_detected_litters=None):
        # 主更新流程：接收本幀通過前處理的 litter，更新軌跡與違規者集合。
        if frame_index is None:
            frame_index = self._fallback_frame_index
            self._fallback_frame_index += 1
        frame_index = int(frame_index)
        self._current_frame_index = frame_index

        self._record_raw_litter_frame(raw_detected_litters, frame_index)

        self._record_actor_frame(actors, frame_index, frame=frame)
        self._drain_backward_results(vehicle_history=vehicle_history)

        if person_vehicle_map:
            for p_id, vehicle_key_or_id in person_vehicle_map.items():
                veh_key = self._normalize_vehicle_like_key(vehicle_key_or_id)
                self.person_to_vehicle_history[int(p_id)] = veh_key
                self._record_dismount_edge(p_id, veh_key, frame_index)
        self._prune_dismount_edges(frame_index)

        for actor_key in list(self.violators.keys()):
            v_data = self.violators[actor_key]
            if (
                v_data.get('until_plate_found', False) and
                actor_key[0] in ('vehicle', 'scooter') and
                vehicle_history is not None
            ):
                plate_entry = vehicle_history.get(actor_key[1], {})
                if plate_entry.get('license_plate') is None:
                    v_data['ttl'] = max(v_data.get('ttl', 0), CONFIRMED_VIOLATOR_TTL)
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
                    CONFIRMED_SHAPE_CHANGE_RATIO
                    if prev_state == 'confirmed'
                    else PENDING_SHAPE_CHANGE_RATIO
                )
                is_shape_consistent = (w_diff_ratio <= shape_thr) and (h_diff_ratio <= shape_thr)

                # confirmed 軌跡優先保持連續，避免尺寸波動導致 ID 斷裂。
                allow_confirmed_dist_only = (
                    prev_state == 'confirmed' and dist < (self.distance_threshold * 0.7)
                )

                if dist < self.distance_threshold and (is_shape_consistent or allow_confirmed_dist_only) and dist < min_dist:
                    # 靜止鎖定的 pending litter 不吸收新偵測：讓新的偵測另開新軌跡，
                    # 避免靜止舊垃圾鎖住真正丟棄的垃圾軌跡起點。
                    if l_data.get('stationary_locked', False) and prev_state == 'pending':
                        continue
                    min_dist = dist
                    best_id = l_id

            if best_id is not None:
                # 第二段：延續既有 litter，更新 history、age、shape reference。
                l_data = self.active_litters[best_id]
                l_data['history'].append(centroid)
                if len(l_data['history']) > self.trajectory_history_len:
                    l_data['history'].pop(0)
                history_boxes = list(l_data.get('history_boxes', []))
                history_confidences = list(l_data.get('history_confidences', []))
                history_boxes.append(tuple(map(float, litter_box[:4])))
                history_confidences.append(float(litter_box[4]))
                history_boxes = history_boxes[-self.trajectory_history_len:]
                history_confidences = history_confidences[-self.trajectory_history_len:]

                age = l_data.get('age', 1) + 1
                state = l_data.get('state', 'pending')
                backward_submitted = bool(l_data.get('backward_submitted', False))
                
                # 繼承剛出生時記錄的肇事者，並在 pending 階段依 homography 座標重新評分。
                thrower_key = l_data.get('thrower_key')
                thrower_center = l_data.get('thrower_center')
                birth_thrower_key = l_data.get('birth_thrower_key')

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
                        MIN_CONFIRM_ABS_DISPLACEMENT,
                    )
                    downward_disp = float(centroid[1] - start_centroid[1])
                    is_downward_enough = downward_disp >= MIN_CONFIRM_DOWNWARD_DISPLACEMENT
                    horizontal_disp = abs(float(centroid[0] - start_centroid[0]))
                    is_horizontal_enough = horizontal_disp >= MIN_CONFIRM_HORIZONTAL_DISPLACEMENT

                    # 2. 使用軌跡物理特徵檢查（至少要有足夠歷史幀）
                    is_physics_valid = False
                    if age >= self.min_confirm_age:
                        is_physics_valid, _ = validate_trajectory(l_data['history'])

                    # 2.5 靜止舊垃圾抑制：歷史中心點最大跨距 (x/y 軸 span 取大)。
                    # 真正被丟出的垃圾在 age >= 3 時 span 通常 > 30 px；
                    # 靜止舊垃圾即使 detector 抖動也很少同時在 x/y 上拉出 > 18 px。
                    history_xs = [p[0] for p in l_data['history']]
                    history_ys = [p[1] for p in l_data['history']]
                    if len(history_xs) >= 2:
                        history_max_span = max(
                            max(history_xs) - min(history_xs),
                            max(history_ys) - min(history_ys),
                        )
                    else:
                        history_max_span = 0.0

                    # 短期靜止：在最低 confirm age 之上，但 span 仍小 → 視為靜止候選不 confirm。
                    is_static_candidate = (
                        age >= self.static_candidate_min_age and
                        history_max_span < self.min_history_span_for_confirm
                    )

                    # 長期靜止鎖：若曾連續 stationary_lock_age 幀都在小半徑內，
                    # 永久標記 stationary_locked，避免後續單一大抖動衝過 confirm。
                    if (
                        age >= self.stationary_lock_age and
                        history_max_span < self.stationary_lock_span and
                        not l_data.get('stationary_locked', False)
                    ):
                        l_data['stationary_locked'] = True
                    stationary_locked = bool(l_data.get('stationary_locked', False))

                    # 3. confirmed 條件 (litter object-event branch 唯一入口)：
                    # - 必須有明確 thrower (避免靜止舊垃圾沿用遠方 actor)
                    # - 不可為靜止候選或被靜止鎖鎖定
                    # - 軌跡符合物理特性且有向下位移
                    # - 或 有明顯位移 + 向下位移 + 具 thrower 關聯

                    # 車輛/機車 thrower 的加嚴確認條件：
                    # 分析顯示 FP 案例幾乎都是 vehicle thrower + age=2；
                    # 加嚴 min_age 與向下位移門檻，排除車輛部件或快速車輛造成的假陽性。
                    is_vehicle_thrower = (
                        thrower_key is not None and
                        thrower_key[0] in ('vehicle', 'scooter')
                    )
                    effective_min_age = (
                        self.min_confirm_age_vehicle
                        if is_vehicle_thrower
                        else self.min_confirm_age
                    )
                    effective_min_downward = (
                        self.min_confirm_downward_displacement_vehicle
                        if is_vehicle_thrower
                        else self.min_confirm_downward_displacement
                    )
                    is_downward_enough_effective = (
                        downward_disp >= effective_min_downward
                    )
                    ys_for_release = [float(p[1]) for p in l_data['history']]
                    apex_index = int(np.argmin(ys_for_release)) if ys_for_release else 0
                    fall_from_apex = (
                        float(ys_for_release[-1] - ys_for_release[apex_index])
                        if ys_for_release and apex_index < len(ys_for_release) - 1
                        else 0.0
                    )
                    descent_steps_after_apex = max(
                        len(ys_for_release) - apex_index - 1, 0
                    )
                    has_arc_descent = (
                        apex_index > 0 and
                        descent_steps_after_apex >= 2 and
                        fall_from_apex >= effective_min_downward
                    )
                    is_downward_enough_effective = (
                        is_downward_enough_effective or has_arc_descent
                    )
                    # Attribution can rank several actors later, but event
                    # confirmation itself needs release-time causal support.
                    # A passer-by acquired only after birth must not turn noise
                    # into a confirmed litter event.
                    release_actor_supported = (
                        birth_thrower_key is not None and
                        thrower_key is not None
                    )
                    # 若 thrower 為車輛，且水平位移遠大於向下位移（純水平滑動），則不 confirm。
                    is_horiz_ratio_ok = True
                    if is_vehicle_thrower and downward_disp > 0:
                        horiz_ratio = horizontal_disp / max(downward_disp, 1e-6)
                        if horiz_ratio > MAX_HORIZ_TO_DOWN_RATIO_VEHICLE:
                            is_horiz_ratio_ok = False

                    # 車輛 thrower：軌跡任意相鄰幀最大位移過大 → 車輛本體移動誤觸發，拒絕 confirm。
                    # 真實丟棄物的單步位移通常 < 80px；車輛部件可達 100-250px。
                    is_step_velocity_ok = True
                    if is_vehicle_thrower and len(l_data['history']) >= 2:
                        hist_pts = l_data['history']
                        # 以「每幀」速度判斷，而非單步絕對位移：
                        # 小快物件偵測稀疏時，單步可跨多幀（detection gap 大），位移自然大，
                        # 但每幀速度仍小且物理合理；車身瞬移則每幀速度大。除以幀距即可區分。
                        _frames_aligned = list(l_data.get('history_frames', [])) + [frame_index]
                        _m = min(len(hist_pts), len(_frames_aligned))
                        _pts = hist_pts[-_m:]
                        _frs = _frames_aligned[-_m:]
                        _max_step_per_frame = 0.0
                        for i in range(_m - 1):
                            _gap = max(int(_frs[i + 1]) - int(_frs[i]), 1)
                            _step = distance.euclidean(_pts[i], _pts[i + 1]) / _gap
                            if _step > _max_step_per_frame:
                                _max_step_per_frame = _step
                        if _max_step_per_frame > self.max_vehicle_thrower_step_px:
                            is_step_velocity_ok = False

                    # 車身/貨物誤判抑制（以「相對載體車輛的分離」判別，而非靜態重疊）：
                    # 靜態重疊無法區分「被丟出但仍與車重疊的垃圾」與「車身部件」——兩者重疊都可能很高。
                    # 真正的判別是「是否相對車輛分離」：
                    #   - 車身部件 / 貨物 / 車牌 / 後照鏡誤判 → 隨車移動，相對車輛淨位移 ≈ 0。
                    #   - 被丟出的垃圾 → 脫離車輛 (落地/落後)，相對車輛淨位移大。
                    # 僅當 litter 明顯疊在某車輛上 (載體) 時才檢查；落在草地等無載體者不受影響。
                    vehicle_relative_ok = True
                    carrier_key, vehicle_body_overlap, carrier_center = self._carrier_vehicle(
                        litter_box, actors,
                    )
                    vehicle_rel_sep = None
                    if carrier_key is not None and vehicle_body_overlap >= CARRIER_OVERLAP_MIN:
                        vehicle_rel_sep = self._litter_vehicle_separation(
                            carrier_key,
                            carrier_center,
                            l_data.get('birth_centroid', l_data['history'][0]),
                            l_data.get('birth_frame', frame_index),
                            centroid,
                            frame_index,
                        )
                        if (
                            vehicle_rel_sep is not None and
                            vehicle_rel_sep < MIN_VEHICLE_RELATIVE_SEPARATION
                        ):
                            vehicle_relative_ok = False

                    can_confirm_by_trajectory = (
                        age >= effective_min_age and
                        is_physics_valid and
                        is_downward_enough_effective and
                        is_horizontal_enough and
                        is_horiz_ratio_ok and
                        is_step_velocity_ok and
                        vehicle_relative_ok and
                        release_actor_supported and
                        not is_static_candidate and
                        not stationary_locked
                    )
                    can_confirm_by_motion = (
                        age >= effective_min_age and
                        is_moved_enough and
                        is_downward_enough_effective and
                        is_horizontal_enough and
                        is_horiz_ratio_ok and
                        is_step_velocity_ok and
                        vehicle_relative_ok and
                        release_actor_supported and
                        not is_static_candidate and
                        not stationary_locked
                    )
                    # Fall-then-stable：軌跡曾有 cy 下降（落下）+ 後段穩定（落地不動）
                    # 用於 driver-throw 物體：拋擲時 detector 抓到上升 + 落下段非連續，
                    # 但落地後 stable 段是穩定可靠證據。
                    ys = [float(p[1]) for p in l_data['history']]
                    fall_disp_history = 0.0
                    fall_stable_tail_ok = False
                    if len(ys) >= self.fall_stable_tail_window + 2:
                        peak_y = min(ys)
                        peak_idx = ys.index(peak_y)
                        if peak_idx < len(ys) - 1:
                            max_after_peak = max(ys[peak_idx:])
                            fall_disp_history = max_after_peak - peak_y
                        tail = ys[-self.fall_stable_tail_window:]
                        fall_stable_tail_ok = (max(tail) - min(tail)) <= FALL_STABLE_MAX_TAIL_JITTER
                    can_confirm_fall_then_stable = (
                        is_vehicle_thrower and
                        age >= self.fall_stable_min_age and
                        fall_disp_history >= FALL_STABLE_MIN_FALL_DISP and
                        fall_stable_tail_ok and
                        is_step_velocity_ok and
                        vehicle_relative_ok and
                        release_actor_supported and
                        not stationary_locked
                    )

                    # 快速落下特例（age=2 vehicle thrower）：10fps 高速場景中 litter 可能只出現 2 幀。
                    # gap <= 2：兩次偵測必須連續幀，排除「birth 後 holding/missed 再跳到遠處新偵測」的 ID 碰撞。
                    # release_from_thrower：起點貼近車、終點已離開車身 → 排除車邊緣 detector jitter。
                    # horiz/down ratio >= 0.15：真實丟擲都有橫向分量；近乎垂直下落 (ratio < 0.15)
                    #   多為車邊緣/車燈/告示牌等靜態物 detector jitter 在垂直方向偵測到不同位置。
                    _fast_history_frames = l_data.get('history_frames', [])
                    _fast_prev_fi = int(_fast_history_frames[-1]) if _fast_history_frames else frame_index
                    fast_drop_frame_gap = int(frame_index) - _fast_prev_fi
                    fast_drop_release_ok = False
                    if is_vehicle_thrower and thrower_key is not None:
                        for _actor in actors:
                            if self._actor_key(_actor) == thrower_key:
                                fast_drop_release_ok = self._release_origin_near_actor(
                                    l_data['history'], _actor,
                                )
                                break
                    fast_drop_horiz_ratio_ok = (
                        downward_disp > 0 and
                        (horizontal_disp / downward_disp) >= 0.15
                    )
                    actorless_fast_drop_supported = (
                        birth_thrower_key is None and
                        thrower_key is not None and
                        is_vehicle_thrower and
                        downward_disp > 0 and
                        (horizontal_disp / downward_disp)
                        <= FAST_DROP_ACTORLESS_MAX_HORIZ_RATIO
                    )
                    can_confirm_vehicle_fast_drop = (
                        is_vehicle_thrower and
                        age == self.min_confirm_age and
                        fast_drop_frame_gap <= self.fast_drop_max_frame_gap and
                        fast_drop_release_ok and
                        fast_drop_horiz_ratio_ok and
                        is_physics_valid and
                        downward_disp >= FAST_DROP_MIN_DOWNWARD and
                        is_horizontal_enough and
                        is_horiz_ratio_ok and
                        is_step_velocity_ok and
                        vehicle_relative_ok and
                        (release_actor_supported or actorless_fast_drop_supported) and
                        not is_static_candidate and
                        not stationary_locked
                    )

                    # 隨車部件 / 純水平條紋 / 共動 等 FP 篩選已移至 detect 前處理
                    # （litter_candidate_is_vehicle_fp）。tracker 只負責追蹤與軌跡確認，
                    # 不再對候選做 FP「懷疑」；能進到這裡的都是前處理放行的候選。
                    if getattr(self, '_debug', False) and age >= 2:
                        print(
                            f"  [TRK fi={frame_index} lid={best_id} age={age}] "
                            f"span={history_max_span:.1f} moved={moved_dist:.1f} "
                            f"down={downward_disp:.1f} horiz={horizontal_disp:.1f} "
                            f"thrower={thrower_key} birth_thrower={birth_thrower_key} "
                            f"release_actor_ok={release_actor_supported} veh={is_vehicle_thrower} "
                            f"eff_age={effective_min_age} eff_down={effective_min_downward:.0f} "
                            f"horiz_ok={is_horiz_ratio_ok} step_ok={is_step_velocity_ok} "
                            f"body_ok={vehicle_relative_ok} carrier={carrier_key} ov={vehicle_body_overlap:.2f} rel_sep={vehicle_rel_sep} "
                            f"static={is_static_candidate} locked={stationary_locked} "
                            f"traj={is_physics_valid} "
                            f"can_traj={can_confirm_by_trajectory} can_mot={can_confirm_by_motion} "
                            f"can_fast={can_confirm_vehicle_fast_drop} actorless_fast={actorless_fast_drop_supported} "
                            f"gap={fast_drop_frame_gap} rel_ok={fast_drop_release_ok} "
                            f"can_fs={can_confirm_fall_then_stable} fall={fall_disp_history:.0f}"
                        )

                    if (
                        can_confirm_by_trajectory or can_confirm_by_motion or
                        can_confirm_vehicle_fast_drop or can_confirm_fall_then_stable
                    ):
                        state = 'confirmed' # 確認為垃圾！
                        if self._pv_assoc_enabled and best_id not in self._pv_litter_seen_ids:
                            # offline 關聯的硬錨定:confirmed litter 當幀中心(每 litter id 記一次)。
                            self._pv_litter_seen_ids.add(best_id)
                            self._pv_litter_events.append({
                                'frame_index': int(frame_index),
                                'center': tuple(centroid),
                            })
                        # 違規升級門檻：開啟 require_vehicle 時，純行人（無車輛關聯）thrower 不升級
                        # 為違規——不做 backtrack/車牌、不標 violator，藉此壓制行人/小便場景的 FP。
                        escalate_violation = (
                            not self.require_vehicle_for_violation or
                            self._thrower_has_vehicle(thrower_key)
                        )
                        if not escalate_violation and getattr(self, '_debug', False):
                            print(f"  [VIOLATION_SKIP_NO_VEHICLE fi={frame_index} litter={best_id} thrower={thrower_key}]")
                        if best_id not in self._litter_event_seen_ids:
                            # 首次確認:記一筆 litter 事件(events.jsonl 來源)。
                            self._litter_event_seen_ids.add(best_id)
                            _lx1, _ly1, _lx2, _ly2 = (int(v) for v in litter_box[:4])
                            event = {
                                'litter_id': int(best_id),
                                'frame_index': int(frame_index),
                                'birth_frame': int(l_data.get('birth_frame', frame_index)),
                                'confirm_frame': int(frame_index),
                                'bbox': [_lx1, _ly1, _lx2, _ly2],
                                'center': [float(centroid[0]), float(centroid[1])],
                                'detector_confidence': (
                                    float(litter_box[4]) if len(litter_box) > 4 else None
                                ),
                                'thrower_key': list(thrower_key) if thrower_key is not None else None,
                                'vehicle_key': None,
                                'escalated': bool(escalate_violation),
                                'backtrack_status': (
                                    'pending' if self._smart_backtrack_enabled else 'legacy'
                                ),
                            }
                            self._litter_events.append(event)
                            self._litter_events_by_id[int(best_id)] = event
                        # Smart resolver 必須看所有 confirmed litter；舊的 require_vehicle
                        # 只能決定最後是否升級，不能在推理前把候選事件擋掉。
                        if (
                            (self._smart_backtrack_enabled or escalate_violation)
                            and not backward_submitted
                        ):
                            backward_submitted = self._submit_backward_resolution(
                                litter_id=best_id,
                                litter_data=l_data,
                                current_bbox=litter_box,
                                current_centroid=centroid,
                                confirm_frame=frame_index,
                                prev_thrower_key=thrower_key,
                            )

                        if (
                            not self._smart_backtrack_enabled
                            and escalate_violation
                            and thrower_key is not None
                        ):
                            # confirmed 後標記 thrower；若該人綁定車輛，也同步標記車輛。
                            current_actor_center = thrower_center # 預設為舊位置
                            
                            # 去目前的 actors 裡面找他現在開到哪了
                            for actor in actors:
                                if self._actor_key(actor) == thrower_key:
                                    current_actor_center = self._actor_center(actor)
                                    break
                                    
                            self._mark_violator(
                                thrower_key,
                                current_actor_center,
                                ttl=CONFIRMED_VIOLATOR_TTL,
                                action='littering',
                            )

                            cls_name, track_id = thrower_key
                            veh_key = self._bound_vehicle_for_person(track_id) if cls_name == 'person' else None
                            if veh_key is not None:
                                # 去當前畫面找車輛中心點
                                veh_center = current_actor_center
                                for actor in actors:
                                    if self._actor_key(actor) == veh_key:
                                        veh_center = self._actor_center(actor)
                                        break
                                        
                                # 同時將該車輛標記為違規！(讓畫面畫紅框並抓取車牌)
                                self._mark_violator(
                                    veh_key,
                                    veh_center,
                                    ttl=CONFIRMED_VIOLATOR_TTL,
                                    action='littering',
                                )
                
                new_active_litters[best_id] = {
                    'bbox': litter_box,
                    'history': l_data['history'],
                    'missed': 0,
                    'thrower_key': thrower_key,
                    'birth_thrower_key': birth_thrower_key,
                    'thrower_center': thrower_center,
                    'age': age,
                    'state': state,
                    'init_shape': l_data['init_shape'],
                    'birth_frame': l_data.get('birth_frame', frame_index),
                    'birth_centroid': l_data.get('birth_centroid', l_data['history'][0]),
                    'birth_bbox': l_data.get('birth_bbox', l_data.get('bbox')),
                    'history_frames': (l_data.get('history_frames', []) + [frame_index])[-TRAJECTORY_HISTORY_LEN:],
                    'history_boxes': history_boxes,
                    'history_confidences': history_confidences,
                    # Queue 滿或 worker 尚未接受時保持 False，下一次 update 可重試。
                    'backward_submitted': backward_submitted,
                    'backward_result': l_data.get('backward_result'),
                    'stationary_locked': bool(l_data.get('stationary_locked', False)),
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

                # 與既有 stationary_locked 軌跡共位 → 繼承鎖定，避免靜止點生出可誤 confirm 的雙胞胎。
                inherit_locked = False
                for _l_data in self.active_litters.values():
                    if not _l_data.get('stationary_locked', False):
                        continue
                    _pbox = _l_data['bbox']
                    _pc = ((_pbox[0] + _pbox[2]) / 2.0, (_pbox[1] + _pbox[3]) / 2.0)
                    if distance.euclidean(centroid, _pc) <= LOCK_INHERIT_RADIUS:
                        inherit_locked = True
                        break

                litter_id = self.next_id
                new_active_litters[litter_id] = {
                    'bbox': litter_box,
                    'history': [centroid],
                    'missed': 0,
                    'thrower_key': thrower_key,        # 紀錄嫌疑犯
                    'birth_thrower_key': thrower_key,  # release-time causal anchor
                    'thrower_center': thrower_center,
                    'age': 1,
                    'state': 'pending',
                    'init_shape': (curr_w, curr_h),
                    'ref_shape': (curr_w, curr_h),
                    'birth_frame': frame_index,
                    'birth_centroid': centroid,
                    'birth_bbox': litter_box,
                    'history_frames': [frame_index],
                    'history_boxes': [tuple(map(float, litter_box[:4]))],
                    'history_confidences': [float(litter_box[4])],
                    'backward_submitted': False,
                    'backward_result': None,
                    'stationary_locked': inherit_locked,
                }

                self.next_id += 1
        # 第四段：處理本幀沒被配對到的舊 litter；短暫消失可保留，超過門檻移除。
        for l_id, l_data in self.active_litters.items():
            l_data['missed'] += 1
            if l_data['missed'] < self.max_missed_frames:
                new_active_litters[l_id] = l_data
        
        self.active_litters = new_active_litters
        self._drain_backward_results(vehicle_history=vehicle_history)

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

                if jump_dist <= MAX_VIOLATOR_JUMP:
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
                    'ttl': v_data.get('ttl', CONFIRMED_VIOLATOR_TTL),
                    'center': rebound_center,
                    'missed': 0,
                    'until_plate_found': bool(v_data.get('until_plate_found', False)),
                    'action': v_data.get('action'),
                }
                self.violators[rebound_key] = rebound_data
                if rebound_key != violator_key:
                    del self.violators[violator_key]

                active_violator_keys.add(rebound_key)
                occupied_actor_keys.add(rebound_key)
                continue

            # 3) 本幀找不到可延續對象：累積 miss，超過門檻才釋放
            v_data['missed'] = v_data.get('missed', 0) + 1
            if v_data['missed'] > VIOLATOR_MAX_MISSED:
                del self.violators[violator_key]

        return self.active_litters, active_violator_keys

    def close(self, timeout=2.0):
        # idempotent：正常 EOF 會先 finalize；例外路徑則由 close 負責兜底。
        if self._closed:
            return
        self.finalize_backtracking(timeout=timeout)
        self._closed = True

    def finalize_backtracking(self, vehicle_history=None, timeout=8.0):
        """Flush worker, solve final event-expanded MCF, then update events.

        必須在 summary/events 寫檔之前呼叫，否則影片尾端的 attribution
        仍停留在 provisional thrower。
        """
        if hasattr(self, '_final_backtrack_summary'):
            self._drain_backward_results(vehicle_history=vehicle_history)
            return dict(self._final_backtrack_summary)

        self._backward_accepting = False
        deadline = time.monotonic() + max(float(timeout), 0.0)
        while (
            getattr(self._backward_tasks, 'unfinished_tasks', 0) > 0
            and time.monotonic() < deadline
        ):
            self._drain_backward_results(vehicle_history=vehicle_history)
            time.sleep(0.01)

        if not self._backward_stop_sent:
            try:
                remaining = max(0.01, deadline - time.monotonic())
                self._backward_tasks.put(
                    self._backward_stop,
                    timeout=min(remaining, 0.5),
                )
                self._backward_stop_sent = True
            except queue.Full:
                pass
        if self._backward_thread.is_alive():
            self._backward_thread.join(
                timeout=max(0.0, deadline - time.monotonic())
            )
        self._drain_backward_results(vehicle_history=vehicle_history)

        # Final authority: all completed per-event candidate tables are solved
        # together. Actor capacities remain unlimited by design (one car/many people).
        final_resolutions = {}
        if self._smart_resolver is not None and self._smart_candidate_tables:
            try:
                final_resolutions = self._smart_resolver.solve_routes(
                    self._smart_candidate_tables
                )
            except Exception as exc:  # noqa: BLE001
                if self._debug:
                    print(f"  [SMART_BACKTRACK final-flow failed: {exc}]")
        for litter_id, resolution in final_resolutions.items():
            task = self._smart_tasks.get(int(litter_id))
            if task is None:
                continue
            task = dict(task)
            task['revision'] = max(
                int(task.get('revision', 0)),
                int(self._applied_backtrack_revisions.get(int(litter_id), 0)),
            ) + 1
            result = self._smart_resolution_result(resolution, task)
            self._apply_backward_result(result, vehicle_history=vehicle_history)

        statuses = [
            str(event.get('backtrack_status', 'pending'))
            for event in self._litter_events
        ]
        summary = {
            'enabled': bool(self._smart_backtrack_enabled),
            'solver': 'kalman_rts_reverse_trajectory_min_cost_flow',
            'events': len(self._litter_events),
            'candidate_tables': len(self._smart_candidate_tables),
            'resolved': sum(status == 'resolved' for status in statuses),
            'dustbin': sum(status == 'dustbin' for status in statuses),
            'legacy': sum(status == 'legacy' for status in statuses),
            'pending': sum(status == 'pending' for status in statuses),
            'worker_flushed': not self._backward_thread.is_alive(),
            'person_capacity': 'unbounded',
            'vehicle_capacity': 'unbounded',
        }
        # A timeout is observable, not a permanent final state. Do not cache it:
        # a second finalize/close call can still join, drain and re-solve.
        if summary['worker_flushed']:
            self._final_backtrack_summary = summary
        return dict(summary)

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

    def get_litter_events(self):
        # confirmed litter 事件(每 litter id 一筆,首次確認時記錄)。events.jsonl 來源。
        return list(self._litter_events)

    def get_backtrack_candidate_records(
        self,
        input_video,
        output_video=None,
    ):
        """Return JSON-safe research records for every confirmed litter event.

        Candidate diagnostics intentionally live in a separate sidecar rather
        than the frontend events JSONL.  Include incomplete/pending events too,
        otherwise candidate-generation failures disappear from evaluation.
        Call after ``finalize_backtracking()`` for authoritative assignments.
        """

        from pipeline.backtrack.sidecar import build_candidate_record

        records = []
        for event in sorted(
            self._litter_events,
            key=lambda item: (
                int(item.get('frame_index', 0)),
                int(item.get('litter_id', -1)),
            ),
        ):
            litter_id = int(event.get('litter_id', -1))
            records.append(
                build_candidate_record(
                    task=self._smart_tasks.get(litter_id, {}),
                    event=event,
                    routes=self._smart_candidate_tables.get(litter_id, []),
                    input_video=input_video,
                    output_video=output_video,
                )
            )
        return records

    def finalize_associations(self):
        """收尾:對全片輕量歷史跑 offline Person↔Vehicle 關聯(event-anchored + Hungarian + dustbin)。
        回傳 JSON-serializable dict;PV_ASSOC 關閉或無資料時回 None。主迴圈結束後呼叫一次。"""
        # Smart mode 的 Hungarian 僅限 detector frame↔frame identity。
        # 舊 P↔V 1-to-1 Hungarian 不得覆蓋 many-to-many flow assignment。
        if self._smart_backtrack_enabled:
            return None
        if not self._pv_assoc_enabled or not self._pv_full_history:
            return None

        import sys
        from pipeline.paths import REPO_ROOT
        # person_vehicle_assoc.py 位於 repo root(路徑集中於 pipeline.paths)。
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        try:
            from python_tools.person_vehicle_assoc import associate, AssocConfig, LitterEvent
        except Exception as exc:  # noqa: BLE001 — 缺模組不該讓整段 run 失敗
            print(f"[PV_ASSOC] import failed, skip association: {exc}")
            return None

        cfg = AssocConfig(fps=self.fps)
        tau_env = os.environ.get("PV_ASSOC_TAU")
        if tau_env:
            try:
                cfg.tau = float(tau_env)
            except ValueError:
                pass

        litter_events = [
            LitterEvent(frame_index=int(e['frame_index']), center=tuple(e['center']))
            for e in self._pv_litter_events
        ]
        result = associate(self._pv_full_history, litter_events, cfg)

        bindings = []
        for pkey, b in result.bindings.items():
            bindings.append({
                'person_track_id': int(pkey[1]),
                'vehicle_cls': b.vehicle_key[0] if b.vehicle_key else None,
                'vehicle_track_id': int(b.vehicle_key[1]) if b.vehicle_key else None,
                'score': round(float(b.score), 4),
                'birth_frame': int(b.birth_frame),
                'death_frame': int(b.death_frame),
                'ttl_until_frame': int(b.ttl_until_frame),
            })
        bindings.sort(key=lambda d: d['person_track_id'])
        n_bound = sum(1 for d in bindings if d['vehicle_track_id'] is not None)
        return {
            'frames_accumulated': len(self._pv_full_history),
            'litter_events': len(litter_events),
            'persons': len(result.person_keys),
            'confirmed_vehicles': len(result.vehicle_keys),
            'bound_persons': n_bound,
            'unbound_persons': len(bindings) - n_bound,
            'bindings': bindings,
        }

    def _record_actor_frame(self, actors, frame_index, frame=None):
        # 每幀保留 actor 快照。confirmed 延遲出現時，backward worker 可回看出生幀附近。
        actor_snapshots = []
        frame_h = frame.shape[0] if frame is not None else 0
        frame_w = frame.shape[1] if frame is not None else 0
        bev_bottoms = []   # 本幀車輛/機車底邊中心 + 高度，供 BEV 穩定平面累積。

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
                'actor_key': (cls_name, track_id),
                'box': box,
                'center': self._box_center(box),
                'footpoint': ((box[0] + box[2]) / 2.0, box[3]),
                'confidence': float(
                    actor.get('confidence', actor.get('pose_conf', actor.get('conf', 1.0)))
                ),
                'observed': bool(actor.get('observed', True)),
                'source': str(actor.get('source', 'detector')),
                'frame_index': int(frame_index),
                'frame_size': (int(frame_w), int(frame_h)),
            }

            # Track ID 可能被外部 tracker 回收。長時間中斷後重現時增加 epoch，
            # 成本層可用 tracklet_uid 區分物理上不同的軌跡。
            actor_key = (cls_name, track_id)
            uid_state = self._actor_tracklet_epochs.get(actor_key)
            gap_limit = max(int(round(self.fps * 2.0)), 2)
            if uid_state is None:
                uid_state = {'epoch': 0, 'last_frame': int(frame_index)}
            elif int(frame_index) - int(uid_state['last_frame']) > gap_limit:
                uid_state = {
                    'epoch': int(uid_state.get('epoch', 0)) + 1,
                    'last_frame': int(frame_index),
                }
            else:
                uid_state['last_frame'] = int(frame_index)
            self._actor_tracklet_epochs[actor_key] = uid_state
            snapshot['tracklet_uid'] = "{}:{}:{}".format(
                cls_name, track_id, int(uid_state['epoch'])
            )

            if frame is not None and cls_name in ('vehicle', 'scooter'):
                roi = self._crop_actor_roi(frame, box, frame_w, frame_h)
                if roi is not None:
                    snapshot['plate_roi'] = roi

            if self._bev_stable_enabled and cls_name in ('vehicle', 'scooter'):
                bx1, by1, bx2, by2 = box
                bev_bottoms.append((((bx1 + bx2) / 2.0, by2), max(by2 - by1, 1.0)))

            actor_snapshots.append(snapshot)

        with self._actor_history_lock:
            self.actor_frame_history.append({
                'frame_index': int(frame_index),
                'actors': actor_snapshots,
            })
            self._smart_actor_history.append({
                'frame_index': int(frame_index),
                'actors': [
                    {
                        key: value
                        for key, value in snapshot.items()
                        if key != 'plate_roi'
                    }
                    for snapshot in actor_snapshots
                ],
            })

        if self._pv_assoc_enabled:
            # 全片不截斷的輕量歷史(去 plate_roi 省記憶);主執行緒 only,finalize 在 join 後讀。
            self._pv_full_history.append({
                'frame_index': int(frame_index),
                'actors': [
                    {'cls': s['cls'], 'track_id': s['track_id'],
                     'box': s['box'], 'center': s['center'],
                     'confidence': s['confidence'], 'observed': s['observed'],
                     'source': s['source'], 'tracklet_uid': s['tracklet_uid']}
                    for s in actor_snapshots
                ],
            })

        if bev_bottoms:
            with self._bev_lock:
                for bottom, height in bev_bottoms:
                    self._bev_bottoms.append(bottom)
                    self._bev_heights.append(height)

    def _submit_backward_resolution(self, litter_id, litter_data, current_bbox,
                                    current_centroid, confirm_frame, prev_thrower_key=None):
        # task 用 immutable snapshot，worker 不碰 main thread 追蹤狀態。
        if not self._backward_accepting:
            return False
        birth_frame = int(litter_data.get('birth_frame', confirm_frame))
        birth_centroid = tuple(litter_data.get('birth_centroid', litter_data['history'][0]))
        birth_bbox = litter_data.get('birth_bbox', litter_data.get('bbox', current_bbox))
        history = [tuple(p) for p in litter_data.get('history', [])]
        history_frames = [int(f) for f in litter_data.get('history_frames', [])]
        history_boxes = [
            tuple(map(float, box[:4]))
            for box in litter_data.get('history_boxes', [])
        ]
        history_confidences = [
            float(value) for value in litter_data.get('history_confidences', [])
        ]
        if len(history_frames) == len(history) - 1:
            # confirm 發生在 update 迴圈中段:history 已 append 本幀質心(L343),但
            # history_frames 要到幀尾 dict 重建才補 → 這裡用 confirm_frame 補齊,
            # 否則尾端對齊會整體錯位一幀(軌跡擬合的時間座標被污染)。
            history_frames.append(int(confirm_frame))
        if not history or history[-1] != tuple(current_centroid):
            history.append(tuple(current_centroid))
            history_frames.append(int(confirm_frame))
        if len(history_boxes) == len(history) - 1:
            history_boxes.append(tuple(map(float, current_bbox[:4])))
        if len(history_confidences) == len(history) - 1:
            history_confidences.append(
                float(current_bbox[4]) if len(current_bbox) > 4 else 1.0
            )

        accepted_history_frames = list(history_frames)
        recovered_raw_frames = []
        if self._raw_prefix_enabled:
            (
                history,
                history_frames,
                history_boxes,
                history_confidences,
                recovered_raw_frames,
            ) = self._recover_raw_litter_prefix(
                history,
                history_frames,
                history_boxes,
                history_confidences,
            )

        with self._actor_history_lock:
            history_source = (
                self._smart_actor_history
                if self._smart_backtrack_enabled
                else self.actor_frame_history
            )
            pre_birth_frames = (
                self._smart_context_frames
                if self._smart_backtrack_enabled
                else BACKWARD_PRE_BIRTH_FRAMES
            )
            actor_frames = [
                {
                    'frame_index': item['frame_index'],
                    'actors': [dict(actor) for actor in item.get('actors', [])],
                }
                for item in history_source
                if (
                    birth_frame - pre_birth_frames
                    <= int(item.get('frame_index', -1))
                    <= int(confirm_frame) + BACKWARD_POST_BIRTH_FRAMES
                )
            ]
            plate_actor_frames = [
                {
                    'frame_index': item['frame_index'],
                    'actors': [dict(actor) for actor in item.get('actors', [])],
                }
                for item in self.actor_frame_history
                if (
                    birth_frame - BACKWARD_PRE_BIRTH_FRAMES
                    <= int(item.get('frame_index', -1))
                    <= int(confirm_frame) + BACKWARD_POST_BIRTH_FRAMES
                )
            ]

        if not actor_frames:
            return False

        revision = int(self._backtrack_revisions.get(int(litter_id), 0)) + 1
        recovered_raw_frame_set = set(recovered_raw_frames)
        task = {
            'schema_version': 1,
            'litter_id': int(litter_id),
            'revision': revision,
            'fps': float(self.fps),
            'birth_frame': birth_frame,
            'confirm_frame': int(confirm_frame),
            'birth_centroid': birth_centroid,
            'birth_bbox': birth_bbox,
            'current_bbox': current_bbox,
            'current_centroid': tuple(current_centroid),
            'history': history,
            'history_frames': history_frames,
            'history_boxes': history_boxes,
            'history_confidences': history_confidences,
            'history_sources': [
                (
                    'raw_rtdetr_recovered'
                    if int(frame) in recovered_raw_frame_set
                    else 'accepted_tracker'
                )
                for frame in history_frames
            ],
            'accepted_history_frames': accepted_history_frames,
            'recovered_raw_history_frames': recovered_raw_frames,
            'prev_thrower_key': prev_thrower_key,
            'actor_frames': actor_frames,
            'plate_actor_frames': plate_actor_frames,
        }

        try:
            self._backward_tasks.put_nowait(task)
            self._backtrack_revisions[int(litter_id)] = revision
            if self._smart_backtrack_enabled:
                self._smart_tasks[int(litter_id)] = task
            return True
        except queue.Full:
            # confirmed task 不可永久遺失。飽和是罕見情況，主執行緒同步
            # fallback 一次，比把事件永遠標成 submitted 更安全。
            try:
                result = self._resolve_backward_task(task)
                if result is not None:
                    self._backward_results.put(result)
                self._backtrack_revisions[int(litter_id)] = revision
                if self._smart_backtrack_enabled:
                    self._smart_tasks[int(litter_id)] = task
                return True
            except Exception:
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
        """Run smart attribution first; legacy heuristic is an exception fallback."""
        if self._smart_resolver is not None:
            try:
                resolution = self._smart_resolver.resolve_task(task)
                return self._smart_resolution_result(resolution, task)
            except Exception as exc:  # noqa: BLE001 - keep production pipeline alive
                if self._debug:
                    print(
                        f"  [SMART_BACKTRACK litter={task.get('litter_id')} "
                        f"fallback={type(exc).__name__}: {exc}]"
                    )
        legacy_task = dict(task)
        legacy_task['actor_frames'] = task.get(
            'plate_actor_frames', task.get('actor_frames', [])
        )
        result = self._resolve_backward_task_legacy(legacy_task)
        if result is not None:
            actor_key = result.get('actor_key')
            plate_key = result.get('plate_key')
            if actor_key is not None:
                actor_key = tuple(actor_key)
            if plate_key is not None:
                plate_key = tuple(plate_key)
            if actor_key is not None and actor_key[0] == 'person':
                result['person_key'] = actor_key
                result['vehicle_key'] = plate_key
                result['direct_vehicle'] = False
                result['route_type'] = (
                    'person_vehicle' if plate_key is not None else 'person'
                )
            elif actor_key is not None and actor_key[0] in ('vehicle', 'scooter'):
                result['person_key'] = None
                result['vehicle_key'] = actor_key
                result['direct_vehicle'] = True
                result['route_type'] = 'direct_vehicle'
            result['revision'] = int(task.get('revision', 0))
            result['status'] = 'legacy'
        return result

    def _smart_resolution_result(self, resolution, task):
        person_key = resolution.person_key
        vehicle_key = resolution.vehicle_key
        actor_key = resolution.actor_key
        release_frame = (
            int(resolution.release_frame)
            if resolution.release_frame is not None
            else int(task.get('birth_frame', 0))
        )

        def _center_near(key):
            if key is None:
                return None
            frames = sorted(
                task.get('actor_frames', []),
                key=lambda item: abs(
                    int(item.get('frame_index', release_frame)) - release_frame
                ),
            )
            for frame_snapshot in frames:
                center = self._snapshot_center_for_key(
                    key, frame_snapshot.get('actors', [])
                )
                if center is not None:
                    return center
            return None

        mark_items = []
        if person_key is not None:
            mark_items.append({
                'actor_key': person_key,
                'center': _center_near(person_key),
            })
        if vehicle_key is not None:
            mark_items.append({
                'actor_key': vehicle_key,
                'center': _center_near(vehicle_key),
            })
        plate_roi_items = (
            self._plate_roi_items_for_key(
                vehicle_key,
                task.get('plate_actor_frames', task.get('actor_frames', [])),
                int(task.get('birth_frame', release_frame)),
            )
            if vehicle_key is not None
            else []
        )
        return {
            'litter_id': int(task.get('litter_id', resolution.litter_id)),
            'revision': int(task.get('revision', 0)),
            'status': (
                'dustbin' if actor_key is None and vehicle_key is None else 'resolved'
            ),
            'actor_key': actor_key,
            'person_key': person_key,
            'vehicle_key': vehicle_key,
            'direct_vehicle': bool(resolution.direct_vehicle),
            'score': float(resolution.total_cost),
            'margin_to_second': resolution.margin_to_second,
            'actor_margins': dict(resolution.actor_margins),
            'route_type': str(resolution.route_type),
            'route_id': str(resolution.route_id),
            'release_frame': resolution.release_frame,
            'release_point': resolution.release_point,
            'release_covariance': resolution.release_covariance,
            'components': dict(resolution.components),
            'route_candidates': list(resolution.routes),
            'birth_frame': int(task.get('birth_frame', 0)),
            'confirm_frame': int(task.get('confirm_frame', task.get('birth_frame', 0))),
            'mark_items': mark_items,
            'plate_key': vehicle_key,
            'plate_roi_items': plate_roi_items,
            'plate_blocked_since_litter': (
                vehicle_key is not None and not plate_roi_items
            ),
        }

    def _resolve_backward_task_legacy(self, task):
        birth_ref = task.get('birth_bbox')
        if birth_ref is None:
            birth_ref = task.get('birth_centroid')
        birth_anchor = self._litter_ground_anchor(birth_ref)
        if birth_anchor is None:
            return None

        # 取得最後確認時的地面點，作為真實世界座標的基準 (避免空中的 birth_anchor 投影出錯)
        ground_ref = task.get('current_bbox')
        if ground_ref is None:
            ground_ref = task.get('current_centroid')
        ground_anchor = self._litter_ground_anchor(ground_ref)
        if ground_anchor is None:
            ground_anchor = birth_anchor

        history = task.get('history') or []
        prev_thrower_key = task.get('prev_thrower_key')
        birth_frame = int(task.get('birth_frame', 0))
        actor_candidates = None
        if self._trajfit_enabled:
            # 軌跡反演優先;擬合失敗或無時空相交候選 → None → 走既有落點評分 fallback。
            actor_candidates = self._trajfit_actor_candidates(task)
        fallback_frames = task.get('actor_frames', []) if actor_candidates is None else []
        if actor_candidates is None:
            actor_candidates = {}

        for frame_snapshot in fallback_frames:
            frame_index = int(frame_snapshot.get('frame_index', birth_frame))
            actors = []
            for actor_snapshot in frame_snapshot.get('actors', []):
                actor = self._snapshot_to_actor(actor_snapshot)
                if actor is not None:
                    actors.append(actor)
            if not actors:
                continue

            homography = self._estimate_ground_homography(actors, ground_anchor)
            ground_world = self._project_point(ground_anchor, homography)
            start_2d = history[0] if history else None
            end_2d = history[-1] if history else None

            for actor_snapshot in frame_snapshot.get('actors', []):
                actor = self._snapshot_to_actor(actor_snapshot)
                if actor is None:
                    continue

                score, release_like = self._score_backward_actor(
                    actor=actor,
                    birth_anchor=birth_anchor,
                    ground_world=ground_world,
                    start_2d=start_2d,
                    end_2d=end_2d,
                    homography=homography,
                    history=history,
                    frame_index=frame_index,
                    birth_frame=birth_frame,
                    prev_thrower_key=prev_thrower_key,
                )
                if score is None:
                    continue
                if score > BACKWARD_SCORE_LIMIT and not (
                    release_like and score <= BACKWARD_RELEASE_SCORE_LIMIT
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

    def _trajfit_actor_candidates(self, task):
        # 軌跡反演歸因:litter 空中段 → 拋物線擬合 → 向後外插釋放事件 → 時空相交評分。
        # 只掃 [birth - max_back, birth]:釋放必在首次偵測之前;birth 之後軌跡跟的是垃圾不是人。
        # 評分= 外插點到 actor bbox 距離 / margin(margin 由 bbox 尺寸與擬合 sigma 縮放,無常數 px 門檻)。
        # 回傳與既有評分同構的 candidates dict;無法反演回 None(由呼叫端 fallback)。
        history = task.get('history') or []
        hframes = task.get('history_frames') or []
        if len(history) < 2 or len(hframes) < 2:
            if self._debug:
                print(f"  [TRAJFIT litter={task.get('litter_id')} skip: history={len(history)}/{len(hframes)}]")
            return None
        pts, fs = _trajfit_airborne_prefix(history, hframes)
        fit = _trajfit_fit_ballistic(
            pts, fs,
            sigma_floor=self._trajfit_sigma_floor,
            min_pts=self._trajfit_min_air_pts,
            min_speed=self._trajfit_min_speed,
        )
        if fit is None:
            if self._debug:
                print(
                    f"  [TRAJFIT litter={task.get('litter_id')} no-fit: "
                    f"air_pts={len(pts)}/{len(history)} (min={self._trajfit_min_air_pts})]"
                )
            return None

        birth_frame = int(task.get('birth_frame', 0))
        lo = birth_frame - self._trajfit_max_back
        candidates = {}
        for frame_snapshot in task.get('actor_frames', []):
            fi = int(frame_snapshot.get('frame_index', -1))
            if fi < lo or fi > birth_frame:
                continue
            proj = _trajfit_point_at(fit, fi)
            for actor_snapshot in frame_snapshot.get('actors', []):
                actor = self._snapshot_to_actor(actor_snapshot)
                if actor is None:
                    continue
                if str(actor.get('cls', '')).lower() not in ('person', 'vehicle', 'scooter'):
                    continue
                box = actor['box']
                box_dist = self._point_to_box_distance(proj, box)
                bw = float(box[2]) - float(box[0])
                bh = float(box[3]) - float(box[1])
                margin = max(0.35 * math.hypot(bw, bh), 3.0 * fit['sigma'], 16.0)
                score = box_dist / max(margin, 1e-6)
                if score > self._trajfit_gate:
                    continue
                actor_key = self._actor_key(actor)
                candidate = candidates.setdefault(actor_key, {
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
                    candidate['best_frame'] = fi
                    candidate['best_frame_actors'] = frame_snapshot.get('actors', [])

        if self._debug:
            print(
                f"  [TRAJFIT litter={task.get('litter_id')} air_pts={len(pts)} "
                f"g={fit['g']:.2f} sigma={fit['sigma']:.1f} candidates={len(candidates)}]"
            )
        return candidates or None

    def _score_backward_actor(self, actor, birth_anchor, ground_world,
                              start_2d, end_2d, homography, history,
                              frame_index, birth_frame, prev_thrower_key=None):
        cls_name = str(actor.get('cls', '')).lower()
        if cls_name not in ('person', 'vehicle', 'scooter') or ground_world is None:
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
            ground_world[0] - actor_world[0],
            ground_world[1] - actor_world[1],
        )
        world_width = self._projected_actor_width((ax1, ay1, ax2, ay2), homography)
        if cls_name in ('vehicle', 'scooter'):
            threshold = max(160.0, world_width * 0.85)
            origin_margin = max(130.0, (ax2 - ax1) * 1.05, (ay2 - ay1) * 0.45)
        else:
            threshold = max(85.0, world_width * 1.9)
            origin_margin = max(75.0, (ax2 - ax1) * 1.25, (ay2 - ay1) * 0.45)

        score = world_dist / max(threshold, 1e-6)
        
        box_dist = self._point_to_box_distance(birth_anchor, actor['box'])
        if box_dist == 0.0:
            score = min(score, 1.8)

        if actor_key == prev_thrower_key:
            score *= THROWER_PREVIOUS_BONUS

        if start_2d is not None and end_2d is not None:
            score *= self._direction_factor(actor_anchor, start_2d, end_2d, history, homography)

        frame_gap = abs(int(frame_index) - int(birth_frame))
        score *= (1.0 + min(frame_gap, 45) * 0.025)

        release_like = self._release_origin_near_actor(history, actor)
        if release_like:
            score = min(score, 2.2)
            score *= 0.82

        if box_dist <= origin_margin:
            score *= 0.88

        return score, release_like

    def _drain_backward_results(self, vehicle_history=None):
        # worker 結果只能在 main/update thread 套用，避免 shared dict 競爭。
        while True:
            try:
                result = self._backward_results.get_nowait()
            except queue.Empty:
                break
            self._apply_backward_result(result, vehicle_history=vehicle_history)

    @staticmethod
    def _json_actor_key(actor_key):
        return list(actor_key) if actor_key is not None else None

    def _apply_backward_result(self, result, vehicle_history=None):
        if not isinstance(result, dict) or result.get('error'):
            return False
        litter_id = result.get('litter_id')
        if litter_id is None:
            return False
        litter_id = int(litter_id)
        revision = int(result.get('revision', 0))
        if revision < int(self._applied_backtrack_revisions.get(litter_id, -1)):
            return False
        self._applied_backtrack_revisions[litter_id] = revision

        route_candidates = result.get('route_candidates')
        if route_candidates is not None:
            self._smart_candidate_tables[litter_id] = list(route_candidates)

        actor_key = result.get('actor_key')
        person_key = result.get('person_key')
        vehicle_key = result.get('vehicle_key')
        status = str(result.get('status', 'legacy'))
        if actor_key is not None:
            actor_key = tuple(actor_key)
        if person_key is not None:
            person_key = tuple(person_key)
        if vehicle_key is not None:
            vehicle_key = tuple(vehicle_key)

        has_vehicle = (
            vehicle_key is not None
            or (actor_key is not None and actor_key[0] in ('vehicle', 'scooter'))
        )
        escalated = (
            actor_key is not None
            and (not self.require_vehicle_for_violation or has_vehicle)
        )
        assignment_signature = (
            actor_key,
            person_key,
            vehicle_key,
            status,
            result.get('route_id'),
            bool(escalated),
        )
        same_assignment = (
            self._applied_backtrack_signatures.get(litter_id)
            == assignment_signature
        )
        self._applied_backtrack_signatures[litter_id] = assignment_signature

        compact_result = {
            'status': status,
            'actor_key': actor_key,
            'person_key': person_key,
            'vehicle_key': vehicle_key,
            'direct_vehicle': bool(result.get('direct_vehicle', False)),
            'score': result.get('score'),
            'margin_to_second': result.get('margin_to_second'),
            'actor_margins': result.get('actor_margins'),
            'route_type': result.get('route_type'),
            'route_id': result.get('route_id'),
            'release_frame': result.get('release_frame'),
            'release_point': result.get('release_point'),
            'release_covariance': result.get('release_covariance'),
            'components': result.get('components'),
            'birth_frame': result.get('birth_frame'),
            'confirm_frame': result.get('confirm_frame'),
            'revision': revision,
        }
        if litter_id in self.active_litters:
            first_mark = (result.get('mark_items') or [{}])[0]
            self.active_litters[litter_id]['thrower_key'] = actor_key
            self.active_litters[litter_id]['thrower_center'] = first_mark.get('center')
            self.active_litters[litter_id]['backward_result'] = compact_result

        event = self._litter_events_by_id.get(litter_id)
        if event is None:
            # Backward-compatible recovery for tests/old injected events.
            event = next(
                (
                    item for item in self._litter_events
                    if int(item.get('litter_id', -1)) == litter_id
                ),
                None,
            )
            if event is not None:
                self._litter_events_by_id[litter_id] = event
        if event is not None:
            event['thrower_key'] = self._json_actor_key(actor_key)
            event['vehicle_key'] = self._json_actor_key(vehicle_key)
            event['escalated'] = bool(escalated)
            event['backtrack_status'] = status
            event['backtrack'] = compact_result

        new_marked_keys = (
            {
                tuple(mark['actor_key'])
                for mark in result.get('mark_items', [])
                if mark.get('actor_key') is not None
            }
            if escalated
            else set()
        )
        previous_marked_keys = set(
            self._backtrack_marked_keys.get(litter_id, set())
        )
        self._backtrack_marked_keys[litter_id] = new_marked_keys
        for stale_key in previous_marked_keys - new_marked_keys:
            still_referenced = any(
                stale_key in keys
                for other_litter_id, keys in self._backtrack_marked_keys.items()
                if int(other_litter_id) != litter_id
            )
            if (
                not still_referenced
                and self.violators.get(stale_key, {}).get('action') == 'littering'
            ):
                self.violators.pop(stale_key, None)

        # Smart 模式在 committed assignment 前不產生 side effect；dustbin 或
        # require_vehicle 未滿足時也不畫違規框、不送 OCR。
        if not escalated or same_assignment:
            return True

        plate_key = vehicle_key or result.get('plate_key')
        if plate_key is not None:
            plate_key = tuple(plate_key)
        plate_blocked_since_litter = bool(
            result.get('plate_blocked_since_litter', False)
        )
        if (
            plate_blocked_since_litter
            and plate_key is not None
            and plate_key[0] in ('vehicle', 'scooter')
            and vehicle_history is not None
        ):
            plate_entry = vehicle_history[plate_key[1]]
            if plate_entry.get('license_plate') is None:
                plate_entry['plate_search_until_found'] = True
                plate_entry['plate_blocked_since_litter'] = True
                plate_entry['plate_search_birth_frame'] = result.get('birth_frame')
                plate_entry['plate_search_litter_id'] = litter_id

        for mark in result.get('mark_items', []):
            mark_key = mark.get('actor_key')
            if mark_key is None:
                continue
            mark_key = tuple(mark_key)
            self._mark_violator(
                mark_key,
                mark.get('center'),
                ttl=CONFIRMED_VIOLATOR_TTL,
                until_plate_found=(
                    plate_blocked_since_litter and mark_key == plate_key
                ),
                action='littering',
            )

        plate_items = result.get('plate_roi_items') or []
        if plate_items:
            with self._backward_plate_lock:
                self.backward_plate_roi_items.extend(plate_items)
        return True

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
                if len(items) >= BACKWARD_PLATE_ROI_PER_RESULT:
                    return items
        return items

    def _action_actor_frames(self, frame_index=None):
        # STGCN action 沒有 litter birth frame；用最近一段 actor ring buffer 反查人車關聯。
        with self._actor_history_lock:
            if not self.actor_frame_history:
                return []
            latest_frame = (
                int(frame_index)
                if frame_index is not None
                else int(self.actor_frame_history[-1].get('frame_index', 0))
            )
            start_frame = latest_frame - int(ACTION_VEHICLE_BACKTRACK_FRAMES)
            return [
                {
                    'frame_index': item['frame_index'],
                    'actors': [dict(actor) for actor in item.get('actors', [])],
                }
                for item in self.actor_frame_history
                if start_frame <= int(item.get('frame_index', -1)) <= latest_frame
            ]

    def _find_action_vehicle_for_person(self, person_id, actor_frames):
        # 從最新幀往前找同一 person 曾經重疊過的 vehicle/scooter。
        person_key = ('person', int(person_id))
        fallback_person_center = None

        for frame_snapshot in sorted(
            actor_frames,
            key=lambda item: int(item.get('frame_index', -1)),
            reverse=True,
        ):
            frame_actors = frame_snapshot.get('actors', [])
            person_center = self._snapshot_center_for_key(person_key, frame_actors)
            if person_center is not None and fallback_person_center is None:
                fallback_person_center = person_center

            vehicle_key = self._linked_vehicle_key(person_key, frame_actors)
            if vehicle_key is None:
                continue

            vehicle_center = (
                self._snapshot_center_for_key(vehicle_key, frame_actors) or
                person_center or
                fallback_person_center
            )
            return vehicle_key, vehicle_center, fallback_person_center

        return None, None, fallback_person_center

    def _queue_action_vehicle_plate_lookup(self, vehicle_key, actor_frames, frame_index, vehicle_history=None):
        # 歷史 vehicle/scooter 不一定還在當前畫面；若有歷史 ROI，直接交給車牌 OCR。
        if vehicle_key is None or vehicle_key[0] not in ('vehicle', 'scooter'):
            return

        history_entry = None
        if vehicle_history is not None:
            history_entry = vehicle_history[vehicle_key[1]]
            if history_entry.get('stgcn_action_plate_submitted', False):
                return
            if history_entry.get('license_plate') is not None:
                return

        plate_items = self._plate_roi_items_for_key(
            vehicle_key,
            actor_frames,
            int(frame_index) if frame_index is not None else 0,
        )
        if plate_items:
            with self._backward_plate_lock:
                self.backward_plate_roi_items.extend(plate_items)

        if history_entry is not None:
            history_entry['stgcn_action_plate_submitted'] = True
            history_entry['plate_search_until_found'] = True
            history_entry['plate_search_source'] = 'stgcn_action_backtrack'
            history_entry['plate_search_birth_frame'] = frame_index
            history_entry['plate_blocked_since_litter'] = history_entry.get(
                'plate_blocked_since_litter',
                False,
            ) or not bool(plate_items)

    def register_action_violators(
        self,
        person_action_map,
        actors,
        person_vehicle_map=None,
        ttl=None,
        frame_index=None,
        vehicle_history=None,
    ):
        # STGCN 旁路只接受 urinate；littering 一律由 litter object-event branch 確認。
        if not person_action_map:
            return set()

        if frame_index is not None:
            self._current_frame_index = int(frame_index)
        if person_vehicle_map:
            for p_id, vehicle_key_or_id in person_vehicle_map.items():
                veh_key = self._normalize_vehicle_like_key(vehicle_key_or_id)
                self.person_to_vehicle_history[int(p_id)] = veh_key
                self._record_dismount_edge(p_id, veh_key, self._current_frame_index)

        ttl = int(ttl or CONFIRMED_VIOLATOR_TTL)
        actor_center_map = {}
        for actor in actors:
            actor_center_map[self._actor_key(actor)] = self._actor_center(actor)
        action_actor_frames = self._action_actor_frames(frame_index)

        marked_violators = set()
        for raw_person_id, action_info in person_action_map.items():
            if not isinstance(action_info, dict) or not action_info.get('alert', False):
                continue

            action_name = str(action_info.get('action', '')).strip().lower()
            if action_name not in ('urinate', 'urination', 'urinating'):
                continue
            action_name = 'urinate'

            try:
                person_id = int(raw_person_id)
            except (TypeError, ValueError):
                continue

            event_frame = action_info.get('new_urinate_event_frame')
            if event_frame is not None:
                try:
                    self._latest_action_event_frames[person_id] = int(event_frame)
                except (TypeError, ValueError):
                    pass

            person_key = ('person', person_id)
            person_center = actor_center_map.get(person_key)
            historical_vehicle_key, historical_vehicle_center, historical_person_center = (
                self._find_action_vehicle_for_person(person_id, action_actor_frames)
            )
            if person_center is None:
                person_center = historical_person_center
            if person_center is None:
                continue

            self._mark_violator(person_key, person_center, ttl=ttl, action=action_name)
            marked_violators.add(person_key)

            vehicle_key = self._bound_vehicle_for_person(person_id)
            if vehicle_key is None:
                vehicle_key = self._find_vehicle_for_person(person_id, actors)
                if vehicle_key is not None:
                    self.person_to_vehicle_history[person_id] = vehicle_key
            if vehicle_key is None and historical_vehicle_key is not None:
                vehicle_key = historical_vehicle_key
                self.person_to_vehicle_history[person_id] = vehicle_key

            if vehicle_key is not None:
                action_event_frame = self._latest_action_event_frames.get(person_id)
                if action_event_frame is not None:
                    self._action_vehicle_associations.setdefault(
                        (person_id, action_event_frame), vehicle_key
                    )
                vehicle_center = actor_center_map.get(vehicle_key)
                if vehicle_center is None and vehicle_key == historical_vehicle_key:
                    vehicle_center = historical_vehicle_center
                if vehicle_center is None:
                    vehicle_center = person_center
                self._mark_violator(vehicle_key, vehicle_center, ttl=ttl, action=action_name)
                self._queue_action_vehicle_plate_lookup(
                    vehicle_key,
                    action_actor_frames,
                    frame_index,
                    vehicle_history=vehicle_history,
                )
                marked_violators.add(vehicle_key)

        return marked_violators

    def get_action_vehicle_associations(self):
        """回傳 confirmed urinate 人物實際回追到的車輛，不暴露可變內部狀態。"""
        return dict(self._action_vehicle_associations)

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

    def _carrier_vehicle(self, litter_box, actors):
        # 找出與 litter 重疊最高的車輛 (litter 的「載體」)，回傳 (actor_key, overlap, center)。
        # overlap 取 mask 重疊與 bbox 內含的較大值：貨物/車牌(高 mask) 或框內部件(高內含) 皆可偵測。
        # 回傳 (None, 0.0, None) 代表 litter 不在任何車輛上。
        best_key = None
        best_overlap = 0.0
        best_center = None
        for actor in actors or []:
            cls_name = str(actor.get('cls', '')).lower()
            if cls_name not in ('vehicle', 'scooter'):
                continue
            try:
                key = (cls_name, int(actor['track_id']))
            except (KeyError, TypeError, ValueError):
                continue
            ratios = []
            mask_poly = actor.get('mask_poly')
            if mask_poly is not None:
                mr = calculate_mask_overlap_ratio(litter_box, mask_poly)
                if mr is not None:
                    ratios.append(float(mr))
            cr = self._box_containment_ratio(litter_box, actor.get('box'))
            if cr is not None:
                ratios.append(float(cr))
            overlap = max(ratios) if ratios else 0.0
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = key
                best_center = self._actor_center(actor)
        return best_key, best_overlap, best_center

    def _actor_center_near_frame(self, actor_key, target_frame):
        # actor ring buffer 中最接近 target_frame 的中心點 (YOLO-seg 可能隔幀執行，就近取值)。
        best_center = None
        best_gap = None
        target_frame = int(target_frame)
        with self._actor_history_lock:
            for item in self.actor_frame_history:
                gap = abs(int(item.get('frame_index', -1)) - target_frame)
                if best_gap is not None and gap >= best_gap:
                    continue
                for a in item.get('actors', []):
                    try:
                        key = (str(a.get('cls', '')).lower(), int(a['track_id']))
                    except (KeyError, TypeError, ValueError):
                        continue
                    if key != actor_key:
                        continue
                    center = a.get('center')
                    if center is not None:
                        best_gap = gap
                        best_center = center
        return best_center

    def _litter_vehicle_separation(self, carrier_key, carrier_center,
                                   birth_centroid, birth_frame, curr_centroid, curr_frame):
        # litter 相對載體車輛的淨位移：扣除車輛自身在畫面上的位移後，才是垃圾真正脫離車輛的證據。
        # 車身部件/貨物隨車移動 → rel ≈ 0；被丟出的垃圾會分離 → rel 大。回傳 None 代表缺歷史資料。
        v_start = self._actor_center_near_frame(carrier_key, birth_frame)
        v_end = self._actor_center_near_frame(carrier_key, curr_frame)
        if v_end is None:
            v_end = carrier_center
        if v_start is None or v_end is None:
            return None
        veh_dx = float(v_end[0]) - float(v_start[0])
        veh_dy = float(v_end[1]) - float(v_start[1])
        lit_dx = float(curr_centroid[0]) - float(birth_centroid[0])
        lit_dy = float(curr_centroid[1]) - float(birth_centroid[1])
        return math.hypot(lit_dx - veh_dx, lit_dy - veh_dy)

    @staticmethod
    def _box_containment_ratio(litter_box, actor_box):
        # litter bbox 落在 actor bbox 內的面積比例；mask 不可用時的退路。
        if actor_box is None:
            return None
        lx1, ly1, lx2, ly2 = map(float, litter_box[:4])
        ax1, ay1, ax2, ay2 = map(float, actor_box[:4])
        ix1, iy1 = max(lx1, ax1), max(ly1, ay1)
        ix2, iy2 = min(lx2, ax2), min(ly2, ay2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        litter_area = max(lx2 - lx1, 1e-6) * max(ly2 - ly1, 1e-6)
        return min(max(inter / litter_area, 0.0), 1.0)

    def _thrower_has_vehicle(self, thrower_key):
        # thrower 是否關聯到車輛：本身是 vehicle/scooter，或為已綁定車輛的 person。
        if thrower_key is None:
            return False
        cls_name, track_id = thrower_key
        if cls_name in ('vehicle', 'scooter'):
            return True
        if cls_name == 'person' and self._bound_vehicle_for_person(track_id) is not None:
            return True
        return False

    def _bound_vehicle_for_person(self, person_id):
        # 統一的 person→bound vehicle 查詢：dismount edge 開啟時走 TTL-aware 持久邊，
        # 否則走舊的無 TTL person_to_vehicle_history(關閉時行為與改動前完全一致)。
        try:
            pid = int(person_id)
        except (TypeError, ValueError):
            return None
        if self._dismount_edge_enabled:
            return self._dismount_vehicle_for_person(pid)
        return self.person_to_vehicle_history.get(pid)

    def _record_dismount_edge(self, person_id, vehicle_key, frame_index):
        # 累積/刷新 person→vehicle 持久邊。換綁不同車時重置 first_frame；同車則延長 last_bound_frame。
        if not self._dismount_edge_enabled or vehicle_key is None or frame_index is None:
            return
        try:
            pid = int(person_id)
        except (TypeError, ValueError):
            return
        fi = int(frame_index)
        edge = self._dismount_edges.get(pid)
        if edge is None or edge.get('vehicle_key') != vehicle_key:
            self._dismount_edges[pid] = {
                'vehicle_key': vehicle_key,
                'first_frame': fi,
                'last_bound_frame': fi,
                'bound_count': 1,
            }
        else:
            edge['last_bound_frame'] = fi
            edge['bound_count'] = int(edge.get('bound_count', 0)) + 1

    def _dismount_vehicle_for_person(self, person_id, frame_index=None):
        # TTL-aware 查詢：邊需 bound_count >= min_bind(濾路過 fluke)且在 TTL 內(濾 track_id 回收舊綁定)。
        if not self._dismount_edge_enabled:
            return None
        edge = self._dismount_edges.get(int(person_id))
        if edge is None or int(edge.get('bound_count', 0)) < self._dismount_min_bind:
            return None
        fi = self._current_frame_index if frame_index is None else int(frame_index)
        ttl_frames = max(1, int(self._dismount_ttl_sec * self.fps))
        if fi - int(edge['last_bound_frame']) > ttl_frames:
            return None   # 邊已過期
        return edge['vehicle_key']

    def _prune_dismount_edges(self, frame_index):
        # 清掉遠超 TTL 的陳舊邊(保留 3×TTL 緩衝後丟棄)，避免長影片無界成長。
        if not self._dismount_edge_enabled or frame_index is None or not self._dismount_edges:
            return
        keep_frames = max(1, int(self._dismount_ttl_sec * self.fps)) * 3
        fi = int(frame_index)
        stale = [
            pid for pid, e in self._dismount_edges.items()
            if fi - int(e.get('last_bound_frame', fi)) > keep_frames
        ]
        for pid in stale:
            del self._dismount_edges[pid]

    def _mark_violator(self, actor_key, center, ttl, until_plate_found=False, action=None):
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
                'action': action,
            }
            return

        prev['ttl'] = max(prev['ttl'], ttl)
        prev['until_plate_found'] = (
            bool(prev.get('until_plate_found', False)) or bool(until_plate_found)
        )
        if action:
            prev['action'] = action
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
            if d <= VIOLATOR_REBIND_DISTANCE and d < best_dist:
                best_dist = d
                best_key = actor_key
                best_center = actor_center

        return best_key, best_center

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
        start_2d = None
        end_2d = None
        if history and len(history) >= 2:
            start_2d = history[0]
            end_2d = history[-1]

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
                origin_margin = max(130.0, (ax2 - ax1) * 1.05, (ay2 - ay1) * 0.45)
            else:
                threshold = max(90.0, world_width * 1.8)
                origin_margin = max(75.0, (ax2 - ax1) * 1.25, (ay2 - ay1) * 0.45)

            score = world_dist / max(threshold, 1e-6)
            
            box_dist = self._point_to_box_distance(history[0], actor['box']) if history else self._point_to_box_distance(litter_anchor, actor['box'])
            if box_dist == 0.0:
                score = min(score, 1.8)

            if actor_key == prev_thrower_key:
                score *= THROWER_PREVIOUS_BONUS

            if start_2d is not None and end_2d is not None:
                score *= self._direction_factor(actor_anchor, start_2d, end_2d, history, homography)
                
            release_like = self._release_origin_near_actor(history, actor)
            if release_like:
                score = min(score, 2.2)
                score *= 0.82

            if box_dist <= origin_margin:
                score *= 0.88

            if score < fallback_score:
                fallback_score = score
                fallback_actor_key = actor_key
                fallback_center = actor_center

            if (
                score <= THROWER_RELEASE_ORIGIN_SCORE_LIMIT and
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
            # fallback 必須要有 actor bbox 與 litter 邊距夠近的證據；
            # 否則靜止舊垃圾會被遠方路過的 actor 認領為 thrower 並過 confirm。
            fallback_box_dist = float('inf')
            fallback_anchor = history[0] if history else litter_anchor
            for actor in actors:
                try:
                    actor_id_key = (str(actor.get('cls', '')).lower(), int(actor['track_id']))
                except (KeyError, TypeError, ValueError):
                    continue
                if actor_id_key != fallback_actor_key:
                    continue
                fallback_box_dist = self._point_to_box_distance(fallback_anchor, actor['box'])
                break

            if fallback_box_dist <= THROWER_BIRTH_BOX_DIST_LIMIT:
                best_actor_key = fallback_actor_key
                best_center = fallback_center
        elif best_actor_key is None and release_actor_key is not None:
            best_actor_key = release_actor_key
            best_center = release_center

        if best_actor_key is None and history and len(history) >= 2:
            # pseudo-ground score 在近景、極大車框時可能把真正從車框邊緣拋出的
            # 輕物排到所有 fallback 之外。只接受比一般 release_like 更強的證據：
            # 軌跡起點實際在同一車框內，且最後一點已明確脫離。這不是放寬
            # 最近車輛歸因；沒有 exact box-origin 的候選仍維持 NULL。
            edge_release_candidates = []
            for actor in actors:
                cls_name = str(actor.get('cls', '')).lower()
                if cls_name not in ('vehicle', 'scooter'):
                    continue
                try:
                    track_id = int(actor['track_id'])
                    box = actor['box']
                except (KeyError, TypeError, ValueError):
                    continue
                start_distance = self._point_to_box_distance(history[0], box)
                end_distance = self._point_to_box_distance(history[-1], box)
                if (
                    start_distance == 0.0 and
                    end_distance >= THROWER_EDGE_RELEASE_MIN_SEPARATION and
                    self._release_origin_near_actor(history, actor)
                ):
                    edge_release_candidates.append((end_distance, (cls_name, track_id), self._actor_center(actor)))
            if edge_release_candidates:
                _, best_actor_key, best_center = min(edge_release_candidates, key=lambda item: item[0])

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
        if abs(dx) < MIN_CONFIRM_HORIZONTAL_DISPLACEMENT or dy < MIN_CONFIRM_DOWNWARD_DISPLACEMENT:
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
        # BEV 基板開啟且累積觀測足夠時，改用跨幀穩定平面(固定機位一次性標定)，消除逐幀抖動。
        if self._bev_stable_enabled:
            stable = self._stable_ground_homography()
            if stable is not None:
                return stable

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

    def _stable_ground_homography(self):
        # 固定機位一次性 BEV 標定：用跨幀累積的車輛底邊擬合單一穩定地面平面並快取。
        # 與 _estimate_ground_homography 同一套公式，差別在統計量來自整段累積而非單幀，
        # 且 ground_ref_y 由累積底邊決定(平面屬相機幾何，與個別 litter 無關)。
        # 觀測數不足回 None，由呼叫端 fallback 回每幀估計。
        with self._bev_lock:
            n = len(self._bev_bottoms)
            if n < self._bev_min_obs:
                return None
            if (
                self._bev_cached_homography is None or
                (n - self._bev_cache_count) >= self._bev_recompute_every
            ):
                bottom_x = np.asarray([p[0] for p in self._bev_bottoms], dtype=np.float32)
                bottom_y = np.asarray([p[1] for p in self._bev_bottoms], dtype=np.float32)
                heights = np.asarray(self._bev_heights, dtype=np.float32)
                median_h = float(np.median(heights)) if heights.size else 80.0
                y_spread = float(np.max(bottom_y) - np.min(bottom_y)) if bottom_y.size > 1 else 0.0
                center_x = float(np.median(bottom_x))
                ground_ref_y = float(np.max(bottom_y))
                horizon_offset = max(80.0, median_h * 1.2, y_spread * 1.5)
                horizon_y = float(np.min(bottom_y)) - horizon_offset
                ground_scale = max(ground_ref_y - horizon_y, 80.0)
                self._bev_cached_homography = np.asarray([
                    [ground_scale, 0.0, -ground_scale * center_x],
                    [0.0, -ground_scale, ground_scale * ground_ref_y],
                    [0.0, 1.0, -horizon_y],
                ], dtype=np.float32)
                self._bev_cache_count = n
            return self._bev_cached_homography

    def _project_point(self, point, homography):
        # 將影像點投影到 pseudo-ground，並避免深度接近 0 造成數值爆炸。
        if point is None:
            return None

        x, y = map(float, point)
        projected = homography @ np.asarray([x, y, 1.0], dtype=np.float32)
        depth = float(projected[2])
        if abs(depth) < HOMOGRAPHY_MIN_DEPTH:
            depth = HOMOGRAPHY_MIN_DEPTH if depth >= 0.0 else -HOMOGRAPHY_MIN_DEPTH

        return (float(projected[0]) / depth, float(projected[1]) / depth)

    def _projected_actor_width(self, box, homography):
        # 投影後 actor 寬度作為距離容忍門檻，讓遠近車輛尺度更一致。
        ax1, ay1, ax2, ay2 = map(float, box[:4])
        left = self._project_point((ax1, ay2), homography)
        right = self._project_point((ax2, ay2), homography)
        if left is None or right is None:
            return max(ax2 - ax1, 1.0)

        return max(math.hypot(right[0] - left[0], right[1] - left[1]), 1.0)

    def _direction_factor(self, actor_anchor, start_2d, end_2d, history, homography=None):
        # 反追蹤方向因子 dispatch：開啟 revvel 時用早期速度反向外插(連續因子)，
        # 否則(或資訊不足)退回舊的 start→end 3 級 cosine 分桶。
        # 傳入 homography 且 BEV 開啟時，方向 cosine 在 BEV 度量空間計算(物理一致)。
        if self._revvel_enabled:
            factor = self._reverse_velocity_factor(actor_anchor, history, homography)
            if factor is not None:
                return factor
        return self._trajectory_direction_factor_2d(actor_anchor, start_2d, end_2d)

    def _reverse_velocity_factor(self, actor_anchor, history, homography=None):
        # OCM 式：用 litter 早期軌跡速度反向外插出「來源方向」，評分 actor 是否落在反向射線上。
        # 回傳 None 代表資訊不足(軌跡太短/近乎靜止)，由 dispatch 退回舊因子。
        if not history or len(history) < 3:
            return None
        k = min(len(history), self._revvel_early_pts)
        p0 = history[0]          # birth/釋放點
        pk = history[k - 1]      # 早期軌跡點：釋放後前段，最保留拋擲方向
        # 速度有效性 gate 在「影像空間」判斷(門檻以像素校準，BEV 尺度不適用)。
        if math.hypot(float(pk[0]) - float(p0[0]), float(pk[1]) - float(p0[1])) < self._revvel_min_speed:
            return None          # 近乎靜止 → 無方向資訊
        # 方向 cosine 在 BEV 度量空間算(物理一致，消除透視扭曲)；無 BEV 時退回影像空間。
        anchor = actor_anchor
        if homography is not None and self._bev_stable_enabled:
            p0 = self._project_point(p0, homography) or p0
            pk = self._project_point(pk, homography) or pk
            anchor = self._project_point(actor_anchor, homography) or actor_anchor
        vx = float(pk[0]) - float(p0[0])
        vy = float(pk[1]) - float(p0[1])
        vlen = math.hypot(vx, vy)
        if vlen < 1e-6:
            return None
        # 反向單位速度向量(指回拋擲來源)。
        rx, ry = -vx / vlen, -vy / vlen
        ax = float(anchor[0]) - float(p0[0])
        ay = float(anchor[1]) - float(p0[1])
        alen = math.hypot(ax, ay)
        if alen < 1e-6:
            return self._revvel_max_bonus   # actor 幾乎就在釋放點 → 最強 bonus
        # cosine：actor 方向與反向速度的對齊度。+1=actor 正好在來源方向(最可能)，
        # -1=actor 在 litter 飛行前方(非來源)。scale-free，近遠景一致。
        cos = (ax * rx + ay * ry) / alen
        factor = 1.0 - self._revvel_gain * cos
        return max(self._revvel_max_bonus, min(self._revvel_max_penalty, factor))

    @staticmethod
    def _trajectory_direction_factor_2d(actor_anchor, start_2d, end_2d):
        # 使用 2D 影像座標計算軌跡方向，避免空中點投影後產生巨大誤差。
        move_vec = (end_2d[0] - start_2d[0], end_2d[1] - start_2d[1])
        from_actor_vec = (start_2d[0] - actor_anchor[0], start_2d[1] - actor_anchor[1])
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
