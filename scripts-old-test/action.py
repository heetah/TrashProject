# -*- coding: utf-8 -*-
# STGCN++ 動作辨識模組：從 person bbox 對應 pose keypoints，累積序列後判斷 normal/littering/urination。
import os
import sys
import traceback
from collections import deque

import numpy as np
import torch
from ultralytics import YOLO
from timeUtils import profile_block


ACTION_CLASSES = {0: "normal", 1: "littering", 2: "urination"}
URINATION_ACTIONS = {"urination", "urinating"}
VIOLATION_ACTIONS = {"littering", *URINATION_ACTIONS}


def _add_stat(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def _int_env(name, default):
    # 讀取整數環境變數；格式錯誤時回退預設值。
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)


def _safe_fps(fps, default=30.0):
    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        fps_value = float(default)
    if fps_value <= 0:
        return float(default)
    return fps_value


def _can_use_half(device):
    # pose YOLO 只有在 CUDA 上使用 half precision。
    if not torch.cuda.is_available():
        return False
    return str(device).lower() not in ("cpu", "mps")


def _select_device(device=None):
    # STGCN/pose 裝置選擇：CLI 優先，其次 ACTION_DEVICE，最後自動偵測 CUDA。
    requested = device if device is not None else os.environ.get("ACTION_DEVICE")
    if requested is None or str(requested).strip() == "":
        return 0 if torch.cuda.is_available() else "cpu"

    requested_str = str(requested).strip().lower()
    wants_cuda = (
        isinstance(requested, int) or
        requested_str.isdigit() or
        requested_str.startswith("cuda")
    )
    if wants_cuda:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return int(requested_str) if requested_str.isdigit() else requested
        print(f"ACTION_DEVICE={requested} requested but CUDA is unavailable; fallback to CPU")
        return "cpu"

    return requested


def iou_xyxy(box_a, box_b):
    # bbox IoU：用來把 YOLO pose 偵測到的人與主流程 person track 對齊。
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0 else inter_area / union


class STGCNActionModule:
    # 封裝模型載入、pose 擷取、骨架序列快取、STGCN 推理與 alert 維持。
    def __init__(
        self,
        pose_model_path,
        stgcn_weight_path,
        stgcn_config_path,
        # STGCN 判定為 littering/urination 後，conf 需 >= 0.5（預設值）才會啟動違規警報。
        action_threshold=0.5,
        track_iou_threshold=0.2,
        window_size=25,
        alert_frames=35,
        urination_window_sec=10.0,
        urination_min_sec=8.0,
        device=None,
        profiler=None,
    ):
        # 初始化 STGCN 狀態：每個 track_id 都有自己的骨架 history 與 alert counter。
        self.profiler = profiler
        selected_device = _select_device(device)
        self.pose_device = selected_device
        self.device = torch.device(selected_device)
        self.pose_half = _can_use_half(selected_device)
        self.pose_imgsz = _int_env("ACTION_POSE_IMGSZ", 0)
        self.action_threshold = float(action_threshold)
        self.track_iou_threshold = float(track_iou_threshold)
        self.window_size = int(window_size)
        self.alert_frames = int(alert_frames)
        self.urination_window_sec = max(0.0, float(urination_window_sec))
        self.urination_min_sec = max(0.0, float(urination_min_sec))
        if self.urination_min_sec > self.urination_window_sec:
            raise ValueError("urination_min_sec cannot exceed urination_window_sec")
        self.predict_interval = max(1, _int_env("ACTION_PREDICT_INTERVAL", 1))
        self.frame_index = 0
        self.track_history = {}
        self.urination_history = {}
        self.urination_positive_counts = {}
        self.alert_counter = {}
        self.alert_action = {}
        self.last_action = {}
        self._logged_error = False
        self.action_classes = dict(ACTION_CLASSES)
        self.violation_actions = set(VIOLATION_ACTIONS)
        self.urination_actions = set(URINATION_ACTIONS)

        self.model = None
        self.pose_model = None
        self.pose_model_path = None
        if isinstance(pose_model_path, (list, tuple)):
            self.pose_model_candidates = [str(path) for path in pose_model_path if path]
        else:
            self.pose_model_candidates = [str(pose_model_path)] if pose_model_path else []
        self._pose_candidate_index = 0
        self.inference_skeleton = None
        self.loaded = False
        self._load_stgcn(stgcn_weight_path, stgcn_config_path, profiler=profiler)
        if self.loaded:
            # STGCN 成功後才載入 pose model，避免 action 不可用時浪費額外模型成本。
            if not self._load_pose_model(start_index=0, profiler=profiler):
                self.loaded = False

    def _load_pose_model(self, start_index=0, profiler=None):
        # TensorRT engine 若不可用，依候選順序自動回退到 .pt，避免加速失敗時整個 action 掛掉。
        last_error = None
        for idx in range(int(start_index), len(self.pose_model_candidates)):
            pose_model_path = self.pose_model_candidates[idx]
            try:
                if idx > int(start_index):
                    print(f"YOLO pose model retry fallback: {pose_model_path}")
                with profile_block(profiler, "model_load.pose_yolo"):
                    self.pose_model = YOLO(pose_model_path)
                self.pose_model_path = pose_model_path
                self._pose_candidate_index = idx
                print(f"YOLO pose model loaded: {pose_model_path}")
                return True
            except Exception as e:
                last_error = e
                print(f"YOLO pose model load failed: {pose_model_path}: {e}")

        self.pose_model = None
        self.pose_model_path = None
        if last_error is not None:
            print(f"YOLO pose model load failed for all candidates: {last_error}")
        else:
            print("YOLO pose model load failed: no candidate path")
        return False

    def _load_stgcn(self, weight_path, config_path, profiler=None):
        # 載入 MMACTION2 recognizer；若缺檔或 import 失敗，action 自動退回 normal。
        if not (weight_path and os.path.exists(weight_path)):
            print("STGCN weight not found, fallback mode")
            return
        if not (config_path and os.path.exists(config_path)):
            print("STGCN config not found, fallback mode")
            return
        try:
            try:
                # transformers 新舊版本符號位置不同，這裡補相容 alias。
                import transformers.modeling_utils as tf_modeling_utils
                from transformers import pytorch_utils as tf_pt_utils

                for sym in (
                    "apply_chunking_to_forward",
                    "find_pruneable_heads_and_indices",
                    "prune_linear_layer",
                ):
                    if not hasattr(tf_modeling_utils, sym) and hasattr(tf_pt_utils, sym):
                        setattr(tf_modeling_utils, sym, getattr(tf_pt_utils, sym))
            except Exception:
                pass

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            mmaction_repo = os.path.join(project_root, "mmaction2")
            if mmaction_repo not in sys.path:
                # 使用專案內 mmaction2，避免吃到系統其他版本。
                sys.path.insert(0, mmaction_repo)
            from mmaction.apis import init_recognizer, inference_skeleton

            device_str = str(self.device)
            if device_str == "cuda":
                device_str = "cuda:0"

            original_torch_load = torch.load

            def compat_torch_load(*args, **kwargs):
                # MMACTION2 舊 checkpoint 需要 weights_only=False 才能完整載入。
                kwargs.setdefault("weights_only", False)
                return original_torch_load(*args, **kwargs)

            torch.load = compat_torch_load
            try:
                with profile_block(profiler, "model_load.stgcn_recognizer"):
                    self.model = init_recognizer(config_path, weight_path, device=device_str)
            finally:
                torch.load = original_torch_load

            self.inference_skeleton = inference_skeleton
            self.loaded = True
            print("STGCN action module loaded")
        except Exception as e:
            print(f"STGCN action module load failed: {e}")

    @staticmethod
    def _extract_skeleton(pose_result, person_idx):
        # 從 YOLO pose 結果取出單人的 17 點骨架與 confidence。
        try:
            keypoints = pose_result.keypoints
            if keypoints is None or keypoints.xy is None or len(keypoints.xy) <= person_idx:
                return None
            kpts = keypoints.xy[person_idx].cpu().numpy()
            if hasattr(keypoints, "conf") and keypoints.conf is not None:
                conf = keypoints.conf[person_idx].cpu().numpy().reshape(-1, 1)
            else:
                conf = np.ones((kpts.shape[0], 1), dtype=np.float32)
            return np.hstack([kpts, conf]).astype(np.float32)
        except Exception:
            return None

    def _predict_action(
        self,
        skeleton_sequence,
        img_shape=None,
        profiler=None,
        profile_name="action.stgcn_predict",
    ):
        # 將骨架序列轉成 MMACTION2 inference_skeleton 格式並取得分類分數。
        if not self.loaded or self.model is None:
            return "normal", 0.0
        try:
            if img_shape is None:
                img_shape = (1080, 1920)
            img_shape = (int(img_shape[0]), int(img_shape[1]))
            active_profiler = profiler if profiler is not None else self.profiler
            with profile_block(active_profiler, profile_name):
                pose_results = []
                for i in range(skeleton_sequence.shape[0]):
                    pose_results.append(
                        {
                            "keypoints": skeleton_sequence[i : i + 1, :, :2].astype(np.float32),
                            "keypoint_scores": skeleton_sequence[i : i + 1, :, 2].astype(np.float32),
                        }
                    )
                result = self.inference_skeleton(self.model, pose_results, img_shape=img_shape)
                pred_score = result.pred_score.detach().cpu().numpy()
                action_idx = int(np.argmax(pred_score))
                conf = float(pred_score[action_idx])
            return self.action_classes.get(action_idx, "unknown"), conf
        except Exception as e:
            if not self._logged_error:
                self._logged_error = True
                print(f"STGCN inference error: {repr(e)}")
                print(traceback.format_exc())
            return "normal", 0.0

    def warmup(self, profiler=None):
        # 先跑 pose 與 STGCN 假資料，避免正式影片第一段因 backend 初始化變慢。
        active_profiler = profiler if profiler is not None else self.profiler
        if not self.loaded or self.model is None or self.pose_model is None:
            return

        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        pose_kwargs = {
            "conf": 0.01,
            "stream": False,
            "verbose": False,
            "device": self.pose_device,
            "half": self.pose_half,
        }
        if self.pose_imgsz > 0:
            pose_kwargs["imgsz"] = self.pose_imgsz

        try:
            with profile_block(active_profiler, "model_warmup.pose_yolo"):
                self.pose_model.predict(dummy_frame, **pose_kwargs)
            with torch.inference_mode():
                dummy_skeleton = np.zeros((self.window_size, 17, 3), dtype=np.float32)
                self._predict_action(
                    dummy_skeleton,
                    img_shape=dummy_frame.shape[:2],
                    profiler=active_profiler,
                    profile_name="model_warmup.stgcn_predict",
                )
            print("STGCN action module warmed up.")
        except Exception as exc:
            next_index = int(self._pose_candidate_index) + 1
            if next_index < len(self.pose_model_candidates):
                print(f"STGCN pose warmup failed on {self.pose_model_path}; retry fallback: {exc}")
                if self._load_pose_model(start_index=next_index, profiler=active_profiler):
                    self.warmup(profiler=active_profiler)
                    return
            self.loaded = False
            print(f"STGCN action module warmup failed; action disabled: {exc}")

    def _is_urination_action(self, action):
        return str(action or "").strip().lower() in self.urination_actions

    def _record_urination_evidence(self, track_id, action, conf, fps, stats=None):
        # urinating/urination 需在最近 10 秒內累積至少 8 秒 positive，避免單次 STGCN 閃爍誤報。
        fps_value = _safe_fps(fps)
        now_sec = self.frame_index / fps_value
        positive = self._is_urination_action(action) and float(conf) >= self.action_threshold
        history = self.urination_history.setdefault(track_id, deque())
        if track_id not in self.urination_positive_counts:
            self.urination_positive_counts[track_id] = 0

        history.append((now_sec, positive))
        if positive:
            self.urination_positive_counts[track_id] += 1
            _add_stat(stats, "stgcn_urination_evidence_frames")

        cutoff_sec = now_sec - self.urination_window_sec
        while history and history[0][0] < cutoff_sec:
            _, stale_positive = history.popleft()
            if stale_positive:
                self.urination_positive_counts[track_id] = max(
                    0,
                    self.urination_positive_counts[track_id] - 1,
                )

        positive_sec = self.urination_positive_counts[track_id] / fps_value
        observed_sec = min(self.urination_window_sec, len(history) / fps_value)
        confirmed = (
            self.urination_min_sec <= 0.0 or
            positive_sec >= self.urination_min_sec
        )
        return confirmed, positive_sec, observed_sec

    def update(self, frame, persons, fps=30.0, profiler=None, stats=None):
        """
        persons：格式為 {'box': xyxy, 'track_id': int, ...} 的 list。
        回傳：{track_id: {'action': str, 'conf': float, 'stgcn_conf': float, 'alert': bool}}。
        """
        # 每幀更新入口：追蹤每個 person 的骨架 history，視 interval 決定是否跑 STGCN。
        active_profiler = profiler if profiler is not None else self.profiler
        with profile_block(active_profiler, "action.update_total"):
            action_map = {}
            self.frame_index += 1
            if not persons:
                return action_map
            _add_stat(stats, "stgcn_person_frames", len(persons))
            if not self.loaded or self.model is None or self.pose_model is None:
                # action 模組不可用時仍回傳 normal 結果，讓主流程不用分支處理。
                _add_stat(stats, "stgcn_disabled_frames")
                for person in persons:
                    track_id = person.get("track_id")
                    if track_id is None or int(track_id) < 0:
                        continue
                    action_map[int(track_id)] = {
                        "action": "normal",
                        "conf": 0.0,
                        "stgcn_conf": 0.0,
                        "alert": False,
                    }
                return action_map

            pose_kwargs = {
                "conf": 0.3,
                "stream": False,
                "verbose": False,
                "device": self.pose_device,
                "half": self.pose_half,
            }
            if self.pose_imgsz > 0:
                pose_kwargs["imgsz"] = self.pose_imgsz

            with profile_block(active_profiler, "action.pose_predict"):
                # 對整張 frame 跑 pose，再用 IoU 配對回主流程的 tracked person。
                pose_results = self.pose_model.predict(frame, **pose_kwargs)
            pose_result = pose_results[0] if pose_results else None
            pose_boxes = []
            with profile_block(active_profiler, "action.pose_parse"):
                if pose_result is not None and pose_result.boxes is not None and len(pose_result.boxes) > 0:
                    for pxyxy in pose_result.boxes.xyxy:
                        pose_boxes.append([int(c) for c in pxyxy])
            _add_stat(stats, "stgcn_pose_boxes", len(pose_boxes))

            with profile_block(active_profiler, "action.track_match_and_state"):
                # 逐一更新每個 tracked person 的骨架序列與最近一次動作結果。
                for person in persons:
                    track_id = person.get("track_id")
                    if track_id is None or int(track_id) < 0:
                        continue
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, person["box"])

                    if track_id not in self.track_history:
                        self.track_history[track_id] = deque(maxlen=self.window_size)
                        self.alert_counter[track_id] = 0
                        self.last_action[track_id] = ("normal", 0.0)
                        self.alert_action[track_id] = None
                        self.urination_history[track_id] = deque()
                        self.urination_positive_counts[track_id] = 0

                    best_idx = -1
                    best_iou = 0.0
                    for idx, pbox in enumerate(pose_boxes):
                        score = iou_xyxy([x1, y1, x2, y2], pbox)
                        if score > best_iou:
                            best_iou = score
                            best_idx = idx

                    if best_idx >= 0 and best_iou >= self.track_iou_threshold and pose_result is not None:
                        skeleton = self._extract_skeleton(pose_result, best_idx)
                        if skeleton is not None:
                            self.track_history[track_id].append(skeleton)
                            _add_stat(stats, "stgcn_pose_matches")
                    else:
                        _add_stat(stats, "stgcn_pose_unmatched")

                    should_predict = (
                        len(self.track_history[track_id]) == self.window_size and
                        (self.frame_index % self.predict_interval) == 0
                    )
                    if len(self.track_history[track_id]) == self.window_size:
                        _add_stat(stats, "stgcn_window_ready")
                    if should_predict:
                        # 序列滿窗且到達推理間隔時才跑 STGCN，降低每幀推理成本。
                        _add_stat(stats, "stgcn_predict_calls")
                        with torch.inference_mode():
                            action, conf = self._predict_action(
                                np.array(list(self.track_history[track_id])),
                                img_shape=frame.shape[:2],
                                profiler=active_profiler,
                            )
                            self.last_action[track_id] = (action, conf)
                            _add_stat(stats, f"stgcn_pred_{action}")
                            # littering 仍採即時門檻；urination 另走 10 秒內至少 8 秒的持續性確認。
                            if action == "littering" and conf >= self.action_threshold:
                                self.alert_counter[track_id] = self.alert_frames
                                self.alert_action[track_id] = action
                                _add_stat(stats, "stgcn_alerts")

                    action, conf = self.last_action[track_id]
                    urination_confirmed, urination_positive_sec, urination_observed_sec = (
                        self._record_urination_evidence(track_id, action, conf, fps, stats=stats)
                    )
                    if self._is_urination_action(action) and urination_confirmed:
                        already_alerting_urination = (
                            self.alert_counter[track_id] > 0 and
                            self.alert_action.get(track_id) == "urination"
                        )
                        self.alert_counter[track_id] = self.alert_frames
                        self.alert_action[track_id] = "urination"
                        if not already_alerting_urination:
                            _add_stat(stats, "stgcn_alerts")
                            _add_stat(stats, "stgcn_urination_confirmed")

                    reported_action = action
                    if self._is_urination_action(action) and not urination_confirmed:
                        reported_action = "normal"

                    action_result = {
                        "action": reported_action,
                        "raw_action": action,
                        # conf 舊欄位保留相容性；stgcn_conf 明確表示這是 STGCN 動作分類分數，不是 person bbox 分數。
                        "conf": conf,
                        "stgcn_conf": conf,
                        "urination_evidence_sec": urination_positive_sec,
                        "urination_observed_sec": urination_observed_sec,
                        "urination_required_sec": self.urination_min_sec,
                    }
                    if self.alert_counter[track_id] > 0:
                        # alert_frames 讓違規標記維持數幀，避免單幀分類閃爍。
                        self.alert_counter[track_id] -= 1
                        action_result["alert"] = True
                        action_result["action"] = self.alert_action.get(track_id) or reported_action
                    else:
                        action_result["alert"] = False
                        self.alert_action[track_id] = None
                    action_map[track_id] = action_result

            return action_map
