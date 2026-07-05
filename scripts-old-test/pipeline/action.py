# -*- coding: utf-8 -*-
# STGCN++ 動作辨識模組：從 person bbox 對應 pose keypoints，累積序列後只判斷 normal/urinate。
import os
import sys
import traceback
from collections import deque

import numpy as np
import torch
from ultralytics import YOLO
from pipeline.profiling import profile_block
from pipeline.paths import ensure_mmaction_on_path


ACTION_CLASSES = {0: "normal", 1: "urinate"}
URINATION_ACTIONS = {"urinate", "urination", "urinating"}
VIOLATION_ACTIONS = set(URINATION_ACTIONS)


def _add_stat(stats, key, amount=1):
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def _int_env(name, default):
    # 讀取整數環境變數；格式錯誤時回退預設值。
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)


def _float_env(name, default):
    # 讀取浮點環境變數；格式錯誤時回退預設值。
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


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

class STGCNActionModule:
    # 封裝模型載入、pose 擷取、骨架序列快取、STGCN 推理與 alert 維持。
    def __init__(
        self,
        pose_model_path,
        stgcn_weight_path,
        stgcn_config_path,
        # STGCN 判定為 urinate 後，conf 需 >= high 門檻（預設 0.5）且累積滿時間才會啟動違規警報。
        action_threshold=0.5,
        urinate_conf_high=None,
        urinate_conf_low=None,
        track_iou_threshold=0.2,
        window_size=100,
        alert_frames=35,
        urination_window_sec=8.0,
        urination_min_sec=5.0,
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
        # urinate 雙門檻（double thresholding / 遲滯）：
        #   conf >= high          → strong，開啟一段連續 urinate 區間；
        #   low <= conf < high    → weak，只有在 strong 已開啟的區間內才採信（補回信心暫降的幀）；
        #   預測為 normal          → 中斷該區間。
        # 可用 ACTION_URINATE_CONF_HIGH / ACTION_URINATE_CONF_LOW 覆寫。
        high = urinate_conf_high if urinate_conf_high is not None else _float_env(
            "ACTION_URINATE_CONF_HIGH", self.action_threshold
        )
        low = urinate_conf_low if urinate_conf_low is not None else _float_env(
            "ACTION_URINATE_CONF_LOW", 0.3
        )
        self.urinate_conf_high = float(high)
        self.urinate_conf_low = min(float(low), self.urinate_conf_high)
        # 【正規化】當前訓練 pkl（garbage_new_nohead_balanced.pkl）是對「人物裁切短片」跑 YOLO-Pose，
        # 存的是 crop 空間座標，img_shape 為裁切尺寸（如 478×286）。PreNormalize2D 以 img_shape
        # 中心做位移+縮放：訓練時人體填滿 crop → x_norm ≈ [-1,1]。若推論時傳全幀 img_shape
        # (1080×1920)，人體只佔一小角，歸一化結果完全不同 → 模型全輸出 normal。
        # 因此預設開啟 bbox 正規化（ACTION_BBOX_NORM=1）：用關鍵點外接框當 crop 尺寸，
        # 與訓練分布一致。若之後重新用全幀訓練，可設 ACTION_BBOX_NORM=0 關閉。
        self.bbox_normalize = _int_env("ACTION_BBOX_NORM", 1) != 0
        self.bbox_norm_pad = 0.15
        self.bbox_norm_conf = 0.3
        # 關鍵點時序補值/平滑：對 window 內低信心關鍵點做線性內插，再做輕量移動平均去抖。
        self.kp_smooth_enable = _int_env("ACTION_KP_SMOOTH", 1) != 0
        self.kp_smooth_window = max(1, _int_env("ACTION_KP_SMOOTH_WIN", 3))
        self.kp_interp_conf = _float_env("ACTION_KP_INTERP_CONF", 0.3)
        self.window_size = int(window_size)
        # 【時間 stride 對齊】訓練 UniformSampleFrames(clip_len=100) 是把「整段 clip」稀疏取樣成
        # 100 幀；推論若只餵最近 100 連續幀(stride=1, ~3.3s)會與較長訓練 clip 的時間覆蓋/速度分布
        # 不一致。ACTION_SAMPLE_SPAN 設成 > window_size 時，改為從最近 span 幀「等間隔取樣」window_size
        # 幀，擴大時間覆蓋並貼近訓練取樣方式。預設 = window_size（行為不變，可 A/B 開啟）。
        self.sample_span = max(self.window_size, _int_env("ACTION_SAMPLE_SPAN", self.window_size))
        self.alert_frames = int(alert_frames)
        self.urination_window_sec = max(0.0, float(urination_window_sec))
        self.urination_min_sec = max(0.0, float(urination_min_sec))
        if self.urination_min_sec > self.urination_window_sec:
            raise ValueError("urination_min_sec cannot exceed urination_window_sec")
        # 【top-p 證據累積】ACTION_URINATE_TOPP=1（預設）：確認機制從「遲滯二值 positive 秒數」
        # 改為「機率質量累積」——每幀累積 urinate 類別機率 p_t（2-class softmax 下 normal 幀
        # 貢獻 1-conf，不再歸零），視窗內 Σ p_t / fps ≥ ACTION_URINATE_TOPP_MASS（機率·秒，
        # 預設 = urination_min_sec）即確認。中途誤判 normal 只「稀釋」證據、不再「中斷」區間，
        # 對信心波動的緩衝更平滑。ACTION_URINATE_TOPP=0 退回舊遲滯二值模式。
        self.urinate_topp_enabled = _int_env("ACTION_URINATE_TOPP", 1) != 0
        # 質量門檻校準：binary 模式把一個 conf=0.55 的 strong 幀計滿 1.0 秒，top-p 只計 0.55，
        # 等效召回點 ≈ urination_min_sec × E[conf|urinate] ≈ 5.0 × 0.7 = 3.5 機率·秒。
        # FP 安全：normal 場景 p_t 多 < floor(0.2) 不累積，M ≈ 0，降門檻不增 normal FP
        #（實測 normal_case108 M ≈ 0.9 << 3.5）。
        self.urinate_topp_mass = max(0.0, _float_env(
            "ACTION_URINATE_TOPP_MASS", 0.7 * self.urination_min_sec
        ))
        # 機率下限（雜訊閘）：p_t < floor 不累積，避免 normal 場景低基線機率緩慢堆積成 FP。
        self.urinate_topp_floor = min(1.0, max(0.0, _float_env("ACTION_URINATE_TOPP_FLOOR", 0.2)))
        # ACTION_EVIDENCE_DEBUG=1：定期印出各 track 的證據累積狀態（診斷確認門檻用）。
        self._evidence_debug = _int_env("ACTION_EVIDENCE_DEBUG", 0) != 0
        self.predict_interval = max(1, _int_env("ACTION_PREDICT_INTERVAL", 1))
        self.frame_index = 0
        self.track_history = {}
        self.urination_history = {}
        # float 證據累積：binary 模式 = 視窗內 positive 幀數；top-p 模式 = 視窗內 Σ p_t。
        self.urination_evidence = {}
        # urinate 雙門檻遲滯狀態：track 目前是否處於 strong 觸發的連續區間（僅 binary 模式用）。
        self.urination_active = {}
        self.alert_counter = {}
        self.alert_action = {}
        self.last_action = {}
        # 確認 urinate 事件記錄(events.jsonl 來源):每個確認 episode 一筆。
        self._urinate_events = []
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

            # 使用專案內 mmaction2，避免吃到系統其他版本(路徑集中於 pipeline.paths)。
            ensure_mmaction_on_path()
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

    def _sample_window(self, skeleton_list):
        # 從 buffer（最多 sample_span 幀）等間隔取出 window_size 幀，貼近訓練的
        # UniformSampleFrames(test_mode) 取樣（整段均勻覆蓋）。當 buffer 長度 == window_size
        # 時退化為原本的「連續 window_size 幀」。
        n = len(skeleton_list)
        if n <= self.window_size:
            return np.array(skeleton_list)
        idx = np.linspace(0, n - 1, self.window_size)
        idx = np.round(idx).astype(int)
        idx = np.clip(idx, 0, n - 1)
        return np.array([skeleton_list[j] for j in idx])

    @staticmethod
    def _moving_average_time(arr, k):
        # arr: (T, V)；沿時間軸做邊緣感知移動平均（window k，奇數），降低關鍵點抖動。
        T = arr.shape[0]
        if k < 3 or T < k:
            return arr
        pad = k // 2
        kernel = np.ones(k, dtype=np.float32) / float(k)
        padded = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        out = np.empty_like(arr)
        for v in range(arr.shape[1]):
            out[:, v] = np.convolve(padded[:, v], kernel, mode="valid")
        return out

    def _interpolate_and_smooth(self, skeleton_sequence):
        # 對整個 window 的骨架序列做時序補值與平滑：
        #   1. 每個關節在信心 < kp_interp_conf 的幀，用同一關節在有效幀間做線性內插；
        #   2. 全程都無效的關節保持原值（多半為 0），不硬補；
        #   3. 補完後對 x/y 做輕量移動平均去抖。confidence 通道保持不變。
        seq = np.asarray(skeleton_sequence, dtype=np.float32).copy()
        if seq.ndim != 3 or seq.shape[0] < 2:
            return seq
        T, V, _ = seq.shape
        conf = seq[..., 2]
        valid = conf >= self.kp_interp_conf
        xs = seq[..., 0]
        ys = seq[..., 1]
        t_idx = np.arange(T)
        for v in range(V):
            m = valid[:, v]
            n_valid = int(m.sum())
            if n_valid == 0 or n_valid == T:
                continue
            xs[:, v] = np.interp(t_idx, t_idx[m], xs[m, v])
            ys[:, v] = np.interp(t_idx, t_idx[m], ys[m, v])
        if self.kp_smooth_enable and self.kp_smooth_window >= 3:
            xs = self._moving_average_time(xs, self.kp_smooth_window)
            ys = self._moving_average_time(ys, self.kp_smooth_window)
        seq[..., 0] = xs
        seq[..., 1] = ys
        return seq

    def _bbox_normalize(self, keypoints, scores, img_shape):
        # 以整個 window 內可信關鍵點的外接框，把骨架平移/縮放成「人物填滿畫面」。
        # 回傳 (平移後的關鍵點, (h, w))，與訓練時裁切短片的正規化方式一致。
        mask = scores > self.bbox_norm_conf
        if int(np.count_nonzero(mask)) < 3:
            # 可信點太少時退回整張 frame 正規化，避免外接框不穩定。
            return keypoints, img_shape
        xs = keypoints[..., 0][mask]
        ys = keypoints[..., 1][mask]
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        pad = self.bbox_norm_pad
        x0 -= w * pad
        y0 -= h * pad
        w *= (1.0 + 2.0 * pad)
        h *= (1.0 + 2.0 * pad)
        shifted = keypoints.copy()
        shifted[..., 0] = keypoints[..., 0] - x0
        shifted[..., 1] = keypoints[..., 1] - y0
        return shifted, (int(round(h)), int(round(w)))

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
            # 先做時序補值/平滑，再進入正規化與 MMACTION2 pipeline。
            skeleton_sequence = self._interpolate_and_smooth(skeleton_sequence)
            keypoints = skeleton_sequence[..., :2].astype(np.float32)
            scores = skeleton_sequence[..., 2].astype(np.float32)
            scores[:, :5] = 0.0  # suppress head nodes (nose/eyes/ears) to match training
            if self.bbox_normalize:
                keypoints, img_shape = self._bbox_normalize(keypoints, scores, img_shape)
            active_profiler = profiler if profiler is not None else self.profiler
            with profile_block(active_profiler, profile_name):
                # Training pkl stores (num_person=2, T, V, 2); second slot is zeros when only
                # one person is tracked. Must match that format at inference or the model gives
                # wrong predictions (single-person (1,V,2) vs padded (2,V,2) behave differently
                # due to cross-person edges in the STGCN graph).
                T, V = keypoints.shape[:2]
                kp_pad = np.zeros((1, V, 2), dtype=np.float32)
                sc_pad = np.zeros((1, V), dtype=np.float32)
                pose_results = []
                for i in range(T):
                    pose_results.append(
                        {
                            "keypoints": np.concatenate([keypoints[i : i + 1], kp_pad], axis=0),
                            "keypoint_scores": np.concatenate([scores[i : i + 1], sc_pad], axis=0),
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

    @staticmethod
    def _normalize_track_id_set(track_ids):
        normalized = set()
        if track_ids is None:
            return normalized
        for track_id in track_ids:
            try:
                normalized.add(int(track_id))
            except (TypeError, ValueError):
                continue
        return normalized

    def _clear_urination_state(self, track_id):
        history = self.urination_history.get(track_id)
        if history is not None:
            history.clear()
        self.urination_evidence[track_id] = 0.0
        self.urination_active[track_id] = False
        action, _ = self.last_action.get(track_id, ("normal", 0.0))
        if self._is_urination_action(action):
            self.last_action[track_id] = ("normal", 0.0)
        if self.alert_action.get(track_id) == "urinate":
            self.alert_counter[track_id] = 0
            self.alert_action[track_id] = None

    def _resolve_urination_positive(self, track_id, action, conf, stats=None):
        # double thresholding（遲滯）：以 strong 幀開啟連續區間，weak 幀只在區間內採信，
        # 預測為 normal 則中斷區間。回傳本幀是否計為 positive 證據。
        is_urinate = self._is_urination_action(action)
        conf = float(conf)
        active = self.urination_active.get(track_id, False)
        if is_urinate and conf >= self.urinate_conf_high:
            active = True
            positive = True
            _add_stat(stats, "stgcn_urinate_strong")
        elif is_urinate and conf >= self.urinate_conf_low:
            # weak candidate：僅在 strong 已觸發的區間內補回，否則視為雜訊丟棄。
            positive = active
            _add_stat(stats, "stgcn_urinate_weak_kept" if positive else "stgcn_urinate_weak_dropped")
        else:
            # 預測為 normal（argmax=normal）才中斷區間；urinate 但低於 low 門檻僅不計分、不中斷。
            if not is_urinate:
                active = False
            positive = False
        self.urination_active[track_id] = active
        return positive

    def _urinate_probability(self, action, conf):
        # 從 (argmax action, conf) 還原 2-class softmax 的 urinate 機率：
        #   pred=urinate → p = conf；pred=normal → p = 1 - conf。
        # 防護：conf < 0.5 的 normal 是 sentinel/fallback（模型未載入回傳 ("normal", 0.0)、
        # 錯誤路徑 ("normal", 0.01)），非真 softmax 輸出，不可解讀成 p >= 0.5 的證據 → 回 0。
        conf = float(conf)
        if self._is_urination_action(action):
            return conf
        if conf >= 0.5:
            return 1.0 - conf
        return 0.0

    def _record_urination_evidence(self, track_id, action, conf, fps, stats=None):
        # urinate 需在最近 urination_window_sec 內累積足夠證據才確認，避免單次 STGCN 閃爍誤報。
        # top-p 模式：累積 urinate 機率質量（機率·秒），誤判 normal 只稀釋、不中斷；
        # binary 模式（ACTION_URINATE_TOPP=0）：沿用遲滯二值 positive 幀秒數。
        fps_value = _safe_fps(fps)
        now_sec = self.frame_index / fps_value
        if self.urinate_topp_enabled:
            p_t = self._urinate_probability(action, conf)
            contribution = p_t if p_t >= self.urinate_topp_floor else 0.0
            required_sec = self.urinate_topp_mass
        else:
            positive = self._resolve_urination_positive(track_id, action, conf, stats=stats)
            contribution = 1.0 if positive else 0.0
            required_sec = self.urination_min_sec
        history = self.urination_history.setdefault(track_id, deque())
        if track_id not in self.urination_evidence:
            self.urination_evidence[track_id] = 0.0

        history.append((now_sec, contribution))
        if contribution > 0.0:
            self.urination_evidence[track_id] += contribution
            _add_stat(stats, "stgcn_urination_evidence_frames")

        cutoff_sec = now_sec - self.urination_window_sec
        while history and history[0][0] < cutoff_sec:
            _, stale_contribution = history.popleft()
            if stale_contribution > 0.0:
                self.urination_evidence[track_id] = max(
                    0.0,
                    self.urination_evidence[track_id] - stale_contribution,
                )

        positive_sec = self.urination_evidence[track_id] / fps_value
        observed_sec = min(self.urination_window_sec, len(history) / fps_value)
        confirmed = required_sec <= 0.0 or positive_sec >= required_sec
        if self._evidence_debug and (self.frame_index % 30 == 0 or confirmed):
            print(
                f"[EVID fi={self.frame_index} tid={track_id}] act={action} conf={float(conf):.2f} "
                f"M={positive_sec:.2f}/{required_sec:.2f} obs={observed_sec:.1f} confirmed={confirmed}"
            )
        return confirmed, positive_sec, observed_sec

    def detect_persons(self, frame, profiler=None, stats=None):
        """YOLO-Pose 同時負責 person 偵測 + 追蹤 + 關鍵點擷取（單次推理，免 IoU 配對）。

        回傳 (persons, frame_skeletons)：
          persons         : [{'box': xyxy(np.float32), 'track_id': int, 'cls': 'person',
                              'mask_poly': None, 'pose_conf': float}]，作為主流程唯一的 person 來源。
          frame_skeletons : {track_id: skeleton(17,3) ndarray}，供 classify_actions 對齊使用。
        並推進 self.frame_index（每幀一次，維持 urinate 時間累積連續）。
        """
        active_profiler = profiler if profiler is not None else self.profiler
        self.frame_index += 1
        persons = []
        frame_skeletons = {}
        if not self.loaded or self.pose_model is None:
            # pose 不可用：回傳空 person，由主流程決定是否 fallback 到 YOLO-Seg。
            return persons, frame_skeletons

        pose_kwargs = {
            "conf": 0.3,
            "persist": True,
            "verbose": False,
            "device": self.pose_device,
            "half": self.pose_half,
            "tracker": "botsort.yaml",
        }
        if self.pose_imgsz > 0:
            pose_kwargs["imgsz"] = self.pose_imgsz

        with profile_block(active_profiler, "action.pose_track"):
            # 在整張 frame 上偵測+追蹤 person，keypoints 為全畫面絕對座標（與訓練分布一致）。
            pose_results = self.pose_model.track(frame, **pose_kwargs)
        pose_result = pose_results[0] if pose_results else None
        if pose_result is None or pose_result.boxes is None or pose_result.boxes.id is None:
            # tracker 尚未指派 id（如首幀或全低信心）→ 本幀無可用 person。
            return persons, frame_skeletons

        with profile_block(active_profiler, "action.pose_parse"):
            boxes_xyxy = pose_result.boxes.xyxy.cpu().numpy()
            track_ids = pose_result.boxes.id.cpu().numpy().astype(int)
            if pose_result.boxes.conf is not None:
                box_conf = pose_result.boxes.conf.cpu().numpy()
            else:
                box_conf = np.ones(len(track_ids), dtype=np.float32)
            for idx in range(len(track_ids)):
                track_id = int(track_ids[idx])
                if track_id < 0:
                    continue
                skeleton = self._extract_skeleton(pose_result, idx)
                if skeleton is None:
                    continue
                frame_skeletons[track_id] = skeleton
                persons.append({
                    "box": boxes_xyxy[idx].astype(np.float32),
                    "track_id": track_id,
                    "cls": "person",
                    # pose 沒有 segmentation polygon；holding/tracker 會自動退回 bbox 錨點。
                    "mask_poly": None,
                    "pose_conf": float(box_conf[idx]),
                })
        _add_stat(stats, "stgcn_pose_boxes", len(persons))
        return persons, frame_skeletons

    def classify_actions(
        self,
        frame,
        persons,
        frame_skeletons,
        fps=30.0,
        blocked_urination_track_ids=None,
        profiler=None,
        stats=None,
    ):
        """以 detect_persons 取得的 persons + frame_skeletons 跑 STGCN。

        骨架已依 track_id 對齊（無需 IoU），回傳
        {track_id: {'action','raw_action','conf','stgcn_conf','alert', ...}}。
        """
        active_profiler = profiler if profiler is not None else self.profiler
        blocked_urination_track_ids = self._normalize_track_id_set(blocked_urination_track_ids)
        frame_skeletons = frame_skeletons or {}
        action_map = {}
        with profile_block(active_profiler, "action.classify_total"):
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

            with profile_block(active_profiler, "action.track_state"):
                # 逐一更新每個 tracked person 的骨架序列與最近一次動作結果。
                for person in persons:
                    track_id = person.get("track_id")
                    if track_id is None or int(track_id) < 0:
                        continue
                    track_id = int(track_id)

                    if track_id not in self.track_history:
                        self.track_history[track_id] = deque(maxlen=self.sample_span)
                        self.alert_counter[track_id] = 0
                        self.last_action[track_id] = ("normal", 0.0)
                        self.alert_action[track_id] = None
                        self.urination_history[track_id] = deque()
                        self.urination_evidence[track_id] = 0.0
                        self.urination_active[track_id] = False

                    urination_blocked = track_id in blocked_urination_track_ids
                    if urination_blocked:
                        self._clear_urination_state(track_id)

                    # 骨架由 detect_persons 依 track_id 直接對齊，免 IoU 配對。
                    skeleton = frame_skeletons.get(track_id)
                    if skeleton is not None:
                        self.track_history[track_id].append(skeleton)
                        _add_stat(stats, "stgcn_pose_matches")
                    else:
                        _add_stat(stats, "stgcn_pose_unmatched")

                    buffer_len = len(self.track_history[track_id])
                    should_predict = (
                        buffer_len >= self.window_size and
                        (self.frame_index % self.predict_interval) == 0
                    )
                    if buffer_len >= self.window_size:
                        _add_stat(stats, "stgcn_window_ready")
                    if should_predict:
                        # 序列滿窗且到達推理間隔時才跑 STGCN，降低每幀推理成本。
                        # 從 buffer（最多 sample_span 幀）等間隔取 window_size 幀，對齊訓練取樣。
                        _add_stat(stats, "stgcn_predict_calls")
                        sampled = self._sample_window(list(self.track_history[track_id]))
                        with torch.inference_mode():
                            action, conf = self._predict_action(
                                sampled,
                                img_shape=frame.shape[:2],
                                profiler=active_profiler,
                            )
                            _add_stat(stats, f"stgcn_pred_{action}")
                            if urination_blocked and self._is_urination_action(action):
                                _add_stat(stats, "stgcn_urinate_blocked_on_vehicle")
                                action, conf = "normal", 0.0
                            self.last_action[track_id] = (action, conf)
                    action, conf = self.last_action[track_id]
                    urination_confirmed, urination_positive_sec, urination_observed_sec = (
                        self._record_urination_evidence(track_id, action, conf, fps, stats=stats)
                    )
                    if self._is_urination_action(action) and urination_confirmed:
                        already_alerting_urination = (
                            self.alert_counter[track_id] > 0 and
                            self.alert_action.get(track_id) == "urinate"
                        )
                        self.alert_counter[track_id] = self.alert_frames
                        self.alert_action[track_id] = "urinate"
                        if not already_alerting_urination:
                            _add_stat(stats, "stgcn_alerts")
                            _add_stat(stats, "stgcn_urinate_confirmed")
                            # 新確認的 urinate episode:記一筆事件(frame_index 為 action 模組
                            # 內部幀計數,作為相對時間戳)。
                            self._urinate_events.append({
                                "track_id": int(track_id),
                                "frame_index": int(self.frame_index),
                                "conf": float(conf),
                                "evidence_sec": float(urination_positive_sec),
                            })

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
                        "urination_required_sec": (
                            self.urinate_topp_mass if self.urinate_topp_enabled
                            else self.urination_min_sec
                        ),
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

    def get_urinate_events(self):
        # 確認 urinate 事件(每個確認 episode 一筆)。events.jsonl 來源。
        return list(self._urinate_events)
