from ultralytics import YOLO, RTDETR
import os
import cv2
from tqdm import tqdm
import argparse
import subprocess
from collections import deque
import numpy as np
import torch
import sys
import warnings
import torch.nn.functional as F
import json
import traceback

warnings.filterwarnings('ignore')

# === STGCN++ 相關設定 ===
WINDOW_SIZE = 30  # STGCN++ 需要固定長度的幀數
ACTION_THRESHOLD = 0.3  # 動作判斷的信心度閾值
STGCN_NUM_FRAMES = 30  # STGCN++ 模型輸入幀數
DEFAULT_FG_MASK_SCALE = 0.5
DEFAULT_MOTION_DIFF_THRESHOLD = 10
DEFAULT_MOTION_DILATE_ITERATIONS = 2


def _check_motion(fg_mask, coords, threshold, mask_scale=1.0):
    # 與 main.py 相同的 motion 檢查：前景像素比例達到門檻才算有動作。
    x1, y1, x2, y2 = coords
    scale = float(mask_scale or 1.0)
    if scale != 1.0:
        x1 = int(np.floor(float(x1) * scale))
        y1 = int(np.floor(float(y1) * scale))
        x2 = int(np.ceil(float(x2) * scale))
        y2 = int(np.ceil(float(y2) * scale))
    h, w = fg_mask.shape[:2]

    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    mask_crop = fg_mask[y1:y2, x1:x2]
    if mask_crop.size == 0:
        return False

    white_pixels = np.sum(mask_crop == 255)
    total_pixels = mask_crop.size
    return (white_pixels / total_pixels) >= threshold


class MotionMaskBuilder:
    # 與 main.py 相同的前景 mask 建立器：預設用 temporal diff，必要時可切回 MOG2。
    def __init__(self, mode="temporal", scale_factor=DEFAULT_FG_MASK_SCALE,
                 diff_threshold=DEFAULT_MOTION_DIFF_THRESHOLD,
                 dilate_iterations=DEFAULT_MOTION_DILATE_ITERATIONS,
                 mog2_history=300, mog2_var_threshold=25, mog2_detect_shadows=True):
        self.mode = str(mode or "temporal").lower()
        self.scale_factor = float(scale_factor or 1.0)
        self.diff_threshold = int(diff_threshold)
        self.dilate_iterations = max(int(dilate_iterations or 0), 0)
        self.prev_gray = None
        self.temporal_kernel = np.ones((3, 3), dtype=np.uint8) if self.dilate_iterations > 0 else None
        self.back_sub = None

        if self.mode == "mog2":
            self.back_sub = cv2.createBackgroundSubtractorMOG2(
                history=int(mog2_history),
                varThreshold=float(mog2_var_threshold),
                detectShadows=bool(mog2_detect_shadows),
            )
        elif self.mode != "temporal":
            raise ValueError(f"Unsupported motion mask mode: {mode}")

    def build(self, frame):
        if self.mode == "mog2":
            return self._build_mog2(frame)
        return self._build_temporal(frame)

    def _scaled_frame(self, frame):
        if self.scale_factor == 1.0:
            return frame
        return cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)

    def _build_mog2(self, frame):
        mask_frame = self._scaled_frame(frame)
        fg_mask = self.back_sub.apply(mask_frame, learningRate=0.005)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        return fg_mask

    def _build_temporal(self, frame):
        mask_frame = self._scaled_frame(frame)
        gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return np.full(gray.shape, 255, dtype=np.uint8)

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray

        _, motion_mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        if self.temporal_kernel is not None:
            motion_mask = cv2.dilate(motion_mask, self.temporal_kernel, iterations=self.dilate_iterations)
        return motion_mask

def iou_xyxy(box_a, box_b):
    """計算兩個 xyxy box 的 IoU。"""
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
    if union <= 0:
        return 0.0
    return inter_area / union

# 模擬的 STGCN++ 推論類別
class STGCN_Predictor:
    def __init__(self, weight_path, config_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        print(f"📥 準備載入真實 STGCN++ 模型至 {device}...")
        self.device = torch.device(device)
        self.loaded = False
        self.mode = "fallback"
        self.use_bbox_normalization = True
        self._logged_mmaction_error = False
        self.last_error = None
        
        # 定義動作類別對應字典 (請依據你訓練時的 label 設定修改！)
        self.action_classes = {
            0: "normal",
            1: "littering",
            # 2: "other_action"...
        }
        
        try:
            self.model = None
            # 優先使用 mmaction2 的 config + checkpoint 載入（支援 MMEngine 訓練檔）
            if config_path and os.path.exists(config_path) and weight_path and os.path.exists(weight_path):
                try:
                    # transformers 新舊版相容：某些 mmaction/mmengine 依賴仍從 modeling_utils 匯入舊函式
                    try:
                        import transformers.modeling_utils as _tf_modeling_utils
                        from transformers import pytorch_utils as _tf_pt_utils

                        compat_symbols = [
                            "apply_chunking_to_forward",
                            "find_pruneable_heads_and_indices",
                            "prune_linear_layer",
                        ]
                        for _sym in compat_symbols:
                            if not hasattr(_tf_modeling_utils, _sym) and hasattr(_tf_pt_utils, _sym):
                                setattr(_tf_modeling_utils, _sym, getattr(_tf_pt_utils, _sym))
                    except Exception:
                        # 若本機無 transformers 或不需要修補，直接略過
                        pass

                    project_root = os.path.dirname(os.path.abspath(__file__))
                    mmaction_repo = os.path.join(project_root, "mmaction2")
                    if mmaction_repo not in sys.path:
                        sys.path.insert(0, mmaction_repo)
                    from mmaction.apis import init_recognizer, inference_skeleton

                    device_str = str(self.device)
                    if device_str == "cuda":
                        device_str = "cuda:0"
                    # PyTorch 2.6+ 預設 weights_only=True，會讓部分 mmengine checkpoint 載入失敗
                    original_torch_load = torch.load
                    def _compat_torch_load(*args, **kwargs):
                        kwargs.setdefault("weights_only", False)
                        return original_torch_load(*args, **kwargs)
                    torch.load = _compat_torch_load
                    try:
                        self.model = init_recognizer(config_path, weight_path, device=device_str)
                    finally:
                        torch.load = original_torch_load
                    self.inference_skeleton = inference_skeleton
                    self.mode = "mmaction"
                    self.use_bbox_normalization = False
                    self.loaded = True
                    print("✓ 以 mmaction2 (config + checkpoint) 載入 STGCN++ 成功")
                    return
                except Exception as e:
                    print(f"⚠️ mmaction2 載入失敗，改嘗試 TorchScript/checkpoint：{e}")

            if weight_path and os.path.exists(weight_path):
                # 先嘗試 TorchScript（最通用，不需要原始模型 class）
                try:
                    self.model = torch.jit.load(weight_path, map_location=self.device)
                    self.model.to(self.device)
                    self.model.eval()
                    self.mode = "torchscript"
                    self.loaded = True
                    print("✓ 以 TorchScript 方式載入 STGCN++ 成功")
                except Exception:
                    # 再嘗試一般 checkpoint（需自帶可直接推論的 model 物件）
                    checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and "model" in checkpoint and hasattr(checkpoint["model"], "eval"):
                        self.model = checkpoint["model"].to(self.device)
                        self.model.eval()
                        self.mode = "checkpoint_model"
                        self.loaded = True
                        print("✓ 從 checkpoint['model'] 載入 STGCN++ 成功")
                    else:
                        print("⚠️ STGCN 權重非 TorchScript，也未包含可直接推論的 model 物件。")
                        print("   請將 STGCN 匯出成 TorchScript，或在此檔案補上模型架構並 load_state_dict。")
            else:
                print("⚠️ 找不到 STGCN 權重檔，將以安全回退模式執行。")
                
        except Exception as e:
            print(f"❌ 載入模型時出錯：{e}")

    def predict_action(self, skeleton_sequence):
        """
        :param skeleton_sequence: numpy array, shape (T, V, C) -> (30, 17, 3)
        :return: (action_name, confidence)
        """
        if not self.loaded or self.model is None:
            # 模型未載入成功時的保護機制
            return "normal", 0.0

        try:
            self.last_error = None
            if self.mode == "mmaction":
                # 使用 mmaction 官方 skeleton inference 介面，避免 fake_anno 欄位差異造成失敗
                pose_results = []
                t = skeleton_sequence.shape[0]
                for i in range(t):
                    pose_results.append({
                        "keypoints": skeleton_sequence[i:i + 1, :, :2].astype(np.float32),
                        "keypoint_scores": skeleton_sequence[i:i + 1, :, 2].astype(np.float32),
                    })
                result = self.inference_skeleton(self.model, pose_results, img_shape=(1080, 1920))
                pred_score = result.pred_score.detach().cpu().numpy()
                action_idx = int(np.argmax(pred_score))
                confidence = float(pred_score[action_idx])
                action_name = self.action_classes.get(action_idx, "unknown")
                return action_name, confidence

            # === 1. 前處理：維度轉換 ===
            # 目前 skeleton_sequence shape: (T, V, C) = (30, 17, 3)
            # 我們需要將其轉為 PyTorch tensor
            data = torch.tensor(skeleton_sequence, dtype=torch.float32)
            
            # 將維度從 (T, V, C) 轉換為 (C, T, V)
            # permute(2, 0, 1) 代表將原本的 index 2 (C) 放第一位，index 0 (T) 放第二位，index 1 (V) 放第三位
            data = data.permute(2, 0, 1) 
            
            # 增加 Batch (N) 和 Person (M) 維度，變成 (N, C, T, V, M)
            # unsqueeze(0) 在最前面加一維 -> (1, C, T, V)
            # unsqueeze(-1) 在最後面加一維 -> (1, C, T, V, 1)
            data = data.unsqueeze(0).unsqueeze(-1) 
            
            # 將資料送入 GPU/CPU
            data = data.to(self.device)
            
            # === 2. 模型推論 ===
            with torch.no_grad(): # 推論時不需要計算梯度
                logits = self.model(data) 
                
                # 通常 STGCN 輸出形狀為 (N, num_classes)，若是 (N, num_classes, T, V, M) 需做 mean pooling
                if logits.dim() > 2:
                    logits = logits.mean(dim=[2, 3, 4]) # 將多餘維度平均掉
                
                # 透過 Softmax 將 logits 轉換為 0~1 的機率分佈
                probs = F.softmax(logits, dim=1)
                
            # === 3. 後處理：取得最高機率的動作 ===
            # 取得最大機率的值與對應的 index
            max_prob, max_index = torch.max(probs, dim=1)
            
            confidence = max_prob.item()
            action_idx = max_index.item()
            
            # 透過字典轉換為文字標籤
            action_name = self.action_classes.get(action_idx, "unknown")
            
            return action_name, confidence

        except Exception as e:
            self.last_error = repr(e)
            print(f"⚠️ 模型推論出錯：{repr(e)}")
            if self.mode == "mmaction" and not self._logged_mmaction_error:
                self._logged_mmaction_error = True
                print("⚠️ mmaction 推論錯誤詳情（僅顯示一次）:")
                print(traceback.format_exc())
            return "normal", 0.0
            
# 從 YOLO-Pose 模型提取骨架
def extract_skeleton_from_pose_model(frame, pose_results, person_idx=0):
    """
    從 YOLO-Pose 結果提取骨架關鍵點
    :param frame: 輸入幀
    :param pose_results: YOLO-Pose 預測結果
    :param person_idx: 人物索引
    :return: 骨架關鍵點 (V, 3) - V個關節點的 (x, y, conf)
    """
    try:
        if hasattr(pose_results, "keypoints") and pose_results.keypoints is not None:
            keypoints = pose_results.keypoints
            if keypoints.xy is not None and len(keypoints.xy) > person_idx:
                # keypoints.xy: (N, V, 2), keypoints.conf: (N, V)
                kpts = keypoints.xy[person_idx].cpu().numpy()
                if hasattr(keypoints, "conf") and keypoints.conf is not None:
                    conf = keypoints.conf[person_idx].cpu().numpy().reshape(-1, 1)
                else:
                    conf = np.ones((kpts.shape[0], 1), dtype=np.float32)
                skeleton = np.hstack([kpts, conf]).astype(np.float32)
                return skeleton
    except Exception as e:
        print(f"⚠️  提取骨架出錯：{e}")
    
    # 回退：返回零骨架
    return np.zeros((17, 3))

# ==========================================

COLOR_DICT = {
    'person': (0, 255, 0),    
    'litter': (0, 0, 255),    
    'vehicle': (255, 0, 0),   
    'littering_alert': (0, 0, 255) # 亂丟垃圾的警告色 (紅色)
}

# ==================== 配置 ====================
# 檢查並設定模型路徑
def find_model_path(model_name, search_paths=None):
    """搜尋模型文件"""
    if search_paths is None:
        search_paths = [
            '.',
            './resources',
            './modules_weight',
            './test_ocr',
            '/home/se_copilot/trashProject',
        ]
    if model_name and os.path.isabs(model_name) and os.path.exists(model_name):
        return os.path.abspath(model_name)
    for path in search_paths:
        full_path = os.path.join(path, model_name)
        if os.path.exists(full_path):
            return os.path.abspath(full_path)
    return model_name  # 回傳原始名稱，由 YOLO 自行處理

# 模型配置
YOLO_MODEL_PATH = find_model_path('best-yolo-seg_v3.pt')
RTDETR_MODEL_PATH = find_model_path('best-rtdetr-seg.pt')
STGCN_WEIGHT_PATH = find_model_path('/home/se_copilot/trashProject/mmaction2/work_dirs/stgcnpp_garbage_v1/best_acc_top1_epoch_13.pth')
POSE_MODEL_PATH = find_model_path('yolo26x-pose.pt')
STGCN_CONFIG_PATH = find_model_path('mmaction2/configs/skeleton/stgcnpp/custom_trash_stgcnpp.py')
PADDLEOCR_IMAGE_PATH = find_model_path('test5.png')

MODULE_NAMES = ("yolo", "rtdetr", "stgcn", "pose", "paddleocr")


def add_module_toggle(parser, module_name, label):
    """加入 --enable-* / --disable-* / --no-* 開關，預設全部開啟。"""
    dest = f"enable_{module_name}"
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--enable-{module_name}",
        dest=dest,
        action="store_true",
        default=True,
        help=f"啟用 {label}（預設）",
    )
    group.add_argument(
        f"--disable-{module_name}",
        f"--no-{module_name}",
        dest=dest,
        action="store_false",
        help=f"關閉 {label}",
    )


def apply_only_selection(args):
    if not args.only:
        return
    selected = set(args.only)
    for module_name in MODULE_NAMES:
        setattr(args, f"enable_{module_name}", module_name in selected)


def model_class_name(model, cls_id):
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        raw_name = names.get(int(cls_id), str(cls_id))
    elif isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
        raw_name = names[int(cls_id)]
    else:
        raw_name = str(cls_id)
    return str(raw_name).strip().lower()


def empty_module_status(enabled):
    if enabled:
        return {"enabled": True, "loaded": False, "success": False, "detail": "", "error": None}
    return {"enabled": False, "loaded": False, "success": False, "detail": "已關閉", "error": None}


def count_result_boxes(results, model, target_names=None):
    total = 0
    target_total = 0
    max_conf = 0.0
    class_counts = {}

    if target_names is not None:
        target_names = {str(name).lower() for name in target_names}

    for result in results or []:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [0.0] * len(classes)
        total += len(classes)
        for cls_id, conf in zip(classes, confs):
            class_name = model_class_name(model, cls_id)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            max_conf = max(max_conf, float(conf))
            if target_names is None or class_name in target_names:
                target_total += 1

    return {
        "total": total,
        "target_total": target_total,
        "max_conf": max_conf,
        "class_counts": class_counts,
    }


def draw_model_boxes(frame, results, model, color, prefix, target_names=None):
    if target_names is not None:
        target_names = {str(name).lower() for name in target_names}

    for result in results or []:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [0.0] * len(classes)
        for box, cls_id, conf in zip(boxes, classes, confs):
            class_name = model_class_name(model, cls_id)
            if target_names is not None and class_name not in target_names:
                continue
            x1, y1, x2, y2 = map(int, box)
            label = f"{prefix}:{class_name} {float(conf):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_motion_filtered_litter_boxes(frame, results, model, color, prefix, fg_mask,
                                      fg_mask_scale=1.0,
                                      motion_threshold=0.25,
                                      core_motion_threshold=0.3):
    kept_count = 0
    if fg_mask is None:
        return kept_count

    for result in results or []:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else [0.0] * len(classes)
        for box, cls_id, conf in zip(boxes, classes, confs):
            class_name = model_class_name(model, cls_id)
            if class_name != 'litter':
                continue

            x1, y1, x2, y2 = map(int, box)
            if not _check_motion(fg_mask, (x1, y1, x2, y2), motion_threshold, mask_scale=fg_mask_scale):
                continue

            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            if width >= 8 and height >= 8:
                core_x1 = int(x1 + 0.2 * width)
                core_y1 = int(y1 + 0.2 * height)
                core_x2 = int(x2 - 0.2 * width)
                core_y2 = int(y2 - 0.2 * height)
                if not _check_motion(fg_mask, (core_x1, core_y1, core_x2, core_y2), core_motion_threshold, mask_scale=fg_mask_scale):
                    continue

            label = f"{prefix}:{class_name} {float(conf):.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            kept_count += 1

    return kept_count


def append_pose_skeleton_for_stgcn(track_history, track_id, skeleton, stgcn_model, box=None):
    if track_id not in track_history:
        track_history[track_id] = deque(maxlen=WINDOW_SIZE)

    if box is not None and stgcn_model.use_bbox_normalization:
        x1, y1, x2, y2 = map(float, box)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        skeleton = skeleton.copy()
        skeleton[:, 0] = (skeleton[:, 0] - x1) / bw
        skeleton[:, 1] = (skeleton[:, 1] - y1) / bh

    track_history[track_id].append(skeleton)
    return len(track_history[track_id]) == WINDOW_SIZE


def run_stgcn_smoke(stgcn_model):
    if stgcn_model is None or not stgcn_model.loaded:
        return False, "模型未載入"
    dummy = np.zeros((WINDOW_SIZE, 17, 3), dtype=np.float32)
    action, conf = stgcn_model.predict_action(dummy)
    if stgcn_model.last_error:
        return False, stgcn_model.last_error
    return True, f"{action} {conf:.3f}"


def extract_ocr_texts(raw_result):
    texts = []
    scores = []

    def add_text(text, score=None):
        text = str(text).strip()
        if not text:
            return
        texts.append(text)
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            scores.append(None)

    def walk(obj):
        if obj is None:
            return

        json_payload = getattr(obj, "json", None)
        if json_payload is not None:
            try:
                payload = json_payload() if callable(json_payload) else json_payload
                if isinstance(payload, str):
                    payload = json.loads(payload)
                walk(payload)
                return
            except Exception:
                pass

        if isinstance(obj, dict):
            if isinstance(obj.get("rec_texts"), list):
                rec_scores = obj.get("rec_scores") or [None] * len(obj["rec_texts"])
                for text, score in zip(obj["rec_texts"], rec_scores):
                    add_text(text, score)
                return
            if "rec_text" in obj:
                add_text(obj.get("rec_text"), obj.get("rec_score"))
                return
            if "text" in obj and ("score" in obj or "confidence" in obj):
                add_text(obj.get("text"), obj.get("score", obj.get("confidence")))
                return
            for value in obj.values():
                walk(value)
            return

        if isinstance(obj, (list, tuple)):
            if (
                len(obj) >= 2 and
                isinstance(obj[1], (list, tuple)) and
                len(obj[1]) >= 2 and
                isinstance(obj[1][0], str)
            ):
                add_text(obj[1][0], obj[1][1])
                return
            for item in obj:
                walk(item)

    walk(raw_result)

    unique_texts = []
    unique_scores = []
    seen = set()
    for text, score in zip(texts, scores):
        if text in seen:
            continue
        seen.add(text)
        unique_texts.append(text)
        unique_scores.append(score)
    return unique_texts, unique_scores


def run_paddleocr_check(image_path, lang):
    if not image_path or not os.path.exists(image_path):
        return False, [], [], f"找不到 OCR 測試圖片：{image_path}"

    image = cv2.imread(image_path)
    if image is None:
        return False, [], [], f"無法讀取 OCR 測試圖片：{image_path}"

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    primary_error = None
    try:
        from paddleocr import PaddleOCR
        try:
            ocr = PaddleOCR(use_angle_cls=False, lang=lang)
        except TypeError:
            ocr = PaddleOCR(lang=lang)

        if hasattr(ocr, "predict"):
            raw_result = ocr.predict(image)
        else:
            raw_result = ocr.ocr(image, cls=False)
        texts, scores = extract_ocr_texts(raw_result)
        return len(texts) > 0, texts, scores, None
    except Exception as exc:
        primary_error = repr(exc)

    try:
        from paddlex import create_model
        print(f"⚠️  PaddleOCR pipeline 失敗，改用 paddlex PP-OCRv5 rec 回退：{primary_error}")
        ocr_device = os.environ.get("PLATE_OCR_DEVICE", "cpu")
        rec_model = create_model(model_name="en_PP-OCRv5_mobile_rec", device=ocr_device)
        raw_result = list(rec_model.predict(image))
        texts, scores = extract_ocr_texts(raw_result)
        return len(texts) > 0, texts, scores, None
    except Exception as exc:
        return False, [], [], f"PaddleOCR={primary_error}; paddlex-rec={repr(exc)}"


def print_status_table(module_status):
    print("\n" + "=" * 60)
    print("🧪 模組偵測狀態")
    print("=" * 60)
    for module_name in MODULE_NAMES:
        status = module_status[module_name]
        if not status["enabled"]:
            marker = "OFF"
        elif status["success"]:
            marker = "OK"
        else:
            marker = "FAIL"
        detail = status["detail"]
        if status.get("error"):
            detail = f"{detail} | error={status['error']}" if detail else f"error={status['error']}"
        print(f"{module_name.upper():10s} {marker:4s} {detail}")
    print("=" * 60)


def safe_destroy_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

resources_dir = 'resources'
video_name = 'resize5.mp4'
output_dir = 'output'


def main():
    parser = argparse.ArgumentParser(description='垃圾亂丟檢測系統')
    parser.add_argument("file", nargs="?", help="file name in resources/", default=video_name)
    parser.add_argument("--seg-model", default=YOLO_MODEL_PATH, help="YOLO 分割/追蹤模型路徑")
    parser.add_argument("--rtdetr-model", default=RTDETR_MODEL_PATH, help="RTDETR 垃圾偵測模型路徑")
    parser.add_argument("--pose-model", default=POSE_MODEL_PATH, help="YOLO Pose 模型路徑")
    parser.add_argument("--stgcn-weight", default=STGCN_WEIGHT_PATH, help="STGCN++ 權重路徑")
    parser.add_argument("--stgcn-config", default=STGCN_CONFIG_PATH, help="STGCN++ config 路徑")
    parser.add_argument("--yolo-conf", type=float, default=0.3, help="YOLO 置信度閾值")
    parser.add_argument("--rtdetr-conf", type=float, default=0.5, help="RTDETR litter 置信度閾值")
    parser.add_argument("--track-iou", type=float, default=0.25, help="seg person 與 pose person 對齊 IoU 閾值")
    parser.add_argument("--ocr-image", default=PADDLEOCR_IMAGE_PATH, help="PaddleOCR 獨立測試圖片")
    parser.add_argument("--ocr-lang", default="en", help="PaddleOCR 語言設定")
    parser.add_argument("--fg-mask-scale", type=float, default=DEFAULT_FG_MASK_SCALE,
                        help="motion mask scale; 0.5 keeps previous low-res mask behavior")
    parser.add_argument("--motion-mask-mode", choices=("temporal", "mog2"), default="temporal",
                        help="temporal uses fast frame differencing; mog2 uses the previous background subtractor")
    parser.add_argument("--motion-diff-threshold", type=int, default=DEFAULT_MOTION_DIFF_THRESHOLD,
                        help="pixel difference threshold for temporal motion mask")
    parser.add_argument("--motion-dilate-iterations", type=int, default=DEFAULT_MOTION_DILATE_ITERATIONS,
                        help="dilation iterations for temporal motion mask")
    parser.add_argument("--mog2-no-shadows", action="store_true",
                        help="disable MOG2 shadow detection when --motion-mask-mode mog2")
    parser.add_argument("--only", nargs="+", choices=MODULE_NAMES, help="只啟用指定模組，例如 --only yolo rtdetr")
    parser.add_argument("--max-frames", type=int, default=0, help="最多處理幾個取樣幀；0 代表全片")
    parser.add_argument("--frame-step", type=int, default=1, help="每 N 幀取樣一次")
    parser.add_argument("--no-video-output", action="store_true", help="只跑偵測統計，不輸出標註影片")
    parser.add_argument("--report", action="store_true", help="輸出量化評估報告（json）")
    add_module_toggle(parser, "yolo", "YOLO Seg")
    add_module_toggle(parser, "rtdetr", "RTDETR litter")
    add_module_toggle(parser, "stgcn", "STGCN++")
    add_module_toggle(parser, "pose", "YOLO Pose")
    add_module_toggle(parser, "paddleocr", "PaddleOCR")
    args = parser.parse_args()
    apply_only_selection(args)
    args.frame_step = max(1, int(args.frame_step))

    os.makedirs(output_dir, exist_ok=True)

    module_status = {
        module_name: empty_module_status(getattr(args, f"enable_{module_name}"))
        for module_name in MODULE_NAMES
    }
    metrics = {
        "frames_total": 0,
        "yolo_detections": 0,
        "yolo_frames_with_detection": 0,
        "rtdetr_detections": 0,
        "rtdetr_litter_detections": 0,
        "rtdetr_frames_with_detection": 0,
        "seg_person_detections": 0,
        "pose_person_detections": 0,
        "pose_keypoint_sets": 0,
        "pose_frames_with_detection": 0,
        "matched_pose_count": 0,
        "stgcn_inference_count": 0,
        "stgcn_littering_count": 0,
        "stgcn_smoke_success": 0,
        "paddleocr_text_count": 0,
    }
    yolo_class_counts = {}
    rtdetr_class_counts = {}
    track_history = {}
    alert_history = {}
    track_stats = {}
    paddleocr_texts = []
    paddleocr_scores = []

    print("=" * 60)
    print("🎬 垃圾亂丟檢測系統 - 配置信息")
    print("=" * 60)
    print(f"YOLO      : {'ON' if args.enable_yolo else 'OFF'} | {args.seg_model}")
    print(f"RTDETR    : {'ON' if args.enable_rtdetr else 'OFF'} | {args.rtdetr_model}")
    print(f"Pose      : {'ON' if args.enable_pose else 'OFF'} | {args.pose_model}")
    print(f"STGCN++   : {'ON' if args.enable_stgcn else 'OFF'} | {args.stgcn_weight}")
    print(f"PaddleOCR : {'ON' if args.enable_paddleocr else 'OFF'} | {args.ocr_image}")
    print(f"Foreground: mode={args.motion_mask_mode}, scale={args.fg_mask_scale}, diff={args.motion_diff_threshold}, dilate={args.motion_dilate_iterations}")
    print(f"輸出目錄  : {os.path.abspath(output_dir)}")
    print("=" * 60)

    yolo_model = None
    if args.enable_yolo:
        try:
            print("\n📥 載入 YOLO Seg 模型中...")
            yolo_model = YOLO(args.seg_model)
            module_status["yolo"]["loaded"] = True
            module_status["yolo"]["detail"] = "模型載入成功，等待影片偵測"
            print("✓ YOLO Seg 模型載入成功")
        except Exception as e:
            module_status["yolo"]["error"] = repr(e)
            print(f"❌ 無法載入 YOLO 模型：{e}")
            sys.exit(1)
    else:
        print("\n⏭️  YOLO Seg 已關閉")

    rtdetr_model = None
    if args.enable_rtdetr:
        try:
            print("📥 載入 RTDETR 模型中...")
            rtdetr_model = RTDETR(args.rtdetr_model)
            module_status["rtdetr"]["loaded"] = True
            module_status["rtdetr"]["detail"] = "模型載入成功，等待影片偵測"
            print("✓ RTDETR 模型載入成功")
        except Exception as e:
            module_status["rtdetr"]["error"] = repr(e)
            print(f"❌ 無法載入 RTDETR 模型：{e}")
            sys.exit(1)
    else:
        print("⏭️  RTDETR 已關閉")

    pose_model = None
    if args.enable_pose:
        try:
            print("📥 載入 YOLO Pose 模型中...")
            pose_model = YOLO(args.pose_model)
            module_status["pose"]["loaded"] = True
            module_status["pose"]["detail"] = "模型載入成功，等待影片偵測"
            print("✓ YOLO Pose 模型載入成功")
        except Exception as e:
            module_status["pose"]["error"] = repr(e)
            print(f"❌ 無法載入 YOLO Pose 模型：{e}")
            sys.exit(1)
    else:
        print("⏭️  YOLO Pose 已關閉")

    stgcn_model = None
    stgcn_smoke_ok = False
    stgcn_smoke_detail = ""
    if args.enable_stgcn:
        try:
            print("📥 載入 STGCN++ 模型中...")
            stgcn_model = STGCN_Predictor(args.stgcn_weight, args.stgcn_config)
            module_status["stgcn"]["loaded"] = bool(stgcn_model.loaded)
            if stgcn_model.loaded:
                stgcn_smoke_ok, stgcn_smoke_detail = run_stgcn_smoke(stgcn_model)
                metrics["stgcn_smoke_success"] = int(stgcn_smoke_ok)
                module_status["stgcn"]["detail"] = f"模型載入成功，smoke={stgcn_smoke_detail}"
                print("✓ STGCN++ 模型載入成功")
                print(f"{'✓' if stgcn_smoke_ok else '⚠️'} STGCN++ 假資料推論：{stgcn_smoke_detail}")
                if not args.enable_pose:
                    print("⚠️  STGCN 已啟用但 Pose 關閉；只能確認模型載入/假資料推論，無法從影片擷取骨架。")
            else:
                module_status["stgcn"]["detail"] = "模型未成功載入"
                print("⚠️  STGCN++ 模型未成功載入，無法確認動作推論")
        except Exception as e:
            module_status["stgcn"]["error"] = repr(e)
            print(f"❌ 無法載入 STGCN++ 模型：{e}")
            sys.exit(1)
    else:
        print("⏭️  STGCN++ 已關閉")

    if args.enable_paddleocr:
        print("📥 執行 PaddleOCR 獨立圖片測試中...")
        ocr_ok, paddleocr_texts, paddleocr_scores, ocr_error = run_paddleocr_check(args.ocr_image, args.ocr_lang)
        metrics["paddleocr_text_count"] = len(paddleocr_texts)
        module_status["paddleocr"]["loaded"] = ocr_error is None
        module_status["paddleocr"]["success"] = ocr_ok
        if ocr_ok:
            preview = " | ".join(paddleocr_texts[:3])
            module_status["paddleocr"]["detail"] = f"辨識文字 {len(paddleocr_texts)} 筆：{preview[:120]}"
            print(f"✓ PaddleOCR 辨識成功，共 {len(paddleocr_texts)} 筆文字")
        else:
            module_status["paddleocr"]["detail"] = "未辨識到文字"
            module_status["paddleocr"]["error"] = ocr_error
            print(f"⚠️  PaddleOCR 未確認成功：{ocr_error or '沒有辨識文字'}")
    else:
        print("⏭️  PaddleOCR 已關閉")

    needs_video = args.enable_yolo or args.enable_rtdetr or args.enable_pose
    if needs_video:
        if not os.path.exists(resources_dir):
            print(f"⚠️  警告：找不到資源目錄 '{resources_dir}'")
            print(f"   請確保 '{resources_dir}' 目錄存在且包含視頻文件")

        video_path = os.path.join(resources_dir, args.file)
        if not os.path.exists(video_path):
            print(f"❌ 錯誤：視頻文件不存在 - {video_path}")
            print(f"   請確保視頻文件在 {resources_dir}/ 目錄中")
            sys.exit(1)

        print(f"📹 開始處理視頻：{video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 錯誤：無法打開視頻文件 - {video_path}")
            sys.exit(1)

        temp_output_path = os.path.join(output_dir, f"temp_{args.file}")
        output_path = os.path.join(output_dir, f"detection_{args.file}")
        temp_output_parent = os.path.dirname(temp_output_path)
        output_parent = os.path.dirname(output_path)
        if temp_output_parent:
            os.makedirs(temp_output_parent, exist_ok=True)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        print(f"📊 視頻信息：{width}x{height} @ {fps}fps, 共 {total_frames} 幀\n")

        out = None
        if not args.no_video_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                print("❌ 錯誤：無法創建輸出視頻文件")
                cap.release()
                sys.exit(1)
        else:
            print("⏭️  已使用 --no-video-output，跳過標註影片輸出")

        motion_masker = None
        if rtdetr_model is not None:
            motion_masker = MotionMaskBuilder(
                mode=args.motion_mask_mode,
                scale_factor=args.fg_mask_scale,
                diff_threshold=args.motion_diff_threshold,
                dilate_iterations=args.motion_dilate_iterations,
                mog2_detect_shadows=not args.mog2_no_shadows,
            )

        raw_frame_count = 0
        processed_sample_count = 0
        print("🔄 開始處理影片，請稍候...")
        with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frame_count += 1
                pbar.update(1)
                is_sample_frame = (raw_frame_count - 1) % args.frame_step == 0
                if not is_sample_frame:
                    if motion_masker is not None:
                        motion_masker.build(frame)
                    continue
                if args.max_frames > 0 and processed_sample_count >= args.max_frames:
                    break

                processed_sample_count += 1
                metrics["frames_total"] += 1

                try:
                    fg_mask = motion_masker.build(frame) if motion_masker is not None else None
                    yolo_results = []
                    if yolo_model is not None:
                        yolo_results = yolo_model.track(
                            frame,
                            persist=True,
                            conf=args.yolo_conf,
                            stream=False,
                            verbose=False
                        )
                        yolo_summary = count_result_boxes(yolo_results, yolo_model)
                        metrics["yolo_detections"] += yolo_summary["total"]
                        if yolo_summary["total"] > 0:
                            metrics["yolo_frames_with_detection"] += 1
                        for name, count in yolo_summary["class_counts"].items():
                            yolo_class_counts[name] = yolo_class_counts.get(name, 0) + count

                    if rtdetr_model is not None:
                        rtdetr_results = rtdetr_model.predict(
                            frame,
                            conf=args.rtdetr_conf,
                            stream=False,
                            verbose=False
                        )
                        rtdetr_summary = count_result_boxes(rtdetr_results, rtdetr_model, target_names={"litter"})
                        metrics["rtdetr_detections"] += rtdetr_summary["total"]
                        motion_litter_count = draw_motion_filtered_litter_boxes(
                            frame,
                            rtdetr_results,
                            rtdetr_model,
                            COLOR_DICT['litter'],
                            "RTDETR",
                            fg_mask,
                            fg_mask_scale=args.fg_mask_scale,
                        )
                        metrics["rtdetr_litter_detections"] += motion_litter_count
                        if motion_litter_count > 0:
                            metrics["rtdetr_frames_with_detection"] += 1
                        for name, count in rtdetr_summary["class_counts"].items():
                            rtdetr_class_counts[name] = rtdetr_class_counts.get(name, 0) + count

                    pose_result = None
                    pose_boxes = []
                    if pose_model is not None:
                        pose_results = pose_model.predict(
                            frame,
                            conf=args.yolo_conf,
                            stream=False,
                            verbose=False
                        )
                        pose_result = pose_results[0] if len(pose_results) > 0 else None
                        if pose_result is not None and pose_result.boxes is not None and len(pose_result.boxes) > 0:
                            metrics["pose_person_detections"] += len(pose_result.boxes)
                            metrics["pose_frames_with_detection"] += 1
                            for pxyxy in pose_result.boxes.xyxy:
                                pose_boxes.append([int(c) for c in pxyxy])
                            if pose_result.keypoints is not None and pose_result.keypoints.xy is not None:
                                metrics["pose_keypoint_sets"] += len(pose_result.keypoints.xy)
                            draw_model_boxes(frame, [pose_result], pose_model, (0, 180, 255), "POSE")

                    if yolo_model is not None:
                        for result in yolo_results:
                            if result.boxes is None or len(result.boxes) == 0:
                                continue
                            mask_polys = getattr(getattr(result, "masks", None), "xy", None)
                            track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(result.boxes)
                            for det_idx, (cls, conf, xyxy, track_id) in enumerate(zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy, track_ids)):
                                class_name = model_class_name(yolo_model, int(cls))
                                x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                                polygon = None
                                if mask_polys is not None and det_idx < len(mask_polys):
                                    polygon = np.asarray(mask_polys[det_idx], dtype=np.int32).reshape((-1, 1, 2))
                                    if polygon.shape[0] < 3:
                                        polygon = None
                                if class_name == 'person' and track_id is not None:
                                    metrics["seg_person_detections"] += 1
                                    if track_id not in alert_history:
                                        alert_history[track_id] = 0
                                        track_stats[track_id] = {
                                            "frames_seen": 0,
                                            "pose_matched": 0,
                                            "stgcn_inference": 0,
                                            "littering_alerts": 0
                                        }
                                    track_stats[track_id]["frames_seen"] += 1

                                    if stgcn_model is not None and pose_result is not None:
                                        best_pose_idx = -1
                                        best_iou = 0.0
                                        for p_idx, pbox in enumerate(pose_boxes):
                                            score = iou_xyxy([x1, y1, x2, y2], pbox)
                                            if score > best_iou:
                                                best_iou = score
                                                best_pose_idx = p_idx
                                        if best_pose_idx >= 0 and best_iou >= args.track_iou:
                                            metrics["matched_pose_count"] += 1
                                            track_stats[track_id]["pose_matched"] += 1
                                            skeleton = extract_skeleton_from_pose_model(frame, pose_result, best_pose_idx)
                                            ready = append_pose_skeleton_for_stgcn(
                                                track_history,
                                                track_id,
                                                skeleton,
                                                stgcn_model,
                                                box=[x1, y1, x2, y2],
                                            )
                                            if ready:
                                                metrics["stgcn_inference_count"] += 1
                                                track_stats[track_id]["stgcn_inference"] += 1
                                                action, act_conf = stgcn_model.predict_action(np.array(list(track_history[track_id])))
                                                if action == "littering" and act_conf > ACTION_THRESHOLD:
                                                    metrics["stgcn_littering_count"] += 1
                                                    track_stats[track_id]["littering_alerts"] += 1
                                                    alert_history[track_id] = 30

                                    if alert_history.get(track_id, 0) > 0:
                                        color = COLOR_DICT['littering_alert']
                                        label = f"ID:{track_id} LITTERING!"
                                        alert_history[track_id] -= 1
                                    else:
                                        color = COLOR_DICT['person']
                                        label = f"YOLO person ID:{track_id}"
                                    if polygon is not None:
                                        cv2.polylines(frame, [polygon], True, color, 2)
                                    else:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                elif class_name in COLOR_DICT:
                                    color = COLOR_DICT[class_name]
                                    label = f"YOLO {class_name} {float(conf):.2f}"
                                    if polygon is not None:
                                        cv2.polylines(frame, [polygon], True, color, 2)
                                    else:
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if yolo_model is None and stgcn_model is not None and pose_result is not None and pose_boxes:
                        skeleton = extract_skeleton_from_pose_model(frame, pose_result, 0)
                        ready = append_pose_skeleton_for_stgcn(
                            track_history,
                            "pose_only_0",
                            skeleton,
                            stgcn_model,
                            box=pose_boxes[0],
                        )
                        if ready:
                            metrics["stgcn_inference_count"] += 1
                            action, act_conf = stgcn_model.predict_action(np.array(list(track_history["pose_only_0"])))
                            if action == "littering" and act_conf > ACTION_THRESHOLD:
                                metrics["stgcn_littering_count"] += 1
                except Exception as e:
                    print(f"\n⚠️  幀 {processed_sample_count} 處理出錯：{e}")
                    continue

                if out is not None:
                    out.write(frame)

        cap.release()
        if out is not None:
            out.release()
        safe_destroy_windows()

        if out is not None:
            print("\n✓ OpenCV 畫面處理完畢。準備使用 FFmpeg 進行編碼轉換...")
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',
                '-i', temp_output_path,
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '22',
                output_path
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                print("🎉 影片匯出成功！")
                print(f"📁 已儲存至： {os.path.abspath(output_path)}")
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
            except FileNotFoundError:
                print("❌ 錯誤：系統中找不到 FFmpeg")
                print(f"   暫存影片保留於：{temp_output_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg 轉換失敗，錯誤代碼：{e.returncode}")
                print(f"   暫存影片保留於：{temp_output_path}")
    else:
        print("\nℹ️  影片相關模型都已關閉，跳過影片讀取。")

    if args.enable_yolo:
        module_status["yolo"]["success"] = metrics["yolo_detections"] > 0
        module_status["yolo"]["detail"] = (
            f"偵測 {metrics['yolo_detections']} 個 box，"
            f"有偵測幀 {metrics['yolo_frames_with_detection']}，classes={yolo_class_counts}"
        )
    if args.enable_rtdetr:
        module_status["rtdetr"]["success"] = metrics["rtdetr_litter_detections"] > 0
        module_status["rtdetr"]["detail"] = (
            f"總 box {metrics['rtdetr_detections']}，litter {metrics['rtdetr_litter_detections']}，"
            f"有偵測幀 {metrics['rtdetr_frames_with_detection']}，classes={rtdetr_class_counts}"
        )
    if args.enable_pose:
        module_status["pose"]["success"] = metrics["pose_keypoint_sets"] > 0
        module_status["pose"]["detail"] = (
            f"person boxes {metrics['pose_person_detections']}，"
            f"keypoint sets {metrics['pose_keypoint_sets']}，有偵測幀 {metrics['pose_frames_with_detection']}"
        )
    if args.enable_stgcn:
        has_real_stgcn = metrics["stgcn_inference_count"] > 0
        module_status["stgcn"]["success"] = has_real_stgcn or stgcn_smoke_ok
        if has_real_stgcn:
            module_status["stgcn"]["detail"] = (
                f"影片推論 {metrics['stgcn_inference_count']} 次，"
                f"littering {metrics['stgcn_littering_count']} 次"
            )
        else:
            module_status["stgcn"]["detail"] = (
                f"smoke={'OK' if stgcn_smoke_ok else 'FAIL'}({stgcn_smoke_detail})；"
                "影片骨架未累積到 STGCN window"
            )

    print_status_table(module_status)

    seg_person = max(1, metrics["seg_person_detections"])
    stgcn_inf = max(1, metrics["stgcn_inference_count"])
    pose_match_rate = metrics["matched_pose_count"] / seg_person
    littering_rate = metrics["stgcn_littering_count"] / stgcn_inf
    active_tracks = len(track_stats)

    print("\n" + "=" * 60)
    print("📈 測試量化指標")
    print("=" * 60)
    print(f"取樣處理幀數: {metrics['frames_total']}")
    print(f"YOLO detections: {metrics['yolo_detections']} | classes: {yolo_class_counts}")
    print(f"RTDETR detections: {metrics['rtdetr_detections']} | litter: {metrics['rtdetr_litter_detections']} | classes: {rtdetr_class_counts}")
    print(f"Seg person 偵測數: {metrics['seg_person_detections']}")
    print(f"Pose person 偵測數: {metrics['pose_person_detections']}")
    print(f"Pose keypoint sets: {metrics['pose_keypoint_sets']}")
    print(f"Seg-Pose 對齊數: {metrics['matched_pose_count']}")
    print(f"骨架對齊率: {pose_match_rate:.4f}")
    print(f"STGCN 推論次數: {metrics['stgcn_inference_count']}")
    print(f"STGCN littering 次數: {metrics['stgcn_littering_count']}")
    print(f"Littering 觸發率: {littering_rate:.4f}")
    print(f"PaddleOCR 文字筆數: {metrics['paddleocr_text_count']}")
    print(f"有效追蹤 ID 數: {active_tracks}")
    print("=" * 60)

    if args.report:
        report_path = os.path.join(output_dir, f"report_{os.path.splitext(args.file)[0]}.json")
        report_payload = {
            "video": args.file,
            "enabled_modules": {
                module_name: getattr(args, f"enable_{module_name}")
                for module_name in MODULE_NAMES
            },
            "module_status": module_status,
            "models": {
                "seg": args.seg_model,
                "rtdetr": args.rtdetr_model,
                "pose": args.pose_model,
                "stgcn": args.stgcn_weight,
                "paddleocr_image": args.ocr_image,
            },
            "thresholds": {
                "yolo_conf": args.yolo_conf,
                "rtdetr_conf": args.rtdetr_conf,
                "track_iou": args.track_iou,
                "action_threshold": ACTION_THRESHOLD,
                "window_size": WINDOW_SIZE,
                "frame_step": args.frame_step,
                "max_frames": args.max_frames,
                    "fg_mask_scale": args.fg_mask_scale,
                    "motion_mask_mode": args.motion_mask_mode,
                    "motion_diff_threshold": args.motion_diff_threshold,
                    "motion_dilate_iterations": args.motion_dilate_iterations,
            },
            "metrics": {
                **metrics,
                "pose_match_rate": pose_match_rate,
                "littering_trigger_rate": littering_rate,
                "active_tracks": active_tracks,
                "yolo_class_counts": yolo_class_counts,
                "rtdetr_class_counts": rtdetr_class_counts,
                "paddleocr_texts": paddleocr_texts,
                "paddleocr_scores": paddleocr_scores,
            },
            "tracks": {str(k): v for k, v in track_stats.items()}
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, ensure_ascii=False, indent=2)
        print(f"🧾 已輸出評估報告：{os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
