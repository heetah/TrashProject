# -*- coding: utf-8 -*-
"""
單獨測試 4c RTDETR litter 偵測（不跑 tracker / holding / OCR / STGCN）。
4c 模型輸入需 RGB + change map (4 channels)，本腳本逐幀構造 4ch frame 並餵給 RTDETR。

用法:
  conda run -n rtdetr python rtdetr_4c_test.py <video.mp4>
    [--model modules_weight/best-rtdetr-4c.pt]
    [--conf 0.5] [--no-output]

輸出:
  - 終端統計: 總幀數、總偵測數、有偵測的幀數、平均 conf
  - output/<name>_rtdetr4c.mp4 標註影片（除非 --no-output）
"""
import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import RTDETR

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "modules_weight", "best-rtdetr-4c.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def compute_pixel_change_map(prev_frame, curr_frame):
    # 與 scripts-old-test/detect.py 完全一致的 change channel 構造方式。
    if prev_frame is None:
        return np.zeros(curr_frame.shape[:2], dtype=np.uint8)
    diff = cv2.absdiff(prev_frame, curr_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return np.clip(diff_gray.astype(np.float32) * 2.0, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--no-output", action="store_true", help="只跑統計不寫標註影片")
    ap.add_argument("--print-dets", action="store_true", help="印出每幀每個 detection 的 cx,cy,wh,conf")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"video not found: {args.video}")
    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")

    print(f"📥 載入 RTDETR 4c 模型: {args.model}")
    model = RTDETR(args.model)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if not args.no_output:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        name = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{name}_rtdetr4c.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        print(f"📤 輸出標註影片: {out_path}")

    prev_frame = None
    total_dets = 0
    frames_with_det = 0
    confs = []
    fi = 0

    pbar = tqdm(total=total or None, unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        change_map = compute_pixel_change_map(prev_frame, frame)
        frame_4ch = np.dstack((frame, change_map))  # (H, W, 4) BGR + change

        results = model.predict(frame_4ch, conf=args.conf, verbose=False)

        n_det = 0
        for r in results:
            if r.boxes is None:
                continue
            n_det += len(r.boxes)
            for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                confs.append(float(c))
                x1, y1, x2, y2 = map(int, box[:4])
                if args.print_dets:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    print(f"  [DET fi={fi} cx={cx},{cy} wh={x2-x1}x{y2-y1} conf={c:.3f}]")
                if writer is not None:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{c:.2f}", (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        total_dets += n_det
        if n_det > 0:
            frames_with_det += 1

        if writer is not None:
            cv2.putText(frame, f"fi={fi} det={n_det}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            writer.write(frame)

        prev_frame = frame.copy() if prev_frame is None else frame
        fi += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    avg_conf = float(np.mean(confs)) if confs else 0.0
    print("=" * 60)
    print(f"frames           : {fi}")
    print(f"total detections : {total_dets}")
    print(f"frames with det  : {frames_with_det}")
    print(f"avg conf         : {avg_conf:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
