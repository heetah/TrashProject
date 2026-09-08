#!/usr/bin/env /usr/bin/python3
"""Create the dynamic-Homography modification overview as an editable PPTX.

Uses LibreOffice/UNO already installed on the workstation; no network or
presentation package is required.  Metrics are read from reproducible JSON
artifacts so the slides do not silently drift from the validation report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

import uno
from com.sun.star.awt import Point, Size
from com.sun.star.beans import PropertyValue
from com.sun.star.drawing.FillStyle import NONE as FILL_NONE, SOLID
from com.sun.star.drawing.LineStyle import NONE as LINE_NONE, SOLID as LINE_SOLID


W, H = 33867, 19050
BLACK, GRAY, LIGHT, BLUE, ORANGE, GREEN, RED = (
    0x111111, 0x666666, 0xEEEEEE, 0x4285F4, 0xFFAB40, 0x0097A7, 0xC62828
)
FONT = "Noto Sans CJK TC"


def prop(name, value):
    item = PropertyValue()
    item.Name = name
    item.Value = value
    return item


def connect_office():
    profile = Path(tempfile.mkdtemp(prefix="codex-impress-"))
    process = subprocess.Popen(
        [
            "libreoffice", "--headless", "--norestore", "--nodefault",
            "--nolockcheck", f"-env:UserInstallation=file://{profile}",
            "--accept=socket,host=127.0.0.1,port=2083;urp;StarOffice.ComponentContext",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local
    )
    for _ in range(80):
        try:
            context = resolver.resolve(
                "uno:socket,host=127.0.0.1,port=2083;urp;StarOffice.ComponentContext"
            )
            desktop = context.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", context
            )
            return process, desktop
        except Exception:
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("could not connect to LibreOffice")


def set_geometry(shape, x, y, w, h):
    shape.Position = Point(int(x), int(y))
    shape.Size = Size(int(w), int(h))


def text(doc, page, value, x, y, w, h, *, size=20, color=BLACK,
         bold=False, align=0, fill=None, line=None, margin=180):
    shape = doc.createInstance("com.sun.star.drawing.TextShape")
    set_geometry(shape, x, y, w, h)
    shape.TextLeftDistance = margin
    shape.TextRightDistance = margin
    shape.TextUpperDistance = margin
    shape.TextLowerDistance = margin
    shape.FillStyle = SOLID if fill is not None else FILL_NONE
    if fill is not None:
        shape.FillColor = int(fill)
    shape.LineStyle = LINE_SOLID if line is not None else LINE_NONE
    if line is not None:
        shape.LineColor = int(line)
    page.add(shape)
    # A freshly created UNO TextShape does not retain text assigned before it
    # belongs to a draw page. Insert it first, then style the actual text
    # cursor so PPTX export contains runs rather than empty paragraphs.
    shape.String = str(value)
    cursor = shape.getText().createTextCursor()
    cursor.gotoEnd(True)
    cursor.CharFontName = FONT
    cursor.CharFontNameAsian = FONT
    cursor.CharHeight = float(size)
    cursor.CharColor = int(color)
    cursor.CharWeight = 150.0 if bold else 100.0
    cursor.ParaAdjust = int(align)
    return shape


def rect(doc, page, x, y, w, h, *, fill=LIGHT, line=GRAY, radius=False):
    service = "com.sun.star.drawing.RectangleShape"
    shape = doc.createInstance(service)
    set_geometry(shape, x, y, w, h)
    shape.FillStyle = SOLID
    shape.FillColor = int(fill)
    shape.LineStyle = LINE_SOLID if line is not None else LINE_NONE
    if line is not None:
        shape.LineColor = int(line)
    if radius:
        try:
            shape.CornerRadius = 250
        except Exception:
            pass
    page.add(shape)
    return shape


def line(doc, page, x1, y1, x2, y2, *, color=GRAY, width=45):
    shape = doc.createInstance("com.sun.star.drawing.LineShape")
    shape.Position = Point(int(x1), int(y1))
    shape.Size = Size(int(x2 - x1), int(y2 - y1))
    shape.LineColor = int(color)
    shape.LineWidth = int(width)
    page.add(shape)
    return shape


def slide(doc, index, title, tag="反向\n追蹤"):
    pages = doc.getDrawPages()
    page = pages.getByIndex(0) if index == 0 else pages.insertNewByIndex(index)
    page.Width, page.Height = W, H
    text(doc, page, title, 1150, 650, 27000, 1500, size=25, bold=False)
    line(doc, page, 1150, 2500, 32500, 2500, color=BLACK, width=20)
    circle = doc.createInstance("com.sun.star.drawing.EllipseShape")
    set_geometry(circle, 29900, 0, 3967, 3967)
    circle.FillStyle, circle.FillColor, circle.LineStyle = SOLID, LIGHT, LINE_NONE
    page.add(circle)
    text(doc, page, tag, 30220, 500, 3000, 2400, size=17, bold=True, align=3)
    text(doc, page, str(index + 1), 31500, 18000, 900, 500, size=10, color=GRAY, align=3)
    return page


def cover(doc):
    page = doc.getDrawPages().getByIndex(0)
    page.Width, page.Height = W, H
    text(doc, page, "環保科技執法專題", 6500, 6500, 21000, 1500, size=32, align=3)
    text(doc, page, "動態 Pseudo-Homography 串接與 58 部驗證", 5000, 8200, 24000, 1100, size=20, align=3)
    text(doc, page, "2026/09/07  修改總覽", 9000, 9700, 16000, 800, size=15, color=GRAY, align=3)
    text(doc, page, "張哲誠　張宇誠", 11000, 11800, 12000, 900, size=13, align=3)


def card(doc, page, title_value, body, x, y, w, h, *, color=BLUE):
    rect(doc, page, x, y, w, h, fill=0xFAFAFA, line=0xBBBBBB, radius=True)
    rect(doc, page, x, y, 160, h, fill=color, line=None)
    text(doc, page, title_value, x + 450, y + 350, w - 800, 700, size=17, bold=True)
    text(doc, page, body, x + 450, y + 1250, w - 800, h - 1500, size=13, color=GRAY)


def metric_bar(doc, page, label, value, target, x, y, width, *, color=BLUE):
    text(doc, page, label, x, y - 100, 7800, 650, size=15)
    rect(doc, page, x + 8000, y, width, 500, fill=LIGHT, line=None)
    rect(doc, page, x + 8000, y, int(width * value), 500, fill=color, line=None)
    tx = x + 8000 + int(width * target)
    line(doc, page, tx, y - 180, tx, y + 700, color=RED, width=35)
    text(doc, page, f"{value*100:.1f}%", x + 8000 + width + 350, y - 120, 2300, 700, size=15, bold=True)


def build(doc, baseline, dynamic, audit):
    cover(doc)
    page = slide(doc, 1, "本次修改：把校正 H 安全地接進歸因，而不是強制套用")
    card(doc, page, "事件內固定", "每個 litter event 只使用一份 immutable H snapshot，避免線上更新造成同一事件座標漂移。", 1300, 3900, 9800, 5200, color=BLUE)
    card(doc, page, "證據閘門", "僅 LOCKED、confidence ≥ 0.65、完整幾何驗證通過時，才切換到 pseudo-ground。", 12050, 3900, 9800, 5200, color=GREEN)
    card(doc, page, "安全退回", "任何缺失、低信心或無效矩陣，都保留原 image-space 成本與完整 NULL route。", 22800, 3900, 9800, 5200, color=ORANGE)
    text(doc, page, "重點：Homography 只改變空間特徵的座標系；Kalman 證據、時間成本與 Min-Cost Flow 拓樸不被偷換。", 2500, 11600, 28800, 1700, size=18, bold=True, align=3)

    page = slide(doc, 2, "新版動態 Homography 資料流")
    labels = ["vehicle mask\n底部 98% 中位點", "長時間 trajectory\nquality filtering", "motion field\nflow clustering", "bounded H search\nrobust losses", "confidence + state\nLOCKED 才可用"]
    for i, label in enumerate(labels):
        x = 900 + i * 6500
        rect(doc, page, x, 5000, 5200, 3000, fill=0xFAFAFA, line=GRAY, radius=True)
        text(doc, page, label, x + 250, 5650, 4700, 1600, size=15, align=3, bold=(i == 4))
        if i < len(labels) - 1:
            text(doc, page, "→", x + 5300, 5700, 1200, 1000, size=27, color=BLUE, align=3)
    text(doc, page, "禁止：把移動車輛 t−1 與 t 的位置誤當同一 world point 直接解 H", 3800, 10100, 26000, 1000, size=17, color=RED, bold=True, align=3)

    page = slide(doc, 3, "事件歸因中的座標轉換")
    text(doc, page, "p = [u, v, 1]ᵀ", 1800, 4200, 6000, 900, size=24, bold=True, align=3)
    text(doc, page, "→", 7600, 4200, 1800, 900, size=28, color=BLUE, align=3)
    text(doc, page, "q = π(H_event p)", 9000, 4200, 8800, 900, size=24, bold=True, align=3)
    text(doc, page, "→", 17900, 4200, 1800, 900, size=28, color=BLUE, align=3)
    text(doc, page, "D_H = dist(qᵣ, H(B)) / diag(H(B))", 19700, 4200, 12500, 900, size=20, bold=True, align=3)
    card(doc, page, "C_BA", "release → person 上半身區域；尺度採投影後 person height。", 1800, 7200, 9000, 4200, color=BLUE)
    card(doc, page, "C_BC", "release → vehicle 四邊形；尺度採投影後 vehicle diagonal。", 12400, 7200, 9000, 4200, color=GREEN)
    card(doc, page, "C_AC", "person / vehicle footpoint 距離；避免重複使用 litter anchor。", 23000, 7200, 9000, 4200, color=ORANGE)
    text(doc, page, "continuous pseudo-ground relative scale ≠ 公尺；成本仍為無因次比例。", 5000, 13800, 24000, 900, size=17, color=GRAY, align=3)

    page = slide(doc, 4, "三道安全閘門與事件級 fallback")
    stages = [("1  狀態", "status = LOCKED"), ("2  信心", "confidence ≥ 0.65"), ("3  幾何", "finite / non-singular / bounded")]
    for i, (a, b) in enumerate(stages):
        x = 1800 + i * 10300
        rect(doc, page, x, 4400, 8500, 3200, fill=0xFAFAFA, line=GRAY, radius=True)
        text(doc, page, a, x + 300, 4850, 7900, 700, size=18, bold=True, color=[BLUE, GREEN, ORANGE][i], align=3)
        text(doc, page, b, x + 300, 5900, 7900, 700, size=15, align=3)
    text(doc, page, "全部通過", 4200, 9300, 5000, 800, size=17, bold=True, color=GREEN, align=3)
    text(doc, page, "→ pseudo-ground 成本", 9000, 9300, 8500, 800, size=18, bold=True)
    text(doc, page, "任一失敗", 4200, 11200, 5000, 800, size=17, bold=True, color=ORANGE, align=3)
    text(doc, page, "→ image-space 成本 + 原因 sidecar", 9000, 11200, 16500, 800, size=18, bold=True)
    text(doc, page, "NULL route 永遠存在；不因校正失敗強制配對。", 6200, 14200, 22000, 900, size=18, bold=True, align=3)

    page = slide(doc, 5, "實作修改範圍")
    rows = [
        ("calibration/state.py", "事件 snapshot 增加 status、image_shape、runtime H"),
        ("backtrack/spatial.py", "矩陣驗證、bbox 投影、正規化距離與向量轉換"),
        ("backtrack/costs.py", "C_BA / C_BC / C_AC 接受同一 event transform"),
        ("backtrack/resolver.py", "選擇座標系並在 candidate diagnostics 留痕"),
        ("litter_tracker.py", "confirm 時凍結 H；低信心逐事件 fallback"),
        ("readiness evaluator", "58 部完整分母；coverage 與 accuracy 分開"),
    ]
    for i, (a, b) in enumerate(rows):
        y = 3400 + i * 2050
        rect(doc, page, 1700, y, 8200, 1350, fill=LIGHT if i % 2 == 0 else 0xFAFAFA, line=None)
        text(doc, page, a, 1900, y + 220, 7800, 750, size=14, bold=True)
        text(doc, page, b, 10400, y + 220, 21000, 750, size=14)

    page = slide(doc, 6, "驗證設計：三種證據不可混為一談")
    card(doc, page, "程式正確性", "Pipeline 319 passed、12 skipped。涵蓋矩陣拒絕、成本一致性、applied/fallback 路徑與既有回歸。", 1600, 3800, 9600, 6200, color=BLUE)
    card(doc, page, "58 部真實短片", "每部獨立啟動；驗證整合不崩潰、輸出完整、fallback 不退化。不能驗證 30–100 秒收斂。", 12100, 3800, 9600, 6200, color=GREEN)
    card(doc, page, "上線準確率", "分母固定 58。漏檢、NULL、錯車都保留；negative clips 缺失，因此 FPR gate 仍 blocked。", 22600, 3800, 9600, 6200, color=ORANGE)
    text(doc, page, "合成測試通過 ≠ 動態 H 提升真實準確率", 5500, 12900, 23000, 1100, size=22, color=RED, bold=True, align=3)

    page = slide(doc, 7, "58 部結果：先看完整分母，再看條件式歸因")
    base_event = baseline["event_detection_sensitivity"]["rate"]
    base_e2e = baseline["end_to_end_vehicle_correctness"]["rate"]
    dyn_event = dynamic["event_detection_sensitivity"]["rate"]
    dyn_e2e = dynamic["end_to_end_vehicle_correctness"]["rate"]
    metric_bar(doc, page, "Baseline event detection", base_event, 0.85, 1800, 4700, 13500, color=GRAY)
    metric_bar(doc, page, "Dynamic-H run event detection", dyn_event, 0.85, 1800, 6800, 13500, color=GREEN)
    metric_bar(doc, page, "Baseline end-to-end vehicle", base_e2e, 0.85, 1800, 9500, 13500, color=GRAY)
    metric_bar(doc, page, "Dynamic-H run end-to-end vehicle", dyn_e2e, 0.85, 1800, 11600, 13500, color=BLUE)
    text(doc, page, "紅線 = 85%", 27100, 3600, 4300, 700, size=14, color=RED, bold=True, align=3)
    text(doc, page, "舊 34/58 是人工口徑；嚴格上線口徑為 33/58，因 case 74 本次沒有系統事件，不能算 end-to-end 正確。", 2700, 15100, 28500, 1200, size=15, color=GRAY, align=3)

    page = slide(doc, 8, "Homography 實際使用情況")
    applied = int(audit.get("applied_events", 0))
    total_events = int(audit.get("event_snapshots", 0))
    fallback = total_events - applied
    text(doc, page, f"applied  {applied}/{total_events}", 2500, 4500, 8500, 1200, size=30, bold=True, color=GREEN, align=3)
    text(doc, page, f"fallback  {fallback}/{total_events}", 12500, 4500, 9000, 1200, size=30, bold=True, color=ORANGE, align=3)
    text(doc, page, "主要原因", 23500, 4200, 5000, 800, size=17, bold=True)
    reason = audit.get("dominant_fallback_reason", "calibration_not_locked")
    text(doc, page, reason, 23500, 5300, 8200, 1000, size=17, color=ORANGE)
    text(doc, page, "58 部皆為短片且每片重置 calibrator；不足以累積 30–100 秒、多車流與 spatial coverage，因此正確行為就是不套用。", 2800, 9000, 28000, 1900, size=18, align=3)
    text(doc, page, "結論：本輪證明 safe integration / no-regression；尚未證明 learned H 可提高歸因準確率。", 3500, 13000, 27000, 1200, size=20, bold=True, color=RED, align=3)

    page = slide(doc, 9, "距離 85% 上線門檻仍有多少差距？")
    failures = dynamic["failure_counts"]
    total = int(dynamic["usable_clip_count"])
    required = int(dynamic["target_counts"]["point_estimate_at_least_target"])
    correct = int(dynamic["end_to_end_vehicle_correctness"]["successes"])
    values = [
        ("正確車輛", correct, GREEN),
        ("事件漏檢", int(failures["missed_event"]), ORANGE),
        ("錯誤車輛", int(failures["wrong_vehicle"]), RED),
        ("NULL / person-only", int(failures["unresolved_or_person_only"]), GRAY),
    ]
    x0, y0, fullw = 3800, 5000, 26000
    cursor = x0
    for label, count, color in values:
        width = int(fullw * count / max(total, 1))
        if width:
            rect(doc, page, cursor, y0, width, 1600, fill=color, line=None)
        cursor += width
        text(doc, page, f"{label} {count}", 3800, 8000 + values.index((label, count, color)) * 1150, 10000, 650, size=15, color=color, bold=True)
    text(doc, page, f"點估計至少需要 {required}/{total}；目前 {correct}/{total}，仍差 {max(required-correct,0)} 部。", 15000, 8200, 16000, 1000, size=21, bold=True)
    text(doc, page, "若要求 Wilson 95% 下界也 ≥85%，需 55/58。", 15000, 10000, 16000, 900, size=17)
    text(doc, page, "沒有 reviewed negative clips → 無法估計誤報率，產品上線仍被阻擋。", 15000, 11900, 16000, 1100, size=17, color=RED, bold=True)

    page = slide(doc, 10, "本輪可以下的結論／不能下的結論")
    card(doc, page, "已證明", "• 新 H 已進入三個空間成本\n• 每事件座標一致\n• 低信心安全 fallback\n• 既有測試與 58 部流程無崩潰", 1800, 3800, 14000, 8500, color=GREEN)
    card(doc, page, "尚未證明", "• learned H 提升真實 ID accuracy\n• 0.65 是跨鏡頭最優常數\n• 產品 accuracy ≥85%\n• negative-video false-positive rate", 18000, 3800, 14000, 8500, color=RED)
    text(doc, page, "因此維持 opt-in；不可在缺少連續攝影機驗證前預設啟用。", 5000, 14500, 24000, 900, size=19, bold=True, align=3)

    page = slide(doc, 11, "達到 2026/10 上線標準的最短路徑")
    steps = [
        ("A", "先補 detection", "17 部漏檢中 16 部已有 RT-DETR candidate；針對 confirmation/track continuity 做逐例診斷。"),
        ("B", "再修 attribution", "8 部錯車做 release frame、候選距離與 ID mapping 人工複核；不以 threshold 硬湊。"),
        ("C", "補真實 H 證據", "每 camera 收集 ≥30–100 秒連續影片，做 frozen-H paired comparison。"),
        ("D", "獨立驗收", "development 與 holdout camera 分離；加入 reviewed negative clips 與人工複核 SOP。"),
    ]
    for i, (letter, title_value, body) in enumerate(steps):
        y = 3400 + i * 3200
        circle = doc.createInstance("com.sun.star.drawing.EllipseShape")
        set_geometry(circle, 1900, y, 1500, 1500)
        circle.FillStyle, circle.FillColor, circle.LineStyle = SOLID, [BLUE, ORANGE, GREEN, RED][i], LINE_NONE
        page.add(circle)
        text(doc, page, letter, 2050, y + 220, 1200, 700, size=18, color=0xFFFFFF, bold=True, align=3)
        text(doc, page, title_value, 4100, y, 6000, 800, size=18, bold=True)
        text(doc, page, body, 10100, y, 21500, 1500, size=14, color=GRAY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    dynamic = json.loads(args.dynamic.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    process, desktop = connect_office()
    doc = None
    try:
        doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0, ())
        build(doc, baseline, dynamic, audit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        target = uno.systemPathToFileUrl(str(args.output.resolve()))
        doc.storeAsURL(target, (prop("FilterName", "Impress MS PowerPoint 2007 XML"),))
    finally:
        if doc is not None:
            doc.close(True)
        process.terminate()
        process.wait(timeout=10)


if __name__ == "__main__":
    main()
