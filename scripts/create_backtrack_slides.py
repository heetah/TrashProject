#!/usr/bin/env python3
"""Create an editable Smart Backtrack comparison deck.

The deck is authored as editable ODP XML and emitted as a direct OOXML PPTX;
no slide is a flattened screenshot. Equations are Unicode text boxes so they
remain editable in PowerPoint/Impress.
"""
from __future__ import annotations

import html
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "presentations"
ODP_PATH = OUT_DIR / "smart_backtrack_version_comparison_20260819.odp"
PPTX_PATH = OUT_DIR / "smart_backtrack_version_comparison_20260819.pptx"

W, H = 33.867, 19.05
BLACK = "111111"
BG = "FFFFFF"
PANEL = "F3F3F3"
PANEL2 = "FAFAFA"
# Kept as named aliases because the page model uses the old semantic names.
# All text styles below explicitly use BLACK; these values only colour editable bars/panels.
WHITE = BLACK
MUTED = BLACK
CYAN = "222222"
ORANGE = "3A3A3A"
GREEN = "4A4A4A"
RED = "555555"
PURPLE = "666666"
GRID = "B8B8B8"


def esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def style_block() -> str:
    text_styles = []
    for name, size, color, bold in [
        ("PTitle", 27, BLACK, True),
        ("PSub", 13, BLACK, False),
        ("PHead", 18, BLACK, True),
        ("PBody", 11, BLACK, False),
        ("PSmall", 8.3, BLACK, False),
        ("PFormula", 12, BLACK, False),
        ("PFormulaSmall", 9.3, BLACK, False),
        ("PKpi", 23, BLACK, True),
        ("PLabel", 8, BLACK, True),
        ("PFoot", 7.2, BLACK, False),
    ]:
        text_styles.append(
            f'<style:style style:name="{name}" style:family="paragraph">'
            f'<style:paragraph-properties fo:text-align="left" fo:line-height="112%"/>'
            f'<style:text-properties fo:font-family="Noto Sans CJK TC" '
            f'style:font-family-generic="sans" fo:font-size="{size}pt" '
            f'fo:color="#{color}" '
            f'{"fo:font-weight=\"bold\"" if bold else ""}/></style:style>'
        )
    graphic = "".join([
        '<style:style style:name="none" style:family="graphic"><style:graphic-properties draw:fill="none" draw:stroke="none"/></style:style>',
        f'<style:style style:name="bg" style:family="graphic"><style:graphic-properties draw:fill="solid" draw:fill-color="#{BG}" draw:stroke="none"/></style:style>',
        f'<style:style style:name="panel" style:family="graphic"><style:graphic-properties draw:fill="solid" draw:fill-color="#{PANEL}" draw:stroke="none"/></style:style>',
        f'<style:style style:name="panel2" style:family="graphic"><style:graphic-properties draw:fill="solid" draw:fill-color="#{PANEL2}" draw:stroke="none"/></style:style>',
        f'<style:style style:name="cyan" style:family="graphic"><style:graphic-properties draw:fill="solid" draw:fill-color="#{CYAN}" draw:stroke="none"/></style:style>',
        f'<style:style style:name="orange" style:family="graphic"><style:graphic-properties draw:fill="solid" draw:fill-color="#{ORANGE}" draw:stroke="none"/></style:style>',
        f'<style:style style:name="green" style:family="graphic"><style:graphic-properties draw:fill="solid" draw:fill-color="#{GREEN}" draw:stroke="none"/></style:style>',
        f'<style:style style:name="red" style:family="graphic"><style:graphic-properties draw:fill="solid" draw:fill-color="#{RED}" draw:stroke="none"/></style:style>',
        f'<style:style style:name="purple" style:family="graphic"><style:graphic-properties draw:fill="solid" draw:fill-color="#{PURPLE}" draw:stroke="none"/></style:style>',
        f'<style:style style:name="line" style:family="graphic"><style:graphic-properties draw:fill="none" draw:stroke="solid" svg:stroke-color="#{GRID}" svg:stroke-width="0.04cm"/></style:style>',
    ])
    return (
        '<office:automatic-styles>'
        '<style:page-layout style:name="PM1"><style:page-layout-properties fo:page-width="33.867cm" fo:page-height="19.05cm" style:print-orientation="landscape"/></style:page-layout>'
        '<style:style style:name="dp1" family="drawing-page"><style:drawing-page-properties draw:background-size="border"/></style:style>'
        + "".join(text_styles) + graphic +
        '</office:automatic-styles>'
    )


def frame(style: str, x: float, y: float, w: float, h: float, text: str,
          text_style: str = "PBody", lines: bool = True) -> str:
    paragraphs = []
    for line in str(text).split("\n") if lines else [str(text)]:
        paragraphs.append(f'<text:p text:style-name="{text_style}">{esc(line)}</text:p>')
    return (
        f'<draw:frame draw:style-name="{style}" draw:text-style-name="{text_style}" '
        f'svg:x="{x:.3f}cm" svg:y="{y:.3f}cm" svg:width="{w:.3f}cm" svg:height="{h:.3f}cm" '
        'presentation:class="text"><draw:text-box>' + "".join(paragraphs) + '</draw:text-box></draw:frame>'
    )


def rect(style: str, x: float, y: float, w: float, h: float) -> str:
    return f'<draw:rect draw:style-name="{style}" svg:x="{x:.3f}cm" svg:y="{y:.3f}cm" svg:width="{w:.3f}cm" svg:height="{h:.3f}cm"/>'


def line(x: float, y: float, w: float, h: float = 0.0) -> str:
    return f'<draw:line draw:style-name="line" svg:x1="{x:.3f}cm" svg:y1="{y:.3f}cm" svg:x2="{x+w:.3f}cm" svg:y2="{y+h:.3f}cm"/>'


def page_header(title: str, subtitle: str = "") -> str:
    out = [rect("bg", 0, 0, W, H), rect("cyan", 0, 0, 0.22, H)]
    out.append(frame("none", 1.1, 0.55, 28.5, 0.8, title, "PTitle"))
    if subtitle:
        out.append(frame("none", 1.12, 1.35, 29.0, 0.45, subtitle, "PSub"))
    return "".join(out)


def footer(number: int, label: str = "SMART BACKTRACK · VERSION STUDY") -> str:
    return frame("none", 1.12, 18.35, 30.8, 0.3, f"{label}                                      {number:02d}", "PFoot")


def card(x, y, w, h, title, body, accent=CYAN, body_style="PBody"):
    style = {CYAN: "cyan", ORANGE: "orange", GREEN: "green", RED: "red", PURPLE: "purple"}.get(accent, "cyan")
    return (
        rect("panel", x, y, w, h) + rect(style, x, y, 0.12, h) +
        frame("none", x + 0.35, y + 0.22, w - 0.55, 0.45, title, "PHead") +
        frame("none", x + 0.35, y + 0.82, w - 0.55, h - 1.0, body, body_style)
    )


def pill(x, y, w, label, accent=CYAN):
    style = {CYAN: "cyan", ORANGE: "orange", GREEN: "green", RED: "red", PURPLE: "purple"}.get(accent, "cyan")
    return rect(style, x, y, w, 0.52) + frame("none", x + 0.12, y + 0.09, w - 0.24, 0.32, label, "PLabel")


def make_pages() -> list[str]:
    pages = []
    # 01
    p = [rect("bg", 0, 0, W, H), rect("cyan", 0, 0, 0.28, H), rect("orange", 1.1, 3.0, 4.8, 0.12)]
    p += [frame("none", 1.1, 1.15, 29.5, 1.0, "反追蹤模組：前版 vs 新版", "PTitle"),
          frame("none", 1.1, 2.0, 28.5, 0.8, "從「最近的 actor」到「release causality route」", "PSub"),
          frame("none", 1.1, 3.45, 14.8, 1.5, "Smart Backtrack\n可驗證的時空歸因與成本模型", "PHead"),
          frame("none", 1.1, 6.2, 20.0, 1.3, "正式數學模型 · Kalman / RTS · 反向外插 · Min-Cost Flow\n2026-08-19  |  production code 對照版", "PBody"),
          card(23.0, 5.6, 8.5, 2.6, "一句話", "在可能的 release 時間範圍內，選擇最符合運動、空間、時間與人車關係的完整路徑。", ORANGE),
          footer(1, "SMART BACKTRACK · EDITABLE DECK")]
    pages.append("".join(p))

    # 02
    p = [page_header("先給結論", "最大的升級不是單一算法，而是把歸因變成可重建的因果時空決策")]
    p += [frame("none", 1.15, 2.05, 31.0, 1.0, "「不是看誰離垃圾最近，而是重建誰最可能在 release 時刻造成垃圾。」", "PHead"),
          card(1.15, 4.0, 9.8, 4.9, "前版：heuristic association", "距離 + 時間\n\n單次最佳 actor\n\n容易被前景 bbox、短暫消失、微小成本 tie 影響\n\n證據難以 replay", RED),
          card(12.0, 4.0, 9.8, 4.9, "新版：causal route resolution", "release hypothesis\n+ Kalman / RTS uncertainty\n+ C_BA / C_AC / C_BC\n+ Min-Cost Flow + NULL\n+ actor-specific margin\n+ compact resolver sidecar", GREEN),
          card(22.85, 4.0, 9.8, 4.9, "技術價值", "把「判斷結果」變成\n可追溯的：\n\n輸入 → 假設 → gate → cost components → route → margin\n\n可離線重播、可做 ablation", CYAN), footer(2)]
    pages.append("".join(p))

    # 03
    p = [page_header("前版與新版：責任邊界的改變", "成熟算法本身不是創新；創新在於它們被放入同一個可驗證因果模型")]
    p += [rect("panel", 1.1, 2.0, 31.7, 0.55), frame("none", 1.35, 2.12, 6.0, 0.3, "比較面向", "PLabel"), frame("none", 9.0, 2.12, 10.0, 0.3, "前版", "PLabel"), frame("none", 19.2, 2.12, 11.8, 0.3, "新版", "PLabel")]
    rows = [
        ("時間語意", "候選出現附近的時間", "T_birth − Δt = release hypothesis；保留區間"),
        ("位置模型", "bbox / track 的直線式使用", "Kalman + RTS；位置與 covariance 分開"),
        ("成本", "距離 + 時間總分", "C_BA、C_AC、C_BC；各自 gate、raw feature、weight"),
        ("多 actor", "單一 winner / 容易被 tie-break 影響", "完整 route + unbounded capacity + NULL"),
        ("不確定性", "通常只看 confidence", "R、P、Mahalanobis、observation-gap、margin"),
        ("研究證據", "看輸出影片判斷", "resolver_input sidecar + deterministic replay"),
    ]
    y = 2.75
    for i, (a, b, c) in enumerate(rows):
        bg = "panel2" if i % 2 == 0 else "panel"
        p += [rect(bg, 1.1, y, 31.7, 1.02), frame("none", 1.35, y + .19, 7.1, .62, a, "PBody"), frame("none", 9.0, y + .19, 9.6, .62, b, "PSmall"), frame("none", 19.2, y + .16, 12.2, .7, c, "PSmall"), line(1.1, y + 1.02, 31.7)]
        y += 1.02
    p.append(footer(3)); pages.append("".join(p))

    # 04
    p = [page_header("新版架構：一個事件如何被歸因", "確認事件與歸因分層；下游不能偽造上游證據")]
    steps = [
        ("1", "litter track", "birth / history", CYAN),
        ("2", "release hypotheses", "ballistic / 2-point", ORANGE),
        ("3", "actor state", "Kalman + RTS", PURPLE),
        ("4", "cost cells", "C_BA / C_AC / C_BC", GREEN),
        ("5", "route graph", "Min-Cost Flow", CYAN),
        ("6", "decision", "person / vehicle / NULL", ORANGE),
    ]
    x = 1.1
    for i, (num, title, sub, color) in enumerate(steps):
        p += [rect("panel", x, 4.3, 4.75, 3.5), pill(x + .35, 4.65, .65, num, color), frame("none", x + .35, 5.45, 4.0, .5, title, "PHead"), frame("none", x + .35, 6.15, 4.0, .45, sub, "PSmall")]
        if i < len(steps) - 1:
            p.append(frame("none", x + 4.82, 5.72, .45, .35, "→", "PHead"))
        x += 5.25
    p += [card(1.1, 9.1, 15.0, 3.1, "關鍵責任", "Hungarian：只處理同一 object 的跨 frame identity。\n不做 person↔vehicle attribution。", PURPLE), card(16.55, 9.1, 16.25, 3.1, "安全出口", "每個事件永遠保留完整 NULL route。\n如果 physical gate、uncertainty 或 route margin 不足，保留人工複核。", ORANGE), footer(4)]
    pages.append("".join(p))

    # 05
    p = [page_header("release 時間：不是一個神諭，而是一組 hypothesis", "T_birth 是第一個有效 litter track；Δt 由軌跡模型產生")]
    p += [rect("panel", 1.1, 2.2, 31.7, 4.3), line(3.2, 4.65, 27.0),
          frame("none", 2.0, 2.55, 6.0, .35, "觀測到 litter", "PLabel"), frame("none", 8.5, 2.55, 7.0, .35, "可能 release window", "PLabel"), frame("none", 22.8, 2.55, 7.0, .35, "後續 litter observations", "PLabel"),
          pill(2.0, 4.15, 2.1, "T_release", ORANGE), pill(14.2, 4.15, 2.1, "T_birth", CYAN), pill(26.0, 4.15, 2.1, "T_last", GREEN),
          frame("none", 4.65, 4.25, 8.8, .35, "← Δt = inferred flight time", "PSmall"), frame("none", 16.5, 4.25, 8.0, .35, "observed / predicted states", "PSmall"),
          frame("none", 1.55, 7.15, 31.0, 1.15, "T_release(h) = T_birth − Δt(h)\n對每個 h：取 release position、velocity、covariance，再套 actor hard gate 與 cost。", "PFormula"),
          card(1.1, 9.0, 10.0, 3.3, "≥ 3 points", "x(t) = aₓt + bₓ\ny(t) = aᵧt² + bᵧt + cᵧ\n可向前補最多 0.5 s", CYAN, "PFormulaSmall"),
          card(11.9, 9.0, 10.0, 3.3, "2 points", "constant velocity\n最多反向 0.4 s\n沿速度方向 covariance 增長較快", ORANGE, "PFormulaSmall"),
          card(22.7, 9.0, 10.1, 3.3, "1 point", "birth fallback\n高 prior / dustbin bias\n避免假造運動證據", RED, "PFormulaSmall"), footer(5)]
    pages.append("".join(p))

    # 06
    p = [page_header("Kalman Filter：位置不是答案，位置 + 不確定度才是狀態", "box state 使用 footpoint 與 log-size；dynamic 用秒，查詢用 frame")]
    p += [card(1.1, 2.0, 15.0, 4.9, "狀態與觀測", "狀態：\nxₖ = [uₖ, vₖ, log wₖ, log hₖ, ũₖ, ṽₖ, ġwₖ, ġhₖ]ᵀ\n\n觀測：\nzₖ = [uₖ, vₖ, log wₖ, log hₖ]ᵀ\n      = Hxₖ + νₖ", CYAN, "PFormula"),
          card(16.7, 2.0, 16.1, 4.9, "Predict / Update", "F(Δt) = [ I₄  ΔtI₄ ; 0  I₄ ]\n\n xₖ⁻ = Fₖxₖ₋₁\n Pₖ⁻ = FₖPₖ₋₁Fₖᵀ + Qₖ\n\nνₖ = zₖ − Hxₖ⁻\nSₖ = HPₖ⁻Hᵀ + Rₖ\nKₖ = Pₖ⁻HᵀSₖ⁻¹\n xₖ = xₖ⁻ + Kₖνₖ", ORANGE, "PFormulaSmall"),
          frame("none", 1.25, 7.55, 31.2, .8, "Pₖ 變大 ⇒ Sₖ 變大 ⇒ Kₖ 變小：預測較不可信時，更新較保守。", "PHead"),
          frame("none", 1.25, 8.65, 31.2, 1.2, "不是把缺失 detection 當真實點；缺失期間只做 predict，並將 observation gap / covariance 反映到成本。", "PBody"), footer(6)]
    pages.append("".join(p))

    # 07
    p = [page_header("不確定度如何進入系統", "confidence 影響 measurement covariance；covariance 影響更新與 gate")]
    p += [frame("none", 1.15, 2.0, 31.0, .55, "Rₖ(c) = R₀ · c⁻ᵖ,   c ∈ [c_min, 1],   p = 2", "PFormula"),
          card(1.1, 3.1, 9.7, 4.5, "高 confidence", "c ↑\nR ↓\nK ↑\n觀測更能拉動狀態\n\n結果：較相信 detector", GREEN, "PBody"),
          card(11.4, 3.1, 9.7, 4.5, "低 confidence", "c ↓\nR ↑\nK ↓\n更新較弱\n\n結果：較依賴 prediction", ORANGE, "PBody"),
          card(21.7, 3.1, 11.1, 4.5, "物理安全限制", "uncertainty ratio = √λ_max(P_release + P_actor) / h_actor\n\n若 > 1.5：reject candidate\n\n注意：covariance 只能提高 cost / reject，不能放寬距離 gate。", RED, "PFormulaSmall"),
          frame("none", 1.15, 8.3, 31.0, .9, "Mahalanobis：d_M² = rᵀ(P_release + P_actor)⁻¹r\n距離仍先過 physical gate；不確定度不能把遠處 actor 變合理。", "PFormula"), footer(7)]
    pages.append("".join(p))

    # 08
    p = [page_header("RTS smoothing：利用未來觀測修正過去狀態", "online Kalman 只看過去；RTS 在 replay / finalize 階段反向平滑")]
    p += [rect("panel", 1.1, 2.0, 31.7, 3.5), line(4.0, 3.85, 25.8),
          pill(2.0, 3.35, 1.6, "xₖ", CYAN), pill(8.3, 3.35, 1.6, "xₖ₊₁", PURPLE), pill(14.6, 3.35, 1.6, "xₖ₊₂", GREEN), pill(20.9, 3.35, 1.6, "xₖ₊₃", ORANGE), pill(27.2, 3.35, 1.6, "future", CYAN),
          frame("none", 2.0, 2.5, 28.2, .5, "forward Kalman pass  →  ←  backward RTS pass", "PLabel"),
          frame("none", 1.35, 6.15, 31.2, 1.5, "Gₖ = PₖFₖᵀ(Pₖ₊₁⁻)⁻¹\n\nx̂ₖˢ = x̂ₖ + Gₖ(x̂ₖ₊₁ˢ − x̂ₖ₊₁⁻)", "PFormula"),
          card(1.1, 8.45, 15.2, 3.0, "直觀", "後面 frame 看見 actor 回來，能修正前面短暫消失的預測；但不會創造不存在的 detector evidence。", PURPLE),
          card(16.7, 8.45, 16.1, 3.0, "歸因限制", "D+T 的 time 仍使用最近真實 observation 的 gap。\nRTS 給 geometry，不把 stale prediction 偽裝成同步觀測。", ORANGE), footer(8)]
    pages.append("".join(p))

    # 09
    p = [page_header("成本矩陣的共同原則：先 gate，再 normalized feature，再加權", "不同單位不能直接相加；distance/time 先轉成無因次量")]
    p += [frame("none", 1.15, 2.05, 31.0, .75, "d̃ = min(d / d_max, 1)       t̃ = min(|t_obs − t_release| / t_max, 1)", "PFormula"),
          card(1.1, 3.25, 9.7, 4.2, "Hard gate", "d > d_max ⇒ invalid\n|Δt| > t_max ⇒ invalid\nuncertainty ratio > 1.5 ⇒ invalid\n\n不能靠提高權重救回 invalid route", RED, "PFormulaSmall"),
          card(11.4, 3.25, 9.7, 4.2, "Soft cost", "通過 gate 後：\nC = Σᵢ wᵢ fᵢ\n\nfᵢ 是 raw feature\nwᵢ 是 trial config\n\n每個 component 可重算", CYAN, "PFormulaSmall"),
          card(21.7, 3.25, 11.1, 4.2, "目前關鍵設定", "C_AC overlap weight = 0\n\n理由：錯誤深度的 vehicle bbox 可能覆蓋 person；footpoint distance 才是主要幾何。\n\n兩點回推上限 = 0.4 s", GREEN, "PBody"),
          frame("none", 1.15, 8.2, 31.0, 1.4, "正式報告要分開寫：\n(1) candidate 被 gate reject；(2) candidate 通過但成本較高；(3) route 被 NULL / 其他 actor 勝出。", "PBody"), footer(9)]
    pages.append("".join(p))

    # 10
    p = [page_header("C_BA：litter → person 的 release compatibility", "人使用上半身 release zone，不使用 footpoint 作為丟垃圾位置")]
    p += [frame("none", 1.15, 2.0, 31.0, .7, "Z_A = [x₁, y₁, x₂, y₁ + 0.72h_A]", "PFormula"),
          frame("none", 1.15, 3.0, 31.0, .7, "r_BA = point_to_rect(p_release, Z_A)\n\nd_BA = ‖r_BA‖₂ / h_A", "PFormula"),
          frame("none", 1.15, 4.05, 31.0, 1.55, "C_BA(A,t) = w_d d̃_BA + w_u log det(I + Σ_BA / h_A²)\n                 + w_t t̃_A + w_q q_A + w_dir f_dir + w_r p_release(t)", "PFormula"),
          card(1.1, 6.25, 10.1, 3.2, "direction", "若 release velocity 與人上半身 displacement 反向：\nf_dir = max(0, −cos θ)\n\n只加成本，不直接 reject。", PURPLE, "PFormulaSmall"),
          card(11.9, 6.25, 10.1, 3.2, "quality", "q_A = (1 − confidence_A)\n       + 0.35 · 1[predicted]\n\n低信心或補點 actor 成本較高。", ORANGE, "PFormulaSmall"),
          card(22.7, 6.25, 10.1, 3.2, "物理 gate", "d_BA ≤ 0.85\nobservation gap ≤ 0.25 s\nuncertainty ratio ≤ 1.5\n\n否則 C_BA invalid。", RED, "PFormulaSmall"),
          frame("none", 1.15, 10.25, 31.0, .75, "C_BA = min_t C_BA(A,t)：先在 release hypotheses 中選最佳時刻，再進入 route 組合。", "PBody"), footer(10)]
    pages.append("".join(p))

    # 11
    p = [page_header("C_AC：person → vehicle 的 sustained association", "避免把一幀路過者，誤當成丟垃圾者的車主")]
    p += [frame("none", 1.15, 2.0, 31.0, .8, "f_AC = [d_foot, 1 − IoM, Δt, q_pair, u_pair, 1 − support_ratio]", "PFormula"),
          frame("none", 1.15, 3.0, 31.0, 1.0, "C_AC(A,C) = w_d d̃_foot + w_o(1 − IoM) + w_t t̃\n                  + w_q q_pair + w_u u_pair + w_c(1 − ρ_support)", "PFormula"),
          card(1.1, 4.7, 10.1, 3.8, "sustained support", "至少一段 observed person + observed vehicle 的連續支援。\n\nmin dwell = 0.15 s\nmax support gap = 0.50 s", GREEN, "PFormulaSmall"),
          card(11.9, 4.7, 10.1, 3.8, "endpoint transition", "允許短暫上下車：\n至少兩個 temporal samples\n且端點 IoM ≥ 0.20\n\n單幀 passerby 不足以成立。", ORANGE, "PFormulaSmall"),
          card(22.7, 4.7, 10.1, 3.8, "本次消融結論", "overlap raw feature 仍保存\n但 production weight = 0\n\n原因：錯深度 vehicle bbox 會產生假 overlap；case 42 因此修正。", CYAN, "PBody"),
          frame("none", 1.15, 9.25, 31.0, .8, "C_AC 不使用 litter anchor；它只回答：「這個 person 與這台 vehicle 是否有長期／轉換關係？」", "PHead"), footer(11)]
    pages.append("".join(p))

    # 12
    p = [page_header("C_BC：litter → vehicle 的直接 release gate", "direct vehicle route 的核心；person route 中只作 bounded support")]
    p += [frame("none", 1.15, 2.0, 31.0, .75, "Z_C = bbox_C ⊕ (0.18w_C, 0.15h_C)\n\nr_BC = point_to_rect(p_release, Z_C)", "PFormula"),
          frame("none", 1.15, 3.0, 31.0, 1.15, "C_BC(C,t) = w_d d̃_BC + w_M d_M + w_u u_BC + w_t t̃_C\n                  + w_q q_C + w_r p_release(t)\n                  + w_rev f_rev + w_exit f_exit + w_rel f_rel", "PFormulaSmall"),
          card(1.1, 5.0, 9.7, 3.6, "direct vehicle", "C_BC 是 hard physical release gate。\n\nnormalized_distance ≤ 0.8\nobservation gap ≤ 0.25 s\nuncertainty ratio ≤ 1.5", CYAN, "PFormulaSmall"),
          card(11.4, 5.0, 9.7, 3.6, "person route", "B-C 不要求硬成立。\n\n它只在同一 release frame 提供 support，避免限制「人離開車後才丟」。", ORANGE, "PBody"),
          card(21.7, 5.0, 11.1, 3.6, "研究結果", "reverse / exit / relative features 已加入 sidecar。\n\n七案方向不一致 → production weight = 0。\n\n診斷 ≠ 已證明有效的 reward。", RED, "PBody"),
          frame("none", 1.15, 9.2, 31.0, .85, "這個設計刻意把「物理不可行」和「成本較差」分離，避免靠權重放寬物理距離。", "PHead"), footer(12)]
    pages.append("".join(p))

    # 13
    p = [page_header("完整 route 與 Min-Cost Flow", "Flow 選 route；Hungarian 不參與 person↔vehicle 歸因")]
    p += [card(1.1, 2.0, 15.1, 4.4, "四條完整 route", "r₁: litter → person → vehicle\nr₂: litter → person\nr₃: litter → vehicle\nr₄: litter → NULL\n\nC_PV = max(0, C_BA + αC_AC − βe^(−C_BC))\nC_direct = C_BC + λ_direct\nC_person = C_BA + λ_null", CYAN, "PFormulaSmall"),
          card(16.7, 2.0, 16.1, 4.4, "整數化目標", "x_e,r ∈ {0,1}\n\nmin Σₑ Σᵣ c_e,r x_e,r\nsubject to:\nΣᵣ x_e,r = 1,  ∀ event e\n\nNULL route 永遠存在。\nactor capacity unbounded：同車多人、同人多事件合法。", ORANGE, "PFormulaSmall"),
          frame("none", 1.15, 7.25, 31.0, .75, "COST_SCALE = 10⁶：保留 0.001 以下的真實差異，避免 route-ID 取代數學最小值。", "PHead"),
          card(1.1, 8.45, 10.1, 3.1, "Flow graph", "source → event node → route node → sink\n\n每個 event 必送 1 unit flow。", PURPLE, "PFormulaSmall"),
          card(11.9, 8.45, 10.1, 3.1, "目前數學限制", "event-expanded\n無跨事件 coupling\n\n因此目前等價於每事件獨立最短 route。", RED, "PBody"),
          card(22.7, 8.45, 10.1, 3.1, "安全語意", "若 NULL cost < all valid route：\n保留 NULL / manual review\n\n不強迫配對。", GREEN, "PBody"), footer(13)]
    pages.append("".join(p))

    # 14
    p = [page_header("Margin：區分「路徑 tie」與「actor tie」", "同一 actor 的不同 release route，不應被誤報成第二個 actor")]
    p += [frame("none", 1.15, 2.0, 31.0, .9, "M_route = C_(2nd route) − C_(best route)", "PFormula"),
          frame("none", 1.15, 3.05, 31.0, .9, "M_vehicle = min_{C ≠ C*} C(vehicle=C) − C(vehicle=C*)\nM_person  = min_{A ≠ A*} C(person=A)  − C(person=A*)", "PFormulaSmall"),
          card(1.1, 4.75, 9.7, 4.0, "M_route 小", "可能只是：\n同一台車、不同 release frame\n或 direct / person route 競爭\n\n不一定代表兩台車真的不確定。", PURPLE, "PBody"),
          card(11.4, 4.75, 9.7, 4.0, "M_actor 小", "不同 vehicle identity 成本接近。\n\n才是多 actor attribution ambiguity。\n\ncase 50 / 51 顯示：正確答案也可能 low margin。", ORANGE, "PBody"),
          card(21.7, 4.75, 11.1, 4.0, "NULL margin", "M_NULL = C_NULL − C_best_nonNULL\n\nM_NULL > 0：non-NULL 較便宜\nM_NULL < 0：NULL 較安全\n\nmargin 是診斷，不是 accuracy。", GREEN, "PFormulaSmall"),
          frame("none", 1.15, 9.55, 31.0, .9, "Sidecar 保存 raw feature、weight、weighted component、gate reason、route ranking；同 input + config 必須 deterministic。", "PHead"), footer(14)]
    pages.append("".join(p))

    # 15
    p = [page_header("研究證據：新版確實改善了什麼？", "只報 reviewed seven-case conditional attribution；63 clips 只報 regression / coverage")]
    p += [card(1.1, 2.0, 9.6, 4.7, "Seven-case reviewed", "Exact Route Top-1\n5 / 7  = 71.4%\n\nRecall@3\n7 / 7  = 100%\n\nRelease hit\n7 / 7\nMAE = 0 frame", GREEN, "PKpi"),
          card(11.3, 2.0, 9.6, 4.7, "修正的機制", "case 42\nC_AC overlap 0 → person 2 → vehicle 1\n\ncase 50 litter 5\nflow scale 10⁶ → vehicle 9\n\ncase 24 / litter 8\nconfirmation reject", CYAN, "PBody"),
          card(21.5, 2.0, 11.3, 4.7, "仍未解決", "case 25：vehicle 1 → 應為 vehicle 2\ncase 168：vehicle 3 → 應為 vehicle 1\n\n兩者正確 actor 都在 Recall@3，\n但 release 區域有多車遮擋。", RED, "PBody"),
          frame("none", 1.15, 7.55, 31.0, .9, "63-clips scan：63/63 pipeline exit 0；18 clips / 23 confirmed litter；尚無完整人工 route labels，因此不是 accuracy 結論。", "PHead"),
          frame("none", 1.15, 9.05, 31.0, 1.15, "研究結論：目前最可靠的增益來自 release timing、confirmation safety、flow precision、C_AC 深度誤差修正；\n不是把所有新 feature 一起打開。", "PBody"), footer(15)]
    pages.append("".join(p))

    # 16
    p = [page_header("最後一頁：技術貢獻與下一步", "把可以證明的創新，和仍需要資料的問題分開")]
    p += [card(1.1, 2.0, 15.0, 4.8, "可以正式主張", "1. release-synchronized causal route\n2. uncertainty-aware state + gates\n3. explicit C_BA / C_AC / C_BC decomposition\n4. NULL-safe Min-Cost Flow\n5. actor-level ambiguity margin\n6. replayable evidence sidecar", GREEN),
          card(16.7, 2.0, 16.1, 4.8, "下一個研究問題", "case 25 / 168 是 instance-level occlusion：\n\n• release-origin instance mask\n• vehicle depth / occlusion ordering\n• hand / throwing evidence\n• 多標註者 admissible route\n\n禁止用兩個案例專屬權重硬修。", ORANGE),
          frame("none", 1.15, 7.65, 31.0, 1.25, "最終一句話：\n新版不是「用了更多算法」，而是「把算法組成一條可解釋、可拒絕、可重播、可逐步驗證的因果證據鏈」。", "PHead"),
          frame("none", 1.15, 10.15, 31.0, .8, "Code：scripts/pipeline/backtrack/   ·   Deck generated from live model contracts   ·   2026-08-19", "PFoot"), footer(16)]
    pages.append("".join(p))
    return pages


def build_odp() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = make_pages()
    content_ns = (
        'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
        'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
        'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0" '
        'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" '
        'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" '
        'xmlns:config="urn:oasis:names:tc:opendocument:xmlns:config:1.0" '
        'xmlns:chart="urn:oasis:names:tc:opendocument:xmlns:chart:1.0" '
        'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
        'xmlns:dr3d="urn:oasis:names:tc:opendocument:xmlns:dr3d:1.0" '
        'office:version="1.3"'
    )
    pages_xml = []
    for i, body in enumerate(pages, 1):
        pages_xml.append(
            f'<draw:page draw:name="page{i}" draw:style-name="dp1">{body}</draw:page>'
        )
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-content {content_ns}>'
        f'{style_block()}<office:body><office:presentation>{"".join(pages_xml)}</office:presentation></office:body>'
        '</office:document-content>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<office:document-styles {content_ns}>'
        '<office:styles><style:default-style style:family="graphic"><style:graphic-properties draw:fill="none"/></style:default-style></office:styles>'
        '<office:automatic-styles><style:page-layout style:name="PM1"><style:page-layout-properties fo:page-width="33.867cm" fo:page-height="19.05cm"/></style:page-layout></office:automatic-styles>'
        '<office:master-styles><style:master-page style:name="Default" style:page-layout-name="PM1"/></office:master-styles>'
        '</office:document-styles>'
    )
    meta = '<?xml version="1.0" encoding="UTF-8"?><office:document-meta xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><office:meta><dc:title>Smart Backtrack Version Comparison</dc:title><dc:creator>Codex</dc:creator><dc:subject>Kalman RTS Min-Cost Flow cost matrix</dc:subject></office:meta></office:document-meta>'
    settings = '<?xml version="1.0" encoding="UTF-8"?><office:document-settings xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:config="urn:oasis:names:tc:opendocument:xmlns:config:1.0"><office:settings/></office:document-settings>'
    manifest = ('<?xml version="1.0" encoding="UTF-8"?>'
                '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.3">'
                '<manifest:file-entry manifest:media-type="application/vnd.oasis.opendocument.presentation" manifest:full-path="/"/>'
                '<manifest:file-entry manifest:media-type="text/xml" manifest:full-path="content.xml"/>'
                '<manifest:file-entry manifest:media-type="text/xml" manifest:full-path="styles.xml"/>'
                '<manifest:file-entry manifest:media-type="text/xml" manifest:full-path="meta.xml"/>'
                '<manifest:file-entry manifest:media-type="text/xml" manifest:full-path="settings.xml"/>'
                '</manifest:manifest>')
    with zipfile.ZipFile(ODP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mimetype", "application/vnd.oasis.opendocument.presentation", compress_type=zipfile.ZIP_STORED)
        z.writestr("content.xml", content)
        z.writestr("styles.xml", styles)
        z.writestr("meta.xml", meta)
        z.writestr("settings.xml", settings)
        z.writestr("META-INF/manifest.xml", manifest)


EMU_PER_CM = 360000
PPTX_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
ODF_DRAW = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}"
ODF_TEXT = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
ODF_SVG = "{urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0}"


def _emu(value: str) -> int:
    return int(round(float(str(value).replace("cm", "")) * EMU_PER_CM))


def _attr(node, key: str, default="0") -> str:
    return node.attrib.get(ODF_SVG + key, default)


def _pptx_root(tag: str, attrs="") -> str:
    ns = " ".join(f'xmlns:{k}="{v}"' for k, v in PPTX_NS.items())
    return f'<{tag} {ns} {attrs}>'


def _xfrm(x, y, w, h):
    return f'<a:xfrm><a:off x="{_emu(x)}" y="{_emu(y)}"/><a:ext cx="{_emu(w)}" cy="{_emu(h)}"/></a:xfrm>'


FILL = {
    "bg": BG, "panel": PANEL, "panel2": PANEL2, "cyan": CYAN,
    "orange": ORANGE, "green": GREEN, "red": RED, "purple": PURPLE,
}
TEXT_STYLE = {
    "PTitle": (27, BLACK, True), "PSub": (13, BLACK, False),
    "PHead": (18, BLACK, True), "PBody": (11, BLACK, False),
    "PSmall": (8.3, BLACK, False), "PFormula": (12, BLACK, False),
    "PFormulaSmall": (9.3, BLACK, False), "PKpi": (23, BLACK, True),
    "PLabel": (8, BLACK, True), "PFoot": (7.2, BLACK, False),
}


def _shape_xml(node, shape_id: int) -> str:
    x, y, w, h = (_attr(node, key) for key in ("x", "y", "width", "height"))
    style = node.attrib.get("{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}style-name", "none")
    fill = FILL.get(style)
    name = f"shape{shape_id}"
    if node.tag == ODF_DRAW + "rect":
        fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else '<a:noFill/>'
        return (
            f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x,y,w,h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}<a:ln><a:noFill/></a:ln></p:spPr></p:sp>'
        )
    if node.tag == ODF_DRAW + "line":
        x2 = node.attrib.get(ODF_SVG + "x2", x)
        y2 = node.attrib.get(ODF_SVG + "y2", y)
        # Horizontal/vertical study guides become editable thin rectangles.
        ww = max(abs(_emu(x2) - _emu(x)), 9000) / EMU_PER_CM
        hh = max(abs(_emu(y2) - _emu(y)), 9000) / EMU_PER_CM
        return (
            f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x,y,ww,hh)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{GRID}"/></a:solidFill><a:ln><a:noFill/></a:ln></p:spPr></p:sp>'
        )
    text_box = node.find(ODF_DRAW + "text-box")
    if text_box is None:
        return ""
    text_style = node.attrib.get("{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}text-style-name", "PBody")
    size, color, bold = TEXT_STYLE.get(text_style, TEXT_STYLE["PBody"])
    paragraphs = []
    for paragraph in text_box.findall(ODF_TEXT + "p"):
        text_value = "".join(paragraph.itertext())
        run = f'<a:r><a:rPr lang="zh-TW" sz="{int(size*100)}" {"b=\"1\"" if bold else ""}><a:latin typeface="Noto Sans CJK TC"/><a:ea typeface="Noto Sans CJK TC"/></a:rPr><a:t>{esc(text_value)}</a:t></a:r>'
        paragraphs.append(f'<a:p><a:pPr algn="l"/><a:rPr lang="zh-TW" sz="{int(size*100)}" {"b=\"1\"" if bold else ""}><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:latin typeface="Noto Sans CJK TC"/><a:ea typeface="Noto Sans CJK TC"/></a:rPr>{run}<a:endParaRPr lang="zh-TW" sz="{int(size*100)}"/></a:p>')
    return (
        f'<p:sp><p:nvSpPr><p:cNvPr id="{shape_id}" name="{name}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr>{_xfrm(x,y,w,h)}<a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
        f'<p:txBody><a:bodyPr wrap="square" rtlCol="0"/><a:lstStyle/>{"".join(paragraphs)}</p:txBody></p:sp>'
    )


def _slide_xml(page, index: int) -> str:
    shapes = []
    shape_id = 2
    for child in list(page):
        if child.tag in {ODF_DRAW + "rect", ODF_DRAW + "frame", ODF_DRAW + "line"}:
            value = _shape_xml(child, shape_id)
            if value:
                shapes.append(value)
                shape_id += 1
    return (
        f'<p:sld xmlns:a="{PPTX_NS["a"]}" xmlns:r="{PPTX_NS["r"]}" xmlns:p="{PPTX_NS["p"]}" showMasterSp="1">'
        '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvGrpSpPr/></p:nvGrpSpPr><p:grpSpPr/>'
        + "".join(shapes) + '</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'
    )


def _pptx_parts(slides: list[str]) -> dict[str, str]:
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    content_types = ['<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>', '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>', '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>', '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>', '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>']
    content_types += [f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, len(slides)+1)]
    content_types.append('</Types>')
    root_rels = f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>'
    pres_rels = ['<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="%s">' % rel_ns, '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>']
    for i in range(1, len(slides)+1):
        pres_rels.append(f'<Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>')
    pres_rels.append('</Relationships>')
    master_rels = f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/></Relationships>'
    presentation = ['<?xml version="1.0" encoding="UTF-8"?><p:presentation xmlns:a="%s" xmlns:r="%s" xmlns:p="%s"><p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst><p:sldIdLst>' % (PPTX_NS['a'], PPTX_NS['r'], PPTX_NS['p'])]
    for i in range(1, len(slides)+1):
        presentation.append(f'<p:sldId id="{255+i}" r:id="rId{i+1}"/>')
    presentation.append('</p:sldIdLst><p:sldSz cx="12192000" cy="6858000" type="screen16x9"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>')
    theme = f'<?xml version="1.0" encoding="UTF-8"?><a:theme xmlns:a="{PPTX_NS["a"]}" name="Smart Backtrack"><a:themeElements><a:clrScheme name="Smart"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1><a:dk2><a:srgbClr val="0B1220"/></a:dk2><a:lt2><a:srgbClr val="F8FAFC"/></a:lt2><a:accent1><a:srgbClr val="38BDF8"/></a:accent1><a:accent2><a:srgbClr val="F59E0B"/></a:accent2><a:accent3><a:srgbClr val="22C55E"/></a:accent3><a:accent4><a:srgbClr val="A78BFA"/></a:accent4><a:accent5><a:srgbClr val="F87171"/></a:accent5><a:accent6><a:srgbClr val="A9B7CC"/></a:accent6><a:hlink><a:srgbClr val="38BDF8"/></a:hlink><a:folHlink><a:srgbClr val="A78BFA"/></a:folHlink></a:clrScheme><a:fontScheme name="Smart"><a:majorFont><a:latin typeface="Noto Sans CJK TC"/><a:ea typeface="Noto Sans CJK TC"/><a:cs typeface="Noto Sans CJK TC"/></a:majorFont><a:minorFont><a:latin typeface="Noto Sans CJK TC"/><a:ea typeface="Noto Sans CJK TC"/><a:cs typeface="Noto Sans CJK TC"/></a:minorFont></a:fontScheme><a:fmtScheme name="Smart"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme></a:themeElements></a:theme>'
    master = f'<?xml version="1.0" encoding="UTF-8"?><p:sldMaster xmlns:a="{PPTX_NS["a"]}" xmlns:r="{PPTX_NS["r"]}" xmlns:p="{PPTX_NS["p"]}"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvGrpSpPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMap accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" bg1="lt1" bg2="lt2" tx1="dk1" tx2="dk2" hlink="hlink" folHlink="folHlink"/><p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst><p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>'
    layout = f'<?xml version="1.0" encoding="UTF-8"?><p:sldLayout xmlns:a="{PPTX_NS["a"]}" xmlns:r="{PPTX_NS["r"]}" xmlns:p="{PPTX_NS["p"]}" matchingName="Blank" type="blank" preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvGrpSpPr/></p:nvGrpSpPr><p:grpSpPr/></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>'
    return {
        "[Content_Types].xml": "".join(content_types),
        "_rels/.rels": root_rels,
        "ppt/presentation.xml": "".join(presentation),
        "ppt/_rels/presentation.xml.rels": "".join(pres_rels),
        "ppt/theme/theme1.xml": theme,
        "ppt/slideMasters/slideMaster1.xml": master,
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": master_rels,
        "ppt/slideLayouts/slideLayout1.xml": layout,
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/></Relationships>',
        **{
            f"ppt/slides/_rels/slide{i}.xml.rels": f'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="{rel_ns}"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/></Relationships>'
            for i in range(1, len(slides) + 1)
        },
        **{f"ppt/slides/slide{i}.xml": slide for i, slide in enumerate(slides, 1)},
    }


def convert() -> None:
    with zipfile.ZipFile(ODP_PATH, "r") as source:
        root = ET.fromstring(source.read("content.xml"))
    slides = [_slide_xml(page, i) for i, page in enumerate(root.findall(".//" + ODF_DRAW + "page"), 1)]
    parts = _pptx_parts(slides)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(PPTX_PATH, "w", zipfile.ZIP_DEFLATED) as package:
        for path, value in parts.items():
            package.writestr(path, value)


if __name__ == "__main__":
    build_odp()
    convert()
    print(PPTX_PATH)
