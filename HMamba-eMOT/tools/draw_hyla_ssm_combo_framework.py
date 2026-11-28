#!/usr/bin/env python3
"""Paper-style framework: HyLA-MLP detector + SSM-mod-mean tracker."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 2600, 2360
BG = "#FFFFFF"
TEXT = "#1B1F27"
MUTED = "#5C6773"
LINE = "#2C3340"

BLUE, BLUE_E = "#D6E6F7", "#4E86C6"
ORANGE, ORANGE_E = "#F8D9C4", "#D07A3A"
PINK, PINK_E = "#F4CFD4", "#C45C6A"
GREEN, GREEN_E = "#D5F0E0", "#2F9A66"
YELLOW, YELLOW_E = "#F8E7B4", "#C9A227"
PURPLE, PURPLE_E = "#E4D4F3", "#7B56B0"
GRAY, GRAY_E = "#E6EBF1", "#667384"
TEAL, TEAL_E = "#CDE8E3", "#2E8A7A"
HEAD, HEAD_E = "#E8F0D8", "#6B8F3A"
WHITE = "#FFFFFF"
DASH = "#8A93A0"

FONT = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
FONTB = "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc"

img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)


def font(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


def box(xy, fill, edge, text, size=18, rad=12, width=2, tfill=TEXT):
    d.rounded_rectangle(xy, radius=rad, fill=fill, outline=edge, width=width)
    x1, y1, x2, y2 = xy
    bb = d.multiline_textbbox((0, 0), text, font=font(size, True), align="center", spacing=3)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.multiline_text(
        ((x1 + x2 - tw) / 2.0, (y1 + y2 - th) / 2.0 - 1),
        text,
        font=font(size, True),
        fill=tfill,
        align="center",
        spacing=3,
    )
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def arrow(a, b, color=LINE, width=3):
    d.line([a, b], fill=color, width=width)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    length, spread = 11, 0.52
    p1 = (b[0] - length * math.cos(ang - spread), b[1] - length * math.sin(ang - spread))
    p2 = (b[0] - length * math.cos(ang + spread), b[1] - length * math.sin(ang + spread))
    d.polygon([b, p1, p2], fill=color)


def elbow(pts, color=LINE, width=3, head=True):
    d.line(pts, fill=color, width=width)
    if head:
        arrow(pts[-2], pts[-1], color, width)


def node(cx, cy, letter, edge, r=16, size=15):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=WHITE, outline=edge, width=2)
    d.text((cx, cy + 1), letter, font=font(size, True), fill=edge, anchor="mm")


def dashed_line(a, b, color=DASH, width=2, dash=9, gap=6):
    x1, y1 = a
    x2, y2 = b
    length = math.hypot(x2 - x1, y2 - y1) or 1.0
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    t = 0.0
    while t < length:
        t2 = min(t + dash, length)
        d.line(
            [(x1 + ux * t, y1 + uy * t), (x1 + ux * t2, y1 + uy * t2)],
            fill=color,
            width=width,
        )
        t = t2 + gap


def dashed_rect(xy, color, width=2, rad=8):
    x1, y1, x2, y2 = xy
    dashed_line((x1 + rad, y1), (x2 - rad, y1), color, width)
    dashed_line((x2, y1 + rad), (x2, y2 - rad), color, width)
    dashed_line((x2 - rad, y2), (x1 + rad, y2), color, width)
    dashed_line((x1, y2 - rad), (x1, y1 + rad), color, width)
    d.arc((x1, y1, x1 + 2 * rad, y1 + 2 * rad), 180, 270, fill=color, width=width)
    d.arc((x2 - 2 * rad, y1, x2, y1 + 2 * rad), 270, 360, fill=color, width=width)
    d.arc((x2 - 2 * rad, y2 - 2 * rad, x2, y2), 0, 90, fill=color, width=width)
    d.arc((x1, y2 - 2 * rad, x1 + 2 * rad, y2), 90, 180, fill=color, width=width)


def mini_scene(xy, accent, title):
    x1, y1, x2, y2 = xy
    d.rounded_rectangle(xy, radius=12, fill=WHITE, outline=accent, width=2)
    d.rectangle((x1 + 3, y1 + 3, x2 - 3, y1 + 36), fill="#F4F7FB")
    d.text(((x1 + x2) / 2, y1 + 19), title, font=font(16, True), fill=TEXT, anchor="ma")
    rink = (x1 + 12, y1 + 44, x2 - 12, y2 - 12)
    d.rounded_rectangle(rink, radius=6, fill="#D9E7F2", outline="#B7C9D8")
    mid_y = (rink[1] + rink[3]) / 2
    d.line((rink[0] + 8, mid_y, rink[2] - 8, mid_y), fill="#C5D5E4", width=2)
    players = [
        (22, 48, accent),
        (78, 58, "#E35D5D"),
        (48, 92, "#3D7EDB"),
        (108, 84, "#6DBE45"),
        (142, 50, "#9B6BD8"),
    ]
    for px, py, c in players:
        d.rounded_rectangle(
            (x1 + px, y1 + py, x1 + px + 24, y1 + py + 30),
            radius=3,
            fill=WHITE,
            outline=c,
            width=2,
        )


def det_stack(x, y, colors):
    for i, c in enumerate(colors):
        ox, oy = (i % 3) * 108, (i // 3) * 88
        d.rounded_rectangle(
            (x + ox, y + oy, x + ox + 92, y + oy + 72),
            radius=8,
            fill=WHITE,
            outline=c,
            width=3,
        )
        d.rectangle((x + ox + 16, y + oy + 14, x + ox + 76, y + oy + 54), outline=c, width=2)


def pred_stack(x, y, colors):
    for i, c in enumerate(colors):
        ox, oy = (i % 2) * 150, (i // 2) * 78
        dashed_rect((x + ox, y + oy, x + ox + 132, y + oy + 62), c, 2, 8)
        for k in range(4):
            d.ellipse(
                (x + ox + 16 + k * 28, y + oy + 22, x + ox + 32 + k * 28, y + oy + 38),
                outline=c,
                width=1,
            )


def track_trail(x, y, color, n=5):
    for k in range(n):
        xx = x + k * 28
        d.ellipse((xx, y, xx + 16, y + 16), outline=color, width=2)
        if k:
            d.line((xx - 12, y + 8, xx, y + 8), fill=color, width=2)


# -----------------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------------
d.text((W / 2, 26), "HyLA-MLP  +  SSM-mod-mean", font=font(38, True), fill=TEXT, anchor="ma")
d.text(
    (W / 2, 70),
    "双模态检测融合  ×  空间超图均值调制的 Mamba 运动预测",
    font=font(22),
    fill=MUTED,
    anchor="ma",
)

# -----------------------------------------------------------------------------
# (a) Overall
# -----------------------------------------------------------------------------
d.text((70, 108), "(a)  总体工作流程", font=font(26, True), fill=TEXT)
d.rounded_rectangle((40, 145, 2560, 980), radius=18, outline="#D3DBE4", width=2)

mini_scene((70, 200, 320, 370), BLUE_E, "RGB  帧 i")
mini_scene((70, 400, 320, 570), ORANGE_E, "Event  帧 i")

d.rounded_rectangle((360, 230, 700, 530), radius=16, outline=GREEN_E, width=3)
box((385, 270, 675, 430), TEAL, TEAL_E, "Dual YOLOX\nHyLA-MLP", 24)
d.text((530, 462), "创新 ①", font=font(17, True), fill=GREEN_E, anchor="ma")
d.text((530, 495), "见 (b)", font=font(15), fill=MUTED, anchor="ma")
arrow((320, 285), (360, 340), BLUE_E)
arrow((320, 485), (360, 420), ORANGE_E)

d.rounded_rectangle((740, 185, 1160, 470), radius=16, outline=YELLOW_E, width=2)
d.text((950, 208), "第 i 帧检测框", font=font(18, True), fill=TEXT, anchor="ma")
det_stack(770, 240, ("#E35D5D", "#3D7EDB", "#E3A23C", "#6DBE45", "#9B6BD8"))
arrow((700, 380), (740, 380))

d.rounded_rectangle((1200, 185, 1580, 470), radius=16, outline=GRAY_E, width=2)
d.text((1390, 208), "轨迹预测框", font=font(18, True), fill=TEXT, anchor="ma")
pred_stack(1230, 245, ("#E35D5D", "#3D7EDB", "#E3A23C", "#6DBE45"))

d.rounded_rectangle((1620, 230, 2000, 500), radius=16, outline=PURPLE_E, width=3)
box((1645, 255, 1975, 400), GRAY, GRAY_E, "Motion Predictor\nSSM-mod-mean", 22)
d.text((1810, 430), "创新 ②", font=font(17, True), fill=PURPLE_E, anchor="ma")
d.text((1810, 462), "见 (c)", font=font(15), fill=MUTED, anchor="ma")
arrow((1620, 360), (1580, 360), GRAY_E)

d.rounded_rectangle((2040, 185, 2525, 760), radius=16, outline=GRAY_E, width=2)
d.text((2282, 210), "历史轨迹", font=font(18, True), fill=TEXT, anchor="ma")
for i, c in enumerate(("#E35D5D", "#3D7EDB", "#E3A23C", "#6DBE45", "#9B6BD8", "#5AA9E6")):
    track_trail(2100, 255 + i * 78, c, 6)
arrow((2040, 360), (2000, 360), GRAY_E)

box((740, 520, 1160, 600), PINK, PINK_E, "第一阶段关联    IoU + Hungarian", 18)
arrow((950, 470), (950, 520))
elbow([(1390, 470), (1390, 560), (1160, 560)])

box((1200, 520, 1580, 600), YELLOW, YELLOW_E, "线性分配", 18)
arrow((1160, 560), (1200, 560))

box((740, 670, 1160, 750), PINK, PINK_E, "第二阶段关联    Lost 轨迹补匹配", 18)
arrow((870, 600), (870, 670))
d.text((1020, 635), "未匹配检测 / 未匹配轨迹", font=font(15), fill=MUTED, anchor="ma")

box((740, 800, 1690, 880), BLUE, BLUE_E, "轨迹管理     更新   ·   激活新生   ·   标记丢失   ·   超时删除", 18)
arrow((950, 750), (950, 800))
elbow([(1390, 600), (1390, 800)])

box((1770, 800, 2150, 880), HEAD, HEAD_E, "输出当前帧轨迹", 18)
arrow((1690, 840), (1770, 840))

d.rounded_rectangle((70, 610, 330, 880), radius=14, outline="#C9D3DE", width=2)
for i, (lab, col) in enumerate(
    (("更新轨迹", GREEN_E), ("新生轨迹", HEAD_E), ("丢失轨迹", YELLOW_E), ("删除轨迹", PINK_E))
):
    yy = 640 + i * 56
    d.ellipse((95, yy, 119, yy + 24), outline=col, width=3)
    d.text((145, yy + 12), lab, font=font(16, True), fill=TEXT, anchor="lm")

d.text(
    (1280, 945),
    "HyLA-MLP 只改检测器；SSM-mod-mean 只改运动预测。关联仍是两阶段 IoU。",
    font=font(16),
    fill=MUTED,
    anchor="ma",
)

# -----------------------------------------------------------------------------
# (b) HyLA-MLP
# -----------------------------------------------------------------------------
d.text((70, 1005), "(b)  创新模块 ①  HyLA-MLP", font=font(26, True), fill=TEXT)
d.rounded_rectangle((40, 1040, 1280, 1880), radius=18, outline=GREEN_E, width=3)

box((80, 1110, 175, 1185), BLUE, BLUE_E, "R", 26)
box((80, 1485, 175, 1560), ORANGE, ORANGE_E, "E", 26)

d.rounded_rectangle((215, 1240, 560, 1455), radius=14, outline=PURPLE_E, width=2)
d.text((387, 1265), "共享残差 MLP   同一组 θ", font=font(16, True), fill=TEXT, anchor="ma")
box((240, 1310, 310, 1360), BLUE, BLUE_E, "1×1", 16)
node(350, 1335, "σ", PURPLE_E)
box((385, 1310, 455, 1360), BLUE, BLUE_E, "1×1", 16)
node(510, 1335, "+", GRAY_E)
arrow((310, 1335), (334, 1335))
arrow((366, 1335), (385, 1335))
arrow((455, 1335), (494, 1335))
d.line((240, 1395, 240, 1425, 510, 1425, 510, 1351), fill=LINE, width=2)
d.text((387, 1435), "RGB / Event 共用", font=font(14), fill=MUTED, anchor="ma")
arrow((175, 1147), (215, 1320), BLUE_E)
arrow((175, 1522), (215, 1370), ORANGE_E)

d.rounded_rectangle((590, 1085, 1245, 1575), radius=14, outline=GREEN_E, width=2)
d.text((917, 1108), "Bi-HyLA", font=font(18, True), fill=TEXT, anchor="ma")
arrow((560, 1347), (590, 1347), GREEN_E)
d.line((590, 1192, 590, 1452), fill=GREEN_E, width=3)

box((615, 1165, 705, 1220), BLUE, BLUE_E, "R̃", 18)
box((615, 1425, 705, 1480), ORANGE, ORANGE_E, "Ẽ", 18)
arrow((590, 1192), (615, 1192), BLUE_E)
arrow((590, 1452), (615, 1452), ORANGE_E)
node(780, 1325, "C", PINK_E, 18)
d.text((780, 1360), "Query", font=font(13), fill=MUTED, anchor="ma")
arrow((705, 1192), (765, 1310), BLUE_E)
arrow((705, 1452), (765, 1340), ORANGE_E)

box((840, 1185, 920, 1235), BLUE, BLUE_E, "Q", 18)
box((840, 1295, 920, 1345), ORANGE, ORANGE_E, "K", 18)
box((840, 1405, 920, 1455), GREEN, GREEN_E, "V", 18)
arrow((798, 1315), (840, 1210))
arrow((798, 1325), (840, 1320))
arrow((798, 1335), (840, 1430))

box((970, 1265, 1100, 1385), YELLOW, YELLOW_E, "划窗", 18)
arrow((920, 1210), (970, 1290))
arrow((920, 1320), (970, 1325))
arrow((920, 1430), (970, 1360))

box((1140, 1165, 1225, 1235), GREEN, GREEN_E, "A", 22)
box((1130, 1290, 1235, 1410), YELLOW, YELLOW_E, "×A\nσ\n×A", 15)
arrow((1100, 1305), (1140, 1200))
arrow((1100, 1335), (1130, 1350))
d.text((1182, 1448), "同一 A 作用两次", font=font(13), fill=MUTED, anchor="ma")

d.text((917, 1545), "交叉注意力后仍保持两路增强特征", font=font(14), fill=MUTED, anchor="ma")

box((90, 1630, 190, 1690), BLUE, BLUE_E, "R′", 18)
box((90, 1730, 190, 1790), ORANGE, ORANGE_E, "E′", 18)
node(280, 1710, "C", PINK_E, 18)
node(390, 1710, "σ", PURPLE_E, 18)
box((450, 1670, 620, 1750), YELLOW, YELLOW_E, "门控校正 δ", 16)
node(700, 1710, "+", GRAY_E, 20)
box((780, 1640, 1020, 1780), TEAL, TEAL_E, "Fl", 32)
arrow((190, 1660), (264, 1695), BLUE_E)
arrow((190, 1760), (264, 1725), ORANGE_E)
arrow((296, 1710), (374, 1710), PINK_E)
arrow((406, 1710), (450, 1710))
arrow((620, 1710), (680, 1710))
arrow((720, 1710), (780, 1710))
d.text((280, 1762), "拼接", font=font(13), fill=MUTED, anchor="ma")
d.text((700, 1798), "R′ + E′ + δ", font=font(14), fill=MUTED, anchor="ma")
d.text(
    (660, 1838),
    "每个尺度出口是单个张量 Fl，再送入共享 PA-FPN / Head",
    font=font(16),
    fill=MUTED,
    anchor="ma",
)

# -----------------------------------------------------------------------------
# (c) SSM-mod-mean
# -----------------------------------------------------------------------------
d.text((1330, 1005), "(c)  创新模块 ②  SSM-mod-mean", font=font(26, True), fill=TEXT)
d.rounded_rectangle((1310, 1040, 2560, 1880), radius=18, outline=PURPLE_E, width=3)

box((1350, 1095, 1595, 1215), BLUE, BLUE_E, "当前帧目标\n运动差分顶点", 18)

d.rounded_rectangle((1645, 1075, 2075, 1300), radius=14, fill=YELLOW, outline=YELLOW_E, width=2)
d.text((1860, 1096), "规则超边", font=font(17, True), fill=TEXT, anchor="ma")
box((1670, 1128, 1850, 1182), PINK, PINK_E, "spatial", 16)
box((1865, 1128, 2045, 1182), BLUE, BLUE_E, "rgb", 16)
box((1670, 1200, 1850, 1254), ORANGE, ORANGE_E, "evt", 16)
box((1865, 1200, 2045, 1254), GREEN, GREEN_E, "motion", 16)
d.text((1860, 1278), "超边 → 顶点", font=font(14), fill=MUTED, anchor="ma")
arrow((1595, 1155), (1645, 1155))

d.rounded_rectangle((2120, 1075, 2525, 1335), radius=14, outline=PURPLE_E, width=2)
d.text((2322, 1098), "均匀均值  mean", font=font(16, True), fill=TEXT, anchor="ma")
for i, c in enumerate(("#E35D5D", "#3D7EDB", "#6DBE45")):
    cx = 2205 + i * 95
    if i == 1:
        dashed_line((cx - 16, 1165), (cx + 16, 1197), MUTED, 2, 4, 3)
        dashed_line((cx + 16, 1165), (cx - 16, 1197), MUTED, 2, 4, 3)
        d.ellipse((cx - 16, 1165, cx + 16, 1197), outline=MUTED, width=2)
        d.text((cx, 1220), "去掉自身", font=font(13), fill=MUTED, anchor="ma")
    else:
        d.ellipse((cx - 16, 1165, cx + 16, 1197), outline=c, width=3)
        dashed_line((cx, 1197), (2322, 1260), c, 2)
node(2322, 1280, "u", PURPLE_E, 18)
d.text((2322, 1318), "leave-one-out", font=font(13), fill=MUTED, anchor="ma")
arrow((2075, 1205), (2120, 1205), PURPLE_E)

box((1350, 1420, 1595, 1545), GRAY, GRAY_E, "Mamba 参数\nΔ     B     C", 18)

d.rounded_rectangle((1645, 1385, 2075, 1620), radius=14, outline=PURPLE_E, width=2)
d.text((1860, 1408), "残差调制", font=font(17, True), fill=TEXT, anchor="ma")
node(1710, 1488, "C", PINK_E, 16)
d.text((1710, 1520), "x , u", font=font(13), fill=MUTED, anchor="ma")
box((1775, 1458, 1910, 1518), PURPLE, PURPLE_E, "MLP", 16)
node(1975, 1488, "σ", PURPLE_E, 16)
arrow((1726, 1488), (1775, 1488))
arrow((1910, 1488), (1959, 1488))
d.text((1860, 1560), "Δ  加法修正", font=font(15, True), fill=TEXT, anchor="ma")
d.text((1860, 1595), "B / C  乘法门控", font=font(15, True), fill=TEXT, anchor="ma")
arrow((1595, 1482), (1645, 1482))
elbow([(2322, 1335), (2322, 1488), (2075, 1488)], PURPLE_E)

box((2120, 1420, 2525, 1595), TEAL, TEAL_E, "Bi-Mamba 扫描\n预测下一框", 20)
arrow((2075, 1507), (2120, 1507))

d.text(
    (1935, 1688),
    "mean：邻居不做注意力加权，只做均匀平均，再去调制 SSM 选择参数",
    font=font(16),
    fill=MUTED,
    anchor="ma",
)
d.text(
    (1935, 1740),
    "空间先验只进入运动模型，不改变检测特征，也不改关联代价",
    font=font(16),
    fill=MUTED,
    anchor="ma",
)
d.text(
    (1935, 1792),
    "输出是预测框，送回 (a) 与检测框做 IoU 匹配",
    font=font(16),
    fill=MUTED,
    anchor="ma",
)

# -----------------------------------------------------------------------------
# Legend
# -----------------------------------------------------------------------------
d.rounded_rectangle((40, 1910, 2560, 2320), radius=16, outline="#D3DBE4", width=2)
d.text((80, 1940), "图例", font=font(22, True), fill=TEXT)

d.rounded_rectangle((80, 1995, 150, 2040), radius=8, fill=WHITE, outline="#E35D5D", width=3)
d.text((166, 2017), "检测框", font=font(16, True), fill=TEXT, anchor="lm")
dashed_rect((300, 1995, 370, 2040), GRAY_E, 2, 8)
d.text((386, 2017), "预测框", font=font(16, True), fill=TEXT, anchor="lm")
d.rounded_rectangle((500, 1995, 570, 2040), radius=8, fill=TEAL, outline=TEAL_E, width=2)
d.text((586, 2017), "融合特征 Fl", font=font(16, True), fill=TEXT, anchor="lm")
d.rounded_rectangle((780, 1995, 850, 2040), radius=8, fill=PURPLE, outline=PURPLE_E, width=2)
d.text((866, 2017), "空间先验 u", font=font(16, True), fill=TEXT, anchor="lm")
node(1100, 2017, "C", PINK_E, 16)
d.text((1130, 2017), "拼接", font=font(16, True), fill=TEXT, anchor="lm")
node(1260, 2017, "σ", PURPLE_E, 16)
d.text((1290, 2017), "非线性", font=font(16, True), fill=TEXT, anchor="lm")
node(1440, 2017, "+", GRAY_E, 16)
d.text((1470, 2017), "残差 / 相加", font=font(16, True), fill=TEXT, anchor="lm")
node(1680, 2017, "u", PURPLE_E, 16)
d.text((1710, 2017), "leave-one-out 均值先验", font=font(16, True), fill=TEXT, anchor="lm")

d.text((80, 2095), "流程要点", font=font(20, True), fill=TEXT)
d.text(
    (80, 2145),
    "检测：RGB 与 Event 经双路 CSPDarknet，在 dark3 / dark4 / dark5 做 HyLA-MLP。共享 MLP 对齐两模态，Bi-HyLA 用拼接 Query 做窗口交叉注意力，出口把两路增强特征相加并门控校正，得到单个 Fl。",
    font=font(17),
    fill=MUTED,
)
d.text(
    (80, 2210),
    "跟踪：历史目标构成 spatial / rgb / evt / motion 超图。mean 聚合得到 u，残差调制 Mamba 的 Δ、B、C 后双向扫描，预测当前框，再与检测做两阶段 IoU 关联。",
    font=font(17),
    fill=MUTED,
)
d.text(
    (80, 2270),
    "组合方式：两个模块分别改检测器与运动预测器，不在同一张量上串联。",
    font=font(17),
    fill=MUTED,
)

out = Path("/home/lyt/mymodel/Mamba_Trackers/docs/hyla_mlp_ssm_mod_mean_framework.png")
out.parent.mkdir(parents=True, exist_ok=True)
img.save(out, "PNG", optimize=True, dpi=(180, 180))
print(out)
