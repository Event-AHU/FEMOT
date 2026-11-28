#!/usr/bin/env python3
"""Paper-style Backbone / Neck / Head figure for dual-modal YOLOX + HyLA."""

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2480, 1480
BG = "#FFFFFF"
TEXT = "#1A1A1A"
MUTED = "#5A5A5A"
LINE = "#2B2B2B"

BLUE = "#D7E8F8"
BLUE_E = "#4F86C4"
YELLOW = "#FFF3C4"
YELLOW_E = "#C9A227"
PINK = "#F7D4D4"
PINK_E = "#C56B6B"
GREEN = "#D8F3E4"
GREEN_E = "#2E9A64"
PURPLE = "#E8D6F3"
PURPLE_E = "#8A5AAF"
ORANGE = "#FFE4CC"
ORANGE_E = "#D07A2C"
DARK = "#4E7FBA"
DARK_T = "#FFFFFF"
GRAY = "#F3F4F6"
GRAY_E = "#8A93A0"

FONT_PATH = "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc"
FONT_BOLD_PATH = "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD_PATH if bold else FONT_PATH, size)


img = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img)


def box(xy, fill, edge, text, size=20, radius=10, width=2, tfill=TEXT):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=edge, width=width)
    x1, y1, x2, y2 = xy
    fnt = font(size, True)
    bbox = draw.multiline_textbbox((0, 0), text, font=fnt, align="center", spacing=4)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        ((x1 + x2 - tw) / 2, (y1 + y2 - th) / 2),
        text,
        font=fnt,
        fill=tfill,
        align="center",
        spacing=4,
    )
    return ((x1 + x2) / 2, (y1 + y2) / 2, x1, y1, x2, y2)


def arrow(start, end, color=LINE, width=2):
    draw.line([start, end], fill=color, width=width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    length, spread = 10, 0.5
    p1 = (end[0] - length * math.cos(ang - spread), end[1] - length * math.sin(ang - spread))
    p2 = (end[0] - length * math.cos(ang + spread), end[1] - length * math.sin(ang + spread))
    draw.polygon([end, p1, p2], fill=color)


def hline(x1, x2, y, color=LINE, width=2):
    draw.line((x1, y, x2, y), fill=color, width=width)


def vline(x, y1, y2, color=LINE, width=2):
    draw.line((x, y1, x, y2), fill=color, width=width)


def letter_circle(cx, cy, letter, r=13, edge=GREEN_E):
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#FFFFFF", outline=edge, width=2)
    draw.text((cx, cy + 1), letter, font=font(16, True), fill=edge, anchor="mm")


# Title
draw.text((WIDTH // 2, 28), "Dual-modal YOLOX  (Shared MLP + HyLA)", font=font(36, True), fill=TEXT, anchor="ma")

# Column guides
col = [70, 820, 1680, 2410]
for x in col[1:3]:
    draw.line((x, 430, x, 1380), fill="#E4E7EC", width=2)
draw.text(((col[0] + col[1]) / 2, 1420), "Backbone", font=font(28, True), fill=TEXT, anchor="ma")
draw.text(((col[1] + col[2]) / 2, 1420), "Neck", font=font(28, True), fill=TEXT, anchor="ma")
draw.text(((col[2] + col[3]) / 2, 1420), "Head", font=font(28, True), fill=TEXT, anchor="ma")

# ---------------------------------------------------------------------------
# Top dashed insets
# ---------------------------------------------------------------------------
def inset(xy, title):
    draw.rounded_rectangle(xy, radius=12, outline="#B8C0CC", width=2)
    draw.text(((xy[0] + xy[2]) / 2, xy[1] + 18), title, font=font(18, True), fill=TEXT, anchor="ma")


# Shared MLP
inset((90, 70, 560, 400), "Shared MLP")
box((150, 150, 230, 195), BLUE, BLUE_E, "1×1", 18)
letter_circle(280, 172, "σ", edge=PURPLE_E)
box((330, 150, 410, 195), BLUE, BLUE_E, "1×1", 18)
letter_circle(455, 172, "+", edge=GRAY_E)
arrow((230, 172), (267, 172))
arrow((293, 172), (330, 172))
arrow((410, 172), (442, 172))
draw.line((150, 230, 150, 250, 455, 250, 455, 185), fill=LINE, width=2)
box((180, 290, 470, 345), YELLOW, YELLOW_E, "RGB / Event 共用参数", 17)

# HyLA
inset((590, 70, 1320, 400), "HyLA")
box((620, 130, 700, 170), BLUE, BLUE_E, "RGB", 16)
box((620, 200, 700, 240), ORANGE, ORANGE_E, "Event", 16)
letter_circle(750, 185, "C", edge=PINK_E)
box((800, 145, 860, 185), BLUE, BLUE_E, "Q", 18)
box((800, 205, 860, 245), ORANGE, ORANGE_E, "K/V", 16)
arrow((700, 150), (737, 175))
arrow((700, 220), (737, 195))
arrow((763, 175), (800, 165))
arrow((763, 195), (800, 225))
box((900, 155, 1020, 235), GREEN, GREEN_E, "划窗\nQ K V", 16)
arrow((860, 165), (900, 180))
arrow((860, 225), (900, 210))
box((1060, 140, 1140, 190), GREEN, GREEN_E, "A", 20)
box((1060, 210, 1280, 270), YELLOW, YELLOW_E, "V → ×A → σ → ×A", 16)
arrow((1020, 195), (1060, 165))
arrow((1020, 195), (1060, 240))
box((1180, 310, 1280, 355), PINK, PINK_E, "残差 +", 16)

# Decoupled head inset
inset((1350, 70, 1780, 400), "Decoupled Head")
box((1420, 145, 1710, 185), BLUE, BLUE_E, "Stem  1×1", 18)
vline(1565, 185, 215)
hline(1480, 1650, 215)
box((1410, 230, 1540, 275), PURPLE, PURPLE_E, "Cls 3×3×2", 16)
box((1580, 230, 1710, 275), YELLOW, YELLOW_E, "Reg 3×3×2", 16)
arrow((1480, 215), (1475, 230))
arrow((1650, 215), (1645, 230))
box((1410, 310, 1540, 355), BLUE, BLUE_E, "Classification", 15)
box((1580, 310, 1640, 355), PURPLE, PURPLE_E, "Reg", 15)
box((1650, 310, 1710, 355), GREEN, GREEN_E, "Obj", 15)
arrow((1475, 275), (1475, 310))
arrow((1645, 275), (1610, 310))
arrow((1645, 275), (1680, 310))

# PA-FPN inset
inset((1810, 70, 2390, 400), "PA-FPN block")
box((1850, 145, 1960, 185), PINK, PINK_E, "Concatenate", 15)
box((2000, 145, 2110, 185), YELLOW, YELLOW_E, "CSPLayer", 16)
box((2150, 145, 2260, 185), BLUE, BLUE_E, "Upsample", 15)
box((1850, 230, 1960, 270), BLUE, BLUE_E, "1×1 Conv", 15)
box((2000, 230, 2110, 270), BLUE, BLUE_E, "3×3 Down", 15)
box((2150, 230, 2350, 270), GREEN, GREEN_E, "HyLA Fusion", 16)
box((1880, 310, 2320, 360), GRAY, GRAY_E, "自上而下融合后再自下而上回传", 16)

def poly(points, color=LINE, width=2):
    draw.line(points, fill=color, width=width)
    arrow(points[-2], points[-1], color, width)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
bw = 145
rgb_x, evt_x = 95, 270
y_in, y_stem, y_d2, y_d3, y_d4, y_d5 = 470, 545, 630, 760, 900, 1040

box((rgb_x, y_in, rgb_x + bw, y_in + 42), BLUE, BLUE_E, "RGB", 18)
box((evt_x, y_in, evt_x + bw, y_in + 42), ORANGE, ORANGE_E, "Event", 18)
box((rgb_x, y_stem, rgb_x + bw, y_stem + 42), BLUE, BLUE_E, "Stem", 17)
box((evt_x, y_stem, evt_x + bw, y_stem + 42), ORANGE, ORANGE_E, "Stem", 17)
box((rgb_x, y_d2, rgb_x + bw, y_d2 + 42), YELLOW, YELLOW_E, "Dark2", 17)
box((evt_x, y_d2, evt_x + bw, y_d2 + 42), YELLOW, YELLOW_E, "Dark2", 17)
box((rgb_x, y_d3, rgb_x + bw, y_d3 + 42), YELLOW, YELLOW_E, "Dark3", 17)
box((evt_x, y_d3, evt_x + bw, y_d3 + 42), YELLOW, YELLOW_E, "Dark3", 17)
box((rgb_x, y_d4, rgb_x + bw, y_d4 + 42), YELLOW, YELLOW_E, "Dark4", 17)
box((evt_x, y_d4, evt_x + bw, y_d4 + 42), YELLOW, YELLOW_E, "Dark4", 17)
box((rgb_x, y_d5, rgb_x + bw, y_d5 + 42), DARK, DARK, "Dark5", 17, tfill=DARK_T)
box((evt_x, y_d5, evt_x + bw, y_d5 + 42), DARK, DARK, "Dark5", 17, tfill=DARK_T)

for x in (rgb_x + bw / 2, evt_x + bw / 2):
    for a, b in (
        (y_in + 42, y_stem),
        (y_stem + 42, y_d2),
        (y_d2 + 42, y_d3),
        (y_d3 + 42, y_d4),
        (y_d4 + 42, y_d5),
    ):
        arrow((x, a), (x, b))

hyla_x1, hyla_x2 = 455, 680
for y, name in ((y_d3, "HyLA"), (y_d4, "HyLA"), (y_d5, "HyLA")):
    box((hyla_x1, y, hyla_x2, y + 42), GREEN, GREEN_E, name, 18)
    arrow((rgb_x + bw, y + 21), (hyla_x1, y + 21), BLUE_E)
    arrow((evt_x + bw, y + 21), (hyla_x1, y + 21), ORANGE_E)

# ---------------------------------------------------------------------------
# Neck
# ---------------------------------------------------------------------------
fx = 720
box((fx, y_d3, fx + 80, y_d3 + 42), GREEN, GREEN_E, "F3", 18)
box((fx, y_d4, fx + 80, y_d4 + 42), GREEN, GREEN_E, "F4", 18)
box((fx, y_d5, fx + 80, y_d5 + 42), GREEN, GREEN_E, "F5", 18)
for y in (y_d3, y_d4, y_d5):
    arrow((hyla_x2, y + 21), (fx, y + 21), GREEN_E)

# top-down FPN
box((840, y_d5, 960, y_d5 + 42), BLUE, BLUE_E, "1×1", 17)
arrow((fx + 80, y_d5 + 21), (840, y_d5 + 21))
box((840, 970, 960, 1012), BLUE, BLUE_E, "Upsample", 15)
arrow((900, y_d5), (900, 1012))
box((1000, y_d4, 1160, y_d4 + 42), PINK, PINK_E, "Concatenate", 15)
arrow((fx + 80, y_d4 + 21), (1000, y_d4 + 21))
arrow((900, 970), (1000, y_d4 + 21))
box((1000, 830, 1180, 878), YELLOW, YELLOW_E, "CSPLayer", 16)
arrow((1080, y_d4), (1080, 878))

box((1230, 830, 1350, 878), BLUE, BLUE_E, "1×1", 17)
arrow((1180, 854), (1230, 854))
box((1230, 780, 1350, 818), BLUE, BLUE_E, "Upsample", 15)
arrow((1290, 830), (1290, 818))
box((1000, y_d3, 1160, y_d3 + 42), PINK, PINK_E, "Concatenate", 15)
arrow((fx + 80, y_d3 + 21), (1000, y_d3 + 21))
poly([(1290, 780), (1290, 781), (1180, y_d3 + 21)])
box((1000, 690, 1220, 742), YELLOW, YELLOW_E, "CSPLayer", 16)
arrow((1080, y_d3), (1080, 742))
box((1260, 690, 1360, 742), GREEN, GREEN_E, "P3", 20)
arrow((1220, 716), (1260, 716))

# bottom-up PAN
box((1000, 610, 1180, 652), BLUE, BLUE_E, "3×3 Down", 16)
arrow((1080, 690), (1080, 652))
box((1000, 540, 1160, 582), PINK, PINK_E, "Concatenate", 15)
arrow((1080, 610), (1080, 582))
poly([(1290, 854), (1410, 854), (1410, 561), (1160, 561)])
box((1000, 470, 1220, 522), YELLOW, YELLOW_E, "CSPLayer", 16)
arrow((1080, 540), (1080, 522))
box((1260, 470, 1360, 522), GREEN, GREEN_E, "P4", 20)
arrow((1220, 496), (1260, 496))

box((1410, 470, 1530, 522), BLUE, BLUE_E, "3×3 Down", 16)
arrow((1360, 496), (1410, 496))
box((1000, y_d5, 1160, y_d5 + 42), PINK, PINK_E, "Concatenate", 15)
poly([(960, y_d5 + 21), (980, y_d5 + 21), (1000, y_d5 + 21)])
poly([(1470, 522), (1470, y_d5 + 21), (1160, y_d5 + 21)])
box((1230, y_d5, 1410, y_d5 + 42), YELLOW, YELLOW_E, "CSPLayer", 16)
arrow((1160, y_d5 + 21), (1230, y_d5 + 21))
box((1450, y_d5, 1550, y_d5 + 42), GREEN, GREEN_E, "P5", 20)
arrow((1410, y_d5 + 21), (1450, y_d5 + 21))

# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------
def draw_head(hx, hy, title):
    box((hx, hy, hx + 170, hy + 38), YELLOW, YELLOW_E, title, 16)
    box((hx, hy + 50, hx + 170, hy + 90), BLUE, BLUE_E, "Stem 1×1", 16)
    arrow((hx + 85, hy + 38), (hx + 85, hy + 50))
    box((hx + 200, hy + 8, hx + 390, hy + 42), PURPLE, PURPLE_E, "Classification", 15)
    box((hx + 200, hy + 52, hx + 390, hy + 86), YELLOW, YELLOW_E, "Regression", 15)
    box((hx + 200, hy + 96, hx + 390, hy + 130), GREEN, GREEN_E, "Objectness", 15)
    arrow((hx + 170, hy + 70), (hx + 200, hy + 25))
    arrow((hx + 170, hy + 70), (hx + 200, hy + 69))
    arrow((hx + 170, hy + 70), (hx + 200, hy + 113))


draw_head(1760, 458, "P4  /  16")
draw_head(1760, 678, "P3  /  8")
draw_head(1760, 1010, "P5  /  32")
poly([(1360, 496), (1680, 496), (1680, 477), (1760, 477)])
poly([(1360, 716), (1680, 716), (1680, 697), (1760, 697)])
poly([(1550, y_d5 + 21), (1680, y_d5 + 21), (1680, 1029), (1760, 1029)])

output = Path(__file__).resolve().parents[1] / "docs" / "hyla_shared_mlp_architecture.png"
output.parent.mkdir(parents=True, exist_ok=True)
img.save(output, format="PNG", optimize=True, dpi=(180, 180))
print(output)
