#!/usr/bin/env python3
"""Redraw the three MOT schematics in a sparse Visio-like paper style.

The scene dictionaries are rendered to PNG for preview and serialized to
native, editable VSDX shapes. The same geometry is therefore used in both
outputs; no screenshot is embedded in the Visio document.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont

try:
    from export_diagram_vsdx import export_vsdx
except ImportError:  # Support importing the script as tools.draw_hyla_ssm_three_figures.
    from tools.export_diagram_vsdx import export_vsdx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)

FONT = "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"
FONT_B = "/usr/share/fonts/liberation/LiberationSans-Bold.ttf"
FONT_I = "/usr/share/fonts/liberation/LiberationSans-Italic.ttf"

BLUE = "#3566B8"
BLUE_LIGHT = "#EAF1FB"
BLUE_MID = "#B8CBE9"
PURPLE = "#7651B6"
PURPLE_LIGHT = "#F0EAF9"
ORANGE = "#D56B1F"
ORANGE_LIGHT = "#FFF0E4"
GREEN = "#4D8C3F"
GREEN_LIGHT = "#EAF4E7"
YELLOW = "#C18A00"
YELLOW_LIGHT = "#FFF7D5"
RED = "#C84B4B"
RED_LIGHT = "#FAE7E7"
INK = "#1B2430"
GRAY = "#8794A5"
GRAY_LIGHT = "#F5F7FA"
WHITE = "#FFFFFF"


def scn(name: str, width: int, height: int) -> Dict[str, Any]:
    return {"name": name, "width": width, "height": height, "items": []}


def add(s: Dict[str, Any], kind: str, **kw: Any) -> Dict[str, Any]:
    item = {"type": kind, **kw}
    s["items"].append(item)
    return item


def rect(s, x, y, w, h, fill=WHITE, stroke=BLUE, sw=1.5, r=8, dashed=False):
    return add(s, "rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
               width=sw, r=r, dashed=dashed)


def ellipse(s, cx, cy, rx, ry, fill=WHITE, stroke=BLUE, sw=1.5, dashed=False):
    return add(s, "ellipse", cx=cx, cy=cy, rx=rx, ry=ry, fill=fill,
               stroke=stroke, width=sw, dashed=dashed)


def poly(s, points, fill=WHITE, stroke=BLUE, sw=1.5, dashed=False):
    return add(s, "polygon", points=points, fill=fill, stroke=stroke,
               width=sw, dashed=dashed)


def path(s, points, color=BLUE, sw=1.5, arrow=True, dashed=False):
    return add(s, "polyline", points=points, fill=None, stroke=color,
               width=sw, arrow=arrow, dashed=dashed)


def label(s, x, y, value, size=14, color=INK, bold=False, italic=False,
          align="center", w=None, h=None):
    item = {"text": value, "x": x, "y": y, "size": size, "color": color,
            "font": "Arial", "bold": bold, "italic": italic, "align": align}
    if w is not None:
        item["w"] = w
    if h is not None:
        item["h"] = h
    return add(s, "text", **item)


def title(s, number, heading, subtitle):
    label(s, s["width"] / 2, 26, f"({number}) {heading}", 22, PURPLE, True)
    label(s, s["width"] / 2, 53, subtitle, 11, GRAY, False, True)


def font(size, bold=False, italic=False):
    return ImageFont.truetype(FONT_B if bold else FONT_I if italic else FONT, max(8, int(size)))


def dashed_line(draw, pts, fill, width, scale):
    for p0, p1 in zip(pts[:-1], pts[1:]):
        x0, y0 = p0[0] * scale, p0[1] * scale
        x1, y1 = p1[0] * scale, p1[1] * scale
        dist = math.hypot(x1 - x0, y1 - y0) or 1
        ux, uy = (x1 - x0) / dist, (y1 - y0) / dist
        t = 0
        while t < dist:
            a, b = t, min(t + 7 * scale, dist)
            draw.line((x0 + ux*a, y0 + uy*a, x0 + ux*b, y0 + uy*b), fill=fill, width=max(1, int(width*scale)))
            t += 12 * scale


def render(scene: Dict[str, Any], scale=2):
    im = Image.new("RGB", (int(scene["width"] * scale), int(scene["height"] * scale)), WHITE)
    draw = ImageDraw.Draw(im)
    for item in scene["items"]:
        kind = item["type"]
        if kind == "rect":
            xy = tuple(int(v * scale) for v in (item["x"], item["y"], item["x"]+item["w"], item["y"]+item["h"]))
            if item.get("dashed"):
                draw.rounded_rectangle(xy, radius=int(item.get("r", 0)*scale), fill=item.get("fill"), outline=None)
                dashed_line(draw, [(item["x"], item["y"]), (item["x"]+item["w"], item["y"]), (item["x"]+item["w"], item["y"]+item["h"]), (item["x"], item["y"]+item["h"]), (item["x"], item["y"])], item.get("stroke", INK), item.get("width", 1), scale)
            else:
                draw.rounded_rectangle(xy, radius=int(item.get("r", 0)*scale), fill=item.get("fill"), outline=item.get("stroke"), width=max(1, int(item.get("width", 1)*scale)))
        elif kind == "ellipse":
            xy = tuple(int(v * scale) for v in (item["cx"]-item["rx"], item["cy"]-item["ry"], item["cx"]+item["rx"], item["cy"]+item["ry"]))
            draw.ellipse(xy, fill=item.get("fill"), outline=item.get("stroke"), width=max(1, int(item.get("width", 1)*scale)))
        elif kind == "polygon":
            points = [(int(x*scale), int(y*scale)) for x, y in item["points"]]
            draw.polygon(points, fill=item.get("fill"), outline=item.get("stroke"))
        elif kind == "polyline":
            points = [(int(x*scale), int(y*scale)) for x, y in item["points"]]
            if item.get("dashed"):
                dashed_line(draw, item["points"], item.get("stroke", INK), item.get("width", 1), scale)
            else:
                draw.line(points, fill=item.get("stroke", INK), width=max(1, int(item.get("width", 1)*scale)), joint="curve")
            if item.get("arrow", False):
                x0, y0 = points[-2]; x1, y1 = points[-1]
                ang = math.atan2(y1-y0, x1-x0)
                p1 = (x1 - 9*scale*math.cos(ang-.45), y1 - 9*scale*math.sin(ang-.45))
                p2 = (x1 - 9*scale*math.cos(ang+.45), y1 - 9*scale*math.sin(ang+.45))
                draw.polygon(((x1, y1), p1, p2), fill=item.get("stroke", INK))
        elif kind == "text":
            xy = (item["x"]*scale, item["y"]*scale)
            anchor = {"center": "mm", "left": "lm", "right": "rm"}.get(item.get("align", "center"), "mm")
            draw.multiline_text(xy, item["text"], fill=item.get("color", INK), font=font(item.get("size", 14)*scale, item.get("bold", False), item.get("italic", False)), anchor=anchor, align="center", spacing=2*scale)
    return im


def sheet(s, x, y, colors, w=30, h=62, gap=2, outline=GRAY):
    for i, color in enumerate(colors):
        rect(s, x+i*(w+gap), y, w, h, color, outline, 0.7, 2)


def frame_pair(s, x, y):
    rect(s, x, y, 46, 37, "#E9E9E9", RED, 1.2, 3)
    rect(s, x+5, y+5, 36, 27, "#D3D3D3", "#777777", .7, 1)
    path(s, [(x+8, y+28), (x+33, y+7)], "#777777", .7, False)


def hypergraph(s, cx, cy, scale=1):
    groups = [(-36, -18, 62, 48, RED_LIGHT, RED), (8, -24, 60, 46, BLUE_LIGHT, BLUE), (-15, 15, 72, 45, ORANGE_LIGHT, ORANGE), (38, 14, 56, 42, GREEN_LIGHT, GREEN)]
    for dx, dy, rx, ry, fill, edge in groups:
        ellipse(s, cx+dx*scale, cy+dy*scale, rx*scale, ry*scale, fill, edge, 1, True)
    nodes = [(-55, -25, RED), (-23, -7, BLUE), (7, -30, ORANGE), (36, -4, GREEN), (-33, 28, YELLOW), (5, 23, PURPLE), (42, 28, RED)]
    for dx, dy, color in nodes:
        ellipse(s, cx+dx*scale, cy+dy*scale, 8*scale, 8*scale, color, WHITE, .8)
    for a, b, color in [((-55, -25), (-23, -7), RED), ((-23, -7), (36, -4), BLUE), ((7, -30), (5, 23), ORANGE), ((36, -4), (42, 28), GREEN)]:
        path(s, [(cx+a[0]*scale, cy+a[1]*scale), (cx+b[0]*scale, cy+b[1]*scale)], color, 1, False)


def tensor_stack(s, x, y, w, h, colors, offset=5, outline="#8794A5"):
    """Draw a compact stacked tensor block, closer to paper figure notation."""
    for i, color in enumerate(colors):
        rect(s, x + i * offset, y - i * offset, w, h, color, outline, .7, 1)


def incidence_matrix(s, x, y, rows=4, cols=8, cell=13, gap=2,
                     colors=(RED, BLUE, ORANGE, GREEN)):
    """Small typed incidence matrix with one row per hyperedge family."""
    for r in range(rows):
        for c in range(cols):
            # A sparse, deterministic pattern makes the matrix readable while
            # still conveying that each edge selects several vertices.
            active = ((c + 2 * r) % 5 != 1 and (c + r) % 4 != 3)
            fill = colors[r % len(colors)] if active else WHITE
            rect(s, x + c * (cell + gap), y + r * (cell + gap), cell, cell,
                 fill, "#B8C2D0", .45, 1)


def hypergraph_construction(s, x, y, w=330, h=220):
    """Reference-like HyperConv panel used in the SSM detail figure.

    The construction is deliberately schematic: colored closed curves encode
    the four typed hyperedges, while small circles are the object/trajectory
    vertices.  The lower row makes the V2E/E2V operations explicit.
    """
    rect(s, x, y, w, h, "#FFF9E7", "#D8C99C", 1.0, 10)
    label(s, x + w / 2, y + 16, "Hypergraph construction", 12, INK, True)
    # upper construction strip
    construction_w = min(430, max(260, w * .43))
    rect(s, x + 16, y + 31, construction_w, 53, WHITE, GRAY, .9, 5, True)
    label(s, x + 26, y + 41, "objects", 8, GRAY, True, align="left")
    nx0 = x + 62
    nodes = [(nx0, y+63, GRAY), (nx0+42, y+57, GRAY), (nx0+86, y+69, BLUE),
             (nx0+138, y+54, RED), (nx0+190, y+70, GREEN), (nx0+239, y+48, ORANGE),
             (nx0+115, y+43, PURPLE)]
    # Four typed hyperedges, drawn as colored loops and short links.
    ellipse(s, x+135, y+57, 55, 20, "#FDEBEC", RED, 1.7, True)
    ellipse(s, x+208, y+61, 57, 19, "#EAF1FB", BLUE, 1.7, True)
    ellipse(s, x+185, y+47, 42, 28, "#FFF0E4", ORANGE, 1.5, True)
    ellipse(s, x+275, y+60, 42, 18, "#EAF4E7", GREEN, 1.5, True)
    for nx, ny, color in nodes:
        ellipse(s, nx, ny, 5.5, 5.5, color, WHITE, .7)
    label(s, x + 27, y + 75, "objects", 8, GRAY, False, italic=True, align="left")
    path(s, [(x+45, y+61), (x+79, y+61), (x+104, y+56)], GRAY, .7, False)
    # incidence matrix on the upper right
    matrix_x = x + max(500, w * .63)
    label(s, matrix_x, y + 40, "Incidence matrix", 9, INK, True, align="left")
    incidence_matrix(s, matrix_x, y + 49, rows=4, cols=7, cell=10, gap=2)
    label(s, matrix_x + 5, y + 107, "4 typed relations", 8, GRAY, False, italic=True, align="left")
    label(s, x + w - 235, y + 51, "N objects", 8, INK, False, align="left")
    label(s, x + w - 235, y + 69, "E hyperedges", 8, INK, False, align="left")
    label(s, x + w - 235, y + 87, "L history frames", 8, INK, False, align="left")

    # lower V2E -> E2V pathway
    tensor_stack(s, x + 22, y + 137, 42, 42,
                 ["#C8C8C8", "#D4D4D4", "#BEBEBE"], offset=4, outline="#777777")
    label(s, x + 43, y + 188, "vertex\nfeatures", 8, INK, False)
    path(s, [(x+70, y+158), (x+118, y+158)], BLUE, 1.1)
    # V2E gathering trapezoid
    poly(s, [(x+120, y+135), (x+255, y+145), (x+255, y+180), (x+120, y+190)],
         WHITE, BLUE, 1.0)
    label(s, x + 188, y + 158, "V2E Gathering", 9, BLUE, True)
    label(s, x + 188, y + 179, "uniform mean", 7, YELLOW, True, italic=True)
    path(s, [(x+256, y+158), (x+310, y+158)], BLUE, 1.1)
    tensor_stack(s, x + 312, y + 137, 54, 42,
                 [RED, BLUE, ORANGE, GREEN], offset=5, outline="#8794A5")
    label(s, x + 340, y + 188, "edge\nfeatures", 8, INK, False)
    path(s, [(x+370, y+158), (x+485, y+158)], ORANGE, 1.1)
    # E2V scattering trapezoid
    poly(s, [(x+485, y+145), (x+625, y+135), (x+625, y+190), (x+485, y+180)],
         WHITE, ORANGE, 1.0)
    label(s, x + 555, y + 158, "E2V Scattering", 9, ORANGE, True)
    label(s, x + 555, y + 179, "leave-one-out", 7, ORANGE, False, italic=True)
    path(s, [(x+626, y+158), (x+770, y+158)], ORANGE, 1.1)
    tensor_stack(s, x + 772, y + 137, 45, 42,
                 ["#F3B3B3", "#B9CBEA", "#B8D7A9", "#F2D77A"], offset=4, outline="#8794A5")
    label(s, x + 795, y + 188, "vertex\ncontext", 8, INK, False)
    return (x + w, y + h / 2)


def operator(s, x, y, symbol, fill=WHITE, edge=BLUE, r=15, size=15):
    ellipse(s, x, y, r, r, fill, edge, 1.2)
    label(s, x, y, symbol, size, edge, True)


def overall_scene():
    s = scn("Overall", 1600, 900)
    title(s, 1, "Overall multimodal MOT framework", "YOLOX-HyLA detection -> two-stage IoU association -> SSM-mod-mean prediction")
    rect(s, 35, 85, 1530, 315, BLUE_LIGHT, BLUE, 1.2, 22)
    rect(s, 35, 430, 1530, 395, GREEN_LIGHT, GREEN, 1.2, 22)
    label(s, 62, 110, "DETECTION", 14, BLUE, True, align="left")
    label(s, 62, 455, "PREDICTION + ASSOCIATION", 14, GREEN, True, align="left")
    frame_pair(s, 90, 158)
    rect(s, 150, 158, 46, 37, "#EDF2FA", BLUE, 1.2, 3)
    rect(s, 155, 163, 36, 27, "#DCE5F5", BLUE, .7, 1)
    label(s, 113, 206, "RGB", 9, RED, True); label(s, 173, 206, "Event", 9, BLUE, True)
    path(s, [(136, 176), (220, 176), (220, 179), (258, 179)], BLUE, 1.5)
    path(s, [(196, 176), (232, 176), (232, 283), (258, 283)], BLUE, 1.5)
    rect(s, 258, 150, 190, 58, WHITE, BLUE, 1.4, 10); label(s, 353, 179, "RGB CSPDarknet", 13, BLUE, True)
    rect(s, 258, 254, 190, 58, WHITE, BLUE, 1.4, 10); label(s, 353, 283, "Event CSPDarknet", 13, BLUE, True)
    sheet(s, 466, 160, [BLUE_MID, BLUE, PURPLE, BLUE_MID], 18, 52); sheet(s, 466, 266, [ORANGE_LIGHT, ORANGE, RED, ORANGE_LIGHT], 18, 52)
    path(s, [(448, 179), (466, 179)], BLUE, 1.5); path(s, [(448, 283), (466, 283)], BLUE, 1.5)
    ellipse(s, 570, 230, 76, 75, PURPLE_LIGHT, PURPLE, 1.7)
    label(s, 570, 205, "HyLA", 19, PURPLE, True); label(s, 570, 230, "shared MLP +", 11); label(s, 570, 250, "Bi-cross attention", 11); label(s, 570, 330, "dark3 / dark4 / dark5", 10, PURPLE, False, True)
    path(s, [(646, 230), (746, 230)], PURPLE, 1.6)
    rect(s, 746, 194, 130, 72, WHITE, BLUE, 1.5, 10); label(s, 811, 230, "PA-FPN", 16, BLUE, True); path(s, [(876, 230), (948, 230)], BLUE, 1.6)
    rect(s, 948, 194, 125, 72, WHITE, BLUE, 1.5, 10); label(s, 1010, 230, "YOLOX head", 15, BLUE, True); path(s, [(1073, 230), (1140, 230)], BLUE, 1.6)
    sheet(s, 1140, 202, ["#F6C94A", "#9BC77A", "#9CB4D7", "#F2AE78"], 24, 56, 3); label(s, 1190, 185, "D_t", 14, INK, True, True)
    label(s, 100, 498, "track history", 11, GRAY, True)
    for i, color in enumerate([BLUE_MID, GREEN, BLUE, YELLOW]):
        rect(s, 95+i*11, 525-i*6, 94, 40, color, WHITE, .8, 3); label(s, 142+i*11, 545-i*6, f"t-{4-i}", 10, BLUE if color == BLUE_MID else WHITE, True)
    path(s, [(190, 545), (274, 545)], GREEN, 1.6); hypergraph(s, 338, 548); label(s, 338, 612, "H_r: spatial / RGB / Event / motion", 10, GREEN, True); path(s, [(402, 548), (455, 548)], GREEN, 1.6)
    operator(s, 475, 548, "mean", YELLOW_LIGHT, YELLOW, 21, 10); path(s, [(496, 548), (540, 548)], YELLOW, 1.6); sheet(s, 540, 520, [RED, BLUE, ORANGE, GREEN], 16, 55, 2); label(s, 574, 595, "relation contexts", 10, GRAY, False, True)
    operator(s, 640, 548, "softmax", ORANGE_LIGHT, ORANGE, 27, 9); path(s, [(668, 548), (724, 548)], ORANGE, 1.6)
    rect(s, 724, 505, 180, 86, WHITE, PURPLE, 1.6, 14); label(s, 814, 535, "u (64-D prior)", 16, PURPLE, True); label(s, 814, 560, "broadcast across q", 10, GRAY, False, True); path(s, [(904, 548), (961, 548)], PURPLE, 1.6)
    rect(s, 961, 486, 250, 124, WHITE, BLUE, 1.6, 16); label(s, 1086, 516, "SSM-mod-mean", 17, BLUE, True); label(s, 1086, 544, "Delta / B / C", 12); label(s, 1086, 566, "forward + reverse-input", 11); path(s, [(1210, 548), (1270, 548)], BLUE, 1.6)
    sheet(s, 1270, 518, [BLUE_MID, GREEN, YELLOW, ORANGE], 21, 60, 2); label(s, 1320, 595, "predicted boxes", 10, GRAY, False, True); operator(s, 1410, 548, "2-stage", ORANGE_LIGHT, ORANGE, 30, 10); label(s, 1410, 592, "IoU + Hungarian", 10, ORANGE, True)
    path(s, [(1370, 548), (1440, 548)], ORANGE, 1.5); path(s, [(1190, 230), (1460, 230), (1460, 510), (1410, 510)], ORANGE, 1.1); label(s, 1490, 503, "D_t", 10, ORANGE, True)
    path(s, [(1410, 578), (1410, 686), (120, 686), (120, 565)], PURPLE, 1.2, True, True); label(s, 755, 705, "updated track memory for next frame", 10, PURPLE, False, True); label(s, 1460, 640, "stage 1: tracked <-> detections", 10, ORANGE, False, True); label(s, 1460, 660, "stage 2: unmatched + lost", 10, ORANGE, False, True)
    return s


def ssm_scene():
    s = scn("SSM-mod-mean", 1600, 900)
    title(s, 2, "SSM-mod-mean: hypergraph-conditioned motion prediction", "Latest track state -> typed H_r -> mean context -> Delta/B/C modulation -> BiMamba")
    label(s, 70, 103, "latest history", 13, BLUE, True, align="left")
    for i, color in enumerate([BLUE_MID, GREEN, BLUE, YELLOW]):
        rect(s, 78+i*13, 135-i*8, 112, 46, color, WHITE, .8, 3); label(s, 134+i*13, 158-i*8, f"t-{4-i}", 11, BLUE if color == BLUE_MID else WHITE, True)
    label(s, 136, 226, "x_t in R^(q x 4)", 11, GRAY, False, True)
    path(s, [(205, 160), (220, 160)], BLUE, 1.5)
    # Reference-like hypergraph construction and typed V2E/E2V aggregation.
    panel_right, panel_mid = hypergraph_construction(s, 220, 88, 1060, 220)
    label(s, 245, 100, "H_r", 12, BLUE, True, align="left")
    path(s, [(panel_right, panel_mid), (1300, panel_mid)], GREEN, 1.5)
    rect(s, 1300, 150, 220, 90, WHITE, PURPLE, 1.5, 12)
    label(s, 1410, 178, "u_i", 21, PURPLE, True)
    label(s, 1410, 205, "64-D prior", 11, GRAY, False, True)
    label(s, 1410, 224, "relation-softmax context", 9, PURPLE, False, True)
    rect(s, 60, 340, 1480, 440, GRAY_LIGHT, BLUE, 1.2, 18); label(s, 85, 368, "SSM modulation and bidirectional scan", 14, BLUE, True, align="left")
    rect(s, 100, 430, 170, 66, WHITE, BLUE, 1.3, 10); label(s, 185, 463, "x_t", 19, BLUE, True); label(s, 185, 486, "history sequence", 10, GRAY, False, True); path(s, [(270, 463), (335, 463)], BLUE, 1.6)
    rect(s, 335, 420, 205, 86, WHITE, BLUE, 1.3, 10); label(s, 438, 448, "Mamba pre-scan", 14, BLUE, True); label(s, 438, 475, "Conv1D -> x,z,dt,B,C", 10, GRAY, False, True); path(s, [(540, 463), (600, 463)], BLUE, 1.6)
    path(s, [(1410, 240), (1410, 310), (705, 310), (705, 420)], PURPLE, 1.2, True, True); label(s, 1030, 303, "u_i broadcast across q", 10, PURPLE, False, True)
    rect(s, 600, 390, 290, 155, PURPLE_LIGHT, PURPLE, 1.5, 13); label(s, 745, 414, "[x_t ; u_i]", 15, PURPLE, True)
    for yy, name, formula in [(450, "Delta'", "dt + a_dt MLP_dt"), (490, "B'", "B * (1 + a_b tanh)"), (530, "C'", "C * (1 + a_c tanh)")]:
        rect(s, 625, yy, 110, 27, WHITE, PURPLE, .9, 5); label(s, 680, yy+14, name, 10, PURPLE, True); path(s, [(735, yy+14), (760, yy+14)], PURPLE, 1.0); label(s, 822, yy+14, formula, 9, INK, False, True)
    path(s, [(890, 463), (950, 463)], PURPLE, 1.6); rect(s, 950, 390, 300, 155, WHITE, BLUE, 1.5, 15); label(s, 1100, 412, "BiMamba block x 3", 15, BLUE, True)
    rect(s, 1000, 450, 115, 32, BLUE_LIGHT, BLUE, 1, 6); label(s, 1057, 466, "Forward scan", 10, BLUE, True); rect(s, 1000, 495, 115, 32, GREEN_LIGHT, GREEN, 1, 6); label(s, 1057, 511, "Reverse-input", 10, GREEN, True); path(s, [(1115, 466), (1180, 466), (1200, 447)], BLUE, 1.3); path(s, [(1115, 511), (1180, 511), (1200, 531)], GREEN, 1.3); operator(s, 1210, 489, "+", WHITE, INK, 17, 16); path(s, [(1227, 489), (1285, 489)], BLUE, 1.5)
    rect(s, 1285, 450, 150, 78, WHITE, BLUE, 1.3, 10); label(s, 1360, 475, "residual MLP", 13, BLUE, True); label(s, 1360, 500, "LN + projection", 10, GRAY, False, True); path(s, [(1435, 489), (1480, 489)], BLUE, 1.5); sheet(s, 1480, 450, [BLUE, GREEN, YELLOW, ORANGE], 18, 78, 2); label(s, 1520, 558, "bbox(t+1)", 10, GRAY, True, True)
    label(s, 1040, 620, "dt' = dt + a_dt MLP([x;u])    B' = B(1+a_b tanh MLP)    C' = C(1+a_c tanh MLP)", 10, INK, False, True); path(s, [(1520, 540), (1520, 700), (140, 700), (140, 500)], PURPLE, 1.2, True, True); label(s, 760, 730, "predicted box updates the next frame's history", 10, PURPLE, False, True)
    return s


def hyla_scene():
    s = scn("HyLA", 1600, 900)
    title(s, 3, "HyLA-MLP fusion", "Shared alignment -> joint-query bidirectional HyLA -> residual gate synthesis")
    rect(s, 45, 92, 1510, 655, WHITE, BLUE, 1.3, 28, True); label(s, 72, 120, "HyLA at one dark scale", 13, BLUE, True, align="left")
    rect(s, 90, 182, 85, 45, BLUE_LIGHT, BLUE, 1.2, 8); label(s, 132, 205, "R", 17, BLUE, True); rect(s, 90, 338, 85, 45, ORANGE_LIGHT, ORANGE, 1.2, 8); label(s, 132, 361, "E", 17, ORANGE, True); path(s, [(175, 205), (235, 205)], BLUE, 1.4); path(s, [(175, 361), (235, 361)], ORANGE, 1.4)
    rect(s, 235, 180, 120, 50, PURPLE_LIGHT, PURPLE, 1.2, 8); label(s, 295, 205, "Shared MLP", 13, PURPLE, True); rect(s, 235, 336, 120, 50, PURPLE_LIGHT, PURPLE, 1.2, 8); label(s, 295, 361, "Shared MLP", 13, PURPLE, True); path(s, [(295, 230), (295, 336)], PURPLE, 1.0, False, True); label(s, 310, 283, "share", 10, PURPLE, False, True, "left")
    path(s, [(355, 205), (415, 272)], PURPLE, 1.4); path(s, [(355, 361), (415, 290)], PURPLE, 1.4); operator(s, 435, 281, "C", RED_LIGHT, RED, 19, 14); label(s, 435, 317, "J = [R~ ; E~]", 10, RED, False, True)
    path(s, [(454, 274), (505, 215)], BLUE, 1.4); path(s, [(454, 288), (505, 371)], ORANGE, 1.4); rect(s, 505, 190, 180, 50, BLUE_LIGHT, BLUE, 1.2, 8); label(s, 595, 215, "H(J, E~)  R<-E", 12, BLUE, True); rect(s, 505, 346, 180, 50, ORANGE_LIGHT, ORANGE, 1.2, 8); label(s, 595, 371, "H(J, R~)  E<-R", 12, ORANGE, True); label(s, 595, 257, "QK^T / RMS-head -> ReLU -> Sigma", 9, GRAY, False, True); label(s, 595, 413, "QK^T / RMS-head -> ReLU -> Sigma", 9, GRAY, False, True)
    operator(s, 730, 215, "a_R", YELLOW_LIGHT, YELLOW, 18, 10); operator(s, 730, 371, "a_E", YELLOW_LIGHT, YELLOW, 18, 10); path(s, [(685, 215), (712, 215)], BLUE, 1.3); path(s, [(685, 371), (712, 371)], ORANGE, 1.3); operator(s, 790, 215, "+", WHITE, INK, 16, 15); operator(s, 790, 371, "+", WHITE, INK, 16, 15)
    path(s, [(355, 205), (390, 205), (390, 140), (790, 140), (790, 199)], BLUE, 1.1); path(s, [(355, 361), (375, 361), (375, 430), (790, 430), (790, 387)], ORANGE, 1.1); rect(s, 835, 190, 100, 50, BLUE_LIGHT, BLUE, 1.2, 8); label(s, 885, 215, "R'", 18, BLUE, True); rect(s, 835, 346, 100, 50, ORANGE_LIGHT, ORANGE, 1.2, 8); label(s, 885, 371, "E'", 18, ORANGE, True); path(s, [(806, 215), (835, 215)], BLUE, 1.3); path(s, [(806, 371), (835, 371)], ORANGE, 1.3)
    rect(s, 995, 250, 180, 80, PURPLE_LIGHT, PURPLE, 1.3, 12); label(s, 1085, 276, "g = sigmoid", 13, PURPLE, True); label(s, 1085, 302, "Wg [R' ; E']", 11, INK); path(s, [(935, 215), (965, 215), (965, 270), (995, 270)], BLUE, 1.2); path(s, [(935, 371), (965, 371), (965, 310), (995, 310)], ORANGE, 1.2)
    operator(s, 1240, 215, "x", YELLOW_LIGHT, YELLOW, 16, 15); operator(s, 1240, 371, "x", YELLOW_LIGHT, YELLOW, 16, 15); label(s, 1240, 178, "g * R'", 10, BLUE, True); label(s, 1240, 408, "(1-g) * E'", 10, ORANGE, True); path(s, [(1175, 270), (1197, 215)], PURPLE, 1.2); path(s, [(1175, 310), (1197, 371)], PURPLE, 1.2); path(s, [(935, 215), (1100, 215), (1197, 215)], BLUE, 1.0); path(s, [(935, 371), (1100, 371), (1197, 371)], ORANGE, 1.0)
    operator(s, 1320, 293, "+", WHITE, INK, 18, 16); path(s, [(1256, 215), (1295, 280)], YELLOW, 1.2); path(s, [(1256, 371), (1295, 306)], YELLOW, 1.2); rect(s, 1360, 265, 125, 56, WHITE, BLUE, 1.3, 9); label(s, 1422, 293, "+ W_o", 15, BLUE, True); path(s, [(1338, 293), (1360, 293)], BLUE, 1.4); path(s, [(935, 215), (1050, 215), (1050, 160), (1320, 160), (1320, 275)], BLUE, 1.0); path(s, [(935, 371), (1080, 371), (1080, 430), (1320, 430), (1320, 311)], ORANGE, 1.0); rect(s, 1510, 265, 55, 56, GREEN_LIGHT, GREEN, 1.2, 8); label(s, 1538, 293, "F", 18, GREEN, True); path(s, [(1485, 293), (1510, 293)], GREEN, 1.4)
    rect(s, 170, 790, 1260, 70, GRAY_LIGHT, BLUE, 1.0, 13, True); label(s, 800, 815, "HyLA operator", 12, BLUE, True); label(s, 800, 840, "QK^T / sqrt(d)  ->  RMS-head  ->  Sigma_h(V)  ->  ReLU  ->  Sigma_k  ->  1x1 projection", 11, INK, False, True); label(s, 1350, 710, "repeat at dark3 / dark4 / dark5    dim 64 / 96 / 128    window 7 / 7 / global", 10, GRAY, False, True, "right")
    return s


def main():
    scenes = [overall_scene(), ssm_scene(), hyla_scene()]
    files = ["01_overall_two_stage_tracking", "02_ssm_mod_mean_detail", "03_hyla_mlp_detail"]
    pages = []
    for scene, stem in zip(scenes, files):
        render(scene, 2).save(OUT / f"{stem}.png", "PNG", optimize=True, dpi=(220, 220))
        page = {"name": scene["name"], "width": scene["width"], "height": scene["height"], "items": scene["items"]}
        pages.append(page)
        print(f"{stem}: {export_vsdx([page], OUT / f'{stem}.vsdx')}")
    print(f"combined: {export_vsdx(pages, OUT / 'hyla_ssm_revised_framework.vsdx')}")


if __name__ == "__main__":
    main()
