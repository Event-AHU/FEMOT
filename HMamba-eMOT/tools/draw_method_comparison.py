#!/usr/bin/env python3
"""Draw an editable comparison of YOLOX fusion and hypergraph-SSM placement."""
from __future__ import annotations

import copy
from pathlib import Path

from PIL import Image

from export_diagram_vsdx import export_vsdx
import draw_hyla_ssm_three_figures as renderer
from draw_hyla_ssm_three_figures import scn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#252630"
GRAY = "#737782"
LIGHT_GRAY = "#F3F4F6"
RGB = "#B94B55"
RGB_LIGHT = "#FBECEE"
EVENT = "#3C68B8"
EVENT_LIGHT = "#EAF0FB"
PURPLE = "#7651B6"
PURPLE_LIGHT = "#F1EBF9"
GREEN = "#4D8C55"
GREEN_LIGHT = "#EAF4EB"
WHITE = "#FFFFFF"
FONT = "Times New Roman"


def build_scene():
    s = scn("Method comparison", 1100, 1380)

    def add(kind, **kwargs):
        s["items"].append({"type": kind, **kwargs})

    def box(x, y, w, h, fill=WHITE, stroke=INK, sw=1.2, r=8):
        add("rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, width=sw, r=r)

    def oval(cx, cy, rx, ry, fill=WHITE, stroke=INK, sw=1.1):
        add("ellipse", cx=cx, cy=cy, rx=rx, ry=ry,
            fill=fill, stroke=stroke, width=sw)

    def txt(x, y, value, size=16, color=INK, bold=False, italic=False,
            align="center", w=None):
        item = {"type": "text", "x": x, "y": y, "text": value,
                "size": size, "color": color, "font": FONT,
                "bold": bold, "italic": italic, "align": align}
        if w is not None:
            item["w"] = w
        s["items"].append(item)

    def line(points, color=INK, sw=1.4, arrow=True, dashed=False):
        add("polyline", points=points, stroke=color, width=sw, fill=None,
            arrow=arrow, dashed=dashed)

    def dual_inputs(y):
        box(74, y, 94, 38, RGB_LIGHT, RGB, 1.0, 5)
        txt(121, y+19, "RGB", 14, RGB, True)
        box(74, y+50, 94, 38, EVENT_LIGHT, EVENT, 1.0, 5)
        txt(121, y+69, "Event", 14, EVENT, True)

    def dual_backbones(y):
        box(225, y, 126, 38, WHITE, RGB, 1.0, 5)
        txt(288, y+19, "Backbone", 14, INK)
        box(225, y+50, 126, 38, WHITE, EVENT, 1.0, 5)
        txt(288, y+69, "Backbone", 14, INK)

    def feature_pyramids(y):
        box(405, y, 142, 38, LIGHT_GRAY, GRAY, .9, 4)
        txt(476, y+19, "RGB features", 13, GRAY)
        box(405, y+50, 142, 38, LIGHT_GRAY, GRAY, .9, 4)
        txt(476, y+69, "Event features", 13, GRAY)

    def detector_fusion_row(y, method):
        dual_inputs(y)
        dual_backbones(y)
        feature_pyramids(y)
        box(610, y+19, 126, 50,
            PURPLE_LIGHT if method == "HyLA" else WHITE,
            PURPLE if method == "HyLA" else GRAY, 1.15, 6)
        if method == "HyLA":
            txt(673, y+36, "HyLA", 15, PURPLE, True)
            txt(673, y+56, "dark3 / 4 / 5", 11, PURPLE)
        else:
            oval(673, y+44, 14, 14, WHITE, GRAY, 1.0)
            txt(673, y+44, "+", 17, GRAY, True)
        box(812, y+19, 154, 50, WHITE,
            PURPLE if method == "HyLA" else GRAY, 1.1, 6)
        txt(889, y+44, "YOLOX head", 14, INK, True)
        for sy, color in ((y+19, RGB), (y+69, EVENT)):
            line([(168, sy), (225, sy)], color, 1.1)
            line([(351, sy), (405, sy)], color, 1.1)
            line([(547, sy), (610, y+44)], color, 1.1)
        line([(736, y+44), (812, y+44)],
             PURPLE if method == "HyLA" else GRAY, 1.3)

    txt(550, 40, "Where Our Multimodal Method Differs", 26, INK, True)
    box(44, 92, 1012, 570, WHITE, "#C8C7CE", .95, 8)
    txt(78, 128, "(a)  Multimodal detection fusion", 20, INK, True,
        align="left")
    txt(79, 185, "Dual-stream baseline", 15, GRAY, True, align="left")
    detector_fusion_row(218, "sum")
    txt(80, 337,
        "Vanilla YOLOX is single-stream; feature addition is a simple dual-stream baseline.",
        13, GRAY, italic=True, align="left", w=890)
    txt(79, 397, "Ours: feature-level fusion", 15, PURPLE, True,
        align="left")
    detector_fusion_row(430, "HyLA")
    txt(80, 563,
        "HyLA combines RGB and event features at three backbone scales.",
        13, PURPLE, italic=True, align="left", w=890)

    box(44, 695, 1012, 630, WHITE, "#C8C7CE", .95, 8)
    txt(78, 731, "(b)  Hypergraph integration with state-space modeling",
        20, INK, True, align="left")
    txt(79, 788, "Input-side integration", 15, GRAY, True,
        align="left")

    box(91, 842, 174, 66, WHITE, GRAY, 1.1, 6)
    txt(178, 875, "Trajectory", 15, INK)
    box(345, 842, 184, 66, LIGHT_GRAY, GRAY, 1.15, 6)
    oval(387, 870, 25, 13, RGB_LIGHT, RGB, .8)
    oval(410, 878, 25, 13, EVENT_LIGHT, EVENT, .8)
    txt(474, 875, "HConv", 15, GRAY, True)
    box(610, 842, 140, 66, WHITE, GRAY, 1.1, 6)
    txt(680, 875, "SSM", 16, INK, True)
    box(832, 842, 150, 66, WHITE, GRAY, 1.1, 6)
    txt(907, 875, "Output", 15, INK, True)
    line([(265, 875), (345, 875)], GRAY, 1.3)
    line([(529, 875), (610, 875)], GRAY, 1.3)
    line([(750, 875), (832, 875)], GRAY, 1.3)
    txt(550, 947, "Hypergraph transforms the sequence before the SSM.",
        13, GRAY, italic=True)

    txt(79, 1022, "Ours: internal parameter modulation", 15, PURPLE, True,
        align="left")
    box(91, 1137, 174, 72, WHITE, PURPLE, 1.1, 6)
    txt(178, 1173, "Trajectory", 15, INK)
    box(359, 1083, 426, 181, WHITE, PURPLE, 1.35, 8)
    txt(386, 1112, "Mamba / SSM", 17, PURPLE, True, align="left")
    txt(572, 1147, "selective state update", 14, GRAY, italic=True)
    box(523, 1181, 82, 48, PURPLE_LIGHT, PURPLE, 1.0, 5)
    txt(564, 1205, "Δ", 17, PURPLE, True)
    box(626, 1181, 82, 48, PURPLE_LIGHT, PURPLE, 1.0, 5)
    txt(667, 1205, "C", 17, PURPLE, True)
    box(886, 1137, 150, 72, WHITE, PURPLE, 1.1, 6)
    txt(961, 1173, "Output", 15, INK, True)
    box(546, 1036, 146, 40, PURPLE_LIGHT, PURPLE, 1.0, 6)
    txt(619, 1056, "Hypergraph prior", 13, PURPLE, True)
    line([(265, 1173), (359, 1173)], PURPLE, 1.3)
    line([(785, 1173), (886, 1173)], PURPLE, 1.3)
    line([(594, 1076), (564, 1181)], PURPLE, 1.35)
    line([(644, 1076), (667, 1181)], PURPLE, 1.35)
    txt(619, 1292,
        "The prior acts inside Mamba, modulating Δ and C.",
        13, PURPLE, italic=True)
    return s


def write_preview(scene, path: Path):
    renderer.FONT = "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf"
    renderer.FONT_B = "/usr/share/fonts/liberation/LiberationSerif-Bold.ttf"
    renderer.FONT_I = "/usr/share/fonts/liberation/LiberationSerif-Italic.ttf"
    renderer.render(scene, scale=2).save(path, "PNG", optimize=True, dpi=(220, 220))


def main():
    scene = build_scene()
    png = OUT / "04_method_comparison.png"
    vsdx = OUT / "04_method_comparison.vsdx"
    write_preview(scene, png)
    report = export_vsdx([scene], vsdx)
    print("png:", png)
    print("vsdx:", vsdx)
    print(report)


if __name__ == "__main__":
    main()
