#!/usr/bin/env python3
"""Draw a compact, editable explanation of Mamba's Delta and C parameters."""
from __future__ import annotations

from pathlib import Path

from export_diagram_vsdx import export_vsdx
import draw_hyla_ssm_three_figures as renderer
from draw_hyla_ssm_three_figures import scn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#252630"
GRAY = "#737782"
LIGHT_GRAY = "#F3F4F6"
RED = "#B94B55"
RED_LIGHT = "#FBECEE"
BLUE = "#3C68B8"
BLUE_LIGHT = "#EAF0FB"
PURPLE = "#7651B6"
PURPLE_LIGHT = "#F1EBF9"
WHITE = "#FFFFFF"
FONT = "Times New Roman"


def build_scene():
    s = scn("Mamba Delta and C", 1580, 850)

    def add(kind, **kwargs):
        s["items"].append({"type": kind, **kwargs})

    def box(x, y, w, h, fill=WHITE, stroke=INK, sw=1.2, r=7):
        add("rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
            width=sw, r=r)

    def oval(cx, cy, rx, ry, fill=WHITE, stroke=INK, sw=1.1):
        add("ellipse", cx=cx, cy=cy, rx=rx, ry=ry,
            fill=fill, stroke=stroke, width=sw)

    def txt(x, y, value, size=16, color=INK, bold=False, italic=False,
            align="center", w=None, h=None):
        item = {"type": "text", "x": x, "y": y, "text": value,
                "size": size, "color": color, "font": FONT,
                "bold": bold, "italic": italic, "align": align}
        if w is not None:
            item["w"] = w
        if h is not None:
            item["h"] = h
        s["items"].append(item)

    def line(points, color=INK, sw=1.4, arrow=True):
        add("polyline", points=points, stroke=color, width=sw,
            fill=None, arrow=arrow)

    txt(790, 36, "Selective State-Space Parameters: Δ and C", 28, INK, True)
    txt(790, 74,
        "Input-dependent parameters with distinct roles in Mamba",
        16, GRAY, italic=True)

    box(50, 125, 720, 590, WHITE, "#C8C7CE", .95, 8)
    box(810, 125, 720, 590, WHITE, "#C8C7CE", .95, 8)
    txt(82, 160, "A   Δ_t sets the state-update scale", 20, RED, True,
        align="left")
    txt(842, 160, "B   C_t reads from the hidden state", 20, BLUE, True,
        align="left")

    # The effective positive step is formed after bias and softplus.
    txt(386, 213, "input-dependent step", 13, GRAY, italic=True)
    box(91, 240, 154, 66, WHITE, RED, 1.2, 6)
    txt(168, 273, "dt_t + bias", 16, INK, True)
    box(290, 240, 126, 66, RED_LIGHT, RED, 1.2, 6)
    txt(353, 273, "softplus", 16, RED, True)
    box(462, 240, 138, 66, RED_LIGHT, RED, 1.3, 6)
    txt(531, 273, "Δ_t > 0", 18, RED, True)
    line([(245, 273), (290, 273)], RED, 1.4)
    line([(416, 273), (462, 273)], RED, 1.4)

    box(405, 382, 252, 82, WHITE, RED, 1.25, 7)
    txt(531, 408, "discretize dynamics", 15, RED, True)
    txt(531, 438, "A_bar_t = exp(Δ_t A)", 15, INK)
    line([(531, 306), (531, 382)], RED, 1.4)

    box(92, 544, 132, 62, LIGHT_GRAY, GRAY, 1.0, 6)
    txt(158, 575, "h_{t-1}", 17, INK, True)
    box(280, 513, 300, 124, WHITE, RED, 1.25, 7)
    txt(430, 540, "state update", 15, RED, True)
    txt(430, 577, "h_t = A_bar_t h_{t-1}", 15, INK)
    txt(430, 604, "+ B_bar_t x_t", 15, INK)
    box(620, 544, 92, 62, RED_LIGHT, RED, 1.1, 6)
    txt(666, 575, "h_t", 17, RED, True)
    line([(224, 575), (280, 575)], GRAY, 1.35)
    line([(580, 575), (620, 575)], RED, 1.4)
    line([(531, 464), (472, 513)], RED, 1.4)

    txt(245, 672, "small Δ_t: retain more history", 14, GRAY)
    txt(574, 672, "large Δ_t: update faster", 14, RED)

    # C_t is a time-varying readout vector that weights state components.
    txt(992, 229, "hidden state  h_t", 14, GRAY, True)
    box(865, 248, 262, 94, BLUE_LIGHT, BLUE, 1.1, 6)
    for i, shade in enumerate(("#B8CBE9", "#D3DEF1", "#9DB7E0", "#CAD8EF")):
        x = 886 + i * 57
        box(x, 269, 48, 48, shade, BLUE, .8, 3)
        txt(x + 24, 293, f"h{i+1}", 13, BLUE, True)

    txt(992, 391, "readout weights  C_t", 14, GRAY, True)
    box(865, 410, 262, 94, WHITE, BLUE, 1.1, 6)
    for i, shade in enumerate(("#EAF0FB", "#C9D8F0", "#AFC4E6", "#D8E2F4")):
        x = 886 + i * 57
        box(x, 431, 48, 48, shade, BLUE, .8, 3)
        txt(x + 24, 455, f"c{i+1}", 13, BLUE, True)

    box(1190, 320, 142, 110, WHITE, BLUE, 1.2, 7)
    txt(1261, 351, "weighted sum", 14, BLUE, True)
    txt(1261, 390, "Σ C_{t,i} h_{t,i}", 14, INK)
    box(1390, 340, 105, 70, BLUE_LIGHT, BLUE, 1.15, 6)
    txt(1442, 375, "y_t", 19, BLUE, True)
    line([(1127, 295), (1190, 350)], BLUE, 1.35)
    line([(1127, 457), (1190, 402)], BLUE, 1.35)
    line([(1332, 375), (1390, 375)], BLUE, 1.4)
    txt(1180, 543, "C_t weights state components to form the output.",
        14, GRAY, italic=True, w=320)
    txt(1180, 575, "y_t = C_t^T h_t  (+ D x_t skip)", 15, INK, w=320)

    box(372, 758, 836, 54, PURPLE_LIGHT, PURPLE, 1.0, 7)
    txt(790, 785,
        "In our tracker, the hypergraph prior modulates both Δ_t and C_t.",
        16, PURPLE, True)
    return s


def write_preview(scene, path: Path):
    renderer.FONT = "/usr/share/fonts/liberation/LiberationSerif-Regular.ttf"
    renderer.FONT_B = "/usr/share/fonts/liberation/LiberationSerif-Bold.ttf"
    renderer.FONT_I = "/usr/share/fonts/liberation/LiberationSerif-Italic.ttf"
    renderer.render(scene, scale=2).save(path, "PNG", optimize=True,
                                         dpi=(220, 220))


def main():
    scene = build_scene()
    png = OUT / "05_mamba_delta_c.png"
    vsdx = OUT / "05_mamba_delta_c.vsdx"
    write_preview(scene, png)
    report = export_vsdx([scene], vsdx)
    print("png:", png)
    print("vsdx:", vsdx)
    print(report)


if __name__ == "__main__":
    main()
