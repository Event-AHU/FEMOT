#!/usr/bin/env python3
"""Draw a compact comparison of motion- and appearance/spatial hypergraphs."""
from __future__ import annotations

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "06_hypergraph_modeling_comparison.png"

W, H, SCALE = 1700, 960, 2
INK = "#252630"
GRAY = "#737782"
LIGHT = "#F5F6F8"
RED = "#B94B55"
RED_LIGHT = "#FBECEE"
BLUE = "#3C68B8"
BLUE_LIGHT = "#EAF0FB"
GREEN = "#4D8C55"
GREEN_LIGHT = "#EAF4EB"
PURPLE = "#7651B6"
PURPLE_LIGHT = "#F1EBF9"
WHITE = "#FFFFFF"


def main():
    image = Image.new("RGBA", (W*SCALE, H*SCALE), WHITE)
    draw = ImageDraw.Draw(image, "RGBA")
    regular = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", 30)
    italic = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Italic.ttf", 27)
    bold = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Bold.ttf", 38)
    panel_head = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Bold.ttf", 34)
    small = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", 25)
    tiny = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSerif-Regular.ttf", 22)

    def xy(point):
        return tuple(round(v*SCALE) for v in point)

    def text(x, y, value, font=regular, color=INK, anchor="mm"):
        draw.text(xy((x, y)), value, font=font, fill=color, anchor=anchor)

    def rounded(rect, fill=WHITE, outline=INK, width=1, radius=8):
        draw.rounded_rectangle(tuple(round(v*SCALE) for v in rect),
                               radius=round(radius*SCALE), fill=fill,
                               outline=outline, width=max(1, round(width*SCALE)))

    def line(points, color=INK, width=1.5):
        draw.line([xy(p) for p in points], fill=color,
                  width=max(1, round(width*SCALE)), joint="curve")

    def arrow(start, end, color=INK, width=1.7, head=10):
        x0, y0 = start
        x1, y1 = end
        line([start, end], color, width)
        angle = math.atan2(y1-y0, x1-x0)
        left = (x1-head*math.cos(angle-0.48), y1-head*math.sin(angle-0.48))
        right = (x1-head*math.cos(angle+0.48), y1-head*math.sin(angle+0.48))
        draw.polygon([xy(end), xy(left), xy(right)], fill=color)

    def ellipse(rect, outline, fill=None, width=2):
        draw.ellipse(tuple(round(v*SCALE) for v in rect), fill=fill,
                     outline=outline, width=max(1, round(width*SCALE)))

    def hyperedge(rect, color, light):
        rgba = tuple(bytes.fromhex(light.lstrip("#"))) + (72,)
        draw.ellipse(tuple(round(v*SCALE) for v in rect), fill=rgba,
                     outline=color, width=round(3*SCALE))

    def person(cx, cy, shirt, ident, text_color=INK):
        # Compact pictogram: head, torso, arms and legs remain legible at print size.
        r = 11
        ellipse((cx-r, cy-47-r, cx+r, cy-47+r), INK, WHITE, 2)
        torso = (cx-14, cy-31, cx+14, cy+3)
        draw.rounded_rectangle(tuple(round(v*SCALE) for v in torso),
                               radius=round(5*SCALE), fill=shirt,
                               outline=INK, width=round(1.7*SCALE))
        line([(cx-10, cy-25), (cx-22, cy-5)], INK, 2)
        line([(cx+10, cy-25), (cx+22, cy-5)], INK, 2)
        line([(cx-6, cy+3), (cx-11, cy+25)], INK, 2.4)
        line([(cx+6, cy+3), (cx+11, cy+25)], INK, 2.4)
        text(cx, cy+48, ident, tiny, text_color)

    # Header and two clean comparison columns.
    text(W/2, 49, "Hypergraph Modeling for Multi-Object Tracking", bold, INK)
    text(W/2, 98, "Motion-sequence reasoning versus spatial and appearance relations",
         italic, GRAY)
    rounded((40, 145, 825, 885), WHITE, "#D5D6DB", 1, 8)
    rounded((875, 145, 1660, 885), WHITE, "#D5D6DB", 1, 8)
    line([(850, 175), (850, 850)], "#D5D6DB", 1.1)

    text(452, 190, "(a) HyperSSM", panel_head, GRAY)
    text(452, 232, "Motion-vector input to HyperConv", small, GRAY)
    text(1268, 190, "(b) Our method", panel_head, PURPLE)
    text(1268, 232, "Appearance + spatial relations", small, PURPLE)

    # HyperSSM: target trajectories are represented by their motion vectors.
    left_people = [(190, 433, RED, "T1", (54, -54)),
                   (365, 365, BLUE, "T2", (58, -48)),
                   (530, 452, GREEN, "T3", (55, -50)),
                   (705, 360, "#9A7BBE", "T4", (8, 65))]
    hyperedge((107, 304, 785, 550), BLUE, BLUE_LIGHT)
    for cx, cy, shirt, ident, vec in left_people:
        person(cx, cy, shirt, ident)
        start = (cx+24, cy-8)
        end = (start[0]+vec[0], start[1]+vec[1])
        arrow(start, end, BLUE if ident != "T4" else GRAY, 2.0, 11)
    # Proposed method: overlapping relation-specific edges share target vertices.
    # Targets 1 and 2 are nearby; targets 1, 3 and 4 share appearance.
    hyperedge((1010, 348, 1265, 575), RED, RED_LIGHT)
    hyperedge((1000, 300, 1580, 480), BLUE, BLUE_LIGHT)
    right_people = [(1085, 395, RED, "T1"),
                    (1175, 500, GREEN, "T2"),
                    (1350, 380, RED, "T3"),
                    (1490, 450, RED, "T4")]
    for cx, cy, shirt, ident in right_people:
        person(cx, cy, shirt, ident)
    text(452, 600, "Target trajectories  →  motion vectors  →  HyperConv",
         small, INK)
    text(1268, 600, "One target can belong to multiple relation-specific hyperedges.",
         small, INK)
    line([(145, 637), (760, 637)], "#E1E2E6", 1)
    line([(930, 637), (1605, 637)], "#E1E2E6", 1)

    # Small legend makes the node features explicit without adding a pipeline.
    ellipse((1000, 660, 1032, 692), RED, RED_LIGHT, 2)
    text(1160, 676, "Spatial proximity", tiny, RED, anchor="lm")
    ellipse((1328, 660, 1360, 692), BLUE, BLUE_LIGHT, 2)
    text(1380, 676, "RGB appearance", tiny, BLUE, anchor="lm")
    text(1268, 737, "Vertices are tracked targets; colored loops are hyperedges.",
         tiny, GRAY)

    # The left-side vector legend separates the feature cue from the relation.
    text(452, 672, "Motion vectors provide features for hypergraph state modeling.",
         tiny, GRAY)
    arrow((250, 716), (330, 716), BLUE, 2.0, 11)
    text(368, 716, "motion vector", tiny, GRAY, anchor="lm")
    text(452, 794, "Model relations from how targets move", tiny, BLUE)
    text(1268, 794, "Model relations from who looks alike and who is nearby",
         tiny, PURPLE)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").resize((W*SCALE, H*SCALE), Image.Resampling.LANCZOS).save(
        OUT, "PNG", optimize=True, dpi=(220, 220))
    print(OUT)


if __name__ == "__main__":
    main()
