#!/usr/bin/env python3
"""Draw a polished two-stage HyLA + SSM-mod-mean MOT overview.

The scene is intentionally compact: detection is a clean multimodal YOLOX
chain with a highlighted HyLA fusion module; prediction/association reuses
the sparse purple/blue/green language of the SSM module diagram.
"""
from __future__ import annotations

from pathlib import Path

try:
    from draw_hyla_ssm_three_figures import (
        BLUE, BLUE_LIGHT, BLUE_MID, GRAY, GRAY_LIGHT, GREEN, GREEN_LIGHT,
        INK, ORANGE, ORANGE_LIGHT, PURPLE, PURPLE_LIGHT, RED, RED_LIGHT,
        WHITE, add, ellipse, export_vsdx, label, operator, path, rect, render,
        scn,
    )
except ImportError:
    from tools.draw_hyla_ssm_three_figures import (
        BLUE, BLUE_LIGHT, BLUE_MID, GRAY, GRAY_LIGHT, GREEN, GREEN_LIGHT,
        INK, ORANGE, ORANGE_LIGHT, PURPLE, PURPLE_LIGHT, RED, RED_LIGHT,
        WHITE, add, ellipse, export_vsdx, label, operator, path, rect, render,
        scn,
    )

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
OUT.mkdir(parents=True, exist_ok=True)


def bars(s, x, y, colors, w=18, h=54, gap=4, stroke=GRAY):
    for i, color in enumerate(colors):
        rect(s, x + i * (w + gap), y, w, h, color, stroke, .7, 2)


def frame_icon(s, x, y, w=86, h=62, color=BLUE):
    rect(s, x, y, w, h, "#F3F7FC", color, 1.0, 4)
    rect(s, x + 7, y + 7, w - 14, h - 14, "#DFEAF7", "#B9C9DB", .7, 2)
    for dx, dy, c in [(16, 16, color), (39, 31, RED), (60, 13, PURPLE)]:
        rect(s, x + dx, y + dy, 12, 16, WHITE, c, 1.0, 1)


def trail(s, x, y, color, n=6, gap=27):
    for i in range(n):
        xx = x + i * gap
        ellipse(s, xx, y, 7, 7, WHITE, color, 1.4)
        if i:
            path(s, [(xx - gap + 8, y), (xx - 8, y)], color, 1.0, False)


def hypergraph_prior(s, x, y):
    """Compact two-edge spatial/RGB prior sketch."""
    label(s, x + 145, y + 4, "Hypergraph prior", 16, GREEN, True)
    ellipse(s, x + 39, y + 64, 43, 22, "#FDEBEC", RED, 1.3)
    ellipse(s, x + 100, y + 73, 48, 24, BLUE_LIGHT, BLUE, 1.3)
    for dx, dy in [(10, 61), (43, 51), (68, 77), (91, 57), (119, 83), (145, 68)]:
        ellipse(s, x + dx, y + dy, 5, 5, WHITE, GRAY, .9)
    ellipse(s, x + 8, y + 11, 3.5, 3.5, RED, RED, .4)
    label(s, x + 19, y + 11, "Spatial", 10, INK, align="left")
    ellipse(s, x + 84, y + 11, 3.5, 3.5, BLUE, BLUE, .4)
    label(s, x + 95, y + 11, "RGB", 10, INK, align="left")
    label(s, x + 83, y + 115, "V2E mean → E2V aggregation", 10, INK)


def detection_stage(s):
    rect(s, 45, 92, 1910, 330, BLUE_LIGHT, BLUE, 1.2, 12)
    label(s, 70, 118, "STAGE 1  |  MULTIMODAL DETECTION", 14, BLUE, True, align="left")
    label(s, 70, 142, "RGB / Event features are aligned and fused before YOLOX prediction", 11, GRAY, italic=True, align="left")

    frame_icon(s, 85, 193, color=BLUE)
    label(s, 128, 275, "RGB frame", 11, BLUE, True)
    frame_icon(s, 85, 306, color=ORANGE)
    label(s, 128, 388, "Event frame", 11, ORANGE, True)

    rect(s, 220, 183, 180, 74, WHITE, BLUE, 1.3, 7)
    label(s, 310, 211, "CSPDarknet", 15, BLUE, True)
    label(s, 310, 237, "RGB backbone", 10, GRAY, italic=True)
    rect(s, 220, 296, 180, 74, WHITE, ORANGE, 1.3, 7)
    label(s, 310, 324, "CSPDarknet", 15, ORANGE, True)
    label(s, 310, 350, "Event backbone", 10, GRAY, italic=True)
    path(s, [(171, 224), (220, 220)], BLUE, 1.4)
    path(s, [(171, 337), (220, 333)], ORANGE, 1.4)

    # The backbone emits three explicit pyramid levels.  They are kept as
    # compact feature strips so the overview remains readable while making
    # it unambiguous that HyLA is applied at dark3, dark4 and dark5.
    bars(s, 425, 199, [BLUE_MID, BLUE, PURPLE, BLUE_MID], 17, 54, 3)
    bars(s, 425, 312, [ORANGE_LIGHT, ORANGE, RED, ORANGE_LIGHT], 17, 54, 3)
    label(s, 471, 181, "multi-scale", 10, GRAY, italic=True)
    # One compact legend per modality avoids tiny labels colliding under the
    # four-bar feature strips while still naming all three pyramid levels.
    label(s, 470, 269, "dark3   dark4   dark5", 8, BLUE, True)
    label(s, 470, 382, "dark3   dark4   dark5", 8, ORANGE, True)
    path(s, [(400, 220), (425, 220)], BLUE, 1.3)
    path(s, [(400, 333), (425, 333)], ORANGE, 1.3)

    # A single HyLA panel contains three scale-wise fusion lanes.  Each lane
    # receives RGB/Event features and runs the same Shared MLP -> Bi-HyLA
    # block, which mirrors the module detail figure without overloading the
    # overview with repeated large containers.
    rect(s, 540, 145, 455, 265, WHITE, GREEN, 1.7, 11)
    label(s, 767, 165, "HyLA-enhanced fusion", 17, GREEN, True)
    label(s, 767, 184, "dark3 / dark4 / dark5 scale-wise gated fusion", 10, GRAY, italic=True)
    lane_y = [201, 255, 309]
    scales = ["dark3", "dark4", "dark5"]
    for yy, scale_name in zip(lane_y, scales):
        label(s, 556, yy + 18, scale_name, 9, GREEN, True, align="left")
        rect(s, 606, yy + 4, 23, 27, BLUE_LIGHT, BLUE, .8, 3)
        label(s, 617.5, yy + 18, "R", 9, BLUE, True)
        rect(s, 634, yy + 4, 23, 27, ORANGE_LIGHT, ORANGE, .8, 3)
        label(s, 645.5, yy + 18, "E", 9, ORANGE, True)
        path(s, [(657, yy + 18), (673, yy + 18)], GRAY, 1.0)
        rect(s, 673, yy, 108, 35, PURPLE_LIGHT, PURPLE, 1.0, 5)
        label(s, 727, yy + 18, "Shared MLP", 10, PURPLE, True)
        path(s, [(781, yy + 18), (795, yy + 18)], PURPLE, 1.1)
        rect(s, 795, yy, 124, 35, GREEN_LIGHT, GREEN, 1.0, 5)
        label(s, 857, yy + 18, "Bi-HyLA", 11, GREEN, True)
        path(s, [(919, yy + 18), (978, yy + 18)], GREEN, 1.0)
    label(s, 945, 388, "fused pyramid", 9, GREEN, italic=True, align="right")
    path(s, [(511, 226), (540, 226)], BLUE, 1.2)
    path(s, [(511, 339), (540, 339)], ORANGE, 1.2)

    rect(s, 1040, 206, 170, 80, WHITE, BLUE, 1.3, 7)
    label(s, 1125, 234, "PA-FPN", 16, BLUE, True)
    label(s, 1125, 260, "shared pyramid", 10, GRAY, italic=True)
    path(s, [(995, 275), (1040, 246)], GREEN, 1.4)
    path(s, [(1210, 246), (1240, 246)], BLUE, 1.3)
    rect(s, 1240, 206, 170, 80, WHITE, BLUE, 1.3, 7)
    label(s, 1325, 234, "YOLOX head", 15, BLUE, True)
    label(s, 1325, 260, "D_t detections", 10, GRAY, italic=True)
    path(s, [(1410, 246), (1450, 246)], BLUE, 1.3)
    bars(s, 1452, 216, [RED_LIGHT, BLUE_LIGHT, ORANGE_LIGHT, GREEN_LIGHT], 21, 60, 3)
    label(s, 1500, 296, "current detections  D_t", 10, BLUE, italic=True)


def prediction_stage(s):
    rect(s, 45, 458, 1910, 465, PURPLE_LIGHT, PURPLE, 1.2, 12)
    label(s, 70, 484, "STAGE 2  |  MOTION PREDICTION + ASSOCIATION", 14, PURPLE, True, align="left")
    label(s, 70, 508, "Hypergraph prior modulates both forward and backward SSM scans", 11, GRAY, italic=True, align="left")

    # Left history and prior branch.
    label(s, 100, 562, "track memory", 12, BLUE, True)
    trail(s, 90, 596, BLUE, 6)
    trail(s, 90, 635, GREEN, 6)
    trail(s, 90, 674, ORANGE, 6)
    label(s, 170, 720, "tracked / lost trajectories", 10, GRAY, italic=True)
    path(s, [(252, 635), (290, 635)], BLUE, 1.4)
    rect(s, 290, 580, 205, 112, WHITE, GREEN, 1.3, 8)
    hypergraph_prior(s, 300, 586)
    path(s, [(495, 635), (535, 635)], GREEN, 1.4)
    rect(s, 535, 594, 130, 83, WHITE, PURPLE, 1.4, 8)
    label(s, 600, 621, "u_i", 22, PURPLE, True)
    label(s, 600, 647, "64-D prior", 10, GRAY, italic=True)
    label(s, 600, 664, "relation softmax", 9, PURPLE, italic=True)

    # The overview deliberately keeps the SSM abstract: one motion stream is
    # split into two scans, while a single hypergraph-prior bus modulates both.
    rect(s, 700, 537, 800, 350, "#FAF8FD", PURPLE, 1.6, 10)
    label(s, 1100, 561, "SSM-mod-mean  × 3", 18, PURPLE, True)
    rect(s, 730, 675, 145, 58, WHITE, BLUE, 1.2, 6)
    label(s, 802, 699, "motion sequence", 11, BLUE, True)
    label(s, 802, 718, "x₁ … x_q", 9, GRAY, italic=True)
    ellipse(s, 900, 704, 6, 6, BLUE, BLUE, .8)
    path(s, [(875, 704), (894, 704)], BLUE, 1.3)

    # One prior bus, with two short arrows, is the only modulation cue.
    rect(s, 1015, 580, 250, 38, PURPLE_LIGHT, PURPLE, 1.2, 7)
    label(s, 1140, 599, "hypergraph prior", 12, PURPLE, True)
    path(s, [(665, 635), (700, 635), (700, 599), (1015, 599)], PURPLE, 1.2, True)
    path(s, [(1140, 618), (1140, 630), (1040, 630), (1040, 650)], PURPLE, 1.2, True)
    path(s, [(1140, 630), (1190, 630), (1190, 760), (1150, 760)], PURPLE, 1.2, True)
    label(s, 1200, 635, "modulates both scans", 9, PURPLE, True, italic=True, align="left")

    # Two clean scan branches.
    path(s, [(906, 704), (930, 704), (930, 680)], BLUE, 1.2, True)
    path(s, [(906, 704), (912, 704), (912, 785), (930, 785)], GREEN, 1.2, True)
    rect(s, 930, 645, 220, 75, WHITE, BLUE, 1.3, 8)
    label(s, 1040, 674, "Forward scan", 15, BLUE, True)
    label(s, 1040, 697, "SSM branch", 10, GRAY, italic=True)
    rect(s, 930, 750, 220, 75, WHITE, GREEN, 1.3, 8)
    label(s, 1040, 779, "Backward scan", 15, GREEN, True)
    label(s, 1040, 802, "SSM branch", 10, GRAY, italic=True)
    path(s, [(1150, 682), (1240, 682), (1240, 720)], BLUE, 1.2)
    path(s, [(1150, 787), (1240, 787), (1240, 720)], GREEN, 1.2)
    operator(s, 1240, 720, "+", WHITE, INK, 16, 15)
    path(s, [(1256, 720), (1280, 720)], BLUE, 1.3)
    rect(s, 1280, 687, 110, 66, WHITE, BLUE, 1.1, 5)
    label(s, 1335, 712, "residual", 11, BLUE, True)
    label(s, 1335, 731, "MLP", 11, BLUE, True)
    path(s, [(1390, 720), (1420, 720)], BLUE, 1.3)
    bars(s, 1422, 691, ["#E7CFB7", "#DDB58D", "#D1955C", "#C97830"], 16, 60, 3)
    label(s, 1460, 775, "predicted Δ bbox", 11, ORANGE, True)

    # Association receives both detector boxes and predicted motion.
    path(s, [(1545, 246), (1670, 246), (1670, 644), (1695, 644)], BLUE, 1.2)
    label(s, 1640, 300, "D_t", 10, BLUE, italic=True)
    path(s, [(1495, 720), (1695, 721)], ORANGE, 1.3)
    rect(s, 1695, 625, 190, 116, WHITE, ORANGE, 1.5, 8)
    label(s, 1790, 652, "Two-stage association", 14, ORANGE, True)
    label(s, 1790, 678, "IoU cost + Hungarian", 10, GRAY, italic=True)
    label(s, 1790, 704, "1  tracked ↔ D_t", 10, INK)
    label(s, 1790, 724, "2  unmatched + lost", 10, INK)
    path(s, [(1790, 741), (1790, 795)], GREEN, 1.2, True)
    rect(s, 1695, 795, 190, 52, GREEN_LIGHT, GREEN, 1.2, 6)
    label(s, 1790, 821, "output tracks", 13, GREEN, True)


def overall_scene():
    s = scn("Overall revised", 2000, 970)
    label(s, 1000, 34, "HyLA-MLP + SSM-mod-mean", 27, PURPLE, True)
    label(s, 1000, 64, "multimodal YOLOX detection  →  hypergraph-conditioned motion prediction  →  two-stage association", 11, GRAY, italic=True)
    detection_stage(s)
    prediction_stage(s)
    return s


def main():
    scene = overall_scene()
    png = OUT / "overall_framework_revised.png"
    vsdx = OUT / "overall_framework_revised.vsdx"
    render(scene, 2).save(png, "PNG", optimize=True, dpi=(220, 220))
    print(f"png: {png}")
    print(f"vsdx: {export_vsdx([scene], vsdx)}")


if __name__ == "__main__":
    main()
