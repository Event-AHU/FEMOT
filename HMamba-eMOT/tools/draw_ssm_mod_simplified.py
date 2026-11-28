#!/usr/bin/env python3
"""Unify the supplied SSM-mod Visio diagram into one compact module.

The input VSDX unfolds HyperConv, SSM modulation, Mamba and BiMamba in
separate panels. This overview retains the motion path and the conditioning
branch without repeating those nested details. PNG and VSDX share geometry.
"""
from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from PIL import ImageDraw, ImageFont

try:
    from draw_hyla_ssm_three_figures import (
        BLUE, BLUE_LIGHT, GRAY, GREEN, INK, ORANGE, PURPLE, PURPLE_LIGHT,
        RED, WHITE, ellipse, label, operator, path, rect, render, scn,
    )
    from export_diagram_vsdx import export_vsdx, validate_vsdx
except ImportError:
    from tools.draw_hyla_ssm_three_figures import (
        BLUE, BLUE_LIGHT, GRAY, GREEN, INK, ORANGE, PURPLE, PURPLE_LIGHT,
        RED, WHITE, ellipse, label, operator, path, rect, render, scn,
    )
    from tools.export_diagram_vsdx import export_vsdx, validate_vsdx


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"


def history(s):
    """Four offset components along the history axis, drawn once."""
    label(s, 168, 519, "Motion history", 18, BLUE, True)
    colors = ["#C0D1EB", "#8AAAD6", "#5F88C4", "#3566B8"]
    for i, color in enumerate(colors):
        rect(s, 112 + 28 * i, 548, 24, 66, color, None, 0, 3)
    label(s, 166, 644, "q × 4 motion offsets", 12, GRAY)
    label(s, 166, 669, "t−L  ···  t−2  t−1", 11, GRAY, italic=True)


def math_label(s, x, y, text, size=18, color=INK, bold=False):
    """Typeset short subscripts as separately editable text runs."""
    text = text.replace("Ā", "Ā")
    runs = []
    start = 0
    for match in re.finditer(r"_(?:\{([^}]+)\}|(.))", text):
        if match.start() > start:
            runs.append((text[start:match.start()], size, 0))
        runs.append((match.group(1) or match.group(2), size * .68, size * .25))
        start = match.end()
    if start < len(text):
        runs.append((text[start:], size, 0))
    measured = []
    for value, height, offset in runs:
        plain = []
        overbars = []
        for character in value:
            if character == "\u0304":
                overbars.append(len(plain) - 1)
            else:
                plain.append(character)
        value = "".join(plain)
        face = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSerif-Italic.ttf",
                                  round(height * 4))
        bars = [(face.getlength(value[:index]) / 4,
                 face.getlength(value[:index + 1]) / 4) for index in overbars]
        measured.append((value, height, offset, face.getlength(value) / 4, bars))
    left = x - sum(run[3] for run in measured) / 2
    for value, height, offset, width, bars in measured:
        item = label(s, left + width / 2, y + offset, value, height,
                     color, bold, italic=True, w=width + 3, h=height * 1.4)
        item["font"] = "Cambria Math"
        for begin, end in bars:
            bar_y = y + offset - height * .44 - 1.5
            path(s, [(left + begin + 2, bar_y), (left + end, bar_y)],
                 color, 1.0, False)
        left += width


def state(s, x, y, text, color=BLUE, width=65, height=45):
    rect(s, x - width / 2, y - height / 2, width, height,
         WHITE, color, 1.5, 0)
    math_label(s, x, y, text, 25, color)


def hypergraph_prior(s):
    """Ablated typed hypergraph, mean aggregation and one context vector.

    The retained ablation uses only spatial and RGB hyperedges.  Event and
    motion relations are intentionally omitted from this architecture view.
    """
    label(s, 423, 166, "Hypergraph prior", 18, GREEN, True)
    for x, text, color in [(360, "Spatial", RED), (452, "RGB", BLUE)]:
        ellipse(s, x, 201, 3.5, 3.5, color, color, .5)
        label(s, x + 9, 201, text, 12, INK, align="left")

    # Overlapping closed curves are the two retained hyperedges; white
    # circles are vertices.  The other two relation families are removed by
    # the ablation and are not shown in the legend or aggregation sketch.
    ellipse(s, 393, 264, 65, 31, "#FAECEC", RED, 1.3)
    ellipse(s, 448, 278, 70, 33, "#EDF2FB", BLUE, 1.3)
    for x, y in [(349, 261), (385, 252), (407, 280), (426, 240),
                 (447, 267), (479, 283)]:
        ellipse(s, x, y, 6, 6, WHITE, "#65758A", 1.0)
    label(s, 422, 334, "V2E mean → E2V aggregation", 13, INK)

    path(s, [(519, 272), (607, 272)], GREEN, 1.4)
    label(s, 561, 247, "relation softmax", 11, GRAY, italic=True)
    # A single thin context vector replaces the repeated prior tensors.
    for i, color in enumerate(["#DCD1EF", "#B9A3DC", "#9273C5", PURPLE]):
        rect(s, 616 + 12 * i, 249, 9, 46, color, None, 0, 2)
    math_label(s, 637, 324, "u_i", 24, PURPLE)
    label(s, 637, 349, "64-D context", 12, PURPLE)
    path(s, [(665, 272), (752, 272), (752, 310), (800, 310)], PURPLE, 1.6)
    label(s, 708, 247, "broadcast", 11, PURPLE, italic=True)


def modulated_bimamba(s):
    """Show the two modulation operators alongside the recurrent SSM core."""
    rect(s, 720, 140, 1040, 704, "#FAF8FD", PURPLE, 1.25, 8)
    label(s, 1240, 172, "SSM-modulated BiMamba × 3", 21, PURPLE, True)

    # Motion splits into the illustrated forward core and a compact reverse
    # path. Input/output projections and the z gate are folded into X/Y.
    path(s, [(720, 580), (743, 580)], BLUE, 1.5, False)
    ellipse(s, 743, 580, 3, 3, BLUE, BLUE, .5)
    path(s, [(743, 580), (743, 221), (1380, 221), (1380, 284)], BLUE, 1.4)
    label(s, 1110, 207, "input projection", 11, BLUE, italic=True)
    label(s, 1500, 255, "Forward scan", 18, BLUE, True)
    math_label(s, 1350, 261, "X", 24, INK)

    rect(s, 1317, 284, 126, 54, WHITE, BLUE, 1.4, 0)
    label(s, 1380, 311, "Conv1D\n+ SiLU", 15, BLUE, True)
    path(s, [(1380, 338), (1380, 400)], BLUE, 1.5)

    # The prior is concatenated with conv features, rather than injected
    # directly into the motion embedding.
    label(s, 972, 267, "Hypergraph-conditioned parameters", 16, PURPLE, True)
    rect(s, 800, 285, 345, 50, PURPLE_LIGHT, PURPLE, 1.0, 4)
    math_label(s, 972.5, 310, "f_t = [x_t ; u_i]", 23, PURPLE)
    path(s, [(1380, 359), (1245, 359), (1245, 310), (1145, 310)], BLUE, 1.2)
    label(s, 1235, 289, "conv feature", 11, BLUE, italic=True)

    # Delta: additive residual followed by the scan's bias/softplus.
    path(s, [(812, 335), (812, 350), (775, 350), (775, 642)], PURPLE, 1.5, False)
    ellipse(s, 775, 420, 2.5, 2.5, PURPLE, PURPLE, .5)
    path(s, [(775, 420), (1054, 420)], PURPLE, 1.5)
    label(s, 914, 368, "Step-size modulation", 15, PURPLE, True)
    math_label(s, 936, 396, "α_Δ MLP_Δ(f_t)", 19, PURPLE)
    operator(s, 1070, 420, "+", WHITE, PURPLE, 16, 19)
    math_label(s, 1070, 369, "Δ_{raw} + b_Δ", 18, INK)
    path(s, [(1070, 385), (1070, 404)], INK, 1.1)
    path(s, [(1086, 420), (1150, 420)], PURPLE, 1.5)
    label(s, 1121, 399, "softplus", 11, PURPLE, italic=True)
    state(s, 1180, 420, "Δ′_t", PURPLE, 60, 42)
    path(s, [(1210, 420), (1347.5, 422)], PURPLE, 1.5, True, True)
    label(s, 1278, 397, "discretize (A, B)", 12, PURPLE, italic=True)

    # The forward state-space core follows the reference's vertical layout.
    # B is not gated; its discrete coefficient still depends on Delta.
    state(s, 1380, 422, "B̄_t")
    path(s, [(1380, 444.5), (1380, 509.5)], BLUE, 1.5)
    state(s, 1380, 532, "h_t")
    state(s, 1510, 532, "Ā_t")
    path(s, [(1477.5, 532), (1412.5, 532)], BLUE, 1.5)
    path(s, [(1380, 554.5), (1380, 574), (1570, 574),
             (1570, 475), (1510, 475), (1510, 509.5)], BLUE, 1.15)
    math_label(s, 1525, 455, "h_{t-1}", 20, BLUE)
    label(s, 1502, 591, "one-step delay", 11, GRAY, italic=True)
    math_label(s, 1518, 392, "Ā_t = exp(Δ′_t A)", 18, PURPLE)
    math_label(s, 1518, 426, "B̄_t = Δ′_t B_t", 18, PURPLE)
    # A short same-symbol reference makes the Delta/A relation explicit
    # without adding a long connector through the state-input path.
    math_label(s, 1440, 480, "Δ′_t", 19, PURPLE)
    path(s, [(1460, 482), (1492, 509.5)], PURPLE, 1.4, True, True)

    # C: a bounded residual gate multiplying the original readout matrix.
    path(s, [(775, 642), (1054, 642)], PURPLE, 1.5)
    label(s, 930, 585, "Readout modulation", 15, PURPLE, True)
    math_label(s, 920, 614, "1 + α_C tanh(MLP_C(f_t))", 19, PURPLE)
    ellipse(s, 1070, 642, 16, 16, WHITE, PURPLE, 1.1)
    math_label(s, 1070, 642, "⊙", 22, PURPLE)
    math_label(s, 1070, 591, "C_t", 20, INK)
    path(s, [(1070, 607), (1070, 626)], INK, 1.1)
    path(s, [(1086, 642), (1347.5, 642)], PURPLE, 1.6)
    path(s, [(1380, 554.5), (1380, 619.5)], BLUE, 1.5)
    state(s, 1380, 642, "C′_t", PURPLE)
    path(s, [(1380, 664.5), (1380, 676)], BLUE, 1.4)
    operator(s, 1380, 691, "+", WHITE, BLUE, 15, 18)

    # Mamba's D skip bypasses the state recurrence, as in the reference.
    path(s, [(1380, 359), (1618, 359), (1618, 691), (1395, 691)], BLUE, 1.1)
    math_label(s, 1540, 337, "D ⊙ x_t", 18, BLUE)
    path(s, [(1380, 706), (1380, 740), (1619, 740)], BLUE, 1.5)
    math_label(s, 1351, 723, "Y", 25, INK)
    label(s, 1502, 722, "z gate · output projection", 12, BLUE, italic=True)

    path(s, [(743, 580), (743, 785), (1570, 785), (1619, 752)], GREEN, 1.5)
    label(s, 1150, 768, "Backward scan", 18, GREEN, True)
    operator(s, 1635, 740, "+", WHITE, INK, 16, 19)
    path(s, [(1651, 740), (1670, 740)], BLUE, 1.3)
    rect(s, 1670, 712, 74, 56, WHITE, BLUE, 1.0, 4)
    label(s, 1707, 740, "residual\nMLP", 12, BLUE, True)
    path(s, [(1744, 740), (1760, 740)], BLUE, 1.3, False)
    label(s, 1016, 476, "B has no direct modulation", 12, GRAY, italic=True)
    label(s, 1010, 695, "C′ controls the state readout", 12, GRAY, italic=True)
    label(s, 1240, 824, "Δ / C modulation in both scan directions", 12, GRAY, italic=True)


def scene():
    s = scn("SSM-mod unified", 2160, 920)
    label(s, 1080, 42, "SSM-mod-mean", 28, PURPLE, True)
    # Only the whole module and the scan operation need enclosing frames.
    rect(s, 52, 96, 2056, 794, WHITE, "#AEC0D7", 1.1, 8)

    history(s)
    modulated_bimamba(s)
    hypergraph_prior(s)

    # History provides latest motion for the prior and a temporal sequence
    # for Mamba. Modal relations are the second input to the prior branch.
    path(s, [(224, 559), (266, 559), (266, 272), (325, 272)], BLUE, 1.3)
    label(s, 269, 247, "latest state", 11, GRAY, italic=True)
    path(s, [(224, 580), (315, 580)], BLUE, 1.4)
    rect(s, 315, 551, 137, 58, BLUE_LIGHT, BLUE, 1.1, 4)
    label(s, 383.5, 580, "Linear\nembedding", 15, BLUE, True)
    path(s, [(452, 580), (720, 580)], BLUE, 1.4)
    label(s, 586, 556, "motion sequence", 13, BLUE, italic=True)

    path(s, [(1760, 740), (1800, 740)], BLUE, 1.4)
    rect(s, 1800, 698, 145, 86, "#FFF6ED", ORANGE, 1.1, 4)
    label(s, 1872.5, 724, "Prediction head", 15, ORANGE, True)
    label(s, 1872.5, 754, "AvgPool + MLP", 12, INK)
    path(s, [(1945, 740), (1980, 740)], ORANGE, 1.4)
    for i, color in enumerate(["#E7CFB7", "#DDB58D", "#D1955C", "#C97830"]):
        rect(s, 1990 + 20 * i, 710, 16, 60, color, None, 0, 2)
    label(s, 2027, 685, "Predicted motion", 14, ORANGE, True)
    label(s, 2027, 800, "Δ bbox", 15, ORANGE, True, italic=True)
    return s


def render_preview(s, scale=2):
    """Use matching serif math labels in both preview and editable output."""
    im = render({**s, "items": [item for item in s["items"]
                                if item.get("font") != "Cambria Math"]}, scale)
    draw = ImageDraw.Draw(im)
    for item in s["items"]:
        if item.get("font") != "Cambria Math":
            continue
        face = "BoldItalic" if item.get("bold") else "Italic"
        font = ImageFont.truetype(
            f"/usr/share/fonts/dejavu/DejaVuSerif-{face}.ttf",
            round(item["size"] * scale),
        )
        draw.text((item["x"] * scale, item["y"] * scale), item["text"],
                  fill=item["color"], font=font, anchor="mm")
    return im


def group_native_module(file_path, width, height):
    """Group all native shapes with an identity transform for easy reuse.

    Children retain page coordinates: a group covering the entire page has
    its lower-left local origin at the page origin, so no connector, geometry
    or text transform needs adjustment. Visio can still edit each member.
    """
    v = "http://schemas.microsoft.com/office/visio/2012/main"
    with zipfile.ZipFile(file_path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    contents = ET.fromstring(parts["visio/pages/page1.xml"])
    shapes = contents.find(f"{{{v}}}Shapes")
    ident = max(int(s.get("ID")) for s in shapes.iter(f"{{{v}}}Shape")) + 1
    group = ET.Element(f"{{{v}}}Shape", {
        "ID": str(ident), "NameU": "SSM-mod-mean", "Name": "SSM-mod-mean",
        "Type": "Group", "LineStyle": "0", "FillStyle": "0", "TextStyle": "0",
    })
    for name, value in {
        "PinX": width / 144, "PinY": height / 144,
        "Width": width / 72, "Height": height / 72,
        "LocPinX": width / 144, "LocPinY": height / 144,
        "Angle": 0, "FlipX": 0, "FlipY": 0,
    }.items():
        ET.SubElement(group, f"{{{v}}}Cell", {"N": name, "V": str(value)})
    contents.remove(shapes)
    group.append(shapes)
    ET.SubElement(contents, f"{{{v}}}Shapes").append(group)
    parts["visio/pages/page1.xml"] = ET.tostring(contents, encoding="utf-8", xml_declaration=True)

    core = ET.fromstring(parts["docProps/core.xml"])
    dc = "http://purl.org/dc/elements/1.1/"
    for name, value in {
        "title": "SSM-mod-mean: Spatial/RGB ablation module",
        "subject": "Simplified editable architecture based on ssm_mod.vsdx",
        "description": "One grouped native-vector module; Spatial/RGB hyperedges, additive Delta and multiplicative C modulation, recurrent Forward SSM core and compact Backward scan.",
    }.items():
        node = core.find(f"{{{dc}}}{name}")
        if node is None:
            node = ET.SubElement(core, f"{{{dc}}}{name}")
        node.text = value
    parts["docProps/core.xml"] = ET.tostring(core, encoding="utf-8", xml_declaration=True)
    staging = file_path.with_suffix(".tmp.vsdx")
    with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    staging.replace(file_path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = scene()
    render_preview(s, 2).save(OUT / "ssm_mod_simplified.png", "PNG", optimize=True, dpi=(220, 220))
    vsdx_path = OUT / "ssm_mod_simplified.vsdx"
    export_vsdx([s], vsdx_path)
    group_native_module(vsdx_path, s["width"], s["height"])
    print(validate_vsdx(vsdx_path))


if __name__ == "__main__":
    main()
