#!/usr/bin/env python3
"""Create the sparse paper-style concept figure from the final uploaded VSDX.

Only HyLA and the hypergraph-guided bidirectional SSM are highlighted.  The
RGB, event and track images are copied byte-for-byte from the source package;
feature stacks, trajectories and the hypergraph are explicitly schematic.
All diagram text, modules and arrows remain native editable Visio objects.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import html
import json
import math
import pathlib
import posixpath
import zipfile
import xml.etree.ElementTree as ET

import cairosvg

from embed_real_data_images import image_shape
from export_diagram_vsdx import export_vsdx, validate_vsdx


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = pathlib.Path(
    "/home/lyt/.codex/attachments/1e84ad6a-612e-4b05-82c4-a1b403e9a488/"
    "overall_framework_revised.vsdx"
)
V = "http://schemas.microsoft.com/office/visio/2012/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"
C = "http://schemas.openxmlformats.org/package/2006/content-types"
DC = "http://purl.org/dc/elements/1.1/"
INK = "#171717"
GRAY = "#575460"
PURPLE = "#793BB7"
LAVENDER = "#F4EDFB"
RGB_COLOR = "#BC454C"
EVENT_COLOR = "#355CC4"
FONT = "Times New Roman"
W, H = 1415, 425


def q(tag: str, ns: str = V) -> str:
    return "{" + ns + "}" + tag


def extract_images(source: pathlib.Path, asset_dir: pathlib.Path) -> dict:
    """Follow the final document's ForeignData relationships, not a dataset path."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    roles = {"rgb_input": "205", "event_input": "207", "output_tracks": "206"}
    images = {}
    with zipfile.ZipFile(source) as archive:
        page = ET.fromstring(archive.read("visio/pages/page1.xml"))
        rels = ET.fromstring(archive.read("visio/pages/_rels/page1.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        for role, ident in roles.items():
            shape = page.find(".//" + q("Shape") + "[@ID='" + ident + "']")
            if shape is None or shape.get("Type") != "Foreign":
                raise ValueError(f"Missing source image shape {ident}: {role}")
            rel = shape.find(".//" + q("Rel"))
            rid = rel.get(q("id", R))
            part = posixpath.normpath(posixpath.join("visio/pages", targets[rid]))
            data = archive.read(part)
            destination = asset_dir / (role + pathlib.Path(part).suffix)
            destination.write_bytes(data)
            images[role] = {
                "path": str(destination.resolve()), "part": part,
                "shape_id": ident, "relationship": rid,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    return images


def make_scene(images: dict) -> dict:
    scene = {"name": "Concept overview", "width": W, "height": H, "items": []}
    items = scene["items"]

    def add(kind, **kwargs):
        items.append({"type": kind, **kwargs})

    def rect(x, y, w, h, fill="#FFFFFF", stroke=INK, width=1.3, r=10):
        add("rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, width=width, r=r)

    def text(x, y, value, size=21, color=INK, angle=0, italic=False):
        add("text", x=x, y=y, text=value, size=size, font=FONT,
            color=color, angle=angle, italic=italic, align="center")

    def line(points, color=INK, width=1.6, arrow=True):
        add("polyline", points=points, stroke=color, width=width, fill=None, arrow=arrow)

    def circle(x, y, radius, fill="#FFFFFF", stroke=INK, width=1):
        add("ellipse", cx=x, cy=y, rx=radius, ry=radius,
            fill=fill, stroke=stroke, width=width)

    def curve(points, color):
        # Sample one cubic Bezier to keep the Visio curve a native connector.
        sampled = []
        for i in range(25):
            t = i / 24
            sampled.append(tuple(
                (1-t)**3*points[0][a] + 3*(1-t)**2*t*points[1][a]
                + 3*(1-t)*t*t*points[2][a] + t**3*points[3][a]
                for a in (0, 1)
            ))
        line(sampled, color, 1.8)

    def image(role, x, y, width):
        # Every source image is 1038 x 780. Preserve the full field of view.
        height = width * 780 / 1038
        add("image", role=role, path=images[role]["path"], x=x, y=y,
            w=width, h=height)
        rect(x, y, width, height, fill=None, stroke=INK, width=.85, r=0)

    def features(x, y, colors):
        for i in (2, 1, 0):
            rect(x + i*5, y - i*4, 74, 60, fill=colors[i],
                 stroke=colors[3], width=.85, r=0)

    # The two input streams follow the visual grammar of the paper reference.
    image("rgb_input", 26, 34, 160)
    image("event_input", 26, 246, 160)
    text(106, 175, "RGB input")
    text(106, 387, "Event input")
    for y in (52, 264):
        rect(226, y, 42, 86, r=12)
        text(247, y+43, "Backbone", size=20, angle=90)
        line([(186, y+43), (226, y+43)])
        line([(268, y+43), (312, y+43)])

    features(312, 65, ["#D9E3EF", "#E8EFF6", "#F4F7FB", "#687A91"])
    features(312, 277, ["#E7DCEA", "#F0E8F2", "#F8F3F9", "#8B6C91"])
    text(354, 151, "RGB features", 19)
    text(354, 363, "Event features", 19)

    # A single concept block represents fusion at all three backbone scales.
    rect(440, 122, 64, 158, LAVENDER, PURPLE, 1.8, 18)
    text(472, 201, "HyLA fusion", 23, PURPLE, angle=90)
    curve([(396, 95), (442, 95), (472, 100), (472, 122)], RGB_COLOR)
    curve([(396, 307), (443, 307), (472, 302), (472, 280)], EVENT_COLOR)
    text(493, 330, "dark3 / dark4 / dark5", 17, PURPLE)

    features(551, 167, ["#DAD2EB", "#E9E2F4", "#F4EFFA", "#815BA5"])
    line([(504, 201), (551, 201)], PURPLE, 1.8)
    text(593, 255, "Fused features", 19)
    line([(635, 201), (683, 201)])
    rect(683, 171, 128, 60, r=12)
    text(747, 201, "Neck & head", 20)
    text(747, 255, "YOLOX detection", 18)
    line([(811, 201), (1092, 201)])
    text(950, 179, "Detections", 19)

    # The motion branch is deliberately one compact block, with no matrices,
    # aggregation operators, residual connections or separate scan boxes.
    history_colors = ("#527BA0", "#799063", "#AA7769")
    for row, color in enumerate(history_colors):
        points = [(610 + i*27, 306 + row*18 + [3, -2, 1, -3, 0][i])
                  for i in range(5)]
        line(points, color, 1.2, False)
        for j, (x, y) in enumerate(points):
            circle(x, y, 3.5, color if j == 4 else "#FFFFFF", color, .9)
    text(664, 380, "Track history", 20)
    line([(727, 326), (775, 326)])
    rect(775, 272, 260, 123, LAVENDER, PURPLE, 1.8, 13)
    text(905, 293, "SSM-mod-mean", 22, PURPLE)

    # Overlapping spatial/RGB hyperedges, shown as one relation-prior icon.
    add("ellipse", cx=837, cy=348, rx=24, ry=13,
        fill="#FBE3E4", stroke=RGB_COLOR, width=1.0)
    add("ellipse", cx=857, cy=353, rx=24, ry=14,
        fill="#E1EAFB", stroke=EVENT_COLOR, width=1.0)
    for x, y in [(819, 348), (835, 341), (846, 353),
                 (853, 345), (865, 358), (878, 350)]:
        circle(x, y, 3.3, "#FFFFFF", "#7E8190", .8)
    text(848, 377, "Spatial + RGB prior", 14)
    line([(883, 350), (913, 350)], PURPLE, 1.5)
    # Two opposed scan strokes are a conceptual symbol for the bidirectional
    # SSM; the hypergraph icon points to the symbol as a whole.
    line([(925, 341), (1007, 341)], PURPLE, 1.5)
    line([(1007, 360), (925, 360)], PURPLE, 1.5)
    text(967, 377, "Bi-SSM", 17)

    line([(1035, 341), (1064, 341), (1064, 308), (1092, 308)])
    text(1064, 367, "Motion", 15, GRAY, italic=True)
    rect(1092, 166, 46, 172, r=12)
    text(1115, 252, "Association", 22, angle=90)
    line([(1138, 252), (1185, 252)])
    image("output_tracks", 1185, 176, 202)
    text(1286, 354, "Output tracks", 21)
    return scene


def svg_text(item: dict) -> str:
    angle = item.get("angle", 0)
    transform = (f' transform="rotate({angle} {item["x"]} {item["y"]})"'
                 if angle else "")
    italic = ' font-style="italic"' if item.get("italic") else ""
    return (
        f'<text x="{item["x"]}" y="{item["y"]}" text-anchor="middle" '
        f'dominant-baseline="central" font-family="Times New Roman, Liberation Serif, serif" '
        f'font-size="{item["size"]}" fill="{item["color"]}"{italic}{transform}>'
        + html.escape(item["text"]) + '</text>'
    )


def write_svg(scene: dict, output: pathlib.Path) -> None:
    body = []
    for item in scene["items"]:
        kind = item["type"]
        stroke = item.get("stroke") or "none"
        fill = item.get("fill") or "none"
        style = f'fill="{fill}" stroke="{stroke}" stroke-width="{item.get("width", 1)}"'
        if kind == "rect":
            body.append(f'<rect x="{item["x"]}" y="{item["y"]}" '
                        f'width="{item["w"]}" height="{item["h"]}" '
                        f'rx="{item.get("r", 0)}" {style}/>')
        elif kind == "ellipse":
            body.append(f'<ellipse cx="{item["cx"]}" cy="{item["cy"]}" '
                        f'rx="{item["rx"]}" ry="{item["ry"]}" {style}/>')
        elif kind == "text":
            body.append(svg_text(item))
        elif kind == "polyline":
            points = item["points"]
            coords = " ".join(f'{x:.4f},{y:.4f}' for x, y in points)
            body.append(f'<polyline points="{coords}" {style} stroke-linecap="round" '
                        'stroke-linejoin="round"/>')
            if item.get("arrow"):
                x, y = points[-1]
                px, py = points[-2]
                theta = math.atan2(y-py, x-px)
                length = 7.0
                wings = [(x-length*math.cos(theta-a), y-length*math.sin(theta-a))
                         for a in (.42, -.42)]
                coords = " ".join(f'{a:.4f},{b:.4f}' for a, b in [(x, y), *wings])
                body.append(f'<polygon points="{coords}" fill="{stroke}"/>')
        elif kind == "image":
            data = base64.b64encode(pathlib.Path(item["path"]).read_bytes()).decode()
            body.append(f'<image x="{item["x"]}" y="{item["y"]}" '
                        f'width="{item["w"]}" height="{item["h"]}" '
                        f'href="data:image/jpeg;base64,{data}" preserveAspectRatio="xMidYMid meet"/>')
        else:
            raise ValueError(kind)
    output.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="14in" height="{14*H/W:.8f}in" '
        f'viewBox="0 0 {W} {H}">\n'
        '<title>HyLA fusion and hypergraph-guided bidirectional SSM for multimodal MOT</title>\n'
        '<desc>Paired real RGB and event inputs are fused by HyLA at dark3, dark4 and dark5. '
        'YOLOX detections and hypergraph-guided SSM motion predictions enter association '
        'to produce tracked objects with IDs. Real inputs and output are extracted '
        'unchanged from the final overall-framework Visio document.</desc>\n'
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>\n'
        + '\n'.join(body) + '\n</svg>\n', encoding="utf-8"
    )


def scale_for_visio(scene: dict) -> dict:
    result = copy.deepcopy(scene)
    scale = 14*72/W
    result["width"] *= scale
    result["height"] *= scale
    for item in result["items"]:
        for key in ("x", "y", "w", "h", "cx", "cy", "rx", "ry", "r", "size", "width"):
            if key in item:
                item[key] *= scale
        if "points" in item:
            item["points"] = [(x*scale, y*scale) for x, y in item["points"]]
    return result


def write_visio(scene: dict, output: pathlib.Path) -> dict:
    scaled = scale_for_visio(scene)
    plain = copy.deepcopy(scaled)
    for item in plain["items"]:
        if item["type"] == "image":
            item.update(type="rect", fill=None, stroke=None, width=0, r=0)
    export_vsdx([plain], output)
    with zipfile.ZipFile(output) as archive:
        parts = {n: archive.read(n) for n in archive.namelist()}
    page = ET.fromstring(parts["visio/pages/page1.xml"])
    shapes = page.find(q("Shapes"))
    relationships = ET.Element(q("Relationships", P))
    types = ET.fromstring(parts["[Content_Types].xml"])
    if types.find(q("Default", C) + "[@Extension='jpeg']") is None:
        ET.SubElement(types, q("Default", C), {"Extension": "jpeg", "ContentType": "image/jpeg"})
    for index, item in enumerate(scaled["items"], 1):
        shape = shapes.find(q("Shape") + "[@ID='" + str(index) + "']")
        if item["type"] == "text" and item.get("angle"):
            shape.find(q("Cell") + "[@N='Angle']").set("V", str(-math.radians(item["angle"])))
        if item["type"] != "image":
            continue
        rid = "rId" + str(index)
        filename = item["role"] + ".jpeg"
        data = pathlib.Path(item["path"]).read_bytes()
        parts["visio/media/" + filename] = data
        ET.SubElement(relationships, q("Relationship", P), {
            "Id": rid,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "Target": "../media/" + filename,
        })
        replacement = image_shape(
            index, (item["x"]+item["w"]/2)/72,
            (scaled["height"]-item["y"]-item["h"]/2)/72,
            item["w"]/72, item["h"]/72, rid, item["role"]
        )
        replacement.find(q("ForeignData")).set("CompressionType", "JPEG")
        for name in ("LinePattern", "FillPattern"):
            replacement.insert(0, ET.Element(q("Cell"), {"N": name, "V": "0"}))
        position = list(shapes).index(shape)
        shapes.remove(shape)
        shapes.insert(position, replacement)
    core = ET.fromstring(parts["docProps/core.xml"])
    for tag, value in (
        (q("title", DC), "HyLA + SSM-mod-mean: conceptual overview"),
        (q("description", DC), "A concept figure with editable native shapes and three original embedded images from the final framework VSDX."),
    ):
        element = core.find(tag)
        if element is None:
            element = ET.SubElement(core, tag)
        element.text = value
    for name, root in (
        ("visio/pages/page1.xml", page),
        ("visio/pages/_rels/page1.xml.rels", relationships),
        ("[Content_Types].xml", types), ("docProps/core.xml", core),
    ):
        parts[name] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in parts.items():
            archive.writestr(name, data)
    return validate_vsdx(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-vsdx", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT/"docs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = extract_images(args.input_vsdx, args.output_dir/"concept_framework_assets")
    scene = make_scene(images)
    prefix = args.output_dir/"overall_framework_concept"
    write_svg(scene, prefix.with_suffix(".svg"))
    cairosvg.svg2png(url=str(prefix.with_suffix(".svg")),
                    write_to=str(prefix.with_suffix(".png")),
                    output_width=W*3, output_height=H*3)
    cairosvg.svg2pdf(url=str(prefix.with_suffix(".svg")),
                    write_to=str(prefix.with_suffix(".pdf")))
    report = write_visio(scene, prefix.with_suffix(".vsdx"))
    manifest = {
        "source_vsdx": str(args.input_vsdx.resolve()),
        "source_sha256": hashlib.sha256(args.input_vsdx.read_bytes()).hexdigest(),
        "images": images,
        "schematic_elements": ["feature stacks", "track history", "hypergraph", "scan directions"],
        "visio_validation": report,
    }
    prefix.with_name(prefix.name+"_sources").with_suffix(".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for suffix in (".vsdx", ".png", ".svg", ".pdf"):
        print(prefix.with_suffix(suffix))


if __name__ == "__main__":
    main()
