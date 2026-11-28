#!/usr/bin/env python3
"""Embed real FEMOT raster frames into an existing one-page Visio drawing.

The input VSDX is treated as a ZIP package.  Existing XML is kept intact as
much as possible: placeholder shapes are hidden, image parts and relationships
are added, and three native Visio ``Foreign`` bitmap shapes are appended to the
page.  This avoids rebuilding the user's hand-edited diagram.

Usage::

    python tools/embed_real_data_images.py \
        --input /path/to/overall_framework_revised.vsdx \
        --output docs/overall_framework_real_data.vsdx \
        --rgb rgb.png --event event.png --detections detections.png

Images are converted to PNG before embedding.  The script does not alter the
source VSDX.
"""

from __future__ import annotations

import argparse
import copy
import io
import mimetypes
import pathlib
import re
import uuid
import zipfile
import xml.etree.ElementTree as ET

from PIL import Image


V = "http://schemas.microsoft.com/office/visio/2012/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("", V)
ET.register_namespace("r", R)


def q(ns: str, tag: str) -> str:
    return "{" + ns + "}" + tag


def cell(name: str, value: object, formula: str | None = None) -> ET.Element:
    attrs = {"N": name, "V": str(value)}
    if formula is not None:
        attrs["F"] = formula
    return ET.Element(q(V, "Cell"), attrs)


def image_shape(shape_id: int, pin_x: float, pin_y: float, width: float,
                height: float, rel_id: str, name: str) -> ET.Element:
    """Build a page-level Foreign bitmap shape matching Visio's own output."""
    s = ET.Element(q(V, "Shape"), {
        "ID": str(shape_id),
        "NameU": name,
        "Name": name,
        "Type": "Foreign",
        # The style IDs are harmless in this document and match ordinary image
        # shapes exported by Visio.  The actual pixels come from ForeignData.
        "LineStyle": "2",
        "FillStyle": "2",
        "TextStyle": "2",
        "UniqueID": "{" + str(uuid.uuid4()).upper() + "}",
    })
    # PinX/PinY are the center of a page shape.  ImgWidth/ImgHeight make the
    # image fill its shape and prevent Visio from inheriting stale crop values.
    for n, v, f in (
        ("PinX", pin_x, None), ("PinY", pin_y, None),
        ("Width", width, None), ("Height", height, None),
        ("LocPinX", width / 2.0, "Width*0.5"),
        ("LocPinY", height / 2.0, "Height*0.5"),
        ("Angle", 0, None), ("FlipX", 0, None), ("FlipY", 0, None),
        ("ResizeMode", 0, None),
        ("ImgOffsetX", 0, "ImgWidth*0"),
        ("ImgOffsetY", 0, "ImgHeight*0"),
        ("ImgWidth", width, "Width*1"),
        ("ImgHeight", height, "Height*1"),
        ("ClippingPath", "", None),
        ("TxtPinX", width / 2.0, "Width*0.5"),
        ("TxtPinY", 0, "Height*0"),
        ("TxtWidth", width, "Width*1"),
        ("TxtHeight", 0, "Height*0"),
        ("TxtLocPinX", width / 2.0, "TxtWidth*0.5"),
        ("TxtLocPinY", 0, "TxtHeight*0.5"),
        ("TxtAngle", 0, None),
        ("VerticalAlign", 0, None),
        # These cells are present on Visio's own picture shapes.  They are
        # useful when the file is opened and edited in desktop Visio, while
        # remaining harmless for other VSDX readers.
        ("Relationships", 0, None),
        ("ObjType", 1, None),
        ("ShapeShdwShow", 2, None),
    ):
        c = cell(n, v, f)
        if n == "ClippingPath":
            c.set("E", "#N/A")
        s.append(c)

    geom = ET.SubElement(s, q(V, "Section"), {"N": "Geometry", "IX": "0"})
    for n, v in (("NoFill", 0), ("NoLine", 0), ("NoShow", 0),
                 ("NoSnap", 0), ("NoQuickDrag", 0)):
        geom.append(cell(n, v))
    rows = (
        ("RelMoveTo", 1, 0, 0),
        ("RelLineTo", 2, 1, 0),
        ("RelLineTo", 3, 1, 1),
        ("RelLineTo", 4, 0, 1),
        ("RelLineTo", 5, 0, 0),
    )
    for row_type, ix, x, y in rows:
        row = ET.SubElement(geom, q(V, "Row"), {"T": row_type, "IX": str(ix)})
        row.append(cell("X", x))
        row.append(cell("Y", y))

    foreign = ET.SubElement(s, q(V, "ForeignData"), {
        "ForeignType": "Bitmap", "CompressionType": "PNG"
    })
    ET.SubElement(foreign, q(V, "Rel"), {q(R, "id"): rel_id})
    return s


def _hide_shape(root: ET.Element, shape_id: str) -> bool:
    shape = root.find(".//" + q(V, "Shape") + "[@ID='" + shape_id + "']")
    if shape is None:
        return False
    # Keep the original objects in the document for editability, but make their
    # fill/line invisible so they cannot obscure the embedded photograph.
    for n in ("FillPattern", "LinePattern"):
        c = shape.find(q(V, "Cell") + "[@N='" + n + "']")
        if c is None:
            c = ET.SubElement(shape, q(V, "Cell"), {"N": n})
        c.set("V", "0")
    return True


def _next_shape_id(root: ET.Element) -> int:
    ids = []
    for shape in root.findall(".//" + q(V, "Shape")):
        try:
            ids.append(int(shape.get("ID", "0")))
        except ValueError:
            pass
    return max(ids, default=0) + 1


def _png_bytes(path: pathlib.Path, target_ratio: float | None = None) -> bytes:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if target_ratio and target_ratio > 0:
            ratio = im.width / im.height
            if ratio > target_ratio:
                new_w = max(1, round(im.height * target_ratio))
                left = (im.width - new_w) // 2
                im = im.crop((left, 0, left + new_w, im.height))
            elif ratio < target_ratio:
                new_h = max(1, round(im.width / target_ratio))
                top = (im.height - new_h) // 2
                im = im.crop((0, top, im.width, top + new_h))
        # Keep the package compact while retaining details visible at the small
        # frame sizes in the overall figure.
        im.thumbnail((1200, 900), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        im.save(out, format="PNG", optimize=True)
        return out.getvalue()


def _ensure_png_default(types_root: ET.Element) -> None:
    if types_root.find(q(CONTENT_TYPES, "Default") + "[@Extension='png']") is None:
        ET.SubElement(types_root, q(CONTENT_TYPES, "Default"), {
            "Extension": "png", "ContentType": "image/png"
        })


def embed(input_vsdx: pathlib.Path, output_vsdx: pathlib.Path,
          rgb: pathlib.Path, event: pathlib.Path, detections: pathlib.Path) -> None:
    with zipfile.ZipFile(input_vsdx, "r") as zin:
        parts = {name: zin.read(name) for name in zin.namelist()}

    page_name = "visio/pages/page1.xml"
    rels_name = "visio/pages/_rels/page1.xml.rels"
    types_name = "[Content_Types].xml"
    if page_name not in parts or types_name not in parts:
        raise ValueError("input is missing page1.xml or Content_Types.xml")

    page_root = ET.fromstring(parts[page_name])
    if rels_name in parts:
        rels_root = ET.fromstring(parts[rels_name])
    else:
        # Visio omits a page .rels part when the page has no external assets.
        # Adding this standard empty relationships part is enough to support
        # the image relationships below.
        rels_root = ET.Element(q(PKG_REL, "Relationships"))
    types_root = ET.fromstring(parts[types_name])

    # Preserve the hand-edited outer frame and captions.  Hide only the nested
    # placeholder art and four detection bars.
    for sid in ("7", "8", "9", "10", "13", "14", "15", "16",
                "90", "91", "92", "93"):
        _hide_shape(page_root, sid)

    max_id = _next_shape_id(page_root)
    media = (
        ("rgb_frame.png", rgb, 1.77777777778, 10.3611111111, 1.0, 0.666666666667),
        ("event_frame.png", event, 1.77777777778, 8.79166666667, 1.0, 0.666666666667),
        ("detections.png", detections, 20.8125, 10.0555555556, 1.291666666666, 0.833333333333),
    )
    shapes_parent = page_root.find(q(V, "Shapes"))
    if shapes_parent is None:
        raise ValueError("page1.xml has no Shapes element")

    # Existing relationship IDs are arbitrary; choose the next numeric suffix.
    used_rids = [int(m.group(1)) for e in rels_root
                 for m in [re.fullmatch(r"rId(\d+)", e.get("Id", ""))] if m]
    next_rid = max(used_rids, default=0) + 1
    _ensure_png_default(types_root)
    for filename, source, x, y, w, h in media:
        rid = "rId" + str(next_rid)
        next_rid += 1
        data = _png_bytes(source, target_ratio=w / h)
        parts["visio/media/" + filename] = data
        ET.SubElement(rels_root, q(PKG_REL, "Relationship"), {
            "Id": rid,
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "Target": "../media/" + filename,
        })
        shapes_parent.append(image_shape(max_id, x, y, w, h, rid, filename.rsplit(".", 1)[0]))
        max_id += 1

    def serialize(root: ET.Element) -> bytes:
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    parts[page_name] = serialize(page_root)
    parts[rels_name] = serialize(rels_root)
    parts[types_name] = serialize(types_root)

    output_vsdx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_vsdx, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in parts.items():
            zout.writestr(name, data)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=pathlib.Path, required=True)
    p.add_argument("--output", type=pathlib.Path, required=True)
    p.add_argument("--rgb", type=pathlib.Path, required=True)
    p.add_argument("--event", type=pathlib.Path, required=True)
    p.add_argument("--detections", type=pathlib.Path, required=True)
    args = p.parse_args()
    embed(args.input, args.output, args.rgb, args.event, args.detections)
    print(args.output)


if __name__ == "__main__":
    main()
