"""Export a simple diagram scene to editable native Visio (.vsdx) shapes.

The scene uses top-left coordinates with y increasing downward, in points
(72 scene units = 1 inch).  It is deliberately the same scene used by the
SVG/PNG renderer; no flattened diagram image is placed in the Visio file.

Public API::

    export_vsdx([{'name': 'Overview', 'width': 1200, 'height': 720,
                  'items': [{'type': 'rect', 'x': 10, 'y': 10,
                             'w': 80, 'h': 40, 'fill': '#dfeaff'}]}], path)
    report = validate_vsdx(path)

Supported types: rect, ellipse, polygon, polyline, text.  Text uses an x
anchor determined by ``align`` and a vertically centered y anchor.  Lines
are native one-dimensional shapes with unglued, independently editable endpoints.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import posixpath
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import xml.etree.ElementTree as ET


V = "http://schemas.microsoft.com/office/visio/2012/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
P = "http://schemas.openxmlformats.org/package/2006/relationships"
C = "http://schemas.openxmlformats.org/package/2006/content-types"
E = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC = "http://purl.org/dc/elements/1.1/"
DCT = "http://purl.org/dc/terms/"
XML = "http://www.w3.org/XML/1998/namespace"

for _prefix, _uri in [("v", V), ("r", R), ("pr", P), ("ct", C),
                      ("ep", E), ("vt", VT), ("cp", CP), ("dc", DC),
                      ("dcterms", DCT)]:
    ET.register_namespace(_prefix, _uri)

DEFAULT_TEMPLATE = (Path(__file__).resolve().parents[1] / "docs" /
                    "hyla_shared_mlp_architecture_editable.vsdx")
_NONE = {None, "", "none", "transparent"}


def _q(name: str, ns: str = V) -> str:
    return "{%s}%s" % (ns, name)


def _num(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Diagram coordinates must be finite")
    return format(0.0 if abs(value) < 1e-12 else value, ".12g")


def _cell(parent: ET.Element, name: str, value: Any,
          formula: Optional[str] = None) -> ET.Element:
    attrs = {"N": name, "V": _num(value) if isinstance(value, (int, float)) else str(value)}
    if formula is not None:
        attrs["F"] = formula
    return ET.SubElement(parent, _q("Cell"), attrs)


def _xml(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _paint(shape: ET.Element, item: Mapping[str, Any], *, nofill: bool = False) -> None:
    fill = item.get("fill")
    stroke = item.get("stroke", "#222222")
    weight = float(item.get("width", item.get("sw", 1.0)))
    _cell(shape, "LinePattern", 0 if stroke in _NONE or weight <= 0 else
          (2 if item.get("dashed", item.get("dash", False)) else 1))
    _cell(shape, "LineColor", "#222222" if stroke in _NONE else stroke)
    _cell(shape, "LineWeight", max(weight, 0) / 72)
    _cell(shape, "LineCap", 1)
    _cell(shape, "FillPattern", 0 if fill in _NONE or nofill else 1)
    _cell(shape, "FillForegnd", "#FFFFFF" if fill in _NONE else fill)
    _cell(shape, "FillForegndTrans", float(item.get("fill_opacity", 1.0)) * -1 + 1)
    _cell(shape, "LineColorTrans", float(item.get("stroke_opacity", 1.0)) * -1 + 1)
    _cell(shape, "ShdwPattern", 0)
    _cell(shape, "BeginArrow", 4 if item.get("arrow_start") else 0)
    _cell(shape, "EndArrow", 4 if item.get("arrow", item.get("arrow_end", False)) else 0)
    _cell(shape, "BeginArrowSize", 1)
    _cell(shape, "EndArrowSize", 1)


def _shape(shapes: ET.Element, ident: int, kind: str) -> ET.Element:
    return ET.SubElement(shapes, _q("Shape"), {
        "ID": str(ident), "NameU": "%s.%d" % (kind, ident),
        "Type": "Shape", "LineStyle": "0", "FillStyle": "0", "TextStyle": "0",
    })


def _xform(shape: ET.Element, x: float, y: float, w: float, h: float,
           page_height: float) -> None:
    for name, value, formula in [
        ("PinX", (x + w / 2) / 72, None),
        ("PinY", (page_height - y - h / 2) / 72, None),
        ("Width", w / 72, None), ("Height", h / 72, None),
        ("LocPinX", w / 144, "Width*0.5"),
        ("LocPinY", h / 144, "Height*0.5"),
        ("Angle", 0, None), ("FlipX", 0, None), ("FlipY", 0, None),
        ("ResizeMode", 0, None),
    ]:
        _cell(shape, name, value, formula)


def _geometry(shape: ET.Element, nofill: bool = False, noline: bool = False) -> ET.Element:
    geom = ET.SubElement(shape, _q("Section"), {"N": "Geometry", "IX": "0"})
    for key, val in [("NoFill", int(nofill)), ("NoLine", int(noline)),
                     ("NoShow", 0), ("NoSnap", 0), ("NoQuickDrag", 0)]:
        _cell(geom, key, val)
    return geom


def _row(geom: ET.Element, kind: str, ix: int,
         values: Mapping[str, Any]) -> ET.Element:
    row = ET.SubElement(geom, _q("Row"), {"T": kind, "IX": str(ix)})
    for name, value in values.items():
        _cell(row, name, value)
    return row


def _rectangle(shapes: ET.Element, ident: int, item: Mapping[str, Any], ph: float) -> None:
    shape = _shape(shapes, ident, "Rectangle")
    _xform(shape, item["x"], item["y"], item["w"], item["h"], ph)
    _paint(shape, item)
    _cell(shape, "Rounding", float(item.get("r", item.get("rx", 0))) / 72)
    geom = _geometry(shape, item.get("fill") in _NONE)
    for ix, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)], 1):
        _row(geom, "RelMoveTo" if ix == 1 else "RelLineTo", ix, {"X": x, "Y": y})


def _ellipse(shapes: ET.Element, ident: int, item: Mapping[str, Any], ph: float) -> None:
    if "cx" in item:
        x, y = item["cx"] - item["rx"], item["cy"] - item["ry"]
        w, h = 2 * item["rx"], 2 * item["ry"]
    else:
        x, y, w, h = item["x"], item["y"], item["w"], item["h"]
    shape = _shape(shapes, ident, "Ellipse")
    _xform(shape, x, y, w, h, ph)
    _paint(shape, item)
    geom = _geometry(shape, item.get("fill") in _NONE)
    row = _row(geom, "Ellipse", 1, {
        "X": w / 144, "Y": h / 144,
        "A": w / 72, "B": h / 144,
        "C": w / 144, "D": h / 72,
    })
    for cell in row:
        cell.set("F", {"X": "Width*0.5", "Y": "Height*0.5", "A": "Width",
                       "B": "Height*0.5", "C": "Width*0.5", "D": "Height"}[cell.get("N")])


def _polygon(shapes: ET.Element, ident: int, item: Mapping[str, Any], ph: float) -> None:
    points = [(float(x), float(y)) for x, y in item["points"]]
    if len(points) < 3:
        raise ValueError("A polygon requires at least three points")
    xmin, ymin = min(x for x, _ in points), min(y for _, y in points)
    w = max(x for x, _ in points) - xmin
    h = max(y for _, y in points) - ymin
    if w <= 0 or h <= 0:
        raise ValueError("A polygon must have nonzero width and height")
    shape = _shape(shapes, ident, "Polygon")
    _xform(shape, xmin, ymin, w, h, ph)
    _paint(shape, item)
    geom = _geometry(shape, item.get("fill") in _NONE)
    if points[-1] != points[0]:
        points.append(points[0])
    for ix, (x, y) in enumerate(points, 1):
        _row(geom, "RelMoveTo" if ix == 1 else "RelLineTo", ix,
             {"X": (x - xmin) / w, "Y": 1 - (y - ymin) / h})


def _polyline(shapes: ET.Element, ident: int, item: Mapping[str, Any], ph: float) -> None:
    """Emit a native 1-D shape; geometry is expressed in its chord frame."""
    points = [(float(x) / 72, (ph - float(y)) / 72) for x, y in item["points"]]
    if len(points) < 2:
        raise ValueError("A polyline requires at least two points")
    bx, by = points[0]
    ex, ey = points[-1]
    length = math.hypot(ex - bx, ey - by)
    if length < 1e-10:
        # Closed unfilled polylines are 2-D paths, not zero-length 1-D lines.
        closed = dict(item, fill=item.get("fill"))
        _polygon(shapes, ident, closed, ph)
        return
    angle = math.atan2(ey - by, ex - bx)
    shape = _shape(shapes, ident, "Line")
    for name, value, formula in [
        ("PinX", (bx + ex) / 2, "(BeginX+EndX)/2"),
        ("PinY", (by + ey) / 2, "(BeginY+EndY)/2"),
        ("Width", length, "SQRT((EndX-BeginX)^2+(EndY-BeginY)^2)"),
        ("Height", 0, None), ("LocPinX", length / 2, "Width*0.5"),
        ("LocPinY", 0, None),
        # ShapeSheet ATAN2 accepts x then y, unlike Python's math.atan2.
        ("Angle", angle, "ATAN2(EndX-BeginX,EndY-BeginY)"),
        ("FlipX", 0, None), ("FlipY", 0, None),
        ("BeginX", bx, None), ("BeginY", by, None),
        ("EndX", ex, None), ("EndY", ey, None),
        ("GlueType", 0, None), ("NoAlignBox", 1, None),
    ]:
        _cell(shape, name, value, formula)
    _paint(shape, item, nofill=True)
    geom = _geometry(shape, True)
    cs, sn = math.cos(angle), math.sin(angle)
    for ix, (x, y) in enumerate(points, 1):
        u, v = (x - bx) * cs + (y - by) * sn, -(x - bx) * sn + (y - by) * cs
        row = _row(geom, "MoveTo" if ix == 1 else "LineTo", ix, {"X": u, "Y": v})
        # These formulae preserve the route's proportions after resizing.
        for cell in row:
            scale = u / length if cell.get("N") == "X" else v / length
            cell.set("F", "Width*%s" % _num(scale))


def _text_extent(text: str, size: float, font: str, bold: bool,
                 italic: bool) -> tuple:
    lines = text.split("\n") or [""]
    try:
        from PIL import ImageFont
        serif = "times" in font.lower() or "serif" in font.lower()
        name = "DejaVuSerif" if serif else "DejaVuSans"
        if bold and italic:
            name += "-BoldItalic" if serif else "-BoldOblique"
        elif bold:
            name += "-Bold"
        elif italic:
            name += "-Italic" if serif else "-Oblique"
        face = ImageFont.truetype(name + ".ttf", round(size * 4))
        width = max(float(face.getlength(line)) / 4 for line in lines)
    except (ImportError, OSError, AttributeError):
        width = max(sum(1.0 if ord(c) > 255 else 0.58 for c in line) * size for line in lines)
    # Small breathing room guards against font substitution in Visio.
    return max(width * 1.04 + size * .2, size), max(len(lines) * size * 1.35, size * 1.35)


def _text(shapes: ET.Element, ident: int, item: Mapping[str, Any], ph: float,
          fonts: Mapping[str, int]) -> None:
    value = str(item["text"])
    size = float(item.get("size", 14))
    font = str(item.get("font", "Arial"))
    bold, italic = bool(item.get("bold", False)), bool(item.get("italic", False))
    align = item.get("align", "center")
    w, h = _text_extent(value, size, font, bold, italic)
    if "w" in item:
        w = float(item["w"])
    if "h" in item:
        h = float(item["h"])
    x = float(item["x"]) - ({"left": 0, "center": w / 2, "right": w}.get(align, w / 2))
    y = float(item["y"]) - h / 2
    shape = _shape(shapes, ident, "Text")
    _xform(shape, x, y, w, h, ph)
    _paint(shape, {"fill": None, "stroke": None, "width": 0})
    for key, val in [("VerticalAlign", 1), ("LeftMargin", 0), ("RightMargin", 0),
                     ("TopMargin", 0), ("BottomMargin", 0), ("TextBkgndTrans", 1)]:
        _cell(shape, key, val)
    _geometry(shape, True, True)
    chars = ET.SubElement(shape, _q("Section"), {"N": "Character"})
    row = ET.SubElement(chars, _q("Row"), {"IX": "0"})
    _cell(row, "Font", fonts[font], 'FONT("%s")' % font.replace('"', '""'))
    _cell(row, "Size", size / 72)
    _cell(row, "Color", item.get("color", "#222222"))
    _cell(row, "Style", (1 if bold else 0) + (2 if italic else 0))
    _cell(row, "Case", 0)
    _cell(row, "Pos", 0)
    para = ET.SubElement(shape, _q("Section"), {"N": "Paragraph"})
    row = ET.SubElement(para, _q("Row"), {"IX": "0"})
    _cell(row, "HorzAlign", {"left": 0, "center": 1, "right": 2}.get(align, 1))
    _cell(row, "IndFirst", 0)
    _cell(row, "IndLeft", 0)
    _cell(row, "IndRight", 0)
    _cell(row, "SpBefore", 0)
    _cell(row, "SpAfter", 0)
    _cell(row, "SpLine", -1.0)
    ET.SubElement(shape, _q("Text")).text = value


def _font_table(document: ET.Element, pages: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    faces = document.find(_q("FaceNames"))
    if faces is None:
        faces = ET.SubElement(document, _q("FaceNames"))
    found = {}
    for index, face in enumerate(faces):
        ident = int(face.get("ID", index))
        face.set("ID", str(ident))
        found[face.get("NameU", face.get("Name", "Calibri"))] = ident
    desired = {str(item.get("font", "Arial")) for page in pages for item in page["items"]
               if item.get("type", item.get("kind")) == "text"}
    for name in sorted(desired):
        if name in found:
            continue
        ident = max(found.values(), default=-1) + 1
        ET.SubElement(faces, _q("FaceName"), {"ID": str(ident), "NameU": name,
                                             "Name": name, "Flags": "0"})
        found[name] = ident
    return found


def _metadata(parts: Dict[str, bytes], pages: Sequence[Mapping[str, Any]]) -> None:
    core = ET.fromstring(parts["docProps/core.xml"])
    values = {
        _q("title", DC): "Multimodal MOT: YOLOX-HyLA and SSM-mod-mean",
        _q("subject", DC): "Editable native Visio scientific architecture diagrams",
        _q("description", DC): "Three architecture diagrams. Shapes, text, and lines are individually editable.",
        _q("modified", DCT): datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for tag, value in values.items():
        node = core.find(tag)
        if node is None:
            node = ET.SubElement(core, tag)
        node.text = value
    parts["docProps/core.xml"] = _xml(core)
    if "docProps/app.xml" in parts:
        app = ET.fromstring(parts["docProps/app.xml"])
        for child in list(app):
            if child.tag in {_q("HeadingPairs", E), _q("TitlesOfParts", E)}:
                app.remove(child)
        heading = ET.SubElement(app, _q("HeadingPairs", E))
        vector = ET.SubElement(heading, _q("vector", VT), {"size": "2", "baseType": "variant"})
        variant = ET.SubElement(vector, _q("variant", VT))
        ET.SubElement(variant, _q("lpstr", VT)).text = "Pages"
        variant = ET.SubElement(vector, _q("variant", VT))
        ET.SubElement(variant, _q("i4", VT)).text = str(len(pages))
        titles = ET.SubElement(app, _q("TitlesOfParts", E))
        vector = ET.SubElement(titles, _q("vector", VT), {"size": str(len(pages)), "baseType": "lpstr"})
        for page in pages:
            ET.SubElement(vector, _q("lpstr", VT)).text = str(page["name"])
        parts["docProps/app.xml"] = _xml(app)


def export_vsdx(pages: Sequence[Mapping[str, Any]], output_path: Any,
                template_path: Any = None) -> Dict[str, Any]:
    """Write one or more scene pages as a native editable Visio document.

    Unknown scene item types fail explicitly instead of silently disappearing.
    The return value is the result of :func:`validate_vsdx`.
    """
    if not pages:
        raise ValueError("At least one page is required")
    template = Path(template_path) if template_path else DEFAULT_TEMPLATE
    with zipfile.ZipFile(template) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    old_pages = ET.fromstring(parts["visio/pages/pages.xml"])
    source_sheet = old_pages.find("%s/%s" % (_q("Page"), _q("PageSheet")))
    for name in list(parts):
        if re.match(r"visio/pages/(?:_rels/)?page\d+\.xml(?:\.rels)?$", name):
            del parts[name]
    # A copied preview would misleadingly depict the previous document.
    parts.pop("docProps/thumbnail.emf", None)
    root_rels = ET.fromstring(parts["_rels/.rels"])
    for rel in list(root_rels):
        if rel.get("Type", "").endswith("/thumbnail"):
            root_rels.remove(rel)
    parts["_rels/.rels"] = _xml(root_rels)
    document = ET.fromstring(parts["visio/document.xml"])
    fonts = _font_table(document, pages)
    parts["visio/document.xml"] = _xml(document)
    pages_xml = ET.Element(_q("Pages"), {_q("space", XML): "preserve"})
    page_rels = ET.Element(_q("Relationships", P))
    types = ET.fromstring(parts["[Content_Types].xml"])
    for child in list(types):
        if re.match(r"/visio/pages/page\d+\.xml$", child.get("PartName", "")):
            types.remove(child)
    dispatch = {"rect": _rectangle, "ellipse": _ellipse, "polygon": _polygon,
                "polyline": _polyline}
    for index, page in enumerate(pages, 1):
        pw, ph = float(page["width"]), float(page["height"])
        if pw <= 0 or ph <= 0:
            raise ValueError("Page dimensions must be positive")
        entry = ET.SubElement(pages_xml, _q("Page"), {
            "ID": str(index - 1), "NameU": str(page["name"]), "Name": str(page["name"]),
            "ViewScale": "0.6", "ViewCenterX": _num(pw / 144), "ViewCenterY": _num(ph / 144),
        })
        sheet = copy.deepcopy(source_sheet) if source_sheet is not None else ET.Element(_q("PageSheet"))
        for name, value in [("PageWidth", pw / 72), ("PageHeight", ph / 72)]:
            cell = sheet.find("%s[@N='%s']" % (_q("Cell"), name))
            if cell is None:
                _cell(sheet, name, value)
            else:
                cell.set("V", _num(value))
                cell.attrib.pop("F", None)
        entry.append(sheet)
        ET.SubElement(entry, _q("Rel"), {_q("id", R): "rId%d" % index})
        ET.SubElement(page_rels, _q("Relationship", P), {
            "Id": "rId%d" % index, "Type": "http://schemas.microsoft.com/visio/2010/relationships/page",
            "Target": "page%d.xml" % index,
        })
        contents = ET.Element(_q("PageContents"), {_q("space", XML): "preserve"})
        shapes = ET.SubElement(contents, _q("Shapes"))
        for ident, item in enumerate(page["items"], 1):
            kind = item.get("type", item.get("kind"))
            if kind == "text":
                _text(shapes, ident, item, ph, fonts)
            elif kind in dispatch:
                dispatch[kind](shapes, ident, item, ph)
            else:
                raise ValueError("Unsupported diagram item type: %r" % kind)
        parts["visio/pages/page%d.xml" % index] = _xml(contents)
        ET.SubElement(types, _q("Override", C), {
            "PartName": "/visio/pages/page%d.xml" % index,
            "ContentType": "application/vnd.ms-visio.page+xml",
        })
    parts["visio/pages/pages.xml"] = _xml(pages_xml)
    parts["visio/pages/_rels/pages.xml.rels"] = _xml(page_rels)
    parts["[Content_Types].xml"] = _xml(types)
    _metadata(parts, pages)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    return validate_vsdx(output)


write_vsdx = export_vsdx


def validate_vsdx(path: Any) -> Dict[str, Any]:
    """Check package/XML integrity, native shapes, relationship targets, and IDs.

    This is a structural check, not a substitute for rendering in Microsoft
    Visio. It raises ValueError for malformed documents and returns counts.
    """
    report = {"path": str(Path(path).resolve()), "pages": [], "shape_count": 0,
              "text_count": 0, "connector_count": 0, "foreign_count": 0}
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt zip member")
        names = set(archive.namelist())
        parsed = {name: ET.fromstring(archive.read(name)) for name in names
                  if name.endswith(".xml") or name.endswith(".rels")}
        for name, root in parsed.items():
            if not name.endswith(".rels"):
                continue
            source = "" if name == "_rels/.rels" else name.replace("/_rels/", "/")[:-5]
            base = posixpath.dirname(source)
            for rel in root:
                if rel.get("TargetMode") == "External":
                    continue
                target = rel.get("Target", "")
                resolved = posixpath.normpath(posixpath.join(base, target)).lstrip("/")
                if resolved not in names:
                    raise ValueError("Broken relationship %s -> %s" % (name, target))
        for child in parsed["[Content_Types].xml"]:
            part = child.get("PartName")
            if part and part.lstrip("/") not in names:
                raise ValueError("Content type points at missing part: " + part)
        page_names = [e.get("Name") for e in parsed["visio/pages/pages.xml"]]
        for index, name in enumerate(page_names, 1):
            root = parsed["visio/pages/page%d.xml" % index]
            shapes = root.findall(".//" + _q("Shape"))
            ids = [shape.get("ID") for shape in shapes]
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate shape IDs on page " + str(index))
            texts = sum(s.find(_q("Text")) is not None for s in shapes)
            connectors = sum(s.find("%s[@N='BeginX']" % _q("Cell")) is not None for s in shapes)
            foreign = sum(s.get("Type") == "Foreign" for s in shapes)
            report["pages"].append({"name": name, "shapes": len(shapes),
                                    "texts": texts, "connectors": connectors, "foreign": foreign})
            report["shape_count"] += len(shapes)
            report["text_count"] += texts
            report["connector_count"] += connectors
            report["foreign_count"] += foreign
    report["valid"] = True
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON scene or existing VSDX to validate")
    parser.add_argument("output", type=Path, nargs="?", help="Output VSDX for a JSON scene")
    parser.add_argument("--template", type=Path, default=None)
    args = parser.parse_args()
    if args.input.suffix.lower() == ".vsdx":
        result = validate_vsdx(args.input)
    else:
        if args.output is None:
            parser.error("An output VSDX path is required for a JSON scene")
        value = json.loads(args.input.read_text())
        result = export_vsdx(value["pages"] if isinstance(value, dict) else value,
                             args.output, args.template)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
