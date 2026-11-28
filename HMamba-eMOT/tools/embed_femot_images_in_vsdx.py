"""Embed real FEMOT frames into the user's editable Visio framework.

The source document is intentionally treated as an opaque Visio package.  The
page XML is edited in place so hand-adjusted shapes, labels, and connectors are
kept intact; only the three image placeholders are replaced with Foreign
Bitmap shapes and the corresponding package relationships are added.
"""

from __future__ import annotations

import re
import uuid
import zipfile
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

from PIL import Image


V = "http://schemas.microsoft.com/office/visio/2012/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"

ET.register_namespace("", V)
ET.register_namespace("r", R)
ET.register_namespace("", REL)
ET.register_namespace("", CT)


SOURCE = Path(
    "/home/lyt/.codex/attachments/e8986c85-ea2f-4526-8540-06894421065f/"
    "overall_framework_revised.vsdx"
)
OUT = Path("/home/lyt/mymodel/Mamba_Trackers/docs/overall_framework_real_data.vsdx")
ASSET_DIR = Path("/home/lyt/mymodel/Mamba_Trackers/docs/femot_real_data")

SEQUENCE = "dvSave-2024_06_04_18_07_37"
DATA_ROOT = Path("/rydata/wangx_dataset/_dataset/FEMOT/test") / SEQUENCE
RGB_SOURCE = DATA_ROOT / f"{SEQUENCE}_aps/00001.jpg"
EVENT_SOURCE = DATA_ROOT / f"{SEQUENCE}_dvs/00001.jpg"
DET_SOURCE = Path(
    "/home/lyt/mymodel/Mamba_Trackers/det_results/femot_dual/vis_test/"
) / SEQUENCE / "00001.jpg"


def _cell(shape: ET.Element, name: str) -> ET.Element | None:
    for item in shape.findall(f"{{{V}}}Cell"):
        if item.attrib.get("N") == name:
            return item
    return None


def _number(shape: ET.Element, name: str) -> float:
    item = _cell(shape, name)
    if item is None:
        raise ValueError(f"shape {shape.attrib.get('ID')} has no {name} cell")
    return float(item.attrib["V"])


def _bounds(shape: ET.Element) -> tuple[float, float, float, float]:
    x, y = _number(shape, "PinX"), _number(shape, "PinY")
    w, h = _number(shape, "Width"), _number(shape, "Height")
    return x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0


def _union(shapes: Iterable[ET.Element]) -> tuple[float, float, float, float]:
    boxes = [_bounds(shape) for shape in shapes]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _shape_by_id(root: ET.Element, shape_id: str) -> ET.Element:
    shape = root.find(f".//{{{V}}}Shape[@ID='{shape_id}']")
    if shape is None:
        raise ValueError(f"source VSDX is missing shape {shape_id}")
    return shape


def _hide(shape: ET.Element) -> None:
    """Make a placeholder invisible while retaining it for editability."""
    for name in ("FillPattern", "LinePattern"):
        item = _cell(shape, name)
        if item is not None:
            item.set("V", "0")


def _rectangle_geometry(shape: ET.Element) -> None:
    section = ET.SubElement(shape, f"{{{V}}}Section", {"N": "Geometry", "IX": "0"})
    ET.SubElement(section, f"{{{V}}}Cell", {"N": "NoFill", "V": "1"})
    ET.SubElement(section, f"{{{V}}}Cell", {"N": "NoLine", "V": "0"})
    ET.SubElement(section, f"{{{V}}}Cell", {"N": "NoShow", "V": "0"})
    ET.SubElement(section, f"{{{V}}}Cell", {"N": "NoSnap", "V": "0"})
    ET.SubElement(section, f"{{{V}}}Cell", {"N": "NoQuickDrag", "V": "0"})
    for ix, (row_type, x, y) in enumerate(
        (
            ("RelMoveTo", "0", "0"),
            ("RelLineTo", "1", "0"),
            ("RelLineTo", "1", "1"),
            ("RelLineTo", "0", "1"),
            ("RelLineTo", "0", "0"),
        ),
        1,
    ):
        row = ET.SubElement(section, f"{{{V}}}Row", {"T": row_type, "IX": str(ix)})
        ET.SubElement(row, f"{{{V}}}Cell", {"N": "X", "V": x})
        ET.SubElement(row, f"{{{V}}}Cell", {"N": "Y", "V": y})


def _foreign_shape(
    shape_id: int,
    bounds: tuple[float, float, float, float],
    relationship_id: str,
) -> ET.Element:
    left, bottom, right, top = bounds
    width, height = right - left, top - bottom
    shape = ET.Element(
        f"{{{V}}}Shape",
        {
            "ID": str(shape_id),
            "Type": "Foreign",
            "LineStyle": "0",
            "FillStyle": "0",
            "TextStyle": "0",
            "UniqueID": "{" + str(uuid.uuid4()).upper() + "}",
        },
    )
    pin_x, pin_y = (left + right) / 2.0, (bottom + top) / 2.0
    values = (
        ("PinX", pin_x),
        ("PinY", pin_y),
        ("Width", width),
        ("Height", height),
        ("LocPinX", width / 2.0),
        ("LocPinY", height / 2.0),
        ("Angle", 0),
        ("FlipX", 0),
        ("FlipY", 0),
        ("ResizeMode", 0),
        ("ImgOffsetX", 0),
        ("ImgOffsetY", 0),
        ("ImgWidth", width),
        ("ImgHeight", height),
        ("ClippingPath", ""),
        ("TxtPinX", width / 2.0),
        ("TxtPinY", 0),
        ("TxtWidth", width),
        ("TxtHeight", 0),
        ("TxtLocPinX", width / 2.0),
        ("TxtLocPinY", 0),
        ("TxtAngle", 0),
        ("VerticalAlign", 0),
        ("Relationships", 0),
        ("ObjType", 1),
        ("ShapeShdwShow", 2),
    )
    for name, value in values:
        cell = ET.SubElement(shape, f"{{{V}}}Cell", {"N": name, "V": str(value)})
        if name == "LocPinX":
            cell.set("F", "Width*0.5")
        elif name == "LocPinY":
            cell.set("F", "Height*0.5")
        elif name == "ImgWidth":
            cell.set("F", "Width*1")
        elif name == "ImgHeight":
            cell.set("F", "Height*1")
        elif name == "TxtPinX":
            cell.set("F", "Width*0.5")
        elif name == "TxtWidth":
            cell.set("F", "Width*1")
        elif name == "TxtLocPinX":
            cell.set("F", "TxtWidth*0.5")
        elif name == "TxtLocPinY":
            cell.set("F", "TxtHeight*0.5")
        elif name == "Relationships":
            cell.set("F", "SUM(DEPENDSON(4))")
    _rectangle_geometry(shape)
    foreign = ET.SubElement(
        shape,
        f"{{{V}}}ForeignData",
        {"ForeignType": "Bitmap", "CompressionType": "JPEG"},
    )
    ET.SubElement(foreign, f"{{{V}}}Rel", {f"{{{R}}}id": relationship_id})
    return shape


def _prepare_image(source: Path, target: Path, aspect: float) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    image = Image.open(source).convert("RGB")
    width, height = image.size
    wanted_height = max(1, round(width / aspect))
    if wanted_height < height:
        top = (height - wanted_height) // 2
        image = image.crop((0, top, width, top + wanted_height))
    elif wanted_height > height:
        wanted_width = max(1, round(height * aspect))
        left = max(0, (width - wanted_width) // 2)
        image = image.crop((left, 0, left + min(width, wanted_width), height))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=95, optimize=True)


def _next_relationship_id(root: ET.Element) -> int:
    ids = []
    for rel in root.findall(f"{{{REL}}}Relationship"):
        match = re.fullmatch(r"rId(\d+)", rel.attrib.get("Id", ""))
        if match:
            ids.append(int(match.group(1)))
    return max(ids, default=0) + 1


def build() -> Path:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rgb_asset = ASSET_DIR / "femot_rgb_00001.jpg"
    event_asset = ASSET_DIR / "femot_event_00001.jpg"
    det_asset = ASSET_DIR / "femot_detections_00001.jpg"
    _prepare_image(RGB_SOURCE, rgb_asset, aspect=1.5)
    _prepare_image(EVENT_SOURCE, event_asset, aspect=1.5)
    _prepare_image(DET_SOURCE, det_asset, aspect=1.55)

    with zipfile.ZipFile(SOURCE, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}

    page_root = ET.fromstring(members["visio/pages/page1.xml"])
    shapes_root = page_root.find(f"{{{V}}}Shapes")
    if shapes_root is None:
        raise ValueError("source VSDX page has no Shapes element")

    # Keep the original frame outlines, but remove the abstract inner icons and
    # detection bars that would otherwise compete with the photographs.
    for shape_id in ("7", "8", "9", "10", "13", "14", "15", "16", "90", "91", "92", "93"):
        _hide(_shape_by_id(page_root, shape_id))

    rgb_bounds = _bounds(_shape_by_id(page_root, "7"))
    event_bounds = _bounds(_shape_by_id(page_root, "13"))
    det_bounds = _union(_shape_by_id(page_root, str(i)) for i in (90, 91, 92, 93))
    max_id = max(int(shape.attrib.get("ID", "0")) for shape in shapes_root.findall(f"{{{V}}}Shape"))

    rels_name = "visio/pages/_rels/page1.xml.rels"
    if rels_name in members:
        rels_root = ET.fromstring(members[rels_name])
    else:
        rels_root = ET.Element(f"{{{REL}}}Relationships")
    next_rel = _next_relationship_id(rels_root)

    specs = (
        (rgb_asset, rgb_bounds),
        (event_asset, event_bounds),
        (det_asset, det_bounds),
    )
    for offset, (asset, bounds) in enumerate(specs, 1):
        rel_id = f"rId{next_rel + offset - 1}"
        target = f"../media/{asset.name}"
        ET.SubElement(
            rels_root,
            f"{{{REL}}}Relationship",
            {
                "Id": rel_id,
                "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                "Target": target,
            },
        )
        shapes_root.append(_foreign_shape(max_id + offset, bounds, rel_id))
        members[f"visio/media/{asset.name}"] = asset.read_bytes()

    types_root = ET.fromstring(members["[Content_Types].xml"])
    if not any(
        node.tag == f"{{{CT}}}Default" and node.attrib.get("Extension", "").lower() == "jpg"
        for node in types_root
    ):
        ET.SubElement(types_root, f"{{{CT}}}Default", {"Extension": "jpg", "ContentType": "image/jpeg"})

    members["visio/pages/page1.xml"] = ET.tostring(page_root, encoding="utf-8", xml_declaration=True)
    members[rels_name] = ET.tostring(rels_root, encoding="utf-8", xml_declaration=True)
    members["[Content_Types].xml"] = ET.tostring(types_root, encoding="utf-8", xml_declaration=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return OUT


if __name__ == "__main__":
    result = build()
    print(result)
