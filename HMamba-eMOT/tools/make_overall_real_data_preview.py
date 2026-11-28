"""Make a quick raster preview of the real-data VSDX variant.

The uploaded document is kept as native Visio XML.  In environments without
Visio/LibreOffice, this preview uses the matching editable framework raster as
the background and composites the same embedded photographs at their page
coordinates.
"""

from pathlib import Path
from PIL import Image, ImageOps


ROOT = Path("/home/lyt/mymodel/Mamba_Trackers")
BACKGROUND = ROOT / "docs/overall_framework_revised.png"
OUT = ROOT / "docs/overall_framework_real_data.png"
SEQUENCE = "dvSave-2024_06_04_18_07_37"
FRAME = "00044"
DATA_ROOT = Path("/rydata/wangx_dataset/_dataset/FEMOT/test") / SEQUENCE
RGB_SOURCE = DATA_ROOT / f"{SEQUENCE}_aps/{FRAME}.jpg"
EVENT_SOURCE = DATA_ROOT / f"{SEQUENCE}_dvs/{FRAME}.jpg"
DET_SOURCE = ROOT / "det_results/femot_dual/vis_test" / SEQUENCE / f"{FRAME}.jpg"


def paste_cover(canvas: Image.Image, source: Path, box: tuple[int, int, int, int]) -> None:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    image = Image.open(source).convert("RGB")
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    canvas.paste(image, (left, top))


def main() -> None:
    canvas = Image.open(BACKGROUND).convert("RGB")
    # The editable page uses 144 px per Visio drawing unit and a 13.5-unit
    # bottom-origin page in the matching raster export.
    scale = 144.0
    page_height_px = 1944.0

    def box(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
        return (
            round((x - w / 2.0) * scale),
            round(page_height_px - (y + h / 2.0) * scale),
            round((x + w / 2.0) * scale),
            round(page_height_px - (y - h / 2.0) * scale),
        )

    paste_cover(canvas, RGB_SOURCE, box(1.77777777778, 10.3611111111, 1.0, 0.666666666667))
    paste_cover(canvas, EVENT_SOURCE, box(1.77777777778, 8.79166666667, 1.0, 0.666666666667))
    paste_cover(canvas, DET_SOURCE, box(20.8125, 10.0555555556, 1.291666666667, 0.833333333333))
    canvas.save(OUT, format="PNG", optimize=True)
    print(OUT)


if __name__ == "__main__":
    main()
