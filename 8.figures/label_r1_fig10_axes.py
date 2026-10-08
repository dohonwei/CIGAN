"""Correct the axis labels in the original-layout R1 Figure 10."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
SOURCE = PROJECT_ROOT / "paper" / "CIGAN_TIM_R1" / "Fig10.png"
OUTPUT_DIR = HERE.parents[0] / "outputs" / "r1_original_layout_pdf"
OUTPUT_PNG = OUTPUT_DIR / "Fig10_axis_corrected.png"
OUTPUT_PDF = OUTPUT_DIR / "Fig10_axis_corrected.pdf"
EXPECTED_SIZE = (3568, 2966)
X_LABEL = "Mean band power (\u00b5V\u00b2)"
Y_LABEL = "Reference - reconstruction band power (\u00b5V\u00b2)"

# Each rectangle contains only an existing x-axis title in the original image.
LABEL_BOXES = (
    (390, 1400, 1395, 1468),
    (2174, 1400, 3179, 1468),
    (390, 2878, 1395, 2958),
    (2174, 2878, 3179, 2958),
)

Y_LABEL_BOXES = (
    (0, 210, 82, 1250),
    (1750, 210, 1855, 1250),
    (0, 1680, 82, 2730),
    (1750, 1680, 1855, 2730),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    raise FileNotFoundError("No suitable TrueType font found")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCE) as opened:
        image = opened.convert("RGB")
    if image.size != EXPECTED_SIZE:
        raise ValueError(f"Unexpected source size: {image.size}, expected {EXPECTED_SIZE}")

    draw = ImageDraw.Draw(image)
    x_font = find_font(46)
    for left, top, right, bottom in LABEL_BOXES:
        draw.rectangle((left, top, right, bottom), fill="white")
        draw.text(
            ((left + right) / 2, (top + bottom) / 2),
            X_LABEL,
            font=x_font,
            fill="black",
            anchor="mm",
        )

    y_font = find_font(40)
    for left, top, right, bottom in Y_LABEL_BOXES:
        draw.rectangle((left, top, right, bottom), fill="white")
        label_layer = Image.new("RGBA", (bottom - top, right - left), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_layer)
        label_draw.text(
            (label_layer.width / 2, label_layer.height / 2),
            Y_LABEL,
            font=y_font,
            fill="black",
            anchor="mm",
        )
        rotated = label_layer.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        x = int((left + right - rotated.width) / 2)
        y = int((top + bottom - rotated.height) / 2)
        image.paste(rotated, (x, y), rotated)

    image.save(OUTPUT_PNG, dpi=(300, 300), optimize=True)
    width_in = image.width / 300
    height_in = image.height / 300
    figure = plt.figure(figsize=(width_in, height_in), dpi=300)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.imshow(np.asarray(image), interpolation="none", aspect="auto")
    axes.set_axis_off()
    figure.savefig(OUTPUT_PDF, format="pdf", bbox_inches=None, pad_inches=0, dpi=300)
    plt.close(figure)

    manifest = {
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "output_png": str(OUTPUT_PNG.resolve()),
        "output_pdf": str(OUTPUT_PDF.resolve()),
        "pixels": list(image.size),
        "layout_changed": False,
        "data_changed": False,
        "change": {
            "x_axis": f"Replaced four x-axis titles with: {X_LABEL}",
            "y_axis": f"Replaced four y-axis titles with: {Y_LABEL}",
        },
    }
    manifest_path = OUTPUT_DIR / "Fig10_axis_corrected_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_PDF.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
