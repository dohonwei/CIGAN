"""Re-layout the four R1 figures flagged for readability without changing data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "paper" / "CIGAN_TIM_R1"
DEFAULT_OUTPUT = HERE.parents[0] / "outputs" / "r1_readability_only"


SPECS = {
    "Fig5": {
        "filename": "Fig5.png",
        "size": (2160, 1920),
        "layout": "grid",
        "crops": (
            (130, 20, 940, 880),
            (1180, 20, 2000, 880),
            (130, 910, 940, 1870),
            (1180, 910, 2000, 1870),
        ),
        "gap": 34,
    },
    "Fig7": {
        "filename": "Fig7BlandAltmanDEAP.png",
        "size": (4770, 1765),
        "layout": "stacked",
        "crops": ((0, 0, 3010, 1765), (3060, 0, 4770, 1765)),
        "gap": 45,
    },
    "Fig8": {
        "filename": "Fig8.png",
        "size": (5355, 1755),
        "layout": "two_plus_one",
        "crops": ((0, 0, 1775, 1755), (1780, 0, 3570, 1755), (3575, 0, 5355, 1755)),
        "gap": 34,
    },
    "Fig10": {
        "filename": "Fig10.png",
        "size": (3568, 2966),
        "layout": "grid",
        "crops": ((0, 0, 1784, 1483), (1784, 0, 3568, 1483),
                  (0, 1483, 1784, 2966), (1784, 1483, 3568, 2966)),
        "gap": 12,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(box).convert("RGB")


def white_canvas(width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def grid_layout(parts: list[Image.Image], gap: int) -> Image.Image:
    if len(parts) != 4:
        raise ValueError("A 2x2 grid requires four panels")
    left = max(parts[0].width, parts[2].width)
    right = max(parts[1].width, parts[3].width)
    top = max(parts[0].height, parts[1].height)
    bottom = max(parts[2].height, parts[3].height)
    canvas = white_canvas(left + gap + right, top + gap + bottom)
    positions = ((0, 0), (left + gap, 0), (0, top + gap), (left + gap, top + gap))
    for panel, (x, y) in zip(parts, positions):
        canvas.paste(panel, (x, y))
    return canvas


def stacked_layout(parts: list[Image.Image], gap: int) -> Image.Image:
    width = max(panel.width for panel in parts)
    height = sum(panel.height for panel in parts) + gap * (len(parts) - 1)
    canvas = white_canvas(width, height)
    y = 0
    for panel in parts:
        x = (width - panel.width) // 2
        canvas.paste(panel, (x, y))
        y += panel.height + gap
    return canvas


def two_plus_one_layout(parts: list[Image.Image], gap: int) -> Image.Image:
    if len(parts) != 3:
        raise ValueError("A 2+1 layout requires three panels")
    top_width = parts[0].width + gap + parts[1].width
    top_height = max(parts[0].height, parts[1].height)
    width = max(top_width, parts[2].width)
    canvas = white_canvas(width, top_height + gap + parts[2].height)
    top_x = (width - top_width) // 2
    canvas.paste(parts[0], (top_x, 0))
    canvas.paste(parts[1], (top_x + parts[0].width + gap, 0))
    canvas.paste(parts[2], ((width - parts[2].width) // 2, top_height + gap))
    return canvas


def save_package(image: Image.Image, stem: str, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}_readable.png"
    pdf = output_dir / f"{stem}_readable.pdf"
    image.save(png, dpi=(300, 300), optimize=True)

    width_in = 7.2
    height_in = width_in * image.height / image.width
    fig = plt.figure(figsize=(width_in, height_in), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(np.asarray(image), interpolation="none")
    ax.axis("off")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    return png, pdf


def build(figure_id: str, source_dir: Path, output_dir: Path) -> dict:
    spec = SPECS[figure_id]
    source = source_dir / spec["filename"]
    if not source.exists():
        raise FileNotFoundError(source)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    if image.size != spec["size"]:
        raise ValueError(f"Unexpected size for {source}: {image.size}, expected {spec['size']}")

    parts = [crop(image, box) for box in spec["crops"]]
    if spec["layout"] == "grid":
        composed = grid_layout(parts, spec["gap"])
    elif spec["layout"] == "stacked":
        composed = stacked_layout(parts, spec["gap"])
    elif spec["layout"] == "two_plus_one":
        composed = two_plus_one_layout(parts, spec["gap"])
    else:
        raise ValueError(spec["layout"])

    png, pdf = save_package(composed, figure_id, output_dir)
    return {
        "figure_id": figure_id,
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "source_pixels": list(image.size),
        "output_pixels": list(composed.size),
        "layout_only": True,
        "png": str(png.resolve()),
        "pdf": str(pdf.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", nargs="+", choices=tuple(SPECS), default=tuple(SPECS))
    args = parser.parse_args()

    records = [build(figure_id, args.source_dir, args.output_dir) for figure_id in args.figures]
    manifest = args.output_dir / "readability_manifest.json"
    manifest.write_text(json.dumps({"figures": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in records:
        print(f"{record['figure_id']}: {record['pdf']}")
    print(f"Manifest: {manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
