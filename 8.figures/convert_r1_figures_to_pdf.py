"""Convert the four reviewer-flagged R1 PNG figures to PDFs unchanged."""
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
DEFAULT_OUTPUT = HERE.parents[0] / "outputs" / "r1_original_layout_pdf"
FIGURES = (
    "Fig5.png",
    "Fig7BlandAltmanDEAP.png",
    "Fig8.png",
    "Fig10.png",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, output_dir: Path, dpi: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{source.stem}.pdf"

    with Image.open(source) as opened:
        image = opened.convert("RGB")
        width_px, height_px = image.size
        width_in = width_px / dpi
        height_in = height_px / dpi

        figure = plt.figure(figsize=(width_in, height_in), dpi=dpi)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.imshow(np.asarray(image), interpolation="none", aspect="auto")
        axes.set_axis_off()
        figure.savefig(output, format="pdf", bbox_inches=None, pad_inches=0, dpi=dpi)
        plt.close(figure)

    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "source_pixels": [width_px, height_px],
        "output": str(output.resolve()),
        "page_points": [width_in * 72.0, height_in * 72.0],
        "dpi": dpi,
        "layout_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    records = []
    for filename in FIGURES:
        source = args.source_dir / filename
        if not source.exists():
            raise FileNotFoundError(source)
        record = convert(source, args.output_dir, args.dpi)
        records.append(record)
        print(f"{source.name} -> {record['output']}")

    manifest = args.output_dir / "conversion_manifest.json"
    manifest.write_text(
        json.dumps({"figures": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Manifest: {manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
