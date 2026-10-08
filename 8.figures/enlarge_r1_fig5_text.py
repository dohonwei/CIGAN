"""Enlarge text in the original R1 Figure 5 without recomputing t-SNE."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
SOURCE = PROJECT_ROOT / "paper" / "CIGAN_TIM_R1" / "Fig5.png"
OUTPUT_DIR = HERE.parents[0] / "outputs" / "r1_original_layout_pdf"
OUTPUT_PNG = OUTPUT_DIR / "Fig5.png"
OUTPUT_PDF = OUTPUT_DIR / "Fig5.pdf"
EXPECTED_SIZE = (2160, 1920)


PANELS = (
    {
        "bounds": (264, 208, 904, 844),
        "title": "t-SNE (full)",
        "x_ticks": ((269, "-2"), (394, "-1"), (520, "0"), (646, "1"), (770, "2"), (896, "3")),
        "y_ticks": ((817, "-2"), (694, "-1"), (571, "0"), (448, "1"), (325, "2")),
    },
    {
        "bounds": (1275, 208, 1915, 844),
        "title": "t-SNE (no_causal)",
        "x_ticks": tuple(zip((1276, 1357, 1438, 1519, 1600, 1681, 1762, 1843),
                             ("-20", "-15", "-10", "-5", "0", "5", "10", "15"))),
        "y_ticks": tuple(zip((843, 768, 693, 617, 542, 466, 391, 315, 240),
                             ("-20", "-15", "-10", "-5", "0", "5", "10", "15", "20"))),
    },
    {
        "bounds": (264, 1143, 904, 1779),
        "title": "t-SNE (no_fen)",
        "x_ticks": tuple(zip((301, 394, 487, 579, 671, 764, 856),
                             ("-15", "-10", "-5", "0", "5", "10", "15"))),
        "y_ticks": tuple(zip((1740, 1647, 1555, 1462, 1370, 1279, 1186),
                             ("-15", "-10", "-5", "0", "5", "10", "15"))),
    },
    {
        "bounds": (1275, 1143, 1915, 1779),
        "title": "t-SNE (no_psd)",
        "x_ticks": tuple(zip((1319, 1390, 1462, 1533, 1604, 1675, 1747, 1818, 1890),
                             ("-20", "-15", "-10", "-5", "0", "5", "10", "15", "20"))),
        "y_ticks": tuple(zip((1742, 1602, 1462, 1323, 1183),
                             ("-20", "-10", "0", "10", "20"))),
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cover(axis, x: float, y: float, width: float, height: float) -> None:
    axis.add_patch(Rectangle((x, y), width, height, facecolor="white", edgecolor="none", zorder=2))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCE) as opened:
        image = opened.convert("RGB")
    if image.size != EXPECTED_SIZE:
        raise ValueError(f"Unexpected source size: {image.size}; expected {EXPECTED_SIZE}")

    width, height = image.size
    figure = plt.figure(figsize=(width / 300, height / 300), dpi=300)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(image, extent=(0, width, height, 0), interpolation="none", zorder=0)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")

    for panel in PANELS:
        left, top, right, bottom = panel["bounds"]

        # Clear only the original small text outside the plotting rectangle.
        cover(axis, left + 120, top - 47, right - left - 240, 43)
        cover(axis, left - 90, top - 4, 80, bottom - top + 22)
        cover(axis, left - 28, bottom + 8, right - left + 56, 48)
        cover(axis, right - 150, top + 7, 143, 91)

        axis.text((left + right) / 2, top - 24, panel["title"], ha="center", va="center",
                  fontsize=10, color="black", zorder=3)
        for x, label in panel["x_ticks"]:
            axis.text(x, bottom + 17, label, ha="center", va="top", fontsize=8.5,
                      color="black", zorder=3)
        for y, label in panel["y_ticks"]:
            axis.text(left - 17, y, label, ha="right", va="center", fontsize=8.5,
                      color="black", zorder=3)

        # Replace the compact raster legend with a readable vector legend.
        handles = (
            Line2D([], [], linestyle="none", marker="o", markersize=4.5,
                   markerfacecolor="#1f77b4", markeredgewidth=0, label="Real"),
            Line2D([], [], linestyle="none", marker="o", markersize=4.5,
                   markerfacecolor="#ff7f0e", markeredgewidth=0, label="Fake"),
        )
        legend = axis.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(right / width - 0.004, 1 - (top + 7) / height),
            bbox_transform=axis.transAxes,
            fontsize=8.5,
            frameon=True,
            framealpha=1.0,
            borderpad=0.35,
            handletextpad=0.35,
            labelspacing=0.25,
        )
        legend.set_zorder(4)
        axis.add_artist(legend)

    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
    figure.savefig(OUTPUT_PDF, format="pdf", dpi=300, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(figure)

    manifest = {
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "output_png": str(OUTPUT_PNG.resolve()),
        "output_pdf": str(OUTPUT_PDF.resolve()),
        "source_pixels": list(image.size),
        "layout_changed": False,
        "embedding_recomputed": False,
        "data_changed": False,
        "change": "Replaced subplot titles, tick labels, and legends with larger text.",
    }
    manifest_path = OUTPUT_DIR / "Fig5_text_overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_PDF.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
