"""Shared paths, styles, and provenance helpers for E8 figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

E8_ROOT = Path(__file__).resolve().parents[0]
REVISION_ROOT = E8_ROOT.parent
DEFAULT_OUTPUT = E8_ROOT / "outputs"
PALETTE = {"blue": "#0077BB", "cyan": "#33BBEE", "teal": "#009988",
           "orange": "#EE7733", "red": "#CC3311", "magenta": "#EE3377",
           "grey": "#999999", "black": "#000000"}
COLORS = [PALETTE[k] for k in ("blue", "cyan", "teal", "orange", "red", "magenta", "grey", "black")]


def configure_matplotlib() -> None:
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def save_figure(fig, output_dir: Path, figure_id: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = output_dir / f"{figure_id}.png", output_dir / f"{figure_id}.pdf"
    fig.savefig(png, dpi=300, facecolor="white"); fig.savefig(pdf, facecolor="white")
    return png, pdf


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
