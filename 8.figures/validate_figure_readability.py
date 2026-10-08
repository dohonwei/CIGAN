"""Validate E8 figure files, dimensions, provenance, and readability metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from e8_common import DEFAULT_OUTPUT, sha256, write_json


def validate_item(item: dict) -> dict:
    checks, errors = {}, []
    png, pdf = Path(item["png"]), Path(item["pdf"])
    checks["png_exists"] = png.exists() and png.stat().st_size > 10_000
    checks["pdf_exists"] = pdf.exists() and pdf.stat().st_size > 1_000
    if not checks["png_exists"]: errors.append("PNG missing or unexpectedly small")
    if not checks["pdf_exists"]: errors.append("PDF missing or unexpectedly small")
    if checks["png_exists"]:
        with Image.open(png) as image:
            width_px, height_px = image.size; dpi = image.info.get("dpi", (0, 0))
            checks.update({"width_px": width_px, "height_px": height_px,
                           "dpi_x": float(dpi[0]), "dpi_y": float(dpi[1])})
            checks["minimum_300_dpi"] = min(dpi) >= 299
            checks["publication_width"] = width_px >= float(item["width_in"]) * float(item["dpi"]) * .94
            if not checks["minimum_300_dpi"]: errors.append(f"DPI below 300: {dpi}")
            if not checks["publication_width"]: errors.append(f"PNG width too small: {width_px}px")
    checks["minimum_font_8pt"] = float(item.get("minimum_font_pt", 0)) >= 8
    checks["axis_labels_declared"] = len(item.get("axis_labels", [])) >= 2 and all(item.get("axis_labels", []))
    checks["caption_present"] = bool(item.get("caption", "").strip())
    checks["note_present"] = bool(item.get("note", "").strip())
    checks["legend_contract"] = isinstance(item.get("legend_required"), bool)
    for key in ("minimum_font_8pt", "axis_labels_declared", "caption_present", "note_present", "legend_contract"):
        if not checks[key]: errors.append(key.replace("_", " ") + " failed")
    source_checks = []
    for source in item.get("sources", []):
        path = Path(source["path"]); ok = path.exists() and sha256(path) == source["sha256"]
        source_checks.append({"path": str(path), "hash_matches": ok})
        if not ok: errors.append(f"Source missing or changed: {path}")
    checks["sources"] = source_checks
    return {"figure_id": item["figure_id"], "status": "PASS" if not errors else "FAIL", "checks": checks, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT / "figure_manifest.json")
    parser.add_argument("--report", type=Path, default=DEFAULT_OUTPUT / "readability_report.json"); args = parser.parse_args()
    if not args.manifest.exists(): raise FileNotFoundError(f"Generate E8 figures first: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")); results = [validate_item(x) for x in manifest.get("figures", [])]
    if not results: raise ValueError("Manifest contains no figures")
    report = {"status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL", "n_figures": len(results), "results": results}
    write_json(args.report, report)
    for result in results:
        print(f"{result['figure_id']}: {result['status']}")
        for error in result["errors"]: print(f"  - {error}")
    print(f"Overall: {report['status']}; report: {args.report}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
