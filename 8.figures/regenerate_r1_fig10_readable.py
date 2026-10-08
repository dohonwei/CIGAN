"""Replot R1 Figure 10 with larger typography and unchanged source data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
SOURCE = PROJECT_ROOT / "generated" / "waste" / "deap" / "generated_data_loso.npz"
OUTPUT_DIR = HERE.parents[0] / "outputs" / "r1_original_layout_pdf"
OUTPUT_PNG = OUTPUT_DIR / "Fig10_axis_corrected.png"
OUTPUT_PDF = OUTPUT_DIR / "Fig10_axis_corrected.pdf"
FS = 128
BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def band_powers(signals: np.ndarray) -> dict[str, np.ndarray]:
    frequencies, spectrum = welch(
        signals,
        fs=FS,
        nperseg=min(256, signals.shape[-1]),
        axis=-1,
    )
    result = {}
    for name, (low, high) in BANDS.items():
        selected = (frequencies >= low) & (frequencies <= high)
        result[name] = np.trapz(spectrum[:, selected], frequencies[selected], axis=-1)
    return result


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE, allow_pickle=False) as archive:
        reference = np.asarray(archive["real_fp"])[:, 0, :]
        reconstruction = np.asarray(archive["fake_fp"])[:, 0, :]

    reference_power = band_powers(reference)
    reconstruction_power = band_powers(reconstruction)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    records = []
    for axis, (band, _) in zip(axes.flat, BANDS.items()):
        real_values = reference_power[band]
        fake_values = reconstruction_power[band]
        means = (real_values + fake_values) / 2
        differences = real_values - fake_values
        bias = float(np.mean(differences))
        std = float(np.std(differences))
        lower = bias - 1.96 * std
        upper = bias + 1.96 * std

        axis.scatter(means, differences, s=15, alpha=0.6, color="royalblue")
        axis.axhline(bias, color="black", linewidth=2, label=f"Bias: {bias:.2f}")
        axis.axhline(upper, color="red", linestyle="--", linewidth=2, label="+1.96SD")
        axis.axhline(lower, color="red", linestyle="--", linewidth=2, label="-1.96SD")
        axis.set_title(f"{band} Band")
        axis.set_xlabel("Mean band power (\xb5V\xb2)")
        axis.set_ylabel("Reference - reconstruction\nband power (\xb5V\xb2)")
        axis.tick_params(axis="both", labelsize=14)
        axis.legend(loc="upper right", frameon=True)
        axis.grid(True, alpha=0.3)
        records.append(
            {
                "band": band.lower(),
                "bias": bias,
                "loa_lower": lower,
                "loa_upper": upper,
                "n_points": int(differences.size),
            }
        )

    figure.tight_layout(pad=1.2, h_pad=1.4, w_pad=1.4)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    manifest = {
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "output_png": str(OUTPUT_PNG.resolve()),
        "output_pdf": str(OUTPUT_PDF.resolve()),
        "layout": "original 2x2 panel arrangement",
        "data_changed": False,
        "difference_direction": "reference minus reconstruction",
        "records": records,
    }
    manifest_path = OUTPUT_DIR / "Fig10_axis_corrected_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    print(OUTPUT_PDF.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
