"""Replot R1 Figure 7 with larger typography and unchanged source data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import probplot


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
SOURCE = PROJECT_ROOT / "generated" / "waste" / "deap" / "generated_data_loso.npz"
OUTPUT_DIR = HERE.parents[0] / "outputs" / "r1_original_layout_pdf"
OUTPUT_PNG = OUTPUT_DIR / "Fig7BlandAltmanDEAP.png"
OUTPUT_PDF = OUTPUT_DIR / "Fig7BlandAltmanDEAP.pdf"
N_TRIALS = 50
SEED = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE, allow_pickle=False) as archive:
        reference = np.asarray(archive["real_fp"])[:N_TRIALS, 0, :].reshape(-1)
        reconstruction = np.asarray(archive["fake_fp"])[:N_TRIALS, 0, :].reshape(-1)

    means = (reference + reconstruction) / 2
    differences = reference - reconstruction
    bias = float(np.mean(differences))
    std = float(np.std(differences))
    lower = bias - 1.96 * std
    upper = bias + 1.96 * std

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
    )
    figure, (ba_axis, qq_axis) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [2, 1]},
    )

    density = ba_axis.hexbin(means, differences, gridsize=50, cmap="Blues", mincnt=1)
    colorbar = figure.colorbar(density, ax=ba_axis)
    colorbar.set_label("Point Density", fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    ba_axis.axhline(bias, color="black", linewidth=2, label=f"Bias: {bias:.2f}")
    ba_axis.axhline(upper, color="red", linestyle="--", linewidth=2, label=f"+1.96SD: {upper:.2f}")
    ba_axis.axhline(lower, color="red", linestyle="--", linewidth=2, label=f"-1.96SD: {lower:.2f}")
    ba_axis.set_title("Bland-Altman Plot (Fp1)")
    ba_axis.set_xlabel("Mean of reference and reconstruction signal (\xb5V)")
    ba_axis.set_ylabel("Difference: reference - reconstruction (\xb5V)")
    ba_axis.tick_params(axis="both", labelsize=14)
    ba_axis.legend(loc="upper right")
    ba_axis.grid(True, alpha=0.3)

    rng = np.random.default_rng(SEED)
    sample_size = min(10_000, differences.size)
    sample = differences[rng.choice(differences.size, size=sample_size, replace=False)]
    probplot(sample, dist="norm", plot=qq_axis)
    qq_axis.set_title("Q-Q Plot of Residuals")
    qq_axis.set_xlabel("Theoretical quantiles")
    qq_axis.set_ylabel("Ordered values")
    qq_axis.tick_params(axis="both", labelsize=14)
    qq_axis.grid(True, alpha=0.3)

    figure.tight_layout(pad=1.0, w_pad=1.4)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    manifest = {
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "output_png": str(OUTPUT_PNG.resolve()),
        "output_pdf": str(OUTPUT_PDF.resolve()),
        "layout": "original 1x2 panel arrangement with 2:1 width ratio",
        "data_changed": False,
        "n_trials": N_TRIALS,
        "qq_sampling_seed": SEED,
        "difference_direction": "reference minus reconstruction",
        "bias": bias,
        "loa_lower": lower,
        "loa_upper": upper,
    }
    manifest_path = OUTPUT_DIR / "Fig7BlandAltmanDEAP_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
