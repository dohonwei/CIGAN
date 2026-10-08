"""Replot R1 Figure 8 with a fixed perturbation seed and larger typography."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]
SOURCE = PROJECT_ROOT / "generated" / "deap" / "CIGAN_generated.npz"
OUTPUT_DIR = HERE.parents[0] / "outputs" / "r1_original_layout_pdf"
OUTPUT_PNG = OUTPUT_DIR / "Fig8.png"
OUTPUT_PDF = OUTPUT_DIR / "Fig8.pdf"
OUTPUT_CSV = OUTPUT_DIR / "Fig8_fixed_seed_metrics.csv"
SEED = 42
N_TRIALS = 50
SNR_LEVELS = (30, 20, 10, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pearson_per_trial(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    reference_centered = reference - reference.mean(axis=1, keepdims=True)
    estimate_centered = estimate - estimate.mean(axis=1, keepdims=True)
    numerator = np.sum(reference_centered * estimate_centered, axis=1)
    denominator = np.sqrt(
        np.sum(reference_centered**2, axis=1) * np.sum(estimate_centered**2, axis=1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(reference.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def evaluate(reference: np.ndarray, reconstruction: np.ndarray) -> list[dict[str, float]]:
    rng = np.random.RandomState(SEED)
    records = []
    for snr_db in SNR_LEVELS:
        signal_power = np.mean(reference**2, axis=1, keepdims=True)
        noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10.0)))
        perturbed = reconstruction + rng.normal(size=reconstruction.shape) * noise_std
        residual = reference - perturbed
        records.append(
            {
                "snr_db": float(snr_db),
                "pearson": float(np.nanmean(pearson_per_trial(reference, perturbed))),
                "rmse_uv": float(np.mean(np.sqrt(np.mean(residual**2, axis=1)))),
                "mae_uv": float(np.mean(np.abs(residual), axis=1).mean()),
            }
        )
    return records


def save_csv(records: list[dict[str, float]]) -> None:
    header = "snr_db,pearson,rmse_uv,mae_uv\n"
    rows = [
        f"{row['snr_db']:.0f},{row['pearson']:.9f},{row['rmse_uv']:.9f},{row['mae_uv']:.9f}"
        for row in records
    ]
    OUTPUT_CSV.write_text(header + "\n".join(rows) + "\n", encoding="ascii")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE, allow_pickle=False) as archive:
        reference = np.asarray(archive["real_fp"])[:N_TRIALS, 0, :].astype(np.float64)
        reconstruction = np.asarray(archive["fake_fp"])[:N_TRIALS, 0, :].astype(np.float64)

    records = evaluate(reference, reconstruction)
    save_csv(records)
    ordered = sorted(records, key=lambda row: row["snr_db"])
    snr = np.asarray([row["snr_db"] for row in ordered])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    panels = (
        ("pearson", "Reconstruction Correlation vs Perturbation SNR\n(Fp1)", "Pearson correlation", "blue", "o"),
        ("rmse_uv", "Reconstruction RMSE vs Perturbation SNR\n(Fp1)", "RMSE (\xb5V)", "red", "s"),
        ("mae_uv", "Reconstruction MAE vs Perturbation SNR\n(Fp1)", "MAE (\xb5V)", "green", "^"),
    )
    for axis, (key, title, ylabel, color, marker) in zip(axes, panels):
        values = [row[key] for row in ordered]
        axis.plot(snr, values, marker=marker, markersize=8, linewidth=2.5, color=color)
        axis.set_title(title)
        axis.set_xlabel("Perturbation SNR (dB)")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(True, alpha=0.3)
        axis.set_xticks((5, 10, 15, 20, 25, 30))
    axes[0].set_ylim(0, 1)

    figure.tight_layout(pad=1.0, w_pad=1.4)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)

    manifest = {
        "source": str(SOURCE.resolve()),
        "source_sha256": sha256(SOURCE),
        "output_png": str(OUTPUT_PNG.resolve()),
        "output_pdf": str(OUTPUT_PDF.resolve()),
        "output_metrics": str(OUTPUT_CSV.resolve()),
        "layout": "original 1x3 panel arrangement",
        "seed": SEED,
        "n_trials": N_TRIALS,
        "channel": "Fp1",
        "protocol": "Gaussian perturbation added to reconstructed Fp1 before comparison with reference Fp1",
        "records": records,
    }
    manifest_path = OUTPUT_DIR / "Fig8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))
    print(OUTPUT_PDF.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
