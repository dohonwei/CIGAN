"""Regenerate manuscript Figures 5, 7, 8, 9, and 10 for Reviewer 6.4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from e8_common import COLORS, DEFAULT_OUTPUT, REVISION_ROOT, configure_matplotlib, save_figure, sha256, write_json

MODEL_LABELS = {"cigan": "CIGAN", "hveegnet": "hvEEGNet", "tieeegnet": "TIE-EEGNet",
                "wavenet": "WaveNet", "encoder_decoder": "Encoder-decoder"}

REVIEWER_FONT_PT = 10


def enlarge_axis_text(ax, *, label_size=11, tick_size=10, title_size=12, legend_size=10):
    """Keep labels readable after the figure is scaled in the two-column layout."""
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.title.set_size(title_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)


def load_folds(pattern: Path, folds=range(1, 6)) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    fake, real, paths = [], [], []
    for fold in folds:
        path = Path(str(pattern).format(fold=fold))
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as z:
            fake.append(z["fake_uv"]); real.append(z["real_uv"])
        paths.append(path)
    return np.concatenate(fake), np.concatenate(real), paths


def provenance(paths: list[Path]) -> list[dict]:
    return [{"path": str(p.resolve()), "sha256": sha256(p)} for p in dict.fromkeys(paths)]


def entry(fid, png, pdf, width, labels, caption, note, sources, legend=True):
    return {"figure_id": fid, "png": str(png.resolve()), "pdf": str(pdf.resolve()),
            "width_in": width, "dpi": 300, "minimum_font_pt": REVIEWER_FONT_PT, "axis_labels": labels,
            "caption": caption, "note": note, "legend_required": legend, "sources": provenance(sources)}


def compact_features(x: np.ndarray, fs: float) -> np.ndarray:
    # Trial-level temporal and spectral descriptors retain both frontal channels.
    feats = []
    for ch in range(x.shape[1]):
        y = x[:, ch]
        basic = np.column_stack((y.mean(1), y.std(1), np.quantile(y, .1, axis=1),
                                 np.quantile(y, .5, axis=1), np.quantile(y, .9, axis=1)))
        f, p = welch(y, fs=fs, axis=1, nperseg=min(256, y.shape[1]))
        bands = []
        for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, min(45, fs / 2))):
            mask = (f >= lo) & (f < hi)
            bands.append(np.trapz(p[:, mask], f[mask], axis=1))
        feats.append(np.column_stack((basic, *bands)))
    return np.concatenate(feats, axis=1)


def figure5(out: Path, seed: int):
    base = REVISION_ROOT / "1.prior_benchmark/predictions/hci/fold_{fold}/granger_seed_42_fixed_prior_welch_e100.npz"
    full, real, paths = load_folds(base)
    variants = {"Full model": full}
    for key, label in (("no_causal", "No causality"), ("no_fen", "No FEN"), ("no_psd", "No PSD loss")):
        pat = REVISION_ROOT / f"3.ablation/predictions/hci/fold_{{fold}}/{key}_seed_42_e3_welch_e100.npz"
        fake, check_real, src = load_folds(pat)
        if fake.shape != full.shape or not np.allclose(check_real, real, atol=1e-5):
            raise ValueError(f"HCI fold mapping mismatch for {key}")
        variants[label] = fake; paths += src
    blocks = [compact_features(real, 256)] + [compact_features(v, 256) for v in variants.values()]
    matrix = np.concatenate(blocks); matrix = (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-8)
    pca = PCA(n_components=min(20, matrix.shape[1]), random_state=seed).fit_transform(matrix)
    emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", random_state=seed).fit_transform(pca)
    n = len(real); real_xy = emb[:n]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), sharex=True, sharey=True)
    for i, (ax, (label, _)) in enumerate(zip(axes.flat, variants.items())):
        fake_xy = emb[n * (i + 1):n * (i + 2)]
        ax.scatter(real_xy[:, 0], real_xy[:, 1], s=7, alpha=.35, color=COLORS[0], label="Reference", rasterized=True)
        ax.scatter(fake_xy[:, 0], fake_xy[:, 1], s=7, alpha=.35, color=COLORS[3], label="Reconstruction", rasterized=True)
        ax.set_title(label); ax.set_xlabel("Shared t-SNE dimension 1"); ax.set_ylabel("Shared t-SNE dimension 2")
        ax.legend(frameon=False, markerscale=1.5)
        enlarge_axis_text(ax)
    fig.tight_layout(); png, pdf = save_figure(fig, out, "Fig5_revised"); plt.close(fig)
    return entry("Fig5_revised", png, pdf, 7.2, ["Shared t-SNE dimension 1", "Shared t-SNE dimension 2"],
                 "Shared-embedding visualization of reference and reconstructed HCI frontal EEG for the full and ablated models.",
                 "One PCA/t-SNE embedding was fitted jointly to all conditions; the plot is descriptive and is not a statistical test.", paths)


def deap_full():
    pat = REVISION_ROOT / "1.prior_benchmark/predictions/deap/fold_{fold}/granger_seed_42_fixed_prior_welch_e100.npz"
    return load_folds(pat)


def figure7(out: Path, seed: int):
    fake, real, paths = deap_full(); rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    for ch, ax in enumerate(axes):
        r, f = real[:, ch].ravel(), fake[:, ch].ravel(); d = f - r; m = (f + r) / 2
        bias, sd = float(d.mean()), float(d.std(ddof=1)); loa = (bias - 1.96 * sd, bias + 1.96 * sd)
        take = rng.choice(len(m), min(150000, len(m)), replace=False)
        ax.hexbin(m[take], d[take], gridsize=75, mincnt=1, bins="log", cmap="viridis", rasterized=True)
        ax.axhline(bias, color=COLORS[4], lw=1.2, label=f"Bias = {bias:.2f}")
        ax.axhline(loa[0], color="black", ls="--", lw=1, label=f"95% LoA = [{loa[0]:.2f}, {loa[1]:.2f}]")
        ax.axhline(loa[1], color="black", ls="--", lw=1)
        ax.set_title(("Fp1", "Fp2")[ch]); ax.set_xlabel(r"Paired mean amplitude ($\mu$V)")
        ax.set_ylabel(r"Reconstruction - reference ($\mu$V)"); ax.legend(frameon=False, loc="upper right")
        enlarge_axis_text(ax)
    png, pdf = save_figure(fig, out, "Fig7_revised"); plt.close(fig)
    return entry("Fig7_revised", png, pdf, 7.2, ["Mean amplitude (reference and reconstruction) (uV)", "Difference: reconstruction - reference (uV)"],
                 "Bland-Altman analysis for DEAP Fp1 and Fp2 reconstruction.",
                 "Bias and limits of agreement use all samples; a fixed random subset is displayed to control overplotting.", paths)


def clean_metrics(fake, real):
    d = fake - real
    # Match E6: compute each trial-channel metric first, then pool observations.
    rmse = np.sqrt(np.mean(d * d, axis=-1))
    mae = np.mean(np.abs(d), axis=-1)
    return float(rmse.mean()), float(mae.mean())


def figure8(out: Path):
    csv = REVISION_ROOT / "6.robustness/outputs/formal_input_noise/summary_metrics.csv"
    df = pd.read_csv(csv); models = [m for m in MODEL_LABELS if m in set(df.model)]
    clean, sources = {}, [csv]
    cigan, real, src = deap_full(); clean["cigan"] = clean_metrics(cigan, real); sources += src
    for model in models:
        if model == "cigan": continue
        pat = REVISION_ROOT / f"4.baselines/predictions/deap/fold_{{fold}}/{model}_seed_42_e4_stabilized_e100.npz"
        fake, r, src = load_folds(pat); clean[model] = clean_metrics(fake, r); sources += src
    noises = [x for x in ("white", "pink", "impulsive") if x in set(df.noise_type)]
    fig, axes = plt.subplots(2, len(noises), figsize=(7.2, 5.0), sharex=True)
    for col, noise in enumerate(noises):
        for mi, model in enumerate(models):
            z = df[(df.noise_type == noise) & (df.model == model)].sort_values("snr_db")
            color = COLORS[mi]; label = MODEL_LABELS[model]
            axes[0, col].plot(z.snr_db, z.rmse_mean, marker="o", ms=3, lw=1.2, color=color, label=label)
            axes[1, col].plot(z.snr_db, z.rmse_mean - clean[model][0], marker="o", ms=3, lw=1.2, color=color, label=label)
        axes[0, col].set_title(noise.capitalize() + " noise"); axes[1, col].set_xlabel("Input SNR (dB)")
        axes[1, col].axhline(0, color="black", lw=.8)
    axes[0, 0].set_ylabel(r"RMSE ($\mu$V)"); axes[1, 0].set_ylabel(r"RMSE increase from clean ($\mu$V)")
    axes[0, -1].legend(frameon=False, bbox_to_anchor=(1.03, 1), loc="upper left")
    for ax in axes.flat:
        enlarge_axis_text(ax)
    fig.tight_layout(); png, pdf = save_figure(fig, out, "Fig8_revised"); plt.close(fig)
    return entry("Fig8_revised", png, pdf, 7.2, ["Input SNR (dB)", "RMSE (uV) / RMSE increase from clean (uV)"],
                 "DEAP reconstruction robustness when noise is added to non-frontal input EEG before inference.",
                 "Top: absolute RMSE. Bottom: degradation relative to each model's clean-test RMSE. Fixed checkpoints were not retrained.", sources)


def figure9(out: Path):
    path = REVISION_ROOT / "5.uncertainty/outputs/deap/revision_mc50/deap_revision_mc50.npz"
    with np.load(path, allow_pickle=False) as z:
        pred, std, real, fs, passes = z["fake_fp"], z["fake_std"], z["real_fp"], float(z["sampling_rate_hz"]), int(z["passes"])
    corrs = np.array([np.corrcoef(pred[i, 0], real[i, 0])[0, 1] for i in range(len(pred))])
    idx = int(np.argsort(np.nan_to_num(corrs, nan=-2))[len(corrs)//2]); count = min(pred.shape[-1], int(5 * fs)); t = np.arange(count) / fs
    mu, sigma, y = pred[idx, 0, :count], std[idx, 0, :count], real[idx, 0, :count]
    fig, ax = plt.subplots(figsize=(7.2, 3.0)); ax.plot(t, y, color="black", lw=.8, label="Reference")
    ax.plot(t, mu, color=COLORS[0], lw=1, label="Predictive mean")
    ax.fill_between(t, mu - 1.96*sigma, mu + 1.96*sigma, color=COLORS[0], alpha=.2,
                    label="Uncalibrated MC-dropout band")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(r"Amplitude ($\mu$V)"); ax.set_title(f"DEAP Fp1 example ({passes} stochastic passes)")
    ax.legend(frameon=False, ncol=3)
    enlarge_axis_text(ax, label_size=12, tick_size=11, title_size=13, legend_size=10.5)
    fig.tight_layout(); png, pdf = save_figure(fig, out, "Fig9_revised"); plt.close(fig)
    return entry("Fig9_revised", png, pdf, 7.2, ["Time (s)", "Amplitude (uV)"],
                 "Example DEAP Fp1 reconstruction with MC-dropout predictive dispersion.",
                 "The shaded region is an uncalibrated uncertainty band, not a 95% frequentist confidence interval; the trial was selected deterministically at median correlation.", [path])


def figure10(out: Path):
    fake, real, paths = deap_full(); fs = 128.; bands = {"Delta": (1,4), "Theta": (4,8), "Alpha": (8,13), "Beta": (13,30)}
    f, pf = welch(fake[:, 0], fs=fs, axis=1, nperseg=256); _, pr = welch(real[:, 0], fs=fs, axis=1, nperseg=256)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for ax, (name, (lo, hi)) in zip(axes.flat, bands.items()):
        mask = (f >= lo) & (f < hi); bf = np.trapz(pf[:, mask], f[mask], axis=1); br = np.trapz(pr[:, mask], f[mask], axis=1)
        mean, diff = (bf + br) / 2, bf - br; bias, sd = diff.mean(), diff.std(ddof=1)
        ax.scatter(mean, diff, s=8, alpha=.28, color=COLORS[0], rasterized=True)
        ax.axhline(bias, color=COLORS[4], lw=1.1, label=f"Bias = {bias:.2f}")
        ax.axhline(bias-1.96*sd, color="black", ls="--", lw=.9, label="95% LoA"); ax.axhline(bias+1.96*sd, color="black", ls="--", lw=.9)
        ax.set_title(name); ax.legend(frameon=False)
        enlarge_axis_text(ax, tick_size=11, title_size=13, legend_size=10.5)
    fig.supxlabel(r"Mean band power of reference and reconstruction ($\mu V^2$)", y=.015, fontsize=12)
    fig.supylabel(r"Band-power difference: reconstruction - reference ($\mu V^2$)", x=.015, fontsize=12)
    fig.tight_layout(rect=(.08, .07, 1, 1), h_pad=2.0, w_pad=2.0)
    png, pdf = save_figure(fig, out, "Fig10_revised"); plt.close(fig)
    return entry("Fig10_revised", png, pdf, 7.2, ["Mean band power (reference and reconstruction) (uV^2)", "Difference: reconstruction - reference (uV^2)"],
                 "Fp1 band-power Bland-Altman analysis on the DEAP test folds.",
                 "The horizontal axis is the mean band power of the paired measurements, not time step or sample number.", paths)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "reviewer6_figures")
    parser.add_argument("--seed", type=int, default=42); args = parser.parse_args(); configure_matplotlib()
    figures = [figure5(args.output_dir, args.seed), figure7(args.output_dir, args.seed), figure8(args.output_dir), figure9(args.output_dir), figure10(args.output_dir)]
    manifest = DEFAULT_OUTPUT / "reviewer6_figure_manifest.json"; write_json(manifest, {"figures": figures})
    captions = DEFAULT_OUTPUT / "reviewer6_captions.md"
    captions.write_text("\n\n".join(f"### {x['figure_id']}\n\n{x['caption']}\n\nNote: {x['note']}" for x in figures), encoding="utf-8")
    print(f"Generated {len(figures)} figures in {args.output_dir}"); print(f"Manifest: {manifest}"); return 0


if __name__ == "__main__": raise SystemExit(main())
