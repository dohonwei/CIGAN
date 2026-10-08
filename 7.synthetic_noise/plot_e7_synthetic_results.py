"""Create E7 quantitative curves and the revised representative Fig. 4."""
from __future__ import annotations

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from e7_common import MODELS, OUTPUT_DIR, PREDICTION_DIR, load_data

LABELS = {"ridge": "Ridge", "hveegnet": "hvEEGNet", "tieeegnet": "TIE-EEGNet", "cigan_granger": "CIGAN-Granger", "cigan_uniform": "CIGAN-uniform", "cigan_structural": "CIGAN-structural (oracle)"}
COLORS = {"ridge": "#555555", "hveegnet": "#0072B2", "tieeegnet": "#009E73", "cigan_granger": "#D55E00", "cigan_uniform": "#E69F00", "cigan_structural": "#CC79A7"}


def save(fig, stem):
    directory = OUTPUT_DIR / "figures"; directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight"); fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig)


def metric_curves(models, metrics_tag, output_tag):
    suffix = f"_{metrics_tag}" if metrics_tag else ""
    with (OUTPUT_DIR / f"e7_metrics_raw{suffix}.csv").open(encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    specs = (("pearson", "Pearson correlation"), ("rmse", "RMSE"), ("output_snr_db", "Output SNR (dB)"), ("log_psd_rmse", "Log-PSD RMSE"))
    fig = plt.figure(figsize=(8.6, 6.6))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(.14, 1, 1),
        left=.10,
        right=.98,
        bottom=.10,
        top=.98,
        wspace=.28,
        hspace=.22,
    )
    axes = np.asarray([
        [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])],
    ])
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    snrs = [-5, 0, 5, 10, 20]
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for ax, (metric, ylabel), panel_label in zip(axes.flat, specs, panel_labels):
        for model in models:
            means, sems = [], []
            for snr in snrs:
                values = np.asarray([float(r[metric]) for r in rows if r["model"] == model and int(r["snr_db"]) == snr])
                means.append(values.mean()); sems.append(values.std(ddof=1) / np.sqrt(len(values)))
            ax.errorbar(
                snrs,
                means,
                yerr=sems,
                marker="o",
                markersize=4.5,
                linewidth=1.4,
                elinewidth=1.0,
                capsize=2.5,
                label=LABELS[model],
                color=COLORS[model],
            )
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(snrs)
        ax.grid(axis="y", color="#D0D0D0", linewidth=.55, alpha=.7)
        ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, labelsize=10)
        ax.set_title(panel_label, loc="left", fontsize=11, fontweight="bold", pad=4)
        for spine in ax.spines.values():
            spine.set_linewidth(.8)
    for ax in axes[-1]: ax.set_xlabel("Input SNR (dB)", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = legend_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=len(models),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        fontsize=10.5,
        handlelength=2.2,
        columnspacing=1.5,
        borderpad=.5,
    )
    legend.get_frame().set_linewidth(0.8)
    output_suffix = f"_{output_tag}" if output_tag else ""
    save(fig, f"e7_snr_metric_curves{output_suffix}")


def waveform_grid(models, prediction_tag, output_tag):
    suffix = f"_{prediction_tag}" if prediction_tag else ""; data = load_data()
    predictions = {m: np.load(PREDICTION_DIR / f"{m}{suffix}.npz")["prediction"] for m in models}
    signals, snrs = ("multisine", "chirp", "erp"), (20, 5, -5); fig, axes = plt.subplots(3, 3, figsize=(12, 7), sharex=True)
    for i, signal in enumerate(signals):
        for j, snr in enumerate(snrs):
            match = np.flatnonzero((data["signal_type"] == signal) & (data["noise_type"] == "white") & (data["snr_db"] == snr) & (data["repeat"] == 16))[0]
            truth = data["true_target_bank"][int(data["source_id"][match]), 0]; fs = float(data["fs"]); length = min(len(truth), int(2 * fs)); t = np.arange(length) / fs
            ax = axes[i, j]; ax.plot(t, truth[:length], color="black", linewidth=1.4, label="Ground truth")
            for model in models: ax.plot(t, predictions[model][match, 0, :length], color=COLORS[model], linewidth=.9, alpha=.85, label=LABELS[model])
            if i == 0: ax.set_title(f"{snr} dB")
            if j == 0: ax.set_ylabel({"multisine": "Multisine", "chirp": "Chirp", "erp": "ERP-like"}[signal] + "\nAmplitude")
            if i == 2: ax.set_xlabel("Time (s)")
            ax.grid(alpha=.18)
    handles, labels = axes[0, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    output_suffix = f"_{output_tag}" if output_tag else ""
    fig.tight_layout(rect=(0, 0, 1, .90)); save(fig, f"e7_revised_figure4_waveforms{output_suffix}")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS)); parser.add_argument("--run-tag", default="", help="Prediction-file tag"); parser.add_argument("--metrics-tag", default=None); parser.add_argument("--output-tag", default=None); args = parser.parse_args()
    metrics_tag = args.run_tag if args.metrics_tag is None else args.metrics_tag
    output_tag = args.run_tag if args.output_tag is None else args.output_tag
    metric_curves(args.models, metrics_tag, output_tag); waveform_grid(args.models, args.run_tag, output_tag); print(f"Figures written to {OUTPUT_DIR / 'figures'}")


if __name__ == "__main__": main()
