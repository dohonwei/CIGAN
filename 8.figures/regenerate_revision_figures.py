"""Generate publication-ready revision figures from final E1--E5 results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e8_common import COLORS, DEFAULT_OUTPUT, PALETTE, REVISION_ROOT, configure_matplotlib, save_figure, sha256, write_json

FIGURE_IDS = ("e1_prior_pilot", "e2_stability", "e3_ablation", "e4_baselines", "e5_calibration")
DISPLAY = {"granger": "Granger", "mutual_info": "MI", "pearson": "Pearson", "random": "Random",
    "uniform": "Uniform", "full": "Full", "no_causal": "No causal", "no_fen": "No FEN",
    "no_psd": "No PSD", "cigan": "CIGAN", "encoder_decoder": "Encoder-decoder",
    "hveegnet": "hvEEGNet", "tieeegnet": "TIE-EEGNet", "wavenet": "WaveNet", "spline": "Spline"}


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists(): raise FileNotFoundError(path)
    frame = pd.read_csv(path); numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all(): raise ValueError(f"Non-finite values in {path}")
    return frame


def panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def bars(ax, labels, values, errors, ylabel, colors=None) -> None:
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=errors, capsize=2.5, color=colors or COLORS[:len(labels)], edgecolor="black",
           linewidth=0.45, error_kw={"elinewidth": 0.7})
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel(ylabel)
    ax.axhline(0, color="#777777", linewidth=0.6)


def entry(figure_id, png, pdf, sources, caption, note, axis_labels, legend_required) -> dict:
    return {"figure_id": figure_id, "png": str(png.resolve()), "pdf": str(pdf.resolve()),
        "width_in": 6.9, "dpi": 300, "minimum_font_pt": 8, "axis_labels": axis_labels,
        "legend_required": legend_required, "caption": caption, "note": note,
        "sources": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in sources]}


def figure_e1(output: Path) -> dict:
    source = REVISION_ROOT / "1.prior_benchmark/pilot_subset/results/deap/subjects16_test8_seed42/prior_summary.csv"
    order = ["granger", "mutual_info", "pearson", "random", "uniform"]
    df = require_csv(source).set_index("prior").loc[order].reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.1), constrained_layout=True)
    specs = [("pearson", "Pearson correlation"), ("rmse", "RMSE (uV)"), ("mae", "MAE (uV)"), ("dtw", "DTW distance")]
    labels = [DISPLAY[x] for x in df.prior]
    for label, ax, (metric, ylabel) in zip("abcd", axes.flat, specs):
        bars(ax, labels, df[f"{metric}_mean"], df[f"{metric}_std"], ylabel); panel_label(ax, label)
    png, pdf = save_figure(fig, output, "e1_prior_pilot"); plt.close(fig)
    return entry("e1_prior_pilot", png, pdf, [source], "Prior comparison in the fixed random-subject DEAP pilot.",
        "Bars show means across eight held-out subjects; error bars show SD. This analysis is exploratory and no Holm-adjusted comparison was significant.",
        ["Prior definition", "Pearson correlation", "RMSE (uV)", "MAE (uV)", "DTW distance"], False)


def figure_e2(output: Path) -> dict:
    records, sources = [], []
    for dataset in ("deap", "hci"):
        for level in ("fold", "subject", "lag"):
            path = REVISION_ROOT / f"2.granger_stability/outputs/{dataset}/{level}_pairwise_stability.csv"
            df = require_csv(path); sources.append(path)
            for value in df.spearman: records.append({"dataset": dataset.upper() if dataset == "deap" else "HCI", "level": level.title(), "spearman": value})
    data = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.15), sharey=True, constrained_layout=True)
    for panel, (ax, dataset) in enumerate(zip(axes, ("DEAP", "HCI"))):
        groups = [data[(data.dataset == dataset) & (data.level == level)].spearman.to_numpy() for level in ("Fold", "Subject", "Lag")]
        bp = ax.boxplot(groups, labels=["Fold", "Subject", "Lag"], patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.1}, showfliers=False)
        for patch, color in zip(bp["boxes"], COLORS[:3]): patch.set_facecolor(color); patch.set_alpha(.78)
        rng = np.random.default_rng(42)
        for pos, group in enumerate(groups, 1): ax.scatter(pos + rng.uniform(-.08, .08, len(group)), group, s=7, color="#333333", alpha=.32, linewidths=0)
        ax.set_title(dataset); ax.set_xlabel("Stability level"); ax.set_ylim(-1.02, 1.02); ax.axhline(0, color="#777777", linewidth=.6); panel_label(ax, "ab"[panel])
    axes[0].set_ylabel("Pairwise Spearman correlation")
    png, pdf = save_figure(fig, output, "e2_stability"); plt.close(fig)
    return entry("e2_stability", png, pdf, sources, "Granger prior ranking stability across folds, subjects, and lag settings.",
        "Boxes summarize pairwise Spearman correlations; points show individual comparisons. Stability must not be interpreted as direct neural causality.",
        ["Stability level", "Pairwise Spearman correlation"], False)


def two_dataset_metric_figure(output: Path, figure_id: str, folder: str, filename: str, key: str,
                              order: list[str], caption: str, note: str) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.2), constrained_layout=True); sources = []
    for row, dataset in enumerate(("deap", "hci")):
        source = REVISION_ROOT / f"{folder}/results/{dataset}/legacy_reuse/{filename}"
        df = require_csv(source).set_index(key).loc[order].reset_index(); sources.append(source)
        labels = [DISPLAY[x] for x in df[key]]
        for col, (metric, ylabel) in enumerate((("pearson", "Pearson correlation"), ("dtw", "DTW distance"))):
            ax = axes[row, col]
            colors = [PALETTE["blue"] if x in {"full", "cigan"} else COLORS[(i + 1) % len(COLORS)] for i, x in enumerate(df[key])]
            bars(ax, labels, df[f"{metric}_mean"], df[f"{metric}_std"], ylabel, colors)
            name = "DEAP" if dataset == "deap" else "HCI"
            ax.set_title(name + (" - correlation" if col == 0 else " - distance")); panel_label(ax, "abcd"[row * 2 + col])
    png, pdf = save_figure(fig, output, figure_id); plt.close(fig)
    return entry(figure_id, png, pdf, sources, caption, note,
                 ["Model condition", "Pearson correlation", "DTW distance"], False)


def figure_e3(output: Path) -> dict:
    return two_dataset_metric_figure(output, "e3_ablation", "3.ablation", "ablation_summary.csv", "variant",
        ["full", "no_causal", "no_fen", "no_psd"], "Subject-level reconstruction metrics for the ablation conditions.",
        "Bars show subject means and error bars show SD. The full model is not uniformly superior on every metric.")


def figure_e4(output: Path) -> dict:
    return two_dataset_metric_figure(output, "e4_baselines", "4.baselines", "baseline_summary.csv", "model",
        ["cigan", "hveegnet", "tieeegnet", "wavenet", "encoder_decoder", "spline"],
        "Unified subject-level comparison with reconstruction baselines.",
        "Bars show subject means and error bars show SD under one common metric implementation.")


def figure_e5(output: Path) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), constrained_layout=True); sources = []
    methods = {
        "uncalibrated_mc_dropout": ("MC dropout (uncalibrated)", PALETTE["orange"]),
        "cross_fold_conformal_scaled": ("Cross-fold conformal scaling", PALETTE["blue"]),
    }
    for row, dataset in enumerate(("deap", "hci")):
        source = REVISION_ROOT / (
            f"5.uncertainty/outputs/{dataset}/cross_fold_conformal_revision/"
            "uq_uncalibrated_vs_conformal.csv"
        )
        df = require_csv(source); sources.append(source); name = "DEAP" if dataset == "deap" else "MAHNOB-HCI"
        required = {"method", "nominal_coverage", "picp_mean", "picp_std", "mpiw_mean_uv", "mpiw_std_uv"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{source} is missing columns: {sorted(missing)}")
        if not set(methods).issubset(set(df.method)):
            raise ValueError(f"{source} does not contain both raw and calibrated E5 results")

        ax = axes[row, 0]
        ax.plot([.5, 1], [.5, 1], linestyle="--", color=PALETTE["black"], label="Ideal calibration")
        for method, (label, color) in methods.items():
            values = df[df.method == method].sort_values("nominal_coverage")
            ax.plot(values.nominal_coverage, values.picp_mean, marker="o", color=color, label=label)
            ax.fill_between(
                values.nominal_coverage,
                np.maximum(0, values.picp_mean - values.picp_std),
                np.minimum(1, values.picp_mean + values.picp_std),
                color=color, alpha=.12,
            )
        ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(.49, 1.0), ylim=(0, 1.0), title=f"{name} - calibration")
        ax.legend(frameon=False, loc="upper left"); panel_label(ax, "ac"[row])

        ax = axes[row, 1]
        for method, (label, color) in methods.items():
            values = df[df.method == method].sort_values("nominal_coverage")
            ax.errorbar(
                values.nominal_coverage, values.mpiw_mean_uv, yerr=values.mpiw_std_uv,
                marker="o", color=color, capsize=2.5, linewidth=1.4, label=label,
            )
        ax.set(xlabel="Nominal coverage", ylabel="Mean interval width (uV)", title=f"{name} - interval width")
        ax.set_ylim(bottom=0); ax.legend(frameon=False, loc="upper left"); panel_label(ax, "bd"[row])

        raw_95 = df[(df.method == "uncalibrated_mc_dropout") & np.isclose(df.nominal_coverage, .95)]
        calibrated_95 = df[(df.method == "cross_fold_conformal_scaled") & np.isclose(df.nominal_coverage, .95)]
        if len(raw_95) != 1 or len(calibrated_95) != 1:
            raise ValueError(f"{source} must contain exactly one 95% row per method")
        raw_95, calibrated_95 = raw_95.iloc[0], calibrated_95.iloc[0]
        print(
            f"{name} at 95%: raw coverage={raw_95.picp_mean:.4f}, raw width={raw_95.mpiw_mean_uv:.2f} uV; "
            f"calibrated coverage={calibrated_95.picp_mean:.4f}, "
            f"calibrated width={calibrated_95.mpiw_mean_uv:.2f} uV",
            flush=True,
        )
    png, pdf = save_figure(fig, output, "e5_calibration"); plt.close(fig)
    return entry("e5_calibration", png, pdf, sources,
        "Raw MC-dropout and cross-fold conformal calibration with coverage-width trade-offs.",
        "Lines show subject-macro means; shaded regions and error bars show across-subject SD. "
        "Conformal scaling brings coverage close to nominal levels at the cost of wider intervals.",
        ["Nominal coverage", "Empirical coverage", "Mean interval width (uV)"], True)


GENERATORS = {"e1_prior_pilot": figure_e1, "e2_stability": figure_e2, "e3_ablation": figure_e3,
              "e4_baselines": figure_e4, "e5_calibration": figure_e5}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--figures", nargs="+", choices=FIGURE_IDS, default=list(FIGURE_IDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "figures"); args = parser.parse_args()
    configure_matplotlib(); entries = []
    for figure_id in args.figures: print(f"Generating {figure_id}...", flush=True); entries.append(GENERATORS[figure_id](args.output_dir))
    manifest = {"schema_version": 1, "figures": entries,
                "existing_e7_figures": str((REVISION_ROOT / "7.synthetic_noise/outputs/figures").resolve())}
    manifest_path = args.output_dir.parent / "figure_manifest.json"; write_json(manifest_path, manifest)
    captions = []
    for item in entries:
        captions.extend([f"## {item['figure_id']}", "", item["caption"], "", f"Note. {item['note']}", "", "```latex",
            "\\begin{figure}[t]", "    \\centering", f"    \\includegraphics[width=\\linewidth]{{{Path(item['pdf']).name}}}",
            f"    \\caption{{{item['caption']}}}", f"    \\label{{fig:{item['figure_id']}}}", "\\end{figure}", "```", ""])
    (args.output_dir.parent / "captions_and_latex.md").write_text("\n".join(captions), encoding="utf-8")
    print(f"Saved {len(entries)} figure packages and manifest: {manifest_path}"); return 0


if __name__ == "__main__": raise SystemExit(main())
