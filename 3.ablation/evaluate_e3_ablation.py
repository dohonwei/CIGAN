"""Evaluate E3 variants and run subject-level paired statistics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import rankdata, wilcoxon
from tqdm.auto import tqdm

from e3_common import ABLATION_VARIANTS, E3_ROOT, prediction_path

REVISION_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REVISION_ROOT / "1.prior_benchmark"
DATASETS = ("deap", "hci")
sys.path.insert(0, str(REVISION_ROOT / "code"))
from legacy_artifacts import (
    PROJECT_ROOT, canonical_targets, fold_ids, load_legacy_prediction,
)


def e1_prediction_path(dataset, fold, prior, seed, run_tag):
    suffix = f"_{run_tag}" if run_tag else ""
    return E1_ROOT / "predictions" / dataset / f"fold_{fold}" / f"{prior}_seed_{seed}{suffix}.npz"

BANDS = {"delta_diff": (1., 4.), "theta_diff": (4., 8.), "alpha_diff": (8., 13.), "beta_diff": (13., 30.)}
METRICS = ("cosine", "pearson", "dtw", *BANDS)
LOWER_IS_BETTER = {"dtw", *BANDS}


def cosine(x, y):
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean(); denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def powers(x, fs):
    frequencies, spectrum = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    return {name: float(np.trapz(spectrum[(frequencies >= low) & (frequencies <= high)], frequencies[(frequencies >= low) & (frequencies <= high)])) for name, (low, high) in BANDS.items()}


def dtw(x, y):
    try: from fastdtw import fastdtw
    except ImportError as exc: raise RuntimeError("fastdtw is required for manuscript-aligned DTW") from exc
    return float(fastdtw(x[::4], y[::4], dist=2)[0])


LEGACY_PATHS = {
    "deap": {
        "full": "ablation/deap/ablation_full_fake_fp.npy",
        "no_causal": "ablation/deap/ablation_no_causal_fake_fp.npy",
        "no_fen": "ablation/deap/ablation_no_fen_fake_fp.npy",
        "no_psd": "ablation/deap/ablation_no_psd_fake_fp.npy",
    },
    "hci": {
        "full": "ablation/hci/ablation_full_fake_fp.npy",
        "no_causal": "ablation/hci/ablation_no_causal_fake_fp.npy",
        "no_fen": "ablation/hci/ablation_no_fen_fake_fp.npy",
        "no_psd": "ablation/hci/ablation_no_psd_fake_fp.npy",
    },
}


def evaluate_arrays(
    fake, real, test_idx, subjects, folds, fs, dataset, variant, show_progress=True
):
    if fake.shape != real.shape or fake.shape[1:] != (2, 3840):
        raise ValueError(f"Unexpected prediction shape: {fake.shape}/{real.shape}")
    if not np.isfinite(fake).all() or not np.isfinite(real).all():
        raise ValueError("Non-finite prediction")
    rows = []
    trials = tqdm(
        range(len(fake)),
        desc=f"E3 {dataset}:{variant}",
        unit="trial",
        disable=not show_progress,
    )
    for trial in trials:
        for channel, channel_name in enumerate(("Fp1", "Fp2")):
            generated, reference = fake[trial, channel].astype(float), real[trial, channel].astype(float)
            generated_power, reference_power = powers(generated, fs), powers(reference, fs)
            row = {"dataset": dataset, "fold": int(folds[trial]), "variant": variant,
                   "global_trial_idx": int(test_idx[trial]), "subject": str(subjects[trial]), "channel": channel_name,
                   "cosine": cosine(reference, generated), "pearson": pearson(reference, generated), "dtw": dtw(reference, generated)}
            row.update({band: abs(reference_power[band] - generated_power[band]) for band in BANDS}); rows.append(row)
    return rows


def evaluate_archive(path, dataset, fold, variant, show_progress=True):
    if not path.exists(): raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        fake, real = archive["fake_uv"], archive["real_uv"]
        test_idx, subjects, fs = archive["test_idx"], archive["test_subjects"], float(archive["sampling_rate_hz"])
    if fake.shape != real.shape or fake.shape[1:] != (2, 3840): raise ValueError(f"Unexpected prediction shape: {path}: {fake.shape}/{real.shape}")
    if not np.isfinite(fake).all() or not np.isfinite(real).all(): raise ValueError(f"Non-finite prediction: {path}")
    folds = np.full(len(fake), fold, dtype=np.int64)
    return evaluate_arrays(
        fake, real, test_idx, subjects, folds, fs, dataset, variant, show_progress
    )


def evaluate_legacy(dataset, variant, show_progress=True):
    protocol = "legacy" if dataset == "hci" else "revision"
    path = PROJECT_ROOT / LEGACY_PATHS[dataset][variant]
    if path.suffix == ".npy":
        fake = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        real, subjects, sampling_rate = canonical_targets(dataset, protocol)
        if fake.shape != real.shape or not np.isfinite(fake).all():
            raise ValueError(
                f"Legacy prediction does not match canonical targets: {path}: "
                f"fake={fake.shape}, canonical={real.shape}"
            )
        artifact = {
            "fake": fake, "real": real, "subjects": subjects,
            "folds": fold_ids(dataset, len(fake)),
            "sampling_rate_hz": sampling_rate,
        }
    else:
        artifact = load_legacy_prediction(path, dataset, protocol=protocol)
    test_idx = np.arange(len(artifact["fake"]), dtype=np.int64)
    return evaluate_arrays(
        artifact["fake"], artifact["real"], test_idx, artifact["subjects"],
        artifact["folds"], float(artifact["sampling_rate_hz"]), dataset, variant,
        show_progress,
    )


def holm(p_values):
    p = np.asarray(p_values, dtype=float); order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p); adjusted[order] = np.minimum(adjusted_sorted, 1.); return adjusted


def rank_biserial(difference):
    nonzero = difference[difference != 0]
    if not len(nonzero): return 0.
    ranks = rankdata(np.abs(nonzero)); return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def comparisons(subject_df):
    rows = []
    for variant in ABLATION_VARIANTS:
        for metric in METRICS:
            pivot = subject_df.pivot(index="subject", columns="variant", values=metric)
            paired = pivot[["full", variant]].dropna(); full, ablated = paired["full"].to_numpy(), paired[variant].to_numpy()
            difference = full - ablated
            try: statistic, p_raw = wilcoxon(full, ablated)
            except ValueError: statistic, p_raw = 0., 1.
            favorable = -difference if metric in LOWER_IS_BETTER else difference
            rows.append({"variant": variant, "metric": metric, "n_subjects": len(paired), "full_mean": float(full.mean()),
                         "variant_mean": float(ablated.mean()), "mean_difference_full_minus_variant": float(difference.mean()),
                         "full_favorable_mean": bool(favorable.mean() > 0), "wilcoxon_statistic": float(statistic),
                         "p_raw": float(p_raw), "rank_biserial_raw_difference": rank_biserial(difference)})
    result = pd.DataFrame(rows); result["p_holm"] = holm(result["p_raw"]); result["significant_holm_0.05"] = result["p_holm"] < .05
    return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--run-tag", default="e3_welch_e100")
    parser.add_argument("--full-run-tag", default="fixed_prior_welch_e100")
    parser.add_argument("--source", choices=("revision", "legacy"), default="revision")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    show_progress = not args.no_progress
    rows = []
    if args.source == "legacy":
        for variant in ("full", *ABLATION_VARIANTS):
            rows.extend(evaluate_legacy(args.dataset, variant, show_progress))
    else:
        for fold in range(1, 6):
            rows.extend(evaluate_archive(e1_prediction_path(args.dataset, fold, "granger", args.seed, args.full_run_tag), args.dataset, fold, "full", show_progress))
            for variant in ABLATION_VARIANTS:
                rows.extend(evaluate_archive(prediction_path(args.dataset, fold, variant, args.seed, args.run_tag), args.dataset, fold, variant, show_progress))
    trial_df = pd.DataFrame(rows)
    # Ensure every paired method was evaluated on exactly the same held-out trials.
    counts = trial_df.groupby(["variant", "global_trial_idx", "channel"]).size()
    if not (counts == 1).all(): raise ValueError("Duplicate E3 trial/channel rows detected")
    trial_sets = [set(zip(g.global_trial_idx, g.channel)) for _, g in trial_df.groupby("variant")]
    if any(items != trial_sets[0] for items in trial_sets[1:]): raise ValueError("E3 variants do not share identical held-out trials")
    result_tag = "legacy_reuse" if args.source == "legacy" else f"seed_{args.seed}_{args.run_tag}"
    output = E3_ROOT / "results" / args.dataset / result_tag; output.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(output / "trial_channel_metrics.csv", index=False)
    subject_df = trial_df.groupby(["dataset", "variant", "subject"], as_index=False)[list(METRICS)].mean()
    subject_df.to_csv(output / "subject_metrics.csv", index=False)
    fold_df = trial_df.groupby(["dataset", "variant", "fold"], as_index=False)[list(METRICS)].mean()
    fold_df.to_csv(output / "fold_metrics.csv", index=False)
    summary = subject_df.groupby(["dataset", "variant"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]; summary.reset_index().to_csv(output / "ablation_summary.csv", index=False)
    stats = comparisons(subject_df); stats.insert(0, "dataset", args.dataset); stats.to_csv(output / "full_vs_ablation_statistics.csv", index=False)
    print(f"Saved E3 results: {output}"); return 0


if __name__ == "__main__": raise SystemExit(main())
