"""Evaluate E4 baselines against the formal E1 CIGAN predictions."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import rankdata, wilcoxon

from e4_common import BASELINES, E4_ROOT, prediction_path
from e1_common import DATASETS, prediction_path as e1_prediction_path

BANDS = {"delta_diff": (1., 4.), "theta_diff": (4., 8.), "alpha_diff": (8., 13.), "beta_diff": (13., 30.)}
METRICS = ("cosine", "pearson", "dtw", *BANDS)
LOWER_IS_BETTER = {"dtw", *BANDS}


def cosine(x, y):
    denominator = np.linalg.norm(x) * np.linalg.norm(y); return float(np.dot(x, y) / denominator) if denominator else np.nan


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean(); denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def powers(x, fs):
    frequencies, spectrum = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs))); result = {}
    for name, (low, high) in BANDS.items():
        selected = (frequencies >= low) & (frequencies <= high)
        result[name] = float(np.trapz(spectrum[selected], frequencies[selected]))
    return result


def dtw(x, y):
    try: from fastdtw import fastdtw
    except ImportError as exc: raise RuntimeError("fastdtw is required for manuscript-aligned DTW") from exc
    return float(fastdtw(x[::4], y[::4], dist=2)[0])


def evaluate_archive(path, dataset, fold, model):
    if not path.exists(): raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        fake, real = archive["fake_uv"], archive["real_uv"]
        test_idx, subjects, fs = archive["test_idx"], archive["test_subjects"], float(archive["sampling_rate_hz"])
    if fake.shape != real.shape or fake.shape[1:] != (2, 3840): raise ValueError(f"Unexpected shape: {path}: {fake.shape}/{real.shape}")
    if not np.isfinite(fake).all() or not np.isfinite(real).all(): raise ValueError(f"Non-finite values: {path}")
    rows = []
    for trial in range(len(fake)):
        for channel, channel_name in enumerate(("Fp1", "Fp2")):
            generated, reference = fake[trial, channel].astype(float), real[trial, channel].astype(float)
            generated_power, reference_power = powers(generated, fs), powers(reference, fs)
            row = {"dataset": dataset, "fold": fold, "model": model, "global_trial_idx": int(test_idx[trial]), "subject": str(subjects[trial]), "channel": channel_name,
                   "cosine": cosine(reference, generated), "pearson": pearson(reference, generated), "dtw": dtw(reference, generated)}
            row.update({name: abs(reference_power[name] - generated_power[name]) for name in BANDS}); rows.append(row)
    return rows


def holm(values):
    p = np.asarray(values, dtype=float); order = np.argsort(p); sorted_adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p); adjusted[order] = np.minimum(sorted_adjusted, 1.); return adjusted


def rank_biserial(difference):
    nonzero = difference[difference != 0]
    if not len(nonzero): return 0.
    ranks = rankdata(np.abs(nonzero)); return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def comparisons(subject_df):
    rows = []
    for baseline in BASELINES:
        for metric in METRICS:
            pivot = subject_df.pivot(index="subject", columns="model", values=metric); paired = pivot[["cigan", baseline]].dropna()
            cigan, other = paired["cigan"].to_numpy(), paired[baseline].to_numpy(); difference = cigan - other
            try: statistic, p_raw = wilcoxon(cigan, other)
            except ValueError: statistic, p_raw = 0., 1.
            favorable = -difference if metric in LOWER_IS_BETTER else difference
            rows.append({"baseline": baseline, "metric": metric, "n_subjects": len(paired), "cigan_mean": float(cigan.mean()), "baseline_mean": float(other.mean()),
                         "mean_difference_cigan_minus_baseline": float(difference.mean()), "cigan_favorable_mean": bool(favorable.mean() > 0),
                         "wilcoxon_statistic": float(statistic), "p_raw": float(p_raw), "rank_biserial_raw_difference": rank_biserial(difference)})
    result = pd.DataFrame(rows); result["p_holm"] = holm(result["p_raw"]); result["significant_holm_0.05"] = result["p_holm"] < .05; return result


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--run-tag", default="e4_final_e100")
    parser.add_argument("--cigan-run-tag", default="fixed_prior_final_e100"); args = parser.parse_args(); rows = []
    for fold in range(1, 6):
        rows.extend(evaluate_archive(e1_prediction_path(args.dataset, fold, "granger", args.seed, args.cigan_run_tag), args.dataset, fold, "cigan"))
        for model in BASELINES: rows.extend(evaluate_archive(prediction_path(args.dataset, fold, model, args.seed, args.run_tag), args.dataset, fold, model))
    trial_df = pd.DataFrame(rows); counts = trial_df.groupby(["model", "global_trial_idx", "channel"]).size()
    if not (counts == 1).all(): raise ValueError("Duplicate E4 trial/channel rows detected")
    trial_sets = [set(zip(g.global_trial_idx, g.channel)) for _, g in trial_df.groupby("model")]
    if any(items != trial_sets[0] for items in trial_sets[1:]): raise ValueError("E4 models do not share identical held-out trials")
    output = E4_ROOT / "results" / args.dataset / f"seed_{args.seed}_{args.run_tag}"; output.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(output / "trial_channel_metrics.csv", index=False)
    subject_df = trial_df.groupby(["dataset", "model", "subject"], as_index=False)[list(METRICS)].mean(); subject_df.to_csv(output / "subject_metrics.csv", index=False)
    fold_df = trial_df.groupby(["dataset", "model", "fold"], as_index=False)[list(METRICS)].mean(); fold_df.to_csv(output / "fold_metrics.csv", index=False)
    summary = subject_df.groupby(["dataset", "model"])[list(METRICS)].agg(["mean", "std"]); summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "baseline_summary.csv", index=False)
    stats = comparisons(subject_df); stats.insert(0, "dataset", args.dataset); stats.to_csv(output / "cigan_vs_baselines_statistics.csv", index=False)
    print(f"Saved E4 results: {output}"); return 0


if __name__ == "__main__": raise SystemExit(main())
