"""Evaluate the frozen-generator prior substitutions at subject level."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import rankdata, wilcoxon


E1_ROOT = Path(__file__).resolve().parents[0]
PREDICTION_ROOT = E1_ROOT / "fixed_generator/predictions/deap"
PRIORS = ("granger", "pearson", "mutual_info", "random", "uniform")
BANDS = {"delta_diff": (1., 4.), "theta_diff": (4., 8.), "alpha_diff": (8., 13.), "beta_diff": (13., 30.)}
METRICS = ("cosine", "pearson", "rmse", "mae", "dtw", *BANDS)
LOWER_IS_BETTER = {"rmse", "mae", "dtw", *BANDS}


def cosine(x, y):
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def powers(x, fs):
    frequencies, spectrum = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    return {name: float(np.trapz(spectrum[(frequencies >= low) & (frequencies <= high)], frequencies[(frequencies >= low) & (frequencies <= high)])) for name, (low, high) in BANDS.items()}


def dtw_distance(x, y):
    try:
        from fastdtw import fastdtw
    except ImportError as exc:
        raise RuntimeError("fastdtw is required for the manuscript-aligned DTW metric") from exc
    return float(fastdtw(x[::4], y[::4], dist=2)[0])


def evaluate(prior_name, seed):
    path = PREDICTION_ROOT / f"{prior_name}_seed_{seed}.npz"
    with np.load(path, allow_pickle=False) as archive:
        fake, real = archive["fake_fp"], archive["real_fp"]
        subjects, fs = archive["subjects"].astype(str), float(archive["sampling_rate_hz"])
    if fake.shape != real.shape or fake.shape[1:] != (2, 3840): raise ValueError(f"Shape mismatch: {path}")
    rows = []
    for trial in range(len(fake)):
        for channel, channel_name in enumerate(("Fp1", "Fp2")):
            generated, reference = fake[trial, channel].astype(float), real[trial, channel].astype(float)
            generated_power, reference_power = powers(generated, fs), powers(reference, fs)
            row = {"dataset": "deap", "prior": prior_name, "subject": subjects[trial],
                   "global_trial_idx": trial, "channel": channel_name,
                   "cosine": cosine(reference, generated), "pearson": pearson(reference, generated),
                   "rmse": float(np.sqrt(np.mean((reference - generated) ** 2))),
                   "mae": float(np.mean(np.abs(reference - generated))),
                   "dtw": dtw_distance(reference, generated)}
            row.update({name: abs(reference_power[name] - generated_power[name]) for name in BANDS}); rows.append(row)
    return rows


def rank_biserial(difference):
    nonzero = difference[difference != 0]
    if not len(nonzero): return 0.
    ranks = rankdata(np.abs(nonzero))
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum())


def holm(values):
    values = np.asarray(values, dtype=float); order = np.argsort(values)
    adjusted_ordered = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    adjusted = np.empty_like(values); adjusted[order] = np.minimum(adjusted_ordered, 1.)
    return adjusted


def comparisons(subject_df):
    rows = []
    for comparator in PRIORS[1:]:
        for metric in METRICS:
            pivot = subject_df.pivot(index="subject", columns="prior", values=metric)
            paired = pivot[["granger", comparator]].dropna()
            difference = paired["granger"].to_numpy() - paired[comparator].to_numpy()
            try: statistic, p_raw = wilcoxon(paired["granger"], paired[comparator])
            except ValueError: statistic, p_raw = 0., 1.
            favorable = -difference if metric in LOWER_IS_BETTER else difference
            rows.append({"comparator": comparator, "metric": metric, "n_subjects": len(paired),
                         "granger_mean": float(paired["granger"].mean()), "comparator_mean": float(paired[comparator].mean()),
                         "mean_difference_granger_minus_comparator": float(difference.mean()),
                         "granger_favorable_mean": bool(favorable.mean() > 0), "wilcoxon_statistic": float(statistic),
                         "p_raw": float(p_raw), "rank_biserial_raw_difference": rank_biserial(difference)})
    frame = pd.DataFrame(rows); frame["p_holm"] = holm(frame["p_raw"])
    frame["significant_holm_0.05"] = frame["p_holm"] < .05
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="deap", choices=("deap",))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    trial_df = pd.DataFrame(row for prior in PRIORS for row in evaluate(prior, args.seed))
    trial_sets = [set(zip(group.global_trial_idx, group.channel)) for _, group in trial_df.groupby("prior")]
    if any(items != trial_sets[0] for items in trial_sets[1:]): raise ValueError("Prior outputs are not trial-aligned")
    output = E1_ROOT / "fixed_generator/results/deap"; output.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(output / "trial_channel_metrics.csv", index=False)
    subject_df = trial_df.groupby(["dataset", "prior", "subject"], as_index=False)[list(METRICS)].mean()
    subject_df.to_csv(output / "subject_metrics.csv", index=False)
    summary = subject_df.groupby(["dataset", "prior"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "prior_summary.csv", index=False)
    comparisons(subject_df).to_csv(output / "granger_vs_priors_statistics.csv", index=False)
    print(f"Saved fixed-generator E1 results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
