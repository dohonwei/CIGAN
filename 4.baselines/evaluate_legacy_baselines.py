"""Re-evaluate manuscript-matched baseline predictions with one metric implementation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import rankdata, wilcoxon
from tqdm.auto import tqdm

E4_ROOT = Path(__file__).resolve().parents[0]
REVISION_ROOT = E4_ROOT.parent
sys.path.insert(0, str(REVISION_ROOT / "code"))
from legacy_artifacts import PROJECT_ROOT, canonical_targets, fold_ids


PATHS = {
    "deap": {
        "cigan": "ablation/deap/ablation_full_fake_fp.npy",
        "hveegnet": "ablation/deap/hveegnet_fake_fp.npy",
        "tieeegnet": "ablation/deap/tie_eegnet_fake_fp.npy",
        "wavenet": "ablation/deap/wavenet_fake_fp.npy",
        "spline": "ablation/deap/spline_fake_fp.npy",
        "encoder_decoder": "ablation/deap/encdec_fake_fp.npy",
    },
    "hci": {
        "cigan": "ablation/hci/ablation_full_fake_fp.npy",
        "hveegnet": "ablation/hci/hveegnet_fake_fp.npy",
        "tieeegnet": "ablation/hci/tie_eegnet_fake_fp.npy",
        "wavenet": "ablation/hci/wavenet_fake_fp.npy",
        "spline": "ablation/hci/spline_fake_fp.npy",
        "encoder_decoder": "ablation/hci/encdec_fake_fp.npy",
    },
}
BANDS = {"delta_diff": (1., 4.), "theta_diff": (4., 8.), "alpha_diff": (8., 13.), "beta_diff": (13., 30.)}
METRICS = ("cosine", "pearson", "dtw", *BANDS)
LOWER_IS_BETTER = {"dtw", *BANDS}


def cosine(x, y):
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else np.nan


def band_powers(x, fs):
    frequencies, spectrum = welch(x, fs=fs, nperseg=min(len(x), int(2 * fs)))
    return {
        name: float(np.trapz(spectrum[(frequencies >= low) & (frequencies <= high)], frequencies[(frequencies >= low) & (frequencies <= high)]))
        for name, (low, high) in BANDS.items()
    }


def dtw(x, y):
    try:
        from fastdtw import fastdtw
    except ImportError as exc:
        raise RuntimeError("fastdtw is required for manuscript-aligned DTW") from exc
    return float(fastdtw(x[::4], y[::4], dist=2)[0])


def evaluate_model(dataset, model):
    protocol = "legacy" if dataset == "hci" else "revision"
    path = PROJECT_ROOT / PATHS[dataset][model]
    fake_all = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    real_all, subjects, sampling_rate = canonical_targets(dataset, protocol)
    folds = fold_ids(dataset, len(real_all))
    if fake_all.shape != real_all.shape or fake_all.shape[1:] != (2, 3840):
        raise ValueError(f"Prediction/canonical shape mismatch: {path}: {fake_all.shape}/{real_all.shape}")
    if not np.isfinite(fake_all).all() or not np.isfinite(real_all).all():
        raise ValueError(f"Non-finite prediction/reference: {path}")
    rows = []
    for trial in tqdm(range(len(fake_all)), desc=f"E4 {dataset}:{model}", unit="trial"):
        for channel, channel_name in enumerate(("Fp1", "Fp2")):
            fake = fake_all[trial, channel].astype(float)
            real = real_all[trial, channel].astype(float)
            fake_power = band_powers(fake, sampling_rate)
            real_power = band_powers(real, sampling_rate)
            row = {
                "dataset": dataset, "model": model, "fold": int(folds[trial]),
                "global_trial_idx": trial, "subject": str(subjects[trial]), "channel": channel_name,
                "cosine": cosine(real, fake), "pearson": pearson(real, fake),
                "dtw": dtw(real, fake),
            }
            row.update({name: abs(real_power[name] - fake_power[name]) for name in BANDS})
            rows.append(row)
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


def comparisons(subject_df, models):
    rows = []
    for model in models:
        if model == "cigan": continue
        for metric in METRICS:
            pivot = subject_df.pivot(index="subject", columns="model", values=metric)
            paired = pivot[["cigan", model]].dropna()
            difference = paired["cigan"].to_numpy() - paired[model].to_numpy()
            try: statistic, p_raw = wilcoxon(paired["cigan"], paired[model])
            except ValueError: statistic, p_raw = 0., 1.
            favorable = -difference if metric in LOWER_IS_BETTER else difference
            rows.append({
                "comparator": model, "metric": metric, "n_subjects": len(paired),
                "cigan_mean": float(paired["cigan"].mean()), "comparator_mean": float(paired[model].mean()),
                "mean_difference_cigan_minus_comparator": float(difference.mean()),
                "cigan_favorable_mean": bool(favorable.mean() > 0), "wilcoxon_statistic": float(statistic),
                "p_raw": float(p_raw), "rank_biserial_raw_difference": rank_biserial(difference),
            })
    frame = pd.DataFrame(rows); frame["p_holm"] = holm(frame["p_raw"])
    frame["significant_holm_0.05"] = frame["p_holm"] < .05
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="deap", choices=("deap", "hci"))
    args = parser.parse_args()
    models = PATHS[args.dataset]
    trial_df = pd.DataFrame(row for model in models for row in evaluate_model(args.dataset, model))
    trial_sets = [set(zip(group.global_trial_idx, group.channel)) for _, group in trial_df.groupby("model")]
    if any(items != trial_sets[0] for items in trial_sets[1:]):
        raise ValueError("Legacy baseline predictions are not aligned to identical trials")
    output = E4_ROOT / "results" / args.dataset / "legacy_reuse"; output.mkdir(parents=True, exist_ok=True)
    trial_df.to_csv(output / "trial_channel_metrics.csv", index=False)
    subject_df = trial_df.groupby(["dataset", "model", "subject"], as_index=False)[list(METRICS)].mean()
    subject_df.to_csv(output / "subject_metrics.csv", index=False)
    fold_df = trial_df.groupby(["dataset", "model", "fold"], as_index=False)[list(METRICS)].mean()
    fold_df.to_csv(output / "fold_metrics.csv", index=False)
    summary = subject_df.groupby(["dataset", "model"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "baseline_summary.csv", index=False)
    comparisons(subject_df, models).to_csv(output / "cigan_vs_baselines_statistics.csv", index=False)
    print(f"Saved validated legacy E4 results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
