"""Reviewer 4.2: Granger ranking stability across folds, subjects, and lags."""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
import torch

HERE = Path(__file__).resolve()
E2 = HERE.parents[0]
E1 = E2.parent / "1.prior_benchmark"
sys.path.insert(0, str(E1))

from e1_common import (  # noqa: E402
    CHANNEL_NAMES, DATASETS, SOURCE_INDICES, load_dataset, load_fold,
    normalize_nonnegative,
)
from compute_priors import apply_csd, granger_prior  # noqa: E402

LAGS = {"deap": [3, 5, 8], "hci": [5, 10, 15]}
METRICS = ("spearman", "kendall_tau", "top5_overlap", "top10_overlap")


def one(csd_eeg, lag):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw, details = granger_prior(csd_eeg, lag, device)
    return normalize_nonnegative(raw), details


def stability_metrics(a, b):
    return {
        "spearman": float(spearmanr(a, b).statistic),
        "kendall_tau": float(kendalltau(a, b).statistic),
        "top5_overlap": len(set(np.argsort(a)[-5:]) & set(np.argsort(b)[-5:])) / 5,
        "top10_overlap": len(set(np.argsort(a)[-10:]) & set(np.argsort(b)[-10:])) / 10,
    }


def comparisons(weights, labels):
    rows = []
    for i, j in combinations(range(len(weights)), 2):
        rows.append({"item_a": labels[i], "item_b": labels[j],
                     **stability_metrics(weights[i], weights[j])})
    return rows


def output_dir(dataset):
    out = E2 / "outputs" / dataset
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_pairwise(dataset, kind, weights, labels, meta):
    out = output_dir(dataset)
    np.savez(
        out / f"{kind}_weights.npz", weights=np.stack(weights),
        labels=np.asarray(labels), source_indices=SOURCE_INDICES,
        source_names=CHANNEL_NAMES[SOURCE_INDICES],
    )
    pd.DataFrame(comparisons(weights, labels)).to_csv(
        out / f"{kind}_pairwise_stability.csv", index=False)
    pd.DataFrame(meta).to_csv(out / f"{kind}_metadata.csv", index=False)


def bootstrap_interval(values, rng, n_resamples):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    sampled_means = rng.choice(
        values, size=(n_resamples, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def save_cluster_summary(rows, cluster_column, path, seed, n_resamples):
    """Treat subjects or folds, rather than repeated resamples, as independent."""
    cluster_means = rows.groupby(cluster_column, sort=True)[list(METRICS)].mean()
    rng = np.random.default_rng(seed)
    summary = []
    for metric in METRICS:
        mean, low, high = bootstrap_interval(
            cluster_means[metric].to_numpy(), rng, n_resamples)
        summary.append({
            "metric": metric,
            "n_independent_clusters": len(cluster_means),
            "mean": mean,
            "bootstrap_ci95_low": low,
            "bootstrap_ci95_high": high,
            "bootstrap_resamples": n_resamples,
        })
    pd.DataFrame(summary).to_csv(path, index=False)


def run_fold(dataset, csd_eeg, base_lag):
    weights, labels, meta = [], [], []
    for fold in range(1, 6):
        train_idx, _ = load_fold(dataset, fold)
        weight, details = one(csd_eeg[train_idx], base_lag)
        weights.append(weight)
        labels.append(f"fold_{fold}")
        meta.append({"label": labels[-1], "lag": base_lag,
                     "n_trials": len(train_idx),
                     "significant_edges": details["n_significant_edges"]})
    save_pairwise(dataset, "fold", weights, labels, meta)


def run_subject(dataset, csd_eeg, subjects, base_lag):
    """Across-subject agreement measures heterogeneity, not repeatability."""
    weights, labels, meta = [], [], []
    for subject in np.unique(subjects):
        idx = np.where(subjects == subject)[0]
        weight, details = one(csd_eeg[idx], base_lag)
        weights.append(weight)
        labels.append(str(subject))
        meta.append({"label": labels[-1], "lag": base_lag,
                     "n_trials": len(idx),
                     "significant_edges": details["n_significant_edges"]})
    save_pairwise(dataset, "subject", weights, labels, meta)


def run_lag(dataset, csd_eeg, fs):
    weights, labels, meta = [], [], []
    for fold in range(1, 6):
        train_idx, _ = load_fold(dataset, fold)
        for lag in LAGS[dataset]:
            weight, details = one(csd_eeg[train_idx], lag)
            weights.append(weight)
            labels.append(f"fold_{fold}_p{lag}")
            meta.append({"label": labels[-1], "fold": fold, "lag": lag,
                         "delay_ms": 1000 * lag / fs,
                         "n_trials": len(train_idx),
                         "significant_edges": details["n_significant_edges"]})
    save_pairwise(dataset, "lag", weights, labels, meta)


def run_within_subject(dataset, csd_eeg, subjects, base_lag, seed, repeats,
                       ci_resamples):
    """Estimate repeated split-half reliability within each DEAP subject."""
    if dataset != "deap":
        raise ValueError("within_subject is restricted to the reported DEAP analysis")
    rng = np.random.default_rng(seed)
    rows, weight_rows, weight_labels = [], [], []
    unique_subjects = np.unique(subjects)
    total, done = len(unique_subjects) * repeats, 0
    for subject in unique_subjects:
        subject_idx = np.where(subjects == subject)[0]
        if len(subject_idx) < 4:
            raise ValueError(f"Subject {subject} has fewer than four trials")
        for repeat in range(1, repeats + 1):
            shuffled = rng.permutation(subject_idx)
            half = len(shuffled) // 2
            idx_a, idx_b = shuffled[:half], shuffled[half:2 * half]
            weight_a, details_a = one(csd_eeg[idx_a], base_lag)
            weight_b, details_b = one(csd_eeg[idx_b], base_lag)
            rows.append({
                "subject": str(subject), "repeat": repeat, "seed": seed,
                "lag": base_lag, "n_trials_half_a": len(idx_a),
                "n_trials_half_b": len(idx_b),
                "significant_edges_half_a": details_a["n_significant_edges"],
                "significant_edges_half_b": details_b["n_significant_edges"],
                **stability_metrics(weight_a, weight_b),
            })
            weight_rows.extend([weight_a, weight_b])
            weight_labels.extend([f"{subject}_r{repeat}_a",
                                  f"{subject}_r{repeat}_b"])
            done += 1
            print(f"within_subject: {done}/{total} subject-splits completed",
                  flush=True)

    out = output_dir(dataset)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "within_subject_split_half_stability.csv", index=False)
    save_cluster_summary(
        frame, "subject", out / "within_subject_split_half_summary.csv",
        seed, ci_resamples)
    np.savez(
        out / "within_subject_split_half_weights.npz",
        weights=np.stack(weight_rows), labels=np.asarray(weight_labels),
        source_indices=SOURCE_INDICES,
        source_names=CHANNEL_NAMES[SOURCE_INDICES],
    )


def run_group_bootstrap(dataset, csd_eeg, subjects, base_lag, seed, repeats,
                        ci_resamples):
    """Test each fold-level prior under subject-level bootstrap resampling."""
    if dataset != "deap":
        raise ValueError("group_bootstrap is restricted to the reported DEAP analysis")
    rng = np.random.default_rng(seed)
    rows, weight_rows, weight_labels = [], [], []
    total, done = 5 * repeats, 0
    for fold in range(1, 6):
        train_idx, _ = load_fold(dataset, fold)
        train_subjects = np.unique(subjects[train_idx])
        reference, reference_details = one(csd_eeg[train_idx], base_lag)
        for repeat in range(1, repeats + 1):
            sampled_subjects = rng.choice(
                train_subjects, size=len(train_subjects), replace=True)
            sampled_idx = np.concatenate([
                train_idx[subjects[train_idx] == subject]
                for subject in sampled_subjects
            ])
            weight, details = one(csd_eeg[sampled_idx], base_lag)
            rows.append({
                "fold": fold, "repeat": repeat, "seed": seed,
                "lag": base_lag, "n_train_subjects": len(train_subjects),
                "n_unique_bootstrap_subjects": len(np.unique(sampled_subjects)),
                "n_bootstrap_trials": len(sampled_idx),
                "reference_significant_edges": reference_details["n_significant_edges"],
                "bootstrap_significant_edges": details["n_significant_edges"],
                **stability_metrics(reference, weight),
            })
            weight_rows.append(weight)
            weight_labels.append(f"fold_{fold}_bootstrap_{repeat}")
            done += 1
            print(f"group_bootstrap: {done}/{total} estimates completed",
                  flush=True)

    out = output_dir(dataset)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "group_subject_bootstrap_stability.csv", index=False)
    save_cluster_summary(
        frame, "fold", out / "group_subject_bootstrap_summary.csv",
        seed, ci_resamples)
    np.savez(
        out / "group_subject_bootstrap_weights.npz",
        weights=np.stack(weight_rows), labels=np.asarray(weight_labels),
        source_indices=SOURCE_INDICES,
        source_names=CHANNEL_NAMES[SOURCE_INDICES],
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument(
        "--mode", required=True,
        choices=("fold", "subject", "lag", "within_subject",
                 "group_bootstrap", "enhanced", "all"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repeats", type=int, default=20)
    parser.add_argument("--ci-resamples", type=int, default=10_000)
    args = parser.parse_args()
    if args.split_repeats < 1 or args.bootstrap_repeats < 1:
        parser.error("repeat counts must be positive")
    if args.ci_resamples < 1_000:
        parser.error("--ci-resamples must be at least 1000")
    if args.dataset != "deap" and args.mode in {
        "within_subject", "group_bootstrap", "enhanced"}:
        parser.error(f"--mode {args.mode} is restricted to --dataset deap")
    return args


def main():
    args = parse_args()
    data = load_dataset(args.dataset)
    eeg, subjects = data["eeg"], data["subjects"]
    fs, base_lag = DATASETS[args.dataset]["fs"], DATASETS[args.dataset]["lag"]
    print("Applying CSD once to the complete dataset...", flush=True)
    csd_eeg = apply_csd(eeg, fs)
    if args.mode in ("fold", "all"):
        run_fold(args.dataset, csd_eeg, base_lag)
    if args.mode in ("subject", "all"):
        run_subject(args.dataset, csd_eeg, subjects, base_lag)
    if args.mode in ("lag", "all"):
        run_lag(args.dataset, csd_eeg, fs)
    if args.dataset == "deap" and args.mode in ("within_subject", "enhanced", "all"):
        run_within_subject(
            args.dataset, csd_eeg, subjects, base_lag, args.seed,
            args.split_repeats, args.ci_resamples)
    if args.dataset == "deap" and args.mode in ("group_bootstrap", "enhanced", "all"):
        run_group_bootstrap(
            args.dataset, csd_eeg, subjects, base_lag, args.seed,
            args.bootstrap_repeats, args.ci_resamples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
