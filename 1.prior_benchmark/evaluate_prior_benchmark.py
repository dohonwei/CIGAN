"""Evaluate completed E1 predictions and compare Granger with simpler priors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import rankdata, wilcoxon

from e1_common import DATASETS, PRIORS, E1_ROOT, prediction_path


BANDS = {
    "delta_diff": (1.0, 4.0),
    "theta_diff": (4.0, 8.0),
    "alpha_diff": (8.0, 13.0),
    "beta_diff": (13.0, 30.0),
}
LOWER_IS_BETTER = {
    "rmse", "mae", "dtw", "delta_diff", "theta_diff", "alpha_diff", "beta_diff"
}
METRICS = [
    "cosine", "pearson", "rmse", "mae", "dtw",
    "delta_diff", "theta_diff", "alpha_diff", "beta_diff",
]


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 0 else np.nan


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(np.dot(x_centered, y_centered) / denominator) if denominator > 0 else np.nan


def band_powers(signal: np.ndarray, fs: float) -> dict[str, float]:
    frequencies, spectrum = welch(signal, fs=fs, nperseg=min(len(signal), int(2 * fs)))
    powers = {}
    for name, (low, high) in BANDS.items():
        selected = (frequencies >= low) & (frequencies <= high)
        powers[name] = float(np.trapz(spectrum[selected], frequencies[selected]))
    return powers


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from fastdtw import fastdtw
    except ImportError as exc:
        raise RuntimeError("fastdtw is required to reproduce the manuscript DTW metric") from exc
    # Match the established project evaluation protocol: downsample by four for DTW.
    distance, _ = fastdtw(x[::4], y[::4], dist=2)
    return float(distance)


def evaluate_file(
    dataset: str, fold: int, prior: str, seed: int, run_tag: str | None = None
) -> list[dict]:
    path = prediction_path(dataset, fold, prior, seed, run_tag)
    if not path.exists():
        raise FileNotFoundError(f"Missing E1 prediction: {path}")
    with np.load(path, allow_pickle=False) as archive:
        fake = archive["fake_uv"]
        real = archive["real_uv"]
        test_idx = archive["test_idx"]
        subjects = archive["test_subjects"]
        fs = float(archive["sampling_rate_hz"])
    if fake.shape != real.shape or fake.shape[1:] != (2, 3840):
        raise ValueError(f"Unexpected prediction shape in {path}: {fake.shape}/{real.shape}")
    if not np.isfinite(fake).all() or not np.isfinite(real).all():
        raise ValueError(f"Non-finite values in {path}")

    rows = []
    for local_idx in range(len(fake)):
        for channel_idx, channel_name in enumerate(("Fp1", "Fp2")):
            generated = fake[local_idx, channel_idx].astype(np.float64)
            reference = real[local_idx, channel_idx].astype(np.float64)
            reference_bands = band_powers(reference, fs)
            generated_bands = band_powers(generated, fs)
            row = {
                "dataset": dataset,
                "fold": fold,
                "prior": prior,
                "seed": seed,
                "global_trial_idx": int(test_idx[local_idx]),
                "subject": str(subjects[local_idx]),
                "channel": channel_name,
                "cosine": cosine_similarity(reference, generated),
                "pearson": pearson_correlation(reference, generated),
                "rmse": float(np.sqrt(np.mean((reference - generated) ** 2))),
                "mae": float(np.mean(np.abs(reference - generated))),
                "dtw": dtw_distance(reference, generated),
            }
            for band_name in BANDS:
                row[band_name] = abs(reference_bands[band_name] - generated_bands[band_name])
            rows.append(row)
    return rows


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / ranks.sum())


def statistical_comparisons(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparators = [prior for prior in PRIORS if prior != "granger"]
    for comparator in comparators:
        for metric in METRICS:
            pivot = subject_df.pivot(index="subject", columns="prior", values=metric)
            paired = pivot[["granger", comparator]].dropna()
            granger_values = paired["granger"].to_numpy()
            comparator_values = paired[comparator].to_numpy()
            differences = granger_values - comparator_values
            try:
                statistic, p_value = wilcoxon(granger_values, comparator_values)
            except ValueError:
                statistic, p_value = 0.0, 1.0
            favorable = -differences if metric in LOWER_IS_BETTER else differences
            rows.append({
                "comparator": comparator,
                "metric": metric,
                "n_subjects": len(paired),
                "granger_mean": float(granger_values.mean()),
                "comparator_mean": float(comparator_values.mean()),
                "mean_difference_granger_minus_comparator": float(differences.mean()),
                "wilcoxon_statistic": float(statistic),
                "p_raw": float(p_value),
                "rank_biserial_raw_difference": rank_biserial(differences),
                "granger_favorable_mean": bool(favorable.mean() > 0),
            })
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"].tolist())
    result["significant_holm_0.05"] = result["p_holm"] < 0.05
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()
    result_suffix = f"_{args.run_tag}" if args.run_tag else ""
    output_dir = E1_ROOT / f"results/{args.dataset}/seed_{args.seed}{result_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for prior in PRIORS:
        for fold in range(1, 6):
            rows.extend(
                evaluate_file(args.dataset, fold, prior, args.seed, args.run_tag)
            )
    trial_df = pd.DataFrame(rows)
    trial_df.to_csv(output_dir / "trial_channel_metrics.csv", index=False)

    subject_df = (
        trial_df.groupby(["dataset", "seed", "prior", "subject"], as_index=False)[METRICS]
        .mean()
    )
    subject_df.to_csv(output_dir / "subject_metrics.csv", index=False)

    summary = subject_df.groupby(["dataset", "seed", "prior"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output_dir / "prior_summary.csv", index=False)

    statistics = statistical_comparisons(subject_df)
    statistics.insert(0, "dataset", args.dataset)
    statistics.insert(1, "seed", args.seed)
    statistics.to_csv(output_dir / "granger_vs_priors_statistics.csv", index=False)
    print(f"Saved E1 results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
