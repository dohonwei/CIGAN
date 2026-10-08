"""Compute one fold-specific channel prior using training subjects only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import f as f_distribution
from sklearn.feature_selection import mutual_info_regression

from e1_common import (
    CHANNEL_NAMES, DATASETS, FRONTAL_INDICES, PRIORS, SOURCE_INDICES,
    load_dataset, load_fold, load_pilot_split, normalize_nonnegative,
    pilot_prior_path, prior_path,
    sha256_indices, validate_args, write_json,
)


def apply_csd(train_eeg: np.ndarray, fs: float) -> np.ndarray:
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("MNE is required because the manuscript states CSD preprocessing") from exc
    info = mne.create_info(CHANNEL_NAMES.tolist(), sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020")
    epochs = mne.EpochsArray(train_eeg, info, verbose=False)
    csd = mne.preprocessing.compute_current_source_density(epochs, verbose=False)
    return csd.get_data(copy=True).astype(np.float32)


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    flat = p_values.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    thresholds = alpha * np.arange(1, len(flat) + 1) / len(flat)
    passed = ranked <= thresholds
    cutoff = np.where(passed)[0].max() if passed.any() else -1
    significant = np.zeros(len(flat), dtype=bool)
    if cutoff >= 0:
        significant[order[: cutoff + 1]] = True
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(flat) / np.arange(1, len(flat) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted.reshape(p_values.shape), significant.reshape(p_values.shape)


def granger_prior(train_eeg: np.ndarray, lag: int, device: torch.device) -> tuple[np.ndarray, dict]:
    data = torch.from_numpy(train_eeg)
    n_trials, n_channels, n_times = data.shape
    n_vars = n_channels * (lag + 1)
    covariance = torch.zeros((n_vars, n_vars), dtype=torch.float64, device=device)
    k_total = n_trials * (n_times - lag)
    for start in range(0, n_trials, 64):
        batch = data[start : start + 64].to(device=device, dtype=torch.float64)
        current = batch[:, :, lag:]
        lags = [batch[:, :, lag - step : n_times - step] for step in range(1, lag + 1)]
        design = torch.cat([current] + lags, dim=1)
        flat = design.transpose(1, 2).reshape(-1, n_vars).T
        covariance += flat @ flat.T

    f_stats = np.zeros((len(SOURCE_INDICES), len(FRONTAL_INDICES)), dtype=np.float64)
    p_values = np.ones_like(f_stats)
    d1 = lag
    d2 = k_total - 2 * lag
    if d2 <= 0:
        raise ValueError(f"Invalid Granger denominator degrees of freedom: {d2}")

    for target_pos, target in enumerate(FRONTAL_INDICES):
        target_lags = [int(target + step * n_channels) for step in range(1, lag + 1)]
        r_yy = covariance[target, target]
        r_yx_reduced = covariance[target, target_lags].unsqueeze(0)
        r_xx_reduced = covariance[target_lags][:, target_lags]
        rss_reduced = (
            r_yy - r_yx_reduced @ torch.linalg.pinv(r_xx_reduced) @ r_yx_reduced.T
        ).item()
        for source_pos, source in enumerate(SOURCE_INDICES):
            source_lags = [int(source + step * n_channels) for step in range(1, lag + 1)]
            full_lags = target_lags + source_lags
            r_yx_full = covariance[target, full_lags].unsqueeze(0)
            r_xx_full = covariance[full_lags][:, full_lags]
            rss_full = (r_yy - r_yx_full @ torch.linalg.pinv(r_xx_full) @ r_yx_full.T).item()
            improvement = max(0.0, rss_reduced - rss_full)
            f_stat = (improvement / d1) / (max(rss_full, 1e-12) / d2)
            f_stats[source_pos, target_pos] = f_stat
            p_values[source_pos, target_pos] = f_distribution.sf(f_stat, d1, d2)

    adjusted_p, significant = fdr_bh(p_values)
    edge_scores = np.where(significant, np.log1p(f_stats), 0.0)
    raw_scores = edge_scores.mean(axis=1)
    details = {
        "f_stats": f_stats,
        "p_values": p_values,
        "adjusted_p": adjusted_p,
        "significant": significant,
        "n_significant_edges": int(significant.sum()),
        "n_total_edges": int(significant.size),
        "d1": int(d1),
        "d2": int(d2),
    }
    return raw_scores, details


def pearson_prior(train_eeg: np.ndarray) -> tuple[np.ndarray, dict]:
    raw_scores = np.zeros(len(SOURCE_INDICES), dtype=np.float64)
    target_flat = [train_eeg[:, target, :].reshape(-1) for target in FRONTAL_INDICES]
    for source_pos, source in enumerate(SOURCE_INDICES):
        x = train_eeg[:, source, :].reshape(-1)
        correlations = [abs(float(np.corrcoef(x, y)[0, 1])) for y in target_flat]
        raw_scores[source_pos] = np.mean(correlations)
    return raw_scores, {}


def mutual_information_prior(
    train_eeg: np.ndarray, seed: int, max_samples: int
) -> tuple[np.ndarray, dict]:
    total = train_eeg.shape[0] * train_eeg.shape[2]
    n_samples = min(total, max_samples)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(total, size=n_samples, replace=False))
    flat = train_eeg.transpose(0, 2, 1).reshape(-1, train_eeg.shape[1])
    sampled = flat[selected]
    raw_scores = np.zeros(len(SOURCE_INDICES), dtype=np.float64)
    for source_pos, source in enumerate(SOURCE_INDICES):
        x = sampled[:, [source]]
        values = []
        for target in FRONTAL_INDICES:
            value = mutual_info_regression(
                x, sampled[:, target], random_state=seed, n_neighbors=3
            )[0]
            values.append(float(value))
        raw_scores[source_pos] = np.mean(values)
    return raw_scores, {"mi_samples": int(n_samples), "mi_total_observations": int(total)}


def save_prior(
    dataset: str, fold: int, prior: str, seed: int, weights: np.ndarray,
    raw_scores: np.ndarray, train_idx: np.ndarray, train_subjects: np.ndarray,
    details: dict, output_path=None, split_id: str | None = None,
) -> None:
    path = output_path or prior_path(dataset, fold, prior, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": weights.astype(np.float32),
        "raw_scores": raw_scores.astype(np.float64),
        "source_indices": SOURCE_INDICES,
        "source_names": CHANNEL_NAMES[SOURCE_INDICES],
        "frontal_indices": FRONTAL_INDICES,
        "frontal_names": CHANNEL_NAMES[FRONTAL_INDICES],
        "train_indices_sha256": np.asarray(sha256_indices(train_idx)),
        "train_subjects": train_subjects,
        "dataset": np.asarray(dataset),
        "fold": np.int64(fold),
        "split_id": np.asarray(split_id or ""),
        "prior": np.asarray(prior),
        "seed": np.int64(seed),
    }
    for key, value in details.items():
        if isinstance(value, np.ndarray):
            payload[key] = value
    np.savez(path, **payload)

    csv_path = path.with_suffix(".csv")
    order = np.argsort(weights)[::-1]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "source_index", "source_channel", "raw_score", "weight"])
        for rank, pos in enumerate(order, start=1):
            writer.writerow([rank, SOURCE_INDICES[pos], CHANNEL_NAMES[SOURCE_INDICES[pos]], raw_scores[pos], weights[pos]])

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "fold": fold,
        "split_id": split_id,
        "prior": prior,
        "seed": seed,
        "train_indices_sha256": sha256_indices(train_idx),
        "train_subjects": train_subjects.tolist(),
        "n_train_trials": int(len(train_idx)),
        "sampling_rate_hz": DATASETS[dataset]["fs"],
        "lag_order": DATASETS[dataset]["lag"] if prior == "granger" else None,
        "weights_sum": float(weights.sum()),
        "details": {key: value for key, value in details.items() if not isinstance(value, np.ndarray)},
    }
    write_json(path.with_suffix(".json"), metadata)
    print(f"Saved: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--fold", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--prior", required=True, choices=PRIORS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-mi-samples", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot-split", type=str, default=None)
    args = parser.parse_args()
    validate_args(args.dataset, args.fold, args.prior)
    split_id = None
    if args.pilot_split:
        train_idx, _, split_id = load_pilot_split(Path(args.pilot_split), args.dataset)
        output = pilot_prior_path(args.dataset, split_id, args.prior, args.seed)
    else:
        train_idx, _ = load_fold(args.dataset, args.fold)
        output = prior_path(args.dataset, args.fold, args.prior, args.seed)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    data = load_dataset(args.dataset)
    train_subjects = np.unique(data["subjects"][train_idx])
    train_eeg = data["eeg"][train_idx]

    if args.prior in {"granger", "pearson", "mutual_info"}:
        train_eeg = apply_csd(train_eeg, DATASETS[args.dataset]["fs"])

    if args.prior == "granger":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        raw_scores, details = granger_prior(
            train_eeg, DATASETS[args.dataset]["lag"], device
        )
    elif args.prior == "pearson":
        raw_scores, details = pearson_prior(train_eeg)
    elif args.prior == "mutual_info":
        raw_scores, details = mutual_information_prior(
            train_eeg, args.seed, args.max_mi_samples
        )
    elif args.prior == "random":
        raw_scores = np.random.default_rng(args.seed).random(len(SOURCE_INDICES))
        details = {"random_seed": args.seed}
    else:
        raw_scores = np.ones(len(SOURCE_INDICES), dtype=np.float64)
        details = {}

    weights = normalize_nonnegative(raw_scores)
    save_prior(
        args.dataset, args.fold, args.prior, args.seed, weights, raw_scores,
        train_idx, train_subjects, details, output_path=output, split_id=split_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
