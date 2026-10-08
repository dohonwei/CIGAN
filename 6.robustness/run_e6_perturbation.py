"""Evaluate pretrained DEAP models under controlled source-channel noise."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from tqdm import tqdm

HERE = Path(__file__).resolve()
E6_ROOT = HERE.parents[0]
REVISION_ROOT = E6_ROOT.parent
E1_CODE = REVISION_ROOT / "1.prior_benchmark"
E4_CODE = REVISION_ROOT / "4.baselines"
sys.path[:0] = [str(E1_CODE), str(E4_CODE)]

from e1_common import (DATASETS, FRONTAL_INDICES, SOURCE_INDICES,
                       checkpoint_path as cigan_checkpoint_path,
                       load_dataset, load_fold,
                       prediction_path as cigan_prediction_path,
                       set_deterministic_seed)
from e1_model import CIGANGenerator
from e4_common import (checkpoint_path as baseline_checkpoint_path,
                       make_generator,
                       prediction_path as baseline_prediction_path)

DEFAULT_MODELS = ("cigan", "hveegnet", "tieeegnet", "wavenet", "encoder_decoder")
NOISE_TYPES = ("white", "pink", "impulsive")
METRICS = ("pearson", "rmse", "mae", "output_snr_db", "log_psd_rmse")


def add_noise(sources: np.ndarray, kind: str, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if kind == "white":
        noise = rng.normal(size=sources.shape)
    elif kind == "pink":
        frequencies = np.fft.rfftfreq(sources.shape[-1])
        scale = np.zeros_like(frequencies)
        scale[1:] = 1.0 / np.sqrt(frequencies[1:])
        spectrum = (rng.normal(size=(*sources.shape[:-1], len(frequencies))) +
                    1j * rng.normal(size=(*sources.shape[:-1], len(frequencies))))
        noise = np.fft.irfft(spectrum * scale, n=sources.shape[-1], axis=-1)
    elif kind == "impulsive":
        noise = np.zeros_like(sources, dtype=np.float64)
        count = max(1, sources.shape[-1] // 100)
        for trial in range(sources.shape[0]):
            for channel in range(sources.shape[1]):
                indices = rng.choice(sources.shape[-1], count, replace=False)
                noise[trial, channel, indices] = rng.normal(0.0, 10.0, count)
    else:
        raise ValueError(kind)
    noise -= noise.mean(axis=-1, keepdims=True)
    noise /= noise.std(axis=-1, keepdims=True) + 1e-12
    power = np.mean(np.square(sources, dtype=np.float64), axis=-1, keepdims=True)
    noisy = sources + noise * np.sqrt(power / (10.0 ** (snr_db / 10.0)))
    return noisy.astype(np.float32)


def load_model(model_name: str, fold: int, args, device: torch.device):
    if model_name == "cigan":
        path = cigan_checkpoint_path("deap", fold, "granger", args.seed,
                                     args.cigan_run_tag)
        archive_path = cigan_prediction_path("deap", fold, "granger", args.seed,
                                             args.cigan_run_tag)
        model = CIGANGenerator()
    else:
        path = baseline_checkpoint_path("deap", fold, model_name, args.seed,
                                        args.baseline_run_tag)
        archive_path = baseline_prediction_path("deap", fold, model_name, args.seed,
                                                args.baseline_run_tag)
        model = make_generator(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing formal prediction for replay check: {archive_path}")
    # These are locally generated project checkpoints containing NumPy metadata.
    # PyTorch 2.6 defaults to weights_only=True, which rejects that metadata.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch releases that lack weights_only.
        checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["generator_ema"])
    model.to(device).eval()
    mean = np.asarray(checkpoint["channel_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["channel_std"], dtype=np.float32)
    prior = None
    if model_name == "cigan":
        prior_path = Path(checkpoint["prior_path"])
        if not prior_path.exists():
            prior_path = (REVISION_ROOT / "1.prior_benchmark" / "priors" / "deap" /
                          f"fold_{fold}" / f"granger_seed_{args.seed}.npz")
        if not prior_path.exists():
            raise FileNotFoundError(f"Missing Granger prior: {prior_path}")
        with np.load(prior_path, allow_pickle=False) as prior_archive:
            prior = torch.from_numpy(prior_archive["weights"].astype(np.float32)).to(device)
    return model, mean, std, prior, archive_path


@torch.no_grad()
def infer(model, model_name: str, standardized_sources: np.ndarray,
          prior: torch.Tensor | None, device: torch.device, batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(standardized_sources), batch_size):
        batch = torch.from_numpy(standardized_sources[start:start + batch_size]).to(device)
        output = model(batch, prior, DATASETS["deap"]["fs"]) if model_name == "cigan" else model(batch)
        chunks.append(output.cpu().numpy())
    return np.concatenate(chunks)


def metric_row(reference: np.ndarray, prediction: np.ndarray, fs: float) -> dict:
    r, p = reference.astype(np.float64), prediction.astype(np.float64)
    error = r - p
    denominator = np.linalg.norm(r - r.mean()) * np.linalg.norm(p - p.mean())
    pearson = float(np.dot(r - r.mean(), p - p.mean()) / denominator) if denominator else np.nan
    frequency, psd_r = welch(r, fs=fs, nperseg=min(len(r), int(2 * fs)))
    _, psd_p = welch(p, fs=fs, nperseg=min(len(p), int(2 * fs)))
    selected = (frequency >= 4.0) & (frequency <= 45.0)
    return {
        "pearson": pearson,
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "output_snr_db": float(10.0 * np.log10((np.mean(r ** 2) + 1e-12) /
                                                (np.mean(error ** 2) + 1e-12))),
        "log_psd_rmse": float(np.sqrt(np.mean((np.log10(psd_r[selected] + 1e-12) -
                                                np.log10(psd_p[selected] + 1e-12)) ** 2))),
    }


def plot_curves(subject: pd.DataFrame, output: Path, models: list[str]) -> None:
    labels = {"cigan": "CIGAN-Granger", "hveegnet": "hvEEGNet",
              "tieeegnet": "TIE-EEGNet", "wavenet": "WaveNet",
              "encoder_decoder": "Encoder-decoder"}
    colors = {"cigan": "#D55E00", "hveegnet": "#0072B2",
              "tieeegnet": "#009E73", "wavenet": "#CC79A7",
              "encoder_decoder": "#555555"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    specs = (("pearson", "Pearson correlation"), ("rmse", "RMSE (uV)"),
             ("mae", "MAE (uV)"), ("log_psd_rmse", "Log-PSD RMSE"))
    pooled = subject.groupby(["model", "snr_db"], as_index=False)[list(METRICS)].mean()
    for ax, (metric, ylabel) in zip(axes.flat, specs):
        for model_name in models:
            subset = pooled[pooled.model == model_name].sort_values("snr_db")
            ax.plot(subset.snr_db, subset[metric], marker="o", linewidth=1.7,
                    label=labels[model_name], color=colors[model_name])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    for ax in axes[-1]:
        ax.set_xlabel("Input SNR (dB)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(output / "e6_deap_input_noise_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / "e6_deap_input_noise_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS,
                        default=list(DEFAULT_MODELS))
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 6),
                        default=list(range(1, 6)))
    parser.add_argument("--noise-types", nargs="+", choices=NOISE_TYPES,
                        default=list(NOISE_TYPES))
    parser.add_argument("--snr-db", nargs="+", type=float, default=[20, 10, 5, 0, -5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cigan-run-tag", default="fixed_prior_welch_e100")
    parser.add_argument("--baseline-run-tag", default="e4_stabilized_e100")
    parser.add_argument("--output-tag", default="formal_input_noise")
    parser.add_argument("--replay-atol", type=float, default=0.5,
                        help="Maximum pointwise clean-replay error in uV")
    parser.add_argument("--replay-rmse-atol", type=float, default=1e-2,
                        help="Maximum clean-replay RMSE in uV")
    parser.add_argument("--replay-min-corr", type=float, default=0.99999,
                        help="Minimum correlation with archived clean prediction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    set_deterministic_seed(args.seed)
    device = torch.device(args.device)
    data = load_dataset("deap")
    fs = DATASETS["deap"]["fs"]
    rows = []
    output = E6_ROOT / "outputs" / args.output_tag
    output.mkdir(parents=True, exist_ok=True)

    for fold in args.folds:
        train_idx, test_idx = load_fold("deap", fold)
        if np.intersect1d(np.unique(data["subjects"][train_idx]),
                          np.unique(data["subjects"][test_idx])).size:
            raise ValueError(f"Subject leakage in fold {fold}")
        raw_sources = data["eeg"][test_idx][:, SOURCE_INDICES].astype(np.float32)
        references = data["eeg"][test_idx][:, FRONTAL_INDICES].astype(np.float32)
        loaded = {name: load_model(name, fold, args, device) for name in args.models}

        for name, (model, mean, std, prior, archive_path) in loaded.items():
            standardized = ((raw_sources - mean[SOURCE_INDICES][None, :, None]) /
                            std[SOURCE_INDICES][None, :, None]).astype(np.float32)
            clean_standardized = infer(model, name, standardized, prior, device, args.batch_size)
            clean_uv = (clean_standardized * std[FRONTAL_INDICES][None, :, None] +
                        mean[FRONTAL_INDICES][None, :, None])
            with np.load(archive_path, allow_pickle=False) as archive:
                archived = archive["fake_uv"]
                archived_idx = archive["test_idx"]
            if not np.array_equal(archived_idx, test_idx):
                raise ValueError(f"Test-index mismatch: {archive_path}")
            replay_error = clean_uv - archived
            max_error = float(np.max(np.abs(replay_error)))
            replay_rmse = float(np.sqrt(np.mean(replay_error.astype(np.float64) ** 2)))
            replay_corr = float(np.corrcoef(clean_uv.ravel(), archived.ravel())[0, 1])
            if (max_error > args.replay_atol or replay_rmse > args.replay_rmse_atol or
                    replay_corr < args.replay_min_corr):
                raise ValueError(
                    f"Checkpoint replay failed for {name} fold {fold}: "
                    f"max_abs={max_error:g}, rmse={replay_rmse:g}, corr={replay_corr:g}"
                )
            print(
                f"Replay OK: {name} fold {fold}, max_abs={max_error:.3g} uV, "
                f"rmse={replay_rmse:.3g} uV, corr={replay_corr:.9f}",
                flush=True,
            )

        conditions = [(kind, snr) for kind in args.noise_types for snr in args.snr_db]
        for condition_id, (kind, snr) in enumerate(tqdm(conditions, desc=f"DEAP fold {fold}")):
            noisy_sources = add_noise(raw_sources, kind, snr,
                                      args.seed + fold * 1000 + condition_id)
            for name, (model, mean, std, prior, _) in loaded.items():
                standardized = ((noisy_sources - mean[SOURCE_INDICES][None, :, None]) /
                                std[SOURCE_INDICES][None, :, None]).astype(np.float32)
                predicted_standardized = infer(model, name, standardized, prior,
                                               device, args.batch_size)
                predictions = (predicted_standardized * std[FRONTAL_INDICES][None, :, None] +
                               mean[FRONTAL_INDICES][None, :, None])
                for trial in range(len(test_idx)):
                    for channel, channel_name in enumerate(("Fp1", "Fp2")):
                        row = {"dataset": "deap", "fold": fold, "model": name,
                               "noise_type": kind, "snr_db": snr,
                               "global_trial_idx": int(test_idx[trial]),
                               "subject": str(data["subjects"][test_idx[trial]]),
                               "channel": channel_name}
                        row.update(metric_row(references[trial, channel],
                                              predictions[trial, channel], fs))
                        rows.append(row)

    raw = pd.DataFrame(rows)
    raw.to_csv(output / "trial_channel_metrics.csv", index=False)
    subject = raw.groupby(["dataset", "fold", "model", "noise_type", "snr_db", "subject"],
                          as_index=False)[list(METRICS)].mean()
    subject.to_csv(output / "subject_metrics.csv", index=False)
    summary = subject.groupby(["dataset", "model", "noise_type", "snr_db"])[list(METRICS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "summary_metrics.csv", index=False)
    plot_curves(subject, output, args.models)
    print(f"Saved E6 real-EEG input-noise results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
