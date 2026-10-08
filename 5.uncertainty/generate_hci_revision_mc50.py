"""Generate revision-protocol MC-dropout predictions for DEAP or HCI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

HERE = Path(__file__).resolve()
E5_ROOT = HERE.parents[0]
E1_ROOT = E5_ROOT.parent / "1.prior_benchmark"
sys.path.insert(0, str(E1_ROOT))
from e1_common import (
    DATASETS, FRONTAL_INDICES, SOURCE_INDICES, checkpoint_path, load_dataset,
    load_fold, prior_path, set_deterministic_seed, transform_eeg,
)
from e1_model import CIGANGenerator


def enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout1d):
            module.train()
            count += 1
    if count != 8:
        raise ValueError(f"Expected 8 Dropout1d layers, found {count}")


def generate_fold(args: argparse.Namespace, fold: int, data: dict[str, np.ndarray],
                  output_dir: Path, device: torch.device) -> Path:
    output_path = output_dir / f"fold_{fold}_mc{args.passes}.npz"
    if output_path.exists() and not args.overwrite:
        print(f"Reusing {output_path}")
        return output_path

    checkpoint = checkpoint_path(args.dataset, fold, "granger", args.seed, args.run_tag)
    prior_file = prior_path(args.dataset, fold, "granger", args.seed)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing formal {args.dataset.upper()} checkpoint: {checkpoint}\n"
            f"Complete this E1 {args.dataset.upper()} fold or pass the matching --run-tag."
        )
    if not prior_file.exists():
        raise FileNotFoundError(prior_file)

    _, test_idx = load_fold(args.dataset, fold)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    required = {"generator_ema", "channel_mean", "channel_std"}
    missing = required.difference(state)
    if missing:
        raise KeyError(f"{checkpoint} is missing keys: {sorted(missing)}")
    mean = np.asarray(state["channel_mean"], dtype=np.float32)
    std = np.asarray(state["channel_std"], dtype=np.float32)
    sources = transform_eeg(data["eeg"][test_idx], mean, std)[:, SOURCE_INDICES]
    real = data["eeg"][test_idx][:, FRONTAL_INDICES].astype(np.float32, copy=False)
    with np.load(prior_file, allow_pickle=False) as archive:
        prior = archive["weights"].astype(np.float32, copy=False)

    model = CIGANGenerator().to(device)
    model.load_state_dict(state["generator_ema"])
    enable_mc_dropout(model)
    prior_tensor = torch.from_numpy(prior).to(device)
    draws = []
    with torch.inference_mode():
        for _ in tqdm(range(args.passes), desc=f"{args.dataset.upper()} fold {fold} MC passes"):
            batches = []
            for start in range(0, len(test_idx), args.batch_size):
                batch = torch.from_numpy(sources[start:start + args.batch_size]).to(device)
                batches.append(model(batch, prior_tensor, DATASETS[args.dataset]["fs"]).cpu().numpy())
            draws.append(np.concatenate(batches, axis=0))
    draws = np.stack(draws, axis=0)
    frontal_mean = mean[FRONTAL_INDICES][None, None, :, None]
    frontal_std = std[FRONTAL_INDICES][None, None, :, None]
    draws = draws * frontal_std + frontal_mean
    prediction = draws.mean(axis=0, dtype=np.float64).astype(np.float32)
    uncertainty = draws.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
    if not np.isfinite(prediction).all() or not np.isfinite(uncertainty).all():
        raise ValueError(f"Non-finite MC output in fold {fold}")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, fake_fp=prediction, fake_std=uncertainty, real_fp=real,
        test_idx=test_idx, subjects=data["subjects"][test_idx].astype(str),
        fold=np.int64(fold), sampling_rate_hz=np.float32(DATASETS[args.dataset]["fs"]),
        window_seconds=np.float32(data["eeg"].shape[-1] / DATASETS[args.dataset]["fs"]),
        sample_count=np.int64(data["eeg"].shape[-1]), dataset=args.dataset,
        passes=np.int64(args.passes), seed=np.int64(args.seed), run_tag=args.run_tag,
        checkpoint=str(checkpoint), protocol="revision",
    )
    print(f"Saved {output_path}")
    return output_path


def merge_folds(output_dir: Path, dataset: str, passes: int,
                expected_samples: int, sampling_rate: float, sample_count: int) -> Path | None:
    paths = [output_dir / f"fold_{fold}_mc{passes}.npz" for fold in range(1, 6)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print("Merge deferred; missing fold outputs:")
        print("\n".join(missing))
        return None
    parts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            parts.append({key: archive[key] for key in ("fake_fp", "fake_std", "real_fp", "test_idx", "subjects")})
    indices = np.concatenate([part["test_idx"] for part in parts])
    if len(indices) != expected_samples or not np.array_equal(np.sort(indices), np.arange(expected_samples)):
        raise ValueError(f"Fold MC outputs do not cover the {dataset.upper()} canonical dataset exactly once")
    order = np.argsort(indices)
    merged = output_dir / f"{dataset}_revision_mc{passes}.npz"
    np.savez_compressed(
        merged,
        fake_fp=np.concatenate([part["fake_fp"] for part in parts])[order],
        fake_std=np.concatenate([part["fake_std"] for part in parts])[order],
        real_fp=np.concatenate([part["real_fp"] for part in parts])[order],
        test_idx=indices[order],
        subjects=np.concatenate([part["subjects"] for part in parts])[order],
        sampling_rate_hz=np.float32(sampling_rate),
        window_seconds=np.float32(sample_count / sampling_rate),
        sample_count=np.int64(sample_count), passes=np.int64(passes),
        dataset=dataset, protocol="revision",
    )
    print(f"Saved merged MC output: {merged}")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="hci")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-tag", default="fixed_prior_welch_e100")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if any(fold not in range(1, 6) for fold in args.folds):
        parser.error("--folds must contain values in 1..5")
    if args.passes < 2 or args.batch_size < 1:
        parser.error("--passes must be >=2 and --batch-size must be positive")

    set_deterministic_seed(args.seed)
    data = load_dataset(args.dataset)
    expected_fs = DATASETS[args.dataset]["fs"]
    if data["eeg"].shape[1:] != (32, 3840) or not np.isclose(data["sampling_rate_hz"], expected_fs):
        raise ValueError(
            f"{args.dataset.upper()} input is not the revision {expected_fs:g} Hz, "
            "3840-sample dataset"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_seconds = data["eeg"].shape[-1] / expected_fs
    print(
        f"Device: {device}; {args.dataset.upper()} shape: {data['eeg'].shape}; "
        f"fs={expected_fs:g} Hz; window={window_seconds:g} s"
    )
    output_dir = E5_ROOT / "outputs" / args.dataset / "revision_mc50"
    for fold in args.folds:
        generate_fold(args, fold, data, output_dir, device)
    merge_folds(
        output_dir, args.dataset, args.passes, len(data["eeg"]),
        expected_fs, data["eeg"].shape[-1]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
