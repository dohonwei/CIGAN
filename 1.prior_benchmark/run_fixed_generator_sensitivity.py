"""Run five priors through one frozen original CIGAN checkpoint without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from original_cigan_model import OriginalCIGANGenerator


E1_ROOT = Path(__file__).resolve().parents[0]
PROJECT_ROOT = E1_ROOT.parent
DATA_PATH = PROJECT_ROOT / "0.preprocessing/canonical_data/deap_canonical_30s_128hz.npz"
SOURCE_INDICES_PATH = PROJECT_ROOT / "causality_gpu/deap/other_indices.npy"
CHECKPOINT = PROJECT_ROOT / "comparison/deap/CIGAN_G.pth"
REFERENCE = PROJECT_ROOT / "comparison/deap/CIGAN_generated_data_single.npz"
PRIOR_ROOT = E1_ROOT / "fixed_generator/priors/deap"
OUTPUT_ROOT = E1_ROOT / "fixed_generator/predictions/deap"
PRIORS = ("granger", "pearson", "mutual_info", "random", "uniform")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(model, sources, prior, sampling_rate, batch_size, device):
    outputs = []
    prior_tensor = torch.from_numpy(prior).to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sources), batch_size):
            batch = torch.from_numpy(sources[start:start + batch_size]).to(device)
            outputs.append(model(batch, prior_tensor, sampling_rate).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="deap", choices=("deap",))
    parser.add_argument("--priors", nargs="+", default=list(PRIORS), choices=PRIORS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-replay-check", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        eeg = archive["eeg"].astype(np.float32, copy=False)
        subjects = archive["subjects"].astype(str)
        sampling_rate = float(archive["sampling_rate_hz"])
    source_indices = np.load(SOURCE_INDICES_PATH).astype(np.int64)
    if source_indices.shape != (30,) or set(source_indices) != set(range(32)) - {0, 16}:
        raise ValueError(f"Unexpected original source indices: {source_indices}")
    sources, real = eeg[:, source_indices], eeg[:, [0, 16]]

    model = OriginalCIGANGenerator(in_channels=30, hidden=64).to(device)
    state = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    checkpoint_hash = sha256(CHECKPOINT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    predictions = {}
    for prior_name in args.priors:
        prior_path = PRIOR_ROOT / f"{prior_name}_seed_{args.seed}.npz"
        if not prior_path.exists():
            raise FileNotFoundError(f"Missing prior {prior_path}; run compute_fixed_generator_priors.py first")
        with np.load(prior_path, allow_pickle=False) as archive:
            prior = archive["weights"].astype(np.float32)
        output = OUTPUT_ROOT / f"{prior_name}_seed_{args.seed}.npz"
        if output.exists() and not args.overwrite:
            with np.load(output, allow_pickle=False) as archive:
                predictions[prior_name] = archive["fake_fp"]
            print(f"Reuse existing prediction: {output}")
            continue
        fake = predict(model, sources, prior, sampling_rate, args.batch_size, device)
        predictions[prior_name] = fake
        np.savez_compressed(
            output, fake_fp=fake, real_fp=real, subjects=subjects,
            source_indices=source_indices, prior=prior, prior_name=np.asarray(prior_name),
            seed=np.int64(args.seed), sampling_rate_hz=np.float32(sampling_rate),
            window_seconds=np.float32(30.0), checkpoint_sha256=np.asarray(checkpoint_hash),
            analysis=np.asarray("fixed-generator prior substitution sensitivity"),
        )
        print(f"Saved: {output}")

    replay = {}
    if not args.skip_replay_check:
        if "granger" not in predictions:
            with np.load(PRIOR_ROOT / f"granger_seed_{args.seed}.npz", allow_pickle=False) as archive:
                granger = archive["weights"].astype(np.float32)
            predictions["granger"] = predict(model, sources, granger, sampling_rate, args.batch_size, device)
        with np.load(REFERENCE, allow_pickle=False) as archive:
            stored = archive["fake_fp"].astype(np.float32, copy=False)
        generated = predictions["granger"]
        replay = {
            "reference": str(REFERENCE.relative_to(PROJECT_ROOT)),
            "mae_uv": float(np.mean(np.abs(generated - stored))),
            "max_abs_diff_uv": float(np.max(np.abs(generated - stored))),
            "pearson": float(np.corrcoef(generated.ravel(), stored.ravel())[0, 1]),
        }
        # The exact original generation may differ slightly across CUDA implementations.
        if replay["pearson"] < 0.99999 or replay["mae_uv"] > 0.01:
            raise RuntimeError(f"Frozen-checkpoint replay failed: {replay}")
        print(f"Replay verified: {replay}")
    metadata = {
        "analysis": "fixed-generator prior substitution sensitivity",
        "dataset": "deap", "protocol": "128 Hz/30 s/3840 samples",
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_hash, "priors": args.priors,
        "seed": args.seed, "replay_check": replay,
        "interpretation_limit": "Inference sensitivity only; not independently retrained prior models.",
    }
    (OUTPUT_ROOT / f"run_seed_{args.seed}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
