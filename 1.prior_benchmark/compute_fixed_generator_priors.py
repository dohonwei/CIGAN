"""Compute global priors for the frozen original-CIGAN sensitivity analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compute_priors import apply_csd, mutual_information_prior, pearson_prior
from e1_common import CHANNEL_NAMES, FRONTAL_INDICES, SOURCE_INDICES, normalize_nonnegative


E1_ROOT = Path(__file__).resolve().parents[0]
PROJECT_ROOT = E1_ROOT.parent
DATA_PATH = PROJECT_ROOT / "0.preprocessing/canonical_data/deap_canonical_30s_128hz.npz"
ORIGINAL_GRANGER = PROJECT_ROOT / "causality_gpu/deap/frontal_weights.npy"
OUTPUT_ROOT = E1_ROOT / "fixed_generator/priors/deap"
PRIORS = ("granger", "pearson", "mutual_info", "random", "uniform")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="deap", choices=("deap",))
    parser.add_argument("--priors", nargs="+", default=list(PRIORS), choices=PRIORS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-mi-samples", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        eeg = archive["eeg"].astype(np.float32, copy=False)
        sampling_rate = float(archive["sampling_rate_hz"])
    csd = None
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for prior_name in args.priors:
        output = OUTPUT_ROOT / f"{prior_name}_seed_{args.seed}.npz"
        if output.exists() and not args.overwrite:
            print(f"Reuse existing prior: {output}")
            continue
        details = {}
        if prior_name == "granger":
            weights = np.load(ORIGINAL_GRANGER).astype(np.float32)
            raw_scores = weights.astype(np.float64)
            details["source"] = str(ORIGINAL_GRANGER.relative_to(PROJECT_ROOT))
        elif prior_name in ("pearson", "mutual_info"):
            if csd is None:
                csd = apply_csd(eeg, sampling_rate)
            if prior_name == "pearson":
                raw_scores, _ = pearson_prior(csd)
            else:
                raw_scores, mi_details = mutual_information_prior(csd, args.seed, args.max_mi_samples)
                details.update(mi_details)
            weights = normalize_nonnegative(raw_scores)
        elif prior_name == "random":
            raw_scores = np.random.default_rng(args.seed).random(len(SOURCE_INDICES))
            weights = normalize_nonnegative(raw_scores)
        else:
            raw_scores = np.ones(len(SOURCE_INDICES), dtype=np.float64)
            weights = normalize_nonnegative(raw_scores)
        if weights.shape != (30,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError(f"Invalid {prior_name} weights")
        if not np.isclose(weights.sum(), 1.0, atol=1e-5):
            raise ValueError(f"{prior_name} weights do not sum to one: {weights.sum()}")
        np.savez(
            output, weights=weights.astype(np.float32), raw_scores=np.asarray(raw_scores),
            source_indices=SOURCE_INDICES, source_names=CHANNEL_NAMES[SOURCE_INDICES],
            frontal_indices=FRONTAL_INDICES, dataset=np.asarray("deap"),
            protocol=np.asarray("128 Hz/30 s/3840 samples"), seed=np.int64(args.seed),
        )
        metadata = {
            "analysis": "fixed-generator prior substitution sensitivity",
            "dataset": "deap", "protocol": "128 Hz/30 s/3840 samples",
            "prior": prior_name, "seed": args.seed, "weights_sum": float(weights.sum()),
            "data_sha256": file_sha256(DATA_PATH), **details,
        }
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
