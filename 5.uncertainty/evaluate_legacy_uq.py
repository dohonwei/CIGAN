"""Recompute calibration/coverage-width from validated original DEAP MC50 output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

E5_ROOT = Path(__file__).resolve().parents[0]
REVISION_ROOT = E5_ROOT.parent
sys.path.insert(0, str(REVISION_ROOT / "code"))
from legacy_artifacts import PROJECT_ROOT, canonical_targets, fold_ids


LEVELS = np.asarray([.50, .60, .70, .80, .90, .95, .99])
PATHS = {
    "deap": PROJECT_ROOT / "comparison/deap/CIGAN_generated_data_mc50.npz",
    "hci": PROJECT_ROOT / "comparison/hci/CIGAN_generated_data_mc50.npz",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=PATHS)
    args = parser.parse_args()
    path = PATHS[args.dataset]
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["fake_fp"].astype(np.float32, copy=False)
        uncertainty = archive["fake_std"].astype(np.float32, copy=False)
        reference = archive["real_fp"].astype(np.float32, copy=False)
    protocol = "legacy" if args.dataset == "hci" else "revision"
    canonical, subjects, _ = canonical_targets(args.dataset, protocol)
    if prediction.shape != uncertainty.shape or reference.shape != canonical.shape:
        raise ValueError(f"Unexpected MC50 shapes in {path}")
    if not np.array_equal(reference, canonical):
        raise ValueError(
            f"Original {args.dataset} MC50 reference does not match E0 {protocol} canonical data; "
            "legacy UQ reuse is prohibited"
        )
    if not np.isfinite(prediction).all() or not np.isfinite(uncertainty).all():
        raise ValueError(f"Non-finite MC50 output in {path}")
    if np.any(uncertainty < 0):
        raise ValueError(f"Negative predictive standard deviation in {path}")
    folds = fold_ids(args.dataset, len(prediction))
    rows = []
    for level in LEVELS:
        z_score = norm.ppf((1 + level) / 2)
        lower, upper = prediction - z_score * uncertainty, prediction + z_score * uncertainty
        covered, width = (reference >= lower) & (reference <= upper), upper - lower
        for subject in np.unique(subjects):
            mask = subjects == subject
            subject_folds = np.unique(folds[mask])
            if len(subject_folds) != 1:
                raise ValueError(f"Subject {subject} occurs in multiple test folds")
            rows.append({
                "dataset": args.dataset, "fold": int(subject_folds[0]), "subject": str(subject),
                "nominal_coverage": level, "picp": float(covered[mask].mean()),
                "mpiw_uv": float(width[mask].mean()),
                "rmse_uv": float(np.sqrt(np.mean((prediction[mask] - reference[mask]) ** 2))),
                "source": "original_project_mc50", "protocol": protocol,
            })
    output = E5_ROOT / "outputs" / args.dataset / "legacy_reuse"
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows); raw.to_csv(output / "uq_subject_calibration.csv", index=False)
    raw.groupby("nominal_coverage").agg(
        picp_mean=("picp", "mean"), picp_std=("picp", "std"),
        mpiw_mean_uv=("mpiw_uv", "mean"), mpiw_std_uv=("mpiw_uv", "std"),
        rmse_mean_uv=("rmse_uv", "mean"),
    ).reset_index().to_csv(output / "uq_calibration_summary.csv", index=False)
    print(f"Saved validated legacy E5 results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
