"""Cross-fold empirical conformal scaling for the original MC-dropout output.

For each held-out fold, the other four folds determine the scale multiplier.
The held-out fold is used only for evaluation. The original Gaussian MC-dropout
intervals are retained as an uncalibrated reference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def conformal_quantile(scores: np.ndarray, level: float) -> float:
    """Finite-sample corrected empirical quantile using the higher order statistic."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        raise ValueError("No finite calibration scores")
    rank = min(int(np.ceil((len(scores) + 1) * level)), len(scores))
    return float(np.partition(scores, rank - 1)[rank - 1])


def add_rows(rows, *, dataset, fold, subjects, prediction, reference,
             half_width, level, method, multiplier, std_floor, protocol):
    covered = np.abs(reference - prediction) <= half_width
    width = 2.0 * half_width
    for subject in np.unique(subjects):
        mask = subjects == subject
        rows.append({
            "dataset": dataset,
            "fold": fold,
            "subject": str(subject),
            "nominal_coverage": level,
            "method": method,
            "picp": float(covered[mask].mean()),
            "mpiw_uv": float(width[mask].mean()),
            "rmse_uv": float(np.sqrt(np.mean((prediction[mask] - reference[mask]) ** 2))),
            "calibration_multiplier": multiplier,
            "std_floor_uv": std_floor,
            "protocol": protocol,
            "source": "original_project_mc50",
        })


def save_calibration_figure(summary: pd.DataFrame, output: Path, dataset: str) -> None:
    labels = {
        "uncalibrated_mc_dropout": "MC dropout (uncalibrated)",
        "cross_fold_conformal_scaled": "Cross-fold conformal scaling",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    axes[0].plot([.5, .99], [.5, .99], color="0.55", linestyle="--", label="Ideal")
    for method, group in summary.groupby("method"):
        group = group.sort_values("nominal_coverage")
        axes[0].plot(group["nominal_coverage"], group["picp_mean"], marker="o",
                     label=labels.get(method, method))
        axes[1].plot(group["nominal_coverage"], group["mpiw_mean_uv"], marker="o",
                     label=labels.get(method, method))
    axes[0].set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(.48, 1.0), ylim=(0, 1.0))
    axes[1].set(xlabel="Nominal coverage", ylabel="Mean interval width (uV)", xlim=(.48, 1.0))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{dataset.upper()} uncertainty calibration")
    fig.savefig(output / "uq_calibration_coverage_width.png", dpi=300)
    fig.savefig(output / "uq_calibration_coverage_width.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=PATHS)
    parser.add_argument("--input", type=Path, help="Optional MC50 NPZ overriding the legacy artifact")
    parser.add_argument("--protocol", choices=("revision", "legacy"),
                        help="Required with --input for HCI; defaults to the artifact's original protocol")
    parser.add_argument(
        "--std-floor-percentile", type=float, default=1.0,
        help="Calibration-fold percentile used as a numerical floor for MC standard deviations.",
    )
    args = parser.parse_args()
    if not 0 <= args.std_floor_percentile < 100:
        parser.error("--std-floor-percentile must be in [0, 100)")

    path = args.input.resolve() if args.input else PATHS[args.dataset]
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        prediction = archive["fake_fp"].astype(np.float32, copy=False)
        uncertainty = archive["fake_std"].astype(np.float32, copy=False)
        reference = archive["real_fp"].astype(np.float32, copy=False)

    protocol = args.protocol or ("legacy" if args.dataset == "hci" and not args.input else "revision")
    canonical, subjects, _ = canonical_targets(args.dataset, protocol)
    if prediction.shape != uncertainty.shape or reference.shape != canonical.shape:
        raise ValueError(f"Unexpected MC50 shapes in {path}")
    if not np.array_equal(reference, canonical):
        raise ValueError(f"MC50 reference does not match E0 {protocol} canonical data")
    if not all(np.isfinite(array).all() for array in (prediction, uncertainty, reference)):
        raise ValueError(f"Non-finite values in {path}")
    if np.any(uncertainty < 0):
        raise ValueError(f"Negative predictive standard deviation in {path}")

    folds = fold_ids(args.dataset, len(prediction))
    rows = []
    calibration_rows = []
    for fold in range(1, 6):
        calibration_mask = folds != fold
        evaluation_mask = folds == fold
        calibration_std = uncertainty[calibration_mask]
        positive_std = calibration_std[calibration_std > 0]
        if not len(positive_std):
            raise ValueError(f"Fold {fold} has no positive calibration uncertainties")
        std_floor = float(np.percentile(positive_std, args.std_floor_percentile))
        scaled_std = np.maximum(uncertainty, std_floor)
        calibration_scores = (
            np.abs(reference[calibration_mask] - prediction[calibration_mask])
            / scaled_std[calibration_mask]
        )

        for level in LEVELS:
            z_score = float(norm.ppf((1.0 + level) / 2.0))
            multiplier = conformal_quantile(calibration_scores, float(level))
            calibration_rows.append({
                "dataset": args.dataset, "evaluation_fold": fold,
                "nominal_coverage": level, "calibration_multiplier": multiplier,
                "std_floor_uv": std_floor,
                "calibration_point_count": int(calibration_scores.size),
                "protocol": protocol,
            })
            common = dict(
                dataset=args.dataset, fold=fold,
                subjects=subjects[evaluation_mask], prediction=prediction[evaluation_mask],
                reference=reference[evaluation_mask], level=float(level), protocol=protocol,
            )
            add_rows(
                rows, **common, half_width=z_score * uncertainty[evaluation_mask],
                method="uncalibrated_mc_dropout", multiplier=z_score,
                std_floor=0.0,
            )
            add_rows(
                rows, **common, half_width=multiplier * scaled_std[evaluation_mask],
                method="cross_fold_conformal_scaled", multiplier=multiplier,
                std_floor=std_floor,
            )

    output_name = (
        "cross_fold_conformal_revision"
        if args.input and protocol == "revision" else "cross_fold_conformal"
    )
    output = E5_ROOT / "outputs" / args.dataset / output_name
    output.mkdir(parents=True, exist_ok=True)
    subject = pd.DataFrame(rows)
    subject.to_csv(output / "uq_cross_conformal_subject.csv", index=False)
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(output / "uq_cross_conformal_calibration.csv", index=False)

    fold = subject.groupby(["method", "fold", "nominal_coverage"], as_index=False).agg(
        picp=("picp", "mean"), mpiw_uv=("mpiw_uv", "mean"),
        rmse_uv=("rmse_uv", "mean"),
        calibration_multiplier=("calibration_multiplier", "first"),
        std_floor_uv=("std_floor_uv", "first"), subject_count=("subject", "nunique"),
    )
    fold.to_csv(output / "uq_cross_conformal_fold.csv", index=False)
    summary = subject.groupby(["method", "nominal_coverage"], as_index=False).agg(
        picp_mean=("picp", "mean"), picp_std=("picp", "std"),
        mpiw_mean_uv=("mpiw_uv", "mean"), mpiw_std_uv=("mpiw_uv", "std"),
        rmse_mean_uv=("rmse_uv", "mean"), subject_count=("subject", "nunique"),
    )
    summary.to_csv(output / "uq_cross_conformal_summary.csv", index=False)
    summary.to_csv(output / "uq_uncalibrated_vs_conformal.csv", index=False)
    save_calibration_figure(summary, output, args.dataset)
    print(f"Saved cross-fold empirical calibration results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
