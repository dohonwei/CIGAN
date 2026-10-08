"""Evaluate whether MC-dropout dispersion identifies high-error EEG windows."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

E5_ROOT = Path(__file__).resolve().parents[0]
REVISION_ROOT = E5_ROOT.parent
sys.path.insert(0, str(REVISION_ROOT / "code"))
from legacy_artifacts import PROJECT_ROOT, canonical_targets, fold_ids


PATHS = {
    "deap": PROJECT_ROOT / "comparison/deap/CIGAN_generated_data_mc50.npz",
    "hci": PROJECT_ROOT / "comparison/hci/CIGAN_generated_data_mc50.npz",
}
COVERAGES = np.asarray([.10, .20, .30, .40, .50, .60, .70, .80, .90, 1.00])


def bootstrap_subject_mean(values: np.ndarray, rng: np.random.Generator,
                           repetitions: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [.025, .975]))


def safe_wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not np.any(values):
        return 1.0
    return float(wilcoxon(values, alternative="greater").pvalue)


def risk_at_coverage(error: np.ndarray, order: np.ndarray, coverage: float) -> float:
    retained = max(1, int(np.ceil(len(error) * coverage)))
    return float(error[order[:retained]].mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=PATHS)
    parser.add_argument("--input", type=Path, help="Optional MC50 NPZ overriding the legacy artifact")
    parser.add_argument("--protocol", choices=("revision", "legacy"),
                        help="Required with --input for HCI; defaults to the artifact's original protocol")
    parser.add_argument("--high-error-quantile", type=float, default=.80)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.high_error_quantile < 1:
        parser.error("--high-error-quantile must be between 0 and 1")
    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")

    path = args.input.resolve() if args.input else PATHS[args.dataset]
    if not path.exists():
        if args.dataset == "hci" and args.input and args.protocol == "revision":
            raise FileNotFoundError(
                f"Missing merged revision MC file: {path}\n"
                "Run generate_hci_revision_mc50.py first. The merged file is created only "
                "after fold_1 through fold_5 MC outputs all exist."
            )
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
    if not all(np.isfinite(x).all() for x in (prediction, uncertainty, reference)):
        raise ValueError(f"Non-finite values in {path}")

    folds = fold_ids(args.dataset, len(prediction))
    window_mae = np.mean(np.abs(prediction - reference), axis=(1, 2))
    window_rmse = np.sqrt(np.mean((prediction - reference) ** 2, axis=(1, 2)))
    window_uncertainty = np.mean(uncertainty, axis=(1, 2))
    subject_rows, risk_rows = [], []

    for subject in np.unique(subjects):
        mask = subjects == subject
        error = window_mae[mask]
        error_rmse = window_rmse[mask]
        dispersion = window_uncertainty[mask]
        subject_folds = np.unique(folds[mask])
        if len(subject_folds) != 1:
            raise ValueError(f"Subject {subject} occurs in multiple folds")
        rho, rho_p = spearmanr(dispersion, error)
        high_error = error >= np.quantile(error, args.high_error_quantile)
        auc = roc_auc_score(high_error, dispersion) if len(np.unique(high_error)) == 2 else np.nan
        uncertainty_order = np.argsort(dispersion)
        oracle_order = np.argsort(error)
        full_mae = float(error.mean())
        retained_50 = risk_at_coverage(error, uncertainty_order, .50)
        subject_rows.append({
            "dataset": args.dataset, "fold": int(subject_folds[0]), "subject": str(subject),
            "window_count": int(mask.sum()), "spearman_rho": float(rho),
            "spearman_p": float(rho_p), "high_error_auroc": float(auc),
            "full_mae_uv": full_mae, "full_rmse_uv": float(error_rmse.mean()),
            "retained_50_mae_uv": retained_50,
            "relative_mae_reduction_at_50": float((full_mae - retained_50) / full_mae),
            "protocol": protocol, "source": "original_project_mc50",
        })
        for coverage in COVERAGES:
            risk_rows.extend([
                {"dataset": args.dataset, "fold": int(subject_folds[0]), "subject": str(subject),
                 "coverage": coverage, "ordering": "mc_dropout_uncertainty",
                 "risk_mae_uv": risk_at_coverage(error, uncertainty_order, coverage)},
                {"dataset": args.dataset, "fold": int(subject_folds[0]), "subject": str(subject),
                 "coverage": coverage, "ordering": "oracle_error",
                 "risk_mae_uv": risk_at_coverage(error, oracle_order, coverage)},
                {"dataset": args.dataset, "fold": int(subject_folds[0]), "subject": str(subject),
                 "coverage": coverage, "ordering": "no_rejection",
                 "risk_mae_uv": full_mae},
            ])

    subject_df = pd.DataFrame(subject_rows)
    risk_df = pd.DataFrame(risk_rows)
    rng = np.random.default_rng(args.seed)
    rho_ci = bootstrap_subject_mean(subject_df["spearman_rho"].to_numpy(), rng, args.bootstrap_repetitions)
    reduction_ci = bootstrap_subject_mean(
        subject_df["relative_mae_reduction_at_50"].to_numpy(), rng, args.bootstrap_repetitions
    )
    summary = pd.DataFrame([{
        "dataset": args.dataset, "subject_count": subject_df["subject"].nunique(),
        "window_count": len(window_mae),
        "spearman_macro_mean": subject_df["spearman_rho"].mean(),
        "spearman_macro_std": subject_df["spearman_rho"].std(),
        "spearman_bootstrap_ci_low": rho_ci[0], "spearman_bootstrap_ci_high": rho_ci[1],
        "spearman_positive_wilcoxon_p": safe_wilcoxon(subject_df["spearman_rho"].to_numpy()),
        "spearman_fisher_combined_p": combine_pvalues(subject_df["spearman_p"], method="fisher")[1],
        "high_error_auroc_macro_mean": subject_df["high_error_auroc"].mean(),
        "high_error_auroc_macro_std": subject_df["high_error_auroc"].std(),
        "relative_mae_reduction_at_50_mean": subject_df["relative_mae_reduction_at_50"].mean(),
        "relative_mae_reduction_at_50_ci_low": reduction_ci[0],
        "relative_mae_reduction_at_50_ci_high": reduction_ci[1],
        "subjects_positive_reduction_at_50": int((subject_df["relative_mae_reduction_at_50"] > 0).sum()),
        "protocol": protocol,
    }])

    # Global uncertainty deciles are descriptive only; inferential results remain subject-level.
    decile = pd.qcut(window_uncertainty, 10, labels=False, duplicates="drop") + 1
    decile_df = pd.DataFrame({
        "uncertainty_decile": decile, "window_mae_uv": window_mae,
        "window_rmse_uv": window_rmse, "window_uncertainty_uv": window_uncertainty,
    }).groupby("uncertainty_decile", as_index=False).agg(
        window_count=("window_mae_uv", "size"), mean_mae_uv=("window_mae_uv", "mean"),
        mean_rmse_uv=("window_rmse_uv", "mean"),
        mean_uncertainty_uv=("window_uncertainty_uv", "mean"),
    )

    output_name = "uq_utility_revision" if args.input and protocol == "revision" else "uq_utility"
    output = E5_ROOT / "outputs" / args.dataset / output_name
    output.mkdir(parents=True, exist_ok=True)
    subject_df.to_csv(output / "uq_utility_subject.csv", index=False)
    risk_df.to_csv(output / "uq_risk_coverage_subject.csv", index=False)
    risk_summary = risk_df.groupby(["ordering", "coverage"], as_index=False).agg(
        risk_mae_mean_uv=("risk_mae_uv", "mean"), risk_mae_std_uv=("risk_mae_uv", "std")
    )
    risk_summary.to_csv(output / "uq_risk_coverage_summary.csv", index=False)
    decile_df.to_csv(output / "uq_error_by_uncertainty_decile.csv", index=False)
    summary.to_csv(output / "uq_utility_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    labels = {"mc_dropout_uncertainty": "MC-dropout uncertainty", "oracle_error": "Oracle",
              "no_rejection": "No rejection"}
    for ordering, group in risk_summary.groupby("ordering"):
        axes[0].plot(group["coverage"], group["risk_mae_mean_uv"], marker="o",
                     label=labels[ordering])
    axes[0].set(xlabel="Retained coverage", ylabel="Retained-window MAE (uV)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(decile_df["uncertainty_decile"], decile_df["mean_mae_uv"], marker="o")
    axes[1].set(xlabel="MC-dropout uncertainty decile", ylabel="Window MAE (uV)", xticks=range(1, 11))
    fig.suptitle(f"{args.dataset.upper()} MC-dropout utility")
    fig.savefig(output / "uq_risk_ranking.png", dpi=300)
    fig.savefig(output / "uq_risk_ranking.pdf")
    plt.close(fig)
    print(summary.to_string(index=False))
    print(f"Saved UQ utility results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
