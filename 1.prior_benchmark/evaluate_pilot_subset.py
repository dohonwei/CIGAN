"""Evaluate one fixed-subject E1 pilot without treating trials as independent."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from e1_common import E1_ROOT, PRIORS, load_pilot_split, pilot_prediction_path
from evaluate_prior_benchmark import METRICS, statistical_comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="deap", choices=("deap", "hci"))
    parser.add_argument("--pilot-split", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    _, _, split_id = load_pilot_split(Path(args.pilot_split), args.dataset)
    rows = []
    prior_progress = tqdm(
        PRIORS, desc="E1 pilot priors", unit="prior", disable=args.no_progress
    )
    for prior in prior_progress:
        prior_progress.set_postfix_str(prior)
        path = pilot_prediction_path(args.dataset, split_id, prior, args.seed)
        # evaluate_file uses the standard path, so temporarily evaluate the same archive contract here.
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            fake, real = archive["fake_uv"], archive["real_uv"]
            test_idx, subjects = archive["test_idx"], archive["test_subjects"]
            fs = float(archive["sampling_rate_hz"])
        from evaluate_prior_benchmark import band_powers, cosine_similarity, pearson_correlation, dtw_distance, BANDS
        trial_progress = tqdm(
            range(len(fake)),
            desc=f"E1 pilot {prior}",
            unit="trial",
            leave=False,
            disable=args.no_progress,
        )
        for i in trial_progress:
            for channel, channel_name in enumerate(("Fp1", "Fp2")):
                generated, reference = fake[i, channel].astype(float), real[i, channel].astype(float)
                gp, rp = band_powers(generated, fs), band_powers(reference, fs)
                row = {"dataset": args.dataset, "split_id": split_id, "prior": prior,
                       "global_trial_idx": int(test_idx[i]), "subject": str(subjects[i]), "channel": channel_name,
                       "cosine": cosine_similarity(reference, generated),
                       "pearson": pearson_correlation(reference, generated),
                       "rmse": float(np.sqrt(np.mean((reference-generated)**2))),
                       "mae": float(np.mean(np.abs(reference-generated))),
                       "dtw": dtw_distance(reference, generated)}
                row.update({band: abs(rp[band]-gp[band]) for band in BANDS})
                rows.append(row)
    output = E1_ROOT / "pilot_subset/results" / args.dataset / split_id
    output.mkdir(parents=True, exist_ok=True)
    trial_df = pd.DataFrame(rows)
    trial_df.to_csv(output / "trial_channel_metrics.csv", index=False)
    subject_df = trial_df.groupby(["dataset", "split_id", "prior", "subject"], as_index=False)[METRICS].mean()
    subject_df.to_csv(output / "subject_metrics.csv", index=False)
    summary = subject_df.groupby(["dataset", "split_id", "prior"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "prior_summary.csv", index=False)
    stats = statistical_comparisons(subject_df)
    stats.insert(0, "dataset", args.dataset); stats.insert(1, "split_id", split_id)
    stats["pilot_only"] = True
    stats.to_csv(output / "granger_vs_priors_statistics.csv", index=False)
    print(f"Saved pilot results: {output}")
    if subject_df.subject.nunique() < 5:
        print("WARNING: fewer than 5 test subjects; inferential p-values are exploratory only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
