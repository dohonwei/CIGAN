"""Evaluate E7 predictions on held-out repeats 16--19."""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy.signal import welch

from e7_common import MODELS, OUTPUT_DIR, PREDICTION_DIR, TEST_REPEATS, load_data


def metrics(reference, prediction, fs):
    r, p = reference.ravel(), prediction.ravel(); error = r - p
    pearson = float(np.corrcoef(r, p)[0, 1]) if r.std() > 0 and p.std() > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(error ** 2))); mae = float(np.mean(np.abs(error)))
    output_snr = float(10 * np.log10((np.mean(r ** 2) + 1e-12) / (np.mean(error ** 2) + 1e-12)))
    freq, psd_r = welch(r, fs=fs, nperseg=min(len(r), int(2 * fs)))
    _, psd_p = welch(p, fs=fs, nperseg=min(len(p), int(2 * fs)))
    selected = (freq >= 4) & (freq <= 45)
    log_psd_rmse = float(np.sqrt(np.mean((np.log10(psd_r[selected] + 1e-12) - np.log10(psd_p[selected] + 1e-12)) ** 2)))
    return {"pearson": pearson, "rmse": rmse, "mae": mae, "output_snr_db": output_snr, "log_psd_rmse": log_psd_rmse}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS)); parser.add_argument("--run-tag", default="", help="Prediction-file tag"); parser.add_argument("--output-tag", default=None, help="Output CSV tag; defaults to --run-tag"); args = parser.parse_args()
    data = load_data(); mask = np.isin(data["repeat"], TEST_REPEATS); row_ids = np.flatnonzero(mask); fs = float(data["fs"])
    raw = []
    for model in args.models:
        stem = f"{model}_{args.run_tag}" if args.run_tag else model; path = PREDICTION_DIR / f"{stem}.npz"
        if not path.exists(): raise FileNotFoundError(f"Missing {path}; train {model} first.")
        prediction = np.load(path)["prediction"]
        if prediction.shape != (len(data["repeat"]), 2, data["noisy_sources"].shape[-1]): raise ValueError(f"Unexpected {model} prediction shape: {prediction.shape}")
        for row in row_ids:
            reference = data["true_target_bank"][int(data["source_id"][row])]
            for channel in range(2):
                record = {"model": model, "row_id": int(row), "signal_type": str(data["signal_type"][row]), "noise_type": str(data["noise_type"][row]), "snr_db": int(data["snr_db"][row]), "repeat": int(data["repeat"][row]), "target_channel": channel + 1}
                record.update(metrics(reference[channel], prediction[row, channel], fs)); raw.append(record)
    metric_names = ("pearson", "rmse", "mae", "output_snr_db", "log_psd_rmse")
    groups = {}
    for row in raw: groups.setdefault(tuple(row[k] for k in ("model", "signal_type", "noise_type", "snr_db")), []).append(row)
    summary = []
    for key, values in sorted(groups.items()):
        record = dict(zip(("model", "signal_type", "noise_type", "snr_db"), key)); record["n"] = len(values)
        for metric in metric_names:
            vector = np.asarray([v[metric] for v in values], dtype=float)
            record[f"{metric}_mean"] = float(np.nanmean(vector)); record[f"{metric}_std"] = float(np.nanstd(vector, ddof=1)); record[f"{metric}_sem"] = float(np.nanstd(vector, ddof=1) / np.sqrt(np.isfinite(vector).sum()))
        summary.append(record)
    raw_fields = list(raw[0]); summary_fields = list(summary[0])
    output_tag = args.run_tag if args.output_tag is None else args.output_tag
    suffix = f"_{output_tag}" if output_tag else ""
    write_csv(OUTPUT_DIR / f"e7_metrics_raw{suffix}.csv", raw, raw_fields); write_csv(OUTPUT_DIR / f"e7_metrics_summary{suffix}.csv", summary, summary_fields)
    print(f"Wrote {len(raw)} raw rows and {len(summary)} summary rows to {OUTPUT_DIR}")


if __name__ == "__main__": main()
