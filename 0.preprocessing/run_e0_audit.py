"""E0: audit data provenance and freeze subject-independent folds.

This script reads the original project data without modifying it. All generated
artifacts are written under the revision workspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold


SCRIPT_PATH = Path(__file__).resolve()
E0_ROOT = SCRIPT_PATH.parents[0]
REVISION_ROOT = E0_ROOT.parent
PROJECT_ROOT = REVISION_ROOT
OUTPUT_DIR = E0_ROOT / "outputs"
CANONICAL_DIR = E0_ROOT / "canonical_data"

CHANNEL_NAMES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]

DATASETS = {
    "deap": {
        "source": E0_ROOT / "canonical_data/deap_canonical_30s_128hz.npz",
        "sampling_rate_hz": 128.0,
        "expected_duration_seconds": 30.0,
        "expected_subjects": 32,
        "expected_trials": 1280,
        "provenance": "E0 canonical DEAP copy at 128 Hz and 30 s",
    },
    "hci": {
        "source": E0_ROOT / "canonical_data/hci_canonical_15s_256hz.npz",
        "sampling_rate_hz": 256.0,
        "expected_duration_seconds": 15.0,
        "expected_subjects": 27,
        "expected_trials": 527,
        "provenance": "E0 rebuild from numeric-sorted HCI CSV trials at 256 Hz and 15 s",
    },
}

KNOWN_RATE_REFERENCES = {
    "data/hci/hcipreprocess.py": 128.0,
    "data/deap/deappreprocess.py": 128.0,
    "1_train.py (HCI branch)": 256.0,
    "2_R_Q_evaluation.py (HCI branch)": 256.0,
    "causality_gpu/causilty.py (HCI config)": 256.0,
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def distribution(values: np.ndarray) -> dict[str, int]:
    counts = Counter(str(json_scalar(v)) for v in values.tolist())
    return dict(sorted(counts.items()))


def build_folds(dataset: str, subjects: np.ndarray, n_splits: int = 5) -> list[dict]:
    splitter = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(subjects), dtype=np.uint8)
    folds = []
    npz_payload = {}

    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(dummy, groups=subjects), start=1
    ):
        train_subjects = sorted(str(x) for x in np.unique(subjects[train_idx]))
        test_subjects = sorted(str(x) for x in np.unique(subjects[test_idx]))
        overlap = sorted(set(train_subjects).intersection(test_subjects))
        if overlap:
            raise RuntimeError(f"{dataset} fold {fold_id} subject leakage: {overlap}")

        npz_payload[f"train_idx_fold_{fold_id}"] = train_idx.astype(np.int64)
        npz_payload[f"test_idx_fold_{fold_id}"] = test_idx.astype(np.int64)
        folds.append(
            {
                "dataset": dataset,
                "fold": fold_id,
                "n_train_trials": int(len(train_idx)),
                "n_test_trials": int(len(test_idx)),
                "n_train_subjects": len(train_subjects),
                "n_test_subjects": len(test_subjects),
                "train_subjects": train_subjects,
                "test_subjects": test_subjects,
                "subject_overlap": overlap,
            }
        )

    np.savez(OUTPUT_DIR / f"shared_folds_{dataset}.npz", **npz_payload)
    return folds


def write_fold_csv(all_folds: list[dict]) -> None:
    path = OUTPUT_DIR / "fold_summary.csv"
    fieldnames = [
        "dataset", "fold", "n_train_trials", "n_test_trials",
        "n_train_subjects", "n_test_subjects", "train_subjects",
        "test_subjects", "subject_overlap",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for fold in all_folds:
            row = dict(fold)
            for key in ("train_subjects", "test_subjects", "subject_overlap"):
                row[key] = "|".join(row[key])
            writer.writerow(row)


def write_experiment_manifest(e0_complete: bool) -> None:
    rows = [
        ("E0", "Data and protocol audit", "completed" if e0_complete else "in_progress", "None"),
        ("E1", "Prior comparison", "not_started", "E0"),
        ("E2", "Granger stability", "not_started", "E0"),
        ("E3", "Ablation statistics", "not_started", "E0"),
        ("E4", "Unified baseline evaluation", "not_started", "E0"),
        ("E5", "UQ calibration", "not_started", "E0,E4"),
        ("E6", "Band-range and gamma sensitivity", "not_started", "E0"),
        ("E7", "Synthetic-signal noise robustness", "not_started", "E0"),
        ("E8", "Publication figure regeneration", "not_started", "E1-E7"),
    ]
    with (OUTPUT_DIR / "experiment_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment_id", "name", "status", "dependencies"])
        writer.writerows(rows)


def write_canonical_copy(dataset: str, config: dict, arrays: dict) -> Path:
    output_path = CANONICAL_DIR / f"{dataset}_canonical_30s_128hz.npz"
    np.savez(
        output_path,
        eeg=arrays["eeg"],
        arousal=arrays["arousal"],
        valence=arrays["valence"],
        subjects=arrays["subjects"],
        sampling_rate_hz=np.float32(config["sampling_rate_hz"]),
        window_seconds=np.float32(
            arrays["eeg"].shape[-1] / config["sampling_rate_hz"]
        ),
        channel_names=np.asarray(CHANNEL_NAMES),
        source_sha256=np.asarray(sha256_file(config["source"])),
    )
    return output_path


def audit_dataset(dataset: str, config: dict, write_canonical: bool) -> tuple[dict, list[dict]]:
    source = config["source"]
    if not source.exists():
        raise FileNotFoundError(source)

    with np.load(source, allow_pickle=False) as archive:
        required = {"eeg", "arousal", "valence", "subjects"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"{dataset} missing arrays: {missing}")
        arrays = {key: archive[key] for key in required}

    eeg = arrays["eeg"]
    arousal = arrays["arousal"]
    valence = arrays["valence"]
    subjects = arrays["subjects"]
    errors = []
    warnings = []

    if eeg.ndim != 3:
        errors.append(f"EEG ndim is {eeg.ndim}, expected 3")
    if eeg.shape[1] != len(CHANNEL_NAMES):
        errors.append(f"EEG has {eeg.shape[1]} channels, expected 32")
    for name, values in (("arousal", arousal), ("valence", valence), ("subjects", subjects)):
        if len(values) != len(eeg):
            errors.append(f"{name} length {len(values)} != trials {len(eeg)}")

    unique_subjects = np.unique(subjects)
    duration_seconds = eeg.shape[-1] / config["sampling_rate_hz"]
    if len(eeg) != config["expected_trials"]:
        warnings.append(
            f"trial count {len(eeg)} differs from expected {config['expected_trials']}"
        )
    if len(unique_subjects) != config["expected_subjects"]:
        warnings.append(
            f"subject count {len(unique_subjects)} differs from expected {config['expected_subjects']}"
        )
    expected_duration = config["expected_duration_seconds"]
    if not np.isclose(duration_seconds, expected_duration):
        errors.append(
            f"duration is {duration_seconds:.6g} s, expected {expected_duration:g} s"
        )

    finite_sample = eeg.reshape(-1)[:: max(1, eeg.size // 1_000_000)]
    if not np.isfinite(finite_sample).all():
        errors.append("sampled EEG values contain NaN or infinity")

    folds = build_folds(dataset, subjects)
    canonical_path = source if not errors else None

    report = {
        "dataset": dataset,
        "source_path": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256_file(source),
        "source_arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "sampling_rate_hz": config["sampling_rate_hz"],
        "sampling_rate_provenance": config["provenance"],
        "duration_seconds": duration_seconds,
        "channel_names": CHANNEL_NAMES,
        "frontal_target_indices": [0, 16],
        "frontal_target_names": [CHANNEL_NAMES[0], CHANNEL_NAMES[16]],
        "n_subjects": int(len(unique_subjects)),
        "trials_per_subject": distribution(subjects),
        "arousal_distribution": distribution(arousal),
        "valence_distribution": distribution(valence),
        "fold_file": str(OUTPUT_DIR / f"shared_folds_{dataset}.npz"),
        "canonical_path": str(canonical_path) if canonical_path else None,
        "errors": errors,
        "warnings": warnings,
        "status": "PASS" if not errors else "FAIL",
    }
    return report, folds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-canonical",
        action="store_true",
        help="Write metadata-enriched NPZ copies after the audit passes.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

    dataset_reports = []
    all_folds = []
    for dataset, config in DATASETS.items():
        report, folds = audit_dataset(dataset, config, args.write_canonical)
        dataset_reports.append(report)
        all_folds.extend(folds)

    write_fold_csv(all_folds)

    audit_passed = all(not report["errors"] for report in dataset_reports)
    canonical_written = all(report["canonical_path"] for report in dataset_reports)
    e0_complete = audit_passed and canonical_written
    write_experiment_manifest(e0_complete)

    rate_values = sorted(set(KNOWN_RATE_REFERENCES.values()))
    audit = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "revision_root": str(REVISION_ROOT),
        "fold_protocol": {
            "splitter": "GroupKFold",
            "n_splits": 5,
            "group_key": "subjects",
            "randomized": False,
            "rule": "All learned priors and normalization parameters must use train indices only.",
        },
        "known_hci_sampling_rate_references_hz": KNOWN_RATE_REFERENCES,
        "legacy_hci_sampling_rate_conflict_detected": len(rate_values) > 1,
        "resolution": (
            "The revision protocol rebuilds HCI from the source CSV files at the original "
            "256 Hz rate using a 15 s (3840-sample) centered window. Legacy 128-Hz HCI "
            "canonical data and legacy HCI results are not valid revision inputs."
        ),
        "datasets": dataset_reports,
        "canonical_data_written": canonical_written,
        "overall_status": "PASS" if e0_complete else "FAIL",
        "blockers": [] if e0_complete else [
            "Build hci_canonical_15s_256hz.npz before running the E0 audit."
        ],
        "mandatory_downstream_rule": (
            "Do not reuse global Granger priors; recompute every learned prior inside "
            "each training fold using train indices only."
        ),
    }

    report_path = OUTPUT_DIR / "data_audit_report.json"
    report_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"E0 audit status: {audit['overall_status']}")
    print(f"Report: {report_path}")
    print(f"Fold summary: {OUTPUT_DIR / 'fold_summary.csv'}")
    if args.write_canonical:
        print(f"Canonical data: {CANONICAL_DIR}")
    return 0 if audit["overall_status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
