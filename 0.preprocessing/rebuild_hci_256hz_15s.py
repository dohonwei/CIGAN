"""Rebuild the MAHNOB-HCI canonical dataset at 256 Hz and 15 seconds.

The source CSV files are treated as read-only. Trial files are sorted by their
numeric trial IDs so labels cannot be shifted by lexicographic filename order.
Filtering is performed on the full available trial before the centered crop to
reduce crop-boundary artifacts. The output remains in microvolts and is not
normalized; normalization must be fitted inside each downstream training fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


SCRIPT_PATH = Path(__file__).resolve()
E0_ROOT = SCRIPT_PATH.parents[0]
DEFAULT_RAW_ROOT = Path(
    os.environ.get("CIGAN_HCI_PREPROC_DIR", "data/hci/raw")
)
DEFAULT_OUTPUT = E0_ROOT / "canonical_data/hci_canonical_15s_256hz.npz"
DEFAULT_AUDIT_CSV = E0_ROOT / "outputs/hci_256hz_rebuild_trials.csv"
DEFAULT_MANIFEST = E0_ROOT / "outputs/hci_256hz_rebuild_manifest.json"

SAMPLING_RATE_HZ = 256.0
WINDOW_SECONDS = 15.0
TARGET_SAMPLES = int(SAMPLING_RATE_HZ * WINDOW_SECONDS)
N_CHANNELS = 32
LOWCUT_HZ = 4.0
HIGHCUT_HZ = 45.0
BANDPASS_ORDER = 4
NOTCH_HZ = 50.0
NOTCH_Q = 30.0
LABEL_THRESHOLD = 5.0
VOLTS_TO_MICROVOLTS = 1e6

CHANNEL_NAMES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_trial_id(path: Path) -> int:
    prefix = path.stem.split("_", maxsplit=1)[0]
    try:
        return int(prefix)
    except ValueError as exc:
        raise ValueError(f"Cannot parse numeric trial ID from {path.name}") from exc


def load_label_vector(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    values = np.loadtxt(path, delimiter=",", ndmin=1, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Invalid label vector: {path}")
    return values


def design_filters():
    bandpass_sos = butter(
        BANDPASS_ORDER,
        [LOWCUT_HZ, HIGHCUT_HZ],
        btype="bandpass",
        fs=SAMPLING_RATE_HZ,
        output="sos",
    )
    notch_b, notch_a = iirnotch(NOTCH_HZ, NOTCH_Q, fs=SAMPLING_RATE_HZ)
    return bandpass_sos, notch_b, notch_a


def preprocess_trial(
    source: Path,
    bandpass_sos: np.ndarray,
    notch_b: np.ndarray,
    notch_a: np.ndarray,
) -> tuple[np.ndarray, dict]:
    # CSV layout is samples x channels; calculations remain in volts here.
    samples_by_channels = np.loadtxt(source, delimiter=",", dtype=np.float64)
    if samples_by_channels.ndim != 2:
        raise ValueError(f"{source} is not a two-dimensional CSV")
    if samples_by_channels.shape[1] != N_CHANNELS:
        raise ValueError(
            f"{source} has {samples_by_channels.shape[1]} channels; expected {N_CHANNELS}"
        )
    if not np.isfinite(samples_by_channels).all():
        raise ValueError(f"{source} contains NaN or infinity")

    n_samples = samples_by_channels.shape[0]
    if n_samples < TARGET_SAMPLES:
        raise ValueError(
            f"{source} has {n_samples} samples; at least {TARGET_SAMPLES} are required"
        )

    channels_by_samples = samples_by_channels.T
    # Apply the stated 50-Hz notch and then the 4-45-Hz bandpass on full trials.
    filtered = filtfilt(notch_b, notch_a, channels_by_samples, axis=-1)
    filtered = sosfiltfilt(bandpass_sos, filtered, axis=-1)

    crop_start = (n_samples - TARGET_SAMPLES) // 2
    crop_end = crop_start + TARGET_SAMPLES
    cropped_uv = filtered[:, crop_start:crop_end] * VOLTS_TO_MICROVOLTS
    cropped_uv = cropped_uv.astype(np.float32, copy=False)

    audit = {
        "source_samples": int(n_samples),
        "source_duration_seconds": float(n_samples / SAMPLING_RATE_HZ),
        "crop_start_sample": int(crop_start),
        "crop_end_sample_exclusive": int(crop_end),
        "crop_duration_seconds": WINDOW_SECONDS,
        "output_min_uv": float(cropped_uv.min()),
        "output_max_uv": float(cropped_uv.max()),
        "output_mean_uv": float(cropped_uv.mean()),
        "output_std_uv": float(cropped_uv.std()),
    }
    return cropped_uv, audit


def discover_subjects(raw_root: Path) -> list[Path]:
    subjects = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if not subjects:
        raise RuntimeError(f"No subject directories found under {raw_root}")
    return subjects


def rebuild(raw_root: Path, output_path: Path, overwrite: bool) -> None:
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    bandpass_sos, notch_b, notch_a = design_filters()

    eeg_trials = []
    arousal_raw_all = []
    valence_raw_all = []
    subjects_all = []
    trial_ids_all = []
    source_files_all = []
    audit_rows = []

    for subject_dir in discover_subjects(raw_root):
        trial_files = sorted(subject_dir.glob("*_EEG.csv"), key=numeric_trial_id)
        if not trial_files:
            raise RuntimeError(f"No EEG trial files found in {subject_dir}")

        arousal_raw = load_label_vector(subject_dir / "labels_feltArsl.csv")
        valence_raw = load_label_vector(subject_dir / "labels_feltVlnc.csv")
        if len(arousal_raw) != len(trial_files) or len(valence_raw) != len(trial_files):
            raise ValueError(
                f"{subject_dir.name}: trials/arousal/valence counts differ: "
                f"{len(trial_files)}/{len(arousal_raw)}/{len(valence_raw)}"
            )

        trial_ids = [numeric_trial_id(path) for path in trial_files]
        expected_ids = list(range(len(trial_files)))
        if trial_ids != expected_ids:
            raise ValueError(
                f"{subject_dir.name}: expected trial IDs {expected_ids}, found {trial_ids}"
            )

        for trial_file, trial_id in zip(trial_files, trial_ids):
            eeg_uv, audit = preprocess_trial(
                trial_file, bandpass_sos, notch_b, notch_a
            )
            eeg_trials.append(eeg_uv)
            arousal_raw_all.append(arousal_raw[trial_id])
            valence_raw_all.append(valence_raw[trial_id])
            subjects_all.append(subject_dir.name)
            trial_ids_all.append(trial_id)
            source_files_all.append(str(trial_file.relative_to(raw_root)))
            audit_rows.append(
                {
                    "subject": subject_dir.name,
                    "trial_id": trial_id,
                    "source_file": str(trial_file.relative_to(raw_root)),
                    "arousal_raw": float(arousal_raw[trial_id]),
                    "valence_raw": float(valence_raw[trial_id]),
                    **audit,
                }
            )

    eeg = np.stack(eeg_trials).astype(np.float32, copy=False)
    arousal_raw_array = np.asarray(arousal_raw_all, dtype=np.float32)
    valence_raw_array = np.asarray(valence_raw_all, dtype=np.float32)
    arousal_binary = (arousal_raw_array >= LABEL_THRESHOLD).astype(np.int64)
    valence_binary = (valence_raw_array >= LABEL_THRESHOLD).astype(np.int64)
    subjects = np.asarray(subjects_all)
    trial_ids = np.asarray(trial_ids_all, dtype=np.int64)
    source_files = np.asarray(source_files_all)

    expected_shape = (len(eeg_trials), N_CHANNELS, TARGET_SAMPLES)
    if eeg.shape != expected_shape:
        raise RuntimeError(f"Unexpected output shape {eeg.shape}; expected {expected_shape}")

    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez(
        temporary_path,
        eeg=eeg,
        arousal=arousal_binary,
        valence=valence_binary,
        arousal_raw=arousal_raw_array,
        valence_raw=valence_raw_array,
        subjects=subjects,
        trial_ids=trial_ids,
        source_files=source_files,
        sampling_rate_hz=np.float32(SAMPLING_RATE_HZ),
        window_seconds=np.float32(WINDOW_SECONDS),
        channel_names=np.asarray(CHANNEL_NAMES),
        amplitude_unit=np.asarray("microvolt"),
        normalization=np.asarray("none; fit on training fold downstream"),
        bandpass_hz=np.asarray([LOWCUT_HZ, HIGHCUT_HZ], dtype=np.float32),
        bandpass_order=np.int64(BANDPASS_ORDER),
        notch_hz=np.float32(NOTCH_HZ),
        notch_q=np.float32(NOTCH_Q),
        label_threshold=np.float32(LABEL_THRESHOLD),
    )
    os.replace(temporary_path, output_path)

    audit_fields = list(audit_rows[0].keys())
    with DEFAULT_AUDIT_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        writer.writerows(audit_rows)

    manifest = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "shape": list(eeg.shape),
        "dtype": str(eeg.dtype),
        "sampling_rate_hz": SAMPLING_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "target_samples": TARGET_SAMPLES,
        "n_subjects": int(len(np.unique(subjects))),
        "n_trials": int(len(eeg)),
        "amplitude_unit": "microvolt",
        "normalization": "none; fit on training fold downstream",
        "preprocessing": {
            "notch_hz": NOTCH_HZ,
            "notch_q": NOTCH_Q,
            "bandpass_hz": [LOWCUT_HZ, HIGHCUT_HZ],
            "bandpass_order": BANDPASS_ORDER,
            "filter_application_scope": "full trial before centered crop",
            "crop": "centered",
        },
        "label_mapping": {
            "source": "numeric trial ID indexes the corresponding label row",
            "binary_rule": f"label >= {LABEL_THRESHOLD:g}",
        },
        "channel_order_assumption": (
            "The 32 CSV columns follow the MAHNOB-HCI/DEAP-compatible order "
            "stored in channel_names."
        ),
    }
    DEFAULT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Saved canonical HCI data: {output_path}")
    print(f"Saved trial audit: {DEFAULT_AUDIT_CSV}")
    print(f"Saved rebuild manifest: {DEFAULT_MANIFEST}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rebuild(args.raw_root.resolve(), args.output.resolve(), args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
