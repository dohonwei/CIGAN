"""Validated access to prediction artifacts produced by the original project."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


REVISION_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REVISION_ROOT
E0_ROOT = REVISION_ROOT / "0.preprocessing"
FRONTAL_INDICES = np.asarray([0, 16], dtype=np.int64)

CANONICAL = {
    "deap": E0_ROOT / "canonical_data" / "deap_canonical_30s_128hz.npz",
    "hci": E0_ROOT / "canonical_data" / "hci_canonical_15s_256hz.npz",
}
LEGACY_CANONICAL = {
    "deap": CANONICAL["deap"],
    "hci": E0_ROOT / "canonical_data" / "hci_canonical_30s_128hz.npz",
}
FOLDS = {
    "deap": E0_ROOT / "outputs" / "shared_folds_deap.npz",
    "hci": E0_ROOT / "outputs" / "shared_folds_hci.npz",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_targets(dataset: str, protocol: str = "revision") -> tuple[np.ndarray, np.ndarray, float]:
    if protocol not in ("revision", "legacy"):
        raise ValueError(f"Unknown protocol: {protocol}")
    path = (LEGACY_CANONICAL if protocol == "legacy" else CANONICAL)[dataset]
    with np.load(path, allow_pickle=False) as archive:
        targets = archive["eeg"][:, FRONTAL_INDICES].astype(np.float32, copy=False)
        subjects = archive["subjects"].astype(str)
        sampling_rate = float(archive["sampling_rate_hz"])
    return targets, subjects, sampling_rate


def fold_ids(dataset: str, sample_count: int) -> np.ndarray:
    result = np.full(sample_count, -1, dtype=np.int64)
    with np.load(FOLDS[dataset], allow_pickle=False) as archive:
        for fold in range(1, 6):
            indices = archive[f"test_idx_fold_{fold}"].astype(np.int64)
            if np.any(result[indices] != -1):
                raise ValueError(f"Overlapping test folds in {FOLDS[dataset]}")
            result[indices] = fold
    if np.any(result == -1):
        raise ValueError(f"Folds do not cover all {sample_count} {dataset} samples")
    return result


def load_legacy_prediction(
    path: Path,
    dataset: str,
    *,
    fake_key: str = "fake_fp",
    real_key: str = "real_fp",
    protocol: str = "revision",
) -> dict[str, np.ndarray | float]:
    """Load a legacy prediction only when its reference exactly matches E0 data."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if fake_key not in archive.files or real_key not in archive.files:
            raise KeyError(f"{path} must contain {fake_key!r} and {real_key!r}")
        fake = archive[fake_key].astype(np.float32, copy=False)
        real = archive[real_key].astype(np.float32, copy=False)

    canonical, subjects, sampling_rate = canonical_targets(dataset, protocol)
    if fake.shape != canonical.shape or real.shape != canonical.shape:
        raise ValueError(
            f"Legacy shape mismatch for {path}: fake={fake.shape}, real={real.shape}, "
            f"canonical={canonical.shape}"
        )
    if not np.isfinite(fake).all() or not np.isfinite(real).all():
        raise ValueError(f"Non-finite values in {path}")
    if not np.array_equal(real, canonical):
        maximum = float(np.max(np.abs(real - canonical)))
        correlation = float(np.corrcoef(real.ravel(), canonical.ravel())[0, 1])
        raise ValueError(
            f"Legacy reference does not match the E0 {dataset} {protocol} canonical targets: "
            f"max_abs_diff={maximum:.6g}, correlation={correlation:.6g}. "
            "This artifact cannot be used in the revision analysis."
        )
    return {
        "fake": fake,
        "real": real,
        "subjects": subjects,
        "folds": fold_ids(dataset, len(fake)),
        "sampling_rate_hz": sampling_rate,
        "protocol": protocol,
    }
