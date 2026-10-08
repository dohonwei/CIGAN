"""Shared paths, data contracts, and utilities for E1."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
E1_ROOT = SCRIPT_PATH.parents[0]
REVISION_ROOT = E1_ROOT.parent
E0_ROOT = REVISION_ROOT / "0.preprocessing"

DATASETS = {
    "deap": {
        "data": E0_ROOT / "canonical_data/deap_canonical_30s_128hz.npz",
        "folds": E0_ROOT / "outputs/shared_folds_deap.npz",
        "fs": 128.0,
        "lag": 5,
    },
    "hci": {
        "data": E0_ROOT / "canonical_data/hci_canonical_15s_256hz.npz",
        "folds": E0_ROOT / "outputs/shared_folds_hci.npz",
        "fs": 256.0,
        "lag": 10,
    },
}

PRIORS = ("granger", "pearson", "mutual_info", "random", "uniform")
FRONTAL_INDICES = np.asarray([0, 16], dtype=np.int64)
SOURCE_INDICES = np.asarray([i for i in range(32) if i not in FRONTAL_INDICES], dtype=np.int64)
CHANNEL_NAMES = np.asarray([
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
    "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
])


def validate_args(dataset: str, fold: int, prior: str | None = None) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if fold not in range(1, 6):
        raise ValueError("fold must be in 1..5")
    if prior is not None and prior not in PRIORS:
        raise ValueError(f"Unknown prior: {prior}")


def load_dataset(dataset: str) -> dict[str, np.ndarray]:
    path = DATASETS[dataset]["data"]
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    eeg = result["eeg"]
    if eeg.ndim != 3 or eeg.shape[1:] != (32, 3840):
        raise ValueError(f"Unexpected {dataset} EEG shape: {eeg.shape}")
    if not np.isfinite(eeg).all():
        raise ValueError(f"{dataset} EEG contains NaN or infinity")
    fs = float(result["sampling_rate_hz"])
    if not np.isclose(fs, DATASETS[dataset]["fs"]):
        raise ValueError(f"{dataset} metadata fs={fs}, expected {DATASETS[dataset]['fs']}")
    return result


def load_fold(dataset: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    validate_args(dataset, fold)
    path = DATASETS[dataset]["folds"]
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        train_idx = archive[f"train_idx_fold_{fold}"].astype(np.int64)
        test_idx = archive[f"test_idx_fold_{fold}"].astype(np.int64)
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError(f"{dataset} fold {fold} has train/test index overlap")
    return train_idx, test_idx


def fit_channel_scaler(eeg: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = eeg[train_idx].astype(np.float64, copy=False)
    mean = train.mean(axis=(0, 2), dtype=np.float64)
    std = train.std(axis=(0, 2), dtype=np.float64)
    if np.any(std < 1e-8):
        raise ValueError("At least one training-fold channel has near-zero standard deviation")
    return mean.astype(np.float32), std.astype(np.float32)


def transform_eeg(eeg: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((eeg - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def normalize_nonnegative(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    scores = np.maximum(scores, 0.0)
    total = scores.sum()
    if total <= 0:
        return np.full(scores.shape, 1.0 / len(scores), dtype=np.float32)
    return (scores / total).astype(np.float32)


def prior_path(dataset: str, fold: int, prior: str, seed: int) -> Path:
    return E1_ROOT / f"priors/{dataset}/fold_{fold}/{prior}_seed_{seed}.npz"


def prediction_path(
    dataset: str, fold: int, prior: str, seed: int, run_tag: str | None = None
) -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return E1_ROOT / f"predictions/{dataset}/fold_{fold}/{prior}_seed_{seed}{suffix}.npz"


def checkpoint_path(
    dataset: str, fold: int, prior: str, seed: int, run_tag: str | None = None
) -> Path:
    suffix = f"_{run_tag}" if run_tag else ""
    return E1_ROOT / f"checkpoints/{dataset}/fold_{fold}/{prior}_seed_{seed}{suffix}.pth"


def pilot_split_path(dataset: str, seed: int, n_subjects: int, n_test_subjects: int) -> Path:
    return E1_ROOT / (
        f"pilot_subset/splits/{dataset}_subjects{n_subjects}_test{n_test_subjects}_seed{seed}.npz"
    )


def pilot_prior_path(dataset: str, split_id: str, prior: str, seed: int) -> Path:
    return E1_ROOT / f"pilot_subset/priors/{dataset}/{split_id}/{prior}_seed_{seed}.npz"


def pilot_prediction_path(dataset: str, split_id: str, prior: str, seed: int) -> Path:
    return E1_ROOT / f"pilot_subset/predictions/{dataset}/{split_id}/{prior}_seed_{seed}.npz"


def pilot_checkpoint_path(dataset: str, split_id: str, prior: str, seed: int) -> Path:
    return E1_ROOT / f"pilot_subset/checkpoints/{dataset}/{split_id}/{prior}_seed_{seed}.pth"


def load_pilot_split(path: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        stored_dataset = str(archive["dataset"].item())
        train_idx = archive["train_idx"].astype(np.int64)
        test_idx = archive["test_idx"].astype(np.int64)
        split_id = str(archive["split_id"].item())
    if stored_dataset != dataset:
        raise ValueError(f"Pilot split dataset={stored_dataset}, requested={dataset}")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError("Pilot split has train/test index overlap")
    return train_idx, test_idx, split_id


def sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
