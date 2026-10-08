"""Shared paths and contracts for the E3 ablation experiment."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
E3_ROOT = HERE.parents[0]
REVISION_ROOT = E3_ROOT.parent
E1_CODE = REVISION_ROOT / "1.prior_benchmark"
sys.path.insert(0, str(E1_CODE))

VARIANTS = ("full", "no_causal", "no_fen", "no_psd")
ABLATION_VARIANTS = VARIANTS[1:]


def tagged_stem(variant: str, seed: int, run_tag: str | None) -> str:
    suffix = f"_{run_tag}" if run_tag else ""
    return f"{variant}_seed_{seed}{suffix}"


def prediction_path(dataset: str, fold: int, variant: str, seed: int, run_tag: str | None) -> Path:
    return E3_ROOT / "predictions" / dataset / f"fold_{fold}" / f"{tagged_stem(variant, seed, run_tag)}.npz"


def checkpoint_path(dataset: str, fold: int, variant: str, seed: int, run_tag: str | None) -> Path:
    return E3_ROOT / "checkpoints" / dataset / f"fold_{fold}" / f"{tagged_stem(variant, seed, run_tag)}.pth"
