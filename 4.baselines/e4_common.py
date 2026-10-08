"""Shared E4 paths, model names, and established baseline architectures."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve()
E4_ROOT = HERE.parents[0]
REVISION_ROOT = E4_ROOT.parent
E1_CODE = REVISION_ROOT / "1.prior_benchmark"
sys.path.insert(0, str(E1_CODE))

from e1_model import UltraEncoder

BASELINES = ("hveegnet", "tieeegnet", "wavenet", "encoder_decoder", "spline")
DEEP_BASELINES = BASELINES[:-1]


class HVEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.short = nn.Conv1d(30, 32, 3, padding=1)
        self.medium = nn.Conv1d(30, 32, 7, padding=3)
        self.long = nn.Conv1d(30, 32, 11, padding=5)
        self.merge = nn.Sequential(nn.Conv1d(96, 64, 1), nn.ELU(inplace=True), nn.Conv1d(64, 2, 3, padding=1))

    def forward(self, x): return self.merge(torch.cat((self.short(x), self.medium(x), self.long(x)), dim=1))


class TIEEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv1d(30, 64, 1)
        self.temporal = nn.Conv1d(64, 64, 15, padding=7)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(64, 16, 1), nn.ReLU(inplace=True), nn.Conv1d(16, 64, 1), nn.Sigmoid())
        self.output = nn.Conv1d(64, 2, 3, padding=1)

    def forward(self, x):
        hidden = F.gelu(self.temporal(F.gelu(self.spatial(x))))
        return self.output(hidden * self.attention(hidden))


class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(30, 64, 7, padding=3), nn.ReLU(inplace=True),
                                     nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(inplace=True),
                                     nn.Conv1d(128, 64, 5, padding=2), nn.ReLU(inplace=True),
                                     nn.Conv1d(64, 2, 3, padding=1))

    def forward(self, x): return self.network(x)


def make_generator(name: str) -> nn.Module:
    if name == "hveegnet": return HVEEGNet()
    if name == "tieeegnet": return TIEEEGNet()
    if name == "wavenet": return UltraEncoder()
    if name == "encoder_decoder": return EncoderDecoder()
    raise ValueError(f"No trainable generator for {name}")


def tagged_stem(model: str, seed: int, run_tag: str | None) -> str:
    suffix = f"_{run_tag}" if run_tag else ""
    return f"{model}_seed_{seed}{suffix}"


def prediction_path(dataset: str, fold: int, model: str, seed: int, run_tag: str | None) -> Path:
    return E4_ROOT / "predictions" / dataset / f"fold_{fold}" / f"{tagged_stem(model, seed, run_tag)}.npz"


def checkpoint_path(dataset: str, fold: int, model: str, seed: int, run_tag: str | None) -> Path:
    return E4_ROOT / "checkpoints" / dataset / f"fold_{fold}" / f"{tagged_stem(model, seed, run_tag)}.pth"
