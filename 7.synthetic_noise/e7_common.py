"""Shared data, models, and paths for the reviewer-requested E7 benchmark."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import f as f_distribution

ROOT = Path(__file__).resolve().parents[0]
DATA_FILE = ROOT / "data" / "synthetic_noise_benchmark.npz"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
PRIOR_DIR = OUTPUT_DIR / "priors"
MODELS = ("ridge", "hveegnet", "tieeegnet", "cigan_granger", "cigan_uniform", "cigan_structural")
DEEP_MODELS = MODELS[1:]
TRAIN_REPEATS = tuple(range(12))
VALID_REPEATS = tuple(range(12, 16))
TEST_REPEATS = tuple(range(16, 20))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(mmap_mode: str | None = None):
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}; run generate_synthetic_data.py first.")
    return np.load(DATA_FILE, mmap_mode=mmap_mode)


def clean_indices(data, repeats: tuple[int, ...]) -> np.ndarray:
    # clean banks contain one row per signal type and repeat, in source_id order.
    source_ids = np.unique(data["source_id"][np.isin(data["repeat"], repeats)])
    return source_ids.astype(int)


def fit_scalers(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    x_mean = x.mean(axis=(0, 2), keepdims=True).astype("float32")
    x_std = (x.std(axis=(0, 2), keepdims=True) + 1e-6).astype("float32")
    y_mean = y.mean(axis=(0, 2), keepdims=True).astype("float32")
    y_std = (y.std(axis=(0, 2), keepdims=True) + 1e-6).astype("float32")
    return x_mean, x_std, y_mean, y_std


class HVEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.short = nn.Conv1d(30, 32, 3, padding=1)
        self.medium = nn.Conv1d(30, 32, 7, padding=3)
        self.long = nn.Conv1d(30, 32, 11, padding=5)
        self.merge = nn.Sequential(nn.Conv1d(96, 64, 1), nn.ELU(), nn.Conv1d(64, 2, 3, padding=1))

    def forward(self, x):
        return self.merge(torch.cat((self.short(x), self.medium(x), self.long(x)), dim=1))


class TIEEEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Conv1d(30, 64, 1)
        self.temporal = nn.Conv1d(64, 64, 15, padding=7)
        self.attention = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(64, 16, 1), nn.ReLU(), nn.Conv1d(16, 64, 1), nn.Sigmoid())
        self.output = nn.Conv1d(64, 2, 3, padding=1)

    def forward(self, x):
        h = F.gelu(self.temporal(F.gelu(self.spatial(x))))
        return self.output(h * self.attention(h))


class ResidualBlock(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.f = nn.Conv1d(64, 64, 3, padding=dilation, dilation=dilation)
        self.g = nn.Conv1d(64, 64, 3, padding=dilation, dilation=dilation)
        self.o = nn.Conv1d(64, 64, 1)
        self.dropout = nn.Dropout1d(0.1)

    def forward(self, x):
        return x + self.o(self.dropout(torch.tanh(self.f(x)) * torch.sigmoid(self.g(x))))


class CIGAN(nn.Module):
    def __init__(self, prior: np.ndarray, fs: float):
        super().__init__()
        self.input = nn.Conv1d(30, 64, 1)
        self.blocks = nn.ModuleList(ResidualBlock(d) for d in (1, 2, 4, 8, 16, 32, 64, 128))
        self.encoder_output = nn.Conv1d(64, 2, 1)
        self.final = nn.Conv1d(4, 2, 1)
        self.register_buffer("prior", torch.as_tensor(prior, dtype=torch.float32).view(1, 30, 1))
        self.fs = fs

    def forward(self, x):
        h = self.input(x)
        for block in self.blocks:
            h = block(h)
        generated = self.encoder_output(h)
        weighted = (x * (self.prior * x.shape[1])).mean(dim=1, keepdim=True).repeat(1, 2, 1)
        spectrum = torch.fft.rfft(weighted, dim=-1)
        frequency = torch.fft.rfftfreq(x.shape[-1], d=1.0 / self.fs).to(x.device)
        fen = torch.fft.irfft(spectrum * ((frequency >= 1) & (frequency <= 30)).view(1, 1, -1), n=x.shape[-1])
        return self.final(torch.cat((generated, fen), dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(2, 32, 7, 2, 3), nn.LeakyReLU(.2), nn.Conv1d(32, 64, 7, 2, 3), nn.LeakyReLU(.2), nn.Conv1d(64, 128, 7, 2, 3), nn.LeakyReLU(.2), nn.AdaptiveAvgPool1d(1))
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        return self.output(self.features(x).squeeze(-1))


def structural_prior() -> np.ndarray:
    # Mean absolute source contribution across the two known simulation equations.
    p = np.zeros(30, dtype="float32")
    p[[0, 4, 12]] += np.array([.45, .30, .25], dtype="float32") / 2
    p[[1, 5, 18]] += np.array([.40, .35, .25], dtype="float32") / 2
    return p / p.sum()


def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    flat = p_values.ravel()
    order = np.argsort(flat)
    ranked = flat[order]
    passed = ranked <= alpha * np.arange(1, len(flat) + 1) / len(flat)
    cutoff = np.where(passed)[0].max() if passed.any() else -1
    significant = np.zeros(len(flat), dtype=bool)
    if cutoff >= 0:
        significant[order[:cutoff + 1]] = True
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(flat) / np.arange(1, len(flat) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted.reshape(p_values.shape), significant.reshape(p_values.shape)


def granger_prior(x_train: np.ndarray, y_train: np.ndarray, lag: int,
                  device: torch.device) -> tuple[np.ndarray, dict]:
    """Estimate source-to-target Granger weights from training repeats only."""
    train_eeg = np.concatenate((x_train, y_train), axis=1)
    data = torch.from_numpy(train_eeg)
    n_trials, n_channels, n_times = data.shape
    n_vars = n_channels * (lag + 1)
    covariance = torch.zeros((n_vars, n_vars), dtype=torch.float64, device=device)
    for start in range(0, n_trials, 64):
        batch = data[start:start + 64].to(device=device, dtype=torch.float64)
        current = batch[:, :, lag:]
        lags = [batch[:, :, lag - step:n_times - step] for step in range(1, lag + 1)]
        flat = torch.cat([current] + lags, dim=1).transpose(1, 2).reshape(-1, n_vars).T
        covariance += flat @ flat.T

    f_stats = np.zeros((30, 2), dtype=np.float64)
    p_values = np.ones_like(f_stats)
    d1 = lag
    d2 = n_trials * (n_times - lag) - 2 * lag
    for target_pos, target in enumerate((30, 31)):
        target_lags = [target + step * n_channels for step in range(1, lag + 1)]
        r_yy = covariance[target, target]
        reduced_yx = covariance[target, target_lags].unsqueeze(0)
        reduced_rss = (r_yy - reduced_yx @ torch.linalg.pinv(
            covariance[target_lags][:, target_lags]) @ reduced_yx.T).item()
        for source in range(30):
            source_lags = [source + step * n_channels for step in range(1, lag + 1)]
            full_lags = target_lags + source_lags
            full_yx = covariance[target, full_lags].unsqueeze(0)
            full_rss = (r_yy - full_yx @ torch.linalg.pinv(
                covariance[full_lags][:, full_lags]) @ full_yx.T).item()
            improvement = max(0.0, reduced_rss - full_rss)
            stat = (improvement / d1) / (max(full_rss, 1e-12) / d2)
            f_stats[source, target_pos] = stat
            p_values[source, target_pos] = f_distribution.sf(stat, d1, d2)

    adjusted_p, significant = fdr_bh(p_values)
    raw_scores = np.where(significant, np.log1p(f_stats), 0.0).mean(axis=1)
    if not np.any(raw_scores > 0):
        raise RuntimeError("No Granger edge survived BH-FDR; refusing an implicit uniform fallback.")
    weights = (raw_scores / raw_scores.sum()).astype("float32")
    return weights, {"f_stats": f_stats, "p_values": p_values,
                     "adjusted_p": adjusted_p, "significant": significant,
                     "n_significant_edges": int(significant.sum()),
                     "n_total_edges": int(significant.size), "lag": lag,
                     "d1": d1, "d2": d2}


def make_generator(name: str, fs: float, granger_weights: np.ndarray | None = None) -> nn.Module:
    if name == "hveegnet": return HVEEGNet()
    if name == "tieeegnet": return TIEEEGNet()
    if name == "cigan_granger":
        if granger_weights is None: raise ValueError("cigan_granger requires granger_weights")
        return CIGAN(granger_weights, fs)
    if name == "cigan_uniform": return CIGAN(np.ones(30, dtype="float32") / 30, fs)
    if name == "cigan_structural": return CIGAN(structural_prior(), fs)
    raise ValueError(name)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
