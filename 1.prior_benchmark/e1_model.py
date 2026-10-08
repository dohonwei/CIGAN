"""CIGAN architecture and fixed manuscript-aligned losses for E1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNetResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout1d(p=0.1)
        self.output = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        return x + self.output(self.dropout(gated))


class UltraEncoder(nn.Module):
    def __init__(self, in_channels: int = 30, hidden: int = 64):
        super().__init__()
        self.input = nn.Conv1d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList(
            WaveNetResBlock(hidden, dilation)
            for dilation in [1, 2, 4, 8, 16, 32, 64, 128]
        )
        self.output = nn.Conv1d(hidden, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class CIGANGenerator(nn.Module):
    """The same generator is used for every E1 prior condition."""

    def __init__(self, in_channels: int = 30):
        super().__init__()
        self.encoder = UltraEncoder(in_channels)
        self.final_merge = nn.Conv1d(4, 2, 1)

    @staticmethod
    def frequency_enhancement(x: torch.Tensor, fs: float) -> torch.Tensor:
        n_times = x.shape[-1]
        spectrum = torch.fft.rfft(x, dim=-1)
        frequencies = torch.fft.rfftfreq(n_times, d=1.0 / fs).to(x.device)
        # Preserve the original CIGAN FEN definition for a one-variable E1 comparison.
        mask = (frequencies >= 1.0) & (frequencies <= 30.0)
        return torch.fft.irfft(spectrum * mask.view(1, 1, -1), n=n_times, dim=-1)

    def forward(self, sources: torch.Tensor, prior: torch.Tensor, fs: float) -> torch.Tensor:
        generated = self.encoder(sources)
        channel_gain = (prior * sources.shape[1]).view(1, -1, 1)
        weighted = sources * channel_gain
        fen = weighted.mean(dim=1, keepdim=True).repeat(1, 2, 1)
        fen = self.frequency_enhancement(fen, fs)
        return self.final_merge(torch.cat([generated, fen], dim=1))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, 7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x).squeeze(-1))


def temporal_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(fake, real)


def amplitude_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(fake.abs().mean(dim=-1), real.abs().mean(dim=-1))


def _welch_psd(x: torch.Tensor, fs: float, nperseg: int = 256) -> torch.Tensor:
    """Differentiable one-sided Welch density matching SciPy's default scaling."""
    if x.shape[-1] < nperseg:
        raise ValueError(f"Welch PSD requires at least {nperseg} samples")
    flat = x.reshape(-1, x.shape[-1])
    segments = flat.unfold(-1, nperseg, nperseg // 2)
    segments = segments - segments.mean(dim=-1, keepdim=True)
    window = torch.hann_window(
        nperseg, periodic=True, device=x.device, dtype=x.dtype
    )
    spectrum = torch.fft.rfft(segments * window, dim=-1)
    density = spectrum.abs().square() / (float(fs) * window.square().sum())
    if nperseg % 2 == 0:
        density[..., 1:-1] *= 2.0
    else:
        density[..., 1:] *= 2.0
    return density.mean(dim=-2)


def psd_loss(fake: torch.Tensor, real: torch.Tensor, fs: float) -> torch.Tensor:
    return F.l1_loss(_welch_psd(fake, fs), _welch_psd(real, fs))


def local_stft_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(64, device=fake.device)
    fake_flat = fake.reshape(-1, fake.shape[-1])
    real_flat = real.reshape(-1, real.shape[-1])
    fake_stft = torch.stft(
        fake_flat, n_fft=64, hop_length=32, window=window, return_complex=True
    )
    real_stft = torch.stft(
        real_flat, n_fft=64, hop_length=32, window=window, return_complex=True
    )
    return F.l1_loss(fake_stft.abs(), real_stft.abs())


def gradient_penalty(
    discriminator: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
    coefficient: float = 10.0,
) -> torch.Tensor:
    batch = real.shape[0]
    alpha = torch.rand(batch, 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
    scores = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(batch, -1)
    return coefficient * (gradients.norm(2, dim=1) - 1.0).square().mean()


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float = 0.999) -> None:
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
        target_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)
