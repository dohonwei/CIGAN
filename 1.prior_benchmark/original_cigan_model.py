"""Original-project CIGAN generator used by the frozen-checkpoint sensitivity test."""
from __future__ import annotations

import torch
import torch.nn as nn


class WaveNetResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout1d(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))
        return x + self.out(self.drop(gated))


class UltraEncoder(nn.Module):
    def __init__(self, in_channels: int = 30, hidden: int = 64):
        super().__init__()
        self.inp = nn.Conv1d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList(
            WaveNetResBlock(hidden, dilation)
            for dilation in (1, 2, 4, 8, 16, 32, 64, 128)
        )
        self.out = nn.Conv1d(hidden, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.inp(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.out(hidden)


class OriginalCIGANGenerator(nn.Module):
    """Architecture whose parameter names exactly match comparison/*/CIGAN_G.pth."""

    def __init__(self, in_channels: int = 30, hidden: int = 64):
        super().__init__()
        self.encoder = UltraEncoder(in_channels, hidden)
        self.w_fc = nn.Linear(in_channels, in_channels)
        self.final_merge = nn.Conv1d(4, 2, 1)

    @staticmethod
    def frequency_enhancement(x: torch.Tensor, sampling_rate_hz: float) -> torch.Tensor:
        n_times = x.shape[-1]
        spectrum = torch.fft.rfft(x, dim=-1)
        frequencies = torch.fft.rfftfreq(n_times, d=1.0 / sampling_rate_hz).to(x.device)
        mask = ((frequencies >= 1.0) & (frequencies <= 30.0)).view(1, 1, -1)
        return torch.fft.irfft(spectrum * mask, n=n_times, dim=-1)

    def forward(
        self, sources: torch.Tensor, prior: torch.Tensor, sampling_rate_hz: float
    ) -> torch.Tensor:
        generated = self.encoder(sources)
        transformed_prior = self.w_fc(prior).view(1, -1, 1)
        cue = (sources * transformed_prior).mean(dim=1, keepdim=True).repeat(1, 2, 1)
        enhanced = self.frequency_enhancement(cue, sampling_rate_hz)
        return self.final_merge(torch.cat([generated, enhanced], dim=1))
