"""Train one canonical E3 dataset x fold x ablation task."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from e3_common import VARIANTS, checkpoint_path, prediction_path
from e1_common import (DATASETS, FRONTAL_INDICES, SOURCE_INDICES,
                       fit_channel_scaler, load_dataset, load_fold,
                       set_deterministic_seed, sha256_indices, transform_eeg,
                       validate_args)
from e1_model import (CIGANGenerator, Discriminator, amplitude_loss,
                      gradient_penalty, local_stft_loss, psd_loss,
                      temporal_loss, update_ema)
from train_prior_benchmark import load_verified_prior


class AblationGenerator(CIGANGenerator):
    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant

    def forward(self, sources: torch.Tensor, prior: torch.Tensor, fs: float) -> torch.Tensor:
        generated = self.encoder(sources)
        if self.variant == "no_fen":
            return self.final_merge(torch.cat((generated, torch.zeros_like(generated)), dim=1))
        if self.variant == "no_causal":
            prior = torch.full_like(prior, 1.0 / sources.shape[1])
        channel_gain = (prior * sources.shape[1]).view(1, -1, 1)
        fen = (sources * channel_gain).mean(dim=1, keepdim=True).repeat(1, 2, 1)
        fen = self.frequency_enhancement(fen, fs)
        return self.final_merge(torch.cat((generated, fen), dim=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--fold", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--adversarial-ramp-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--run-tag", default="e3_welch_e100")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    validate_args(args.dataset, args.fold, "granger")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs must satisfy 0 <= warmup < epochs")
    if args.adversarial_ramp_epochs <= 0:
        raise ValueError("--adversarial-ramp-epochs must be positive")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must satisfy 0 <= decay < 1")
    set_deterministic_seed(args.seed)

    output_path = prediction_path(args.dataset, args.fold, args.variant, args.seed, args.run_tag)
    model_path = checkpoint_path(args.dataset, args.fold, args.variant, args.seed, args.run_tag)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    data = load_dataset(args.dataset)
    train_idx, test_idx = load_fold(args.dataset, args.fold)
    subjects = data["subjects"]
    if np.intersect1d(np.unique(subjects[train_idx]), np.unique(subjects[test_idx])).size:
        raise ValueError("Subject leakage detected")
    granger_prior_weights, used_prior_path = load_verified_prior(
        args.dataset, args.fold, "granger", args.seed, train_idx
    )
    effective_prior_weights = (
        np.full(30, 1.0 / 30, dtype=np.float32)
        if args.variant == "no_causal" else granger_prior_weights
    )
    channel_mean, channel_std = fit_channel_scaler(data["eeg"], train_idx)
    train_eeg = transform_eeg(data["eeg"][train_idx], channel_mean, channel_std)
    test_eeg = transform_eeg(data["eeg"][test_idx], channel_mean, channel_std)
    train_set = TensorDataset(
        torch.from_numpy(train_eeg[:, SOURCE_INDICES]),
        torch.from_numpy(train_eeg[:, FRONTAL_INDICES]),
    )
    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = DATASETS[args.dataset]["fs"]
    prior = torch.from_numpy(effective_prior_weights).to(device)
    generator = AblationGenerator(args.variant).to(device)
    generator_ema = copy.deepcopy(generator).to(device)
    discriminator = Discriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            phase = "warmup"
            adversarial_weight = 0.0
        else:
            phase = "full"
            adversarial_weight = min(
                1.0,
                (epoch - args.warmup_epochs) / args.adversarial_ramp_epochs,
            )
        generator.train(); discriminator.train()
        totals = {key: 0.0 for key in ("g", "d", "gan", "tc", "amp", "psd", "stft")}
        n_batches = 0
        progress = tqdm(loader, desc=f"E3 {args.dataset} {args.variant} f{args.fold} e{epoch}", leave=False)
        for sources_cpu, targets_cpu in progress:
            sources = sources_cpu.to(device, non_blocking=True)
            targets = targets_cpu.to(device, non_blocking=True)
            if adversarial_weight > 0.0:
                with torch.no_grad(): fake_detached = generator(sources, prior, fs)
                loss_d = discriminator(fake_detached).mean() - discriminator(targets).mean() + gradient_penalty(discriminator, targets, fake_detached)
                optimizer_d.zero_grad(set_to_none=True); loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
                optimizer_d.step()
            else:
                loss_d = torch.zeros((), device=device)

            fake = generator(sources, prior, fs)
            loss_tc = temporal_loss(fake, targets)
            if phase == "warmup":
                loss_gan = torch.zeros((), device=device)
                loss_amp = torch.zeros((), device=device)
                loss_psd = torch.zeros((), device=device)
                loss_stft = torch.zeros((), device=device)
                loss_g = loss_tc
            else:
                loss_gan = -discriminator(fake).mean()
                loss_amp = amplitude_loss(fake, targets)
                if args.variant == "no_psd":
                    loss_psd = torch.zeros((), device=device)
                    loss_stft = torch.zeros((), device=device)
                else:
                    loss_psd = psd_loss(fake, targets, fs)
                    loss_stft = local_stft_loss(fake, targets)
                reconstruction = 5.0 * loss_tc + 3.0 * loss_amp + loss_psd + loss_stft
                loss_g = reconstruction + adversarial_weight * loss_gan
            optimizer_g.zero_grad(set_to_none=True); loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            optimizer_g.step()
            update_ema(generator_ema, generator, decay=args.ema_decay)
            for key, value in (("g", loss_g), ("d", loss_d), ("gan", loss_gan), ("tc", loss_tc), ("amp", loss_amp), ("psd", loss_psd), ("stft", loss_stft)):
                totals[key] += float(value.detach().cpu())
            n_batches += 1
        if epoch == args.warmup_epochs:
            generator_ema.load_state_dict(generator.state_dict())
        row = {"epoch": epoch, "phase": phase, "adversarial_weight": adversarial_weight,
               **{key: value / n_batches for key, value in totals.items()}}
        history.append(row); print(json.dumps(row), flush=True)

    generator_ema.eval(); test_sources = torch.from_numpy(test_eeg[:, SOURCE_INDICES]); predictions = []
    with torch.no_grad():
        for start in range(0, len(test_sources), args.batch_size):
            predictions.append(generator_ema(test_sources[start:start + args.batch_size].to(device), prior, fs).cpu().numpy())
    fake_standardized = np.concatenate(predictions)
    fake_uv = fake_standardized * channel_std[FRONTAL_INDICES][None, :, None] + channel_mean[FRONTAL_INDICES][None, :, None]
    real_uv = data["eeg"][test_idx][:, FRONTAL_INDICES]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, fake_uv=fake_uv.astype(np.float32), real_uv=real_uv.astype(np.float32), test_idx=test_idx,
             test_subjects=subjects[test_idx], effective_prior_weights=effective_prior_weights,
             reference_granger_prior_weights=granger_prior_weights, channel_mean=channel_mean, channel_std=channel_std,
             history_json=np.asarray(json.dumps(history)), dataset=np.asarray(args.dataset), fold=np.int64(args.fold),
             variant=np.asarray(args.variant), seed=np.int64(args.seed), run_tag=np.asarray(args.run_tag),
             warmup_epochs=np.int64(args.warmup_epochs), adversarial_ramp_epochs=np.int64(args.adversarial_ramp_epochs),
             grad_clip=np.float32(args.grad_clip), ema_decay=np.float32(args.ema_decay),
             ema_reset_after_warmup=np.bool_(args.warmup_epochs > 0),
             psd_definition=np.asarray("differentiable_welch_density_nperseg256_50pct_overlap"),
             sampling_rate_hz=np.float32(fs), generated_at_utc=np.asarray(datetime.now(timezone.utc).isoformat()))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator_ema": generator_ema.state_dict(), "dataset": args.dataset, "fold": args.fold,
                "variant": args.variant, "seed": args.seed, "run_tag": args.run_tag, "epochs": args.epochs,
                "batch_size": args.batch_size, "prior_path": str(used_prior_path),
                "warmup_epochs": args.warmup_epochs, "adversarial_ramp_epochs": args.adversarial_ramp_epochs,
                "grad_clip": args.grad_clip, "ema_decay": args.ema_decay,
                "ema_reset_after_warmup": args.warmup_epochs > 0,
                "psd_definition": "differentiable_welch_density_nperseg256_50pct_overlap",
                "effective_prior_weights": effective_prior_weights,
                "reference_granger_prior_weights": granger_prior_weights,
                "train_indices_sha256": sha256_indices(train_idx), "channel_mean": channel_mean,
                "channel_std": channel_std, "history": history}, model_path)
    print(f"Saved predictions: {output_path}"); print(f"Saved checkpoint: {model_path}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
