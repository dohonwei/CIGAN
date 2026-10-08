"""Train or generate one canonical E4 baseline fold."""
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from e4_common import BASELINES, checkpoint_path, make_generator, prediction_path
from e1_common import (DATASETS, FRONTAL_INDICES, SOURCE_INDICES,
                       fit_channel_scaler, load_dataset, load_fold,
                       set_deterministic_seed, sha256_indices, transform_eeg,
                       validate_args)
from e1_model import Discriminator, gradient_penalty, temporal_loss, update_ema


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--fold", required=True, type=int, choices=range(1, 6)); parser.add_argument("--model", required=True, choices=BASELINES)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--run-tag", default="e4_final_e100")
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--adversarial-ramp-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--overwrite", action="store_true"); args = parser.parse_args()
    if not 0 <= args.warmup_epochs < args.epochs: raise ValueError("--warmup-epochs must satisfy 0 <= warmup < epochs")
    if args.adversarial_ramp_epochs <= 0: raise ValueError("--adversarial-ramp-epochs must be positive")
    if args.grad_clip <= 0: raise ValueError("--grad-clip must be positive")
    if not 0. <= args.ema_decay < 1.: raise ValueError("--ema-decay must satisfy 0 <= decay < 1")
    validate_args(args.dataset, args.fold); set_deterministic_seed(args.seed)
    output_path = prediction_path(args.dataset, args.fold, args.model, args.seed, args.run_tag)
    model_path = checkpoint_path(args.dataset, args.fold, args.model, args.seed, args.run_tag)
    if output_path.exists() and not args.overwrite: raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    data = load_dataset(args.dataset); train_idx, test_idx = load_fold(args.dataset, args.fold); subjects = data["subjects"]
    if np.intersect1d(np.unique(subjects[train_idx]), np.unique(subjects[test_idx])).size: raise ValueError("Subject leakage detected")
    channel_mean, channel_std = fit_channel_scaler(data["eeg"], train_idx)
    train_eeg = transform_eeg(data["eeg"][train_idx], channel_mean, channel_std)
    test_eeg = transform_eeg(data["eeg"][test_idx], channel_mean, channel_std)
    history = []

    if args.model == "spline":
        import mne
        from e1_common import CHANNEL_NAMES
        # Perform actual spherical-spline EEG interpolation using standard 10-20 positions.
        n_test, _, n_times = data["eeg"][test_idx].shape
        flattened = np.zeros((32, n_test * n_times), dtype=np.float64)
        flattened[SOURCE_INDICES] = data["eeg"][test_idx][:, SOURCE_INDICES].transpose(1, 0, 2).reshape(30, -1)
        info = mne.create_info(CHANNEL_NAMES.tolist(), sfreq=DATASETS[args.dataset]["fs"], ch_types="eeg", verbose="ERROR")
        raw = mne.io.RawArray(flattened, info, verbose="ERROR")
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="raise", verbose="ERROR")
        raw.info["bads"] = ["Fp1", "Fp2"]
        raw.interpolate_bads(reset_bads=False, mode="accurate", verbose="ERROR")
        fake_uv = raw.get_data(picks=["Fp1", "Fp2"]).reshape(2, n_test, n_times).transpose(1, 0, 2).astype(np.float32)
    else:
        train_set = TensorDataset(torch.from_numpy(train_eeg[:, SOURCE_INDICES]), torch.from_numpy(train_eeg[:, FRONTAL_INDICES]))
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                            generator=torch.Generator().manual_seed(args.seed), pin_memory=torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generator = make_generator(args.model).to(device); generator_ema = copy.deepcopy(generator).to(device); discriminator = Discriminator().to(device)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(.5, .9)); optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(.5, .9))
        for epoch in range(1, args.epochs + 1):
            phase = "warmup" if epoch <= args.warmup_epochs else "full"
            adversarial_weight = 0. if phase == "warmup" else min(1., (epoch - args.warmup_epochs) / args.adversarial_ramp_epochs)
            generator.train(); discriminator.train(); total_g = total_d = total_gan = total_tc = 0.; n_batches = 0
            for sources_cpu, targets_cpu in tqdm(loader, desc=f"E4 {args.dataset} {args.model} f{args.fold} e{epoch}", leave=False):
                sources, targets = sources_cpu.to(device, non_blocking=True), targets_cpu.to(device, non_blocking=True)
                if adversarial_weight > 0.:
                    with torch.no_grad(): fake_detached = generator(sources)
                    loss_d = discriminator(fake_detached).mean() - discriminator(targets).mean() + gradient_penalty(discriminator, targets, fake_detached)
                    optimizer_d.zero_grad(set_to_none=True); loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip); optimizer_d.step()
                else:
                    loss_d = torch.zeros((), device=device)
                fake = generator(sources); loss_tc = temporal_loss(fake, targets)
                loss_gan = -discriminator(fake).mean() if adversarial_weight > 0. else torch.zeros((), device=device)
                loss_g = loss_tc if phase == "warmup" else 10. * loss_tc + adversarial_weight * loss_gan
                optimizer_g.zero_grad(set_to_none=True); loss_g.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip); optimizer_g.step()
                update_ema(generator_ema, generator, args.ema_decay)
                total_g += float(loss_g.detach().cpu()); total_d += float(loss_d.detach().cpu()); total_gan += float(loss_gan.detach().cpu()); total_tc += float(loss_tc.detach().cpu()); n_batches += 1
            if epoch == args.warmup_epochs: generator_ema.load_state_dict(generator.state_dict())
            row = {"epoch": epoch, "phase": phase, "adversarial_weight": adversarial_weight,
                   "g": total_g / n_batches, "d": total_d / n_batches, "gan": total_gan / n_batches, "tc": total_tc / n_batches}
            history.append(row); print(json.dumps(row), flush=True)
        generator_ema.eval(); test_sources = torch.from_numpy(test_eeg[:, SOURCE_INDICES]); predictions = []
        with torch.no_grad():
            for start in range(0, len(test_sources), args.batch_size): predictions.append(generator_ema(test_sources[start:start + args.batch_size].to(device)).cpu().numpy())
        fake_standardized = np.concatenate(predictions)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generator_ema": generator_ema.state_dict(), "dataset": args.dataset, "fold": args.fold, "model": args.model,
                    "seed": args.seed, "run_tag": args.run_tag, "epochs": args.epochs, "batch_size": args.batch_size,
                    "warmup_epochs": args.warmup_epochs, "adversarial_ramp_epochs": args.adversarial_ramp_epochs,
                    "grad_clip": args.grad_clip, "ema_decay": args.ema_decay,
                    "ema_reset_after_warmup": args.warmup_epochs > 0,
                    "train_indices_sha256": sha256_indices(train_idx), "channel_mean": channel_mean, "channel_std": channel_std,
                    "loss_definition": "WGAN-GP + 10 * temporal L1", "history": history}, model_path)

    if args.model != "spline":
        fake_uv = fake_standardized * channel_std[FRONTAL_INDICES][None, :, None] + channel_mean[FRONTAL_INDICES][None, :, None]
    real_uv = data["eeg"][test_idx][:, FRONTAL_INDICES]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, fake_uv=fake_uv.astype(np.float32), real_uv=real_uv.astype(np.float32), test_idx=test_idx,
             test_subjects=subjects[test_idx], channel_mean=channel_mean, channel_std=channel_std,
             history_json=np.asarray(json.dumps(history)), dataset=np.asarray(args.dataset), fold=np.int64(args.fold), model=np.asarray(args.model),
             seed=np.int64(args.seed), run_tag=np.asarray(args.run_tag), sampling_rate_hz=np.float32(DATASETS[args.dataset]["fs"]),
             warmup_epochs=np.int64(args.warmup_epochs), adversarial_ramp_epochs=np.int64(args.adversarial_ramp_epochs),
             grad_clip=np.float32(args.grad_clip), ema_decay=np.float32(args.ema_decay),
             generated_at_utc=np.asarray(datetime.now(timezone.utc).isoformat()))
    print(f"Saved predictions: {output_path}");
    if args.model != "spline": print(f"Saved checkpoint: {model_path}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
