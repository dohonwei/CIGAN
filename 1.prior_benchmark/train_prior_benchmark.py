"""Train one dataset x fold x prior x seed E1 task."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from e1_common import (
    DATASETS, FRONTAL_INDICES, PRIORS, SOURCE_INDICES, checkpoint_path,
    fit_channel_scaler, load_dataset, load_fold, load_pilot_split,
    pilot_checkpoint_path, pilot_prediction_path, pilot_prior_path,
    prediction_path, prior_path,
    set_deterministic_seed, sha256_indices, transform_eeg, validate_args,
)
from e1_model import (
    CIGANGenerator, Discriminator, amplitude_loss, gradient_penalty,
    local_stft_loss, psd_loss, temporal_loss, update_ema,
)


def load_verified_prior(dataset: str, fold: int, prior: str, seed: int, train_idx: np.ndarray, path=None):
    path = path or prior_path(dataset, fold, prior, seed)
    if not path.exists():
        raise FileNotFoundError(f"Run compute_priors.py first: {path}")
    with np.load(path, allow_pickle=False) as archive:
        weights = archive["weights"].astype(np.float32)
        stored_hash = str(archive["train_indices_sha256"].item())
        stored_sources = archive["source_indices"].astype(np.int64)
    if stored_hash != sha256_indices(train_idx):
        raise ValueError("Prior was not computed from the current training fold")
    if not np.array_equal(stored_sources, SOURCE_INDICES):
        raise ValueError("Prior source-channel order differs from E1 source order")
    if weights.shape != (30,) or not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError("Invalid prior weights")
    return weights, path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--fold", required=True, type=int, choices=range(1, 6))
    parser.add_argument("--prior", required=True, choices=PRIORS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--adversarial-ramp-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pilot-split", type=str, default=None)
    args = parser.parse_args()
    validate_args(args.dataset, args.fold, args.prior)
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs must satisfy 0 <= warmup < epochs")
    if args.adversarial_ramp_epochs <= 0:
        raise ValueError("--adversarial-ramp-epochs must be positive")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must satisfy 0 <= decay < 1")
    set_deterministic_seed(args.seed)

    split_id = None
    if args.pilot_split:
        train_idx, test_idx, split_id = load_pilot_split(Path(args.pilot_split), args.dataset)
        output_path = pilot_prediction_path(args.dataset, split_id, args.prior, args.seed)
        model_path = pilot_checkpoint_path(args.dataset, split_id, args.prior, args.seed)
        selected_prior_path = pilot_prior_path(args.dataset, split_id, args.prior, args.seed)
    else:
        train_idx, test_idx = load_fold(args.dataset, args.fold)
        output_path = prediction_path(args.dataset, args.fold, args.prior, args.seed, args.run_tag)
        model_path = checkpoint_path(args.dataset, args.fold, args.prior, args.seed, args.run_tag)
        selected_prior_path = prior_path(args.dataset, args.fold, args.prior, args.seed)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")

    data = load_dataset(args.dataset)
    subjects = data["subjects"]
    if np.intersect1d(np.unique(subjects[train_idx]), np.unique(subjects[test_idx])).size:
        raise ValueError("Subject leakage detected")

    prior_weights, used_prior_path = load_verified_prior(
        args.dataset, args.fold, args.prior, args.seed, train_idx, selected_prior_path
    )
    channel_mean, channel_std = fit_channel_scaler(data["eeg"], train_idx)
    train_standardized = transform_eeg(data["eeg"][train_idx], channel_mean, channel_std)
    test_standardized = transform_eeg(data["eeg"][test_idx], channel_mean, channel_std)

    sources_train = torch.from_numpy(train_standardized[:, SOURCE_INDICES, :])
    targets_train = torch.from_numpy(train_standardized[:, FRONTAL_INDICES, :])
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(sources_train, targets_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = DATASETS[args.dataset]["fs"]
    prior_tensor = torch.from_numpy(prior_weights).to(device)
    generator = CIGANGenerator().to(device)
    discriminator = Discriminator().to(device)
    generator_ema = copy.deepcopy(generator).to(device)
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
        generator.train()
        discriminator.train()
        totals = {key: 0.0 for key in ("g", "d", "gan", "tc", "amp", "psd", "stft")}
        n_batches = 0
        progress = tqdm(loader, desc=f"{args.dataset} {args.prior} f{args.fold} e{epoch}", leave=False)
        for sources_cpu, targets_cpu in progress:
            sources = sources_cpu.to(device, non_blocking=True)
            targets = targets_cpu.to(device, non_blocking=True)

            if adversarial_weight > 0.0:
                with torch.no_grad():
                    fake_detached = generator(sources, prior_tensor, fs)
                loss_d = (
                    discriminator(fake_detached).mean()
                    - discriminator(targets).mean()
                    + gradient_penalty(discriminator, targets, fake_detached)
                )
                optimizer_d.zero_grad(set_to_none=True)
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip)
                optimizer_d.step()
            else:
                loss_d = torch.zeros((), device=device)

            fake = generator(sources, prior_tensor, fs)
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
                loss_psd = psd_loss(fake, targets, fs)
                loss_stft = local_stft_loss(fake, targets)
                # Full-phase coefficients exactly match the revision manuscript.
                reconstruction = (
                    5.0 * loss_tc
                    + 3.0 * loss_amp
                    + loss_psd
                    + loss_stft
                )
                loss_g = reconstruction + adversarial_weight * loss_gan
            optimizer_g.zero_grad(set_to_none=True)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), args.grad_clip)
            optimizer_g.step()
            update_ema(generator_ema, generator, decay=args.ema_decay)

            for key, value in (
                ("g", loss_g), ("d", loss_d), ("gan", loss_gan), ("tc", loss_tc),
                ("amp", loss_amp), ("psd", loss_psd), ("stft", loss_stft),
            ):
                totals[key] += float(value.detach().cpu())
            n_batches += 1

        if epoch == args.warmup_epochs:
            generator_ema.load_state_dict(generator.state_dict())

        row = {
            "epoch": epoch,
            "phase": phase,
            "adversarial_weight": adversarial_weight,
            **{key: value / n_batches for key, value in totals.items()},
        }
        history.append(row)
        print(json.dumps(row))

    generator_ema.eval()
    test_sources = torch.from_numpy(test_standardized[:, SOURCE_INDICES, :])
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_sources), args.batch_size):
            batch = test_sources[start : start + args.batch_size].to(device)
            predictions.append(generator_ema(batch, prior_tensor, fs).cpu().numpy())
    fake_standardized = np.concatenate(predictions, axis=0)
    frontal_mean = channel_mean[FRONTAL_INDICES][None, :, None]
    frontal_std = channel_std[FRONTAL_INDICES][None, :, None]
    fake_uv = fake_standardized * frontal_std + frontal_mean
    real_uv = data["eeg"][test_idx][:, FRONTAL_INDICES, :]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fake_uv=fake_uv.astype(np.float32),
        real_uv=real_uv.astype(np.float32),
        test_idx=test_idx,
        test_subjects=subjects[test_idx],
        prior_weights=prior_weights,
        channel_mean=channel_mean,
        channel_std=channel_std,
        history_json=np.asarray(json.dumps(history)),
        dataset=np.asarray(args.dataset),
        fold=np.int64(args.fold),
        split_id=np.asarray(split_id or ""),
        prior=np.asarray(args.prior),
        seed=np.int64(args.seed),
        run_tag=np.asarray(args.run_tag or ""),
        warmup_epochs=np.int64(args.warmup_epochs),
        adversarial_ramp_epochs=np.int64(args.adversarial_ramp_epochs),
        grad_clip=np.float32(args.grad_clip),
        ema_decay=np.float32(args.ema_decay),
        ema_reset_after_warmup=np.bool_(args.warmup_epochs > 0),
        psd_definition=np.asarray("differentiable_welch_density_nperseg256_50pct_overlap"),
        sampling_rate_hz=np.float32(fs),
        generated_at_utc=np.asarray(datetime.now(timezone.utc).isoformat()),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_ema": generator_ema.state_dict(),
            "dataset": args.dataset,
            "fold": args.fold,
            "split_id": split_id,
            "prior": args.prior,
            "seed": args.seed,
            "run_tag": args.run_tag,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "warmup_epochs": args.warmup_epochs,
            "adversarial_ramp_epochs": args.adversarial_ramp_epochs,
            "grad_clip": args.grad_clip,
            "ema_decay": args.ema_decay,
            "ema_reset_after_warmup": args.warmup_epochs > 0,
            "psd_definition": "differentiable_welch_density_nperseg256_50pct_overlap",
            "prior_path": str(used_prior_path),
            "train_indices_sha256": sha256_indices(train_idx),
            "channel_mean": channel_mean,
            "channel_std": channel_std,
            "history": history,
        },
        model_path,
    )
    print(f"Saved predictions: {output_path}")
    print(f"Saved checkpoint: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
