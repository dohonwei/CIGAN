"""Train E7 methods on clean training repeats and predict every noisy condition."""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset

from e7_common import (CHECKPOINT_DIR, MODELS, PREDICTION_DIR, PRIOR_DIR, TRAIN_REPEATS,
                       VALID_REPEATS, Discriminator, clean_indices, fit_scalers,
                       granger_prior, load_data, make_generator, save_json, seed_everything)


def normalized(x, mean, std): return (x - mean) / std


def predict_deep(model, x, x_mean, x_std, y_mean, y_std, device, batch_size):
    model.eval(); predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(normalized(x[start:start + batch_size], x_mean, x_std)).to(device)
            predictions.append((model(xb).cpu().numpy() * y_std + y_mean).astype("float32"))
    return np.concatenate(predictions)


def tagged(name, run_tag):
    return f"{name}_{run_tag}" if run_tag else name


def train_ridge(data, seed, run_tag):
    train_ids, valid_ids = clean_indices(data, TRAIN_REPEATS), clean_indices(data, VALID_REPEATS)
    x_train, y_train = data["clean_source_bank"][train_ids], data["true_target_bank"][train_ids]
    x_valid, y_valid = data["clean_source_bank"][valid_ids], data["true_target_bank"][valid_ids]
    xt = x_train.transpose(0, 2, 1).reshape(-1, 30); yt = y_train.transpose(0, 2, 1).reshape(-1, 2)
    xv = x_valid.transpose(0, 2, 1).reshape(-1, 30); yv = y_valid.transpose(0, 2, 1).reshape(-1, 2)
    best = None
    for alpha in (.01, .1, 1., 10.):
        model = Ridge(alpha=alpha).fit(xt, yt)
        score = float(np.mean(np.abs(model.predict(xv) - yv)))
        if best is None or score < best[0]: best = (score, alpha, model)
    _, alpha, model = best; chunks = []
    noisy = data["noisy_sources"]
    for start in range(0, len(noisy), 32):
        x = noisy[start:start + 32]; flat = x.transpose(0, 2, 1).reshape(-1, 30)
        pred = model.predict(flat).reshape(len(x), x.shape[-1], 2).transpose(0, 2, 1)
        chunks.append(pred.astype("float32"))
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    stem = tagged("ridge", run_tag)
    np.savez_compressed(PREDICTION_DIR / f"{stem}.npz", prediction=np.concatenate(chunks))
    save_json(CHECKPOINT_DIR / f"{stem}.json", {"alpha": alpha, "validation_mae": best[0], "seed": seed, "run_tag": run_tag})
    print(f"ridge: alpha={alpha:g}, validation_mae={best[0]:.6f}")


def gradient_penalty(discriminator, real, fake):
    alpha = torch.rand(real.shape[0], 1, 1, device=real.device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True); score = discriminator(mixed)
    gradient = torch.autograd.grad(score, mixed, torch.ones_like(score), create_graph=True)[0]
    return 10 * (gradient.reshape(real.shape[0], -1).norm(2, dim=1) - 1).square().mean()


def spectral_loss(fake, real, n_fft):
    window = torch.hann_window(n_fft, device=fake.device)
    f = torch.stft(fake.flatten(0, 1), n_fft, n_fft // 2, window=window, return_complex=True).abs()
    r = torch.stft(real.flatten(0, 1), n_fft, n_fft // 2, window=window, return_complex=True).abs()
    if n_fft == 256: f, r = f.square().mean(-1), r.square().mean(-1)
    return F.l1_loss(f, r)


def train_deep(name, data, args, device):
    train_ids, valid_ids = clean_indices(data, TRAIN_REPEATS), clean_indices(data, VALID_REPEATS)
    x_train, y_train = data["clean_source_bank"][train_ids], data["true_target_bank"][train_ids]
    x_valid, y_valid = data["clean_source_bank"][valid_ids], data["true_target_bank"][valid_ids]
    scalers = fit_scalers(x_train, y_train); x_mean, x_std, y_mean, y_std = scalers
    train_set = TensorDataset(torch.from_numpy(normalized(x_train, x_mean, x_std)), torch.from_numpy(normalized(y_train, y_mean, y_std)))
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    prior_weights, prior_details = None, None
    if name == "cigan_granger":
        prior_weights, prior_details = granger_prior(x_train, y_train, args.granger_lag, device)
        PRIOR_DIR.mkdir(parents=True, exist_ok=True)
        prior_stem = tagged("cigan_granger", args.run_tag)
        np.savez(PRIOR_DIR / f"{prior_stem}.npz", weights=prior_weights,
                 train_repeats=np.asarray(TRAIN_REPEATS), **prior_details)
        save_json(PRIOR_DIR / f"{prior_stem}.json", {
            "method": "Granger source-to-target prior with BH-FDR",
            "lag": args.granger_lag, "train_repeats": list(TRAIN_REPEATS),
            "validation_repeats_used_for_prior": [], "test_repeats_used_for_prior": [],
            "n_significant_edges": prior_details["n_significant_edges"],
            "n_total_edges": prior_details["n_total_edges"],
            "weights": prior_weights.tolist(), "seed": args.seed})
    model, discriminator = make_generator(name, float(data["fs"]), prior_weights).to(device), Discriminator().to(device)
    opt_g = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(.5, .9)); opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(.5, .9))
    xv = torch.from_numpy(normalized(x_valid, x_mean, x_std)).to(device); yv = torch.from_numpy(normalized(y_valid, y_mean, y_std)).to(device)
    best_loss, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            adversarial_weight = min(1.0, max(0.0, (epoch - args.warmup_epochs) / args.adversarial_ramp_epochs))
            if adversarial_weight > 0:
                for _ in range(args.critic_steps):
                    opt_d.zero_grad(set_to_none=True); fake = model(xb).detach()
                    d_loss = discriminator(fake).mean() - discriminator(yb).mean() + gradient_penalty(discriminator, yb, fake)
                    d_loss.backward(); torch.nn.utils.clip_grad_norm_(discriminator.parameters(), args.grad_clip); opt_d.step()
            opt_g.zero_grad(set_to_none=True); fake = model(xb)
            temporal = F.l1_loss(fake, yb); amplitude = F.l1_loss(fake.abs().mean(-1), yb.abs().mean(-1))
            if epoch <= args.warmup_epochs:
                # Establish the known source-to-target mapping before adversarial training.
                loss = temporal
            else:
                reconstruction = 5 * temporal + 3 * amplitude + spectral_loss(fake, yb, 256) + spectral_loss(fake, yb, 64)
                loss = reconstruction - adversarial_weight * discriminator(fake).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); opt_g.step()
        model.eval()
        with torch.no_grad(): val_loss = float(F.l1_loss(model(xv), yv).cpu())
        history.append({"epoch": epoch, "phase": "warmup" if epoch <= args.warmup_epochs else "full", "validation_mae_normalized": val_loss})
        print(f"{name}: epoch {epoch:03d}, phase={history[-1]['phase']}, validation_mae={val_loss:.6f}", flush=True)
        if val_loss < best_loss - args.min_delta: best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                print(f"{name}: early stopping at epoch {epoch}"); break
    model.load_state_dict(best_state); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True); stem = tagged(name, args.run_tag)
    torch.save({"model": name, "state_dict": best_state, "scalers": [torch.from_numpy(v) for v in scalers], "fs": float(data["fs"]), "best_validation_mae_normalized": best_loss, "args": vars(args), "prior_weights": prior_weights, "prior_details": prior_details}, CHECKPOINT_DIR / f"{stem}.pt")
    prediction = predict_deep(model, data["noisy_sources"], x_mean, x_std, y_mean, y_std, device, args.predict_batch_size)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True); np.savez_compressed(PREDICTION_DIR / f"{stem}.npz", prediction=prediction)
    save_json(CHECKPOINT_DIR / f"{stem}_history.json", {"history": history, "best_epoch": min(history, key=lambda row: row["validation_mae_normalized"])["epoch"], "run_tag": args.run_tag})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=100); parser.add_argument("--batch-size", type=int, default=4); parser.add_argument("--predict-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--critic-steps", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=20); parser.add_argument("--adversarial-ramp-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0); parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping (default)")
    parser.add_argument("--min-delta", type=float, default=1e-4); parser.add_argument("--workers", type=int, default=0); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", default="", help="Suffix used to keep reruns separate from prior outputs")
    parser.add_argument("--granger-lag", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); args = parser.parse_args()
    if args.adversarial_ramp_epochs <= 0: parser.error("--adversarial-ramp-epochs must be positive")
    if not 0 <= args.warmup_epochs < args.epochs: parser.error("--warmup-epochs must be in [0, epochs)")
    seed_everything(args.seed); data, device, started = load_data(), torch.device(args.device), time.time()
    for name in args.models:
        if name == "ridge": train_ridge(data, args.seed, args.run_tag)
        else: train_deep(name, data, args, device)
    print(f"Completed {args.models} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__": main()
