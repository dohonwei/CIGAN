"""Run selected E4 baselines while skipping completed predictions."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from e4_common import BASELINES, prediction_path
from e1_common import DATASETS

CODE_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True, choices=[*DATASETS, "all"])
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5]); parser.add_argument("--models", nargs="+", choices=BASELINES, default=list(BASELINES))
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128); parser.add_argument("--run-tag", default="e4_final_e100")
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--adversarial-ramp-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    args = parser.parse_args()
    if not 0 <= args.warmup_epochs < args.epochs: raise ValueError("--warmup-epochs must satisfy 0 <= warmup < epochs")
    if args.adversarial_ramp_epochs <= 0: raise ValueError("--adversarial-ramp-epochs must be positive")
    if args.grad_clip <= 0: raise ValueError("--grad-clip must be positive")
    if not 0. <= args.ema_decay < 1.: raise ValueError("--ema-decay must satisfy 0 <= decay < 1")
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    if any(fold not in range(1, 6) for fold in args.folds): raise ValueError("folds must be in 1..5")
    for dataset in datasets:
        for fold in args.folds:
            for model in args.models:
                output = prediction_path(dataset, fold, model, args.seed, args.run_tag)
                if output.exists(): print(f"SKIP existing prediction: {output}", flush=True); continue
                command = [sys.executable, str(CODE_DIR / "run_e4_baselines.py"), "--dataset", dataset, "--fold", str(fold),
                           "--model", model, "--seed", str(args.seed), "--epochs", str(args.epochs),
                           "--batch-size", str(args.batch_size), "--run-tag", args.run_tag,
                           "--warmup-epochs", str(args.warmup_epochs),
                           "--adversarial-ramp-epochs", str(args.adversarial_ramp_epochs),
                           "--grad-clip", str(args.grad_clip), "--ema-decay", str(args.ema_decay)]
                print("RUN:", " ".join(command), flush=True); subprocess.run(command, check=True, cwd=CODE_DIR.parents[0])
    return 0


if __name__ == "__main__": raise SystemExit(main())
