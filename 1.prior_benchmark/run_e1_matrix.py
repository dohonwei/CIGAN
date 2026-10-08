"""Run the required E1 matrix while skipping completed artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from e1_common import DATASETS, PRIORS, prediction_path, prior_path


CODE_DIR = Path(__file__).resolve().parent


def run_command(arguments: list[str]) -> None:
    print("RUN:", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True, cwd=CODE_DIR.parents[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=[*DATASETS, "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--adversarial-ramp-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--priors", nargs="+", choices=PRIORS, default=list(PRIORS))
    parser.add_argument("--stage", choices=["priors", "train", "all"], default="all")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("--warmup-epochs must satisfy 0 <= warmup < epochs")
    if args.adversarial_ramp_epochs <= 0:
        raise ValueError("--adversarial-ramp-epochs must be positive")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must satisfy 0 <= decay < 1")

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for fold in args.folds:
        if fold not in range(1, 6):
            raise ValueError(f"Invalid fold: {fold}")

    for dataset in datasets:
        for fold in args.folds:
            for prior in args.priors:
                prior_output = prior_path(dataset, fold, prior, args.seed)
                prediction_output = prediction_path(
                    dataset, fold, prior, args.seed, args.run_tag
                )

                if args.stage in {"priors", "all"}:
                    if prior_output.exists():
                        print(f"SKIP existing prior: {prior_output}", flush=True)
                    else:
                        run_command([
                            sys.executable,
                            str(CODE_DIR / "compute_priors.py"),
                            "--dataset", dataset,
                            "--fold", str(fold),
                            "--prior", prior,
                            "--seed", str(args.seed),
                        ])

                if args.stage in {"train", "all"}:
                    if prediction_output.exists():
                        print(f"SKIP existing prediction: {prediction_output}", flush=True)
                    else:
                        if not prior_output.exists():
                            raise FileNotFoundError(
                                f"Missing {prior_output}; run with --stage priors or --stage all first"
                            )
                        run_command([
                            sys.executable,
                            str(CODE_DIR / "train_prior_benchmark.py"),
                            "--dataset", dataset,
                            "--fold", str(fold),
                            "--prior", prior,
                            "--seed", str(args.seed),
                            "--epochs", str(args.epochs),
                            "--batch-size", str(args.batch_size),
                            "--warmup-epochs", str(args.warmup_epochs),
                            "--adversarial-ramp-epochs", str(args.adversarial_ramp_epochs),
                            "--grad-clip", str(args.grad_clip),
                            "--ema-decay", str(args.ema_decay),
                            *(["--run-tag", args.run_tag] if args.run_tag else []),
                        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
