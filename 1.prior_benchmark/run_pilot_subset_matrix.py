"""Compute and independently train every prior on one fixed pilot split."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from e1_common import PRIORS, load_pilot_split, pilot_prediction_path, pilot_prior_path

CODE_DIR = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    print("RUN:", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=CODE_DIR.parents[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="deap", choices=("deap", "hci"))
    parser.add_argument("--pilot-split", required=True)
    parser.add_argument("--priors", nargs="+", choices=PRIORS, default=list(PRIORS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--adversarial-ramp-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    args = parser.parse_args()
    _, _, split_id = load_pilot_split(Path(args.pilot_split), args.dataset)
    for prior in args.priors:
        prior_file = pilot_prior_path(args.dataset, split_id, prior, args.seed)
        if not prior_file.exists():
            run([sys.executable, str(CODE_DIR / "compute_priors.py"), "--dataset", args.dataset,
                 "--fold", "1", "--prior", prior, "--seed", str(args.seed),
                 "--pilot-split", args.pilot_split])
        else:
            print(f"SKIP existing prior: {prior_file}")
        prediction = pilot_prediction_path(args.dataset, split_id, prior, args.seed)
        if prediction.exists():
            print(f"SKIP existing prediction: {prediction}")
            continue
        run([sys.executable, str(CODE_DIR / "train_prior_benchmark.py"),
             "--dataset", args.dataset, "--fold", "1", "--prior", prior,
             "--seed", str(args.seed), "--epochs", str(args.epochs),
             "--batch-size", str(args.batch_size), "--warmup-epochs", str(args.warmup_epochs),
             "--adversarial-ramp-epochs", str(args.adversarial_ramp_epochs),
             "--grad-clip", str(args.grad_clip), "--ema-decay", str(args.ema_decay),
             "--pilot-split", args.pilot_split])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
