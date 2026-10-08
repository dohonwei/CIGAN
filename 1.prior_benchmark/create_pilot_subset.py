"""Create an immutable, outcome-blind subject subset for the E1 pilot."""
from __future__ import annotations

import argparse
import json

import numpy as np

from e1_common import DATASETS, load_dataset, pilot_split_path, sha256_indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="deap", choices=DATASETS)
    parser.add_argument("--n-subjects", type=int, default=8)
    parser.add_argument("--n-test-subjects", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = load_dataset(args.dataset)
    subjects = np.unique(data["subjects"])
    if not 2 <= args.n_subjects <= len(subjects):
        raise ValueError("--n-subjects must be between 2 and the dataset subject count")
    if not 1 <= args.n_test_subjects < args.n_subjects:
        raise ValueError("--n-test-subjects must be positive and smaller than --n-subjects")

    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(subjects, size=args.n_subjects, replace=False))
    test_subjects = np.sort(rng.choice(selected, size=args.n_test_subjects, replace=False))
    train_subjects = np.setdiff1d(selected, test_subjects)
    train_idx = np.flatnonzero(np.isin(data["subjects"], train_subjects)).astype(np.int64)
    test_idx = np.flatnonzero(np.isin(data["subjects"], test_subjects)).astype(np.int64)
    split_id = f"subjects{args.n_subjects}_test{args.n_test_subjects}_seed{args.seed}"
    output = pilot_split_path(args.dataset, args.seed, args.n_subjects, args.n_test_subjects)
    payload = {
        "dataset": args.dataset, "split_id": split_id, "seed": args.seed,
        "selected_subjects": selected.tolist(), "train_subjects": train_subjects.tolist(),
        "test_subjects": test_subjects.tolist(), "n_train_trials": len(train_idx),
        "n_test_trials": len(test_idx), "train_indices_sha256": sha256_indices(train_idx),
        "test_indices_sha256": sha256_indices(test_idx),
        "selection_rule": "uniform random subjects before model training; subject-disjoint split",
    }
    if output.exists():
        with np.load(output, allow_pickle=False) as archive:
            existing = {"train_idx": archive["train_idx"], "test_idx": archive["test_idx"]}
        if not np.array_equal(existing["train_idx"], train_idx) or not np.array_equal(existing["test_idx"], test_idx):
            raise FileExistsError(f"Existing immutable split differs: {output}")
        print(f"Verified existing immutable pilot split: {output}")
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, train_idx=train_idx, test_idx=test_idx,
             dataset=np.asarray(args.dataset), split_id=np.asarray(split_id), seed=np.int64(args.seed),
             selected_subjects=selected, train_subjects=train_subjects, test_subjects=test_subjects)
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved immutable pilot split: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
