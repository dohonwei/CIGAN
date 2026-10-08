"""Check whether the original CIGAN output can be reused in Reviewer 4.1 analyses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REVISION_ROOT / "code"))
from legacy_artifacts import PROJECT_ROOT, load_legacy_prediction, sha256_file


PATHS = {
    "deap": PROJECT_ROOT / "comparison/deap/CIGAN_generated_data_single.npz",
    "hci": PROJECT_ROOT / "comparison/hci/cigan_main_generated_loso.npz",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=PATHS)
    args = parser.parse_args()
    path = PATHS[args.dataset]
    report = {
        "dataset": args.dataset,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path) if path.exists() else None,
        "status": "rejected",
        "permitted_use": "none",
    }
    try:
        protocol = "legacy" if args.dataset == "hci" else "revision"
        artifact = load_legacy_prediction(path, args.dataset, protocol=protocol)
        report.update(
            status="reference_verified",
            prediction_shape=list(artifact["fake"].shape),
            protocol=protocol,
            permitted_use="fixed-generator sensitivity reference only",
            limitation=(
                "This output is the Granger-trained arm only. It cannot be combined with "
                "newly retrained alternative-prior arms as a controlled training comparison."
            ),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        report["reason"] = str(exc)
    output = REVISION_ROOT / "1.prior_benchmark/results" / f"legacy_reuse_audit_{args.dataset}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "reference_verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
