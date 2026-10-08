"""Build the revision's auditable inventory of reusable original-project outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from legacy_artifacts import PROJECT_ROOT, REVISION_ROOT, canonical_targets, sha256_file


ARTIFACTS = (
    ("E1", "deap", "granger_full_model", "comparison/deap/CIGAN_generated_data_single.npz", "revision"),
    ("E1", "hci", "granger_full_model", "comparison/hci/cigan_main_generated_loso.npz", "legacy"),
    ("E3", "deap", "full", "ablation/deap/ablation_full_fake_fp.npy", "revision"),
    ("E3", "deap", "no_causal", "ablation/deap/ablation_no_causal_fake_fp.npy", "revision"),
    ("E3", "deap", "no_fen", "ablation/deap/ablation_no_fen_fake_fp.npy", "revision"),
    ("E3", "deap", "no_psd", "ablation/deap/ablation_no_psd_fake_fp.npy", "revision"),
    ("E3", "hci", "full", "ablation/hci/ablation_full_fake_fp.npy", "legacy"),
    ("E3", "hci", "no_causal", "ablation/hci/ablation_no_causal_fake_fp.npy", "legacy"),
    ("E3", "hci", "no_fen", "ablation/hci/ablation_no_fen_fake_fp.npy", "legacy"),
    ("E3", "hci", "no_psd", "ablation/hci/ablation_no_psd_fake_fp.npy", "legacy"),
    ("E4", "deap", "cigan", "ablation/deap/ablation_full_fake_fp.npy", "revision"),
    ("E4", "deap", "hveegnet", "ablation/deap/hveegnet_fake_fp.npy", "revision"),
    ("E4", "deap", "tieeegnet", "ablation/deap/tie_eegnet_fake_fp.npy", "revision"),
    ("E4", "deap", "wavenet", "ablation/deap/wavenet_fake_fp.npy", "revision"),
    ("E4", "deap", "spline", "ablation/deap/spline_fake_fp.npy", "revision"),
    ("E4", "deap", "encoder_decoder", "ablation/deap/encdec_fake_fp.npy", "revision"),
    ("E4", "hci", "cigan", "ablation/hci/ablation_full_fake_fp.npy", "legacy"),
    ("E4", "hci", "hveegnet", "ablation/hci/hveegnet_fake_fp.npy", "legacy"),
    ("E4", "hci", "tieeegnet", "ablation/hci/tie_eegnet_fake_fp.npy", "legacy"),
    ("E4", "hci", "wavenet", "ablation/hci/wavenet_fake_fp.npy", "legacy"),
    ("E4", "hci", "spline", "ablation/hci/spline_fake_fp.npy", "legacy"),
    ("E4", "hci", "encoder_decoder", "ablation/hci/encdec_fake_fp.npy", "legacy"),
    ("E5", "deap", "cigan_mc50", "comparison/deap/CIGAN_generated_data_mc50.npz", "revision"),
    ("E5", "hci", "cigan_mc50", "comparison/hci/CIGAN_generated_data_mc50.npz", "legacy"),
)


def inspect(experiment: str, dataset: str, role: str, relative: str, protocol: str) -> dict:
    path = PROJECT_ROOT / relative
    row = {
        "experiment": experiment,
        "dataset": dataset,
        "role": role,
        "path": relative,
        "protocol": protocol,
        "exists": path.exists(),
        "sha256": "",
        "fake_shape": "",
        "reference_match": False,
        "status": "missing",
        "allowed_use": "none",
        "reason": "file is missing",
    }
    if not path.exists():
        return row
    row["sha256"] = sha256_file(path)
    try:
        canonical, _, _ = canonical_targets(dataset, protocol)
        if path.suffix == ".npy":
            fake = np.load(path, allow_pickle=False)
            row["fake_shape"] = "x".join(map(str, fake.shape))
            if not np.isfinite(fake).all():
                row.update(status="rejected", reason="artifact contains non-finite values")
            elif fake.shape != canonical.shape:
                row.update(status="rejected_protocol", reason="shape differs from E0 canonical targets")
            else:
                row.update(
                    status="verified_reusable",
                    allowed_use="common-metric re-evaluation",
                    reason=(
                        f"prediction-only array matches the E0 {protocol} canonical shape; "
                        "canonical targets are supplied by index during evaluation"
                    ),
                )
            return row
        with np.load(path, allow_pickle=False) as archive:
            fake = archive["fake_fp"]
            real = archive["real_fp"]
        row["fake_shape"] = "x".join(map(str, fake.shape))
        finite = bool(np.isfinite(fake).all() and np.isfinite(real).all())
        shape_match = fake.shape == real.shape == canonical.shape
        reference_match = bool(shape_match and np.array_equal(real, canonical))
        row["reference_match"] = reference_match
        if finite and reference_match:
            allowed_use = "common-metric re-evaluation"
            if experiment == "E1":
                allowed_use = "fixed-generator sensitivity reference only"
            elif experiment == "E5":
                allowed_use = "MC-dropout calibration re-evaluation"
            row.update(
                status="verified_reusable",
                allowed_use=allowed_use,
                reason=f"real_fp exactly equals the E0 {protocol} canonical frontal targets",
            )
        elif not finite:
            row.update(status="rejected", reason="artifact contains non-finite values")
        elif not shape_match:
            row.update(status="rejected_protocol", reason="shape differs from E0 canonical targets")
        else:
            maximum = float(np.max(np.abs(real - canonical)))
            correlation = float(np.corrcoef(real.ravel(), canonical.ravel())[0, 1])
            row.update(
                status="rejected_protocol",
                reason=f"real_fp differs from E0 canonical data (max_abs_diff={maximum:.6g}, r={correlation:.6g})",
            )
    except (KeyError, ValueError, OSError) as exc:
        row.update(status="rejected", reason=str(exc))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REVISION_ROOT / "reuse_inventory")
    args = parser.parse_args()
    rows = [inspect(*entry) for entry in ARTIFACTS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "legacy_artifact_inventory.csv", index=False, encoding="utf-8-sig")
    summary = frame.groupby(["experiment", "dataset", "status"]).size().reset_index(name="count")
    payload = {
        "policy": "Only verified_reusable artifacts may enter common-metric revision analyses.",
        "inventory": rows,
        "summary": summary.to_dict(orient="records"),
    }
    (args.output_dir / "legacy_artifact_inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved inventory to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
