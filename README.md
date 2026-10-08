# CIGAN code

Code-only public snapshot of the CIGAN project and its revision experiments. No EEG recordings, labels, trained weights, predictions, or generated figures are included. The original project was not modified.

## Layout

| Directory | Purpose |
| --- | --- |
| `0.preprocessing` | DEAP and MAHNOB-HCI preprocessing and protocol audit |
| `1.prior_benchmark` | Prior construction, CIGAN training, and pilot comparison |
| `2.granger_stability` | Granger ranking stability analysis |
| `3.ablation` | CIGAN ablation training and evaluation |
| `4.baselines` | Unified baseline training and evaluation |
| `5.uncertainty` | MC-dropout and uncertainty calibration |
| `6.robustness` | Real-EEG perturbation and existing-result analyses |
| `7.synthetic_noise` | Controlled synthetic-noise experiments |
| `8.figures` | Figure regeneration |
| `code` | Shared artifact validation helpers |
| `core`, `pipeline` | Historical project scripts retained for provenance; they are not the preferred revision workflow |

Each numbered directory contains its stage's Python scripts directly. Run a stage script from the repository root with `python 1.prior_benchmark/<script>.py ...` after preparing the required input data. The scripts are a research snapshot, not an automated end-to-end package. Some analyses explicitly require historical artifacts that are not redistributed.

## Data placement

Place your own authorized datasets under `data/` or pass paths where a script exposes an input argument. The E0 audit expects canonical archives under `0.preprocessing/canonical_data/`, and writes fold assignments under `0.preprocessing/outputs/`. The HCI 256-Hz rebuild script reads `CIGAN_HCI_PREPROC_DIR` when set, otherwise `data/hci/raw`. Later stages read the E0 outputs. All these paths are ignored by Git.

The `requirements.txt` lists direct Python dependencies without version pinning. Use a PyTorch build compatible with your accelerator, or a CPU build for non-training checks. This snapshot has passed syntax checks, but the full experimental pipeline has not been rerun without the private datasets and artifacts.

DEAP data and MAHNOB-HCI data have separate licenses and are not part of this repository. Do not upload the original recordings, derived subject-level archives, model checkpoints, predictions, or local logs.

## Safety check before publishing

Run `git status --short` and `git ls-files` after adding files. Confirm that only `.py`, `.md`, `.txt`, and `.gitignore` files are staged. The `.gitignore` is a guardrail, not a substitute for reviewing the staged file list.
