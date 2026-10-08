"""
pipeline_2_evaluate_quality_and_classification.py
(100% \u5b8c\u6574\u878d\u5408\u7248)
\u81ea\u52a8\u9002\u914d 128Hz/256Hz\uff0c\u5e76\u8bc4\u4f30\u6240\u6709 CIGAN\u53d8\u4f53\u3001\u4f20\u7edf\u63d2\u503c \u4ee5\u53ca \u6df1\u5ea6\u57fa\u7ebf(hvEEGNet, TIE-EEGNet)\u3002
"""

import os
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#                    GLOBAL CONFIG & PATHS
# ============================================================
DATASET = "deap"  # \u2605 \u5728\u8fd9\u91cc\u5207\u6362 "deap" \u6216 "hci"\uff0c\u52a1\u5fc5\u4e0e pipeline_1 \u4fdd\u6301\u4e00\u81f4\uff01
FS = 128 if DATASET == "deap" else 256  # \u2605 \u81ea\u52a8\u9002\u914d\u91c7\u6837\u7387

BASE_DIR = r"."
DATA_NPZ = os.path.join(BASE_DIR, "data", DATASET, f"{DATASET}_30s_preprocessed_checked_labels.npz")
GEN_DIR  = os.path.join(BASE_DIR, "pipeline_out", DATASET, "data")
OUT_DIR  = os.path.join(BASE_DIR, "pipeline_out", DATASET, "reports")
os.makedirs(OUT_DIR, exist_ok=True)

# \u2605 \u6838\u5fc3\u4fee\u6b63\uff1a\u8865\u5168\u6240\u6709\u6df1\u5ea6\u57fa\u7ebf\u6a21\u578b
MODEL_NAMES = [
    "CIGAN_Full",
    "Ablation_NoFEN",
    "Ablation_NoCausal",
    "Ablation_NoPSD",
    "Vanilla_WGAN_GP",
    "hvEEGNet",        # \u2605 \u6062\u590d hvEEGNet
    "TIE_EEGNet",      # \u2605 \u6062\u590d TIE-EEGNet
    "Linear_Interp",
    "Spline_Interp"
]

# ============================================================
#                    METRICS UTILITIES
# ============================================================
def compute_mmd(X, Y, sigma=50.0):
    XX = np.exp(-np.square(np.subtract.outer(X, X)) / (2 * sigma ** 2)).mean()
    YY = np.exp(-np.square(np.subtract.outer(Y, Y)) / (2 * sigma ** 2)).mean()
    XY = np.exp(-np.square(np.subtract.outer(X, Y)) / (2 * sigma ** 2)).mean()
    return float(XX + YY - 2 * XY)

def extract_psd_features(eeg_fp):
    feats = []
    for ch in range(eeg_fp.shape[0]):
        # \u2605 \u52a8\u6001\u4f7f\u7528 FS \u63d0\u53d6 PSD
        f, Pxx = welch(eeg_fp[ch], fs=FS, nperseg=256)
        valid_idx = (f >= 1) & (f <= 45)
        feats.extend(Pxx[valid_idx])
    return np.array(feats)

# ============================================================
#                    CLASSIFICATION (TRTR & TSTR)
# ============================================================
def run_trtr(X_real, labels):
    """ TRTR: Real Data Train, Real Data Test """
    X_tr, X_te, y_tr, y_te = train_test_split(X_real, labels, test_size=0.2, random_state=42, stratify=labels)
    clf = LGBMClassifier(random_state=42, verbose=-1)
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    return accuracy_score(y_te, preds), f1_score(y_te, preds, average='macro')

def run_tstr(X_real, X_fake, labels):
    """ TSTR: Fake Data Train, Real Data Test """
    idx = np.arange(len(labels))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)
    clf = LGBMClassifier(random_state=42, verbose=-1)
    clf.fit(X_fake[idx_tr], labels[idx_tr])  # Train on Synthetic
    preds = clf.predict(X_real[idx_te])      # Test on Real
    return accuracy_score(labels[idx_te], preds), f1_score(labels[idx_te], preds, average='macro')

# ============================================================
#                      MAIN EXECUTION
# ============================================================
def main():
    print("="*60)
    print(f"PIPELINE 2: UNIFIED MMD & TSTR EVALUATION ({DATASET.upper()})")
    print("="*60)

    data_real = np.load(DATA_NPZ)
    labels_arousal = data_real["arousal"]
    labels_valence = data_real["valence"]

    # \u83b7\u53d6\u771f\u5b9e\u6570\u636e\u4f5c\u4e3a\u53c2\u8003\u57fa\u51c6
    ref_npz = np.load(os.path.join(GEN_DIR, "CIGAN_Full_generated.npz"))
    real_fp = ref_npz["real_fp"]
    N_all = min(len(labels_arousal), len(real_fp))
    real_fp = real_fp[:N_all]

    print(f"\n[1/3] Extracting Features for Ground Truth (TRTR) using FS={FS}Hz...")
    X_real = np.array([extract_psd_features(real_fp[i]) for i in tqdm(range(N_all))])

    # \u8bc4\u6d4b Ground Truth
    acc_r_a, f1_r_a = run_trtr(X_real, labels_arousal[:N_all])
    acc_r_v, f1_r_v = run_trtr(X_real, labels_valence[:N_all])

    report = {
        "Ground_Truth (TRTR)": {"MMD": 0.0000, "Arousal_ACC": acc_r_a, "Arousal_F1": f1_r_a, "Valence_ACC": acc_r_v, "Valence_F1": f1_r_v}
    }

    print("\n[2/3] Evaluating Generative Models (MMD + TSTR)...")
    for name in MODEL_NAMES:
        path = os.path.join(GEN_DIR, f"{name}_generated.npz")
        if not os.path.exists(path):
            print(f"\u26a0\ufe0f Skipped {name}, file not found.")
            continue

        gen_data = np.load(path)
        fake_fp = gen_data["fake_fp"][:N_all]

        # 1. \u8ba1\u7b97 MMD (\u53d6\u53cc\u901a\u9053\u5e73\u5747)
        mmd_ch0 = np.mean([compute_mmd(real_fp[i, 0], fake_fp[i, 0]) for i in range(min(N_all, 150))])
        mmd_ch1 = np.mean([compute_mmd(real_fp[i, 1], fake_fp[i, 1]) for i in range(min(N_all, 150))])
        mmd_avg = (mmd_ch0 + mmd_ch1) / 2

        # 2. \u63d0\u53d6\u5047\u7279\u5f81\u5e76\u8dd1 TSTR
        X_fake = np.array([extract_psd_features(fake_fp[i]) for i in range(N_all)])
        acc_a, f1_a = run_tstr(X_real, X_fake, labels_arousal[:N_all])
        acc_v, f1_v = run_tstr(X_real, X_fake, labels_valence[:N_all])

        report[name] = {"MMD": mmd_avg, "Arousal_ACC": acc_a, "Arousal_F1": f1_a, "Valence_ACC": acc_v, "Valence_F1": f1_v}
        print(f"  -> {name} evaluated. MMD: {mmd_avg:.5f}")

    # ============================================================
    # \u8f93\u51fa\u5bf9\u9f50\u7684\u6700\u7ec8\u5927\u8868
    # ============================================================
    print("\n[3/3] Generating Unified Report...")
    out_file = os.path.join(OUT_DIR, "Unified_Evaluation_Report.txt")
    with open(out_file, "w") as f:
        header = f"{'Model Name':<22} | {'MMD':<8} | {'Arou_ACC':<8} | {'Arou_F1':<8} | {'Val_ACC':<8} | {'Val_F1':<8}"
        print(f"\n{header}\n" + "-" * 75)
        f.write(f"{header}\n" + "-" * 75 + "\n")

        for m_name, res in report.items():
            line = f"{m_name:<22} | {res['MMD']:.5f}  | {res['Arousal_ACC']:.4f}   | {res['Arousal_F1']:.4f}  | {res['Valence_ACC']:.4f}  | {res['Valence_F1']:.4f}"
            print(line)
            f.write(line + "\n")

    print(f"\n\U0001f389 PIPELINE 2 COMPLETED for {DATASET.upper()}! Report saved to {out_file}")

if __name__ == "__main__":
    main()