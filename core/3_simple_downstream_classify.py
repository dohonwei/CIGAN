"""
3_simple_downstream_classify_loso.py (TIM Revision - LOSO Subject-Independent)
\u5168\u81ea\u52a8\u63d0\u53d6\u9891\u57df\u7279\u5f81\uff0c\u5e76\u6267\u884c\u4e25\u683c\u7684\u8de8\u88ab\u8bd5 (Subject-Independent) TRTR & TSTR \u4e0b\u6e38\u60c5\u7eea\u5206\u7c7b\u8bc4\u4f30
"""

import os
import numpy as np
from scipy.signal import welch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# ==============================================================
#                    GLOBAL CONFIG & PATHS
# ==============================================================
DATASET = "hci"  # \u2605 \u5728\u8fd9\u91cc\u4e00\u952e\u5207\u6362 "deap" \u6216 "hci"
FS = 128 if DATASET == "deap" else 256  # \u2605 \u81ea\u52a8\u9002\u914d\u91c7\u6837\u7387

BASE_DIR = r"."
NPZ_REAL = os.path.join(BASE_DIR, "data", DATASET, f"{DATASET}_30s_preprocessed_checked_labels.npz")

# \u2605 \u66f4\u65b0\u5bf9\u9f50\uff1a\u5bf9\u5e94\u6700\u65b0\u7684\u6a21\u578b\u547d\u540d\u548c\u5168\u5c0f\u5199\u7684\u6587\u4ef6\u540d
MODEL_PATHS = {
    "CIGAN_Main": os.path.join(BASE_DIR, "comparison", DATASET, "cigan_main_generated_loso.npz"),
    # "CIGAN_with_PSD": os.path.join(BASE_DIR, "comparison", DATASET, "cigan_with_psd_generated_loso.npz"),
    "Ablation_NoFEN": os.path.join(BASE_DIR, "comparison", DATASET, "ablation_nofen_generated_loso.npz"),
    "Ablation_NoCausal": os.path.join(BASE_DIR, "comparison", DATASET, "ablation_nocausal_generated_loso.npz"),
    "Ablation_Vanilla": os.path.join(BASE_DIR, "comparison", DATASET, "ablation_vanilla_generated_loso.npz"),
    "hvEEGNet": os.path.join(BASE_DIR, "comparison", DATASET, "hveegnet_generated_loso.npz"),
    "TIE_EEGNet": os.path.join(BASE_DIR, "comparison", DATASET, "tie_eegnet_generated_loso.npz"),
    "Linear_Interp": os.path.join(BASE_DIR, "comparison", DATASET, "linear_generated_loso.npz"),
    "Spline_Interp": os.path.join(BASE_DIR, "comparison", DATASET, "spline_generated_loso.npz")
}

# ==============================================================
#                    FEATURE EXTRACTION
# ==============================================================
def extract_features(eeg_data, fs):
    """\u63d0\u53d6 4 \u4e2a\u9891\u6bb5\u7684 PSD \u7279\u5f81 (Delta, Theta, Alpha, Beta)"""
    N, C, T = eeg_data.shape
    f, P = welch(eeg_data, fs=fs, nperseg=256, axis=-1)
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    features = []

    for k, (f1, f2) in bands.items():
        idx = (f >= f1) & (f <= f2)
        power = np.trapezoid(P[:, :, idx], f[idx], axis=-1) if np.any(idx) else np.zeros((N, C))
        features.append(power)

    return np.concatenate(features, axis=1) # Shape: (N, C * 4)

# ==============================================================
#                    EVALUATION (LOSO)
# ==============================================================
def run_tstr_loso(feats_real, feats_fake, labels, groups):
    """
    LOSO \u7b56\u7565\u4e0b\u7684 TRTR / TSTR \u8bc4\u4f30
    """
    logo = LeaveOneGroupOut()
    acc_list, f1_list = [], []

    for train_idx, test_idx in logo.split(feats_real, labels, groups):
        # TSTR: \u8bad\u7ec3\u96c6\u4f7f\u7528 Fake (\u751f\u6210\u6570\u636e)\uff0c\u6d4b\u8bd5\u96c6\u4f7f\u7528 Real (\u771f\u5b9e\u6570\u636e)
        X_train, y_train = feats_fake[train_idx], labels[train_idx]
        X_test, y_test   = feats_real[test_idx], labels[test_idx]

        clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc_list.append(accuracy_score(y_test, preds))
        f1_list.append(f1_score(y_test, preds, average='macro'))

    return np.mean(acc_list), np.std(acc_list), np.mean(f1_list), np.std(f1_list)

# ==============================================================
#                    MAIN LOOP
# ==============================================================
def main():
    print(f"==================================================")
    print(f"\U0001f680 Simple Downstream Classify (LOSO) - {DATASET.upper()}")
    print(f"==================================================")

    # 1. Load Real Data
    npz_real = np.load(NPZ_REAL)
    real_fp = npz_real["eeg"][:, [0, 16], :]
    subjects = npz_real["subjects"]
    labels_arousal = npz_real["arousal"]
    labels_valence = npz_real["valence"]

    N_all = len(real_fp)

    print("\U0001f9e0 Extracting Features for Real Data (TRTR)...")
    feats_real = extract_features(real_fp, FS)

    report = []

    # 2. Run TRTR (Upper Bound Baseline)
    trtr_a_m, trtr_a_s, trtr_f1_a_m, trtr_f1_a_s = run_tstr_loso(feats_real, feats_real, labels_arousal, subjects)
    trtr_v_m, trtr_v_s, trtr_f1_v_m, trtr_f1_v_s = run_tstr_loso(feats_real, feats_real, labels_valence, subjects)

    report.append({
        "Model": "TRTR (Real Target)",
        "A_ACC": f"{trtr_a_m:.4f}\xb1{trtr_a_s:.4f}", "A_F1": f"{trtr_f1_a_m:.4f}\xb1{trtr_f1_a_s:.4f}",
        "V_ACC": f"{trtr_v_m:.4f}\xb1{trtr_v_s:.4f}", "V_F1": f"{trtr_f1_v_m:.4f}\xb1{trtr_f1_v_s:.4f}"
    })

    # 3. Loop over all generated fake data
    for m_name, m_path in MODEL_PATHS.items():
        if not os.path.exists(m_path):
            print(f"\u26a0\ufe0f Warning: File not found for {m_name}. Skipping...")
            continue

        print(f"\U0001f50d Evaluating {m_name} (TSTR)...")
        gen_data = np.load(m_path)
        fake_fp = gen_data["fake_fp"][:N_all, :2, :] if "fake_fp" in gen_data else gen_data["fake"][:N_all, :2, :]

        feats_fake = extract_features(fake_fp, FS)

        ts_a_m, ts_a_s, f1_a_m, f1_a_s = run_tstr_loso(feats_real, feats_fake, labels_arousal, subjects)
        ts_v_m, ts_v_s, f1_v_m, f1_v_s = run_tstr_loso(feats_real, feats_fake, labels_valence, subjects)

        report.append({
            "Model": m_name,
            "A_ACC": f"{ts_a_m:.4f}\xb1{ts_a_s:.4f}", "A_F1": f"{f1_a_m:.4f}\xb1{f1_a_s:.4f}",
            "V_ACC": f"{ts_v_m:.4f}\xb1{ts_v_s:.4f}", "V_F1": f"{f1_v_m:.4f}\xb1{f1_v_s:.4f}"
        })

    # ==============================================================
    #                    PRINT FINAL REPORT
    # ==============================================================
    df_report = pd.DataFrame(report)
    print("\n" + "="*80)
    print("\U0001f3c6 FINAL DOWNSTREAM CLASSIFICATION REPORT (TSTR)")
    print("="*80)
    print(df_report.to_string(index=False))

    # Save to CSV
    save_csv = os.path.join(BASE_DIR, "comparison", DATASET, f"Master_Classify_Summary_{DATASET.upper()}.csv")
    df_report.to_csv(save_csv, index=False)
    print(f"\n\U0001f4c1 Classification Summary saved to {save_csv}")

if __name__ == "__main__":
    main()