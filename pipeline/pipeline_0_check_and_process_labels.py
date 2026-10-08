# 0_check_and_process_labels.py
import os
import numpy as np

# ======================================================
# \u9700\u8981\u68c0\u67e5\u548c\u5904\u7406\u7684 npz \u6587\u4ef6\u8def\u5f84\uff08\u6309\u9700\u4fee\u6539\uff09
# ======================================================
# HCI
# NPZ_PATH = r"./data/hci/hci_30s_preprocessed.npz"
# DEAP\uff08\u9700\u8981\u65f6\u89e3\u9664\u6ce8\u91ca\uff09
NPZ_PATH = r"./data/deap/deap_30s_preprocessed.npz"

OUTPUT_NPZ = NPZ_PATH.replace(".npz", "_checked_labels.npz")

# ======================================================
# \u6807\u7b7e\u9608\u503c\uff08HCI / DEAP \u9ed8\u8ba4 1\u20139\uff09
# ======================================================
AROUSAL_THRESHOLD = 5
VALENCE_THRESHOLD = 5

if __name__ == "__main__":
    print(f"\u52a0\u8f7d\u6570\u636e: {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)

    eeg = data["eeg"]              # (N, C, T)
    arousal_raw = data["arousal"]  # (N,)
    valence_raw = data["valence"]  # (N,)
    subjects = data["subjects"]    # (N,)

    print("====================================")
    print("\u6570\u636e\u7ed3\u6784\u68c0\u67e5 (D3)")
    print("====================================")
    print("EEG shape      :", eeg.shape)
    print("arousal shape  :", arousal_raw.shape)
    print("valence shape  :", valence_raw.shape)
    print("subjects shape :", subjects.shape)

    N = eeg.shape[0]
    if not (len(arousal_raw) == len(valence_raw) == len(subjects) == N):
        raise ValueError("\u274c \u6807\u7b7e\u957f\u5ea6\u4e0e EEG trial \u6570\u4e0d\u4e00\u81f4\uff0c\u8bf7\u68c0\u67e5\u6570\u636e\uff01")

    print("\u2714 \u6570\u636e\u7ed3\u6784 OK\uff1aEEG / arousal / valence / subjects \u6570\u91cf\u4e00\u81f4\n")

    print("====================================")
    print("\u6807\u7b7e\u4e8c\u5206\u7c7b\u5904\u7406 (D4)")
    print("====================================")

    arousal_binary = (arousal_raw >= AROUSAL_THRESHOLD).astype(int)
    valence_binary = (valence_raw >= VALENCE_THRESHOLD).astype(int)

    print("\u793a\u4f8b\uff1a(\u539f\u59cb arousal \u2192 \u4e8c\u5206\u7c7b\u524d10\u4e2a)")
    print(list(zip(arousal_raw[:10], arousal_binary[:10])))

    print("\n\u793a\u4f8b\uff1a(\u539f\u59cb valence \u2192 \u4e8c\u5206\u7c7b\u524d10\u4e2a)")
    print(list(zip(valence_raw[:10], valence_binary[:10])))

    np.savez(
        OUTPUT_NPZ,
        eeg=eeg,
        arousal=arousal_binary,
        valence=valence_binary,
        subjects=subjects
    )

    print("\n====================================")
    print("\U0001f389 \u5b8c\u6210\uff1aD3+D4 \u5904\u7406\u6210\u529f\uff01")
    print("\u5df2\u4fdd\u5b58\u65b0\u6587\u4ef6\uff1a", OUTPUT_NPZ)
    print("====================================")
