import os
import numpy as np
from tqdm import tqdm

# ======================================================================
# \u8def\u5f84\u8bbe\u7f6e
# ======================================================================
data_dir = "data/deap/raw"
label_dir = "data/deap/labels"

OUT_DIR = "data/deap"
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================================
# Step 1\uff1a\u5408\u5e76\u6240\u6709 s01 ~ s32 \u7684\u539f\u59cb\u6570\u636e
# ======================================================================

subjects = [f"s{idx:02d}" for idx in range(1, 33)]

all_data = []      # (40, 40, 8064)
all_labels = []    # (40, 4)
all_subject_ids = []  # \u6bcf\u4e2a trial \u5bf9\u5e94\u7684\u88ab\u8bd5\u7f16\u53f7

print("\u6b63\u5728\u5408\u5e76\u6240\u6709\u88ab\u8bd5\u7684\u539f\u59cb\u6570\u636e...")

for subj in subjects:
    data_path = os.path.join(data_dir, f"{subj}.npy")
    label_path = os.path.join(label_dir, f"{subj}.npy")

    if not os.path.exists(data_path):
        print(f"[\u8b66\u544a] \u672a\u627e\u5230\u6570\u636e\u6587\u4ef6: {data_path}\uff0c\u8df3\u8fc7")
        continue
    if not os.path.exists(label_path):
        print(f"[\u8b66\u544a] \u672a\u627e\u5230\u6807\u7b7e\u6587\u4ef6: {label_path}\uff0c\u8df3\u8fc7")
        continue

    data = np.load(data_path)       # shape (40, 40, 8064)
    labels = np.load(label_path)    # shape (40, 4)

    if data.shape[0] != labels.shape[0]:
        print(f"[\u8b66\u544a] trial \u6570\u4e0d\u5339\u914d: {subj} \u2014 \u8df3\u8fc7")
        continue

    all_data.append(data)
    all_labels.append(labels)

    # \u8bb0\u5f55\u6bcf\u4e2a trial \u5bf9\u5e94\u7684\u88ab\u8bd5\u7f16\u53f7
    all_subject_ids.extend([subj] * data.shape[0])

    print(f"{subj} \u5bfc\u5165\u6210\u529f: data{data.shape}, labels{labels.shape}")

if len(all_data) == 0:
    raise RuntimeError("\u274c \u672a\u80fd\u52a0\u8f7d\u4efb\u4f55\u88ab\u8bd5\u6570\u636e\uff0c\u8bf7\u68c0\u67e5\u8def\u5f84\u3002")

merged_data = np.concatenate(all_data, axis=0)      # (1280,40,8064)
merged_labels = np.concatenate(all_labels, axis=0)  # (1280,4)
merged_subjects = np.array(all_subject_ids)         # (1280,)

print("\u5408\u5e76\u540e\u6570\u636e\u5f62\u72b6:", merged_data.shape)
print("\u5408\u5e76\u540e\u6807\u7b7e\u5f62\u72b6:", merged_labels.shape)

# ======================================================================
# Step 2\uff1a\u7edf\u4e00\u4fdd\u7559 EEG \u524d 32 \u901a\u9053 + 30 \u79d2\u7a97\u53e3\uff083840 \u70b9\uff09
# ======================================================================

SAMPLING_RATE = 128
WINDOW_SECONDS = 30
T_TARGET = SAMPLING_RATE * WINDOW_SECONDS  # 30s \u2192 3840 \u70b9

print(f"\n\u5f00\u59cb 30 \u79d2\u7a97\u53e3\u9884\u5904\u7406\u2026 T_target = {T_TARGET}")

# \u53ea\u4fdd\u7559\u524d 32 \u901a\u9053 EEG
EEG_CHANNELS = list(range(32))
data_eeg = merged_data[:, EEG_CHANNELS, :]  # (1280, 32, 8064)

print("\u4ec5 EEG \u901a\u9053 shape:", data_eeg.shape)

# \u88c1\u526a\u524d 3840 \u70b9
if data_eeg.shape[2] >= T_TARGET:
    data_eeg_30s = data_eeg[:, :, :T_TARGET].astype(np.float32)
else:
    pad_len = T_TARGET - data_eeg.shape[2]
    data_eeg_30s = np.pad(
        data_eeg,
        ((0, 0), (0, 0), (0, pad_len)),
        mode='constant'
    ).astype(np.float32)

print("\u88c1\u526a\u540e\u7684 shape:", data_eeg_30s.shape)

# ======================================================================
# Step 3\uff1a\u4fdd\u5b58\u4e3a\u548c HCI \u4e00\u81f4\u7684 npz \u6587\u4ef6
# ======================================================================

FINAL_OUTPUT = os.path.join(OUT_DIR, "deap_30s_preprocessed.npz")

np.savez(
    FINAL_OUTPUT,
    eeg=data_eeg_30s,        # (1280, 32, 3840)
    arousal=merged_labels[:,1],
    valence=merged_labels[:,0],
    subjects=merged_subjects # (1280,)
)

print("\n===================================")
print("\U0001f389 DEAP 30 \u79d2\u7a97\u53e3 npz \u9884\u5904\u7406\u5b8c\u6210!")
print("\u5df2\u4fdd\u5b58:", FINAL_OUTPUT)
print("===================================\n")
