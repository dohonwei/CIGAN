import os
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_ROOT = "data/hci/raw"
OUTPUT_PATH = "data/hci/hci_30s_preprocessed.npz"

# \u91c7\u6837\u7387\u4e0e\u76ee\u6807\u65f6\u957f
FS = 128
WIN_SEC = 30
T_TARGET = FS * WIN_SEC

print("\u76ee\u6807\u957f\u5ea6:", T_TARGET)

all_eeg = []
all_arousal = []
all_valence = []
all_subjects = []


def extract_middle_segment(data, target_len):
    """
    \u4ece EEG \u7684\u4e2d\u95f4\u88c1\u526a target_len \u70b9\uff0c\u907f\u514d baseline \u533a\u57df\u3002
    """
    C, T = data.shape
    if T <= target_len:
        pad_len = target_len - T
        return np.pad(data, ((0,0),(0,pad_len)), 'constant')

    # \u4e2d\u95f4\u622a\u53d6
    start = (T - target_len) // 2
    end = start + target_len
    return data[:, start:end]


subject_folders = sorted(os.listdir(DATA_ROOT))

for subject in tqdm(subject_folders):
    subj_path = os.path.join(DATA_ROOT, subject)
    if not os.path.isdir(subj_path):
        continue

    arousal_path = os.path.join(subj_path, "labels_feltArsl.csv")
    valence_path = os.path.join(subj_path, "labels_feltVlnc.csv")

    arousal_labels = pd.read_csv(arousal_path, header=None).iloc[:,0].values
    valence_labels = pd.read_csv(valence_path, header=None).iloc[:,0].values

    trial_files = sorted([f for f in os.listdir(subj_path) if f.endswith(".csv") and "label" not in f])

    for i, fname in enumerate(trial_files):

        df = pd.read_csv(os.path.join(subj_path, fname), header=None)
        data = df.values.T               # (C, T)
        data = data * 1e6                # <-- \u8f6c \u03bcV \u5173\u952e\u4fee\u590d\uff01

        # \u622a\u53d6\u4e2d\u95f4 30 \u79d2\uff08\u6bd4\u524d 30 \u79d2\u597d\u5f88\u591a\uff09
        data = extract_middle_segment(data, T_TARGET)

        all_eeg.append(data.astype(np.float32))
        all_arousal.append(arousal_labels[i])
        all_valence.append(valence_labels[i])
        all_subjects.append(subject)


all_eeg = np.array(all_eeg)
all_arousal = np.array(all_arousal)
all_valence = np.array(all_valence)
all_subjects = np.array(all_subjects)

print("\u6700\u7ec8 EEG \u5f62\u72b6:", all_eeg.shape)
np.savez(
    OUTPUT_PATH,
    eeg=all_eeg,
    arousal=all_arousal,
    valence=all_valence,
    subjects=all_subjects
)

print("\U0001f389 \u5df2\u4fdd\u5b58\u4fee\u590d\u540e\u7684 HCI \u6570\u636e\uff1a", OUTPUT_PATH)
