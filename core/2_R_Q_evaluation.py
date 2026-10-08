import os
import numpy as np
import pandas as pd
import torch  # \u5f15\u5165 PyTorch \u4f7f\u7528 GPU
from scipy.signal import welch
from fastdtw import fastdtw
from joblib import Parallel, delayed  # \u7528\u4e8e CPU \u591a\u8fdb\u7a0b\u52a0\u901f

# ==========================================
# 1. \u53c2\u6570\u914d\u7f6e
# ==========================================
DATASET = "hci"
SAMPLING_RATE = 128 if DATASET == "deap" else 256

EEG_BANDS = {
    'delta_diff': (1, 4),
    'theta_diff': (4, 8),
    'alpha_diff': (8, 13),
    'beta_diff': (13, 30)
}

BASE_DIR = r"."
FAKE_DATA_DIR = os.path.join(BASE_DIR, "ablation",DATASET)
npz_path = os.path.join(BASE_DIR, "data", DATASET, f"{DATASET}_30s_preprocessed_checked_labels.npz")

original_data = np.load(npz_path)
real_data = original_data["eeg"][:, [0, 16], :]
print("\u771f\u5b9e\u6570\u636e\u52a0\u8f7d\u6210\u529f\uff0c\u5f62\u72b6\u4e3a:", real_data.shape)

FAKE_DATA_FILES = {
    "Full Model": "ablation_full_fake_fp.npy",
    "w/o Causal": "ablation_no_causal_fake_fp.npy",
    "w/o FEN": "ablation_no_fen_fake_fp.npy",
    "w/o PSD": "ablation_no_psd_fake_fp.npy",
    "hveegnet": "hveegnet_fake_fp.npy",
    "tie-eegnet": "tie_eegnet_fake_fp.npy",
    "wavenet": "wavenet_fake_fp.npy",
    "spline": "spline_fake_fp.npy",
    "encdec": "encdec_fake_fp.npy"
}

# \U0001f31f \u81ea\u52a8\u68c0\u6d4b\u5e76\u6fc0\u6d3b GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\U0001f525 \u5f53\u524d\u8ba1\u7b97\u8bbe\u5907: {device}")


# ==========================================
# 2. GPU \u52a0\u901f\u6307\u6807\u51fd\u6570 (Cosine & Pearson)
# ==========================================
def calculate_basic_metrics_gpu(real_flat, fake_flat):
    """\u4f7f\u7528 PyTorch \u5728 GPU \u4e0a\u79d2\u6740\u5927\u89c4\u6a21\u77e9\u9635\u7684\u65f6\u57df\u6307\u6807"""
    # \u5c06\u6570\u636e\u8f6c\u6362\u4e3a Tensor \u5e76\u63a8\u9001\u5230 GPU
    r = torch.from_numpy(real_flat).float().to(device)
    f = torch.from_numpy(fake_flat).float().to(device)

    # 1. GPU \u8ba1\u7b97 Cosine \u76f8\u4f3c\u5ea6
    cos_sim = torch.nn.functional.cosine_similarity(r, f, dim=0).item()

    # 2. GPU \u8ba1\u7b97 Pearson \u76f8\u5173\u7cfb\u6570
    r_mean = torch.mean(r)
    f_mean = torch.mean(f)
    r_diff = r - r_mean
    f_diff = f - f_mean
    pearson_corr = (torch.sum(r_diff * f_diff) / (
                torch.sqrt(torch.sum(r_diff ** 2)) * torch.sqrt(torch.sum(f_diff ** 2)))).item()

    return cos_sim, pearson_corr


# ==========================================
# 3. \u9010\u4e2a\u6837\u672c\u4f18\u5316\u51fd\u6570 (\u7528\u4e8e\u591a\u8fdb\u7a0b)
# ==========================================
def _worker_trial_metrics(real_trial, fake_trial, fs):
    """\u5355\u4e2a\u6837\u672c\u7684\u590d\u6742\u6307\u6807\u8ba1\u7b97 (DTW + PSD)"""

    # 1. DTW \u6027\u80fd\u4f18\u5316\uff1a
    # 3840\u4e2a\u70b9\u5bf9DTW\u6765\u8bf4\u4f9d\u7136\u504f\u957f\uff0c\u5efa\u8bae\u6bcf\u9694 4 \u4e2a\u70b9\u4e0b\u91c7\u6837\u4e00\u6b21\uff08\u4e0d\u5f71\u54cd\u5b8f\u89c2\u65f6\u95f4\u5bf9\u9f50\u8d8b\u52bf\u7279\u5f81\uff09
    # \u8fd9\u6837\u70b9\u6570\u4ece 3840 \u964d\u5230 960\uff0c\u8ba1\u7b97\u901f\u5ea6\u4f1a\u63d0\u5347\u5341\u51e0\u500d\uff01
    real_sub = real_trial[::4]
    fake_sub = fake_trial[::4]

    try:
        dtw_dist, _ = fastdtw(real_sub, fake_sub, dist=2)
    except:
        dtw_dist = np.nan

    # 2. \u8ba1\u7b97\u8111\u7535\u5404\u9891\u5e26\u529f\u7387
    nperseg = min(len(real_trial), fs * 2)

    # \u771f\u5b9e\u6570\u636e\u4e0e\u751f\u6210\u6570\u636e\u7684 PSD
    freqs_r, psd_r = welch(real_trial, fs=fs, nperseg=nperseg)
    _, psd_f = welch(fake_trial, fs=fs, nperseg=nperseg)

    band_diffs = {}
    for band_name, (low, high) in EEG_BANDS.items():
        idx_band = np.logical_and(freqs_r >= low, freqs_r <= high)

        # \U0001f4a1 \u3010\u6838\u5fc3\u4fee\u590d\u3011\u5c06 np.trapz \u66ff\u6362\u4e3a\u65b0\u7248 NumPy \u652f\u6301\u7684 np.trapezoid
        if hasattr(np, 'trapezoid'):
            power_r = np.trapezoid(psd_r[idx_band], freqs_r[idx_band])
            power_f = np.trapezoid(psd_f[idx_band], freqs_r[idx_band])
        else:
            # \u517c\u5bb9\u8001\u7248\u672c NumPy
            power_r = np.trapz(psd_r[idx_band], freqs_r[idx_band])
            power_f = np.trapz(psd_f[idx_band], freqs_r[idx_band])

        band_diffs[band_name] = abs(power_r - power_f)

    return dtw_dist, band_diffs


def calculate_complex_metrics_parallel(real_data, fake_data, fs):
    """\u5229\u7528 CPU \u591a\u6838\u5e76\u884c\u8ba1\u7b97\u6240\u6709\u6837\u672c\u7684 DTW \u548c PSD \u5dee\u5f02"""
    # \u5c06\u591a\u7ef4\u8111\u7535\u6570\u636e\u5c55\u5e73\u6210 (\u6837\u672c\u6570, \u4fe1\u53f7\u70b9\u6570) \u65b9\u4fbf\u9010\u4e2a Trial \u5904\u7406
    r_samples = real_data.reshape(-1, real_data.shape[-1])
    f_samples = fake_data.reshape(-1, fake_data.shape[-1])
    num_samples = r_samples.shape[0]

    # \u5f00\u542f\u591a\u8fdb\u7a0b\u5e76\u884c\uff08n_jobs=-1 \u4ee3\u8868\u7528\u5c3d\u4f60\u7535\u8111\u7684\u6240\u6709 CPU \u6838\u5fc3\uff09
    results = Parallel(n_jobs=-1)(
        delayed(_worker_trial_metrics)(r_samples[i], f_samples[i], fs) for i in range(num_samples)
    )

    # \u805a\u5408\u591a\u8fdb\u7a0b\u7ed3\u679c
    dtw_list = [r[0] for r in results]
    mean_dtw = np.nanmean(dtw_list)

    mean_psd_diffs = {}
    for band in EEG_BANDS.keys():
        mean_psd_diffs[band] = np.mean([r[1][band] for r in results])

    return mean_dtw, mean_psd_diffs


# ==========================================
# 4. \u4e3b\u7a0b\u5e8f\u5faa\u73af
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print(" \U0001f680 GPU\u52a0\u901f + CPU\u591a\u8fdb\u7a0b\u5206\u5e03\u5f0f\u8bc4\u4f30\u7cfb\u7edf\u542f\u52a8 \U0001f680 ")
    print("=" * 60)

    results_list = []

    for model_name, file_path in FAKE_DATA_FILES.items():
        full_file_path = os.path.join(FAKE_DATA_DIR, file_path)
        if not os.path.exists(full_file_path):
            print(f"\u274c \u627e\u4e0d\u5230\u6587\u4ef6: {full_file_path}\uff0c\u8df3\u8fc7\u8be5\u7ec4\u3002")
            continue
        print(f"\n[\u8fd0\u884c\u4e2d] \u6b63\u5728\u5e76\u884c\u8bc4\u4f30: {model_name}...")
        fake_data = np.load(full_file_path)
        print(real_data.shape, fake_data.shape)
        assert real_data.shape == fake_data.shape, f"\u9519\u8bef\uff1a{model_name} \u5f62\u72b6\u4e0d\u5339\u914d\uff01"

        # 1. \u5c55\u5e73\u6570\u636e\uff0c\u4f7f\u7528 GPU \u79d2\u6740 Cosine \u548c Pearson
        real_flat = real_data.flatten()
        fake_flat = fake_data.flatten()
        cos_sim, pearson_corr = calculate_basic_metrics_gpu(real_flat, fake_flat)

        # 2. \u8fd0\u7528\u591a\u8fdb\u7a0b\u5e76\u884c\u673a\u5236\uff0c\u5206\u644a\u8ba1\u7b97 DTW \u548c PSD \u5dee\u5f02
        dtw_dist, psd_diffs = calculate_complex_metrics_parallel(real_data, fake_data, fs=SAMPLING_RATE)

        # 3. \u6574\u5408
        row = {
            "Model Group": model_name,
            "Cosine \u2191": cos_sim,
            "Pearson \u2191": pearson_corr,
            "DTW \u2193": dtw_dist,
            "\u03b4_diff \u2193": psd_diffs["delta_diff"],
            "\u03b8_diff \u2193": psd_diffs["theta_diff"],
            "\u03b1_diff \u2193": psd_diffs["alpha_diff"],
            "\u03b2_diff \u2193": psd_diffs["beta_diff"]
        }
        results_list.append(row)
        print(f"\u2705 {model_name} \u8bc4\u4f30\u5b8c\u6210\uff01")

    # ==========================================
    # 5. \u7ed3\u679c\u5c55\u793a\u4e0e\u4fdd\u5b58
    # ==========================================
    if results_list:
        df_results = pd.DataFrame(results_list)
        print("\n" + "=" * 40 + " \u6d88\u878d\u5b9e\u9a8c\u5bf9\u6bd4\u8868\u683c " + "=" * 40)
        print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=" * 98)

        output_csv = os.path.join(BASE_DIR, "results-R1", f"ablation_study_{DATASET}.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        print(f"\n\U0001f389 \u5b8c\u7f8e\u8fd0\u884c\uff01\u6700\u7ec8\u6c47\u603b\u8868\u5df2\u4fdd\u5b58\u81f3: {output_csv}")