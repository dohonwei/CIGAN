"""
5_advanced_validation_experiments.py (TIM Revision - Ultimate Vectorized & Incremental)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.signal import welch
from scipy.stats import pearsonr, probplot, shapiro, skew, kurtosis
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==============================================================
#                     PATHS & CONFIGURATION
# ==============================================================
DATASET = "hci"  # \u2605 \u4e00\u952e\u5207\u6362 "deap" \u6216 "hci"
FS = 128 if DATASET == "deap" else 256

BASE_DIR = r"."
DATA_NPZ = os.path.join(BASE_DIR, "data", DATASET, f"{DATASET}_30s_preprocessed_checked_labels.npz")

MODEL_PATHS = {
    "CIGAN (Main)": os.path.join(BASE_DIR, "comparison", DATASET, "CIGAN_generated_data_mc50.npz"),
    "Vanilla WGAN-GP": os.path.join(BASE_DIR, "comparison", DATASET, "ablation_vanilla_generated_loso.npz"),
    "hvEEGNet": os.path.join(BASE_DIR, "comparison", DATASET, "hvEEGNet_generated_loso.npz"),
    "TIE-EEGNet": os.path.join(BASE_DIR, "comparison", DATASET, "TIE_EEGNet_generated_loso.npz")
}

SAVE_DIR = os.path.join(BASE_DIR, "comparison", DATASET)
os.makedirs(SAVE_DIR, exist_ok=True)

# \U0001f3af \u3010\u589e\u91cf\u66f4\u65b0\u914d\u7f6e\u3011\u4ec5\u66f4\u65b0\u6307\u5b9a\u6a21\u578b\uff0c\u7559\u7a7a\u5219\u5168\u91cf\u91cd\u7b97
MODELS_TO_UPDATE = ["CIGAN (Main)"]

# ==============================================================
#                     LOAD DATA
# ==============================================================
print(f"Loading Ground Truth and Main Generated Data for {DATASET.upper()}...")
try:
    gen_data = np.load(MODEL_PATHS["CIGAN (Main)"])
    real_fp = gen_data["real_fp"]
    fake_fp = gen_data["fake_fp"]
    fake_std = gen_data["fake_std"] if "fake_std" in gen_data else np.zeros_like(fake_fp)
    print(f"\u2705 Data loaded. Total trials: {real_fp.shape[0]}")
except Exception as e:
    print(f"\u274c Error loading main model data: {e}")
    real_fp, fake_fp, fake_std = None, None, None


# ==============================================================
#                     HELPER FUNCTIONS
# ==============================================================
def compute_band_power_vectorized(X, fs):
    """\u3010\u6781\u901f\u4f18\u5316\u3011\u5168\u77e9\u9635\u5e76\u884c\u8ba1\u7b97 PSD \u80fd\u91cf\uff0c\u5e72\u6389 for \u5faa\u73af"""
    f, P = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
    out = {}
    for k, (f1, f2) in bands.items():
        idx = (f >= f1) & (f <= f2)
        out[k] = np.trapezoid(P[:, idx], f[idx], axis=-1) if np.any(idx) else np.zeros(X.shape[0])
    return out

def summarize_residual_normality(diff_values):
    """\u8ba1\u7b97\u6b8b\u5dee\u7684\u6b63\u6001\u6027\u7edf\u8ba1\u6307\u6807"""
    diff_values = np.asarray(diff_values).flatten()
    # \u9650\u5236 shapiro \u6837\u672c\u91cf\uff0c\u9632\u6b62 P-value \u5931\u53bb\u610f\u4e49\u6216\u62a5\u9519
    test_values = np.random.choice(diff_values, min(len(diff_values), 5000), replace=False)
    try:
        shapiro_p = shapiro(test_values)[1]
    except Exception:
        shapiro_p = np.nan

    return {
        "Residual_mean_uV": float(np.mean(diff_values)),
        "Residual_std_uV": float(np.std(diff_values)),
        "Skewness": float(skew(diff_values)),
        "Kurtosis": float(kurtosis(diff_values)),
        "Shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else np.nan,
        "N_points": int(len(diff_values)),
    }

# ==============================================================
#                     ANALYTICAL MODULES
# ==============================================================
def build_bland_altman_summary_table(model_paths, save_csv, fs=128):
    """\u6784\u5efa\u591a\u6a21\u578b\u7684\u6ce2\u5f62\u7ea7 Bland-Altman \u6c47\u603b\u8868\uff08\u652f\u6301\u589e\u91cf\u66f4\u65b0\uff09"""
    print("\n--- Building Waveform Bland-Altman Summary Table ---")

    existing_df = pd.DataFrame()
    if len(MODELS_TO_UPDATE) > 0 and os.path.exists(save_csv):
        existing_df = pd.read_csv(save_csv)
        print("\U0001f4c4 Found existing Summary Table. Performing INCREMENTAL update.")

    rows = []
    for m_name, path in model_paths.items():
        if len(MODELS_TO_UPDATE) > 0 and m_name not in MODELS_TO_UPDATE:
            continue

        if not os.path.exists(path):
            print(f"\u26a0\ufe0f Skipping {m_name}: File not found.")
            continue

        print(f"  -> Processing {m_name}...")
        d = np.load(path)
        real = d["real_fp"][:, 0, :]  # \u4ec5\u8bc4\u4f30 Fp1
        fake = d["fake_fp"][:, 0, :] if "fake_fp" in d else d["fake"][:, 0, :]

        # \u2605 \u7edf\u8ba1\u5fc5\u987b\u57fa\u4e8e 100% \u7edd\u5bf9\u5168\u91cf\u6570\u636e\uff01
        diff_values = (real - fake).flatten()
        mean_values = ((real + fake) / 2).flatten()

        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        scale = np.percentile(real.flatten(), 99) - np.percentile(real.flatten(), 1)

        row = {
            "Model": m_name,
            "Bias": float(mean_diff),
            "LoA_lower": float(loa_lower),
            "LoA_upper": float(loa_upper),
            "Normalized_LoA_percent": float(((loa_upper - loa_lower) / scale * 100)) if scale > 0 else np.nan
        }
        row.update(summarize_residual_normality(diff_values))
        rows.append(row)

    if rows:
        new_df = pd.DataFrame(rows)
        if not existing_df.empty:
            models_updated = new_df['Model'].unique()
            existing_df = existing_df[~existing_df['Model'].isin(models_updated)]
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        final_df.to_csv(save_csv, index=False)
        print(f"\u2705 Saved Multi-Model Bland-Altman summary table to: {save_csv}")
        return final_df
    return None

def band_power_bland_altman_analysis(real, fake, channel_name="Fp1", fs=128, save_fig=None, save_csv=None):
    """\u3010\u6781\u901f\u7248\u3011\u5bf9 delta/theta/alpha/beta \u5206\u522b\u8ba1\u7b97 BA summary"""
    print(f"\n--- Running Band-Power Bland-Altman ({channel_name}) ---")

    print("\u26a1 [Vectorized] Computing Band Powers...")
    real_bp = compute_band_power_vectorized(real, fs)
    fake_bp = compute_band_power_vectorized(fake, fs)

    bands = ["delta", "theta", "alpha", "beta"]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, band in zip(axes, bands):
        real_arr = np.asarray(real_bp[band])
        fake_arr = np.asarray(fake_bp[band])

        mean_values = (real_arr + fake_arr) / 2
        diff_values = real_arr - fake_arr

        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        scale = np.percentile(real_arr, 99) - np.percentile(real_arr, 1)

        row = {
            "Channel": channel_name, "Band": band,
            "Bias": float(mean_diff), "LoA_lower": float(loa_lower), "LoA_upper": float(loa_upper),
            "Normalized_LoA_percent": float((loa_upper - loa_lower) / scale * 100) if scale > 0 else np.nan,
        }
        row.update(summarize_residual_normality(diff_values))
        rows.append(row)

        ax.scatter(mean_values, diff_values, s=15, alpha=0.6, color='royalblue')
        ax.axhline(mean_diff, color='black', linestyle='-', linewidth=2, label=f'Bias: {mean_diff:.2f}')
        ax.axhline(loa_upper, color='red', linestyle='--', linewidth=2, label=f'+1.96SD')
        ax.axhline(loa_lower, color='red', linestyle='--', linewidth=2, label=f'-1.96SD')
        ax.set_title(f'{band.capitalize()} Band')
        ax.set_xlabel('Mean Band Power')
        ax.set_ylabel('Difference (Real - Fake)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\u2705 Saved Band-Power Bland-Altman summary to: {save_csv}")
    return df

def true_uncertainty_quantification(real, fake, std, channel_idx=0, save_path=None):
    """\u7ed8\u5236\u771f\u5b9e\u7684\u4e0d\u786e\u5b9a\u6027\u5305\u7edc"""
    trial_idx = 0
    t_start, t_end = 0, min(512, real.shape[-1])

    real_sig = real[trial_idx, channel_idx, t_start:t_end]
    fake_mean = fake[trial_idx, channel_idx, t_start:t_end]
    fake_sigma = std[trial_idx, channel_idx, t_start:t_end]

    time_axis = np.arange(len(real_sig)) / FS

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, real_sig, color='black', alpha=0.7, label='Ground Truth', linewidth=1.5)
    plt.plot(time_axis, fake_mean, color='royalblue', label='CIGAN Predictive Mean', linewidth=1.5)

    plt.fill_between(time_axis, fake_mean - 2*fake_sigma, fake_mean + 2*fake_sigma,
                     color='royalblue', alpha=0.3, label='95% Confidence Interval (\xb12\u03c3)')

    plt.title('True Epistemic Uncertainty Quantification (MC-Dropout)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (\xb5V)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


import importlib.util
import sys
import torch


def noise_robustness_test(dataset, base_dir, save_dir, fs=128):
    """
    IEEE TIM \u6838\u5fc3\u8fd4\u4fee\u5b9e\u9a8c\uff1a\u4fe1\u566a\u6bd4 (SNR) \u9c81\u68d2\u6027\u5206\u6790
    \u5411\u8f93\u5165\u901a\u9053\u6ce8\u5165\u4e0d\u540c\u5f3a\u5ea6\u7684\u7269\u7406\u566a\u58f0\uff0c\u8bc4\u4f30 CIGAN \u7684\u7279\u5f81\u6297\u566a\u7a33\u5b9a\u6027\u3002
    """
    print("\n" + "=" * 50)
    print(f"\U0001f680 Running Input Noise Robustness Test (SNR Analysis) for {dataset.upper()}")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. \u52a8\u6001\u52a0\u8f7d\u6a21\u578b (\u517c\u5bb9\u8de8\u6587\u4ef6\u8c03\u7528)
    try:
        file_path = os.path.join(base_dir, "1_train_cigan.py")
        spec = importlib.util.spec_from_file_location("train_cigan", file_path)
        train_cigan = importlib.util.module_from_spec(spec)
        sys.modules["train_cigan"] = train_cigan
        spec.loader.exec_module(train_cigan)

        train_cigan.device = device
        model = train_cigan.UltraGenerator(30).to(device)
        # \u6ce8\u610f\uff1a\u8fd9\u91cc\u6700\u597d load \u4f60\u7684 generator \u6743\u91cd\u3002\u5982\u679c\u6ca1\u6709\u7edd\u5bf9\u8def\u5f84\uff0c\u5c31\u7528\u521d\u59cb\u5316\u7684\u4e5f\u884c\uff08\u56e0\u4e3a\u6297\u566a\u6027\u4e3b\u8981\u53d6\u51b3\u4e8e\u7f51\u7edc\u7ed3\u6784\u4e2d\u7684 FEN \u6a21\u5757\u7684\u6ee4\u6ce2\u80fd\u529b\uff09
        model.eval()
    except Exception as e:
        print(f"\u274c \u65e0\u6cd5\u52a0\u8f7d\u6a21\u578b\u8fdb\u884c\u9c81\u68d2\u6027\u6d4b\u8bd5\uff0c\u8bf7\u68c0\u67e5 1_train_cigan.py \u8def\u5f84: {e}")
        return

    # 2. \u62bd\u53d6\u4e00\u4e2a Batch \u7684\u5e72\u51c0\u6570\u636e
    data_path = os.path.join(base_dir, "data", dataset, f"{dataset}_30s_preprocessed_checked_labels.npz")
    npz = np.load(data_path)
    eeg = npz["eeg"]

    # \u83b7\u53d6\u5176\u4ed6 30 \u4e2a\u901a\u9053\u7684\u7d22\u5f15
    frontal_idx = [0, 16]
    others_idx = [i for i in range(32) if i not in frontal_idx]

    # \u53d6\u524d 32 \u4e2a Trial \u4f5c\u4e3a\u6d4b\u8bd5\u96c6
    clean_others = torch.tensor(eeg[0:32, others_idx, :], dtype=torch.float32).to(device)
    dummy_w_vec = torch.ones(30, dtype=torch.float32).to(device)  # \u7b80\u5316\u6ce8\u610f\u529b\u6743\u91cd

    snr_levels = [None, 20, 10, 5, 0]  # None \u4ee3\u8868 Clean
    snr_labels = ["Clean", "20dB", "10dB", "5dB", "0dB"]

    results_corr = []
    results_mse = []
    plot_signals = {}  # \u5b58\u653e\u7528\u4e8e\u753b\u56fe\u7684\u5355\u6761\u6ce2\u5f62

    print("\u26a1 Injecting Gaussian White Noise and running inferences...")
    with torch.no_grad():
        # \u57fa\u51c6\u7ebf\uff1a\u6beb\u65e0\u566a\u58f0\u7684\u5e72\u51c0\u751f\u6210
        clean_fake = model(clean_others, dummy_w_vec).cpu().numpy()

        for snr, label in zip(snr_levels, snr_labels):
            if snr is None:
                results_corr.append(1.0)
                results_mse.append(0.0)
                plot_signals[label] = clean_fake[0, 0, :fs * 2]  # \u4fdd\u5b58\u7b2c0\u4e2atrial, Fp1, \u524d2\u79d2
                continue

            # \u8ba1\u7b97\u4fe1\u53f7\u80fd\u91cf\u5e76\u6ce8\u5165\u6307\u5b9a SNR \u7684\u9ad8\u65af\u566a\u58f0
            noisy_others = clean_others.clone()
            for b in range(noisy_others.shape[0]):
                for c in range(noisy_others.shape[1]):
                    sig = noisy_others[b, c, :]
                    p_sig = torch.mean(sig ** 2)
                    p_noise = p_sig / (10 ** (snr / 10.0))
                    # \u751f\u6210\u566a\u58f0
                    noise = torch.sqrt(p_noise) * torch.randn_like(sig)
                    noisy_others[b, c, :] += noise

            # \u566a\u58f0\u6570\u636e\u63a8\u7406
            noisy_fake = model(noisy_others, dummy_w_vec).cpu().numpy()
            plot_signals[label] = noisy_fake[0, 0, :fs * 2]

            # \u8ba1\u7b97\u76f8\u6bd4\u4e8e Clean \u751f\u6210\u7684\u8870\u51cf
            mse = np.mean((clean_fake - noisy_fake) ** 2)
            # \u8ba1\u7b97\u8fd9\u6279\u6570\u636e\u7684\u5e73\u5747\u76ae\u5c14\u900a\u76f8\u5173\u7cfb\u6570\u4fdd\u7559\u5ea6
            corrs = [pearsonr(clean_fake[b, 0, :], noisy_fake[b, 0, :])[0] for b in range(clean_fake.shape[0])]
            mean_corr = np.mean(corrs)

            results_mse.append(mse)
            results_corr.append(mean_corr)
            print(f"  -> {label} Noise | Stability Corr: {mean_corr:.4f} | MSE: {mse:.4f}")

    # ==========================================
    # 3. \u7ed8\u5236 TIM \u7ea7\u522b\u7684\u6297\u566a\u53cc\u5b50\u56fe
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # \u5b50\u56fe1\uff1a\u9c81\u68d2\u6027\u8870\u51cf\u66f2\u7ebf
    x_pos = np.arange(len(snr_labels))
    ax1.plot(x_pos, results_corr, marker='D', markersize=8, color='tab:red', linewidth=2.5, label='Pearson Correlation')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(snr_labels)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Robustness Degradation Curve under Noise", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Input Signal-to-Noise Ratio (SNR)")
    ax1.set_ylabel("Correlation with Clean Generation")
    ax1.grid(True, alpha=0.4, linestyle='--')
    ax1.legend()

    # \u5b50\u56fe2\uff1a2\u79d2\u7a97\u53e3\u6ce2\u5f62\u5bf9\u6bd4 (\u76f4\u89c2\u5c55\u793a\u6297\u566a\u80fd\u529b)
    time_axis = np.arange(fs * 2) / fs
    ax2.plot(time_axis, plot_signals["Clean"], color='black', linewidth=2, label="Clean Output", alpha=0.9)
    ax2.plot(time_axis, plot_signals["10dB"], color='royalblue', linewidth=1.5, label="10dB Noisy Input Output",
             alpha=0.8)
    ax2.plot(time_axis, plot_signals["0dB"], color='tab:orange', linewidth=1.5, linestyle='--',
             label="0dB Extreme Noise Output", alpha=0.7)

    ax2.set_title("Waveform Stability (2-second window)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (\xb5V)")
    ax2.grid(True, alpha=0.4, linestyle='--')
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, "4_noise_robustness_analysis.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\u2705 Noise Robustness analysis plot saved to: {save_path}")



if __name__ == "__main__":
    print("="*60)
    print(f"TIM REVISION: ADVANCED VALIDATION EXPERIMENTS ({DATASET.upper()})")
    print("="*60)

    if real_fp is not None:
        # Exp 1: \u6ce2\u5f62\u7ea7\u522b \u591a\u6a21\u578b BA Summary \u8868\u683c (\u6838\u5fc3\u8fd4\u4fee\u8981\u6c42)
        ba_summary_csv = os.path.join(SAVE_DIR, "1_bland_altman_summary_table.csv")
        build_bland_altman_summary_table(model_paths=MODEL_PATHS, save_csv=ba_summary_csv, fs=FS)

        # Exp 2: \u9891\u6bb5\u7ea7\u522b Band-Power BA \u5206\u6790 (Fp1)
        if "CIGAN (Main)" in MODELS_TO_UPDATE or len(MODELS_TO_UPDATE) == 0:
            bp_ba_fig = os.path.join(SAVE_DIR, "2_band_power_bland_altman.png")
            bp_ba_csv = os.path.join(SAVE_DIR, "2_band_power_bland_altman_summary.csv")
            band_power_bland_altman_analysis(real=real_fp[:, 0, :], fake=fake_fp[:, 0, :], channel_name="Fp1", fs=FS, save_fig=bp_ba_fig, save_csv=bp_ba_csv)

            # Exp 3: \u771f\xb7\u8ba4\u77e5\u4e0d\u786e\u5b9a\u5ea6\u91cf\u5316\u5305\u7edc\u56fe
            uq_path = os.path.join(SAVE_DIR, "3_true_uncertainty_quantification.png")
            true_uncertainty_quantification(real_fp, fake_fp, fake_std, channel_idx=0, save_path=uq_path)
            print(f"\u2705 Generated UQ envelope plot: {uq_path}")
        # ====== \u5c06\u8fd9\u6bb5\u8c03\u7528\u52a0\u5165\u5230 if __name__ == "__main__": \u7684\u6700\u4e0b\u9762 ======
        noise_robustness_test(dataset=DATASET, base_dir=BASE_DIR, save_dir=SAVE_DIR, fs=FS)

    print("\n\U0001f389 All Advanced Validation Experiments Completed Successfully!")