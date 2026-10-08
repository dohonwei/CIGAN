import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# ==========================================
# \u8def\u5f84\u914d\u7f6e
# ==========================================
BASE_DIR = r"./ablation_eswa_deap"

SINGLE_PATH = os.path.join(BASE_DIR, "Full_Model_single.npz")
MC50_PATH = os.path.join(BASE_DIR, "Full_Model_mc50.npz")


# ==========================================
# \u57fa\u7840\u6307\u6807
# ==========================================
def compute_metrics(real, pred):
    pearson_corr, _ = pearsonr(real.flatten(), pred.flatten())

    rmse = np.sqrt(
        mean_squared_error(
            real.flatten(),
            pred.flatten()
        )
    )

    return pearson_corr, rmse


# ==========================================
# UQ\u6307\u6807
# ==========================================
def compute_uq_metrics(real_gt,
                       mean_pred,
                       std_pred,
                       z_score=1.96):

    lower = mean_pred - z_score * std_pred
    upper = mean_pred + z_score * std_pred

    covered = (
        (real_gt >= lower) &
        (real_gt <= upper)
    )

    picp = np.mean(covered)

    data_range = (
        np.max(real_gt) -
        np.min(real_gt)
    )

    interval_width = upper - lower

    pinaw = (
        np.mean(interval_width) /
        data_range
    )

    return picp, pinaw


# ==========================================
# \u6838\u5fc3\uff1a\u771f\u5b9e MC sensitivity
# ==========================================
def generate_real_sensitivity_table():

    print("\n" + "=" * 70)
    print("\U0001f4ca Real Sensitivity Analysis of MC-Dropout")
    print("=" * 70)

    # ------------------------------
    # \u52a0\u8f7d\u6570\u636e
    # ------------------------------
    single_data = np.load(SINGLE_PATH)
    mc_data = np.load(MC50_PATH)

    real_gt = mc_data["real_gt"]

    # K=1 deterministic
    single_pred = (
        single_data["fake_fp"]
        if "fake_fp" in single_data
        else single_data["fake_mean"]
    )

    # \u771f\u5b9e50\u6b21 stochastic prediction
    if "mc_preds" not in mc_data:
        raise ValueError(
            "\u274c mc_preds \u4e0d\u5b58\u5728\uff01\n"
            "\u8bf7\u5148\u4fee\u6539 4_dropout.py "
            "\u4fdd\u5b58 mc_preds \u540e\u91cd\u65b0\u8fd0\u884c mc50 \u63a8\u7406\u3002"
        )

    mc_preds = mc_data["mc_preds"]

    print(f"\u2705 mc_preds shape: {mc_preds.shape}")

    K_list = [1, 5, 10, 20, 30, 40, 50]

    results = []

    for K in K_list:

        print(f"\n\U0001f50d Evaluating K={K}")

        # ---------------------------------
        # K=1\uff08deterministic\uff09
        # ---------------------------------
        if K == 1:

            pearson_corr, rmse = compute_metrics(
                real_gt,
                single_pred
            )

            picp = np.nan
            pinaw = np.nan

        # ---------------------------------
        # \u771f\u6b63\u4f7f\u7528\u524dK\u6b21MC\u9884\u6d4b
        # ---------------------------------
        else:

            selected_preds = mc_preds[:K]

            mean_pred = np.mean(
                selected_preds,
                axis=0
            )

            std_pred = np.std(
                selected_preds,
                axis=0
            )

            pearson_corr, rmse = compute_metrics(
                real_gt,
                mean_pred
            )

            picp, pinaw = compute_uq_metrics(
                real_gt,
                mean_pred,
                std_pred
            )

        results.append({
            "Forward Passes (K)": K,
            "Pearson (CC) \u2191": f"{pearson_corr:.4f}",

            "RMSE \u2193":
                f"{rmse:.4f}",

            "PICP (%) \u2191":
                (
                    "N/A"
                    if K == 1
                    else f"{picp * 100:.2f}"
                ),

            "PINAW \u2193":
                (
                    "N/A"
                    if K == 1
                    else f"{pinaw:.4f}"
                )
        })

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print(df.to_markdown(index=False))
    print("=" * 70)

    save_path = os.path.join(
        BASE_DIR,
        "Table_MC_Dropout_Sensitivity_REAL.csv"
    )

    df.to_csv(save_path, index=False)

    print(f"\n\U0001f4e6 Saved to:")
    print(save_path)

    return df


def plot_uq_figure(npz_path, save_dir, sample_idx=0, channel=0, start=100, end=600, K=50):
    """
    \u7ed8\u5236 ESWA \u671f\u520a\u98ce\u683c\u7684\u5e26\u9634\u5f71\u7684\u4e0d\u786e\u5b9a\u6027\u91cf\u5316\u56fe
    \u4ece mc_preds \u52a8\u6001\u8ba1\u7b97\u5747\u503c\u548c\u6807\u51c6\u5dee\uff0c\u786e\u4fdd\u4e0e\u8868\u683c\u6570\u636e\u4e00\u81f4
    """
    print(f"\n\U0001f3a8 \u6b63\u5728\u7ed8\u5236 UQ \u7f6e\u4fe1\u533a\u95f4\u56fe (ESWA Style, K={K})...")
    data = np.load(npz_path)

    # \u2605 \u4ece mc_preds \u52a8\u6001\u8ba1\u7b97\uff0c\u4e0e\u8868\u683c\u4fdd\u6301\u4e00\u81f4
    if "mc_preds" in data:
        mc_preds = data["mc_preds"][:K]  # \u53d6\u524dK\u6b21
        mean_all = np.mean(mc_preds, axis=0)
        std_all = np.std(mc_preds, axis=0)
        gt_all = data['real_gt']

        mean = mean_all[sample_idx, channel, start:end]
        std = std_all[sample_idx, channel, start:end]
        gt = gt_all[sample_idx, channel, start:end]
    else:
        # \u964d\u7ea7\u65b9\u6848\uff1a\u4f7f\u7528\u9884\u8ba1\u7b97\u7684\u5747\u503c\u548c\u6807\u51c6\u5dee
        mean = data['fake_mean'][sample_idx, channel, start:end]
        std = data['fake_std'][sample_idx, channel, start:end]
        gt = data['real_gt'][sample_idx, channel, start:end]

    t = np.arange(len(mean))

    plt.figure(figsize=(10, 4), dpi=300)  # \u9ad8\u6e05 300 DPI

    # 95% \u7f6e\u4fe1\u533a\u95f4\u9634\u5f71
    plt.fill_between(t, mean - 1.96 * std, mean + 1.96 * std,
                     color='#add8e6', alpha=0.6, label='95% CI (Epistemic Uncertainty)')

    # \u9884\u6d4b\u5747\u503c
    plt.plot(t, mean, color='#0047ab', linewidth=1.8, label='Predicted Mean (CIGAN)')

    # \u771f\u5b9e\u8111\u7535\u6ce2 Ground Truth
    plt.plot(t, gt, color='#d62728', linewidth=1.5, linestyle='--', label='Ground Truth EEG')

    plt.title('Uncertainty Quantification of Reconstructed EEG Signal', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Amplitude (\u03bcV)', fontsize=12)

    # \u5b66\u672f\u98ce\u7f51\u683c\u4e0e\u56fe\u4f8b
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right', framealpha=0.9, fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'Fig_UQ_Confidence_Interval_{idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"\u2705 UQ \u56fe\u5df2\u6210\u529f\u4fdd\u5b58\u81f3: {save_path}")



# ==========================================
# \u4e3b\u7a0b\u5e8f
# ==========================================
if __name__ == "__main__":

    print("\U0001f680 Running REAL MC-Dropout sensitivity analysis")

    # generate_real_sensitivity_table()
    for idx in [0, 5, 10,15]:
        plot_uq_figure(MC50_PATH, BASE_DIR, sample_idx=idx, channel=0, K=50)
    print("\n\U0001f389 Done.")