"""
causal_stability_report.py
\u8bfb\u53d6 Bootstrap \u7ed3\u679c\uff0c\u8ba1\u7b97\u6bcf\u4e2a\u901a\u9053\u7684\u201cTop-10 \u5165\u9009\u7387\u201d\u53ca\u201c\u6743\u91cd\u7a33\u5b9a\u6027\u201d\uff0c\u751f\u6210\u8fd4\u4fee\u4e13\u5c5e\u53ef\u89c6\u5316\u56fe\u8868\u3002
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================
# 1. \u8def\u5f84\u4e0e\u914d\u7f6e
# ==============================================================
DATASET = "hci"  # \u53ef\u5207\u6362 hci
BASE_DIR = r"."
CAUSAL_DIR = os.path.join(BASE_DIR, "causality_gpu", DATASET)

# \u7edf\u4e00\u8f93\u51fa\u5230 final_paper_tables \u6587\u4ef6\u5939\uff0c\u65b9\u4fbf\u540e\u7eed\u5199\u8bba\u6587\u76f4\u63a5\u62ff
SAVE_DIR = os.path.join(BASE_DIR, "final_paper_tables", DATASET)
os.makedirs(SAVE_DIR, exist_ok=True)

# ==============================================================
# 2. \u6570\u636e\u52a0\u8f7d\u4e0e\u5904\u7406
# ==============================================================
weights_path = os.path.join(CAUSAL_DIR, "boot_strap_weights.npy")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"\u627e\u4e0d\u5230 {weights_path}\uff0c\u8bf7\u5148\u5728 causilty.py \u5f00\u542f RUN_STABILITY_ANALYSIS")

# weights shape: (n_bootstraps, 30)
weights = np.load(weights_path)
n_boots, n_channels = weights.shape

# \u8fd8\u539f\u901a\u9053\u540d\u79f0 (\u6392\u9664 Fp1 0, Fp2 16)
all_channels = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
    "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4",
    "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
    "PO4", "O2"
]
frontal_idx = [0, 16]
other_names = [all_channels[i] for i in range(len(all_channels)) if i not in frontal_idx]

# ==============================================================
# 3. \u6838\u5fc3\u6307\u6807\u8ba1\u7b97\uff1a\u5747\u503c\u3001\u6807\u51c6\u5dee\u3001Top-10 \u547d\u4e2d\u7387
# ==============================================================
mean_w = np.mean(weights, axis=0)
std_w = np.std(weights, axis=0)

# \u8ba1\u7b97\u6bcf\u4e2a\u901a\u9053\u5728 N \u6b21 Bootstrap \u4e2d\uff0c\u8fdb\u5165\u524d 10 \u540d\u7684\u9891\u7387
top10_freq = np.zeros(n_channels)
for i in range(n_boots):
    # argsort \u4ece\u5c0f\u5230\u5927\u6392\uff0c\u53d6\u6700\u540e10\u4e2a\u5373\u4e3a Top-10 \u7684\u7d22\u5f15
    top10_idx = np.argsort(weights[i])[-10:]
    top10_freq[top10_idx] += 1

top10_prob = (top10_freq / n_boots) * 100

# \u6784\u5efa DataFrame \u5e76\u6309\u5e73\u5747\u6743\u91cd\u964d\u5e8f\u6392\u5217
df = pd.DataFrame({
    "Channel": other_names,
    "Mean_Weight": mean_w,
    "Std_Weight": std_w,
    "Top10_Selection_Prob(%)": top10_prob
})
df = df.sort_values(by="Mean_Weight", ascending=False).reset_index(drop=True)

# \u5bfc\u51fa CSV \u8868\u683c
csv_path = os.path.join(SAVE_DIR, f"Causal_Prior_Stability_{DATASET.upper()}.csv")
df.to_csv(csv_path, index=False)
print(f"\u2705 \u7a33\u5b9a\u6027\u7edf\u8ba1\u8868\u5df2\u5bfc\u51fa\u81f3: {csv_path}")

# ==============================================================
# 4. \u7ed8\u5236\u4f9b\u8bba\u6587\u4f7f\u7528\u7684\u53ef\u89c6\u5316\u56fe\u8868
# ==============================================================
plt.rcParams.update({"font.family": "serif", "font.size": 12})

# \u6211\u4eec\u53ea\u753b\u6392\u540d\u524d 15 \u7684\u901a\u9053\uff0c\u907f\u514d\u56fe\u8868\u8fc7\u4e8e\u62e5\u6324
plot_top_k = 15
sub_df = df.head(plot_top_k)

fig, ax1 = plt.subplots(figsize=(12, 6))

x_pos = np.arange(plot_top_k)
# \u7ed8\u5236\u5e26\u6709\u8bef\u5dee\u68d2\u7684\u5e73\u5747\u6743\u91cd\u6761\u5f62\u56fe
bars = ax1.bar(x_pos, sub_df["Mean_Weight"], yerr=sub_df["Std_Weight"],
               capsize=5, color='royalblue', alpha=0.8, edgecolor='black', label="Mean Weight \xb1 Std")

ax1.set_xticks(x_pos)
ax1.set_xticklabels(sub_df["Channel"], rotation=45, ha='right')
ax1.set_ylabel("Causal Attention Weight (Normalized)", color='royalblue', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='royalblue')
ax1.set_title(f"Robustness of Causal Prior: Top {plot_top_k} Channels over {n_boots} Bootstraps ({DATASET.upper()})")

# \u589e\u52a0\u7b2c\u4e8c y \u8f74\uff0c\u5c55\u793a\u8be5\u901a\u9053\u7684 Top-10 \u547d\u4e2d\u7387
ax2 = ax1.twinx()
ax2.plot(x_pos, sub_df["Top10_Selection_Prob(%)"], color='darkorange', marker='D',
         linestyle='--', linewidth=2, markersize=8, label="Top-10 Selection Prob")
ax2.set_ylabel("Probability of Selection in Top-10 (%)", color='darkorange', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(0, 105)

# \u5408\u5e76\u56fe\u4f8b
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()

# \u4fdd\u5b58\u56fe\u8868
plot_path = os.path.join(SAVE_DIR, f"Causal_Prior_Robustness_Plot_{DATASET.upper()}.png")
plt.savefig(plot_path, dpi=300)
print(f"\u2705 \u8bef\u5dee\u68d2\u4e0e\u6982\u7387\u53cc\u8f74\u56fe\u5df2\u5bfc\u51fa\u81f3: {plot_path}")
plt.show()