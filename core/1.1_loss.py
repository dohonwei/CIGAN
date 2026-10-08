import os
import numpy as np
import matplotlib.pyplot as plt

# \U0001f4c2 \u8bf7\u6839\u636e\u4f60\u7684\u6570\u636e\u96c6\u7c7b\u578b\u4fee\u6539\u8fd9\u91cc\uff1a"hci" \u6216 "deap"
DATASET = "hci"
BASE_DIR = r"."
LOSS_FILE = os.path.join(BASE_DIR, "comparison", DATASET, "losslogtimrevisionnopsd.npz")


def plot_training_curves():
    if not os.path.exists(LOSS_FILE):
        print(f"\u274c \u627e\u4e0d\u5230\u65e5\u5fd7\u6587\u4ef6\uff1a{LOSS_FILE}\n\u8bf7\u786e\u4fdd\u5148\u8fd0\u884c\u4e3b\u4ee3\u7801\u751f\u6210\u4e86\u8be5\u6587\u4ef6\u3002")
        return

    # 1. \u52a0\u8f7d\u6570\u636e (shape: 5_folds x Epochs)
    data = np.load(LOSS_FILE)

    # \u53d6 5 \u4f2a\u6298\u7684\u5e73\u5747\u503c\uff0c\u5f97\u5230\u5e73\u6ed1\u4e14\u7a33\u5b9a\u7684\u4e3b\u7ebf (shape: Epochs)
    Lgan = np.mean(data["Lgan"], axis=0)
    Ltc = np.mean(data["Ltc"], axis=0)
    Lamp = np.mean(data["Lamp"], axis=0)
    Lstft = np.mean(data["Lstft"], axis=0)

    Gloss = np.mean(data["Gloss"], axis=0)
    Dloss = np.mean(data["Dloss"], axis=0)

    epochs = np.arange(1, len(Lgan) + 1)

    # 2. \u5f00\u59cb\u753b\u56fe\u914d\u7f6e
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # ------------------ (a) Component Losses ------------------
    ax1 = axes[0]
    ax1.plot(epochs, Ltc, label=r'$L_{tc}$ (Temporal)', color='#1f77b4', linewidth=2)
    ax1.plot(epochs, Lstft, label=r'$L_{stft}$ (Time-Freq)', color='#ff7f0e', linewidth=2)
    ax1.plot(epochs, Lamp, label=r'$L_{amp}$ (Amplitude)', color='#2ca02c', linewidth=2)
    ax1.plot(epochs, Lgan, label=r'$L_{gan}$ (Adversarial)', color='#d62728', linewidth=2, linestyle='--')

    ax1.set_title("(a) Generator Component Losses", fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss Value", fontsize=12)
    ax1.legend(loc="upper right", frameon=True, fontsize=11)

    # ------------------ (b) Total G and D Losses ------------------
    ax2 = axes[1]
    ax2.plot(epochs, Gloss, label=r'Total $G_{loss}$', color='#9467bd', linewidth=2.5)
    ax2.plot(epochs, Dloss, label=r'Total $D_{loss}$', color='#8c564b', linewidth=2.5)

    ax2.set_title("(b) Adversarial Training Dynamics", fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Total Loss", fontsize=12)
    ax2.legend(loc="upper right", frameon=True, fontsize=11)

    # 3. \u5e03\u5c40\u8c03\u6574\u4e0e\u4fdd\u5b58
    plt.tight_layout()
    save_fig_path = os.path.join(BASE_DIR, "comparison", DATASET, "Fig2_Loss_Curves.png")
    plt.savefig(save_fig_path, bbox_inches='tight')
    plt.show()
    print(f"\U0001f389 \u753b\u56fe\u5b8c\u6210\uff01\u56fe\u7247\u5df2\u4fdd\u5b58\u81f3: {save_fig_path}")


if __name__ == "__main__":
    plot_training_curves()