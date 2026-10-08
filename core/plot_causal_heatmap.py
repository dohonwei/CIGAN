import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(W, channel_names, save_path=None, title="Causality Matrix"):
    """
    W: (C, C) \u56e0\u679c\u77e9\u9635
    channel_names: list of channel labels
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(W, xticklabels=channel_names, yticklabels=channel_names,
                cmap="viridis", square=True)
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print("\u4fdd\u5b58\u70ed\u529b\u56fe:", save_path)

    plt.show()
