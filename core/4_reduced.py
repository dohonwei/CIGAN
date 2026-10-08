import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

# ============================================================
# 0. \u5168\u5c40\u914d\u7f6e\u4e0e\u8fd0\u884c\u77e9\u9635
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\U0001f525 \u4f7f\u7528\u8ba1\u7b97\u8bbe\u5907: {device}")

# \u57fa\u7840\u76ee\u5f55 (\u8bf7\u786e\u4fdd\u4e0e\u4f60\u7684\u672c\u5730\u73af\u5883\u4e00\u81f4)
BASE_DIR = r"."
OUT_DIR_BASE = os.path.join(BASE_DIR, "comparison")

DATASETS = [ "deap"]

# \u8fd0\u884c\u77e9\u9635\u5b9a\u4e49
FIXED_SETTINGS = []
RANDOM_SETTINGS = ["Random-10", "Random-5"]
RANDOM_SEEDS = [42, 43, 44]  # 3 \u4e2a Seed\uff0c\u6700\u540e\u7b97\u5747\u503c\u65b9\u5dee
FIXED_SEEDS = [ 42, 43, 44]
# \u8bad\u7ec3\u8d85\u53c2\u6570 (\u4e25\u683c\u5bf9\u9f50 1.all.py)
EPOCHS = 100
BATCH_SIZE = 32
LR_G = 1e-4
LR_D = 1e-4


# ============================================================
# 1. \u6838\u5fc3\u7f51\u7edc\u67b6\u6784 (\u76f4\u63a5\u4ece 1.all.py \u590d\u5236\u8fc7\u6765)
# ============================================================
def align_length(a, b):
    Ta, Tb = a.shape[-1], b.shape[-1]
    if Ta == Tb: return a, b
    T = min(Ta, Tb)
    return a[..., :T], b[..., :T]


def apply_fen_auto_differentiable(x, fs):
    B, C, T = x.shape
    X_f = torch.fft.rfft(x, dim=-1, norm="forward")
    freqs = torch.fft.rfftfreq(T, 1 / fs).to(x.device)
    mask = ((freqs >= 1.0) & (freqs <= 30.0)).float().unsqueeze(0).unsqueeze(0)
    return torch.fft.irfft(X_f * mask, n=T, dim=-1, norm="forward")


class WaveNetResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        return x + self.out(torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x)))


class UltraEncoder(nn.Module):
    def __init__(self, in_ch=30, hidden=64):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([WaveNetResBlock(hidden, d) for d in [1, 2, 4, 8, 16, 32, 64, 128]])
        self.out = nn.Conv1d(hidden, 2, 1)

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks: h = blk(h)
        return self.out(h)


class UltraGenerator(nn.Module):
    """
    \u6ce8\u610f\uff1a\u8fd9\u91cc\u7684 O \u51b3\u5b9a\u4e86\u8f93\u5165\u901a\u9053\u6570\u3002
    \u5f53\u4f20\u7ed9\u5b83\u7684 others \u662f\u964d\u7ef4\u540e\u7684\u6570\u636e\u65f6\uff0cw_fc \u4e5f\u4f1a\u81ea\u9002\u5e94\u53d8\u5c0f\u3002
    """

    def __init__(self, O, fs):
        super().__init__()
        self.fs = fs
        self.encoder = UltraEncoder(in_ch=O)
        self.w_fc = nn.Linear(O, O)
        self.final_merge = nn.Conv1d(4, 2, kernel_size=1)
        nn.init.zeros_(self.final_merge.weight)
        nn.init.zeros_(self.final_merge.bias)

    def forward(self, others, w_vec):
        gen = self.encoder(others)
        w = self.w_fc(w_vec).unsqueeze(0).unsqueeze(-1)
        w_oth = others * w
        fen_in = w_oth.mean(dim=1, keepdim=True).repeat(1, 2, 1)
        fen = apply_fen_auto_differentiable(fen_in, self.fs)
        gen, fen = align_length(gen, fen)
        return gen + self.final_merge(torch.cat([gen, fen], dim=1))


class UltraDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 7, stride=2, padding=3), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 7, stride=2, padding=3), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1))


# --- \u635f\u5931\u51fd\u6570 ---
def loss_tc(fake, real):
    fake, real = align_length(fake, real)
    return F.l1_loss(fake, real)


def loss_amp(fake, real):
    fake, real = align_length(fake, real)
    return F.l1_loss(fake.abs().mean(dim=-1), real.abs().mean(dim=-1))


def loss_psd_torch(fake, real):
    F_fake = torch.fft.rfft(fake, norm="forward")
    F_real = torch.fft.rfft(real, norm="forward")
    return F.l1_loss(torch.abs(F_fake), torch.abs(F_real))


def loss_stft_torch(fake, real):
    B, C, T = fake.shape
    win = torch.hann_window(64).to(fake.device)
    F_fake = torch.stft(fake.view(-1, T), n_fft=64, hop_length=32, window=win, return_complex=True)
    F_real = torch.stft(real.view(-1, T), n_fft=64, hop_length=32, window=win, return_complex=True)
    return F.l1_loss(torch.abs(F_fake) / 64.0, torch.abs(F_real) / 64.0)


def gradient_penalty(D, real, fake, lambda_gp=10.0):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out = D(inter)
    grad = torch.autograd.grad(outputs=out, inputs=inter, grad_outputs=torch.ones_like(out),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    return lambda_gp * ((grad.view(B, -1).norm(2, dim=1) - 1.0) ** 2).mean()


# ============================================================
# 2. \u6570\u636e\u52a0\u8f7d\u4e0e\u901a\u9053\u6620\u5c04\u6a21\u5757
# ============================================================
def get_subset_indices(dataset_name, setting_name, seed=42):
    """
    \u6838\u5fc3\uff1a\u6839\u636e setting \u8fd4\u56de\u8981\u4fdd\u7559\u7684\u901a\u9053\u7d22\u5f15\u96c6\u5408 (0-29 \u5185)
    """
    if setting_name == "Full-30":
        return np.arange(30)

    k = int(setting_name.split("-")[1])

    if "Random" in setting_name:
        np.random.seed(seed)
        random.seed(seed)
        return np.sort(np.random.choice(30, k, replace=False))

    elif "Top" in setting_name:
        csv_path = os.path.join(BASE_DIR, "causality_gpu", dataset_name, "causal_channel_ranking.csv")
        # \u517c\u5bb9\u6027\u5904\u7406\uff1a\u5982\u679c\u4f60\u6ca1\u6709\u5206\u522b\u5b58\u653e hci/deap \u7684 ranking\uff0c\u56de\u9000\u5230\u4e3b\u76ee\u5f55
        if not os.path.exists(csv_path):
            csv_path = os.path.join(BASE_DIR, "causality_gpu", "causal_channel_ranking.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"\u627e\u4e0d\u5230\u6392\u5e8f\u6587\u4ef6: {csv_path}\u3002\u8bf7\u68c0\u67e5\u8def\u5f84\u3002")

        df = pd.read_csv(csv_path)
        # \u53d6 CSV \u524d K \u884c\u5bf9\u5e94\u7684\u7d22\u5f15 (\u5047\u8bbe CSV \u5df2\u7ecf\u6309\u7167\u91cd\u8981\u6027\u6392\u597d\u5e8f\uff0c\u4e14\u7b2c\u4e00\u5217/Index \u662f\u539f\u59cb\u901a\u9053\u53f7)
        # \u6839\u636e\u4f60\u4e4b\u524d\u63d0\u4f9b\u7684\u4fe1\u606f\uff0c\u901a\u5e38\u7528 DataFrame \u7684 index \u5373\u53ef
        return np.sort(df.index[:k].values)


def load_full_dataset(dataset_name):
    """\u52a0\u8f7d\u539f\u59cb\u5b8c\u6574\u6570\u636e\u96c6"""
    npz_path = os.path.join(BASE_DIR, "data", dataset_name, f"{dataset_name}_30s_preprocessed_checked_labels.npz")
    w_path = os.path.join(BASE_DIR, "causality_gpu", dataset_name, "frontal_weights.npy")
    idx_path = os.path.join(BASE_DIR, "causality_gpu", dataset_name, "other_indices.npy")

    npz = np.load(npz_path)
    eeg = npz["eeg"]

    others = eeg[:, np.load(idx_path), :]
    frontal = eeg[:, [0, 16], :]
    w_vec = np.load(w_path)
    fs = 128 if dataset_name == "deap" else 256

    return others, frontal, w_vec, fs


# ============================================================
# 3. \u8fd0\u884c\u5c01\u88c5\u6a21\u5757 (\u540c 1.all.py \u8bad\u7ec3\u903b\u8f91)
# ============================================================
def run_setting_experiment(dataset_name, seed,setting_name, others_data, frontal_data, w_vec_data, fs, save_dir):
    """\u5355\u72ec\u8dd1\u4e00\u4e2a setting \u7684\u5168\u91cf\u8bad\u7ec3\u548c\u751f\u6210"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    N, O, T = others_data.shape
    print(f"\n\U0001f680 \u5f00\u59cb\u8bad\u7ec3: {setting_name} (\u901a\u9053\u6570 O={O}, \u79cd\u5b50={seed})")
    N, O, T = others_data.shape
    print(f"\n\U0001f680 \u5f00\u59cb\u8bad\u7ec3: {setting_name} (\u901a\u9053\u6570 O={O})")

    # \u5b9e\u4f8b\u5316\u81ea\u9002\u5e94\u5c3a\u5bf8\u7684\u6a21\u578b
    G = UltraGenerator(O, fs).to(device)
    D = UltraDisc().to(device)

    optG = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.9))

    # \u8f6c\u4e3a Tensor
    o_ts = torch.tensor(others_data, dtype=torch.float32, device=device)
    f_ts = torch.tensor(frontal_data, dtype=torch.float32, device=device)
    w_ts = torch.tensor(w_vec_data, dtype=torch.float32, device=device)

    idx_all = torch.arange(N)

    # --- \u8bad\u7ec3\u5faa\u73af (\u4e25\u683c\u5bf9\u9f50 1.all.py \u7684 CIGAN \u8bad\u7ec3) ---
    for ep in range(1, EPOCHS + 1):
        perm = idx_all[torch.randperm(N)]
        o_shuf, f_shuf = o_ts[perm], f_ts[perm]
        G.train();
        D.train()
        g_loss_accum = 0.0

        pbar = tqdm(range(0, N, BATCH_SIZE), desc=f"Epoch {ep}/{EPOCHS}", leave=False)
        for i in pbar:
            o_b, r_b = o_shuf[i:i + BATCH_SIZE], f_shuf[i:i + BATCH_SIZE]
            if o_b.size(0) < 2: continue

            # Train D
            fake = G(o_b, w_ts)
            fake, r_align = align_length(fake, r_b)
            loss_D = D(fake.detach()).mean() - D(r_align).mean() + gradient_penalty(D, r_align, fake.detach())
            optD.zero_grad();
            loss_D.backward();
            optD.step()

            # Train G
            fake = G(o_b, w_ts)
            fake, r_align = align_length(fake, r_b)
            loss_G = -D(fake).mean() + 5.0 * loss_tc(fake, r_align) + 3.0 * loss_amp(fake, r_align)
            loss_G += 1.0 * loss_psd_torch(fake, r_align) + 1.0 * loss_stft_torch(fake, r_align)
            optG.zero_grad();
            loss_G.backward();
            optG.step()

            g_loss_accum += loss_G.item()
            pbar.set_postfix(G_Loss=f"{g_loss_accum / (i / BATCH_SIZE + 1):.3f}")

    # --- \u751f\u6210\u9636\u6bb5 ---
    print(f"  \u2705 \u8bad\u7ec3\u5b8c\u6210\uff0c\u6b63\u5728\u751f\u6210\u5168\u96c6\u9884\u6d4b...")
    G.eval()
    fake_list = []
    with torch.no_grad():
        for i in range(0, N, BATCH_SIZE):
            fake_list.append(G(o_ts[i:i + BATCH_SIZE], w_ts)[..., :T].cpu().numpy())

    fake_all = np.concatenate(fake_list, axis=0)

    # \u786e\u4fdd\u6587\u4ef6\u5939\u5b58\u5728
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "fake_fp.npy")
    np.save(save_path, fake_all)

    print(f"  \U0001f4e6 \u9884\u6d4b\u7ed3\u679c\u5df2\u4fdd\u5b58\u81f3: {save_path} (\u5f62\u72b6: {fake_all.shape})")


# ============================================================
# 4. \u7ec8\u6781\u8c03\u5ea6\u4e3b\u5f15\u64ce
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" \U0001f3af \u72ec\u7acb\u6a21\u5757: Reduced-Source \u51cf\u5c11\u6e90\u901a\u9053\u6d88\u878d\u5b9e\u9a8c ")
    print("=" * 60)

    # 2. \u5c06 Top \u548c Random \u90fd\u653e\u5165\u591a\u79cd\u5b50\u77e9\u9635\u4e2d
    MULTI_SEED_SETTINGS = ["Top-10", "Top-5"]
    RANDOM_SEEDS = [42, 43, 44]

    for dataset in DATASETS:
        print(f"\n\U0001f4c2 \u6b63\u5728\u52a0\u8f7d\u6570\u636e\u96c6: {dataset.upper()}")
        others_all, frontal_all, w_vec_all, fs = load_full_dataset(dataset)

        # \u81ea\u52a8\u6784\u5efa\u5168\u77e9\u9635 Setting \u5217\u8868
        all_settings_to_run = FIXED_SETTINGS.copy()
        for ms in MULTI_SEED_SETTINGS:
            for seed in RANDOM_SEEDS:
                all_settings_to_run.append(f"{ms}_seed{seed}")

        # \u5f00\u59cb\u5faa\u73af\u6267\u884c
        for setting in all_settings_to_run:
            print(f"\n\u5f53\u524d\u6267\u884c: {dataset.upper()} | Setting: {setting}")

            current_seed = 42
            base_setting = setting
            if "_seed" in setting:
                base_setting, seed_str = setting.split("_seed")
                current_seed = int(seed_str)

            indices = get_subset_indices(dataset, base_setting, seed=current_seed)
            others_sub = others_all[:, indices, :]
            w_vec_sub = w_vec_all[indices]

            # \u4e13\u5c5e\u8f93\u51fa\u76ee\u5f55\uff0c\u5982: reducedsource/Top-5_seed42/
            save_dir = os.path.join(OUT_DIR_BASE, dataset, "reduced_source", setting)

            # \u3010\u6838\u5fc3\u4fee\u6539\u3011\uff1a\u4f20\u5165\u5f53\u524d\u7684\u79cd\u5b50 current_seed
            run_setting_experiment(
                dataset_name=dataset,
                setting_name=setting,
                others_data=others_sub,
                frontal_data=frontal_all,
                w_vec_data=w_vec_sub,
                fs=fs,
                save_dir=save_dir,
                seed=current_seed  # <-- \u786e\u4fdd\u8fd9\u4e00\u884c\u4f20\u5165\u4e86\u79cd\u5b50
            )

    print("\n\U0001f389 \u606d\u559c\uff01Reduced-Source \u964d\u7ef4\u77e9\u9635\u5b9e\u9a8c\u5168\u90e8\u8c03\u5ea6\u5b8c\u6210\uff01")