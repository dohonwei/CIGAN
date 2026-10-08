import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ============================================================
# 1. \u5168\u5c40\u914d\u7f6e\u4e0e\u73af\u5883
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\U0001f525 \u4f7f\u7528\u8ba1\u7b97\u8bbe\u5907: {device}")

BASE_DIR = r"."
DATASETS = ["deap"]  # \u53ef\u6309\u9700\u52a0\u5165 "deap"
EPOCHS = 100
BATCH_SIZE = 128
LR_G = 1e-4
LR_D = 1e-4


# ============================================================
# 2. \u6838\u5fc3\u5de5\u5177\u4e0e\u635f\u5931\u51fd\u6570 (\u7eaf PyTorch \u5b9e\u73b0)
# ============================================================
def align_length(a, b):
    Ta, Tb = a.shape[-1], b.shape[-1]
    if Ta == Tb: return a, b
    T = min(Ta, Tb)
    return a[..., :T], b[..., :T]


def apply_fen_auto_differentiable(x, fs):
    """\u7eaf PyTorch \u5b9e\u73b0\u7684\u53ef\u5fae FEN \u6a21\u5757 (\u63d0\u53d6 1-30Hz)"""
    B, C, T = x.shape
    X_f = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(T, 1 / fs).to(x.device)
    mask = ((freqs >= 1.0) & (freqs <= 30.0)).float().unsqueeze(0).unsqueeze(0)
    return torch.fft.irfft(X_f * mask, n=T, dim=-1)


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
# 3. \u52a8\u6001\u6d88\u878d\u7f51\u7edc\u7ed3\u6784 (\u542b MC-Dropout \u7684\u7075\u9b42)
# ============================================================
class WaveNetResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()

        self.filter = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )

        self.gate = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )

        self.out = nn.Conv1d(
            channels,
            channels,
            kernel_size=1
        )

        # \u2b50 \u66f4\u5f3a MC-Dropout
        self.drop = nn.Dropout1d(
            p=0.25
        )

    def forward(self, x):

        f = torch.tanh(
            self.filter(x)
        )

        g = torch.sigmoid(
            self.gate(x)
        )

        h = f * g

        # stochastic masking
        h = self.drop(h)

        out = self.out(h)

        return x + out


class UltraEncoder(nn.Module):
    def __init__(self,
                 in_ch=30,
                 hidden=64):

        super().__init__()

        self.inp = nn.Conv1d(
            in_ch,
            hidden,
            kernel_size=1
        )

        # \u2b50 \u8f93\u5165\u5c42 stochasticity
        self.input_drop = nn.Dropout1d(
            p=0.20
        )

        self.blocks = nn.ModuleList([
            WaveNetResBlock(hidden, d)
            for d in [1, 2, 4, 8, 16, 32, 64, 128]
        ])

        self.out = nn.Conv1d(
            hidden,
            2,
            kernel_size=1
        )

    def forward(self, x):

        h = self.inp(x)

        # \u2b50 input-level dropout
        h = self.input_drop(h)

        for blk in self.blocks:
            h = blk(h)

        out = self.out(h)

        return out


class UltraGeneratorAblation(nn.Module):
    def __init__(self, O, fs, use_causal=True, use_fen=True):
        super().__init__()
        self.fs = fs
        self.use_causal = use_causal
        self.use_fen = use_fen
        self.encoder = UltraEncoder(in_ch=O)

        if self.use_causal:
            self.w_fc = nn.Linear(O, O)

        if self.use_fen:
            self.final_merge = nn.Conv1d(4, 2, kernel_size=1)
            nn.init.zeros_(self.final_merge.weight)
            nn.init.zeros_(self.final_merge.bias)

    def forward(self, others, w_vec):
        gen = self.encoder(others)

        if not self.use_fen:
            return gen

        if self.use_causal:
            w = self.w_fc(w_vec).unsqueeze(0).unsqueeze(-1)
            w_oth = others * w
        else:
            w_oth = others

        fen_in = w_oth.mean(dim=1, keepdim=True).repeat(1, 2, 1)
        fen = apply_fen_auto_differentiable(fen_in, self.fs)

        gen, fen = align_length(gen, fen)
        out = gen + self.final_merge(torch.cat([gen, fen], dim=1))
        return out


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


# ============================================================
# 4. \u6570\u636e\u52a0\u8f7d
# ============================================================
def load_dataset(dataset_name):
    npz_path = os.path.join(BASE_DIR, "data", dataset_name, f"{dataset_name}_30s_preprocessed_checked_labels.npz")
    w_path = os.path.join(BASE_DIR, "causality_gpu", dataset_name, "frontal_weights.npy")
    idx_path = os.path.join(BASE_DIR, "causality_gpu", dataset_name, "other_indices.npy")

    npz = np.load(npz_path)
    eeg = npz["eeg"]
    others = torch.tensor(eeg[:, np.load(idx_path), :], dtype=torch.float32, device=device)
    frontal = torch.tensor(eeg[:, [0, 16], :], dtype=torch.float32, device=device)
    w_vec = torch.tensor(np.load(w_path), dtype=torch.float32, device=device)

    fs = 128 if dataset_name == "deap" else 256
    return others, frontal, w_vec, fs


# ============================================================
# 5. \u8bad\u7ec3\u5f15\u64ce (\u8d1f\u8d23\u8bad\u7ec3\u5e76\u8fd4\u56de\u6700\u4f18\u6a21\u578b)
# ============================================================
def train_and_generate_ablation(dataset_name, ablation_name, config, others, frontal, w_vec, fs):
    N, O, T = others.shape

    G = UltraGeneratorAblation(O, fs, use_causal=config["use_causal"], use_fen=config["use_fen"]).to(device)
    D = UltraDisc().to(device)

    optG = torch.optim.Adam(G.parameters(), lr=LR_G, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.5, 0.9))

    print(f"\n\U0001f680 \u5f00\u59cb\u8bad\u7ec3 [{ablation_name}] on [{dataset_name.upper()}]")
    idx_all = torch.arange(N)

    for ep in range(1, EPOCHS + 1):
        perm = idx_all[torch.randperm(N)]
        o_shuf, f_shuf = others[perm], frontal[perm]
        G.train();
        D.train()
        g_loss_accum = 0.0

        pbar = tqdm(range(0, N, BATCH_SIZE), desc=f"Epoch {ep}/{EPOCHS}", leave=False)
        for i in pbar:
            o_b, r_b = o_shuf[i:i + BATCH_SIZE], f_shuf[i:i + BATCH_SIZE]
            if o_b.size(0) < 2: continue

            # --- Train D ---
            fake = G(o_b, w_vec)
            fake, r_align = align_length(fake, r_b)
            loss_D = D(fake.detach()).mean() - D(r_align).mean() + gradient_penalty(D, r_align, fake.detach())
            optD.zero_grad();
            loss_D.backward();
            optD.step()

            # --- Train G ---
            fake = G(o_b, w_vec)
            fake, r_align = align_length(fake, r_b)

            loss_G = -D(fake).mean() + 5.0 * loss_tc(fake, r_align) + 3.0 * loss_amp(fake, r_align)
            if config["use_freq_loss"]:
                loss_G += 1.0 * loss_psd_torch(fake, r_align) + 1.0 * loss_stft_torch(fake, r_align)

            optG.zero_grad();
            loss_G.backward();
            optG.step()
            g_loss_accum += loss_G.item()
            pbar.set_postfix(G_Loss=f"{g_loss_accum / (i / BATCH_SIZE + 1):.3f}")

    print(f"\u2705 [{ablation_name}] \u8bad\u7ec3\u5b8c\u6210\uff01")
    return G  # \u2605 \u5173\u952e\uff1a\u8fd4\u56de\u8bad\u7ec3\u597d\u7684\u6a21\u578b\uff0c\u4ea4\u7ed9\u4e0b\u6e38\u63a8\u7406\u51fd\u6570


# ============================================================
# 6. \u63a8\u7406\u4e0e\u6821\u51c6\u5f15\u64ce (\u652f\u6301 Single \u548c MC-Dropout)
# ============================================================
def generate_cigan_inference_ablation(
        G,
        others,
        frontal,
        w_vec,
        dataset_name,
        exp_name,
        mode='single',
        num_passes=50,
        batch_size=64):

    print(f"\n{'-' * 50}")
    print(
        f"\U0001f50d \u542f\u52a8\u63a8\u7406 | "
        f"\u5b9e\u9a8c: [{exp_name}] | "
        f"\u6a21\u5f0f: [{mode.upper()}]"
    )

    N, C, T = others.shape

    G.eval()

    # ===================================
    # \u6b63\u786e\u542f\u7528 MC-Dropout
    # ===================================
    if mode == 'mc50':

        for m in G.modules():

            if isinstance(
                m,
                (nn.Dropout, nn.Dropout1d)
            ):
                m.train()

    elif mode == 'single':

        num_passes = 1

        for m in G.modules():

            if isinstance(
                m,
                (nn.Dropout, nn.Dropout1d)
            ):
                m.eval()

    else:
        raise ValueError(
            "mode \u5fc5\u987b\u662f "
            "'single' \u6216 'mc50'"
        )

    mc_preds_all = []

    with torch.no_grad():

        for i in tqdm(
                range(0, N, batch_size),
                desc="Generating",
                leave=False):

            o_b = others[i:i + batch_size]

            if o_b.size(0) == 0:
                continue

            batch_preds = []

            for _ in range(num_passes):

                fake = G(
                    o_b,
                    w_vec
                )[..., :T]

                batch_preds.append(
                    fake.unsqueeze(0)
                )

            batch_preds = torch.cat(
                batch_preds,
                dim=0
            )

            mc_preds_all.append(
                batch_preds.cpu().numpy()
            )

    mc_preds = np.concatenate(
        mc_preds_all,
        axis=1
    )

    mean_pred = np.mean(
        mc_preds,
        axis=0
    )

    if mode == 'mc50':

        std_pred = np.std(
            mc_preds,
            axis=0
        )

    else:

        std_pred = np.zeros_like(
            mean_pred
        )

    real_gt = frontal[
              ..., :T
              ].cpu().numpy()

    # ===================================
    # \u4e0d\u505a\u4eba\u5de5 calibration
    # ===================================
    final_std = std_pred

    # ===================================
    # Debug uncertainty
    # ===================================
    if mode == 'mc50':

        z_score = 1.96

        lower_bound = (
                mean_pred
                - z_score * std_pred
        )

        upper_bound = (
                mean_pred
                + z_score * std_pred
        )

        picp = np.mean(
            (real_gt >= lower_bound)
            &
            (real_gt <= upper_bound)
        )

        print("\n\U0001f4ca MC Uncertainty Statistics")

        print(
            f"Mean STD : "
            f"{np.mean(std_pred):.4f}"
        )

        print(
            f"Max STD  : "
            f"{np.max(std_pred):.4f}"
        )

        print(
            f"PICP     : "
            f"{picp * 100:.2f}%"
        )

    # ===================================
    # \u4fdd\u5b58
    # ===================================
    save_dir = os.path.join(
        BASE_DIR,
        f"ablation_eswa_{dataset_name}"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    save_name = os.path.join(
        save_dir,
        f"{exp_name}_{mode}.npz"
    )

    np.savez(
        save_name,

        fake_mean=mean_pred,

        fake_std=final_std,

        # \u2b50 \u4fdd\u5b58\u6240\u6709 MC samples
        mc_preds=mc_preds,

        real_gt=real_gt
    )

    print(
        f"\U0001f4e6 \u6570\u636e\u4fdd\u5b58\u81f3: "
        f"{save_name}"
    )

    return (
        mean_pred,
        final_std,
        real_gt
    )


# ============================================================
# 7. \u603b\u63a7\u6d41\u6c34\u7ebf
# ============================================================
ABLATION_EXPERIMENTS = {
    "Full Model": {"use_causal": True, "use_fen": True, "use_freq_loss": True},
    # "wo Causal": {"use_causal": False, "use_fen": True, "use_freq_loss": True},
    # "wo FEN": {"use_causal": True, "use_fen": False, "use_freq_loss": True},
    # "wo PSD": {"use_causal": True, "use_fen": True, "use_freq_loss": False},
    # "wo CFP": {"use_causal": False, "use_fen": False, "use_freq_loss": False},
}

if __name__ == "__main__":
    print("=" * 60)
    print(" \U0001f52c \u7ec8\u6781\u6d88\u878d\u5b9e\u9a8c\u77e9\u9635\u542f\u52a8 (\u9002\u914d\u6700\u65b0\u67b6\u6784 & \u5ba1\u7a3f\u4eba\u9632\u7ebf\u7248) \U0001f52c ")
    print("=" * 60)

    for dataset in DATASETS:
        print(f"\n{'#' * 60}")
        print(f" \u52a0\u8f7d\u6570\u636e\u96c6: {dataset.upper()} ")
        print(f"{'#' * 60}")

        others, frontal, w_vec, fs = load_dataset(dataset)

        for exp_name, config in ABLATION_EXPERIMENTS.items():

            # 1. \u8bad\u7ec3\u83b7\u53d6\u5f53\u524d\u6d88\u878d\u914d\u7f6e\u7684\u6a21\u578b
            G = train_and_generate_ablation(dataset, exp_name, config, others, frontal, w_vec, fs)
            safe_exp_name = exp_name.replace(" ", "_")

            # 2. \u751f\u6210\u57fa\u7840 Single \u6570\u636e (\u6240\u6709\u5b9e\u9a8c\u90fd\u8dd1\uff0c\u7528\u4e8e\u8ba1\u7b97 DTW/MMD \u7b49\u6307\u6807)
            # generate_cigan_inference_ablation(G, others, frontal, w_vec, dataset, safe_exp_name, mode="single")
            #
            # # 3. \U0001f3af \u6838\u5fc3\u62a4\u57ce\u6cb3\uff1a\u53ea\u9488\u5bf9 Full Model \u8dd1 MC-Dropout \u4e0d\u786e\u5b9a\u6027\u5206\u6790
            if exp_name == "Full Model":
                generate_cigan_inference_ablation(G, others, frontal, w_vec, dataset, safe_exp_name, mode="mc50")

    print("\n\U0001f389 \u6240\u6709\u5b9e\u9a8c\u53ca\u63a8\u7406\u751f\u6210\u4efb\u52a1\u5df2\u5168\u90e8\u5b8c\u6210\uff01")