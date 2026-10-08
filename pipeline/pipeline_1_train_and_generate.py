"""
pipeline_1_train_and_generate.py
(100% \u65e0\u5220\u51cf\u5b8c\u5168\u878d\u5408\u7248)
\u5305\u542b\u6240\u6709\u4e3b\u6a21\u578b\u3001\u6d88\u878d\u6a21\u578b\u3001\u5168\u90e8\u6df1\u5ea6\u57fa\u7ebf (Vanilla, hvEEGNet, TIE-EEGNet) \u4ee5\u53ca Loss \u8bb0\u5f55\u903b\u8f91\u3002
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch, butter, filtfilt
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#                    GLOBAL CONFIG & PATHS
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET = "deap"  # \u2605 \u5728\u8fd9\u91cc\u5207\u6362 "deap" \u6216 "hci"
FS = 128 if DATASET == "deap" else 256  # \u2605 \u81ea\u52a8\u9002\u914d\u91c7\u6837\u7387

BASE_DIR = r"."
DATA_NPZ = os.path.join(BASE_DIR, "data", DATASET, f"{DATASET}_30s_preprocessed_checked_labels.npz")

W_PATH   = os.path.join(BASE_DIR, "causality_gpu", DATASET, "frontal_weights.npy")
IDX_PATH = os.path.join(BASE_DIR, "causality_gpu", DATASET, "other_indices.npy")

OUT_DIR_MODELS = os.path.join(BASE_DIR, "pipeline_out", DATASET, "models")
OUT_DIR_DATA   = os.path.join(BASE_DIR, "pipeline_out", DATASET, "data")
os.makedirs(OUT_DIR_MODELS, exist_ok=True)
os.makedirs(OUT_DIR_DATA, exist_ok=True)

# ============================================================
#                    DSP UTILITIES
# ============================================================
def welch_psd(x, fs=FS):
    f, p = welch(x, fs=fs, nperseg=256)
    return f, p

def bandpass_filter(x, low, high, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, x)

def apply_fen_auto(x):
    B, C, T = x.shape
    x_np = x.detach().cpu().numpy()
    bands = [(1,4), (4,8), (8,12), (12,30)]
    out = np.zeros_like(x_np)
    for i in range(B):
        for c in range(C):
            raw = x_np[i, c]
            acc = np.zeros_like(raw)
            for (f1, f2) in bands:
                try:
                    acc += bandpass_filter(raw, f1, f2, fs=FS)
                except Exception:
                    acc += raw
            out[i, c] = acc
    return torch.tensor(out, dtype=torch.float32, device=device)

def align_length(a, b):
    T = min(a.shape[-1], b.shape[-1])
    return a[..., :T], b[..., :T]

def ema_update(target, source, beta=0.999):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(beta).add_(sp.data, alpha=1-beta)

# ============================================================
#                    LOSS FUNCTIONS
# ============================================================
def loss_tc(fake, real):
    f, r = align_length(fake, real)
    return F.l1_loss(f, r)

def loss_amp(fake, real):
    f, r = align_length(fake, real)
    return F.l1_loss(f.abs().mean(dim=-1), r.abs().mean(dim=-1))

def loss_welch(fake, real):
    f, r = align_length(fake, real)
    B, C, T = f.shape
    L = 0.0
    for i in range(B):
        for c in range(C):
            _, Pr = welch(r[i, c].detach().cpu().numpy(), fs=FS, nperseg=256)
            _, Pf = welch(f[i, c].detach().cpu().numpy(), fs=FS, nperseg=256)
            m = min(len(Pr), len(Pf))
            L += np.mean(np.abs(Pr[:m] - Pf[:m]))
    return torch.tensor(L / (B*C), dtype=torch.float32, device=device)

def loss_stft_s(fake, real):
    f, r = align_length(fake, real)
    B, C, T = f.shape
    win = torch.hann_window(64, device=device)
    L = 0.0
    for i in range(B):
        for c in range(C):
            F1 = torch.abs(torch.stft(r[i, c], n_fft=64, hop_length=32, window=win, return_complex=True))
            F2 = torch.abs(torch.stft(f[i, c], n_fft=64, hop_length=32, window=win, return_complex=True))
            m1, m2 = align_length(F1, F2)
            L += F.l1_loss(m1, m2)
    return L / (B*C)

def compute_gp(D, real, fake):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out = D(inter)
    grad = torch.autograd.grad(outputs=out, inputs=inter, grad_outputs=torch.ones_like(out),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((grad.view(B, -1).norm(2, dim=1) - 1.0) ** 2).mean() * 10.0

# ============================================================
#               MODELS 1: CIGAN (Ours & Ablations)
# ============================================================
class WaveNetResBlock(nn.Module):
    def __init__(self, channels, d):
        super().__init__()
        self.f = nn.Conv1d(channels, channels, 3, padding=d, dilation=d)
        self.g = nn.Conv1d(channels, channels, 3, padding=d, dilation=d)
        self.o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout1d(p=0.1)

    def forward(self, x):
        y = self.drop(torch.tanh(self.f(x)) * torch.sigmoid(self.g(x)))
        return x + self.o(y)

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
    def __init__(self, O=30, mode="full"):
        super().__init__()
        self.mode = mode
        self.encoder = UltraEncoder(in_ch=O)
        self.w_fc = nn.Linear(O, O)
        self.final = nn.Conv1d(4, 2, 1)

    def forward(self, others, w_vec):
        gen = self.encoder(others)
        if self.mode == "no_causal": w_vec = torch.ones_like(w_vec)
        w = self.w_fc(w_vec).unsqueeze(0).unsqueeze(-1) if self.mode != "no_causal" else w_vec.unsqueeze(0).unsqueeze(-1)
        fen_in = (others * w).mean(dim=1, keepdim=True).repeat(1, 2, 1)
        fen = apply_fen_auto(fen_in) if self.mode != "no_fen" else torch.zeros_like(gen)
        gen, fen = align_length(gen, fen)
        return self.final(torch.cat([gen, fen], dim=1))

class UltraDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, 2, 3), nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 7, 2, 3), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 7, 2, 3), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x): return self.fc(self.net(x).squeeze(-1))

# ============================================================
#               MODELS 2: DEEP BASELINES (Restored)
# ============================================================
class VanillaGenerator(nn.Module):
    def __init__(self, in_ch=30, out_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, out_ch, kernel_size=3, padding=1)
        )
    def forward(self, x): return self.net(x)

class VanillaDiscriminator(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, 1)
    def forward(self, x): return self.fc(self.net(x).squeeze(-1))

class hvEEGNetGenerator(nn.Module):
    def __init__(self, in_ch=30, out_ch=2):
        super().__init__()
        self.conv_short = nn.Conv1d(in_ch, 32, kernel_size=3, padding=1)
        self.conv_med = nn.Conv1d(in_ch, 32, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(in_ch, 32, kernel_size=11, padding=5)
        self.merge = nn.Sequential(
            nn.Conv1d(32 * 3, 64, kernel_size=1), nn.ELU(inplace=True),
            nn.Conv1d(64, out_ch, kernel_size=3, padding=1)
        )
    def forward(self, x):
        h1 = self.conv_short(x)
        h2 = self.conv_med(x)
        h3 = self.conv_long(x)
        return self.merge(torch.cat([h1, h2, h3], dim=1))

class TIEEEGNetGenerator(nn.Module):
    def __init__(self, in_ch=30, out_ch=2):
        super().__init__()
        self.spatial_conv = nn.Conv1d(in_ch, 64, kernel_size=1)
        self.temporal_conv = nn.Conv1d(64, 64, kernel_size=15, padding=7)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(64, 16, kernel_size=1),
            nn.ReLU(inplace=True), nn.Conv1d(16, 64, kernel_size=1), nn.Sigmoid()
        )
        self.out = nn.Conv1d(64, out_ch, kernel_size=3, padding=1)
    def forward(self, x):
        h = F.gelu(self.spatial_conv(x))
        h = F.gelu(self.temporal_conv(h))
        return self.out(h * self.attn(h))

# ============================================================
#               TRAINING ENGINES & DATA LOADERS
# ============================================================
def load_data():
    npz = np.load(DATA_NPZ)
    eeg = npz["eeg"]
    idx, w = np.load(IDX_PATH), np.load(W_PATH)
    others, frontal = eeg[:, idx, :], eeg[:, [0, 16], :]
    return (torch.tensor(others, dtype=torch.float32, device=device),
            torch.tensor(frontal, dtype=torch.float32, device=device),
            torch.tensor(w, dtype=torch.float32, device=device), others, frontal)

def train_and_generate_cigan(mode, others_t, frontal_t, w_vec, epochs=40, mc_passes=50):
    print(f"\n\U0001f680 Training CIGAN Variant: [{mode.upper()}] on {DATASET.upper()} ({FS} Hz)")
    G, D = UltraGenerator(mode=mode).to(device), UltraDisc().to(device)
    G_ema = UltraGenerator(mode=mode).to(device)
    G_ema.load_state_dict(G.state_dict())

    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    N = others_t.shape[0]

    # \u2605 \u6062\u590d ESWA Loss \u4fdd\u5b58\u65e5\u5fd7
    loss_log_eswa = {"G_loss": [], "D_loss": [], "L_gan": [], "L_tc": [], "L_amp": [], "L_psd": [], "L_stft": []}

    for ep in range(1, epochs + 1):
        perm = torch.randperm(N)
        others_shuf, frontal_shuf = others_t[perm], frontal_t[perm]

        G_sum, D_sum, gan_sum, tc_sum, amp_sum, psd_sum, stft_sum = 0, 0, 0, 0, 0, 0, 0
        step = 0

        for i in range(0, N, 32):
            o, r = others_shuf[i:i+32], frontal_shuf[i:i+32]
            if o.size(0) < 32: continue

            # Train D
            fake = G(o, w_vec).detach()
            fake, r = align_length(fake, r)
            loss_D = D(fake).mean() - D(r).mean() + compute_gp(D, r, fake)
            optD.zero_grad(); loss_D.backward(); optD.step()

            # Train G
            fake = G(o, w_vec)
            fake, r = align_length(fake, r)

            L_gan = -D(fake).mean()
            L_tc = loss_tc(fake, r)
            L_amp = loss_amp(fake, r)
            L_psd = loss_welch(fake, r) if mode != "no_psd" else torch.tensor(0.0)
            L_stft = loss_stft_s(fake, r) if mode != "no_psd" else torch.tensor(0.0)

            loss_G = L_gan + 5.0*L_tc + 3.0*L_amp + 1.0*L_psd + 1.0*L_stft

            optG.zero_grad(); loss_G.backward(); optG.step()
            ema_update(G_ema, G)

            # Accumulate logs
            step += 1
            G_sum += loss_G.item(); D_sum += loss_D.item()
            gan_sum += L_gan.item(); tc_sum += L_tc.item(); amp_sum += L_amp.item()
            psd_sum += L_psd.item() if mode != "no_psd" else 0
            stft_sum += L_stft.item() if mode != "no_psd" else 0

        # Save epoch logs
        loss_log_eswa["G_loss"].append(G_sum/step); loss_log_eswa["D_loss"].append(D_sum/step)
        loss_log_eswa["L_gan"].append(gan_sum/step); loss_log_eswa["L_tc"].append(tc_sum/step)
        loss_log_eswa["L_amp"].append(amp_sum/step); loss_log_eswa["L_psd"].append(psd_sum/step)
        loss_log_eswa["L_stft"].append(stft_sum/step)

        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch {ep}/{epochs} | Loss G: {G_sum/step:.3f} | Loss D: {D_sum/step:.3f}")

    name_map = {"full": "CIGAN_Full", "no_fen": "Ablation_NoFEN", "no_causal": "Ablation_NoCausal", "no_psd": "Ablation_NoPSD"}

    # \u2605 \u6062\u590d npz loss \u4fdd\u5b58\u903b\u8f91 (\u4ec5\u4e3a\u4e3b\u6a21\u578b\u4fdd\u5b58)
    if mode == "full":
        np.savez(os.path.join(OUT_DIR_MODELS, f"loss_log_eswa_{DATASET}.npz"), **{k: np.array(v) for k, v in loss_log_eswa.items()})

    G_ema.eval()
    for m in G_ema.modules():
        if m.__class__.__name__.startswith('Dropout'): m.train()

    fake_means, fake_stds = [], []
    with torch.no_grad():
        for i in tqdm(range(0, N, 32), desc="MC-Dropout Generating"):
            o = others_t[i:i+32]
            if o.size(0) == 0: continue
            mc_preds = torch.cat([align_length(G_ema(o, w_vec), frontal_t[0:1])[0].unsqueeze(0) for _ in range(mc_passes)], dim=0)
            fake_means.append(mc_preds.mean(dim=0).cpu().numpy())
            fake_stds.append(mc_preds.std(dim=0).cpu().numpy())

    np.savez(os.path.join(OUT_DIR_DATA, f"{name_map[mode]}_generated.npz"),
             fake_fp=np.concatenate(fake_means, axis=0), fake_std=np.concatenate(fake_stds, axis=0), real_fp=frontal_t.cpu().numpy())
    torch.save(G_ema.state_dict(), os.path.join(OUT_DIR_MODELS, f"{name_map[mode]}.pth"))

def train_and_generate_baseline(name, G, D, others_t, frontal_t, epochs=40):
    print(f"\n\U0001f680 Training Baseline: [{name}] on {DATASET.upper()} ({FS} Hz)")
    optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
    N = others_t.shape[0]

    for ep in range(1, epochs + 1):
        perm = torch.randperm(N)
        others_shuf, frontal_shuf = others_t[perm], frontal_t[perm]
        for i in range(0, N, 32):
            o, r = others_shuf[i:i+32], frontal_shuf[i:i+32]
            if o.size(0) < 32: continue

            fake = G(o).detach()
            fake, r = align_length(fake, r)
            loss_D = D(fake).mean() - D(r).mean() + compute_gp(D, r, fake)
            optD.zero_grad(); loss_D.backward(); optD.step()

            fake = align_length(G(o), r)[0]
            loss_G = -D(fake).mean()
            optG.zero_grad(); loss_G.backward(); optG.step()

    G.eval()
    fake_list = []
    with torch.no_grad():
        for i in range(0, N, 32):
            o = others_t[i:i+32]
            if o.size(0) > 0: fake_list.append(G(o).cpu().numpy())

    np.savez(os.path.join(OUT_DIR_DATA, f"{name}_generated.npz"),
             fake_fp=np.concatenate(fake_list, axis=0), real_fp=frontal_t.cpu().numpy())

def run_interpolations(others_np, frontal_np):
    print("\n\u23f3 Running Mathematical Interpolations...")
    N, _, T = others_np.shape
    f_lin, f_spl = np.zeros_like(frontal_np), np.zeros_like(frontal_np)
    x_old = np.arange(T)
    for i in tqdm(range(N), desc="Interpolating"):
        mean_sig = np.mean(others_np[i], axis=0)
        f_lin[i, 0], f_lin[i, 1] = mean_sig * 1.05, mean_sig * 0.95
        f_spl[i, 0], f_spl[i, 1] = CubicSpline(x_old, mean_sig * 1.02)(x_old), CubicSpline(x_old, mean_sig * 0.98)(x_old)

    np.savez(os.path.join(OUT_DIR_DATA, "Linear_Interp_generated.npz"), fake_fp=f_lin, real_fp=frontal_np)
    np.savez(os.path.join(OUT_DIR_DATA, "Spline_Interp_generated.npz"), fake_fp=f_spl, real_fp=frontal_np)

# ============================================================
#                      MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print(f"PIPELINE 1: UNIFIED TRAINING AND GENERATION ({DATASET.upper()})")
    print("="*60)

    o_t, f_t, w, o_np, f_np = load_data()

    # 1. CIGAN & Ablations (\u5305\u542b loss_log \u8bb0\u5f55)
    for mode in ["full", "no_fen", "no_causal", "no_psd"]:
        train_and_generate_cigan(mode, o_t, f_t, w, epochs=40)

    # 2. Vanilla WGAN Baseline
    train_and_generate_baseline("Vanilla_WGAN_GP", VanillaGenerator().to(device), VanillaDiscriminator().to(device), o_t, f_t, epochs=40)

    # 3. hvEEGNet Baseline (\u2605 \u6062\u590d)
    train_and_generate_baseline("hvEEGNet", hvEEGNetGenerator().to(device), VanillaDiscriminator().to(device), o_t, f_t, epochs=40)

    # 4. TIE-EEGNet Baseline (\u2605 \u6062\u590d)
    train_and_generate_baseline("TIE_EEGNet", TIEEEGNetGenerator().to(device), VanillaDiscriminator().to(device), o_t, f_t, epochs=40)

    # 5. Interpolations
    run_interpolations(o_np, f_np)

    print(f"\n\U0001f389 PIPELINE 1 COMPLETED for {DATASET.upper()}! All models trained and .npz data generated.")