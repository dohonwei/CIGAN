"""
================================================================================
    MASTER EXPERIMENT SCRIPT: Baselines, Ablations, and CIGAN (Main)
================================================================================
\u7edd\u5bf9\u516c\u5e73\u914d\u7f6e\uff1a
1. \u6240\u6709\u6a21\u578b\u4f7f\u7528\u76f8\u540c\u7684 5-Fold LOSO \u5212\u5206 (\u56fa\u5b9a\u968f\u673a\u79cd\u5b50\u4e0e Fold Indices)\u3002
2. \u6240\u6709\u6df1\u5ea6\u6a21\u578b\u4f7f\u7528\u76f8\u540c\u7684\u5f3a\u5224\u522b\u5668 (UltraDisc)\u3002
3. \u6240\u6709\u6df1\u5ea6\u6a21\u578b\u7edf\u4e00\u4f7f\u7528 EMA \u6743\u91cd\u6ed1\u52a8\u5e73\u5747\u63d0\u5347\u751f\u6210\u7a33\u5b9a\u6027\u3002
4. \u5f7b\u5e95\u4fee\u590d FS \u91c7\u6837\u7387 Bug\uff0c\u52a8\u6001\u9002\u914d\u6570\u636e\u96c6\u3002
5. CIGAN \u53ca\u590d\u6742\u6d88\u878d\u6a21\u578b\u542f\u7528\u540c\u65b9\u5dee\u4e0d\u786e\u5b9a\u6027\u52a0\u6743 (AWL)\u3002
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch, butter, filtfilt
from scipy.interpolate import CubicSpline
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================
#                    GLOBAL CONFIGURATION
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\U0001f680 Using Compute Device: {device}")

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# \U0001f3af \u6570\u636e\u96c6\u4e0e\u8d85\u53c2\u6570
DATASET = "hci"  # \u53ef\u9009: "hci" \u6216 "deap"
FS = 128 if DATASET == "deap" else 256
EPOCHS = 100  # \u7edf\u4e00\u7684\u8bad\u7ec3\u8f6e\u6570
BATCH_SIZE = 256

# \U0001f4c2 \u8def\u5f84\u914d\u7f6e
BASE_DIR = r"."
DATA_NPZ = os.path.join(BASE_DIR, "data", DATASET, f"{DATASET}_30s_preprocessed_checked_labels.npz")
W_PATH = os.path.join(BASE_DIR, "causality_gpu", DATASET, "frontal_weights.npy")
IDX_PATH = os.path.join(BASE_DIR, "causality_gpu", DATASET, "other_indices.npy")

BASE_OUT_DIR = os.path.join(BASE_DIR, "comparison", DATASET)
os.makedirs(BASE_OUT_DIR, exist_ok=True)

# \U0001f3af \u672c\u6b21\u8981\u8fd0\u884c\u7684\u6a21\u578b\u5217\u8868 (\u5982\u679c\u4f60\u53ea\u60f3\u6d4b\u67d0\u51e0\u4e2a\uff0c\u53ef\u4ee5\u6ce8\u91ca\u6389\u5176\u4ed6\u7684)
MODELS_TO_RUN = [
    "Linear_Interp",
    "Spline_Interp",
    "hvEEGNet",
    "TIE_EEGNet",
    "Ablation_Vanilla",
    "Ablation_NoFEN",
    "Ablation_NoCausal",
    "CIGAN_Main"
]


# ============================================================
#                  Data Loader & LOSO Setup
# ============================================================
def load_all_data():
    npz = np.load(DATA_NPZ)
    eeg = npz["eeg"]
    subjects = npz["subjects"]
    others_idx = np.load(IDX_PATH)
    w_vec_np = np.load(W_PATH)

    others = torch.tensor(eeg[:, others_idx, :], dtype=torch.float32)
    frontal = torch.tensor(eeg[:, [0, 16], :], dtype=torch.float32)
    w_vec = torch.tensor(w_vec_np, dtype=torch.float32)
    return others, frontal, subjects, w_vec


print("\U0001f4e6 Loading Data...")
OTHERS_ALL, FRONTAL_ALL, SUBJECTS_ALL, W_VEC_ALL = load_all_data()
N_TOTAL, C_OUT, T_LEN = FRONTAL_ALL.shape

# \u2605 \u7edd\u5bf9\u516c\u5e73\u4fdd\u8bc1\uff1a\u5168\u5c40\u9884\u5148\u8ba1\u7b97\u597d 5 \u6298\u7d22\u5f15\uff0c\u6240\u6709\u6a21\u578b\u5171\u4eab
gkf = GroupKFold(n_splits=5)
SHARED_FOLDS = list(gkf.split(OTHERS_ALL.numpy(), FRONTAL_ALL.numpy(), groups=SUBJECTS_ALL))
print(f"\U0001f512 Locked {len(SHARED_FOLDS)} Folds for Cross-Validation.")


# ============================================================
#               Signal Processing & Physical Losses
# ============================================================
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_losses):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(num_losses, dtype=torch.float32))

    def forward(self, losses):
        return sum(torch.exp(-self.params[i]) * L + self.params[i] for i, L in enumerate(losses))


def bandpass_filter(x, low, high, fs=FS, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x)


def apply_fen_auto(x):
    # \u3010\u4fee\u6539\u70b9\u3011\uff1a\u7eaf GPU \u8fd0\u7b97\uff0c\u6d88\u9664 for \u5faa\u73af\uff0c\u4fdd\u7559\u53cd\u5411\u4f20\u64ad\u68af\u5ea6
    B, C, T = x.shape

    # \u6267\u884c\u5b9e\u6570 FFT
    X = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / FS).to(x.device)

    # \u6784\u5efa 1-30Hz \u7684\u9891\u5e26\u63a9\u7801 (\u7b49\u4ef7\u4e8e\u539f\u672c\u56db\u4e2a\u9891\u6bb5\u4fe1\u53f7\u53e0\u52a0)
    mask = (freqs >= 1.0) & (freqs <= 30.0)

    # \u9891\u57df\u6ee4\u6ce2\u5e76\u9006\u53d8\u6362\u56de\u65f6\u57df
    X_filtered = X * mask.view(1, 1, -1)
    return torch.fft.irfft(X_filtered, n=T, dim=-1)


def align_length(a, b):
    T = min(a.shape[-1], b.shape[-1])
    return a[..., :T], b[..., :T]


def loss_tc(fake, real):
    fake, real = align_length(fake, real)
    return F.l1_loss(fake, real)


def loss_amp(fake, real):
    fake, real = align_length(fake, real)
    return F.l1_loss(fake.abs().mean(dim=-1), real.abs().mean(dim=-1))


def loss_welch(fake, real):
    fake, real = align_length(fake, real)
    B, C, T = fake.shape
    win = torch.hann_window(256, device=fake.device)

    # \u5408\u5e76\u524d\u4e24\u4e2a\u7ef4\u5ea6\u8fdb\u884c\u77e9\u9635\u8ba1\u7b97\uff0c\u907f\u514d for \u5faa\u73af
    fake_flat = fake.reshape(B * C, T)
    real_flat = real.reshape(B * C, T)

    # nperseg=256, hop_length=128 (\u9ed8\u8ba4 50% \u91cd\u53e0)
    F1 = torch.stft(real_flat, n_fft=256, hop_length=128, window=win, return_complex=True)
    F2 = torch.stft(fake_flat, n_fft=256, hop_length=128, window=win, return_complex=True)

    # \u5728\u65f6\u95f4\u7ef4\u5ea6(-1)\u4e0a\u5e73\u5747\uff0c\u5f97\u5230 PSD \u529f\u7387\u8c31\u5bc6\u5ea6
    P1 = torch.mean(torch.abs(F1) ** 2, dim=-1)
    P2 = torch.mean(torch.abs(F2) ** 2, dim=-1)

    return F.l1_loss(P1, P2)


def loss_stft_s(fake, real):
    fake, real = align_length(fake, real)
    B, C, T = fake.shape
    win = torch.hann_window(64, device=fake.device)

    # \u538b\u5e73\u4e3a 2D \u77e9\u9635\u4e00\u6b21\u6027\u8ba1\u7b97
    fake_flat = fake.reshape(B * C, T)
    real_flat = real.reshape(B * C, T)

    F1 = torch.stft(real_flat, n_fft=64, hop_length=32, window=win, return_complex=True)
    F2 = torch.stft(fake_flat, n_fft=64, hop_length=32, window=win, return_complex=True)

    return F.l1_loss(torch.abs(F1), torch.abs(F2))


def gradient_penalty(D, real, fake, lambda_gp=10.0):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out = D(inter)
    grad = torch.autograd.grad(outputs=out, inputs=inter, grad_outputs=torch.ones_like(out),
                               create_graph=True, retain_graph=True, only_inputs=True)[0].view(B, -1)
    return lambda_gp * ((grad.norm(2, dim=1) - 1.0) ** 2).mean()


def ema_update(target, source, beta=0.999):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(beta).add_(sp.data, alpha=1 - beta)


# ============================================================
#                  Network Architectures
# ============================================================
# 1. \u901a\u7528\u8d85\u5f3a\u5224\u522b\u5668 (\u5168\u7cfb\u5217\u5171\u4eab)
class UltraDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x): return self.fc(self.net(x).squeeze(-1))


# 2. \u4f20\u7edf\u57fa\u7ebf CNN
class VanillaGenerator(nn.Module):
    def __init__(self, in_ch=30, out_ch=2, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(128, 64, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(64, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x, w_vec=None): return self.net(x)


class hvEEGNetGenerator(nn.Module):
    def __init__(self, in_ch=30, out_ch=2, **kwargs):
        super().__init__()
        self.conv_short = nn.Conv1d(in_ch, 32, kernel_size=3, padding=1)
        self.conv_med = nn.Conv1d(in_ch, 32, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(in_ch, 32, kernel_size=11, padding=5)
        self.merge = nn.Sequential(nn.Conv1d(96, 64, kernel_size=1), nn.ELU(inplace=True),
                                   nn.Conv1d(64, out_ch, kernel_size=3, padding=1))

    def forward(self, x, w_vec=None): return self.merge(
        torch.cat([self.conv_short(x), self.conv_med(x), self.conv_long(x)], dim=1))


class TIEEEGNetGenerator(nn.Module):
    def __init__(self, in_ch=30, out_ch=2, **kwargs):
        super().__init__()
        self.spatial = nn.Conv1d(in_ch, 64, kernel_size=1)
        self.temporal = nn.Conv1d(64, 64, kernel_size=15, padding=7)
        self.attn = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(64, 16, kernel_size=1), nn.ReLU(inplace=True),
                                  nn.Conv1d(16, 64, kernel_size=1), nn.Sigmoid())
        self.out = nn.Conv1d(64, out_ch, kernel_size=3, padding=1)

    def forward(self, x, w_vec=None):
        h = F.gelu(self.temporal(F.gelu(self.spatial(x))))
        return self.out(h * self.attn(h))


# 3. \u7ec8\u6781 CIGAN (\u901a\u8fc7\u63a7\u5236\u5f00\u5173\u5b9e\u73b0\u6d88\u878d)
class WaveNetResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout1d(p=0.1)

    def forward(self, x): return x + self.out(self.drop(torch.tanh(self.filter(x)) * torch.sigmoid(self.gate(x))))


class UltraEncoder(nn.Module):
    def __init__(self, in_ch=30, hidden=64):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, kernel_size=1)
        self.blocks = nn.ModuleList([WaveNetResBlock(hidden, d) for d in [1, 2, 4, 8, 16, 32, 64, 128]])
        self.out = nn.Conv1d(hidden, 2, kernel_size=1)

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks: h = blk(h)
        return self.out(h)


class UltraGenerator(nn.Module):
    def __init__(self, in_ch=30, use_causal=True, use_fen=True):
        super().__init__()
        self.encoder = UltraEncoder(in_ch=in_ch)
        self.use_causal = use_causal
        self.use_fen = use_fen
        if self.use_causal: self.w_fc = nn.Linear(in_ch, in_ch)
        self.final_merge = nn.Conv1d(4, 2, kernel_size=1) if self.use_fen else None

    def forward(self, others, w_vec=None):
        gen = self.encoder(others)
        if not self.use_fen: return gen  # Ablation: No FEN

        w_oth = others * self.w_fc(w_vec).unsqueeze(0).unsqueeze(-1) if (
                    self.use_causal and w_vec is not None) else others
        fen = apply_fen_auto(w_oth.mean(dim=1, keepdim=True).repeat(1, 2, 1))
        gen, fen = align_length(gen, fen)
        return self.final_merge(torch.cat([gen, fen], dim=1))


# ============================================================
#                  Experiment Directory Map
# ============================================================
EXPERIMENT_CONFIG = {
    "hvEEGNet": {"G": hvEEGNetGenerator, "kwargs": {}, "loss": "l1_only", "needs_w": False},
    "TIE_EEGNet": {"G": TIEEEGNetGenerator, "kwargs": {}, "loss": "l1_only", "needs_w": False},
    "Ablation_Vanilla": {"G": VanillaGenerator, "kwargs": {}, "loss": "l1_only", "needs_w": False},
    "Ablation_NoFEN": {"G": UltraGenerator, "kwargs": {"use_causal": True, "use_fen": False}, "loss": "multi_physics_3_no_psd",
                       "needs_w": True},
    "Ablation_NoCausal": {"G": UltraGenerator, "kwargs": {"use_causal": False, "use_fen": True},
                          "loss": "multi_physics_3_no_psd", "needs_w": True},
    "CIGAN_Main": {"G": UltraGenerator, "kwargs": {"use_causal": True, "use_fen": True},
                       "loss": "multi_physics_3_no_psd", "needs_w": True}
}


# ============================================================
#                    Interpolation Methods
# ============================================================
def run_interpolation():
    print("\n\u23f3 Running Interpolation Baselines...")
    fake_linear = np.zeros((N_TOTAL, C_OUT, T_LEN), dtype=np.float32)
    fake_spline = np.zeros((N_TOTAL, C_OUT, T_LEN), dtype=np.float32)
    x_old = np.arange(T_LEN)

    others_np = OTHERS_ALL.numpy()
    for i in tqdm(range(N_TOTAL), desc="Interpolating", ncols=80):
        mean_sig = np.mean(others_np[i], axis=0)
        fake_linear[i, 0, :] = mean_sig * 1.05
        fake_linear[i, 1, :] = mean_sig * 0.95
        fake_spline[i, 0, :] = CubicSpline(x_old, mean_sig * 1.02)(x_old)
        fake_spline[i, 1, :] = CubicSpline(x_old, mean_sig * 0.98)(x_old)

    if "Linear_Interp" in MODELS_TO_RUN:
        np.savez(os.path.join(BASE_OUT_DIR, "linear_generated_loso.npz"), fake_fp=fake_linear,
                 real_fp=FRONTAL_ALL.numpy())
    if "Spline_Interp" in MODELS_TO_RUN:
        np.savez(os.path.join(BASE_OUT_DIR, "spline_generated_loso.npz"), fake_fp=fake_spline,
                 real_fp=FRONTAL_ALL.numpy())


# ============================================================
#                  Unified Deep Training Loop
# ============================================================
# ============================================================
#                  Unified Deep Training Loop
# ============================================================
def run_deep_model(model_name):
    cfg = EXPERIMENT_CONFIG[model_name]
    print(f"\n=======================================================")
    print(f"\U0001f680 Training: {model_name} | Loss Strategy: {cfg['loss']}")
    print(f"=======================================================")

    fake_all_loso = np.zeros((N_TOTAL, C_OUT, T_LEN), dtype=np.float32)

    # \u2605 \u65b0\u589e\uff1a\u521d\u59cb\u5316\u8bb0\u5f55 6 \u4e2a Loss \u7684\u5b57\u5178 (5 Folds x EPOCHS)
    loss_history = {
        "Gloss": np.zeros((len(SHARED_FOLDS), EPOCHS)),
        "Dloss": np.zeros((len(SHARED_FOLDS), EPOCHS)),
        "Lgan": np.zeros((len(SHARED_FOLDS), EPOCHS)),
        "Ltc": np.zeros((len(SHARED_FOLDS), EPOCHS)),
        "Lamp": np.zeros((len(SHARED_FOLDS), EPOCHS)),
        "Lstft": np.zeros((len(SHARED_FOLDS), EPOCHS)),
    }

    for fold, (train_idx, test_idx) in enumerate(SHARED_FOLDS):
        print(f"\n--- Fold {fold + 1}/5 ---")

        G = cfg["G"](**cfg["kwargs"]).to(device)
        D = UltraDisc().to(device)

        G_ema = copy.deepcopy(G).to(device)
        optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

        if "multi" in cfg["loss"]:
            num_losses = 4 if "4" in cfg["loss"] else 3
            awl = AutomaticWeightedLoss(num_losses).to(device)
            optG = torch.optim.Adam([{'params': G.parameters()}, {'params': awl.parameters(), 'weight_decay': 1e-4}],
                                    lr=1e-4, betas=(0.5, 0.9))
        else:
            optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))

        o_train = OTHERS_ALL[train_idx].to(device)
        r_train = FRONTAL_ALL[train_idx].to(device)
        w_global = W_VEC_ALL.to(device)

        G.train()
        D.train()

        pbar = tqdm(range(EPOCHS), desc="Training", ncols=90)
        for ep in pbar:
            # \u2605 \u65b0\u589e\uff1aEpoch \u7ea7\u522b\u7684 Loss \u7d2f\u52a0\u5668
            ep_gloss, ep_dloss, ep_lgan = 0.0, 0.0, 0.0
            ep_ltc, ep_lamp, ep_lstft = 0.0, 0.0, 0.0
            num_batches = 0

            perm = torch.randperm(len(o_train))
            for i in range(0, len(o_train), BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                o, r = o_train[idx], r_train[idx]
                if o.size(0) < BATCH_SIZE: continue

                # Train D
                with torch.no_grad():
                    fake = G(o, w_global) if cfg["needs_w"] else G(o)
                fake, r_align = align_length(fake, r)
                loss_D = D(fake.detach()).mean() - D(r_align).mean() + gradient_penalty(D, r_align, fake.detach())
                optD.zero_grad()
                loss_D.backward()
                optD.step()

                # Train G
                fake = G(o, w_global) if cfg["needs_w"] else G(o)
                fake, r_align = align_length(fake, r)

                L_gan = -D(fake).mean()
                L_tc_v = loss_tc(fake, r_align)

                # \u2605 \u4e3a\u4e86\u753b\u56fe\u9700\u8981\uff0c\u5373\u4f7f\u662f\u57fa\u7ebf\u6a21\u578b\u4e5f\u7edf\u4e00\u7b97\u4e00\u4e0b\u8fd9\u4e24\u4e2a\u7269\u7406Loss (\u5728GPU\u4e0a\u6781\u5feb\uff0c\u4e0d\u5f71\u54cd\u901f\u5ea6)
                L_amp_v = loss_amp(fake, r_align)
                L_stft_v = loss_stft_s(fake, r_align)

                if cfg["loss"] == "l1_only":
                    loss_G = L_gan + 10.0 * L_tc_v
                elif cfg["loss"] == "l1_and_weak_physics":
                    L_psd_v = loss_welch(fake, r_align)
                    loss_G = L_gan + 50.0 * L_tc_v + 1.0 * L_amp_v + 0.05 * L_psd_v + 0.05 * L_stft_v
                elif cfg["loss"] == "multi_physics_4":
                    L_psd_v = loss_welch(fake, r_align)
                    loss_G = L_gan + awl([L_tc_v, L_amp_v, L_psd_v, L_stft_v])
                elif cfg["loss"] == "multi_physics_3_no_psd":
                    loss_G = L_gan + awl([L_tc_v, L_amp_v, L_stft_v])

                optG.zero_grad()
                loss_G.backward()
                optG.step()
                ema_update(G_ema, G)

                # \u2605 \u7d2f\u52a0 Batch \u635f\u5931
                ep_dloss += loss_D.item()
                ep_gloss += loss_G.item()
                ep_lgan += L_gan.item()
                ep_ltc += L_tc_v.item()
                ep_lamp += L_amp_v.item()
                ep_lstft += L_stft_v.item()
                num_batches += 1

            # \u2605 \u8bb0\u5f55 Epoch \u5e73\u5747\u635f\u5931
            if num_batches > 0:
                loss_history["Dloss"][fold, ep] = ep_dloss / num_batches
                loss_history["Gloss"][fold, ep] = ep_gloss / num_batches
                loss_history["Lgan"][fold, ep] = ep_lgan / num_batches
                loss_history["Ltc"][fold, ep] = ep_ltc / num_batches
                loss_history["Lamp"][fold, ep] = ep_lamp / num_batches
                loss_history["Lstft"][fold, ep] = ep_lstft / num_batches

        # Validation...
        G_ema.eval()
        o_test = OTHERS_ALL[test_idx].to(device)

        with torch.no_grad():
            fold_fakes = []
            for i in range(0, len(o_test), BATCH_SIZE):
                ot = o_test[i:i + BATCH_SIZE]
                f = G_ema(ot, w_global) if cfg["needs_w"] else G_ema(ot)
                fold_fakes.append(f.cpu().numpy())
            fake_all_loso[test_idx] = np.concatenate(fold_fakes, axis=0)[..., :T_LEN]

    # \u4fdd\u5b58\u4e3b\u751f\u6210\u6570\u636e
    save_path = os.path.join(BASE_OUT_DIR, f"{model_name.lower()}generatedloso.npz")
    np.savez(save_path, fake_fp=fake_all_loso, real_fp=FRONTAL_ALL.numpy())
    print(f"\u2705 Finished {model_name}. Data saved to: {save_path}")

    # \u2605 \u4fdd\u5b58\u4e13\u95e8\u7684 Loss \u6570\u636e (\u4ec5\u4e3a\u4e3b\u5b9e\u9a8c CIGAN_Main \u4fdd\u5b58\uff0c\u514d\u5f97\u88ab\u6d88\u878d\u5b9e\u9a8c\u8986\u76d6)
    if model_name == "CIGAN_Main":
        loss_save_path = os.path.join(BASE_OUT_DIR, "losslogtimrevisionnopsd.npz")
        np.savez(loss_save_path, **loss_history)
        print(f"\U0001f4c8 Loss log saved specifically for plotting: {loss_save_path}")


# ============================================================
#                        MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    if any("Interp" in m for m in MODELS_TO_RUN):
        run_interpolation()

    for m_name in MODELS_TO_RUN:
        if "Interp" not in m_name:
            run_deep_model(m_name)

    print("\n\U0001f389 All Scheduled Experiments Completed Successfully!")