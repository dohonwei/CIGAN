# 1_train_cigan_eswa.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch, butter, filtfilt
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
# =====================================
# Loss logging for ESWA figures
# =====================================
loss_log = {
    "L_gan":  [],
    "L_tc":   [],
    "L_amp":  [],
    "L_psd":  [],
    "L_stft": [],
    "G_loss": [],
    "D_loss": []
}

# ============================================================
#                    DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
# ============================================================
#                    PATHS (\u6309\u9700\u4fee\u6539)
# ============================================================
# \u5df2\u7ecf D3+D4 \u5904\u7406\u540e\u7684 npz
DATASET = "hci"  # \u2605 \u5728\u8fd9\u91cc\u5207\u6362 "deap" \u6216 "hci"
FS = 128 if DATASET == "deap" else 256  # \u2605 \u81ea\u52a8\u9002\u914d\u91c7\u6837\u7387
BASE_DIR = r"."
DATA_NPZ = os.path.join(BASE_DIR, "data", DATASET, f"{DATASET}_30s_preprocessed_checked_labels.npz")
W_PATH   = os.path.join(BASE_DIR, "causality_gpu", DATASET, "frontal_weights.npy")
IDX_PATH = os.path.join(BASE_DIR, "causality_gpu", DATASET, "other_indices.npy")

SAVE_MODEL_G   = os.path.join(BASE_DIR, "comparison", DATASET, "CIGAN_G.pth")
SAVE_MODEL_D   = os.path.join(BASE_DIR, "comparison", DATASET, "CIGAN_D.pth")
SAVE_GEN_NPZ   = os.path.join(BASE_DIR, "comparison", DATASET, "CIGAN_generated_data.npz")
SAVE_LOSS_FIG  = os.path.join(BASE_DIR, "comparison", DATASET, "CIGAN_loss_curves.png")
os.makedirs(os.path.dirname(SAVE_MODEL_G), exist_ok=True)
os.makedirs(os.path.dirname(SAVE_GEN_NPZ), exist_ok=True)
os.makedirs(os.path.dirname(SAVE_LOSS_FIG), exist_ok=True)



# ============================================================
#                    Band-pass Filter + PSD
# ============================================================
def welch_psd(x, fs=128):
    f, p = welch(x, fs=fs, nperseg=256)
    return f, p

def bandpass_filter(x, low, high, fs=128, order=4):
    nyq = fs / 2
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x)

def apply_fen_auto(x, fs=FS):
    B, C, T = x.shape
    X_fft = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(T, d=1.0/fs).to(x.device)
    mask = ((freqs >= 1.0) & (freqs <= 30.0)).unsqueeze(0).unsqueeze(0).float()
    X_fft_filtered = X_fft * mask
    return torch.fft.irfft(X_fft_filtered, n=T, dim=-1)

# ============================================================
#                     Length Alignment
# ============================================================
def align_length(a, b):
    """
    force a and b to same T (last dimension)
    """
    Ta = a.shape[-1]
    Tb = b.shape[-1]
    if Ta == Tb:
        return a, b
    T = min(Ta, Tb)
    return a[..., :T], b[..., :T]

# ============================================================
#                     Losses
# ============================================================
def loss_tc(fake, real):
    fake, real = align_length(fake, real)
    return F.l1_loss(fake, real)

def loss_amp(fake, real):
    fake, real = align_length(fake, real)
    af = fake.abs().mean(dim=-1)
    ar = real.abs().mean(dim=-1)
    return F.l1_loss(af, ar)

def loss_welch(fake, real):
    fake, real = align_length(fake, real)
    B, C, T = fake.shape
    fs = FS
    L = 0.0
    for i in range(B):
        for c in range(C):
            r_np = real[i, c].detach().cpu().numpy()
            f_np = fake[i, c].detach().cpu().numpy()
            fr, Pr = welch(r_np, fs=fs, nperseg=256)
            ff, Pf = welch(f_np, fs=fs, nperseg=256)
            m = min(len(Pr), len(Pf))
            L += np.mean(np.abs(Pr[:m] - Pf[:m]))
    return torch.tensor(L / (B*C), dtype=torch.float32, device=fake.device)

def loss_stft_s(fake, real):
    fake, real = align_length(fake, real)
    B, C, T = fake.shape
    win = torch.hann_window(64, device=device)
    L = 0.0
    for i in range(B):
        for c in range(C):
            F1 = torch.stft(real[i, c], n_fft=64, hop_length=32,
                            window=win, return_complex=True)
            F2 = torch.stft(fake[i, c], n_fft=64, hop_length=32,
                            window=win, return_complex=True)
            m1 = torch.abs(F1)
            m2 = torch.abs(F2)
            m1, m2 = align_length(m1, m2)
            L += F.l1_loss(m1, m2)
    return L / (B*C)

# ============================================================
#          WaveNet Residual Block (Dilated Convolution)
# ============================================================
class WaveNetResBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.filter = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gate   = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.out    = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop   = nn.Dropout1d(p=0.1) # \u2605 \u65b0\u589e MC-Dropout

    def forward(self, x):
        f = torch.tanh(self.filter(x))
        g = torch.sigmoid(self.gate(x))
        y = self.drop(f * g) # \u2605 \u5728\u975e\u7ebf\u6027\u6fc0\u6d3b\u540e\u65bd\u52a0 Dropout
        y = self.out(y)
        return x + y

# ============================================================
#      Ultra-Pro WaveNet Temporal Encoder (30\u21922 channels)
# ============================================================
class UltraEncoder(nn.Module):
    def __init__(self, in_ch=30, hidden=128):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, kernel_size=1)
        dil_list = [1, 2, 4, 8, 16, 32, 64, 128]
        self.blocks = nn.ModuleList([
            WaveNetResBlock(hidden, d) for d in dil_list
        ])
        self.out = nn.Conv1d(hidden, 2, kernel_size=1)

    def forward(self, x):
        # x: (B,30,T)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        y = self.out(h)    # (B,2,T)
        return y

# ============================================================
#         Ultra-Pro Generator (WaveNet + FEN fusion)
# ============================================================
class UltraGenerator(nn.Module):
    def __init__(self, O):
        super().__init__()
        self.encoder = UltraEncoder(in_ch=O)
        self.w_fc = nn.Linear(O, O)
        self.final_merge = nn.Conv1d(2+2, 2, kernel_size=1)

    def forward(self, others, w_vec):
        """
        others : (B,30,T)
        w_vec  : (30,) frontal causal weights
        """
        B, C, T = others.shape
        gen = self.encoder(others)             # (B,2,T)

        # Weighted frontal cue (w\xb7others)
        w = self.w_fc(w_vec).unsqueeze(0).unsqueeze(-1)  # (1,30,1)
        w_oth = others * w                        # (B,30,T)
        fen_in = w_oth.mean(dim=1, keepdim=True)  # (B,1,T)
        fen_in = fen_in.repeat(1, 2, 1)           # (B,2,T)

        # FEN filter
        fen = apply_fen_auto(fen_in)              # (B,2,T)

        # Merge
        gen, fen = align_length(gen, fen)
        out = self.final_merge(torch.cat([gen, fen], dim=1))
        return out

# ============================================================
#                   Discriminator (1D CNN)
# ============================================================
class UltraDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B,2,T)
        h = self.net(x)          # (B,128,1)
        h = h.squeeze(-1)        # (B,128)
        out = self.fc(h)         # (B,1)
        return out

# ============================================================
#                   Gradient Penalty
# ============================================================
def gradient_penalty(D, real, fake, lambda_gp=10.0):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    inter = alpha * real + (1 - alpha) * fake
    inter.requires_grad_(True)
    out = D(inter)
    grad = torch.autograd.grad(
        outputs=out,
        inputs=inter,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad = grad.view(B, -1)
    gp = ((grad.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp

# ============================================================
#                        EMA Helper
# ============================================================
def ema_update(target, source, beta=0.999):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(beta).add_(sp.data, alpha=1-beta)

# ============================================================
#                    Data Loader helper
# ============================================================
def load_hci_for_cigan():
    npz = np.load(DATA_NPZ)
    eeg = npz["eeg"]           # (N,32,T)
    subjects = npz["subjects"] # (N,)
    others_idx = np.load(IDX_PATH)
    w_vec_np   = np.load(W_PATH)

    others = eeg[:, others_idx, :]     # (N,30,T)
    frontal = eeg[:, [0,16], :]       # (N,2,T)

    others = torch.tensor(others, dtype=torch.float32, device=device)
    frontal = torch.tensor(frontal, dtype=torch.float32, device=device)
    w_vec   = torch.tensor(w_vec_np, dtype=torch.float32, device=device)

    return others, frontal, w_vec

# ============================================================
#                        Training
# ============================================================

class AutomaticWeightedLoss(nn.Module):
    """
    \u57fa\u4e8e\u540c\u65b9\u5dee\u4e0d\u786e\u5b9a\u6027\u7684\u52a8\u6001\u6743\u91cd\u8c03\u6574\u6a21\u5757
    \u9488\u5bf9 4 \u4e2a\u7269\u7406\u91cd\u5efa Loss (TC, AMP, PSD, STFT)
    """
    def __init__(self, num_losses=4):
        super(AutomaticWeightedLoss, self).__init__()
        # \u521d\u59cb\u5316\u4e3a 0\uff0c\u5373\u521d\u59cb\u6743\u91cd exp(0) = 1
        self.params = nn.Parameter(torch.zeros(num_losses, dtype=torch.float32))

    def forward(self, losses):
        """
        losses: \u5305\u542b\u5404\u9879\u539f\u59cb loss \u6807\u91cf\u7684 list \u6216 tensor
        """
        total_loss = 0.0
        for i, L in enumerate(losses):
            # exp(-s) * L + s
            total_loss += torch.exp(-self.params[i]) * L + self.params[i]
        return total_loss



def train_cigan_eswa(epochs=100, batch_size=256, val_ratio=0.2):
    # 1. \u52a0\u8f7d\u6570\u636e
    others, frontal, w_vec = load_hci_for_cigan()

    # ==================== Subject-wise Validation Split ====================
    subjects = np.load(DATA_NPZ)['subjects']
    unique_subjects = np.unique(subjects)
    n_val = max(1, int(len(unique_subjects) * val_ratio))
    val_subs = np.random.choice(unique_subjects, n_val, replace=False)

    val_mask = np.isin(subjects, val_subs)
    others_train = others[~val_mask]
    frontal_train = frontal[~val_mask]
    others_val = others[val_mask]
    frontal_val = frontal[val_mask]

    print(f"Training subjects: {len(unique_subjects) - n_val} | Val subjects: {n_val}")

    O = others.shape[1]
    G = UltraGenerator(O).to(device)
    D = UltraDisc().to(device)
    G_ema = UltraGenerator(O).to(device)
    G_ema.load_state_dict(G.state_dict())

    # ==================== Optimizer & AWL ====================
    awl = AutomaticWeightedLoss(num_losses=4).to(device)

    optG = torch.optim.Adam([
        {'params': G.parameters(), 'lr': 1e-4},
        {'params': awl.parameters(), 'lr': 1e-3, 'weight_decay': 1e-5}
    ], betas=(0.5, 0.9))

    optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

    # \u5b66\u4e60\u7387\u8c03\u5ea6
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optG, T_max=epochs, eta_min=1e-6)

    best_pearson = -1.0
    best_epoch = 0
    best_path = os.path.join(os.path.dirname(SAVE_MODEL_G), "CIGAN_G_ema_best.pth")

    loss_history = {"G": [], "D": [], "L_gan": [], "L_tc": [], "L_amp": [], "L_psd": [], "L_stft": [],
                    "Val_Pearson": []}

    # 4. \u5f00\u59cb Epoch \u5faa\u73af
    for ep in range(1, epochs + 1):
        # \u6253\u4e71\u8bad\u7ec3\u96c6
        perm = torch.randperm(len(others_train))
        o_train_shuf = others_train[perm]
        r_train_shuf = frontal_train[perm]

        G.train()
        D.train()

        G_loss_sum = D_loss_sum = gan_sum = tc_sum = amp_sum = psd_sum = stft_sum = 0.0
        step = 0

        pbar = tqdm(range(0, len(o_train_shuf), batch_size), desc=f"Epoch {ep}/{epochs}", ncols=100)
        for i in pbar:
            o = o_train_shuf[i:i + batch_size]
            r = r_train_shuf[i:i + batch_size]
            if o.size(0) < batch_size: continue

            # ----------------- Train D -----------------
            with torch.no_grad():
                fake = G(o, w_vec)
            fake, r_aligned = align_length(fake, r)
            fake_det = fake.detach()

            D_real = D(r_aligned)
            D_fake = D(fake_det)
            gp = gradient_penalty(D, r_aligned, fake_det)
            loss_D = D_fake.mean() - D_real.mean() + gp

            optD.zero_grad()
            loss_D.backward()
            optD.step()

            # ----------------- Train G -----------------
            fake = G(o, w_vec)
            fake, r_aligned = align_length(fake, r)

            D_fake = D(fake)
            L_gan = -D_fake.mean()

            L_tc = loss_tc(fake, r_aligned)
            L_amp = loss_amp(fake, r_aligned)
            L_psd = loss_welch(fake, r_aligned)
            L_stft = loss_stft_s(fake, r_aligned)

            # \u2605 \u4fee\u6539\uff1a\u8ba1\u7b97\u52a8\u6001\u52a0\u6743\u7684\u7269\u7406 Loss
            # \u5bf9\u6297\u635f\u5931 L_gan \u4f5c\u4e3a\u4e3b\u7ebf\u4efb\u52a1\u4e0d\u53c2\u4e0e\u52a0\u6743\uff0c\u7269\u7406\u7ea6\u675f\u4f5c\u4e3a\u8f85\u52a9\u4efb\u52a1\u8fdb\u884c\u4e0d\u786e\u5b9a\u6027\u52a0\u6743
            physical_losses = [L_tc, L_amp, L_psd, L_stft]
            weighted_physical_loss = awl(physical_losses)

            loss_G = L_gan + weighted_physical_loss

            optG.zero_grad()
            loss_G.backward()
            optG.step()

            # \u66f4\u65b0 EMA \u6a21\u578b
            ema_update(G_ema, G)

            # ======== \u7d2f\u8ba1 Loss ========
            G_loss_sum += loss_G.item();
            D_loss_sum += loss_D.item()
            gan_sum += L_gan.item();
            tc_sum += L_tc.item()
            amp_sum += L_amp.item();
            psd_sum += L_psd.item()
            stft_sum += L_stft.item()
            step += 1

            pbar.set_postfix(G=f"{G_loss_sum / step:.3f}", D=f"{D_loss_sum / step:.3f}")


        # ====================================================
        # \u2605 \u65b0\u589e\uff1aValidation \u9a8c\u8bc1\u5faa\u73af (\u57fa\u4e8e G_ema)
        # ====================================================
        G_ema.eval()
        val_pearson_list = []
        with torch.no_grad():
            for i in range(0, len(others_val), batch_size):
                o_v = others_val[i:i + batch_size]
                r_v = frontal_val[i:i + batch_size]
                if o_v.size(0) == 0: continue

                fake_v = G_ema(o_v, w_vec)
                fake_v, r_v = align_length(fake_v, r_v)

                # \u4f7f\u7528 scipy \u8ba1\u7b97\u771f\u5b9e Pearson\uff08\u66f4\u53ef\u9760\uff09
                for c in range(2):  # Fp1 \u548c Fp2 \u5206\u522b\u7b97
                    p = pearsonr(r_v[:, c].cpu().numpy().flatten(),
                                 fake_v[:, c].cpu().numpy().flatten())[0]
                    val_pearson_list.append(p)

        mean_val_pearson = np.mean(val_pearson_list)

        # \u5224\u65ad\u5e76\u4fdd\u5b58\u6700\u4f73\u6a21\u578b
        if mean_val_pearson > best_pearson:
            best_pearson = mean_val_pearson
            best_epoch = ep
            torch.save(G_ema.state_dict(), best_path)
            print(f"  \U0001f31f [New Best] Val Pearson: {best_pearson:.4f} @ Epoch {best_epoch} (Saved!)")
        else:
            print(f"  - Val Pearson: {mean_val_pearson:.4f} (Best was {best_pearson:.4f} @ Epoch {best_epoch})")

    # ====================================================
    # \u2605 \u8bad\u7ec3\u7ed3\u675f\uff1a\u52a0\u8f7d Best Checkpoint \u5e76\u8986\u76d6\u4fdd\u5b58
    # ====================================================
    print(f"\n\U0001f389 \u8bad\u7ec3\u7ed3\u675f! \u52a0\u8f7d Epoch {best_epoch} \u7684\u6700\u4f73\u6743\u91cd...")
    G_ema.load_state_dict(torch.load(best_path))

    # \u5c06\u6700\u597d\u7684\u6743\u91cd\u6b63\u5f0f\u4fdd\u5b58\u4e3a\u6700\u7ec8\u6a21\u578b
    torch.save(G_ema.state_dict(), SAVE_MODEL_G)
    torch.save(D.state_dict(), SAVE_MODEL_D)
    print(f"\u2705 \u6700\u7ec8 G_ema \u5df2\u66f4\u65b0\u5e76\u4fdd\u5b58\u81f3: {SAVE_MODEL_G}")

    # ====================================================
    # \u4fdd\u5b58 loss_log_eswa.npz\uff08\u7528\u4e8e Fig.2 ESWA \u98ce\u683c\u56fe\uff09
    # ====================================================


    # ====================================================
    # \u7ed8\u56fe\u903b\u8f91
    # ====================================================
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history["G"], label="G")
    plt.plot(loss_history["D"], label="D")
    plt.legend();
    plt.xlabel("Epoch");
    plt.ylabel("Loss")
    plt.title("Generator / Discriminator Loss")
    plt.tight_layout()
    plt.savefig(SAVE_LOSS_FIG.replace(".png", "_GD.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for k in ["L_gan", "L_tc", "L_amp", "L_psd", "L_stft"]:
        plt.plot(loss_history[k], label=k)
    plt.legend();
    plt.xlabel("Epoch");
    plt.ylabel("Loss")
    plt.title("Loss Components")
    plt.tight_layout()
    plt.savefig(SAVE_LOSS_FIG, dpi=160)
    plt.close()

    # \u5355\u72ec\u753b\u4e00\u5f20 Pearson \u7684\u8d8b\u52bf\u56fe
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history["Val_Pearson"], label="Val Pearson", color='green', marker='o', markersize=4)
    plt.axvline(x=best_epoch - 1, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.legend();
    plt.xlabel("Epoch");
    plt.ylabel("Pearson Correlation")
    plt.title("Validation Pearson over Epochs")
    plt.tight_layout()
    plt.savefig(SAVE_LOSS_FIG.replace(".png", "_Pearson.png"), dpi=160)
    plt.close()

    return G_ema

# ============================================================
#            \u4f7f\u7528 EMA \u751f\u6210 HCI \u524d\u989d EEG
# ============================================================
# ============================================================
#       \u4f7f\u7528 MC-Dropout (EMA) \u751f\u6210 HCI \u524d\u989d EEG \u53ca\u4e0d\u786e\u5b9a\u6027\u91cf\u5316
# ============================================================
def generate_cigan_inference(G_ema, mode='single', num_passes=50):
    """
    \u7edf\u4e00\u7684 CIGAN \u63a8\u7406\u51fd\u6570\uff1a\u540c\u65f6\u652f\u6301 Single \u548c MC50 \u6a21\u5f0f
    :param G_ema: \u8bad\u7ec3\u597d\u7684\u751f\u6210\u5668\u6a21\u578b
    :param mode: 'single' (\u5355\u6b21\u786e\u5b9a\u6027\u524d\u5411) \u6216 'mc50' (\u8499\u7279\u5361\u6d1b Dropout \u524d\u5411)
    :param num_passes: MC \u6a21\u5f0f\u4e0b\u7684\u91c7\u6837\u6b21\u6570 (\u9ed8\u8ba4\u4e3a 50)
    """
    print(f"\n{'=' * 50}")
    print(f"\U0001f680 \u5f00\u59cb CIGAN \u63a8\u7406\u6d41\u7a0b | \u5f53\u524d\u6a21\u5f0f: [{mode.upper()}]")
    print(f"{'=' * 50}")

    npz = np.load(DATA_NPZ)
    eeg = npz["eeg"]  # (N,32,T)
    N, C, T = eeg.shape

    others_idx = np.load(IDX_PATH)
    w_vec_np = np.load(W_PATH)

    others = eeg[:, others_idx, :]  # (N,30,T)
    frontal = eeg[:, [0, 16], :]  # (N,2,T)

    others = torch.tensor(others, dtype=torch.float32, device=device)
    w_vec = torch.tensor(w_vec_np, dtype=torch.float32, device=device)

    # 1. \u57fa\u7840\u8bbe\u7f6e\uff1a\u9996\u5148\u5168\u90fd\u5207\u6362\u5230 evaluation \u6a21\u5f0f (\u56fa\u5b9a BatchNorm)
    G_ema.eval()

    # 2. \u2605 \u6838\u5fc3\u903b\u8f91\u5206\u6d41 \u2605
    if mode == 'mc50':
        print(f"\u2699\ufe0f MC-Dropout \u5df2\u6fc0\u6d3b: \u5f3a\u5236\u5f00\u542f Dropout \u5c42, \u5c06\u91c7\u6837 {num_passes} \u6b21...")
        for m in G_ema.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    elif mode == 'single':
        print("\u2699\ufe0f Single \u6a21\u5f0f\u5df2\u6fc0\u6d3b: \u7eaf\u51c0 Eval \u72b6\u6001, \u5355\u6b21\u786e\u5b9a\u6027\u63a8\u7406...")
        num_passes = 1  # \u5f3a\u5236\u6539\u4e3a\u5355\u6b21\u524d\u5411
    else:
        raise ValueError("mode \u53c2\u6570\u5fc5\u987b\u662f 'single' \u6216 'mc50'")

    fake_mean_list = []
    fake_std_list = []

    with torch.no_grad():
        for i in tqdm(range(0, N, 32), desc=f"Generating"):
            o = others[i:i + 32]
            if o.size(0) == 0:
                continue

            mc_preds = []
            # 3. \u524d\u5411\u4f20\u64ad\u5faa\u73af
            for _ in range(num_passes):
                fake = G_ema(o, w_vec)  # (b,2,Tf)
                fake = fake[:, :, :T]  # \u622a\u65ad\u5bf9\u9f50\u957f\u5ea6
                mc_preds.append(fake.unsqueeze(0))  # (1, b, 2, T)

            mc_preds = torch.cat(mc_preds, dim=0)  # (num_passes, b, 2, T)

            # 4. \u7edf\u8ba1\u5904\u7406\u5206\u6d41
            if mode == 'mc50':
                # \u8ba1\u7b97\u5747\u503c\u548c\u6807\u51c6\u5dee
                mean_pred = mc_preds.mean(dim=0)
                std_pred = mc_preds.std(dim=0)
            else:  # single \u6a21\u5f0f
                # \u76f4\u63a5\u53d6\u552f\u4e00\u7684\u4e00\u6b21\u7ed3\u679c\uff0c\u6807\u51c6\u5dee\u586b\u96f6\uff08\u9632\u6b62\u4e0b\u6e38\u753b\u56fe\u4ee3\u7801\u62a5\u9519\uff09
                mean_pred = mc_preds.squeeze(0)
                std_pred = torch.zeros_like(mean_pred)

            fake_mean_list.append(mean_pred.cpu().numpy())
            fake_std_list.append(std_pred.cpu().numpy())

    # 5. \u62fc\u63a5\u5168\u90e8\u7ed3\u679c
    fake_all_mean = np.concatenate(fake_mean_list, axis=0)  # (N,2,T)
    fake_all_std = np.concatenate(fake_std_list, axis=0)  # (N,2,T)
    real_all = frontal  # (N,2,T) Ground Truth

    # 6. \u2605 \u52a8\u6001\u4fdd\u5b58\u6587\u4ef6\u540d\uff0c\u9632\u6b62 single \u548c mc50 \u4e92\u76f8\u8986\u76d6 \u2605
    # \u5047\u8bbe\u4f60\u7684 SAVE_GEN_NPZ \u53eb 'hci_generated_eswa.npz'
    save_name = SAVE_GEN_NPZ.replace('.npz', f'_{mode}.npz')

    np.savez(
        save_name,
        fake_fp=fake_all_mean,
        fake_std=fake_all_std,
        real_fp=real_all
    )

    print(f"\u2705 \u63a8\u7406\u5b8c\u6210\uff01\u7ed3\u679c\u5df2\u6210\u529f\u4fdd\u5b58\u81f3: {save_name}\n")
    return fake_all_mean, fake_all_std, real_all


# ==============================================================
#                      \u4e3b\u7a0b\u5e8f\u5165\u53e3
# ==============================================================
if __name__ == "__main__":
    # \u5982\u679c\u5df2\u7ecf\u8bad\u7ec3\u597d\u4e86\uff0c\u53ef\u4ee5\u76f4\u63a5 load \u6a21\u578b\uff0c\u4e0d\u7528\u6bcf\u6b21\u90fd train
    # G_best = UltraGenerator(O).to(device)
    # G_best.load_state_dict(torch.load(best_path))
    # print(f"\u2705 Loaded Best Checkpoint for inference: {best_path}")

    # \u5047\u8bbe\u8fd9\u662f\u4f60\u7684\u8bad\u7ec3\u6d41\u7a0b
    G_best = train_cigan_eswa(epochs=100, batch_size=128)

    # ---------------------------------------------------------
    # \U0001f4a1 \u5b8c\u7f8e\u6d41\u6c34\u7ebf\uff1a\u8bad\u7ec3\u5b8c\u540e\uff0c\u4e00\u53e3\u6c14\u628a\u4e24\u4e2a\u7248\u672c\u7684\u6570\u636e\u90fd\u751f\u6210\u51fa\u6765\uff01
    # ---------------------------------------------------------

    # 1. \u751f\u6210 Single \u7248\u672c\uff1a\u7528\u4e8e\u8dd1 DTW, MMD, \u9891\u6bb5\u7279\u5f81\uff0c\u4ee5\u53ca\u5206\u7c7b\u8868\u683c
    generate_cigan_inference(G_best, mode='single')

    # 2. \u751f\u6210 MC50 \u7248\u672c\uff1a\u4e13\u4f9b\u753b\u4e0d\u786e\u5b9a\u6027\u533a\u95f4\u56fe (Fig 9)\uff0c\u4ee5\u53ca\u7f6e\u4fe1\u5ea6\u8ba1\u7b97
    # generate_cigan_inference(G_best, mode='mc50', num_passes=50)
