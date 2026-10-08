import os
import json
import csv
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import f, spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. GPU \u8bbe\u7f6e
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ============================================
# 2. \u6838\u5fc3\u6570\u5b66\u51fd\u6570 (Yule-Walker \u534f\u65b9\u5dee\u77e9\u9635\u6cd5)
# ============================================
def build_yule_walker_covariance(eeg_tensor, p, batch_size=128):
    """
    \u3010\u5ba1\u7a3f\u4eba\u6ee1\u5206\u7ea7\u4f18\u5316\u3011\u53ea\u9700\u904d\u5386\u4e00\u6b21\u6570\u636e\uff0c\u6784\u5efa\u5168\u5c40 Yule-Walker \u5757\u6258\u666e\u5229\u8328\u534f\u65b9\u5dee\u77e9\u9635 R
    \u6781\u5927\u964d\u4f4e\u663e\u5b58\u5360\u7528\uff0c\u901f\u5ea6\u63d0\u5347\u767e\u500d\uff0c\u5f7b\u5e95\u89e3\u51b3\u591a\u53d8\u91cf\u5197\u4f59\u5d29\u584c\u3002
    """
    N, C, T = eeg_tensor.shape
    num_vars = C * (p + 1)
    work_dtype = torch.float64

    # R \u77e9\u9635\u5305\u542b\u6240\u6709\u901a\u9053\u7684 0 \u9636\u5230 p \u9636\u81ea\u76f8\u5173\u4e0e\u4e92\u76f8\u5173
    R = torch.zeros((num_vars, num_vars), device=device, dtype=work_dtype)
    K_total = int(N * (T - p))

    print("\u6784\u5efa\u5168\u5c40 Yule-Walker \u534f\u65b9\u5dee\u77e9\u9635...")
    for start_idx in tqdm(range(0, N, batch_size)):
        end_idx = min(start_idx + batch_size, N)
        batch_eeg = eeg_tensor[start_idx:end_idx].to(device=device, dtype=work_dtype)

        # \u6784\u5efa\u5f53\u524d\u6279\u6b21\u7684 Z \u77e9\u9635: [Y, X_lag1, X_lag2, ..., X_lagp]
        Y_batch = batch_eeg[:, :, p:]  # (batch, C, T-p)
        X_lags = [batch_eeg[:, :, p - lag: T - lag] for lag in range(1, p + 1)]

        # \u62fc\u63a5\u5e76\u5728\u6279\u6b21\u548c\u65f6\u95f4\u7ef4\u5ea6\u4e0a\u5c55\u5e73
        Z_batch = torch.cat([Y_batch] + X_lags, dim=1)  # (batch, C*(p+1), T-p)
        Z_batch_flat = Z_batch.transpose(1, 2).reshape(-1, num_vars).t()  # (C*(p+1), batch*(T-p))

        # \u7d2f\u52a0\u534f\u65b9\u5dee (\u5916\u79ef)
        R += Z_batch_flat @ Z_batch_flat.t()

        del batch_eeg, Y_batch, X_lags, Z_batch, Z_batch_flat
        torch.cuda.empty_cache()

    return R, K_total


def fdr_bh_correction(pval_mat, alpha=0.05):
    """FDR-BH \u9519\u8bef\u7387\u63a7\u5236\u6821\u6b63"""
    orig_shape = pval_mat.shape
    p_flat = pval_mat.flatten()
    m = len(p_flat)
    sort_idx = np.argsort(p_flat)
    p_sorted = p_flat[sort_idx]
    v = np.arange(1, m + 1)
    line = (v / m) * alpha
    under_line = p_sorted <= line
    max_sig_idx = np.max(np.where(under_line)[0]) if np.any(under_line) else -1

    sig_mask_flat = np.zeros(m, dtype=bool)
    if max_sig_idx >= 0:
        sig_mask_flat[sort_idx[:max_sig_idx + 1]] = True

    p_adj_flat = np.zeros(m)
    prev_p = 1.0
    for idx in reversed(range(m)):
        curr_p = p_sorted[idx] * (m / (idx + 1))
        curr_p = min(prev_p, curr_p)
        p_adj_flat[sort_idx[idx]] = curr_p
        prev_p = curr_p

    p_adj_flat = np.clip(p_adj_flat, 0.0, 1.0)
    return p_adj_flat.reshape(orig_shape), sig_mask_flat.reshape(orig_shape)


def summarize_channel_scores(scores, weights, othernames, topk=10):
    """\u751f\u6210\u6392\u5e8f\u6458\u8981\uff0c\u4fbf\u4e8e\u5199 json / csv / rebuttal\u3002"""
    ranking = []
    for i, ch in enumerate(othernames):
        ranking.append({
            "sourcechannel": ch,
            "rawscore": float(scores[i]),
            "weight": float(weights[i])
        })
    ranking = sorted(ranking, key=lambda x: x["weight"], reverse=True)
    return ranking[:topk], ranking


def save_channel_ranking_csv(csvpath, othernames, rawscores, weights,
                             sigmask=None, pvalfdr=None, frontalnames=None):
    """\u4fdd\u5b58\u901a\u9053\u6392\u5e8f\u8868\uff0c\u65b9\u4fbf\u8bba\u6587\u548c rebuttal \u76f4\u63a5\u5f15\u7528\u3002"""
    rows = []
    for i, ch in enumerate(othernames):
        row = {
            "sourcechannel": ch,
            "rawscore": float(rawscores[i]),
            "weight": float(weights[i]),
        }
        if sigmask is not None and frontalnames is not None:
            for j, tgt in enumerate(frontalnames):
                row[f"sigto{tgt}"] = int(sigmask[i, j])
        if pvalfdr is not None and frontalnames is not None:
            for j, tgt in enumerate(frontalnames):
                row[f"pfdrto{tgt}"] = float(pvalfdr[i, j])
        rows.append(row)

    rows = sorted(rows, key=lambda x: x["weight"], reverse=True)

    fieldnames = list(rows[0].keys()) if rows else ["sourcechannel", "rawscore", "weight"]
    with open(csvpath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(jsonpath, summarydict):
    with open(jsonpath, "w", encoding="utf-8") as f:
        json.dump(summarydict, f, indent=2, ensure_ascii=False)


def normalize_weights(s_i):
    """\u5e26\u6e29\u5ea6\u7cfb\u6570\u7684 Softmax"""
    s_i = np.asarray(s_i, dtype=np.float64)
    s_i = np.maximum(s_i, 0.0)
    total = s_i.sum()
    if total <= 0:
        # \u4fee\u590d\uff1anp.ones_like \u6f0f\u4e86\u4e0b\u5212\u7ebf
        return np.ones_like(s_i, dtype=np.float64) / len(s_i)
    return s_i / total


# \u4fee\u590d\uff1a\u53d6\u6d88\u4e86\u591a\u4f59\u7684\u7f29\u8fdb
def compute_causal_prior_from_eeg(
        eeg_processed,
        channelnames,
        sfreq,
        bestp,
        alphafdr=0.05,
        frontalindices=(0, 16),
):
    """
    \u4ece\u9884\u5904\u7406\u540e\u7684 EEG \u6570\u636e\u76f4\u63a5\u8ba1\u7b97 frontal causal prior\u3002
    \u8fd4\u56de\u6240\u6709\u5173\u952e\u4e2d\u95f4\u7ed3\u679c\uff0c\u4fbf\u4e8e\u4e3b\u6d41\u7a0b\u5b58\u76d8\u548c\u540e\u7eed\u7a33\u5b9a\u6027\u5206\u6790\u590d\u7528\u3002
    """
    frontalindices = list(frontalindices)
    otherindices = [i for i in range(len(channelnames)) if i not in frontalindices]
    frontalnames = [channelnames[i] for i in frontalindices]
    othernames = [channelnames[i] for i in otherindices]

    eegtorchcpu = torch.from_numpy(eeg_processed).float()
    C = eegtorchcpu.shape[1]

    print("\n\u6784\u5efa\u5168\u5c40 Yule-Walker \u534f\u65b9\u5dee\u77e9\u9635...")
    # \u4fee\u590d\uff1a\u4f20\u53c2 batch_size \u8865\u5168\u4e0b\u5212\u7ebf
    R, Ktotal = build_yule_walker_covariance(eegtorchcpu, bestp, batch_size=128)

    print("\n\U0001f50d \u5f00\u59cb\u6267\u884c\u6210\u5bf9\u53cc\u53d8\u91cf Granger \u68c0\u9a8c...")
    Fmat = np.zeros((len(otherindices), len(frontalindices)), dtype=np.float64)
    pvalmat = np.zeros((len(otherindices), len(frontalindices)), dtype=np.float64)

    d1 = bestp
    # \u4fee\u590d\uff1a\u6f0f\u6389\u7684\u4e58\u53f7 *
    d2 = Ktotal - 2 * bestp

    if d2 <= 0:
        raise ValueError(f"\u81ea\u7531\u5ea6\u4e0d\u8db3\uff1ad2={d2}\uff0c\u8bf7\u68c0\u67e5 bestp={bestp} \u4e0e\u6570\u636e\u957f\u5ea6\u3002")

    for jidx, tgt in enumerate(frontalindices):
        Ytgtidx = tgt
        # \u4fee\u590d\uff1a\u6f0f\u6389\u7684\u4e58\u53f7 *
        Xtgtlags = [tgt + lag * C for lag in range(1, bestp + 1)]

        Ryy = R[Ytgtidx, Ytgtidx]
        Ryxred = R[Ytgtidx, Xtgtlags].unsqueeze(0)
        Rxxred = R[Xtgtlags][:, Xtgtlags]

        RSSred = Ryy - Ryxred @ torch.linalg.pinv(Rxxred) @ Ryxred.t()
        RSSred = RSSred.item()

        for iidx, src in enumerate(otherindices):
            # \u4fee\u590d\uff1a\u6f0f\u6389\u7684\u4e58\u53f7 *
            Xsrclags = [src + lag * C for lag in range(1, bestp + 1)]
            Xfulllags = Xtgtlags + Xsrclags

            Ryxfull = R[Ytgtidx, Xfulllags].unsqueeze(0)
            Rxxfull = R[Xfulllags][:, Xfulllags]

            RSSfull = Ryy - Ryxfull @ torch.linalg.pinv(Rxxfull) @ Ryxfull.t()
            RSSfull = RSSfull.item()

            diff = max(0.0, RSSred - RSSfull)
            Fstat = (diff / d1) / (max(1e-12, RSSfull) / d2)

            Fmat[iidx, jidx] = Fstat
            pvalmat[iidx, jidx] = f.sf(Fstat, d1, d2)

    print(f"\n--- \U0001f4a1 \u6570\u636e\u76d1\u63a7 ---")
    print(f"Fmat \u6700\u5927\u503c: {Fmat.max():.4f}, \u6700\u5c0f\u503c: {Fmat.min():.4f}")
    print(f"Pval \u6700\u5c0f\u503c: {pvalmat.min():.2e}")

    print("\n\u6b63\u5728\u8fdb\u884c\u7edf\u8ba1\u6821\u6b63\u4e0e\u6743\u91cd\u805a\u5408...")
    pvalfdr, sigmask = fdr_bh_correction(pvalmat, alpha=alphafdr)

    print(f"\u901a\u8fc7 FDR \u68c0\u9a8c\u7684\u8fb9\u6570\u91cf: {sigmask.sum()} / {sigmask.size}")
    if sigmask.sum() == sigmask.size:
        print("\u2139\ufe0f \u63d0\u793a\uff1a\u6837\u672c\u91cf\u8f83\u5927\uff0c\u6240\u6709\u8fb9\u5747\u663e\u8457\uff0c\u8fd9\u5728\u5927\u6837\u672c\u7edf\u8ba1\u4e2d\u53ef\u80fd\u51fa\u73b0\u3002")

    scores = np.where(sigmask, np.log1p(Fmat), 0.0)
    rawscores = scores.mean(axis=1)
    weights = normalize_weights(rawscores)

    # \u4f9b\u70ed\u529b\u56fe\u663e\u793a\u7684\u77e9\u9635
    Wsub = np.zeros((len(otherindices), len(frontalindices)), dtype=np.float64)
    for iidx in range(len(otherindices)):
        for jidx in range(len(frontalindices)):
            if sigmask[iidx, jidx]:
                Wsub[iidx, jidx] = np.log1p(Fmat[iidx, jidx])

    if Wsub.max() > 0:
        Wsub = Wsub / Wsub.max()

    # \u7a20\u5bc6 CxC \u90bb\u63a5\u77e9\u9635\uff0c\u4fbf\u4e8e\u4e0b\u6e38\u8bb0\u5f55
    Wgrangergpu = np.zeros((C, C), dtype=np.float64)
    for iidx, src in enumerate(otherindices):
        for jidx, tgt in enumerate(frontalindices):
            if sigmask[iidx, jidx]:
                Wgrangergpu[src, tgt] = Fmat[iidx, jidx]

    return {
        "Fmat": Fmat,
        "pvalmat": pvalmat,
        "pvalfdr": pvalfdr,
        "sigmask": sigmask,
        "rawscores": rawscores,
        "weights": weights,
        "Wsub": Wsub,
        "Wgrangergpu": Wgrangergpu,
        "otherindices": np.array(otherindices),
        "frontalindices": np.array(frontalindices),
        "othernames": othernames,
        "frontalnames": frontalnames,
        "d1": int(d1),
        "d2": int(d2),
        "nchannels": int(C),
        "ntrials": int(eeg_processed.shape[0]),
        "ntimes": int(eeg_processed.shape[2]),
    }


# \u4fee\u590d\uff1a\u53d6\u6d88\u4e86\u591a\u4f59\u7684\u7f29\u8fdb
def boot_strap_stability_analysis(
        eeg_processed,
        channelnames,
        sfreq,
        bestp,
        alphafdr=0.05,
        frontalindices=(0, 16),
        nbootstrap=5,
        sampleratio=0.8,
        randomseed=42,
):
    """
    \u901a\u8fc7 trial-level bootstrap / subsampling \u68c0\u67e5 causal prior \u7a33\u5b9a\u6027\u3002
    """
    rng = np.random.default_rng(randomseed)
    N = eeg_processed.shape[0]
    # \u4fee\u590d\uff1a\u6f0f\u6389\u7684\u4e58\u53f7 *
    samplen = max(2, int(N * sampleratio))

    allweights = []
    for runidx in range(nbootstrap):
        sel = rng.choice(N, size=samplen, replace=False)
        eegsub = eeg_processed[sel]
        res = compute_causal_prior_from_eeg(
            eeg_processed=eegsub,
            channelnames=channelnames,
            sfreq=sfreq,
            bestp=bestp,
            alphafdr=alphafdr,
            frontalindices=frontalindices,
        )
        allweights.append(res["weights"])

    allweights = np.stack(allweights, axis=0)  # (nbootstrap, notherchannels)

    pearsonvals = []
    spearmanvals = []
    top5overlaps = []

    for i in range(nbootstrap):
        for j in range(i + 1, nbootstrap):
            wi = allweights[i]
            wj = allweights[j]

            pearsonvals.append(float(pearsonr(wi, wj)[0]))
            spearmanvals.append(float(spearmanr(wi, wj)[0]))

            topi = set(np.argsort(wi)[-5:])
            topj = set(np.argsort(wj)[-5:])
            top5overlaps.append(len(topi & topj) / 5.0)

    stability = {
        "weightsmean": allweights.mean(axis=0).tolist(),
        "weightsstd": allweights.std(axis=0).tolist(),
        "pairwisepearsonmean": float(np.mean(pearsonvals)) if pearsonvals else None,
        "pairwisespearmanmean": float(np.mean(spearmanvals)) if spearmanvals else None,
        "pairwisetop5overlapmean": float(np.mean(top5overlaps)) if top5overlaps else None,
        "allweights": allweights.tolist(),
    }
    return stability


# ============================================
# 3. \u4e3b\u6d41\u7a0b\u6267\u884c (\u591a\u6570\u636e\u96c6\u81ea\u52a8\u914d\u7f6e\u7248)
# ============================================
if __name__ == "__main__":

    # ----------------------------------------
    # \U0001f3af \u53ea\u9700\u8981\u5728\u8fd9\u91cc\u4fee\u6539\u6570\u636e\u96c6\u540d\u79f0\uff01("HCI" \u6216 "DEAP")
    # ----------------------------------------
    TARGET_DATASET = "HCI"

    # ----------------------------------------
    # \u2699\ufe0f \u6570\u636e\u96c6\u5168\u5c40\u914d\u7f6e\u5b57\u5178
    # ----------------------------------------
    DATASET_CONFIG = {
        "HCI": {
            "sfreq": 256.0,
            "best_p": 10,  # 256Hz \u4e0b 10 \u4e2a lag \u7ea6\u7b49\u4e8e 39ms
            "npz_path": r"./data/hci/hci_30s_preprocessed_checked_labels.npz",
            "out_dir": r"./causality_gpu/hci"
        },
        "DEAP": {
            "sfreq": 128.0,
            "best_p": 5,  # 128Hz \u4e0b 5 \u4e2a lag \u7ea6\u7b49\u4e8e 39ms
            "npz_path": r"./data/deap/deap_30s_preprocessed_checked_labels.npz",
            "out_dir": r"./causality_gpu/deap"
        }
    }

    # \u68c0\u67e5\u8f93\u5165\u662f\u5426\u5408\u6cd5
    if TARGET_DATASET not in DATASET_CONFIG:
        raise ValueError(f"\u672a\u77e5\u7684\u6570\u636e\u96c6: {TARGET_DATASET}\uff0c\u8bf7\u5728 'HCI' \u6216 'DEAP' \u4e2d\u9009\u62e9\u3002")

    # \u81ea\u52a8\u52a0\u8f7d\u5bf9\u5e94\u53c2\u6570
    cfg = DATASET_CONFIG[TARGET_DATASET]
    SFREQ = cfg["sfreq"]
    BEST_P = cfg["best_p"]
    NPZ_PATH = cfg["npz_path"]
    OUT_DIR = cfg["out_dir"]
    print(f"==================================================")
    print(f"\U0001f680 \u6b63\u5728\u521d\u59cb\u5316\u56e0\u679c\u5206\u6790\u6d41\u6c34\u7ebf")
    print(f"\U0001f4c2 \u5f53\u524d\u76ee\u6807\u6570\u636e\u96c6 : {TARGET_DATASET}")
    print(f"\u23f1\ufe0f \u91c7\u6837\u7387 (sfreq) : {SFREQ} Hz")
    print(f"\U0001f9e0 \u6ede\u540e\u9636\u6570 (p)   : {BEST_P} (\u5bf9\u5e94\u7269\u7406\u5ef6\u8fdf ~39ms)")
    print(f"\U0001f4c1 \u8f93\u51fa\u76ee\u5f55       : {OUT_DIR}")
    print(f"==================================================")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n\u52a0\u8f7d\u6570\u636e...")
    data = np.load(NPZ_PATH, allow_pickle=True)
    eeg = data["eeg"]

    channel_names = [
        "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
        "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4",
        "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
        "PO4", "O2"
    ]

    # --- 1. CSD \u7a7a\u95f4\u6ee4\u6ce2 (\u6297\u5bb9\u79ef\u4f20\u5bfc) ---
    print("\n\U0001f9e0 \u6b63\u5728\u5e94\u7528 CSD \u7a7a\u95f4\u6ee4\u6ce2\u4ee5\u6d88\u9664\u5bb9\u79ef\u4f20\u5bfc\u6548\u5e94...")
    csd_applied = False
    csd_error_message = None

    try:
        import mne

        # \u4fee\u590d\uff1a\u4fee\u6b63\u4e86 MNE \u6240\u6709\u7684\u62fc\u5199\u9519\u8bef\u548c\u5f03\u7528\u7684 API
        info = mne.create_info(ch_names=channel_names, sfreq=SFREQ, ch_types='eeg')
        info.set_montage('standard_1020')
        epochs = mne.EpochsArray(eeg, info, verbose=False)
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs, verbose=False)
        eeg_processed = epochs_csd.get_data(copy=False)

        csd_applied = True
        print("\u2705 CSD \u7a7a\u95f4\u6ee4\u6ce2\u5b8c\u6210\uff01")
    except Exception as e:
        csd_error_message = str(e)
        print(f"\u26a0\ufe0f CSD \u5904\u7406\u5931\u8d25 ({e})\uff0c\u56de\u9000\u5230\u539f\u59cb\u9884\u5904\u7406\u6570\u636e\u3002")
        eeg_processed = eeg

    frontalindices = [0, 16]

    # --- 2. \u8ba1\u7b97 causal prior ---
    result = compute_causal_prior_from_eeg(
        eeg_processed=eeg_processed,
        channelnames=channel_names,
        sfreq=SFREQ,
        bestp=BEST_P,
        alphafdr=0.05,
        frontalindices=frontalindices,
    )

    Fmat = result["Fmat"]
    pvalmat = result["pvalmat"]
    pvalfdr = result["pvalfdr"]
    sigmask = result["sigmask"]
    rawscores = result["rawscores"]
    wsoftmax = result["weights"]
    Wsub = result["Wsub"]
    Wgrangergpu = result["Wgrangergpu"]
    otherindices = result["otherindices"]
    frontalindices = result["frontalindices"]
    othernames = result["othernames"]
    frontalnames = result["frontalnames"]
    C = result["nchannels"]

    # --- 3. \u7ed8\u56fe\u5c55\u793a ---
    heatmapname = f"grangergpuheatmap{TARGET_DATASET.lower()}.png"
    plot_causal_heatmap_path = os.path.join(OUT_DIR, heatmapname)

    plt.figure(figsize=(6, 10))
    sns.heatmap(
        Wsub,
        xticklabels=frontalnames,
        yticklabels=othernames,
        cmap="viridis",
        cbar_kws={'label': 'Log F-statistic (Normalized)'}
    )
    plt.title(f"Pairwise Directed Predictive Influence to Frontal Region ({TARGET_DATASET})")
    plt.xlabel("Target (Frontal)")
    plt.ylabel("Source (Others)")
    plt.tight_layout()
    plt.savefig(plot_causal_heatmap_path, dpi=300)
    plt.close()

    # --- 4. \u7ed3\u6784\u5316\u6458\u8981 ---
    top10, fullranking = summarize_channel_scores(
        scores=rawscores,
        weights=wsoftmax,
        othernames=othernames,
        topk=10
    )

    summary = {
        "targetdataset": TARGET_DATASET,
        "sfreqhz": float(SFREQ),
        "bestp": int(BEST_P),
        "physicaldelayms": float(1000.0 * BEST_P / SFREQ),
        "ntrials": int(result["ntrials"]),
        "nchannels": int(result["nchannels"]),
        "ntimes": int(result["ntimes"]),
        "alphafdr": 0.05,
        "methodtype": "pairwisebivariategrangerwithfdraggregation",
        "aggregationrule": "mean(log1p(F)) over significant edges, then L1 normalization",
        "numsignificantedges": int(sigmask.sum()),
        "numtotaledges": int(sigmask.size),
        "top10channels": top10,
        "csd_applied": bool(csd_applied)
    }

    # --- 5. \u5b58\u76d8\u5bf9\u63a5\u4e0b\u6e38 ---
    print("\n\U0001f4be \u6b63\u5728\u4fdd\u5b58\u5bf9\u63a5\u4e0b\u6e38\u6a21\u578b\u53ca\u5ba1\u7a3f\u5907\u67e5\u6240\u9700\u7684\u6587\u4ef6...")

    np.save(os.path.join(OUT_DIR, "frontal_weights.npy"), wsoftmax)
    np.save(os.path.join(OUT_DIR, "other_indices.npy"), np.array(otherindices))
    np.save(os.path.join(OUT_DIR, "W_granger_gpu.npy"), Wgrangergpu)
    np.save(os.path.join(OUT_DIR, "granger_F_matrix.npy"), Fmat)
    np.save(os.path.join(OUT_DIR, "granger_sig_mask.npy"), sigmask)
    np.save(os.path.join(OUT_DIR, "granger_pvals.npy"), pvalmat)
    np.save(os.path.join(OUT_DIR, "granger_pvals_fdr.npy"), pvalfdr)
    np.save(os.path.join(OUT_DIR, "raw_scores.npy"), rawscores)

    save_summary_json(os.path.join(OUT_DIR, "causal_summary.json"), summary)

    save_channel_ranking_csv(
        csvpath=os.path.join(OUT_DIR, "causal_channel_ranking.csv"),
        othernames=othernames,
        rawscores=rawscores,
        weights=wsoftmax,
        sigmask=sigmask,
        pvalfdr=pvalfdr,
        frontalnames=frontalnames
    )

    # \u590d\u5236\u5907\u7528\u56fa\u5b9a\u540d\u70ed\u529b\u56fe
    fixed_heatmap_path = os.path.join(OUT_DIR, "granger_gpu_heatmap.png")
    if os.path.exists(plot_causal_heatmap_path) and plot_causal_heatmap_path != fixed_heatmap_path:
        import shutil

        shutil.copy(plot_causal_heatmap_path, fixed_heatmap_path)

    # \u5982\u679c\u9700\u8981\u8dd1\u7a33\u5b9a\u6027\u5206\u6790\uff0c\u5c06\u8fd9\u91cc\u6539\u4e3a True
    RUN_STABILITY_ANALYSIS = True

    if RUN_STABILITY_ANALYSIS:
        # \u5c06 nbootstrap \u4ece 5 \u63d0\u5347\u5230 20 \u6216 50\uff0c\u4ee5\u83b7\u5f97\u66f4\u5f3a\u7edf\u8ba1\u5b66\u652f\u6491
        N_BOOTSTRAPS = 50

        print(f"\n\U0001f4ca \u5f00\u59cb\u6267\u884c Bootstrap \u7a33\u5b9a\u6027\u5206\u6790 (N={N_BOOTSTRAPS} \u6b21\u91cd\u91c7\u6837) ...")
        stability = boot_strap_stability_analysis(
            eeg_processed=eeg_processed,
            channelnames=channel_names,
            sfreq=SFREQ,
            bestp=BEST_P,
            alphafdr=0.05,
            frontalindices=(0, 16),
            nbootstrap=N_BOOTSTRAPS,
            sampleratio=0.8,
            randomseed=42,
        )

        # \u786e\u4fdd\u5b58\u4e0b\u6bcf\u4e00\u6b21 Bootstrap \u7684\u5b8c\u6574\u6743\u91cd\u77e9\u9635 (N_BOOTSTRAPS, 30)
        np.save(os.path.join(OUT_DIR, "boot_strap_weights.npy"), np.array(stability["allweights"]))
        np.save(os.path.join(OUT_DIR, "boot_strap_weights_mean.npy"), np.array(stability["weightsmean"]))
        np.save(os.path.join(OUT_DIR, "boot_strap_weights_std.npy"), np.array(stability["weightsstd"]))

        stabilitysummary = {
            "pairwisepearsonmean": stability["pairwisepearsonmean"],
            "pairwisespearmanmean": stability["pairwisespearmanmean"],
            "pairwisetop5overlapmean": stability["pairwisetop5overlapmean"],
            "nbootstrap": N_BOOTSTRAPS,
            "sampleratio": 0.8
        }
        save_summary_json(os.path.join(OUT_DIR, "causal_stability_summary.json"), stabilitysummary)
        print("\u2705 Bootstrap \u7a33\u5b9a\u6027\u6570\u636e\u63d0\u53d6\u5b8c\u6bd5\u5e76\u5df2\u5b58\u76d8\uff01")

    print(f"\U0001f389 \u8fd0\u884c\u5b8c\u6bd5\uff01\u6240\u6709\u5bf9\u63a5\u6587\u4ef6\u5df2\u751f\u6210\u81f3: {OUT_DIR}")