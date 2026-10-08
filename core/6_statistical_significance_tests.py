"""
6_statistical_significance_tests.py (TIM Revision)
\u6267\u884c\u914d\u5bf9\u975e\u53c2\u6570\u7edf\u8ba1\u68c0\u9a8c (Wilcoxon Signed-Rank Test)\uff0c\u751f\u6210\u4f9b IEEE TIM \u8bba\u6587\u76f4\u63a5\u4f7f\u7528\u7684\u663e\u8457\u6027\u62a5\u544a\u3002
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# ==============================================================
#                    CONFIG & PATHS
# ==============================================================
DATASET = "deap"
BASE_DIR = r"."
RAW_DATA_CSV = os.path.join(BASE_DIR, "evaluate", DATASET, f"Master_Quality_Raw_Data_{DATASET.upper()}.csv")
SAVE_DIR = os.path.join(BASE_DIR, "evaluate", DATASET)

TARGET_MODEL = "CIGAN (Main)"  # \u4f60\u7684\u4e3b\u6a21\u578b\u540d\u79f0
METRICS_TO_TEST = ["MMD", "DTW", "Cosine", "Theta_Diff"]  # \u4f60\u60f3\u68c0\u9a8c\u7684\u5173\u952e\u6307\u6807


def get_significance_stars(p_value):
    """\u6839\u636e p \u503c\u8fd4\u56de\u7edf\u8ba1\u663e\u8457\u6027\u661f\u53f7"""
    if p_value < 0.001: return "***"
    if p_value < 0.01:  return "**"
    if p_value < 0.05:  return "*"
    return "ns"  # Not Significant


def main():
    print("=" * 60)
    print(f"STATISTICAL SIGNIFICANCE TESTING ({DATASET.upper()})")
    print("=" * 60)

    if not os.path.exists(RAW_DATA_CSV):
        raise FileNotFoundError(f"\u627e\u4e0d\u5230\u539f\u59cb\u6570\u636e {RAW_DATA_CSV}\uff0c\u8bf7\u5148\u8fd0\u884c\u5347\u7ea7\u7248\u7684 3_evaluate_quality.py")

    df = pd.read_csv(RAW_DATA_CSV)

    # \u83b7\u53d6\u6240\u6709\u7684\u5bf9\u6bd4\u6a21\u578b\uff08\u9664\u4e86\u76ee\u6807\u4e3b\u6a21\u578b\uff09
    baselines = [m for m in df['Model'].unique() if m != TARGET_MODEL]

    results = []

    # \u4e3a\u4e86\u7b80\u5316\u8868\u683c\uff0c\u6211\u4eec\u8fd9\u91cc\u4ee5 Fp1 \u9891\u9053\u7684\u9a8c\u8bc1\u4e3a\u4f8b\uff08\u4f60\u4e5f\u53ef\u4ee5\u8dd1\u5168\u901a\u9053\u6216 Fp2\uff09
    df_fp1 = df[df['Channel'] == 'Fp1']

    for baseline in baselines:
        for metric in METRICS_TO_TEST:
            # \u63d0\u53d6\u6210\u5bf9\u7684 Trial \u6570\u636e
            target_scores = df_fp1[df_fp1['Model'] == TARGET_MODEL].sort_values('Trial_Idx')[metric].values
            base_scores = df_fp1[df_fp1['Model'] == baseline].sort_values('Trial_Idx')[metric].values

            # \u786e\u4fdd\u6837\u672c\u5b8c\u5168\u5bf9\u9f50
            if len(target_scores) != len(base_scores) or len(target_scores) == 0:
                continue

            # \u6267\u884c Wilcoxon \u914d\u5bf9\u7b26\u53f7\u79e9\u68c0\u9a8c
            # \u6ce8\u610f\uff1a\u5982\u679c\u4e24\u7ec4\u6570\u636e\u5b8c\u5168\u76f8\u540c\uff08\u5dee\u503c\u4e3a0\uff09\uff0cwilcoxon \u4f1a\u62a5\u9519\uff0c\u6211\u4eec\u52a0\u4e2a try-except
            try:
                stat, p_val = wilcoxon(target_scores, base_scores)
            except ValueError:
                p_val = 1.0

                # \u8ba1\u7b97\u6548\u5e94\u91cf (Effect Size) - \u8fd9\u91cc\u4f7f\u7528 Cohen's d \u7684\u7b80\u5316\u7248\u8fd1\u4f3c\uff0c\u6216\u8005\u7edd\u5bf9\u5747\u503c\u5dee
            mean_diff = np.mean(target_scores) - np.mean(base_scores)

            # MMD, DTW, Theta_Diff \u662f\u8d8a\u5c0f\u8d8a\u597d\uff1bCosine \u662f\u8d8a\u5927\u8d8a\u597d\u3002\u6211\u4eec\u8981\u6807\u660e\u662f\u8c01\u66f4\u597d\u3002
            if metric in ["MMD", "DTW", "Theta_Diff", "Delta_Diff", "Alpha_Diff", "Beta_Diff"]:
                cigan_better = mean_diff < 0
            else:
                cigan_better = mean_diff > 0

            results.append({
                "Baseline Model": baseline,
                "Metric": metric,
                "CIGAN Mean": f"{np.mean(target_scores):.4f}",
                "Baseline Mean": f"{np.mean(base_scores):.4f}",
                "p-value": f"{p_val:.2e}",
                "Significance": get_significance_stars(p_val),
                "CIGAN Wins?": "Yes" if cigan_better and p_val < 0.05 else "No"
            })

    # \u8f93\u51fa\u6f02\u4eae\u7684\u8868\u683c
    df_results = pd.DataFrame(results)

    print("\n\U0001f3c6 \u914d\u5bf9\u663e\u8457\u6027\u68c0\u9a8c\u7ed3\u679c (Target: CIGAN vs Others):")
    print(df_results.to_string(index=False))

    out_csv = os.path.join(SAVE_DIR, f"Statistical_Significance_Report_{DATASET.upper()}.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\n\u2705 \u663e\u8457\u6027\u68c0\u9a8c\u62a5\u544a\u5df2\u5bfc\u51fa\u81f3: {out_csv}")


if __name__ == "__main__":
    main()