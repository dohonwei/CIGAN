import torch
import time
import numpy as np
from thop import profile
from thop import clever_format
import importlib.util
import sys
import warnings

warnings.filterwarnings('ignore')

# ==============================================================
# \u52a8\u6001\u52a0\u8f7d 1_train_cigan.py
# ==============================================================
file_path = "1_train_cigan.py"
spec = importlib.util.spec_from_file_location("train_cigan", file_path)
train_cigan = importlib.util.module_from_spec(spec)
sys.modules["train_cigan"] = train_cigan
spec.loader.exec_module(train_cigan)

# \u2605 \u6838\u5fc3\u4fee\u590d\uff1a\u5f3a\u5236\u8986\u76d6\u539f\u6587\u4ef6\u4e2d\u7684\u5168\u5c40 device \u53d8\u91cf\u4e3a CPU\uff0c\u89e3\u51b3\u8de8\u8bbe\u5907\u62fc\u63a5\u62a5\u9519
edge_device = torch.device("cpu")
train_cigan.device = edge_device

# \u6210\u529f\u5bfc\u5165\u6a21\u578b\u7c7b
UltraGenerator = train_cigan.UltraGenerator


def evaluate_edge_metrics():
    print("=" * 50)
    print("   IoHT Edge-Computing Metrics Profiler")
    print("=" * 50)

    print("Target Hardware: Edge CPU (Simulated)")

    # \u521d\u59cb\u5316\u4f60\u7684\u6a21\u578b (\u8f93\u5165 30 \u901a\u9053)
    model = UltraGenerator(O=30).to(edge_device)
    model.eval()

    # \u6a21\u62df\u5355\u6b21\u63a8\u7406\u7684\u8f93\u5165 (Batch=1, Channels=30, Time=3840 \u5bf9\u5e94 30\u79d2 128Hz)
    dummy_others = torch.randn(1, 30, 3840).to(edge_device)
    dummy_w_vec = torch.randn(30).to(edge_device)

    # ==========================================
    # 1. \u8ba1\u7b97 FLOPs (\u8ba1\u7b97\u590d\u6742\u5ea6) \u548c Params (\u5185\u5b58\u5360\u7528)
    # ==========================================
    print("\n[1] Profiling FLOPs and Parameters...")
    try:
        flops, params = profile(model, inputs=(dummy_others, dummy_w_vec), verbose=False)
        flops_formatted, params_formatted = clever_format([flops, params], "%.2f")
        print(f"Total Parameters (Model Size): {params_formatted} (Raw: {params})")
        print(f"Total FLOPs (Computation)  : {flops_formatted} (Raw: {flops})")

        # \u4f30\u7b97\u6a21\u578b\u5b58\u50a8\u4f53\u79ef (MB) = \u53c2\u6570\u91cf * 4 Bytes (float32) / 1024 / 1024
        model_size_mb = params * 4 / (1024 * 1024)
        print(f"Estimated Memory Footprint : {model_size_mb:.2f} MB")
    except Exception as e:
        print("Error calculating FLOPs. \u8bf7\u786e\u4fdd\u5b89\u88c5\u4e86 thop: pip install thop")
        print(e)
        return

    # ==========================================
    # 2. \u8ba1\u7b97 Inference Latency (\u63a8\u65ad\u5ef6\u8fdf)
    # ==========================================
    print("\n[2] Benchmarking Inference Latency (CPU)...")

    # \u9884\u70ed (Warm-up) \u907f\u514d\u51b7\u542f\u52a8\u8bef\u5dee
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_others, dummy_w_vec)

    # \u6b63\u5f0f\u6d4b\u901f (\u6d4b 30 \u6b21\u53d6\u5e73\u5747)
    latencies = []
    with torch.no_grad():
        for _ in range(30):
            start_time = time.perf_counter()
            _ = model(dummy_others, dummy_w_vec)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # \u8f6c\u6362\u4e3a\u6beb\u79d2 (ms)

    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    print(f"Average CPU Inference Latency: {avg_latency:.2f} \xb1 {std_latency:.2f} ms per 30-sec epoch")

    # ==========================================
    # 3. \u8f93\u51fa\u8bba\u6587\u4e13\u7528\u603b\u7ed3
    # ==========================================
    print("\n" + "=" * 50)
    print("\U0001f4cb PAPER REPORTING SUMMARY (Copy to Response Letter)")
    print("=" * 50)
    print(f"To address the reviewer's concern regarding edge-intelligence feasibility, ")
    print(f"we profiled CIGAN on a standard CPU node. The model maintains a highly ")
    print(f"compact footprint of {model_size_mb:.2f} MB ({params_formatted} parameters). ")
    print(f"Processing a 30-second multi-channel EEG epoch requires {flops_formatted} FLOPs, ")
    print(f"with an average inference latency of {avg_latency:.2f} ms. ")
    print(f"This sub-second processing time is well within the real-time constraints ")
    print(f"of wearable IoHT applications.")


if __name__ == "__main__":
    evaluate_edge_metrics()