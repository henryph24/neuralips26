"""Inference latency and memory benchmarks for deployment motivation.

Measures per-batch inference time and peak GPU memory for:
- RR-MoA (Top-2, 5 experts)
- SR-MoA (Self-Routed, K=5 dense, per-expert sigmoid gates on raw input)
- Residual-IA+ (Self-Routed Residual-IA, shared NLinear raw branch + gated backbone residual)
- Single adapter (conv head)
- DLinear (from-scratch baseline)

Usage:
    python scripts/benchmark_inference.py --device cuda
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from scripts.run_rr_moa import (
    MeanPoolHead, LastTokenHead, MaxPoolHead, AttentionPoolHead, Conv1dPoolHead,
    RawRoutedMoA,
)
from scripts.run_self_routed_moa import SelfRoutedMoA
from scripts.run_sr_ria import SelfRoutedResidualIA
from feasibility.standard_data import _detect_backbone_type


class DLinear(nn.Module):
    """Simple DLinear baseline for benchmarking."""
    def __init__(self, input_len=512, output_len=96):
        super().__init__()
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x):
        return self.linear(x)


def benchmark_forward(fn, warmup=20, repeats=200, device="cuda"):
    """Benchmark a callable with CUDA events for accurate timing."""
    # Warmup
    for _ in range(warmup):
        fn()

    if device == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(repeats):
            start_event.record()
            fn()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
    else:
        times = []
        for _ in range(repeats):
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            fn()
            if device == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def measure_peak_memory(fn, device="cuda"):
    """Measure peak GPU memory during a forward pass."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return {"peak_mb": float(peak)}
    return {"peak_mb": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    B = args.batch_size
    L = args.input_len
    H = args.horizon

    print("Loading backbone: %s" % args.backbone)
    model = load_backbone(args.backbone, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Dummy input
    x_raw = torch.randn(B, L, device=device)
    bx = x_raw.unsqueeze(1)  # (B, 1, L)
    mask = torch.ones(B, L, device=device)

    results = {}

    # --- 1. Backbone feature extraction (shared cost) ---
    print("\n1. Backbone feature extraction...")
    with torch.no_grad():
        feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
    print("   Feature shape: %s" % str(feat.shape))

    def backbone_fn():
        with torch.no_grad():
            _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)

    results["backbone"] = benchmark_forward(backbone_fn, device=device)
    results["backbone"].update(measure_peak_memory(backbone_fn, device=device))
    results["backbone"]["params"] = sum(p.numel() for p in model.parameters())
    print("   Latency: %.2f +/- %.2f ms" % (results["backbone"]["mean_ms"], results["backbone"]["std_ms"]))

    # --- 2. Single adapter (conv) ---
    print("\n2. Single adapter (conv head)...")
    conv_head = Conv1dPoolHead(hdim, H).to(device).eval()
    adapter_params = sum(p.numel() for p in conv_head.parameters())

    def single_adapter_fn():
        with torch.no_grad():
            f = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
            conv_head(f)

    results["single_adapter"] = benchmark_forward(single_adapter_fn, device=device)
    results["single_adapter"].update(measure_peak_memory(single_adapter_fn, device=device))
    results["single_adapter"]["adapter_params"] = adapter_params
    print("   Latency: %.2f +/- %.2f ms, params: %d" % (
        results["single_adapter"]["mean_ms"], results["single_adapter"]["std_ms"], adapter_params))

    # --- 3. RR-MoA (Top-2, 5 experts) ---
    print("\n3. RR-MoA (Top-2, K=5)...")
    rrmoa = RawRoutedMoA(hdim, H, input_len=L, K=5, top_k=2).to(device).eval()
    rrmoa_params = sum(p.numel() for p in rrmoa.parameters())

    def rrmoa_fn():
        with torch.no_grad():
            f = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
            rrmoa(f, x_raw)

    results["rrmoa_top2"] = benchmark_forward(rrmoa_fn, device=device)
    results["rrmoa_top2"].update(measure_peak_memory(rrmoa_fn, device=device))
    results["rrmoa_top2"]["adapter_params"] = rrmoa_params
    print("   Latency: %.2f +/- %.2f ms, params: %d" % (
        results["rrmoa_top2"]["mean_ms"], results["rrmoa_top2"]["std_ms"], rrmoa_params))

    # --- 4. SR-MoA (Self-Routed, K=5 dense, gated mode) ---
    print("\n4. SR-MoA (Self-Routed, K=5 dense)...")
    srmoa = SelfRoutedMoA(
        hdim, H, input_len=L, K=5, hidden=64,
        routing_mode="gated", gate_hidden=16,
    ).to(device).eval()
    srmoa_params = sum(p.numel() for p in srmoa.parameters())

    def srmoa_fn():
        with torch.no_grad():
            f = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
            srmoa(f, x_raw)

    results["srmoa_dense"] = benchmark_forward(srmoa_fn, device=device)
    results["srmoa_dense"].update(measure_peak_memory(srmoa_fn, device=device))
    results["srmoa_dense"]["adapter_params"] = srmoa_params
    print("   Latency: %.2f +/- %.2f ms, params: %d" % (
        results["srmoa_dense"]["mean_ms"], results["srmoa_dense"]["std_ms"], srmoa_params))

    # --- 5. Residual-IA+ (Self-Routed Residual-IA, shared NLinear raw branch) ---
    print("\n5. Residual-IA+ (SR-RIA, K=5)...")
    sr_ria = SelfRoutedResidualIA(
        hdim, H, input_len=L, K=5, hidden=64,
        gate_hidden=16, blend_init_bias=-2.0, raw_arch="nlinear",
    ).to(device).eval()
    sr_ria_params = sum(p.numel() for p in sr_ria.parameters())

    def sr_ria_fn():
        with torch.no_grad():
            f = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
            sr_ria(f, x_raw)

    results["residual_ia_plus"] = benchmark_forward(sr_ria_fn, device=device)
    results["residual_ia_plus"].update(measure_peak_memory(sr_ria_fn, device=device))
    results["residual_ia_plus"]["adapter_params"] = sr_ria_params
    print("   Latency: %.2f +/- %.2f ms, params: %d" % (
        results["residual_ia_plus"]["mean_ms"], results["residual_ia_plus"]["std_ms"], sr_ria_params))

    # --- 6. DLinear (no backbone) ---
    print("\n4. DLinear (from scratch, no backbone)...")
    dlinear = DLinear(L, H).to(device).eval()
    dlinear_params = sum(p.numel() for p in dlinear.parameters())

    def dlinear_fn():
        with torch.no_grad():
            dlinear(x_raw)

    results["dlinear"] = benchmark_forward(dlinear_fn, device=device)
    results["dlinear"].update(measure_peak_memory(dlinear_fn, device=device))
    results["dlinear"]["params"] = dlinear_params
    print("   Latency: %.2f +/- %.2f ms, params: %d" % (
        results["dlinear"]["mean_ms"], results["dlinear"]["std_ms"], dlinear_params))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK SUMMARY (batch=%d, device=%s)" % (B, device))
    print("=" * 60)
    print("%-20s %10s %10s %12s" % ("Method", "Latency", "Peak MB", "Adapter Params"))
    print("-" * 60)
    for name, r in results.items():
        latency = "%.1f ms" % r["mean_ms"]
        peak = "%.0f" % r.get("peak_mb", 0)
        params = r.get("adapter_params", r.get("params", 0))
        print("%-20s %10s %10s %12s" % (name, latency, peak, "{:,}".format(params)))

    # Save (device-tagged so MPS / CPU runs do not overwrite the canonical A10G run)
    os.makedirs("results/benchmark", exist_ok=True)
    suffix = "" if device == "cuda" else "_%s" % device
    fname = "results/benchmark/inference_%s_B%d%s.json" % (
        args.backbone.split("/")[-1], B, suffix)
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: %s" % fname)


if __name__ == "__main__":
    main()
