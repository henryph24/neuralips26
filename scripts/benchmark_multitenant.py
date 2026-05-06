"""Multi-tenant adapter swapping benchmark.

Simulates a multi-tenant deployment where one frozen backbone serves
N tenants via hot-swappable adapters. Measures adapter swap latency,
throughput, and memory vs. per-tenant model reloading.

Usage:
    python scripts/benchmark_multitenant.py --device cuda
"""

import argparse
import json
import os
import sys
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from scripts.run_rr_moa import RawRoutedMoA
from feasibility.standard_data import _detect_backbone_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device
    B = args.batch_size
    L = 512
    H = 96

    print("Loading backbone: %s" % args.backbone)
    model = load_backbone(args.backbone, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Create dummy input
    x_raw = torch.randn(B, L, device=device)
    bx = x_raw.unsqueeze(1)
    mask = torch.ones(B, L, device=device)

    # Pre-compute features (shared across tenants)
    with torch.no_grad():
        feat = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)

    # Create N different adapter state dicts (simulating different tenants)
    print("\nCreating adapter variants for N tenants...")
    template = RawRoutedMoA(hdim, H, input_len=L, K=5, top_k=2).to(device).eval()
    adapter_size_bytes = sum(p.nelement() * p.element_size() for p in template.parameters())
    adapter_size_mb = adapter_size_bytes / (1024 * 1024)
    print("  Adapter size: %.2f MB (%d params)" % (
        adapter_size_mb, sum(p.numel() for p in template.parameters())))

    # Save N different adapter state dicts to CPU (simulating host RAM)
    N_MAX = 1000
    tenant_weights = []
    for i in range(N_MAX):
        sd = {}
        for k, v in template.state_dict().items():
            sd[k] = v.cpu() + torch.randn_like(v.cpu()) * 0.01
        tenant_weights.append(sd)

    results = {}

    # ================================================================
    # Benchmark 1: Adapter swap latency
    # ================================================================
    print("\n=== Benchmark 1: Adapter Swap Latency ===")
    for N in [1, 10, 100, 500, 1000]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N):
            # Load adapter weights from "host RAM" to GPU
            sd = tenant_weights[i % N_MAX]
            template.load_state_dict({k: v.to(device) for k, v in sd.items()})
            # Run inference
            with torch.no_grad():
                template(feat, x_raw)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        per_tenant_ms = elapsed / N * 1000
        throughput = N / elapsed
        print("  N=%4d: total=%.2fs  per_tenant=%.2fms  throughput=%.0f tenants/sec" % (
            N, elapsed, per_tenant_ms, throughput))
        results["swap_N%d" % N] = {
            "n_tenants": N,
            "total_sec": elapsed,
            "per_tenant_ms": per_tenant_ms,
            "throughput_per_sec": throughput,
        }

    # ================================================================
    # Benchmark 2: Memory footprint comparison
    # ================================================================
    print("\n=== Benchmark 2: Memory Footprint ===")
    backbone_params = sum(p.numel() for p in model.parameters())
    adapter_params = sum(p.numel() for p in template.parameters())

    backbone_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    adapter_mb = adapter_size_mb

    for N in [1, 10, 100, 1000]:
        # RR-MoA approach: 1 backbone + N adapters (adapters in host RAM, 1 on GPU)
        rrmoa_gpu_mb = backbone_mb + adapter_mb  # constant
        rrmoa_host_mb = N * adapter_mb
        rrmoa_total_mb = rrmoa_gpu_mb + rrmoa_host_mb

        # Per-tenant DLinear: N separate models
        dlinear_params = L * H  # simple linear
        dlinear_mb_each = dlinear_params * 4 / (1024 * 1024)  # float32
        dlinear_total_mb = N * dlinear_mb_each

        # Per-tenant full model: N copies of backbone
        full_total_mb = N * backbone_mb

        print("  N=%4d:  RR-MoA=%.1fMB(GPU)+%.1fMB(host)  "
              "DLinear=%.1fMB  FullModel=%.1fMB" % (
            N, rrmoa_gpu_mb, rrmoa_host_mb, dlinear_total_mb, full_total_mb))
        results["memory_N%d" % N] = {
            "n_tenants": N,
            "rrmoa_gpu_mb": rrmoa_gpu_mb,
            "rrmoa_host_mb": rrmoa_host_mb,
            "dlinear_total_mb": dlinear_total_mb,
            "full_model_total_mb": full_total_mb,
        }

    # ================================================================
    # Benchmark 3: Backbone forward (shared cost)
    # ================================================================
    print("\n=== Benchmark 3: Backbone Forward (shared, amortized) ===")
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        with torch.no_grad():
            _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    backbone_ms = np.mean(times)
    print("  Backbone forward: %.1f ms (shared across all tenants)" % backbone_ms)
    results["backbone_forward_ms"] = backbone_ms

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-TENANT DEPLOYMENT SUMMARY")
    print("=" * 60)
    print("Backbone: %s (%.1f MB GPU)" % (args.backbone, backbone_mb))
    print("Adapter:  RR-MoA Top-2 (%.2f MB, %d params)" % (adapter_mb, adapter_params))
    print("")
    print("At N=100 tenants:")
    r100 = results["swap_N100"]
    m100 = results["memory_N100"]
    print("  Adapter swap: %.1f ms/tenant (%.0f tenants/sec)" % (
        r100["per_tenant_ms"], r100["throughput_per_sec"]))
    print("  RR-MoA memory:  %.1f MB GPU + %.1f MB host RAM" % (
        m100["rrmoa_gpu_mb"], m100["rrmoa_host_mb"]))
    print("  Full-model alt: %.1f MB (%.0fx more)" % (
        m100["full_model_total_mb"],
        m100["full_model_total_mb"] / (m100["rrmoa_gpu_mb"] + m100["rrmoa_host_mb"])))

    # Save
    os.makedirs("results/benchmark", exist_ok=True)
    fname = "results/benchmark/multitenant_%s_B%d.json" % (
        args.backbone.split("/")[-1], B)
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: %s" % fname)


if __name__ == "__main__":
    main()
