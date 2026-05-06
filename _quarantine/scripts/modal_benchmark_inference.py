"""Modal.com GPU inference benchmark for RR-MoA, SR-MoA, Residual-IA+.

Replaces the retired GPU VM for the inference table. The container is
ephemeral: Modal serverless functions auto-scale to 0 immediately after
the benchmark dictionary is returned to the local CLI, so no idle GPU
charges accrue.

Usage (one-shot):
    modal run scripts/modal_benchmark_inference.py

Output: results/benchmark/inference_modal_A10G_B128.json
"""

import json
import os
import sys

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "momentfm",
        "peft>=0.7.0",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "huggingface_hub",
    )
    .add_local_dir(os.path.join(REPO_ROOT, "feasibility"), "/root/feasibility")
    .add_local_dir(os.path.join(REPO_ROOT, "scripts"), "/root/scripts")
)

app = modal.App("rrmoa-inference-benchmark", image=image)


@app.function(gpu="A10G", timeout=900, memory=16384)
def run_inference_benchmark(
    backbone: str = "AutonLab/MOMENT-1-small",
    batch_size: int = 128,
    input_len: int = 512,
    horizon: int = 96,
    warmup: int = 20,
    repeats: int = 200,
) -> dict:
    """Benchmark backbone, single adapter, RR-MoA, SR-MoA, Residual-IA+, DLinear."""
    sys.path.insert(0, "/root")

    import numpy as np
    import torch
    import torch.nn as nn

    from feasibility.model import (
        load_backbone, _get_encoder_blocks, _get_hidden_dim,
        _disable_gradient_checkpointing,
    )
    from feasibility.finetune import _extract_features_batch
    from scripts.run_rr_moa import RawRoutedMoA, Conv1dPoolHead
    from scripts.run_self_routed_moa import SelfRoutedMoA
    from scripts.run_sr_ria import SelfRoutedResidualIA
    from scripts.run_standard_evolution import _detect_backbone_type

    class DLinear(nn.Module):
        def __init__(self, input_len_, output_len_):
            super().__init__()
            self.linear = nn.Linear(input_len_, output_len_)

        def forward(self, x):
            return self.linear(x)

    device = "cuda"
    B, L, H = batch_size, input_len, horizon

    print("[modal] Loading backbone: %s" % backbone)
    model = load_backbone(backbone, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(backbone)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    x_raw = torch.randn(B, L, device=device)
    bx = x_raw.unsqueeze(1)
    mask = torch.ones(B, L, device=device)

    def benchmark(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(repeats):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
        }

    def peak_memory(fn):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        return float(torch.cuda.max_memory_allocated() / (1024 * 1024))

    results = {}

    print("[modal] (1/6) Backbone-only forward...")

    def backbone_fn():
        with torch.no_grad():
            _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)

    results["backbone"] = benchmark(backbone_fn)
    results["backbone"]["peak_mb"] = peak_memory(backbone_fn)
    results["backbone"]["params"] = sum(p.numel() for p in model.parameters())

    print("[modal] (2/6) Single adapter (Conv1dPool)...")
    conv = Conv1dPoolHead(hdim, H).to(device).eval()

    def conv_fn():
        with torch.no_grad():
            f = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
            conv(f)

    results["single_adapter"] = benchmark(conv_fn)
    results["single_adapter"]["peak_mb"] = peak_memory(conv_fn)
    results["single_adapter"]["adapter_params"] = sum(p.numel() for p in conv.parameters())

    print("[modal] (3/6) RR-MoA (Top-2, K=5)...")
    rrmoa = RawRoutedMoA(hdim, H, input_len=L, K=5, top_k=2).to(device).eval()

    def rr_fn():
        with torch.no_grad():
            f = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
            rrmoa(f, x_raw)

    results["rrmoa_top2"] = benchmark(rr_fn)
    results["rrmoa_top2"]["peak_mb"] = peak_memory(rr_fn)
    results["rrmoa_top2"]["adapter_params"] = sum(p.numel() for p in rrmoa.parameters())

    print("[modal] (4/6) SR-MoA (dense, K=5)...")
    srmoa = SelfRoutedMoA(
        hdim, H, input_len=L, K=5, hidden=64,
        routing_mode="gated", gate_hidden=16,
    ).to(device).eval()

    def sr_fn():
        with torch.no_grad():
            f = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
            srmoa(f, x_raw)

    results["srmoa_dense"] = benchmark(sr_fn)
    results["srmoa_dense"]["peak_mb"] = peak_memory(sr_fn)
    results["srmoa_dense"]["adapter_params"] = sum(p.numel() for p in srmoa.parameters())

    print("[modal] (5/6) Residual-IA+ (SR-RIA)...")
    sr_ria = SelfRoutedResidualIA(
        hdim, H, input_len=L, K=5, hidden=64, gate_hidden=16,
        blend_init_bias=-2.0, raw_arch="nlinear",
    ).to(device).eval()

    def ria_fn():
        with torch.no_grad():
            f = _extract_features_batch(model, blocks, bx, mask, backbone_type=bb_type)
            sr_ria(f, x_raw)

    results["residual_ia_plus"] = benchmark(ria_fn)
    results["residual_ia_plus"]["peak_mb"] = peak_memory(ria_fn)
    results["residual_ia_plus"]["adapter_params"] = sum(p.numel() for p in sr_ria.parameters())

    print("[modal] (6/6) DLinear (no backbone)...")
    dlinear = DLinear(L, H).to(device).eval()

    def dl_fn():
        with torch.no_grad():
            dlinear(x_raw)

    results["dlinear"] = benchmark(dl_fn)
    results["dlinear"]["peak_mb"] = peak_memory(dl_fn)
    results["dlinear"]["params"] = sum(p.numel() for p in dlinear.parameters())

    results["_meta"] = {
        "backbone": backbone,
        "device": "cuda (Modal A10G)",
        "batch_size": B,
        "input_len": L,
        "horizon": H,
        "warmup": warmup,
        "repeats": repeats,
    }
    print("[modal] Done.")
    return results


@app.local_entrypoint()
def main(
    backbone: str = "AutonLab/MOMENT-1-small",
    batch_size: int = 128,
    repeats: int = 200,
):
    print("Submitting inference benchmark to Modal A10G (auto-scale-to-0 after run)...")
    results = run_inference_benchmark.remote(
        backbone=backbone,
        batch_size=batch_size,
        repeats=repeats,
    )

    print("\n" + "=" * 78)
    print("INFERENCE BENCHMARK (Modal A10G, batch=%d)" % batch_size)
    print("=" * 78)
    header = "%-22s %18s %12s %18s"
    print(header % ("Method", "Latency (ms)", "Peak MB", "Adapter Params"))
    print("-" * 78)
    order = [
        "backbone", "single_adapter", "rrmoa_top2",
        "srmoa_dense", "residual_ia_plus", "dlinear",
    ]
    for name in order:
        r = results.get(name, {})
        if not r:
            continue
        latency = "%.2f +/- %.2f" % (r.get("mean_ms", 0), r.get("std_ms", 0))
        peak = "%.0f" % r.get("peak_mb", 0)
        params = r.get("adapter_params", r.get("params", 0))
        print(header % (name, latency, peak, "{:,}".format(params)))

    out_path = os.path.join(
        REPO_ROOT, "results", "benchmark", "inference_modal_A10G_B%d.json" % batch_size
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: %s" % out_path)
    print("Modal container has been released; GPU auto-scaled to 0.")
