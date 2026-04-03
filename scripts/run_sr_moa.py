"""Spectral-Routed Mixture of Adapters (SR-MoA).

Fallback for RR-MoA: routes on the FREQUENCY DOMAIN of the raw input
instead of the time domain. FFT provides mathematically stable routing
signals immune to both backbone LayerNorms and temporal noise.

Usage:
    python scripts/run_sr_moa.py --dataset ETTh1
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _get_hidden_dim,
    _disable_gradient_checkpointing,
)
from feasibility.finetune import _extract_features_batch
from feasibility.code_evolution import SEED_ADAPTERS
from scripts.run_standard_evolution import (
    load_standard_data, train_adapter, _detect_backbone_type,
)
from scripts.run_rr_moa import (
    MeanPoolHead, LastTokenHead, MaxPoolHead, AttentionPoolHead, Conv1dPoolHead,
    HEAD_CLASSES, HEAD_NAMES,
)


class SpectralRoutedMoA(nn.Module):
    """Spectral-Routed Mixture of Adapters.

    Routes on FFT amplitude spectrum of raw input.
    Smooth low-freq signals → Linear/MLP experts.
    Noisy high-freq signals → Conv/Attention experts.
    """
    def __init__(self, d_model, output_dim, input_len=512, K=5, hidden=64):
        super().__init__()
        self.K = K
        self.input_len = input_len

        # K expert adapter heads
        self.adapters = nn.ModuleList([
            HEAD_CLASSES[i % len(HEAD_CLASSES)](d_model, output_dim, hidden)
            for i in range(K)
        ])

        # Spectral router: operates on FFT amplitude spectrum
        n_freqs = input_len // 2 + 1  # 257 for input_len=512
        self.router = nn.Sequential(
            nn.Linear(n_freqs, 32),
            nn.GELU(),
            nn.Linear(32, K),
        )

        self.load_balance_coeff = 0.01

    def _get_spectrum(self, raw_input):
        """Compute amplitude spectrum via FFT."""
        spectrum = torch.fft.rfft(raw_input, dim=-1)
        amplitude = spectrum.abs()  # (B, n_freqs)
        # Log-scale for numerical stability
        return torch.log1p(amplitude)

    def forward(self, hidden_states, raw_input):
        # Route based on spectral content
        spectrum = self._get_spectrum(raw_input)  # (B, n_freqs)
        logits = self.router(spectrum)  # (B, K)
        weights = F.softmax(logits, dim=-1)  # (B, K)

        # Expert outputs from hidden states
        outputs = torch.stack([a(hidden_states) for a in self.adapters], dim=1)

        return (weights.unsqueeze(-1) * outputs).sum(dim=1)

    def get_routing_stats(self, raw_input):
        with torch.no_grad():
            spectrum = self._get_spectrum(raw_input)
            logits = self.router(spectrum)
            return F.softmax(logits, dim=-1)

    def load_balance_loss(self, raw_input):
        spectrum = self._get_spectrum(raw_input)
        logits = self.router(spectrum)
        weights = F.softmax(logits, dim=-1)
        f_i = weights.mean(dim=0)
        p_i = F.softmax(logits, dim=-1).mean(dim=0)
        return self.K * (f_i * p_i).sum()

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def train_sr_moa(model, blocks, X_train, Y_train, X_test, Y_test,
                 device="cuda", n_epochs=15, forecast_horizon=96, batch_size=128,
                 backbone_type="moment", K=5, hidden=64):
    """Train SR-MoA."""
    hdim = _get_hidden_dim(model)
    adapter = SpectralRoutedMoA(hdim, forecast_horizon, K=K, hidden=hidden).to(device)

    trainable = list(adapter.parameters())
    pids = {id(p) for p in trainable}
    for p in model.parameters():
        if p.requires_grad and id(p) not in pids:
            trainable.append(p); pids.add(id(p))

    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train(); adapter.train()
        for bx, by in train_loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                pred = adapter(feat, bx_raw)
                loss = mse_fn(pred, by) + adapter.load_balance_coeff * adapter.load_balance_loss(bx_raw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    preds, tgts, all_routing = [], [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx_raw = bx.to(device)
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
            preds.append(adapter(feat, bx_raw).float().cpu())
            tgts.append(by.cpu())
            all_routing.append(adapter.get_routing_stats(bx_raw).cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    routing = torch.cat(all_routing)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()
    mean_routing = routing.mean(dim=0).tolist()
    routing_entropy = -(routing * torch.log(routing + 1e-10)).sum(dim=-1).mean().item()
    routing_max = routing.max(dim=-1).values.mean().item()

    return {
        "mse": mse, "mae": mae, "param_count": adapter.param_count(),
        "routing": {HEAD_NAMES[i]: round(w, 3) for i, w in enumerate(mean_routing[:len(HEAD_NAMES)])},
        "routing_entropy": routing_entropy, "routing_max_weight": routing_max,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/sr_moa", exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks)-4), len(blocks)):
        for p in blocks[i].parameters():
            p.requires_grad = True

    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]

    print("SR-MoA: %s K=%d seed=%d" % (args.dataset, args.K, args.seed))
    start = time.time()
    result = train_sr_moa(model, blocks, X_train, Y_train, X_test, Y_test,
                          device=args.device, forecast_horizon=args.horizon,
                          backbone_type=bb_type, K=args.K)
    elapsed = time.time() - start

    print("SR-MoA: MSE=%.4f  params=%d  time=%.0fs" % (result["mse"], result["param_count"], elapsed))
    print("Routing: %s  entropy=%.3f  max=%.3f" % (
        result["routing"], result["routing_entropy"], result["routing_max_weight"]))

    # Baselines
    model2 = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model2)
    blocks2 = _get_encoder_blocks(model2)
    for p in model2.parameters():
        p.requires_grad = False
    for i in range(max(0, len(blocks2)-4), len(blocks2)):
        for p in blocks2[i].parameters():
            p.requires_grad = True

    baselines = {"linear": SEED_ADAPTERS[0], "attention": SEED_ADAPTERS[3], "conv": SEED_ADAPTERS[4]}
    baseline_results = {}
    for name, code in baselines.items():
        try:
            tr = train_adapter(code, model2, blocks2, X_train, Y_train, X_test, Y_test,
                               device=args.device, n_epochs=15, forecast_horizon=args.horizon,
                               backbone_type=bb_type)
            baseline_results[name] = tr
        except Exception:
            pass

    best_bl = min(baseline_results.values(), key=lambda x: x["mse"])["mse"]
    best_bl_name = min(baseline_results, key=lambda k: baseline_results[k]["mse"])
    delta = (result["mse"] - best_bl) / best_bl * 100
    winner = "SR-MoA" if result["mse"] < best_bl else "BASELINE"
    print(">>> %s: SR-MoA=%.4f vs %s=%.4f  delta=%+.1f%%" % (winner, result["mse"], best_bl_name, best_bl, delta))

    save_data = {
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed, "K": args.K,
        "sr_moa": result, "elapsed": elapsed,
        "baselines": {k: v for k, v in baseline_results.items()},
        "winner": winner, "delta_pct": delta,
    }
    path = "results/sr_moa/%s_H%d_K%d_%d.json" % (args.dataset, args.horizon, args.K, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
