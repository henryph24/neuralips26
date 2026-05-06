"""Learnable normalization dose: alpha as nn.Parameter.

Tests whether gradient descent discovers the optimal normalization level
per dataset, and whether the learned alpha correlates with R(D).

If alpha stays near 0 on high-R(D) datasets (where raw routing is optimal)
and drifts positive on low-R(D) datasets, the model autonomously discovers
Proposition 2's prediction.

Usage:
    python scripts/run_learnable_alpha.py --dataset ETTh1 --seed 42
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
from feasibility.standard_data import load_standard_data, _detect_backbone_type
from scripts.run_rr_moa import HEAD_CLASSES, HEAD_NAMES


class LearnableAlphaRRMoA(nn.Module):
    """RR-MoA with learnable normalization dose alpha."""
    def __init__(self, d_model, output_dim, input_len=512, K=5, hidden=64):
        super().__init__()
        self.K = K
        self.output_dim = output_dim
        self.adapters = nn.ModuleList([
            HEAD_CLASSES[i % len(HEAD_CLASSES)](d_model, output_dim, hidden)
            for i in range(K)
        ])
        self.router = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=32, stride=16, padding=8),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.router_head = nn.Linear(64, K)
        # Learnable alpha: initialized at 0 (pure raw)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        self.load_balance_coeff = 0.01

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def _compute_logits(self, raw_input):
        alpha = self.alpha
        mu = raw_input.mean(dim=-1, keepdim=True)
        sigma = raw_input.std(dim=-1, keepdim=True) + 1e-5
        revin = (raw_input - mu) / sigma
        x = (1 - alpha) * raw_input + alpha * revin
        x = x.unsqueeze(1)
        router_feat = self.router(x).flatten(1)
        return self.router_head(router_feat)

    def forward(self, hidden_states, raw_input):
        logits = self._compute_logits(raw_input)
        weights = F.softmax(logits, dim=-1)
        outputs = torch.stack([a(hidden_states) for a in self.adapters], dim=1)
        return (weights.unsqueeze(-1) * outputs).sum(dim=1)

    def load_balance_loss(self, raw_input):
        logits = self._compute_logits(raw_input)
        weights = F.softmax(logits, dim=-1)
        f_i = weights.mean(dim=0)
        return self.K * (f_i * F.softmax(logits, dim=-1).mean(dim=0)).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/learnable_alpha", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = load_backbone(args.backbone, args.device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    for p in model.parameters():
        p.requires_grad = False  # frozen

    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]

    adapter = LearnableAlphaRRMoA(hdim, args.horizon, input_len=512, K=5).to(args.device)
    trainable = list(adapter.parameters())
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = args.device == "cuda"

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=128, shuffle=True)

    # Track alpha trajectory
    alpha_trajectory = []

    for epoch in range(args.epochs):
        model.train(); adapter.train()
        for bx, by in train_loader:
            bx_raw = bx.to(args.device)
            bx_enc = bx.to(args.device).unsqueeze(1)
            by = by.to(args.device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=args.device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=bb_type)
                pred = adapter(feat, bx_raw)
                loss = mse_fn(pred, by)
                loss = loss + adapter.load_balance_coeff * adapter.load_balance_loss(bx_raw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        alpha_val = adapter.alpha.item()
        alpha_trajectory.append({"epoch": epoch, "alpha": alpha_val})
        print("  Epoch %d: alpha=%.4f (logit=%.4f)" % (epoch, alpha_val, adapter.alpha_logit.item()))

    # Evaluate
    model.eval(); adapter.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=128)

    preds, tgts = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx_raw = bx.to(args.device)
            bx_enc = bx.to(args.device).unsqueeze(1)
            by = by.to(args.device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=args.device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=bb_type)
            preds.append(adapter(feat, bx_raw).float().cpu())
            tgts.append(by.cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    mse = nn.MSELoss()(preds, tgts).item()
    final_alpha = adapter.alpha.item()

    print("\n%s: MSE=%.4f  final_alpha=%.4f" % (args.dataset, mse, final_alpha))

    save_data = {
        "dataset": args.dataset, "horizon": args.horizon, "seed": args.seed,
        "mse": mse,
        "final_alpha": final_alpha,
        "alpha_logit": adapter.alpha_logit.item(),
        "alpha_trajectory": alpha_trajectory,
    }
    path = "results/learnable_alpha/%s_H%d_%d.json" % (args.dataset, args.horizon, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
