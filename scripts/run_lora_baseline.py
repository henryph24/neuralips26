"""LoRA baseline for the RR-MoA multi-seed comparison.

Trains a LoRA adapter (low-rank perturbations on q,v projections across all
encoder blocks) with a linear forecast head, using the identical train/val/test
protocol as run_rr_moa.py so the numbers are directly comparable.

This script exists because the reviewer asked for a modern PEFT baseline, and
LoRA is the canonical one. The paper's freeze-level narrative holds for LoRA
too: under --unfreeze frozen, only LoRA's low-rank perturbations are trainable;
the rest of the backbone has requires_grad=False.

Usage:
    python scripts/run_lora_baseline.py --dataset ETTh1 --seed 42 --unfreeze frozen
    python scripts/run_lora_baseline.py --dataset ETTm1 --seed 43 --unfreeze frozen --rank 8
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
    _disable_gradient_checkpointing, _apply_unfreeze,
    attach_lora, discover_module_names,
)
from feasibility.finetune import _extract_features_batch
from feasibility.config import AdapterConfig
from scripts.run_standard_evolution import load_standard_data, _detect_backbone_type


class LoRALinearHead(nn.Module):
    """Linear forecast head on top of mean-pooled hidden states."""
    def __init__(self, d_model, horizon):
        super().__init__()
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, hidden_states):
        # hidden_states: (B, T, d_model); mean over T
        return self.fc(hidden_states.mean(dim=1))


def build_lora_model(backbone_name, device, rank, unfreeze):
    """Load MOMENT, attach LoRA to q,v across all blocks, apply freeze policy."""
    model = load_backbone(backbone_name, device)
    _disable_gradient_checkpointing(model)

    # First discover T5 attention module paths, then attach LoRA.
    module_map = discover_module_names(model)

    cfg = AdapterConfig(
        adapter_type="lora",
        lora_rank=rank,
        lora_alpha=rank * 2,
        target_modules_key="qv",
        layer_placement="all",
        unfreeze=unfreeze,
        head_type="linear",
        pooling="mean",
        config_id=f"lora_r{rank}_qv_all_{unfreeze}",
    )
    model = attach_lora(model, cfg, module_map)

    # Freeze everything first, then apply the chosen freeze policy, then turn on LoRA params.
    for p in model.parameters():
        p.requires_grad = False

    # Re-discover encoder blocks on the PEFT-wrapped model and apply unfreeze policy
    blocks = _get_encoder_blocks(model)
    _apply_unfreeze(model, unfreeze)  # no-op for "frozen"

    # Enable LoRA trainable params (matches feasibility/finetune.py pattern)
    lora_params = []
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            lora_params.append(p)

    return model, blocks, lora_params, cfg


def train_lora_baseline(model, blocks, head, X_train, Y_train, X_test, Y_test,
                        trainable_params, device="cuda", n_epochs=15,
                        forecast_horizon=96, batch_size=128, backbone_type="moment"):
    """Training loop mirrors run_rr_moa.py structure."""
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    mse_fn = nn.MSELoss()
    use_amp = device == "cuda"

    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float(),
    ), batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train(); head.train()
        for bx, by in train_loader:
            bx_enc = bx.to(device).unsqueeze(1)  # (B, 1, 512)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
                pred = head(feat)
                loss = mse_fn(pred, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test evaluation
    model.eval(); head.eval()
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float(),
    ), batch_size=batch_size)

    preds, tgts = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for bx, by in test_loader:
            bx_enc = bx.to(device).unsqueeze(1)
            by = by.to(device)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=backbone_type)
            preds.append(head(feat).float().cpu())
            tgts.append(by.cpu())

    preds, tgts = torch.cat(preds), torch.cat(tgts)
    mse = nn.MSELoss()(preds, tgts).item()
    mae = nn.L1Loss()(preds, tgts).item()
    return mse, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ETTh1")
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--unfreeze", default="frozen", choices=["frozen", "last2", "last4", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbone", default="AutonLab/MOMENT-1-small")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs("results/lora_baseline", exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build LoRA-wrapped model
    model, blocks, lora_params, cfg = build_lora_model(
        args.backbone, args.device, args.rank, args.unfreeze,
    )
    hdim = _get_hidden_dim(model)
    bb_type = _detect_backbone_type(args.backbone)

    # Forecast head (trainable)
    head = LoRALinearHead(hdim, args.horizon).to(args.device)
    trainable_params = lora_params + list(head.parameters())

    n_lora = sum(p.numel() for p in lora_params)
    n_head = sum(p.numel() for p in head.parameters())
    n_total_trainable = sum(p.numel() for p in trainable_params)
    print("LoRA r=%d unfreeze=%s: lora_params=%d, head_params=%d, total_trainable=%d" % (
        args.rank, args.unfreeze, n_lora, n_head, n_total_trainable))

    # Data
    splits, _ = load_standard_data(args.dataset, args.horizon)
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    print("%s H=%d: train=%d, test=%d" % (args.dataset, args.horizon, len(X_train), len(X_test)))

    # Train + evaluate
    print("\nLoRA r=%d: %s unfreeze=%s seed=%d" % (args.rank, args.dataset, args.unfreeze, args.seed))
    start = time.time()
    mse, mae = train_lora_baseline(
        model, blocks, head, X_train, Y_train, X_test, Y_test,
        trainable_params, device=args.device, n_epochs=args.epochs,
        forecast_horizon=args.horizon, backbone_type=bb_type,
    )
    elapsed = time.time() - start

    print("LoRA: MSE=%.4f  MAE=%.4f  time=%.0fs" % (mse, mae, elapsed))

    save_data = {
        "dataset": args.dataset,
        "horizon": args.horizon,
        "seed": args.seed,
        "rank": args.rank,
        "unfreeze": args.unfreeze,
        "backbone": args.backbone,
        "lora_mse": mse,
        "lora_mae": mae,
        "lora_params": n_lora,
        "head_params": n_head,
        "total_trainable_params": n_total_trainable,
        "elapsed": elapsed,
        "config_id": cfg.config_id,
    }
    path = "results/lora_baseline/%s_H%d_r%d_%s_%d.json" % (
        args.dataset, args.horizon, args.rank, args.unfreeze, args.seed)
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Saved to %s" % path)


if __name__ == "__main__":
    main()
