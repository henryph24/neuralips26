"""SMD anomaly-detection mechanism-check (rebuttal task-diversity: jemj Q1 / 8b2Z W5).

RR-MoA reconstructs each length-`win` window on a strictly frozen MOMENT-small
(route on the raw window, experts on frozen hidden states, output = the window).
Trained on the normal `train` split; anomaly score on `test` = per-window
reconstruction MSE. Reports ROC-AUC / PR-AUC and, crucially, the ROUTING ENTROPY
-- the mechanism check: entropy should stay healthy (no collapse), exactly as on
forecasting, showing the anti-collapse mechanism is not forecasting-specific.

Frozen features cap absolute detection quality, so this is framed as
mechanism-generalization, not a SOTA anomaly-detection claim.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from input_len_ablation import (
    load_backbone,
    _get_encoder_blocks,
    _disable_gradient_checkpointing,
    _get_hidden_dim,
    _detect_backbone_type,
    _mask_last_l,
    RawRoutedMoA,
)
from feasibility.finetune import _extract_features_batch

BACKBONE = "AutonLab/MOMENT-1-small"


def _windows(arr, win, stride):
    """arr: (T, C). Each (start, channel) becomes one univariate length-`win` window."""
    T, C = arr.shape
    Xs, starts = [], []
    for s in range(0, T - win + 1, stride):
        for c in range(C):
            Xs.append(arr[s:s + win, c])
            starts.append(s)
    return np.asarray(Xs, np.float32), np.asarray(starts)


def run_anomaly_cell(machine, seed, win=512, stride=256, n_epochs=15, device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    base = f"/root/data/SMD/{machine}"
    tr = np.loadtxt(base + "_train.txt", delimiter=",").astype(np.float32)
    te = np.loadtxt(base + "_test.txt", delimiter=",").astype(np.float32)
    lb = np.loadtxt(base + "_test_label.txt").astype(np.float32)

    Xtr, _ = _windows(tr, win, stride)
    Xte, s_te = _windows(te, win, stride)
    yte = np.array([1.0 if lb[s:s + win].max() > 0 else 0.0 for s in s_te], np.float32)

    model = load_backbone(BACKBONE, device)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False  # strictly frozen
    bb = _detect_backbone_type(BACKBONE)
    hdim = _get_hidden_dim(model)

    adapter = RawRoutedMoA(hdim, win, input_len=win, K=5, hidden=64, top_k=2,
                           router_input_mode="raw").to(device)
    opt = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=1e-3)
    mse = nn.MSELoss()
    use_amp = device == "cuda"

    trl = DataLoader(TensorDataset(torch.from_numpy(Xtr).float()), batch_size=128, shuffle=True)
    t0 = time.time()
    for _ in range(n_epochs):
        model.train(); adapter.train()
        for (bx,) in trl:
            bx = bx.to(device)                       # (B, win) reconstruct itself
            bx_enc = bx.unsqueeze(1)                 # (B, 1, win)
            mask = _mask_last_l(bx.shape[0], win, win, device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=bb)
                pred = adapter(feat, bx)
                loss = mse(pred, bx) + adapter.load_balance_coeff * adapter.load_balance_loss(bx, hidden_states=feat)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); adapter.eval()
    tel = DataLoader(TensorDataset(torch.from_numpy(Xte).float()), batch_size=128)
    scores, routing = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
        for (bx,) in tel:
            bx = bx.to(device); bx_enc = bx.unsqueeze(1)
            mask = _mask_last_l(bx.shape[0], win, win, device)
            feat = _extract_features_batch(model, blocks, bx_enc, mask, backbone_type=bb)
            pred = adapter(feat, bx)
            scores.append(((pred.float() - bx.float()) ** 2).mean(dim=1).cpu())
            routing.append(adapter.get_routing_stats(bx, hidden_states=feat).cpu())
    scores = torch.cat(scores).numpy()
    routing = torch.cat(routing)
    entropy = -(routing * torch.log(routing + 1e-10)).sum(-1).mean().item()
    max_w = routing.max(dim=-1).values.mean().item()

    two_class = yte.min() != yte.max()
    return {
        "exp": "anomaly", "machine": machine, "seed": int(seed), "win": win, "stride": stride,
        "roc_auc": float(roc_auc_score(yte, scores)) if two_class else float("nan"),
        "pr_auc": float(average_precision_score(yte, scores)) if two_class else float("nan"),
        "routing_entropy": entropy, "routing_max_weight": max_w,
        "n_test_windows": int(len(scores)), "anom_frac": float(yte.mean()),
        "param_count": adapter.param_count(), "elapsed": time.time() - t0,
    }
