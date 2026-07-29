"""Is (M,Sigma) decodable from the hidden states the router reads? (Pm4m round 2)

Pm4m asks which claim is correct:
  main text  : "non-learnable I/O scaling PRESERVES the (M,Sigma) statistics in
                the hidden states the router reads" (regime (ii) not vulnerable)
  our W3 reply: Moirai's non-learnable normalization "also STRIPS per-window
                location-scale from the hidden states"

Neither was ever measured. This measures it directly, with no training involved.

Protocol: run each frozen backbone forward, take exactly the tensor a
hidden-state router consumes (run_adamix.py:185 -> hidden_states.mean(dim=1)),
and ridge-regress the window's own (mu, log sigma) out of it. Held-out R^2 is
the decodability of the statistics from the router's input.

Controls that make the number interpretable:
  * raw window            -> R^2 must be ~1.0 (the statistics are its own moments)
  * MOMENT with RevIN off -> upper bound for this backbone with normalization removed
  * Chronos / Timer-XL    -> no instance normalization, so nothing should be removed

Reading: high R^2 on Moirai with low R^2 on MOMENT vindicates the main text.
Both low means the main-text clause is wrong and needs correcting in revision.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from feasibility.model import (
    load_backbone, _get_encoder_blocks, _disable_gradient_checkpointing,
)
from feasibility.standard_data import load_standard_data, _detect_backbone_type
from feasibility.finetune import _extract_features_batch


def _ridge_r2(F_tr, y_tr, F_te, y_te, alphas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Ridge with alpha picked on a validation slice; returns held-out R^2.

    Standardized features, closed-form solve. Sweeping alpha means a low R^2
    cannot be blamed on a badly chosen regularizer.
    """
    mu, sd = F_tr.mean(0, keepdims=True), F_tr.std(0, keepdims=True) + 1e-8
    A, B = (F_tr - mu) / sd, (F_te - mu) / sd
    n_val = max(1, int(0.2 * A.shape[0]))
    A_fit, y_fit, A_val, y_val = A[:-n_val], y_tr[:-n_val], A[-n_val:], y_tr[-n_val:]

    def _fit(X, y, alpha):
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        G = X1.T @ X1 + alpha * np.eye(X1.shape[1])
        return np.linalg.solve(G, X1.T @ y)

    def _pred(X, w):
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ w

    best, best_a = -np.inf, alphas[0]
    for a in alphas:
        w = _fit(A_fit, y_fit, a)
        r2 = 1.0 - ((y_val - _pred(A_val, w)) ** 2).sum() / (
            ((y_val - y_val.mean()) ** 2).sum() + 1e-12)
        if r2 > best:
            best, best_a = r2, a
    w = _fit(A, y_tr, best_a)
    pred = _pred(B, w)
    return float(1.0 - ((y_te - pred) ** 2).sum() / (
        ((y_te - y_te.mean()) ** 2).sum() + 1e-12)), float(best_a)


def _pooled_hidden(model, blocks, X, backbone_type, device, batch_size=64):
    """Exactly the router's input: last-block hidden states, mean-pooled."""
    feats = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            bx = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            bx_enc = bx.unsqueeze(1)
            mask = torch.ones(bx_enc.shape[0], bx_enc.shape[2], device=device)
            h = _extract_features_batch(model, blocks, bx_enc, mask,
                                        backbone_type=backbone_type)
            feats.append(h.float().mean(dim=1).cpu().numpy())
    return np.concatenate(feats, 0)


def run_cell(backbone, dataset="ETTh1", disable_revin=False, n_train=2000,
             n_test=1000, horizon=96, device="cuda", label=None):
    t0 = time.time()
    model = load_backbone(backbone, device, disable_revin=disable_revin)
    _disable_gradient_checkpointing(model)
    blocks = _get_encoder_blocks(model)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    bb = _detect_backbone_type(backbone)

    splits, _ = load_standard_data(dataset, horizon)
    X_train = splits["train"][0][:n_train]
    X_test = splits["test"][0][:n_test]

    # Targets: the window's own location and scale, i.e. what normalization removes.
    def _targets(X):
        return X.mean(axis=1), np.log(X.std(axis=1) + 1e-6)

    mu_tr, ls_tr = _targets(X_train)
    mu_te, ls_te = _targets(X_test)

    H_tr = _pooled_hidden(model, blocks, X_train, bb, device)
    H_te = _pooled_hidden(model, blocks, X_test, bb, device)

    r2_mu, a_mu = _ridge_r2(H_tr, mu_tr, H_te, mu_te)
    r2_ls, a_ls = _ridge_r2(H_tr, ls_tr, H_te, ls_te)
    # Sanity ceiling: decode the same targets from the raw window.
    r2_mu_raw, _ = _ridge_r2(X_train, mu_tr, X_test, mu_te)
    r2_ls_raw, _ = _ridge_r2(X_train, ls_tr, X_test, ls_te)

    # Degeneracy diagnostics: a near-zero R^2 is only meaningful if the features
    # actually carry variance. Constant/collapsed features would fake a "stripped"
    # verdict, so report the feature scale and effective rank alongside it.
    f_std = float(H_tr.std(axis=0).mean())
    sv = np.linalg.svd(H_tr - H_tr.mean(0, keepdims=True), compute_uv=False)
    eff_rank = int((sv > (sv.max() * 1e-3)).sum()) if sv.max() > 0 else 0

    return {
        "feat_std_mean": f_std, "feat_eff_rank": eff_rank,
        "label": label or backbone, "backbone": backbone, "dataset": dataset,
        "disable_revin": bool(disable_revin), "hidden_dim": int(H_tr.shape[1]),
        "r2_mu_hidden": r2_mu, "r2_logsigma_hidden": r2_ls,
        "r2_mu_raw_ceiling": r2_mu_raw, "r2_logsigma_raw_ceiling": r2_ls_raw,
        "alpha_mu": a_mu, "alpha_logsigma": a_ls,
        "n_train": int(H_tr.shape[0]), "n_test": int(H_te.shape[0]),
        "elapsed": time.time() - t0,
    }
