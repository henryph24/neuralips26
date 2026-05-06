"""Unit tests for T1.A: denormalized MSE reporting.

Run with: ``python tests/test_t1a_denorm.py``

Covers:

- ``load_standard_data`` exposes ``_scaler`` and ``<name>_ch`` in the splits dict
  and preserves the ``(X, Y)`` tuple contract for existing callers.
- ``compute_denorm_mse`` returns exactly zero when preds == tgts.
- ``compute_denorm_mse`` with per-sample per-channel scales behaves as the
  element-wise analytical formula ``mean((pred*scale + mean - tgt*scale - mean)**2)``
  = ``mean((pred - tgt)**2 * scale**2)``.
- RawRoutedMoA's three router modes produce distinct behavior; ``uniform``
  yields exact 1/K softmax weights.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feasibility.standard_data import (
    load_standard_data, compute_denorm_mse,
)
from scripts.run_rr_moa import RawRoutedMoA, EXPERT_POOLS


def _assert_close(a, b, tol=1e-5, msg=""):
    if not np.isclose(a, b, atol=tol):
        raise AssertionError("%s expected %s got %s" % (msg, b, a))


def test_load_standard_data_contract():
    splits, n_ch = load_standard_data("ETTh1", forecast_horizon=96, max_samples=100)
    assert n_ch == 7, n_ch
    # Extra metadata keys are present without breaking the old interface.
    for k in ("train", "val", "test", "train_ch", "val_ch", "test_ch", "_scaler"):
        assert k in splits, "missing key %s" % k
    # Existing unpacking pattern still works.
    X_train, Y_train = splits["train"]
    X_test, Y_test = splits["test"]
    assert X_train.shape[1] == 512
    assert Y_train.shape[1] == 96
    # Channel indices align with X/Y by row.
    assert len(splits["test_ch"]) == len(X_test)
    # All channel indices are valid.
    assert splits["test_ch"].min() >= 0
    assert splits["test_ch"].max() < n_ch
    print("  load_standard_data_contract: OK")


def test_compute_denorm_mse_zero_error():
    splits, _ = load_standard_data("ETTh1", forecast_horizon=96, max_samples=100)
    _, Y_test = splits["test"]
    mse_d, mae_d = compute_denorm_mse(Y_test, Y_test, splits["test_ch"], splits["_scaler"])
    _assert_close(mse_d, 0.0, msg="identity denorm MSE")
    _assert_close(mae_d, 0.0, msg="identity denorm MAE")
    print("  compute_denorm_mse_zero_error: OK")


def test_compute_denorm_mse_analytical():
    """Under a constant shift delta in normalized space, the denormalized
    MSE should equal ``delta**2 * mean(scale_test[ch]**2)`` where scale_test
    is aggregated over the test-set channel distribution (NOT the uniform
    mean over channels)."""
    splits, _ = load_standard_data("ETTh1", forecast_horizon=96, max_samples=100)
    _, Y_test = splits["test"]
    ch = splits["test_ch"]
    scaler = splits["_scaler"]
    delta = 0.25
    preds = Y_test + delta
    mse_d, _ = compute_denorm_mse(preds, Y_test, ch, scaler)
    # Analytical: (delta * scale_ch)**2 averaged over all (sample, horizon)
    # positions. Horizon is uniform, so this reduces to delta**2 *
    # mean(scale_ch**2) over samples.
    scales = scaler.scale_[ch]
    expected = float(np.mean((delta * scales) ** 2))
    _assert_close(mse_d, expected, tol=1e-4, msg="analytical denorm MSE")
    print("  compute_denorm_mse_analytical: OK (got %.6f, expected %.6f)" % (mse_d, expected))


def test_rawroutedmoa_uniform_mode():
    model = RawRoutedMoA(d_model=16, output_dim=8, input_len=32, K=5, hidden=8,
                         top_k=None, router_input_mode="uniform")
    B = 4
    raw = torch.randn(B, 32)
    logits = model._compute_logits(raw)
    weights = torch.softmax(logits, dim=-1)
    assert torch.allclose(weights, torch.full((B, 5), 0.2), atol=1e-6), weights
    hidden = torch.randn(B, 10, 16)
    out = model(hidden, raw)
    assert out.shape == (B, 8), out.shape
    print("  rawroutedmoa_uniform_mode: OK")


def test_rawroutedmoa_raw_vs_revin():
    torch.manual_seed(0)
    raw_model = RawRoutedMoA(d_model=16, output_dim=8, input_len=32, K=5, hidden=8,
                             router_input_mode="raw")
    torch.manual_seed(0)
    revin_model = RawRoutedMoA(d_model=16, output_dim=8, input_len=32, K=5, hidden=8,
                               router_input_mode="revin")
    # With matched init, the only difference in _compute_logits is the
    # per-window normalization. A signal with a strong DC offset + varying
    # amplitude should produce different logits under raw vs revin.
    raw = torch.zeros(2, 32)
    raw[0] = 2.0 + 0.01 * torch.randn(32)   # high mean, low amplitude
    raw[1] = -1.0 + 5.0 * torch.randn(32)   # low mean, high amplitude
    raw_logits = raw_model._compute_logits(raw).detach()
    revin_logits = revin_model._compute_logits(raw).detach()
    # They should not be element-wise equal: revin has stripped the
    # per-window mean/variance that raw retains.
    assert not torch.allclose(raw_logits, revin_logits, atol=1e-3), "raw vs revin logits identical"
    print("  rawroutedmoa_raw_vs_revin: OK")


def test_rrmoa_macro_expert_pool():
    """T3.A: macro expert pool must satisfy RR-MoA's expert contract and
    propagate gradients under sparse routing."""
    assert "macro" in EXPERT_POOLS
    model = RawRoutedMoA(d_model=64, output_dim=16, input_len=128, K=5,
                         hidden=32, top_k=2, router_input_mode="raw",
                         expert_pool="macro")
    B = 8
    raw = torch.randn(B, 128)
    # MOMENT's encoder emits (B, T, d_model) hidden states; pick T=16 for speed.
    hidden = torch.randn(B, 16, 64)
    out = model(hidden, raw)
    assert out.shape == (B, 16), out.shape

    # Param count should equal the sum of the 5 individual expert param counts
    # plus the router / router_head. Non-zero.
    total = sum(p.numel() for p in model.parameters())
    expert_sum = sum(sum(p.numel() for p in e.parameters()) for e in model.adapters)
    assert total >= expert_sum, (total, expert_sum)

    # Forward/backward: loss grad should flow to at least 2 experts (top_k=2)
    # and to the router.
    tgt = torch.randn(B, 16)
    loss = torch.nn.functional.mse_loss(out, tgt) + 0.01 * model.load_balance_loss(raw)
    loss.backward()
    n_with_grad = sum(
        1 for e in model.adapters
        if any(p.grad is not None and p.grad.abs().sum() > 0 for p in e.parameters())
    )
    assert n_with_grad >= 2, "expected at least 2 experts with grad, got %d" % n_with_grad

    # The expert names exposed on the model should match MACRO_EXPERT_NAMES.
    from feasibility.rrmoa_macro_experts import MACRO_EXPERT_NAMES
    assert model._expert_names == MACRO_EXPERT_NAMES
    print("  rrmoa_macro_expert_pool: OK (params=%d, experts_with_grad=%d/5)"
          % (total, n_with_grad))


def main():
    print("=== T1.A denorm + RR-MoA mode unit tests ===")
    test_load_standard_data_contract()
    test_compute_denorm_mse_zero_error()
    test_compute_denorm_mse_analytical()
    test_rawroutedmoa_uniform_mode()
    test_rawroutedmoa_raw_vs_revin()
    test_rrmoa_macro_expert_pool()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
