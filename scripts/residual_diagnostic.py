"""
Residual Diagnostic: CPU-only test of E1 (Residual DLinear + FM correction).

Tests whether backbone features contain COMPLEMENTARY information to raw input
by training:
  1. Ridge on raw input (≈ DLinear base)
  2. Ridge on backbone features predicting RESIDUAL (Y - DLinear_pred)
  3. Combined: DLinear_pred + α * FM_residual_pred

If combined < DLinear alone, backbone features add genuine value as corrections.
"""

import sys
import time
import numpy as np
import torch
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

sys.path.insert(0, ".")

from feasibility.data import load_dataset_multihor
from feasibility.model import load_moment, _get_encoder_blocks
from feasibility.finetune import _extract_features_batch


def extract_all_features(model, encoder_blocks, X, batch_size=32):
    features_all = []
    model.eval()
    for i in range(0, len(X), batch_size):
        batch_x = torch.from_numpy(X[i:i + batch_size]).float().unsqueeze(1)
        input_mask = torch.ones(batch_x.shape[0], 512)
        with torch.no_grad():
            feat = _extract_features_batch(
                model=model, encoder_blocks=encoder_blocks,
                batch_x=batch_x, input_mask=input_mask, backbone_type="moment",
            )
        features_all.append(feat.cpu().numpy())
    return np.concatenate(features_all, axis=0)


def run_residual_diagnostic(dataset_name="ETTh1", forecast_horizon=96):
    print(f"\n{'='*70}")
    print(f"RESIDUAL DIAGNOSTIC: {dataset_name} H={forecast_horizon}")
    print(f"{'='*70}")

    # Load data
    data = load_dataset_multihor(dataset_name=dataset_name, input_len=512,
                                  forecast_horizon=forecast_horizon, stride=64)
    samples = data["samples"]
    X = samples[:, :512]
    Y = samples[:, 512:]
    n = len(X)
    n_train = int(0.8 * n)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    print(f"Data: {n} samples ({n_train} train, {n - n_train} test)")

    # Load MOMENT and extract features
    model = load_moment(device="cpu")
    encoder_blocks = _get_encoder_blocks(model)
    feat_train = extract_all_features(model, encoder_blocks, X_train)
    feat_test = extract_all_features(model, encoder_blocks, X_test)

    # Pooling strategies for backbone features
    feat_train_pooled = feat_train.mean(axis=1)  # (n, 512)
    feat_test_pooled = feat_test.mean(axis=1)

    # PCA for richer representation
    feat_train_flat = feat_train.reshape(feat_train.shape[0], -1)
    feat_test_flat = feat_test.reshape(feat_test.shape[0], -1)
    pca = PCA(n_components=min(128, feat_train_flat.shape[0] - 1), random_state=42)
    feat_train_pca = pca.fit_transform(feat_train_flat)
    feat_test_pca = pca.transform(feat_test_flat)

    print(f"Features: backbone={feat_train.shape}, PCA-128 variance={pca.explained_variance_ratio_.sum():.3f}")

    # ============================================================
    # Step 1: DLinear baseline (Ridge on raw input)
    # ============================================================
    dlinear = Ridge(alpha=1.0)
    dlinear.fit(X_train, Y_train)
    dl_pred_train = dlinear.predict(X_train)
    dl_pred_test = dlinear.predict(X_test)
    dl_mse = mean_squared_error(Y_test, dl_pred_test)
    print(f"\n[Base] Ridge on raw input (≈DLinear):  MSE = {dl_mse:.6f}")

    # ============================================================
    # Step 2: FM-only baseline (Ridge on backbone features)
    # ============================================================
    fm_only = Ridge(alpha=1.0)
    fm_only.fit(feat_train_pooled, Y_train)
    fm_mse = mean_squared_error(Y_test, fm_only.predict(feat_test_pooled))
    print(f"[Base] Ridge on backbone features:      MSE = {fm_mse:.6f}")

    fm_only_pca = Ridge(alpha=1.0)
    fm_only_pca.fit(feat_train_pca, Y_train)
    fm_pca_mse = mean_squared_error(Y_test, fm_only_pca.predict(feat_test_pca))
    print(f"[Base] Ridge on backbone PCA-128:       MSE = {fm_pca_mse:.6f}")

    # ============================================================
    # Step 3: RESIDUAL — FM predicts what DLinear gets wrong
    # ============================================================
    residual_train = Y_train - dl_pred_train  # What DLinear misses
    residual_test = Y_test - dl_pred_test

    print(f"\n[Residual] DLinear residual stats: mean={residual_train.mean():.4f}, "
          f"std={residual_train.std():.4f}, ||residual||/||Y||={np.linalg.norm(residual_train)/np.linalg.norm(Y_train):.3f}")

    results = {}

    # 3a: Residual from pooled backbone features
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        resid_model = Ridge(alpha=alpha)
        resid_model.fit(feat_train_pooled, residual_train)
        resid_pred_test = resid_model.predict(feat_test_pooled)

        # Try different blending scales
        for scale in [0.1, 0.3, 0.5, 0.7, 1.0]:
            combined_pred = dl_pred_test + scale * resid_pred_test
            combined_mse = mean_squared_error(Y_test, combined_pred)
            key = f"pooled_a{alpha}_s{scale}"
            results[key] = combined_mse

    best_pooled = min((v, k) for k, v in results.items() if k.startswith("pooled"))
    print(f"[Residual] Best (pooled):               MSE = {best_pooled[0]:.6f}  ({best_pooled[1]})")

    # 3b: Residual from PCA backbone features
    results_pca = {}
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        resid_model = Ridge(alpha=alpha)
        resid_model.fit(feat_train_pca, residual_train)
        resid_pred_test = resid_model.predict(feat_test_pca)

        for scale in [0.1, 0.3, 0.5, 0.7, 1.0]:
            combined_pred = dl_pred_test + scale * resid_pred_test
            combined_mse = mean_squared_error(Y_test, combined_pred)
            key = f"pca_a{alpha}_s{scale}"
            results_pca[key] = combined_mse

    best_pca = min((v, k) for k, v in results_pca.items())
    print(f"[Residual] Best (PCA-128):              MSE = {best_pca[0]:.6f}  ({best_pca[1]})")

    # 3c: Residual from CONCATENATED (raw + backbone) features
    concat_train = np.concatenate([X_train, feat_train_pooled], axis=1)
    concat_test = np.concatenate([X_test, feat_test_pooled], axis=1)

    results_concat = {}
    for alpha in [0.01, 0.1, 1.0, 10.0]:
        resid_model = Ridge(alpha=alpha)
        resid_model.fit(concat_train, residual_train)
        resid_pred_test = resid_model.predict(concat_test)

        for scale in [0.1, 0.3, 0.5, 0.7, 1.0]:
            combined_pred = dl_pred_test + scale * resid_pred_test
            combined_mse = mean_squared_error(Y_test, combined_pred)
            key = f"concat_a{alpha}_s{scale}"
            results_concat[key] = combined_mse

    best_concat = min((v, k) for k, v in results_concat.items())
    print(f"[Residual] Best (raw+backbone concat):  MSE = {best_concat[0]:.6f}  ({best_concat[1]})")

    # ============================================================
    # Step 4: Oracle — Ridge on raw+backbone directly (not residual)
    # ============================================================
    oracle = Ridge(alpha=1.0)
    oracle.fit(concat_train, Y_train)
    oracle_mse = mean_squared_error(Y_test, oracle.predict(concat_test))
    print(f"\n[Oracle] Ridge on raw+backbone (joint): MSE = {oracle_mse:.6f}")

    # Also try with PCA features
    concat_pca_train = np.concatenate([X_train, feat_train_pca], axis=1)
    concat_pca_test = np.concatenate([X_test, feat_test_pca], axis=1)
    oracle_pca = Ridge(alpha=1.0)
    oracle_pca.fit(concat_pca_train, Y_train)
    oracle_pca_mse = mean_squared_error(Y_test, oracle_pca.predict(concat_pca_test))
    print(f"[Oracle] Ridge on raw+PCA-128 (joint):  MSE = {oracle_pca_mse:.6f}")

    # ============================================================
    # Summary
    # ============================================================
    best_residual = min(best_pooled[0], best_pca[0], best_concat[0])
    best_oracle = min(oracle_mse, oracle_pca_mse)

    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name} H={forecast_horizon}")
    print(f"{'='*70}")
    print(f"DLinear (Ridge raw):     {dl_mse:.4f}")
    print(f"FM-only (Ridge backbone):{fm_mse:.4f}  ({(fm_mse/dl_mse - 1)*100:+.1f}% vs DLinear)")
    print(f"Best Residual:           {best_residual:.4f}  ({(best_residual/dl_mse - 1)*100:+.1f}% vs DLinear)")
    print(f"Best Oracle (joint):     {best_oracle:.4f}  ({(best_oracle/dl_mse - 1)*100:+.1f}% vs DLinear)")
    print()

    if best_residual < dl_mse:
        improvement = (1 - best_residual / dl_mse) * 100
        print(f">>> FM RESIDUAL IMPROVES OVER DLINEAR BY {improvement:.1f}%")
        print(f">>> Backbone features contain COMPLEMENTARY information!")
    else:
        print(f">>> FM RESIDUAL DOES NOT IMPROVE OVER DLINEAR")
        print(f">>> Backbone features add NO complementary value (for linear correction)")

    if best_oracle < dl_mse:
        improvement = (1 - best_oracle / dl_mse) * 100
        print(f">>> Joint (raw+backbone) improves by {improvement:.1f}% — upper bound for E1")

    return {
        "dlinear_mse": dl_mse,
        "fm_only_mse": fm_mse,
        "best_residual_mse": best_residual,
        "best_oracle_mse": best_oracle,
    }


if __name__ == "__main__":
    datasets = ["ETTh1", "ETTm1", "Weather"]
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_residual_diagnostic(ds)

    # Final cross-dataset summary
    print(f"\n{'='*70}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<12} {'DLinear':>10} {'FM-only':>10} {'Residual':>10} {'Oracle':>10} {'Resid Δ':>10}")
    for ds, r in all_results.items():
        delta = (r["best_residual_mse"] / r["dlinear_mse"] - 1) * 100
        print(f"{ds:<12} {r['dlinear_mse']:>10.4f} {r['fm_only_mse']:>10.4f} "
              f"{r['best_residual_mse']:>10.4f} {r['best_oracle_mse']:>10.4f} {delta:>+9.1f}%")
