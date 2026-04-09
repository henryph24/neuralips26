"""
k-NN Diagnostic: Is the DLinear gap closable from backbone features alone?

Zero-GPU experiment. Estimates the Bayes-optimal MSE floor for backbone features
by using k-NN regression (a universal approximator) on MOMENT hidden states.

Results:
- k-NN MSE ≈ DLinear MSE → Info IS in backbone features, need better adapter
- k-NN MSE >> DLinear MSE → Info NOT in features, must bypass backbone
- k-NN MSE < DLinear MSE → Backbone adds value beyond raw input
"""

import sys
import time
import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

sys.path.insert(0, ".")

from feasibility.data import load_dataset_multihor
from feasibility.model import load_moment, _get_encoder_blocks
from feasibility.finetune import _extract_features_batch


def extract_all_features(model, encoder_blocks, X, batch_size=32):
    """Extract MOMENT hidden states for all samples."""
    features_all = []
    model.eval()
    for i in range(0, len(X), batch_size):
        batch_x = torch.from_numpy(X[i:i + batch_size]).float().unsqueeze(1)
        input_mask = torch.ones(batch_x.shape[0], 512)
        with torch.no_grad():
            feat = _extract_features_batch(
                model=model,
                encoder_blocks=encoder_blocks,
                batch_x=batch_x,
                input_mask=input_mask,
                backbone_type="moment",
            )
        features_all.append(feat.cpu().numpy())
    return np.concatenate(features_all, axis=0)


def run_diagnostic(dataset_name="ETTh1", forecast_horizon=96):
    print(f"\n{'='*60}")
    print(f"k-NN Diagnostic: {dataset_name} H={forecast_horizon}")
    print(f"{'='*60}")

    # 1. Load data
    t0 = time.time()
    data = load_dataset_multihor(
        dataset_name=dataset_name,
        input_len=512,
        forecast_horizon=forecast_horizon,
        stride=64,
    )
    samples = data["samples"]
    X = samples[:, :512]
    Y = samples[:, 512:]
    print(f"Data loaded: {X.shape[0]} samples, X={X.shape}, Y={Y.shape} ({time.time()-t0:.1f}s)")

    # Train/test split (80/20, same as training code)
    n = len(X)
    n_train = int(0.8 * n)
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    print(f"Split: {n_train} train, {n - n_train} test")

    # 2. Load MOMENT
    t0 = time.time()
    model = load_moment(device="cpu", model_name="AutonLab/MOMENT-1-small")
    encoder_blocks = _get_encoder_blocks(model)
    print(f"MOMENT loaded ({time.time()-t0:.1f}s)")

    # 3. Extract features
    t0 = time.time()
    feat_train = extract_all_features(model, encoder_blocks, X_train)
    feat_test = extract_all_features(model, encoder_blocks, X_test)
    print(f"Features extracted: train={feat_train.shape}, test={feat_test.shape} ({time.time()-t0:.1f}s)")

    # 4. Reduce dimensionality for k-NN (393K dims is too many)
    feat_train_flat = feat_train.reshape(feat_train.shape[0], -1)
    feat_test_flat = feat_test.reshape(feat_test.shape[0], -1)

    # Use mean-pooled features (512,768) -> (768) — same as most adapters
    feat_train_pooled = feat_train.mean(axis=1)  # (n_train, 768)
    feat_test_pooled = feat_test.mean(axis=1)

    # Also try PCA on flattened features
    print("Running PCA on flattened features (512*768 -> 256)...")
    t0 = time.time()
    pca = PCA(n_components=min(256, feat_train_flat.shape[0] - 1), random_state=42)
    feat_train_pca = pca.fit_transform(feat_train_flat)
    feat_test_pca = pca.transform(feat_test_flat)
    print(f"PCA done, explained variance: {pca.explained_variance_ratio_.sum():.3f} ({time.time()-t0:.1f}s)")

    # 5. k-NN regression (multiple settings)
    results = {}

    # 5a. k-NN on mean-pooled features
    for k in [3, 5, 10, 20]:
        t0 = time.time()
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(feat_train_pooled, Y_train)
        Y_pred = knn.predict(feat_test_pooled)
        mse = mean_squared_error(Y_test, Y_pred)
        results[f"knn_pooled_k{k}"] = mse
        print(f"k-NN (pooled, k={k}): MSE = {mse:.6f} ({time.time()-t0:.1f}s)")

    # 5b. k-NN on PCA features
    for k in [3, 5, 10, 20]:
        t0 = time.time()
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(feat_train_pca, Y_train)
        Y_pred = knn.predict(feat_test_pca)
        mse = mean_squared_error(Y_test, Y_pred)
        results[f"knn_pca256_k{k}"] = mse
        print(f"k-NN (PCA-256, k={k}): MSE = {mse:.6f} ({time.time()-t0:.1f}s)")

    # 5c. k-NN on RAW input (baseline — what DLinear sees)
    for k in [3, 5, 10, 20]:
        t0 = time.time()
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        mse = mean_squared_error(Y_test, Y_pred)
        results[f"knn_raw_k{k}"] = mse
        print(f"k-NN (raw input, k={k}): MSE = {mse:.6f} ({time.time()-t0:.1f}s)")

    # 5d. Linear regression on pooled features (= what a linear adapter does)
    from sklearn.linear_model import Ridge
    t0 = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(feat_train_pooled, Y_train)
    Y_pred = ridge.predict(feat_test_pooled)
    mse = mean_squared_error(Y_test, Y_pred)
    results["ridge_pooled"] = mse
    print(f"Ridge (pooled features): MSE = {mse:.6f} ({time.time()-t0:.1f}s)")

    # 5e. Linear regression on raw input (= DLinear equivalent)
    ridge_raw = Ridge(alpha=1.0)
    ridge_raw.fit(X_train, Y_train)
    Y_pred = ridge_raw.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    results["ridge_raw"] = mse
    print(f"Ridge (raw input, ≈DLinear): MSE = {mse:.6f}")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {dataset_name} H={forecast_horizon}")
    print(f"{'='*60}")
    print(f"DLinear reference:     ~0.417 (trained, H=96)")
    print(f"Ridge on raw input:     {results['ridge_raw']:.4f}  (≈DLinear)")
    print(f"Ridge on backbone feat: {results['ridge_pooled']:.4f}  (≈frozen linear adapter)")
    best_backbone = min(v for k, v in results.items() if "pooled" in k or "pca" in k)
    best_raw = min(v for k, v in results.items() if "raw" in k)
    print(f"Best k-NN (backbone):   {best_backbone:.4f}")
    print(f"Best k-NN (raw input):  {best_raw:.4f}")
    print()
    if best_backbone < results["ridge_raw"]:
        print(">>> BACKBONE FEATURES CONTAIN MORE INFO THAN RAW INPUT")
        print(">>> A better adapter CAN close the gap")
    elif best_backbone > results["ridge_raw"] * 1.2:
        print(">>> BACKBONE FEATURES ARE WORSE THAN RAW INPUT")
        print(">>> Must bypass backbone (E1/A2) to close the gap")
    else:
        print(">>> BACKBONE FEATURES ≈ RAW INPUT (similar info content)")

    return results


if __name__ == "__main__":
    datasets = ["ETTh1"]
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_diagnostic(ds)
