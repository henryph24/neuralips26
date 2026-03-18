"""Wrapper around TEMPLATE transferability score functions."""

import sys
import os
import numpy as np

# Add template_code/utils to path for imports
_TEMPLATE_UTILS = os.path.join(os.path.dirname(__file__), "..", "template_code", "utils")
sys.path.insert(0, _TEMPLATE_UTILS)

from scipy.stats import pearsonr


def _power_iteration(A: np.ndarray, max_iter: int = 100) -> tuple:
    """Power iteration to find dominant eigenvector."""
    n = A.shape[0]
    x = np.random.randn(n)
    x /= np.linalg.norm(x)
    for _ in range(max_iter):
        x = A @ x
        norm = np.linalg.norm(x)
        if norm > 0:
            x /= norm
    eigenvalue = x.T @ A @ x
    return eigenvalue, x


def dl_score(feature: np.ndarray, trend_feature: np.ndarray) -> float:
    """Data Loading score: correlation of dominant eigenvectors between
    feature and trend-feature covariance matrices."""
    def _dominant_eigvec(X):
        if X.shape[0] > X.shape[1]:
            M = X.T @ X
        else:
            M = X @ X.T
        _, vec = _power_iteration(M)
        return vec

    v1 = _dominant_eigvec(feature)
    v2 = _dominant_eigvec(trend_feature)

    # Align lengths if needed (can differ when n > d vs n <= d)
    min_len = min(len(v1), len(v2))
    corr, _ = pearsonr(v1[:min_len], v2[:min_len])
    return float(corr)


def pl_score(feature: np.ndarray) -> float:
    """Power Law score: ratio of max singular value to nuclear norm."""
    if feature.shape[0] > feature.shape[1]:
        M = feature.T @ feature
    else:
        M = feature @ feature.T

    eigenvalues = np.linalg.eigvalsh(M)
    eigenvalues = np.maximum(eigenvalues, 0)
    singular_values = np.sqrt(eigenvalues)

    nuclear_norm = np.sum(singular_values)
    max_sv = np.max(singular_values)

    return float(max_sv / nuclear_norm) if nuclear_norm > 0 else 0.0


def ta_score(feature: np.ndarray, first_feature: np.ndarray, device: str = "auto") -> float:
    """Task Alignment score: kernel CKA between first and last layer features.

    Uses CUDA CKA when available, falls back to linear CKA on CPU.
    """
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        return _ta_score_cuda(feature, first_feature)
    else:
        return _ta_score_cpu(feature, first_feature)


def _ta_score_cuda(feature: np.ndarray, first_feature: np.ndarray) -> float:
    """TA score using kernel CKA on GPU (matches original TEMPLATE)."""
    import torch
    sys.path.insert(0, _TEMPLATE_UTILS)
    from CKA import CudaCKA

    device = torch.device("cuda")
    cuda_cka = CudaCKA(device)

    feat = torch.tensor(feature, dtype=torch.float32).to(device)
    first_feat = torch.tensor(first_feature, dtype=torch.float32).to(device)

    score = cuda_cka.kernel_CKA(first_feat, feat, sigma=None)
    return float(score.cpu().numpy())


def _ta_score_cpu(feature: np.ndarray, first_feature: np.ndarray) -> float:
    """TA score using linear CKA on CPU (fallback for local testing)."""
    sys.path.insert(0, _TEMPLATE_UTILS)
    from CKA import CKA as CKAClass

    cka = CKAClass()
    score = cka.linear_CKA(first_feature, feature)
    return float(score)


def compute_template_scores(
    feature: np.ndarray,
    first_feature: np.ndarray,
    trend_feature: np.ndarray,
    device: str = "auto",
) -> dict:
    """Compute all three TEMPLATE transferability scores.

    Returns dict with keys: dl, pl, ta, composite.
    """
    dl = dl_score(feature, trend_feature)
    pl = pl_score(feature)
    ta = ta_score(feature, first_feature, device=device)

    # Composite: simple average (can be weighted later)
    composite = (abs(dl) + pl + ta) / 3.0

    return {
        "dl": dl,
        "pl": pl,
        "ta": ta,
        "composite": composite,
    }
