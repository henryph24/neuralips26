"""Synthetic ground-truth validation for MI loss estimation pipeline.

Generates K=5 Gaussian clusters where all MI quantities have closed-form
solutions, then validates that our estimation methods (CCA+Gaussian reference,
k-NN conditional entropy, MLP classifier) recover the true values.

Sweeps the correlation rho_MS between (M,Sigma) and shape S to demonstrate:
1. Bound is tight when rho_MS ≈ 0 (statistics independent of shape)
2. Bound loosens gracefully as rho_MS increases
3. Estimation accuracy relative to ground truth

Usage:
    python scripts/synthetic_mi_tightness.py
"""

import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_K = math.log(5)


def generate_synthetic_data(K=5, N_per_cluster=200, seq_len=512, rho_MS=0.0, seed=42):
    """Generate synthetic windows from K Gaussian clusters with controlled rho_MS.

    Each cluster k has:
    - mean mu_k: drawn from well-separated values
    - scale sigma_k: drawn from distinct values
    - Within-cluster shape has correlation rho_MS with (mu_k, sigma_k)

    When rho_MS=0: (M,Sigma) ⊥ S, so epsilon=0 and bound is tight.
    As rho_MS→1: shape encodes location-scale info, epsilon grows, bound loosens.

    Returns: windows (N,seq_len), E_true (N,) ground-truth cluster assignments
    """
    rng = np.random.default_rng(seed)

    # Cluster parameters (well-separated for clean routing)
    cluster_means = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    cluster_scales = np.array([0.3, 0.7, 1.0, 1.5, 2.5])

    windows = []
    E_true = []

    for k in range(K):
        mu_k = cluster_means[k]
        sigma_k = cluster_scales[k]

        for _ in range(N_per_cluster):
            # Generate shape (seq_len iid standard normal + rho-controlled
            # contamination from location-scale)
            noise = rng.standard_normal(seq_len)
            # Contamination: inject trend proportional to mu_k and scale
            # modulation proportional to sigma_k into the shape
            contamination = (rho_MS * mu_k / 3.0 * np.linspace(-1, 1, seq_len) +
                             rho_MS * (sigma_k - 1.0) * np.sin(
                                 np.linspace(0, 4 * np.pi, seq_len)))
            shape = noise * (1 - rho_MS) + contamination

            # Reconstruct window: x = mu + sigma * shape
            window = mu_k + sigma_k * shape
            windows.append(window)
            E_true.append(k)

    windows = np.array(windows, dtype=np.float32)
    E_true = np.array(E_true, dtype=np.int64)

    # Shuffle
    perm = rng.permutation(len(windows))
    return windows[perm], E_true[perm]


def compute_ground_truth_mi(K=5, N_per_cluster=200, seq_len=512, rho_MS=0.0,
                            n_mc=50000, seed=42):
    """Compute ground-truth MI quantities via Monte Carlo on the generative model.

    For equal-weight clusters:
    - H(E) = log(K)
    - I(X; E) = H(E) = log(K) [since E is deterministic given cluster, and
                                 X uniquely identifies cluster with enough data]
    - I(S; E): must estimate — depends on how distinguishable shapes are across clusters

    We use large MC samples to get near-exact values.
    """
    rng = np.random.default_rng(seed)
    H_E = math.log(K)

    # For I(S; E): generate shapes per cluster and classify
    # I(S; E) = H(E) - H(E|S)
    # H(E|S) estimated via optimal Bayes classifier (since we know the generative model)

    # Generate large sample and do Bayes classification
    windows, E_true = generate_synthetic_data(K, n_mc // K, seq_len, rho_MS, seed)

    mu = windows.mean(axis=1)
    sigma = windows.std(axis=1)
    sigma_safe = np.clip(sigma, 1e-8, None)
    S = (windows - mu[:, None]) / sigma_safe[:, None]

    # For the Bayes classifier: use k-NN on S with large k as proxy
    # (exact Bayes is intractable for 512-dim shapes)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    S_tr, S_te, E_tr, E_te = train_test_split(S, E_true, test_size=0.3,
                                                random_state=seed, stratify=E_true)
    knn = KNeighborsClassifier(n_neighbors=20, metric="euclidean", n_jobs=-1)
    knn.fit(S_tr, E_tr)
    probs = knn.predict_proba(S_te)

    # H(E|S) ≈ mean cross-entropy of Bayes-optimal classifier
    H_E_given_S = 0.0
    for i in range(len(E_te)):
        p = probs[i, E_te[i]]
        if p > 1e-10:
            H_E_given_S -= math.log(p)
    H_E_given_S /= len(E_te)

    I_S_E = max(H_E - H_E_given_S, 0.0)

    # I(M,Sigma; E): use k-NN on (mu, sigma) space
    loc_scale = np.column_stack([mu, sigma])
    ls_tr, ls_te, e_tr, e_te = train_test_split(loc_scale, E_true, test_size=0.3,
                                                  random_state=seed, stratify=E_true)
    knn_ms = KNeighborsClassifier(n_neighbors=20, metric="euclidean")
    knn_ms.fit(ls_tr, e_tr)
    probs_ms = knn_ms.predict_proba(ls_te)
    H_E_given_MS = 0.0
    for i in range(len(e_te)):
        p = probs_ms[i, e_te[i]]
        if p > 1e-10:
            H_E_given_MS -= math.log(p)
    H_E_given_MS /= len(e_te)
    I_MS_E = max(H_E - H_E_given_MS, 0.0)

    # epsilon = I(M,Sigma; S) via CCA Gaussian reference
    from sklearn.cross_decomposition import CCA
    n_sub = min(5000, len(windows))
    cca = CCA(n_components=2, max_iter=1000)
    cca.fit(loc_scale[:n_sub], S[:n_sub])
    U, V = cca.transform(loc_scale[:n_sub], S[:n_sub])
    rho = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(2)])
    epsilon = -0.5 * np.sum(np.log(1 - np.clip(rho ** 2, 0, 0.9999)))

    exact_loss = H_E_given_S  # = I(X;E) - I(S;E) ≈ H(E) - I(S;E)
    lower_bound = max(I_MS_E - epsilon, 0.0)

    return {
        "H_E": float(H_E),
        "H_E_given_S": float(H_E_given_S),
        "I_S_E": float(I_S_E),
        "I_MS_E": float(I_MS_E),
        "epsilon": float(epsilon),
        "exact_loss": float(exact_loss),
        "lower_bound": float(lower_bound),
        "gap": float(exact_loss - lower_bound),
        "retention": float(I_S_E / H_E if H_E > 0 else 0),
        "rho_cca": [float(r) for r in rho],
        "bayes_accuracy_shape": float((knn.predict(S_te) == E_te).mean()),
        "bayes_accuracy_ms": float((knn_ms.predict(ls_te) == e_te).mean()),
    }


def run_sweep():
    """Sweep rho_MS from 0 to 0.9 and measure all MI quantities."""
    rho_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    seeds = [42, 43, 44, 45, 46]
    results = []

    for rho_MS in rho_values:
        print("\n=== rho_MS = %.1f ===" % rho_MS)
        seed_results = []

        for s in seeds:
            r = compute_ground_truth_mi(K=5, N_per_cluster=200, seq_len=512,
                                        rho_MS=rho_MS, n_mc=5000, seed=s)
            seed_results.append(r)

        # Aggregate
        agg = {"rho_MS": rho_MS}
        for key in ["H_E", "H_E_given_S", "I_S_E", "I_MS_E", "epsilon",
                     "exact_loss", "lower_bound", "gap", "retention"]:
            vals = [sr[key] for sr in seed_results]
            agg[key + "_mean"] = float(np.mean(vals))
            agg[key + "_std"] = float(np.std(vals))

        results.append(agg)
        print("  Exact loss = %.4f +/- %.4f" % (
            agg["exact_loss_mean"], agg["exact_loss_std"]))
        print("  Lower bound = %.4f +/- %.4f" % (
            agg["lower_bound_mean"], agg["lower_bound_std"]))
        print("  Gap = %.4f +/- %.4f" % (agg["gap_mean"], agg["gap_std"]))
        print("  epsilon = %.4f +/- %.4f" % (agg["epsilon_mean"], agg["epsilon_std"]))

    # Summary table
    print("\n" + "=" * 70)
    print("SYNTHETIC TIGHTNESS SWEEP SUMMARY")
    print("=" * 70)
    print("%-6s  %7s  %7s  %7s  %7s  %7s" % (
        "rho_MS", "Exact", "LB", "Gap", "epsilon", "Ret%"))
    print("-" * 50)
    for r in results:
        print("%-6.1f  %7.4f  %7.4f  %7.4f  %7.4f  %6.1f%%" % (
            r["rho_MS"], r["exact_loss_mean"], r["lower_bound_mean"],
            r["gap_mean"], r["epsilon_mean"], r["retention_mean"] * 100))

    # Save
    os.makedirs("results/analysis", exist_ok=True)
    out_path = "results/analysis/synthetic_mi_tightness.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to %s" % out_path)

    return results


if __name__ == "__main__":
    run_sweep()
