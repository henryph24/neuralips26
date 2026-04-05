"""Self-verification script for the multi-seed RR-MoA ablation evidence.

Re-reads every JSON result file in evidence_vm/{rr_moa,adamix}/ and
recomputes the per-dataset mean+-std from the raw seed values, then checks
each number against the values claimed in main.tex Tables 3, 4, and 5.

Exits with code 0 if all values match within 0.005 tolerance. Otherwise
prints a discrepancy report and exits with code 1.

Usage:
    python3 evidence_vm/verify.py
"""

import glob
import json
import os
import sys
from collections import defaultdict

EVID = os.path.dirname(os.path.abspath(__file__))
TOL = 0.005  # tolerance for numeric comparison (MSE values are ~0.1-1.5)
ENT_TOL = 0.01  # tolerance for entropy values

# ----- Paper's claimed numbers (from main.tex Tables 3-5) -----

TAB3_RRMOA = {
    # (dataset, freeze_level) -> (mean, std) RR-MoA test MSE
    ("ETTh1", "frozen"):  (0.690, 0.021),
    ("ETTh1", "last2"):   (0.727, 0.074),
    ("ETTh1", "last4"):   (0.749, 0.036),
    ("ETTm1", "frozen"):  (0.572, 0.073),
    ("ETTm1", "last2"):   (0.623, 0.032),
    ("ETTm1", "last4"):   (0.571, 0.034),
    ("Weather", "frozen"): (0.289, 0.008),
    ("Weather", "last2"):  (0.251, 0.005),
    ("Weather", "last4"):  (0.256, 0.014),
}

TAB3_BASELINE = {
    ("ETTh1", "frozen"):  (1.220, 0.023),
    ("ETTh1", "last2"):   (1.030, 0.139),
    ("ETTh1", "last4"):   (1.101, 0.120),
    ("ETTm1", "frozen"):  (1.169, 0.006),
    ("ETTm1", "last2"):   (0.891, 0.049),
    ("ETTm1", "last4"):   (0.866, 0.016),
    ("Weather", "frozen"): (0.522, 0.003),
    ("Weather", "last2"):  (0.478, 0.025),
    ("Weather", "last4"):  (0.497, 0.033),
}

TAB4_ADAMIX_MSE = {
    ("ETTh1", "frozen"):  (1.105, 0.026),
    ("ETTh1", "last2"):   (1.153, 0.000),
    ("ETTh1", "last4"):   (1.154, 0.001),
    ("ETTm1", "frozen"):  (1.008, 0.012),
    ("ETTm1", "last2"):   (1.061, 0.088),
    ("ETTm1", "last4"):   (1.123, 0.000),
    ("Weather", "frozen"): (0.459, 0.017),
    ("Weather", "last2"):  (0.607, 0.002),
    ("Weather", "last4"):  (0.607, 0.002),
}

TAB4_ADAMIX_ENTROPY = {
    ("ETTh1", "frozen"):  (0.629, 0.436),
    ("ETTh1", "last2"):   (0.000, 0.000),
    ("ETTh1", "last4"):   (0.000, 0.000),
    ("ETTm1", "frozen"):  (0.487, 0.371),
    ("ETTm1", "last2"):   (0.218, 0.309),
    ("ETTm1", "last4"):   (0.000, 0.000),
    ("Weather", "frozen"): (0.509, 0.307),
    ("Weather", "last2"):  (0.000, 0.000),
    ("Weather", "last4"):  (0.000, 0.000),
}

TAB5_TOPK = {
    # top_k -> (mean, std) RR-MoA MSE on ETTh1 last2, 3 seeds
    1: (1.268, 0.072),
    2: (0.727, 0.074),
    3: (0.679, 0.042),
    "dense": (0.550, 0.029),
}


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, var ** 0.5


def close(a, b, tol):
    return abs(a - b) <= tol


def main():
    errors = []
    checks = 0

    # --- Table 3: RR-MoA freeze ablation + baselines ---
    rr_groups = defaultdict(list)
    bl_groups = defaultdict(list)
    for f in sorted(glob.glob(f"{EVID}/rr_moa/*_top2_*_4?.json")):
        d = json.load(open(f))
        if not d.get("baselines"):
            continue
        key = (d["dataset"], d["unfreeze"])
        rr_groups[key].append(d["rr_moa"]["mse"])
        bl_groups[key].append(min(x["mse"] for x in d["baselines"].values()))

    for key, expected in TAB3_RRMOA.items():
        checks += 1
        if key not in rr_groups:
            errors.append(f"Table 3 RR-MoA {key}: NO DATA")
            continue
        got = mean_std(rr_groups[key])
        if not (close(got[0], expected[0], TOL) and close(got[1], expected[1], TOL)):
            errors.append(
                f"Table 3 RR-MoA {key}: paper={expected}, json={got}"
            )

    for key, expected in TAB3_BASELINE.items():
        checks += 1
        if key not in bl_groups:
            errors.append(f"Table 3 baseline {key}: NO DATA")
            continue
        got = mean_std(bl_groups[key])
        if not (close(got[0], expected[0], TOL) and close(got[1], expected[1], TOL)):
            errors.append(
                f"Table 3 baseline {key}: paper={expected}, json={got}"
            )

    # --- Table 4: AdaMix collapse ---
    am_mse = defaultdict(list)
    am_ent = defaultdict(list)
    for f in sorted(glob.glob(f"{EVID}/adamix/*_4?.json")):
        d = json.load(open(f))
        if "unfreeze" not in d:
            continue
        key = (d["dataset"], d["unfreeze"])
        am_mse[key].append(d["adamix"]["mse"])
        am_ent[key].append(d["adamix"]["routing_entropy"])

    for key, expected in TAB4_ADAMIX_MSE.items():
        checks += 1
        if key not in am_mse:
            errors.append(f"Table 4 AdaMix MSE {key}: NO DATA")
            continue
        got = mean_std(am_mse[key])
        if not (close(got[0], expected[0], TOL) and close(got[1], expected[1], TOL)):
            errors.append(
                f"Table 4 AdaMix MSE {key}: paper={expected}, json={got}"
            )

    for key, expected in TAB4_ADAMIX_ENTROPY.items():
        checks += 1
        if key not in am_ent:
            errors.append(f"Table 4 AdaMix entropy {key}: NO DATA")
            continue
        got = mean_std(am_ent[key])
        if not (close(got[0], expected[0], ENT_TOL) and close(got[1], expected[1], ENT_TOL)):
            errors.append(
                f"Table 4 AdaMix entropy {key}: paper={expected}, json={got}"
            )

    # --- Table 5: Top-k ablation on ETTh1 last-2 ---
    topk_groups = defaultdict(list)
    for f in sorted(glob.glob(f"{EVID}/rr_moa/ETTh1_H96_K5_top*_last2_4?.json")):
        d = json.load(open(f))
        topk_groups[d.get("top_k", 5)].append(d["rr_moa"]["mse"])
    for f in sorted(glob.glob(f"{EVID}/rr_moa/ETTh1_H96_K5_dense_last2_4?.json")):
        d = json.load(open(f))
        topk_groups["dense"].append(d["rr_moa"]["mse"])

    for k, expected in TAB5_TOPK.items():
        checks += 1
        if k not in topk_groups:
            errors.append(f"Table 5 Top-{k}: NO DATA")
            continue
        got = mean_std(topk_groups[k])
        if not (close(got[0], expected[0], TOL) and close(got[1], expected[1], TOL)):
            errors.append(f"Table 5 Top-{k}: paper={expected}, json={got}")

    # --- 27/27 wins audit ---
    total = 0
    wins = 0
    for f in sorted(glob.glob(f"{EVID}/rr_moa/*_top2_*_4?.json")):
        d = json.load(open(f))
        if not d.get("baselines"):
            continue
        total += 1
        rr = d["rr_moa"]["mse"]
        bl = min(x["mse"] for x in d["baselines"].values())
        if rr < bl:
            wins += 1
    checks += 1
    if (wins, total) != (27, 27):
        errors.append(f"27/27 wins: got {wins}/{total}")

    # --- Report ---
    print(f"Ran {checks} checks against {len(glob.glob(f'{EVID}/rr_moa/*.json'))} "
          f"RR-MoA + {len(glob.glob(f'{EVID}/adamix/*.json'))} AdaMix JSON files.")
    if errors:
        print(f"FAIL: {len(errors)} discrepancies:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"PASS: all {checks} numeric claims in main.tex Tables 3-5 "
              f"match the raw JSON evidence within tolerance "
              f"(MSE {TOL}, entropy {ENT_TOL}).")
        print(f"RR-MoA wins: {wins}/{total}")
        sys.exit(0)


if __name__ == "__main__":
    main()
