"""Self-verification script for the multi-seed RR-MoA ablation evidence.

Re-reads every JSON result file in evidence_vm/{rr_moa,adamix}/ and
recomputes the per-dataset mean+-std from the raw seed values, then checks
each number against the values claimed in main.tex Tables 3, 4, and 5.

Additionally, checks arithmetic consistency of derived percentages across
main.tex Tables 4 (baselines), 14 (horizon DLinear gap), and 26 (cross-
backbone). For each "{-XX\\%}" style cell, recomputes the percentage from
the neighboring RR-MoA / baseline MSE values and flags any cell whose
claimed percentage disagrees with the computed value by more than 1pp.
This catches the class of "stale percentage that no longer matches the
updated cell value" errors (e.g. the -35% vs -32% ETTh1 MOMENT-large
discrepancy identified by the 2026-04-10 audit).

Exits with code 0 if all values match within 0.005 tolerance. Otherwise
prints a discrepancy report and exits with code 1.

Usage:
    python3 evidence_vm/verify.py
"""

import glob
import json
import os
import re
import sys
from collections import defaultdict

EVID = os.path.dirname(os.path.abspath(__file__))
TOL = 0.005  # tolerance for numeric comparison (MSE values are ~0.1-1.5)
ENT_TOL = 0.01  # tolerance for entropy values
PCT_TOL = 1.0  # tolerance for percentage arithmetic (1 percentage point)

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

# Table 2 green rows: RevIN ablation (no_revin, last4)
TAB2_NOREVIN_MSE = {
    ("ETTh1", "last4"):  (0.575, 0.075),
    ("ETTm1", "last4"):  (0.426, 0.072),
    ("Weather", "last4"): (0.232, 0.045),
}
TAB2_NOREVIN_ENTROPY = {
    ("ETTh1", "last4"):  (1.315, 0.148),
    ("ETTm1", "last4"):  (1.103, 0.376),
    ("Weather", "last4"): (0.661, 0.501),
}

TAB5_TOPK = {
    # top_k -> (mean, std) RR-MoA MSE on ETTh1 last2, 3 seeds
    1: (1.268, 0.072),
    2: (0.727, 0.074),
    3: (0.679, 0.042),
    "dense": (0.550, 0.029),
}

# --- Cross-backbone table (tab:backbone in main.tex, lines 1686-1697) ---
# Each cell is (rr_moa_mse, best_fixed_mse, claimed_percentage_improvement).
# check_percentage() verifies (fixed - rr_moa) / fixed * 100 == claimed_pct.
TAB_BACKBONE_PCT = {
    ("ETTh1", "moment-small"):   (0.690, 1.220, 43),
    ("ETTh1", "moment-large"):   (0.803, 1.173, 32),  # was -35 pre-audit (2026-04-10)
    ("ETTh1", "moirai"):         (0.471, 0.664, 29),
    ("ETTm1", "moment-small"):   (0.572, 1.169, 51),
    ("ETTm1", "moment-large"):   (0.704, 1.126, 38),
    ("ETTm1", "moirai"):         (0.396, 0.471, 16),
    ("Weather", "moment-small"): (0.289, 0.522, 45),
    ("Weather", "moment-large"): (0.267, 0.606, 56),
    ("Weather", "moirai"):       (0.209, 0.238, 12),
}

# --- Multi-horizon DLinear-gap narrative (tab:horizon caption + body text) ---
# Each cell is (rr_moa_mse, dlinear_mse, claimed_percentage_gap).
# check_percentage() verifies (rr_moa - dlinear) / dlinear * 100 == claimed_pct.
# Only the cells explicitly named in the body text at lines 510 and 958 are
# checked here; the full 12-cell horizon grid is out of scope.
TAB_HORIZON_GAP = {
    ("ETTh1",   96):  (0.680, 0.420, 62),   # 5-seed update (was 0.690/0.417/66)
    ("ETTh1",  720):  (0.816, 0.553, 48),   # 5-seed update (was 0.838/0.566/48)
    ("Weather", 96):  (0.276, 0.207, 33),   # 5-seed update (was 0.329/0.208/58)
    ("Weather",720):  (0.401, 0.350, 15),   # 5-seed update (was 0.400/0.350/14)
}

# --- Baseline comparison LoRA row (tab:baselines, line 343) ---
# The full 108-run sweep's best config per dataset, copied from the bolded
# rows of tab:lora_sweep (appendix, lines 1212-1254). These are the values
# that MUST appear in the main-text Best LoRA row. The check greps main.tex
# directly to ensure the numbers haven't drifted.
TAB_BASELINES_LORA = {
    "ETTh1":   1.154,  # was 1.135 pre-audit (2026-04-10)
    "ETTm1":   0.956,  # was 0.895 pre-audit
    "Weather": 0.600,  # was 0.575 pre-audit
}


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    return mu, var ** 0.5


def close(a, b, tol):
    return abs(a - b) <= tol


def check_improvement_pct(larger, smaller, claimed_pct):
    """Recompute (larger - smaller) / larger * 100 and compare to claimed.

    Used for improvement cells where a method beats a baseline: RR-MoA MSE
    (smaller) vs fixed-baseline MSE (larger). Returns the computed percentage
    so callers can include the actual value in the error report.
    """
    actual = (larger - smaller) / larger * 100
    return actual, close(actual, claimed_pct, PCT_TOL)


def check_gap_pct(higher, lower, claimed_pct):
    """Recompute (higher - lower) / lower * 100 for gap-to-baseline claims.

    Used when a method is WORSE than a reference (e.g. RR-MoA vs DLinear):
    higher = RR-MoA MSE, lower = DLinear MSE, gap = (higher - lower) / lower.
    """
    actual = (higher - lower) / lower * 100
    return actual, close(actual, claimed_pct, PCT_TOL)


def grep_main_tex_lora_row():
    """Extract the three ETTh1/ETTm1/Weather values from the Best-LoRA row
    of tab:baselines in main.tex. Returns dict of dataset -> mean, or None
    if the regex does not match (caller reports failure).
    """
    main_tex = os.path.join(os.path.dirname(EVID), "main.tex")
    if not os.path.exists(main_tex):
        return None
    with open(main_tex) as f:
        content = f.read()
    # Line 343 format:
    #   Best LoRA (108-run sweep) & $X.XXX \pm Y.YYY$ & $X.XXX \pm Y.YYY$ & $X.XXX \pm Y.YYY$ & ${<}0.001^{***}$ \\
    m = re.search(
        r"Best LoRA \(108-run sweep\)\s*&\s*"
        r"\$([0-9.]+)\s*\\pm\s*([0-9.]+)\$\s*&\s*"
        r"\$([0-9.]+)\s*\\pm\s*([0-9.]+)\$\s*&\s*"
        r"\$([0-9.]+)\s*\\pm\s*([0-9.]+)\$",
        content,
    )
    if not m:
        return None
    return {
        "ETTh1":   float(m.group(1)),
        "ETTm1":   float(m.group(3)),
        "Weather": float(m.group(5)),
    }


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

    # --- Table 2 green rows: RevIN ablation ---
    nr_mse = defaultdict(list)
    nr_ent = defaultdict(list)
    for f in sorted(glob.glob(f"{EVID}/adamix/*_no_revin.json")):
        d = json.load(open(f))
        key = (d["dataset"], d.get("unfreeze", "last4"))
        nr_mse[key].append(d["adamix"]["mse"])
        nr_ent[key].append(d["adamix"]["routing_entropy"])

    for key, expected in TAB2_NOREVIN_MSE.items():
        checks += 1
        if key not in nr_mse:
            errors.append(f"Table 2 no-RevIN MSE {key}: NO DATA")
            continue
        got = mean_std(nr_mse[key])
        if not (close(got[0], expected[0], TOL) and close(got[1], expected[1], TOL)):
            errors.append(
                f"Table 2 no-RevIN MSE {key}: paper={expected}, json={got}"
            )

    for key, expected in TAB2_NOREVIN_ENTROPY.items():
        checks += 1
        if key not in nr_ent:
            errors.append(f"Table 2 no-RevIN entropy {key}: NO DATA")
            continue
        got = mean_std(nr_ent[key])
        if not (close(got[0], expected[0], ENT_TOL) and close(got[1], expected[1], ENT_TOL)):
            errors.append(
                f"Table 2 no-RevIN entropy {key}: paper={expected}, json={got}"
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

    # --- Cross-backbone table (tab:backbone) arithmetic consistency ---
    # Checks that every claimed {-XX\%} cell matches the actual percentage
    # recomputed from the neighboring rr_moa / best_fixed MSE values.
    for key, (rrmoa, fixed, claimed_pct) in TAB_BACKBONE_PCT.items():
        checks += 1
        actual, ok = check_improvement_pct(fixed, rrmoa, claimed_pct)
        if not ok:
            errors.append(
                f"tab:backbone {key}: claimed -{claimed_pct}%, "
                f"actual -{actual:.1f}% (rr_moa={rrmoa}, fixed={fixed})"
            )

    # --- Multi-horizon DLinear gap claims (body text + tab:horizon caption) ---
    # Body text at line 510 and caption at line 958 both claim specific gap
    # narrowing percentages. Verify each against (rr_moa - dlinear) / dlinear.
    for key, (rrmoa, dlinear, claimed_pct) in TAB_HORIZON_GAP.items():
        checks += 1
        actual, ok = check_gap_pct(rrmoa, dlinear, claimed_pct)
        if not ok:
            errors.append(
                f"tab:horizon {key}: claimed +{claimed_pct}%, "
                f"actual +{actual:.1f}% (rr_moa={rrmoa}, dlinear={dlinear})"
            )

    # --- Baseline comparison LoRA row (tab:baselines, line 343) ---
    # The main-text best-LoRA cells must match the bolded appendix sweep rows
    # (tab:lora_sweep). This is a direct string/value check against main.tex.
    lora_cells = grep_main_tex_lora_row()
    if lora_cells is None:
        checks += 1
        errors.append("tab:baselines: could not locate 'Best LoRA (108-run sweep)' row in main.tex")
    else:
        for ds, expected in TAB_BASELINES_LORA.items():
            checks += 1
            got = lora_cells.get(ds)
            if got is None or not close(got, expected, TOL):
                errors.append(
                    f"tab:baselines LoRA {ds}: expected {expected} (from appendix tab:lora_sweep), "
                    f"main.tex has {got}"
                )

    # --- Report ---
    print(f"Ran {checks} checks against {len(glob.glob(f'{EVID}/rr_moa/*.json'))} "
          f"RR-MoA + {len(glob.glob(f'{EVID}/adamix/*.json'))} AdaMix JSON files.")
    if errors:
        print(f"FAIL: {len(errors)} discrepancies:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"PASS: all {checks} numeric claims in main.tex Tables 3-5, "
              f"tab:baselines LoRA row, tab:horizon DLinear gaps, and "
              f"tab:backbone cross-backbone percentages match within tolerance "
              f"(MSE {TOL}, entropy {ENT_TOL}, pct {PCT_TOL}pp).")
        print(f"RR-MoA wins: {wins}/{total}")
        sys.exit(0)


if __name__ == "__main__":
    main()
