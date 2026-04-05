# Experimental Evidence Audit — Multi-Seed RR-MoA Ablation

**Purpose.** This directory contains the raw artifacts generated during the multi-seed
RR-MoA + AdaMix freeze-level ablation reported in the paper. Every number in
Tables 3–5 and Figure 2 of the paper can be traced back to one of these files.
This document is a forensic manifest: it lists the files, their hashes, the VM
where they were produced, and the execution timestamps. It is intended to
rebut any accusation that the experiments were fabricated.

## 1. Execution environment

| Field | Value |
|---|---|
| Host | `ip-10-110-161-2` (AWS, `ap-southeast-2`) |
| Kernel | `Linux 6.8.0-1044-aws x86_64` (Ubuntu 22.04 AMI) |
| GPU | NVIDIA A10G, 23028 MiB, driver 550.144.03, compute capability 8.6 |
| CUDA | 12.4 |
| PyTorch | 2.4.0+cu124 |
| SSH key | `hungphanphd.pem` (RACE VM credentials) |
| VM IP | `ec2-13-238-161-176.ap-southeast-2.compute.amazonaws.com` |

Captured with `nvidia-smi`, `uname -a`, and `torch.cuda.get_device_name(0)`
over SSH at the time the evidence was retrieved (see `system_metadata.txt`).

## 2. Files in this directory

| File | Description | Size |
|---|---|---|
| `multiseed_run.log` | Full stdout of the overnight multi-seed sweep (`scripts/run_multiseed.sh`) | 1,232 lines |
| `freeze_ablation_run.log` | Stdout of the earlier single-seed freeze ablation (`scripts/run_freeze_ablation.py`) | 553 lines |
| `vm_file_listing.txt` | `ls -la --time-style=full-iso` of `results/rr_moa/` and `results/adamix/` on the VM | 108 lines |
| `rr_moa/*.json` | 54 per-experiment JSON result files for RR-MoA (all variants) | — |
| `adamix/*.json` | 45 per-experiment JSON result files for AdaMix collapse control | — |
| `manifest_md5.txt` | MD5 hash of every log and JSON file (for tamper evidence) | 99 lines |
| `FORENSIC_AUDIT.md` | This document | — |

## 3. Experimental runs captured

### 3a. Freeze-level ablation (the 27/27 wins claim)
- **27 RR-MoA experiments** = 3 datasets (ETTh1, ETTm1, Weather) × 3 freeze levels (frozen, last-2, last-4) × 3 seeds (42, 43, 44), each with Top-2 sparse routing and 15 training epochs.
- **27 AdaMix control experiments** = same 3×3×3 grid with hidden-state routing (the collapse control).
- Each RR-MoA run additionally trains three fixed baselines (linear / attention / conv) on the same data and freeze level, stored inside the same JSON.
- Total Python processes launched: 54 (27 RR-MoA + 27 AdaMix), each a separate `python3 scripts/run_rr_moa.py` or `python3 scripts/run_adamix.py` invocation.

### 3b. Top-k sparsity ablation
- **12 RR-MoA experiments** = 4 sparsity levels (k=1, 2, 3, dense K=5) × 3 seeds (42, 43, 44), all on ETTh1 last-2, using `--no-baselines` for speed.

### 3c. Single-seed smoke test (earlier, kept as historical record)
- **18 old-style RR-MoA runs** (6 datasets × 3 seeds at `last-4` unfreeze, dense K=5 — pre-refactor naming convention `*_K5_{seed}.json`). Not used in Tables 3–5 of the current paper, but kept on disk and included here so that the reviewer can see the full file set.

## 4. Execution timeline (VM-side wall clock)

```
Earliest file:  2026-04-03 17:52:08 AEST   (single-seed smoke test start)
Latest file:    2026-04-05 16:26:39 AEST   (re-run of ETTh1 last-2 top-2 for 3 seeds)
Total span:     46.6 hours across 3 calendar days
Files per day:
  2026-04-03 : 18 files   [17:52:08 -> 18:43:30]  ← pre-refactor smoke test
  2026-04-04 : 18 files   [03:12:29 -> 04:03:52]  ← pre-refactor smoke test (second batch)
  2026-04-05 : 67 files   [04:51:54 -> 16:26:39]  ← multi-seed sweep (THIS PAPER)
```

The multi-seed sweep used in the paper started at **04:51:54 AEST on 2026-04-05**
and completed at **07:13:18 AEST** (2h 21min) for the 66 orchestrated runs,
followed by a short re-run at 16:26 to fill in three ETTh1-last-2 files that
were overwritten by the Top-k ablation (filename collision, since both use
`top2`). Every one of these timestamps is visible in the file listing and
correlates with the embedded `[HH:MM:SS]` markers in `multiseed_run.log` (66
such markers, one per experiment launch).

## 5. Paper number ↔ JSON file traceability

| Paper claim | Source file(s) | Verification |
|---|---|---|
| Table 3 ETTh1 frozen RR-MoA = $0.690\pm0.021$, Best fixed $= 1.220\pm0.023$ | `rr_moa/ETTh1_H96_K5_top2_frozen_{42,43,44}.json` | $\mathrm{mean}(0.6671, 0.7175, 0.6850)=0.6899$; $\mathrm{std}=0.0209$ ✓ |
| Table 3 ETTm1 frozen RR-MoA = $0.572\pm0.073$ | `rr_moa/ETTm1_H96_K5_top2_frozen_{42,43,44}.json` | $\mathrm{mean}(0.4965, 0.5473, 0.6706)=0.5715$; $\mathrm{std}=0.0731$ ✓ |
| Table 3 Weather frozen RR-MoA = $0.289\pm0.008$ | `rr_moa/Weather_H96_K5_top2_frozen_{42,43,44}.json` | $\mathrm{mean}(0.2818, 0.2993, 0.2856)=0.2889$; $\mathrm{std}=0.0075$ ✓ |
| Table 4 ETTh1 last-2 AdaMix entropy $= 0.000\pm0.000$ | `adamix/ETTh1_H96_K5_last2_{42,43,44}.json` | routing_entropy = $\{3.0\!\times\!10^{-5}, 1.2\!\times\!10^{-5}, 0.0\}$ ✓ |
| Table 4 ETTh1 last-4 AdaMix entropy $= 0.000\pm0.000$ | `adamix/ETTh1_H96_K5_last4_{42,43,44}.json` | routing_entropy = $\{0.0, 3.5\!\times\!10^{-5}, -0.0\}$ ✓ |
| Table 4 ETTh1 frozen AdaMix entropy $= 0.629\pm0.436$ | `adamix/ETTh1_H96_K5_frozen_{42,43,44}.json` | routing_entropy = $\{1.083, 0.764, 0.040\}$; mean $= 0.629$, std $= 0.436$ ✓ |
| Table 5 Top-2 ETTh1 last-2 $= 0.727\pm0.074$ | `rr_moa/ETTh1_H96_K5_top2_last2_{42,43,44}.json` | $\mathrm{mean}(0.6826, 0.8318, 0.6680)=0.7274$; $\mathrm{std}=0.0740$ ✓ |
| Table 5 Top-1 ETTh1 last-2 collapses | `rr_moa/ETTh1_H96_K5_top1_last2_{42,43,44}.json` | MSE $= \{1.279, 1.348, 1.176\}$; collapses to single expert ✓ |
| Figure 2 ETTh1 routing weights | `rr_moa/ETTh1_H96_K5_top2_frozen_{42,43,44}.json` | mean = $0.210\pm0.045$ (last 0.217, max 0.188, attn 0.180, conv1d 0.205) ✓ |
| **27/27 wins** | All 27 `rr_moa/*_top2_{frozen,last2,last4}_*.json` files | Each file has `"winner": "RR-MoA"` and RR-MoA MSE < min(baseline MSEs) ✓ |

Verification script: `python3 evidence_vm/verify.py` (see next section).

## 6. Self-verification

Run from the repository root:

```bash
python3 evidence_vm/verify.py
```

This script re-reads every JSON file, recomputes the per-dataset mean±std
from the raw seed values, and checks each number against the values in
Tables 3, 4, 5 of `main.tex`. The script exits with code 0 if all values
match within 0.005 tolerance, and prints a discrepancy report otherwise.

## 7. Hash manifest

See `manifest_md5.txt` — MD5 hash of every log file and every JSON file in
this directory. Any modification to any file will change its hash. The logs
and JSONs here are bit-for-bit identical to the ones currently on the VM at
`~/neuralips26/results/{rr_moa,adamix}/`.

## 8. How to reproduce from scratch

On an A10G machine with the repo checked out to commit `337ffa3` or later:

```bash
# One-shot: runs all 66 experiments used in the paper
nohup python3 scripts/run_freeze_ablation.py --experiment all --device cuda \
  --seed 42 --epochs 15 > results/freeze_ablation_run.log 2>&1 &

# Multi-seed sweep (used for Tables 3–5): runs seeds {42,43,44} end-to-end
bash scripts/run_multiseed.sh        # ~2.5 hours on a single A10G
```

Each invocation of `run_rr_moa.py` / `run_adamix.py` is idempotent and writes
a single self-contained JSON file under `results/rr_moa/` or `results/adamix/`
whose name encodes the dataset, horizon, K, sparsity, freeze-level, and seed.
Every JSON contains the full hyperparameter state, the seed, the backbone
trainable parameter count, the final MSE/MAE, the routing weight vector, and
the Shannon entropy of the routing distribution. There is no hidden state
between runs.

---

_Evidence retrieved and audited on 2026-04-05 at the request of the author
in response to a reviewer challenge alleging result fabrication._

---

## Addendum (2026-04-05): LoRA baseline

In response to a second reviewer's request for a modern PEFT baseline, we ran
9 LoRA experiments on the same RACE VM (A10G), strictly frozen backbone,
rank $r{=}8$ on $q,v$ projections across all 8 T5 encoder blocks, linear
forecast head, 15 epochs, seeds $\{42,43,44\}$, datasets $\{$ETTh1, ETTm1,
Weather$\}$. ~10 minutes of wall-clock time total. Raw artifacts:
`lora_baseline/*.json` (9 files). The updated `verify.py` now checks 44
numeric claims (41 previous + 3 new LoRA means) against the JSON evidence
and still exits 0.

Measured LoRA numbers (mean $\pm$ std over 3 seeds, strictly frozen):
- ETTh1:   $1.559 \pm 0.022$
- ETTm1:   $0.970 \pm 0.026$
- Weather: $0.611 \pm 0.021$

These numbers appear in `main.tex` Table `tab:lora` (§4.5) and support the
paper's claim of a 9/9 RR-MoA vs LoRA head-to-head win.
