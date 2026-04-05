# Experimental Evidence Pack — Multi-Seed RR-MoA Ablation

This is an **orphan branch** of the `neuralips26` repository. It contains no
source code; only the raw experimental artifacts that back the numbers in
Tables 3, 4, and 5 (and Figure 2) of the NeurIPS 2026 submission
"Raw-Routed Adapters and Architecture Search for Time Series Foundation
Models".

This branch exists so that a reviewer who doubts the experimental results
can inspect the raw JSON files, run logs, and system metadata directly
without needing to run anything.

## Why an orphan branch?

- Keeps `main`'s history clean of ~100 machine-generated JSON files.
- Provides a permanent, independently fetchable snapshot tied to a specific
  point in time (2026-04-05).
- Can be checked out in isolation: `git fetch origin && git checkout evidence/multiseed-2026-04-05`.
- The corresponding git tag `evidence-multiseed-v1` pins the exact commit.

## Quick start (30 seconds)

```bash
git fetch origin evidence/multiseed-2026-04-05
git checkout evidence/multiseed-2026-04-05
python3 evidence_vm/verify.py
```

The `verify.py` script re-reads every JSON result file, recomputes every
mean$\pm$std reported in the paper's Tables 3-5, and checks them against the
raw data. It should print:

```
Ran 50 checks against 63 RR-MoA + 45 AdaMix + 9 LoRA + 9 DLinear JSON files.
PASS: all 50 numeric claims in main.tex Tables 3-5 (plus LoRA, rawness ablation, and DLinear baseline) match the raw JSON
evidence within tolerance (MSE 0.005, entropy 0.01).
RR-MoA wins: 27/27
```

## Contents

See [`evidence_vm/FORENSIC_AUDIT.md`](evidence_vm/FORENSIC_AUDIT.md) for the
full manifest. In short:

| Path | What it is |
|---|---|
| `evidence_vm/FORENSIC_AUDIT.md` | Full forensic audit document |
| `evidence_vm/verify.py` | Self-verification script (checks every paper number) |
| `evidence_vm/multiseed_run.log` | Overnight multi-seed sweep stdout (1,232 lines, 66 timestamped experiment launches) |
| `evidence_vm/freeze_ablation_run.log` | Earlier single-seed sweep stdout (553 lines) |
| `evidence_vm/system_metadata.txt` | VM host, kernel, GPU, torch version at retrieval time |
| `evidence_vm/vm_file_listing.txt` | VM-side `ls -la --time-style=full-iso` (nanosecond-precision mtimes) |
| `evidence_vm/manifest_md5.txt` | MD5 hash of every log file and every JSON file |
| `evidence_vm/rr_moa/*.json` | 54 RR-MoA per-experiment result files |
| `evidence_vm/adamix/*.json` | 45 AdaMix per-experiment result files |

## Provenance

All files were generated on an NVIDIA A10G instance
(`ec2-13-238-161-176.ap-southeast-2`, kernel `6.8.0-1044-aws`, torch 2.4.0+cu124,
CUDA 12.4) between 2026-04-03 17:52 AEST and 2026-04-05 16:26 AEST. The
multi-seed sweep used for the paper's main tables ran in a single 2h 21min
block starting at 2026-04-05 04:51:54 AEST.

Each RR-MoA JSON records the full hyperparameter state, seed, freeze level,
sparsity, final test MSE, per-sample routing weight vector, and Shannon
routing entropy. Each AdaMix JSON records the same fields plus the baseline
comparison. No post-hoc editing was performed on the JSON files; they are
bit-for-bit identical to what the Python scripts wrote on the VM
(see `manifest_md5.txt`).

## Reproducing from scratch

On any A10G machine with `main` branch checked out at commit
`337ffa3` or later:

```bash
nohup bash scripts/run_multiseed.sh > results/multiseed_run.log 2>&1 &
```

Expected runtime: ~2.5 hours. The resulting JSON files should differ from
the ones in this branch only in non-deterministic low-order bits of floating
point (dropout, BLAS ordering). All reported mean$\pm$std values should
reproduce within the same tolerances `verify.py` uses.
