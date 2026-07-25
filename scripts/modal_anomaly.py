"""Modal A10G driver for the SMD anomaly-detection mechanism-check (rebuttal).

    # preflight one machine/seed (validates + warms cache)
    ~/.venv-modal/bin/modal run scripts/modal_anomaly.py --machines machine-1-1 --seeds 42

    # full: a few machines x 3 seeds
    ~/.venv-modal/bin/modal run scripts/modal_anomaly.py

Results written to results/anomaly_smd/.
"""
import json
import os

import modal

CACHE_DIR = "/root/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch>=2.1.0", "momentfm", "numpy", "scipy", "scikit-learn", "pandas")
    .env({"HF_HOME": CACHE_DIR})
    .add_local_dir("feasibility", "/root/feasibility")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("data/SMD", "/root/data/SMD")
)

app = modal.App("rrmoa-anomaly-smd", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=2400, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(machine: str, seed: int, n_epochs: int = 15) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from anomaly_core import run_anomaly_cell

    res = run_anomaly_cell(machine, seed, n_epochs=n_epochs, device="cuda")
    hf_cache.commit()
    return res


@app.local_entrypoint()
def main(machines: str = "machine-1-1,machine-2-1,machine-3-1",
         seeds: str = "42,43,44", n_epochs: int = 15,
         out_dir: str = "results/anomaly_smd"):
    ms = [m.strip() for m in machines.split(",") if m.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(m, s, n_epochs) for m in ms for s in sd]
    print(f"launching {len(cells)} anomaly cells on A10G ({len(ms)} machines x {len(sd)} seeds)")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        with open(os.path.join(out_dir, f"{res['machine']}_s{res['seed']}.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  {res['machine']:<14} s={res['seed']}  roc_auc={res['roc_auc']:.3f} "
              f"pr_auc={res['pr_auc']:.3f}  entropy={res['routing_entropy']:.3f}  "
              f"(anom_frac={res['anom_frac']:.2f})")
    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
