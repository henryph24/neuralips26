"""Modal A10G driver: Residual-IA+ vs DLinear on off-suite datasets (rebuttal W4).

    # preflight (1 cell): Exchange seed 42
    modal run scripts/modal_ria_offsuite.py --datasets Exchange --seeds 42
    # full: Exchange/Solar/Traffic x 3 seeds
    modal run scripts/modal_ria_offsuite.py
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
    .add_local_file("data/weather.csv", "/root/data/weather.csv")
    .add_local_file("data/electricity.csv", "/root/data/electricity.csv")
    .add_local_file("data/exchange_rate.csv", "/root/data/exchange_rate.csv")
    .add_local_file("data/solar.csv", "/root/data/solar.csv")
    .add_local_file("data/traffic.csv", "/root/data/traffic.csv")
)

app = modal.App("rrmoa-ria-offsuite", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=2400, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, seed: int) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from ria_offsuite_core import run_cell
    result = run_cell(dataset, seed, device="cuda")
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(datasets: str = "Exchange,Solar,Traffic", seeds: str = "42,43,44",
         out_dir: str = "results/ria_offsuite"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(d, s) for d in ds for s in sd]
    print(f"Launching {len(cells)} Residual-IA+ vs DLinear cells on A10G")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        path = os.path.join(out_dir, f"{res['dataset']}_s{res['seed']}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  {res['dataset']:<10} s{res['seed']}  RIA+={res['ria_mse']:.4f}  "
              f"DLinear={res['dlinear_mse']:.4f}  gap={res['gap_pct']:+.1f}%  "
              f"{'MoB' if res['match_or_beat'] else 'loss'}  ent={res.get('ria_entropy')}")
    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
