"""Modal A10G driver: Residual-IA+ vs DLinear at long horizons (jemj Q3).

Reuses the exact off-suite RIA+ recipe (ria_offsuite_core.run_cell) at H in
{1000, 2000} on the non-stationary datasets where the TSFM helps at long H.

    ~/.venv-modal/bin/modal run scripts/modal_ria_horizon.py

Results -> results/ria_horizon/.
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
)

app = modal.App("rrmoa-ria-horizon", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=3000, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, seed: int, horizon: int) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from ria_offsuite_core import run_cell

    res = run_cell(dataset, seed, horizon=horizon, device="cuda")
    hf_cache.commit()
    return res


@app.local_entrypoint()
def main(datasets: str = "ETTm2,Weather,Electricity", horizons: str = "1000,2000",
         seeds: str = "42,43,44", out_dir: str = "results/ria_horizon"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    hs = [int(x) for x in horizons.split(",") if x.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(d, s, h) for d in ds for h in hs for s in sd]
    print(f"launching {len(cells)} RIA+ long-horizon cells on A10G")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        with open(os.path.join(out_dir, f"{res['dataset']}_H{res['horizon']}_s{res['seed']}.json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  {res['dataset']:<11} H={res['horizon']:<5} s={res['seed']}  "
              f"RIA+={res['ria_mse']:.4f} DLin={res['dlinear_mse']:.4f} "
              f"gap={res['gap_pct']:+.1f}% {'WIN' if res['match_or_beat'] else 'loss'} "
              f"ent={res.get('ria_entropy')}")
    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
