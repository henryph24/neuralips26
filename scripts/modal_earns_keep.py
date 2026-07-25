"""Modal A10G driver: TSFM-earns-keep (Residual-IA+ vs DLinear vs Raw-MLP MoE), rebuttal Pm4m W1/W4.

Shows a setting where Residual-IA+ (with frozen TSFM) beats BOTH a 49K DLinear AND a
backbone-free Raw-MLP MoE, rebutting "the TSFM is dead weight." Primary 6 datasets at
H=192 (where the paper reports the TSFM contributes maximally, 6/6).

    # preflight one cell (validate the composed core + warm cache)
    modal run scripts/modal_earns_keep.py --datasets ETTh2 --seeds 42
    # full: 6 datasets x 3 seeds @ H=192
    modal run scripts/modal_earns_keep.py

Results return to results/tsfm_earns_keep/.
"""

import json
import os

import modal

CACHE_DIR = "/root/hf_cache"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "momentfm",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
    )
    .env({"HF_HOME": CACHE_DIR})
    .add_local_dir("feasibility", "/root/feasibility")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_file("data/weather.csv", "/root/data/weather.csv")
    .add_local_file("data/electricity.csv", "/root/data/electricity.csv")
)

app = modal.App("rrmoa-tsfm-earns-keep", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=3600, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(dataset: str, seed: int, horizon: int = 192) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")

    from tsfm_earns_keep_core import run_cell

    result = run_cell(dataset, seed, horizon=horizon, device="cuda")
    hf_cache.commit()
    return result


@app.local_entrypoint()
def main(datasets: str = "ETTh1,ETTh2,ETTm1,ETTm2,Weather,Electricity",
         seeds: str = "42,43,44",
         horizon: int = 192,
         out_dir: str = "results/tsfm_earns_keep"):
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    sd = [int(x) for x in seeds.split(",") if x.strip()]
    cells = [(d, s, horizon) for d in ds for s in sd]

    print(f"Launching {len(cells)} earns-keep cells on A10G @ H={horizon} "
          f"({len(ds)} datasets x {len(sd)} seeds)")

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        path = os.path.join(out_dir, f"{res['dataset']}_H{res['horizon']}_s{res['seed']}.json")
        with open(path, "w") as f:
            json.dump(res, f, indent=2, default=str)
        print(f"  {res['dataset']:<12} s{res['seed']}  "
              f"RIA+={res['ria_mse']:.4f}  DLin={res['dlinear_mse']:.4f}  "
              f"RawMLP={res['rawmlp_mse']:.4f}  "
              f"vsDL={res['gap_vs_dlinear_pct']:+.1f}%  vsRaw={res['gap_vs_rawmlp_pct']:+.1f}%  "
              f"{'BEATS BOTH' if res['beats_both'] else 'no'}  ent={res.get('ria_entropy')}")

    n_both = sum(1 for r in results if r.get("beats_both"))
    with open(os.path.join(out_dir, "_all.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/  "
          f"| Residual-IA+ beats BOTH DLinear and Raw-MLP on {n_both}/{len(results)} cells")
