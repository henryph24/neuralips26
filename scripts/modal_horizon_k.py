"""Modal A10G driver for two rebuttal experiments:

  longer horizons (jemj Q3 + 8b2Z limitation) and larger expert pools (8b2Z W3).

Usage (run from repo root):

    # 1) preflight: 1 cell, warms the weight cache + validates the harness
    ~/.venv-modal/bin/modal run scripts/modal_horizon_k.py --mode preflight

    # 2) larger-K experiment (ETTh1, K in {5,10,15,20}, 3 seeds)
    ~/.venv-modal/bin/modal run scripts/modal_horizon_k.py --mode k

    # 3) longer-horizon experiment (H in {1000,2000}, 3 seeds)
    ~/.venv-modal/bin/modal run scripts/modal_horizon_k.py --mode horizon

    # or everything
    ~/.venv-modal/bin/modal run scripts/modal_horizon_k.py --mode all

Results are written to results/horizon_k/.
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

app = modal.App("rrmoa-horizon-k", image=image)
hf_cache = modal.Volume.from_name("hf-cache-momentfm", create_if_missing=True)


@app.function(gpu="A10G", timeout=2400, volumes={CACHE_DIR: hf_cache})
def run_cell_remote(kind: str, dataset: str, param: int, seed: int, n_epochs: int = 15) -> dict:
    import sys
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/scripts")
    from horizon_k_core import run_horizon_cell, run_k_cell

    if kind == "horizon":
        res = run_horizon_cell(dataset, param, seed, n_epochs=n_epochs, device="cuda")
    else:
        res = run_k_cell(dataset, param, seed, n_epochs=n_epochs, device="cuda")
    hf_cache.commit()
    return res


def _build_cells(mode, n_epochs):
    SEEDS = [42, 43, 44]
    k_cells = [("K", "ETTh1", k, s, n_epochs) for k in (5, 10, 15, 20) for s in SEEDS]
    h_cells = (
        [("horizon", d, h, s, n_epochs)
         for d in ("ETTm2", "Weather", "Electricity") for h in (1000, 2000) for s in SEEDS]
        + [("horizon", "ETTh2", 1000, s, n_epochs) for s in SEEDS]  # ETTh2 too short for H=2000
    )
    if mode == "preflight":
        return [("K", "ETTh1", 5, 42, n_epochs)]
    if mode == "k":
        return k_cells
    if mode == "horizon":
        return h_cells
    return k_cells + h_cells  # all


@app.local_entrypoint()
def main(mode: str = "all", n_epochs: int = 15, out_dir: str = "results/horizon_k"):
    cells = _build_cells(mode, n_epochs)
    print(f"[{mode}] launching {len(cells)} cells on A10G")
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for res in run_cell_remote.starmap(cells, return_exceptions=True):
        if isinstance(res, Exception):
            print(f"  CELL FAILED: {res}")
            continue
        results.append(res)
        if res.get("exp") == "K":
            tag = f"K_{res['dataset']}_K{res['K']}_s{res['seed']}"
        else:
            tag = f"H_{res['dataset']}_H{res['horizon']}_s{res['seed']}"
        with open(os.path.join(out_dir, tag + ".json"), "w") as f:
            json.dump(res, f, indent=2, default=str)
        mse = res.get("rrmoa_raw_mse")
        ent = res.get("rrmoa_raw_entropy")
        dvf = res.get("delta_vs_fixed_pct")
        dvd = res.get("delta_vs_dlinear_pct")
        extra = f" vs_DL={dvd:+.1f}%" if dvd is not None else ""
        print(f"  {tag:<28} mse={mse:.4f} ent={ent:.3f} vs_fixed={dvf:+.1f}%{extra}"
              if mse is not None and ent is not None and dvf is not None else f"  {tag}: {res}")

    with open(os.path.join(out_dir, f"_all_{mode}.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)}/{len(cells)} cells to {out_dir}/")
