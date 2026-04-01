"""Self-improvement iteration: collect new winners → merge → fine-tune → save.

Each iteration:
1. Collects winning adapters from latest evolution results
2. Merges with prior training data (dedup by code)
3. Fine-tunes Qwen on merged dataset
4. Saves new LoRA adapter for next evolution round

Usage:
    python scripts/self_improve_iterate.py --iteration 1
    python scripts/self_improve_iterate.py --iteration 2
"""

import argparse
import json
import glob
import os
import sys
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feasibility.code_evolution import CODE_SYSTEM_PROMPT, validate_adapter_code


def collect_local_winners():
    """Collect winning adapters from local evolution results."""
    winners = []
    seen_codes = set()

    for filepath in sorted(glob.glob("results/local_evolution/self_improving_*.json")):
        with open(filepath) as f:
            data = json.load(f)
        for v in data.get("validated", []):
            if v.get("error") is not None:
                continue
            mse = v.get("mse_15ep", v.get("mse", None))
            if mse is None or mse >= 999:
                continue
            code = v["code"].strip()
            if code in seen_codes:
                continue
            seen_codes.add(code)
            winners.append({
                "code": v["code"],
                "mse": mse,
                "param_count": v.get("param_count", 0),
                "source": os.path.basename(filepath),
            })

    # Also collect from final_population (top performers at 3 epochs)
    for filepath in sorted(glob.glob("results/local_evolution/self_improving_*.json")):
        with open(filepath) as f:
            data = json.load(f)
        for ind in data.get("final_population", [])[:5]:
            if ind.get("error") is not None:
                continue
            code = ind["code"].strip()
            if code in seen_codes:
                continue
            seen_codes.add(code)
            winners.append({
                "code": ind["code"],
                "mse": ind.get("mse", float('inf')),
                "param_count": ind.get("param_count", 0),
                "source": os.path.basename(filepath) + "_pop",
            })

    return winners


def format_sft_example(code, mse=None, reasoning=""):
    """Format a single adapter as an SFT training example."""
    user_prompt = (
        "Design a PyTorch adapter module for time series forecasting. "
        "The adapter should map encoder hidden states to forecast values. "
        "Target MSE should be as low as possible. "
        "Return the complete Adapter class definition."
    )
    assistant_response = json.dumps({
        "adapters": [{
            "reasoning": reasoning or "Optimized adapter architecture",
            "code": code,
        }]
    })
    return {
        "messages": [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        "metadata": {"mse": mse or 0, "source": "iteration"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    iter_n = args.iteration
    print("=" * 60)
    print("Self-Improvement Iteration %d" % iter_n)
    print("=" * 60)

    # 1. Load existing training data
    base_dataset = "experiments/sft_dataset.jsonl"
    existing = []
    existing_codes = set()
    if os.path.exists(base_dataset):
        with open(base_dataset) as f:
            for line in f:
                ex = json.loads(line)
                existing.append(ex)
                # Extract code for dedup
                try:
                    resp = json.loads(ex["messages"][2]["content"])
                    code = resp["adapters"][0]["code"].strip()
                    existing_codes.add(code)
                except (KeyError, IndexError, json.JSONDecodeError):
                    pass
    print("Existing training examples: %d" % len(existing))

    # 2. Collect new winners from local evolution
    new_winners = collect_local_winners()
    print("New winners found: %d" % len(new_winners))

    # 3. Merge (dedup)
    added = 0
    for w in new_winners:
        code = w["code"].strip()
        if code not in existing_codes:
            existing_codes.add(code)
            existing.append(format_sft_example(w["code"], w["mse"]))
            added += 1
    print("New unique examples added: %d" % added)
    print("Total training examples: %d" % len(existing))

    # 4. Save merged dataset
    iter_dataset = "experiments/sft_dataset_iter%d.jsonl" % iter_n
    with open(iter_dataset, "w") as f:
        for ex in existing:
            f.write(json.dumps(ex) + "\n")
    print("Saved to %s" % iter_dataset)

    # 5. Fine-tune
    output_dir = "models/qwen-adapter-coder-iter%d" % iter_n
    print("\nFine-tuning on %d examples → %s" % (len(existing), output_dir))
    cmd = [
        "python3", "scripts/finetune_qwen.py",
        "--model", args.base_model,
        "--dataset", iter_dataset,
        "--output-dir", output_dir,
        "--epochs", str(args.epochs),
    ]
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 60)
    print("Iteration %d complete!" % iter_n)
    print("New adapter: %s" % output_dir)
    print("Use: --adapter-path %s" % output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
