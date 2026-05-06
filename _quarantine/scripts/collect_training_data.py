"""Collect training data for self-improving LLM fine-tuning.

Parses all experiment results (evo_logs, validated, ablation) to extract
(prompt, code, fitness) tuples for SFT training.

Usage:
    python scripts/collect_training_data.py
"""

import json
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feasibility.code_evolution import CODE_SYSTEM_PROMPT, validate_adapter_code


def collect_winning_adapters():
    """Extract winning adapter codes from validated results."""
    winners = []
    seen_codes = set()

    # Sources: validated results from all experiments
    patterns = [
        "results/code_evolution/validated_*_4*.json",        # Code Evo seeds 42-44
        "results/code_evolution/validated_*_gpt-4o.json",    # gpt-4o runs
        "results/code_evolution/ablation_random_code_validated_*_42.json",  # Random Code winners
        "results/code_evolution/ablation_llm_no_evo_validated_*_42.json",   # LLM No Evo winners
    ]

    for pattern in patterns:
        for filepath in sorted(glob.glob(pattern)):
            with open(filepath) as f:
                validated = json.load(f)

            basename = os.path.basename(filepath)
            for v in validated:
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
                    "source": basename,
                    "reasoning": v.get("reasoning", ""),
                })

    return winners


def collect_failed_adapters():
    """Extract failed adapter codes from evolution logs for negative examples."""
    failures = []
    seen_codes = set()

    for filepath in sorted(glob.glob("results/code_evolution/evo_log_*_42.json")):
        with open(filepath) as f:
            log_data = json.load(f)

        # Evolution logs contain final_population or all_reasonings
        # but not individual failed codes directly.
        # Skip — we'll use the winning codes only for SFT.

    return failures


def format_sft_dataset(winners):
    """Format winning adapters as SFT training examples.

    Each example: system prompt + user context → assistant generates winning code.
    """
    examples = []

    for w in winners:
        # User prompt: ask for adapter design (simplified, no population context)
        user_prompt = (
            "Design a PyTorch adapter module for time series forecasting. "
            "The adapter should map encoder hidden states to forecast values. "
            "Target MSE should be as low as possible. "
            "Return the complete Adapter class definition."
        )

        # Assistant response: the winning code
        assistant_response = json.dumps({
            "adapters": [{
                "reasoning": w.get("reasoning", "Optimized adapter architecture"),
                "code": w["code"],
            }]
        })

        examples.append({
            "messages": [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response},
            ],
            "metadata": {
                "mse": w["mse"],
                "param_count": w["param_count"],
                "source": w["source"],
            },
        })

    return examples


def main():
    os.makedirs("experiments", exist_ok=True)

    # Collect winners
    winners = collect_winning_adapters()
    print("Winning adapters collected: %d" % len(winners))

    if winners:
        mses = [w["mse"] for w in winners]
        print("  MSE range: %.4f - %.4f" % (min(mses), max(mses)))
        print("  Sources: %d files" % len(set(w["source"] for w in winners)))

        # Validate all codes are still valid
        valid_count = 0
        for w in winners:
            val = validate_adapter_code(w["code"])
            if val["valid"]:
                valid_count += 1
        print("  Valid: %d/%d (%.0f%%)" % (valid_count, len(winners), 100 * valid_count / len(winners)))

    # Format as SFT dataset
    sft_data = format_sft_dataset(winners)
    print("\nSFT training examples: %d" % len(sft_data))

    # Save raw winners
    with open("experiments/finetune_training_data.json", "w") as f:
        json.dump(winners, f, indent=2)
    print("Saved raw data to experiments/finetune_training_data.json")

    # Save SFT-formatted dataset
    # JSONL format for trl/transformers SFT
    with open("experiments/sft_dataset.jsonl", "w") as f:
        for example in sft_data:
            f.write(json.dumps(example) + "\n")
    print("Saved SFT dataset to experiments/sft_dataset.jsonl")

    # Also save just the codes for quick reference
    with open("experiments/winning_codes.txt", "w") as f:
        for i, w in enumerate(sorted(winners, key=lambda x: x["mse"])):
            f.write("# Adapter %d — MSE=%.4f, params=%d, source=%s\n" % (
                i + 1, w["mse"], w["param_count"], w["source"]))
            f.write(w["code"])
            f.write("\n\n" + "=" * 60 + "\n\n")
    print("Saved codes to experiments/winning_codes.txt")


if __name__ == "__main__":
    main()
