"""Fine-tune Qwen 2.5 3B with QLoRA on winning adapter codes.

Runs on RACE VM (A10G, 23GB VRAM). Uses 4-bit quantization + LoRA
to fit in memory. Trains on SFT dataset of winning adapter codes.

Usage:
    python scripts/finetune_qwen.py
    python scripts/finetune_qwen.py --model Qwen/Qwen2.5-3B-Instruct --epochs 3
    python scripts/finetune_qwen.py --model Qwen/Qwen2.5-7B-Instruct  # fallback if 3B too weak
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", default="experiments/sft_dataset.jsonl")
    parser.add_argument("--output-dir", default="models/qwen-adapter-coder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    print("=" * 60)
    print("Fine-tuning %s with QLoRA" % args.model)
    print("Dataset: %s" % args.dataset)
    print("Output: %s" % args.output_dir)
    print("=" * 60)

    # Load dataset
    examples = []
    with open(args.dataset) as f:
        for line in f:
            examples.append(json.loads(line))
    print("Training examples: %d" % len(examples))

    # Load tokenizer first for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format as text using chat template
    def format_to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list([format_to_text(ex) for ex in examples])
    print("Dataset loaded: %d examples" % len(dataset))

    # Quantization config (4-bit for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print("Loading model (4-bit quantized)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("Trainable params: %d / %d (%.2f%%)" % (trainable, total, 100 * trainable / total))

    # Training config
    os.makedirs(args.output_dir, exist_ok=True)

    training_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
    )

    # Train
    print("\nStarting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_config,
    )

    trainer.train()

    # Save LoRA adapter
    print("\nSaving LoRA adapter to %s" % args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("Adapter saved to: %s" % args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
