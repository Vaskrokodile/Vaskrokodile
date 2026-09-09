"""
lazyRLGYM + Devin-in-the-loop — DPO Training with LoRA
=======================================================
Iterative DPO training using preference pairs constructed by Devin.

Key memory optimization (from lazyrlGYM):
  - LoRA + ref_model=None → TRL uses adapter-disabled base as reference
  - Only ONE model in VRAM (no double loading)
  - Gradient checkpointing for activation savings

Usage:
    python3 dpo_train.py --model /root/models/AIAAH-1 \
        --prefs /root/prefs_iter0.jsonl \
        --output /root/dpo_iter0 \
        --epochs 3 --lr 1e-5 --lora-r 32 --batch-size 2 --grad-accum 4
"""
import argparse
import json
import gc
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_prefs(path: str) -> list[dict]:
    """Load preference pairs from JSONL."""
    pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"[DPO] Loaded {len(pairs)} preference pairs from {path}")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model path")
    parser.add_argument("--prefs", required=True, help="Preference pairs JSONL")
    parser.add_argument("--output", required=True, help="Output directory for LoRA adapter")
    # DPO hyperparams
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-prompt-length", type=int, default=768)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preference pairs
    pairs = load_prefs(args.prefs)
    if not pairs:
        print("[DPO] ERROR: No preference pairs found!")
        sys.exit(1)

    # Format for TRL DPOTrainer: {"prompt": str, "chosen": str, "rejected": str}
    dpo_data = []
    for p in pairs:
        dpo_data.append({
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })
    hf_dataset = Dataset.from_list(dpo_data)
    print(f"[DPO] Dataset: {len(hf_dataset)} pairs")

    # Load tokenizer
    print(f"[DPO] Loading tokenizer from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Load base model (single model — serves as both policy and reference)
    print(f"[DPO] Loading base model from {args.model} (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    vram_after_load = torch.cuda.memory_allocated() / 1e9
    print(f"[DPO] Model loaded. VRAM: {vram_after_load:.1f} GB")

    # TRL 1.12 exposes warmup_steps rather than warmup_ratio.
    # Preserve the intended 10% warmup while adapting to this dataset size.
    effective_batches = max(1, (len(hf_dataset) + args.batch_size * args.grad_accum - 1) // (args.batch_size * args.grad_accum))
    warmup_steps = max(1, round(args.epochs * effective_batches * 0.1))

    # DPO config
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,
        beta=args.beta,
        max_length=args.max_length,
        loss_type="sigmoid",
        precompute_ref_log_probs=False,
        # bitsandbytes is unavailable on this pod; use full-precision AdamW on the 80GB A100.
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    # Create trainer — ref_model=None means TRL uses adapter-disabled base as reference
    print("[DPO] Creating DPOTrainer (ref_model=None → adapter-disabled base = reference)")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("[DPO] Starting training...")
    train_result = trainer.train()

    # Save LoRA adapter
    print(f"[DPO] Saving LoRA adapter to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Extract metrics
    loss = train_result.training_loss
    reward_accuracy = 0.0
    for entry in reversed(trainer.state.log_history):
        if "rewards/accuracies" in entry:
            reward_accuracy = float(entry["rewards/accuracies"])
            break

    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[DPO] Training complete!")
    print(f"[DPO]   Loss: {loss:.4f}")
    print(f"[DPO]   Reward accuracy: {reward_accuracy:.4f}")
    print(f"[DPO]   Peak VRAM: {vram_peak:.1f} GB")
    print(f"[DPO]   Adapter saved to: {output_dir}")

    # Save metrics
    metrics = {
        "loss": float(loss),
        "reward_accuracy": float(reward_accuracy),
        "peak_vram_gb": float(vram_peak),
        "n_pairs": len(dpo_data),
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "seed": args.seed,
        "warmup_steps": warmup_steps,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "weight_decay": 0.01,
    }
    with open(output_dir / "dpo_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Cleanup
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    print("[DPO] Cleanup done.")


if __name__ == "__main__":
    import sys
    main()
