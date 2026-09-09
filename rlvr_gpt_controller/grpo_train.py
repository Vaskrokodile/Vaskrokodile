import argparse
import collections
import gc
import json
import math
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_records(rollouts_path, correctness_path, min_group_std):
    scores = {}
    with open(correctness_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                scores[(x["prompt_id"], int(x["sample_idx"]))] = x

    groups = collections.defaultdict(list)
    with open(rollouts_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            c = scores[(x["prompt_id"], int(x["sample_idx"]))]
            score = float(c["correctness"])
            if x.get("finish_reason") == "length":
                score -= 0.10
            x["reward"] = max(-1.0, score)
            x["prompt_token_ids"] = list(x.get("prompt_token_ids", []))
            x["response_token_ids"] = list(x["response_token_ids"])
            groups[x["prompt_id"]].append(x)

    kept = []
    dropped = collections.Counter()
    for prompt_id, rows in groups.items():
        rewards = [r["reward"] for r in rows]
        if len(rewards) < 2:
            dropped["too_small"] += 1
            continue
        if max(rewards) == min(rewards):
            dropped["zero_variance"] += 1
            continue
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        if math.sqrt(variance) < min_group_std:
            dropped["low_std"] += 1
            continue
        # Mean-only group normalization avoids the length bias of std-normalized GRPO.
        for row in rows:
            row["advantage"] = row["reward"] - mean
            kept.append(row)
    return kept, groups, dropped


def sequence_logprob(model, input_ids, attention_mask, response_start, response_len):
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    start = response_start - 1
    end = start + response_len
    return token_lp[:, start:end]


def make_batch(rows, tokenizer, device):
    sequences = []
    starts = []
    lengths = []
    for row in rows:
        prompt_ids = row["prompt_token_ids"]
        response_ids = row["response_token_ids"]
        if not prompt_ids:
            prompt_ids = tokenizer.encode(row["rendered_prompt"], add_special_tokens=False)
        if not response_ids:
            response_ids = tokenizer.encode(row["response"], add_special_tokens=False)
        if not response_ids:
            continue
        sequences.append(prompt_ids + response_ids)
        starts.append(len(prompt_ids))
        lengths.append(len(response_ids))
        row["_prompt_len"] = len(prompt_ids)
        row["_response_len"] = len(response_ids)
    if not sequences:
        return None
    max_len = max(len(s) for s in sequences)
    pad = tokenizer.pad_token_id
    ids = torch.full((len(sequences), max_len), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, :len(seq)] = 1
    return ids, mask, starts, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--correctness", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=7e-7)
    ap.add_argument("--clip-low", type=float, default=0.20)
    ap.add_argument("--clip-high", type=float, default=0.28)
    ap.add_argument("--kl-beta", type=float, default=0.01)
    ap.add_argument("--min-group-std", type=float, default=0.05)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=2718)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    rows, all_groups, dropped = read_records(
        args.rollouts, args.correctness, args.min_group_std
    )
    if not rows:
        raise RuntimeError("No informative reward groups remain after filtering")
    random.Random(args.seed).shuffle(rows)
    print(json.dumps({
        "all_groups": len(all_groups),
        "kept_groups": len({r["prompt_id"] for r in rows}),
        "kept_rows": len(rows),
        "dropped": dropped,
        "reward_mean": sum(r["reward"] for r in rows) / len(rows),
    }, sort_keys=True))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # Keep the policy-gradient update deterministic; rollout diversity
        # already comes from sampling, and this dataset is reward sparse.
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=0.01,
    )

    # Compute old-policy sequence log-probabilities before any update. This is
    # still one model in memory: the LoRA adapter starts at zero, so this is the
    # rollout policy and the reference anchor at the start of the batch.
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), args.batch_size):
            batch_rows = rows[offset:offset + args.batch_size]
            batch = make_batch(batch_rows, tokenizer, device)
            if batch is None:
                continue
            ids, mask, starts, lengths = batch
            lps = sequence_logprob(model, ids, mask, starts[0], lengths[0]) if len(batch_rows) == 1 else None
            if lps is None:
                out = model(input_ids=ids, attention_mask=mask, use_cache=False)
                lp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
                labels = ids[:, 1:]
                all_lp = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                for i, row in enumerate(batch_rows):
                    row["old_logp"] = all_lp[i, starts[i]-1:starts[i]-1+lengths[i]].detach().float().cpu()
            else:
                batch_rows[0]["old_logp"] = lps[0].detach().float().cpu()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    running = []
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(rows)
        for offset in range(0, len(rows), args.batch_size):
            batch_rows = rows[offset:offset + args.batch_size]
            batch = make_batch(batch_rows, tokenizer, device)
            if batch is None:
                continue
            ids, mask, starts, lengths = batch
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            logits = out.logits[:, :-1, :]
            labels = ids[:, 1:]
            token_lp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            losses = []
            for i, row in enumerate(batch_rows):
                cur = token_lp[i, starts[i]-1:starts[i]-1+lengths[i]]
                old = row["old_logp"].to(device)
                n = min(cur.numel(), old.numel())
                cur, old = cur[:n], old[:n]
                advantage = float(row["advantage"])
                ratio = torch.exp((cur - old).clamp(-10, 10))
                clipped = torch.clamp(ratio, 1.0 - args.clip_low, 1.0 + args.clip_high)
                pg = torch.minimum(ratio * advantage, clipped * advantage)
                # Positive second-order approximation to KL(pi_current || pi_old),
                # available without keeping a second model in memory.
                delta = (cur - old).clamp(-10, 10)
                kl = (torch.exp(delta) - 1.0 - delta).mean()
                losses.append(-pg.mean() + args.kl_beta * kl)
            loss = torch.stack(losses).mean() / args.grad_accum
            loss.backward()
            running.append(float(loss.detach().cpu()) * args.grad_accum)
            micro_step = (offset // args.batch_size) + 1
            is_last = offset + args.batch_size >= len(rows)
            if micro_step % args.grad_accum == 0 or is_last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates % 10 == 0:
                    print(json.dumps({"epoch": epoch, "update": updates, "loss": sum(running[-10:]) / 10}))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(json.dumps({
        "output": args.output,
        "updates": updates,
        "mean_loss": sum(running) / max(1, len(running)),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
    }, sort_keys=True))
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
