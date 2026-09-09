import argparse, gc, hashlib, json, os, shutil, time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def build_prompt(task_text, test_list, setup_code=""):
    tests = "\n".join(test_list)
    setup = f"\n# Setup code:\n{setup_code}\n" if setup_code else ""
    return (
        "You are a Python programming expert. Write a Python function that solves "
        "the following problem. Your code must pass all the given test assertions.\n\n"
        f"Problem:\n{task_text}\n{setup}Tests:\n{tests}\n\n"
        "Reason internally, then output only a complete Python solution inside a "
        "```python code block. Keep the final answer concise."
    )


def merge_adapter(base_path, adapter_path):
    out = f"/tmp/rollout_policy_merge_{os.getpid()}"
    shutil.rmtree(out, ignore_errors=True)
    print(f"[Rollout] Merging adapter {adapter_path} into {base_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base_path, trust_remote_code=True).save_pretrained(out)
    del base, model
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--split", choices=["train", "validation", "test"], default="train")
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--task-ids-file")
    ap.add_argument("--n-samples", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--force-no-think", action="store_true")
    args = ap.parse_args()

    ds = load_dataset("google-research-datasets/mbpp", split=args.split)
    task_ids = None
    if args.task_ids_file:
        task_ids = {int(x.strip()) for x in Path(args.task_ids_file).read_text().splitlines() if x.strip()}
    items = [x for x in ds if task_ids is None or int(x["task_id"]) in task_ids]
    items = items[:args.n_prompts] if task_ids is None else items

    prompts_data = []
    for item in items:
        prompt = build_prompt(item["text"], item["test_list"], item.get("test_setup_code", ""))
        prompts_data.append({
            "prompt_id": f"mbpp_{args.split}_{int(item['task_id'])}",
            "prompt": prompt,
            "task_text": item["text"],
            "test_list": item["test_list"],
            "test_setup_code": item.get("test_setup_code", ""),
            "reference_code": item.get("code", ""),
            "task_id": int(item["task_id"]),
        })

    model_path = merge_adapter(args.model, args.adapter) if args.adapter else args.model
    print(f"[Rollout] {len(prompts_data)} tasks, {args.n_samples} samples each")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()
    chat_prompts = []
    prompt_token_ids = []
    for p in prompts_data:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        if args.force_no_think:
            rendered += "</think>\n"
        chat_prompts.append(rendered)
        # LLM.generate(str_prompts) uses vLLM's completion tokenizer defaults,
        # which add special tokens. Record the exact same context for training.
        prompt_token_ids.append(tokenizer.encode(rendered, add_special_tokens=True))
    sampling = SamplingParams(
        n=args.n_samples, temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, seed=args.seed
    )
    t0 = time.time()
    outputs = llm.generate(chat_prompts, sampling)
    elapsed = time.time() - t0
    records = []
    for prompt_index, (pd, out) in enumerate(zip(prompts_data, outputs)):
        for j, completion in enumerate(out.outputs):
            records.append({
                **pd,
                "sample_idx": j,
                "response": completion.text,
                "gen_tokens": len(completion.token_ids),
                "prompt_token_ids": prompt_token_ids[prompt_index],
                "response_token_ids": list(completion.token_ids),
                "rendered_prompt": chat_prompts[prompt_index],
                "seed": args.seed,
                "prompt_hash": hashlib.sha256(pd["prompt"].encode()).hexdigest(),
                "finish_reason": getattr(completion, "finish_reason", None),
                "policy_model": args.model,
                "policy_adapter": args.adapter,
            })
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({
        "rollouts": len(records), "tasks": len(prompts_data),
        "avg_tokens": sum(r["gen_tokens"] for r in records) / len(records),
        "generation_seconds": elapsed, "seed": args.seed,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
