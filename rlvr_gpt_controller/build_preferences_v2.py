import argparse, collections, hashlib, json
from pathlib import Path


def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--correctness", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--max-pairs-per-task", type=int, default=6)
    args = ap.parse_args()
    rollouts = {}
    for line in Path(args.rollouts).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); rollouts[(r["prompt_id"], int(r["sample_idx"]))] = r
    checks = {}
    for line in Path(args.correctness).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); checks[(r["prompt_id"], int(r["sample_idx"]))] = r
    groups = collections.defaultdict(list)
    for key, r in rollouts.items():
        c = checks.get(key)
        if c is None or not r.get("response", "").strip() or int(r.get("gen_tokens", 0)) >= args.max_tokens:
            continue
        x = dict(r); x["score"] = float(c.get("correctness", 0.0)); x["passed"] = bool(c.get("passed"))
        groups[r["prompt_id"]].append(x)
    pairs = []
    for prompt_id in sorted(groups):
        xs = groups[prompt_id]
        chosen = sorted([x for x in xs if x["score"] > 0.0], key=lambda x: (-x["score"], int(x["gen_tokens"]), int(x["sample_idx"])))
        rejected = sorted([x for x in xs if x["score"] < 1.0], key=lambda x: (x["score"], -int(x["gen_tokens"]), int(x["sample_idx"])))
        seen = set()
        for ch in chosen:
            for rej in rejected:
                if ch["score"] <= rej["score"]: continue
                key = (sha(ch["response"]), sha(rej["response"]))
                if key in seen: continue
                seen.add(key)
                pairs.append({
                    "task_id": ch.get("task_id"), "prompt_id": prompt_id,
                    "prompt_hash": ch.get("prompt_hash") or sha(ch["prompt"]), "prompt": ch["prompt"],
                    "chosen": ch["response"], "rejected": rej["response"],
                    "chosen_passed": ch["passed"], "rejected_passed": rej["passed"],
                    "chosen_score": ch["score"], "rejected_score": rej["score"],
                    "chosen_tokens": int(ch["gen_tokens"]), "rejected_tokens": int(rej["gen_tokens"]),
                    "chosen_sample_idx": int(ch["sample_idx"]), "rejected_sample_idx": int(rej["sample_idx"]),
                    "chosen_hash": key[0], "rejected_hash": key[1],
                    "preference_rule": "higher_verified_test_fraction; length_used_only_as_tiebreaker",
                    "judge_model": "external_controller_verifier_v2", "judge_temperature": 0.0,
                })
                if sum(1 for p in pairs if p["prompt_id"] == prompt_id) >= args.max_pairs_per_task: break
            if sum(1 for p in pairs if p["prompt_id"] == prompt_id) >= args.max_pairs_per_task: break
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in pairs: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps({"pairs": len(pairs), "tasks_with_pairs": len(set(p["prompt_id"] for p in pairs)),
                      "rollouts": len(rollouts), "verified": len(checks),
                      "mean_chosen_score": sum(p["chosen_score"] for p in pairs) / len(pairs) if pairs else 0.0,
                      "mean_rejected_score": sum(p["rejected_score"] for p in pairs) / len(pairs) if pairs else 0.0}, sort_keys=True))


if __name__ == "__main__": main()
