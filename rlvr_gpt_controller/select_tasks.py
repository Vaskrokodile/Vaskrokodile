import argparse, collections, json, random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correctness", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-tasks", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1729)
    args = ap.parse_args()
    groups = collections.defaultdict(list)
    with open(args.correctness, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                groups[int(r["task_id"])].append(float(r.get("correctness", 0.0)))
    stats = []
    for task_id, values in groups.items():
        rate = sum(values) / len(values)
        stats.append((task_id, rate, len(values)))
    mixed = sorted([x for x in stats if 0.0 < x[1] < 1.0], key=lambda x: (abs(x[1] - 0.5), x[1], x[0]))
    hard = sorted([x for x in stats if x[1] == 0.0], key=lambda x: x[0])
    hard += sorted([x for x in stats if 0.0 < x[1] < 1.0], key=lambda x: (x[1], x[0]))
    chosen, seen = [], set()
    for row in mixed + hard:
        if row[0] not in seen and len(chosen) < args.n_tasks:
            chosen.append(row); seen.add(row[0])
    remaining = [x for x in stats if x[0] not in seen]
    random.Random(args.seed).shuffle(remaining)
    for row in remaining:
        if len(chosen) >= args.n_tasks:
            break
        chosen.append(row); seen.add(row[0])
    chosen.sort(key=lambda x: x[0])
    Path(args.output).write_text("\n".join(str(x[0]) for x in chosen) + "\n", encoding="utf-8")
    print(json.dumps({
        "selected": len(chosen),
        "mixed": sum(0.0 < x[1] < 1.0 for x in chosen),
        "mean_task_pass_rate": sum(x[1] for x in chosen) / len(chosen) if chosen else 0.0,
        "output": args.output,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
