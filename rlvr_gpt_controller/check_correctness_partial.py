import argparse, json, os, re, resource, signal, subprocess, tempfile
from pathlib import Path


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*(?:\r?\n)?(.*?)(?:```|$)", text, re.IGNORECASE | re.DOTALL)
    if blocks:
        blocks = [x.strip() for x in blocks if x.strip()]
        if blocks:
            return max(blocks, key=lambda x: (int(bool(re.search(r"\b(def|class|import|from)\b", x))), len(x)))
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if "```" in text:
        text = text.split("```", 1)[1]
        text = re.sub(r"^(?:python|py)\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def limit_child():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024**2, 10 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def run_tests(code, tests, setup, timeout):
    test_literal = repr(list(tests))
    body = (setup + "\n" if setup else "") + code + "\n"
    body += "import json as __json\n"
    body += f"__tests__ = {test_literal}\n__results__ = []\n__errors__ = []\n"
    body += "for __test__ in __tests__:\n"
    body += "    try:\n        exec(__test__, globals(), globals())\n        __results__.append(1)\n        __errors__.append('')\n"
    body += "    except BaseException as __e:\n        __results__.append(0)\n        __errors__.append(type(__e).__name__ + ': ' + str(__e)[:200])\n"
    body += "print('__PARTIAL__' + __json.dumps({'results': __results__, 'errors': __errors__}))\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(body); f.flush(); path = f.name
    try:
        env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"}
        proc = subprocess.Popen(["python3", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                cwd="/tmp", env=env, preexec_fn=limit_child, start_new_session=True)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError): proc.kill()
            proc.communicate()
            return [0] * len(tests), ["timeout"] * len(tests)
        marker = "__PARTIAL__"
        line = next((x for x in stdout.splitlines() if x.startswith(marker)), "")
        if not line:
            err = (stderr or stdout or "no result")[:300]
            return [0] * len(tests), [err] * len(tests)
        result = json.loads(line[len(marker):])
        return result["results"], result["errors"]
    except Exception as e:
        return [0] * len(tests), [str(e)] * len(tests)
    finally:
        try: os.unlink(path)
        except FileNotFoundError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()
    rollouts = [json.loads(x) for x in Path(args.rollouts).read_text(encoding="utf-8").splitlines() if x.strip()]
    results, total = [], 0
    for i, r in enumerate(rollouts, 1):
        code = extract_code(r["response"])
        flags, errors = run_tests(code, r["test_list"], r.get("test_setup_code", ""), args.timeout)
        passed = sum(flags); n = len(flags); total += passed
        results.append({
            "prompt_id": r["prompt_id"], "task_id": r.get("task_id"), "sample_idx": r["sample_idx"],
            "correctness": passed / n if n else 0.0, "passed": passed == n and n > 0,
            "passed_tests": passed, "n_tests": n,
            "error": next((e for e, ok in zip(errors, flags) if not ok), "")[:300],
            "code": code[:2000], "gen_tokens": r["gen_tokens"],
        })
        if i % 20 == 0: print(f"  {i}/{len(rollouts)} tested, mean test score {total / max(1, i * n):.3f}")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"rollouts": len(results), "fully_passed": sum(r["passed"] for r in results),
                      "mean_correctness": sum(r["correctness"] for r in results) / len(results) if results else 0.0,
                      "output": str(out)}, sort_keys=True))


if __name__ == "__main__": main()
