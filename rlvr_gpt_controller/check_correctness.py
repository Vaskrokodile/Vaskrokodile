"""
lazyRLGYM + Devin-in-the-loop — Correctness Checker
=====================================================
Runs on the pod. Takes a rollouts JSONL file, executes each rollout's code
against the MBPP tests, and outputs a correctness JSONL.

This is the "verifiable reward" part — Devin uses these results plus
his own quality judgment to construct preference pairs.

Usage:
    python3 check_correctness.py --rollouts /root/rollouts_iter0.jsonl \
        --output /root/correctness_iter0.jsonl
"""
import argparse
import json
import os
import re
import resource
import signal
import subprocess
import tempfile
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

def _limit_child_resources():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024**2, 10 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))


def run_tests(code: str, test_list: list[str], setup_code: str = "", timeout: int = 10) -> tuple[bool, str]:
    full_code = ""
    if setup_code:
        full_code += setup_code + "\n"
    full_code += code + "\n"
    for test in test_list:
        full_code += test + "\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(full_code)
        f.flush()
        tmp_path = f.name

    proc = None
    try:
        child_env = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
        }
        proc = subprocess.Popen(
            ["python3", tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/tmp",
            env=child_env,
            preexec_fn=_limit_child_resources,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.communicate()
            return False, "timeout"
        if proc.returncode == 0:
            return True, "all tests passed"
        return False, (stderr or stdout)[:500]
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args()

    rollouts = []
    with open(args.rollouts, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rollouts.append(json.loads(line))

    print(f"[Check] Testing {len(rollouts)} rollouts...")

    results = []
    n_pass = 0
    for i, r in enumerate(rollouts):
        code = extract_code(r["response"])
        passed, error = run_tests(
            code,
            r["test_list"],
            r.get("test_setup_code", ""),
            timeout=args.timeout,
        )
        if passed:
            n_pass += 1
        results.append({
            "prompt_id": r["prompt_id"],
            "task_id": r.get("task_id"),
            "sample_idx": r["sample_idx"],
            "correctness": 1.0 if passed else 0.0,
            "passed": passed,
            "error": error[:300] if not passed else "",
            "code": code[:2000],
            "gen_tokens": r["gen_tokens"],
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(rollouts)} tested, {n_pass} passed so far")

    print(f"[Check] Done: {n_pass}/{len(rollouts)} passed ({n_pass/len(rollouts):.3f})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Check] Saved to {output_path}")


if __name__ == "__main__":
    main()
