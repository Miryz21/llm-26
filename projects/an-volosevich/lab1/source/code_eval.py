"""Checks generated code against hidden tests and ranks the models.

Model code runs in a separate `python3 -I` process with a time limit and memory/CPU caps.
Reads results/code_runs.jsonl, writes results/code_scores.jsonl and results/code_summary.md.
"""
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from collections import defaultdict

import code_tasks
from analyze import RESULTS_DIR, mean, strip_think

PROBLEMS = {p["id"]: p for p in code_tasks.PROBLEMS}
TIMEOUT_S = 10

RUNNER = r"""
import copy, json, sys

def norm(v):
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    return v

task = json.load(sys.stdin)
ns = {}
try:
    exec(task["code"], ns)
    fn = ns[task["func"]]
except Exception as e:
    print(json.dumps({"passed": 0, "errors": [f"load: {type(e).__name__}: {e}"]}))
    sys.exit()
passed, errors = 0, []
for args, expected in task["cases"]:
    try:
        got = norm(fn(*copy.deepcopy(args)))
        if got == expected:
            passed += 1
        else:
            errors.append(f"{args!r}: expected {expected!r}, got {got!r}")
    except Exception as e:
        errors.append(f"{args!r}: {type(e).__name__}: {e}")
print(json.dumps({"passed": passed, "errors": errors[:3]}))
"""


def extract_code(text):
    text = strip_think(text)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return max(blocks, key=len) if blocks else text


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S + 5, TIMEOUT_S + 5))


def run_tests(code, problem):
    task = {"code": code, "func": problem["func"], "cases": problem["cases"]}
    total = len(problem["cases"])
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", RUNNER], input=json.dumps(task),
                                  capture_output=True, text=True, timeout=TIMEOUT_S, cwd=tmp,
                                  preexec_fn=_limits)
            res = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            res = {"passed": 0, "errors": ["timeout"]}
        except (json.JSONDecodeError, IndexError):
            res = {"passed": 0, "errors": [f"crash: {proc.stderr.strip()[-200:]}"]}
    return {"passed": res["passed"], "total": total, "errors": res["errors"]}


def main():
    with open(os.path.join(RESULTS_DIR, "code_runs.jsonl"), encoding="utf-8") as f:
        runs = [json.loads(line) for line in f]
    scores = []
    for r in runs:
        res = run_tests(extract_code(r["content"]), PROBLEMS[r["problem_id"]])
        scores.append({k: r[k] for k in ("model", "mode", "problem_id", "repeat")} | res)
    with open(os.path.join(RESULTS_DIR, "code_scores.jsonl"), "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    groups = defaultdict(list)
    for s, r in zip(scores, runs):
        groups[(s["model"], s["mode"], s["problem_id"])].append((s, r))
    models = list(dict.fromkeys(s["model"] for s in scores))

    def stats(model, mode, pid):
        g = groups.get((model, mode, pid), [])
        return {
            "rate": mean(s["passed"] / s["total"] for s, _ in g),
            "solved": mean(float(s["passed"] == s["total"]) for s, _ in g),
            "out_tok": mean(r["usage"].get("completion_tokens") for _, r in g),
            "wall_s": mean(r["wall_s"] for _, r in g),
        }

    lines = ["# Coding suite (English)\n", "## Ranking (B_tuned, greedy decoding)\n",
             f"| rank | model | problems solved (of {len(PROBLEMS)}) | tests passed | A_base tests passed |",
             "|---|---|---|---|---|"]
    ranking = []
    for m in models:
        b = [stats(m, "B_tuned", pid) for pid in PROBLEMS]
        a = [stats(m, "A_base", pid) for pid in PROBLEMS]
        ranking.append((sum(x["solved"] for x in b), mean(x["rate"] for x in b), m, mean(x["rate"] for x in a)))
    ranking.sort(reverse=True)
    for i, (solved, rate, m, a_rate) in enumerate(ranking, 1):
        lines.append(f"| {i} | {m} | {solved:.0f} | {rate:.0%} | {a_rate:.0%} |")

    lines += ["\n## Per problem\n",
              "| model | problem | mode | tests passed | solved | out_tok | wall_s |",
              "|---|---|---|---|---|---|---|"]
    for m in models:
        for pid in PROBLEMS:
            for mode in ("A_base", "B_tuned"):
                st = stats(m, mode, pid)
                lines.append(f"| {m} | {pid} | {mode} | {st['rate']:.0%} | {st['solved']:.0%} | "
                             f"{st['out_tok']:.0f} | {st['wall_s']:.1f} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS_DIR, "code_summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
