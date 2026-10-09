"""Runs the English coding suite on every model in modes A/B; appends to results/code_runs.jsonl.

One run per problem and mode (7 x 2 = 14 requests per model): greedy decoding in mode B is
deterministic, so a single run is enough for a reproducible ranking. Runs already present in
the results file are skipped, so an interrupted run can be resumed.

Usage:
    python3 run_code.py
    python3 run_code.py --models llama-3.1-8b-instruct --repeats 2
"""
import argparse
import json
import os
from datetime import datetime, timezone

import code_tasks
import config
from run_bench import RESULTS_DIR, chat, start_server, stop_server

MODES = {
    "A_base": {},
    "B_tuned": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "repeat_penalty": 1.0, "max_tokens": 8192},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "code_runs.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {(r["model"], r["mode"], r["problem_id"], r["repeat"]) for r in map(json.loads, f)}
    for model in (m for m in config.MODELS if not args.models or m["name"] in args.models):
        todo = [(mode, p, rep) for mode in MODES for p in code_tasks.PROBLEMS for rep in range(1, args.repeats + 1)
                if (model["name"], mode, p["id"], rep) not in done]
        if not todo:
            continue
        proc = start_server(model, os.path.join(RESULTS_DIR, f"server_{model['name']}.log"))
        try:
            chat(model["name"], "Hello!", {"max_tokens": 16})
            for mode, problem, rep in todo:
                params = MODES[mode]
                r = chat(model["name"], code_tasks.prompt(problem), params)
                rec = {"ts": datetime.now(timezone.utc).isoformat(), "model": model["name"],
                       "mode": mode, "problem_id": problem["id"], "repeat": rep, "params": params, **r}
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"{model['name']:30} {mode:8} {problem['id']:24} #{rep} "
                      f"out={r['usage'].get('completion_tokens')} wall={r['wall_s']:.1f}s "
                      f"finish={r['finish_reason']}", flush=True)
        finally:
            stop_server(proc)


if __name__ == "__main__":
    main()
