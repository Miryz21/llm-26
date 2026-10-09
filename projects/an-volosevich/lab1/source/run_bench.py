"""Поднимает каждую модель в llama-server и прогоняет промпты в режимах A/B.

Все прогоны дописываются в results/runs.jsonl (одна строка — один запрос).
Использование:
    python3 run_bench.py                     # все модели
    python3 run_bench.py --models llama-3.1-8b-instruct --repeats 1
    python3 run_bench.py --no-server         # сервер уже запущен вручную
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import config

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
BASE_URL = f"http://{config.HOST}:{config.PORT}"


def http_json(method, path, payload=None, timeout=1800):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def start_server(model, log_path):
    cmd = [config.LLAMA_SERVER, "-m", os.path.join(config.MODELS_DIR, model["gguf"]),
           "--host", config.HOST, "--port", str(config.PORT),
           "--alias", model["name"], *model["args"]]
    print(f"[server] {' '.join(cmd)}", flush=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    deadline = time.time() + model["load_timeout"]
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}, see {log_path}")
        try:
            if http_json("GET", "/health", timeout=5).get("status") == "ok":
                return proc
        except (urllib.error.URLError, ConnectionError, json.JSONDecodeError):
            pass
        time.sleep(2)
    proc.terminate()
    raise TimeoutError(f"llama-server did not become healthy, see {log_path}")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()


def tokenize(prompt):
    """Отдельный замер скорости токенизации через POST /tokenize (без инференса)."""
    t0 = time.perf_counter()
    tokens = http_json("POST", "/tokenize", {"content": prompt})["tokens"]
    return {"n_tokens": len(tokens), "ms": round((time.perf_counter() - t0) * 1000, 3)}


def chat(model_name, prompt, params):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        # Отключаем кэш префикса, чтобы повторы честно измеряли скорость prefill.
        "cache_prompt": False,
        **params,
    }
    t0 = time.perf_counter()
    resp = http_json("POST", "/v1/chat/completions", payload)
    wall = time.perf_counter() - t0
    msg = resp["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "usage": resp.get("usage", {}),
        "timings": resp.get("timings", {}),
        "wall_s": round(wall, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="имена моделей из config.MODELS")
    ap.add_argument("--repeats", type=int, default=config.REPEATS)
    ap.add_argument("--no-server", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "runs.jsonl")
    models = [m for m in config.MODELS if not args.models or m["name"] in args.models]

    for model in models:
        proc = None
        if not args.no_server:
            proc = start_server(model, os.path.join(RESULTS_DIR, f"server_{model['name']}.log"))
        try:
            # Прогрев: подтягивает веса из page cache/SSD, в замеры не входит.
            chat(model["name"], "Привет!", {"max_tokens": 16})
            for mode, per_prompt in config.MODES.items():
                for prompt_id, prompt in config.PROMPTS.items():
                    params = per_prompt[prompt_id]
                    for rep in range(1, args.repeats + 1):
                        r = chat(model["name"], prompt, params)
                        r["tokenize"] = tokenize(prompt)
                        rec = {"ts": datetime.now(timezone.utc).isoformat(),
                               "model": model["name"], "family": model["family"],
                               "mode": mode, "prompt_id": prompt_id, "repeat": rep,
                               "params": params, **r}
                        with open(out_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        t = r["timings"]
                        print(f"{model['name']:30} {mode:8} {prompt_id:18} #{rep} "
                              f"out={r['usage'].get('completion_tokens')} "
                              f"pp={t.get('prompt_per_second', 0):.0f}t/s "
                              f"tg={t.get('predicted_per_second', 0):.1f}t/s "
                              f"wall={r['wall_s']:.1f}s finish={r['finish_reason']}",
                              flush=True)
        finally:
            if proc:
                stop_server(proc)


if __name__ == "__main__":
    sys.exit(main())
