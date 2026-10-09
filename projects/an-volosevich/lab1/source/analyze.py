"""Считает метрики по results/runs.jsonl и пишет results/summary.md.

Метрики на группу (модель × промпт × режим), усреднённые по повторам:
  - out_tok    — токены ответа (completion_tokens, включая reasoning у Qwen)
  - in_tok     — токены промпта
  - wall_s     — время отклика (полный запрос, без стриминга)
  - pp_tps     — скорость обработки промпта (prefill), ток/с — из timings llama-server
  - tg_tps     — скорость генерации (decode), ток/с
  - tok_tps    — скорость токенизации (POST /tokenize, вкл. HTTP-накладные), ток/с
  - quality    — доля выполненных требований задачи (см. score_* ниже), 0..1
  - uniq       — число различных ответов среди повторов (1 = детерминизм)
  - sim        — средняя попарная похожесть ответов повторов (difflib), 0..1
  - rep3       — доля повторяющихся словесных 3-грамм внутри ответа (зацикливание)
  - rep3_r     — то же для блока рассуждений (Qwen)
  - judge      — средний overall 1..5 от LLM-судьи (если есть results/judge/scores.jsonl)
"""
import difflib
import itertools
import json
import os
import re
import statistics
from collections import defaultdict

import config

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def extract_json(text):
    text = strip_think(text)
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def score_generation(text):
    """Проверки фактов из промпта + ограничение по длине (120–180 слов)."""
    text = strip_think(text)
    words = len(re.findall(r"\w+", text))
    plain = re.sub(r"[«»\"*_]", "", text)  # кавычки/markdown не считаем ошибкой
    checks = {
        "order": "48213" in text,
        "promo": "SORRY10" in text,
        "discount": "10%" in text or "10 %" in text,
        "deadline": "31 декабря" in text,
        "email": "support@technomir.ru" in text,
        "signature": "команда техномир" in plain.lower(),
        "length": 120 <= words <= 180,
    }
    return sum(checks.values()) / len(checks), {"words": words, **checks}


def score_classification(text):
    pred = extract_json(text)
    gold = [label for _, label in config.P2_ITEMS]
    if not isinstance(pred, list):
        return 0.0, {"valid_json": False}
    correct = sum(1 for p, g in zip(pred, gold) if str(p).strip() == g)
    return correct / len(gold), {"valid_json": True, "n_pred": len(pred),
                                 "pred": pred}


def _norm(v):
    if isinstance(v, str):
        v = v.strip().lower().replace("ё", "е").strip("«»\"' ")
        try:
            return float(v.replace(",", ".").replace(" ", ""))
        except ValueError:
            return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return v


def score_extraction(text):
    pred = extract_json(text)
    if not isinstance(pred, dict):
        return 0.0, {"valid_json": False}
    wrong = {k: pred.get(k) for k, g in config.P3_GOLD.items() if _norm(pred.get(k)) != _norm(g)}
    score = 1 - len(wrong) / len(config.P3_GOLD)
    return score, {"valid_json": True, "wrong": wrong}


SCORERS = {
    "P1_generation": score_generation,
    "P2_classification": score_classification,
    "P3_extraction": score_extraction,
}


def rep3(text):
    words = re.findall(r"\w+", strip_think(text).lower())
    grams = list(zip(words, words[1:], words[2:]))
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else float("nan")


def main():
    runs = [json.loads(line) for line in open(os.path.join(RESULTS_DIR, "runs.jsonl"), encoding="utf-8")]
    from judge import load_scores  # ленивый импорт: judge сам импортирует analyze
    judge_scores = load_scores()
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["prompt_id"], r["mode"])].append(r)

    rows, details = [], []
    for (model, pid, mode), rs in sorted(groups.items()):
        scores = [SCORERS[pid](r["content"]) for r in rs]
        contents = [strip_think(r["content"]) for r in rs]
        sims = [difflib.SequenceMatcher(None, a, b).ratio()
                for a, b in itertools.combinations(contents, 2)]
        rows.append({
            "model": model, "prompt": pid, "mode": mode, "n": len(rs),
            "in_tok": mean(r["usage"].get("prompt_tokens") for r in rs),
            "out_tok": mean(r["usage"].get("completion_tokens") for r in rs),
            "reason_ch": mean(len(r["reasoning"]) for r in rs),
            "wall_s": mean(r["wall_s"] for r in rs),
            "pp_tps": mean(r["timings"].get("prompt_per_second") for r in rs),
            "tg_tps": mean(r["timings"].get("predicted_per_second") for r in rs),
            "tok_tps": mean(r["tokenize"]["n_tokens"] / r["tokenize"]["ms"] * 1000
                            for r in rs if r.get("tokenize")),
            "quality": mean(s for s, _ in scores),
            "uniq": len(set(contents)),
            "sim": mean(sims) if sims else 1.0,
            "rep3": mean(rep3(c) for c in contents),
            "rep3_r": mean(rep3(r["reasoning"]) for r in rs),
            "truncated": sum(r["finish_reason"] == "length" for r in rs),
            "judge": mean(j.get("overall") for j in judge_scores.get((model, pid, mode), [])),
        })
        for r, (s, info) in zip(rs, scores):
            details.append({"model": model, "prompt": pid, "mode": mode,
                            "repeat": r["repeat"], "score": round(s, 3), **info})

    cols = ["model", "prompt", "mode", "n", "in_tok", "out_tok", "reason_ch", "wall_s",
            "pp_tps", "tg_tps", "tok_tps", "quality", "judge", "uniq", "sim", "rep3", "rep3_r", "truncated"]
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for row in rows]

    # Сводка по моделям: скорость и качество, режим A vs B.
    agg = ["| model | mode | avg pp_tps | avg tg_tps | avg wall_s | avg quality | avg judge | avg sim |",
           "|---|---|---|---|---|---|---|---|"]
    for (model, mode), rs in itertools.groupby(sorted(rows, key=lambda x: (x["model"], x["mode"])),
                                               key=lambda x: (x["model"], x["mode"])):
        rs = list(rs)
        agg.append(f"| {model} | {mode} | {mean(r['pp_tps'] for r in rs):.0f} | "
                   f"{mean(r['tg_tps'] for r in rs):.1f} | {mean(r['wall_s'] for r in rs):.1f} | "
                   f"{mean(r['quality'] for r in rs):.2f} | {mean(r['judge'] for r in rs):.2f} | "
                   f"{mean(r['sim'] for r in rs):.2f} |")

    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Сводка прогонов\n\n## По моделям\n\n" + "\n".join(agg)
                + "\n\n## Детально (модель × промпт × режим)\n\n" + "\n".join(lines) + "\n")
    with open(os.path.join(RESULTS_DIR, "auto_scores.jsonl"), "w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("\n".join(agg) + "\n\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
