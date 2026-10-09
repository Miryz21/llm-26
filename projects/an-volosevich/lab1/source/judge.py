"""LLM-as-a-judge: слепая оценка ответов.

Шаг 1:  python3 judge.py prepare
        Для каждого промпта пишет results/judge/<prompt_id>.md: промпт, рубрику и
        перемешанные ответы под анонимными id (без модели/режима). Соответствие
        id -> прогон хранится в results/judge/mapping.json и судье не показывается.
Шаг 2:  судья читает файлы и записывает оценки в results/judge/scores.jsonl —
        одна строка на ответ: {"id", <критерии 1..5>, "hallucination": bool,
        "overall": 1..5, "comment": "..."}.
Шаг 3:  python3 judge.py report — агрегирует оценки в results/judge/report.md
        (analyze.py тоже подхватывает overall в сводную таблицу).
"""
import hashlib
import json
import os
import random
import sys
from collections import defaultdict

import config
from analyze import RESULTS_DIR, mean, strip_think

JUDGE_DIR = os.path.join(RESULTS_DIR, "judge")

RUBRICS = {
    "P1_generation": {
        "faithfulness": "нет выдуманных фактов (сроки, суммы, имена, обещания вне списка)",
        "completeness": "присутствуют все факты: имя, № заказа, причина, 3 дня, промокод, 10%, срок, контакт, подпись",
        "tone": "деловой и тёплый тон, уместные извинения",
        "language": "грамотный естественный русский, без калек и смешения языков",
        "format": "форма письма, длина 120–180 слов, нет лишних пояснений вне письма",
    },
    "P2_classification": {
        "correctness": "доля верных меток (сверь с заданием сам, без подсказок)",
        "instruction_following": "только JSON-массив, допустимые метки, порядок и количество сохранены",
    },
    "P3_extraction": {
        "accuracy": "значения полей соответствуют тексту (числа нормализованы: 1 год -> 12 мес.)",
        "no_hallucination": "отсутствующие в тексте поля = null, ничего не додумано",
        "instruction_following": "только JSON, все ключи, числа числами",
    },
}

SCALE = ("Шкала каждого критерия 1–5 (5 — идеально). Дополнительно: hallucination "
         "(true, если в ответе есть утверждения, не следующие из промпта), overall 1–5 "
         "и короткий comment на русском.")


def load_runs():
    with open(os.path.join(RESULTS_DIR, "runs.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def run_key(r):
    return f"{r['model']}|{r['mode']}|{r['prompt_id']}|{r['repeat']}|{r['ts']}"


def prepare():
    os.makedirs(JUDGE_DIR, exist_ok=True)
    mapping = {}
    by_prompt = defaultdict(list)
    for r in load_runs():
        aid = hashlib.sha1(run_key(r).encode()).hexdigest()[:8]
        mapping[aid] = {k: r[k] for k in ("model", "mode", "prompt_id", "repeat", "ts")}
        by_prompt[r["prompt_id"]].append((aid, strip_think(r["content"])))

    rng = random.Random(42)
    for pid, answers in by_prompt.items():
        rng.shuffle(answers)
        crit = "\n".join(f"- **{k}**: {v}" for k, v in RUBRICS[pid].items())
        parts = [f"# Оценка: {pid}\n\n## Промпт\n\n```\n{config.PROMPTS[pid]}\n```\n",
                 f"## Рубрика\n\n{crit}\n\n{SCALE}\n\n## Ответы ({len(answers)})\n"]
        parts += [f"\n### id: {aid}\n\n```\n{text}\n```\n" for aid, text in answers]
        with open(os.path.join(JUDGE_DIR, f"{pid}.md"), "w", encoding="utf-8") as f:
            f.write("".join(parts))
    with open(os.path.join(JUDGE_DIR, "mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)
    print(f"prepared {len(mapping)} answers in {JUDGE_DIR}")


def load_scores():
    path = os.path.join(JUDGE_DIR, "scores.jsonl")
    if not os.path.exists(path):
        return {}
    with open(os.path.join(JUDGE_DIR, "mapping.json"), encoding="utf-8") as f:
        mapping = json.load(f)
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            m = mapping[s["id"]]
            out[(m["model"], m["prompt_id"], m["mode"])].append(s)
    return out


def report():
    scores = load_scores()
    lines = ["# LLM-as-a-judge (слепая оценка)\n"]
    for pid, rubric in RUBRICS.items():
        cols = list(rubric) + ["overall", "halluc_rate"]
        lines += [f"\n## {pid}\n", "| model | mode | " + " | ".join(cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
        for (model, p, mode), ss in sorted(scores.items()):
            if p != pid:
                continue
            vals = [mean(s.get(c) for s in ss) for c in list(rubric) + ["overall"]]
            vals.append(mean(float(s.get("hallucination", False)) for s in ss))
            lines.append(f"| {model} | {mode} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(JUDGE_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    {"prepare": prepare, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "prepare"]()
