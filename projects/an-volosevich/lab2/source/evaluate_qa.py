"""Оценка ответов RAG-пайплайна на вопросах из questions.json и вопросах вне корпуса.

Автоматические метрики:
- key facts   — доля ключевых фактов (`answer_keys`: числа, названия), встретившихся в ответе;
- cite.source — доля процитированных фрагментов из нужной статьи;
- cite.evid.  — есть ли среди процитированных фрагмент с фразой-доказательством;
- cited       — сослалась ли модель хотя бы на один фрагмент;
- для вопросов вне корпуса — отказалась ли модель отвечать.
Все ответы с источниками сохраняются в results/qa_answers.md для ручной проверки.
"""
import argparse
import json
import os
import statistics

from config import QUESTIONS_PATH, RESULTS_DIR, TRANSLATE_QUERY, add_index_args, index_config_from_args
from evaluate import norm, relevant
from load_to_vector_store import client, load
from rag import answer

OUT_OF_CORPUS = [
    "Какой FID получает Stable Diffusion 3 на MS-COCO?",
    "Сколько параметров у модели Sora и на каких данных её обучали?",
]
REFUSAL = ("нет ответа", "не содержится", "не указ", "нет информации", "отсутствует")


def key_coverage(text: str, keys: list[str]) -> float:
    t = norm(text)
    return sum(norm(k) in t for k in keys) / len(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_index_args(parser)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--lang", choices=("ru", "en"), default="ru")
    parser.add_argument("--translate", action=argparse.BooleanOptionalAction, default=TRANSLATE_QUERY)
    parser.add_argument("--tag", default="", help="суффикс имён файлов результатов")
    args = parser.parse_args()
    cfg = index_config_from_args(args)
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    qc = client()
    load(cfg, qc=qc)

    mode = "перевод запроса" if args.translate else "без перевода"
    rows, md = [], [f"# Ответы RAG ({cfg.index_id}, top-{args.top_n}, {mode})", ""]
    for q in questions:
        text = q["question"] if args.lang == "ru" else q["question_en"]
        res = answer(text, cfg, args.top_n, qc=qc, translate=args.translate)
        cited = [res["contexts"][c["n"] - 1] for c in res["citations"]]
        row = {
            "id": q["id"],
            "key_facts": key_coverage(res["answer"], q["answer_keys"]),
            "cited": float(bool(cited)),
            "cite_source": statistics.mean(c["arxiv_id"] in q["sources"] for c in cited) if cited else 0.0,
            "cite_evidence": float(any(relevant(c, q) for c in cited)),
            "llm_s": res["timings"]["llm_s"],
            "tokens_in": res["usage"].get("prompt_tokens", 0),
        }
        rows.append(row)
        md += [f"## {q['id']}. {text}", "", res["answer"], "", "**Источники:**", ""]
        md += [f"- [{c['n']}] {c['paper']}, {c['section'] or '—'}, стр. {', '.join(map(str, c['pages']))} — "
               f"«{c['snippet']}»" for c in res["citations"]] or ["- —"]
        md += ["", f"*Эталон:* {q['reference']}", "",
               f"*Ключевые факты:* {row['key_facts']:.2f} · *цитаты из нужной статьи:* {row['cite_source']:.2f} · "
               f"*генерация:* {row['llm_s']} с", ""]
        print(f"{q['id']} facts {row['key_facts']:.2f} src {row['cite_source']:.2f} evid {row['cite_evidence']:.0f}")

    refusals = []
    for text in OUT_OF_CORPUS:
        res = answer(text, cfg, args.top_n, qc=qc, translate=args.translate)
        refused = any(r in res["answer"].lower() for r in REFUSAL)
        refusals.append(refused)
        md += [f"## Вне корпуса: {text}", "", res["answer"], "", f"*Отказ:* {'да' if refused else 'нет'}", ""]

    summary = {k: statistics.mean(r[k] for r in rows) for k in rows[0] if k != "id"}
    summary["refusal_rate"] = sum(refusals) / len(refusals)
    summary_md = ["| Метрика | Значение |", "|---|---|",
                  f"| ключевые факты в ответе | {summary['key_facts']:.2f} |",
                  f"| ответы с цитатами | {summary['cited']:.2f} |",
                  f"| цитаты из нужной статьи | {summary['cite_source']:.2f} |",
                  f"| среди цитат есть фрагмент-доказательство | {summary['cite_evidence']:.2f} |",
                  f"| отказ на вопросах вне корпуса | {sum(refusals)}/{len(refusals)} |",
                  f"| генерация, с (среднее) | {summary['llm_s']:.1f} |",
                  f"| токенов промпта (среднее) | {summary['tokens_in']:.0f} |"]
    md[2:2] = summary_md + [""]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"qa_answers{args.tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(RESULTS_DIR, f"qa{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"index_id": cfg.index_id, "top_n": args.top_n, "translate": args.translate,
                   "summary": summary, "questions": rows},
                  f, ensure_ascii=False, indent=1)
    print("\n".join(summary_md))


if __name__ == "__main__":
    main()
