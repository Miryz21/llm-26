"""Оценка retrieval: Recall@k, Precision@k, MRR, Hit@k для набора конфигураций индекса.

Эталон задан не id чанков (они меняются при каждом разбиении), а фразами-доказательствами
из текста статей (`evidence` в questions.json). Чанк релевантен, если он из нужной статьи
и содержит хотя бы одну фразу; сравнение — по строке из одних букв и цифр в нижнем регистре,
так что переносы, пробелы и разметка не мешают.

- Recall@k   — доля фраз-доказательств вопроса, покрытых top-k;
- Precision@k — доля релевантных чанков среди top-k;
- MRR        — 1 / ранг первого релевантного чанка (в пределах top-10);
- Hit@k      — есть ли в top-k хотя бы один релевантный чанк.

Каждая конфигурация проверяется на русских формулировках вопросов и на их английском переводе.
"""
import argparse
import json
import os
import re
import statistics
import time

from build_index import build, index_dir
from build_index import load as load_artifacts
from config import QUESTIONS_PATH, RESULTS_DIR, IndexConfig
from load_to_vector_store import DEFAULT_HNSW, client, load
from retriever import search

KS = (5, 10)

# Эксперименты шага 5: базовая конфигурация и вариации по одному параметру.
# (название, конфигурация индекса, параметры поиска, языки вопросов); «ru→en» — русский вопрос,
# переведённый LLM на английский перед поиском.
BASE = IndexConfig(splitter="markdown", chunk_size=400, overlap=60, headers=True, vectors="hybrid", model="bge-m3")
DENSE_BASE = IndexConfig(**{**BASE.to_dict(), "vectors": "dense"})
C800 = IndexConfig(**{**BASE.to_dict(), "chunk_size": 800, "overlap": 100})
BOTH, TRANSLATED = ("ru", "en"), ("ru→en",)
EXPERIMENTS = [
    ("база: hybrid bge-m3 + BM25", BASE, {}, BOTH),
    ("только dense bge-m3", DENSE_BASE, {}, BOTH),
    ("только sparse BM25", IndexConfig(**{**BASE.to_dict(), "vectors": "sparse"}), {}, BOTH),
    ("dense multilingual-e5-large", IndexConfig(**{**BASE.to_dict(), "vectors": "dense", "model": "multilingual-e5-large"}), {}, BOTH),
    ("hybrid multilingual-e5-large + BM25", IndexConfig(**{**BASE.to_dict(), "model": "multilingual-e5-large"}), {}, BOTH),
    ("dense e5-large-v2", IndexConfig(**{**BASE.to_dict(), "vectors": "dense", "model": "e5-large-v2"}), {}, BOTH),
    ("dense bge-base-en", IndexConfig(**{**BASE.to_dict(), "vectors": "dense", "model": "bge-base-en"}), {}, BOTH),
    ("RRF k = 60 вместо 2", BASE, {"rrf_k": 60}, BOTH),
    ("recursive-сплиттер", IndexConfig(**{**BASE.to_dict(), "splitter": "recursive"}), {}, BOTH),
    ("без заголовков в чанке", IndexConfig(**{**BASE.to_dict(), "headers": False}), {}, BOTH),
    ("чанк 200 / overlap 30", IndexConfig(**{**BASE.to_dict(), "chunk_size": 200, "overlap": 30}), {}, BOTH),
    ("чанк 800 / overlap 100", C800, {}, BOTH),
    ("чанк 400 без overlap", IndexConfig(**{**BASE.to_dict(), "overlap": 0}), {}, BOTH),
    ("база + перевод запроса", BASE, {"translate": True}, TRANSLATED),
    ("dense bge-m3 + перевод запроса", DENSE_BASE, {"translate": True}, TRANSLATED),
    ("чанк 800 + перевод запроса", C800, {"translate": True}, TRANSLATED),
]
# (название, параметры HNSW, квантование int8, ef при поиске, точный перебор)
HNSW_VARIANTS = [
    ("точный перебор (exact)", DEFAULT_HNSW, False, None, True),
    ("m=16, ef_construct=100, ef=64", DEFAULT_HNSW, False, 64, False),
    ("m=16, ef_construct=100, ef=8", DEFAULT_HNSW, False, 8, False),
    ("m=4, ef_construct=8, ef=4", {"m": 4, "ef_construct": 8}, False, 4, False),
    ("m=16, ef_construct=100, ef=64, int8", DEFAULT_HNSW, True, 64, False),
]

def norm(text: str) -> str:
    return re.sub(r"[^0-9a-zа-яё]", "", text.lower())


def relevant(chunk: dict, q: dict) -> list[int]:
    """Индексы фраз-доказательств вопроса, найденных в чанке."""
    if chunk["arxiv_id"] not in q["sources"]:
        return []
    text = norm(chunk["text"])
    return [i for i, phrase in enumerate(q["evidence"]) if norm(phrase) in text]


def score(results: list[dict], q: dict) -> dict:
    hits = [relevant(c, q) for c in results]
    out = {}
    for k in KS:
        covered = set().union(*hits[:k]) if hits[:k] else set()
        out[f"recall@{k}"] = len(covered) / len(q["evidence"])
        out[f"precision@{k}"] = sum(bool(h) for h in hits[:k]) / k
        out[f"hit@{k}"] = float(any(hits[:k]))
    first = next((i for i, h in enumerate(hits) if h), None)
    out["mrr"] = 0.0 if first is None else 1 / (first + 1)
    return out


def check_evidence(questions: list[dict], cfg: IndexConfig) -> None:
    """Каждая фраза-доказательство должна находиться хотя бы в одном чанке — иначе эталон сломан."""
    chunks, _, _ = load_artifacts(cfg)
    for q in questions:
        for i, phrase in enumerate(q["evidence"]):
            if not any(i in relevant(c, q) for c in chunks):
                raise SystemExit(f"{q['id']}: фраза «{phrase}» не найдена ни в одном чанке {cfg.index_id}")


def evaluate(qc, cfg: IndexConfig, questions: list[dict], lang: str, ef: int | None = 64, **opts) -> dict:
    per_question, latencies = [], []
    for q in questions:
        text = q["question_en"] if lang == "en" else q["question"]
        t = time.perf_counter()
        results = search(qc, cfg, text, max(KS), ef=ef, **opts)
        latencies.append(time.perf_counter() - t)
        per_question.append({"id": q["id"], **score(results, q),
                             "top": [f"{r['paper']} p.{','.join(map(str, r['pages']))}" for r in results[:5]]})
    metrics = {m: statistics.mean(p[m] for p in per_question) for m in per_question[0] if m not in ("id", "top")}
    metrics["latency_ms"] = 1000 * statistics.median(latencies)
    return {"metrics": metrics, "questions": per_question}


def build_info(cfg: IndexConfig) -> dict:
    with open(os.path.join(index_dir(cfg), "config.json"), encoding="utf-8") as f:
        return json.load(f)


def fmt(x: float) -> str:
    return f"{x:.3f}"


def table(rows: list[tuple[str, str, dict]], extra: list[str]) -> list[str]:
    head = ["Конфигурация", "Вопросы", "R@5", "R@10", "P@5", "P@10", "MRR", "Hit@5"] + extra
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, lang, m in rows:
        cells = [name, lang] + [fmt(m[k]) for k in ("recall@5", "recall@10", "precision@5", "precision@10",
                                                     "mrr", "hit@5")]
        cells += [str(m[e]) if not isinstance(m[e], float) else f"{m[e]:.1f}" for e in extra]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="номера экспериментов (с 1) для быстрой проверки; файлы результатов не пишутся")
    args = parser.parse_args()
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)
    qc = client()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    report, rows = {"experiments": [], "hnsw": []}, []
    selected = [e for i, e in enumerate(EXPERIMENTS, 1) if not args.only or str(i) in args.only]
    for name, cfg, opts, langs in selected:
        build(cfg)
        check_evidence(questions, cfg)
        load(cfg, qc=qc)
        info = build_info(cfg)
        for lang in langs:
            res = evaluate(qc, cfg, questions, lang, **opts)
            res["metrics"].update(chunks=info["chunks"])
            report["experiments"].append({"name": name, "index_id": cfg.index_id, "lang": lang, **res})
            rows.append((f"{name} (`{cfg.index_id}`)", lang, res["metrics"]))
            print(f"{name:<36} {lang:<5}  R@5 {res['metrics']['recall@5']:.3f}  MRR {res['metrics']['mrr']:.3f}")

    if args.only:
        return

    hnsw_rows = []
    for name, hnsw, quant, ef, exact in HNSW_VARIANTS:
        load(DENSE_BASE, hnsw, quant, drop=True, qc=qc)
        for lang in BOTH:
            res = evaluate(qc, DENSE_BASE, questions, lang, ef=ef, exact=exact)
            report["hnsw"].append({"name": name, "lang": lang, **res})
            hnsw_rows.append((name, lang, res["metrics"]))
    load(DENSE_BASE, drop=True, qc=qc)

    with open(os.path.join(RESULTS_DIR, "retrieval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    md = ["# Retrieval: сравнение конфигураций", "",
          f"{len(questions)} вопросов, k = 5 и 10. Латентность — медиана на запрос, включая эмбеддинг запроса "
          "(и перевод запроса LLM в строках ru→en).", ""]
    md += table(rows, ["chunks", "latency_ms"])
    md += ["", "## HNSW (dense bge-m3, база)", ""]
    md += table(hnsw_rows, ["latency_ms"])
    with open(os.path.join(RESULTS_DIR, "retrieval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
