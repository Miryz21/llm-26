"""Поиск по коллекции Qdrant: dense, sparse (BM25) или гибрид с RRF-слиянием на стороне Qdrant.

Опционально запрос сначала переводится на английский той же LLM, что генерирует ответы:
корпус англоязычный, а BM25 и англоязычные эмбеддинги не сопоставляют русские слова с английскими.
"""
import functools
import json
import re
import urllib.request

from qdrant_client import QdrantClient, models

from config import LLM_URL, IndexConfig
from encoders import bm25_query, dense_query
from load_to_vector_store import collection_name

PREFETCH = 30
# RRF: score = Σ w / (k + rank). k = 2 — значение Qdrant по умолчанию: малое k сильнее награждает первые
# места каждого списка. Без весов фрагменты с «зеркальными» рангами (1-й в dense и 3-й в BM25 против
# 3-го и 1-го) получают одинаковый score, и Qdrant возвращает их в произвольном порядке; чуть меньший
# вес BM25 детерминированно разрешает такие ничьи в пользу dense.
RRF_K = 2
RRF_WEIGHTS = [1.0, 0.99]
TRANSLATE = ("You are a translator. The user sends a search query in Russian inside <query> tags. "
             "Do NOT answer the query - only translate it into English. Keep names of methods, models, datasets "
             "and numbers unchanged. Output only the English translation as plain text, without tags.")
TRANSLATE_REQUEST = "Translate into English, do not answer:\n<query>{}</query>"
# Без явной разметки и примеров модель отвечает на вопрос вместо перевода:
# «Из каких двух компонентов состоит Direct3D?» → рассказ о Microsoft Direct3D 9/10/11.
TRANSLATE_EXAMPLES = [
    ("Сколько шагов DDIM используется в ViewCrafter?", "How many DDIM steps are used in ViewCrafter?"),
    ("Из каких модулей состоит DreamPainter и для чего нужен каждый?",
     "What modules does DreamPainter consist of and what is each of them for?"),
]


@functools.cache
def translate_query(query: str, url: str = LLM_URL) -> str:
    if not re.search(r"[а-яё]", query, re.IGNORECASE):
        return query
    messages = [{"role": "system", "content": TRANSLATE}]
    for ru, en in TRANSLATE_EXAMPLES:
        messages += [{"role": "user", "content": TRANSLATE_REQUEST.format(ru)}, {"role": "assistant", "content": en}]
    messages.append({"role": "user", "content": TRANSLATE_REQUEST.format(query)})
    body = json.dumps({"messages": messages, "temperature": 0, "max_tokens": 200, "cache_prompt": False}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = json.load(resp)["choices"][0]["message"]["content"]
    return re.sub(r"[*_`]+|</?query>", "", text).strip().strip('"«»')  # модель иногда выделяет термины markdown


def search(qc: QdrantClient, cfg: IndexConfig, query: str, k: int = 10,
           ef: int | None = None, exact: bool = False, translate: bool = False, rrf_k: int = RRF_K) -> list[dict]:
    if translate:
        query = translate_query(query)
    params = models.SearchParams(hnsw_ef=ef, exact=exact)
    name = collection_name(cfg)
    dense = dense_query(cfg.model, query) if cfg.uses_dense else None
    sparse = models.SparseVector(**bm25_query(query)) if cfg.uses_sparse else None

    if cfg.vectors == "hybrid":
        res = qc.query_points(
            name,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=PREFETCH, params=params),
                models.Prefetch(query=sparse, using="bm25", limit=PREFETCH),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k, weights=RRF_WEIGHTS)),
            limit=k,
        )
    elif cfg.vectors == "dense":
        res = qc.query_points(name, query=dense, using="dense", limit=k, search_params=params)
    else:
        res = qc.query_points(name, query=sparse, using="bm25", limit=k)
    return [{**p.payload, "score": p.score} for p in res.points]
