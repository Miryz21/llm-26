"""Векторизация: dense (sentence-transformers) и sparse BM25.

BM25 разложен между клиентом и Qdrant: клиент хранит в разреженном векторе документа
насыщенную частоту термина tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl)), а IDF считает сам Qdrant
по всей коллекции (`modifier=IDF`). Вектор запроса — единицы для уникальных терминов,
так что скалярное произведение даёт ровно BM25.
"""
import functools
import re
import zlib
from collections import Counter

import numpy as np

from config import DENSE_MODELS

K1, B = 1.2, 0.75
STOPWORDS = set("""
a an and are as at be by for from has have in into is it its of on or that the their them then there these this
those to was were which with we our us can also than such via using used use based both each other while when
where how what does do""".split())
TOKEN = re.compile(r"\w+(?:[-.]\w+)*", re.UNICODE)


def stem(token: str) -> str:
    """Минимальный стеммер для английского: убирает окончания множественного числа."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def terms(text: str) -> list[str]:
    return [stem(t) for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def term_id(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def bm25_documents(texts: list[str]) -> list[dict]:
    docs = [Counter(terms(t)) for t in texts]
    avgdl = sum(sum(d.values()) for d in docs) / len(docs)
    vectors = []
    for doc in docs:
        dl = sum(doc.values())
        weights = {}
        for term, tf in doc.items():
            weights[term_id(term)] = weights.get(term_id(term), 0.0) + \
                tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))
        ids = sorted(weights)
        vectors.append({"indices": ids, "values": [round(weights[i], 4) for i in ids]})
    return vectors


def bm25_query(text: str) -> dict:
    ids = sorted({term_id(t) for t in terms(text)})
    return {"indices": ids, "values": [1.0] * len(ids)}


@functools.cache
def dense_model(name: str):
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(DENSE_MODELS[name]["hf"], device="cuda" if torch.cuda.is_available() else "cpu",
                                model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = min(model.max_seq_length, 1024)
    return model


def dense_documents(name: str, texts: list[str]) -> np.ndarray:
    prefix = DENSE_MODELS[name]["doc"]
    return dense_model(name).encode([prefix + t for t in texts], batch_size=8, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)


def dense_query(name: str, text: str) -> list[float]:
    prefix = DENSE_MODELS[name]["query"]
    return dense_model(name).encode(prefix + text, normalize_embeddings=True).astype(np.float32).tolist()
