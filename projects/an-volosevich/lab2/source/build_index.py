"""Чанки и эмбеддинги → artifacts/<index_id>/.

Каталог артефактов именуется по параметрам (splitter, chunk_size, overlap, заголовки,
тип векторов, модель), поэтому разные конфигурации не затирают друг друга. Повторный запуск
с теми же параметрами ничего не пересчитывает, если не изменился исходный Markdown
(сверяется хэш); `--force` пересобирает принудительно.

Артефакты: config.json — параметры и хэш корпуса, chunks.jsonl — чанки с метаданными,
dense.npy — матрица эмбеддингов, sparse.jsonl — BM25-веса.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from chunking import build_chunks
from config import ARTIFACTS_DIR, MARKDOWN_DIR, PAPERS, IndexConfig, add_index_args, index_config_from_args
from encoders import bm25_documents, dense_documents


def corpus_hash() -> str:
    h = hashlib.sha256()
    for paper in PAPERS:
        with open(os.path.join(MARKDOWN_DIR, f"{paper['arxiv_id']}.md"), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def index_dir(cfg: IndexConfig) -> str:
    return os.path.join(ARTIFACTS_DIR, cfg.index_id)


def is_fresh(cfg: IndexConfig) -> bool:
    path = os.path.join(index_dir(cfg), "config.json")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    return saved["index"] == cfg.to_dict() and saved["corpus"] == corpus_hash()


def build(cfg: IndexConfig, force: bool = False) -> str:
    out = index_dir(cfg)
    if is_fresh(cfg) and not force:
        return out
    started = time.perf_counter()
    chunks = build_chunks(cfg)
    texts = [c["embed_text"] for c in chunks]
    os.makedirs(out, exist_ok=True)
    timings = {"chunking_s": round(time.perf_counter() - started, 2)}

    if cfg.uses_dense:
        t = time.perf_counter()
        np.save(os.path.join(out, "dense.npy"), dense_documents(cfg.model, texts))
        timings["dense_s"] = round(time.perf_counter() - t, 2)
    if cfg.uses_sparse:
        with open(os.path.join(out, "sparse.jsonl"), "w", encoding="utf-8") as f:
            for vec in bm25_documents(texts):
                f.write(json.dumps(vec) + "\n")
    with open(os.path.join(out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    # config.json пишется последним: его наличие означает, что артефакты собраны целиком.
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"index": cfg.to_dict(), "index_id": cfg.index_id, "corpus": corpus_hash(),
                   "chunks": len(chunks), "timings": timings}, f, ensure_ascii=False, indent=2)
    return out


def load(cfg: IndexConfig) -> tuple[list[dict], np.ndarray | None, list[dict] | None]:
    out = index_dir(cfg)
    with open(os.path.join(out, "chunks.jsonl"), encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    dense = np.load(os.path.join(out, "dense.npy")) if cfg.uses_dense else None
    sparse = None
    if cfg.uses_sparse:
        with open(os.path.join(out, "sparse.jsonl"), encoding="utf-8") as f:
            sparse = [json.loads(line) for line in f]
    return chunks, dense, sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_index_args(parser)
    parser.add_argument("--force", action="store_true", help="пересобрать, даже если артефакты актуальны")
    args = parser.parse_args()
    cfg = index_config_from_args(args)
    fresh = is_fresh(cfg) and not args.force
    out = build(cfg, args.force)
    with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
        info = json.load(f)
    state = "актуален, пропущено" if fresh else "собран"
    print(f"{cfg.index_id}: {info['chunks']} чанков, {state} → {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
