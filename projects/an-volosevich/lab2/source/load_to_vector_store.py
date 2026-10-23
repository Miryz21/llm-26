"""Загрузка индекса из artifacts/<index_id>/ в коллекцию Qdrant `papers-<index_id>`.

Схема коллекции: именованный dense-вектор `dense` (косинус, HNSW, опционально скалярное
квантование int8) и sparse-вектор `bm25` с модификатором IDF; payload — чанк с метаданными
(статья, раздел, страницы, URL, текст).

Идемпотентность: id точки — UUID5 от id чанка; если коллекция уже актуальна, запись пропускается,
иначе — upsert и удаление лишних точек. После загрузки скрипт ждёт статуса green (HNSW достроен).
Параметры индекса и хэш корпуса хранятся в метаданных коллекции: если они расходятся с текущими,
скрипт останавливается и просит `--drop-and-reindex`. `--reset` удаляет коллекцию(и).
"""
import argparse
import time
import uuid

from qdrant_client import QdrantClient, models

import build_index
from config import QDRANT_URL, IndexConfig, add_index_args, index_config_from_args

PREFIX = "papers-"
DEFAULT_HNSW = {"m": 16, "ef_construct": 100}


def collection_name(cfg: IndexConfig) -> str:
    return PREFIX + cfg.index_id


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, timeout=60)


def wanted_metadata(cfg: IndexConfig, hnsw: dict, quantization: bool) -> dict:
    return {"index": cfg.to_dict(), "corpus": build_index.corpus_hash(), "hnsw": hnsw, "quantization": quantization}


def create(qc: QdrantClient, name: str, cfg: IndexConfig, dim: int | None, hnsw: dict, quantization: bool,
           metadata: dict) -> None:
    vectors, sparse = {}, {}
    if cfg.uses_dense:
        vectors["dense"] = models.VectorParams(size=dim, distance=models.Distance.COSINE)
    if cfg.uses_sparse:
        sparse["bm25"] = models.SparseVectorParams(modifier=models.Modifier.IDF)
    qc.create_collection(
        name,
        vectors_config=vectors,
        sparse_vectors_config=sparse,
        # Коллекция маленькая: без низких порогов (10 KB — минимум) Qdrant обошёлся бы полным перебором и не строил HNSW.
        hnsw_config=models.HnswConfigDiff(**hnsw, full_scan_threshold=10),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10),
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True),
        ) if quantization else None,
        metadata=metadata,
    )


def load(cfg: IndexConfig, hnsw: dict | None = None, quantization: bool = False,
         drop: bool = False, qc: QdrantClient | None = None) -> str:
    """Гарантирует, что коллекция для `cfg` существует и совпадает с артефактами. Возвращает её имя."""
    qc = qc or client()
    hnsw = hnsw or DEFAULT_HNSW
    build_index.build(cfg)
    chunks, dense, sparse = build_index.load(cfg)
    name = collection_name(cfg)
    metadata = wanted_metadata(cfg, hnsw, quantization)

    if qc.collection_exists(name):
        current = qc.get_collection(name).config.metadata or {}
        if drop:
            qc.delete_collection(name)
        elif current != metadata:
            raise SystemExit(f"{name}: параметры коллекции отличаются от запрошенных — "
                             f"запустите с --drop-and-reindex")
    if not qc.collection_exists(name):
        create(qc, name, cfg, None if dense is None else dense.shape[1], hnsw, quantization, metadata)

    points = []
    for i, chunk in enumerate(chunks):
        vector = {}
        if dense is not None:
            vector["dense"] = dense[i].tolist()
        if sparse is not None:
            vector["bm25"] = models.SparseVector(**sparse[i])
        payload = {k: v for k, v in chunk.items() if k != "embed_text"}
        points.append(models.PointStruct(id=point_id(chunk["id"]), vector=vector, payload=payload))

    keep, existing, offset = {p.id for p in points}, set(), None
    while True:
        batch, offset = qc.scroll(name, limit=256, offset=offset, with_payload=False)
        existing |= {str(p.id) for p in batch}
        if offset is None:
            break
    # Метаданные совпали (те же параметры и корпус) и набор точек тот же — коллекция актуальна.
    # Повторный upsert не нужен: он лишь плодит старые версии точек, которые до оптимизации
    # сегментов искажают IDF для BM25.
    if existing != keep:
        qc.upsert(name, points=points, wait=True)
        if existing - keep:
            qc.delete(name, points_selector=models.PointIdsList(points=list(existing - keep)), wait=True)
    wait_ready(qc, name)
    return name


def wait_ready(qc: QdrantClient, name: str, timeout: float = 120) -> None:
    """Ждёт, пока оптимизатор достроит HNSW и коллекция станет green."""
    deadline = time.monotonic() + timeout
    while qc.get_collection(name).status != models.CollectionStatus.GREEN and time.monotonic() < deadline:
        time.sleep(0.2)


def reset(qc: QdrantClient, name: str | None) -> list[str]:
    names = [name] if name else [c.name for c in qc.get_collections().collections if c.name.startswith(PREFIX)]
    for n in names:
        if qc.collection_exists(n):
            qc.delete_collection(n)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_index_args(parser)
    parser.add_argument("--hnsw-m", type=int, default=DEFAULT_HNSW["m"])
    parser.add_argument("--ef-construct", type=int, default=DEFAULT_HNSW["ef_construct"])
    parser.add_argument("--quantization", action="store_true", help="скалярное квантование dense-векторов в int8")
    parser.add_argument("--drop-and-reindex", action="store_true", help="удалить коллекцию и загрузить заново")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать артефакты build_index перед загрузкой")
    parser.add_argument("--reset", action="store_true", help="только удалить коллекцию текущей конфигурации")
    parser.add_argument("--reset-all", action="store_true", help=f"удалить все коллекции {PREFIX}*")
    args = parser.parse_args()
    cfg = index_config_from_args(args)
    qc = client()

    if args.reset or args.reset_all:
        removed = reset(qc, None if args.reset_all else collection_name(cfg))
        print("удалены:", ", ".join(removed) or "—")
        return
    if args.rebuild:
        build_index.build(cfg, force=True)
    name = load(cfg, {"m": args.hnsw_m, "ef_construct": args.ef_construct}, args.quantization,
                args.drop_and_reindex, qc)
    info = qc.get_collection(name)
    print(f"{name}: {info.points_count} точек, статус {info.status.value}, "
          f"индексировано векторов: {info.indexed_vectors_count}")


if __name__ == "__main__":
    main()
