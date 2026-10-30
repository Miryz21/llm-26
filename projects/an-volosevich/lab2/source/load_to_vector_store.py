"""Step 4: load an index build into a Qdrant collection.

Reads data/index/<index>/ (dense.npy and/or sparse.jsonl) and data/chunks/<chunks>/chunks.jsonl,
writes the Qdrant collection "lab2-<index>": one point per chunk with its vectors and payload.

Requires a running Qdrant server (see QDRANT_START below).
Usage: python load_to_vector_store.py [--index markdown-800-100-h__hybrid-bge-m3] [--hnsw-m 16]
       [--ef-construct 100] [--quantization int8] [--rebuild] [--drop-and-reindex] [--reset] [--reset-all]
"""
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import typer
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient, models

from build_index import IndexSettings, build_index

log = logging.getLogger(__name__)

PREFIX = "lab2-"  # every collection of this lab; --reset-all deletes only these
QDRANT_START = ("cd ~/llms/qdrant && QDRANT__STORAGE__STORAGE_PATH=~/llms/qdrant/storage "
                "QDRANT__STORAGE__SNAPSHOTS_PATH=~/llms/qdrant/snapshots QDRANT__TELEMETRY_DISABLED=true ./qdrant")


# --- Settings ----------------------------------------------------------------------------

class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_QDRANT_")

    url: str = "http://localhost:6333"
    index: str = "markdown-800-100-h__hybrid-bge-m3"  # name of an index build from build_index.py
    hnsw_m: int = 16  # links per node in the HNSW graph (dense vectors only)
    ef_construct: int = 100  # search depth while building the graph
    quantization: Literal["none", "int8"] = "none"  # int8: 4x less RAM for dense vectors
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"

    @property
    def collection(self) -> str:
        return PREFIX + self.index

    @property
    def schema(self) -> dict:
        """Everything that is fixed when the collection is created."""
        return {"index": self.index, "hnsw_m": self.hnsw_m, "ef_construct": self.ef_construct,
                "quantization": self.quantization}


# --- Reading the index build -------------------------------------------------------------

def read_index(settings: VectorStoreSettings) -> tuple[list[dict], np.ndarray | None, list[dict] | None]:
    """Chunks, dense matrix and sparse vectors of the index build; row i belongs to chunk i."""
    index_dir = settings.data_dir / "index" / settings.index
    if not (index_dir / "config.json").exists():
        raise FileNotFoundError(f"No index build {settings.index!r}: run build_index.py, or use --rebuild")

    chunks_name = json.loads((index_dir / "config.json").read_text(encoding="utf-8"))["settings"]["chunks"]
    with (settings.data_dir / "chunks" / chunks_name / "chunks.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    dense = np.load(index_dir / "dense.npy") if (index_dir / "dense.npy").exists() else None
    sparse = None
    if (index_dir / "sparse.jsonl").exists():
        with (index_dir / "sparse.jsonl").open(encoding="utf-8") as f:
            sparse = [json.loads(line) for line in f]
    return chunks, dense, sparse


def fingerprint(settings: VectorStoreSettings, chunks: list[dict]) -> str:
    """sha256 of the vectors and chunks: if it matches the collection, nothing needs loading."""
    digest = hashlib.sha256(json.dumps(chunks, sort_keys=True).encode())
    for name in ("dense.npy", "sparse.jsonl"):
        path = settings.data_dir / "index" / settings.index / name
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


# --- Qdrant ------------------------------------------------------------------------------

def connect(settings: VectorStoreSettings) -> QdrantClient:
    client = QdrantClient(settings.url)
    try:
        client.get_collections()
    except Exception as error:
        raise SystemExit(f"Qdrant is not reachable at {settings.url} ({error}).\nStart it with:\n  {QDRANT_START}")
    return client


def create_collection(client: QdrantClient, settings: VectorStoreSettings, dense: np.ndarray | None,
                      sparse: list[dict] | None) -> None:
    dense_config = {}
    if dense is not None:
        # vectors are L2-normalized, so cosine = dot product
        dense_config["dense"] = models.VectorParams(size=dense.shape[1], distance=models.Distance.COSINE)
    sparse_config = {}
    if sparse is not None:
        # plain dot product of bge-m3 weights = its lexical score, so no IDF modifier
        sparse_config["sparse"] = models.SparseVectorParams()

    quantization = None
    if settings.quantization == "int8":
        quantization = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True))

    client.create_collection(
        settings.collection,
        vectors_config=dense_config,
        sparse_vectors_config=sparse_config,
        # Thresholds at Qdrant's minimum (10 KB): our ~400 KB collection is far below the
        # 10 000 KB default, and Qdrant would otherwise never build HNSW and just scan everything.
        hnsw_config=models.HnswConfigDiff(m=settings.hnsw_m, ef_construct=settings.ef_construct,
                                          full_scan_threshold=10),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10),
        quantization_config=quantization,
        metadata={"schema": settings.schema, "fingerprint": None},  # fingerprint is set after upload
    )
    log.info("Created collection %s", settings.collection)


def upload(client: QdrantClient, settings: VectorStoreSettings, chunks: list[dict],
           dense: np.ndarray | None, sparse: list[dict] | None) -> None:
    """Upsert one point per chunk and delete points of chunks that no longer exist."""
    points = []
    for i, chunk in enumerate(chunks):
        vector = {}
        if dense is not None:
            vector["dense"] = dense[i].tolist()
        if sparse is not None:
            vector["sparse"] = models.SparseVector(indices=sparse[i]["indices"], values=sparse[i]["values"])
        # payload: everything needed for the prompt and citations (embed_text is not needed)
        payload = {key: value for key, value in chunk.items() if key != "embed_text"}
        points.append(models.PointStruct(id=point_id(chunk["chunk_id"]), vector=vector, payload=payload))
    client.upload_points(settings.collection, points, batch_size=64, wait=True)

    # chunks that disappeared after a rebuild would otherwise stay in the collection
    new_ids = {p.id for p in points}
    old_ids = {str(p.id) for p in client.scroll(settings.collection, limit=100_000, with_payload=False)[0]}
    stale = list(old_ids - new_ids)
    if stale:
        client.delete(settings.collection, points_selector=models.PointIdsList(points=stale))
    log.info("Uploaded %d points, deleted %d stale ones", len(points), len(stale))


def point_id(chunk_id: str) -> str:
    """Stable UUID from the chunk id, so loading again overwrites points instead of adding new ones."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def wait_until_indexed(client: QdrantClient, collection: str, timeout: float = 120) -> None:
    """Qdrant builds HNSW in the background; status green means it is done."""
    start = time.time()
    while client.get_collection(collection).status != models.CollectionStatus.GREEN:
        if time.time() - start > timeout:
            raise TimeoutError(f"{collection} was not indexed within {timeout} s")
        time.sleep(0.5)


# --- Load --------------------------------------------------------------------------------

def load(settings: VectorStoreSettings | None = None, drop: bool = False) -> str:
    """Make the collection match the index build; return the collection name.

    Missing → create and upload. Different schema (HNSW, quantization) → stop: it is fixed at
    creation, so this needs drop=True. Same data → nothing to do. Otherwise → upload again.
    """
    settings = settings or VectorStoreSettings()
    client = connect(settings)
    chunks, dense, sparse = read_index(settings)
    current = fingerprint(settings, chunks)

    if drop and client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
        log.info("Dropped collection %s", settings.collection)

    if not client.collection_exists(settings.collection):
        create_collection(client, settings, dense, sparse)
    else:
        metadata = client.get_collection(settings.collection).config.metadata or {}
        if metadata.get("schema") != settings.schema:
            raise SystemExit(f"{settings.collection} was created with {metadata.get('schema')}, "
                             f"now {settings.schema}: run with --drop-and-reindex")
        if metadata.get("fingerprint") == current:
            log.info("Collection %s is up to date", settings.collection)
            return settings.collection

    upload(client, settings, chunks, dense, sparse)
    wait_until_indexed(client, settings.collection)
    # written last: a matching fingerprint means a complete upload
    client.update_collection(settings.collection, metadata={"schema": settings.schema, "fingerprint": current})
    return settings.collection


def reset(settings: VectorStoreSettings, everything: bool = False) -> None:
    """Delete this collection, or every lab2-* collection."""
    client = connect(settings)
    names = [c.name for c in client.get_collections().collections]
    targets = [n for n in names if n.startswith(PREFIX)] if everything else [n for n in names if n == settings.collection]
    for name in targets:
        client.delete_collection(name)
    log.info("Deleted %d collection(s): %s", len(targets), ", ".join(targets) or "none")


# --- CLI ---------------------------------------------------------------------------------

def main(
    index: str | None = typer.Option(None, help="Index build to load, e.g. markdown-800-100-h__hybrid-bge-m3."),
    hnsw_m: int | None = typer.Option(None, help="HNSW links per node."),
    ef_construct: int | None = typer.Option(None, help="HNSW search depth while building."),
    quantization: str | None = typer.Option(None, help="none or int8."),
    rebuild: bool = typer.Option(False, "--rebuild", help="(Re)build the index build first if it is missing or stale."),
    drop_and_reindex: bool = typer.Option(False, "--drop-and-reindex", help="Delete the collection and load it again."),
    reset_one: bool = typer.Option(False, "--reset", help="Only delete this collection."),
    reset_all: bool = typer.Option(False, "--reset-all", help="Only delete all lab2-* collections."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide every HTTP request to Qdrant

    # flags that were given override the settings (defaults or RAG_QDRANT_* variables)
    flags = {"index": index, "hnsw_m": hnsw_m, "ef_construct": ef_construct, "quantization": quantization}
    settings = VectorStoreSettings(**{k: v for k, v in flags.items() if v is not None})

    if reset_one or reset_all:
        reset(settings, everything=reset_all)
        return
    if rebuild:
        # the index name is "<chunks>__<vectors>-<model>", e.g. "markdown-800-100-h__hybrid-bge-m3"
        chunks, vectors_model = settings.index.rsplit("__", 1)
        build_index(IndexSettings(chunks=chunks, vectors=vectors_model.split("-", 1)[0]))
    log.info("Loaded into %s", load(settings, drop=drop_and_reindex))


if __name__ == "__main__":
    typer.run(main)
