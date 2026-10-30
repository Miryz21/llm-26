"""Step 3b: embed the chunks with bge-m3, as dense and/or sparse vectors.

Vector types, chosen with --vectors:
    dense    one 1024-dim vector per chunk (meaning)
    sparse   bge-m3's learned word weights per chunk (exact terms, multilingual)
    hybrid   both; the weighted score dense + w·sparse is computed at search time,
             so the weight can be tuned without rebuilding the index

Reads data/chunks/<chunks>/chunks.jsonl, writes data/index/<chunks>__<vectors>-bge-m3/:
    dense.npy     float32 matrix, one normalized row per chunk (dense, hybrid)
    sparse.jsonl  {"indices": [...], "values": [...]} per chunk (sparse, hybrid)
    config.json   settings + statistics

Usage: python build_index.py [--chunks markdown-800-100-h] [--vectors hybrid] [--force]
"""
import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import typer
from FlagEmbedding import BGEM3FlagModel
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


# --- Settings ----------------------------------------------------------------------------

class IndexSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_INDEX_")

    chunks: str = "markdown-800-100-h"  # name of a chunk build from data_chunking.py
    vectors: Literal["dense", "sparse", "hybrid"] = "hybrid"
    model: str = "BAAI/bge-m3"
    # FlagEmbedding cuts inputs at 512 tokens by default; bge-m3 itself accepts 8192.
    # 1024 covers our largest chunks; longer inputs stop the build instead of being cut.
    max_length: int = 1024
    batch_size: int = 16
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"

    @property
    def name(self) -> str:
        """Readable build name, e.g. "markdown-800-100-h__hybrid-bge-m3"; also the output folder."""
        return f"{self.chunks}__{self.vectors}-{self.model.split('/')[-1].lower()}"

    @property
    def chunks_file(self) -> Path:
        return self.data_dir / "chunks" / self.chunks / "chunks.jsonl"


# --- Steps -------------------------------------------------------------------------------

def is_up_to_date(settings: IndexSettings, out_dir: Path) -> bool:
    """A build can be reused if it was made with the same settings and after its chunks."""
    config_file = out_dir / "config.json"
    if not config_file.exists():
        return False
    saved = json.loads(config_file.read_text(encoding="utf-8"))["settings"]
    if saved != settings.model_dump(mode="json", exclude={"data_dir"}):
        return False
    return config_file.stat().st_mtime > settings.chunks_file.stat().st_mtime


def load_texts(settings: IndexSettings) -> list[str]:
    """The embed_text of every chunk, in chunks.jsonl order (row i of the index = line i)."""
    if not settings.chunks_file.exists():
        raise FileNotFoundError(f"No chunk build {settings.chunks!r}: run data_chunking.py with matching options first")
    with settings.chunks_file.open(encoding="utf-8") as f:
        return [json.loads(line)["embed_text"] for line in f]


def check_input_length(texts: list[str], model: BGEM3FlagModel, max_length: int) -> int:
    """Stop if any text is longer than the model will read; return the longest length."""
    # count with the model's own tokenizer, plus its 2 special tokens (start and end)
    lengths = [len(model.tokenizer.tokenize(t)) + 2 for t in texts]
    too_long = [i for i, n in enumerate(lengths) if n > max_length]
    if too_long:
        raise ValueError(f"{len(too_long)} chunks are longer than max_length={max_length} "
                         f"(longest: {max(lengths)} tokens); they would be cut off silently")
    return max(lengths)


def embed(texts: list[str], model: BGEM3FlagModel, settings: IndexSettings) -> dict:
    """One bge-m3 pass returns both the dense vectors and the sparse word weights."""
    output = model.encode(
        texts,
        batch_size=settings.batch_size,
        max_length=settings.max_length,
        return_dense=settings.vectors in ("dense", "hybrid"),
        return_sparse=settings.vectors in ("sparse", "hybrid"),
    )
    # FlagEmbedding returns None for outputs that were not requested; drop them
    return {key: value for key, value in output.items() if value is not None}


def save(output: dict, settings: IndexSettings, out_dir: Path, stats: dict) -> None:
    """Write the vectors, then config.json last: its presence marks the build as complete."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if "dense_vecs" in output:
        # already L2-normalized by FlagEmbedding, so dot product = cosine similarity
        np.save(out_dir / "dense.npy", output["dense_vecs"].astype(np.float32))

    if "lexical_weights" in output:
        # {token_id: weight} → sparse vector. Their dot product with the query's weights is
        # exactly bge-m3's lexical matching score, so no IDF or other weighting is needed.
        with (out_dir / "sparse.jsonl").open("w", encoding="utf-8") as f:
            for weights in output["lexical_weights"]:
                indices = [int(token) for token in weights]
                values = [float(w) for w in weights.values()]
                f.write(json.dumps({"indices": indices, "values": values}) + "\n")

    config = {"settings": settings.model_dump(mode="json", exclude={"data_dir"}), "stats": stats}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def build_index(settings: IndexSettings | None = None, force: bool = False) -> Path:
    """Embed one chunk build (or reuse an up-to-date index) and return the index folder."""
    settings = settings or IndexSettings()
    out_dir = settings.data_dir / "index" / settings.name
    texts = load_texts(settings)
    if not force and is_up_to_date(settings, out_dir):
        log.info("Index %s is up to date", settings.name)
        return out_dir

    model = BGEM3FlagModel(settings.model, use_fp16=True)  # fp16 on the GPU: half the memory, same ranking
    longest = check_input_length(texts, model, settings.max_length)

    start = time.time()
    output = embed(texts, model, settings)
    stats = {"chunks": len(texts), "longest_input_tokens": longest, "seconds": round(time.time() - start, 1)}
    if "dense_vecs" in output:
        stats["dense_dim"] = int(output["dense_vecs"].shape[1])
    if "lexical_weights" in output:
        stats["sparse_terms_mean"] = round(float(np.mean([len(w) for w in output["lexical_weights"]])))

    save(output, settings, out_dir, stats)
    log.info("%s: %s", settings.name, stats)
    return out_dir


# --- CLI ---------------------------------------------------------------------------------

def main(
    chunks: str | None = typer.Option(None, help="Chunk build to embed, e.g. markdown-800-100-h."),
    vectors: str | None = typer.Option(None, help="dense, sparse or hybrid."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if up to date."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # hide Hugging Face request logs and FlagEmbedding's model-loading notes
    for name in ("httpx", "FlagEmbedding"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # flags that were given override the settings (defaults or RAG_INDEX_* variables)
    flags = {"chunks": chunks, "vectors": vectors}
    settings = IndexSettings(**{k: v for k, v in flags.items() if v is not None})
    log.info("Index is in %s", build_index(settings, force=force))


if __name__ == "__main__":
    typer.run(main)
