"""Step 5a: find the chunks that answer a question, in a Qdrant collection from load_to_vector_store.py.

Search modes, chosen with --mode (a hybrid collection serves all three):
    dense    meaning: bge-m3 dense vector, HNSW search
    sparse   exact terms: bge-m3 sparse word weights
    hybrid   weighted score dense + w·sparse, computed by Qdrant in one request

Optionally (--translate) a Russian question is first translated into English by the LLM from
lab 1 (llama-server), because the papers are in English.

Requires running Qdrant and, with --translate, llama-server (see LLAMA_START below).
Usage: python retriever.py "Какой сэмплер использует Direct3D?" [--mode hybrid] [--top-k 5] [--translate]
"""
import logging
import re
from typing import Literal

import openai
import typer
from FlagEmbedding import BGEM3FlagModel
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient, models

from load_to_vector_store import PREFIX, QDRANT_START

log = logging.getLogger(__name__)

LLAMA_START = ("~/llms/llama.cpp/llama-b11524/llama-server "
               "-m ~/llms/models/ministral3-8b/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf "
               "-c 8192 --parallel 1 -ngl 99 -fa on --host 127.0.0.1 --port 8080")
TRANSLATE_PROMPT = ("Translate this question about scientific papers into English. Keep names, "
                    "abbreviations and numbers unchanged. Reply with the translation only, as plain text.\n\n{question}")
CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
# translations made in this process; temperature 0 makes them repeatable, so asking again is wasted time
TRANSLATIONS: dict[str, str] = {}


# --- Settings and result -----------------------------------------------------------------

class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_RETRIEVE_")

    index: str = "markdown-800-100-h__hybrid-bge-m3"  # the loaded collection is "lab2-<index>"
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    top_k: int = 5
    sparse_weight: float = 0.3  # hybrid score = dense + sparse_weight · sparse (bge-m3's recommended mix)
    candidates: int = 100  # chunks taken from each vector type before mixing; 100 covers our whole collection
    hnsw_ef: int = 64  # ef_search: how many graph nodes HNSW visits per query
    exact: bool = False  # search all dense vectors without HNSW, for comparison
    query_max_length: int = 512  # longer questions are rejected instead of being cut silently
    translate: bool = False  # translate Russian questions into English before searching
    model: str = "BAAI/bge-m3"
    qdrant_url: str = "http://localhost:6333"
    llm_url: str = "http://127.0.0.1:8080/v1"

    @property
    def collection(self) -> str:
        return PREFIX + self.index


class Hit(BaseModel):
    """One found chunk: its payload from Qdrant plus its place in the ranking."""

    rank: int
    score: float
    chunk_id: str
    arxiv_id: str
    title: str
    section: str
    pages: list[int]
    text: str


class QueryTooLongError(ValueError):
    """The question is longer than the model reads; meant to be shown to the user."""


# --- Retriever ---------------------------------------------------------------------------

class Retriever:
    """Loads bge-m3 and connects to Qdrant (and the LLM) once, then answers many questions."""

    def __init__(self, settings: RetrievalSettings | None = None, model: BGEM3FlagModel | None = None):
        """model: an already loaded bge-m3, so several retrievers (e.g. in evaluation) can share it."""
        self.settings = settings or RetrievalSettings()
        self.qdrant = QdrantClient(self.settings.qdrant_url)
        try:
            exists = self.qdrant.collection_exists(self.settings.collection)
        except Exception as error:
            raise SystemExit(f"Qdrant is not reachable ({error}).\nStart it with:\n  {QDRANT_START}")
        if not exists:
            raise SystemExit(f"No collection {self.settings.collection}: run load_to_vector_store.py --index {self.settings.index}")

        self.model = model or BGEM3FlagModel(self.settings.model, use_fp16=True)
        self.llm = openai.OpenAI(base_url=self.settings.llm_url, api_key="unused") if self.settings.translate else None

    def search(self, question: str) -> list[Hit]:
        """The top_k chunks for a question, best first."""
        self.check_query(question)
        query = question
        if self.settings.translate:
            query = self.translate(question)
            self.check_query(query)  # the translation is what gets encoded, so it must fit too
        dense, sparse = self.encode(query)
        points = self.query_qdrant(dense, sparse)
        return [Hit(rank=i + 1, score=p.score, **p.payload) for i, p in enumerate(points)]

    def check_query(self, question: str) -> None:
        # same count as for chunks: the model's tokenizer plus its 2 special tokens
        length = len(self.model.tokenizer.tokenize(question)) + 2
        if length > self.settings.query_max_length:
            raise QueryTooLongError(f"The question is too long: {length} tokens, "
                                    f"the maximum is {self.settings.query_max_length}")

    def translate(self, question: str) -> str:
        """Russian → English with the LLM; questions without Cyrillic letters are returned as they are."""
        if not CYRILLIC.search(question):
            return question
        if question in TRANSLATIONS:
            return TRANSLATIONS[question]
        try:
            reply = self.llm.chat.completions.create(
                model="local",  # llama-server serves one model and ignores the name
                messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(question=question)}],
                temperature=0,  # the same question always gets the same translation
                # a translation has about as many tokens as the question; twice the question
                # limit leaves room, and a longer reply is caught below instead of being used cut
                max_tokens=2 * self.settings.query_max_length,
            )
        except openai.APIConnectionError:
            raise SystemExit(f"The LLM is not reachable at {self.settings.llm_url}.\nStart it with:\n  {LLAMA_START}")
        if reply.choices[0].finish_reason == "length":
            raise QueryTooLongError("The translation of the question was cut off: the question is too long")
        # the model sometimes adds Markdown bold (**Direct3D**); asterisks are not search terms
        translation = reply.choices[0].message.content.replace("*", "").strip()
        log.info("Translated: %s", translation)
        TRANSLATIONS[question] = translation
        return translation

    def encode(self, query: str) -> tuple[list[float] | None, models.SparseVector | None]:
        """One bge-m3 pass; only the vectors the mode needs."""
        mode = self.settings.mode
        output = self.model.encode([query], max_length=self.settings.query_max_length,
                                   return_dense=mode in ("dense", "hybrid"),
                                   return_sparse=mode in ("sparse", "hybrid"))
        dense = output["dense_vecs"][0].tolist() if output["dense_vecs"] is not None else None
        sparse = None
        if output["lexical_weights"] is not None:
            weights = output["lexical_weights"][0]
            sparse = models.SparseVector(indices=[int(t) for t in weights], values=[float(w) for w in weights.values()])
        return dense, sparse

    def query_qdrant(self, dense: list[float] | None, sparse: models.SparseVector | None) -> list[models.ScoredPoint]:
        s = self.settings
        # HNSW settings for the dense part; rescore re-ranks int8-quantized results with the original vectors
        dense_params = models.SearchParams(hnsw_ef=s.hnsw_ef, exact=s.exact,
                                           quantization=models.QuantizationSearchParams(rescore=True))

        if s.mode == "dense":
            result = self.qdrant.query_points(s.collection, query=dense, using="dense", limit=s.top_k,
                                              search_params=dense_params)
        elif s.mode == "sparse":
            result = self.qdrant.query_points(s.collection, query=sparse, using="sparse", limit=s.top_k)
        else:
            # two candidate lists, then Qdrant scores their union with: dense + w · sparse.
            # A chunk missing from one list counts 0 for that part.
            prefetch = [
                models.Prefetch(query=dense, using="dense", limit=s.candidates, params=dense_params),
                models.Prefetch(query=sparse, using="sparse", limit=s.candidates),
            ]
            formula = models.FormulaQuery(
                formula=models.SumExpression(sum=["$score[0]", models.MultExpression(mult=[s.sparse_weight, "$score[1]"])]),
                defaults={"$score[0]": 0.0, "$score[1]": 0.0},
            )
            result = self.qdrant.query_points(s.collection, prefetch=prefetch, query=formula, limit=s.top_k)
        return result.points


# --- CLI ---------------------------------------------------------------------------------

def main(
    question: str = typer.Argument(..., help="The question to search for."),
    mode: str | None = typer.Option(None, help="dense, sparse or hybrid."),
    top_k: int | None = typer.Option(None, help="Number of chunks to return."),
    translate: bool | None = typer.Option(None, "--translate/--no-translate", help="Translate Russian questions first."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # hide every HTTP request to Qdrant (httpx) and the LLM (httpx2, used by openai), and model-loading notes
    for name in ("httpx", "httpx2", "FlagEmbedding"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # flags that were given override the settings (defaults or RAG_RETRIEVE_* variables)
    flags = {"mode": mode, "top_k": top_k, "translate": translate}
    settings = RetrievalSettings(**{k: v for k, v in flags.items() if v is not None})
    try:
        hits = Retriever(settings).search(question)
    except QueryTooLongError as error:
        raise SystemExit(str(error))

    for hit in hits:
        pages = ", ".join(map(str, hit.pages))
        typer.echo(f"{hit.rank}. [{hit.score:.3f}] {hit.chunk_id}  {hit.section or '-'}  (p. {pages})")
        typer.echo(f"   {' '.join(hit.text.split())[:160]}…")


if __name__ == "__main__":
    typer.run(main)
