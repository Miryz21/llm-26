"""Общая конфигурация: корпус, пути, параметры индекса, Qdrant и LLM.

Любой параметр индекса задаётся тремя способами (по убыванию приоритета):
флаг CLI → переменная окружения RAG_* → значение по умолчанию ниже.
"""
import argparse
import os
from dataclasses import asdict, dataclass

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SOURCE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
MARKDOWN_DIR = os.path.join(DATA_DIR, "markdown")
ARTIFACTS_DIR = os.path.join(SOURCE_DIR, "artifacts")
RESULTS_DIR = os.path.join(SOURCE_DIR, "results")
QUESTIONS_PATH = os.path.join(SOURCE_DIR, "questions.json")

# --- Корпус --------------------------------------------------------------------
PAPERS = [
    {
        "arxiv_id": "2405.14832",
        "short": "Direct3D",
        "title": "Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer",
        "authors": "Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, Yao Yao",
        "date": "2024-05-23",
    },
    {
        "arxiv_id": "2409.02048",
        "short": "ViewCrafter",
        "title": "ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis",
        "authors": "Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, "
                   "Tien-Tsin Wong, Ying Shan, Yonghong Tian",
        "date": "2024-09-03",
    },
    {
        "arxiv_id": "2508.02155",
        "short": "DreamPainter",
        "title": "DreamPainter: Image Background Inpainting for E-commerce Scenarios",
        "authors": "Sijie Zhao, Jing Cheng, Yaoyao Wu, Hao Xu, Shaohui Jiao",
        "date": "2025-08-04",
    },
    {
        "arxiv_id": "2503.08729",
        "short": "Preserving Product Fidelity",
        "title": "Preserving Product Fidelity in Large Scale Image Recontextualization with Diffusion Models",
        "authors": "Ishaan Malhi, Praneet Dutta, Ellie Talius, Sally Ma, Brendan Driscoll, Krista Holden, "
                   "Garima Pruthi, Arunachalam Narayanaswamy",
        "date": "2025-03-11",
    },
]
PAPERS_BY_ID = {p["arxiv_id"]: p for p in PAPERS}

# --- Эмбеддинги ----------------------------------------------------------------
# Префиксы — из карточек моделей: e5 и bge-en обучены с инструкциями для запроса/документа.
DENSE_MODELS = {
    "bge-m3": {"hf": "BAAI/bge-m3", "query": "", "doc": ""},
    "multilingual-e5-large": {"hf": "intfloat/multilingual-e5-large", "query": "query: ", "doc": "passage: "},
    "e5-large-v2": {"hf": "intfloat/e5-large-v2", "query": "query: ", "doc": "passage: "},
    "bge-base-en": {"hf": "BAAI/bge-base-en-v1.5",
                    "query": "Represent this sentence for searching relevant passages: ", "doc": ""},
}
# Длина чанков считается одним токенизатором для всех конфигураций, чтобы chunk_size был сравним.
TOKENIZER = "BAAI/bge-m3"
SPLITTERS = ("markdown", "recursive")
VECTORS = ("dense", "sparse", "hybrid")

# --- Qdrant и LLM --------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
# Перевод запроса на английский перед поиском (лучший результат на шаге 5).
TRANSLATE_QUERY = os.environ.get("RAG_TRANSLATE", "1") == "1"


@dataclass(frozen=True)
class IndexConfig:
    splitter: str = os.environ.get("RAG_SPLITTER", "markdown")
    chunk_size: int = int(os.environ.get("RAG_CHUNK_SIZE", 800))
    overlap: int = int(os.environ.get("RAG_OVERLAP", 100))
    headers: bool = os.environ.get("RAG_HEADERS", "1") == "1"
    vectors: str = os.environ.get("RAG_VECTORS", "hybrid")
    model: str = os.environ.get("RAG_MODEL", "bge-m3")

    def __post_init__(self):
        assert self.splitter in SPLITTERS, self.splitter
        assert self.vectors in VECTORS, self.vectors
        assert self.model in DENSE_MODELS, self.model
        assert 0 <= self.overlap < self.chunk_size

    @property
    def chunk_id(self) -> str:
        """Ключ разбиения: от него зависят только чанки."""
        return f"{self.splitter}-c{self.chunk_size}-o{self.overlap}-{'h' if self.headers else 'nh'}"

    @property
    def index_id(self) -> str:
        """Ключ индекса: имя каталога артефактов и коллекции Qdrant."""
        emb = "bm25" if self.vectors == "sparse" else self.model
        return f"{self.chunk_id}-{self.vectors}-{emb}"

    @property
    def uses_dense(self) -> bool:
        return self.vectors in ("dense", "hybrid")

    @property
    def uses_sparse(self) -> bool:
        return self.vectors in ("sparse", "hybrid")

    def to_dict(self) -> dict:
        return asdict(self)


def add_index_args(parser: argparse.ArgumentParser) -> None:
    d = IndexConfig()
    parser.add_argument("--splitter", choices=SPLITTERS, default=d.splitter,
                        help="markdown — секции h1–h3 + окно; recursive — сплошное окно без учёта секций")
    parser.add_argument("--chunk-size", type=int, default=d.chunk_size, help="токенов в чанке")
    parser.add_argument("--overlap", type=int, default=d.overlap, help="токенов перекрытия")
    parser.add_argument("--headers", action=argparse.BooleanOptionalAction, default=d.headers,
                        help="добавлять путь заголовков в текст чанка")
    parser.add_argument("--vectors", choices=VECTORS, default=d.vectors)
    parser.add_argument("--model", choices=list(DENSE_MODELS), default=d.model, help="dense-модель")


def index_config_from_args(args: argparse.Namespace) -> IndexConfig:
    return IndexConfig(args.splitter, args.chunk_size, args.overlap, args.headers, args.vectors, args.model)
