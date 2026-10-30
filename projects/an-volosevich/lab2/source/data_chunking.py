"""Step 3a: split the parsed papers into chunks for retrieval.

Three splitters, chosen with --splitter:
    markdown   LangChain: split by h1–h3 sections, then a token window inside each section
    recursive  LangChain: one token window over the whole paper, headings stay in the text
    docling    docling HybridChunker on the saved docling JSON

Output: data/chunks/<splitter>-<size>-<overlap>-<h|noh>/chunks.jsonl and config.json

Usage: python data_chunking.py [--splitter docling] [--chunk-size 400] [--overlap 0] [--no-headings] [--force]
"""
import json
import logging
import re
from pathlib import Path
from statistics import mean
from typing import Literal, NamedTuple

import typer
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from transformers import AutoTokenizer

log = logging.getLogger(__name__)

PAGE_MARKER = re.compile(r"<!-- page: (\d+) -->")  # written by data_parsing.py before every page
HEADING = re.compile(r"^(#{2,3}) (.+)$")  # "## 3 Methods", "### 3.1 ..."

# Paragraph breaks first, then lines, sentences, words. MarkdownHeaderTextSplitter joins
# lines with "  \n" and drops blank lines, so "  \n" also marks a paragraph.
SEPARATORS = ["\n\n", "  \n", "\n", ". ", " ", ""]


# --- Settings and data model -------------------------------------------------------------

class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_CHUNK_")

    splitter: Literal["markdown", "recursive", "docling"] = "markdown"
    chunk_size: int = 800  # tokens
    overlap: int = 100  # tokens; HybridChunker has no overlap, so docling ignores it
    headings: bool = True  # prepend "title | section" to the embedded text
    heading_max_tokens: int = 64  # the heading line is cut to this length and reserved in chunk_size
    tokenizer: str = "BAAI/bge-m3"  # one tokenizer for all configs keeps sizes comparable
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"

    @property
    def text_budget(self) -> int:
        """Tokens left for the chunk text, so that heading line + text never exceed chunk_size."""
        return self.chunk_size - self.heading_max_tokens if self.headings else self.chunk_size

    @property
    def name(self) -> str:
        """Readable build name, e.g. "markdown-800-100-h"; it is also the output folder."""
        overlap = 0 if self.splitter == "docling" else self.overlap
        return f"{self.splitter}-{self.chunk_size}-{overlap}-{'h' if self.headings else 'noh'}"


class Piece(NamedTuple):
    """What a splitter returns: one piece of a paper, before it becomes a Chunk."""

    section: str
    pages: list[int]
    text: str


class Chunk(BaseModel):
    """One line of chunks.jsonl, read by indexing, the vector store, evaluation and RAG."""

    chunk_id: str  # "<arxiv_id>:<index>"
    arxiv_id: str
    title: str
    section: str  # "4 Experiments > 4.1 Implementation Details"
    pages: list[int]
    text: str  # clean text, for the prompt and citation snippets
    embed_text: str  # what gets embedded


# --- Splitters ---------------------------------------------------------------------------

def split_markdown(paper: dict, settings: ChunkingSettings, tokenizer) -> list[Piece]:
    """LangChain splitters on the paper's Markdown (the "markdown" and "recursive" splitters)."""
    text = (settings.data_dir / "markdown" / f"{paper['arxiv_id']}.md").read_text(encoding="utf-8")
    body = text.split("---\n", 2)[2]  # drop the YAML front matter

    # token window, measured with the shared tokenizer
    window = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=settings.text_budget,
        chunk_overlap=settings.overlap,
        separators=SEPARATORS,
        keep_separator="end",  # a sentence keeps its period instead of passing it to the next chunk
    )

    raw_pieces = []  # (section or None, text)
    if settings.splitter == "recursive":
        # one window over the whole paper; sections are recovered from the headings later
        for piece in window.split_text(body):
            raw_pieces.append((None, piece))
    else:
        # sections first, so a chunk never crosses a section boundary
        sections = MarkdownHeaderTextSplitter([("#", "h1"), ("##", "h2"), ("###", "h3")]).split_text(body)
        for section in sections:
            path = " > ".join(filter(None, [section.metadata.get("h2"), section.metadata.get("h3")]))
            for piece in window.split_text(section.page_content):
                raw_pieces.append((path, piece))

    markers = [int(n) for n in PAGE_MARKER.findall(body)]
    return assign_pages(raw_pieces, markers)


def split_docling(paper: dict, settings: ChunkingSettings, tokenizer) -> list[Piece]:
    """docling HybridChunker: document elements merged under their headings, up to the text budget."""
    doc = DoclingDocument.load_from_json(settings.data_dir / "docling" / f"{paper['arxiv_id']}.json")
    chunker = HybridChunker(
        tokenizer=HuggingFaceTokenizer(tokenizer=tokenizer, max_tokens=settings.text_budget),
        repeat_table_header=True,  # a split table keeps its header row in every part
    )

    pieces = []
    for c in chunker.chunk(doc):
        # every docling element knows its page, so no page markers are needed here
        pages = sorted({prov.page_no for item in c.meta.doc_items for prov in item.prov})
        section = " > ".join(c.meta.headings or [])
        pieces.append(Piece(section, pages, c.text))
    return pieces


# --- Pages and sections for the LangChain splitters --------------------------------------

def assign_pages(raw_pieces: list[tuple[str | None, str]], markers: list[int]) -> list[Piece]:
    """Give each piece its pages (and, for "recursive", its section); remove the page markers.

    The pieces come in document order, so we walk through them and keep track of the current
    page and the current h2/h3 heading.
    """
    pieces = []
    page = markers[0]
    h2 = h3 = ""

    for given_section, text in raw_pieces:
        # Text before a piece's first marker sits on the page of the marker before it.
        # This also puts overlapped text (repeated from the previous chunk) on the right page.
        first_marker = PAGE_MARKER.search(text)
        if first_marker:
            earlier = [m for m in markers if m < int(first_marker[1])]
            page = earlier[-1] if earlier else int(first_marker[1])

        section = given_section
        pages = []
        lines = []
        for line in text.splitlines():
            marker = PAGE_MARKER.fullmatch(line.strip())
            heading = HEADING.match(line)

            if marker:
                # a new page starts here; the marker itself is not kept in the text
                page = int(marker[1])
                continue

            if heading:
                # only the "recursive" splitter keeps headings in its text
                level, name = heading.groups()
                if level == "##":
                    h2, h3 = name, ""
                else:
                    h3 = name
            elif line.strip():
                # a piece belongs to the section of its first real line
                if section is None:
                    section = " > ".join(filter(None, [h2, h3]))
                if page not in pages:
                    pages.append(page)

            lines.append(line.rstrip())

        # a piece with only markers or headings has no content
        if pages:
            pieces.append(Piece(section, pages, "\n".join(lines).strip()))

    return pieces


# --- Build -------------------------------------------------------------------------------

def is_up_to_date(settings: ChunkingSettings, out_dir: Path) -> bool:
    """A build can be reused if it was made with the same settings and after every input file."""
    chunks_file = out_dir / "chunks.jsonl"
    if not chunks_file.exists():
        return False

    # settings missing from the folder name (e.g. heading_max_tokens) must match too
    saved = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))["settings"]
    if saved != settings.model_dump(mode="json", exclude={"data_dir"}):
        return False

    input_dir = settings.data_dir / ("docling" if settings.splitter == "docling" else "markdown")
    newest_input = max(f.stat().st_mtime for f in input_dir.iterdir())
    return chunks_file.stat().st_mtime > newest_input


def paper_card(paper: dict) -> Piece:
    """One extra chunk per paper, built from metadata.json, so questions about authors or dates
    have a clean match. Other metadata (url, dates) is joined by arxiv_id when answering."""
    lines = [
        paper["title"],
        f"Authors: {', '.join(paper['authors'])}",
        f"Published on arXiv: {paper['published']}; this version ({paper['arxiv_id']}): {paper['updated']}",
        f"Link: {paper['url']}",
    ]
    # page 1 is where title and authors are printed, so citations of the card point there
    return Piece(section="Paper information", pages=[1], text="\n".join(lines))


def heading_line(paper: dict, piece: Piece, max_tokens: int, tokenizer) -> str:
    """ "Title | Section > Subsection", cut to max_tokens so a broken heading cannot crowd out the text."""
    line = " | ".join(filter(None, [paper["title"], piece.section]))
    tokens = tokenizer.tokenize(line)
    if len(tokens) > max_tokens:
        # a real heading line is 16–58 tokens; a longer one usually means a parsing error
        log.warning("Heading line of %s cut from %d to %d tokens", paper["arxiv_id"], len(tokens), max_tokens)
        line = tokenizer.convert_tokens_to_string(tokens[:max_tokens])
    return line


def to_chunk(paper: dict, index: int, piece: Piece, settings: ChunkingSettings, tokenizer) -> Chunk:
    embed_text = piece.text
    if settings.headings:
        # e.g. "Direct3D: ... | 4 Experiments > 4.1 Implementation Details", so the chunk
        # knows which paper and section it is from. Its tokens are reserved in text_budget.
        embed_text = f"{heading_line(paper, piece, settings.heading_max_tokens, tokenizer)}\n{piece.text}"

    return Chunk(
        chunk_id=f"{paper['arxiv_id']}:{index:03d}",
        arxiv_id=paper["arxiv_id"],
        title=paper["title"],
        section=piece.section,
        pages=piece.pages,
        text=piece.text,
        embed_text=embed_text,
    )


def validate(chunks: list[Chunk], sizes: list[int], max_tokens: int) -> None:
    """Stop the build instead of writing broken chunks."""
    for c, size in zip(chunks, sizes):
        if size > max_tokens:
            raise ValueError(f"{c.chunk_id}: embed_text has {size} tokens, the limit is {max_tokens}")
        if not c.pages:
            raise ValueError(f"{c.chunk_id} has no page")
        if PAGE_MARKER.search(c.text):
            raise ValueError(f"{c.chunk_id} still contains a page marker")


def save(chunks: list[Chunk], sizes: list[int], settings: ChunkingSettings, out_dir: Path) -> dict:
    """Write config.json (parameters + statistics) and chunks.jsonl; return the statistics."""
    stats = {  # token counts of embed_text
        "chunks": len(chunks),
        "tokens_min": min(sizes),
        "tokens_mean": round(mean(sizes)),
        "tokens_max": max(sizes),
        "multi_page": sum(len(c.pages) > 1 for c in chunks),
    }
    config = {"settings": settings.model_dump(mode="json", exclude={"data_dir"}), "stats": stats}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    # chunks.jsonl is written last: its timestamp marks the build as complete
    lines = [c.model_dump_json() + "\n" for c in chunks]
    (out_dir / "chunks.jsonl").write_text("".join(lines), encoding="utf-8")
    return stats


def chunk(settings: ChunkingSettings | None = None, force: bool = False) -> Path:
    """Build the chunks of one configuration (or reuse an up-to-date build); return its folder."""
    settings = settings or ChunkingSettings()
    out_dir = settings.data_dir / "chunks" / settings.name
    if not force and is_up_to_date(settings, out_dir):
        log.info("Chunks %s are up to date", settings.name)
        return out_dir

    tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer)
    split = split_docling if settings.splitter == "docling" else split_markdown
    papers = json.loads((settings.data_dir / "metadata.json").read_text(encoding="utf-8"))

    chunks = []
    for paper in papers:
        pieces = [paper_card(paper)] + split(paper, settings, tokenizer)
        for index, piece in enumerate(pieces):
            chunks.append(to_chunk(paper, index, piece, settings, tokenizer))

    # sizes of what the embedding model will receive: heading line + text
    sizes = [len(tokenizer.tokenize(c.embed_text)) for c in chunks]
    validate(chunks, sizes, settings.chunk_size)
    stats = save(chunks, sizes, settings, out_dir)
    log.info("%s: %s", settings.name, stats)
    return out_dir


# --- CLI ---------------------------------------------------------------------------------

def main(
    splitter: str | None = typer.Option(None, help="markdown, recursive or docling."),
    chunk_size: int | None = typer.Option(None, help="Chunk size in tokens."),
    overlap: int | None = typer.Option(None, help="Overlap in tokens (not used by docling)."),
    headings: bool | None = typer.Option(None, "--headings/--no-headings", help="Prepend title and section."),
    force: bool = typer.Option(False, "--force", help="Rebuild even if up to date."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide Hugging Face request logs

    # flags that were given override the settings (defaults or RAG_CHUNK_* variables)
    flags = {"splitter": splitter, "chunk_size": chunk_size, "overlap": overlap, "headings": headings}
    settings = ChunkingSettings(**{k: v for k, v in flags.items() if v is not None})
    log.info("Chunks are in %s", chunk(settings, force=force))


if __name__ == "__main__":
    typer.run(main)
