"""Разбиение нормализованного Markdown на чанки с метаданными (статья, раздел, страницы).

Используются готовые сплиттеры langchain-text-splitters:
- markdown — MarkdownHeaderTextSplitter режет статью по заголовкам h1–h3 (h4 остаются внутри
  родительского h3), затем RecursiveCharacterTextSplitter режет длинные разделы на окна;
- recursive — RecursiveCharacterTextSplitter по всей статье, без учёта заголовков.

Размер окна и overlap считаются в токенах токенизатора bge-m3. Номера страниц восстанавливаются
по положению чанка в исходном тексте относительно маркеров <!-- page: N -->, после чего маркеры
из текста удаляются. Положение ищется по первой и последней строке чанка: `add_start_index`
LangChain ненадёжен — после склейки кусков через разделитель текст чанка не всегда является точной
подстрокой исходного, и смещение приходит равным −1.
"""
import bisect
import functools
import os
import re

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import MARKDOWN_DIR, PAPERS, TOKENIZER, IndexConfig

PAGE_MARK = re.compile(r"<!-- page: (\d+) -->")
HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
# MarkdownHeaderTextSplitter склеивает абзацы через «  \n», поэтому этот разделитель идёт первым.
SEPARATORS = ["\n\n", "  \n", "\n", ". ", " ", ""]


@functools.cache
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def count_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def read_markdown(path: str) -> tuple[dict, str]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    _, front, body = text.split("---\n", 2)
    meta = {}
    for line in front.strip().splitlines():
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return meta, body


def window_splitter(cfg: IndexConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer(), chunk_size=cfg.chunk_size, chunk_overlap=cfg.overlap,
        separators=SEPARATORS, strip_whitespace=True)


def pages_of(text: str, start: int, end: int, first_page: int) -> list[int]:
    """Страница начала фрагмента text[start:end] и все маркеры страниц внутри него."""
    marks = [(m.start(), int(m.group(1))) for m in PAGE_MARK.finditer(text)]
    before = [p for pos, p in marks if pos <= start]
    pages = {before[-1] if before else first_page}
    pages |= {p for pos, p in marks if start < pos < end}
    return sorted(pages)


def clean(text: str) -> str:
    text = PAGE_MARK.sub("", text).replace("  \n", "\n\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def locate(text: str, chunk: str, after: int) -> tuple[int, int]:
    """Начало и конец чанка в исходном тексте: первая и последняя строки — непрерывные куски источника."""
    lines = [l.strip() for l in chunk.splitlines() if l.strip()]
    head, tail = lines[0][:60], lines[-1][-60:]
    start = text.find(head, after)
    if start < 0:
        start = text.find(head)
    end = text.find(tail, start)
    return start, (end + len(tail) if end >= 0 else start + len(chunk))


def windows(text: str, cfg: IndexConfig, first_page: int) -> list[tuple[str, list[int], int]]:
    """Окна RecursiveCharacterTextSplitter: (текст, страницы, смещение начала)."""
    out, after = [], 0
    for piece in window_splitter(cfg).split_text(text):
        start, end = locate(text, piece, after)
        after = start + 1
        body = clean(piece)
        if body:
            out.append((body, pages_of(text, start, end, first_page), start))
    return out


def sections(body: str) -> list[tuple[str, str, int]]:
    """Разделы h1–h3: (имя раздела, текст, страница начала раздела)."""
    out, page = [], 1
    for doc in MarkdownHeaderTextSplitter(HEADERS, strip_headers=True).split_text(body):
        name = " > ".join(doc.metadata[k] for k in ("h2", "h3") if k in doc.metadata)
        out.append((name, doc.page_content, page))
        marks = PAGE_MARK.findall(doc.page_content)
        if marks:
            page = int(marks[-1])
    return out


def chunk_paper(paper: dict, cfg: IndexConfig) -> list[dict]:
    meta, body = read_markdown(os.path.join(MARKDOWN_DIR, f"{paper['arxiv_id']}.md"))
    pieces = []
    if cfg.splitter == "markdown":
        for name, text, first_page in sections(body):
            pieces += [(name, t, pages) for t, pages, _ in windows(text, cfg, first_page)]
    else:
        # Раздел чанка — последний заголовок h2/h3 перед его началом.
        heads = [(m.start(), len(m.group(1)), m.group(2)) for m in HEADING.finditer(body)]
        starts = [h[0] for h in heads]
        for t, pages, start in windows(body, cfg, 1):
            path = {}
            for _, level, title in heads[:bisect.bisect_right(starts, start)]:
                path = {k: v for k, v in path.items() if k < level} | {level: title}
            pieces.append((" > ".join(path[k] for k in (2, 3) if k in path), t, pages))

    chunks = []
    for section, text, pages in pieces:
        header = f"{meta['short']} | {section}" if section else meta["short"]
        chunks.append({
            "id": f"{paper['arxiv_id']}:{len(chunks):04d}",
            "arxiv_id": paper["arxiv_id"],
            "paper": meta["short"],
            "title": meta["title"],
            "url": meta["url"],
            "section": section,
            "pages": pages,
            "text": text,
            "embed_text": f"{header}\n{text}" if cfg.headers else text,
            "tokens": count_tokens(text),
        })
    return chunks


def build_chunks(cfg: IndexConfig) -> list[dict]:
    return [chunk for paper in PAPERS for chunk in chunk_paper(paper, cfg)]
