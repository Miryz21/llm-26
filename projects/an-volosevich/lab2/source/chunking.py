"""Разбиение нормализованного Markdown на чанки с метаданными (статья, раздел, страницы).

Текст режется на предложения (таблица — единый блок), предложения упаковываются в окна
по `chunk_size` токенов, а перекрытие набирается из последних предложений предыдущего окна
(не больше `overlap` токенов), поэтому предложение никогда не обрывается посередине.

- markdown — окна не пересекают границы разделов h1–h3 (подразделы h4 входят в родительский h3);
- recursive — сплошное окно по всей статье без учёта заголовков.
"""
import functools
import os
import re
from dataclasses import dataclass

from transformers import AutoTokenizer

from config import MARKDOWN_DIR, PAPERS, TOKENIZER, IndexConfig

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")
PAGE_MARK = re.compile(r"<!-- page: (\d+) -->")


@dataclass
class Unit:
    text: str
    page: int
    section: tuple[str, ...]
    paragraph: int
    tokens: int


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


def split_long(text: str, limit: int) -> list[str]:
    """Режет слишком длинное предложение/строку таблицы по словам."""
    words, parts, cur = text.split(), [], []
    for word in words:
        cur.append(word)
        if count_tokens(" ".join(cur)) > limit:
            cur.pop()
            parts.append(" ".join(cur))
            cur = [word]
    parts.append(" ".join(cur))
    return [p for p in parts if p]


def units_of(body: str, limit: int) -> list[Unit]:
    units, page, section, paragraph = [], 1, (), 0
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        mark = PAGE_MARK.fullmatch(block)
        if mark:
            page = int(mark.group(1))
            continue
        if not block:
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", block)
        if heading:
            level = len(heading.group(1))
            section = section[:level - 1] + ("",) * max(0, level - 1 - len(section)) + (heading.group(2),)
            continue
        paragraph += 1
        if block.startswith("|"):
            pieces = [block] if count_tokens(block) <= limit else block.splitlines()
        else:
            pieces = SENTENCE_END.split(" ".join(block.split()))
        for piece in pieces:
            for part in ([piece] if count_tokens(piece) <= limit else split_long(piece, limit)):
                units.append(Unit(part, page, section, paragraph, count_tokens(part)))
    return units


def pack(units: list[Unit], size: int, overlap: int) -> list[list[Unit]]:
    windows, cur, cur_tokens = [], [], 0
    for unit in units:
        if cur and cur_tokens + unit.tokens > size:
            windows.append(cur)
            tail, tail_tokens = [], 0
            for prev in reversed(cur):
                if tail_tokens + prev.tokens > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += prev.tokens
            while tail and tail_tokens + unit.tokens > size:
                tail_tokens -= tail.pop(0).tokens
            cur, cur_tokens = tail, tail_tokens
        cur.append(unit)
        cur_tokens += unit.tokens
    if cur:
        windows.append(cur)
    return windows


def join_units(units: list[Unit]) -> str:
    out = units[0].text
    for prev, unit in zip(units, units[1:]):
        sep = " " if prev.paragraph == unit.paragraph and not unit.text.startswith("|") else "\n\n"
        out += sep + unit.text
    return out


def section_name(section: tuple[str, ...]) -> str:
    return " > ".join(s for s in section[1:] if s)


def chunk_paper(paper: dict, cfg: IndexConfig) -> list[dict]:
    meta, body = read_markdown(os.path.join(MARKDOWN_DIR, f"{paper['arxiv_id']}.md"))
    units = units_of(body, cfg.chunk_size)
    if cfg.splitter == "markdown":
        groups, key = [], None
        for unit in units:
            if unit.section[:3] != key:
                groups.append([])
                key = unit.section[:3]
            groups[-1].append(unit)
    else:
        groups = [units]

    chunks = []
    for group in groups:
        for window in pack(group, cfg.chunk_size, cfg.overlap):
            text = join_units(window)
            section = section_name(window[0].section)
            header = f"{meta['short']} | {section}" if section else meta["short"]
            chunks.append({
                "id": f"{paper['arxiv_id']}:{len(chunks):04d}",
                "arxiv_id": paper["arxiv_id"],
                "paper": meta["short"],
                "title": meta["title"],
                "url": meta["url"],
                "section": section,
                "pages": sorted({u.page for u in window}),
                "text": text,
                "embed_text": f"{header}\n{text}" if cfg.headers else text,
                "tokens": sum(u.tokens for u in window),
            })
    return chunks


def build_chunks(cfg: IndexConfig) -> list[dict]:
    return [chunk for paper in PAPERS for chunk in chunk_paper(paper, cfg)]
