"""Показывает, как чанкер режет текст: границы чанков, страницы, длину и перекрытие.

    python show_chunks.py --paper Direct3D --section "4.1"            # раздел статьи из корпуса
    python show_chunks.py --paper ViewCrafter --splitter recursive --chunk-size 200 --overlap 40
    python show_chunks.py --file notes.md                             # свой Markdown-файл
    echo "# Заголовок ..." | python show_chunks.py --file -           # текст из stdin
    python show_chunks.py --section "4.1" --format markdown           # таблица для README

В терминале чанки чередуют цвета, а перекрытие с предыдущим чанком подсвечено жёлтым фоном.
"""
import argparse
import os
import sys

from chunking import chunk_markdown, count_tokens, parse_markdown, read_markdown
from config import MARKDOWN_DIR, PAPERS, add_index_args, index_config_from_args

COLORS = ("\033[36m", "\033[35m")      # голубой / пурпурный — чётные и нечётные чанки
OVERLAP = "\033[30;43m"                # чёрный текст на жёлтом фоне
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
MIN_OVERLAP = 20                        # совпадение короче — случайное, не перекрытие


def overlap_with(prev: str, cur: str) -> str:
    """Самый длинный суффикс предыдущего чанка, с которого начинается текущий."""
    for k in range(min(len(prev), len(cur)), MIN_OVERLAP - 1, -1):
        if prev.endswith(cur[:k]):
            return cur[:k]
    return ""


def load(args) -> tuple[dict, str]:
    if args.file:
        text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
        if text.startswith("---\n"):
            meta, body = parse_markdown(text)
            meta.setdefault("arxiv_id", meta.get("short", "doc"))
            return {"short": meta["arxiv_id"], "title": "", "url": "", **meta}, body
        name = os.path.basename(args.file) if args.file != "-" else "stdin"
        return {"arxiv_id": name, "short": name, "title": name, "url": ""}, text
    wanted = args.paper.lower()
    paper = next((p for p in PAPERS if wanted in (p["arxiv_id"], p["short"].lower())
                  or p["short"].lower().startswith(wanted)), None)
    if paper is None:
        raise SystemExit(f"статья «{args.paper}» не найдена; есть: " + ", ".join(p["short"] for p in PAPERS))
    return read_markdown(os.path.join(MARKDOWN_DIR, f"{paper['arxiv_id']}.md"))


def annotate(chunks: list[dict]) -> list[dict]:
    out, prev = [], ""
    for c in chunks:
        ov = overlap_with(prev, c["text"])
        out.append({**c, "overlap": ov, "overlap_tokens": count_tokens(ov) if ov else 0})
        prev = c["text"]
    return out


def pages(c: dict) -> str:
    p = c["pages"]
    return f"{p[0]}–{p[-1]}" if len(p) > 1 else str(p[0])


def render_terminal(chunks: list[dict], cfg) -> str:
    lines = [f"{BOLD}{cfg.chunk_id} — чанков: {len(chunks)}{RESET}", ""]
    for i, c in enumerate(chunks):
        color = COLORS[i % 2]
        ov = f", перекрытие {c['overlap_tokens']} ток." if c["overlap"] else ""
        lines.append(f"{BOLD}{color}━━ #{i + 1}  {c['id']}  стр. {pages(c)}  {c['tokens']} ток.{ov}{RESET}")
        lines.append(f"{DIM}   раздел: {c['section'] or '—'}{RESET}")
        rest = c["text"][len(c["overlap"]):]
        lines.append(f"{OVERLAP}{c['overlap']}{RESET}{color}{rest}{RESET}")
        lines.append("")
    return "\n".join(lines)


def cell(text: str, limit: int, tail: bool = False) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    if len(text) <= limit:
        return text
    return "…" + text[-limit:].split(" ", 1)[-1] if tail else text[:limit].rsplit(" ", 1)[0] + "…"


def render_markdown(chunks: list[dict], cfg) -> str:
    lines = ["| # | Стр. | Токенов | Перекрытие | Начало чанка | Конец чанка |", "|---|---|---|---|---|---|"]
    for i, c in enumerate(chunks):
        ov = f"{c['overlap_tokens']} ток.: «{cell(c['overlap'], 40)}»" if c["overlap"] else "—"
        lines.append(f"| {i + 1} | {pages(c)} | {c['tokens']} | {ov} | {cell(c['text'], 70)} "
                     f"| {cell(c['text'], 60, tail=True)} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--paper", default="Direct3D", help="arXiv id или короткое имя статьи корпуса")
    parser.add_argument("--section", help="показать только чанки, чей раздел содержит эту строку")
    parser.add_argument("--file", help="свой Markdown-файл вместо статьи корпуса; «-» — stdin")
    parser.add_argument("--format", choices=("terminal", "markdown"), default="terminal")
    add_index_args(parser)
    args = parser.parse_args()
    cfg = index_config_from_args(args)

    meta, body = load(args)
    chunks = annotate(chunk_markdown(meta, body, cfg))
    if args.section:
        chunks = [c for c in chunks if args.section.lower() in c["section"].lower()]
        if not chunks:
            raise SystemExit(f"нет чанков в разделах, содержащих «{args.section}»")
    print(render_terminal(chunks, cfg) if args.format == "terminal" else render_markdown(chunks, cfg))


if __name__ == "__main__":
    main()
