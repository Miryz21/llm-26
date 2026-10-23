"""PDF → нормализованный Markdown с метаданными.

Скачивает недостающие PDF с arXiv, конвертирует pymupdf4llm (двухколоночная вёрстка, таблицы)
и нормализует результат:
- YAML-шапка: название, авторы, arXiv id, URL, путь к PDF, дата публикации и разбора, число страниц;
- маркеры страниц <!-- page: N --> — по ним чанки получают номера страниц;
- уровни заголовков восстанавливаются по нумерации разделов (1 → h2, 1.1 → h3, 1.1.1 → h4,
  приложения A / A.1 — так же): pymupdf4llm назначает их по размеру шрифта, и уровни скачут;
- удаляются колонтитулы с номером страницы, боковая метка arXiv, список литературы и «текст картинок»
  (подписи узлов диаграмм, извлечённые из рисунков: для поиска это шум, подпись Figure N остаётся).
"""
import argparse
import datetime as dt
import os
import re
import urllib.request

import pymupdf4llm

from config import MARKDOWN_DIR, PAPERS, PDF_DIR, SOURCE_DIR

NUMBERED = re.compile(r"^(?:\d+|[A-Z])(?:\.\d+)*\.?\s+\S")
PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.DOTALL)
ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}v\d+\s+\[[\w.-]+\]\s+\d{1,2} \w{3} \d{4}\s*$")


def download(arxiv_id: str) -> str:
    path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
    if not os.path.exists(path):
        os.makedirs(PDF_DIR, exist_ok=True)
        urllib.request.urlretrieve(f"https://arxiv.org/pdf/{arxiv_id}", path)
    return path


def heading_level(text: str) -> int:
    """Уровень заголовка по номеру раздела: «3» → 2, «3.1» → 3, «A.2.1» → 4; без номера → 2."""
    number = text.split()[0].rstrip(".")
    if NUMBERED.match(text):
        return 2 + number.count(".")
    return 2


def clean_inline(text: str) -> str:
    text = re.sub(r"</?sup>", "", text)
    text = re.sub(r"_([^\w\s_]{1,2})_", r"\1", text)          # «0 _._ 6» → «0 . 6», «_×_» → «×»
    text = re.sub(r"(\d) \. (\d)", r"\1.\2", text)
    return text


def normalize_page(text: str, title: str) -> list[str]:
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"\d{1,2}", stripped) or ARXIV_STAMP.match(stripped):
            continue
        plain = re.sub(r"[*_#]", "", stripped).strip()
        if plain.lower() == title.lower():
            continue
        heading = re.match(r"^#{1,6}\s+(.*)$", stripped)
        # Заголовки приложений в некоторых PDF распознаются как пункты списка: «- A.1 SYNTHETIC ...».
        list_heading = re.match(r"^-\s+((?:[A-Z]|\d+)(?:\.\d+)*\s+[A-Z][A-Z0-9 /:&-]+)$", stripped)
        if heading or list_heading:
            name = re.sub(r"\*\*", "", (heading or list_heading).group(1)).strip()
            out += ["", "#" * heading_level(name) + " " + name, ""]
            continue
        out.append(clean_inline(line.rstrip()))
    return out


def drop_references(lines: list[str]) -> list[str]:
    """Вырезает раздел References до следующего заголовка (обычно — приложения) или конца."""
    out, skipping = [], False
    for line in lines:
        if line.startswith("#"):
            skipping = re.sub(r"^#+\s+", "", line).strip().lower() == "references"
            if skipping:
                continue
        if not skipping:
            out.append(line)
    return out


def convert(paper: dict, force: bool = False) -> str:
    out_path = os.path.join(MARKDOWN_DIR, f"{paper['arxiv_id']}.md")
    if os.path.exists(out_path) and not force:
        return out_path
    pdf = download(paper["arxiv_id"])
    pages = pymupdf4llm.to_markdown(pdf, page_chunks=True, show_progress=False)

    body = []
    for page in pages:
        body += ["", f"<!-- page: {page['metadata']['page_number']} -->", ""]
        body += normalize_page(PICTURE_TEXT.sub("", page["text"]), paper["title"])
    body = drop_references(body)

    front = [
        "---",
        f"title: \"{paper['title']}\"",
        f"short: \"{paper['short']}\"",
        f"authors: \"{paper['authors']}\"",
        f"arxiv_id: \"{paper['arxiv_id']}\"",
        f"url: https://arxiv.org/abs/{paper['arxiv_id']}",
        f"source: {os.path.relpath(pdf, SOURCE_DIR)}",
        f"published: {paper['date']}",
        f"parsed: {dt.date.today().isoformat()}",
        f"pages: {len(pages)}",
        f"parser: pymupdf4llm {pymupdf4llm.__version__}",
        "---",
        "",
        f"# {paper['title']}",
    ]
    text = "\n".join(front + body)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="пересобрать уже существующий Markdown")
    args = parser.parse_args()
    for paper in PAPERS:
        path = convert(paper, args.force)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        print(f"{paper['short']:<28} {len(text.split()):>6} слов  → {os.path.relpath(path, SOURCE_DIR)}")


if __name__ == "__main__":
    main()
