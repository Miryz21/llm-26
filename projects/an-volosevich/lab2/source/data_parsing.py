"""Step 2: convert the downloaded PDFs into Markdown with metadata (+ docling JSON for chunking).

Reads data/pdf/ and data/metadata.json, writes data/markdown/<id>.md and data/docling/<id>.json.
Usage: python data_parsing.py [--force]
"""
import json
import logging
import re
import unicodedata
from datetime import date
from importlib.metadata import version
from pathlib import Path

import typer
import yaml
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TitleItem
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

SPLIT_ACCENT = re.compile(r"\s?([¨´ˆ˜])([A-Za-z])")  # PDFs store "Plücker" as "Pl ¨ucker"
COMBINING = {"¨": "̈", "´": "́", "ˆ": "̂", "˜": "̃"}


class ParsingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_PARSE_")

    data_dir: Path = Path(__file__).resolve().parents[1] / "data"
    formula_recognition: bool = True  # formulas → LaTeX; slower


def make_converter(settings: ParsingSettings) -> DocumentConverter:
    options = PdfPipelineOptions(
        do_ocr=False,  # arXiv PDFs have a text layer
        do_formula_enrichment=settings.formula_recognition,
        # heading levels from PDF bookmarks, then numbering, then font style
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
    )
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})


def fix_accents(text: str) -> str:
    return SPLIT_ACCENT.sub(lambda m: unicodedata.normalize("NFC", m[2] + COMBINING[m[1]]), text)


def normalize(doc: DoclingDocument, title: str) -> None:
    """Drop the title and References, fix accents."""
    to_delete, in_references = [], False
    for item, _ in doc.iterate_items():
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            in_references = item.text.strip().lower() == "references"
            if " ".join(item.text.split()).lower() == title.lower():
                to_delete.append(item)  # the title is written from metadata instead
        if in_references:
            to_delete.append(item)
    doc.delete_items(node_items=to_delete)

    for item in doc.texts:
        item.text = fix_accents(item.text)
    for table in doc.tables:
        for cell in table.data.table_cells:
            cell.text = fix_accents(cell.text)


def to_markdown(doc: DoclingDocument, paper: dict) -> str:
    """YAML front matter, the title, then each page behind a <!-- page: N --> marker."""
    front = {**paper, "pages": len(doc.pages), "parser": f"docling {version('docling')}",
             "parsed": date.today().isoformat()}
    pages = [(n, doc.export_to_markdown(page_no=n)) for n in sorted(doc.pages)]
    body = "\n\n".join(f"<!-- page: {n} -->\n\n{text}" for n, text in pages if text.strip())
    return f"---\n{yaml.safe_dump(front, allow_unicode=True, sort_keys=False)}---\n\n# {paper['title']}\n\n{body}\n"


def parse(settings: ParsingSettings | None = None, force: bool = False) -> Path:
    """Convert every paper that has no Markdown yet and return the Markdown folder."""
    settings = settings or ParsingSettings()
    md_dir, json_dir = settings.data_dir / "markdown", settings.data_dir / "docling"
    md_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    papers = json.loads((settings.data_dir / "metadata.json").read_text(encoding="utf-8"))
    todo = [p for p in papers if force or not (md_dir / f"{p['arxiv_id']}.md").exists()]
    if not todo:
        log.info("All papers already parsed, nothing to do")
        return md_dir

    converter = make_converter(settings)  # loads the models, so only when there is work
    for paper in todo:
        log.info("Parsing %s", paper["arxiv_id"])
        doc = converter.convert(settings.data_dir / paper["pdf_file"]).document
        normalize(doc, paper["title"])
        doc.save_as_json(json_dir / f"{paper['arxiv_id']}.json")
        # Markdown is written last: its presence means the paper is fully parsed
        (md_dir / f"{paper['arxiv_id']}.md").write_text(to_markdown(doc, paper), encoding="utf-8")
    return md_dir


def main(force: bool = typer.Option(False, "--force", help="Parse all papers again.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("docling").setLevel(logging.WARNING)
    log.info("Markdown is in %s", parse(force=force))


if __name__ == "__main__":
    typer.run(main)
