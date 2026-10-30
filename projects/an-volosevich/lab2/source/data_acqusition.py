"""Step 1: download the paper corpus (raw PDFs + arXiv metadata) into lab2/data/.

Usage: python data_acqusition.py [--force]
"""
import json
import logging
import shutil
from pathlib import Path

import arxiv
import pooch
import typer
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class AcquisitionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_ACQ_")

    # pinned arXiv version → sha256 of its PDF; pooch verifies every file against it
    papers: dict[str, str] = {
        "2405.14832v2": "338964fff4d4807223af6467ed6d5e993ce7835d2c149ae08ca7dcfb4a516ffc",
        "2409.02048v1": "7cd7050a0b0d12398aacb3b78b0c7549ff079964d985e7a129312fc88a7f8833",
        "2508.02155v1": "01d7d0c8ed2294335341228eaf5226cc0b86f686824b97ff3099d7c444cc3781",
        "2503.08729v1": "a09cc30d31a8b01891da340892f28e854615b3bc6ea2f3cb1f82a9a507b31a98",
    }
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"


def download_pdfs(settings: AcquisitionSettings) -> None:
    """Pooch skips files that are present with the right hash and downloads the rest."""
    corpus = pooch.create(
        path=settings.data_dir / "pdf",
        base_url="",
        urls={f"{pid}.pdf": f"https://arxiv.org/pdf/{pid}" for pid in settings.papers},
        registry={f"{pid}.pdf": f"sha256:{sha}" for pid, sha in settings.papers.items()},
        retry_if_failed=3,
    )
    for name in corpus.registry:
        corpus.fetch(name)

def save_metadata(settings: AcquisitionSettings) -> None:
    """Title, authors and dates of all papers, in one arXiv API query."""
    results = arxiv.Client().results(arxiv.Search(id_list=list(settings.papers)))
    papers = [
        {
            "arxiv_id": r.get_short_id(),
            "title": " ".join(r.title.split()),
            "authors": [a.name for a in r.authors],
            "published": r.published.date().isoformat(),
            "updated": r.updated.date().isoformat(),
            "url": r.entry_id,
            "pdf_file": f"pdf/{r.get_short_id()}.pdf",
        }
        for r in results
    ]
    path = settings.data_dir / "metadata.json"
    path.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")


def acquire(settings: AcquisitionSettings | None = None, force: bool = False) -> Path:
    """Make sure the corpus is in the data folder and return that folder."""
    settings = settings or AcquisitionSettings()
    if force:
        shutil.rmtree(settings.data_dir / "pdf", ignore_errors=True)
        (settings.data_dir / "metadata.json").unlink(missing_ok=True)

    download_pdfs(settings)
    if not (settings.data_dir / "metadata.json").exists():
        save_metadata(settings)
    return settings.data_dir


def main(force: bool = typer.Option(False, "--force", help="Download all papers again.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log.info("Corpus is in %s", acquire(force=force))


if __name__ == "__main__":
    typer.run(main)
