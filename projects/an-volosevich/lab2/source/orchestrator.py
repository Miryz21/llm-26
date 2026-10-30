"""Run the whole lab-2 pipeline, step by step.

    1   download the papers       data_acqusition.py
    2   parse PDF → Markdown      data_parsing.py
    3a  split into chunks         data_chunking.py
    3b  embed the chunks          build_index.py
    4   load into Qdrant          load_to_vector_store.py
    5   evaluate retrieval        evaluate.py   (only with --evaluate)
    6   answer a question         rag.py        (only with --question)

Every step reuses its up-to-date output, so a repeated run only does what changed.
Settings are each step's defaults, or its RAG_* environment variables.
Step 4 needs a running Qdrant, steps 5–6 also llama-server; the steps print how to start them.

Usage: python orchestrator.py [--evaluate] [--question "Какой сэмплер использует Direct3D?"]
"""
import logging

import typer

from build_index import IndexSettings, build_index
from data_acqusition import acquire
from data_chunking import ChunkingSettings, chunk
from data_parsing import parse
from evaluate import evaluate
from load_to_vector_store import VectorStoreSettings, load
from rag import ContextTooSmallError, RagEngine, RagSettings, print_answer
from retriever import QueryTooLongError

log = logging.getLogger("orchestrator")


def main(
    run_evaluation: bool = typer.Option(False, "--evaluate", help="Also run step 5, the retrieval evaluation."),
    question: str | None = typer.Option(None, help="Also run step 6: answer this question."),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # hide library chatter and the retriever's per-question translations
    for name in ("httpx", "httpx2", "FlagEmbedding", "docling", "arxiv", "retriever"):
        logging.getLogger(name).setLevel(logging.WARNING)

    log.info("Step 1: download the papers")
    acquire()

    log.info("Step 2: parse PDF → Markdown")
    parse()

    log.info("Step 3a: split into chunks")
    chunking = ChunkingSettings()
    chunk(chunking)

    log.info("Step 3b: embed the chunks")
    index = IndexSettings(chunks=chunking.name)  # the index is built from the chunks just made
    build_index(index)

    log.info("Step 4: load into Qdrant")
    load(VectorStoreSettings(index=index.name))

    if run_evaluation:
        log.info("Step 5: evaluate retrieval")
        log.info("Results are in %s", evaluate())
    else:
        log.info("Step 5: skipped (add --evaluate)")

    if question:
        log.info("Step 6: answer the question")
        try:
            print_answer(RagEngine(RagSettings(index=index.name)).answer(question))
        except (QueryTooLongError, ContextTooSmallError) as error:
            raise SystemExit(str(error))
    else:
        log.info("Step 6: skipped (add --question \"...\")")


if __name__ == "__main__":
    typer.run(main)
