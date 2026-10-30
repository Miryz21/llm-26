"""Step 5b: measure retrieval quality on the evaluation questions and compare configurations.

Every experiment changes the base configuration in one place (chunking or retrieval). Missing
builds are made on the way; each pipeline step reuses its up-to-date results.

Ground truth: each question in lab2/questions.json has 1–3 evidence phrases copied from the paper
(formulas as written in the paper's LaTeX source). A chunk is relevant if it contains one; texts are
compared without spaces, braces and LaTeX style commands. A phrase found in no chunk of a build is a
parsing gap: it is reported and left out of that build's metrics.

Metrics per experiment and question language, k = 5 and 10:
    Recall@k     share of a question's evidence phrases found in its top k chunks
    Precision@k  share of the top k chunks that contain an evidence phrase
    MRR          1 / rank of the first relevant chunk (0 if none in the top 10)

Output: data/results/retrieval.json (every question) and data/results/retrieval.md (tables).
Requires running Qdrant and llama-server.
Usage: python evaluate.py [--only base,dense,docling]
"""
import json
import logging
import re
import statistics
import time
from pathlib import Path

import typer
from FlagEmbedding import BGEM3FlagModel

from build_index import IndexSettings, build_index
from data_chunking import ChunkingSettings, chunk
from load_to_vector_store import VectorStoreSettings, load
from retriever import CYRILLIC, Hit, RetrievalSettings, Retriever

log = logging.getLogger(__name__)

LAB_DIR = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = LAB_DIR / "questions.json"
RESULTS_DIR = LAB_DIR / "data" / "results"
K = (5, 10)
LANGUAGES = {"ru": "question", "en": "question_en"}


# --- Experiments -------------------------------------------------------------------------

# base: markdown splitter 800/100 with headings → hybrid bge-m3 index → hybrid search with translation
BASE_RETRIEVAL = {"mode": "hybrid", "translate": True, "top_k": max(K)}

EXPERIMENTS = {
    "base": {},
    "dense": {"retrieval": {"mode": "dense"}},
    "sparse": {"retrieval": {"mode": "sparse"}},
    "no-translation": {"retrieval": {"translate": False}},
    "no-headings": {"chunking": {"headings": False}},
    "recursive": {"chunking": {"splitter": "recursive"}},
    "docling": {"chunking": {"splitter": "docling"}},
    "chunk-400": {"chunking": {"chunk_size": 400, "overlap": 60}},
    "exact-search": {"retrieval": {"exact": True}},  # no HNSW: shows what the graph costs in quality
    "hnsw-ef-8": {"retrieval": {"hnsw_ef": 8}},  # a very shallow HNSW search
}


def prepare(experiment: dict) -> tuple[RetrievalSettings, Path]:
    """Build (or reuse) every step the experiment needs; return its retrieval settings and chunks file."""
    chunking = ChunkingSettings(**experiment.get("chunking", {}))
    chunks_dir = chunk(chunking)
    index = IndexSettings(chunks=chunking.name, vectors="hybrid")  # hybrid serves all three search modes
    build_index(index)
    load(VectorStoreSettings(index=index.name))
    retrieval = RetrievalSettings(index=index.name, **{**BASE_RETRIEVAL, **experiment.get("retrieval", {})})
    return retrieval, chunks_dir / "chunks.jsonl"


# --- Ground truth and scoring ------------------------------------------------------------

STYLE_COMMANDS = re.compile(r"\\(text|textbf|mathbf|mathrm|boldsymbol|bm|left|right|displaystyle)(?![a-z])")


def normalize(text: str) -> str:
    """Lowercase, without spaces, braces, $ and LaTeX style commands, so a formula from the paper's
    source ("L_\\text{BCE}") and from docling ("L _ { B C E }") compare equal."""
    text = STYLE_COMMANDS.sub("", text)
    text = re.sub(r"\\[,;:! ]|\\quad|[\s{}$]", "", text)
    return text.lower()


def find_gaps(questions: list[dict], chunks_file: Path) -> dict[str, list[str]]:
    """Evidence phrases that no chunk of the build contains, per question id."""
    with chunks_file.open(encoding="utf-8") as f:
        texts = [normalize(json.loads(line)["text"]) for line in f]
    gaps = {}
    for q in questions:
        missing = [p for p in q["evidence"] if not any(normalize(p) in t for t in texts)]
        if missing:
            gaps[q["id"]] = missing
    return gaps


def score(question: dict, hits: list[Hit], missing: list[str]) -> dict | None:
    """Recall@k, Precision@k and MRR of one question; None if none of its phrases is in the build."""
    phrases = [normalize(p) for p in question["evidence"] if p not in missing]
    if not phrases:
        return None

    texts = [normalize(hit.text) for hit in hits]
    relevant = [any(p in t for p in phrases) for t in texts]  # per rank: does this chunk contain a phrase?
    scores = {}
    for k in K:
        scores[f"R@{k}"] = sum(any(p in t for t in texts[:k]) for p in phrases) / len(phrases)
        scores[f"P@{k}"] = sum(relevant[:k]) / k
    scores["MRR"] = next((1 / (rank + 1) for rank, r in enumerate(relevant) if r), 0.0)
    return scores


# --- Running -----------------------------------------------------------------------------

def run_experiment(experiment: dict, questions: list[dict], model: BGEM3FlagModel) -> dict:
    settings, chunks_file = prepare(experiment)
    retriever = Retriever(settings, model=model)
    gaps = find_gaps(questions, chunks_file)

    rows = []
    for lang, key in LANGUAGES.items():
        for q in questions:
            start = time.time()
            hits = retriever.search(q[key])
            rows.append({
                "id": q["id"], "category": q["category"], "lang": lang,
                "ms": round((time.time() - start) * 1000, 1),
                "hits": [hit.chunk_id for hit in hits],
                "scores": score(q, hits, gaps.get(q["id"], [])),
            })

    with chunks_file.open(encoding="utf-8") as f:
        chunk_count = sum(1 for _ in f)
    return {"settings": settings.model_dump(mode="json"), "chunks": chunk_count, "gaps": gaps, "questions": rows}


def warm_up_translations(questions: list[dict], model: BGEM3FlagModel) -> float:
    """Translate every Russian question once (they are cached), so search times exclude the LLM.
    Returns the median translation time in ms."""
    retriever = Retriever(RetrievalSettings(translate=True), model=model)
    times = []
    for q in questions:
        if CYRILLIC.search(q["question"]):
            start = time.time()
            retriever.translate(q["question"])
            times.append((time.time() - start) * 1000)
    return statistics.median(times)


def average(rows: list[dict], lang: str, categories: tuple[str, ...]) -> dict:
    """Mean of every metric over the scored questions of one language and category set."""
    scored = [r["scores"] for r in rows if r["lang"] == lang and r["category"] in categories and r["scores"]]
    if not scored:
        return {}
    return {metric: statistics.mean(s[metric] for s in scored) for metric in scored[0]} | {"n": len(scored)}


# --- Report ------------------------------------------------------------------------------

def report(results: dict, translation_ms: float) -> str:
    """Markdown tables: all questions, formula questions, parsing gaps."""
    metrics = [f"R@{k}" for k in K] + [f"P@{k}" for k in K] + ["MRR"]
    lines = ["# Retrieval evaluation", "",
             f"Questions per language: {len(results['base']['questions']) // 2}. "
             f"Translation of a Russian question: median {translation_ms:.0f} ms (not included in search time).", "",
             "## All questions", "",
             "| Experiment | Chunks | " + " | ".join(f"ru {m}" for m in metrics) + " | "
             + " | ".join(f"en {m}" for m in metrics) + " | Search ms |",
             "|" + "---|" * (3 + 2 * len(metrics))]
    for name, result in results.items():
        cells = []
        for lang in LANGUAGES:
            avg = average(result["questions"], lang, ("text", "formula", "formula-inline"))
            cells += [f"{avg[m]:.3f}" for m in metrics]
        ms = statistics.median(r["ms"] for r in result["questions"])
        lines.append(f"| {name} | {result['chunks']} | " + " | ".join(cells) + f" | {ms:.0f} |")

    lines += ["", "## Formula questions (only those whose formula was parsed; the others are parsing gaps)", "",
              "| Experiment | ru R@5 | ru MRR | en R@5 | en MRR | questions scored |", "|---|---|---|---|---|---|"]
    for name, result in results.items():
        ru, en = (average(result["questions"], lang, ("formula", "formula-inline")) for lang in LANGUAGES)
        lines.append(f"| {name} | {ru['R@5']:.3f} | {ru['MRR']:.3f} | {en['R@5']:.3f} | {en['MRR']:.3f} | {ru['n']} |")

    # gaps depend only on the chunk build, so each build is listed once
    gaps_by_build = {result["settings"]["index"].split("__")[0]: result["gaps"] for result in results.values()}
    lines += ["", "## Parsing gaps: evidence phrases found in no chunk", "", "| Chunk build | Question | Phrase |", "|---|---|---|"]
    for build, gaps in gaps_by_build.items():
        for qid, phrases in gaps.items():
            for phrase in phrases:
                lines.append(f"| {build} | {qid} | `{phrase}` |")
    return "\n".join(lines) + "\n"


def evaluate(names: list[str] | None = None) -> Path:
    """Run the experiments (all by default), write the results and return the results folder."""
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)  # shared by all experiments
    translation_ms = warm_up_translations(questions, model)

    results = {}
    for name in names or EXPERIMENTS:
        log.info("Experiment %s", name)
        results[name] = run_experiment(EXPERIMENTS[name], questions, model)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "retrieval.md").write_text(report(results, translation_ms), encoding="utf-8")
    return RESULTS_DIR


# --- CLI ---------------------------------------------------------------------------------

def main(only: str | None = typer.Option(None, help=f"Comma-separated experiments, from: {', '.join(EXPERIMENTS)}.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpx2", "FlagEmbedding", "retriever"):  # hide requests, model notes, translations
        logging.getLogger(name).setLevel(logging.WARNING)

    names = only.split(",") if only else None
    unknown = set(names or []) - set(EXPERIMENTS)
    if unknown:
        raise SystemExit(f"Unknown experiments: {', '.join(sorted(unknown))}")
    results_dir = evaluate(names)
    typer.echo((results_dir / "retrieval.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    typer.run(main)
