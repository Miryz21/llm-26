"""Step 6: answer a question about the papers, with citations (RAG).

1. retrieve the top chunks (retriever.py: translation, bge-m3, Qdrant)
2. build the prompt: instructions + numbered fragments + question
3. check that prompt + answer fit the LLM's context; drop the lowest-ranked fragments if not
4. ask the LLM from lab 1 (llama-server, OpenAI-compatible API)
5. return the answer with its citations: paper, section, pages, snippet, link

Requires running Qdrant and llama-server.
Usage: python rag.py "Какой сэмплер использует Direct3D?"
"""
import json
import logging
import re
from pathlib import Path

import httpx
import numpy as np
import openai
import typer
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from retriever import LLAMA_START, Hit, QueryTooLongError, RetrievalSettings, Retriever

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about scientific papers using only the numbered fragments given with the question.
Rules:
- Answer in the language of the question.
- Use only facts from the fragments. If they do not contain the answer, say so.
- After every statement, cite its fragment label in square brackets, e.g. [F2].
- Do not copy the papers' own reference numbers such as [48] into the answer.
- Keep formulas, numbers and names exactly as in the fragments."""

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# fragments are labelled [F1], [F2]…, because the papers' own references look like [1], [48]
CITATION = re.compile(r"\[F(\d+)\]")


# --- Settings and result -----------------------------------------------------------------

class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_ANSWER_")

    index: str = "markdown-800-100-h__hybrid-bge-m3"  # which loaded collection to search
    top_n: int = 5  # fragments given to the LLM
    translate: bool = True  # translate Russian questions for the search (the answer stays in Russian)
    max_answer_tokens: int = 1024  # reserved for the answer inside the LLM's context
    temperature: float = 0.0  # repeatable answers
    llm_url: str = "http://127.0.0.1:8080"  # llama-server; the OpenAI API is under /v1
    data_dir: Path = Path(__file__).resolve().parents[1] / "data"


class Citation(BaseModel):
    number: int  # [F<number>] in the answer
    title: str
    authors: list[str]
    url: str
    section: str
    pages: list[int]
    snippet: str  # the sentence of the fragment closest to what the answer says with [F<number>]


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation]
    prompt_tokens: int  # counted by the LLM's own tokenizer, chat template included
    answer_tokens: int
    context_size: int
    dropped_fragments: int  # lowest-ranked fragments left out to fit the context
    truncated: bool  # the answer hit max_answer_tokens and ends mid-text


class ContextTooSmallError(RuntimeError):
    """Even one fragment does not fit the LLM's context next to the answer reserve."""


# --- Engine ------------------------------------------------------------------------------

class RagEngine:
    """Loads the retriever and connects to the LLM once, then answers many questions."""

    def __init__(self, settings: RagSettings | None = None):
        self.settings = settings or RagSettings()
        s = self.settings
        self.retriever = Retriever(RetrievalSettings(index=s.index, top_k=s.top_n, translate=s.translate))
        self.llm = openai.OpenAI(base_url=f"{s.llm_url}/v1", api_key="unused")
        self.server = httpx.Client(base_url=s.llm_url, timeout=30)
        try:
            props = self.server.get("/props").json()
        except httpx.ConnectError:
            raise SystemExit(f"The LLM is not reachable at {s.llm_url}.\nStart it with:\n  {LLAMA_START}")
        # context size of the running server, e.g. 8192 for "-c 8192"
        self.context_size = props["default_generation_settings"]["n_ctx"]
        if props["total_slots"] > 1:
            # the slots share one KV cache, so parallel requests could together exceed the context
            log.warning("llama-server has %d parallel slots sharing a %d-token context; the budget check "
                        "assumes one request at a time. Start it with --parallel 1.", props["total_slots"], self.context_size)
        # authors and links are not in the chunks; they are joined from metadata.json by arxiv_id
        papers = json.loads((s.data_dir / "metadata.json").read_text(encoding="utf-8"))
        self.papers = {p["arxiv_id"]: p for p in papers}

    def answer(self, question: str) -> Answer:
        hits = self.retriever.search(question)
        messages, hits, prompt_tokens = self.fit_to_context(question, hits)

        reply = self.llm.chat.completions.create(
            model="local",  # llama-server serves one model and ignores the name
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_answer_tokens,
        )
        text = reply.choices[0].message.content.strip()
        truncated = reply.choices[0].finish_reason == "length"
        if truncated:
            log.warning("The answer reached max_answer_tokens=%d and was cut off", self.settings.max_answer_tokens)
        return Answer(
            question=question,
            text=text,
            citations=self.citations(text, hits),
            prompt_tokens=reply.usage.prompt_tokens,
            answer_tokens=reply.usage.completion_tokens,
            context_size=self.context_size,
            dropped_fragments=self.settings.top_n - len(hits),
            truncated=truncated,
        )

    # --- prompt and context budget -----------------------------------------------------

    @staticmethod
    def build_messages(question: str, hits: list[Hit]) -> list[dict]:
        fragments = []
        for number, hit in enumerate(hits, start=1):
            pages = ", ".join(map(str, hit.pages))
            fragments.append(f"[F{number}] {hit.title} — {hit.section or 'front matter'} (p. {pages})\n{hit.text}")
        user = "Fragments:\n\n" + "\n\n".join(fragments) + f"\n\nQuestion: {question}"
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def count_tokens(self, messages: list[dict]) -> int:
        """Exact prompt size: the server renders its chat template and tokenizes the result."""
        prompt = self.server.post("/apply-template", json={"messages": messages}).json()["prompt"]
        # add_special: include the start-of-text token the server adds when generating
        tokens = self.server.post("/tokenize", json={"content": prompt, "add_special": True}).json()["tokens"]
        return len(tokens)

    def fit_to_context(self, question: str, hits: list[Hit]) -> tuple[list[dict], list[Hit], int]:
        """Drop the lowest-ranked fragments until prompt + answer reserve fit the context."""
        budget = self.context_size - self.settings.max_answer_tokens
        while hits:
            messages = self.build_messages(question, hits)
            tokens = self.count_tokens(messages)
            if tokens <= budget:
                return messages, hits, tokens
            log.warning("Prompt has %d tokens, the budget is %d: dropping fragment %d", tokens, budget, len(hits))
            hits = hits[:-1]
        raise ContextTooSmallError(f"No fragment fits into {budget} prompt tokens "
                                   f"(context {self.context_size} minus {self.settings.max_answer_tokens} for the answer)")

    # --- citations ---------------------------------------------------------------------

    def citations(self, text: str, hits: list[Hit]) -> list[Citation]:
        """One citation per fragment number the answer uses, in order of first use."""
        numbers = []
        for match in CITATION.finditer(text):
            number = int(match[1])
            if 1 <= number <= len(hits) and number not in numbers:  # ignore numbers the model made up
                numbers.append(number)

        citations = []
        for number in numbers:
            hit = hits[number - 1]
            paper = self.papers[hit.arxiv_id]
            citations.append(Citation(
                number=number, title=hit.title, authors=paper["authors"], url=paper["url"],
                section=hit.section, pages=hit.pages, snippet=self.snippet(hit.text, text, number),
            ))
        return citations

    def snippet(self, fragment: str, answer: str, number: int) -> str:
        """The fragment sentence most similar (bge-m3, works across languages) to the answer's
        sentences that cite this fragment."""
        claims = " ".join(s for s in SENTENCE_END.split(answer) if f"[F{number}]" in s)
        sentences = [s for s in SENTENCE_END.split(" ".join(fragment.split())) if len(s) > 20] or [fragment]
        vectors = self.retriever.model.encode([claims] + sentences, return_dense=True)["dense_vecs"]
        best = int(np.argmax(vectors[1:] @ vectors[0]))
        return sentences[best][:300]


# --- CLI ---------------------------------------------------------------------------------

def print_answer(answer: Answer) -> None:
    typer.echo(f"\n{answer.text}\n")
    if answer.truncated:
        typer.echo("(the answer was cut off at the answer token limit)")
    if not answer.citations:
        typer.echo("(the answer cites no fragment)")
    for c in answer.citations:
        authors = c.authors[0] + (" et al." if len(c.authors) > 1 else "")
        pages = ", ".join(map(str, c.pages))
        typer.echo(f"[F{c.number}] {c.title} ({authors}) — {c.section or 'front matter'}, p. {pages}\n"
                   f"    {c.url}\n    «{c.snippet}»")
    typer.echo(f"\nTokens: prompt {answer.prompt_tokens} + answer {answer.answer_tokens} "
               f"of {answer.context_size} context; fragments dropped to fit: {answer.dropped_fragments}")


def main(question: str = typer.Argument(..., help="The question about the papers.")) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name in ("httpx", "httpx2", "FlagEmbedding"):  # hide HTTP requests and model-loading notes
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        answer = RagEngine().answer(question)
    except (QueryTooLongError, ContextTooSmallError) as error:
        raise SystemExit(str(error))
    print_answer(answer)


if __name__ == "__main__":
    typer.run(main)
