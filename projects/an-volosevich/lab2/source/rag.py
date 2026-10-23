"""RAG-движок: вопрос → поиск в Qdrant → промпт с контекстами → LLM → ответ с цитатами.

LLM — Ministral-3-8B-Instruct из лабораторной №1, поднятая через llama-server
(OpenAI-совместимый /v1/chat/completions). Режимы:
    python rag.py "вопрос"          — один ответ в терминал;
    python rag.py                   — интерактивный диалог;
    python rag.py --serve 8000      — HTTP-сервис: POST /ask {"question": "..."} → JSON.
"""
import argparse
import json
import re
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import LLM_URL, TRANSLATE_QUERY, IndexConfig, add_index_args, index_config_from_args
from encoders import terms
from load_to_vector_store import client, load
from retriever import search, translate_query

SYSTEM = """Ты — ассистент, который отвечает на вопросы по научным статьям о диффузионных моделях.
Отвечай только на основе фрагментов из контекста. Каждый фрагмент пронумерован: [1], [2] и т.д.
После каждого утверждения ставь ссылку на фрагмент(ы), из которых оно взято, например: «... 50 шагов DDIM [2].»
Если в контексте нет ответа, так и скажи: «В предоставленных фрагментах нет ответа на этот вопрос».
Не придумывай факты, числа и названия, которых нет во фрагментах. Названия методов, моделей и датасетов
не переводи. Отвечай на языке вопроса, кратко: 2–6 предложений."""

# Ссылки статей на их библиографию ([12], [3, 5]) совпадают по формату с нашими номерами фрагментов,
# и модель начинает выдавать их за цитаты — из текста для LLM они убираются.
PAPER_REFS = re.compile(r"\s?\[\d+(?:\s*[,–-]\s*\d+)*\]")

# Жадное декодирование без кэша промпта: переиспользование KV-кэша меняет порядок вычислений,
# и ответы на один и тот же вопрос начинают отличаться между запусками.
GENERATION = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 700, "cache_prompt": False}


def build_prompt(question: str, contexts: list[dict]) -> list[dict]:
    blocks = []
    for i, c in enumerate(contexts, 1):
        pages = ", ".join(map(str, c["pages"]))
        text = PAPER_REFS.sub("", c["text"])
        blocks.append(f"[{i}] {c['paper']} — {c['section'] or 'начало статьи'} (стр. {pages})\n{text}")
    user = "Контекст:\n\n" + "\n\n".join(blocks) + f"\n\nВопрос: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def call_llm(messages: list[dict], url: str = LLM_URL) -> dict:
    body = json.dumps({"messages": messages, **GENERATION}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)


def snippet(text: str, query: str, limit: int = 240) -> str:
    """Предложение фрагмента, больше всего пересекающееся по терминам с запросом.

    Обрывки короче 40 символов («1.», «Section.») — следствие разбиения по точкам — не рассматриваются.
    """
    wanted = set(terms(query))
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    candidates = [s for s in sentences if len(s) >= 40] or sentences
    best = max(candidates, key=lambda s: (len(wanted & set(terms(s))), len(s)))
    return best if len(best) <= limit else best[:limit].rsplit(" ", 1)[0] + "…"


def answer(question: str, cfg: IndexConfig, top_n: int = 5, ef: int = 64, llm_url: str = LLM_URL,
           qc=None, translate: bool = TRANSLATE_QUERY) -> dict:
    qc = qc or client()
    t0 = time.perf_counter()
    query = translate_query(question) if translate else question
    contexts = search(qc, cfg, query, top_n, ef=ef)
    t1 = time.perf_counter()
    resp = call_llm(build_prompt(question, contexts), llm_url)
    t2 = time.perf_counter()
    text = resp["choices"][0]["message"]["content"].strip()

    refs = re.findall(r"\[(\d+(?:\s*[,;–-]\s*\d+)*)\]", text)  # [2], [2, 4], [2–3]
    cited = sorted({int(n) for ref in refs for n in re.findall(r"\d+", ref) if 1 <= int(n) <= len(contexts)})
    citations = [{
        "n": n,
        "paper": contexts[n - 1]["paper"],
        "arxiv_id": contexts[n - 1]["arxiv_id"],
        "section": contexts[n - 1]["section"],
        "pages": contexts[n - 1]["pages"],
        "url": contexts[n - 1]["url"],
        "snippet": snippet(contexts[n - 1]["text"], query + " " + text),
    } for n in cited]
    return {
        "question": question,
        "search_query": query,
        "answer": text,
        "contexts": contexts,
        "citations": citations,
        "retrieved": [{"paper": c["paper"], "arxiv_id": c["arxiv_id"], "pages": c["pages"],
                       "section": c["section"], "score": round(c["score"], 4)} for c in contexts],
        "timings": {"retrieval_s": round(t1 - t0, 3), "llm_s": round(t2 - t1, 2)},
        "usage": resp.get("usage", {}),
    }


def render(result: dict) -> str:
    lines = [result["answer"], "", "Источники:"]
    for c in result["citations"]:
        pages = ", ".join(map(str, c["pages"]))
        lines.append(f"  [{c['n']}] {c['paper']} (arXiv:{c['arxiv_id']}), {c['section'] or '—'}, стр. {pages}")
        lines.append(f"      «{c['snippet']}»")
    if not result["citations"]:
        lines.append("  — (модель не сослалась ни на один фрагмент)")
    t = result["timings"]
    if result["search_query"] != result["question"]:
        lines.append(f"\nпоисковый запрос: {result['search_query']}")
    lines.append(f"поиск {t['retrieval_s']} с, генерация {t['llm_s']} с")
    return "\n".join(lines)


def serve(port: int, cfg: IndexConfig, top_n: int, ef: int, llm_url: str, translate: bool) -> None:
    qc = client()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/ask":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            question = (body.get("question") or "").strip()
            if not question:
                self.send_error(400, "question is required")
                return
            result = answer(question, cfg, body.get("top_n", top_n), ef, llm_url, qc, translate)
            result.pop("contexts")
            data = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"POST http://127.0.0.1:{port}/ask  (коллекция для {cfg.index_id})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", help="вопрос; без него — интерактивный режим")
    add_index_args(parser)
    parser.add_argument("--top-n", type=int, default=5, help="сколько фрагментов передать LLM")
    parser.add_argument("--ef", type=int, default=64, help="hnsw_ef при поиске")
    parser.add_argument("--llm-url", default=LLM_URL)
    parser.add_argument("--translate", action=argparse.BooleanOptionalAction, default=TRANSLATE_QUERY,
                        help="переводить запрос на английский перед поиском")
    parser.add_argument("--json", action="store_true", help="вывести ответ в JSON")
    parser.add_argument("--serve", type=int, metavar="PORT", help="запустить HTTP-сервис")
    args = parser.parse_args()
    cfg = index_config_from_args(args)
    load(cfg)

    if args.serve:
        serve(args.serve, cfg, args.top_n, args.ef, args.llm_url, args.translate)
        return
    questions = [args.question] if args.question else iter(lambda: input("\nВопрос> ").strip(), "")
    for q in questions:
        result = answer(q, cfg, args.top_n, args.ef, args.llm_url, translate=args.translate)
        if args.json:
            result.pop("contexts")
            print(json.dumps(result, ensure_ascii=False, indent=1))
        else:
            print(render(result))


if __name__ == "__main__":
    main()
