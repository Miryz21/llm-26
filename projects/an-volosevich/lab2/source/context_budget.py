"""Бюджет контекста LLM: промпт с найденными фрагментами и ответом должен поместиться в окно модели.

Токены считаются токенизатором самой LLM: llama-server собирает промпт по шаблону чата
(/apply-template) и токенизирует его со служебными токенами (/tokenize). Если
промпт + max_tokens ответа не помещаются в окно:
1. самые релевантные фрагменты (начало выдачи) остаются целиком, пока на остальные хватает
   хотя бы MIN_SUMMARY токенов;
2. остальные фрагменты сжимаются LLM под вопрос — сохраняются факты, числа и названия,
   относящиеся к вопросу; в промпте такой фрагмент помечен как сжатый;
3. если даже по MIN_SUMMARY на фрагмент не хватает, наименее релевантные фрагменты отбрасываются;
   если сжатие вышло длиннее отведённого, последний фрагмент пересжимается плотнее и только потом
   отбрасывается.
Вопрос, который сам по себе не помещается в окно, — ошибка.
"""
import functools
import json
import re
import urllib.error
import urllib.request

from config import N_CTX

MARGIN = 32          # запас на расхождение подсчёта и служебные токены ответа
MIN_SUMMARY = 80     # меньше — от фрагмента остаётся одна фраза, проще его отбросить
JOIN = 2             # токены разделителя между блоками фрагментов
MARK = 12            # пометка «сжато под вопрос» в заголовке сжатого блока
SUMMARY_SYSTEM = (
    "You compress a fragment of a scientific paper for answering a question. Keep every fact, number, "
    "name and term from the fragment that is relevant to the question; drop everything else. Do not add "
    "anything that is not in the fragment and do not answer the question. Write in the language of the "
    "fragment, as plain text, at most {words} words.")


class LLMError(RuntimeError):
    """llama-server недоступен или отклонил запрос; сообщение — для пользователя."""


def base_url(chat_url: str) -> str:
    return re.sub(r"/v1/chat/completions/?$", "", chat_url)


def post(url: str, payload: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except (ValueError, KeyError, TypeError):
            pass
        raise LLMError(f"llama-server вернул {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"llama-server недоступен ({url}): {e.reason}") from e


@functools.cache
def context_size(chat_url: str) -> int:
    if N_CTX:
        return N_CTX
    url = base_url(chat_url) + "/props"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return int(json.load(resp)["default_generation_settings"]["n_ctx"])
    except urllib.error.URLError as e:
        raise LLMError(f"llama-server недоступен ({url}): {e.reason}") from e
    except (KeyError, ValueError, TypeError) as e:
        raise LLMError(f"llama-server не сообщает размер контекста в /props; задайте RAG_N_CTX ({e})") from e


def text_tokens(text: str, chat_url: str) -> int:
    return len(post(base_url(chat_url) + "/tokenize", {"content": text})["tokens"])


def prompt_tokens(messages: list[dict], chat_url: str) -> int:
    prompt = post(base_url(chat_url) + "/apply-template", {"messages": messages})["prompt"]
    return len(post(base_url(chat_url) + "/tokenize", {"content": prompt, "add_special": True})["tokens"])


def summarize(question: str, text: str, limit: int, chat_url: str, max_tokens: int) -> str:
    """Сжатие фрагмента под вопрос до ~limit токенов."""
    words = max(20, int(limit * 0.6))   # ~1.6 токена на слово в корпусе
    system = SUMMARY_SYSTEM.format(words=words)
    # Сам фрагмент тоже должен поместиться в окно вызова сжатия; иначе он обрезается по длине.
    room = context_size(chat_url) - limit - MARGIN - text_tokens(system + question, chat_url) - 32
    while text and text_tokens(text, chat_url) > room:
        text = text[: int(len(text) * 0.8)]
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}\n\nFragment:\n{text}"}]
    resp = post(chat_url, {"messages": messages, "temperature": 0, "max_tokens": min(limit, max_tokens),
                           "cache_prompt": False})
    return resp["choices"][0]["message"]["content"].strip()


def fit(question: str, contexts: list[dict], build_prompt, chat_url: str, max_tokens: int,
        focus: str | None = None) -> tuple[list[dict], dict]:
    """Фрагменты, помещающиеся в окно вместе с ответом, и отчёт о том, что пришлось сделать.

    focus — формулировка вопроса для сжатия фрагментов (поисковый запрос на языке корпуса): с русским
    вопросом модель сжимает английский фрагмент по-русски, теряя термины и тратя больше токенов.
    """
    focus = focus or question
    n_ctx = context_size(chat_url)
    budget = n_ctx - max_tokens - MARGIN
    info = {"n_ctx": n_ctx, "budget": budget, "summarized": [], "dropped": 0}

    total = prompt_tokens(build_prompt(question, contexts), chat_url)
    if total <= budget:
        return contexts, {**info, "prompt_tokens": total}

    base = prompt_tokens(build_prompt(question, []), chat_url)
    if base > budget:
        raise LLMError(f"вопрос слишком длинный: {base} токенов промпта без фрагментов при бюджете {budget} "
                       f"(окно {n_ctx} − ответ {max_tokens})")
    available = budget - base
    costs = [prompt_tokens(build_prompt(question, [c]), chat_url) - base + JOIN for c in contexts]

    kept = list(contexts)
    while len(kept) > 1 and len(kept) * MIN_SUMMARY > available:   # даже по минимуму не хватает
        kept.pop()
    info["dropped"] = len(contexts) - len(kept)

    full = 0                                                        # сколько первых фрагментов — целиком
    while full < len(kept) and sum(costs[:full + 1]) + MIN_SUMMARY * (len(kept) - full - 1) <= available:
        full += 1
    limits = {}
    rest = kept[full:]
    if rest:
        share = (available - sum(costs[:full])) // len(rest)
        for i, c in enumerate(rest, full):
            if costs[i] > share:
                header = costs[i] - text_tokens(c["text"], chat_url)
                limits[i] = max(share - header - MARK, 20)
                kept[i] = {**c, "prompt_text": summarize(focus, c["text"], limits[i], chat_url, max_tokens)}

    total = prompt_tokens(build_prompt(question, kept), chat_url)
    while total > budget and kept:
        last = len(kept) - 1
        if last in limits and limits[last] - (total - budget) - MARK >= MIN_SUMMARY // 2:
            limits[last] -= total - budget + MARK                   # пересжать плотнее
            kept[last] = {**kept[last], "prompt_text": summarize(focus, kept[last]["text"], limits[last],
                                                                 chat_url, max_tokens)}
        elif len(kept) > 1:
            kept.pop()
            limits.pop(last, None)
            info["dropped"] += 1
        else:
            break
        total = prompt_tokens(build_prompt(question, kept), chat_url)
    if total > budget:
        raise LLMError(f"фрагмент не помещается в окно даже после сжатия: {total} токенов при бюджете {budget}")
    info["summarized"] = sorted(i + 1 for i in limits)
    return kept, {**info, "prompt_tokens": total}
