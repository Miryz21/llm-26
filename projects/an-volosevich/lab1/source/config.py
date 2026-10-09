"""Конфигурация эксперимента: модели, промпты, режимы генерации."""
import json
import os

LLMS_DIR = os.path.expanduser(os.environ.get("LLMS_DIR", "~/llms"))
LLAMA_SERVER = os.path.join(LLMS_DIR, "llama.cpp", "llama-b11524", "llama-server")
MODELS_DIR = os.path.join(LLMS_DIR, "models")
HOST, PORT = "127.0.0.1", 8080

# Число повторов каждого прогона — для оценки стабильности/детерминизма.
REPEATS = 3

# --- Модели --------------------------------------------------------------------
# Все три поднимаются через llama-server (OpenAI-совместимый REST /v1/chat/completions).
MODELS = [
    {
        "name": "qwen3.8-flash-next-reap256",
        "family": "Qwen",
        "gguf": "qwen38/Qwen3.8-Flash-Next-UD-Q3_K_XL-reap256-00001-of-00002.gguf",
        # --fit раскладывает эксперты между VRAM и RAM; per_layer_token_embd остаётся в mmap (SSD).
        # reasoning_effort — параметр шаблона чата, а не сэмплинга: одинаков в обоих режимах.
        "args": ["-c", "16384", "--parallel", "1", "-fa", "on", "--fit", "on", "-t", "12",
                 "--chat-template-kwargs", '{"reasoning_effort":"low"}'],
        "load_timeout": 900,
    },
    {
        "name": "llama-3.1-8b-instruct",
        "family": "Llama",
        "gguf": "llama31-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "args": ["-c", "8192", "-ngl", "99", "-fa", "on"],
        "load_timeout": 180,
    },
    {
        "name": "ministral-3-8b-instruct",
        "family": "Mistral",
        "gguf": "ministral3-8b/Ministral-3-8B-Instruct-2512-Q4_K_M.gguf",
        "args": ["-c", "8192", "-ngl", "99", "-fa", "on"],
        "load_timeout": 180,
    },
]

# --- Промпты -------------------------------------------------------------------
P1_GENERATION = """Напиши вежливое письмо клиенту от интернет-магазина «ТехноМир».
Факты:
- клиент: Анна Сергеевна Петрова;
- заказ № 48213 (робот-пылесос) задерживается на 3 рабочих дня из-за сбоя у службы доставки;
- в качестве извинения — промокод SORRY10 на скидку 10% на следующую покупку, действует до 31 декабря;
- контакт поддержки: support@technomir.ru.
Требования: деловой и тёплый тон, 120–180 слов, не добавляй фактов, которых нет в списке.
Подпись: «Команда ТехноМир»."""

P2_LABELS = ["Billing", "Tech support", "Sales"]
P2_ITEMS = [
    ("С карты списали деньги дважды за одну подписку, верните, пожалуйста, лишнее.", "Billing"),
    ("Приложение вылетает при попытке открыть раздел «Отчёты» после обновления.", "Tech support"),
    ("Сколько будет стоить корпоративная лицензия на 50 сотрудников?", "Sales"),
    ("Не могу войти в аккаунт: пишет «неверный токен», хотя пароль правильный.", "Tech support"),
    ("Пришлите, пожалуйста, закрывающие документы и счёт-фактуру за сентябрь.", "Billing"),
    ("Хотим перейти на тариф Enterprise, можно ли получить скидку при оплате за год?", "Sales"),
    ("Почему в счёте за октябрь появилась строка «доп. услуги», мы их не подключали?", "Billing"),
    ("Интеграция с 1С перестала выгружать заказы, в логах ошибка 502.", "Tech support"),
    ("Есть ли у вас партнёрская программа для интеграторов и какие условия?", "Sales"),
]
P2_CLASSIFICATION = (
    "Классифицируй каждое обращение клиента ровно в одну категорию из списка "
    f"{json.dumps(P2_LABELS, ensure_ascii=False)}.\n"
    "Ответь ТОЛЬКО JSON-массивом меток в том же порядке, без пояснений.\n\nОбращения:\n"
    + "\n".join(f"{i}. {text}" for i, (text, _) in enumerate(P2_ITEMS, 1))
)

P3_EXTRACTION = """Извлеки из описания товара поля и верни ТОЛЬКО JSON-объект с ключами:
brand, model, screen_inches, ram_gb, storage_gb, battery_mah, color, price_rub, weight_g, warranty_months.
Числа — числами, если значения нет в тексте — null.

Описание:
«Новинка сезона — смартфон Realme GT 7 Pro в цвете «марсианский оранжевый». Флагман получил
AMOLED-дисплей диагональю 6,78 дюйма, 16 ГБ оперативной и 512 ГБ встроенной памяти, а также
аккумулятор на 6500 мАч с быстрой зарядкой 120 Вт. Официальная гарантия производителя — 1 год.
Цена в нашем магазине — 74 990 ₽, при оформлении предзаказа в подарок фирменный чехол.»"""

P3_GOLD = {
    "brand": "Realme", "model": "GT 7 Pro", "screen_inches": 6.78, "ram_gb": 16,
    "storage_gb": 512, "battery_mah": 6500, "color": "марсианский оранжевый",
    "price_rub": 74990, "weight_g": None, "warranty_months": 12,
}

PROMPTS = {
    "P1_generation": P1_GENERATION,
    "P2_classification": P2_CLASSIFICATION,
    "P3_extraction": P3_EXTRACTION,
}

# --- Режимы --------------------------------------------------------------------
# A — базовый: в запрос не передаётся ни один параметр сэмплинга (дефолты llama-server).
# B — тюнинг: temperature, top_p, top_k, repeat_penalty, max_tokens под тип задачи.
MODES = {
    "A_base": {p: {} for p in PROMPTS},
    "B_tuned": {
        "P1_generation": {"temperature": 0.7, "top_p": 0.9, "top_k": 40,
                          "repeat_penalty": 1.1, "max_tokens": 1024},
        "P2_classification": {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                              "repeat_penalty": 1.0, "max_tokens": 1024},
        "P3_extraction": {"temperature": 0.0, "top_p": 1.0, "top_k": 1,
                          "repeat_penalty": 1.0, "max_tokens": 1024},
    },
}
