"""Тесты оценщиков: python -m pytest -q (из каталога source)."""
import json

import analyze
import config

GOLD_LABELS = [label for _, label in config.P2_ITEMS]

LETTER = ("Уважаемая Анна Сергеевна! " + "слово " * 120 +
          "Заказ № 48213, промокод SORRY10 на скидку 10% до 31 декабря, support@technomir.ru. "
          "С уважением, **Команда «ТехноМир»**")


def test_strip_think_removes_reasoning():
    assert analyze.strip_think("<think>рассуждаю</think>\nОтвет") == "Ответ"


def test_extract_json_from_fenced_block():
    assert analyze.extract_json('Вот:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert analyze.extract_json("нет json") is None


def test_generation_all_checks_pass_with_markdown_signature():
    score, info = analyze.score_generation(LETTER)
    assert info["signature"] and info["length"]
    assert score == 1.0


def test_generation_detects_corrupted_signature():
    _, info = analyze.score_generation(LETTER.replace("ТехноМир", "Технomir"))
    assert not info["signature"]


def test_generation_length_bounds():
    _, info = analyze.score_generation("Заказ 48213")
    assert not info["length"]


def test_classification_perfect_and_extra_label():
    assert analyze.score_classification(json.dumps(GOLD_LABELS))[0] == 1.0
    shifted = ["Billing", "Tech support", "Sales"] * 3 + ["Sales"]
    score, info = analyze.score_classification(json.dumps(shifted))
    assert info["n_pred"] == 10 and score < 1.0


def test_classification_invalid_json():
    assert analyze.score_classification("Billing, Sales")[0] == 0.0


def test_extraction_normalizes_numbers_and_strings():
    pred = dict(config.P3_GOLD, screen_inches="6,78", price_rub="74 990", color="«Марсианский оранжевый»")
    assert analyze.score_extraction(json.dumps(pred, ensure_ascii=False))[0] == 1.0


def test_extraction_penalizes_hallucinated_null_field():
    pred = dict(config.P3_GOLD, weight_g=210)
    score, info = analyze.score_extraction(json.dumps(pred, ensure_ascii=False))
    assert "weight_g" in info["wrong"] and score == 0.9


def test_rep3_detects_loops():
    assert analyze.rep3("раз два три " * 10) > 0.8
    assert analyze.rep3("каждое слово здесь встречается ровно один раз") == 0.0
