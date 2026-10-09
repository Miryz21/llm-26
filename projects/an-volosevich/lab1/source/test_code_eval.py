"""Coding-suite tests: reference solutions must pass every hidden test, broken ones must not."""
import inspect
from collections import Counter

import pytest

import code_eval
import code_tasks


def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]


def merge_intervals(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def int_to_roman(n):
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
             (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for value, sym in table:
        while n >= value:
            out, n = out + sym, n - value
    return out


def top_k_frequent_words(words, k):
    return [w for w, _ in sorted(Counter(words).items(), key=lambda x: (-x[1], x[0]))[:k]]


def decode_string(s):
    stack, cur, num = [], "", 0
    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)
        elif c == "[":
            stack.append((cur, num))
            cur, num = "", 0
        elif c == "]":
            prev, k = stack.pop()
            cur = prev + cur * k
        else:
            cur += c
    return cur


def longest_valid_parentheses(s):
    best, stack = 0, [-1]
    for i, c in enumerate(s):
        if c == "(":
            stack.append(i)
        else:
            stack.pop()
            if stack:
                best = max(best, i - stack[-1])
            else:
                stack.append(i)
    return best


def min_window(s, t):
    need, missing = Counter(t), len(t)
    left, best = 0, (0, float("inf"))
    for right, c in enumerate(s, 1):
        missing -= need[c] > 0
        need[c] -= 1
        if not missing:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if right - left < best[1] - best[0]:
                best = (left, right)
            need[s[left]] += 1
            missing += 1
            left += 1
    return "" if best[1] == float("inf") else s[best[0]:best[1]]


def calculate(expression):
    tokens = [t for t in expression.replace("(", " ( ").replace(")", " ) ").replace("+", " + ")
              .replace("-", " - ").replace("*", " * ").replace("/", " / ").split()]
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def take():
        nonlocal pos
        pos += 1
        return tokens[pos - 1]

    def factor():
        tok = take()
        if tok == "-":
            return -factor()
        if tok == "(":
            value = expr()
            take()
            return value
        return int(tok)

    def term():
        value = factor()
        while peek() in ("*", "/"):
            op, rhs = take(), factor()
            value = value * rhs if op == "*" else int(value / rhs)
        return value

    def expr():
        value = term()
        while peek() in ("+", "-"):
            op, rhs = take(), term()
            value = value + rhs if op == "+" else value - rhs
        return value

    return expr()


REFERENCE = {p["id"]: globals()[p["func"]] for p in code_tasks.PROBLEMS}


@pytest.mark.parametrize("problem", code_tasks.PROBLEMS, ids=lambda p: p["id"])
def test_reference_solution_passes_all_cases(problem):
    code = inspect.getsource(REFERENCE[problem["id"]])
    if problem["func"] in ("top_k_frequent_words", "min_window"):
        code = "from collections import Counter\n" + code
    res = code_eval.run_tests(code, problem)
    assert res["passed"] == res["total"], res["errors"]


def test_wrong_solution_is_detected():
    problem = code_tasks.PROBLEMS[0]
    res = code_eval.run_tests("def is_palindrome(s):\n    return s == s[::-1]\n", problem)
    assert 0 < res["passed"] < res["total"]


def test_eval_shortcut_fails_calculator():
    problem = code_tasks.PROBLEMS[-1]
    res = code_eval.run_tests("def calculate(expression):\n    return int(eval(expression))\n", problem)
    assert res["passed"] < res["total"]


def test_infinite_loop_times_out():
    res = code_eval.run_tests("def is_palindrome(s):\n    while True: pass\n", code_tasks.PROBLEMS[0])
    assert res["passed"] == 0 and res["errors"] == ["timeout"]


def test_extract_code_prefers_python_block():
    text = "<think>plan</think>Here:\n```python\ndef f():\n    return 1\n```\nDone."
    assert code_eval.extract_code(text) == "def f():\n    return 1\n"
