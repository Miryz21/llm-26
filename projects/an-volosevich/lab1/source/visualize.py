"""Строит графики по results/*.jsonl в results/plots/*.png.

  quality_matrix.png    — сводная матрица качества: модель × режим × задача (судья, авто, код)
  speed_quality.png     — компромисс скорость генерации ↔ качество, A и B для каждой модели
  latency_breakdown.png — из чего складывается время ответа: prefill и генерация по задачам
  code_heatmap.png      — задачи по программированию: пройденные тесты по каждой задаче и режиму
  speed.png             — скорость генерации и обработки промпта (две панели, у каждой своя шкала)
  tokens_breakdown.png  — длина ответа: токены ответа и рассуждения по задачам и режимам
  stability_heatmap.png — похожесть ответов между повторами и число уникальных ответов
"""
import difflib
import itertools
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import analyze  # noqa: E402
from judge import load_scores  # noqa: E402

PLOTS_DIR = os.path.join(analyze.RESULTS_DIR, "plots")
PROMPT_RU = {"P1_generation": "P1 генерация", "P2_classification": "P2 классификация",
             "P3_extraction": "P3 извлечение"}
SHORT = {"llama-3.1-8b-instruct": "Llama-3.1-8B", "ministral-3-8b-instruct": "Ministral-3-8B",
         "qwen3.8-flash-next-reap256": "Qwen3.8 REAP-256"}

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Категориальные цвета, различимые при дальтонизме; режимы A/B различаются ещё и формой маркера.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
COLOR_A, COLOR_B = SERIES[0], SERIES[1]
MODES = [("A_base", "A — базовый", COLOR_A, "o"), ("B_tuned", "B — тюнинг", COLOR_B, "D")]
MODE_SHORT = {"A_base": "A", "B_tuned": "B"}
PROBLEM_SHORT = {"C1_palindrome": "палиндром", "C2_merge_intervals": "интервалы", "C3_int_to_roman": "римские",
                 "C4_top_k_words": "топ-k слов", "C5_decode_string": "k[...]", "C6_longest_valid_parens": "скобки",
                 "C7_min_window": "мин. окно", "C8_calculator": "калькулятор"}
# Последовательная шкала одного оттенка (светлый = хуже, тёмный = лучше).
SEQ = ListedColormap(["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                      "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK2,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.titlelocation": "left",
})


def save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def fmt(v, digits):
    return "—" if math.isnan(v) else f"{v:.{digits}f}"


def load_jsonl(name):
    path = os.path.join(analyze.RESULTS_DIR, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def title(ax, text, subtitle):
    ax.set_title(f"{text}\n", loc="left")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)


def heatmap(ax, values, labels, row_names, col_names):
    """values — нормированные 0..1 (цвет), labels — подписи ячеек с исходными значениями."""
    ax.imshow([[0 if math.isnan(v) else v for v in row] for row in values], cmap=SEQ, vmin=0, vmax=1,
              aspect="auto")
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=9,
                    color="white" if not math.isnan(v) and v > 0.55 else INK)
    ax.set_xticks(range(len(col_names)), col_names)
    ax.set_yticks(range(len(row_names)), row_names)
    ax.tick_params(length=0)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(col_names))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(row_names))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)


def quality_table(runs, judge_scores, code_scores):
    """Строки (model, mode) и столбцы: судья P1–P3, авто P1–P3, код. Значения: (норм. 0..1, подпись)."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["prompt_id"], r["mode"])].append(r)
    code = defaultdict(list)
    for s in code_scores:
        code[(s["model"], s["mode"])].append(s)
    n_problems = len({s["problem_id"] for s in code_scores})
    models = list(dict.fromkeys(r["model"] for r in runs))
    cols = ["P1 судья", "P2 судья", "P3 судья", "P1 авто", "P2 авто", "P3 авто", "Код: решено"]
    table = {}
    for m in models:
        for mode in ("A_base", "B_tuned"):
            row = []
            for p in analyze.SCORERS:
                j = analyze.mean(s.get("overall") for s in judge_scores.get((m, p, mode), []))
                row.append(((j - 1) / 4, fmt(j, 1)))
            for p in analyze.SCORERS:
                a = analyze.mean(analyze.SCORERS[p](r["content"])[0] for r in groups[(m, p, mode)])
                row.append((a, fmt(a, 2)))
            solved = sum(s["passed"] == s["total"] for s in code[(m, mode)])
            row.append((solved / n_problems if n_problems else float("nan"),
                        f"{solved}/{n_problems}" if n_problems else "—"))
            table[(m, mode)] = row
    return models, cols, table


def quality_matrix(models, cols, table):
    keys = [(m, mode) for m in models for mode in ("A_base", "B_tuned")]
    fig, ax = plt.subplots(figsize=(9.6, 0.55 * len(keys) + 1.6))
    heatmap(ax, [[v for v, _ in table[k]] for k in keys], [[t for _, t in table[k]] for k in keys],
            [f"{SHORT.get(m, m)} · {MODE_SHORT[mode]}" for m, mode in keys], cols)
    ax.set_title("Сводная матрица качества\n\n", loc="left")
    ax.text(0, 1.13, "Судья — overall 1–5 (слепая оценка); авто — доля пройденных проверок; код — задачи, "
                     "где пройдены все тесты. Цвет — значение, нормированное к 0–1.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "quality_matrix.png")


def speed_quality(runs, models, table):
    tg = defaultdict(list)
    for r in runs:
        tg[(r["model"], r["mode"])].append(r["timings"].get("predicted_per_second"))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.grid(True, axis="both")
    for color, m in zip(SERIES, models):
        pts = []
        for mode, label, _, marker in MODES:
            x = analyze.mean(tg[(m, mode)])
            y = analyze.mean(v for v, _ in table[(m, mode)])
            pts.append((x, y))
            ax.scatter(x, y, s=90, color=color, marker=marker, edgecolors=SURFACE, linewidths=2, zorder=3)
            ax.annotate(f"{SHORT.get(m, m)} · {MODE_SHORT[mode]}  ({y:.2f})", (x, y), xytext=(9, -3 if mode == "A_base" else 6),
                        textcoords="offset points", fontsize=9, color=INK2)
        ax.plot(*zip(*pts), color=color, lw=2, alpha=0.4, zorder=2)
    ax.set_xlabel("Скорость генерации, ток/с")
    ax.set_ylabel("Среднее качество (0–1)")
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 1)
    title(ax, "Скорость против качества",
          "Качество — среднее нормированных столбцов сводной матрицы; линия соединяет режимы A и B")
    handles = [Line2D([], [], marker=m, color=MUTED, linestyle="", markersize=8, label=lab) for _, lab, _, m in MODES]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    save(fig, "speed_quality.png")


def latency_breakdown(runs, code_runs, models):
    rows = [(p, mode) for p in analyze.SCORERS for mode in ("A_base", "B_tuned")] + \
           [("code", mode) for mode in ("A_base", "B_tuned")]
    labels = [f"{PROMPT_RU.get(p, 'Код (сред.)')} · {MODE_SHORT[mode]}" for p, mode in rows]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["prompt_id"], r["mode"])].append(r)
    for r in code_runs:
        groups[(r["model"], "code", r["mode"])].append(r)
    fig, axes = plt.subplots(1, len(models), figsize=(13, 0.42 * len(rows) + 1.8), sharey=True)
    for ax, m in zip(axes, models):
        pre = [analyze.mean(r["timings"]["prompt_ms"] / 1000 for r in groups[(m, p, mode)]) for p, mode in rows]
        gen = [analyze.mean(r["timings"]["predicted_ms"] / 1000 for r in groups[(m, p, mode)]) for p, mode in rows]
        ax.set_axisbelow(True)
        ax.barh(range(len(rows)), pre, height=0.55, color=SERIES[0])
        ax.barh(range(len(rows)), gen, left=pre, height=0.55, color=SERIES[1])
        for i, (a, b) in enumerate(zip(pre, gen)):
            ax.annotate(f"{a + b:.1f} с", xy=(a + b, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, max(a + b for a, b in zip(pre, gen)) * 1.25)
        ax.set_title(SHORT.get(m, m), fontsize=10)
        ax.set_xlabel("секунды (шкала своя у каждой модели)", fontsize=8, color=MUTED)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(color=SERIES[0], label="prefill (обработка промпта)"),
                        Patch(color=SERIES[1], label="генерация (включая рассуждение)")],
               loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Из чего складывается время ответа", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "latency_breakdown.png")


def code_heatmap(code_scores, models):
    problems = list(dict.fromkeys(s["problem_id"] for s in code_scores))
    cell = {(s["model"], s["mode"], s["problem_id"]): s for s in code_scores}
    keys = [(m, mode) for m in models for mode in ("A_base", "B_tuned")]
    values, labels = [], []
    for m, mode in keys:
        row_v, row_l = [], []
        for p in problems:
            s = cell.get((m, mode, p))
            row_v.append(s["passed"] / s["total"] if s else float("nan"))
            row_l.append(f"{s['passed']}/{s['total']}" if s else "—")
        solved = sum(1 for p in problems if (m, mode, p) in cell and
                     cell[(m, mode, p)]["passed"] == cell[(m, mode, p)]["total"])
        row_v.append(solved / len(problems))
        row_l.append(f"{solved}/{len(problems)}")
        values.append(row_v)
        labels.append(row_l)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(keys) + 1.8))
    heatmap(ax, values, labels, [f"{SHORT.get(m, m)} · {MODE_SHORT[mode]}" for m, mode in keys],
            [p.split("_", 1)[0] + "\n" + PROBLEM_SHORT.get(p, p) for p in problems] + ["Решено"])
    ax.axvline(len(problems) - 0.5, color=SURFACE, lw=6)
    ax.set_title("Задачи по программированию: пройденные скрытые тесты\n\n\n", loc="left")
    ax.text(0, 1.2, "Ячейка — тестов пройдено / всего; режим A — дефолты, B — temperature=0, max_tokens=8192",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "code_heatmap.png")


def tokens_breakdown(runs, models):
    """Токены рассуждения оцениваются по доле символов рассуждения в ответе модели."""
    rows = [(p, mode) for p in analyze.SCORERS for mode in ("A_base", "B_tuned")]
    labels = [f"{PROMPT_RU[p]} · {MODE_SHORT[mode]}" for p, mode in rows]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["prompt_id"], r["mode"])].append(r)

    def split(r):
        total = r["usage"].get("completion_tokens", 0)
        chars = len(r["reasoning"]) + len(r["content"])
        reasoning = total * len(r["reasoning"]) / chars if chars else 0
        return total - reasoning, reasoning

    fig, axes = plt.subplots(1, len(models), figsize=(13, 0.42 * len(rows) + 1.8), sharey=True)
    for ax, m in zip(axes, models):
        parts = [[split(r) for r in groups[(m, p, mode)]] for p, mode in rows]
        ans = [analyze.mean(a for a, _ in ps) for ps in parts]
        rea = [analyze.mean(b for _, b in ps) for ps in parts]
        ax.set_axisbelow(True)
        ax.barh(range(len(rows)), ans, height=0.55, color=SERIES[0])
        ax.barh(range(len(rows)), rea, left=ans, height=0.55, color=SERIES[1])
        for i, (a, b) in enumerate(zip(ans, rea)):
            ax.annotate(f"{a + b:.0f}", xy=(a + b, i), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, max(a + b for a, b in zip(ans, rea)) * 1.2)
        ax.set_title(SHORT.get(m, m), fontsize=10)
        ax.set_xlabel("токены (шкала своя у каждой модели)", fontsize=8, color=MUTED)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(color=SERIES[0], label="ответ"), Patch(color=SERIES[1], label="рассуждение (оценка)")],
               loc="upper right", ncol=2, frameon=False)
    fig.suptitle("Длина ответа, токенов (среднее по 3 повторам)", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "tokens_breakdown.png")


def stability_heatmap(runs, models):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["prompt_id"], r["mode"])].append(analyze.strip_think(r["content"]))
    keys = [(m, mode) for m in models for mode in ("A_base", "B_tuned")]
    values, labels = [], []
    for m, mode in keys:
        row_v, row_l = [], []
        for p in analyze.SCORERS:
            texts = groups[(m, p, mode)]
            sims = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in itertools.combinations(texts, 2)]
            sim = analyze.mean(sims) if sims else float("nan")
            row_v.append(sim)
            row_l.append(f"{sim:.2f}\n{len(set(texts))} из {len(texts)}")
        values.append(row_v)
        labels.append(row_l)
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(keys) + 1.8))
    heatmap(ax, values, labels, [f"{SHORT.get(m, m)} · {MODE_SHORT[mode]}" for m, mode in keys],
            [PROMPT_RU[p] for p in analyze.SCORERS])
    ax.set_title("Стабильность между повторами\n\n", loc="left")
    ax.text(0, 1.12, "Похожесть ответов (1 = идентичны) и число уникальных ответов из 3",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    save(fig, "stability_heatmap.png")


def speed(runs, models):
    per_model = defaultdict(list)
    for r in runs:
        per_model[r["model"]].append(r)
    panels = [("Генерация (decode), ток/с", "predicted_per_second", 1),
              ("Обработка промпта (prefill), ток/с", "prompt_per_second", 0)]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 1.0 + 1.1 * len(models)))
    for ax, (panel_title, key, digits) in zip(axes, panels):
        vals = [analyze.mean(r["timings"].get(key) for r in per_model[m]) for m in models]
        ax.set_axisbelow(True)
        bars = ax.barh([SHORT.get(m, m) for m in models], vals, height=0.45, color=COLOR_A)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:,.{digits}f}".replace(",", " "), xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlim(0, max(vals) * 1.15)
    fig.suptitle("Скорость инференса (llama.cpp, RTX 5070 + Ryzen 9 9950X)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "speed.png")


def main():
    runs = load_jsonl("runs.jsonl")
    code_runs = load_jsonl("code_runs.jsonl")
    code_scores = load_jsonl("code_scores.jsonl")
    judge_scores = load_scores()
    models, cols, table = quality_table(runs, judge_scores, code_scores)
    quality_matrix(models, cols, table)
    speed_quality(runs, models, table)
    latency_breakdown(runs, code_runs, models)
    if code_scores:
        code_heatmap(code_scores, models)
    speed(runs, models)
    tokens_breakdown(runs, models)
    stability_heatmap(runs, models)


if __name__ == "__main__":
    main()
