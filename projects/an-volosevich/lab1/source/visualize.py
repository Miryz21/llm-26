"""Строит графики по results/runs.jsonl (+ оценки судьи) в results/plots/*.png.

  speed.png       — скорость генерации и обработки промпта (две панели, у каждой своя шкала)
  quality.png     — автоматическая оценка качества, A -> B (dumbbell)
  judge.png       — оценка LLM-судьи overall 1..5, A -> B
  stability.png   — похожесть ответов между повторами, A -> B
  out_tokens.png  — длина ответа в токенах, A -> B
  latency.png     — время отклика, A -> B
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
from matplotlib.lines import Line2D  # noqa: E402

import analyze  # noqa: E402
from judge import load_scores  # noqa: E402

PLOTS_DIR = os.path.join(analyze.RESULTS_DIR, "plots")
PROMPT_RU = {"P1_generation": "P1 генерация", "P2_classification": "P2 классификация",
             "P3_extraction": "P3 извлечение"}

# Пара цветов, различимая при дальтонизме; A и B различаются ещё и формой маркера.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
MODES = [("A_base", "A — базовый", COLOR_A, "o"), ("B_tuned", "B — тюнинг", COLOR_B, "D")]

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


def dumbbell(name, title, subtitle, rows, xlim=None, digits=2):
    """rows: [(model, prompt_id, a, b)]. Строки сгруппированы по модели."""
    rows = [r for r in rows if not (math.isnan(r[2]) and math.isnan(r[3]))]
    if not rows:
        return
    labels, ys, y = [], [], 0
    group_y = []
    for model in dict.fromkeys(r[0] for r in rows):
        group_y.append((y, model))
        y += 1
        for r in (r for r in rows if r[0] == model):
            ys.append((y, r))
            y += 1
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * y + 1.6))
    for yy, (_, pid, a, b) in ys:
        if not (math.isnan(a) or math.isnan(b)):
            ax.plot([a, b], [yy, yy], color=MUTED, lw=2, alpha=0.5, solid_capstyle="round", zorder=1)
        # Небольшой вертикальный сдвиг, чтобы совпадающие A и B не перекрывали друг друга.
        for (_, _, color, marker), v, dy in zip(MODES, (a, b), (-0.13, 0.13)):
            if not math.isnan(v):
                ax.scatter(v, yy + dy, s=60, color=color, marker=marker, edgecolors=SURFACE,
                           linewidths=2, zorder=3)
        ax.annotate(f"{fmt(a, digits)} → {fmt(b, digits)}", xy=(1.01, yy), xycoords=("axes fraction", "data"),
                    va="center", fontsize=9, color=INK2)
    ax.set_yticks([yy for yy, _ in ys], [PROMPT_RU.get(r[1], r[1]) for _, r in ys])
    for gy, model in group_y:
        ax.annotate(model, xy=(0, gy), xycoords=("figure fraction", "data"), xytext=(8, 0),
                    textcoords="offset points", va="center", fontweight="bold", color=INK)
    ax.set_ylim(y - 0.4, -0.6)
    if xlim:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{title}\n", loc="left")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    handles = [Line2D([], [], marker=m, color=c, linestyle="", markersize=8, markeredgecolor=SURFACE,
                      label=lab) for _, lab, c, m in MODES]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, -0.02 - 1.2 / y), ncol=2, frameon=False)
    save(fig, name)


def speed(runs, models):
    per_model = defaultdict(list)
    for r in runs:
        per_model[r["model"]].append(r)
    panels = [("Генерация (decode), ток/с", "predicted_per_second", 1),
              ("Обработка промпта (prefill), ток/с", "prompt_per_second", 0)]
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 1.0 + 1.1 * len(models)))
    for ax, (title, key, digits) in zip(axes, panels):
        vals = [analyze.mean(r["timings"].get(key) for r in per_model[m]) for m in models]
        ax.set_axisbelow(True)
        bars = ax.barh(models, vals, height=0.45, color=COLOR_A)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:,.{digits}f}".replace(",", " "), xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(vals) * 1.15)
    fig.suptitle("Скорость инференса (llama.cpp, RTX 5070 + Ryzen 9 9950X)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    save(fig, "speed.png")


def main():
    runs = [json.loads(line) for line in open(os.path.join(analyze.RESULTS_DIR, "runs.jsonl"), encoding="utf-8")]
    judge_scores = load_scores()
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], r["prompt_id"], r["mode"])].append(r)
    models = list(dict.fromkeys(r["model"] for r in runs))

    def metric(fn):
        return [(m, p, fn(groups.get((m, p, "A_base"), []), m, p, "A_base"),
                 fn(groups.get((m, p, "B_tuned"), []), m, p, "B_tuned"))
                for m in models for p in analyze.SCORERS]

    def similarity(rs, *_):
        texts = [analyze.strip_think(r["content"]) for r in rs]
        sims = [difflib.SequenceMatcher(None, a, b).ratio() for a, b in itertools.combinations(texts, 2)]
        return analyze.mean(sims) if sims else float("nan")

    dumbbell("quality.png", "Автоматическая оценка качества",
             "Доля выполненных проверок задачи (0–1), среднее по повторам",
             metric(lambda rs, m, p, _: analyze.mean(analyze.SCORERS[p](r["content"])[0] for r in rs)), (0, 1.02))
    if judge_scores:
        dumbbell("judge.png", "Оценка LLM-судьи (слепая)", "Overall 1–5, среднее по повторам",
                 metric(lambda rs, m, p, mode: analyze.mean(s.get("overall") for s in judge_scores.get((m, p, mode), []))),
                 (1, 5.1))
    dumbbell("stability.png", "Стабильность ответов между повторами",
             "Средняя попарная похожесть текстов (difflib), 1 = идентичны", metric(similarity), (0, 1.02))
    dumbbell("out_tokens.png", "Длина ответа", "completion_tokens (у Qwen включая рассуждение), среднее",
             metric(lambda rs, *_: analyze.mean(r["usage"].get("completion_tokens") for r in rs)), digits=0)
    dumbbell("latency.png", "Время отклика", "Полный запрос без стриминга, секунды, среднее",
             metric(lambda rs, *_: analyze.mean(r["wall_s"] for r in rs)), digits=1)
    speed(runs, models)


if __name__ == "__main__":
    main()
