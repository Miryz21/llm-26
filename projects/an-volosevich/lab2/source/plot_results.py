"""График сравнения конфигураций retrieval: Recall@5 и MRR по языкам вопросов (results/plots/)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"ru": ("#2a78d6", "вопрос на русском"),
          "en": ("#eb6834", "вопрос на английском"),
          "ru→en": ("#1baf7a", "русский → перевод LLM")}
MARKERS = {"ru": "o", "en": "s", "ru→en": "D"}
OFFSET = {"ru": 0.14, "en": -0.14, "ru→en": 0.0}  # чтобы совпавшие значения не перекрывали друг друга


def main():
    with open(os.path.join(RESULTS_DIR, "retrieval.json"), encoding="utf-8") as f:
        experiments = json.load(f)["experiments"]
    names = list(dict.fromkeys(e["name"] for e in experiments))
    y = {name: len(names) - 1 - i for i, name in enumerate(names)}

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
                         "xtick.color": TEXT_2, "ytick.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.2), sharey=True, facecolor=SURFACE)
    for ax, metric, title in zip(axes, ("recall@5", "mrr"), ("Recall@5", "MRR")):
        ax.set_facecolor(SURFACE)
        for name in names:
            ax.axhline(y[name], color=GRID, lw=1, zorder=0)
        for lang, (color, _) in SERIES.items():
            pts = [(e["metrics"][metric], y[e["name"]] + OFFSET[lang]) for e in experiments if e["lang"] == lang]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=70, color=color, marker=MARKERS[lang],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.set_xlim(0.2, 1.0)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([y[n] for n in names], names)
    handles = [plt.Line2D([], [], ls="", marker=MARKERS[l], color=c, markersize=8, label=lab)
               for l, (c, lab) in SERIES.items()]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.005, 0.955))
    fig.suptitle("Качество retrieval по конфигурациям (18 вопросов, k = 5)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
    fig.savefig(os.path.join(RESULTS_DIR, "plots", "retrieval.png"), dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()
