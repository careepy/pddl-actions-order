#!/usr/bin/env python3
"""Draw the two manuscript figures from the current ten-order audit."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from run_order_audit import DEFAULT_OUT, MODELS, ORDERS


OUT = DEFAULT_OUT / "manuscript_figures"
LABELS = {
    "canonical": "Исходный",
    "canonical_disp_1": "Смещение 1/3",
    "canonical_disp_2": "Смещение 2/3",
    "canonical_disp_3": "Обратный",
    "plan_front": "План в начале",
    "plan_back": "План в конце",
    "plan_scatter": "План рассеян",
    "reference_disp_1": "От плана: 1/3",
    "reference_disp_2": "От плана: 2/3",
    "reference_reverse": "Обратный к плану",
}
MODEL_LABELS = {
    MODELS[0]: "DeepSeek V4 Flash",
    MODELS[1]: "GPT-OSS-120B",
    MODELS[2]: "Nemotron-3-Super",
}
MODEL_COLORS = ("#3156A3", "#BC5736", "#25816B")
OUTCOME_LABELS = (
    ("goal_reached", "Цель достигнута", "#3500BF"),
    ("executable_no_goal", "Исполним, без цели", "#653AFF"),
    ("execution_error", "Ошибка исполнения", "#638AF4"),
    ("parse_error", "Ошибка разбора", "#AEC4FF"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=240, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def plot_reachability() -> None:
    rows = read_rows(DEFAULT_OUT / "metric_summaries.csv")
    lookup = {(r["model"], r["order"]): r for r in rows if r["metric"] == "reachability"}
    fig, ax = plt.subplots(figsize=(7.7, 5.25))
    y = np.arange(len(ORDERS))
    for model, color, offset in zip(MODELS, MODEL_COLORS, (-0.23, 0.0, 0.23)):
        selected = [lookup[(model, order)] for order in ORDERS]
        means = np.array([float(r["task_mean"]) for r in selected])
        low = np.array([float(r["ci95_low"]) for r in selected])
        high = np.array([float(r["ci95_high"]) for r in selected])
        ax.errorbar(
            means, y + offset, xerr=np.vstack((means - low, high - means)),
            fmt="o", markersize=4.2, capsize=2.1, elinewidth=1.05,
            color=color, label=MODEL_LABELS[model],
        )
    ax.set_yticks(y, [LABELS[order] for order in ORDERS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.81)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Доля планов, достигших цели", fontsize=10)
    ax.grid(axis="x", color="#DFE2E6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, fontsize=8.4, bbox_to_anchor=(0.62, 0.99))
    fig.subplots_adjust(left=0.27, right=0.975, top=0.91, bottom=0.105)
    save(fig, "figure_2_reachability_all_ten")


def plot_outcomes() -> None:
    rows = read_rows(DEFAULT_OUT / "outcome_counts.csv")
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts[(row["model"], row["order"])][row["outcome"]] += int(row["count"])
    fig, axes = plt.subplots(3, 1, figsize=(7.7, 8.35), sharex=True)
    y = np.arange(len(ORDERS))
    for ax, model in zip(axes, MODELS):
        left = np.zeros(len(ORDERS))
        for outcome, label, color in OUTCOME_LABELS:
            values = np.array([counts[(model, order)][outcome] / 300 for order in ORDERS])
            ax.barh(y, values, left=left, height=0.77, label=label,
                    color=color, edgecolor="white", linewidth=0.35)
            left += values
        if not np.allclose(left, 1):
            raise ValueError(f"Outcome proportions do not sum to one for {model}")
        ax.set_yticks(y, [LABELS[order] for order in ORDERS], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(MODEL_LABELS[model], loc="left", fontsize=10, pad=3)
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color="#DFE2E6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[-1].set_xlabel("Доля ответов", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.59, 0.997))
    fig.subplots_adjust(left=0.27, right=0.985, top=0.925, bottom=0.06, hspace=0.28)
    save(fig, "figure_3_four_outcomes_all_ten")


def main() -> None:
    plot_reachability()
    plot_outcomes()
    print(f"Wrote manuscript figures to {OUT}")


if __name__ == "__main__":
    main()
