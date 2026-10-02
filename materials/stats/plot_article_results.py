#!/usr/bin/env python3
"""Render publication-oriented figures from build_article_results.py output."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper-action-order-matplotlib")

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

from build_article_results import (
    DEFAULT_OUTPUT,
    MODEL_LABELS,
    MODEL_SOURCES,
    ORDER_KEYS,
    ORDER_LABELS_RU,
    SERIES_CONFIGS,
)
from experiment_config import PROBLEM_IDS


TEXT_COLOR = "#17182C"
GRID_COLOR = "#D9DCF4"
PALETTE = {
    "dark": "#3E00BE",
    "medium": "#6c3fff",
    "light": "#7197ff",
}
MODEL_COLORS = {
    "deepseek/deepseek-v4-flash": PALETTE["dark"],
    "gpt-oss-120b": PALETTE["medium"],
    "nemotron-3-super": PALETTE["light"],
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titlecolor": TEXT_COLOR,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.9,
        "grid.alpha": 1.0,
        "legend.frameon": False,
    }
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def style_axes(axis: plt.Axes, *, grid_axis: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID_COLOR)
    axis.spines["bottom"].set_color(GRID_COLOR)
    axis.grid(True, axis=grid_axis)
    axis.set_axisbelow(True)


def plot_reachability(summary: list[dict[str, str]], output_dir: Path, series: dict[str, object]) -> None:
    index = {
        (row["model"], row["order"]): row
        for row in summary
        if row["metric"] == "reachability" and row["series"] == series["key"]
    }
    orders = series["orders"]
    figure, axis = plt.subplots(figsize=(9.2, 5.0 if len(orders) > 4 else 4.3))
    y_values = list(range(len(orders)))
    offsets = (-0.18, 0.0, 0.18)

    for offset, model in zip(offsets, series["model_sources"]):
        records = [index[(model, order)] for order in orders]
        rates = [float(record["rate"]) for record in records]
        lower = [rate - float(record["ci95_low_wilson"]) for rate, record in zip(rates, records)]
        upper = [float(record["ci95_high_wilson"]) - rate for rate, record in zip(rates, records)]
        axis.errorbar(
            rates,
            [value + offset for value in y_values],
            xerr=[lower, upper],
            color=MODEL_COLORS[model],
            label=MODEL_LABELS[model],
            marker="o",
            markersize=5.5,
            linestyle="none",
            linewidth=1.45,
            elinewidth=1.25,
            capsize=3,
        )
    axis.set_yticks(y_values, [series["order_labels"][key] for key in orders])
    axis.invert_yaxis()
    axis.set_xlim(0.05, 0.68)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_xlabel("Доля планов, достигших цели")
    axis.set_ylabel("Порядок схем действий")
    axis.set_title("Достижение цели", pad=12)
    style_axes(axis, grid_axis="x")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    figure.subplots_adjust(left=0.20, right=0.98, bottom=0.17, top=0.78)
    save_figure(figure, output_dir, f"figure_reachability_by_order_{series['key']}")


def plot_task_delta_heatmap(task_rates: list[dict[str, str]], output_dir: Path, series: dict[str, object]) -> None:
    reachability = [row for row in task_rates if row["metric"] == "reachability" and row["series"] == series["key"]]
    index = {
        (row["model"], row["problem"], row["order"]): float(row["rate"])
        for row in reachability
    }
    baseline_order = series["baseline"]
    compared_orders = [order for order in series["orders"] if order != baseline_order]
    matrices: dict[str, list[list[float]]] = {}
    max_abs = 0.0
    for model in series["model_sources"]:
        matrix: list[list[float]] = []
        for problem in PROBLEM_IDS:
            baseline = index[(model, problem, baseline_order)]
            row = [100 * (index[(model, problem, order)] - baseline) for order in compared_orders]
            matrix.append(row)
            max_abs = max(max_abs, *(abs(value) for value in row))
        matrices[model] = matrix

    limit = max(10.0, max_abs)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    heatmap_cmap = LinearSegmentedColormap.from_list(
        "violet_white_blue",
        [PALETTE["dark"], "#F8F8FD", PALETTE["light"]],
    )
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 9.2), sharey=True)
    image = None
    for axis, model in zip(axes, series["model_sources"]):
        matrix = matrices[model]
        image = axis.imshow(matrix, cmap=heatmap_cmap, norm=norm, aspect="auto")
        axis.set_title(MODEL_LABELS[model], fontsize=11)
        axis.set_xticks(
            range(len(compared_orders)),
            [series["order_labels"][key].replace(" ", "\n") for key in compared_orders],
            fontsize=8,
        )
        axis.set_yticks(range(len(PROBLEM_IDS)), PROBLEM_IDS, fontsize=8)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_color(GRID_COLOR)
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                text_color = "white" if abs(value) > 0.55 * limit else TEXT_COLOR
                axis.text(x, y, f"{value:+.0f}", ha="center", va="center", fontsize=7, color=text_color)
    axes[0].set_ylabel("Задача")
    figure.subplots_adjust(left=0.06, right=0.88, bottom=0.07, top=0.92, wspace=0.18)
    if image is not None:
        colorbar_axis = figure.add_axes((0.91, 0.18, 0.018, 0.64))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Изменение относительно исходного порядка, п.п.")
    figure.suptitle(f"Неоднородность эффекта порядка между задачами: {series['label']}", fontsize=13, y=0.98)
    save_figure(figure, output_dir, f"figure_task_reachability_delta_heatmap_{series['key']}")


def plot_outcome_composition(outcomes: list[dict[str, str]], output_dir: Path, series: dict[str, object]) -> None:
    index = {(row["model"], row["order"]): row for row in outcomes if row["series"] == series["key"]}
    categories = (
        ("reached", "Цель достигнута", PALETTE["dark"]),
        ("executable_without_goal", "Исполним, цель не достигнута", PALETTE["medium"]),
        ("non_executable", "Неисполним", PALETTE["light"]),
    )
    orders = series["orders"]
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    x_values = list(range(len(orders)))
    x_labels = [series["order_labels"][key].replace(" ", "\n") for key in orders]
    for axis, model in zip(axes, series["model_sources"]):
        bottoms = [0.0] * len(orders)
        for field, label, color in categories:
            values = [
                int(index[(model, order)][field]) / int(index[(model, order)]["total"])
                for order in orders
            ]
            axis.bar(x_values, values, bottom=bottoms, label=label, color=color, width=0.72)
            bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
        axis.set_title(MODEL_LABELS[model], fontsize=11)
        axis.set_xticks(x_values, x_labels, fontsize=8)
        axis.set_ylim(0, 1)
        style_axes(axis, grid_axis="y")
    axes[0].set_ylabel("Доля запусков")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.suptitle(f"Состав исходов генерации планов: {series['label']}", fontsize=13)
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.84, wspace=0.12)
    save_figure(figure, output_dir, f"figure_outcome_composition_{series['key']}")


def write_captions(output_dir: Path) -> None:
    text = """# Подписи к рисункам

1. **Достижение цели при различных порядках схем действий.** Точки показывают долю строго валидных планов, достигших целевого состояния; интервалы — 95% интервалы Уилсона. Для каждой комбинации модели и порядка выполнено 300 независимых запусков.

2. **Неоднородность эффекта порядка между задачами.** В ячейках показано изменение доли достигших цели планов относительно исходного порядка, в процентных пунктах. Отрицательные значения соответствуют ухудшению.

3. **Состав исходов генерации планов.** Исполнимые планы, не достигшие цели, отделены от неисполнимых планов; достижение цели проверено строгим валидатором VAL.
"""
    (output_dir / "figure_captions_ru.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot final article results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(input_dir / "summary_rates.csv")
    task_rates = read_csv(input_dir / "task_rates.csv")
    outcomes = read_csv(input_dir / "outcome_composition.csv")
    for series in SERIES_CONFIGS:
        plot_reachability(summary, output_dir, series)
        plot_task_delta_heatmap(task_rates, output_dir, series)
        plot_outcome_composition(outcomes, output_dir, series)
    write_captions(output_dir)
    print(f"Wrote article figures to {output_dir}")


if __name__ == "__main__":
    main()
