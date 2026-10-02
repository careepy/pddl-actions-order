#!/usr/bin/env python3
"""Render a reference-style dot-and-whisker panel for every metric."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper-action-order-matplotlib")

from matplotlib.ticker import PercentFormatter

from build_article_all_metrics import DEFAULT_ALL_METRICS_OUTPUT, METRICS
from build_article_results import MODEL_LABELS, MODEL_SOURCES, ORDER_KEYS, ORDER_LABELS_RU, SERIES_CONFIGS
from plot_article_results import (
    GRID_COLOR,
    MODEL_COLORS,
    TEXT_COLOR,
    plt,
    save_figure,
    style_axes,
)


AXIS_LIMITS = {
    "executability": (0.0, 1.0),
    "reachability": (0.0, 1.0),
    "conditional_reachability": (0.0, 1.0),
    "optimality_ratio": (0.8, 1.4),
    "plan_length": (0.0, 50.0),
    "execution_progress": (0.0, 1.0),
    "first_failure_step": (0.0, 25.0),
    "non_executable_failure": (0.0, 1.0),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_all_metrics(
    summary: list[dict[str, str]], output_dir: Path, series: dict[str, object],
    *, stem: str | None = None,
) -> None:
    index = {
        (row["model"], row["order"], row["metric"]): row
        for row in summary
        if row["series"] == series["key"]
    }
    figure, axes = plt.subplots(2, 4, figsize=(14.2, 6.35))
    orders = series["orders"]
    y_values = list(range(len(orders)))
    offsets = (-0.18, 0.0, 0.18)

    for panel, (axis, metric) in enumerate(zip(axes.flat, METRICS)):
        metric_key = metric["key"]
        for offset, model in zip(offsets, series["model_sources"]):
            records = [index[(model, order, metric_key)] for order in orders]
            means = [float(record["task_mean"]) for record in records]
            lower = [
                mean - float(record["ci95_low_task_bootstrap"])
                for mean, record in zip(means, records)
            ]
            upper = [
                float(record["ci95_high_task_bootstrap"]) - mean
                for mean, record in zip(means, records)
            ]
            axis.errorbar(
                means,
                [value + offset for value in y_values],
                xerr=[lower, upper],
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
                marker="o",
                markersize=4.1,
                linestyle="none",
                linewidth=1.05,
                elinewidth=1.0,
                capsize=2.2,
            )

        axis.set_title(metric["label_ru"], fontsize=9.5, pad=6)
        axis.set_yticks(y_values)
        if panel % 4 == 0:
            axis.set_yticklabels(
                [series["order_labels"][key] for key in orders], fontsize=8
            )
            axis.set_ylabel("Порядок действий", fontsize=8)
        else:
            axis.set_yticklabels([])
        axis.invert_yaxis()
        is_percent = metric["format"] == "percent"
        axis.set_xlim(*AXIS_LIMITS[metric_key])
        if is_percent:
            axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        axis.set_xlabel("Значение метрики", fontsize=8)
        axis.tick_params(axis="x", labelsize=8)
        style_axes(axis, grid_axis="x")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        columnspacing=2.0,
        handletextpad=0.6,
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.995,
        bottom=0.09,
        top=0.865,
        wspace=0.18,
        hspace=0.42,
    )
    save_figure(figure, output_dir, stem or f"figure_all_metrics_{series['key']}")


def write_caption(output_dir: Path) -> None:
    text = """# Подпись к рисунку

**Все метрики качества планов при различных порядках схем действий.** Точки показывают средние значения метрики по задачам, горизонтальные интервалы — 95% непараметрические bootstrap-интервалы по 20 задачам. Условное достижение цели рассчитано только среди исполнимых планов; длина плана и отношение длины к эталону — только среди планов, достигших цели; первый ошибочный шаг — только для ошибок исполнения, где VAL сообщил номер шага. Поэтому число наблюдений у условных метрик различается между моделями и порядками и приведено в отдельных таблицах.
"""
    (output_dir / "figure_all_metrics_caption_ru.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot every article metric.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ALL_METRICS_OUTPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = read_csv(input_dir / "all_metrics_summary.csv")
    for series in SERIES_CONFIGS:
        plot_all_metrics(summary, output_dir, series)
    write_caption(output_dir)
    print(f"Wrote all-metric article figure to {output_dir}")


if __name__ == "__main__":
    main()
